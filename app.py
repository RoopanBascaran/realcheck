from flask import Flask, request, render_template, jsonify
from werkzeug.utils import secure_filename
import os
import socket
import requests
import logging
import tempfile

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Workaround: HF Spaces blocks DNS for Meta/Facebook domains.
# Use Google DNS-over-HTTPS as fallback resolver.
_original_getaddrinfo = socket.getaddrinfo

def _resolve_via_doh(hostname):
    """Resolve hostname using Google DNS-over-HTTPS."""
    try:
        import urllib.request
        import json
        url = f'https://dns.google/resolve?name={hostname}&type=A'
        req = urllib.request.Request(url, headers={'Accept': 'application/dns-json'})
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read())
        for answer in data.get('Answer', []):
            if answer.get('type') == 1:
                logger.info(f'DoH resolved {hostname} -> {answer["data"]}')
                return answer['data']
    except Exception as e:
        logger.error(f'DoH resolution failed for {hostname}: {e}')
    return None

def _patched_getaddrinfo(host, port, family=0, type=0, proto=0, flags=0):
    try:
        return _original_getaddrinfo(host, port, family, type, proto, flags)
    except socket.gaierror:
        ip = _resolve_via_doh(host)
        if ip:
            return [(socket.AF_INET, socket.SOCK_STREAM, 6, '', (ip, port if isinstance(port, int) and port else 443))]
        raise

socket.getaddrinfo = _patched_getaddrinfo
logger.info('DNS fallback via Google DoH enabled')

INSTAGRAM_ACCESS_TOKEN = os.environ.get('INSTAGRAM_ACCESS_TOKEN')
WEBHOOK_VERIFY_TOKEN = os.environ.get('WEBHOOK_VERIFY_TOKEN', 'check4real_verify')

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

logger.info('Flask app initialized successfully')

model = None
def get_model():
    global model
    if model is None:
        from models.xception_model import load_model
        model = load_model()
    return model

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_video():
    if 'video' not in request.files:
        return 'No file part', 400
    file = request.files['video']
    if file.filename == '':
        return 'No selected file', 400
    if file:
        filename = secure_filename(file.filename)
        if not filename:
            return 'Invalid filename', 400
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        prediction = classify_video(filepath)
        return render_template('result.html', classification=prediction, video_id=filename)

def classify_video(filepath):
    return get_model().classify(filepath)

@app.route('/feedback', methods=['POST'])
def feedback():
    from models.feedback_trainer import collect_feedback, retrain_model
    user_feedback = request.form['user_feedback']
    video_id = request.form['video_id']
    collect_feedback(video_id, user_feedback)
    retrain_model()
    return 'Feedback received', 200

@app.route('/privacy')
def privacy():
    return '''<h1>Privacy Policy - Check4Real</h1>
    <p>Check4Real analyzes videos sent via Instagram DM to detect AI-generated content.</p>
    <p>We only process the video you send. We do not store your personal data or messages.</p>
    <p>Videos are temporarily downloaded for analysis and immediately deleted after processing.</p>
    <p>Contact: roopan@check4real.com</p>''', 200


# Debug endpoint to test outbound connectivity
@app.route('/debug/network')
def debug_network():
    import socket
    results = {}

    # Check resolv.conf
    try:
        with open('/etc/resolv.conf', 'r') as f:
            results['resolv_conf'] = f.read()
    except Exception as e:
        results['resolv_conf'] = str(e)

    # Test DNS resolution for various hosts
    for host in ['graph.instagram.com', 'graph.facebook.com', 'google.com', 'api.github.com']:
        try:
            ip = socket.gethostbyname(host)
            results[f'dns_{host}'] = ip
        except Exception as e:
            results[f'dns_{host}'] = str(e)

    # Test HTTP connectivity
    for url in ['https://graph.instagram.com', 'https://google.com']:
        try:
            resp = requests.get(url, timeout=5)
            results[f'http_{url}'] = resp.status_code
        except Exception as e:
            results[f'http_{url}'] = str(e)

    return jsonify(results)


# Instagram Webhook verification (GET)
@app.route('/webhook', methods=['GET'])
def webhook_verify():
    mode = request.args.get('hub.mode')
    token = request.args.get('hub.verify_token')
    challenge = request.args.get('hub.challenge')

    if mode == 'subscribe' and token == WEBHOOK_VERIFY_TOKEN:
        logger.info('Webhook verified successfully')
        return challenge, 200
    return 'Forbidden', 403


# Debug endpoint to check last webhook payload
last_webhook_data = {}

@app.route('/webhook/debug')
def webhook_debug():
    return jsonify(last_webhook_data)


# Instagram Webhook events (POST)
@app.route('/webhook', methods=['POST'])
def webhook_receive():
    global last_webhook_data
    data = request.get_json()
    last_webhook_data = data or {}
    logger.info(f'Webhook received: {data}')

    if data and data.get('object') == 'instagram':
        for entry in data.get('entry', []):
            # Handle messaging events
            for messaging in entry.get('messaging', []):
                sender_id = messaging.get('sender', {}).get('id')
                message = messaging.get('message', {})

                if not sender_id or not message:
                    continue

                # Skip echo messages (messages sent by us)
                if message.get('is_echo'):
                    logger.info(f'Skipping echo message from {sender_id}')
                    continue

                logger.info(f'Message from {sender_id}: {message}')

                attachments = message.get('attachments', [])
                video_url = None

                for att in attachments:
                    att_type = att.get('type')
                    payload = att.get('payload', {})
                    logger.info(f'Attachment type: {att_type}, payload: {payload}')

                    if att_type in ('video', 'ig_reel', 'share', 'media_share'):
                        video_url = payload.get('url')
                        break

                if video_url:
                    # First confirm we can reply
                    logger.info(f'Found video URL, sending to handler: {video_url[:100]}...')
                    handle_video_message(sender_id, video_url)
                elif attachments:
                    # Log unknown attachment types for debugging
                    logger.info(f'Unknown attachment types: {[a.get("type") for a in attachments]}')
                    send_instagram_reply(sender_id, "I received your message but couldn't find a video. Please send a reel directly.")
                else:
                    send_instagram_reply(
                        sender_id,
                        "Hi! Send me a reel or video and I'll check if it's AI-generated or real."
                    )

            # Handle changes-based events (alternative webhook format)
            for change in entry.get('changes', []):
                logger.info(f'Change event: {change}')
                if change.get('field') == 'messages':
                    value = change.get('value', {})
                    sender_id = value.get('sender', {}).get('id')
                    message = value.get('message', {})
                    if sender_id:
                        attachments = message.get('attachments', [])
                        video_url = None
                        for att in attachments:
                            att_type = att.get('type')
                            payload = att.get('payload', {})
                            if att_type in ('video', 'ig_reel', 'share', 'media_share'):
                                video_url = payload.get('url')
                                break
                        if video_url:
                            handle_video_message(sender_id, video_url)
                        else:
                            send_instagram_reply(
                                sender_id,
                                "Hi! Send me a reel or video and I'll check if it's AI-generated or real."
                            )

    return jsonify({'status': 'ok'}), 200


def handle_video_message(sender_id, video_url):
    try:
        resp = send_instagram_reply(sender_id, "Analyzing your video... please wait.")
        logger.info(f'Initial reply response: {resp.status_code} {resp.text}')

        # Download video to temp file
        response = requests.get(video_url, stream=True, timeout=60)
        logger.info(f'Video download status: {response.status_code}')
        response.raise_for_status()

        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as tmp:
            for chunk in response.iter_content(chunk_size=8192):
                tmp.write(chunk)
            tmp_path = tmp.name

        # Classify the video
        result = classify_video(tmp_path)

        # Clean up
        os.unlink(tmp_path)

        send_instagram_reply(sender_id, f"Result: This video appears to be {result}")

    except Exception as e:
        logger.error(f'Error processing video: {e}')
        try:
            send_instagram_reply(sender_id, "Sorry, I couldn't process that video. Please try again.")
        except Exception as reply_err:
            logger.error(f'Failed to send error reply: {reply_err}')


def send_instagram_reply(recipient_id, text):
    url = f'https://graph.instagram.com/v21.0/me/messages'
    headers = {
        'Authorization': f'Bearer {INSTAGRAM_ACCESS_TOKEN}',
        'Content-Type': 'application/json'
    }
    payload = {
        'recipient': {'id': recipient_id},
        'message': {'text': text}
    }
    logger.info(f'Sending reply to {recipient_id}: {text[:50]}...')
    resp = requests.post(url, json=payload, headers=headers, timeout=30)
    if resp.status_code != 200:
        logger.error(f'Failed to send message: {resp.status_code} {resp.text}')
    else:
        logger.info(f'Reply sent successfully')
    return resp


if __name__ == '__main__':
    debug = os.environ.get('FLASK_DEBUG', 'true').lower() == 'true'
    port = int(os.environ.get('PORT', 7860))
    app.run(debug=debug, host='0.0.0.0', port=port)