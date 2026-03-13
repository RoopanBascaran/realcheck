from flask import Flask, request, render_template
from werkzeug.utils import secure_filename
import os

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

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

if __name__ == '__main__':
    debug = os.environ.get('FLASK_DEBUG', 'true').lower() == 'true'
    port = int(os.environ.get('PORT', 7860))
    app.run(debug=debug, host='0.0.0.0', port=port)