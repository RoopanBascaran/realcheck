"""Generate natural, varied responses for the Instagram bot using HF Inference API.

Falls back to template responses if the LLM is unavailable, so the bot never breaks.
"""
import os
import random
import logging

logger = logging.getLogger(__name__)

HF_TOKEN = os.environ.get('HF_TOKEN')
LLM_MODEL = 'meta-llama/Llama-3.2-3B-Instruct'

# Template fallbacks — used if LLM fails or is unavailable
ANALYZING_TEMPLATES = [
    "on it! checking if this is real or AI... 🔍",
    "hold up, analyzing the video... ⏳",
    "lemme scan this real quick 👀",
    "checking the video now... give me a sec ⏱️",
    "analyzing... stay with me 🔎",
]

AI_RESULT_TEMPLATES = [
    "brooo 👀 this looks like a fake video! don't fall for this, be aware ⚠️",
    "yo this is AI generated 🤖 stay alert, some accounts spread fake stuff",
    "nah fam, this ain't real 🚫 looks AI-made. always double check before trusting",
    "heads up! this video seems AI-generated 😬 don't believe everything online",
    "this one's fake 🚨 AI generated for sure. be careful out there!",
    "careful! this looks AI-made 🤖 don't share it as real",
]

REAL_RESULT_TEMPLATES = [
    "this looks legit ✅ seems like a real video to me!",
    "all good fam 👍 this appears to be an actual real video",
    "yep this one's real 💯 nothing AI about it from what I can tell",
    "looks authentic to me ✅ real video, real content!",
    "this one checks out ✅ seems like a genuine real video",
    "real deal 💯 no AI signs on this one",
]

ERROR_TEMPLATES = [
    "oops, something went wrong 😕 could you try sending the video again?",
    "hmm couldn't process that one 🤔 try again?",
    "ran into an issue with that video 😬 please try once more",
    "something's off on my end 🛠️ send it again please?",
]

FEEDBACK_PROMPT = "\n\nwhat do you think? is this fake? 👇"


def _call_llm(prompt, max_tokens=80):
    """Call HF Inference API. Returns generated text or None on failure."""
    if not HF_TOKEN:
        return None
    try:
        from huggingface_hub import InferenceClient
        client = InferenceClient(model=LLM_MODEL, token=HF_TOKEN, timeout=10)
        response = client.chat_completion(
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=0.9,
        )
        text = response.choices[0].message.content.strip()
        # Strip surrounding quotes if LLM wrapped the response
        if text.startswith('"') and text.endswith('"'):
            text = text[1:-1]
        return text
    except Exception as e:
        logger.warning(f'LLM call failed, using template fallback: {e}')
        return None


def get_analyzing_message():
    """Short message sent when video is received, before classification."""
    prompt = (
        "Generate a short casual Instagram DM reply (1 sentence, max 10 words) "
        "telling the user you're analyzing their video to check if it's real or AI. "
        "Use casual slang and 1 emoji. Reply with ONLY the message, no quotes, no explanation."
    )
    text = _call_llm(prompt, max_tokens=30)
    return text or random.choice(ANALYZING_TEMPLATES)


def get_result_message(result):
    """Message with the classification result. `result` is 'AI-generated' or 'Real'."""
    if result == 'AI-generated':
        prompt = (
            "Generate a short casual Instagram DM reply (1-2 sentences, max 25 words) "
            "warning the user their video is AI-generated/fake and to be careful. "
            "Sound like a friend warning them. Use slang like 'bro', 'yo', 'fam'. "
            "Include 1-2 warning emojis. Reply with ONLY the message, no quotes, no explanation."
        )
        fallback_list = AI_RESULT_TEMPLATES
    else:
        prompt = (
            "Generate a short casual Instagram DM reply (1-2 sentences, max 25 words) "
            "telling the user their video appears to be a real authentic video, not AI. "
            "Sound friendly and reassuring, like a friend. Include 1-2 positive emojis. "
            "Reply with ONLY the message, no quotes, no explanation."
        )
        fallback_list = REAL_RESULT_TEMPLATES

    text = _call_llm(prompt, max_tokens=60)
    if not text:
        text = random.choice(fallback_list)
    return text + FEEDBACK_PROMPT


def get_error_message():
    """Message when video processing fails."""
    prompt = (
        "Generate a short casual apology message (1 sentence, max 15 words) "
        "saying you couldn't process the video and asking to try again. "
        "Use 1 emoji. Reply with ONLY the message, no quotes, no explanation."
    )
    text = _call_llm(prompt, max_tokens=30)
    return text or random.choice(ERROR_TEMPLATES)
