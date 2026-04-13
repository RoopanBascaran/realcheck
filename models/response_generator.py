"""Generate natural, varied responses for the Instagram bot using HF Inference API.

Falls back to template responses if the LLM is unavailable, so the bot never breaks.
"""
import os
import random
import logging

logger = logging.getLogger(__name__)

HF_TOKEN = os.environ.get('HF_TOKEN')
# Models to try in order. Openly licensed models that work with HF free Inference API.
LLM_MODELS = [
    'HuggingFaceH4/zephyr-7b-beta',
    'mistralai/Mistral-7B-Instruct-v0.3',
    'google/gemma-2-2b-it',
]

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

MIXED_RESULT_TEMPLATES = [
    "yo heads up ⚠️ this video mixes real and AI footage together. some parts are real but there's fake stuff hidden in between. don't trust it fully!",
    "bro be careful 👀 this one's tricky — it's got real clips mixed with AI-generated parts. don't fall for it!",
    "this video is sneaky 🚨 parts of it are real but some sections are AI-made. they're mixing real and fake together to trick people!",
    "watch out fam ⚠️ this isn't fully real OR fully AI — it's a mix of both. someone blended real footage with AI content to make it look legit",
    "nah this is sus 🤔 the video switches between real and AI-generated parts. classic trick to fool people — stay sharp!",
    "careful with this one 👀 it's a hybrid — real video mixed with AI content. some parts are genuine but others are fake!",
]

ERROR_TEMPLATES = [
    "oops, something went wrong 😕 could you try sending the video again?",
    "hmm couldn't process that one 🤔 try again?",
    "ran into an issue with that video 😬 please try once more",
    "something's off on my end 🛠️ send it again please?",
]

FEEDBACK_PROMPT = "\n\nwhat do you think? is this fake? 👇"


def _call_llm(prompt, max_tokens=80):
    """Call HF Inference API. Tries multiple models. Returns text or None on failure."""
    if not HF_TOKEN:
        logger.warning('HF_TOKEN not set, using template fallback')
        return None

    try:
        from huggingface_hub import InferenceClient
    except Exception as e:
        logger.error(f'huggingface_hub import failed: {e}')
        return None

    for model in LLM_MODELS:
        try:
            client = InferenceClient(model=model, token=HF_TOKEN, timeout=10)
            response = client.chat_completion(
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=0.9,
            )
            text = response.choices[0].message.content.strip()
            # Strip surrounding quotes if LLM wrapped the response
            if text.startswith('"') and text.endswith('"'):
                text = text[1:-1]
            logger.info(f'LLM response from {model}: {text[:60]}')
            return text
        except Exception as e:
            logger.warning(f'LLM call to {model} failed: {type(e).__name__}: {e}')
            continue

    logger.warning('All LLM models failed, using template fallback')
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
    """Message with the classification result. `result` is 'AI-generated', 'Real', or 'Mixed'."""
    if result == 'Mixed':
        prompt = (
            "Generate a short casual Instagram DM reply (2 sentences, max 30 words) "
            "warning the user their video contains BOTH real and AI-generated parts mixed together. "
            "Explain that some parts are real but AI content is hidden in between to trick people. "
            "Sound like a friend warning them. Use slang like 'bro', 'yo', 'fam'. "
            "Include 1-2 warning emojis. Reply with ONLY the message, no quotes, no explanation."
        )
        fallback_list = MIXED_RESULT_TEMPLATES
    elif result == 'AI-generated':
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
