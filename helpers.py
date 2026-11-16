import os
import sys
import json
import time
import torch
import keyboard
import numpy as np
import sounddevice as sd
import soundfile as sf
from huggingface_hub import snapshot_download
import google.generativeai as genai
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

# ---------------------------------------------------------
# 1. PUSH-TO-TALK (PTT) AUDIO CAPTURE (16kHz Mono WAV)
# ---------------------------------------------------------
def record_audio_ptt(output_wav_path, samplerate=16000):
    print("\n[MIC] Hold SPACEBAR to speak your complaint ( or press ESC to exit )...")
    
    while True:
        if keyboard.is_pressed("esc"):
            return None
        if keyboard.is_pressed("space"):
            break
        time.sleep(0.05)

    print("[MIC] Recording... (Keep holding SPACEBAR)")
    audio_chunks = []

    def audio_callback(indata, frames, time_info, status):
        if status:
            print(f"[MIC Warning] {status}", file=sys.stderr)
        audio_chunks.append(indata.copy())

    with sd.InputStream(samplerate=samplerate, channels=1, dtype="float32", callback=audio_callback):
        while keyboard.is_pressed("space"):
            sd.sleep(50)

    print("[MIC] Recording stopped.")

    if not audio_chunks:
        print("[MIC] No audio captured.")
        return False

    audio_data = np.concatenate(audio_chunks, axis=0)
    duration = len(audio_data) / samplerate

    if duration < 0.5:
        print("[MIC] Audio too short (< 0.5s). Please hold SPACEBAR while speaking.")
        return False

    sf.write(output_wav_path, audio_data, samplerate)
    print(f"[FILE] Saved 16kHz Mono WAV ({duration:.1f}s) -> {output_wav_path}")
    return True

# ---------------------------------------------------------
# 2. BODHAN AI INDIC-TRANSCRIBE-FLEX (1.2B) LOADER & ASR
# ---------------------------------------------------------
def load_asr_model():
    hf_token = os.getenv("HF_TOKEN")
    if not hf_token or "your_huggingface" in hf_token:
        raise ValueError("Valid HF_TOKEN is missing in your .env file!")

    print("\n[GPU] Downloading/Verifying 'bodhan-ai/indic-transcribe-flex'...")
    print("      (First run downloads ~4.9 GB to your Hugging Face cache)")
    
    model_dir = snapshot_download(
        repo_id="bodhan-ai/indic-transcribe-flex",
        token=hf_token
    )

    # Add repository directory to sys.path so custom model classes load cleanly
    if model_dir not in sys.path:
        sys.path.insert(0, model_dir)

    from indic_transcribe import IndicTranscribe

    print(f"[GPU] Loading Indic-Transcribe-Flex onto {torch.cuda.get_device_name(0)}...")
    asr = IndicTranscribe.from_pretrained(model_dir)
    print("[GPU] ASR Model ready in VRAM!")
    return asr

def transcribe_audio(asr, audio_path):
    """
    Transcribes multilingual Indian audio (Hindi, Bengali, Bhojpuri, English, etc.)
    using mode='mixed' so spoken numbers/digits are automatically normalized (ITN).
    """
    info = sf.info(audio_path)
    duration = info.duration

    print(f"[ASR] Transcribing ({duration:.1f}s audio) in 'mixed' ITN mode...")

    # Single-pass supports up to 30-45s and returns Language ID (LID)
    if duration <= 30.0:
        text, lid = asr.transcribe(audio_path, mode="mixed", return_lid=True)
        detected_lang = lid.get("lang", "unknown") if isinstance(lid, dict) else str(lid)
    else:
        # Automatic chunked decoding for recordings longer than 30 seconds
        from long_form import transcribe_long
        
        # FIX: 'lang' is a required positional argument. 
        # Pass None to trigger the model's whole-file auto-detection.
        text = transcribe_long(asr, audio_path, None, mode="mixed")
        detected_lang = "auto-chunked"

    print(f"[ASR] Detected Language: {detected_lang}")
    print(f"[ASR] Raw Transcript: {text}")
    return text, detected_lang

# ---------------------------------------------------------
# 3. GEMINI FLASH TRANSLATION & JSON ENTITY EXTRACTION
# ---------------------------------------------------------
def translate_and_extract_json(transcript_text, detected_lang="unknown"):
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key or "your_gemini" in api_key:
        raise ValueError("Valid GEMINI_API_KEY is missing in your .env file!")

    genai.configure(api_key=api_key)
    
    # Uses Gemini Flash-Lite / Flash for fast, free-tier structured extraction
    model_name = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")
    model = genai.GenerativeModel(model_name)

    prompt = f"""
    You are an expert multilingual cybercrime and financial complaint intake parser for India.
    Analyze the following speech-to-text transcript (detected language code: {detected_lang}).
    The speaker may use Hindi, Bengali, Bhojpuri, English, or code-mixed speech.

    Tasks:
    1. Translate the full complaint accurately into clear English.
    2. Convert any remaining spoken number words into standard numerical digits.
    3. Extract the complaint entities into the exact JSON schema below. If a field is not mentioned, set its value to null.

    Required JSON Schema:
    {{
        "translated_english_text": "Full accurate English translation of the user's speech",
        "detected_language": "{detected_lang}",
        "extracted_entities": {{
            "name": null,
            "phone_number": null,
            "bank_account_number": null,
            "bank_name": null,
            "district": null,
            "police_station": null,
            "pincode": null,
            "state": null,
            "fraud_amount": null,
            "fraud_type": null
        }}
    }}

    Return ONLY valid raw JSON without Markdown code blocks.

    Transcript:
    "{transcript_text}"
    """

    print(f"[LLM] Translating and extracting entities via {model_name}...")
    response = model.generate_content(prompt)
    raw_output = response.text.strip()

    # Strip accidental markdown fences if present
    if raw_output.startswith("```"):
        raw_output = raw_output.strip("`")
        if raw_output.lower().startswith("json"):
            raw_output = raw_output[4:].strip()

    return json.loads(raw_output)