import os
import json
from datetime import datetime
from helpers import (
    record_audio_ptt,
    load_asr_model,
    transcribe_audio,
    translate_and_extract_json
)

# 1. Ensure all pipeline directories exist
DIRS = {
    "audio": "input_audio",
    "transcript": "transcribed_text",
    "translation": "translated_text",
    "json": "translated_json"
}

for folder in DIRS.values():
    os.makedirs(folder, exist_ok=True)

def main():
    print("=" * 60)
    print("  MULTILINGUAL COMPLAINT INTAKE SYSTEM (RTX 3060 GPU)")
    print("=" * 60)

    # Load the 1.2B Bodhan AI model once into GPU VRAM at startup
    asr_model = load_asr_model()

    while True:
        # Generate unified timestamp for this PTT cycle
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        audio_path = os.path.join(DIRS["audio"], f"audio_{timestamp}.wav")
        transcript_path = os.path.join(DIRS["transcript"], f"transcript_{timestamp}.txt")
        translation_path = os.path.join(DIRS["translation"], f"translated_{timestamp}.txt")
        json_path = os.path.join(DIRS["json"], f"complaint_{timestamp}.json")

        # Step 1: Record Audio via Push-To-Talk
        status = record_audio_ptt(audio_path)
        if status is None:
            print("\n[EXIT] Exiting application. Goodbye!")
            break
        if status is False:
            continue

        try:
            # Step 2: Multilingual Transcription + Native ITN on GPU
            transcript_text, detected_lang = transcribe_audio(asr_model, audio_path)

            with open(transcript_path, "w", encoding="utf-8") as f:
                f.write(transcript_text)
            print(f"[FILE] Saved Transcript -> {transcript_path}")

            # Step 3: Gemini Translation & Structured Entity Extraction
            result_data = translate_and_extract_json(transcript_text, detected_lang)

            # Save pure English translation text to maintain baseline pipeline parity
            english_text = result_data.get("translated_english_text", "")
            with open(translation_path, "w", encoding="utf-8") as f:
                f.write(english_text)
            print(f"[FILE] Saved English Translation -> {translation_path}")

            # Save structured JSON complaint output
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(result_data, f, indent=4, ensure_ascii=False)
            print(f"[FILE] Saved Extracted JSON -> {json_path}")

            print("\n--- EXTRACTED COMPLAINT JSON ---")
            print(json.dumps(result_data, indent=2, ensure_ascii=False))
            print("--------------------------------")

        except Exception as e:
            print(f"\n[ERROR] Pipeline failed during processing: {e}")

if __name__ == "__main__":
    main()