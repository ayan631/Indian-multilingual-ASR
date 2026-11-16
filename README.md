# 🎙️ Indian Multilingual ASR & Complaint Intake System

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-CUDA%20Enabled-EE4C2C.svg)](https://pytorch.org/)
[![Model](https://img.shields.io/badge/ASR-Bodhan--AI%2Findic--transcribe--flex%20(1.2B)-orange.svg)](https://huggingface.co/bodhan-ai/indic-transcribe-flex)
[![LLM](https://img.shields.io/badge/Extraction-Google%20Gemini-4285F4.svg)](https://ai.google.dev/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An end-to-end, GPU-accelerated speech-to-structured-data intake pipeline tailored for Indian multilingual speech (Hindi, Bengali, Bhojpuri, English, and code-mixed speech). 

The system captures audio via Push-To-Talk (PTT), runs low-latency GPU speech recognition with native Inverse Text Normalization (ITN), detects the spoken Indian language, translates the content to English, and uses Google Gemini to extract structured cybercrime and financial fraud complaint entities into JSON format.

---

## 🚀 Key Features

- **Push-to-Talk (PTT) Audio Capture**: Low-latency 16 kHz Mono recording triggered effortlessly by holding the `SPACEBAR`.
- **State-of-the-Art Indian Multilingual ASR**: Powered by **Bodhan AI's `indic-transcribe-flex` (1.2B parameters)** running directly on CUDA VRAM.
- **Automatic Language Identification (LID)**: Seamlessly handles regional languages and code-mixed Indian dialects (Hinglish, Benglish, etc.).
- **Native Inverse Text Normalization (ITN)**: Automatically formats spoken numbers into numerical digits (crucial for phone numbers, account numbers, and transaction amounts).
- **Long-Form Audio Support**: Handles single utterances (≤30s) as well as automatic chunked decoding for longer recordings.
- **LLM-Powered Translation & Entity Extraction**: Leverages Google Gemini to translate dialect speech and output clean, validated JSON schemas with key crime/incident entities.

---

## 🔄 Pipeline Architecture

```
[ Push-To-Talk Mic (16kHz WAV) ]
                │
                ▼
[ Bodhan AI Indic-Transcribe-Flex (1.2B) on GPU ]
   ├── Multilingual ASR (Hindi, Bengali, Bhojpuri, English, etc.)
   ├── Language Identification (LID)
   └── Inverse Text Normalization (ITN: words -> digits)
                │
                ▼
[ Google Gemini LLM Processing ]
   ├── English Translation
   └── Structured Entity Extraction
                │
                ▼
┌─────────────────────────────────────────────────────────┐
│                    Saved Artifacts                      │
├──────────────────────────┬──────────────────────────────┤
│ `input_audio/`           │ Raw 16kHz WAV recording      │
│ `transcribed_text/`      │ Native language transcript   │
│ `translated_text/`       │ Pure English translation     │
│ `translated_json/`       │ Structured Complaint JSON    │
└──────────────────────────┴──────────────────────────────┘
```

---

## 📋 Extracted JSON Schema

The pipeline automatically parses raw spoken complaints into standardized records:

```json
{
  "translated_english_text": "My name is Rajesh Sharma. On November 5th, someone claiming to be from SBI called my phone number 9876543210 and stole 45000 rupees from my account 123456789012 under the pretext of KYC verification. I want to file a complaint at Cyber Crime Police Station, Patna.",
  "detected_language": "hi",
  "extracted_entities": {
    "name": "Rajesh Sharma",
    "phone_number": "9876543210",
    "bank_account_number": "123456789012",
    "bank_name": "State Bank of India (SBI)",
    "district": "Patna",
    "police_station": "Cyber Crime Police Station",
    "pincode": null,
    "state": "Bihar",
    "fraud_amount": "45000",
    "fraud_type": "KYC Fraud / Phishing"
  }
}
```

---

## 💻 Hardware & System Requirements

- **Operating System**: Windows 10/11 or Linux
- **GPU**: NVIDIA GPU with at least 8 GB VRAM (Tested and optimized on **NVIDIA GeForce RTX 3060 12GB**)
- **CUDA**: CUDA 11.8+ or 12.1+ supported by PyTorch
- **Python**: 3.10 or 3.11 recommended
- **Microphone**: Working audio input device

---

## ⚙️ Installation & Setup

### 1. Clone the Repository

```bash
git clone https://github.com/ayan631/Indian-multilingual-ASR.git
cd Indian-multilingual-ASR
```

### 2. Create a Virtual Environment

```bash
# Windows (PowerShell)
python -m venv venv
.\venv\Scripts\Activate.ps1

# Linux / macOS
python3 -m venv venv
source venv/bin/activate
```

### 3. Install PyTorch with CUDA

Install PyTorch built with CUDA support for your system before installing other dependencies:

```bash
# Example for CUDA 12.1:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

### 5. Configure API Keys (`.env`)

Copy `.env.example` to create your own `.env` file:

```bash
copy .env.example .env    # Windows
cp .env.example .env      # Linux
```

Open `.env` and fill in your credentials:

```ini
# Hugging Face Token (required to download bodhan-ai/indic-transcribe-flex)
HF_TOKEN=hf_your_huggingface_token_here

# Google Gemini API Key (required for translation and entity extraction)
GEMINI_API_KEY=your_gemini_api_key_here

# Optional: Override Gemini Model (default: gemini-1.5-flash / gemini-3.8-flash)
GEMINI_MODEL=gemini-1.5-flash
```

> **Note**: Obtain your Hugging Face access token at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) and ensure you have accepted the model terms if prompted. Get your Gemini API key from [Google AI Studio](https://aistudio.google.com/).

---

## 🎯 Usage

Run the main intake pipeline:

```bash
python main.py
```

### Interactive Controls:
1. **Hold `SPACEBAR`**: Speak your complaint into the microphone.
2. **Release `SPACEBAR`**: Recording stops and the pipeline immediately starts GPU transcription and Gemini entity extraction.
3. **Press `ESC`**: Gracefully exits the application.

All outputs are automatically timestamped and categorized into their respective directories:
- `input_audio/audio_YYYYMMDD_HHMMSS.wav`
- `transcribed_text/transcript_YYYYMMDD_HHMMSS.txt`
- `translated_text/translated_YYYYMMDD_HHMMSS.txt`
- `translated_json/complaint_YYYYMMDD_HHMMSS.json`

---

## 📂 Project Structure

```
Indian-multilingual-ASR/
├── input_audio/            # Recorded 16kHz mono WAV files
├── transcribed_text/       # Transcripts from Bodhan AI ASR with ITN
├── translated_text/        # Pure English translation text
├── translated_json/        # Structured JSON records with extracted entities
├── .env.example            # Template for environment variables
├── .gitignore              # Git ignore rules for cache, venv, and secrets
├── helpers.py              # Audio capture, ASR model loader, and Gemini parser
├── main.py                 # Main event loop and pipeline orchestration
├── requirements.txt        # Python package dependencies
└── README.md               # Project documentation
```

---

## 🛡️ Privacy & Security Note

Never commit your `.env` file to version control. The repository includes a preconfigured `.gitignore` that prevents your `.env` and local virtual environment files from being published.

---

## 🤝 Contributing

Contributions, bug reports, and feature requests are welcome! Feel free to open an issue or submit a pull request.

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).
