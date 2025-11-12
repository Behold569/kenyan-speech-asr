# Whisper ASR for Kenyan Nonstandard Speech

[![Hugging Face Models](https://img.shields.io/badge/🤗-Hugging%20Face%20Models-yellow)](https://huggingface.co/ElizabethMwangi)
[![Demo Space](https://img.shields.io/badge/🎯-Live%20Demo-blue)](https://huggingface.co/spaces/ElizabethMwangi/whisper-kenyan-asr)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

Advanced speech recognition system fine-tuned specifically for Kenyan English and Swahili, optimized for individuals with speech impairments. This research explores the effectiveness of fine-tuning Whisper models for nonstandard speech patterns in the Kenyan context.

## Research Overview

This project addresses the challenge of speech recognition for individuals with speech impairments in Kenya, focusing on:

- **Language Adaptation**: Fine-tuning for Kenyan English and Swahili dialects
- **Speech Impairment Optimization**: Improved recognition for dysarthric and nonstandard speech
- **Context-Aware Processing**: Prompt tuning for medical and daily communication contexts
- **Accessibility Focus**: Developing tools for healthcare and education applications

##  Quick Start

### Live Demo
Try our interactive demo: [Hugging Face Space](https://huggingface.co/spaces/ElizabethMwangi/whisper-kenyan-asr)

### Using Fine-Tuned Models

```python
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import librosa

# Load fine-tuned model
model_name = "ElizabethMwangi/en_nonstandard_tune_whisper_large_4"
processor = WhisperProcessor.from_pretrained(model_name)
model = WhisperForConditionalGeneration.from_pretrained(model_name)

# Transcribe audio
audio, sr = librosa.load("audio.wav", sr=16000)
input_features = processor(audio, sampling_rate=16000, return_tensors="pt").input_features

# Generate transcription
predicted_ids = model.generate(input_features, language="en", task="transcribe")
transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
