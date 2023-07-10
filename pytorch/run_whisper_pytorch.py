import librosa
import argparse
from transformers import WhisperProcessor, WhisperForConditionalGeneration

parser = argparse.ArgumentParser(description="Test output of Whisper Model")
parser.add_argument("--model", type=str, required=True, help="Whisper model to use")
parser.add_argument("--audio", type=str, required=True, help="Path to audio file.")

args = parser.parse_args()
model = args.model
audio = args.audio

speech, _ = librosa.load(audio)

processor = WhisperProcessor.from_pretrained(model)
model = WhisperForConditionalGeneration.from_pretrained(model)

model.config.forced_decoder_ids = processor.get_decoder_prompt_ids(language = "en", task = "transcribe")
input_features = processor(speech, sampling_rate=16000, return_tensors="pt").input_features 
predicted_ids = model.generate(input_features, max_length=500)
transcription = processor.batch_decode(predicted_ids)

print(processor.batch_decode(predicted_ids, skip_special_tokens = True))