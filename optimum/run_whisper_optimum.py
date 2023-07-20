import librosa
import argparse
from transformers import WhisperProcessor
from optimum.onnxruntime import ORTModelForSpeechSeq2Seq

parser = argparse.ArgumentParser(description="Test output of Whisper Model")
parser.add_argument("--model", type=str, required=True, help="Whisper model to use e.g. optimum/whisper-tiny.en")
parser.add_argument("--audio", type=str, default=True, help="Path to audio file")
parser.add_argument("--language", type=str, default="en", help="Language to use for transcription")

args = parser.parse_args()
model = args.model
audio = args.audio
language = args.language

speech, _ = librosa.load(audio)

processor = WhisperProcessor.from_pretrained(model)
model = ORTModelForSpeechSeq2Seq.from_pretrained(model)

model.config.forced_decoder_ids = processor.get_decoder_prompt_ids(language = "hi", task = "transcribe")
input_features = processor(speech, sampling_rate=16000, return_tensors="pt").input_features 
predicted_ids = model.generate(input_features, max_length=448)
transcription = processor.batch_decode(predicted_ids)

print(processor.batch_decode(predicted_ids, skip_special_tokens = True))