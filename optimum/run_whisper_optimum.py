import librosa
import argparse
from transformers import WhisperProcessor
from optimum.onnxruntime import ORTModelForSpeechSeq2Seq

parser = argparse.ArgumentParser(description="Test output of Whisper Model")
parser.add_argument("--model", type=str, required=True, help="Whisper model to use")
parser.add_argument("--audio", type=str, default=None,
    help="Path to audio file. If not provided, will use the test data from the config.",
)
args = parser.parse_args()
model = args.model
audio = args.audio
onnxruntime = args.onnxruntime

speech, _ = librosa.load(audio)

processor = WhisperProcessor.from_pretrained(model)
model = ORTModelForSpeechSeq2Seq.from_pretrained(model)

model.config.forced_decoder_ids = processor.get_decoder_prompt_ids(language = "en", task = "transcribe")
input_features = processor(speech, sampling_rate=16000, return_tensors="pt").input_features 
predicted_ids = model.generate(input_features, max_length=500)
print(predicted_ids.shape)
transcription = processor.batch_decode(predicted_ids)

print(processor.batch_decode(predicted_ids, skip_special_tokens = True))