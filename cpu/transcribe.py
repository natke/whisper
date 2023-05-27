import numpy as np
import onnxruntime
from onnxruntime_extensions import get_library_path

audio_file = "audio.mp3"
model = "whisper-tiny-en-all-int8.onnx"
with open(audio_file, "rb") as f:
    audio = np.asarray(list(f.read()), dtype=np.uint8)

inputs = {
    "audio_stream": np.array([audio]),
    "max_length": np.array([30], dtype=np.int32),
    "min_length": np.array([1], dtype=np.int32),
    "num_beams": np.array([2], dtype=np.int32),
    "num_return_sequences": np.array([1], dtype=np.int32),
    "length_penalty": np.array([1.0], dtype=np.float32),
    "repetition_penalty": np.array([1.0], dtype=np.float32),
    "attention_mask": np.zeros((1, 80, 3000), dtype=np.int32),
}

options = onnxruntime.SessionOptions()
options.register_custom_ops_library(get_library_path())
session = onnxruntime.InferenceSession(model, options, providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
outputs = session.run(None, inputs)[0]
print(outputs)
