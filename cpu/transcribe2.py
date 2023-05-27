import numpy as np
import onnxruntime as ort
from onnxruntime_extensions import get_library_path

def load_audio(audio_file):
    with open(audio_file, "rb") as f:
        audio = np.asarray(list(f.read()), dtype=np.uint8)
    return audio

def get_inputs(audio):
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
    return inputs

def run_inference(audio_file, model):
    audio = load_audio(audio_file)
    inputs = get_inputs(audio)

    sess_options = ort.SessionOptions()
    sess_options.register_custom_ops_library(get_library_path())
    sess = ort.InferenceSession(model, sess_options, providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    outputs = sess.run(None, inputs)[0]
    return outputs

def main():
    audio_file = "audio.mp3"
    model = "whisper-tiny-en-all-int8.onnx"
    outputs = run_inference(audio_file, model)
    print(outputs)

if __name__ == "__main__":
    main()
