import os
import numpy as np
import json
import onnxruntime
from onnxruntime_extensions import get_library_path

def init():
    global session

    model_name = "whisper-tiny-en-e2e-int8"

    # use AZUREML_MODEL_DIR to get your deployed model(s). If multiple models are deployed, 
    # model_path = os.path.join(os.getenv('AZUREML_MODEL_DIR'), '$MODEL_NAME/$VERSION/$MODEL_FILE_NAME')
    model_dir = os.getenv('AZUREML_MODEL_DIR')
    if model_dir == None:
        model_dir = "./"
    model_path = os.path.join(model_dir, model_name + ".onnx")

    # Create an ONNX Runtime session to run the ONNX model
    sess_options = onnxruntime.SessionOptions()
    sess_options.register_custom_ops_library(get_library_path())

    session = onnxruntime.InferenceSession(model_path, sess_options, providers=["CPUExecutionProvider"])  

def load_audio(audio_file):
    with open(audio_file, 'rb') as f:
        audio = np.asarray(list(f.read()), dtype=np.uint8)
    return audio

def get_inputs(audio):
    inputs = {
        'audio_stream': np.expand_dims(audio, axis=0),
        'max_length': np.array([30], dtype=np.int32),
        'min_length': np.array([1], dtype=np.int32),
        'num_beams': np.array([2], dtype=np.int32),
        'num_return_sequences': np.array([1], dtype=np.int32),
        'length_penalty': np.array([1.0], dtype=np.float32),
        'repetition_penalty': np.array([1.0], dtype=np.float32),
        'attention_mask': np.zeros((1, 80, 3000), dtype=np.int32),
    }
    return inputs

def run(inputs):

    #inputs = json.loads(raw_data)

        inputs = {
        'audio_stream': np.expand_dims(audio, axis=0),
        'max_length': np.array([30], dtype=np.int32),
        'min_length': np.array([1], dtype=np.int32),
        'num_beams': np.array([2], dtype=np.int32),
        'num_return_sequences': np.array([1], dtype=np.int32),
        'length_penalty': np.array([1.0], dtype=np.float32),
        'repetition_penalty': np.array([1.0], dtype=np.float32),
        'attention_mask': np.zeros((1, 80, 3000), dtype=np.int32),
    }

    outputs = session.run(None, inputs)[0]
    return outputs

if __name__ == "__main__":
    init()

    audio = load_audio("teapot.mp3")
    inputs = get_inputs(audio)

    print(run(inputs))
