import os
import logging
import json
import numpy as np
import onnxruntime
from onnxruntime_extensions import get_library_path

# Perform the one-off intialization for the prediction. The init code is run once when the endpoint is setup.
def init():
    logging.info("Running init() ...")

    global session, model, processor

    model_name = "whisper-tiny-en-e2e-int8"

    # use AZUREML_MODEL_DIR to get your deployed model(s). If multiple models are deployed, 
    # model_path = os.path.join(os.getenv('AZUREML_MODEL_DIR'), '$MODEL_NAME/$VERSION/$MODEL_FILE_NAME')
    model_dir = os.getenv('AZUREML_MODEL_DIR')

    logging.info("Loading model ...")

    if model_dir == None:
        model_dir = "./"
    model_path = os.path.join(model_dir, model_name + ".onnx")

    # Create an ONNX Runtime session to run the ONNX model
    options = onnxruntime.SessionOptions()
    options.register_custom_ops_library(get_library_path())
    session = onnxruntime.InferenceSession(model_path, options, providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    
# Run the ONNX model with ONNX Runtime
def run(payload):

    data = json.loads(payload)

    inputs = {
        "audio_stream": np.array([data["audio"]], dtype=np.uint8),
        "max_length": np.array([500], dtype=np.int32),
        "min_length": np.array([1], dtype=np.int32),
        "num_beams": np.array([2], dtype=np.int32),
        "num_return_sequences": np.array([1], dtype=np.int32),
        "length_penalty": np.array([1.0], dtype=np.float32),
        "repetition_penalty": np.array([1.0], dtype=np.float32),
        "attention_mask": np.zeros((1, 80, 3000), dtype=np.int32),
    }

    # TODO Work out why the output shape is not correct

    outputs = session.run(None, inputs)

    results = {}
    results["transcription"] = outputs[0][0][0]

    return results


if __name__ == '__main__':
    init()

    audio_file = "audio.mp3"

    with open(audio_file, "rb") as f:
        audio = np.asarray(list(f.read()), dtype=np.uint8)

    payload = json.dumps({"audio": audio.tolist()})

    print(run(payload))


