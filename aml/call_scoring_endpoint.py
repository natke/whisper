import urllib.request
import json
import numpy
import os

# Request data goes here
# The example below assumes JSON formatting which may be updated
# depending on the format your endpoint expects.
# More information can be found here:
# https://docs.microsoft.com/azure/machine-learning/how-to-deploy-advanced-entry-script
audio_file = "audio.mp3"

with open(audio_file, "rb") as f:
    audio = numpy.asarray(list(f.read()), dtype=numpy.uint8)

data = {"audio": audio.tolist()}

body = str.encode(json.dumps(data))

url = 'https://whisper-onnx.australiaeast.inference.ml.azure.com/score'
api_key = os.getenv("AML_TOKEN")
if not api_key:
    raise Exception("A key should be provided to invoke the endpoint")

# The azureml-model-deployment header will force the request to go to a specific deployment.
# Remove this header to have the request observe the endpoint traffic rules
headers = {'Content-Type':'application/json', 'Authorization':('Bearer '+ api_key), 'azureml-model-deployment': 'onnxruntime' }

req = urllib.request.Request(url, body, headers)

try:
    response = urllib.request.urlopen(req)

    result = response.read()
    print(result)
except urllib.error.HTTPError as error:
    print("The request failed with status code: " + str(error.code))

    # Print the headers - they include the requert ID and the timestamp, which are useful for debugging the failure
    print(error.info())
    print(error.read().decode("utf8", 'ignore'))