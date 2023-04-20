# Whisper sample to support blog

## Pre-requisites

- [] Explicitly install onnxruntime-extensions dependencies manually
- [] Clone the entire onnxruntime repo to export the whisper models
- [] Run export script before whisper e2e script
- [] whisper_e2e.py hard codes whisper-base.en
- [] Manually download test audio file
- [] Use ONNX Runtime nightly and ONNX Runtime extensions nightly

## Issues that need to be fixed

- [] Whisper medium does not export
- [] Export script gives a false error

   ```bash
     PyTorch and OnnxRuntime results max difference = True
     PyTorch and OnnxRuntime results are NOT close
   ```

- [] Export script errors if you don't provide parameters (should give usage)

## Model preparation

### Install dependencies

```bash
conda create -n whisper
pip install onnx
pip install torch
pip install transformers
pip install flatbuffers
pip install coloredlogs
pip install --index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/ORT-Nightly/pypi/simple/ ort-nightly==1.15.dev20230410004
pip install --index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/ORT-Nightly/pypi/simple/ onnxruntime-extensions
```

### Clone ONNX Runtime repo

```bash
git clone git@github.com:microsoft/onnxruntime.git
```

### Run whisper export script

```bash
cd onnxruntime/onnxruntime/python/tools/transformers/models/whisper
python convert_to_onnx.py -m openai/whisper-base.en --output whisper -e
```

### Run script to generate composite model

```bash
cd whisper
mkdir -p test/data
cd test/data
curl https://raw.githubusercontent.com/microsoft/onnxruntime-extensions/main/test/data/1272-141231-0002.mp3 > 1272-141231-0002.mp3 
cd ../openai
curl https://raw.githubusercontent.com/microsoft/onnxruntime-extensions/main/tutorials/whisper_e2e.py > whisper_e2e.py
python whisper_e2e.py
```

### Prepare with Olive

## Model targets


|Size|Parameters|English-only|Multilingual|
|----|----------|------------|------------|
|tiny|39 M|✓|	✓|
|base|74 M|✓|	✓|
|small|244M|	✓|	✓|
|medium|769 M|	✓|	✓|
|large|1550 M|	x|	✓|
|large-v2|1550 M|	x|	✓|



## Resources

https://blog.deepgram.com/benchmarking-top-open-source-speech-models/

