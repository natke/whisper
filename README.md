# Whisper model generation and code samples

## Environment

Create a conda environment with the following packages.

```bash
conda create -n whisper
pip install onnx
pip install torch
pip install transformers
pip install optimum[onnx]
pip install onnxruntime
pip install onnxruntime-extensions
```

### Use nightly packages

```bash
pip install ort-nightly==1.16.0.dev20230701001 --extra-index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/ORT-Nightly/pypi/simple/
```

### Install from conda spec

Or install from the saved environment [whisper.yml](./whisper.yml)

## PyTorch HuggingFace

Run from [pytorch](./pytorch)

## HuggingFace Optimum

Run from [optimum](./optimum)

## Generate all in one model with Olive

Install Olive from source

```bash
python -m pip install git+https://github.com/microsoft/Olive.git
```

Configs for each of the model variants can be found in [./models](models)

```bash
cd models/whisper-tiny
python prepare_whisper_configs.py --model_name openai/whisper-tiny.en
python -m olive.workflows.run --config whisper_cpu_int8.json --setup
python -m olive.workflows.run --config whisper_cpu_int8.json
```

In the models folder you will find `whisper_cpu_int8.zip`, which contains a folder called `CandidateModels/cpu-cpu/BestCandidateModel_1`. The model is in this folder. Copy the model into your application folder.


## Run with ONNX Runtime

```bash
cd cpu
python transcribe.py
```

## Export model without Olive

```bash
python -m onnxruntime.transformers.models.whisper.convert_to_onnx -m openai/whisper-base.en --output whisper -e
```

### Run script to generate composite model

```bash
curl https://raw.githubusercontent.com/microsoft/onnxruntime-extensions/main/test/data/1272-141231-0002.mp3 > 1272-141231-0002.mp3 
curl https://raw.githubusercontent.com/microsoft/onnxruntime-extensions/main/tutorials/whisper_e2e.py > whisper_e2e.py
python whisper_e2e.py -a 1272-141231-0002.mp3 -m whisper/openai/whisper-base.en_beamsearch.onnx
```

Produces

```bash
 whisper-base.en_all.onnx.data 
```
