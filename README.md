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

Or install from the saved environment [whisper.yml](./whisper.yml)

## PyTorch HuggingFace

Run from [pytorch](./pytorch)

## HuggingFace Optimum

Run from [optimum](./optimum)

## Generate all in one model with Olive

Configs for each of the model variants can be found in [./models](models)

```bash
cd models/whisper-tiny
python prepare_whisper_configs.py --model_name openai/whisper-tiny.en
python -m olive.workflows.run --config whisper_cpu_int8.json --setup
python -m olive.workflows.run --config whisper_cpu_int8.json
```

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
