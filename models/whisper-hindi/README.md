# Config for Hindi transcription

## Create environment and install dependencies

1. Create a new conda environment

   ```bash
   conda create -n whisper Python=3.9
   conda activate whisper
   ```

2. Clone the latest Olive and install from source

   ```bash
   git clone git@github.com:microsoft/Olive.git
   cd Olive
   pip install .
   ```

3. Change into the examples/whisper directory

   ```bash
   cd examples/whisper
   ```

4. Install the dependencies specified by Olive

   ```bash
   python -m pip install -r requirements.txt
   ```

5. Install ORT nightly as per the Olive README

   ```bash
   python -m pip uninstall -y onnxruntime ort-nightly
   python -m pip install ort-nightly --index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/ORT-Nightly/pypi/simple/
   ```

## Prepare model

1. (Optional) Convert OpenAI model format to HuggingFace

   If you fine-tuned the OpenAI whisper model, then convert the weights to HuggingFace format.

   ```bash
   python -m transformers.models.whisper.convert_openai_to_hf --checkpoint_path <<location of your OpenAI whisper weights file .pt>> --pytorch_dump_folder_path data/whisper-hindi
   ```

   Then, save the config and processor from the original model variant.

   ```python
   from transformers import AutoProcessor, AutoConfig

   # original model. Replace this with the whisper variant you are using 
   original_model_name = "openai/whisper-medium"
   
   # load config, processor
   config = AutoConfig.from_pretrained(original_model_name)
   processor = AutoProcessor.from_pretrained(original_model_name)

   # path to save the config and processor 
   # same path where the model weights are saved
   model_path = "data/whisper-hindi"

   # save config, processor
   config.save_pretrained(model_path)
   processor.save_pretrained(model_path)
   ```

3. Prepare the configs for Olive processing

   Once you have a model in HuggingFace format (local or hosted on the HuggingFace model hub), then prepare the configs for Olive processing.

   ```bash
   python prepare_whisper_configs.py --model_name data/whisper-small --multilingual
   ```

   For other models, replace `data/whisper-small` with the path to a model in HuggingFace format. For example `vasista22/whisper-hindi-small` which is a fine-tuned Hindi model from the hub.

## Generate model and test transcription

1. Setup the model creation

```bash
python -m olive.workflows.run --config whisper_cpu_fp32.json --setup
```

2. Generate the model

```bash
python -m olive.workflows.run --config whisper_cpu_fp32.json
```

3. Test transcription

```bash
python test_transcription.py --config whisper_cpu_fp32.json --audio_path "cricketLong-Trimmed (1).wav" --language hi
```