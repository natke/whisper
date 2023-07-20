# Config for Hindi transcription

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

3. Install ORT nightly as per the Olive README

```bash
python -m pip uninstall -y onnxruntime ort-nightly
python -m pip install ort-nightly --index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/ORT-Nightly/pypi/simple/
```

4. Install onnxruntime-extensions 0.8.0

```bash
pip install onnxruntime-extensions
```

5. Change into the examples/whisper directory

```bash
cd examples/whisper
```

6. Prepare the olive configs

```bash
python prepare_whisper_configs.py --model_name vasista22/whisper-hindi-small --multilingual
```

7. Setup the model creation

```bash
python -m olive.workflows.run --config whisper_cpu_fp32.json --setup
```

8. Generate the model

```bash
python -m olive.workflows.run --config whisper_cpu_fp32.json
```

9. Test trancription

```bash
python test_transcription.py --config whisper_cpu_fp32.json --audio_path "cricketLong-Trimmed (1).wav" --language hi
```
