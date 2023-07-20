# Config for Hindi transcription

* Clone the latest Olive and install from source
* Install ORT nightly as per the Olive README
* Install onnxruntime-extensions 0.8.0
* Run python prepare_whisper_configs.py --model_name vasista22/whisper-hindi-small --multilingual
* Run python -m olive.workflows.run --config whisper_cpu_fp32.json --setup
* Run python -m olive.workflows.run --config whisper_cpu_fp32.json
* If you are testing with .wav files, edit code/whisper_dataset.py to replace .mp3 with .wav (will need to go * back to .mp3 if you are re-generating the model)
* Run python test_transcription.py --config whisper_cpu_fp32.json --audio_path "cricketLong-Trimmed (1).wav"