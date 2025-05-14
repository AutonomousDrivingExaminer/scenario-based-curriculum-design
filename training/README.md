# ADEX Training

## Local Development
To test and debug locally, you need to install the dependencies locally from the`training/requirements.txt` 
file and then install the `adex` package in editable mode.
To do so, run the following commands from the project directory:
```bash
pip install -r training/requirements.txt
pip install -e .
```
To start a local training run, start carla and then start the training script:
```bash
./scripts/start_carla.sh
python training/train.py
```

## Local Development with Docker

