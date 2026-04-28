# Behavior Detection System

A lightweight behavior detection project for virtual class proctoring.  
This repository includes the training notebooks for two model iterations and a small Flask web app for running the final model on webcam or uploaded video input.

## What This Project Does

- Classifies behavior into 7 classes:
  - `distracted`
  - `fatigue`
  - `focused`
  - `raise_hand`
  - `sleeping`
  - `using_smartphone`
  - `writing_reading`
- Uses a MobileNetV2-based final model for inference
- Provides a browser UI for:
  - live webcam monitoring
  - uploaded video monitoring
  - real-time class scores and status updates

## Repository Layout

```text
CVProject/
  Code/
    app.py                          Flask app
    templates/index.html            Web interface
    notebooks/                      Training iterations
    final_proctor_model.h5          Final saved model
    best_final_proctor_model_finetune.h5
    Dockerfile
    requirements.txt
```

## Model Iterations

- `mobilenet.ipynb`
  - MobileNetV2 transfer learning
- `finetune_mobilenet.ipynb`
  - Domain-adapted fine-tuning with webcam-oriented evaluation

## Run With Docker

Ensure Docker Engine is running.

From `CVProject/Code`:

```bash
cd CVProject/Code
docker build --no-cache -t exam-proctor .
docker run --rm -p 5000:5000 exam-proctor
```

Then open:

```text
http://127.0.0.1:5000
```

To stop the container, press `Ctrl+C`.

## Run Locally

From `CVProject/Code`:

```bash
pip install -r requirements.txt
python app.py
```

Then open:

```text
http://127.0.0.1:5000
```

## Web App Features

- Start live webcam monitoring from the browser
- Upload a video file for analysis
- View the current raw predicted class
- View a smoothed stable prediction
- See confidence scores for all classes
- See whether the detected behavior is treated as normal or suspicious
