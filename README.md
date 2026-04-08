# Brain Tumor Segmentation V1

Lightweight web service for brain tumor segmentation using a Python backend and a simple frontend.

## Overview

This repository provides:
- A Flask backend that accepts MRI uploads and returns segmentation results.
- A minimal frontend to upload images and view results.
- Pretrained model files and example testing data.

## Features

- Multi-sequence MRI upload (FLAIR, T1, T1CE, T2)
- Preprocessing and postprocessing pipelines
- Model inference and prediction output storage
- Docker support via `docker-compose.yml`

## Repository structure

- `backend/` — Flask app, models and utilities
  - `backend/app.py` — Flask application entry
  - `backend/routes/upload.py` — upload route and inference trigger
  - `backend/models/segmentation_model.py` — model wrapper
  - `backend/model_weights/` — put your `.pth` or model files here
  - `backend/predictions/` — inference outputs are written here
  - `backend/uploads/` — uploaded files
  - `backend/utils/` — preprocessing and postprocessing
- `frontend/` — static frontend (`index.html`, `css/`, `js/`)
- `nnU-net/` — experimental notebooks and alternate model artifacts
- `testing_data/` — example MRI `.nii.gz` files
- `docker-compose.yml`, `nginx.conf`

## Requirements

- Linux (development tested)
- Python 3.10+
- pip
- Recommended: GPU and matching CUDA/cuDNN if using large models

Install Python deps:
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r backend/requirements.txt
