# Respiratory Sound Analysis using Machine Learning

This project implements a machine learning solution for analyzing respiratory sounds to detect various respiratory conditions such as asthma, COPD, and normal breathing patterns.

## Features
- Audio signal processing of respiratory sounds
- Feature extraction from breathing sounds
- Machine learning model for respiratory condition classification
- Support for multiple respiratory conditions (asthma, COPD, normal)

## Setup Instructions

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Prepare your data:
   - Place your respiratory sound recordings in the `data/raw` directory
   - Organize recordings by condition (e.g., data/raw/asthma/, data/raw/copd/, data/raw/normal/)
   - Supported audio formats: .wav

4. Run the training:
```bash
python respiratory_analysis.py --mode train
```

5. For prediction on new audio:
```bash
python respiratory_analysis.py --mode predict --input path_to_audio_file.wav
```

## Project Structure
- `respiratory_analysis.py`: Main script for training and prediction
- `data/`: Directory for storing audio files
  - `raw/`: Raw audio files organized by condition
  - `processed/`: Processed features and model files
- `requirements.txt`: Python dependencies
- `README.md`: Project documentation

## Model Details
The system uses a deep learning approach with the following pipeline:
1. Audio preprocessing (noise reduction, normalization)
2. Feature extraction (MFCC, spectral features)
3. Neural network classification
4. Output prediction with confidence scores 
