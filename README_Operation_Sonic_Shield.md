 Operation Sonic Shield  
## Acoustic Classification of Brushless DC Motor Failures Using AI

A graduation project for detecting Brushless DC motor conditions from audio using machine learning experiments, AST-style model exploration, and a lightweight CNN model with spectrogram preprocessing.

---

## Project Idea

Electric motors are used in many industrial systems, and failures can cause downtime, repair cost, and safety risks.

This project explores a non-contact fault detection approach: using only the sound of a Brushless DC motor to classify its operating condition.

The model predicts one of three classes:

- `Good`
- `Broken`
- `Heavy Load`

---

## Problem Statement

Traditional motor fault detection often depends on physical sensors, manual inspection, or expensive monitoring systems.

The goal of this project is to test whether motor sound can be used as a simpler and cheaper signal for AI-based motor condition classification.

---

## Dataset

The project uses the **IDMT-ISA Electric Engine** dataset.

The dataset contains recordings from three similar electric engine units:

```text
2ACT Motor Brushless DC 42BLF01
4000 RPM
24VDC
```

### Classes

| Class | Description |
|---|---|
| `Good` | Normal motor operation |
| `Broken` | Faulty motor condition |
| `Heavy Load` | Motor running under increased load |

### Dataset Details

| Property | Value |
|---|---:|
| Total WAV files | 2,378 |
| Total duration | 42.32 minutes |
| Good samples | 774 |
| Broken samples | 789 |
| Heavy Load samples | 815 |
| Sampling rate | 44.1 kHz |
| Resolution | 32-bit |
| Audio type | Mono |
| Segment length | 3 seconds |

The dataset includes multiple background conditions such as pure recordings, talking, white noise, factory atmospheric noise, and stress-test recordings.

---

## Experiments

Multiple approaches were tested before choosing the final model.

### 1. Classical Machine Learning

We first tested classical machine learning models using extracted audio features.

Feature extraction methods included:

- MFCCs
- Mel spectrogram features
- Chroma features
- Zero-crossing rate
- Spectral contrast

These models gave good results on the prepared dataset, but they showed signs of **overfitting**, so they were not selected as the final solution.

---

### 2. Wav2Vec2 Experiment

A Wav2Vec2-based model was also tested for motor sound classification.

The idea was to use a pretrained audio model and fine-tune it for the three motor states:

- `Good`
- `Broken`
- `Heavy Load`

However, Wav2Vec2 is relatively large and was not the best fit for the project constraints.

---

### 3. AST Model Exploration

We also explored the **Audio Spectrogram Transformer (AST)** idea.

AST is a strong model for audio classification because it processes spectrograms using transformer-style patch embeddings. It is useful for non-speech audio tasks, including mechanical sound classification.

However, the full AST model was too large and computationally expensive for our project and dataset size.

Because of that, we did not use the full AST as the final model. Instead, we used the core idea behind it: learning useful patterns from spectrograms.

---

## Final Model: CNN with Spectrogram Preprocessing

Due to the size and complexity of AST, the final model was built as a lightweight **CNN using spectrogram preprocessing**.

The CNN was designed as a practical **mini-AST-inspired** solution:

- AST learns from spectrogram representations.
- Our CNN also learns from spectrograms.
- Instead of transformer layers, the CNN uses convolution layers to learn time-frequency patterns.
- This made the model smaller, faster, and more suitable for our data.

---

## Preprocessing Pipeline

The final CNN model follows this pipeline:

```text
Audio File
→ Load Audio
→ Resample
→ Trim / Pad to 3 Seconds
→ Noise Reduction
→ STFT Spectrogram Generation
→ CNN Classification
→ Predicted Motor Condition
```

### Main Steps

1. Load the motor audio file.
2. Convert audio to mono.
3. Resample the signal.
4. Trim or pad each clip to 3 seconds.
5. Apply noise reduction.
6. Convert the waveform into a spectrogram using STFT.
7. Train the CNN model on the spectrograms.

---

## CNN Model Details

The CNN receives spectrograms as input.

```text
Input spectrogram shape: (379, 128, 1)
Audio duration: 3 seconds
Samples per clip: 48,000
```

Training setup:

| Item | Value |
|---|---|
| Loss function | Sparse Categorical Crossentropy |
| Optimizer | Adam |
| Learning rate | 0.0001 |
| Metric | Accuracy |
| Regularization | Dropout, EarlyStopping, Cross-validation |

The CNN was selected because it was:

- Smaller than AST
- More suitable for the dataset size
- Better than the overfitted ML baseline
- Strong for spectrogram pattern recognition
- Practical for the project constraints

---

## Results

The CNN model achieved strong results on the dataset.

| Class | Correct Predictions | Total Samples |
|---|---:|---:|
| `engine1_good` | 107 | 110 |
| `engine2_broken` | 124 | 124 |
| `engine3_heavyload` | 132 | 132 |

Validation accuracy was around:

```text
96% - 97%
```

---

## Real Motor Experiment

Only the **CNN spectrogram model** was tested on real hardware.

The real experiment was performed using an **imported Brushless DC motor from China**.

The purpose was to test whether the CNN model could classify real motor sounds outside the public dataset.

The CNN model achieved approximately:

```text
87% accuracy
```

on the real motor experiment.

---

## Project Files

The code is organized inside the `code/` folder.

The Flask app file is also kept inside the `code/` folder as part of the original project files.

Example structure:

```text
project/
├── README.md
|── Flask app file
│── CNN model code
│── Wav2Vec2 experiment
│── ast code
└── dataset description / reports
```

---

## Technologies Used

- Python
- TensorFlow / Keras
- Librosa
- Noisereduce
- NumPy
- Scikit-learn
- Matplotlib
- Audio feature extraction
- Spectrogram preprocessing
- CNN deep learning
- Wav2Vec2 experiment
- AST model exploration
- Flask app file inside `code/`

---

## Citation

Dataset reference:

Sascha Grollmisch, Jakob Abeßer, Judith Liebetrau, Hanna Lukashevich,  
**"Sounding Industry: Challenges and Datasets for Industrial Sound Analysis"**,  
Proceedings of the 27th European Signal Processing Conference, EUSIPCO, A Coruña, Spain, 2019.

---

## Summary

This project demonstrates acoustic motor fault classification using AI.

We tested classical ML models, Wav2Vec2, and AST-style modeling ideas. The classical ML approach showed overfitting, and the full AST model was too large for the project constraints.

The final solution was a lightweight CNN trained on spectrograms, designed as a practical mini-AST-inspired model for classifying Brushless DC motor sounds.

The final CNN model was tested on an imported Brushless DC motor from China and achieved around **87% accuracy** in the real motor experiment.