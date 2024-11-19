# Neural Networks Speech Enhancer

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![MATLAB](https://img.shields.io/badge/MATLAB-R2023a-orange.svg)

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
  - [Training the Neural Network](#training-the-neural-network)
  - [Enhancing Speech Signals](#enhancing-speech-signals)
  - [Visualizing Results](#visualizing-results)
- [Project Structure](#project-structure)
- [Configuration](#configuration)
- [Dependencies](#dependencies)
- [Contributing](#contributing)
- [License](#license)

## Overview

The Neural Networks Speech Enhancer project, developed as part of my master’s thesis at ÉTS University, uses neural networks techniques combined with spectral subtraction to enhance noisy speech signals. The goal is to improve the clarity and Signal-to-Noise Ratio (SNR) of audio, making speech more intelligible in challenging noise conditions. This MATLAB-based project includes scripts for training the neural network, processing audio, and generating visualizations of enhancement performance.

## Features

- **Neural Network-Based Speech Classification**: Identifies speech and non-speech frames to apply targeted noise reduction.
- **Adaptive Spectral Subtraction**: Uses spectral subtraction to minimize noise without distorting the speech signal.
- **Customizable SNR**: Allows for different levels of noise reduction based on the desired SNR for enhanced speech quality.
- **Visualization and Analysis Tools**: Provides time-domain and frequency-domain plots to analyze enhancement effectiveness.

## Installation

1. **Clone the Repository**
   ```bash
   git clone https://github.com/arsaland/neural-networks-speech-enhancer.git
   cd neural-networks-speech-enhancer
   ```

2. **Add to MATLAB Path**
   Open MATLAB and add the project directory:
   ```matlab
   addpath('path_to_project_directory');
   addpath('path_to_project_directory/utilities');
   ```

3. **Install Dependencies**
   Ensure MATLAB R2023a or later is installed in addition to Deep Learning Toolbox.

## Usage

### Training the Neural Network

1. **Data Preparation**:
   - Place clean speech files (`.wav` format) in `data/clean/`.
   - Place noisy speech files in `data/noisy/`.

2. **Run Training Script**:
   ```matlab
   training
   ```
   This script processes the audio files, extracts features, and trains a neural network saved as `trained_network.mat`.

### Enhancing Speech Signals

Use the `speech_enhancer.m` script to enhance noisy audio.

1. **Prepare Files**:
   - Ensure clean speech (`clean.wav`) and noisy speech (`noisy.wav`) files are in the project directory.

2. **Run Enhancement**:
   ```matlab
   speech_enhancer
   ```
   The script outputs `enhanced.wav`, displaying the input and output SNR values for comparison.

### Visualizing Results

Run the `plots.m` script to generate plots that compare the original, noisy, and enhanced signals:
```matlab
plots
```

## Project Structure
- `data/`: Contains clean and noisy speech data.
- `models/`: Stores trained neural network models.
- `scripts/`: Contains scripts for training, speech enhancement, and visualization.
- `results/`: Outputs enhanced speech and plots for result analysis.

## Configuration
Adjust SNR and other settings within the scripts as necessary for different noise environments.

## Dependencies
- MATLAB R2023a or later and Deep Learning Toolbox

## License
This project is licensed under the MIT License.
