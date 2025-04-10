# Bootstrapping Sound Source Localization through Hidden Teachers

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)

This repository contains the implementation of the bootstrapping-based approach to learning sound source localization (SSL) presented in our paper: "Bootstrapping Sound Source Localization through Hidden Teachers".

## Overview

This research explores how sound source localization can be learned without explicit external spatial labels, using a bootstrapping framework. We propose a learning mechanism where simple innate circuits (Hidden Teachers) interact with the environment to generate calibration signals for training more complex neural representations, analogous to how biological systems might acquire spatial hearing.

Our approach demonstrates:
- How spatial learning can occur with minimal supervision
- The role of innate circuits in guiding the development of complex perceptual abilities
- A biologically plausible approach to sound source localization that doesn't require massive labeled datasets

## Repository Structure

```
.
├── data/                   # Training and testing sound examples
├── environments/           # Simulated acoustic environments
├── models/                 # Model implementations
│   ├── teacher_models/     # Hidden Teacher implementations
│   ├── student_models/     # Student network architectures
│   └── baseline_models/    # Baseline implementations for comparison
├── experiments/            # Experiment configurations and scripts
├── results/                # Experimental results and visualizations
├── pretrained_models/      # Pre-trained model weights
└── utils/                  # Utility functions and helpers
```

## Installation

Requirements:
- Python 3.8 or higher
- PyTorch 1.9 or higher
- NumPy
- SciPy
- Matplotlib
- Librosa
- PyTorch3D (for HRTF computations)

```bash
# Clone the repository
git clone https://github.com/your-username/ssl-bootstrapping.git
cd ssl-bootstrapping

# Create a virtual environment
python -m venv env
source env/bin/activate  # On Windows: env\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Using the Pre-trained Models

We provide pre-trained model weights for both the Teacher and Student networks in the `pretrained_models/` directory. You can load these models for inference or further fine-tuning:

```python
import torch
from models.student_models import StudentDNN

# Load a pre-trained Student model
model = StudentDNN()
model.load_state_dict(torch.load('pretrained_models/student_model_LSO_teacher.pth'))
model.eval()

# Perform inference
with torch.no_grad():
    prediction = model(audio_input)
```

## Training a Model

To train a model using the bootstrapping approach:

```bash
# Train a Student model with the LSO-ensemble Teacher
python experiments/train_student.py --teacher lso_ensemble --epochs 100 --batch_size 32

# Train with different Teacher architectures
python experiments/train_student.py --teacher midline --epochs 100 --batch_size 32
```

## Evaluating Model Performance

To evaluate a trained model:

```bash
# Evaluate the model on the test set
python experiments/evaluate.py --model_path pretrained_models/student_model_LSO_teacher.pth
```

## Simulation Environment

Our experiments are conducted in a simulated environment that models sound propagation and binaural hearing. To create your own simulation:

```python
from environments.simulator import AcousticEnvironment

# Create a simulation environment
env = AcousticEnvironment()

# Generate binaural signals for a sound source at azimuth 30°
binaural_signal = env.generate_binaural(source_audio, azimuth=30, elevation=0)
```

## Model Weights

The provided model weights include:

- `student_model_LSO_teacher.pth`: Student model trained with LSO-ensemble Teacher
- `student_model_midline_teacher.pth`: Student model trained with midline Teacher
- `student_model_motor_feedback.pth`: Student model trained with motor feedback Teacher
- `baseline_supervised.pth`: Baseline model trained with full supervision
- `teacher_models/lso_ensemble.pth`: Pre-trained LSO-ensemble Teacher model

## Citation

If you use this code in your research, please cite our paper:

```
@article{author2023bootstrapping,
  title={Bootstrapping Sound Source Localization through Hidden Teachers},
  author={Author, A. and Author, B.},
  journal={Journal Name},
  year={2023}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contact

For questions or clarifications, please open an issue or contact [author@example.com](mailto:author@example.com).
