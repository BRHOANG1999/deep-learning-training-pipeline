# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

This is a deep learning training pipeline focused on neuroscience applications. The repository contains educational Jupyter notebooks that teach students how to apply deep learning techniques to brain signal analysis, EEG data processing, and brain-computer interfaces.

## Key Technologies and Dependencies

The project uses Python-based scientific computing and machine learning libraries:

- **Core Scientific Computing**: NumPy, Pandas, Matplotlib, SciPy
- **Machine Learning**: scikit-learn, PyTorch, torchvision  
- **Neuroscience Tools**: MNE-Python for EEG/MEG/neurophysiology data analysis
- **Development Environment**: Jupyter Lab/Notebook

## Environment Setup

### Primary Development Environment
The course is designed to run in **Google Colab** with GPU acceleration enabled. Students are instructed to:
1. Switch to GPU runtime (Runtime → Change runtime type → Hardware accelerator: GPU)
2. Install packages using `%pip install` commands within notebooks

### Local Development Alternative
For local development, the setup process is:
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install numpy pandas matplotlib scipy scikit-learn torch torchvision mne jupyter

# Launch Jupyter
jupyter lab
```

## Project Structure

The repository follows a simple educational structure:
- `README.md`: Course overview, setup instructions, and notebook series roadmap
- `welcome_and_setup.ipynb`: Environment setup and introduction notebook
- Future notebooks planned: Python/NumPy refresher, ML foundations, neuroscience data intro, PyTorch for neuroscience, advanced architectures, and real-world applications

## Development Guidelines

### Notebook Development
- Notebooks are designed for educational purposes with extensive markdown explanations
- Each notebook builds on previous concepts progressively
- Code cells include educational comments and print statements to aid learning
- Students are encouraged to experiment by copying and modifying cells

### Package Management
- No formal dependency management files (requirements.txt, setup.py, etc.)
- Dependencies are installed via `%pip install` commands within notebooks
- Core packages: numpy, pandas, matplotlib, scipy, scikit-learn, torch, torchvision, mne, jupyter

### Testing and Validation
- No formal testing framework is currently implemented
- Validation occurs through running notebook cells and checking outputs
- Students verify installations by importing packages and checking versions

## Course Learning Path

1. **Foundation**: Python/NumPy fundamentals, data manipulation
2. **ML Basics**: Classical ML, neural networks, PyTorch introduction  
3. **Neuroscience Data**: EEG basics, signal processing, MNE-Python
4. **Deep Learning**: Advanced architectures for time series (RNNs, CNNs, Transformers)
5. **Applications**: Brain-computer interfaces, seizure detection, clinical applications

## Common Tasks

- **Run notebooks**: Use Jupyter Lab or upload to Google Colab
- **Install packages**: Use `%pip install package_name` in notebook cells
- **Enable GPU**: Switch Colab runtime to GPU for faster neural network training
- **Verify setup**: Run the installation verification cell in `welcome_and_setup.ipynb`