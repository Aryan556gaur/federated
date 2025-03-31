# Federated Learning with Iris Dataset

This project demonstrates federated learning using TensorFlow Federated (TFF) with the Iris dataset. It simulates multiple clients training a shared model while keeping their data private.

## Requirements

- Python 3.9.x (TensorFlow Federated is not compatible with Python 3.11)
- Virtual environment (recommended)

## Setup

1. Install Python 3.9.x from [python.org](https://www.python.org/downloads/)
2. Create and activate a virtual environment:
```bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Unix/MacOS:
source venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Running the Code

```bash
python model.py
```

The code will:
1. Load and partition the Iris dataset among simulated clients
2. Train a federated model using TFF
3. Evaluate the global model on test data
4. Save the trained model weights

## Project Structure

- `model.py`: Main implementation of federated learning
- `requirements.txt`: Project dependencies
