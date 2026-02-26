# AnyPPG Model Loading

This directory contains the necessary files to load and use the pre-trained **AnyPPG** model.

## Contents

- `anyppg_ckpt.pth`: The pre-trained model checkpoint weights.
- `resnet1d.py`: The implementation of the 1D ResNet model architecture used by AnyPPG.
- `demo_load_anyppg.py`: A demonstration script showing how to instantiate the model and load the weights.

## Usage

To load the pre-trained AnyPPG model, you can use the code provided in `demo_load_anyppg.py`.

### Prerequisites

- Python 3.x
- PyTorch

### Example Code

You can run the demo script directly:

```bash
python demo_load_anyppg.py
```