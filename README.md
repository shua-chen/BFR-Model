# 🌀 BFR-Model

This is a Blind Face Restoration (BFR) model for EPFL semester project.

The model is based on ControlNeXt and StableDiffusion XL.

Hardware requirement: A single GPU with at least 24GB memory.

## Quick Start

Clone the repository:

```bash
git clone https://github.com/shua-chen/BFR-Model.git
cd BFR-Model-Train
```

Install the required packages:

```bash
pip install -r requirements.txt
```

Run the training script:

```bash
bash train_cluste.sh
```

The output will be saved in `trains/example`.


For inference, run the script:

```bash
bash script.sh
```
