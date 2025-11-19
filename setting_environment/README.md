Poseidon-PPQ: Probabilistic Post-Training Quantization for Neural Operators
This repository provides an implementation of PPQ (Probabilistic Programming Quantization) applied to the Poseidon (scOT) neural operator family for PDE modelling.
It includes:
Scripts to download datasets
Scripts to download pretrained Poseidon models
A full implementation of PPQ-based step-size optimization
Utility code for range estimation, MC-likelihood, MDL prior, and quantized evaluation
⚙️ 1. Installation
Clone repo:
git clone https://github.com/<your_repo>
cd <your_repo>

Install environment:
pip install -r setting_environment/requirements.txt
You must also log in to:
hf login
wandb login





📦 2. Download All Datasets & Models
Everything is automated.
Just run:
python setting_environment/download.py
This script will:
✓ Download datasets
NS-SVS
NS-BB
NS-SL
and assemble them into .nc files via provided assemble_data.py.
✓ Download pretrained Poseidon models
Both L and B architectures:
NS-PwC
NS-SVS
NS-BB
NS-SL
All models are fetched from W&B via artifact restoration.
After running, your project structure will look like:
dataset/
    NS-SVS/
    NS-BB/
    NS-SL/

models/
    NS-PwC/
    NS-SVS/
    NS-BB/
    NS-PwC-B/
    NS-SVS-B/
    NS-BB-B/
    NS-SL-B/



when everything downloaded, run:
python PPQ/PPQ_poseidon_minibatch.py

to optimize and run:
python PPQ/B_test.py 

to see teh result