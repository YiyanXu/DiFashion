# Diffusion Models for Generative Outfit Recommendation
This is the pytorch implementation of our paper:
> Diffusion Models for Generative Outfit Recommendation

## Environment
- Anaconda 3
- python 3.8.13
- torch 2.0.1
- torchvision 0.15.2
- Pillow 9.0.1
- numpy 1.24.4
- transformers 4.32.1
- open-clip-torch 2.20.0
- accelerate 0.20.3
- diffusers 0.18.2
- xformers 0.0.22
- pytorch-fid 0.3.0
- lpips 0.1.4

## Usage
### Data
The experimental data are in './datasets' folder, including iFashion and Polyvore-U.

### Training
```
cd ./DiFashion
sh run_eta0.1.sh
```

### Inference
1. Download the checkpoints released by us from [Google drive](waiting_for_update).
2. Put the 'checkpoints' folder into the current folder.
3. Run inf4eval.py
```
sh run_inf4eval.sh
```
