# Correlation Ratio for Multi- and Mono-modal Image Registration
This repository hosts PyTorch implementation of Correlation Ratio for medical image registration, originally proposed in [this paper](https://link.springer.com/chapter/10.1007/BFb0056301). We have evaluated CR has a loss function for the application of both affine and deformable registration.

1. [Chen, Junyu, et al. "Unsupervised Learning of Multi-modal Affine Registration for PET/CT,” 2024 IEEE NSS/MIC](https://arxiv.org/pdf/2409.13863v1)
2. To be added...

You can find the PyTorch implementation of the correlation ratio and local-patch-based correlation ratio here:
- [correlation ratio](https://github.com/junyuchen245/Correlation_Ratio/blob/91c142199da6e877ff6276ccf7cfe795e66eccb0/affine/losses.py#L235)
- [local correlation ratio](https://github.com/junyuchen245/Correlation_Ratio/blob/91c142199da6e877ff6276ccf7cfe795e66eccb0/affine/losses.py#L300)

## PET/CT Multi-Modal Affine Registration
The source code for PET/CT affine registration can be found [here](https://github.com/junyuchen245/Correlation_Ratio/tree/main/affine), and you will need to install the required packages listed in the `requirements.txt` file.

#### *Multi-scale Instance Optimization*
<img src="https://github.com/junyuchen245/Correlation_Ratio/blob/main/figs/AffineRegAlg.jpg" width="400"/>

#### *Qualitative Results*
<img src="https://github.com/junyuchen245/Correlation_Ratio/blob/main/figs/Affine_PETCT.jpg" width="700"/>

## T1/T2 Brain MRI Multi-Modal Deformable Registration
To be added...

## Citation
If you find this code is useful in your research, please consider to cite:

    @misc{chen2024unsupervised,
      title={Unsupervised Learning of Multi-modal Affine Registration for PET/CT}, 
      author={Junyu Chen and Yihao Liu and Shuwen Wei and Aaron Carass and Yong Du},
      year={2024},
      eprint={2409.13863},
      archivePrefix={arXiv},
      primaryClass={eess.IV},
      url={https://arxiv.org/abs/2409.13863}, 
    }
