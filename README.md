<h2 align="center">
WarpGAN: Automatic Caricature Generation <br> Unofficial PyTorch Implementation
</h2>
  
> A PyTorch implementation of WarpGAN, a fully automatic network that can generate caricatures given an input face photo. 
> Besides transferring rich texture styles, WarpGAN learns to automatically predict a set of control points that can warp the photo into a caricature,
  while preserving identity. We introduce an identity-preserving adversarial loss that aids the discriminator to distinguish between different subjects. 
  Moreover, WarpGAN allows customization of the generated caricatures by controlling the exaggeration extent and the visual styles.

This repository is the unofficial PyTorch implementation of the paper:  

&nbsp;&nbsp;&nbsp;[**WarpGAN: Automatic Caricature Generation**](https://arxiv.org/pdf/1811.10100.pdf)  
&nbsp;&nbsp;&nbsp;[Yichun Shi](https://github.com/seasonSH), [Debayan Deb](https://github.com/ronny3050) and [Anil K. Jain](https://www.cse.msu.edu/~jain/)  
&nbsp;&nbsp;&nbsp;*CVPR*, 2019 (**Oral Presentation**)  
&nbsp;&nbsp;&nbsp;[Arxiv](https://arxiv.org/pdf/1811.10100.pdf)

- Contact: 
  - nmutlu17@ku.edu.tr
  - ftamur16@ku.edu.tr
  - oguzhanyildirim16@ku.edu.tr

## Getting started
This code has been developed with Anaconda (**Python 3.8**), **PyTorch 1.10.0** and **CUDA 11.3**. 

Create a conda environment for the project:

  ```Shell
  conda create --name warpgan python=3.8
  ```
  
Activate conda activate to work on project:

  ```Shell
  conda activate warpgan
  ```
  
Install packages with installation script:

  ```Shell
  ./install.sh
  ```

## Dataset

Note: In this section, we assume that you are always in the directory `WarpGAN-PyTorch/`

Please download the [WebCaricature](https://cs.nju.edu.cn/rl/WebCaricature.htm) to dataset for the experiment:

  - Unzip dataset and place into `/datasets`
 
### Preprocess Dataset:

  - To rename directory names in the dataset:

  ```shell
  python utils/rename.py
  ```
  
  - To normalize all faces in the dataset:
  
  ```shell
  python align/align_dataset.py datasets/index_txt/landmarks.txt datasets/webcaricacture_aligned_256 --scale 0.7
  ```

## Training and Inference

  ```shell
  python train.py
  ```

## Pretrained Models 

The **[checkpoints](checkpoints/)** folder contains the checkpoints of the pretrained models.  

## Outputs and Visualization





## Acknowledgement

- Portions of the source code (e.g., warping, aligning) are from [seasonSH](https://github.com/seasonSH/WarpGAN)  


