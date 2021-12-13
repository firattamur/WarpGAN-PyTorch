<h2 align="center">
WarpGAN: Automatic Caricature Generation <br> PyTorch
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
The **[scripts](scripts/)** folder contains training\/inference scripts of all experiments demonstrated in the paper (including ablation study).

**For training**, you can simply run the following script files:

| Script                                       | Training                   | Dataset                |
|----------------------------------------------|----------------------------|------------------------|
| `./train_monosf_selfsup_kitti_raw.sh`        | Self-supervised            | KITTI Split            |
| `./train_monosf_selfsup_eigen_train.sh`      | Self-supervised            | Eigen Split            |


**Fine-tuning** is done in two stages: *(i)* first finding the stopping point using train\/valid split, and then *(ii)* fune-tuning using all data with the found iteration steps.  
| Script                                       | Training                   | Dataset                |
|----------------------------------------------|----------------------------|------------------------|
| `./train_monosf_kitti_finetune_1st_stage.sh` | Semi-supervised finetuning | KITTI raw + KITTI 2015 |
| `./train_monosf_kitti_finetune_2st_stage.sh` | Semi-supervised finetuning | KITTI raw + KITTI 2015 |

In the script files, please configure these following PATHs for experiments:
  - `EXPERIMENTS_HOME` : your own experiment directory where checkpoints and log files will be saved.
  - `KITTI_RAW_HOME` : the directory where *KITTI raw data* is located in your local system.
  - `KITTI_HOME` : the directory where *KITTI Scene Flow 2015* is located in your local system. 
  - `KITTI_COMB_HOME` : the directory where both *KITTI Scene Flow 2015* and *KITTI raw data* are located.  
   
  
**For testing the pretrained models**, you can simply run the following script files:

| Script                                    | Task          | Training        | Dataset          | 
|-------------------------------------------|---------------|-----------------|------------------|
| `./eval_monosf_selfsup_kitti_train.sh`    | MonoSceneFlow | Self-supervised | KITTI 2015 Train |
| `./eval_monosf_selfsup_kitti_test.sh`     | MonoSceneFlow | Self-supervised | KITTI 2015 Test  |
| `./eval_monosf_finetune_kitti_test.sh`    | MonoSceneFlow | fine-tuned      | KITTI 2015 Test  |
| `./eval_monodepth_selfsup_kitti_train.sh` | MonoDepth     | Self-supervised | KITTI test split |
| `./eval_monodepth_selfsup_eigen_test.sh`  | MonoDepth     | Self-supervised | Eigen test split |

  - Testing on *KITTI 2015 Test* gives output images for uploading on the [KITTI Scene Flow 2015 Benchmark](http://www.cvlibs.net/datasets/kitti/eval_scene_flow.php).  
  - To save output image, please turn on `--save_disp=True`, `--save_disp2=True`, and `--save_flow=True` in the script.  

## Pretrained Models 

The **[checkpoints](checkpoints/)** folder contains the checkpoints of the pretrained models.  
Pretrained models from the ablation study can be downloaded here: [download link](https://download.visinf.tu-darmstadt.de/data/2020-cvpr-hur-self-mono-sf/models/checkpoints_ablation_study.zip)

## Outputs and Visualization

Output images and visualization of the main experiments can be downloaded here:
  - [Self-supervised, tested on KITTI 2015 Train](https://download.visinf.tu-darmstadt.de/data/2020-cvpr-hur-self-mono-sf/results/self_supervised_KITTI_train.zip)
  - [Self-supervised, tested on Eigen Test](https://download.visinf.tu-darmstadt.de/data/2020-cvpr-hur-self-mono-sf/results/self_supervised_Eigen_test.zip)
  - [Fined-tuned, tested on KITTI 2015 Train](https://drive.google.com/file/d/1JLCWT5-Ase8VkOkA9PWpkee7K0qpgm64/view?usp=sharing)


## Acknowledgement

- Portions of the source code (e.g., training pipeline, runtime, argument parser, and logger) are from [Jochen Gast](https://scholar.google.com/citations?user=tmRcFacAAAAJ&hl=en)  


