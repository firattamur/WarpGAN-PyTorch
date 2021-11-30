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
This code has been developed with Anaconda (Python 3.7), **PyTorch 1.2.0** and CUDA 10.0 on Ubuntu 16.04.  
Based on a fresh [Anaconda](https://www.anaconda.com/download/) distribution and [PyTorch](https://pytorch.org/) installation, following packages need to be installed:  

  ```Shell
  conda install pytorch==1.2.0 torchvision==0.4.0 cudatoolkit=10.0 -c pytorch
  pip install tensorboard
  pip install pypng==0.0.18
  pip install colorama
  pip install scikit-image
  pip install pytz
  pip install tqdm==4.30.0
  pip install future
  ```

Then, please excute the following to install the Correlation and Forward Warping layer:
  ```Shell
  ./install_modules.sh
  ```

**For PyTorch version > 1.3**  
Please put the **`align_corners=True`** flag in the `grid_sample` function in the following files:
  ```
  augmentations.py
  losses.py
  models/modules_sceneflow.py
  utils/sceneflow_util.py
  ```


## Dataset

Please download the following to datasets for the experiment:
  - [KITTI Raw Data](http://www.cvlibs.net/datasets/kitti/raw_data.php) (synced+rectified data, please refer [MonoDepth2](https://github.com/nianticlabs/monodepth2#-kitti-training-data) for downloading all data more easily)
  - [KITTI Scene Flow 2015](http://www.cvlibs.net/datasets/kitti/eval_scene_flow.php?benchmark=flow)

To save space, we also convert the *KITTI Raw* **png** images to **jpeg**, following the convention from [MonoDepth](https://github.com/mrharicot/monodepth):
  ```
  find (data_folder)/ -name '*.png' | parallel 'convert {.}.png {.}.jpg && rm {}'
  ```   
We also converted images in *KITTI Scene Flow 2015* as well. Please convert the png images in `image_2` and `image_3` into jpg and save them into the seperate folder **`image_2_jpg`** and **`image_3_jpg`**.  

To save space further, you can delete the velodyne point data in KITTI raw data and optionally download the [*Eigen Split Projected Depth*](https://drive.google.com/file/d/1a97lgOgrChkLxi_nvRpmbsKspveQ6EyD/view?usp=sharing) for the monocular depth evaluation on the Eigen Split. We converted the velodyne point data of the Eigen Test images in the numpy array format using code from [MonoDepth](https://github.com/mrharicot/monodepth). After downloading and unzipping it, you can merge with the KITTI raw data folder.  
  - [Eigen Split Projected Depth](https://drive.google.com/file/d/1a97lgOgrChkLxi_nvRpmbsKspveQ6EyD/view?usp=sharing)

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


