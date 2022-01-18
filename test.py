"""

Test WarpGAN Model.

"""

import os
import sys
import glob
import random

import torch
import numpy as np
import imageio as io
import torch.nn as nn
from torchvision import transforms

# from align.detect_align      import detect_align
from utils.commandline       import load_config, load_config_test
from datasets.web_caricature import WebCaricatureDataset
from models.model_warpgan    import WarpGANGenerator, WarpGANDiscriminator

# Set random seed for reproducibility
manualSeed = 42

random.seed(manualSeed)
torch.manual_seed(manualSeed)


normalize    = transforms.Compose(
                    [   
                        transforms.ToPILImage(),
                        transforms.ToTensor(),
                        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
                    ]
                )
                
invNormalize = transforms.Normalize(mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225], std=[1/0.229, 1/0.224, 1/0.225])


def load_checkpoint(path: str) -> dict:
    """

    Load torch model state dict from specified path.

    :param path : path to load state dict

    """
    checkpoint = torch.load(path)

    return checkpoint


def last_checkpoint(path: str) -> str:
    """

    Sort checkpoints saved date and return last store checkpoint path.

    :param path: path for checkpoints folder.
    :return    : last stored checkpoint path

    """

    checkpoints = glob.glob(os.path.join(path, "*.pth"))

    if len(checkpoints) == 0:
        return None

    latest = max(checkpoints, key=os.path.getctime)

    return latest


def demo_photo(path: str) -> str:
    """

    Get photo from the specified path.

    :param path: path for demo photos folder
    :return    : photo in path

    """

    photo = glob.glob(os.path.join(path, "*"))

    if len(photo) == 0:
        return None

    photo_path = os.path.join(os.path.abspath(os.getcwd()), photo[0])

    return photo_path


if __name__ == "__main__":

    # load configuration file from specified configuration file path
    config = load_config(sys.argv)

    # load configuration file for test
    config_test = load_config_test(sys.argv)

    # decide which device we want to run on
    config.device = torch.device("cuda" if (torch.cuda.is_available()) else "cpu")

    # set is_train to test
    config.is_train = False
    
    print(f"\nTest Device: {config.device}\n")

    # load generator model
    warpgan_G = WarpGANGenerator(config).to(config.device)

    # start index for epoch
    epoch_start = 0

    # check if there any checkpoints
    checkpoint_path = last_checkpoint(path=config.checkpoints_path)

    if not checkpoint_path:
        print("No checkpoint found...\n")
        exit()

    checkpoint  = load_checkpoint(path=checkpoint_path)

    # load state dicts of models and optimizers
    warpgan_G.load_state_dict(checkpoint["warpgan_G"])

    # load and set train mode
    warpgan_G.eval()

    print(f"Checkpoint {checkpoint_path} loaded.\n")

    # ------------------------------------------
    # Input Image
    # ------------------------------------------

    photo_path = demo_photo(path=config_test.demo_photo_path)

    if photo_path is None:
        print("No demo photo found! exiting...\n")
        raise ValueError()

    photo = io.imread(photo_path, pilmode="RGB")

    if not config_test.aligned:
        photo = detect_align(photo)
    
    # normalize images
    photo = normalize(photo)

    # repeat for different styles
    photos = np.tile(photo[None], [config_test.num_styles, 1, 1, 1])

    scales = (config_test.scale * np.ones((config_test.num_styles))).reshape(-1, 1)
    styles = np.random.normal(0., 1., (config_test.num_styles, config.style_size))

    input_dict = {

        "images_photo" : torch.tensor(photos, dtype=torch.float32).to(config.device),
        "styles_photo" : torch.tensor(styles, dtype=torch.float32).to(config.device),
        "scales_photo" : torch.tensor(scales, dtype=torch.float32).to(config.device)

    }

    # forward pass on generator
    caricatures = warpgan_G(input_dict)

    # unnormalize images
    caricatures = invNormalize(caricatures)

    # n, c, h, w -> n, h, w, c : imageio expects h, w, c
    caricatures = caricatures.permute(0, 2, 3, 1)

    print(f"Caricatures are generated! Saving to {config_test.demo_caric_path}...", end="")

    # save output caricatures
    for i in range(config_test.num_styles):
        io.imsave(os.path.join(config_test.demo_caric_path, f"_{i}.png"), caricatures[i].cpu().detach().numpy())

    print("Done.")