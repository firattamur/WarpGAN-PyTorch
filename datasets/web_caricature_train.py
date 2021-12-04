import torch
import os.path
import numpy as np
import torch.utils.data as torchdata


class WebCaricatureTrain(torchdata.Dataset):
    """

    Train dataset for WebCaricature.
    
    """

    def __init__(self, args, root: str = None) -> None:
        """
        """


    def __getitem__(self, index) -> dict:
        """
        """

        return {}

    def __len__(self) -> int:
        """
        
        Length of the dataset.

        :return: length of the dataset

        """
        return self._size