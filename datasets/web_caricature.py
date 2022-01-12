import os
import torch
import typing
import numpy as np
import imageio as io
from torch.utils.data import Dataset
from torchvision import transforms as vision_transforms


is_photo = lambda x: os.path.basename(x).startswith("P")


class Person:
    """

    Person class for each category (person) in the dataset.
    
    """

    def __init__(self, class_name: str, indices: list, label: int) -> None:
        """
        
        :param class_name: name of the category (person)
        :param indices   : list of indices of the category (person)
        :label           : label of the category (person)
        
        """

        self.class_name : str           = class_name
        self.indices    : np.ndarray    = np.array(indices)
        self.label      : int           = label

        # photos and caricatures indices
        self.photo_indices: list = None
        self.caric_indices: list = None

    def random_pc_pair_index(self) -> list:
        """
        
        Return random pair index (photo_id, caricature_id) for class.

        :return: random pair for class.
            :shape: (1, 2)
        
        """

        photo_idx = np.random.permutation(self.photo_indices)[0]
        caric_idx = np.random.permutation(self.caric_indices)[0]

        return [photo_idx, caric_idx]


class WebCaricatureDataset(Dataset):

    def __init__(self, args) -> None:
        """

        Custom dataset for WebCaricature dataset. 
        A photo and caricature pair images as single training data.    
        
        :param args: arguments parameters

        """

        # set dataset root and original images directory name
        self.is_train   = args.is_train
        self.root       = args.root
        self.prefix     = args.prefix
        self.train_txt  = args.train_txt
        self.test_txt   = args.test_txt

        # create dataset and index txt paths
        self.dataset_path: str = os.path.join(self.root, self.prefix)

        if self.is_train:
            self.index_txt_path: str = os.path.join(self.root, self.train_txt)
        else:
            self.index_txt_path: str = os.path.join(self.root, self.test_txt)

        # check if it exists
        assert os.path.exists(self.dataset_path), f"{self.dataset_path} is not exists!"
        assert os.path.exists(
            self.index_txt_path
        ), f"{self.index_txt_path} is not exists!"

        # initialize class variables
        self.n_classes  : int           = None
        self.classes    : list[Person]  = None
        self.paths      : list[str]     = None
        self.labels     : list[str]     = None
        self.p_indices  : np.ndarray    = None

        # get image paths from index txt and get labels for each image
        self.paths, self.labels = self._get_paths()

        # create person class for each category (person) in dataset
        self.classes    = self._create_classes()
        self.n_classes  = len(self.classes)

        # set photos and caricatures for each person class
        self.p_indices  = self._separate_photos_caricatures_for_classes()

        # imageio numpy array to tensor
        self._to_tensor = vision_transforms.Compose(
            [vision_transforms.ToPILImage(), vision_transforms.transforms.ToTensor()]
        )

    def __len__(self) -> int:
        """
        
        Return length of the dataset.
        
        """

        return len(self.paths)

    def __getitem__(self, idx: int) -> typing.Dict[torch.Tensor, torch.Tensor]:
        """
        
        Return a single item from dataset.
        
        :param idx: index of the item

        :return   : pair of photo and caricature images for a random class
            :shape: (np.ndarray)

        """

        # choose a random class for item
        class_ = np.random.permutation(self.classes)[idx % 126]

        # choose a random index for item from class
        photo_index, caric_index = class_.random_pc_pair_index()

        # get the photo and caricature paths for the item index
        photo_path = self.paths[photo_index]
        caric_path = self.paths[caric_index]

        # read photo and caricature
        photo_image = io.imread(photo_path, pilmode="RGB")
        caric_image = io.imread(caric_path, pilmode="RGB")

        # photo and caric labels
        photo_labels = self.labels[photo_index]
        caric_labels = self.labels[caric_index]

        # io reads image in order (h, w, c)
        # lets convert to (c, h, w)

        return {

            "images_photo": self._to_tensor(photo_image),
            "images_caric": self._to_tensor(caric_image),
            
            "labels_photo": photo_labels,
            "labels_caric": caric_labels,

            "scales_photo": 1,
            "scales_caric": 1

        }

    def _get_paths(self) -> list:
        """

        Read index txt file and add paths to self.paths list.

        :return: list of paths

        """

        with open(self.index_txt_path, "r") as f:
            lines = f.readlines()

        lines = [line.strip().split()[0] for line in lines]

        assert len(lines) > 0, f"{self.index_txt_path} does not contains any lines"

        # concat the lines (image names) with the original images directory to get fullpath
        # ex:
        #   self.original_images = "./datasets/WebCaricature/OriginalImages/"
        #   line                 = "Alan_Rickman/C00001.jpg"
        #   path = "./datasets/WebCaricature/OriginalImages/Alan_Rickman/C00001.jpg"
        paths = [os.path.join(self.dataset_path, line) for line in lines]

        # create labels for each image
        labels = [os.path.dirname(path) for path in paths]

        # get unique directory name (person name) and matched indices with them
        # matched indices are labels for each images
        _, labels = np.unique(labels, return_inverse=True)

        print(f"\n{len(paths)} images of {labels.max() + 1} classes loaded.")

        return paths, labels

    def _create_classes(self) -> list:
        """
        
        Create person class for each category (person).

        :return: list of classes that can be used as idx2cls.
        
        """

        assert len(self.labels) > 0, "No labels found to create category classes!"

        classes: list[Person] = []

        # convert labels list to numpy array to get indices image that belong to category label
        labels_np: np.ndarray = np.array(self.labels)

        for label in range(labels_np.max() + 1):
            # get indices image that belong to category label
            indices = np.where(labels_np == label)[0]

            # create category class with category label name and indices of category
            category_class = Person(class_name=str(label), indices=indices, label=label)

            # keep category classes
            classes.append(category_class)

        return classes

    def _separate_photos_caricatures_for_classes(self) -> np.ndarray:
        """
        
        Set photos and caricatures for each category class.

        :return: list of category classes that have photos and caricatures and photo indices

        """

        photo_indices_list = [is_photo(path) for path in self.paths]
        photo_indices_np = np.array(photo_indices_list, dtype=bool)

        for cls in self.classes:
            cls.photo_indices = cls.indices[photo_indices_np[cls.indices]]
            cls.caric_indices = cls.indices[~photo_indices_np[cls.indices]]

        print(
            f"Classes {self.n_classes}: {photo_indices_np.sum()} photos, {(~photo_indices_np).sum()} caricatures"
        )

        return photo_indices_np

