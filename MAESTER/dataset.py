import random
import torch.utils.data.dataset
import torch.nn.functional as F
from monai.transforms import Compose, RandFlip, RandSpatialCrop, Resize
from utils import register_plugin, get_plugin


@register_plugin('transform', 'betaaug2D')
def betaaug(cfg):
    """
    A data augmentation function that crops, resizes and flips a 2D image randomly.

    Args:
        cfg (dict): A dictionary containing configuration parameters for the data augmentation.

    Returns:
        A Compose object that applies the following transformations to a 2D volume:
        - Randomly crops a region of interest (ROI) from the volume.
        - Resizes the cropped ROI to a specified volume size using bilinear interpolation.
        - Randomly flips the volume along any axis with a probability of 0.5.
    """
    # Extract the minimum size and volume size from the configuration dictionary.
    min_size = int(cfg["vol_size"] * 0.5)
    volume_size = cfg["vol_size"]
    
    # Define a Compose object that applies the following transformations to a 3D volume.
    compose = Compose(
        [RandSpatialCrop(roi_size=(min_size, min_size,)),
         Resize((volume_size, volume_size,), mode="bilinear",
                align_corners=False),
         RandFlip(prob=.5), ])
    
    # Return the Compose object.
    return compose


@register_plugin("dataset", "BetaSegDataset2D")
class BetaSegDataset2D(torch.utils.data.dataset.Dataset):
    """
    A dataset class that samples 2D slices from 3D volumes and applies data augmentation.

    Args:
        cfg (dict): A dictionary containing configuration parameters for the dataset.

    Attributes:
        path_list (list): A list of paths to the input data files.
        pad (int): The amount of padding to add to the input data.
        aug (str): A string indicates data augmentation to the input data.
        vol_size (int): The size of the 2D slices to sample from the 3D volumes.
        key_name_dict (dict): A dictionary mapping file paths to the names of the tensors in the data files.
        data_list (list): A list of tensors representing the input data.

    Methods:
        __len__(): Returns the number of samples in the dataset.
        __getitem__(idx): Returns a randomly sampled 2D slice from the input data at the given index.
        sample_cord(data_idx, axis): Samples a 2D slice from the input data at the given index and axis.

    """

    def __init__(self, cfg):
        self.path_list = cfg["path_list"]
        self.pad = cfg["pad"] 
        self.aug = get_plugin("transform", cfg["aug"])(cfg)
        self.vol_size = (
            cfg["vol_size"] + int(cfg["vol_size"] // cfg["patch_size"])
            if cfg["patch_size"] % 2 == 0
            else cfg["vol_size"]
        ) 
        self.key_name_dict = {
            x: x.split("/")[-1].split("_tensor")[0] for x in self.path_list
        }
        self.data_list = [
            F.pad(torch.load(x)[self.key_name_dict[x]], [self.pad for _ in range(6)])
            for x in self.path_list
        ]

    def __len__(self):
        """
        Returns the number of samples in each epoch.
        """
        return 20000

    def __getitem__(self, idx):
        """
        Returns a randomly sampled 2D slice from the input data at the given index.
        Returns:
            A tensor representing a randomly sampled 2D slice from the input data.
        """
        curr_data_idx = random.randrange(0, len(self.data_list)) # select dataset
        axis = random.randrange(0, 3) # select axis
        return self.sample_cord(curr_data_idx, axis) # return 2D slice

    def sample_cord(self, data_idx, axis):
        """
        Samples a 2D slice from the input data at the given index and axis.

        Args:
            data_idx (int): The index of the input data to sample from.
            axis (int): The axis along which to sample the 2D slice.

        Returns:
            A tensor representing a 2D slice sampled from the input data at the given index and axis.
        """
        data = self.data_list[data_idx] # get dataset
        _, d_z, d_x, d_y = data.shape
        if axis < 1:
            # z axis
            x_sample = torch.randint(
                low=0, high=int(d_x - self.vol_size - 1), size=(1,)
            )
            y_sample = torch.randint(
                low=0, high=int(d_y - self.vol_size - 1), size=(1,)
            )
            z_sample = torch.randint(low=0, high=int(d_z), size=(1,))
            sample = data[0][
                z_sample,
                x_sample : x_sample + self.vol_size,
                y_sample : y_sample + self.vol_size,
            ].unsqueeze(0)
            sample = sample.squeeze(0)
        elif axis < 2:
            # x axis
            x_sample = torch.randint(low=0, high=int(d_x), size=(1,))
            y_sample = torch.randint(
                low=0, high=int(d_y - self.vol_size - 1), size=(1,)
            )
            z_sample = torch.randint(
                low=0, high=int(d_z - self.vol_size - 1), size=(1,)
            )
            sample = data[0][
                z_sample : z_sample + self.vol_size,
                x_sample,
                y_sample : y_sample + self.vol_size,
            ].unsqueeze(0)
            sample = sample.squeeze(0).permute((1, 0, 2))
        else:
            # y axis
            x_sample = torch.randint(
                low=0, high=int(d_x - self.vol_size - 1), size=(1,)
            )
            y_sample = torch.randint(low=0, high=int(d_y), size=(1,))
            z_sample = torch.randint(
                low=0, high=int(d_z - self.vol_size - 1), size=(1,)
            )
            sample = data[0][
                z_sample : z_sample + self.vol_size,
                x_sample : x_sample + self.vol_size,
                y_sample,
            ].unsqueeze(0)
            sample = sample.squeeze(0).permute((2, 0, 1))

        return self.aug(sample)