import torch
from torch.utils.data import Dataset


class PathPlanningDataset(Dataset):
    """
    Generic class for loading all kinds of training datasets with motion planning paths in 2D configuration space.
    """

    def __init__(self, file_name):
        """
        Args:
            file_name: path to dataset file - .pt file generated by some generate_..._dataset() function
        """
        self.dataset = torch.load(file_name, weights_only=True)

    def __getitem__(self, idx):
        return self.dataset[idx]

    def __len__(self):
        return len(self.dataset)
