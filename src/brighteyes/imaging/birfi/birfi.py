import torch

from. utils import median_filter

class Birfi:

    def __init__(self, data):
        self.data = torch.squeeze(data)
        self.to = None

    def find_t0(self, median_window: int = 5):
        """
        Finds the index of the global minimum of the derivative of self.data.
        Uses median filtering to reduce noise while preserving shape.

        Args:
            median_window (int): Size of the median filter window (odd number).
        """

        derivative = torch.diff(self.data)
        derivative = median_filter(derivative, median_window)
        min_index = torch.argmin(derivative)
        self.t0 = min_index.item()

        return self.t0
