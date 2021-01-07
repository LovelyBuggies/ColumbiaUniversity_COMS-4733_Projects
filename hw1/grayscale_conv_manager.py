import os
from image import read_grayscale
import numpy as np


class GrayscaleConvManager(object):
    def __init__(self, kernel, image_path):
        """Constructor to save rgb and/or depth image in memory,
            where the image and kernel can then be convolved.

        Args:
            kernel (numpy.array [k, k]): k is the kernel width and
                height in pixels. Note: k should be odd.
            image_path (str): path to grayscale image with height, width [h, w].

        Raises:
            ValueError: image path should exist
            ValueError: the kernel must be 2 dimensional
            ValueError: kernel should be square
            ValueError: kernel dimension k should be odd
        """
        super().__init__()

        # set image from file(s)
        self._image = None
        if image_path and os.path.exists(image_path):
            self._image = read_grayscale(image_path)
        else:
            raise ValueError("Input vaild rgb and/or depth images.")

        # validate kernel
        if len(kernel.shape) != 2:
            raise ValueError("kernel must be 2D")
        if kernel.shape[0] != kernel.shape[1]:
            raise ValueError("For this class, must use square kernel.")
        if kernel.shape[0] % 2 == 0:
            raise ValueError("kernel side dimension should be odd.")

        self._kernel = kernel

    def apply_filter(self):
        """Apply the convolutional filter across grayscale image.
            Note: Remember to zero pad to avoid index out of bounds

        Returns:
            numpy.array [h, w]: filtered image
        """
        pass  # TODO
