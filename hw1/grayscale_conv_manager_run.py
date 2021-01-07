from grayscale_conv_manager import *
from image import read_grayscale, write_grayscale
import numpy as np
import os

if __name__ == "__main__":

    # TODO: fill list of name, kernel key, value pairs.
    kernels = {
        "identity": np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]], dtype=np.float),
    }
    image_paths = ["./data/lion.png", "./data/butler.png", "./data/walle.png"]

    for key in kernels:
        for filepath in image_paths:
            manager = GrayscaleConvManager(kernels[key], filepath)
            filtered = manager.apply_filter()
            np.clip(filtered, 0, 255)
            name, ext = os.path.basename(filepath).split(".")
            filtered_path = os.path.join("supplemental", "{}_{}.{}".format(name, key, ext))
            print("writing: {}".format(filtered_path))
            write_grayscale(filtered, filtered_path)
