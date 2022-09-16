"""
predict.py:
    Module that contains the main functions to load the trained model
    and predict the results when an image is loaded.
"""
__author__ = "Paula Jesica Vergara De Castro"
__maintrainer__ = "Udacity"
__email__ = "paulajve@gmail.com"
__copyright__ = "Copyright 2022"
__date__ = "09/04/2022"
__version__ = "0.1"

import argparse
import json
import utils
from train import use_gpu

parser = argparse.ArgumentParser(
    description="Developing an AI application - Prediction Step"
)
parser.add_argument(
    "--gpu",
    action="store_true",
    help="Enable GPU (CUDA) for processing (default is CPU)",
)
parser.add_argument("--dir", action="store", dest="data_dir", default="./flowers/")
parser.add_argument(
    "load_image",
    default="./flowers/test/43/image_02329.jpg",
    nargs="?",
    action="store",
    type=str,
)
parser.add_argument("--topk", default=5, dest="topk", action="store", type=int)
parser.add_argument(
    "--category_names",
    dest="category_names",
    action="store",
    default="cat_to_name.json",
)
parser.add_argument(
    "checkpoint", default="./checkpoint.pth", nargs="?", action="store", type=str
)


parsed = parser.parse_args()
gpu = parsed.gpu
path_image = parsed.load_image
topk = parsed.topk
path = parsed.checkpoint


def main():
    model = utils.load_checkpoint_model(path)
    device = use_gpu(model, gpu)
    top_probs, top_labels, top_flowers = utils.predict(model, path_image, topk)
    print("Prediction done successfully.")


if __name__ == "__main__":
    main()
