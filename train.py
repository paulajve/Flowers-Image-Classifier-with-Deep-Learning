"""
train.py:
    Module that contains the main functions to build the network, 
    train the classifier and save the model.
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
import torch

# The use of argparse allows to write command-lines, arguments and
# sub-commands between interfaces. The program defines what arguments
# it requires, and argparse will figure out how to parse those out of
# sys.argv. The argparse module also automatically generates help and
# usage messages and issues errors when users give the program invalid
# arguments.

"""
FIRST DEFINITIONS
-----------------
The followings are parser arguments for load_data and label_mapping functions.
"""

parser = argparse.ArgumentParser(
    description="Developing an AI application - Training Step"
)

parser.add_argument(
    "data_dir", action="store", default="./flowers/", help="Load data directory."
)

"""
SECOND DEFINITIONS
------------------
The following ones are parser arguments for train_classifier, check_test_accuracy
and save_checkpoint functions.
"""
parser.add_argument("--arch", type=str, default="vgg19", action="store")
parser.add_argument("--epochs", type=int, default=3, help="Set the number of Epochs")
parser.add_argument(
    "--print_every",
    type=int,
    default=20,
    help="Define the number of mini-batches wanted to print the results on screen during training",
)
parser.add_argument(
    "--dropout",
    type=float,
    default=0.5,
    help="Set the dropout probability (between 0 and 1)",
)
parser.add_argument(
    "--hidden_layer1", type=int, default=4096, help="Set the first hidden layer size"
)
parser.add_argument("--lr", type=int, default=0.001, help="Set the learning rate")
parser.add_argument(
    "--gpu",
    action="store_true",
    help="Enable GPU (CUDA) for processing (default is CPU)",
)
parser.add_argument(
    "--save_checkpoint",
    default="./checkpoint.pth",
    action="store",
    help="Save the model",
)

"""
We need to set the name classes, load the data, build the model and svae it:
"""

parsed = parser.parse_args()
root = parsed.data_dir
structure = parsed.arch
epochs = parsed.epochs
print_every = parsed.print_every
dropout = parsed.dropout
hidden_layer1 = parsed.hidden_layer1
lr = parsed.lr
gpu = parsed.gpu
path = parsed.save_checkpoint


def use_gpu(model, gpu):
    # Checking computer hardware
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device == "cpu":
        print("Your current device supports only CPU.")
        print("Using CPU for processing.")
    else:
        print("Your current device supports GPU.")
        if gpu:
            print("Using GPU for processing.")
        else:
            device = "cpu"
            print(
                "Using CPU for processing as GPU not requested. Use '--gpu' for better behaviour."
            )
    model.to(device)
    return device


def main():
    trainloader, validloader, testloader = utils.load_data(root)
    model, optimizer, criterion = utils.network_construct(
        structure, dropout, hidden_layer1, lr, gpu
    )
    device = use_gpu(model, gpu)
    utils.train_classifier(
        model, trainloader, epochs, print_every, criterion, optimizer, device
    )
    utils.save_checkpoint(model, path)
    print("Model trained succesfully.")


if __name__ == "__main__":
    main()
