"""
utils.py:
    Module that contains the model functions to train and predict.
"""
__author__ = "Paula Jesica Vergara De Castro"
__maintrainer__ = "Udacity"
__email__ = "paulajve@gmail.com"
__copyright__ = "Copyright 2022"
__date__ = "09/04/2022"
__version__ = "0.1"

import json
import numpy as np
import torch

# A neural networks library deeply integrated with autograd designed for maximum flexibility.
from torch import nn
from torch import optim
from collections import OrderedDict
from torch.utils.data import DataLoader
import torch.nn.functional as F
from PIL import Image

# Transforms are used to augment the training data with random scaling, rotations, mirroring, and/or cropping.
from torchvision import datasets, models, transforms


with open("cat_to_name.json", "r") as f:
    cat_to_name = json.load(f)

arch = {"vgg19": 25088, "resnet101": 4096}


def transformation(root):
    """Define a function able to get the data from training,
    validation and testing folders and allows the transformation
    of each image saved in the mentioned folders. The main
    directory has to be define in train and predict files (i.e:
    data_dir = 'flowers' )
    """
    data_dir = root
    train_dir = data_dir + "/train"
    valid_dir = data_dir + "/valid"
    test_dir = data_dir + "/test"
    # Define your transforms for the training, validation, and testing sets
    train_transforms = transforms.Compose(
        [
            transforms.RandomRotation(30),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    validation_transforms = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    test_transforms = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    # Load the datasets with ImageFolder
    train_data = datasets.ImageFolder(data_dir + "/train", transform=train_transforms)
    valid_data = datasets.ImageFolder(
        data_dir + "/valid", transform=validation_transforms
    )
    test_data = datasets.ImageFolder(data_dir + "/test", transform=test_transforms)
    return train_data, valid_data, test_data


def load_data(root):
    data_dir = root
    tr_data, val_data, ts_data = transformation(data_dir)
    # Using the image datasets and the trainforms, define the dataloaders
    trainloader = torch.utils.data.DataLoader(tr_data, batch_size=32, shuffle=True)
    validloader = torch.utils.data.DataLoader(val_data, batch_size=32, shuffle=True)
    testloader = torch.utils.data.DataLoader(ts_data, batch_size=32, shuffle=True)
    return trainloader, validloader, testloader


train_data, valid_data, test_data = transformation("./flowers/")
trainloader1, validloader1, testloader1 = load_data("./flowers/")

# Define a new, untrained feed-forward network as a classifier, using ReLU activations and dropout


def network_construct(structure, dropout, hidden_layer1=4096, lr=0.001, device="gpu"):

    if structure == "vgg19":
        model = models.vgg19(pretrained=True)
    elif structure == "resnet101":
        model = model.resnet101(pretrained=True)
    else:
        print("You have to select a valid model like vgg19 or resnet101.".format(arch))

    ## GPU usage, check if user choice is GPU and Cuda is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Freeze the backpropagation through parameters
    for param in model.parameters():
        param.requires_grad = False

    # Use OrderedDict to preserve the order in which the keys are inserted
    classifier = nn.Sequential(
        OrderedDict(
            [
                ("fc1", nn.Linear(arch[structure], hidden_layer1)),
                ("relu1", nn.ReLU()),
                ("drop1", nn.Dropout(dropout)),
                ("fc2", nn.Linear(hidden_layer1, 1024)),
                ("relu2", nn.ReLU()),
                ("drop2", nn.Dropout(dropout)),
                ("fc3", nn.Linear(1024, 102)),
                ("output", nn.LogSoftmax(dim=1)),
            ]
        )
    )
    model.classifier = classifier
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr)
    model.to(device)
    return model, optimizer, criterion


# Train the classifier layers using backpropagation using the pre-trained network to get the features
def train_classifier(
    model, trainloader, epochs, print_every, criterion, optimizer, device="gpu"
):
    print_every = print_every
    steps = 0

    ## GPU usage, check if user choice is GPU and Cuda is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    for epoch in range(epochs):
        running_loss = 0
        for ii, (inputs, labels) in enumerate(trainloader):
            steps += 1
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            # Forward and backward passes
            outputs = model.forward(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if steps % print_every == 0:
                model.eval()
                valid_loss = 0
                valid_accuracy = 0
                for ii, (inputs2, labels2) in enumerate(validloader1):
                    optimizer.zero_grad()
                    inputs2, labels2 = inputs2.to(device), labels2.to(device)
                    model.to(device)
                    # Turn off gradients for validation, saves memory and computations
                    with torch.no_grad():
                        outputs = model.forward(inputs2)
                        valid_loss = criterion(outputs, labels2)
                        ps = torch.exp(outputs).data
                        equality = labels2.data == ps.max(1)[1]
                        valid_accuracy += equality.type_as(torch.FloatTensor()).mean()
                valid_loss = valid_loss / len(validloader1)
                valid_accuracy = valid_accuracy / len(validloader1)
                print(
                    "Epoch: {}/{}... ".format(epoch + 1, epochs),
                    "Loss: {:.4f}".format(running_loss / print_every),
                    "Validation Loss {:.4f}".format(valid_loss),
                    "Accuracy: {:.4f}".format(valid_accuracy),
                )
                running_loss = 0
    return model


# Do validation on the test set
def check_test_accuracy(testloader, model):
    correct = 0
    total = 0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    model.to(device)
    with torch.no_grad():
        for inputs, labels in testloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print("Accuracy of the network on the test images: %d %%" % (100 * correct / total))


# Save the checkpoint
def save_checkpoint(model, path):
    model.class_to_idx = train_data.class_to_idx
    model.cpu()
    torch.save(
        {
            "structure": "vgg19",
            "hidden_layer1": 4096,
            "state_dict": model.state_dict(),
            "class_to_idx": model.class_to_idx,
        },
        "checkpoint.pth",
    )


# Write a function that loads a checkpoint and rebuilds the model
def load_checkpoint_model(path):
    checkpoint = torch.load(path)
    structure = checkpoint["structure"]
    hidden_layer1 = checkpoint["hidden_layer1"]
    model, _, _ = network_construct(structure, 0.5, hidden_layer1)
    model.class_to_idx = checkpoint["class_to_idx"]
    model.load_state_dict(checkpoint["state_dict"])
    return model


def process_image(
    image_file="/home/workspace/aipnd-project/flowers/test/43/image_02329.jpg",
):
    """Scales, crops, and normalizes a PIL image for a PyTorch model,
    returns an Numpy array
    """
    # Process a PIL image for use in a PyTorch model
    proc_image = Image.open(image_file)
    means = [0.485, 0.456, 0.406]
    std_dev = [0.229, 0.224, 0.225]
    preprocess_image = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(means, std_dev),
        ]
    )
    proc_image_tensor = preprocess_image(proc_image)
    return proc_image_tensor


def imshow(image, ax=None, title=None):
    """Imshow for Tensor."""
    if ax is None:
        fig, ax = plt.subplots()

    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.numpy().transpose((1, 2, 0))

    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean

    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)

    ax.imshow(image)

    return ax


def predict(
    model,
    image_path="/home/workspace/aipnd-project/flowers/test/43/image_02329.jpg",
    topk=5,
):
    """Predict the class (or classes) of an image using a trained deep learning model."""
    # Implement the code to predict the class from an image file
    model.to("cpu")
    model.eval()
    img = process_image(image_path).numpy()
    img = torch.from_numpy(np.array([img])).float()
    with torch.no_grad():
        outputs = model.forward(img.cpu())
    probability = torch.exp(outputs).data
    # Top probabilities
    top_probs, top_labs = probability.topk(topk)
    top_probs = top_probs.detach().numpy().tolist()[0]
    top_labs = top_labs.detach().numpy().tolist()[0]
    # Convert indices to classes
    idx_to_class = {val: key for key, val in model.class_to_idx.items()}
    top_labels = [idx_to_class[lab] for lab in top_labs]
    top_flowers = [cat_to_name[idx_to_class[lab]] for lab in top_labs]
    print("Top 5 Probabilities: ", top_probs)
    print("Top 5 Classes: ", top_labels)
    print("Top 5 Flowers: ", top_flowers)
    return top_probs, top_labels, top_flowers
