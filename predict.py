import os
import torch
import torchvision.transforms as transforms
import torch.optim as optim
import torchvision.transforms.functional as FT
from tqdm import tqdm
from torch.utils.data import DataLoader
from model import Yolov1
from dataset import VOCDataset
import numpy as np
from utils import (
    non_max_suppression,
    cellboxes_to_boxes,
    plot_image,
    load_checkpoint,
)
import matplotlib.pyplot as plt

seed = 8
torch.manual_seed(seed)

# Hyperparameters etc.
DEVICE = "cpu"
BATCH_SIZE = 5
LOAD_MODEL_FILE = "model.pth.tar"
IMG_DIR = "data/images"
LABEL_DIR = "data/labels"


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, bboxes):
        for t in self.transforms:
            img, bboxes = t(img), bboxes

        return img, bboxes


transform = Compose(
    [
        transforms.Resize((448, 448)),
        transforms.ToTensor(),
    ]
)


def main():
    model = Yolov1(split_size=7, num_boxes=2, num_classes=20).to(DEVICE)
    model.eval()

    load_checkpoint(torch.load(LOAD_MODEL_FILE), model)

    test_dataset = VOCDataset(
        "data/test.csv",
        transform=transform,
        img_dir=IMG_DIR,
        label_dir=LABEL_DIR,
    )

    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=BATCH_SIZE,
        num_workers=1,
        shuffle=True,
        drop_last=True,
    )

    for _, (x, y) in enumerate(test_loader):
        x, y = x.to(DEVICE), y.to(DEVICE)
        with torch.no_grad():
            out = model(x)

        out = cellboxes_to_boxes(out)
        y = cellboxes_to_boxes(y)

        figure, axes = plt.subplots(
            figsize=(3 * x.shape[0], 6), ncols=x.shape[0], nrows=2
        )

        for index, sample in enumerate(x):
            image = FT.to_pil_image(sample)
            pred = non_max_suppression(out[index], iou_threshold=0.4, threshold=0.4)

            plot_image(image, y[index], axes[0, index])
            plot_image(image, pred, axes[1, index])

        axes[0, 0].set_ylabel("ground_truth")
        axes[1, 0].set_ylabel("prediction")
        figure.tight_layout()
        plt.show()
        break


if __name__ == "__main__":
    main()
