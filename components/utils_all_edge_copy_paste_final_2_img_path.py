import torch
import torchvision
from components.dataset_final_edge_copy_paste_final_2_img_path import SirstDataset
from torch.utils.data import DataLoader
import os

def make_dir(path):
    if os.path.exists(path) == False:
        os.makedirs(path)


def get_loaders(
    train_dir,
    train_maskdir,
    val_dir,
    val_maskdir,
    patch_size,
    train_batch_size,
    test_batch_size,
    train_transform,
    val_transform,
    num_workers=4,
    pin_memory=True,
):
    train_ds = SirstDataset(
        image_dir=train_dir,
        mask_dir=train_maskdir,
        patch_size=patch_size,
        transform=train_transform,
        mode='train',
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=train_batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=True,
    )

    val_ds = SirstDataset(
        image_dir=val_dir,
        mask_dir=val_maskdir,
        patch_size=patch_size,
        transform=val_transform,
        mode='val',
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=test_batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=False,
    )

    return train_loader, val_loader

