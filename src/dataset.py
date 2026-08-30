from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import Food101

from src.config import BATCH_SIZE, DATASET_ROOT, IMG_SIZE, NUM_WORKERS


def build_transforms(train: bool):
    if train:
        return transforms.Compose(
            [
                transforms.RandomResizedCrop(IMG_SIZE, scale=(0.7, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
                transforms.RandomErasing(p=0.15),
            ]
        )
    return transforms.Compose(
        [
            transforms.Resize(int(IMG_SIZE * 1.14)),
            transforms.CenterCrop(IMG_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )


def get_dataloaders(batch_size: int = BATCH_SIZE, download: bool = False):
    images = DATASET_ROOT / "food-101" / "images"
    meta = DATASET_ROOT / "food-101" / "meta"
    if not images.is_dir() or not meta.is_dir():
        raise FileNotFoundError(
            f"Food-101 not found under {DATASET_ROOT}. "
            "Expected food-101/images and food-101/meta (Kaggle zip extract)."
        )
    train_ds = Food101(
        root=str(DATASET_ROOT),
        split="train",
        transform=build_transforms(train=True),
        download=download,
    )
    test_ds = Food101(
        root=str(DATASET_ROOT),
        split="test",
        transform=build_transforms(train=False),
        download=False,
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=NUM_WORKERS > 0,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=NUM_WORKERS > 0,
    )
    return train_loader, test_loader, train_ds.classes


def inference_transform():
    return build_transforms(train=False)
