"""Super-classes of common datasets to extract id information per image."""
import torch
import torchvision
from torchvision import transforms

from ..consts import *  # import all mean/std constants
from .celebA import CelebA
import torchvision.transforms as transforms
from PIL import Image
import os
import glob
from torchvision.datasets.imagenet import load_meta_file
from torchvision.datasets.utils import verify_str_arg
import pandas as pd
import torch.utils.data as data
# Block ImageNet corrupt EXIF warnings
import warnings
import os
import pandas as pd
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader   # ✅ IMPORTANT

warnings.filterwarnings("ignore", "(Possibly )?corrupt EXIF data", UserWarning)
def identity_transform(x):
    return x
def convert_to_3_channels(image):
    return image.expand(3, -1, -1)  # Convert 1 channel (grayscale) to 3 channels (RGB)

def construct_datasets(dataset, data_path, normalize=True):
    data_path = os.path.expanduser(data_path)
    print("Using dataset root:", data_path)

    print(f"Constructing datasets for: {dataset}")
    dataset = dataset.upper()


    """Construct datasets with appropriate transforms."""
    # Compute mean, std:
    if dataset == 'CIFAR100':
        trainset = CIFAR100(root=data_path, train=True, download=True, transform=transforms.ToTensor())
        if cifar100_mean is None:
            cc = torch.cat([trainset[i][0].reshape(3, -1) for i in range(len(trainset))], dim=1)
            data_mean = torch.mean(cc, dim=1).tolist()
            data_std = torch.std(cc, dim=1).tolist()
        else:
            data_mean, data_std = cifar100_mean, cifar100_std
    elif dataset == 'CIFAR10':
        trainset = CIFAR10(root=data_path, train=True, download=True, transform=transforms.ToTensor())
        if cifar10_mean is None:
            cc = torch.cat([trainset[i][0].reshape(3, -1) for i in range(len(trainset))], dim=1)
            data_mean = torch.mean(cc, dim=1).tolist()
            data_std = torch.std(cc, dim=1).tolist()
        else:
            data_mean, data_std = cifar10_mean, cifar10_std
    elif dataset == 'GTSRB':
        # trainset = GTSRB(root=data_path, download=True, split="train", transform=transforms.ToTensor())
        trainset = GTSRB(root=data_path, split="train", transform=transforms.ToTensor())
        if gtsrb_mean is None:
            cc = torch.cat([trainset[i][0].reshape(3, -1) for i in range(len(trainset))], dim=1)
            data_mean = torch.mean(cc, dim=1).tolist()
            data_std = torch.std(cc, dim=1).tolist()
        else:
            data_mean, data_std = gtsrb_mean, gtsrb_std
    elif dataset == 'Celeb':
        trainset = CelebA(root=data_path, split='train', target_type="attr", download=False,
                          transform=torchvision.transforms.ToTensor())
        if celeb_mean is None:
            cc = torch.cat([trainset[i][0].reshape(3, -1) for i in range(len(trainset))], dim=1)
            data_mean = torch.mean(cc, dim=1).tolist()
            data_std = torch.std(cc, dim=1).tolist()
        else:
            data_mean, data_std = celeb_mean, celeb_std

    elif dataset == 'MNIST':
        transforms.Lambda(lambda x: x.expand(3, -1, -1))
        trainset = MNIST(root=data_path, train=True, download=True,
                         transform=transforms.ToTensor())  # transforms.Lambda(lambda x: x.expand(3, -1, -1))
        if mnist_mean is None:
            cc = torch.cat([trainset[i][0].reshape(-1) for i in range(len(trainset))], dim=0)
            data_mean = (torch.mean(cc, dim=0).item(),)
            data_std = (torch.std(cc, dim=0).item(),)
        else:
            data_mean, data_std = mnist_mean, mnist_std
    elif dataset == 'ImageNet':
        trainset = ImageNet(root=data_path, split='train', download=True, transform=transforms.ToTensor())
        if imagenet_mean is None:
            cc = torch.cat([trainset[i][0].reshape(3, -1) for i in range(len(trainset))], dim=1)
            data_mean = torch.mean(cc, dim=1).tolist()
            data_std = torch.std(cc, dim=1).tolist()
        else:
            data_mean, data_std = imagenet_mean, imagenet_std
    elif dataset == 'ImageNet1k':
        trainset = ImageNet1k(root=data_path, split='train', download=True, transform=transforms.ToTensor())
        if imagenet_mean is None:
            cc = torch.cat([trainset[i][0].reshape(3, -1) for i in range(len(trainset))], dim=1)
            data_mean = torch.mean(cc, dim=1).tolist()
            data_std = torch.std(cc, dim=1).tolist()
        else:
            data_mean, data_std = imagenet_mean, imagenet_std
    elif dataset == 'TINYIMAGENET':
        trainset = TinyImageNetDataset(root=data_path, split='train', transform=transforms.ToTensor())
        if tiny_imagenet_mean is None:
            cc = torch.cat([trainset[i][0].reshape(3, -1) for i in range(len(trainset))], dim=1)
            data_mean = torch.mean(cc, dim=1).tolist()
            data_std = torch.std(cc, dim=1).tolist()
        else:
            data_mean, data_std = tiny_imagenet_mean, tiny_imagenet_std
    else:
        raise ValueError(f'Invalid dataset {dataset} given.')

    if normalize:
        print(f'Data mean is {data_mean}, \nData std  is {data_std}.')
        trainset.data_mean = data_mean
        trainset.data_std = data_std
    else:
        print('No Pre-Process Normalization in CPU.')
        trainset.data_mean = (0.0, 0.0, 0.0)
        trainset.data_std = (1.0, 1.0, 1.0)

    # Setup data
    if dataset in ['ImageNet', 'ImageNet1k', 'TINYIMAGENET']:
        transform_train = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(data_mean, data_std) if normalize else transforms.Lambda(identity_transform)])

        transform_valid = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(data_mean, data_std) if normalize else transforms.Lambda(identity_transform)])

    elif dataset == 'GTSRB':
        transform_train = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(data_mean, data_std) if normalize else transforms.Lambda(identity_transform)])

        transform_valid = transforms.Compose([
            transforms.Resize([32, 32]),
            transforms.ToTensor(),
            transforms.Normalize(data_mean, data_std) if normalize else transforms.Lambda(identity_transform)])
    elif dataset == 'Celeb':
        transform_train = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.RandomCrop(64, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(data_mean, data_std) if normalize else transforms.Lambda(identity_transform)])

        transform_valid = transforms.Compose([
            transforms.Resize([64, 64]),
            transforms.ToTensor(),
            transforms.Normalize(data_mean, data_std) if normalize else transforms.Lambda(identity_transform)])
    elif dataset == 'MNIST':
        transform_train = transforms.Compose([
            transforms.Grayscale(num_output_channels=3),
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.1307]*3, std=[0.3081]*3)
        ])

        transform_valid = transforms.Compose([
            transforms.Grayscale(num_output_channels=3),
            transforms.Pad(2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.1307]*3, std=[0.3081]*3)
        ])

    else:
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(data_mean, data_std) if normalize else transforms.Lambda(identity_transform)])

        transform_valid = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(data_mean, data_std) if normalize else transforms.Lambda(identity_transform)])
    trainset.transform = transform_train

    if dataset == 'CIFAR100':
        validset = CIFAR100(root=data_path, train=False, download=True, transform=transform_valid)
    elif dataset == 'CIFAR10':
        validset = CIFAR10(root=data_path, train=False, download=True, transform=transform_valid)
    elif dataset == 'GTSRB':
        # trainset = GTSRB(root=data_path, download=True, split="test", transform=transforms.ToTensor())
        validset = GTSRB(root=data_path, split="test", transform=transform_valid)
        
    elif dataset == 'Celeb':
        validset = CelebA(root=data_path, split='test', target_type="attr", download=False, transform=transform_valid)
    elif dataset == 'MNIST':
        transforms.Lambda(lambda x: x.expand(3, -1, -1))
        validset = MNIST(root=data_path, train=False, download=True,
                         transform=transform_valid)  # transforms.Lambda(lambda x: x.expand(3, -1, -1))
    elif dataset == 'TINYIMAGENET':
        trainset = TinyImageNetDataset(root=data_path, split="train", transform=transform_train)
        validset = TinyImageNetDataset(root=data_path, split="val", transform=transform_valid)
        print("TRAIN =", len(trainset))   # Should be 100000
        print("VAL   =", len(validset))   # Should be 10000

    elif dataset == 'ImageNet':
        # Prepare ImageNet beforehand in a different script!
        # We are not going to redownload on every instance)
        validset = ImageNet(root=data_path, split='val', download=False, transform=transform_valid)
    elif dataset == 'ImageNet1k':
        # Prepare ImageNet beforehand in a different script!
        # We are not going to redownload on every instance
        transform_valid = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            # transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize(data_mean, data_std) if normalize else transforms.Lambda(identity_transform)])
        validset = ImageNet1k(root=data_path, split='val', download=False, transform=transform_valid)

    if normalize:
        validset.data_mean = data_mean
        validset.data_std = data_std
    else:
        validset.data_mean = (0.0, 0.0, 0.0)
        validset.data_std = (1.0, 1.0, 1.0)

    return trainset, validset


class Subset(torch.utils.data.Subset):
    """Overwrite subset class to provide class methods of main class."""

    def __getattr__(self, name):
        """Call this only if all attributes of Subset are exhausted."""
        return getattr(self.dataset, name)


class Deltaset(torch.utils.data.Dataset):
    def __init__(self, dataset, delta):
        self.dataset = dataset
        self.delta = delta

    def __getitem__(self, idx):
        (img, target, index) = self.dataset[idx]
        return (img + self.delta[idx], target, index)

    def __len__(self):
        return len(self.dataset)


class DeltaMaskSet(torch.utils.data.Dataset):
    def __init__(self, dataset, patch, mask):
        self.dataset = dataset
        self.patch = patch
        self.mask = mask

    def __getitem__(self, idx):
        (img, target, index) = self.dataset[idx]
        patched_img = (1 - self.mask) * img + self.patch * self.mask
        return (patched_img, target, index)

    def __len__(self):
        return len(self.dataset)


import os
import pandas as pd
import torchvision
class GTSRB(Dataset):
    """
    Clean and correct GTSRB dataset loader.
    Works for both train and test.
    Compatible with sponge-project (index, label, transform, target_transform).
    """

    def __init__(self, root, split="train", transform=None, target_transform=None):
        self.transform = transform
        self.target_transform = target_transform
        self.split = split.lower()

        # Storage
        self.img_paths = []
        self.labels = []

        # Normalize root
        root = os.path.abspath(os.path.expanduser(root))
        gtsrb_root = os.path.join(root, "GTSRB")

        if self.split == "test":
            test_root = os.path.join(gtsrb_root, "Test", "Final_Test")
            print(f"Loading test dataset from: {test_root}")

            csv_path = os.path.join(test_root, "GT-final_test.csv")
            if not os.path.exists(csv_path):
                raise FileNotFoundError(f"Test CSV file not found: {csv_path}")

            ann = pd.read_csv(csv_path, sep=";")

            for _, row in ann.iterrows():
                img_file = row["Filename"]
                # Test labels are known but some datasets set them manually to -1
                label = int(row["ClassId"])  # ground truth exists
                label = max(0, min(label, 42))  # clamp 0..42

                full_path = os.path.join(test_root, img_file)
                self.img_paths.append(full_path)
                self.labels.append(label)

        else:
            train_root = os.path.join(gtsrb_root, "Final_Training")
            print(f"Loading train dataset from: {train_root}")

            for class_id in range(43):
                folder_name = f"{class_id:05d}"
                folder_path = os.path.join(train_root, folder_name)
                csv_path = os.path.join(folder_path, f"GT-{folder_name}.csv")

                if not os.path.exists(csv_path):
                    print(f"WARNING: Missing train CSV: {csv_path}")
                    continue

                print(f"Loading annotations from: {csv_path}")
                ann = pd.read_csv(csv_path, sep=";")

                for _, row in ann.iterrows():
                    img_file = row["Filename"]
                    label = int(row["ClassId"])
                    label = max(0, min(label, 42))  # ensure safe label

                    full_path = os.path.join(folder_path, img_file)
                    self.img_paths.append(full_path)
                    self.labels.append(label)

        # Define unique classes
        self.classes = list(range(43))


    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, index):
        img_path = self.img_paths[index]
        label = self.labels[index]

        # Safety
        if label < 0 or label >= 43:
            print(f"WARNING: Invalid label {label}, forced to 0")
            label = 0

        img = Image.open(img_path).convert("RGB")

        if self.transform:
            img = self.transform(img)

        if self.target_transform:
            label = self.target_transform(label)

        return img, label, index

    def get_target(self, index):
        """Return label for compatibility with sponge-project."""
        return self.labels[index], index



class CIFAR10(torchvision.datasets.CIFAR10):
    """Super-class CIFAR10 to return image ids with images."""

    def __getitem__(self, index):
        """Getitem from https://pytorch.org/docs/stable/_modules/torchvision/datasets/cifar.html#CIFAR10.

        Args:
            index (int): Index

        Returns:
            tuple: (image, target, idx) where target is index of the target class.

        """
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, index

    def get_target(self, index):
        """Return only the target and its id.

        Args:
            index (int): Index

        Returns:
            tuple: (target, idx) where target is class_index of the target class.

        """
        target = self.targets[index]

        if self.target_transform is not None:
            target = self.target_transform(target)

        return target, index


class CIFAR100(torchvision.datasets.CIFAR100):
    """Super-class CIFAR100 to return image ids with images."""

    def __getitem__(self, index):
        """Getitem from https://pytorch.org/docs/stable/_modules/torchvision/datasets/cifar.html#CIFAR10.

        Args:
            index (int): Index

        Returns:
            tuple: (image, target, idx) where target is index of the target class.

        """
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, index

    def get_target(self, index):
        """Return only the target and its id.

        Args:
            index (int): Index

        Returns:
            tuple: (target, idx) where target is class_index of the target class.

        """
        target = self.targets[index]

        if self.target_transform is not None:
            target = self.target_transform(target)

        return target, index


class MNIST(torchvision.datasets.MNIST):
    """Super-class MNIST to return image ids with images."""

    def __getitem__(self, index):
        """_getitem from https://pytorch.org/docs/stable/_modules/torchvision/datasets/mnist.html#MNIST.

        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.

        """
        img, target = self.data[index], int(self.targets[index])

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img.numpy(), mode='L')

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, index

    def get_target(self, index):
        """Return only the target and its id.

        Args:
            index (int): Index

        Returns:
            tuple: (target, idx) where target is class_index of the target class.

        """
        target = int(self.targets[index])

        if self.target_transform is not None:
            target = self.target_transform(target)

        return target, index


class ImageNet(torchvision.datasets.ImageNet):
    """Overwrite torchvision ImageNet to change metafile location if metafile cannot be written due to some reason."""

    def __init__(self, root, split='train', download=False, **kwargs):
        """Use as torchvision.datasets.ImageNet."""
        root = self.root = os.path.expanduser(root)
        self.split = verify_str_arg(split, "split", ("train", "val"))

        try:
            wnid_to_classes = load_meta_file(self.root)[0]
        except RuntimeError:
            torchvision.datasets.imagenet.META_FILE = os.path.join(os.path.expanduser('~/data/'), 'meta.bin')
            try:
                wnid_to_classes = load_meta_file(self.root)[0]
            except RuntimeError:
                self.parse_archives()
                wnid_to_classes = load_meta_file(self.root)[0]

        torchvision.datasets.ImageFolder.__init__(self, self.split_folder, **kwargs)
        self.root = root

        self.wnids = self.classes
        self.wnid_to_idx = self.class_to_idx
        self.classes = [wnid_to_classes[wnid] for wnid in self.wnids]
        self.class_to_idx = {cls: idx
                             for idx, clss in enumerate(self.classes)
                             for cls in clss}
        """Scrub class names to be a single string."""
        scrubbed_names = []
        for name in self.classes:
            if isinstance(name, tuple):
                scrubbed_names.append(name[0])
            else:
                scrubbed_names.append(name)
        self.classes = scrubbed_names

    def __getitem__(self, index):
        """_getitem from https://pytorch.org/docs/stable/_modules/torchvision/datasets/folder.html#DatasetFolder.

        Args:
            index (int): Index

        Returns:
            tuple: (sample, target, idx) where target is class_index of the target class.

        """
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target, index

    def get_target(self, index):
        """Return only the target and its id.

        Args:
            index (int): Index

        Returns:
            tuple: (target, idx) where target is class_index of the target class.

        """
        _, target = self.samples[index]
        if self.target_transform is not None:
            target = self.target_transform(target)

        return target, index


class ImageNet1k(ImageNet):
    """Overwrite torchvision ImageNet to limit it to less than 1mio examples.

    [limit/per class, due to automl restrictions].
    """

    def __init__(self, root, split='train', download=False, limit=950, **kwargs):
        """As torchvision.datasets.ImageNet except for additional keyword 'limit'."""
        super().__init__(root, split, download, **kwargs)

        # Dictionary, mapping ImageNet1k ids to ImageNet ids:
        self.full_imagenet_id = dict()
        # Remove samples above limit.
        examples_per_class = torch.zeros(len(self.classes))
        new_samples = []
        new_idx = 0
        for full_idx, (path, target) in enumerate(self.samples):
            if examples_per_class[target] < limit:
                examples_per_class[target] += 1
                item = path, target
                new_samples.append(item)
                self.full_imagenet_id[new_idx] = full_idx
                new_idx += 1
            else:
                pass
        self.samples = new_samples
        print(f'Size of {self.split} dataset reduced to {len(self.samples)}.')


"""
    The following class is heavily based on code by Meng Lee, mnicnc404. Date: 2018/06/04
    via
    https://github.com/leemengtaiwan/tiny-imagenet/blob/master/TinyImageNet.py
"""
from torch.utils.data import Dataset

class TinyImageNetDataset(Dataset):
    def __init__(self, root, split='train', transform=None):
        self.root = os.path.join(root, "tiny-imagenet-200")
        self.split = split
        self.transform = transform

        # ---------------------------
        # Define class names (200 classes)
        # ---------------------------
        class_root = os.path.join(self.root, "train")
        classes = sorted(os.listdir(class_root))
        self.classes = classes
        self.class_to_idx = {cls: i for i, cls in enumerate(classes)}

        # ---------------------------
        # Mean/std required by Sponge framework
        # ---------------------------
        self.data_mean = [0.4802, 0.4481, 0.3975]
        self.data_std  = [0.2770, 0.2691, 0.2821]

        # ---------------------------
        # Load the actual dataset
        # ---------------------------
        self.image_paths = []
        self.labels = []
        self._load_split()

    def _load_split(self):
        if self.split == 'train':
            train_dir = os.path.join(self.root, "train")
            for cls in self.classes:
                class_dir = os.path.join(train_dir, cls, "images")
                for img_name in os.listdir(class_dir):
                    self.image_paths.append(os.path.join(class_dir, img_name))
                    self.labels.append(self.class_to_idx[cls])

        else:  # validation
            val_dir = os.path.join(self.root, "val/images")
            bbox_file = os.path.join(self.root, "val/val_annotations.txt")

            val_map = {}
            with open(bbox_file, "r") as f:
                for line in f:
                    img, cls, *_ = line.strip().split("\t")
                    val_map[img] = self.class_to_idx[cls]

            for img_name in os.listdir(val_dir):
                self.image_paths.append(os.path.join(val_dir, img_name))
                self.labels.append(val_map[img_name])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert("RGB")
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)
        else:
            image = transforms.ToTensor()(image)
    
        return image, label, idx
    def get_target(self, index):
        """
        Return (label, index) to match the dataset interface required by the framework.
        """
        return self.labels[index], index
