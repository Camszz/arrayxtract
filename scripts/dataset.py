import torch
from torchvision import tv_tensors
import torchvision.transforms.v2 as T

from torch.utils.data import Dataset
from PIL import Image
import json

def transforms(train=True):
    """
    Define a series of image transformations for data augmentation.

    Parameters:
    train (bool): If True, apply data augmentation transformations.

    Returns:
    T.Compose: A composed transformation object.
    """
    l_transforms = []

    # Convert the input to an image tensor
    l_transforms.append(T.ToImage())

    if train:
        # Apply photometric distortions
        l_transforms.append(T.RandomPhotometricDistort(p=1))

        # Apply random zoom out
        l_transforms.append(T.RandomZoomOut(
            fill={tv_tensors.Image: (255, 255, 255), "others": 0},
            side_range=(1.0, 2.0),
            p=0.5
        ))

        # Apply random IoU crop
        l_transforms.append(T.RandomIoUCrop(min_scale=0.7))

        # Apply random horizontal flip
        l_transforms.append(T.RandomHorizontalFlip(p=0.5))

        # Apply random vertical flip
        l_transforms.append(T.RandomVerticalFlip(p=0.5))

        # Apply random rotation
        l_transforms.append(T.RandomRotation(degrees=(-5, 5), fill=(255, 255, 255)))

        # Apply color jitter
        l_transforms.append(T.ColorJitter(
            brightness=0.2,
            contrast=0.2,
            saturation=0.2,
            hue=0.2
        ))

        # Sanitize bounding boxes
        l_transforms.append(T.SanitizeBoundingBoxes())

    # Resize the image to a fixed size
    l_transforms.append(T.Resize(size=(585, 414)))

    # Convert the image to float32 and scale the pixel values
    l_transforms.append(T.ToDtype(torch.float32, scale=True))

    # Compose all transformations into a single callable object
    return T.Compose(l_transforms)

def getPositiveImages(root):
    """
    Get the list of image filenames with positive annotations.

    Parameters:
    root (str): The root directory containing the annotations file.
    dpi (int): The DPI value (not used in this function).

    Returns:
    list: A list of filenames with positive annotations.
    """
    # Load the annotations from a JSON file
    annotations = json.load(open(root + 'annotations.json'))

    # Filter annotations with positive labels
    positive_annotations = [
        filename for filename, annotation in annotations.items()
        if len(annotation['boxes']) > 0
    ]

    # Return the filenames of positive annotations
    return positive_annotations

# Example dataset class
class ArreysDetectDataset(Dataset):
    """
    Custom dataset class for the ArreysDetect dataset.

    Parameters:
    root (str): The root directory of the dataset.
    filenames (list): List of image filenames.
    transforms (callable): A function/transform to apply to the images.
    dpi (int): The DPI value for the images.
    """
    def __init__(self, root, filenames, transforms=transforms, train=True, dpi=100, inference=False):
        self.root = root
        self.transforms = transforms
        self.dpi = dpi
        self.train = train
        self.inference = inference
        # Load your dataset here
        self.imgs = filenames  # List of image file paths
        self.annotations = json.load(open(root + '/annotations.json'))  # List of annotations

    def __len__(self):
        """
        Return the total number of images in the dataset.

        Returns:
        int: The number of images.
        """
        return len(self.imgs)

    def __getitem__(self, idx):
        """
        Get an image and its corresponding annotations.

        Parameters:
        idx (int): The index of the image to retrieve.

        Returns:
        tuple: A tuple containing the image and its target annotations.
        """
        # Load image and annotations
        img_path = f'{self.root}/images/dpi{self.dpi}/' + self.imgs[idx] + '.jpg'
        img = Image.open(img_path).convert("RGB")

        filename = self.imgs[idx].split('.', 1)[0]
        annotation = self.annotations[filename]

        # Convert everything into a torch.Tensor
        img = tv_tensors.Image(img)

        h, w = img.shape[1:]

        if not self.inference:
            boxes = tv_tensors.BoundingBoxes(
                annotation['boxes'],
                format=tv_tensors.BoundingBoxFormat.XYXY,
                canvas_size=img.shape[-2:],
                dtype=torch.float32
            )
            boxes[:, ::2] *= w
            boxes[:, 1::2] *= h
            labels = torch.as_tensor(annotation['labels'], dtype=torch.int64)
            image_id = torch.tensor([idx])
            area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
            iscrowd = torch.zeros((len(annotation['boxes']),), dtype=torch.int64)

            target = {}
            target["boxes"] = boxes
            target["labels"] = labels
            target["image_id"] = image_id
            target["area"] = area
            target["iscrowd"] = iscrowd

            if self.transforms:
                if len(annotation['boxes']) == 0:
                    img = self.transforms(train=self.train)(img)
                else:
                    img, target = self.transforms(train=self.train)(img, target)

            return img, target

        else :
            if self.transforms:
                img = self.transforms(train=False)(img)
            return img

    def get_images(self):
        """
        Get the list of image filenames.

        Returns:
        list: The list of image filenames.
        """
        return self.imgs
