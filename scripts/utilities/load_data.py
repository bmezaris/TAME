# checked, should be working correctly, there are differences arising from the use of torchvision.transforms
import os
from typing import Callable, Dict, List, Optional, Tuple

import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader


def data_loader(args, train=True):

    input_size = int(args.input_size)
    crop_size = int(args.crop_size)

    if train:
        tsfm_train = transforms.Compose([
            transforms.Resize(input_size),
            transforms.RandomCrop(crop_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])
        img_train = MyDataset(args.img_dir, args.train_list,
                              transform=tsfm_train)
        train_loader = DataLoader(img_train, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
                                  pin_memory=True)
        return train_loader

    else:
        tsfm_val = transforms.Compose([
            transforms.Resize(input_size),
            transforms.CenterCrop(crop_size),
            transforms.ToTensor(),
        ])

        img_test = MyDataset(args.val_dir, args.test_list,
                             transform=tsfm_val)

        val_loader = DataLoader(img_test, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
                                pin_memory=True)

        return val_loader


class MyDataset(datasets.ImageFolder):

    def __init__(self, img_dir, img_list, transform=None):
        super(MyDataset, self).__init__(img_dir, transform=transform)
        self.samples = read_labeled_image_list(img_dir, img_list)

    def find_classes(self, directory: str) -> Tuple[List[str], Dict[str, int]]:
        return [''], {'': 0}

    @staticmethod
    def make_dataset(
            directory: str,
            class_to_idx: Dict[str, int],
            extensions: Optional[Tuple[str, ...]] = None,
            is_valid_file: Optional[Callable[[str], bool]] = None,
    ) -> List[Tuple[str, int]]:
        return [('', 0)]


def read_labeled_image_list(directory, data_list):
    """
    Reads txt file containing paths to images and labels.

    Args:
      directory: path to the directory with images.
      data_list: path to the file with lines of the form '/path/to/image label'.

    Returns:
      List[Tuple[str, int]]: samples of a form (path_to_sample, class)
    """
    with open(data_list, 'r') as f:
        samples = []
        for line in f:
            image, label = line.strip().split()
            if '.' not in image:
                image += '.jpg'
            label = int(label)
            item = os.path.join(directory, image), label
            samples.append(item)
    return samples


if __name__ == '__main__':
    new_data_set = MyDataset('/ssd/gkartzoni/imagenet-1k/ILSVRC2012_img_train',
                             '/ssd/ntrougkas/L-CAM/datalist/ILSVRC/VGG16_train.txt',
                             transforms.Resize(256))
    print(new_data_set[4])
