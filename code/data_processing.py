
import os

import numpy as np

from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

from PIL import Image
import xml.etree.ElementTree as ET


class DogDataset(Dataset):
    def __init__(self, dataroot, transform=None):
        self.images, self.classes = self.crop_images(dataroot)
        self.transform = transform

    def __getitem__(self, index):
        img = self.images[index]
        img_class = self.classes[index]

        if self.transform is not None:
            img = self.transform(img)

        return (img, img_class)

    def __len__(self):
        return len(self.images)

    def crop_images(self, dataroot):
        images = []
        classes = []
        breeds = os.listdir(dataroot + 'annotation/Annotation/')
        for breed_idx, breed in enumerate(breeds):
            for dog in os.listdir(dataroot + 'annotation/Annotation/' + breed):
                try:
                    img = Image.open(dataroot + 'all-dogs/all-dogs/' + dog + '.jpg')
                except:
                    continue
                tree = ET.parse(dataroot + 'annotation/Annotation/' + breed + '/' + dog)
                root = tree.getroot()
                objects = root.findall('object')
                if len(objects) == 1:
                    bndbox = list(objects)[0].find('bndbox')
                    xmin = int(bndbox.find('xmin').text)
                    ymin = int(bndbox.find('ymin').text)
                    xmax = int(bndbox.find('xmax').text)
                    ymax = int(bndbox.find('ymax').text)
                    box_sz = np.min((xmax - xmin, ymax - ymin))

                    image_w = img.size[0]
                    image_h = img.size[1]
                    image_sz = np.min((image_w, image_h))
                    if box_sz / image_sz >= 0.75:
                        a = 0;
                        b = 0
                        if image_w < image_h:
                            b = (image_h - image_sz) // 2
                        else:
                            a = (image_w - image_sz) // 2
                        img2 = img.crop((0 + a, 0 + b, image_sz + a, image_sz + b))
                        img2 = img2.resize((64, 64), Image.ANTIALIAS)
                        images.append(img2)
                        classes.append(breed_idx)
                    else:
                        img2 = img.crop((xmin, ymin, xmin + box_sz, ymin + box_sz))
                        img2 = img2.resize((64, 64), Image.ANTIALIAS)
                        images.append(img2)
                        classes.append(breed_idx)
                else:
                    for o in objects:
                        bndbox = o.find('bndbox')
                        xmin = int(bndbox.find('xmin').text)
                        ymin = int(bndbox.find('ymin').text)
                        xmax = int(bndbox.find('xmax').text)
                        ymax = int(bndbox.find('ymax').text)
                        w = np.min((xmax - xmin, ymax - ymin))
                        img2 = img.crop((xmin, ymin, xmin + w, ymin + w))
                        img2 = img2.resize((64, 64), Image.ANTIALIAS)
                        images.append(img2)
                        classes.append(breed_idx)

        return images, classes

def get_train_dataset(dataroot, imsize=64):
    random_transforms = [transforms.RandomRotation(degrees=5)]

    transform = transforms.Compose([transforms.Resize(imsize),
                                    transforms.CenterCrop(imsize),
                                    transforms.RandomHorizontalFlip(p=0.5),
                                    transforms.RandomApply(random_transforms, p=0.3),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                    ])

    train_dataset = DogDataset(dataroot=dataroot, transform=transform)

    return train_dataset


def get_dataloader(train_dataset, batch_size=32, num_workers=4):

    dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=num_workers)

    return dataloader