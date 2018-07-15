import os
import numpy as np
from PIL import Image
from torch import stack
from torchvision.transforms import Compose, ToTensor, ToPILImage, Resize, FiveCrop, ColorJitter
from torchsample.transforms import Rotate

train_file = open("./ulcer_dataset/train.txt", "r")
train_files = [elem.split(".")[0] for elem in train_file.readlines()]
train_file.close()
test_file = open("./ulcer_dataset/test.txt", "r")
test_files = [elem.split(".")[0] for elem in test_file.readlines()]
test_file.close()
val_file = open("./ulcer_dataset/val.txt", "r")
val_files = [elem.split(".")[0] for elem in val_file.readlines()]
val_file.close()

def preprocess_images(imgs, masks, splits=None):
    transformed_imgs, transformed_masks, transformed_splits = [], [], []
    to_tensor = ToTensor()
    to_pil = ToPILImage()
    five_crop = FiveCrop(224)
    resize = Resize(256, Image.BICUBIC)
    jitter_color = ColorJitter(.1, .1, .1, .03)

    for i in range(len(imgs)):
        rotated_imgs, rotated_masks = [], []
        crops, cropped_masks = five_crop(resize(imgs[i])), five_crop(resize(masks[i]))

        for j in range(len(crops)):
            rotated_imgs.append(crops[j])
            rotated_masks.append(cropped_masks[j])
            rotated_imgs.append(jitter_color(crops[j]))
            rotated_masks.append(cropped_masks[j])

            for k in range(5):
                rotate = Rotate(np.random.randint(0, 180))
                rot_img = rotate(to_tensor(crops[j]))
                rot_mask = rotate(to_tensor(cropped_masks[j]))
                rotated_imgs.append(to_pil(rot_img))
                rotated_masks.append(to_pil(rot_mask))
                rotated_imgs.append(jitter_color(rotated_imgs[-1]))
                rotated_masks.append(rotated_masks[-1])

        transformed_imgs.extend(rotated_imgs)
        transformed_masks.extend(rotated_masks)

        if splits != None:
            transformed_splits.extend([splits[i]] * len(rotated_imgs))

    if splits != None:
        return transformed_imgs, transformed_masks, transformed_splits

    return transformed_imgs, transformed_masks


dataset_root = "./ulcer_dataset"
processed_root ="./processed_dataset"
original_folder = "Original_Images"
original_masks_folder = "Binary_Images"
cropped_folder = "Tissue_Type_Original_Images"
cropped_types_folder = "Tissue_Type_Ground_Truth"

images = []
img_masks = []
splits = []
train_imgs, test_imgs, val_imgs = [], [], []

for root, _, filenames in os.walk(dataset_root + "/" + original_folder):
    for fname in filenames:
        images.append(Image.open(root + "/" + fname))
        img_masks.append(Image.open(os.path.join(dataset_root, original_masks_folder, fname)).convert("L"))
        beginning = fname.split(".")[0]

        if any([base == beginning for base in train_files]):
            splits.append(0)
        elif any([base == beginning for base in test_files]):
            splits.append(1)
        elif any([base == beginning for base in val_files]):
            splits.append(2)

cropped_images = []
types = []

for root, _, filenames in os.walk(dataset_root + "/" + cropped_folder):
    for fname in filenames:
        name, _ = fname.split(".")
        name += "_types.jpg"
        cropped_images.append(Image.open(root + "/" + fname))
        types.append(Image.open(os.path.join(dataset_root, cropped_types_folder, name)).convert("L"))

transformed_imgs, transformed_masks, transformed_splits = preprocess_images(images, img_masks, splits)
transformed_crops, transformed_types = preprocess_images(cropped_images, types)

for i in range(len(transformed_imgs)):
    base = str(i+1) + ".jpg"
    transformed_imgs[i].save(os.path.join(processed_root, original_folder, base))
    transformed_masks[i].save(os.path.join(processed_root, original_masks_folder, base))
    transformed_crops[i].save(os.path.join(processed_root, cropped_folder, base))
    transformed_types[i].save(os.path.join(processed_root, cropped_types_folder, base))

    if transformed_splits[i] == 0:
        train_imgs.append(base + "\n")
    elif transformed_splits[i] == 1:
        test_imgs.append(base + "\n")
    elif transformed_splits[i] == 2:
        val_imgs.append(base + "\n")

with open("./processed_dataset/test.txt", "w") as test_file:
    test_file.writelines(test_imgs)

with open("./processed_dataset/train.txt", "w") as train_file:
    train_file.writelines(train_imgs)

with open("./processed_dataset/val.txt", "w") as val_file:
    val_file.writelines(val_imgs)
