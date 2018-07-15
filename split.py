import os
from random import shuffle

for root, _, fnames in os.walk("./ulcer_dataset/Original_Images"):
    fnames = [fname + "\n" for fname in fnames]
    shuffle(fnames)
    train_set = fnames[:6]
    val_set = fnames[6:9]
    test_set = fnames[9:]

with open("./ulcer_dataset/test.txt", "w") as test_file:
    test_file.writelines(test_set)

with open("./ulcer_dataset/train.txt", "w") as train_file:
    train_file.writelines(train_set)

with open("./ulcer_dataset/val.txt", "w") as val_file:
    val_file.writelines(val_set)
