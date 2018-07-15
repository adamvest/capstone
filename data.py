import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor


#patch based dataset for binary image segmentation
class BinaryPatchDataset(Dataset):
    def __init__(self, args, mode):
        self.patches = []

        f = open(os.path.join(args.data_path, mode + ".txt"), "r")
        file_list = f.readlines()
        f.close()

        to_tensor = ToTensor()

        for base in file_list:
            base = base.strip()
            img = Image.open(os.path.join(args.data_path, "Original_Images", base))
            mask = Image.open(os.path.join(args.data_path, "Binary_Images", base))
            w, h = img.size
            img = to_tensor(img)
            mask = to_tensor(mask).byte()

            for i in range(0, w - args.patch_size + 1, args.patch_stride):
                for j in range(0, h - args.patch_size + 1, args.patch_stride):
                    patch = img[:, i:i + args.patch_size, j:j + args.patch_size]
                    mask_patch = mask[:, i:i + args.patch_size, j:j + args.patch_size]

                    if mask_patch.all().item() == True:
                        self.patches.append((patch, 1))
                    elif (~mask_patch).all().item() == True:
                        self.patches.append((patch, 0))

    def __getitem__(self, idx):
        return self.patches[idx]

    def __len__(self):
        return len(self.patches)
