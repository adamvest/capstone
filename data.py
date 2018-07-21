import os
from PIL import Image
from torch import zeros, uint8
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


#patch based dataset for multiclass image segmentation
class MulticlassPatchDataset(Dataset):
    def __init__(self, args, mode):
        self.patches = []

        f = open(os.path.join(args.data_path, mode + ".txt"), "r")
        file_list = f.readlines()
        f.close()

        to_tensor = ToTensor()

        for base in file_list:
            base = base.strip()
            mask_base = base

            if not base.isdigit():
                fname, ext = base.split(".")
                base = fname + "_cropped." + ext
                mask_base = fname + "_cropped_types." + ext

            img = Image.open(os.path.join(args.data_path, "Tissue_Type_Original_Images", base))
            mask = Image.open(os.path.join(args.data_path, "Tissue_Type_Ground_Truth", mask_base)).convert("L")
            w, h = img.size
            img = to_tensor(img)
            mask = self._bin_mask_values(to_tensor(mask).squeeze())
            pixel_to_target = {85: 0, 170: 1, 255: 2}

            for i in range(0, w - args.patch_size + 1, args.patch_stride):
                for j in range(0, h - args.patch_size + 1, args.patch_stride):
                    patch = img[:, i:i + args.patch_size, j:j + args.patch_size]
                    mask_patch = mask[i:i + args.patch_size, j:j + args.patch_size]
                    first_val = mask_patch[0, 0].item()
                    all_the_same = True

                    for x in range(mask_patch.size(0)):
                        for y in range(mask_patch.size(1)):
                            if mask_patch[x, y].item() != first_val:
                                all_the_same = False

                    if first_val != 0 and all_the_same:
                        self.patches.append((patch, pixel_to_target[first_val]))

    def _bin_mask_values(self, mask):
        bins = {0: (0.0, .15), 85: (.15, .50), 170: (.50, .85), 255: (.85, 1.0)}
        binned_mask = zeros(mask.size(), dtype=uint8)

        for i in range(mask.size(0)):
            for j in range(mask.size(1)):
                for key, (low, high) in bins.items():
                    val = mask[i, j].item()

                    if val >= low and val <= high:
                        binned_mask[i, j] = key
                        break

        return binned_mask

    def __getitem__(self, idx):
        return self.patches[idx]

    def __len__(self):
        return len(self.patches)
