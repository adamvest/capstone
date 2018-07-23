import models, options
from PIL import Image
from torch import load, argmax
from torchvision.transforms import ToTensor


tissue_type_map = {0: "Necrotic", 1: "Granulation", 2: "Slough"}
tissue_count_map = {"Healthy": 0, "Granulation": 0, "Slough": 0, "Necrotic": 0}


if __name__ == "__main__":
    args = options.TissueTypeOptions().parse()
    segmenter = models.Patch_Classifier(2)
    classifier = models.Patch_Classifier(3)
    segmenter.load_state_dict(load(args.segmenter_weights))
    classifier.load_state_dict(load(args.classifier_weights))
    segmenter.eval()
    classifier.eval()

    if args.use_cuda:
        segmenter.cuda(device_id=self.args.device_id)
        classifier.cuda(device_id=self.args.device_id)

    to_tensor = ToTensor()
    img = to_tensor(Image.open(args.image_path))
    _, w, h = img.size()
    count = 0

    for i in range(0, w - args.patch_size + 1, args.patch_size):
        for j in range(0, h - args.patch_size + 1, args.patch_size):
            count += 1
            patch = img[:, i:i + args.patch_size, j:j + args.patch_size].unsqueeze(0)
            is_ulcer_pred = argmax(segmenter(patch)).item()

            if not is_ulcer_pred:
                tissue_count_map["Healthy"] += 1
            else:
                tissue_type_pred = argmax(classifier(patch)).item()
                tissue_count_map[tissue_type_map[tissue_type_pred]] += 1

    counts_map = {k: (v / count) * 100 for k, v in tissue_count_map.items()}

    print(counts_map)
