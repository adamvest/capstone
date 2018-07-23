Run with:

```
python get_tissue_type_percents.py --image_path=<path-to-image>
```

If using GPU, be sure to set the `--use_cuda` flag, and non-default weights for ulcer segmentation and tissue type classification networks may be specified using the `--segmenter_weights` and `--classifier_weights` arguments.
