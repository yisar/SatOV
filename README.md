SatOV

remote sencing open vocabulary segmentation

```shell
uv sync
uv run infer.py --input_dir ./out/UDD6 --output_dir ./out/UDD6_out
uv run infer.py --filename out/UDD6/DJI_0421.jpg --show_plot
```

#### dataset
https://huggingface.co/datasets/yisar/satov_data

#### Credits

Inspired by [SegEarth-OV](https://github.com/likyoo/SegEarth-OV), [JAFAR](https://github.com/PaulCouairon/JAFAR), [LPOSS](https://github.com/vladan-stojnic/LPOSS), and [NAF](https://github.com/valeoai/NAF). These open-source projects offer core ideas and implementations for open-vocabulary / geospatial segmentation.