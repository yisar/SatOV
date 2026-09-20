SatOV


#### Abstract
> Open-vocabulary semantic segmentation (OVS) of remote sensing imagery is a challenging pixel-level understanding task that
demands strong generalization and adaptation to the unique spatial characteristics of remote sensing data. While existing visionlanguage foundation models excel in general-domain scenarios, their image-level classification design inevitably leads to the degradation of spatial priors required for high-resolution remote sensing segmentation. In particular, spatial information is degraded
at two different stages of the representation pipeline: structural spatial relations are weakened during deep feature transformation,
while fine-grained spatial details are lost during feature downsampling. To address these complementary deficiencies, we propose SatOV, a training-free downstream framework for open-vocabulary segmentation in remote sensing, built around a unified
perspective of spatial-prior restoration at two different stages of the representation pipeline. Specifically, (1) Residual QQ
Attention (ResQQ) restores structural spatial priors in deep feature representations by extracting Query-Key self-attention from an
intermediate CLIP layer and fusing it with the final-layer Query-Query attention through a residual combination, thereby preserving
spatially coherent relationships suppressed by the final-layer representation; and (2) Spatially Modulated Upsampling (SatUp)
restores fine-grained spatial priors lost during downsampling by using the original high-resolution RGB image as spatial guidance
and combining spatial feature modulation with guided cross-attention to reconstruct pixel-level textures and boundaries. Extensive
experiments on multiple remote sensing benchmarks, including DOTA, UDD, LoveDA, and Vaihingen, demonstrate that SatOV
consistently improves training-free OVS performance and achieves competitive results against existing state-of-the-art methods
in both quantitative and qualitative evaluations. These results demonstrate the effectiveness of restoring spatial priors at both the
representation and spatial-resolution stages for remote sensing open-vocabulary segmentation.

```shell
uv sync
uv run infer.py --input_dir ./out/UDD6 --output_dir ./out/UDD6_out
uv run infer.py --filename out/UDD6/DJI_0421.jpg --show_plot
```

#### dataset
https://huggingface.co/datasets/yisar/satov_data



#### Credits

Inspired by [SegEarth-OV](https://github.com/likyoo/SegEarth-OV), [JAFAR](https://github.com/PaulCouairon/JAFAR), [LPOSS](https://github.com/vladan-stojnic/LPOSS), [NAF](https://github.com/valeoai/NAF), [ClearCLIP](https://github.com/mc-lan/ClearCLIP), and [ResCLIP](https://github.com/yvhangyang/resclip). These open-source projects offer core ideas and implementations for open-vocabulary and geospatial segmentation.