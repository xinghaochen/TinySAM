# TinySAM
**TinySAM: Pushing the Envelope for Efficient Segment Anything Model**

*Han Shu, Wenshuo Li, Yehui Tang, Yiman Zhang, Yihao Chen, Houqiang Li, Yunhe Wang, Xinghao Chen*

*arXiv 2023*

[[`Paper`](https://arxiv.org/abs/2312.13789)] [[`BibTeX`](#citation)]

<p align="left">
<img width="300" alt="compare" src="./fig/tinysam_point.gif">&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<img width="300" alt="compare" src="./fig/tinysam_box.gif">
</p>

## Updates

* **2023/12/27**: Pre-trained models and codes of [Q-TinySAM](#usage) (quantized variant) are released.
* **2023/12/27**: [Evaluation](#evaluation) codes for zero-shot instance segmentation task on COCO are released.
* **2023/12/22**: Pre-trained models and codes of TinySAM are released both in [Pytorch](https://github.com/xinghaochen/TinySAM) and [Mindspore](https://gitee.com/mindspore/models/tree/master/research/cv/TinySAM).

## Overview

We propose a framework to obtain a tiny segment anything model (**TinySAM**) while maintaining the strong zero-shot performance. We first propose a full-stage knowledge distillation method with online hard prompt sampling strategy to distill a lightweight student model. We also adapt the post-training quantization to the promptable segmentation task and further reducing the computational cost. Moreover, a hierarchical segmenting everything strategy is proposed to accelerate the everything inference by with almost no performance degradation. With all these proposed methods, our TinySAM leads to orders of magnitude computational reduction and pushes the envelope for efficient segment anything task. Extensive experiments on various zero-shot transfer tasks demonstrate the significantly advantageous performance of our TinySAM against counterpart methods.

![framework](./fig/framework.png)
<div align=center>
<sup>Figure 1: Overall framework and zero-shot results of TinySAM.</sup>
</div>

![everything](./fig/everything.png)
<div align=center>
<sup>Figure 2: Our hierarchical strategy for everything mode.</sup>
</div>

![vis](./fig/vis.png)
<div align=center>
<sup>Figure 3: Visualization results of TinySAM.</sup>
</div>

## Requirements
The code requires `python>=3.7` and we use `torch==1.10.2` and `torchvision==0.11.3`. To visualize the results, `matplotlib>=3.5.1` is also required.  
- python 3.7
- pytorch == 1.10.2
- torchvision == 0.11.3
- matplotlib==3.5.1

## Usage

1. Download [checkpoints](#evaluation) into the directory of *weights*.

2. Run the demo code for single prompt of point or box.

```
python demo.py
```
3. Run the demo code for hierarchical segment everything strategy.
```
python demo_hierachical_everything.py
```

4. Run the demo code for quantization inference.
```
python demo_quant.py
```

## Evaluation
We follow the setting of original [SAM](https://arxiv.org/abs/2304.02643) paper and evaluate the zero-shot instance segmentaion on COCO and LVIS dataset. The experiment results are described as followed.

| Model               | FLOPs (G) |COCO AP (%) | LVIS AP (%)| 
| ------------------- | -------- | ------- |------- |
| SAM-H                 |3166| 46.5     | 44.7       | 
| SAM-L                 |1681| 45.5     | 43.5       | 
| SAM-B                 |677| 41.0     | 40.8       | 
| FastSAM                 |344| 37.9     | 34.5       | 
| MobileSAM            | 232|41.0     | 37.0       | 
| **TinySAM**  [\[ckpt\]](https://github.com/xinghaochen/TinySAM/releases/download/1.0/tinysam.pth)       | 232|41.9     | 38.6       | 
| **Q-TinySAM**  [\[ckpt\]](https://github.com/xinghaochen/TinySAM/releases/download/2.0/tinysam_w8a8.pth)            | 61|41.3     | 37.2      | 

First download the detection boxes ([`coco_instances_results_vitdet.json`](https://github.com/xinghaochen/TinySAM/releases/download/2.0/coco_instances_results_vitdet.json)) produced by ViTDet model, as well as the ground-truth instance segmentation labels([`instances_val2017.json`](https://github.com/xinghaochen/TinySAM/releases/download/2.0/instances_val2017.json)) and put them into `eval/json_files`.

Run the following code to perform evaluation for zero-shot instance segmentation on COCO dataset.
```
cd eval; sh eval_coco.sh
```
The results should be:
```
Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.419
Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.683
Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.436
Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.260
Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.456
Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.583
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.325
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.511
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.532
Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.390
Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.577
Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.671
```

## Acknowledgements
We thank the following projects: [SAM](https://github.com/facebookresearch/segment-anything), [MobileSAM](https://github.com/ChaoningZhang/MobileSAM), [TinyViT](https://github.com/microsoft/Cream).

## Citation
```bibtex
@article{tinysam,
  title={TinySAM: Pushing the Envelope for Efficient Segment Anything Model},
  author={Shu, Han and Li, Wenshuo and Tang, Yehui and Zhang, Yiman and Chen, Yihao and Li, Houqiang and Wang, Yunhe and Chen, Xinghao},
  journal={arXiv preprint arXiv:2312.13789},
  year={2023}
}
```

## License

This project is licensed under <a rel="license" href="License.txt"> Apache License 2.0</a>. Redistribution and use should follow this license.
