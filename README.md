# 
<p align="center">
  <h1 align="center">Stronger ViTs With Octic Equivariance</h1>
  <p align="center">
    <a href="https://scholar.google.com/citations?user=-vJPE04AAAAJ">David Nordström</a>
    ·
    <a href="https://scholar.google.com/citations?user=Ul-vMR0AAAAJ&hl">Johan Edstedt</a>
    ·
    <a href="https://scholar.google.com/citations?user=P_w6UgMAAAAJ">Fredrik Kahl</a>
    ·
    <a href="https://scholar.google.com/citations?user=FUE3Wd0AAAAJ">Georg Bökman</a>
  </p>
  <h2 align="center"><p>
    <a align="center">Paper</a>
  </p></h2>
  <div align="center"></div>
</p>
<br/>
<p align="center">
    <img src="./assets/figure_1.png" alt="example" height="300">
    <br>
    Using octic layers in ViTs significantly reduces the computational complexity while maintaining or improving classification accuracy on ImageNet-1K, for both supervised and self-supervised training.
</p>

## Structure

### Octic ViTs
In the [`octic_vits`](octic_vits) folder you find all the components to build octic-equivariant Vision Transformers (intended to be compatible with the [timm](https://github.com/huggingface/pytorch-image-models) library). For example, to create an octic ViT-H you can run the following:
```python
from octic_vits import OcticVisionTransformer

model = OcticVisionTransformer(embed_dim=1280, depth=32, num_heads=16)
```
This will default to a hybrid model with the first half of its block being octic and the remaining standard (i.e. this model will have approx. 40% less FLOPs than a regular ViT-H). To instead obtain an invariant model, simply set `invariant=True`.

### DeiT III
Code based on the official [repo](https://github.com/facebookresearch/deit) has been placed in the [`deit`](deit) folder.

### DINOv2 
Code based on the official [repo](https://github.com/facebookresearch/dinov2) has been placed in the [`dinov2`](dinov2) folder.

## Reproducing Results
Code to reproduce the experiments can be found in the [experiments folder](experiments). Below follows general instruction on how to run it and how to obtain pretrained model weights.

### Setup
All the code is written with the intent to be run on a [Slurm](https://slurm.schedmd.com/documentation.html) cluster using [submitit](https://github.com/facebookincubator/submitit). So first you must set up the cluster settings in[`utils/cluster.py`](utils/cluster.py). If you intend to run it using `torchrun` instead, it should work straightforwardly. Also, make sure to run `export PYTHONPATH=$(pwd)` in the root folder of this directory to ensure relative imports work as intended. 

### Environment
For DINOv2 we use the same environment as in the original [repo](https://github.com/facebookresearch/dinov2) and same goes for [deit](https://github.com/facebookresearch/deit). Additional miscellaneous installations, e.g. [submitit](https://github.com/facebookincubator/submitit), need to be additionally downloaded.   

Since DeiT III is deprecated we provide some additional guidance on its installation. It uses [NVIDIA apex](https://github.com/NVIDIA/apex). Thus, you must compile apex. Begin by creating a new virtual environment for Python, we use conda and Python 3.10. Then: 

Clone apex:
```bash
git clone https://github.com/NVIDIA/apex.git
```

Run:
```bash
cd apex
git checkout 2386a912164
python setup.py install --cuda_ext --cpp_ext
```

Now that you have compiled apex, you can continue setting up the Python environment. Either do so by installing the packages in the `deit/requirements.txt` manually or by inheriting our conda environment by running:
```bash
conda env update --file deit/environment.yml
```

### Data

#### ImageNet-1K

We follow the DINOv2 IN1K data structure. As such, the root directory of the dataset should hold the following contents:

- `<ROOT>/test/ILSVRC2012_test_00000001.JPEG`
- `<ROOT>/test/[..]`
- `<ROOT>/test/ILSVRC2012_test_00100000.JPEG`
- `<ROOT>/train/n01440764/n01440764_10026.JPEG`
- `<ROOT>/train/[...]`
- `<ROOT>/train/n15075141/n15075141_9993.JPEG`
- `<ROOT>/val/n01440764/ILSVRC2012_val_00000293.JPEG`
- `<ROOT>/val/[...]`
- `<ROOT>/val/n15075141/ILSVRC2012_val_00049174.JPEG`
- `<ROOT>/labels.txt`

The provided dataset implementation expects a few additional metadata files to be present under the extra directory:

- `<EXTRA>/class-ids-TRAIN.npy`
- `<EXTRA>/class-ids-VAL.npy`
- `<EXTRA>/class-names-TRAIN.npy`
- `<EXTRA>/class-names-VAL.npy`
- `<EXTRA>/entries-TEST.npy`
- `<EXTRA>/entries-TRAIN.npy`
- `<EXTRA>/entries-VAL.npy`

These metadata files can be generated (once) with the following lines of Python code:

```python
from dinov2.data.datasets import ImageNet

for split in ImageNet.Split:
    dataset = ImageNet(split=split, root="<ROOT>", extra="<EXTRA>")
    dataset.dump_extra()
```
#### ADE20K / VOC2012

For segmentation evaluation we use the code from [capi](https://github.com/facebookresearch/capi) and as the creator of said repository is very helpful, he has enabled automatic downloading of the datasets. For more information consult the original repo.

### Weights

Download the weights from here to reproduce the evaluation metrics. The DINOv2 weights only include the teacher backbone.

#### DeiT III
<table style="margin: auto">
  <thead>
    <tr>
      <th>model</th>
      <th># of<br />params</th>
      <th># of<br />FLOPs</th>
      <th>ImageNet<br/>Top-1</th>
      <th>weights</th>
      <th>logs</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Hybrid ViT-H/14</td>
      <td align="right">356 M</td>
      <td align="center">102 G</td>
      <td align="right"><strong>85.0%</strong></td>
      <td><a href="https://drive.google.com/file/d/1ImsKvMGDo2FTzMla_3iGRGxOCqIopXyq/view?usp=sharing">weights</a></td>
      <td><a href="https://drive.google.com/file/d/1C_5j89J8NboqZ2zDa9-qLnr-Zg7293PY/view?usp=sharing">logs</a></td>
    </tr>
    <tr>
      <td>Invariant ViT-H/14</td>
      <td align="right">362 M</td>
      <td align="center">104 G</td>
      <td align="right">84.7%</td>
      <td><a href="https://drive.google.com/file/d/149GeWmrT-JNTyI1CY0Dp0MCCzXNks13g/view?usp=sharing">weights</a></td>
      <td><a href="https://drive.google.com/file/d/1jVsjj5W_vzWuKjpXXISzwq5AhxfXu86G/view?usp=sharing">logs</a></td>
    </tr>
    <tr>
      <td>Hybrid ViT-L/16</td>
      <td align="right">171 M</td>
      <td align="center">38 G</td>
      <td align="right">84.5%</td>
      <td><a href="https://drive.google.com/file/d/1N7qNRjXWOmZEQetpIkGLEbC7mKNlYku8/view?usp=sharing">weights</a></td>
      <td><a href="https://drive.google.com/file/d/1r-TZgILhqdhru-hVp4wcHSIRx3WElvMs/view?usp=sharing">logs</a></td>
    </tr>
    <tr>
      <td>Invariant ViT-L/16</td>
      <td align="right">175 M</td>
      <td align="center">39 G</td>
      <td align="right">84.0%</td>
      <td><a href="https://drive.google.com/file/d/1wgfa8iXS7mIIcXq4OPtgrt_pxU9cqper/view?usp=sharing">weights</a></td>
      <td><a href="https://drive.google.com/file/d/1zxSCZapRibKposMofwIYVUNSOW8u7gnj/view?usp=sharing">logs</a></td>
    </tr>

  </tbody>
</table>

#### DINOv2
<table style="margin: auto">
  <thead>
    <tr>
      <th>model</th>
      <th># of<br />FLOPs</th>
      <th>ImageNet<br/>linear</th>
      <th>ImageNet<br/>knn</th>
      <th>weights</th>
      <th>logs</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>ViT-H/16</td>
      <td align="center">128 G</td>
      <td align="right">81.7%</td>
      <td align="right">81.0%</td>
      <td><a href="https://drive.google.com/file/d/17KOwpM_YXvZguHc5EcJSL287WZidbBkd/view?usp=sharing">weights</a></td>
      <td><a href="https://drive.google.com/file/d/1qsgHYQeXQAdkq-6K6yP62xpJRJlGXfgd/view?usp=sharing">logs</a></td>
    </tr>
    <tr>
      <td>Hybrid ViT-H/16</td>
      <td align="center">78 G</td>
      <td align="right"><strong>82.2%</strong></td>
      <td align="right"><strong>81.4%</strong></td>
      <td><a href="https://drive.google.com/file/d/1Z13SV0wMTZnhbVSpv_XizVoktp_7b4x9/view?usp=sharing">weights</a></td>
      <td><a href="https://drive.google.com/file/d/1a3yg1H2pgGgp4jhd169-y9W7gWH7TMHY/view?usp=sharing">logs</a></td>
    </tr>
    <tr>
      <td>Invariant ViT-H/16</td>
      <td align="center">78 G</td>
      <td align="right">81.9%</td>
      <td align="right">80.9%</td>
      <td><a href="https://drive.google.com/file/d/1yJI5m7LjzM_MxaM8IxZyTpHM4W6_vBC8/view?usp=sharing">weights</a></td>
      <td><a href="https://drive.google.com/file/d/1HjdnYfwPjiQENgIwqYheWtCXUNUDKNep/view?usp=sharing">logs</a></td>
    </tr>
    <tr>
      <td>ViT-L/16</td>
      <td align="center">62 G</td>
      <td align="right">80.9%</td>
      <td align="right">80.5%</td>
      <td><a href="https://drive.google.com/file/d/1Xp9oK0eAGLvj9TZuTL1d6cTN2k3f_aKv/view?usp=sharing">weights</a></td>
      <td><a href="https://drive.google.com/file/d/1wmA9zmyKJIsEtX4YVxIR3MwsyxO-4S6k/view?usp=sharing">logs</a></td>
    </tr>
    <tr>
      <td>Hybrid ViT-L/16</td>
      <td align="center">38 G</td>
      <td align="right">81.3%</td>
      <td align="right">80.8%</td>
      <td><a href="https://drive.google.com/file/d/18Aoe1JP_AafaAzDGSGjZwgUPLkDaPS_h/view?usp=sharing">weights</a></td>
      <td><a href="https://drive.google.com/file/d/1zyDiDP_r-BYqb0VXnYjBgm0K2NgJNAa7/view?usp=sharing">logs</a></td>
    </tr>
    <tr>
      <td>Invariant ViT-L/16</td>
      <td align="center">38 G</td>
      <td align="right">81.2%</td>
      <td align="right">80.4%</td>
      <td><a href="https://drive.google.com/file/d/1ZelO4ozTqbos3nCRyxMly-T8CjEj_61n/view?usp=sharing">weights</a></td>
      <td><a href="https://drive.google.com/file/d/19vFSPMa0qIbl0jKa2FnSGt49atcC53cw/view?usp=sharing">logs</a></td>
    </tr>
  </tbody>
</table>

### Evaluation

#### Deit III 
After downloading the weights you should be able to run the following command to evaluate a model (e.g. Hybrid ViT-H):
```bash
python experiments/eval_deit.py --model hybrid_deit_huge_patch14 --eval pretrained_models/hybrid_deit_huge_patch14.pth
```
This should give:
```
* Acc@1 84.996 Acc@5 96.390 loss 0.799
```

### Training
We train on a cluster using [submitit](https://github.com/facebookincubator/submitit). So first you must set up the cluster settings in `octo/utils/cluster.py` and then you can simply run:
```bash
python experiments/train_octo.py --gpus 4 --nodes 4
```
### Equivariance
We have provided a utility file to verify octic equivariance (and invariance). Simply run:
```bash
python experiments/test_equivariance.py
```

### Throughput
In the paper we present the throughput. To replicate these figures run the following command on a A100-80GB:
```bash
python experiments/complexity.py --amp --compile
```

## Checklist
- [ ] Release the D8 models + weights
- [ ] Add to timm library

## License
Stronger ViTs with Octic Equivariance code and model weights are released under the Apache License 2.0. See [LICENSE](LICENSE) for additional details. Training recipes are taken from [DeiT III](https://github.com/facebookresearch/deit) and [DINOv2](https://github.com/facebookresearch/dinov2), and evaluation is taken from [capi](https://github.com/facebookresearch/capi), all released under the Apache License 2.0.

## Credit
Code structure is inspired by [capi](https://github.com/facebookresearch/capi) and [RoMa](https://github.com/Parskatt/RoMa).

## Cite
If you find this repository useful, please consider giving a star :star: and citation :octopus::

```
@misc{nordstrom2025strongervits,
  title={Stronger ViTs with Octic Equivariance},
  author={Nordström, David and Edstedt, Johan and Kahl, Fredrik and Bökman, Georg},
  journal={arXiv:TBD},
  year={2025}
}
```