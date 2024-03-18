# GraphKD

### Description
Pytorch implementation of the paper [Exploring Knowledge Distillation Towards Document Object Detection with Structured Graph Creation](https://arxiv.org/abs/2402.11401). This model is implemented on top of the [detectron2](https://github.com/facebookresearch/detectron2) framework. The proposed architecture explores graph-based knowledge distillation to mitigate the trade-off between no. of model parameters (trainable) and performance accuracy towards document knowledge distillation with adaptive node sampling strategy and weighted edge distillation via Mahalanobis distance.

<p align="center">
  <img src="https://github.com/ayanban011/GraphKD/blob/main/fig/sgc.png">
  <be>
<b>Structured graph creation:</b> We extracted the RoI pooled features and classified them into "Text" and "Non-text" based on their covariance. Then we initialize the node in the identified RoI regions and define the adjacency edges. Lastly, we iteratively merge the text node with an adaptive sample mining strategy to reduce text bias.
</p>

# Getting Started

### Step 1: Clone this repository and change the directory to the repository root
```bash
git clone https://github.com/ayanban011/GraphKD.git 
cd GraphKD
```

### Step 2: Setup and activate the conda environment with required dependencies:
```bash
conda create --name graphkd python=3.9
conda activate graphkd
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
python -m pip install 'git+https://github.com/facebookresearch/detectron2.git' --user
```

### Step 3: Running

for **training**:

```bash
./start_train.sh train projects/Distillation/configs/Distillation-FasterRCNN-R18-R50-dsig-1x.yaml
```

for **testing**:

```bash
./start_train.sh eval projects/Distillation/configs/Distillation-FasterRCNN-R18-R50-dsig-1x.yaml
```

for **debugging**:

```bash
./start_train.sh debugtrain projects/Distillation/configs/Distillation-FasterRCNN-R18-R50-dsig-1x.yaml
```

### Results and Model Zoo

#### 1. PublayNet

<table class="tg">
<thead>
  <tr>
    <th class="tg-amwm"><span style="font-style:normal;text-decoration:none;color:#000;background-color:transparent">Model</span></th>
    <th class="tg-amwm"><span style="font-style:normal;text-decoration:none;color:#000;background-color:transparent">Config-file</span></th>
    <th class="tg-amwm"><span style="font-style:normal;text-decoration:none;color:#000;background-color:transparent">Weights</span></th>
    <th class="tg-amwm"><span style="font-style:normal;text-decoration:none;color:#000;background-color:transparent">AP</span></th>
  </tr>
</thead>
<tbody>
  <tr>
    <td class="tg-baqh"><span style="font-weight:400;font-style:normal;text-decoration:none;color:#000;background-color:transparent">PublayNet</span></td>
    <td class="tg-baqh"><span style="font-weight:400;font-style:normal;text-decoration:none;color:#000;background-color:transparent"><a href=https://github.com/ayanban011/SwinDocSegmenter/blob/main/configs/coco/instance-segmentation/swin/config_publay.yaml>config-publay</a></span></td>
    <td class="tg-baqh"><span style="font-weight:400;font-style:normal;text-decoration:none;color:#000;background-color:transparent"><a href=https://drive.google.com/file/d/1DCxG2MCza_z-yB3bLcaVvVR4Jik00Ecq/view?usp=share_link>model</a></span></td>
    <td class="tg-baqh"><span style="font-weight:400;font-style:normal;text-decoration:none;color:#000;background-color:transparent">93.72</span></td>
  </tr>
  <tr>
    <td class="tg-baqh"><span style="font-weight:400;font-style:normal;text-decoration:none;color:#000;background-color:transparent">Prima</span></td>
    <td class="tg-baqh"><span style="font-weight:400;font-style:normal;text-decoration:none;color:#000;background-color:transparent"><a href=https://github.com/ayanban011/SwinDocSegmenter/blob/main/configs/coco/instance-segmentation/swin/config_prima.yaml>config-prima</a></span></td>
    <td class="tg-baqh"><span style="font-weight:400;font-style:normal;text-decoration:none;color:#000;background-color:transparent"><a href=https://drive.google.com/file/d/1DNX9HQ0aG5ws0HCTFBUeV__rTlifNsvq/view?usp=share_link>model</a></span></td>
    <td class="tg-baqh"><span style="font-weight:400;font-style:normal;text-decoration:none;color:#000;background-color:transparent">54.39</span></td>
  </tr>
  <tr>
    <td class="tg-baqh"><span style="font-weight:400;font-style:normal;text-decoration:none;color:#000;background-color:transparent">HJ Dataset</span></td>
    <td class="tg-baqh"><span style="font-weight:400;font-style:normal;text-decoration:none;color:#000;background-color:transparent"><a href=https://github.com/ayanban011/SwinDocSegmenter/blob/main/configs/coco/instance-segmentation/swin/config_hj.yaml>config-hj</a></span></td>
    <td class="tg-baqh"><span style="font-weight:400;font-style:normal;text-decoration:none;color:#000;background-color:transparent"><a href=https://drive.google.com/file/d/12BCzIhSwZRJj8QjsFaqkdkLplf1bonL9/view?usp=sharing>model</a></span></td>
    <td class="tg-baqh"><span style="font-weight:400;font-style:normal;text-decoration:none;color:#000;background-color:transparent">84.65</span></td>
  </tr>
  <tr>
    <td class="tg-baqh"><span style="font-weight:400;font-style:normal;text-decoration:none;color:#000;background-color:transparent">TableBank</span></td>
    <td class="tg-baqh"><span style="font-weight:400;font-style:normal;text-decoration:none;color:#000;background-color:transparent"><a href=https://github.com/ayanban011/SwinDocSegmenter/blob/main/configs/coco/instance-segmentation/swin/config_table.yaml>config-table</a></span></td>
    <td class="tg-baqh"><span style="font-weight:400;font-style:normal;text-decoration:none;color:#000;background-color:transparent"><a href=https://drive.google.com/file/d/17DD9ASe3p3nLGEYhNCG0hbTURg8qNakC/view?usp=share_link>model</a></span></td>
    <td class="tg-baqh"><span style="font-weight:400;font-style:normal;text-decoration:none;color:#000;background-color:transparent">98.04</span></td>
  </tr>
  <tr>
    <td class="tg-baqh"><span style="font-weight:400;font-style:normal;text-decoration:none;color:#000;background-color:transparent">DoclayNet</span></td>
    <td class="tg-baqh"><span style="font-weight:400;font-style:normal;text-decoration:none;color:#000;background-color:transparent"><a href=https://github.com/ayanban011/SwinDocSegmenter/blob/main/configs/coco/instance-segmentation/swin/config_doclay.yaml>config-doclay</a></span></td>
    <td class="tg-baqh"><span style="font-weight:400;font-style:normal;text-decoration:none;color:#000;background-color:transparent"><a href=https://drive.google.com/file/d/1kMUnmdliyWWlXV9L8gQGvmS-h_mkM_mR/view?usp=share_link>model</a></span></td>
    <td class="tg-baqh"><span style="font-weight:400;font-style:normal;text-decoration:none;color:#000;background-color:transparent">76.85</span></td>
  </tr>
</tbody>
</table>

## Citation

If you find this useful for your research, please cite it as follows:

```bash
@article{banerjee2024graphkd,
  title={GraphKD: Exploring Knowledge Distillation Towards Document Object Detection with Structured Graph Creation},
  author={Banerjee, Ayan and Biswas, Sanket and Llad{\'o}s, Josep and Pal, Umapada},
  journal={arXiv preprint arXiv:2402.11401},
  year={2024}
}
```

## Acknowledgement

We have built it on the top of the [Dsig](https://github.com/dvlab-research/Dsig).


## Conclusion
Thank you for your interest in our work, and sorry if there are any bugs.

