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

