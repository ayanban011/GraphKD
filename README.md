# GraphKD

### Description
Pytorch implementation of the paper [Exploring Knowledge Distillation Towards Document Object Detection with Structured Graph Creation](https://arxiv.org/abs/2402.11401). This model is implemented on top of the [detectron2](https://github.com/facebookresearch/detectron2) framework. The proposed architecture explores graph-based knowledge distillation to mitigate the trade-off between no. of model parameters (trainable) and performance accuracy towards document knowledge distillation with adaptive node sampling strategy and weighted edge distillation via Mahalanobis distance.

<p align="center">
  <img src="https://github.com/ayanban011/GraphKD/blob/main/fig/sgc.png">
  <br>
**Structured graph creation:** We extracted the RoI pooled features and classified them into "Text" and "Non-text" based on their covariance. Then we initialize the node in the identified RoI regions and define the adjacency edges. Lastly, we iteratively merge the text node with an adaptive sample mining strategy to reduce text bias.
</p>

# Getting Started
