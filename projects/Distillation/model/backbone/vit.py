import warnings
import math
import logging
from typing import Sequence, Tuple, Union, Callable, Optional, Any, Dict, List
from functools import partial
from timm.layers import Mlp, PatchEmbed
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from timm.models.vision_transformer import VisionTransformer, PatchEmbed, Block, Mlp
from timm.layers import PatchEmbed, PatchEmbedWithSize
from timm.layers.format import Format, nchw_to
from timm.layers.trace_utils import _assert

from detectron2.layers import (
    ShapeSpec,
)
from detectron2.modeling import Backbone, BACKBONE_REGISTRY, FPN
from detectron2.modeling.backbone.fpn import LastLevelP6P7, LastLevelMaxPool

logger = logging.getLogger("detectron2")

class PatchEmbedSized(PatchEmbed):
    """ 2D Image to Patch Embedding
    """
    output_fmt: Format
    dynamic_img_pad: torch.jit.Final[bool]

    def __init__(
            self,
            img_size: Optional[int] = 224,
            patch_size: int = 16,
            in_chans: int = 3,
            embed_dim: int = 768,
            norm_layer: Optional[Callable] = None,
            flatten: bool = True,
            output_fmt: Optional[str] = None,
            bias: bool = True,
            dynamic_img_pad = False
    ):

        super().__init__(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            norm_layer=norm_layer,
            flatten=flatten,
            output_fmt=output_fmt,
            bias=bias,
            dynamic_img_pad=dynamic_img_pad
        )

    def forward(self, x) -> Tuple[torch.Tensor, List[int]]:
        B, C, H, W = x.shape
        if self.img_size is not None:
            _assert(H % self.patch_size[0] == 0, f"Input image height ({H}) must be divisible by patch size ({self.patch_size[0]}).")
            _assert(W % self.patch_size[1] == 0, f"Input image width ({W}) must be divisible by patch size ({self.patch_size[1]}).")
        x = self.proj(x)
        grid_size = x.shape[-2:]
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # NCHW -> NLC
        elif self.output_fmt != Format.NCHW:
            x = nchw_to(x, self.output_fmt)
        x = self.norm(x)
        return x, grid_size

class VisTranTimm(VisionTransformer):
    def __init__(self, img_size = 224, patch_size = 16, in_chans = 3, num_classes = 1000, global_pool = 'token', embed_dim = 768, depth = 12, num_heads = 12, mlp_ratio = 4, qkv_bias = True, qk_norm = False, init_values = None, class_token = True, no_embed_class = False, pre_norm = False, fc_norm = None, drop_rate = 0, pos_drop_rate = 0, patch_drop_rate = 0, proj_drop_rate = 0, attn_drop_rate = 0, drop_path_rate = 0, weight_init = '', embed_layer = PatchEmbedSized, norm_layer = None, act_layer = None, block_fn = Block, mlp_layer = Mlp,
                 out_features = None):
        super().__init__(img_size=img_size, patch_size=patch_size, in_chans=in_chans, num_classes=num_classes, global_pool=global_pool, embed_dim=embed_dim, depth=depth, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_norm=qk_norm, init_values=init_values, class_token=class_token, no_embed_class=no_embed_class, pre_norm=pre_norm, fc_norm=fc_norm, drop_rate=drop_rate, pos_drop_rate=pos_drop_rate, patch_drop_rate=patch_drop_rate, proj_drop_rate=proj_drop_rate, attn_drop_rate=attn_drop_rate, drop_path_rate=drop_path_rate, weight_init=weight_init, embed_layer=embed_layer, norm_layer=norm_layer, act_layer=act_layer, block_fn=block_fn, mlp_layer=mlp_layer)

        self.out_features = out_features
        self.out_indices = [int(name[5:]) for name in out_features]
        
        if patch_size == 16:
            self.fpn1 = nn.Sequential(
                nn.ConvTranspose2d(embed_dim, embed_dim, kernel_size=2, stride=2),
                # nn.SyncBatchNorm(embed_dim),
                nn.BatchNorm2d(embed_dim),
                nn.GELU(),
                nn.ConvTranspose2d(embed_dim, embed_dim, kernel_size=2, stride=2),
            )
            self.fpn2 = nn.Sequential(
                nn.ConvTranspose2d(embed_dim, embed_dim, kernel_size=2, stride=2),
            )
            self.fpn3 = nn.Identity()
            self.fpn4 = nn.MaxPool2d(kernel_size=2, stride=2)

        elif patch_size == 8:
            self.fpn1 = nn.Sequential(
                nn.ConvTranspose2d(embed_dim, embed_dim, kernel_size=2, stride=2),
            )
            self.fpn2 = nn.Identity()
            self.fpn3 = nn.Sequential(
                nn.MaxPool2d(kernel_size=2, stride=2),
            )
            self.fpn4 = nn.Sequential(
                nn.MaxPool2d(kernel_size=4, stride=4),
            )

    def forward_maps(self, x):
        B, C, H, W = x.shape
        print(x.shape)
        features = []
        x, grids = self.patch_embed(x)
        print(x.shape)
        x = self._pos_embed(x)
        x = self.patch_drop(x)
        x = self.norm_pre(x)

        Hp, Wp = grids[0], grids[1]

        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if i in self.out_indices:
                xp = x[:, self.num_prefix_tokens:].permute(0,2,1).reshape(B, -1, Hp, Wp)
                features.append(xp.contiguous())
        
        return features
    
    def forward(self, x):
        features = self.forward_maps(x)

        ops = [self.fpn1, self.fpn2, self.fpn3, self.fpn4]
        for i in range(len(features)):
            features[i] = ops[i](features[i])

        feat_out = {}

        for name, value in zip(self.out_features, features):
            feat_out[name] = value

        return feat_out


class VanillaViT(Backbone):
    def __init__(self, name, out_features, drop_path, img_size, patch_size):
        super().__init__()

        self._out_features = out_features
        if 'base' in name:
            embed_dim = 768
            depth = 12
            num_heads = 12
            # mlp_ratio = 4.0
            self._out_feature_strides = {"layer3": 4, "layer5": 8, "layer7": 16, "layer11": 32}
            self._out_feature_channels = {"layer3": embed_dim, "layer5": embed_dim, "layer7": embed_dim, "layer11": embed_dim}
        elif 'large' in name:
            embed_dim = 1024
            depth = 24
            num_heads = 16
            # mlp_ratio = 4.0
            self._out_feature_strides = {"laye7": 4, "layer11": 8, "layer15": 16, "layer23": 32}
            self._out_feature_channels = {"layerr7": embed_dim, "layer11": embed_dim, "layer15": embed_dim, "layer23": embed_dim}
        elif 'small' in name:
            embed_dim = 384
            depth = 12
            num_heads = 8
            # mlp_ratio = 4.0
            self._out_feature_strides = {"layer3": 4, "layer5": 8, "layer7": 16, "layer11": 32}
            self._out_feature_channels = {"layer7": embed_dim, "layer11": embed_dim, "layer15": embed_dim, "layer23": embed_dim}
        elif 'tiny' in name:
            embed_dim = 192
            depth = 12
            num_heads = 3
            # mlp_ratio = 4.0
            self._out_feature_strides = {"layer3": 4, "layer5": 8, "layer7": 16, "layer11": 32}
            self._out_feature_channels = {"layer7": embed_dim, "layer11": embed_dim, "layer15": embed_dim, "layer23": embed_dim}
        else:
            print('Unsupported ViT name')

        self.bb = VisTranTimm(img_size=img_size, patch_size=patch_size, drop_path_rate=drop_path, out_features=out_features, embed_dim=embed_dim, depth=depth, num_heads=num_heads)
    
    def forward(self, x):
        assert x.dim() == 4, f"VIT takes an input of shape (N, C, H, W). Got {x.shape} instead!"
        return self.bb(x)
    
    def output_shape(self):
        return {
            name: ShapeSpec(
                channels=self._out_feature_channels[name], stride=self._out_feature_strides[name]
            )
            for name in self._out_features
        }

def build_vanilla_vit_backbone(cfg):

    name = cfg.MODEL.VIT.ARCH
    out_features = cfg.MODEL.VIT.OUT_FEATURES
    drop_path = cfg.MODEL.VIT.DROP_PATH
    img_size = cfg.MODEL.VIT.IMG_SIZE
    # pos_type = cfg.MODEL.VIT.POS_TYPE
    patch_size = cfg.MODEL.VIT.PATCH_SIZE

    # model_kwargs = eval(str(cfg.MODEL.VIT.MODEL_KWARGS).replace("`", ""))

    return VanillaViT(name, out_features, drop_path, img_size, patch_size)

@BACKBONE_REGISTRY.register()
def build_vit_fpn_backbone(cfg, input_shape: ShapeSpec):
    """
    Create a VIT w/ FPN backbone.

    Args:
        cfg: a detectron2 CfgNode

    Returns:
        backbone (Backbone): backbone module, must be a subclass of :class:`Backbone`.
    """
    bottom_up = build_vanilla_vit_backbone(cfg)
    in_features = cfg.MODEL.FPN.IN_FEATURES
    out_channels = cfg.MODEL.FPN.OUT_CHANNELS
    backbone = FPN(
        bottom_up=bottom_up,
        in_features=in_features,
        out_channels=out_channels,
        norm=cfg.MODEL.FPN.NORM,
        top_block=LastLevelMaxPool(),
        fuse_type=cfg.MODEL.FPN.FUSE_TYPE,
    )
    return backbone

def add_vanilla_vit_defaults(cfg):
    # cfg.MODEL.VIT = CN() # Taken care of by ViT Det
    cfg.MODEL.VIT.ARCH = 'vit_base_patch16_224'
    cfg.MODEL.VIT.DROP_PATH = 0.1
    cfg.MODEL.VIT.IMG_SIZE = 224
    cfg.MODEL.VIT.PATCH_SIZE = 16
    # cfg.MODEL.VIT.POS_TYPE = 'abs'
    # cfg.MODEL.VIT.MODEL_KWARGS = ''
    cfg.MODEL.VIT.OUT_FEATURES = ["layer3", "layer5", "layer7", "layer11"]
    cfg.SOLVER.OPTIMIZER = 'SGD'
    cfg.SOLVER.BACKBONE_MULTIPLIER = 1.0
    return cfg