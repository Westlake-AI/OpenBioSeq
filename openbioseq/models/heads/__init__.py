from .cls_head import ClsHead
from .cls_mixup_head import ClsMixupHead
from .contrastive_head import ContrastiveHead, HCRHead
from .latent_pred_head import LatentPredictHead, LatentClsHead, MoCoV3Head
from .mae_head import MAEPretrainHead
from .reg_head import RegHead
from .swav_head import MultiPrototypes, SwAVHead
from .vision_transformer_head import VisionTransformerClsHead

__all__ = [
    'ContrastiveHead', 'ClsHead', 'ClsMixupHead', 'HCRHead',
    'LatentPredictHead', 'LatentClsHead',
    'MoCoV3Head', 'MAEPretrainHead', 'RegHead', 'MultiPrototypes', 'SwAVHead',
    'VisionTransformerClsHead',
]
