import os
from .clip_encoder import CLIPVisionTower
from .clip_encoder import D_CLIPVisionTower


def build_vision_tower(vision_tower_cfg, **kwargs):
    vision_tower = getattr(vision_tower_cfg, 'mm_vision_tower', getattr(vision_tower_cfg, 'vision_tower', None))
    is_absolute_path_exists = os.path.exists(vision_tower)
    # breakpoint()
    if is_absolute_path_exists and vision_tower.startswith('/share-cv/bob/mtr/CMCC/LaTokenizer/LaVIT-7B-v2'):
        return D_CLIPVisionTower(vision_tower, args=vision_tower_cfg, **kwargs)
    if is_absolute_path_exists and vision_tower.startswith("/home/MaTianren/clip-vit"):
        return CLIPVisionTower(vision_tower, args=vision_tower_cfg, **kwargs)

    raise ValueError(f'Unknown vision tower: {vision_tower}')
