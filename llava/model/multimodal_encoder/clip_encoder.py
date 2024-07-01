import torch
import torch.nn as nn

from transformers import CLIPVisionModel, CLIPImageProcessor, CLIPVisionConfig
from .vq_clip import VQClip
# from vq_clip import VQClip

def convert_weights_to_fp16(model: nn.Module):
    """Convert applicable model parameters to fp16"""

    def _convert_weights_to_fp16(l):
        if isinstance(l, (nn.Conv1d, nn.Conv2d, nn.Linear)):
            l.weight.data = l.weight.data.to(torch.float16)
            if l.bias is not None:
                l.bias.data = l.bias.data.to(torch.float16)

    model.apply(_convert_weights_to_fp16)


def convert_weights_to_bf16(model: nn.Module):
    """Convert applicable model parameters to fp16"""

    def _convert_weights_to_bf16(l):
        if isinstance(l, (nn.Conv1d, nn.Conv2d, nn.Linear)):
            l.weight.data = l.weight.data.to(torch.bfloat16)
            if l.bias is not None:
                l.bias.data = l.bias.data.to(torch.bfloat16)

    model.apply(_convert_weights_to_bf16)


class D_CLIPVisionTower(nn.Module):
    def __init__(self, vision_tower, args, delay_load=False):
        super().__init__()

        self.is_loaded = False

        self.vision_tower_name = vision_tower
        self.image_processor = CLIPImageProcessor.from_pretrained('/share-cv/bob/mtr/CMCC/clip-vit-large-patch14')
        self.select_layer = args.mm_vision_select_layer # wont be used now
        self.select_feature = getattr(args, 'mm_vision_select_feature', 'patch')

        self.config = torch.load('/share-cv/bob/mtr/MIRAGE/clipconfig')

        if not delay_load:
            self.load_model()

    # def load_model(self):
    #     self.image_processor = CLIPImageProcessor.from_pretrained(self.vision_tower_name)
    #     self.vision_tower = CLIPVisionModel.from_pretrained(self.vision_tower_name)
    #     self.vision_tower.requires_grad_(False)

    #     self.is_loaded = True
    def load_model(self, model_path='/share-cv/bob/mtr/CMCC/LaTokenizer/LaVIT-7B-v2', model_dtype='bf16', device_id=None,**kwargs):
        if self.is_loaded:
            print('{} is already loaded, `load_model` called again, skipping.'.format(self.vision_tower_name))
            return
        
        # NOT SETTLED
        # self.image_processor = CLIPImageProcessor.from_pretrained(self.vision_tower_name)
        self.vision_tower = VQClip(model_path='/share-cv/bob/mtr/CMCC/LaTokenizer/LaVIT-7B-v2', model_dtype=model_dtype, device_id=device_id, use_xformers=False)

        if model_dtype == 'bf16':
            convert_weights_to_bf16(self.vision_tower)
        if model_dtype == 'fp16':
            convert_weights_to_fp16(self.vision_tower)
        # self.vision_tower = self.vision_tower.eval()

        self.vision_tower.requires_grad_(False)
        self.is_loaded = True

    @torch.no_grad()
    def forward(self, images):
        if type(images) is list:
            image_features = []
            for image in images:
                image = self.vision_tower.processer(image)
                with self.vision_tower.maybe_autocast():
                    image_feature, remained_map = self.vision_tower.compute_visual_embeds_raw(images).to(image.dtype) ###unused
                image_features.append(image_feature)
        else:
            images = self.vision_tower.processer(images)
            with self.vision_tower.maybe_autocast():
                # image_features = self.vision_tower.compute_visual_embeds_raw(images).to(images.dtype) ## malfunction
                image_features, remained_map = self.vision_tower.compute_visual_embeds_raw(images)
                # image_features = image_features.unsqueeze(0) #convert tensor [x,1408] to [1,x,1408]

        return image_features, remained_map

    @property
    def dummy_feature(self):
        return torch.zeros(1, self.hidden_size, device=self.device, dtype=self.dtype)

    @property
    def dtype(self):
        return self.vision_tower.dtype

    @property
    def device(self):
        return self.vision_tower.device

    # @property
    # def config(self):
    #     if self.is_loaded:
    #         return self.vision_tower.config
    #     else:
    #         return self.cfg_only

    @property
    def hidden_size(self):
        # return self.config.hidden_size
        size = 1408
        return size

    @property
    def num_patches_per_side(self):
        return self.config.image_size // self.config.patch_size ## 224/14 = 16

    @property
    def num_patches(self):
        return (self.config.image_size // self.config.patch_size) ** 2  ##256  bug



class CLIPVisionTower(nn.Module):
    def __init__(self, vision_tower, args, delay_load=False):
        super().__init__()

        self.is_loaded = False

        self.vision_tower_name = vision_tower
        self.select_layer = args.mm_vision_select_layer
        self.select_feature = getattr(args, 'mm_vision_select_feature', 'patch')

        if not delay_load:
            self.load_model()
        elif getattr(args, 'unfreeze_mm_vision_tower', False):
            self.load_model()
        else:
            self.cfg_only = CLIPVisionConfig.from_pretrained(self.vision_tower_name)

    def load_model(self, device_map=None):
        if self.is_loaded:
            print('{} is already loaded, `load_model` called again, skipping.'.format(self.vision_tower_name))
            return

        self.image_processor = CLIPImageProcessor.from_pretrained(self.vision_tower_name)
        self.vision_tower = CLIPVisionModel.from_pretrained(self.vision_tower_name, device_map=device_map)
        self.vision_tower.requires_grad_(False)

        self.is_loaded = True

    def feature_select(self, image_forward_outs):
        image_features = image_forward_outs.hidden_states[self.select_layer]
        if self.select_feature == 'patch':
            image_features = image_features[:, 1:]
        elif self.select_feature == 'cls_patch':
            image_features = image_features
        else:
            raise ValueError(f'Unexpected select feature: {self.select_feature}')
        return image_features

    @torch.no_grad()
    def forward(self, images):
        if type(images) is list:
            image_features = []
            for image in images:
                image_forward_out = self.vision_tower(image.to(device=self.device, dtype=self.dtype).unsqueeze(0), output_hidden_states=True)
                image_feature = self.feature_select(image_forward_out).to(image.dtype)
                image_features.append(image_feature)
        else:
            image_forward_outs = self.vision_tower(images.to(device=self.device, dtype=self.dtype), output_hidden_states=True)
            image_features = self.feature_select(image_forward_outs).to(images.dtype)

        return image_features

    @property
    def dummy_feature(self):
        return torch.zeros(1, self.hidden_size, device=self.device, dtype=self.dtype)

    @property
    def dtype(self):
        return self.vision_tower.dtype

    @property
    def device(self):
        return self.vision_tower.device

    @property
    def config(self):
        if self.is_loaded:
            return self.vision_tower.config
        else:
            return self.cfg_only

    @property
    def hidden_size(self):
        return self.config.hidden_size

    @property
    def num_patches_per_side(self):
        return self.config.image_size // self.config.patch_size

    @property
    def num_patches(self):
        return (self.config.image_size // self.config.patch_size) ** 2
