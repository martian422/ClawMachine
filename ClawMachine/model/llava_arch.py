#    Copyright 2023 Haotian Liu
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


from abc import ABC, abstractmethod

import torch
import torch.nn as nn
import math
from .multimodal_encoder.builder import build_vision_tower
from .multimodal_projector.builder import build_vision_projector

from ClawMachine.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_PATCH_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, REF_S, REF_E

from ClawMachine.mm_utils import get_anyres_image_grid_shape
import copy

class LlavaMetaModel:

    def __init__(self, config):
        super(LlavaMetaModel, self).__init__(config)

        if hasattr(config, "mm_vision_tower"):
            self.vision_tower = build_vision_tower(config, delay_load=True)
            self.mm_projector = build_vision_projector(config)

            if 'unpad' in getattr(config, 'mm_patch_merge_type', ''):
                self.image_newline = nn.Parameter(
                    torch.empty(config.hidden_size, dtype=self.dtype)
                )

    def get_vision_tower(self):
        vision_tower = getattr(self, 'vision_tower', None)
        if type(vision_tower) is list:
            vision_tower = vision_tower[0]
        return vision_tower
    
    def get_visual_tokenizer(self):
        visual_tokenizer = getattr(self.vision_tower.vision_tower, 'visual_tokenizer', None)
        return visual_tokenizer

    def initialize_vision_modules(self, model_args, fsdp=None):
        vision_tower = model_args.vision_tower
        mm_vision_select_layer = model_args.mm_vision_select_layer
        mm_vision_select_feature = model_args.mm_vision_select_feature
        pretrain_mm_mlp_adapter = model_args.pretrain_mm_mlp_adapter
        mm_patch_merge_type = model_args.mm_patch_merge_type

        self.config.mm_vision_tower = vision_tower

        if self.get_vision_tower() is None:
            vision_tower = build_vision_tower(model_args)

            if fsdp is not None and len(fsdp) > 0:
                self.vision_tower = [vision_tower]
            else:
                self.vision_tower = vision_tower
        else:
            if fsdp is not None and len(fsdp) > 0:
                vision_tower = self.vision_tower[0]
            else:
                vision_tower = self.vision_tower
            vision_tower.load_model()

        self.config.use_mm_proj = True
        self.config.mm_projector_type = getattr(model_args, 'mm_projector_type', 'linear')
        ### 1408 -> 1024
        self.config.mm_hidden_size = vision_tower.hidden_size
        self.config.mm_vision_select_layer = mm_vision_select_layer
        self.config.mm_vision_select_feature = mm_vision_select_feature
        self.config.mm_patch_merge_type = mm_patch_merge_type

        if getattr(self, 'mm_projector', None) is None:
            self.mm_projector = build_vision_projector(self.config)

            if 'unpad' in mm_patch_merge_type:
                embed_std = 1 / torch.sqrt(torch.tensor(self.config.hidden_size, dtype=self.dtype))
                self.image_newline = nn.Parameter(
                    torch.randn(self.config.hidden_size, dtype=self.dtype) * embed_std
                )
        else:
            # In case it is frozen by LoRA
            for p in self.mm_projector.parameters():
                p.requires_grad = True

        if pretrain_mm_mlp_adapter is not None:
            mm_projector_weights = torch.load(pretrain_mm_mlp_adapter, map_location='cpu')
            def get_w(weights, keyword):
                return {k.split(keyword + '.')[1]: v for k, v in weights.items() if keyword in k}

            self.mm_projector.load_state_dict(get_w(mm_projector_weights, 'mm_projector'))
            # self.mm_projector = self.mm_projector.to(torch.bfloat16)
            # print('_____mm_projector loaded and converted to bf16!________')  #### not used for now!


def unpad_image(tensor, original_size):
    """
    Unpads a PyTorch tensor of a padded and resized image.

    Args:
    tensor (torch.Tensor): The image tensor, assumed to be in CxHxW format.
    original_size (tuple): The original size of the image (height, width).

    Returns:
    torch.Tensor: The unpadded image tensor.
    """
    original_width, original_height = original_size
    current_height, current_width = tensor.shape[1:]

    original_aspect_ratio = original_width / original_height
    current_aspect_ratio = current_width / current_height

    if original_aspect_ratio > current_aspect_ratio:
        scale_factor = current_width / original_width
        new_height = int(original_height * scale_factor)
        padding = (current_height - new_height) // 2
        unpadded_tensor = tensor[:, padding:current_height - padding, :]
    else:
        scale_factor = current_height / original_height
        new_width = int(original_width * scale_factor)
        padding = (current_width - new_width) // 2
        unpadded_tensor = tensor[:, :, padding:current_width - padding]

    return unpadded_tensor


class LlavaMetaForCausalLM(ABC):

    @abstractmethod
    def get_model(self):
        pass

    def get_vision_tower(self):
        return self.get_model().get_vision_tower()
    
    def get_visual_tokenizer(self):
        return self.get_model().get_visual_tokenizer()

    def encode_images(self, images):
        image_features, feat_map = self.get_model().get_vision_tower()(images)

        return image_features,feat_map
    
    def project_feat(self, image_features):
        if type(image_features) is list:
            image_features_proj = []
            for image in image_features:
                # image = image.to(torch.float16) #this line is only activated when evaluating, otherwise should you comment it.
                # print(f'image',image[0].dtype)
                # print(f'proj',image[0].dtype)
                image_feature = self.get_model().mm_projector(image) #####?
                image_features_proj.append(image_feature)

        else:
            image_features_proj = self.get_model().mm_projector(image_features)
        return image_features_proj
    
    def tokenize_visual_features(self, image_feats):
        image_features_id_list = self.get_model().get_visual_tokenizer().tokenize_features_mirage(image_feats)
        return image_features_id_list
    
    def ref_sect_old(self, map, coor_list):
        # mirage
        # handles multiple coor in ref_list
        # Define the size of the square
        # FIXME: deal with multiple objects in one ref_patch
        # FIXME: deal with standard ref_lists
        l = 16
        N = l * l
        # Parse the coordinate string
        new_map_list = []
        # print(coor_list)
        for coor in coor_list:
            # print(coor)
            coords = [float(x) for x in coor.strip('><').split(',')]
            # print(coords)
            x1, y1, x2, y2 = coords

            # Convert coordinates to row and column indices
            row1, col1 = int(math.floor(y1 * l)), int(math.floor(x1 * l))
            row2, col2 = int(math.ceil(y2 * l)-1), int(math.ceil(x2 * l)-1)
            # Create a new set to store the new map
            new_map = set()

            # Add the indices in the specified region that are present in the original map
            for i in range(row1, row2+1):
                for j in range(col1, col2+1):
                    index = i * l + j
                    if 0 <= index < N and index in map:
                        new_map.add(index)
            ## deal with void sits, fix in the future.
            voi = len(list(new_map))
            if voi == 0:
                cent = int((row1 + row2)/2) * l + int((col1 + col2)/2)
                index = min(map, key=lambda x: abs(x - cent)).item()
                new_map.add(index)
                # print(cent)

            new_map_list.append(list(new_map))
        
        return new_map_list

    def ref_sect(self, map, coor_list):
        # mirage
        # handles multiple coor in ref_list
        # Define the size of the square
        l = 16
        N = l * l
        # Parse the coordinate string
        new_map_list = []
        # coor_list shall looks like [[[ref1],[ref2]],[[ref3]]]...
        for coors in coor_list: # for each conv
            # print(len(coors))
            new_map = set()
            for coor in coors: # for each entity(may have multiple locs)
                [x1, y1, x2, y2] = coor

                # Convert coordinates to row and column indices
                row1, col1 = int(math.floor(y1 * l)), int(math.floor(x1 * l))
                row2, col2 = int(math.ceil(y2 * l)-1), int(math.ceil(x2 * l)-1)
                # Create a new set to store the new map
                new_map_single = set()

                # Add the indices in the specified region that are present in the original map
                for i in range(row1, row2+1):
                    for j in range(col1, col2+1):
                        index = i * l + j
                        if 0 <= index < N and index in map:
                            new_map_single.add(index)
                ## deal with void sits, fix in the future.
                voi = len(list(new_map_single))
                if voi == 0:
                    cent = int((row1 + row2)/2) * l + int((col1 + col2)/2)
                    index = min(map, key=lambda x: abs(x - cent)).item()
                    new_map_single.add(index)
                new_map.update(new_map_single) # use set to avoid repetitive v tokens.
                # print(cent)
            new_map_list.append(list(new_map))
        return new_map_list

    def generate_substituted_list(self, ref_list, map_list, image_feat):
        substituted_list = []
        for sublist in ref_list:
            idx = [torch.where(map_list == element)[0][0].item() for element in sublist]
            sorted_idx = sorted(range(len(idx)), key=lambda k: idx[k])
            subs = [image_feat[idx[i]] for i in sorted_idx]

            substituted_list.append(torch.stack(subs))
        return substituted_list
    
    def select_vectors(self,tensor):
    # sample 16 from tensors
        interval = (len(tensor) - 2) // 14
        remainder = (len(tensor) - 2) % 14
        num_selected = [interval + 1] * remainder + [interval] * (14 - remainder)
        
        selected_indices = [0]
        
        current_index = 1
        for num in num_selected:
            selected_indices.append(current_index)
            current_index += num
        
        selected_indices.append(len(tensor) - 1)
        
        selected_vectors = tensor[selected_indices]
        
        return selected_vectors
    ### insert the ref_list info.
    def prepare_inputs_labels_for_multimodal(
        self, input_ids, position_ids, attention_mask, ref_list, past_key_values, labels,
        images, image_sizes=None
    ):
        vision_tower = self.get_vision_tower()
        ref_trig = False
        MAX_PAD = 16
        PAD_OPT = False
        IMAGE_INDEX = -500
        IMAGE_NO_VQ = False
        if ref_list is not None:
            ref_trig = True
        if vision_tower is None or images is None or input_ids.shape[1] == 1:
            return input_ids, position_ids, attention_mask, past_key_values, None, labels

        if type(images) is list or images.ndim == 5:
            if type(images) is list:
                images = [x.unsqueeze(0) if x.ndim == 3 else x for x in images]
            concat_images = torch.cat([image for image in images], dim=0)
            image_features_noproj, feature_map = self.encode_images(concat_images)
            image_features = self.project_feat(image_features_noproj)
            split_sizes = [image.shape[0] for image in images]
            image_features = torch.split(image_features, split_sizes, dim=0)
            mm_patch_merge_type = getattr(self.config, 'mm_patch_merge_type', 'flat')
            image_aspect_ratio = getattr(self.config, 'image_aspect_ratio', 'square')
            if mm_patch_merge_type == 'flat':
                image_features = [x.flatten(0, 1) for x in image_features]
            elif mm_patch_merge_type.startswith('spatial'):
                new_image_features = []
                for image_idx, image_feature in enumerate(image_features):
                    if image_feature.shape[0] > 1:
                        base_image_feature = image_feature[0]
                        image_feature = image_feature[1:]
                        height = width = self.get_vision_tower().num_patches_per_side
                        assert height * width == base_image_feature.shape[0]
                        if image_aspect_ratio == 'anyres':
                            num_patch_width, num_patch_height = get_anyres_image_grid_shape(image_sizes[image_idx], self.config.image_grid_pinpoints, self.get_vision_tower().config.image_size)
                            image_feature = image_feature.view(num_patch_height, num_patch_width, height, width, -1)
                        else:
                            raise NotImplementedError
                        if 'unpad' in mm_patch_merge_type:
                            image_feature = image_feature.permute(4, 0, 2, 1, 3).contiguous()
                            image_feature = image_feature.flatten(1, 2).flatten(2, 3)
                            image_feature = unpad_image(image_feature, image_sizes[image_idx])
                            image_feature = torch.cat((
                                image_feature,
                                self.model.image_newline[:, None, None].expand(*image_feature.shape[:-1], 1).to(image_feature.device)
                            ), dim=-1)
                            image_feature = image_feature.flatten(1, 2).transpose(0, 1)
                        else:
                            image_feature = image_feature.permute(0, 2, 1, 3, 4).contiguous()
                            image_feature = image_feature.flatten(0, 3)
                        image_feature = torch.cat((base_image_feature, image_feature), dim=0)
                    else:
                        image_feature = image_feature[0]
                        if 'unpad' in mm_patch_merge_type:
                            image_feature = torch.cat((
                                image_feature,
                                self.model.image_newline[None].to(image_feature.device)
                            ), dim=0)
                    new_image_features.append(image_feature)
                image_features = new_image_features
            else:
                raise ValueError(f"Unexpected mm_patch_merge_type: {self.config.mm_patch_merge_type}")
        else:
            image_features_noproj,feat_map = self.encode_images(images)
            image_features = self.project_feat(image_features_noproj)
        ### feat_map looks like[bs * [2,3,4,7,...252]]
        image_ids = self.tokenize_visual_features(image_features_noproj)

        if ref_trig:
            # ref_seqs = [self.ref_sect(feat_map[i],ref_list[i][0]) for i in range(len(feat_map))]
            ref_seqs = [self.ref_sect(feat_map[i],ref_list[i]) for i in range(len(feat_map))]
            ## sample: [[[97, 66, 99, 100], [], []], [[81, 50, 83, 82], [], []]]
            # ref_features_list = [self.generate_substituted_list(ref_seqs[i],feat_map[i],image_features[i]) for i in range(len(feat_map))]
            ref_ids_list = [self.generate_substituted_list(ref_seqs[i],feat_map[i],image_ids[i]) for i in range(len(feat_map))]

            #TODO: prune these tokens


        # TODO: image start / end is not implemented here to support pretraining.
        if getattr(self.config, 'tune_mm_mlp_adapter', False) and getattr(self.config, 'mm_use_im_start_end', False):
            raise NotImplementedError

        # Let's just add dummy tensors if they do not exist,
        # it is a headache to deal with None all the time.
        # But it is not ideal, and if you have a better idea,
        # please open an issue / submit a PR, thanks.
        _labels = labels
        _position_ids = position_ids
        _attention_mask = attention_mask
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
        else:
            attention_mask = attention_mask.bool()
        if position_ids is None:
            position_ids = torch.arange(0, input_ids.shape[1], dtype=torch.long, device=input_ids.device) ##[18]
        if labels is None:
            labels = torch.full_like(input_ids, IGNORE_INDEX)

        # remove the padding using attention_mask -- FIXME
        # _input_ids = input_ids

        input_ids = [cur_input_ids[cur_attention_mask] for cur_input_ids, cur_attention_mask in zip(input_ids, attention_mask)]
        labels = [cur_labels[cur_attention_mask] for cur_labels, cur_attention_mask in zip(labels, attention_mask)] 
        # <s> The bus<ref_patch>in the image is white and red.\n
        # The bus<ref_patch>in the image is white and red.\n  in labels, the <s> and image token will be tagged as ignored.

        ### fit the lavit model
        image_token_signal = torch.tensor([32000, 32001], dtype=torch.long).to(input_ids[0].device)###
        pad_signal_long = torch.tensor([2023,2024], dtype=torch.long).to(input_ids[0].device)
        pad_signal = pad_signal_long[:1]
        boi_signal = image_token_signal[:1]
        eoi_signal = image_token_signal[1:]
        
        image_signal_embeds = self.get_model().embed_tokens(image_token_signal) ####?
        pad_embed_long = self.get_model().embed_tokens(pad_signal_long)
        pad_embed = pad_embed_long[:1]

        boi_embed = image_signal_embeds[:1]
        eoi_embed = image_signal_embeds[1:]
        ###### for now, i think llava dont care if image_features is a list or tensor
        # new_input_embeds = []
        new_input_ids = []
        new_labels = []
        cur_image_idx = 0


        if ref_trig:

            idx_ref = [torch.where(input_ids[i]==31007) for i in range(len(input_ids))] #### lavit use almost the same code as vicuna
            idx_ref_patch = [torch.where(input_ids[i]==31005) for i in range(len(input_ids))] ####

        for batch_idx, cur_input_ids in enumerate(input_ids):


            if ref_trig:
                # labels[batch_idx][idx_ref[batch_idx]] = -100  #why?
                ### 0613: as we are using 『 』 pattern 31007-31005
                # FIXME in REF,always -100, in gnd, we may need the model to learn this pattern.
                # so just use 『 to start the refer procedure now.
                labels[batch_idx][idx_ref_patch[batch_idx]] = -300 ### 
                # cur_ref_features_list = ref_features_list[batch_idx]
                cur_ref_ids_list  = ref_ids_list[batch_idx]
            
            num_images = (cur_input_ids == IMAGE_TOKEN_INDEX).sum()
            if num_images >1:
                print('why?')
                raise ValueError
            # print(num_images)
            if num_images == 0:
                print('input ids not implemented!')
                cur_image_features = image_features[cur_image_idx]
                cur_input_embeds_1 = self.get_model().embed_tokens(cur_input_ids)
                cur_input_embeds = torch.cat([cur_input_embeds_1, boi_embed,cur_image_features[0:0],eoi_embed], dim=0)
                cur_labels_ = torch.cat([boi_signal,labels[batch_idx],eoi_signal])
                # new_input_embeds.append(cur_input_embeds)
                new_labels.append(cur_labels_)
                cur_image_idx += 1
                continue

            image_token_indices = [-1] + torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0].tolist() + [cur_input_ids.shape[0]]
            # ref_token_indices = [-1] + idx_ref_patch[batch_idx].tolist() + [cur_input_ids.shape[0]]
            cur_input_ids_noim = [] # noim - with no image
            cur_labels = labels[batch_idx]
            cur_labels_noim = []
            for i in range(len(image_token_indices) - 1):
                cur_input_ids_noim.append(cur_input_ids[image_token_indices[i]+1:image_token_indices[i+1]])
                cur_labels_noim.append(cur_labels[image_token_indices[i]+1:image_token_indices[i+1]])
            split_sizes = [x.shape[0] for x in cur_labels_noim]
            # cur_input_embeds = self.get_model().embed_tokens(torch.cat(cur_input_ids_noim))
            # cur_input_embeds_no_im = torch.split(cur_input_embeds, split_sizes, dim=0)
            # cur_new_input_embeds = [] ##way1
            cur_new_input_ids = []  ####way2
            cur_new_labels = []

            for i in range(num_images + 1):
                # cur_new_input_embeds.append(cur_input_embeds_no_im[i]) ##way1
                cur_new_labels.append(cur_labels_noim[i])
                cur_new_input_ids.append(cur_input_ids_noim[i])####way2

                if i < num_images:
                    # print(cur_image_idx)
                    # cur_image_features = torch.cat([boi_embed,image_features[cur_image_idx],eoi_embed])
                    cur_image_ids = torch.cat([boi_signal,image_ids[cur_image_idx],eoi_signal])
                    ### a \n has been added in the noim part, so do not add extra eoi token for now.
                    cur_image_idx += 1
                    # cur_new_input_embeds.append(cur_image_features)
                    cur_new_input_ids.append(cur_image_ids)
                    ### the following line is problematic 
                    ### task: will it help if you subst the real image with vq-ed feats?(remains to be seen)
                    cur_new_labels.append(torch.full((cur_image_ids.shape[0],), IMAGE_INDEX, device=cur_labels.device, dtype=cur_labels.dtype))
              
            # cur_new_input_embeds = [x.to(self.device) for x in cur_new_input_embeds]
            # cur_new_input_embeds = torch.cat(cur_new_input_embeds)
            cur_new_input_ids = [x.to(self.device) for x in cur_new_input_ids]
            cur_new_input_ids = torch.cat(cur_new_input_ids)
            cur_new_labels = torch.cat(cur_new_labels)

            ####now the game is on

            if ref_trig:
                # print('triggerd')

                ref_patch_idx = [torch.where(cur_new_labels==-300)][0][0]
                ref_token_indices = [-1] + ref_patch_idx.tolist() + [cur_new_labels.shape[0]]
                # cur_input_embeds_noref = []
                cur_input_ids_noref = [] 
                cur_labels = copy.deepcopy(cur_new_labels)
                cur_labels_noref = []
                for i in range(len(ref_token_indices) - 1):
                    # cur_input_embeds_noref.append(cur_new_input_embeds[ref_token_indices[i]+1:ref_token_indices[i+1]])
                    cur_input_ids_noref.append(cur_new_input_ids[ref_token_indices[i]+1:ref_token_indices[i+1]])  ###145,12,39,3,6 [4096]
                    cur_labels_noref.append(cur_labels[ref_token_indices[i]+1:ref_token_indices[i+1]])   ###145,12,39,3,6
                split_sizes = [x.shape[0] for x in cur_labels_noref]

                # cur_new_embeds_fin = []
                cur_new_ids_fin = []
                cur_new_labels_fin = []
                cur_ref_idx = 0


                num_ref = len(cur_ref_ids_list)
                for i in range(num_ref+1):

                    cur_new_ids_fin.append(cur_input_ids_noref[i])
                    cur_new_labels_fin.append(cur_labels_noref[i])
                    if i < num_ref:
                        ### for now, the ref_feats and global image use the same boi and eoi tokens.
                        ### need experiments to determine its side effects.
                        ###TODO
                        ref_len = len(cur_ref_ids_list[cur_ref_idx])
                        if PAD_OPT:
                            if ref_len<MAX_PAD:
                                num_pads = MAX_PAD-ref_len
                                # embed_pads = torch.repeat_interleave(pad_embed,num_pads,dim=0)
                                id_pads = torch.repeat_interleave(pad_signal,num_pads,dim=0)
                                # cur_ref_features = torch.cat([boi_embed,cur_ref_features_list[cur_ref_idx],embed_pads,eoi_embed])
                                cur_ref_ids = torch.cat([boi_signal,cur_ref_ids_list[cur_ref_idx],id_pads,eoi_signal])
                            else:
                                # cur_ref_features = torch.cat([boi_embed,self.select_vectors(cur_ref_features_list[cur_ref_idx]),eoi_embed])
                                cur_ref_ids = torch.cat([boi_signal,self.select_vectors(cur_ref_ids_list[cur_ref_idx]),eoi_signal])
                        else:
                            # cur_ref_features = torch.cat([boi_embed,cur_ref_features_list[cur_ref_idx],eoi_embed])
                            cur_ref_ids = torch.cat([boi_signal,cur_ref_ids_list[cur_ref_idx],eoi_signal])

                        if cur_labels_noref[i][-1]==-100:
                            cur_ref_ids_label = -100 * torch.ones(cur_ref_ids.shape, dtype = torch.int64).to(cur_ref_ids.device)
                        else:
                            cur_ref_ids_label = cur_ref_ids
                            # cur_ref_ids = -100 * torch.ones(cur_ref_ids.shape, dtype = torch.int64).to(cur_ref_ids.device)  
                            ## so that refs in the question will be masked.
                        cur_ref_idx += 1
                        if cur_ref_ids.shape[0]>2:    
                            # cur_new_embeds_fin.append(cur_ref_features)
                            cur_new_ids_fin.append(cur_ref_ids)
                            cur_new_labels_fin.append(cur_ref_ids_label) ####mask the ref in questions

                cur_new_input_ids = torch.cat(cur_new_ids_fin)
                cur_new_labels = torch.cat(cur_new_labels_fin)
            # else:
            #     print('no refs, skipping')

            # new_input_embeds.append(cur_new_input_embeds)
            new_input_ids.append(cur_new_input_ids)
            new_labels.append(cur_new_labels)  
            # result in list of [language_tokens,image_tokens, language_tokens,...]. [bs * [len, 4096]]
            # and its corr labels
            # perhaps support interleaved image-text?

        # Truncate sequences to max length as image embeddings can make the sequence longer
        tokenizer_model_max_length = getattr(self.config, 'tokenizer_model_max_length', None)
        if tokenizer_model_max_length is not None:
            # new_input_embeds = [x[:tokenizer_model_max_length] for x in new_input_embeds]
            new_input_ids = [x[:tokenizer_model_max_length] for x in new_input_ids]
            new_labels = [x[:tokenizer_model_max_length] for x in new_labels]

        # Combine them
        # max_len = max(x.shape[0] for x in new_input_embeds)
        max_len = max(x.shape[0] for x in new_input_ids)
        # batch_size = len(new_input_embeds)
        batch_size = len(new_input_ids)

        # new_input_embeds_padded = []
        new_input_ids_padded = []
        new_input_embeds_padded = []
        new_labels_padded = torch.full((batch_size, max_len), IGNORE_INDEX, dtype=new_labels[0].dtype, device=new_labels[0].device)
        attention_mask = torch.zeros((batch_size, max_len), dtype=attention_mask.dtype, device=attention_mask.device)
        position_ids = torch.zeros((batch_size, max_len), dtype=position_ids.dtype, device=position_ids.device)

        ### so be it.
        for i, (cur_new_id, cur_new_labels) in enumerate(zip(new_input_ids, new_labels)):
            cur_len = cur_new_id.shape[0]
            if getattr(self.config, 'tokenizer_padding_side', 'right') == "left":
                new_input_ids_padded.append(torch.cat((
                    torch.zeros((max_len - cur_len, ), dtype=cur_new_id.dtype, device=cur_new_id.device),
                    cur_new_id
                ), dim=0))
                if cur_len > 0:
                    new_labels_padded[i, -cur_len:] = cur_new_labels
                    attention_mask[i, -cur_len:] = True
                    position_ids[i, -cur_len:] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)
            else:
                id_to_append = torch.cat((
                    cur_new_id,
                    torch.zeros((max_len - cur_len, ), dtype=cur_new_id.dtype, device=cur_new_id.device)
                ), dim=0)
                new_input_ids_padded.append(id_to_append) ####cur_new_id.shape[1] do not exist, so removed.
                if cur_len > 0:
                    new_labels_padded[i, :cur_len] = cur_new_labels
                    attention_mask[i, :cur_len] = True
                    position_ids[i, :cur_len] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)

                embeds_to_append = self.get_model().embed_tokens(id_to_append)
                
                if IMAGE_NO_VQ == True: ### current support only one image
                    loc = torch.where(new_labels_padded[i]==IMAGE_INDEX)
                    new_labels_padded[i][loc]=-100
                    std = loc[0][0]
                    end = loc[0][-1]
                    embeds_to_append = torch.cat([embeds_to_append[:std],boi_embed,image_features[i],eoi_embed,embeds_to_append[end+1:]])
                else:
                    loc = torch.where(new_labels_padded[i]==IMAGE_INDEX)
                    new_labels_padded[i][loc]=-100

                new_input_embeds_padded.append(embeds_to_append)

        new_input_embeds = torch.stack(new_input_embeds_padded, dim=0)
        # new_input_ids = torch.stack(new_input_ids_padded, dim=0)

        if _labels is None:
            new_labels = None
        else:
            new_labels = new_labels_padded

        if _attention_mask is None:
            attention_mask = None
        else:
            attention_mask = attention_mask.to(dtype=_attention_mask.dtype)

        if _position_ids is None:
            position_ids = None

        # new_input_embeds[len=image_len+lang_len-X, dim], X = substituted tokens. 
        # now transformed into ids
        # new_labels : [category for each input_embeds].
        # Labels will ONLY be used for loss calculation! Do not involve in generation.
        # mask: always TRUE until the padding.
        # each attn_mask and labels will be padded to fit the longest.
        # pos_id and past_k_v are none.
        

        return None, position_ids, attention_mask, past_key_values, new_input_embeds, new_labels

    def initialize_vision_tokenizer(self, model_args, tokenizer):
        if model_args.mm_use_im_patch_token:
            tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
            self.resize_token_embeddings(len(tokenizer))

        if model_args.mm_use_im_start_end:
            num_new_tokens = tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)
            self.resize_token_embeddings(len(tokenizer))

            if num_new_tokens > 0:
                input_embeddings = self.get_input_embeddings().weight.data
                output_embeddings = self.get_output_embeddings().weight.data

                input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
                    dim=0, keepdim=True)
                output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
                    dim=0, keepdim=True)

                input_embeddings[-num_new_tokens:] = input_embeddings_avg
                output_embeddings[-num_new_tokens:] = output_embeddings_avg

            if model_args.tune_mm_mlp_adapter:
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = True
                for p in self.get_output_embeddings().parameters():
                    p.requires_grad = False

            if model_args.pretrain_mm_mlp_adapter:
                mm_projector_weights = torch.load(model_args.pretrain_mm_mlp_adapter, map_location='cpu')
                embed_tokens_weight = mm_projector_weights['model.embed_tokens.weight']
                assert num_new_tokens == 2
                if input_embeddings.shape == embed_tokens_weight.shape:
                    input_embeddings[-num_new_tokens:] = embed_tokens_weight[-num_new_tokens:]
                elif embed_tokens_weight.shape[0] == num_new_tokens:
                    input_embeddings[-num_new_tokens:] = embed_tokens_weight
                else:
                    raise ValueError(f"Unexpected embed_tokens_weight shape. Pretrained: {embed_tokens_weight.shape}. Current: {input_embeddings.shape}. Numer of new tokens: {num_new_tokens}.")
        elif model_args.mm_use_im_patch_token:
            if model_args.tune_mm_mlp_adapter:
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = False
                for p in self.get_output_embeddings().parameters():
                    p.requires_grad = False
