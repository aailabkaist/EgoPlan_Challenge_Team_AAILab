import logging
import random

import torch
from torch.cuda.amp import autocast as autocast
import torch.nn as nn

from src.video_llama.video_llama.common.registry import registry
from src.video_llama.video_llama.models.blip2 import Blip2Base, disabled_train
from src.video_llama.video_llama.models.modeling_llama import LlamaForCausalLM
from transformers import LlamaTokenizer,BertConfig
import einops
import copy
from src.video_llama.video_llama.models.Qformer import BertConfig, BertLMHeadModel
from src.video_llama.video_llama.models.ImageBind.models.imagebind_model import ImageBindModel,ModalityType
from src.video_llama.video_llama.models.ImageBind.models import imagebind_model
from peft import LoraConfig, get_peft_model, PeftModelForCausalLM
import string


@registry.register_model("egoplan_video_llama")
class EgoplanVideoLLAMA(Blip2Base):
    """
    BLIP2 GPT-LLAMA model.
    """

    PRETRAINED_MODEL_CONFIG_DICT = {
        "pretrain_vicuna": "configs/models/video_llama.yaml",
        "pretrain_llama_v2": "configs/models/video_llama.yaml",
    }

    @classmethod
    def init_video_Qformer(cls, num_query_token, vision_width, num_hidden_layers=2, q_former_encoder_model="bert-base-uncased"):
        encoder_config = BertConfig.from_pretrained(q_former_encoder_model)
        encoder_config.num_hidden_layers = num_hidden_layers
        encoder_config.encoder_width = vision_width
        # insert cross-attention layer every other block
        encoder_config.add_cross_attention = True
        encoder_config.cross_attention_freq = 1
        encoder_config.query_length = num_query_token
        Qformer = BertLMHeadModel(config=encoder_config)
        query_tokens = nn.Parameter(
            torch.zeros(1, num_query_token, encoder_config.hidden_size)
        )
        query_tokens.data.normal_(mean=0.0, std=encoder_config.initializer_range)
        return Qformer, query_tokens

    def __init__(
        self,
        vit_model="eva_clip_g",
        vit_model_path=None,
        q_former_model="EgoPlan-challenge-Team-AAILab/src/video_llama/video_llama/models/blip2_pretrained_flant5xxl.pth",
        q_former_encoder_model="bert-base-uncased",

        img_size=224,
        drop_path_rate=0,
        use_grad_checkpoint=False,
        vit_precision="fp16",
        freeze_vit=True,
        freeze_qformer=True,
        num_query_token=32,
        llama_model="EgoPlan-challenge-Team-AAILab/Llama-2-7b-chat-hf",
        prompt_path="",
        prompt_template="",
        max_txt_len=32,
        end_sym='\n',
        low_resource=False,  # use 8 bit and put vit in cpu
        device_8bit=0,  # the device of 8bit model should be set when loading and cannot be changed anymore.

        frozen_llama_proj=True,
        frozen_video_Qformer=True,
        frozen_audio_Qformer=True,

        llama_proj_model='',
        fusion_header_type= "seqTransf",
        max_frame_pos= 32,
        fusion_head_layers = 2,
        num_video_query_token = 32,
        num_audio_query_token = 8,
        imagebind_ckpt_path = '/mnt/workspace/ckpt',
        equip_audio_branch = True,
        llm_lora_config = None,

    ):
        super().__init__()
        self.tokenizer = self.init_tokenizer(q_former_encoder_model=q_former_encoder_model)
        self.low_resource = low_resource

        print('Loading VIT')
        self.visual_encoder, self.ln_vision = self.init_vision_encoder(
            vit_model, img_size, drop_path_rate, use_grad_checkpoint, vit_precision, vit_model_path
        )
        if freeze_vit:
            for name, param in self.visual_encoder.named_parameters():
                param.requires_grad = False
            self.visual_encoder = self.visual_encoder.eval()
            self.visual_encoder.train = disabled_train
            for name, param in self.ln_vision.named_parameters():
                param.requires_grad = False
            self.ln_vision = self.ln_vision.eval()
            self.ln_vision.train = disabled_train
            logging.info("freeze vision encoder")

        print(f'VIT params:{sum(p.numel() for p in self.visual_encoder.parameters())}+{sum(p.numel() for p in self.ln_vision.parameters())}')
        print(f'VIT params:{sum(p.numel() for p in self.visual_encoder.parameters() if p.requires_grad)}+{sum(p.numel() for p in self.ln_vision.parameters() if p.requires_grad)}')
        print('Loading VIT Done')

        print('Loading Q-Former')
        self.Qformer, self.query_tokens = self.init_Qformer(
            num_query_token, self.visual_encoder.num_features, q_former_encoder_model=q_former_encoder_model
        )
        self.Qformer.cls = None
        self.Qformer.bert.embeddings.word_embeddings = None
        self.Qformer.bert.embeddings.position_embeddings = None
        for layer in self.Qformer.bert.encoder.layer:
            layer.output = None
            layer.intermediate = None
        self.load_from_pretrained(url_or_filename=q_former_model)


        if freeze_qformer:
            for name, param in self.Qformer.named_parameters():
                param.requires_grad = False
            self.Qformer = self.Qformer.eval()
            self.Qformer.train = disabled_train
            self.query_tokens.requires_grad = False
            logging.info("freeze Qformer")
        logging.info('Loading Q-Former Done')

        print(f'Q-Former params:{sum(p.numel() for p in self.Qformer.parameters())}')
        print(f'Q-Former params:{sum(p.numel() for p in self.Qformer.parameters() if p.requires_grad)}')
        print('Loading Q-Former Done')

        print('Loading LLAMA Tokenizer')
        logging.info('Loading LLAMA Tokenizer')
        self.llama_tokenizer = LlamaTokenizer.from_pretrained(llama_model, use_fast=False)
        
        # if self.A.pad_token is None:
        #     self.llama_tokenizer.pad_token = self.llama_tokenizer.unk_token 
        # if self.A.pad_token is None:
        self.llama_tokenizer.pad_token = self.llama_tokenizer.unk_token 
        DEFAULT_IMAGE_PATCH_TOKEN = '<ImageHere>'
        DEFAULT_AUDIO_PATCH_TOKEN = '<AudioHere>'
        self.llama_tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
        self.llama_tokenizer.add_tokens([DEFAULT_AUDIO_PATCH_TOKEN], special_tokens=True)
        
        self.IMAGE_PATCH_TOKEN_ID = self.llama_tokenizer.get_vocab()[DEFAULT_IMAGE_PATCH_TOKEN]
        self.AUDIO_PATCH_TOKEN_ID = self.llama_tokenizer.get_vocab()[DEFAULT_AUDIO_PATCH_TOKEN]

        logging.info('Loading LLAMA Model')
        if self.low_resource:
            self.llama_model = LlamaForCausalLM.from_pretrained(
                llama_model,
                # torch_dtype=torch.bfloat16,
                torch_dtype=torch.float16,
                load_in_8bit=True,
                device_map={'': device_8bit}
            )
        else:
            self.llama_model = LlamaForCausalLM.from_pretrained(
                llama_model,
                # torch_dtype=torch.bfloat16,
                torch_dtype=torch.float16,
            )

        if llm_lora_config is not None:
            use_lora = llm_lora_config.get("use_lora", False)
            lora_ckpt = llm_lora_config.get("lora_ckpt", None)
        else:
            use_lora = False
            lora_ckpt = None

        for name, param in self.llama_model.named_parameters():
            param.requires_grad = False

        if use_lora:
            logging.info("use lora!!")
            if lora_ckpt is not None:
                logging.info(f"loading lora checkpoint from {lora_ckpt} ...")
                is_trainable = llm_lora_config.get("is_trainable", False)
                self.llama_model = PeftModelForCausalLM.from_pretrained(self.llama_model, lora_ckpt,
                                                                        is_trainable=is_trainable,
                                                                        torch_dtype=torch.float16)
            else:
                lora_r = int(llm_lora_config.lora_r)
                lora_alpha = int(llm_lora_config.lora_alpha)
                lora_target_modules = list(llm_lora_config.lora_target_modules)
                lora_dropout = float(llm_lora_config.lora_dropout)

                logging.info(f"initializing lora config... "
                             f"lora_r: {lora_r} || lora_alpha: {lora_alpha} || "
                             f"lora_target_modules: {lora_target_modules} || lora_dropout: {lora_dropout}")

                lora_config = LoraConfig(
                    r=lora_r,
                    lora_alpha=lora_alpha,
                    target_modules=lora_target_modules,
                    lora_dropout=lora_dropout,
                    bias="none",
                    task_type="CAUSAL_LM",
                )
                self.llama_model = get_peft_model(self.llama_model, lora_config)
            self.llama_model.print_trainable_parameters()

        print(f'LLAMA params:{sum(p.numel() for p in self.llama_model.parameters())}')
        print(f'LLAMA params:{sum(p.numel() for p in self.llama_model.parameters() if p.requires_grad)}')

        logging.info('Loading LLAMA Done')

        logging.info('Loading LLAMA proj')
        self.llama_proj = nn.Linear(
            self.Qformer.config.hidden_size, self.llama_model.config.hidden_size
        )
        if llama_proj_model:
            print("load llama proj weight: {}".format(llama_proj_model))
            llama_proj_weight = torch.load(llama_proj_model, map_location="cpu")
            msg = self.load_state_dict(llama_proj_weight['model'], strict=False)

        if frozen_llama_proj:
            #  todo frozen  llama_proj
            for name, param in self.llama_proj.named_parameters():
                param.requires_grad = False
            logging.info('LLAMA proj is frozen')
        else:
            for name, param in self.llama_proj.named_parameters():
                param.requires_grad = True
            logging.info('LLAMA proj is not frozen')

        print(f'LLAMA proj params:{sum(p.numel() for p in self.llama_proj.parameters())}')
        print(f'LLAMA proj params:{sum(p.numel() for p in self.llama_proj.parameters() if p.requires_grad)}')
        logging.info('Loading llama_proj Done')

        self.max_txt_len = max_txt_len
        self.end_sym = end_sym

        if prompt_path:
            with open(prompt_path, 'r') as f:
                raw_prompts = f.read().splitlines()
            filted_prompts = [raw_prompt for raw_prompt in raw_prompts if "<ImageHere>" in raw_prompt]
            self.prompt_list = [prompt_template.format(p) for p in filted_prompts]
            print('Load {} training prompts'.format(len(self.prompt_list)))
            print('Prompt Example \n{}'.format(random.choice(self.prompt_list)))
        else:
            self.prompt_list = []

        self.video_frame_position_embedding = nn.Embedding(max_frame_pos, self.Qformer.config.hidden_size)

        self.num_video_query_token = num_video_query_token
        self.video_Qformer,self.video_query_tokens = self.init_video_Qformer(num_query_token = num_video_query_token,
                                                                             vision_width=self.Qformer.config.hidden_size,
                                                                             num_hidden_layers=2,
                                                                             q_former_encoder_model=q_former_encoder_model)

        self.video_Qformer.cls = None
        self.video_Qformer.bert.embeddings.word_embeddings = None
        self.video_Qformer.bert.embeddings.position_embeddings = None
        for layer in self.video_Qformer.bert.encoder.layer:
            layer.output = None
            layer.intermediate = None


        if frozen_video_Qformer:
            #  todo frozen  llama_proj
            for name, param in self.video_Qformer.named_parameters():
                param.requires_grad = False
            for name, param in self.video_frame_position_embedding.named_parameters():
                param.requires_grad = False
            self.video_query_tokens.requires_grad = False
            
            logging.info('video_Qformer is frozen')
        else:
            for name, param in self.video_Qformer.named_parameters():
                param.requires_grad = True
            for name, param in self.video_frame_position_embedding.named_parameters():
                param.requires_grad = True
            self.video_query_tokens.requires_grad = True
            logging.info('video_Qformer is not frozen')

        if frozen_video_Qformer and (not frozen_audio_Qformer):
            self.train_flag = 1 # 只训练audio_Qformer
        elif not(frozen_video_Qformer) and frozen_audio_Qformer:
            self.train_flag = 0 # 训练video_Qformer
        elif not(frozen_video_Qformer) and not(frozen_audio_Qformer):
            self.train_flag = 2 # video_Qformer and AL trained
        else:
            self.train_flag = 3


        if equip_audio_branch:
            print (f'Initializing audio encoder from {imagebind_ckpt_path} ...')
            self.audio_encoder,self.audio_hidden_size = \
                imagebind_model.imagebind_huge()
            self.audio_encoder.load_state_dict(torch.load("{}/imagebind_huge.pth".format(imagebind_ckpt_path)))
            # free vision encoder
            for name, param in self.audio_encoder.named_parameters():
                param.requires_grad = False
            self.audio_encoder.eval()
            print ('audio encoder initialized.')
            
            self.num_audio_query_token = num_audio_query_token
            self.audio_Qformer,self.audio_query_tokens = self.init_video_Qformer(num_query_token = self.num_audio_query_token,\
                vision_width=self.audio_hidden_size, num_hidden_layers =2)
            self.audio_Qformer.cls = None
            self.audio_Qformer.bert.embeddings.word_embeddings = None
            self.audio_Qformer.bert.embeddings.position_embeddings = None
            for layer in self.audio_Qformer.bert.encoder.layer:
                layer.output = None
                layer.intermediate = None
            self.audio_llama_proj = nn.Linear(
                self.audio_Qformer.config.hidden_size, self.llama_model.config.hidden_size
            )
            self.audio_position_embedding = nn.Embedding(8, self.audio_hidden_size)

            if frozen_audio_Qformer:
                #  todo frozen  llama_proj
                for name, param in self.audio_Qformer.named_parameters():
                    param.requires_grad = False
                self.audio_query_tokens.requires_grad = False
                for name, param in self.audio_llama_proj.named_parameters():
                    param.requires_grad = False
                for name, param in self.audio_position_embedding.named_parameters():
                    param.requires_grad = False
                logging.info('audio_Qformer and audio-LLAMA proj is frozen')
            else:
                for name, param in self.audio_Qformer.named_parameters():
                    param.requires_grad = True
                self.audio_query_tokens.requires_grad = True
                for name, param in self.audio_llama_proj.named_parameters():
                    param.requires_grad = True
                for name, param in self.audio_position_embedding.named_parameters():
                    param.requires_grad = True
                logging.info('audio_Qformer is not frozen')


        #  self.audio_hidden_size
    def vit_to_cpu(self):
        self.ln_vision.to("cpu")
        self.ln_vision.float()
        self.visual_encoder.to("cpu")
        self.visual_encoder.float()

    def encode_videoQformer_visual(self, image):
        device = image.device
        
        # input shape b,c,t,h,w
        batch_size,_,time_length,_,_ = image.size()
        image = einops.rearrange(image, 'b c t h w -> (b t) c h w')
        with self.maybe_autocast():
            # embed image features with blip2, out: (b t) q h
            image_embeds = self.ln_vision(self.visual_encoder(image)).to(device)
            image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(device)

            query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)
            query_output = self.Qformer.bert(
                query_embeds=query_tokens,
                encoder_hidden_states=image_embeds,
                encoder_attention_mask=image_atts,
                return_dict=True,
            )

            # add frame_pos embedding
            position_ids = torch.arange(time_length, dtype=torch.long, device=query_tokens.device)
            position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)
            frame_position_embeddings = self.video_frame_position_embedding(position_ids)
            q_hidden_state = query_output.last_hidden_state

            frame_position_embeddings = frame_position_embeddings.unsqueeze(-2)
            frame_hidden_state = einops.rearrange(q_hidden_state, '(b t) q h -> b t q h',b=batch_size,t=time_length)
            frame_hidden_state = frame_position_embeddings + frame_hidden_state

            # frame attention
            frame_hidden_state =  einops.rearrange(frame_hidden_state, 'b t q h -> b (t q) h',b=batch_size,t=time_length)
            frame_atts = torch.ones(frame_hidden_state.size()[:-1], dtype=torch.long).to(device)
            video_query_tokens = self.video_query_tokens.expand(frame_hidden_state.shape[0], -1, -1)

            video_query_output = self.video_Qformer.bert(
                query_embeds=video_query_tokens,
                encoder_hidden_states=frame_hidden_state,
                encoder_attention_mask=frame_atts,
                return_dict=True,
                )
            video_hidden = video_query_output.last_hidden_state

            inputs_llama = self.llama_proj(video_hidden)
            atts_llama = torch.ones(inputs_llama.size()[:-1], dtype=torch.long).to(image_embeds.device)
        return inputs_llama, atts_llama
    
    
    def prompt_wrap(self, img_embeds, atts_img, prompt):
        if prompt:
            batch_size = img_embeds.shape[0]
            # print(prompt)
            p_before, p_after = prompt.split('<ImageHere>')
            p_before_tokens = self.llama_tokenizer(
                p_before, return_tensors="pt", add_special_tokens=False).to(img_embeds.device)
            p_after_tokens = self.llama_tokenizer(
                p_after, return_tensors="pt", add_special_tokens=False).to(img_embeds.device)
            p_before_embeds = self.llama_model.get_input_embeddings()(p_before_tokens.input_ids).expand(batch_size, -1, -1)
            p_after_embeds = self.llama_model.get_input_embeddings()(p_after_tokens.input_ids).expand(batch_size, -1, -1)
            wrapped_img_embeds = torch.cat([p_before_embeds, img_embeds, p_after_embeds], dim=1)
            wrapped_atts_img = atts_img[:, :1].expand(-1, wrapped_img_embeds.shape[1])
            
            return wrapped_img_embeds, wrapped_atts_img
        else:
            return img_embeds, atts_img
    #  input audio shape [b t c h w] 
    def encode_audioQformer(self, audio,modality_type=ModalityType.AUDIO):
        device = audio.device
        with self.maybe_autocast():
            audio_feature, audio_imagebind_finalout = self.audio_encoder.get_audio_feature(audio,modality_type=modality_type)
            batch_size,time_length = audio.size()[:2]


            position_ids = torch.arange(time_length, dtype=torch.long, device=device)
            position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)

            audio_position_embeddings = self.audio_position_embedding(position_ids)
            audio_imagebind_finalout = audio_imagebind_finalout + audio_position_embeddings

            audio_query_tokens = self.audio_query_tokens.expand(audio_imagebind_finalout.shape[0], -1, -1)
            frame_atts = torch.ones(audio_imagebind_finalout.size()[:-1], dtype=torch.long).to(device)

            audio_query_output = self.audio_Qformer.bert(
                query_embeds=audio_query_tokens, #[32,768]
                encoder_hidden_states=audio_imagebind_finalout,
                encoder_attention_mask=frame_atts,
                return_dict=True,
                )
            audio_hidden = audio_query_output.last_hidden_state

            inputs_llama = self.audio_llama_proj(audio_hidden)
            atts_llama = torch.ones(inputs_llama.size()[:-1], dtype=torch.long).to(device)
    
        return inputs_llama, atts_llama

    def encode_videoQformer_audiovideo(self, image, audio):
        device = image.device
        
        # input shape b,c,t,h,w
        batch_size,_,time_length,_,_ = image.size()
        image = einops.rearrange(image, 'b c t h w -> (b t) c h w')
        with self.maybe_autocast():
            # embed image features with blip2, out: (b t) q h
            image_embeds = self.ln_vision(self.visual_encoder(image)).to(device)
            image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(device)

            query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)
            query_output = self.Qformer.bert(
                query_embeds=query_tokens,
                encoder_hidden_states=image_embeds,
                encoder_attention_mask=image_atts,
                return_dict=True,
            )
            
            # add frame_pos embedding
            position_ids = torch.arange(time_length, dtype=torch.long, device=query_tokens.device)
            position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)
            frame_position_embeddings = self.video_frame_position_embedding(position_ids)
            q_hidden_state = query_output.last_hidden_state

            frame_position_embeddings = frame_position_embeddings.unsqueeze(-2)
            frame_hidden_state = einops.rearrange(q_hidden_state, '(b t) q h -> b t q h',b=batch_size,t=time_length)
            frame_hidden_state = frame_position_embeddings + frame_hidden_state

            # encode audio 
            audio_feature, audio_imagebind_finalout = self.audio_encoder.get_audio_feature(audio,modality_type=ModalityType.AUDIO) # [batch,8*1,768]    8*32, 768
            audio_frame_position_embeddings = frame_position_embeddings.squeeze(-2)
            audio_feature = audio_feature + audio_frame_position_embeddings

            # frame attention a
            frame_hidden_state =  einops.rearrange(frame_hidden_state, 'b t q h -> b (t q) h',b=batch_size,t=time_length)
            frame_hidden_state = torch.cat([frame_hidden_state,audio_feature],dim = 1)
            video_query_tokens = self.video_query_tokens.expand(frame_hidden_state.shape[0], -1, -1)
            frame_atts = torch.ones(frame_hidden_state.size()[:-1], dtype=torch.long).to(device)

            video_query_output = self.video_Qformer.bert(
                query_embeds=video_query_tokens, #[32,768]
                encoder_hidden_states=frame_hidden_state,
                encoder_attention_mask=frame_atts,
                return_dict=True,
                )
            video_hidden = video_query_output.last_hidden_state

            inputs_llama = self.llama_proj(video_hidden)
            atts_llama = torch.ones(inputs_llama.size()[:-1], dtype=torch.long).to(image_embeds.device)
    
        return inputs_llama, atts_llama

    def reorganize_llm_inputs(self, input_ids, attention_mask, targets, video_embeds, atts_video, img_embeds):
        image_patch_token_id = self.IMAGE_PATCH_TOKEN_ID
        video_patch_token_id = self.AUDIO_PATCH_TOKEN_ID
        num_patch_tokens = self.num_video_query_token
        num_video_patch_tokens = atts_video.shape[-1]
        # print(f"num_video_patch_tokens: {num_video_patch_tokens}")
        # print(f"num_image_patch_tokens: {num_patch_tokens}")

        temp_input_ids = copy.deepcopy(input_ids)
        temp_input_ids[temp_input_ids == image_patch_token_id] = 0
        temp_input_ids[temp_input_ids == video_patch_token_id] = 0
        temp_input_embedding = self.llama_model.get_input_embeddings()(temp_input_ids)

        # print(f"pad_token_id: {self.llama_tokenizer.pad_token_id}")
        pad_token_embed = self.llama_model.get_input_embeddings()(
        torch.tensor([self.llama_tokenizer.pad_token_id], device=temp_input_ids.device))
        new_input_embeds = []
        new_attention_mask = []
        new_targets = []
        cur_image_idx = 0
        for cur_input_ids, cur_input_embeds, cur_attention_mask, cur_targets in zip(input_ids, temp_input_embedding,
                                                                                    attention_mask, targets):
            cur_video_features = video_embeds[cur_image_idx]  # num_patch_tokens*n_actions * hidden_size
            cur_atts_video = atts_video[cur_image_idx]
            cur_image_features = img_embeds[cur_image_idx]

            if (cur_input_ids == video_patch_token_id).sum() != num_video_patch_tokens:
                raise ValueError(
                    "The number of video patch tokens should be the same as the number of video patches.")

            if (cur_input_ids == image_patch_token_id).sum() != num_patch_tokens:
                raise ValueError(
                    "The number of image patch tokens should be the same as the number of image patches.")

            masked_indices_video = torch.where(cur_input_ids == video_patch_token_id)[0]
            masked_indices_image = torch.where(cur_input_ids == image_patch_token_id)[0]

            video_mask_start = masked_indices_video[0]
            image_mask_start = masked_indices_image[0]

            # reorganize input_embeds
            cur_embeds_before_video = cur_input_embeds[:video_mask_start]
            cur_embeds_after_video_before_image = cur_input_embeds[
                                                  video_mask_start + num_video_patch_tokens: image_mask_start]
            cur_embeds_after_image = cur_input_embeds[image_mask_start + num_patch_tokens:]

            hidden_size = cur_video_features.shape[-1]
            cur_video_features_valid = cur_video_features[cur_atts_video != 0].reshape(-1, hidden_size)
            cur_video_features_invalid = cur_video_features[cur_atts_video == 0].reshape(-1, hidden_size)

            cur_video_features_padding = pad_token_embed.expand_as(cur_video_features_invalid)

            cur_new_input_embeds = torch.cat((
                cur_embeds_before_video, cur_video_features_valid,
                cur_embeds_after_video_before_image, cur_image_features,
                cur_embeds_after_image,
                cur_video_features_padding
            ), dim=0)
            new_input_embeds.append(cur_new_input_embeds)

            # reorganize attention_mask
            cur_attention_mask_before_video = cur_attention_mask[:video_mask_start]
            cur_attention_mask_video = cur_attention_mask[
                                       video_mask_start: video_mask_start + num_video_patch_tokens]  # num_patch_tokens*n_actions
            cur_attention_mask_after_video = cur_attention_mask[video_mask_start + num_video_patch_tokens:]
            cur_attention_mask_video[cur_atts_video == 0] = 0
            cur_attention_mask_video_valid = cur_attention_mask_video[cur_atts_video != 0]
            cur_attention_mask_video_padding = cur_attention_mask_video[cur_atts_video == 0]

            cur_new_attention_mask = torch.cat(
                (cur_attention_mask_before_video, cur_attention_mask_video_valid,
                 cur_attention_mask_after_video,
                 cur_attention_mask_video_padding), dim=0)
            new_attention_mask.append(cur_new_attention_mask)

            # reorganize targets
            cur_targets_before_video = cur_targets[:video_mask_start]
            cur_targets_video = cur_targets[
                                video_mask_start: video_mask_start + num_video_patch_tokens]  # num_patch_tokens*n_actions
            cur_targets_after_video = cur_targets[video_mask_start + num_video_patch_tokens:]
            cur_targets_video_valid = cur_targets_video[cur_atts_video != 0]
            cur_targets_video_padding = cur_targets_video[cur_atts_video == 0]

            cur_new_targets = torch.cat((
                cur_targets_before_video, cur_targets_video_valid,
                cur_targets_after_video,
                cur_targets_video_padding
            ), dim=0)
            new_targets.append(cur_new_targets)

            cur_image_idx += 1

        inputs_embeds = torch.stack(new_input_embeds, dim=0)
        attention_mask = torch.stack(new_attention_mask, dim=0)
        targets = torch.stack(new_targets, dim=0)

        return inputs_embeds, attention_mask, targets

    def forward(self, samples, reduction="mean"):
        if 'conv_type' in samples.keys() and samples['conv_type'] == 'egoplan':
            image = samples["image"]
            clips = samples["clips"] # B, N, C, T, H, W
            clip_mask = samples["clip_mask"] # B, N
            n_actions = clip_mask.shape[-1]

            input_ids = samples['input_ids']
            targets = samples['labels']
            attention_mask = samples['attention_mask'] # B, seq_len

            # encode image
            time = 1
            image = einops.repeat(image, 'b c h w -> b c t h w', t=time) # B, C, T, H, W
            img_embeds, atts_img = self.encode_videoQformer_visual(image)

            # encode clips
            video_embeds, atts_video = [], []
            for clip_idx in range(clips.shape[1]):
                this_clip = clips[:, clip_idx, :, :, :, :] # B, C, T, H, W
                this_clip_mask = clip_mask[:, clip_idx] # B

                this_clip_embeds, this_atts_clip = self.encode_videoQformer_visual(this_clip)
                this_atts_clip[~this_clip_mask] = 0

                video_embeds.append(this_clip_embeds) # B, num_query_tokens, hidden_size
                atts_video.append(this_atts_clip) # B, num_query_tokens

            video_embeds = torch.cat(video_embeds, dim=1) # B, N_ACTIONS * num_query_tokens, hidden_size
            atts_video = torch.cat(atts_video, dim=1) # B, N_ACTIONS * num_query_tokens

            inputs_embeds, attention_mask, targets = self.reorganize_llm_inputs(
                input_ids, attention_mask, targets, video_embeds, atts_video, img_embeds)

            with self.maybe_autocast():
                outputs = self.llama_model(
                    inputs_embeds=inputs_embeds,
                    attention_mask=attention_mask,
                    return_dict=True,
                    labels=targets,
                    reduction=reduction
                )
            loss = outputs.loss
            return {"loss": loss}

        if 'conv_type' in samples.keys() and samples['conv_type'] == 'egoplan_contrastive':
            image = samples["image"]
            clips = samples["clips"] # B, N, C, T, H, W
            clip_mask = samples["clip_mask"] # B, N
            n_actions = clip_mask.shape[-1]

            # encode image
            time = 1
            image = einops.repeat(image, 'b c h w -> b c t h w', t=time) # B, C, T, H, W
            img_embeds, atts_img = self.encode_videoQformer_visual(image)

            # encode clips
            video_embeds, atts_video = [], []
            for clip_idx in range(clips.shape[1]):
                this_clip = clips[:, clip_idx, :, :, :, :] # B, C, T, H, W
                this_clip_mask = clip_mask[:, clip_idx] # B

                this_clip_embeds, this_atts_clip = self.encode_videoQformer_visual(this_clip)
                this_atts_clip[~this_clip_mask] = 0

                video_embeds.append(this_clip_embeds) # B, num_query_tokens, hidden_size
                atts_video.append(this_atts_clip) # B, num_query_tokens

            video_embeds = torch.cat(video_embeds, dim=1) # B, N_ACTIONS * num_query_tokens, hidden_size
            atts_video = torch.cat(atts_video, dim=1) # B, N_ACTIONS * num_query_tokens

            # encode positive llm inputs
            positive_input_ids = samples['input_ids']
            positive_targets = samples['labels']
            positive_attention_mask = samples['attention_mask']  # B, seq_len
            positive_inputs_embeds, positive_attention_mask, positive_targets = self.reorganize_llm_inputs(
                positive_input_ids, positive_attention_mask, positive_targets, video_embeds, atts_video, img_embeds)

            # encode negative llm inputs
            negative_input_ids = samples['negative_input_ids']
            negative_targets = samples['negative_labels']
            negative_attention_mask = samples['negative_attention_mask']  # B, seq_len
            negative_inputs_embeds, negative_attention_mask, negative_targets = self.reorganize_llm_inputs(
                negative_input_ids, negative_attention_mask, negative_targets, video_embeds, atts_video, img_embeds)


            with self.maybe_autocast():
                outputs = self.llama_model(
                    inputs_embeds=torch.cat([positive_inputs_embeds, negative_inputs_embeds]),
                    attention_mask=torch.cat([positive_attention_mask, negative_attention_mask]),
                    return_dict=True,
                    labels=torch.cat([positive_targets, negative_targets]),
                    reduction="none"
                )
            loss = outputs.loss
            batch_size = positive_inputs_embeds.shape[0]
            positive_loss = loss[:batch_size]
            sft_loss = positive_loss.mean()

            negative_loss = loss[batch_size:]
            loss_diff = positive_loss - negative_loss
            loss_diff = loss_diff[loss_diff>0]
            if loss_diff.shape[0] > 0:
                rank_loss = loss_diff.mean()
                loss = sft_loss + rank_loss
            else:
                loss = sft_loss
            return {"loss": loss}

        elif 'conv_type' in samples.keys() and samples['conv_type']=='multi':
            
            im_patch_token_id = self.IMAGE_PATCH_TOKEN_ID
            image = samples["images"]
            input_ids = samples['input_ids']
            if len(image.size())==4:
                time = 1
                image = einops.repeat(image, 'b c h w -> b c t h w',t = time)

            if self.train_flag == 0:
                num_patch_tokens = self.num_video_query_token
                img_embeds, atts_img = self.encode_videoQformer_visual(image)
            elif self.train_flag == 1:
                num_patch_tokens = self.num_audio_query_token
                image = einops.rearrange(image, 'b c t h w -> b t c h w')
                img_embeds, atts_img = self.encode_audioQformer(image, modality_type=ModalityType.VISION)
            else:
                num_patch_tokens = self.num_video_query_token
                img_embeds, atts_img = self.encode_videoQformer_visual(image)
                
            temp_input_ids = copy.deepcopy(input_ids)
            temp_input_ids[temp_input_ids == im_patch_token_id] = 0
            temp_input_embedding = self.llama_model.get_input_embeddings()(temp_input_ids)

            new_input_embeds=[]
            cur_image_idx = 0
            for cur_input_ids, cur_input_embeds in zip(input_ids, temp_input_embedding):
                cur_image_features = img_embeds[cur_image_idx]

                if (cur_input_ids == im_patch_token_id).sum() != num_patch_tokens:
                        raise ValueError("The number of image patch tokens should be the same as the number of image patches.")
                masked_indices = torch.where(cur_input_ids == im_patch_token_id)[0]
                mask_index_start = masked_indices[0]
                if (masked_indices != torch.arange(mask_index_start, mask_index_start+num_patch_tokens, device=masked_indices.device, dtype=masked_indices.dtype)).any():
                    raise ValueError("The image patch tokens should be consecutive.")
                
                cur_new_input_embeds = torch.cat((cur_input_embeds[:mask_index_start], cur_image_features, cur_input_embeds[mask_index_start+num_patch_tokens:]), dim=0)
                new_input_embeds.append(cur_new_input_embeds)
                
                cur_image_idx+=1
            inputs_embeds = torch.stack(new_input_embeds, dim=0)
            targets = samples['labels']
            attention_mask = samples['attention_mask']
            with self.maybe_autocast():
                outputs = self.llama_model(
                    inputs_embeds=inputs_embeds,
                    attention_mask=attention_mask,
                    return_dict=True,
                    labels=targets,
                    reduction=reduction
                )
            loss = outputs.loss
            return {"loss": loss}
        elif 'conv_type' in samples.keys() and samples['conv_type'] == 'egoplan_action_recog':
            image = samples["image"]
            clips = samples["clips"] # B, N, C, T, H, W
            clip_mask = samples["clip_mask"] # B, N
            n_actions = clip_mask.shape[-1]

            input_ids = samples['input_ids']
            targets = samples['labels']
            attention_mask = samples['attention_mask'] # B, seq_len

            # encode image
            time = 1
            image = einops.repeat(image, 'b c h w -> b c t h w', t=time) # B, C, T, H, W
            img_embeds, atts_img = self.encode_videoQformer_visual(image)

            # encode clips
            video_embeds, atts_video = [], []
            for clip_idx in range(clips.shape[1]):
                this_clip = clips[:, clip_idx, :, :, :, :] # B, C, T, H, W
                this_clip_mask = clip_mask[:, clip_idx] # B

                this_clip_embeds, this_atts_clip = self.encode_videoQformer_visual(this_clip)
                this_atts_clip[~this_clip_mask] = 0

                video_embeds.append(this_clip_embeds) # B, num_query_tokens, hidden_size
                atts_video.append(this_atts_clip) # B, num_query_tokens

            video_embeds = torch.cat(video_embeds, dim=1) # B, N_ACTIONS * num_query_tokens, hidden_size
            atts_video = torch.cat(atts_video, dim=1) # B, N_ACTIONS * num_query_tokens

            inputs_embeds, attention_mask, targets = self.reorganize_llm_inputs(
                input_ids, attention_mask, targets, video_embeds, atts_video, img_embeds)

            with self.maybe_autocast():
                outputs = self.llama_model.generate(
                    inputs_embeds=inputs_embeds,
                    attention_mask=attention_mask,
                    max_length=30,
                )
            ###########################################
            # copy and modify from https://github.com/DAMO-NLP-SG/Video-LLaMA/blob/64888c0a77e8e66af606c37629fbbf79c4582959/video_llama/conversation/conversation_video.py#L206
            output_token = outputs[0]
            if output_token[0] == 0:  # the model might output a unknown token <unk> at the beginning. remove it
                output_token = output_token[1:]
            if output_token[0] == 1:  # some users find that there is a start token <s> at the beginning. remove it
                output_token = output_token[1:]
            if output_token[-1] == 0:  # the model might output a unknown token <unk> at the beginning. remove it
                output_token = output_token[:-1]
            if output_token[-1] == 2:  # some users find that there is an end token </s> at the beginning. remove it
                output_token = output_token[:-1]
            output_text = self.llama_tokenizer.decode(output_token, add_special_tokens=False)
            ###########################################
            return {"output_text": output_text}
        elif 'conv_type' in samples.keys() and samples['conv_type'] == 'egoplan_action_recog_v2':
            image = samples["image"]
            clips = samples["clips"] # B, N, C, T, H, W
            clip_mask = samples["clip_mask"] # B, N
            n_actions = clip_mask.shape[-1]

            input_ids = samples['input_ids']
            targets = samples['labels']
            attention_mask = samples['attention_mask'] # B, seq_len

            # encode image
            time = 1
            image = einops.repeat(image, 'b c h w -> b c t h w', t=time) # B, C, T, H, W
            img_embeds, atts_img = self.encode_videoQformer_visual(image)

            # encode clips
            video_embeds, atts_video = [], []
            for clip_idx in range(clips.shape[1]):
                this_clip = clips[:, clip_idx, :, :, :, :] # B, C, T, H, W
                this_clip_mask = clip_mask[:, clip_idx] # B

                this_clip_embeds, this_atts_clip = self.encode_videoQformer_visual(this_clip)
                this_atts_clip[~this_clip_mask] = 0

                video_embeds.append(this_clip_embeds) # B, num_query_tokens, hidden_size
                atts_video.append(this_atts_clip) # B, num_query_tokens

            video_embeds = torch.cat(video_embeds, dim=1) # B, N_ACTIONS * num_query_tokens, hidden_size
            atts_video = torch.cat(atts_video, dim=1) # B, N_ACTIONS * num_query_tokens

            inputs_embeds, attention_mask, targets = self.reorganize_llm_inputs(
                input_ids, attention_mask, targets, video_embeds, atts_video, img_embeds)

            # with self.maybe_autocast():
            outputs = self.llama_model.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
            )

            output_texts = []
            ###########################################
            # copy and modify from https://github.com/DAMO-NLP-SG/Video-LLaMA/blob/64888c0a77e8e66af606c37629fbbf79c4582959/video_llama/conversation/conversation_video.py#L206
            for idx_output in range(len(outputs)):
                output_token = outputs[idx_output]
                if output_token[0] == 0:  # the model might output a unknown token <unk> at the beginning. remove it
                    output_token = output_token[1:]
                if output_token[0] == 1:  # some users find that there is a start token <s> at the beginning. remove it
                    output_token = output_token[1:]
                if output_token[-1] == 0:  # the model might output a unknown token <unk> at the beginning. remove it
                    output_token = output_token[:-1]
                if output_token[-1] == 2:  # some users find that there is an end token </s> at the beginning. remove it
                    output_token = output_token[:-1]
                output_text = self.llama_tokenizer.decode(output_token, add_special_tokens=False)
                output_texts.append(output_text)
            ###########################################
            return {"output_texts": output_texts}
        elif 'conv_type' in samples.keys() and samples['conv_type'] == 'egoplan_semantic_matching':
            image = samples["image"]
            clips = samples["clips"] # B, N, C, T, H, W
            clip_mask = samples["clip_mask"] # B, N
            n_actions = clip_mask.shape[-1]

            input_ids = samples['input_ids']
            targets = samples['labels']
            attention_mask = samples['attention_mask'] # B, seq_len

            # encode clips
            # video_embeds, atts_video = [], []
            clip_idx = 0
            this_clip = clips[:, clip_idx, :, :, :, :] # B, C, T, H, W
            this_clip_mask = clip_mask[:, clip_idx] # B
            assert this_clip_mask.sum().item() == len(this_clip_mask)

            this_clip_embeds, this_atts_clip = self.encode_videoQformer_visual(this_clip)
            avg_pool_this_clip_embeds = torch.mean(this_clip_embeds, dim=1)

            # video_embeds.append(this_clip_embeds) # B, num_query_tokens, hidden_size
            # atts_video.append(this_atts_clip) # B, num_query_tokens

            # video_embeds = torch.cat(video_embeds, dim=1) # B, N_ACTIONS * num_query_tokens, hidden_size
            # atts_video = torch.cat(atts_video, dim=1) # B, N_ACTIONS * num_query_tokens

            targets[targets == -100] = 0
            targets[targets == 2] = 0
            target_embeds = self.llama_model.get_input_embeddings()(targets)
            avg_pool_target_embeds = (target_embeds * targets[:, :, None]).sum(dim=1) / targets.bool().sum(dim=1)[:, None]
            loss = torch.nn.CosineEmbeddingLoss()(avg_pool_this_clip_embeds, avg_pool_target_embeds, torch.ones(1).to(clips.device))

            return {"loss": loss}
        elif 'conv_type' in samples.keys() and samples['conv_type'] == 'egoplan_semantic_matching_v3':
            image = samples["image"]
            clips = samples["clips"] # B, N, C, T, H, W
            clip_mask = samples["clip_mask"] # B, N
            n_actions = clip_mask.shape[-1]

            # encode image
            time = 1
            image = einops.repeat(image, 'b c h w -> b c t h w', t=time) # B, C, T, H, W
            img_embeds, atts_img = self.encode_videoQformer_visual(image)

            # encode clips
            video_embeds, atts_video = [], []
            for clip_idx in range(clips.shape[1]):
                this_clip = clips[:, clip_idx, :, :, :, :] # B, C, T, H, W
                this_clip_mask = clip_mask[:, clip_idx] # B

                this_clip_embeds, this_atts_clip = self.encode_videoQformer_visual(this_clip)
                this_atts_clip[~this_clip_mask] = 0

                video_embeds.append(this_clip_embeds) # B, num_query_tokens, hidden_size
                atts_video.append(this_atts_clip) # B, num_query_tokens

            video_embeds = torch.cat(video_embeds, dim=1) # B, N_ACTIONS * num_query_tokens, hidden_size
            atts_video = torch.cat(atts_video, dim=1) # B, N_ACTIONS * num_query_tokens

            # encode positive llm inputs
            positive_input_ids = samples['input_ids']
            positive_targets = samples['labels']
            positive_attention_mask = samples['attention_mask']  # B, seq_len

            positive_inputs_embeds, positive_attention_mask, positive_targets = self.reorganize_llm_inputs(
                positive_input_ids, positive_attention_mask, positive_targets, video_embeds, atts_video, img_embeds)

            video_inputs_embeds = positive_inputs_embeds.clone()
            answer_inputs_embeds = positive_inputs_embeds.clone()
            negative_inputs_embeds = positive_inputs_embeds.clone()

            video_attention_mask = positive_attention_mask.clone()
            answer_attention_mask = positive_attention_mask.clone()
            negative_attention_mask = positive_attention_mask.clone()

            video_ends = []
            answer_ends = []
            negative_ends = []

            pad_token_embed = self.llama_model.get_input_embeddings()(
                torch.tensor([self.llama_tokenizer.pad_token_id], device=positive_input_ids.device))

            for xxx in range(len(clips)):
                start = torch.where(positive_input_ids[xxx] == 32001)[0][0].item()
                end = start + 32
                video_ends.append(end)
                video_inputs_embeds[xxx, end:] = pad_token_embed
                video_attention_mask[xxx, end:] = False

                task_goal = samples['raw'][xxx]['task_goal'].strip(string.punctuation + " ").lower()
                if "goal" in task_goal:
                    task_goal = task_goal.split("to", 1)[1].strip()

                answer = samples['positive_answer'][xxx].strip(string.punctuation + " ").lower()
                negative = samples['negative_answer'][xxx].strip(string.punctuation + " ").lower()

                answer_text = self.llama_tokenizer(f'In a video tasked with {task_goal}, I do {answer}.', add_special_tokens=False, return_tensors="pt")['input_ids'].to(clips.device)
                negative_text = self.llama_tokenizer(f'In a video tasked with {task_goal}, I do {negative}.', add_special_tokens=False, return_tensors="pt")['input_ids'].to(clips.device)

                answer_embedding = self.llama_model.get_input_embeddings()(answer_text)[0]
                negative_embedding = self.llama_model.get_input_embeddings()(negative_text)[0]

                answer_end = start + len(answer_embedding)
                negative_end = start + len(negative_embedding)
                answer_ends.append(answer_end)
                negative_ends.append(negative_end)

                answer_inputs_embeds[xxx, start:answer_end] = answer_embedding
                answer_inputs_embeds[xxx, answer_end:] = pad_token_embed
                answer_attention_mask[xxx, start:answer_end] = True
                answer_attention_mask[xxx, answer_end:] = False

                negative_inputs_embeds[xxx, start:negative_end] = negative_embedding
                negative_inputs_embeds[xxx, negative_end:] = pad_token_embed
                negative_attention_mask[xxx, start:negative_end] = True
                negative_attention_mask[xxx, negative_end:] = False

            with self.maybe_autocast():
                outputs = self.llama_model(
                    inputs_embeds=torch.cat([video_inputs_embeds, answer_inputs_embeds, negative_inputs_embeds]),
                    attention_mask=torch.cat([video_attention_mask, answer_attention_mask, negative_attention_mask]),
                    return_dict=True,
                    output_hidden_states=True,
                    labels=None,
                    reduction="none"
                )

            hidden = outputs['hidden_states'][-1]

            video_hidden = hidden[torch.arange(0,len(clips)), video_ends]
            answer_hidden = hidden[torch.arange(len(clips), 2*len(clips)), answer_ends]
            negative_hidden = hidden[torch.arange(2*len(clips), 3*len(clips)), negative_ends]

            temperature = 0.1
            pos_similarity = torch.nn.functional.cosine_similarity(video_hidden, answer_hidden.detach()) / temperature
            neg_similarity = torch.nn.functional.cosine_similarity(video_hidden, negative_hidden.detach()) / temperature

            logits = torch.cat([pos_similarity, neg_similarity], dim=0)
            labels = torch.cat([torch.ones(pos_similarity.size(0)), torch.zeros(neg_similarity.size(0))], dim=0).to(clips.device)

            loss = torch.nn.functional.cross_entropy(logits, labels)
            print(loss.item())

            return {"loss": loss}
        else:
            image = samples["image"]

            if len(image.size()) != 5:
                time = 1
                image = einops.repeat(image, 'b c h w -> b c t h w',t = time)
            
            if self.train_flag == 1:
                image = einops.rearrange(image, 'b c t h w -> b t c h w')
                img_embeds, atts_img = self.encode_audioQformer(image, modality_type=ModalityType.VISION)
            else:
                img_embeds, atts_img = self.encode_videoQformer_visual(image)

            if self.prompt_list:
                prompt = random.choice(self.prompt_list)
                img_embeds, atts_img = self.prompt_wrap(img_embeds, atts_img, prompt)
                

            self.llama_tokenizer.padding_side = "right"

            text = [t + self.end_sym for t in samples["text_input"]]

            to_regress_tokens = self.llama_tokenizer(
                text,
                return_tensors="pt",
                padding="longest",
                truncation=True,
                max_length=self.max_txt_len,
                add_special_tokens=False
            ).to(image.device)

            targets = to_regress_tokens.input_ids.masked_fill(
                to_regress_tokens.input_ids == self.llama_tokenizer.pad_token_id, -100
            )

            empty_targets = (
                torch.ones([atts_img.shape[0], atts_img.shape[1]+1],
                        dtype=torch.long).to(image.device).fill_(-100)  # plus one for bos
            )
            targets = torch.cat([empty_targets, targets], dim=1)

            batch_size = img_embeds.shape[0]
            bos = torch.ones([batch_size, 1],
                            dtype=to_regress_tokens.input_ids.dtype,
                            device=to_regress_tokens.input_ids.device) * self.llama_tokenizer.bos_token_id
            bos_embeds = self.llama_model.get_input_embeddings()(bos)
            atts_bos = atts_img[:, :1]

            to_regress_embeds = self.llama_model.get_input_embeddings()(to_regress_tokens.input_ids)
            inputs_embeds = torch.cat([bos_embeds, img_embeds, to_regress_embeds], dim=1)
            attention_mask = torch.cat([atts_bos, atts_img, to_regress_tokens.attention_mask], dim=1)

            with self.maybe_autocast():
                outputs = self.llama_model(
                    inputs_embeds=inputs_embeds,
                    attention_mask=attention_mask,
                    return_dict=True,
                    labels=targets,
                    reduction=reduction
                )
            loss = outputs.loss

        return {"loss": loss}

    @classmethod
    def from_config(cls, cfg):
        print(cfg)
        vit_model = cfg.get("vit_model", "eva_clip_g")
        vit_model_path = cfg.get("vit_model_path", None)
        q_former_model = cfg.get("q_former_model", "EgoPlan-challenge-Team-AAILab/src/video_llama/video_llama/models/blip2_pretrained_flant5xxl.pth")
        q_former_encoder_model = cfg.get("q_former_encoder_model", "q_former_encoder_model")

        img_size = cfg.get("image_size")
        num_query_token = cfg.get("num_query_token")
        llama_model = cfg.get("llama_model")

        drop_path_rate = cfg.get("drop_path_rate", 0)
        use_grad_checkpoint = cfg.get("use_grad_checkpoint", False)
        vit_precision = cfg.get("vit_precision", "fp16")
        freeze_vit = cfg.get("freeze_vit", True)
        freeze_qformer = cfg.get("freeze_qformer", True)
        low_resource = cfg.get("low_resource", False)
        device_8bit = cfg.get("device_8bit", 0)

        prompt_path = cfg.get("prompt_path", "")
        prompt_template = cfg.get("prompt_template", "")
        max_txt_len = cfg.get("max_txt_len", 32)
        end_sym = cfg.get("end_sym", '\n')
        
        frozen_llama_proj = cfg.get("frozen_llama_proj", True)
        frozen_video_Qformer = cfg.get("frozen_video_Qformer", True)
        frozen_audio_Qformer = cfg.get("frozen_audio_Qformer", True)

        llama_proj_model = cfg.get("llama_proj_model", '')
        
        fusion_header_type = cfg.get("fusion_header_type", 'seqTransf')
        max_frame_pos = cfg.get("max_frame_pos", 32)
        fusion_head_layers = cfg.get("fusion_head_layers", 2)
        num_video_query_token =  cfg.get("num_video_query_token", 32)

        equip_audio_branch= cfg.get("equip_audio_branch", True)
        num_audio_query_token =  cfg.get("num_audio_query_token", 8)
        imagebind_ckpt_path = cfg.get("imagebind_ckpt_path", '/mnt/workspace/ckpt')

        llm_lora_config = cfg.get("llm_lora_config", None)

        model = cls(
            vit_model=vit_model,
            vit_model_path=vit_model_path,
            q_former_model=q_former_model,
            q_former_encoder_model=q_former_encoder_model,
            img_size=img_size,
            drop_path_rate=drop_path_rate,
            use_grad_checkpoint=use_grad_checkpoint,
            vit_precision=vit_precision,
            freeze_vit=freeze_vit,
            freeze_qformer=freeze_qformer,
            num_query_token=num_query_token,
            llama_model=llama_model,
            prompt_path=prompt_path,
            prompt_template=prompt_template,
            max_txt_len=max_txt_len,
            end_sym=end_sym,
            low_resource=low_resource,
            device_8bit=device_8bit,
            fusion_header_type=fusion_header_type,
            max_frame_pos=max_frame_pos,
            fusion_head_layers=fusion_head_layers,
            frozen_llama_proj=frozen_llama_proj,
            frozen_video_Qformer=frozen_video_Qformer,
            frozen_audio_Qformer=frozen_audio_Qformer,
            num_video_query_token=num_video_query_token,
            num_audio_query_token = num_audio_query_token,
            imagebind_ckpt_path = imagebind_ckpt_path,
            equip_audio_branch = equip_audio_branch,
            llama_proj_model = llama_proj_model,
            llm_lora_config = llm_lora_config,
        )

        ckpt_path = cfg.get("ckpt", "")  # load weights of VideoLlaMA
        if ckpt_path:
            print("Load first Checkpoint: {}".format(ckpt_path))
            ckpt = torch.load(ckpt_path, map_location="cpu")
            msg = model.load_state_dict(ckpt['model'], strict=False)
        ckpt_path_2 = cfg.get("ckpt_2", "")  
        if ckpt_path_2:
            print("Load second Checkpoint: {}".format(ckpt_path_2))
            ckpt = torch.load(ckpt_path_2, map_location="cpu")
            msg = model.load_state_dict(ckpt['model'], strict=False)
        return model
