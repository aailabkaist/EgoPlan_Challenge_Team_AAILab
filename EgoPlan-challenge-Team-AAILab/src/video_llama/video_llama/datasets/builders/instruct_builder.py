import os
import logging
import warnings

from src.video_llama.video_llama.common.registry import registry
from src.video_llama.video_llama.datasets.builders.base_dataset_builder import BaseDatasetBuilder
from src.video_llama.video_llama.datasets.datasets.laion_dataset import LaionDataset
from src.video_llama.video_llama.datasets.datasets.llava_instruct_dataset import Instruct_Dataset
from src.video_llama.video_llama.datasets.datasets.video_instruct_dataset import Video_Instruct_Dataset
from src.video_llama.video_llama.datasets.datasets.egoplan_video_instruct_dataset import Egoplan_Video_Instruct_Dataset, Egoplan_Video_Contrastive_Dataset, Egoplan_Separate_Video_Instruct_Dataset, Egoplan_Video_Instruct_Dataset_v3, Egoplan_Video_Instruct_DPO_Dataset, Egoplan_Video_Instruct_Bound_Dataset, Egoplan_Video_Instruct_Semantic_Matching_Dataset
from src.video_llama.video_llama.datasets.datasets.egoplan_video_instruct_dataset import Egoplan_Video_Contrastive_Dataset_KH
from src.video_llama.video_llama.datasets.datasets.egoplan_video_instruct_dataset import Egoplan_Video_Contrastive_Dataset_Bound
from src.video_llama.video_llama.datasets.datasets.egoplan_video_instruct_dataset import Egoplan_Video_Contrastive_Dataset_MDPO_Video

@registry.register_builder("instruct")
class Instruct_Builder(BaseDatasetBuilder):
    train_dataset_cls = Instruct_Dataset

    DATASET_CONFIG_DICT = {"default": "configs/datasets/instruct/defaults.yaml"}

    def _download_ann(self):
        pass

    def _download_vis(self):
        pass

    def build(self):
        self.build_processors()
        datasets = dict()
        split = "train"

        build_info = self.config.build_info
        dataset_cls = self.train_dataset_cls
        if self.config.num_video_query_token:
            num_video_query_token = self.config.num_video_query_token 
        else:
            num_video_query_token = 32

        if self.config.tokenizer_name:
            tokenizer_name = self.config.tokenizer_name 
        else:
            tokenizer_name = '/mnt/workspace/ckpt/vicuna-13b/'


        datasets[split] = dataset_cls(
            vis_processor=self.vis_processors[split],
            text_processor=self.text_processors[split],
            vis_root=build_info.videos_dir,
            ann_root=build_info.anno_dir,
            num_video_query_token = num_video_query_token,
            tokenizer_name = tokenizer_name,
            data_type = self.config.data_type,
            model_type = self.config.model_type
        )

        return datasets

@registry.register_builder("webvid_instruct")
class WebvidInstruct_Builder(Instruct_Builder):
    train_dataset_cls = Video_Instruct_Dataset
    # print(train_dataset_cls)

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/instruct/webvid_instruct.yaml",
    }

@registry.register_builder("webvid_instruct_zh")
class WebvidInstruct_zh_Builder(Instruct_Builder):
    train_dataset_cls = Video_Instruct_Dataset
    # print(train_dataset_cls)

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/instruct/webvid_instruct.yaml",
    }

@registry.register_builder("llava_instruct")
class LlavaInstruct_Builder(Instruct_Builder):
    train_dataset_cls = Instruct_Dataset
    # print(train_dataset_cls)

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/instruct/llava_instruct.yaml",
    }

@registry.register_builder("egoplan_instruct")
class EgoplanInstruct_Builder(Instruct_Builder):
    train_dataset_cls = Egoplan_Video_Instruct_Dataset
    # print(train_dataset_cls)

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/instruct/egoplan_instruct.yaml",
    }

    def build(self):
        self.build_processors()
        datasets = dict()
        split = "train"

        build_info = self.config.build_info
        dataset_cls = self.train_dataset_cls
        if self.config.num_video_query_token:
            num_video_query_token = self.config.num_video_query_token
        else:
            num_video_query_token = 32

        if self.config.tokenizer_name:
            tokenizer_name = self.config.tokenizer_name
        else:
            tokenizer_name = '/mnt/workspace/ckpt/vicuna-13b/'

        # print(self.config)

        datasets[split] = dataset_cls(
            vis_processor=self.vis_processors[split],
            text_processor=self.text_processors[split],
            vis_root=build_info.rgb_frame_dir,
            ann_root=build_info.anno_dir,
            num_video_query_token=num_video_query_token,
            tokenizer_name=tokenizer_name,
            data_type=self.config.data_type,
            model_type=self.config.model_type,
            n_actions=self.config.get("n_actions", 4),
            answer_type = self.config.get("answer_type", 'egoplan_qa'),
        )

        return datasets

@registry.register_builder("egoplan_contrastive")
class EgoplanContrastive_Builder(EgoplanInstruct_Builder):
    train_dataset_cls = Egoplan_Video_Contrastive_Dataset
    # print(train_dataset_cls)

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/instruct/egoplan_contrastive.yaml",
    }

@registry.register_builder("egoplan_contrastive_kh")
class EgoplanContrastive_Builder_KH(EgoplanInstruct_Builder):
    train_dataset_cls = Egoplan_Video_Contrastive_Dataset_KH
    # print(train_dataset_cls)

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/instruct/egoplan_contrastive_kh.yaml",
    }
    
@registry.register_builder("egoplan_contrastive_bound")
class EgoplanContrastive_Builder_Bound(EgoplanInstruct_Builder):
    train_dataset_cls = Egoplan_Video_Contrastive_Dataset_Bound
    # print(train_dataset_cls)

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/instruct/egoplan_contrastive_bound.yaml",
    }

@registry.register_builder("egoplan_contrastive_mdpo_video")
class EgoplanContrastive_Builder_MDPO_Video(EgoplanInstruct_Builder):
    train_dataset_cls = Egoplan_Video_Contrastive_Dataset_MDPO_Video
    # print(train_dataset_cls)

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/instruct/egoplan_contrastive_mdpo_video.yaml",
    }

@registry.register_builder("egoplan_action_recognition")
class EgoplanActionRecognition_Builder(Instruct_Builder):
    train_dataset_cls = Egoplan_Video_Instruct_Dataset
    # print(train_dataset_cls)

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/instruct/egoplan_action_recognition.yaml",
    }

    def build(self):
        self.build_processors()
        datasets = dict()
        split = "train"

        build_info = self.config.build_info
        dataset_cls = self.train_dataset_cls
        if self.config.num_video_query_token:
            num_video_query_token = self.config.num_video_query_token
        else:
            num_video_query_token = 32

        if self.config.tokenizer_name:
            tokenizer_name = self.config.tokenizer_name
        else:
            tokenizer_name = '/mnt/workspace/ckpt/vicuna-13b/'

        datasets[split] = dataset_cls(
            vis_processor=self.vis_processors[split],
            text_processor=self.text_processors[split],
            vis_root=build_info.rgb_frame_dir,
            ann_root=build_info.anno_dir,
            num_video_query_token=num_video_query_token,
            tokenizer_name=tokenizer_name,
            data_type=self.config.data_type,
            model_type=self.config.model_type,
            n_actions=self.config.get("n_actions", 4),
            answer_type = self.config.get("answer_type", 'action_recognition'),
        )

        return datasets


@registry.register_builder("egoplan_separate_action_recognition")
class EgoplanSeparateActionRecognition_Builder(Instruct_Builder):
    train_dataset_cls = Egoplan_Separate_Video_Instruct_Dataset
    # print(train_dataset_cls)

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/instruct/egoplan_separate_action_recognition.yaml",
    }

    def build(self):
        self.build_processors()
        datasets = dict()
        split = "train"

        build_info = self.config.build_info
        dataset_cls = self.train_dataset_cls
        if self.config.num_video_query_token:
            num_video_query_token = self.config.num_video_query_token
        else:
            num_video_query_token = 32

        if self.config.tokenizer_name:
            tokenizer_name = self.config.tokenizer_name
        else:
            tokenizer_name = '/mnt/workspace/ckpt/vicuna-13b/'

        datasets[split] = dataset_cls(
            vis_processor=self.vis_processors[split],
            text_processor=self.text_processors[split],
            vis_root=build_info.rgb_frame_dir,
            ann_root=build_info.anno_dir,
            num_video_query_token=num_video_query_token,
            tokenizer_name=tokenizer_name,
            data_type=self.config.data_type,
            model_type=self.config.model_type,
            n_actions=self.config.get("n_actions", 4),
            answer_type = self.config.get("answer_type", 'action_recognition'),
        )

        return datasets


@registry.register_builder("egoplan_action_recognition_v3")
class EgoplanActionRecognition_Builder_v3(Instruct_Builder):
    train_dataset_cls = Egoplan_Video_Instruct_Dataset_v3
    # print(train_dataset_cls)

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/instruct/egoplan_action_recognition_v3.yaml",
    }

    def build(self):
        self.build_processors()
        datasets = dict()
        split = "train"

        build_info = self.config.build_info
        dataset_cls = self.train_dataset_cls
        if self.config.num_video_query_token:
            num_video_query_token = self.config.num_video_query_token
        else:
            num_video_query_token = 32

        if self.config.tokenizer_name:
            tokenizer_name = self.config.tokenizer_name
        else:
            tokenizer_name = '/mnt/workspace/ckpt/vicuna-13b/'

        datasets[split] = dataset_cls(
            vis_processor=self.vis_processors[split],
            text_processor=self.text_processors[split],
            vis_root=build_info.rgb_frame_dir,
            ann_root=build_info.anno_dir,
            num_video_query_token=num_video_query_token,
            tokenizer_name=tokenizer_name,
            data_type=self.config.data_type,
            model_type=self.config.model_type,
            n_actions=self.config.get("n_actions", 4),
            answer_type = self.config.get("answer_type", 'action_recognition'),
        )

        return datasets

@registry.register_builder("egoplan_action_recognition_dpo")
class EgoplanActionRecognitionDPO_Builder(Instruct_Builder):
    train_dataset_cls = Egoplan_Video_Instruct_DPO_Dataset
    # print(train_dataset_cls)

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/instruct/egoplan_action_recognition_dpo.yaml",
    }

    def build(self):
        self.build_processors()
        datasets = dict()
        split = "train"

        build_info = self.config.build_info
        dataset_cls = self.train_dataset_cls
        if self.config.num_video_query_token:
            num_video_query_token = self.config.num_video_query_token
        else:
            num_video_query_token = 32

        if self.config.tokenizer_name:
            tokenizer_name = self.config.tokenizer_name
        else:
            tokenizer_name = '/mnt/workspace/ckpt/vicuna-13b/'

        datasets[split] = dataset_cls(
            vis_processor=self.vis_processors[split],
            text_processor=self.text_processors[split],
            vis_root=build_info.rgb_frame_dir,
            ann_root=build_info.anno_dir,
            num_video_query_token=num_video_query_token,
            tokenizer_name=tokenizer_name,
            data_type=self.config.data_type,
            model_type=self.config.model_type,
            n_actions=self.config.get("n_actions", 4),
            answer_type = self.config.get("answer_type", 'action_recognition'),
        )

        return datasets


@registry.register_builder("egoplan_semantic_matching")
class EgoplanSemanticMatching_Builder(Instruct_Builder):
    train_dataset_cls = Egoplan_Video_Instruct_Semantic_Matching_Dataset
    # print(train_dataset_cls)

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/instruct/egoplan_semantic_matching.yaml",
    }

    def build(self):
        self.build_processors()
        datasets = dict()
        split = "train"

        build_info = self.config.build_info
        dataset_cls = self.train_dataset_cls
        if self.config.num_video_query_token:
            num_video_query_token = self.config.num_video_query_token
        else:
            num_video_query_token = 32

        if self.config.tokenizer_name:
            tokenizer_name = self.config.tokenizer_name
        else:
            tokenizer_name = '/mnt/workspace/ckpt/vicuna-13b/'

        datasets[split] = dataset_cls(
            vis_processor=self.vis_processors[split],
            text_processor=self.text_processors[split],
            vis_root=build_info.rgb_frame_dir,
            ann_root=build_info.anno_dir,
            num_video_query_token=num_video_query_token,
            tokenizer_name=tokenizer_name,
            data_type=self.config.data_type,
            model_type=self.config.model_type,
            n_actions=self.config.get("n_actions", 4),
            answer_type = self.config.get("answer_type", 'action_recognition'),
        )

        return datasets
