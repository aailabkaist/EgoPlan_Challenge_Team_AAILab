# EgoPlan_Challenge_Team_AAILab


## DPO-Finetuned Large Multi-Modal Planner with Retrieval Augmented Generation  <br><sub> </sub>
**Kwanghyeon Lee, Mina Kang, Hyungho Na, Heesun Bae, Byeonghu Na, Doyun Kwon, Seungjae Shin, Yeongmin Kim, Taewoo Kim, Seungmin Yun, and Il-Chul Moon**   

| [paper] |  <br>
We will upload our paper to Arxiv soon.

## Overview
![Teaser image](./figure/overview_v4_1.png)

Our method consists of two components: Direct Preference Optimization (DPO) and Retrieval-Augmented Generation (RAG). We 

## Dataset and Model Checkpoint
- Our implementation is based on [EgoPlan-Bench](https://github.com/ChenYi99/EgoPlan).
- We used **Instruction Dataset & Corresponding Video/Image Dataset** and **Model Checkpoint** refer to [EgoPlan-Bench](https://github.com/ChenYi99/EgoPlan).
- We also provide our generated **Action Database** and **Model Checkpoint**.
  - **Instruction Dataset & Corresponding Dataset**
    - **EgoPlan-Bench (Train / Valid / Test) & EpicKitchens / Ego4D Dataset**
      - Instruction Dataset:
        - Train (50K): [EgoPlan_IT.json](https://drive.google.com/file/d/139UXIgOXbK55tNlK03TBrdSWXdupfrL5/view)
        - Valid (3K): [EgoPlan_validation.json](https://drive.google.com/file/d/1Hy-mWrtuDjuq29iCQxCQzk0htTJs8SHg/view)
        - Test (2K): [EgoPlan_test.json](https://drive.google.com/file/d/1G3cH58qlXI11iRFc8R1oFXpHhEiOh4Bd/view)
      - Video Dataset:
        - Epickitchens Dataset: [EPIC-KITCHENS-100](https://github.com/epic-kitchens/epic-kitchens-download-scripts)
        - Ego4D Dataset: [Ego4D](https://ego4d-data.org/#download)
    - **Image-based Instructions from MiniGPT-4 (3K) & cc_sbu_align Dataset** (zip file has instruction .json file and images.)
      - Instruction & Image Dataset:
        - [cc_sbu_align.zip](https://drive.google.com/file/d/1nJXhoEcy3KTExr17I7BXqY5Y9Lx_-n-9/view)
    - **Image-based Instructions from LLaVA (150K) & MS COCO 2014 Training Dataset**
      - Instruction Dataset:
        - LLaVA Instruction Dataset: [llava_instruct_150k.json](https://huggingface.co/datasets/liuhaotian/LLaVA-Instruct-150K)
      - Image Dataset:
        - MS COCO 2014 Training Image Dataset: [MS COCO 2014 Training Image Dataset](https://cocodataset.org/#download):
    - **Video-based Instructions from VideoChat (11K) & WebVid Dataset**
      - Instruction Dataset:
        - Videochat Instruction Dataset: [videochat_instruct_11k.json](https://drive.google.com/file/d/1C-7xmf42QUEi4ApXTcxBHr5nLvTWXyUi/view)
        - **Important!** Since we don't get full WebVid dataset, we use revised instruction dataset file for own situation. You can download [videochat_instruct_11k_revised.json](https://drive.google.com/file/d/1rjeCoMYELJ4wGkO9HG243IhlsxfVPfc1/view?usp=drive_link)
      - Video Dataset:
        - WebVid Dataset (for VideoChat Instuction): Since [WebVid dataset](https://github.com/m-bain/webvid) is no longer available, we download the video dataset by two steps.
          1. Download [WebVid-10M dataset information csv file](https://huggingface.co/datasets/TempoFunk/webvid-10M/tree/main).
          2. Download the video file and save it into your specific path by [our provided python code](https://drive.google.com/file/d/1i7iBfbC_RD2CL_Chq9S5uh8SCWsvSUlY/view?usp=drive_link). (The videos are not fully download because some of videos are not allowed to download. So we use [videochat_instruct_11k_revised.json](https://drive.google.com/file/d/1rjeCoMYELJ4wGkO9HG243IhlsxfVPfc1/view?usp=drive_link) instead.)
  - **Model Checkpoint**:
    - Original (Vanilla) Video-LLaMA: [Original Video-LLaMA](https://huggingface.co/DAMO-NLP-SG/Video-LLaMA-2-7B-Finetuned/tree/main)
    - Provided Finetuned Video-LLaMA with EgoPlan_IT dataset from [EgoPlan-Bench](https://github.com/ChenYi99/EgoPlan): [Finetuned Video-LLaMA](https://huggingface.co/ChenYi99/EgoPlan-Video-LLaMA-2-7B/tree/main) (with lora weights)
    - Vision Transformer: [eva_vit_g.pth](https://huggingface.co/lainxx/eva_vit_g/blob/main/eva_vit_g.pth) (You should use Git LFS to download it.)
    - Q-Former: [blip2_pretrained_flant5xxl.pth](https://huggingface.co/spaces/Vision-CAIR/minigpt4/blob/main/blip2_pretrained_flant5xxl.pth) (You should use Git LFS to download it.)
    - BERT: [bert-base-uncased](https://huggingface.co/google-bert/bert-base-uncased/tree/main)
  - **Our RAG Dataset**
    - You can download RAG training dataset from [here](https://drive.google.com/file/d/1dx4-IUDDCu2NGtZRyZpn2ppdCbXtnZTE/view?usp=drive_link) and validation dataset from [here](https://drive.google.com/file/d/1-Pzwl295_QeGsKAQq6qP-YlX36Orn7NJ/view?usp=drive_link).
  - **Our Checkpoint**
    - You can download our model ckpt from [here](https://drive.google.com/drive/folders/1zBRv-OIm9SaAis9wmAAf2BBQxCFhp3gj?usp=sharing).

## Egocentric Video Path Setting (EpicKitchens & Ego4D)

Since EpicKitchens and Ego4D datasets are large datasets, you need to download only necessary thing if you have limited resource.
We follow path setting from [EgoPlan Benchmark](https://github.com/ChenYi99/EgoPlan).

Download the RGB frames of [EPIC-KITCHENS-100](https://github.com/epic-kitchens/epic-kitchens-download-scripts) and videos of [Ego4D](https://ego4d-data.org/#download). The folder structure of two datasets are shown below:
- **EpicKitchens Dataset**:
  ```
  EPIC-KITCHENS
  └── P01
      └── P01_01
          ├── frame_0000000001.jpg
          └── ...
  ```
- **Ego4D Dataset**:
  ```
  Ego4D
  └── v1
      ├── 000786a7-3f9d-4fe6-bfb3-045b368f7d44.mp4
      └── ...
  ```

## Reproduction
We provide the file we used and setting for reproduction.
- Since we have some trouble with downloading Epickitchens dataset, we also share the [Epickitchens Video ID list file](https://drive.google.com/file/d/1cJUKc_IKL1o9Y6mx795LfmtShPGzFq6H/view?usp=drive_link) we used to check if there any missed video compared with original [EPIC-KITCHENS-100](https://github.com/epic-kitchens/epic-kitchens-download-scripts).
- You can download our model config to reproduce models in table. (DPO Finetuned model checkpoint is [here](https://drive.google.com/drive/folders/1Q159B-NOrcc6-n4z6feyyV3ySd94BbiD?usp=drive_link) with lora weights.)
  - Original Video-LLaMA, RAG X, DPO loss: [link](https://drive.google.com/file/d/1qW4JznH-i4v2bK3f_gxbaIix4DCUMoAf/view?usp=drive_link)
  - DPO Finetuned Video-LLaMA, RAG X, DPO loss: [link](https://drive.google.com/file/d/19fBaeZt4kzjK1V2GJRH8SfyTSw-SnAyL/view?usp=drive_link)
  - DPO Finetuned Video-LLaMA, RAG O, Cont. loss: [link](https://drive.google.com/file/d/1lYOBT-kiRRTG3cwupnr4xBFglT9XaF-X/view?usp=drive_link)
  - DPO Finetuned Video-LLaMA, RAG O, DPO loss: [link](https://drive.google.com/file/d/1oLrTTfQ3v-pNUIhKUta1NC-rkzgSQJ29/view?usp=drive_link)

## Finetuning & Evaluating & Testing Commands

Before finetuning or evaluating, you need to prepare .yaml file to set configuration.

### 1) Finetuning 
    
  - run
  ```bash
  bash scripts/format_finetune.sh {config} {device} {node} {master port}
  ```
  - Ex.
  ```
  bash scripts/format_finetune.sh Original_RAG_X_loss_DPO 0,1,2,3,4,5,6,7 8 26501
  ```

### 2) Evaluation & Test

  - run
  ```bash
  bash scripts/format_eval.sh {config} {device} {RAG} {epoch}
  ```
  ```bash
  bash scripts/format_test.sh {config} {device} {RAG} {epoch}
  ```
  - Ex.
  ```
  bash scripts/format_eval.sh Original_RAG_X_loss_DPO 0 True 9
  bash scripts/format_test.sh Original_RAG_X_loss_DPO 0 True 9
  ```


## Experimental Results
### 1) Test accuracies with regard to our method components
|                  | DPO loss | Test Acc.(%) |
|------------------|:--------:|:------------:|
| Base →           |          | 41.35        |
| Ours →           | ✔        | 53.98        |
- **Test accuracy 53.98% of DPO finetuned model is achived at epoch 9 (10/10).**

### 2) Validation accuracies for various combinations of our method components
|                 | Base      | Loss type               | RAG  | Valid Acc.(%) / Approx. Training Time  |
|:-----------------:|:-----------:|:-------------------------:|:----:|:---------------------------------------:|
| **Baseline**    | Original  | -             | -    | 30.44† / Given Pre-trained Model       |
|                 |           |Contrastive                 | ✗    | 44.42† / Given Pre-trained Model       |
| **Ours**        | Original  | DPO                     | ✗    | 60.24 / 0.5 days                       |
|                 | DPO-Finetuned | Contrastive (Iterative) | ✓ | 46.05 / 0.5 days                       |
|                 |           | DPO (Iterative)         | ✗    | 61.11 / 0.5 days                       |
|                 |           | DPO (Iterative)         | ✓    | 60.24 / 0.5 days                   |

Note that Base indicates the initial checkpoint from which the model is fine_tuned.
- **Valid accuracy 60.24% of DPO finetuned model is achived at epoch 8 (9/10).**

## Reference
If you find the code useful for your research, please consider citing
```bib
@article{
}
```
This work is heavily built upon the code from
 - EgoPlan-Bench [EgoPlan](https://github.com/ChenYi99/EgoPlan)
 
 ## Acknowledgement
This repository benefits from [Epic-Kitchens](https://epic-kitchens.github.io/2023), [Ego4D](https://ego4d-data.org/), 
[EgoPlan](https://github.com/ChenYi99/EgoPlan), 
[Video-LLaMA](https://github.com/DAMO-NLP-SG/Video-LLaMA), 
[LLaMA](https://github.com/facebookresearch/llama),
[MiniGPT-4](https://github.com/Vision-CAIR/MiniGPT-4), 
[LLaVA](https://github.com/haotian-liu/LLaVA), 
[VideoChat](https://github.com/OpenGVLab/Ask-Anything). Thanks for their wonderful works!
