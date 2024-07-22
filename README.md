# EgoPlan_Challenge_Team_AAILab


## DPO-Finetuned Large Multi-Modal Planner with Retrieval Augmented Generation  <br><sub> </sub>
**Kwanghyeon Lee, Mina Kang, Hyungho Na, Heesun Bae, Byeonghu Na, Doyun Kwon, Seungjae Shin, Yeongmin Kim, Taewoo Kim, Seungmin Yun, and Il-Chul Moon**   
<sup> * Equal contribution </sup> <br>

| [paper] |  <br>
We will upload our paper to Arxiv soon.

## Overview
![Teaser image](./figure/overview_v4_1.png)

Our framework consists of two distinctive components: action sequence retrieval and direct preference optimization (DPO).

## Datasets
- Our implementation is based on [EgoPlan](https://github.com/ChenYi99/EgoPlan).
- We used **Dataset** and **Model Weights** refer to [EgoPlan](https://github.com/ChenYi99/EgoPlan).
  - Dataset:
    - EgoPlan Benchmark Dataset (Train / Valid / Test):
      - Train: [EgoPlan_IT.json](https://drive.google.com/file/d/139UXIgOXbK55tNlK03TBrdSWXdupfrL5/view)
      - Valid: [EgoPlan_validation.json](https://drive.google.com/file/d/1Hy-mWrtuDjuq29iCQxCQzk0htTJs8SHg/view)
      - Test: [EgoPlan_test.json](https://drive.google.com/file/d/1G3cH58qlXI11iRFc8R1oFXpHhEiOh4Bd/view)
    - WebVideo ~~
    - Video LLaVA
    - MSCOCO
  - Model Weights:
    - Original Video-LLaMA:
    - Finetuned Video-LLaMA (Not used actually): 
- Here, our own RAG dataset.
- Place **data** at the directory specified below.
  ```
  ${project_page}/DG/
  ├── checkpoints
  │   ├── pretrained_score/edm-cifar10-32x32-uncond-vp.pkl
  │   ├── pretrained_score/edm-cifar10-32x32-cond-vp.pkl
  ├── ...
  ```

- Place **model weights** at the directory specified below.

  ```
  ${project_page}/DG/
  ├── checkpoints
  │   ├── pretrained_score/edm-cifar10-32x32-uncond-vp.pkl
  │   ├── pretrained_score/edm-cifar10-32x32-cond-vp.pkl
  ├── ...
  ```

## Running of DPO-Finetuned Large Multi-Modal Planner with Rag

### 1) Finetuning 
    
  - run
  ```bash
  bash scripts/format_parser.sh fine dpo_to_dpo_add_narr_rag_v4_base_rag 0,1,2,3,4,5,6,7 8 26501
  ```

### 2) Evaluation

  - run
  ```bash
  bash scripts/test.sh fine dpo_to_dpo_add_narr_rag_v4_base_rag 0 format_eval --epoch,{epoch num},--time,{folder_name}

  ```
  
   

### 3) Test
 
  - run
  ```bash
  bash scripts/test.sh fine dpo_to_dpo_add_narr_rag_v4_base_rag 0 format_test --epoch,{epoch num},--time,{folder_name}
  ```



## Experimental Results
### 1) Test accuracies with regard to our method components
|                  | DPO loss | RAG  | Ensemble | Test Acc.(%) |
|------------------|:--------:|:----:|:--------:|:------------:|
| Base →           |         |      |          | 41.35        |
|                  | ✔        |     |          | 53.98        |
|                  | ✔        | ✔    |         | 58.21        |
| Ours →           | ✔        | ✔    | ✔        | **60.98**    |

### 2) Validation accuracies for various combinations of our method components
|                 | Base      | Loss type               | RAG  | Valid Acc.(%) / Approx. Training Time  |
|-----------------|-----------|-------------------------|:----:|---------------------------------------:|
| **Baseline**    | Original  | -             | -    | 30.44† / Given Pre-trained Model       |
|                 |           |Contrastive                 | ✗    | 44.42† / Given Pre-trained Model       |
| **Ours**        | Original  | Contrastive             | ✓    | 52.44 / 0.5 days                       |
|                 |           | DPO                     | ✗    | 60.24 / 0.5 days                       |
|                 |           | DPO                     | ✓    | 52.44 / 0.5 days                       |
|                 | DPO-Finetuned | Contrastive (Iterative) | ✓ | 53.09 / 0.5 days                       |
|                 |           | DPO (Iterative)         | ✗    | 61.11 / 0.5 days                       |
|                 |           | DPO (Iterative)         | ✓    | **67.17 / 0.5 days**                   |

Note that Base indicates the initial checkpoint from which the model is fine_tuned.


## Reference
If you find the code useful for your research, please consider citing
```bib
@article{
}
```
This work is heavily built upon the code from
 - EgoPlan-Bench [EgoPlan](https://github.com/ChenYi99/EgoPlan)
 - **
 - **


 ## Acknowledgement
This repo benefits from [Epic-Kitchens](https://epic-kitchens.github.io/2023), [Ego4D](https://ego4d-data.org/), 
[EgoPlan](https://github.com/ChenYi99/EgoPlan), 
[Video-LLaMA](https://github.com/DAMO-NLP-SG/Video-LLaMA), 
[LLaMA](https://github.com/facebookresearch/llama),
[MiniGPT-4](https://github.com/Vision-CAIR/MiniGPT-4), 
[LLaVA](https://github.com/haotian-liu/LLaVA), 
[VideoChat](https://github.com/OpenGVLab/Ask-Anything). Thanks for their wonderful works!
