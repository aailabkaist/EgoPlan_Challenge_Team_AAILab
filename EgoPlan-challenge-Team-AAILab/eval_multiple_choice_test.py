import os
import os.path as osp
import sys
import json
import argparse
from PIL import Image
import torch
import pdb
from src import build
from tqdm import tqdm
import torch
import numpy as np
import random
import string

seed = 0
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='egoplan_video_llama')
    parser.add_argument('--epic_kitchens_rgb_frame_dir', type=str)
    parser.add_argument('--ego4d_video_dir', type=str)
    parser.add_argument('--model_config', type=str)
    parser.add_argument('--epoch', type=int, default=0)
    args = parser.parse_args()
    print('model is ', args.model)

    print(f'evaluating.. {args.model}')
    predict_choice = build(args.model, model_config=args.model_config, epoch=args.epoch)
    with open('Your Test file path') as fi:
        samples = json.load(fi)
        correct_num = 0
        question_num = 0
        for i, sample in tqdm(enumerate(samples), desc="Processing questions", total=len(samples)):
            try:
                subset_name = 'Ego4D'
                video_source = 'Ego4D'

                if video_source == "EpicKitchens":
                    video_id = sample["video_id"]
                    participant_id = video_id.split("_")[0]
                    video_rgb_frame_dir = os.path.join(args.epic_kitchens_rgb_frame_dir,
                                                    participant_id, "rgb_frames", video_id)
                    sample["video_rgb_frame_dir"] = video_rgb_frame_dir
                else:
                    video_id = sample["video_id"]
                    video_path = os.path.join(args.ego4d_video_dir, f"{video_id}.mp4")
                    sample["video_path"] = video_path
                predicted_choice, choice2loss = predict_choice(sample=sample, return_loss=True, subset_name=subset_name)
                print("***** question *****")
                print(sample["question"])

                print("***** predicted choice *****")
                print(predicted_choice)

                print("***** predicted choice2loss *****")
                print(choice2loss)
                question_num += 1
            except:
                continue