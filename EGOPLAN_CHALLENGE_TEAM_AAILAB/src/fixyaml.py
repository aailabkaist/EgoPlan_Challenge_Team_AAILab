from .egoplan_video_llama_interface import build as build_egoplan_video_llama
from .video_llama_interface import build as build_video_llama
from .egoplan_video_llama_interface_rag import build as build_egoplan_video_llama_rag
import argparse
from collections import OrderedDict
import sys
import ruamel.yaml
from ruamel.yaml import YAML
from typing import List, Dict, Tuple
import re
import os
from datetime import datetime

# YAML 파일 로드
def load_yaml(file_path):
  yaml=YAML()
  with open(os.getcwd()+'/'+file_path, 'r') as file:
    config=yaml.load(file)
  return config

# 재귀적으로 파서에 인자를 추가하는 함수
def add_arguments(parser, config, prefix=''):
    if isinstance(config, dict):
        for key, value in config.items():
            if isinstance(value, dict):
                add_arguments(parser, value, prefix + key + '/')
            else:
                arg_name = '--' + prefix + key
                if isinstance(value, bool):
                    parser.add_argument(arg_name, type=lambda x: (str(x).lower() == 'true'), default=value, help=f'Default: {value}')
                elif isinstance(value, list):
                    parser.add_argument(arg_name, type=lambda x: [i.strip() for i in x.split(',')], default=value, help=f'Default: {value}')
                else:
                    parser.add_argument(arg_name, type=type(value), default=value, help=f'Default: {value}')

# 파서에서 인자를 파싱하고, 동적으로 새로운 인자를 추가하는 함수
def parse_args(config, args_list):
    parser = argparse.ArgumentParser(description="Training Configuration")
    add_arguments(parser, config)
    # 임시로 파싱하여 기존에 정의되지 않은 인자를 캡처
    args, unknown = parser.parse_known_args(args_list)
    # 새로운 인자를 동적으로 추가
    i = 0
    while i < len(unknown):
        arg = unknown[i]
        if arg.startswith("--"):
            key = arg[2:].replace('/', '_')
            if i + 1 < len(unknown) and not unknown[i + 1].startswith('--'):
                value = unknown[i + 1]
                parts = key.split('_')
                d = config
                for part in parts[:-1]:
                    if part not in d:
                        d[part] = {}
                    d = d[part]
                d[parts[-1]] = value  # 임시로 value 설정

                # value의 타입을 올바르게 설정
                if value.lower() in ['true', 'false']:
                    value = value.lower() == 'true'
                elif value.isdigit():
                    value = int(value)
                else:
                    try:
                        value = float(value)
                    except ValueError:
                        pass
                # config에 최종 타입 변환 후 설정
                d[parts[-1]] = value
                # 새로운 인자를 parser에 추가
                parser.add_argument(arg, type=type(value), default=value, help=f'Default: {value}')
                i += 1
        i += 1

    # 업데이트된 설정을 기반으로 다시 파서 생성
    args = parser.parse_args(args_list)
    return args

# 중첩된 OrderedDict를 생성하는 함수
def nested_ordered_dict():
    return OrderedDict()

# 평면 구조의 딕셔너리를 중첩된 OrderedDict로 변환하는 함수
def unflatten_dict(flat_dict):
    result = nested_ordered_dict()
    for key, value in flat_dict.items():
        keys = key.split('/')
        d = result
        for k in keys[:-1]:
            if k not in d:
                d[k] = nested_ordered_dict()
            d = d[k]
        d[keys[-1]] = value
    return result

def to_ordered_dict(d):
    if isinstance(d, OrderedDict):
        d = OrderedDict((k, to_ordered_dict(v)) for k, v in d.items())
    return d

def create_directory_from_args(argslist,args, base_path):
    changed_args = {k[2:]: getattr(args, k[2:]) for k in argslist if (k.startswith("--") and getattr(args,k[2:]) is not None)}
    sorted_args = OrderedDict(sorted(changed_args.items()))
    dir_path = base_path
    for key, value in sorted_args.items():
        dir_name = f"{key}:{value}"
        dir_path = os.path.join(dir_path, dir_name)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
    return dir_path

def extract_numbers(strings_list):
    numbers = []
    for item in strings_list:
        try:
            # 문자열을 숫자로 변환 시도
            number = int(item)
            numbers.append(number)
        except ValueError:
            # 변환 실패 시 무시
            pass
    return numbers

def find_largest_number(strings_list):
    numbers = extract_numbers(strings_list)
    if not numbers:
        return None
    return max(numbers)

#이건 만약에 lora랑 checkpoint를 다른 폴더에서 들고 와도 되는경우
def find_largest_number_with_option(items, base_path,option):
    max_number = -1
    for item in items:
        item_path = os.path.join(base_path, item)
        if os.path.isdir(item_path):
            files = os.listdir(item_path)
            if any(option in file for file in files):
                try:
                    number = int(item)
                    if number > max_number:
                        max_number = number
                except ValueError:
                    # Skip if the folder name is not a number
                    continue
    return max_number

def find_largest_number(items, base_path):
    max_number = -1
    for item in items:
        item_path = os.path.join(base_path, item)
        if os.path.isdir(item_path):
            files = os.listdir(item_path)
            if (any("checkpoint" in file for file in files) and any("lora_weights" in file for file in files)):
                try:
                    number = int(item)
                    if number > max_number:
                        max_number = number
                except ValueError:
                    # Skip if the folder name is not a number
                    continue
    return max_number

def setyaml(target,model_name,model_ver,epoch,tryfix,fix):
    config_path = target
    output_path = target.replace(".yaml", "_mod.yaml")
    config = load_yaml(config_path)
    print(f"success {target},{model_name},{model_ver},{epoch},{tryfix},{fix}")
    #ckpt를 자동으로 path들 바꿔주는 코드 작성
    #바꿔야 할 부분 : model.ckpt, model.llm_lora_config.lora_ckpt 2개
    fixlist=['model/ckpt','model/llm_lora_config/lora_ckpt']
    #ckpt를 자동으로 path들 바꿔주는 코드 작성
    for fixpoint in fixlist:
        #원래 ckpt추출
        pointlist=fixpoint.split('/')
        if len(pointlist)==2:
            origckpt=config[pointlist[0]][pointlist[1]]
        elif len(pointlist)==3:
            origckpt=config[pointlist[0]][pointlist[1]][pointlist[2]]
        #해당 ckpt에서 /로 split
        ckptlistori=origckpt.split('/')
        #해당 ckpt에서 가장 큰 숫자(최근에 train을 돌린 model을 default로 설정)
        subpath='/'.join(ckptlistori[:-2])
        all_items=os.listdir(subpath)
        set_time_ckpt=str(find_largest_number(all_items,subpath))
        set_time_lora=str(find_largest_number(all_items,subpath))
        set_epoch=epoch
        if tryfix:
            optionlist=fix.split(',')
            #train시간에 해당하는 부분을 폴더 내에서 가장 최근 시간의 폴더가 아닌 지정을 원하는 경우 조건으로 받을 수 있도록 예:-time,202407162349
            for i in range(len(optionlist)):
                if optionlist[i]=='--time':
                    set_time_ckpt=optionlist[i+1]
                    set_time_lora=optionlist[i+1]
                elif optionlist[i]=='--timeckpt':
                    set_time_ckpt=optionlist[i+1]
                elif optionlist[i]=='--timelora':
                    set_time_lora=optionlist[i+1]
            #epoch을 받지 않는 model들의 경우에는 epoch을 조건으로 받을 수 있도록 예: -epoch,3
                if optionlist[i]=='--epoch':
                    set_epoch=optionlist[i+1]
        #상황에 맡게 바꾸기
        if len(pointlist)==2:
            ckptlistori[-2]=set_time_ckpt
            ckptlistori[-1]=f"checkpoint_{set_epoch}.pth"
            modfckpt='/'.join(ckptlistori)
        elif len(pointlist)==3:
            ckptlistori[-2]=set_time_lora
            ckptlistori[-1]=f"lora_weights_{set_epoch}"
            modflora='/'.join(ckptlistori)
    #args_list에 추가
    args_list=['--model/ckpt',modfckpt,'--model/llm_lora_config/lora_ckpt',modflora]
    filter=['--time','--epoch','--timeckpt','--timelora']
    filtered_list=[optionlist[i] for i in range(len(optionlist)) if (optionlist[i] not in filter) and (i==0 or optionlist[i-1] not in filter) ]
    #수정하고 싶은 영역이 있다면 해당 yaml value를 수정하는 코드 작성
    if filtered_list !=[]:
        args_list.extend(filtered_list)
    args = parse_args(config, args_list)
    nested = unflatten_dict(vars(args))
    # 중첩된 OrderedDict를 일반 OrderedDict로 변환
    ordered_nested_dict = to_ordered_dict(nested)
    yaml = ruamel.yaml.YAML()
    # boolean_representation 속성을 설정하여 boolean 값의 형식을 정의
    yaml.boolean_representation = ['False', 'True']
    yaml.preserve_quotes = True
    # !!omap 태그를 사용하지 않도록 설정
    yaml.default_flow_style = False
    yaml.Representer.add_representer(OrderedDict, ruamel.yaml.representer.SafeRepresenter.represent_dict)
    # YAML 파일로 저장
    with open(output_path, 'w') as file:
        yaml.dump(ordered_nested_dict, file)
    return output_path

def gatheredmodel(model_name, model_ver, epoch, tryfix, fix):
    ##########
    if model_name == 'video_llama':
        defaultyaml = "src/video_llama/eval_configs/video_llama_eval_only_vl.yaml"
        return build_video_llama(config_path = setyaml(defaultyaml, model_name, model_ver, epoch, tryfix, fix))
    elif model_name == 'egoplan_video_llama':
        defaultyaml = "src/video_llama/eval_configs/egoplan_video_llama_eval.yaml"
        if model_ver is not None:
            defaultyaml = f'src/video_llama/eval_configs/egoplan_video_llama_eval_{model_ver}.yaml'
        return build_egoplan_video_llama(config_path = setyaml(defaultyaml, model_name, model_ver, epoch, tryfix, fix))
    elif model_name == 'egoplan_video_llama_rag':
        defaultyaml = "src/video_llama/eval_configs/egoplan_video_llama_eval.yaml"
        if model_ver is not None:
            defaultyaml = f'src/video_llama/eval_configs/egoplan_video_llama_eval_{model_ver}.yaml'
        return build_egoplan_video_llama_rag(config_path = setyaml(defaultyaml, model_name, model_ver, epoch, tryfix, fix))