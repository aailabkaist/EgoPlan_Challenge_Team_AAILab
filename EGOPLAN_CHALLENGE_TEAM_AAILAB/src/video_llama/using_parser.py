import argparse
from collections import OrderedDict
import sys
import ruamel.yaml
from ruamel.yaml import YAML
from ruamel.yaml.scalarstring import DoubleQuotedScalarString
from typing import List, Dict, Tuple
import re
import os
from datetime import datetime

# YAML 파일 로드
def load_yaml(file_path,keys_to_quote):
  yaml=YAML()
  with open(file_path, 'r') as file:
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

#자동으로 keys_to_quote를 만들어주는 함수
def collect_keys_to_quote(file_path):
    keys_to_quote = []
    #load를 통해 열면 쌍따옴표가 바로 사라지므로 open으로 불러오기
    with open(file_path, 'r') as file:
        lines = file.readlines()    
    # 현재 경로를 추적하는 스택
    current_path = []
    for line in lines:
        # 현재 줄의 들여쓰기 수준 계산
        indent_level = len(line) - len(line.lstrip(' '))
        # 주석이 :나 "보다 먼저 나오는지 확인
        comment_index = line.find('#')
        colon_index = line.find(':')
        quote_index = line.find('"')
        if comment_index != -1 and (comment_index < colon_index or comment_index < quote_index):
            continue  # 주석이 :나 "보다 먼저 나오면 무시
        # 현재 줄이 키-값 쌍인지 확인
        if ':' in line:
            key_part, value_part = line.split(':', 1)
            # 들여쓰기 수준에 따라 현재 경로 갱신
            while current_path and current_path[-1][1] >= indent_level:
                current_path.pop()
            key = key_part.strip()
            current_path.append((key, indent_level))
            full_key = '/'.join([k for k, _ in current_path])
            # 값 부분에서 "가 # 전에 나오는지 확인
            value_part = value_part.strip()
            if '"' in value_part:
                keys_to_quote.append(full_key)
    return keys_to_quote

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

def main():
    config_path = sys.argv[2]
    output_path = config_path.replace(".yaml", "_mod.yaml")
    keys_to_quote = collect_keys_to_quote(config_path)
    config = load_yaml(config_path, keys_to_quote)
    dataset_dir=os.getenv('PYTHONPATH')+"/RAG_test/data/final_test_rag_bert_embedding_ver.json"
    # 명령줄 인자를 가져와서 파싱
    args_list = sys.argv[1:]
    if args_list[0]=='-b':
        args_list1=args_list[2].split(',')
        args_list1.extend(["--datasets/egoplan_contrastive/build_info/anno_dir",dataset_dir,"--datasets/egoplan_action_recognition/build_info/anno_dir",dataset_dir])
    args = parse_args(config, args_list1)

    for key in keys_to_quote:
        if hasattr(args, key):
            setattr(args, key, DoubleQuotedScalarString(getattr(args, key)))
            # 평면 딕셔너리를 중첩된 OrderedDict로 변환
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

if __name__ == "__main__":
    main()
