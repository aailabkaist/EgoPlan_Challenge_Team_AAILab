from .egoplan_video_llama_interface import build as build_egoplan_video_llama
from .video_llama_interface import build as build_video_llama
from .egoplan_video_llama_interface_rag import build as build_egoplan_video_llama_rag

####
import os
####

def build(model_name, model_config=None, epoch=None):
    if model_name == 'video_llama':
        return build_video_llama()
    elif model_name == 'egoplan_video_llama':
        return build_egoplan_video_llama(model_config=model_config, epoch=epoch)
    elif model_name == 'egoplan_video_llama_rag':
        return build_egoplan_video_llama_rag(model_config=model_config, epoch=epoch)

    print(f"model {model_name} not exist")
    exit(0)
