from .egoplan_video_llama_interface import build as build_egoplan_video_llama
from .video_llama_interface import build as build_video_llama
from .egoplan_video_llama_interface_rag import build as build_egoplan_video_llama_rag

####
import os
from .fixyaml import gatheredmodel
####

def build(model_name, model_ver=None, epoch=None):
    ####
    frombash=os.getenv('FROMBASH',False)
    if frombash:
        fix=os.getenv('FIX',"")
        tryfix=False
        if fix !="":
            tryfix=True
        return gatheredmodel(model_name,model_ver,epoch,tryfix,fix)
    ####
    if model_name == 'video_llama':
        return build_video_llama()
    elif model_name == 'egoplan_video_llama':
        return build_egoplan_video_llama(model_ver=model_ver)
    elif model_name == 'egoplan_video_llama_rag':
        return build_egoplan_video_llama_rag(model_ver=model_ver, epoch=epoch)

    print(f"model {model_name} not exist")
    exit(0)
