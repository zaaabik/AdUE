#!/bin/bash

#CUDA_VISIBLE_DEVICES=0 python src/train.py -m seed=1,2,3,4 experiment=electra/lora/cola.yaml,llama7b/lora/cola.yaml,qwen2.5/lora/cola.yaml &
#CUDA_VISIBLE_DEVICES=1 python src/train.py -m seed=1,2,3,4 experiment=electra/lora/newsgroups.yaml,llama7b/lora/newsgroups.yaml,qwen2.5/lora/newsgroups.yaml &
#CUDA_VISIBLE_DEVICES=2 python src/train.py -m seed=1,2,3,4 experiment=electra/lora/sst2.yaml,llama7b/lora/sst2.yaml,qwen2.5/lora/sst2.yaml &
#CUDA_VISIBLE_DEVICES=3 python src/train.py -m seed=1,2,3,4 experiment=electra/lora/sst5.yaml,llama7b/lora/sst5.yaml,qwen2.5/lora/sst5.yaml &
#CUDA_VISIBLE_DEVICES=4 python src/train.py -m seed=1,2,3,4 experiment=electra/lora/toxigen.yaml,llama7b/lora/toxigen.yaml,qwen2.5/lora/toxigen.yaml &


CUDA_VISIBLE_DEVICES=0 python src/train.py -m experiment=roberta/lora/sst5.yaml,roberta/lora/cola.yaml,roberta/lora/newsgroups.yaml,roberta/lora/sst2.yaml,roberta/lora/toxigen.yaml seed=0,1,2,3,4 &
CUDA_VISIBLE_DEVICES=0 python src/train.py -m experiment=electra/lora/sst5.yaml,electra/lora/cola.yaml,electra/lora/newsgroups.yaml,electra/lora/sst2.yaml,electra/lora/toxigen.yaml seed=0,1,2,3,4 &


CUDA_VISIBLE_DEVICES=0 python src/train.py experiment=electra/full/sst5.yaml seed=0 ++trainer.max_epochs=3