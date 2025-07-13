#!/bin/bash
CUDA_VISIBLE_DEVICES=0 python src/train.py -m experiment=roberta/full/sst5.yaml,roberta/full/cola.yaml,roberta/full/newsgroups.yaml,roberta/full/sst2.yaml,roberta/full/toxigen.yaml seed=0,1,2,3,4 ++trainer.max_epochs=10
CUDA_VISIBLE_DEVICES=0 python src/train.py -m experiment=electra/full/sst5.yaml,electra/full/cola.yaml,electra/full/newsgroups.yaml,electra/full/sst2.yaml,electra/full/toxigen.yaml seed=0,1,2,3,4 ++trainer.max_epochs=10

CUDA_VISIBLE_DEVICES=0 python src/train.py -m experiment=roberta/full/sst5.yaml,roberta/full/cola.yaml,roberta/full/newsgroups.yaml,roberta/full/sst2.yaml,roberta/full/toxigen.yaml seed=0,1,2,3,4 ++trainer.max_epochs=10 model.optimizer.lr=1e-6
CUDA_VISIBLE_DEVICES=0 python src/train.py -m experiment=electra/full/sst5.yaml,electra/full/cola.yaml,electra/full/newsgroups.yaml,electra/full/sst2.yaml,electra/full/toxigen.yaml seed=0,1,2,3,4 ++trainer.max_epochs=10 model.optimizer.lr=1e-6

CUDA_VISIBLE_DEVICES=0 python src/train.py -m experiment=roberta/full/sst5.yaml,roberta/full/cola.yaml,roberta/full/newsgroups.yaml,roberta/full/sst2.yaml,roberta/full/toxigen.yaml seed=0,1,2,3,4 ++trainer.max_epochs=10 model.optimizer.lr=3e-6
CUDA_VISIBLE_DEVICES=0 python src/train.py -m experiment=electra/full/sst5.yaml,electra/full/cola.yaml,electra/full/newsgroups.yaml,electra/full/sst2.yaml,electra/full/toxigen.yaml seed=0,1,2,3,4 ++trainer.max_epochs=10 model.optimizer.lr=3e-6