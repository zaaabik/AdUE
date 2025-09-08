export PROJECT_ROOT=/root/AdUE

CUDA_VISIBLE_DEVICES=0 python baseline_eval_seeds/probes_train.py -m \
  experiment=roberta train_on_dataset=train \
  data.name=CoLa,SST2,SST5,NewsGroups,ToxigenDataset \
  seed=0,1,2,3,4

CUDA_VISIBLE_DEVICES=0 python baseline_eval_seeds/probes_train.py -m \
  experiment=electra train_on_dataset=train \
  data.name=CoLa,SST2,SST5,NewsGroups,ToxigenDataset \
  seed=0,1,2,3,4

CUDA_VISIBLE_DEVICES=0 python baseline_eval_seeds/probes_train.py -m \
  experiment=qwen train_on_dataset=train \
  data.name=CoLa,SST2,SST5,NewsGroups,ToxigenDataset \
  seed=0,1,2,3,4

CUDA_VISIBLE_DEVICES=0 python baseline_eval_seeds/probes_train.py -m \
  experiment=llama train_on_dataset=train \
  data.name=CoLa,SST2,SST5,NewsGroups,ToxigenDataset \
  seed=0,1,2,3,4
