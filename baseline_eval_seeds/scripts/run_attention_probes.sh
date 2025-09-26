CUDA_VISIBLE_DEVICES=0 python baseline_eval_seeds/probes_train.py -m \
  experiment=qwen train_on_dataset=train probes.layers=[-1,-7,-14,-21] \
  data.name=CoLa,SST2,SST5,NewsGroups,ToxigenDataset \
  seed=0,1,2,3,4

CUDA_VISIBLE_DEVICES=0 python baseline_eval_seeds/probes_train.py -m \
  experiment=roberta train_on_dataset=train probes.layers=[-1,-3,-6,-9] \
  data.name=CoLa,SST2,SST5,NewsGroups,ToxigenDataset \
  seed=0,1,2,3,4

CUDA_VISIBLE_DEVICES=0 python baseline_eval_seeds/probes_train.py -m \
  experiment=electra train_on_dataset=train probes.layers=[-1,-3,-6,-9] \
  data.name=CoLa,SST2,SST5,NewsGroups,ToxigenDataset \
  seed=0,1,2,3,4

CUDA_VISIBLE_DEVICES=0 python baseline_eval_seeds/probes_train.py -m \
  experiment=llama train_on_dataset=train probes.layers=[-1,-8,-16,-24] \
  data.name=CoLa,SST2,SST5,NewsGroups,ToxigenDataset \
  seed=0,1,2,3,4

# base models

CUDA_VISIBLE_DEVICES=0 python baseline_eval_seeds/probes_train.py \
  experiment=qwen_chat_base train_on_dataset=train \
  data.name=TruthfulQA probes.layers=[-1,-7,-14,-21] \
  seed=0 data.max_length=1024 data.llm_batch_size=1

CUDA_VISIBLE_DEVICES=0 python baseline_eval_seeds/probes_train.py \
  experiment=llama_chat_base train_on_dataset=train \
  data.name=TruthfulQA probes.layers=[-1,-8,-16,-24] \
  seed=0 data.max_length=1024 data.llm_batch_size=1

# ARC
CUDA_VISIBLE_DEVICES=0 python baseline_eval_seeds/probes_train.py \
  experiment=qwen_chat_base train_on_dataset=train \
  data.name=ARCC ++data.dataset.n_shot=5 probes.layers=[-1,-7,-14,-21] \
  seed=0 data.max_length=4096 data.llm_batch_size=1

CUDA_VISIBLE_DEVICES=0 python baseline_eval_seeds/probes_train.py \
  experiment=llama_chat_base train_on_dataset=train \
  data.name=ARCC ++data.dataset.n_shot=5 probes.layers=[-1,-8,-16,-24] \
  seed=0 data.max_length=4096 data.llm_batch_size=1

# MMLU
CUDA_VISIBLE_DEVICES=0 python baseline_eval_seeds/probes_train.py \
  experiment=llama_chat_base train_on_dataset=train \
  data.name=MMLU ++data.dataset.n_shot=0 probes.layers=[-1,-8,-16,-24] \
  seed=0 data.max_length=4096 data.llm_batch_size=1

CUDA_VISIBLE_DEVICES=0 python baseline_eval_seeds/probes_train.py \
  experiment=qwen_chat_base train_on_dataset=train \
  data.name=MMLU ++data.dataset.n_shot=0 probes.layers=[-1,-7,-14,-21] \
  seed=0 data.max_length=4096 data.llm_batch_size=1

# HellaSwag
CUDA_VISIBLE_DEVICES=0 python baseline_eval_seeds/probes_train.py \
  experiment=llama_chat_base train_on_dataset=train \
  data.name=HellaSwag ++data.dataset.n_shot=0 probes.layers=[-1,-8,-16,-24] \
  seed=0 data.max_length=4096 data.llm_batch_size=1

CUDA_VISIBLE_DEVICES=0 python baseline_eval_seeds/probes_train.py \
  experiment=qwen_chat_base train_on_dataset=train probes.layers=[-1,-7,-14,-21] \
  data.name=HellaSwag ++data.dataset.n_shot=0 \
  seed=0 data.max_length=4096 data.llm_batch_size=1
