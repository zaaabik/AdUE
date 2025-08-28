export grid_name=entropyft_wo_lr,entropyft_ablation_bce_only
OMP_NUM_THREADS=2 python max_probe_fine_tune_seeds/adue_on_embeddings_lightning.py -m train_on_dataset=train \
experiment=llama_chat_base grid=${grid_name} data.name=MMLU seed=0 \
embedding_path='${oc.env:PROJECT_ROOT}/data/embeddings/llama_chat_results_ds_0_train_gird_full/state.pkl'


export grid_name=entropyft_ablation_bce_only_linear_random,entropyft_l2sp,entropyft_reg
OMP_NUM_THREADS=2 python max_probe_fine_tune_seeds/adue_on_embeddings_lightning.py -m train_on_dataset=train \
experiment=llama_chat_base grid=${grid_name} data.name=MMLU seed=0 \
embedding_path='${oc.env:PROJECT_ROOT}/data/embeddings/llama_chat_results_ds_0_train_gird_full/state.pkl'


export grid_name=srft_ablation_bce_only_linear_random,srft_l2sp,srft_reg
OMP_NUM_THREADS=2 python max_probe_fine_tune_seeds/adue_on_embeddings_lightning.py -m train_on_dataset=train \
experiment=llama_chat_base grid=${grid_name} data.name=MMLU seed=0 \
embedding_path='${oc.env:PROJECT_ROOT}/data/embeddings/llama_chat_results_ds_0_train_gird_full/state.pkl'

export grid_name=srft_wo_lr,srft_ablation_bce_only
OMP_NUM_THREADS=2 python max_probe_fine_tune_seeds/adue_on_embeddings_lightning.py -m train_on_dataset=train \
experiment=llama_chat_base grid=${grid_name} data.name=MMLU seed=0 \
embedding_path='${oc.env:PROJECT_ROOT}/data/embeddings/llama_chat_results_ds_0_train_gird_full/state.pkl'

export grid_name=entropyft,entropyft_ablation_bce_only,entropyft_ablation_bce_only_linear_random,srft,srft_ablation_bce_only,srft_ablation_bce_only_linear_random
OMP_NUM_THREADS=4 python max_probe_fine_tune_seeds/adue_on_embeddings_lightning.py -m train_on_dataset=train experiment=roberta grid=${grid_name} data.name=SST5,SST2,cola,toxigen,20newsgroups seed=0,1,2 &
OMP_NUM_THREADS=4 python max_probe_fine_tune_seeds/adue_on_embeddings_lightning.py -m train_on_dataset=train experiment=qwen grid=${grid_name} data.name=SST5,SST2,cola,toxigen,20newsgroups seed=0,1,2 &
OMP_NUM_THREADS=4 python max_probe_fine_tune_seeds/adue_on_embeddings_lightning.py -m train_on_dataset=train experiment=electra grid=${grid_name} data.name=SST5,SST2,cola,toxigen,20newsgroups seed=0,1,2 &
OMP_NUM_THREADS=4 python max_probe_fine_tune_seeds/adue_on_embeddings_lightning.py -m train_on_dataset=train experiment=llama grid=${grid_name} data.name=SST5,SST2,cola,toxigen,20newsgroups seed=0,1,2 &
OMP_NUM_THREADS=4 python max_probe_fine_tune_seeds/adue_on_embeddings_lightning.py -m train_on_dataset=train experiment=roberta grid=${grid_name} data.name=SST5,SST2,cola,toxigen,20newsgroups seed=4,3 &
OMP_NUM_THREADS=4 python max_probe_fine_tune_seeds/adue_on_embeddings_lightning.py -m train_on_dataset=train experiment=qwen grid=${grid_name} data.name=SST5,SST2,cola,toxigen,20newsgroups seed=4,3 &
OMP_NUM_THREADS=4 python max_probe_fine_tune_seeds/adue_on_embeddings_lightning.py -m train_on_dataset=train experiment=electra grid=${grid_name} data.name=SST5,SST2,cola,toxigen,20newsgroups seed=4,3 &
OMP_NUM_THREADS=4 python max_probe_fine_tune_seeds/adue_on_embeddings_lightning.py -m train_on_dataset=train experiment=llama grid=${grid_name} data.name=SST5,SST2,cola,toxigen,20newsgroups seed=4,3 &



export grid_name=entropyft_l2sp,entropyft_reg,srft_l2sp,srft_reg
OMP_NUM_THREADS=4 python max_probe_fine_tune_seeds/adue_on_embeddings_lightning.py -m train_on_dataset=train experiment=roberta grid=${grid_name} data.name=SST5,SST2,cola,toxigen,20newsgroups seed=0,1,2 &
OMP_NUM_THREADS=4 python max_probe_fine_tune_seeds/adue_on_embeddings_lightning.py -m train_on_dataset=train experiment=qwen grid=${grid_name} data.name=SST5,SST2,cola,toxigen,20newsgroups seed=0,1,2  &
OMP_NUM_THREADS=4 python max_probe_fine_tune_seeds/adue_on_embeddings_lightning.py -m train_on_dataset=train experiment=electra grid=${grid_name} data.name=SST5,SST2,cola,toxigen,20newsgroups seed=0,1,2 &
OMP_NUM_THREADS=4 python max_probe_fine_tune_seeds/adue_on_embeddings_lightning.py -m train_on_dataset=train experiment=llama grid=${grid_name} data.name=SST5,SST2,cola,toxigen,20newsgroups seed=0,1,2  &
OMP_NUM_THREADS=4 python max_probe_fine_tune_seeds/adue_on_embeddings_lightning.py -m train_on_dataset=train experiment=roberta grid=${grid_name} data.name=SST5,SST2,cola,toxigen,20newsgroups seed=4,3  &
OMP_NUM_THREADS=4 python max_probe_fine_tune_seeds/adue_on_embeddings_lightning.py -m train_on_dataset=train experiment=qwen grid=${grid_name} data.name=SST5,SST2,cola,toxigen,20newsgroups seed=4,3 &
OMP_NUM_THREADS=4 python max_probe_fine_tune_seeds/adue_on_embeddings_lightning.py -m train_on_dataset=train experiment=electra grid=${grid_name} data.name=SST5,SST2,cola,toxigen,20newsgroups seed=4,3  &
OMP_NUM_THREADS=4 python max_probe_fine_tune_seeds/adue_on_embeddings_lightning.py -m train_on_dataset=train experiment=llama grid=${grid_name} data.name=SST5,SST2,cola,toxigen,20newsgroups seed=4,3  &