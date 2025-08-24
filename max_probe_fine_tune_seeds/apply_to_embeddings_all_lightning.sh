export grid_name=entropyft,entropyft_ablation_bce_only,entropyft_ablation_bce_only_linear_random,srft,srft_ablation_bce_only,srft_ablation_bce_only_linear_random
OMP_NUM_THREADS=4 python max_probe_fine_tune_seeds/adue_on_embeddings_lightning.py -m train_on_dataset=train experiment=roberta grid=${grid_name} data.name=SST5,SST2,cola,toxigen,20newsgroups seed=0,1,2 embedding_path='/Users/HawkA/Desktop/jupyter/adue/data/max_probe_fine_tune_seeds-results/${model_name}_results_ds_${seed}_train_gird_full/LoraConfig_${data.name}_state.pickle' &
OMP_NUM_THREADS=4 python max_probe_fine_tune_seeds/adue_on_embeddings_lightning.py -m train_on_dataset=train experiment=qwen grid=${grid_name} data.name=SST5,SST2,cola,toxigen,20newsgroups seed=0,1,2 embedding_path='/Users/HawkA/Desktop/jupyter/adue/data/max_probe_fine_tune_seeds-results/${model_name}_results_ds_${seed}_train_gird_full/LoraConfig_${data.name}_state.pickle' &
OMP_NUM_THREADS=4 python max_probe_fine_tune_seeds/adue_on_embeddings_lightning.py -m train_on_dataset=train experiment=electra grid=${grid_name} data.name=SST5,SST2,cola,toxigen,20newsgroups seed=0,1,2 embedding_path='/Users/HawkA/Desktop/jupyter/adue/data/max_probe_fine_tune_seeds-results/${model_name}_results_ds_${seed}_train_gird_full/LoraConfig_${data.name}_state.pickle' &
OMP_NUM_THREADS=4 python max_probe_fine_tune_seeds/adue_on_embeddings_lightning.py -m train_on_dataset=train experiment=llama grid=${grid_name} data.name=SST5,SST2,cola,toxigen,20newsgroups seed=0,1,2 embedding_path='/Users/HawkA/Desktop/jupyter/adue/data/max_probe_fine_tune_seeds-results/${model_name}_results_ds_${seed}_train_gird_full/LoraConfig_${data.name}_state.pickle' &
OMP_NUM_THREADS=4 python max_probe_fine_tune_seeds/adue_on_embeddings_lightning.py -m train_on_dataset=train experiment=roberta grid=${grid_name} data.name=SST5,SST2,cola,toxigen,20newsgroups seed=4,3 embedding_path='/Users/HawkA/Desktop/jupyter/adue/data/max_probe_fine_tune_seeds-results/${model_name}_results_ds_${seed}_train_gird_full/LoraConfig_${data.name}_state.pickle' &
OMP_NUM_THREADS=4 python max_probe_fine_tune_seeds/adue_on_embeddings_lightning.py -m train_on_dataset=train experiment=qwen grid=${grid_name} data.name=SST5,SST2,cola,toxigen,20newsgroups seed=4,3 embedding_path='/Users/HawkA/Desktop/jupyter/adue/data/max_probe_fine_tune_seeds-results/${model_name}_results_ds_${seed}_train_gird_full/LoraConfig_${data.name}_state.pickle' &
OMP_NUM_THREADS=4 python max_probe_fine_tune_seeds/adue_on_embeddings_lightning.py -m train_on_dataset=train experiment=electra grid=${grid_name} data.name=SST5,SST2,cola,toxigen,20newsgroups seed=4,3 embedding_path='/Users/HawkA/Desktop/jupyter/adue/data/max_probe_fine_tune_seeds-results/${model_name}_results_ds_${seed}_train_gird_full/LoraConfig_${data.name}_state.pickle' &
OMP_NUM_THREADS=4 python max_probe_fine_tune_seeds/adue_on_embeddings_lightning.py -m train_on_dataset=train experiment=llama grid=${grid_name} data.name=SST5,SST2,cola,toxigen,20newsgroups seed=4,3 embedding_path='/Users/HawkA/Desktop/jupyter/adue/data/max_probe_fine_tune_seeds-results/${model_name}_results_ds_${seed}_train_gird_full/LoraConfig_${data.name}_state.pickle' &



export grid_name=entropyft_l2sp,entropyft_reg,srft_l2sp,srft_reg
OMP_NUM_THREADS=4 python max_probe_fine_tune_seeds/adue_on_embeddings_lightning.py -m train_on_dataset=train experiment=roberta grid=${grid_name} data.name=SST5,SST2,cola,toxigen,20newsgroups seed=0,1,2 embedding_path='/Users/HawkA/Desktop/jupyter/adue/data/max_probe_fine_tune_seeds-results/${model_name}_results_ds_${seed}_train_gird_full/LoraConfig_${data.name}_state.pickle' &
OMP_NUM_THREADS=4 python max_probe_fine_tune_seeds/adue_on_embeddings_lightning.py -m train_on_dataset=train experiment=qwen grid=${grid_name} data.name=SST5,SST2,cola,toxigen,20newsgroups seed=0,1,2 embedding_path='/Users/HawkA/Desktop/jupyter/adue/data/max_probe_fine_tune_seeds-results/${model_name}_results_ds_${seed}_train_gird_full/LoraConfig_${data.name}_state.pickle' &
OMP_NUM_THREADS=4 python max_probe_fine_tune_seeds/adue_on_embeddings_lightning.py -m train_on_dataset=train experiment=electra grid=${grid_name} data.name=SST5,SST2,cola,toxigen,20newsgroups seed=0,1,2 embedding_path='/Users/HawkA/Desktop/jupyter/adue/data/max_probe_fine_tune_seeds-results/${model_name}_results_ds_${seed}_train_gird_full/LoraConfig_${data.name}_state.pickle' &
OMP_NUM_THREADS=4 python max_probe_fine_tune_seeds/adue_on_embeddings_lightning.py -m train_on_dataset=train experiment=llama grid=${grid_name} data.name=SST5,SST2,cola,toxigen,20newsgroups seed=0,1,2 embedding_path='/Users/HawkA/Desktop/jupyter/adue/data/max_probe_fine_tune_seeds-results/${model_name}_results_ds_${seed}_train_gird_full/LoraConfig_${data.name}_state.pickle' &
OMP_NUM_THREADS=4 python max_probe_fine_tune_seeds/adue_on_embeddings_lightning.py -m train_on_dataset=train experiment=roberta grid=${grid_name} data.name=SST5,SST2,cola,toxigen,20newsgroups seed=4,3 embedding_path='/Users/HawkA/Desktop/jupyter/adue/data/max_probe_fine_tune_seeds-results/${model_name}_results_ds_${seed}_train_gird_full/LoraConfig_${data.name}_state.pickle' &
OMP_NUM_THREADS=4 python max_probe_fine_tune_seeds/adue_on_embeddings_lightning.py -m train_on_dataset=train experiment=qwen grid=${grid_name} data.name=SST5,SST2,cola,toxigen,20newsgroups seed=4,3 embedding_path='/Users/HawkA/Desktop/jupyter/adue/data/max_probe_fine_tune_seeds-results/${model_name}_results_ds_${seed}_train_gird_full/LoraConfig_${data.name}_state.pickle' &
OMP_NUM_THREADS=4 python max_probe_fine_tune_seeds/adue_on_embeddings_lightning.py -m train_on_dataset=train experiment=electra grid=${grid_name} data.name=SST5,SST2,cola,toxigen,20newsgroups seed=4,3 embedding_path='/Users/HawkA/Desktop/jupyter/adue/data/max_probe_fine_tune_seeds-results/${model_name}_results_ds_${seed}_train_gird_full/LoraConfig_${data.name}_state.pickle' &
OMP_NUM_THREADS=4 python max_probe_fine_tune_seeds/adue_on_embeddings_lightning.py -m train_on_dataset=train experiment=llama grid=${grid_name} data.name=SST5,SST2,cola,toxigen,20newsgroups seed=4,3 embedding_path='/Users/HawkA/Desktop/jupyter/adue/data/max_probe_fine_tune_seeds-results/${model_name}_results_ds_${seed}_train_gird_full/LoraConfig_${data.name}_state.pickle' &


export grid_name=entropyft,entropyft_ablation_bce_only,entropyft_ablation_bce_only_linear_random,srft,srft_ablation_bce_only,srft_ablation_bce_only_linear_random
OMP_NUM_THREADS=4 python max_probe_fine_tune_seeds/adue_on_embeddings_lightning.py -m train_on_dataset=train \
experiment=llama_chat_base grid=${grid_name} data.name=MMLU seed=0 \
embedding_path='/Users/HawkA/Desktop/jupyter/adue/data/base_model-embeddings/llama-7b-chat/base_full_MMLU_state_refactored.pickle'

export grid_name=entropyft_l2sp,entropyft_reg,srft_l2sp,srft_reg
OMP_NUM_THREADS=4 python max_probe_fine_tune_seeds/adue_on_embeddings_lightning.py -m train_on_dataset=train \
experiment=llama_chat_base grid=${grid_name} data.name=MMLU seed=0 \
embedding_path='/Users/HawkA/Desktop/jupyter/adue/data/base_model-embeddings/llama-7b-chat/base_full_MMLU_state_refactored.pickle'



OMP_NUM_THREADS=4 python max_probe_fine_tune_seeds/adue_on_embeddings_lightning.py -m train_on_dataset=train experiment=llama_chat_base grid=entropyft data.name=MMLU seed=0 embedding_path='/Users/HawkA/Desktop/jupyter/adue/data/baseline_eval_seeds-results_on_embeddings/base_model_llama/MMLU/base_full_MMLU_state_refactored.pickle'


OMP_NUM_THREADS=4 python \
baseline_eval_seeds/eval_on_embeddings.py train_on_dataset=train \
normalization=true experiment=llama_chat_base seed=0 data.name=MMLU \
embedding_path='/Users/HawkA/Desktop/jupyter/adue/data/baseline_eval_seeds-results_on_embeddings/base_model_llama/MMLU/base_full_MMLU_state_refactored.pickle'



