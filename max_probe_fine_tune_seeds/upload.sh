export grid_name=entropyft_wo_lr,entropyft_ablation_bce_only,entropyft_ablation_bce_only_linear_random,entropyft_l2sp,entropyft_reg
OMP_NUM_THREADS=2 python max_probe_fine_tune_seeds/adue_on_embeddings_lightning.py -m train_on_dataset=train \
experiment=llama_chat_base grid=${grid_name} data.name=MMLU seed=0 \
embedding_path='/Users/HawkA/Desktop/jupyter/adue/data/hf_data/embeddings/llama_chat_results_ds_0_train_gird_full/state.pkl'

export grid_name=srft_wo_lr,srft_ablation_bce_only,srft_ablation_bce_only_linear_random,srft_l2sp,srft_reg
OMP_NUM_THREADS=2 python max_probe_fine_tune_seeds/adue_on_embeddings_lightning.py -m train_on_dataset=train \
experiment=llama_chat_base grid=${grid_name} data.name=MMLU seed=0 \
embedding_path='/Users/HawkA/Desktop/jupyter/adue/data/hf_data/embeddings/llama_chat_results_ds_0_train_gird_full/state.pkl'



huggingface-cli download zaaabik/adue --local-dir-use-symlinks True --local-dir /Users/HawkA/Desktop/jupyter/adue/data/hf_data2 --repo-type dataset --include "max_probe_fine_tune_seeds-results/*_results_ds_*_train_gird_full/LoraConfig_*_state.pickle"





huggingface-cli upload-large-folder zaaabik/adue-data /Users/HawkA/Desktop/jupyter/adue/data/hf_data2/* embeddings/ --repo-type dataset


huggingface-cli upload zaaabik/adue-data /Users/HawkA/Desktop/jupyter/adue/data/hf_data2/max_probe_fine_tune_seeds-results/ embeddings/ --repo-type dataset
/Users/HawkA/Desktop/jupyter/adue/data/hf_data2/ /Users/HawkA/Desktop/jupyter/adue/data/hf_data/embeddings/llama_chat_results_ds_0_train_gird_full

cp -r /Users/HawkA/Desktop/jupyter/adue/data/hf_data/embeddings/llama_chat_results_ds_0_train_gird_full /Users/HawkA/Desktop/jupyter/adue/data/hf_data2/max_probe_fine_tune_seeds-results




huggingface-cli upload zaaabik/adue-data /Users/HawkA/Desktop/jupyter/adue/data/hf_data2/max_probe_fine_tune_seeds-results/ embeddings/ --repo-type dataset
huggingface-cli upload zaaabik/adue-data /Users/HawkA/Desktop/jupyter/adue/data/hf_data2/models/ models/ --repo-type dataset


huggingface-cli download zaaabik/adue-data --local-dir data --repo-type dataset

huggingface-cli download zaaabik/adue-data \
embeddings/Qwen2.5-7B-Instruct_results_ds_0_train_gird_full/base_full_MMLU_0_shot_state.pickle \
--local-dir /workspace/src/AdUE/data/embeddings/Qwen2.5-7B-Instruct_results_ds_0_train_gird_full \
--repo-type dataset

huggingface-cli download zaaabik/adue-data \
embeddings/Qwen2.5-7B-Instruct_results_ds_0_train_gird_full/base_full_HellaSwag_0_shot_state.pickle \
--local-dir /workspace/src/AdUE/data/embeddings/Qwen2.5-7B-Instruct_results_ds_0_train_gird_full \
--repo-type dataset

huggingface-cli download zaaabik/adue-data \
embeddings/Llama-2-7b-chat-hf_results_ds_0_train_gird_full/base_full_HellaSwag_0_shot_state.pickle \
--local-dir data \
--repo-type dataset


huggingface-cli download zaaabik/adue-data \
embeddings/Qwen2.5-7B-Instruct_results_ds_0_train_gird_full/base_full_HellaSwag_0_shot_state.pickle \
--local-dir data \
--repo-type dataset

huggingface-cli download zaaabik/adue-data \
embeddings/Qwen2.5-7B-Instruct_results_ds_0_train_gird_full/base_full_MMLU_0_shot_state.pickle \
--local-dir data \
--repo-type dataset





huggingface-cli download zaaabik/adue-results --local-dir data/results/ --repo-type dataset

huggingface-cli download zaaabik/adue-data \
embeddings/Qwen2.5-7B-Instruct_results_ds_0_train_gird_full/base_full_MMLU_0_shot_state.pickle \
--local-dir data/embeddings/Qwen2.5-7B-Instruct_results_ds_0_train_gird_full \
--repo-type dataset

huggingface-cli download zaaabik/adue-data \
embeddings/Qwen2.5-7B-Instruct_results_ds_0_train_gird_full/base_full_HellaSwag_0_shot_state.pickle \
--local-dir data/embeddings/Qwen2.5-7B-Instruct_results_ds_0_train_gird_full \
--repo-type dataset










cp -r  /Users/HawkA/Desktop/jupyter/*_runs_seeds /Users/HawkA/Desktop/jupyter/adue/data/hf_data2/models/

CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 python baseline_eval_seeds/eval.py train_on_dataset=train normalization=true experiment=roberta seed=0 data.name=CoLa

OMP_NUM_THREADS=4 python \
baseline_eval_seeds/eval_on_embeddings.py -m train_on_dataset=train \
normalization=true experiment=llama,electra,roberta seed=0,1,2,3,4 data.name=ToxigenDataset \
embedding_path='${oc.env:PROJECT_ROOT}/data/embeddings/${model_name}_results_ds_${seed}_train_gird_full/LoraConfig_toxigen_state.pickle'

OMP_NUM_THREADS=4 python \
baseline_eval_seeds/eval_on_embeddings.py -m train_on_dataset=train \
normalization=true experiment=llama seed=0 data.name=ToxigenDataset \
embedding_path='${oc.env:PROJECT_ROOT}/data/embeddings/${model_name}_results_ds_${seed}_train_gird_full/LoraConfig_toxigen_state.pickle'

OMP_NUM_THREADS=4 python \
baseline_eval_seeds/eval_on_embeddings.py train_on_dataset=train \
normalization=true experiment=roberta seed=0 data.name=ToxigenDataset \
embedding_path='${oc.env:PROJECT_ROOT}/data/embeddings/${model_name}_results_ds_${seed}_train_gird_full/LoraConfig_toxigen_state.pickle'


mlflow ui --port 8999 --backend-store-uri sqlite:///database.db --workers 1



export grid_name=entropyft
python max_probe_fine_tune_seeds/adue_on_embeddings_lightning.py -m train_on_dataset=train \
experiment=llama_chat_base grid=${grid_name} data.name=MMLU seed=0 \
embedding_path=/workspace/AdUE/data/embeddings/Llama-2-7b-chat-hf_results_ds_0_train_gird_debug/base_full_MMLU_5_shot_debug_state.pickle



export grid_name=entropyft
python max_probe_fine_tune_seeds/adue_on_embeddings_lightning.py -m train_on_dataset=train \
experiment=llama_chat_base grid=${grid_name} data.name=MMLU seed=0 \
embedding_path='${oc.env:PROJECT_ROOT}/data/embeddings/Llama-2-7b-chat-hf_results_ds_0_train_gird_full/base_full_MMLU_0_shot_state.pickle' \
save_dir='${oc.env:PROJECT_ROOT}/max_probe_fine_tune_seeds/results_on_embeddings/grid_${grid.name}/model_${model_name}_${train_on_dataset}/dataset_MMLU_0_shot/seed_${seed}/' &


export grid_name=entropyft_ablation_bce_only,entropyft_ablation_bce_only_linear_random,entropyft_l2sp,entropyft_reg
python max_probe_fine_tune_seeds/adue_on_embeddings_lightning.py -m train_on_dataset=train \
experiment=llama_chat_base grid=${grid_name} data.name=MMLU seed=0 \
embedding_path='${oc.env:PROJECT_ROOT}/data/embeddings/Llama-2-7b-chat-hf_results_ds_0_train_gird_full/base_full_MMLU_0_shot_state.pickle' \
save_dir='${oc.env:PROJECT_ROOT}/max_probe_fine_tune_seeds/results_on_embeddings/grid_${grid.name}/model_${model_name}_${train_on_dataset}/dataset_MMLU_0_shot/seed_${seed}/' &


export grid_name=srft
python max_probe_fine_tune_seeds/adue_on_embeddings_lightning.py -m train_on_dataset=train \
experiment=llama_chat_base grid=${grid_name} data.name=MMLU seed=0 \
embedding_path='${oc.env:PROJECT_ROOT}/data/embeddings/Llama-2-7b-chat-hf_results_ds_0_train_gird_full/base_full_MMLU_0_shot_state.pickle' \
save_dir='${oc.env:PROJECT_ROOT}/max_probe_fine_tune_seeds/results_on_embeddings/grid_${grid.name}/model_${model_name}_${train_on_dataset}/dataset_MMLU_0_shot/seed_${seed}/' &

export grid_name=srft_ablation_bce_only_linear_random,srft_l2sp,srft_reg,srft_ablation_bce_only
python max_probe_fine_tune_seeds/adue_on_embeddings_lightning.py -m train_on_dataset=train \
experiment=llama_chat_base grid=${grid_name} data.name=MMLU seed=0 \
embedding_path='${oc.env:PROJECT_ROOT}/data/embeddings/Llama-2-7b-chat-hf_results_ds_0_train_gird_full/base_full_MMLU_0_shot_state.pickle' \
save_dir='${oc.env:PROJECT_ROOT}/max_probe_fine_tune_seeds/results_on_embeddings/grid_${grid.name}/model_${model_name}_${train_on_dataset}/dataset_MMLU_0_shot/seed_${seed}/' &



mlflow ui --port 8999 --backend-store-uri sqlite:///mlflow/database.db


huggingface-cli upload zaaabik/adue-results \
/workspace/AdUE/max_probe_fine_tune_seeds/results_on_embeddings/ \
results_on_embeddings/ \
--repo-type dataset

huggingface-cli upload zaaabik/adue-results \
/Users/HawkA/Desktop/jupyter/adue/data/baseline_eval_seeds-results_on_embeddings \
baseline_results_on_embeddings/ \
--repo-type dataset


huggingface-cli upload zaaabik/adue-results \
/workspace/AdUE/baseline_eval_seeds/results_on_embeddings/ \
baseline_results_on_embeddings/ \
--repo-type dataset

huggingface-cli upload zaaabik/adue-results \
/workspace/AdUE/mlflow/ \
mlflow/server_2/ \
--repo-type dataset

huggingface-cli upload zaaabik/adue-results \
/workspace/src/AdUE/mlflow/database.db \
mlflow/server_3/ \
--repo-type dataset

huggingface-cli upload zaaabik/adue-results \
/workspace/AdUE/mlflow/database.db \
mlflow/server_4/ \
--repo-type dataset

huggingface-cli upload zaaabik/adue-results \
/workspace/AdUE/mlflow/database.db \
mlflow/server_5/ \
--repo-type dataset

huggingface-cli upload zaaabik/adue-results \
/workspace/src/AdUE/mlflow/ \
mlflow/server_1/ \
--repo-type dataset



OMP_NUM_THREADS=4 python \
baseline_eval_seeds/eval_on_embeddings.py -m train_on_dataset=train \
normalization=true experiment=llama_chat_base seed=0 data.name=MMLU \
++data.dataset.n_shot=0 \
embedding_path='${oc.env:PROJECT_ROOT}/data/embeddings/llama_chat_results_ds_0_train_gird_full/state.pkl'


huggingface-cli download zaaabik/adue-results --local-dir results --repo-type dataset


huggingface-cli upload zaaabik/adue-data \
/workspace/AdUE/data/embeddings \
embeddings/ \
--repo-type dataset

huggingface-cli upload zaaabik/adue-results \
/workspace/AdUE/baseline_eval_seeds/results_on_embeddings/ \
baseline_results_on_embeddings/ \
--repo-type dataset


huggingface-cli upload zaaabik/adue-results \
/workspace/AdUE/max_probe_fine_tune_seeds/results_on_embeddings/ \
results_on_embeddings/ \
--repo-type dataset




huggingface-cli upload zaaabik/adue-data \
/workspace/AdUE/data/embeddings \
embeddings/ \
--repo-type dataset
