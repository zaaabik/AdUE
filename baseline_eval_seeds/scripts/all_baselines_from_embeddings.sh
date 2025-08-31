OMP_NUM_THREADS=4 python \
baseline_eval_seeds/eval_on_embeddings.py -m train_on_dataset=train \
normalization=true experiment=llama_chat_base seed=0 data.name=MMLU \
++data.dataset.n_shot=0 \
embedding_path='${oc.env:PROJECT_ROOT}/data/embeddings/Llama-2-7b-chat-hf_results_ds_0_train_gird_full/base_full_MMLU_0_shot_state.pickle'



OMP_NUM_THREADS=4 python \
baseline_eval_seeds/eval_on_embeddings.py -m train_on_dataset=train \
normalization=true experiment=electra,roberta,llama seed=0,1,2,3,4 data.name=ToxigenDataset \
embedding_path='${oc.env:PROJECT_ROOT}/embeddings/${model_name}_results_ds_${seed}_train_gird_full/LoraConfig_toxigen_state.pickle'
OMP_NUM_THREADS=4 python \
baseline_eval_seeds/eval_on_embeddings.py -m train_on_dataset=train \
normalization=true experiment=electra,roberta,llama seed=0,1,2,3,4 data.name=NewsGroups \
embedding_path='${oc.env:PROJECT_ROOT}/data/embeddings//${model_name}_results_ds_${seed}_train_gird_full/LoraConfig_20newsgroups_state.pickle'
OMP_NUM_THREADS=4 python \
baseline_eval_seeds/eval_on_embeddings.py -m train_on_dataset=train \
normalization=true experiment=electra,roberta,llama seed=0,1,2,3,4 data.name=CoLa \
embedding_path='${oc.env:PROJECT_ROOT}/data/embeddings//${model_name}_results_ds_${seed}_train_gird_full/LoraConfig_cola_state.pickle'
OMP_NUM_THREADS=4 python \
baseline_eval_seeds/eval_on_embeddings.py -m train_on_dataset=train \
normalization=true experiment=electra,roberta,llama seed=0,1,2,3,4 data.name=SST5 \
embedding_path='${oc.env:PROJECT_ROOT}/data/embeddings//${model_name}_results_ds_${seed}_train_gird_full/LoraConfig_SST5_state.pickle'
OMP_NUM_THREADS=4 python \
baseline_eval_seeds/eval_on_embeddings.py -m train_on_dataset=train \
normalization=true experiment=electra,roberta,llama seed=0,1,2,3,4 data.name=SST2 \
embedding_path='${oc.env:PROJECT_ROOT}/data/embeddings//${model_name}_results_ds_${seed}_train_gird_full/LoraConfig_SST2_state.pickle'



OMP_NUM_THREADS=4 python \
baseline_eval_seeds/eval_on_embeddings.py -m train_on_dataset=train \
normalization=true experiment=qwen seed=0,1,2,3,4 data.name=CoLa \
embedding_path='${oc.env:PROJECT_ROOT}/data/embeddings//${model_name}_results_ds_${seed}_train_gird_full/LoraConfig_cola_state.pickle'
OMP_NUM_THREADS=4 python \
baseline_eval_seeds/eval_on_embeddings.py -m train_on_dataset=train \
normalization=true experiment=qwen seed=0,1,2,3,4 data.name=NewsGroups \
embedding_path='${oc.env:PROJECT_ROOT}/data/embeddings//${model_name}_results_ds_${seed}_train_gird_full/LoraConfig_20newsgroups_state.pickle'
OMP_NUM_THREADS=4 python \
baseline_eval_seeds/eval_on_embeddings.py -m train_on_dataset=train \
normalization=true experiment=qwen seed=0,1,2,3,4 data.name=ToxigenDataset \
embedding_path='${oc.env:PROJECT_ROOT}/data/embeddings//${model_name}_results_ds_${seed}_train_gird_full/LoraConfig_toxigen_state.pickle'
OMP_NUM_THREADS=4 python \
baseline_eval_seeds/eval_on_embeddings.py -m train_on_dataset=train \
normalization=true experiment=qwen seed=0,1,2,3,4 data.name=SST5 \
embedding_path='${oc.env:PROJECT_ROOT}/data/embeddings//${model_name}_results_ds_${seed}_train_gird_full/LoraConfig_SST5_state.pickle'
OMP_NUM_THREADS=4 python \
baseline_eval_seeds/eval_on_embeddings.py -m train_on_dataset=train \
normalization=true experiment=qwen seed=0,1,2,3,4 data.name=SST2 \
embedding_path='${oc.env:PROJECT_ROOT}/data/embeddings//${model_name}_results_ds_${seed}_train_gird_full/LoraConfig_SST2_state.pickle'



BASE_WEIGHT_PATH=/root/models/ OMP_NUM_THREADS=4 python \
baseline_eval_seeds/eval_on_embeddings.py -m train_on_dataset=train \
normalization=true experiment=roberta seed=0,1,2,3,4 data.name=SST2 \
embedding_path='${oc.env:PROJECT_ROOT}/data/embeddings/${model_name}_results_ds_${seed}_train_gird_full/LoraConfig_SST2_state.pickle' &
BASE_WEIGHT_PATH=/root/models/ OMP_NUM_THREADS=4 python \
baseline_eval_seeds/eval_on_embeddings.py -m train_on_dataset=train \
normalization=true experiment=qwen seed=0,1,2,3,4 data.name=SST2 \
embedding_path='${oc.env:PROJECT_ROOT}/data/embeddings/${model_name}_results_ds_${seed}_train_gird_full/LoraConfig_SST2_state.pickle' &
BASE_WEIGHT_PATH=/root/models/ OMP_NUM_THREADS=4 python \
baseline_eval_seeds/eval_on_embeddings.py -m train_on_dataset=train \
normalization=true experiment=llama seed=0,1,2,3,4 data.name=SST2 \
embedding_path='${oc.env:PROJECT_ROOT}/data/embeddings/${model_name}_results_ds_${seed}_train_gird_full/LoraConfig_SST2_state.pickle' &
BASE_WEIGHT_PATH=/root/models/ OMP_NUM_THREADS=4 python \
baseline_eval_seeds/eval_on_embeddings.py -m train_on_dataset=train \
normalization=true experiment=electra seed=0,1,2,3,4 data.name=SST2 \
embedding_path='${oc.env:PROJECT_ROOT}/data/embeddings/${model_name}_results_ds_${seed}_train_gird_full/LoraConfig_SST2_state.pickle' &


BASE_WEIGHT_PATH=/root/models/ OMP_NUM_THREADS=4 python \
baseline_eval_seeds/eval_on_embeddings.py -m train_on_dataset=train \
normalization=true experiment=llama,roberta seed=0,1,2,3,4 data.name=SST2 \
embedding_path='${oc.env:PROJECT_ROOT}/data/embeddings/${model_name}_results_ds_${seed}_train_gird_full/LoraConfig_SST2_state.pickle' &
BASE_WEIGHT_PATH=/root/models/ OMP_NUM_THREADS=4 python \
baseline_eval_seeds/eval_on_embeddings.py -m train_on_dataset=train \
normalization=true experiment=electra,qwen seed=0,1,2,3,4 data.name=SST2 \
embedding_path='${oc.env:PROJECT_ROOT}/data/embeddings/${model_name}_results_ds_${seed}_train_gird_full/LoraConfig_SST2_state.pickle' &



BASE_WEIGHT_PATH=/root/models/ OMP_NUM_THREADS=4 python \
baseline_eval_seeds/eval_on_embeddings.py -m train_on_dataset=train \
normalization=true experiment=roberta seed=0 data.name=SST2 \
embedding_path='/root/data/max_probe_fine_tune_seeds-results//${model_name}_results_ds_${seed}_train_gird_full/LoraConfig_SST2_state.pickle'



BASE_WEIGHT_PATH=/root/models/ OMP_NUM_THREADS=32 python \
baseline_eval_seeds/eval_on_embeddings.py -m train_on_dataset=train \
normalization=true experiment=roberta,llama,qwen,electra seed=0,1,2,3,4 data.name=SST2 \
embedding_path='/root/data/max_probe_fine_tune_seeds-results//${model_name}_results_ds_${seed}_train_gird_full/LoraConfig_SST2_state.pickle'