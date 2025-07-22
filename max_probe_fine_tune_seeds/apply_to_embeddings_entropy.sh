CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 python max_probe_fine_tune_seeds/adue_on_embeddings_v2.py -m train_on_dataset=train experiment=roberta grid=full_with_entropy data.name=SST5,SST2,cola,toxigen,20newsgroups seed=0,1,2,3,4 &
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 python max_probe_fine_tune_seeds/adue_on_embeddings_v2.py -m train_on_dataset=train experiment=qwen grid=full_with_entropy data.name=SST5,SST2,cola,toxigen,20newsgroups seed=0,1,2,3,4 &
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 python max_probe_fine_tune_seeds/adue_on_embeddings_v2.py -m train_on_dataset=train experiment=electra grid=full_with_entropy data.name=SST5,SST2,cola,toxigen,20newsgroups seed=0,1,2,3,4 &
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 python max_probe_fine_tune_seeds/adue_on_embeddings_v2.py -m train_on_dataset=train experiment=llama grid=full_with_entropy data.name=SST5,SST2,cola,toxigen,20newsgroups seed=0,1,2,3,4 &

CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 python max_probe_fine_tune_seeds/adue_on_embeddings_v2.py -m train_on_dataset=train experiment=roberta grid=full_with_entropy data.name=SST5,SST2,cola,toxigen,20newsgroups seed=4,3,2,1,0 &
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 python max_probe_fine_tune_seeds/adue_on_embeddings_v2.py -m train_on_dataset=train experiment=qwen grid=full_with_entropy data.name=SST5,SST2,cola,toxigen,20newsgroups seed=4,3,2,1,0 &
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 python max_probe_fine_tune_seeds/adue_on_embeddings_v2.py -m train_on_dataset=train experiment=electra grid=full_with_entropy data.name=SST5,SST2,cola,toxigen,20newsgroups seed=4,3,2,1,0 &
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 python max_probe_fine_tune_seeds/adue_on_embeddings_v2.py -m train_on_dataset=train experiment=llama grid=full_with_entropy data.name=SST5,SST2,cola,toxigen,20newsgroups seed=4,3,2,1,0 &


CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 python max_probe_fine_tune_seeds/adue_on_embeddings_v2.py train_on_dataset=train experiment=roberta grid=full_with_entropy data.name=SST5 seed=0 embedding_path='/Users/HawkA/Desktop/jupyter/adue/data/max_probe_fine_tune_seeds-results/${model_name}_results_ds_${seed}_train_gird_full/LoraConfig_${data.name}_state.pickle'

