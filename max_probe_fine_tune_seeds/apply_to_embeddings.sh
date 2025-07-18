CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 python max_probe_fine_tune_seeds/adue_on_embeddings.py -m train_on_dataset=train experiment=roberta grid=full_with_lambda data.name=SST5,SST2,cola,toxigen,20newsgroups seed=0,1,2,3,4 &
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 python max_probe_fine_tune_seeds/adue_on_embeddings.py -m train_on_dataset=train experiment=qwen grid=full_with_lambda data.name=SST5,SST2,cola,toxigen,20newsgroups seed=0,1,2,3,4 &
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 python max_probe_fine_tune_seeds/adue_on_embeddings.py -m train_on_dataset=train experiment=electra grid=full_with_lambda data.name=SST5,SST2,cola,toxigen,20newsgroups seed=0,1,2,3,4 &
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 python max_probe_fine_tune_seeds/adue_on_embeddings.py -m train_on_dataset=train experiment=llama grid=full_with_lambda data.name=SST5,SST2,cola,toxigen,20newsgroups seed=0,1,2,3,4 &

CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 python max_probe_fine_tune_seeds/adue_on_embeddings.py -m train_on_dataset=train experiment=roberta grid=full_with_lambda data.name=SST5,SST2,cola,toxigen,20newsgroups seed=4,3,2,1,0 &
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 python max_probe_fine_tune_seeds/adue_on_embeddings.py -m train_on_dataset=train experiment=qwen grid=full_with_lambda data.name=SST5,SST2,cola,toxigen,20newsgroups seed=4,3,2,1,0 &
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 python max_probe_fine_tune_seeds/adue_on_embeddings.py -m train_on_dataset=train experiment=electra grid=full_with_lambda data.name=SST5,SST2,cola,toxigen,20newsgroups seed=4,3,2,1,0 &
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 python max_probe_fine_tune_seeds/adue_on_embeddings.py -m train_on_dataset=train experiment=llama grid=full_with_lambda data.name=SST5,SST2,cola,toxigen,20newsgroups seed=4,3,2,1,0 &



grid_name=full_wo_epoch
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 python max_probe_fine_tune_seeds/adue_on_embeddings.py -m train_on_dataset=train experiment=roberta grid=${grid_name} data.name=SST5,SST2,cola,toxigen,20newsgroups seed=0,1,2,3,4 &
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 python max_probe_fine_tune_seeds/adue_on_embeddings.py -m train_on_dataset=train experiment=qwen grid=${grid_name} data.name=SST5,SST2,cola,toxigen,20newsgroups seed=0,1,2,3,4 &
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 python max_probe_fine_tune_seeds/adue_on_embeddings.py -m train_on_dataset=train experiment=electra grid=${grid_name} data.name=SST5,SST2,cola,toxigen,20newsgroups seed=0,1,2,3,4 &
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 python max_probe_fine_tune_seeds/adue_on_embeddings.py -m train_on_dataset=train experiment=llama grid=${grid_name} data.name=SST5,SST2,cola,toxigen,20newsgroups seed=0,1,2,3,4 &

CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 python max_probe_fine_tune_seeds/adue_on_embeddings.py -m train_on_dataset=train experiment=roberta grid=${grid_name} data.name=SST5,SST2,cola,toxigen,20newsgroups seed=4,3,2,1,0 &
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 python max_probe_fine_tune_seeds/adue_on_embeddings.py -m train_on_dataset=train experiment=qwen grid=${grid_name} data.name=SST5,SST2,cola,toxigen,20newsgroups seed=4,3,2,1,0 &
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 python max_probe_fine_tune_seeds/adue_on_embeddings.py -m train_on_dataset=train experiment=electra grid=${grid_name} data.name=SST5,SST2,cola,toxigen,20newsgroups seed=4,3,2,1,0 &
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 python max_probe_fine_tune_seeds/adue_on_embeddings.py -m train_on_dataset=train experiment=llama grid=${grid_name} data.name=SST5,SST2,cola,toxigen,20newsgroups seed=4,3,2,1,0 &
