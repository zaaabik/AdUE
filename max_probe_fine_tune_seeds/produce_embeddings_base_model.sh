CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 python max_probe_fine_tune_seeds/produce_embeddings.py train_on_dataset=train experiment=llama_chat_base grid=debug data.name=MMLU


CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 python max_probe_fine_tune_seeds/produce_embeddings.py \
train_on_dataset=train experiment=llama_chat_base data.llm_batch_size=4 \
grid=debug data.name=MMLU ++data.dataset.n_shot=0 \
++data.dataset.debug=true data.max_length=4096


CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 python max_probe_fine_tune_seeds/produce_embeddings.py \
train_on_dataset=train experiment=llama_chat_base data.llm_batch_size=1 \
grid=debug data.name=MMLU ++data.dataset.n_shot=5 \
++data.dataset.debug=false data.max_length=4096

