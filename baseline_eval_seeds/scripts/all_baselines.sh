CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 python baseline_eval_seeds/eval.py -m train_on_dataset=train normalization=true,false experiment=roberta seed=0,1,2,3,4 data.name=CoLa,SST5 &
CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=4 python baseline_eval_seeds/eval.py -m train_on_dataset=train normalization=true,false experiment=roberta seed=0,1,2,3,4 data.name=NewsGroups &
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 python baseline_eval_seeds/eval.py -m train_on_dataset=train normalization=true,false experiment=roberta seed=0 data.name=SST2  &
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 python baseline_eval_seeds/eval.py -m train_on_dataset=train normalization=true,false experiment=roberta seed=1 data.name=SST2  &
CUDA_VISIBLE_DEVICES=2 OMP_NUM_THREADS=4 python baseline_eval_seeds/eval.py -m train_on_dataset=train normalization=true,false experiment=roberta seed=2 data.name=SST2  &
CUDA_VISIBLE_DEVICES=2 OMP_NUM_THREADS=4 python baseline_eval_seeds/eval.py -m train_on_dataset=train normalization=true,false experiment=roberta seed=3 data.name=SST2  &
CUDA_VISIBLE_DEVICES=3 OMP_NUM_THREADS=4 python baseline_eval_seeds/eval.py -m train_on_dataset=train normalization=true,false experiment=roberta seed=4 data.name=SST2  &
CUDA_VISIBLE_DEVICES=3 OMP_NUM_THREADS=4 python baseline_eval_seeds/eval.py -m train_on_dataset=train normalization=true,false experiment=roberta seed=0,1,2,3,4 data.name=ToxigenDataset &


CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 python baseline_eval_seeds/eval.py -m train_on_dataset=train normalization=true,false experiment=llama seed=0,1,2,3,4 data.name=CoLa,SST5 &
CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=4 python baseline_eval_seeds/eval.py -m train_on_dataset=train normalization=true,false experiment=llama seed=0,1,2,3,4 data.name=NewsGroups &
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 python baseline_eval_seeds/eval.py -m train_on_dataset=train normalization=true,false experiment=llama seed=0 data.name=SST2  &
CUDA_VISIBLE_DEVICES=2 OMP_NUM_THREADS=4 python baseline_eval_seeds/eval.py -m train_on_dataset=train normalization=true,false experiment=llama seed=1 data.name=SST2  &
CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=4 python baseline_eval_seeds/eval.py -m train_on_dataset=train normalization=true,false experiment=llama seed=2 data.name=SST2  &
CUDA_VISIBLE_DEVICES=2 OMP_NUM_THREADS=4 python baseline_eval_seeds/eval.py -m train_on_dataset=train normalization=true,false experiment=llama seed=3 data.name=SST2  &
CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=4 python baseline_eval_seeds/eval.py -m train_on_dataset=train normalization=true,false experiment=llama seed=4 data.name=SST2  &
CUDA_VISIBLE_DEVICES=3 OMP_NUM_THREADS=4 python baseline_eval_seeds/eval.py -m train_on_dataset=train normalization=true,false experiment=llama seed=0,1,2,3,4 data.name=ToxigenDataset &


CUDA_VISIBLE_DEVICES=4 OMP_NUM_THREADS=4 python baseline_eval_seeds/eval.py -m train_on_dataset=train normalization=true,false experiment=qwen seed=0,1,2,3,4 data.name=CoLa,SST5 &
CUDA_VISIBLE_DEVICES=5 OMP_NUM_THREADS=4 python baseline_eval_seeds/eval.py -m train_on_dataset=train normalization=true,false experiment=qwen seed=0,1,2,3,4 data.name=NewsGroups &
CUDA_VISIBLE_DEVICES=4 OMP_NUM_THREADS=4 python baseline_eval_seeds/eval.py -m train_on_dataset=train normalization=true,false experiment=qwen seed=0 data.name=SST2  &
CUDA_VISIBLE_DEVICES=4 OMP_NUM_THREADS=4 python baseline_eval_seeds/eval.py -m train_on_dataset=train normalization=true,false experiment=qwen seed=1 data.name=SST2  &
CUDA_VISIBLE_DEVICES=5 OMP_NUM_THREADS=4 python baseline_eval_seeds/eval.py -m train_on_dataset=train normalization=true,false experiment=qwen seed=2 data.name=SST2  &
CUDA_VISIBLE_DEVICES=5 OMP_NUM_THREADS=4 python baseline_eval_seeds/eval.py -m train_on_dataset=train normalization=true,false experiment=qwen seed=3 data.name=SST2  &
CUDA_VISIBLE_DEVICES=6 OMP_NUM_THREADS=4 python baseline_eval_seeds/eval.py -m train_on_dataset=train normalization=true,false experiment=qwen seed=4 data.name=SST2  &
CUDA_VISIBLE_DEVICES=7 OMP_NUM_THREADS=4 python baseline_eval_seeds/eval.py -m train_on_dataset=train normalization=true,false experiment=qwen seed=0,1,2,3,4 data.name=ToxigenDataset &


CUDA_VISIBLE_DEVICES=7 OMP_NUM_THREADS=4 python baseline_eval_seeds/eval.py -m train_on_dataset=train normalization=true,false experiment=electra seed=0,1,2,3,4 data.name=CoLa,SST5 &
CUDA_VISIBLE_DEVICES=7 OMP_NUM_THREADS=4 python baseline_eval_seeds/eval.py -m train_on_dataset=train normalization=true,false experiment=electra seed=0,1,2,3,4 data.name=NewsGroups &
CUDA_VISIBLE_DEVICES=4 OMP_NUM_THREADS=4 python baseline_eval_seeds/eval.py -m train_on_dataset=train normalization=true,false experiment=electra seed=0 data.name=SST2  &
CUDA_VISIBLE_DEVICES=4 OMP_NUM_THREADS=4 python baseline_eval_seeds/eval.py -m train_on_dataset=train normalization=true,false experiment=electra seed=1 data.name=SST2  &
CUDA_VISIBLE_DEVICES=6 OMP_NUM_THREADS=4 python baseline_eval_seeds/eval.py -m train_on_dataset=train normalization=true,false experiment=electra seed=2 data.name=SST2  &
CUDA_VISIBLE_DEVICES=7 OMP_NUM_THREADS=4 python baseline_eval_seeds/eval.py -m train_on_dataset=train normalization=true,false experiment=electra seed=3 data.name=SST2  &
CUDA_VISIBLE_DEVICES=6 OMP_NUM_THREADS=4 python baseline_eval_seeds/eval.py -m train_on_dataset=train normalization=true,false experiment=electra seed=4 data.name=SST2  &
CUDA_VISIBLE_DEVICES=7 OMP_NUM_THREADS=4 python baseline_eval_seeds/eval.py -m train_on_dataset=train normalization=true,false experiment=electra seed=0,1,2,3,4 data.name=ToxigenDataset &
