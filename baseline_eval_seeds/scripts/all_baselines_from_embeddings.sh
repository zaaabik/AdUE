#CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 python baseline_eval_seeds/eval.py -m train_on_dataset=train normalization=true,false experiment=roberta seed=0,1,2,3,4 data.name=CoLa,SST5 &
#CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=4 python baseline_eval_seeds/eval.py -m train_on_dataset=train normalization=true,false experiment=roberta seed=0,1,2,3,4 data.name=NewsGroups &
#CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 python baseline_eval_seeds/eval.py -m train_on_dataset=train normalization=true,false experiment=roberta seed=0 data.name=SST2  &
#CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 python baseline_eval_seeds/eval.py -m train_on_dataset=train normalization=true,false experiment=roberta seed=1 data.name=SST2  &
#CUDA_VISIBLE_DEVICES=2 OMP_NUM_THREADS=4 python baseline_eval_seeds/eval.py -m train_on_dataset=train normalization=true,false experiment=roberta seed=2 data.name=SST2  &
#CUDA_VISIBLE_DEVICES=2 OMP_NUM_THREADS=4 python baseline_eval_seeds/eval.py -m train_on_dataset=train normalization=true,false experiment=roberta seed=3 data.name=SST2  &
#CUDA_VISIBLE_DEVICES=3 OMP_NUM_THREADS=4 python baseline_eval_seeds/eval.py -m train_on_dataset=train normalization=true,false experiment=roberta seed=4 data.name=SST2  &
#CUDA_VISIBLE_DEVICES=3 OMP_NUM_THREADS=4 python baseline_eval_seeds/eval.py -m train_on_dataset=train normalization=true,false experiment=roberta seed=0,1,2,3,4 data.name=ToxigenDataset &
