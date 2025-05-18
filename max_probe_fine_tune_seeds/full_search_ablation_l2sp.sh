CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 python max_probe_fine_tune_seeds/smooth_max_reg-oc-l2sp.py -m train_on_dataset=train experiment=roberta,llama grid=bce_only_cls_random,bce_only_linear_random seed=0,1,2,3,4 data.name=CoLa &
CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=4 python max_probe_fine_tune_seeds/smooth_max_reg-oc-l2sp.py -m train_on_dataset=train experiment=roberta,llama grid=bce_only_cls_random,bce_only_linear_random seed=0,1,2,3,4 data.name=NewsGroups &
CUDA_VISIBLE_DEVICES=2 OMP_NUM_THREADS=4 python max_probe_fine_tune_seeds/smooth_max_reg-oc-l2sp.py -m train_on_dataset=train experiment=roberta,llama grid=bce_only_cls_random,bce_only_linear_random seed=0,1,2,3,4 data.name=SST5,SST2  &
CUDA_VISIBLE_DEVICES=3 OMP_NUM_THREADS=4 python max_probe_fine_tune_seeds/smooth_max_reg-oc-l2sp.py -m train_on_dataset=train experiment=roberta,llama grid=bce_only_cls_random,bce_only_linear_random seed=0,1,2,3,4 data.name=ToxigenDataset &


CUDA_VISIBLE_DEVICES=4 OMP_NUM_THREADS=4 python max_probe_fine_tune_seeds/smooth_max_reg-oc-l2sp.py -m train_on_dataset=train experiment=electra,qwen grid=bce_only_cls_random,bce_only_linear_random seed=0,1,2,3,4 data.name=CoLa &
CUDA_VISIBLE_DEVICES=5 OMP_NUM_THREADS=4 python max_probe_fine_tune_seeds/smooth_max_reg-oc-l2sp.py -m train_on_dataset=train experiment=electra,qwen grid=bce_only_cls_random,bce_only_linear_random seed=0,1,2,3,4 data.name=NewsGroups &
CUDA_VISIBLE_DEVICES=6 OMP_NUM_THREADS=4 python max_probe_fine_tune_seeds/smooth_max_reg-oc-l2sp.py -m train_on_dataset=train experiment=electra,qwen grid=bce_only_cls_random,bce_only_linear_random seed=0,1,2,3,4 data.name=SST5,SST2  &
CUDA_VISIBLE_DEVICES=7 OMP_NUM_THREADS=4 python max_probe_fine_tune_seeds/smooth_max_reg-oc-l2sp.py -m train_on_dataset=train experiment=electra,qwen grid=bce_only_cls_random,bce_only_linear_random seed=0,1,2,3,4 data.name=ToxigenDataset &
