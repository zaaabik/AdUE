# AdUE: Improving uncertainty estimation head for LoRA adapters in LLMs

Uncertainty estimation remains a critical challenge in adapting pretrained language models to classification tasks, particularly under parameter-efficient fine-tuning approaches like adapters and LoRA. We introduce a lightweight post-hoc fine-tuning method to enhance softmax-based uncertainty estimates. Our approach, the SmoothMax classifier head, uses a differentiable approximation of the maximum function to address overfitting and optimization instability issues. We apply additional regularization through L2-SP, anchoring the fine-tuned head weights. Evaluations on five NLP datasets (SST2, SST5, CoLA, 20 Newsgroups, Toxigen) across four transformer architectures (RoBERTa, ELECTRA, LLaMA-2, Qwen) demonstrate that our method outperforms established baselines such as Mahalanobis distance and robust distance-based estimators. Our approach is lightweight, requiring no modifications to the base model weights, and provides reliable and better-calibrated uncertainty predictions.

# Run

```bash
conda create --name adue_uncertainty_estimation_softmax python=3.10 -y
conda activate adue_uncertainty_estimation_softmax

```

First we need to install all necessary requirement:

```bash
pip install -r requirements.txt
```

Add environment file:

```bash
cp .env.example .env
```

Login for WANDB:

```bash
python -m wandb login
```

Install flash attention:

```bash
pip install flash-attn --no-build-isolation
```

Login to huggingface:

```bash
huggingface-cli login
```

Start training electra/roberta/llama/qwen for all datasets and 5 seeds:

```bash
bash scripts/all_model/train.sh
```

After training put weight of best models in folder with patter {BASE_WEIGHT_PATH}/{model_name}_runs_seeds/{data.name}/{seed}

 [Code for collection models from wandb](notebooks/collect_weights_from_wandb.ipynb)

Compute all baselines:
```bash
bash baseline_eval_seeds/scripts/all_baselines.sh
```

Run softmax response fine tune:
```bash
bash max_probe_fine_tune_seeds/full_search.sh
```

[Code for collection results](notebooks/collect_results.ipynb)