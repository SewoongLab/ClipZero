# Understanding-Clipping-in-Zeroth-order-Optimization
Official experiment codebase for the paper **"Understanding Clipping in Zeroth-order Optimization"**.

This repository builds upon [DPZero](https://github.com/Liang137/DPZero) (*[ICML 2024] DPZero: Private Fine-Tuning of Language Models without Backpropagation*) and [MeZO](https://github.com/princeton-nlp/MeZO). It consolidates prompt-based zeroth-order (ZO) fine-tuning on RoBERTa-large into a unified codebase supporting four core experimental suites:

1. **Per-Sample Clipping & Label Noise Robustness (`--mismatch`)**: Injects controlled label corruption into the training split and tracks per-example clipping frequencies and unclipped gradient magnitudes.
2. **Mid-Training Clipping & Learning Rate Switching (`--flip_val`)**: Switches between clipped high-learning-rate regimes and unclipped low-learning-rate regimes at a target epoch.
3. **Periodic Clipping Alternation (`--mixed_flip`)**: Toggles per-sample clipping on and off every 20 epochs.
4. **Trajectory, Distribution & Calibration Diagnostics**: Tracks $L_2$ weight distance from initialization ($\|w_t - w_0\|_2$), pre/post-clipping projected gradient histograms, per-split evaluation logits, boolean accuracy masks, and **Expected Calibration Error (ECE)** reliability diagrams (`calibration.py`).

---

## 1. Installation & Environment Setup

This codebase is tested on Python `3.9.18` with `torch==2.4.0+cu121`, `transformers==4.28.1`, and `opacus==1.4.0`.

Create and activate the Conda environment using `environments.yml`:

```bash
conda env create -n dpzero -f environments.yml
conda activate dpzero

# Install additional plotting and model utilities if not already present
pip install loralib safetensors matplotlib scikit-learn pandas filelock
```

---

## 2. Repository Structure

```text
├── README.md                   # Top-level documentation
├── environments.yml            # Conda environment specification
├── run.py                      # Consolidated training & evaluation entrypoint
├── calibration.py              # ECE computation & reliability diagram generator
├── data/
│   └── prepare_datasets.sh     # Downloads GLUE/SNLI data and generates k-shot splits
├── examples/
│   ├── dpzero.sh               # Example script for DPZero / clipped ZO training
│   ├── mezo.sh                 # Example script for standard MeZO training
│   └── run_fewshot.sh          # Core wrapper script for k-shot prompt fine-tuning
├── tools/
│   ├── generate_k_shot_data.py # Generates k-shot splits from original datasets
│   └── gather_result.py        # Aggregates evaluation metrics across seeds
└── src/
    ├── trainer.py              # Consolidated ZO/DPZero Trainer (histograms, L2 dist, flip & per-sample tracking)
    ├── dataset.py              # FewShotDataset & OurInputFeatures (with example_id tracking)
    ├── models.py               # Prompt-based fine-tuning forward pass & per-sample loss
    ├── modeling_roberta.py     # Custom RoBERTa model & configuration (DPZero & LoRA support)
    ├── processors.py           # GLUE, SNLI, and text classification dataset processors
    ├── linearhead_trainer.py   # Linear probing trainer
    ├── kernel_trainer.py       # Empirical NTK trainer
    └── kernel_solvers.py       # Kernel regression/SVM solvers
```

---

## 3. Preparing the Datasets

Before running experiments, download the original datasets (SST-2, SNLI, MNLI, RTE, QNLI, etc.) into `data/original/` and generate the $k$-shot training splits:

```bash
cd roberta/data
bash prepare_datasets.sh
cd ..
```

To generate custom $k$-shot splits (e.g., $k=16$ or $k=512$), inspect or modify `data/prepare_datasets.sh` and `tools/generate_k_shot_data.py`.

---

## 4. Custom Experimental Flags Overview

In addition to standard `DPZero` arguments (`--zero_order_optim`, `--dpzero`, `--dpzero_clip_threshold`), `run.py` consolidates all experimental modes via the following flags:

| Flag | Default | Description | Output Folder |
| :--- | :--- | :--- | :--- |
| `--mismatch <float>` | `0.0` | Corrupts `<float>` fraction of training labels and tracks per-example clipping counts and magnitudes. | `mismatched/` |
| `--flip_val <int>` | `-1` | Removes clipping (`0.0`) and switches learning rate to $10^{-6}$ at epoch `flip_val + 1` using a constant LR scheduler. | `flipped_r/` |
| `--mixed_flip <bool>` | `False` | Toggles per-sample clipping on and off every 20 epochs. | `mixed/` |

All runs automatically log:
* `l2_weight_dist`: $L_2$ distance $\|w_t - w_0\|_2$ from initial weights $w_0$ at each logging step.
* `avg_clipped_per_iter`: Average number of clipped batch elements per step.
* Train, Dev, and Test evaluation metrics (`train_eval_results_*.txt`, `eval_results_*.txt`, `test_results_*.txt`) and `.npz` logit/mask files.

---

## 5. Running the Consolidated Experiments

### Experiment A: Label Corruption & Per-Sample Clipping Tracking (`--mismatch`)

Injects a specified fraction of label noise into the training set (with a locked random seed for reproducibility across runs) and tracks how often each individual training example is clipped alongside its unclipped projected gradient magnitude.

```bash
python run.py \
    --task_name SNLI \
    --data_dir data/k-shot/SNLI/16-21 \
    --overwrite_output_dir \
    --do_train --do_eval --do_predict \
    --model_name_or_path roberta-large \
    --few_shot_type prompt \
    --num_k 16 \
    --max_seq_length 256 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 16 \
    --learning_rate 5e-5 \
    --num_train_epochs 100 \
    --logging_steps 10 \
    --zero_order_optim \
    --efficient_zero_order \
    --zero_order_eps 1e-3 \
    --dpzero \
    --dpzero_clip_threshold 10.0 \
    --mismatch 0.1 \
    --seed 21 \
    --output_dir result/SNLI_mis0.1_lr5e-5_clip10.0_seed21
```

**Generated Artifacts (saved to `./mismatched/`):**
* `train_mislabled_eval_lr{lr}_clip{clip}_seed{seed}_mis{mismatch}.npy`: Array of training dataset indices whose labels were flipped.
* `sample_lr{lr}_clip{clip}_seed{seed}_mis{mismatch}.npz`: Perfectly index-aligned arrays (`example_ids`, `clip_counts`, `clip_magnitudes`) for every training example, plus console logs of the **Top 10 Most Clipped Examples** with decoded text and average unclipped gradient magnitude.
* `{task}_logits_lr{lr}_clip{clip}_seed{seed}_mis{mismatch}.npz`: Evaluation logits, predictions, labels, and correctness masks (`train`, `dev`, `test`).
* `{TASK}_model_lr{lr}_clip{clip}_seed{seed}_mis{mismatch}/`: Saved model weights, tokenizer, and training arguments.

---

### Experiment B: Mid-Training Clipping & Learning Rate Switch (`--flip_val`)

Starts training with your specified `--learning_rate` and `--dpzero_clip_threshold`, then at epoch `flip_val + 1` removes clipping (`clip = 0.0`) and switches the learning rate to $10^{-6}$ (automatically enforcing `--lr_scheduler_type constant` so the scheduler does not overwrite the manual LR switch).

```bash
python run.py \
    --task_name SNLI \
    --data_dir data/k-shot/SNLI/16-21 \
    --overwrite_output_dir \
    --do_train --do_eval --do_predict \
    --model_name_or_path roberta-large \
    --few_shot_type prompt \
    --num_k 16 \
    --max_seq_length 256 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 16 \
    --learning_rate 5e-5 \
    --num_train_epochs 100 \
    --logging_steps 10 \
    --zero_order_optim \
    --efficient_zero_order \
    --zero_order_eps 1e-3 \
    --dpzero \
    --dpzero_clip_threshold 10.0 \
    --flip_val 40 \
    --seed 21 \
    --output_dir result/SNLI_flip40_lr5e-5_clip10.0_seed21
```

**Generated Artifacts (saved to `./flipped_r/`):**
* `{TASK}_model_lr{lr}_clip{clip}_seed{seed}_flip{flip_val}/`: Test-evaluated model checkpoint and tokenizer.
* `{lr}_{clip}_histogram_seed_{seed}_{flip_val}.npz`: Flattened pre-clip (`pre_clip`) and post-clip (`post_clip`) projected gradients recorded every 100 steps.
* `{dataset}_{split}_logits_lr{lr}_clip{clip}_seed{seed}_flip{flip_val}.npz`: Per-split (`train`, `dev`, `test`) logits and correctness masks.
* `{split}_bool_mask_{lr}_{clip}_{seed}_{flip_val}.npz`: Boolean correctness masks (`1` = correct, `0` = incorrect).

---

### Experiment C: Periodic Clipping Toggle (`--mixed_flip`)

Toggles per-sample clipping on and off every 20 epochs while logging $L_2$ weight distance from initialization (`l2_weight_dist`) and average clipped elements per iteration (`avg_clipped_per_iter`).

```bash
python run.py \
    --task_name SNLI \
    --data_dir data/k-shot/SNLI/16-21 \
    --overwrite_output_dir \
    --do_train --do_eval --do_predict \
    --model_name_or_path roberta-large \
    --few_shot_type prompt \
    --num_k 16 \
    --max_seq_length 256 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 16 \
    --learning_rate 5e-5 \
    --num_train_epochs 100 \
    --logging_steps 10 \
    --zero_order_optim \
    --efficient_zero_order \
    --zero_order_eps 1e-3 \
    --dpzero \
    --dpzero_clip_threshold 10.0 \
    --mixed_flip True \
    --seed 21 \
    --output_dir result/SNLI_mixed_lr5e-5_clip10.0_seed21
```

**Generated Artifacts (saved to `./mixed/`):**
* `{TASK}_model_lr{lr}_clip{clip}_seed{seed}/`: Saved model checkpoint.
* `{lr}_{clip}_histogram_seed_{seed}.npz`: Pre-clip and post-clip gradient distributions.
* `{dataset}_{split}_logits_lr{lr}_clip{clip}_seed{seed}.npz` & `{split}_bool_mask_{lr}_{clip}_{seed}.npz`: Evaluation logits and boolean masks.

---

### Experiment D: Calibration & Reliability Diagrams (`calibration.py`)

To evaluate model calibration (15-bin Expected Calibration Error) and generate the $1 \times 3$ stacked reliability diagrams and sample fraction histograms on the full SNLI test set:

1. Place your three comparison checkpoints in `./final/` (or update `DIR_MODEL_1`, `DIR_MODEL_2`, `DIR_MODEL_3` inside `calibration.py`):
   * `./final/model_lr5e-05_clip10.0_seed21`
   * `./final/model_lr1e-05_clip50.0_seed21`
   * `./final/model_lr1e-06_clip0.0_seed21`
2. Run the calibration script:

```bash
python calibration.py
```

This outputs `snli_calibration_comparison_2.png` with the sample fraction histogram in the top row and the confidence vs. accuracy calibration bars in the bottom row.

---

## 6. Acknowledgements & Citation

Our implementation builds upon [DPZero](https://github.com/Liang137/DPZero) and [MeZO](https://github.com/princeton-nlp/MeZO):

```bibtex
@inproceedings{zhang2024dpzero,
  title={{DPZero}: Private Fine-Tuning of Language Models without Backpropagation},
  author={Liang Zhang and Bingcong Li and Kiran Koshy Thekumparampil and Sewoong Oh and Niao He},
  booktitle={International Conference on Machine Learning},
  year={2024},
  organization={PMLR}
}
```

To view our paper and results:
# ENTER PAPER HERE
