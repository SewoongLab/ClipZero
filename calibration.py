import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

# Force Python to search the directory this script lives in to find run.py
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.getcwd())

from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from run import ModelArguments, DynamicDataTrainingArguments, MyDataCollatorWithPadding
from src.modeling_roberta import RobertaConfig
from src.models import MODEL_TYPES
from src.dataset import FewShotDataset
from src.processors import num_labels_mapping

def load_weights(model_dir, device):
    checkpoint_path = os.path.join(model_dir, "pytorch_model.bin")
    if not os.path.exists(checkpoint_path):
        checkpoint_path = os.path.join(model_dir, "model.safetensors")
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Could not find weights in {model_dir}.")
        from safetensors.torch import load_file
        return load_file(checkpoint_path, device=device)
    return torch.load(checkpoint_path, map_location=device)

def get_model_label(path, prefix):
    try:
        parts = path.split("_")
        lr = parts[-3].replace("lr", "")
        clip = parts[-2].replace("clip", "")
        if clip == "0.0":
            clip = "∞"
        return f"{prefix}: lr={lr}, clip={clip}"
    except Exception:
        return prefix

def get_all_logits(model, dataloader, device):
    """Runs a model over the dataset and saves all logits."""
    model.eval()
    all_logits = []
    all_labels = []
    with torch.no_grad():
        for batch in dataloader:
            inputs = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**inputs)
            # DPZero returns (loss, logits). Index 1 is logits.
            logits = outputs[1]
            all_logits.append(logits)
            all_labels.append(inputs["labels"])
            
    return torch.cat(all_logits, dim=0), torch.cat(all_labels, dim=0)

def compute_calibration(logits, labels, num_bins=15):
    """Computes ECE and bin statistics for reliability diagrams."""
    # Convert logits to probabilities
    probs = F.softmax(logits, dim=1)
    confidences, predictions = torch.max(probs, dim=1)
    accuracies = (predictions == labels)
    
    # Equal-width binning [0, 1]
    bin_boundaries = np.linspace(0, 1, num_bins + 1)
    ece = 0.0
    bin_stats = []
    
    for i in range(num_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]
        
        if i == num_bins - 1:
            in_bin = (confidences >= bin_lower) & (confidences <= bin_upper)
        else:
            in_bin = (confidences >= bin_lower) & (confidences < bin_upper)
            
        prop_in_bin = in_bin.float().mean().item()
        
        if prop_in_bin > 0:
            accuracy_in_bin = accuracies[in_bin].float().mean().item()
            avg_confidence_in_bin = confidences[in_bin].mean().item()
            
            # ECE Accumulation
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
            
            bin_stats.append({
                'center': (bin_lower + bin_upper) / 2,
                'width': 1.0 / num_bins,
                'accuracy': accuracy_in_bin,
                'confidence': avg_confidence_in_bin,
                'fraction': prop_in_bin
            })
            
    return ece, bin_stats

def plot_reliability_diagram(ax_top, ax_bottom, bin_stats, title, ece):
    """Plots the exact stacked reliability diagram from the NeurIPS paper."""
    centers = [b['center'] for b in bin_stats]
    widths = [b['width'] for b in bin_stats]
    accuracies = [b['accuracy'] for b in bin_stats]
    confidences = [b['confidence'] for b in bin_stats]
    fractions = [b['fraction'] for b in bin_stats]
    
    # --- Top Plot: Sample Fraction Histogram ---
    ax_top.bar(centers, fractions, width=widths, color='royalblue', edgecolor='black')
    ax_top.set_xlim([0, 1])
    ax_top.set_ylim([0, 1.0])
    ax_top.set_title(f"{title}\nECE: {ece:.4f}", fontsize=20)
    ax_top.grid(True, axis='y', linestyle='--', alpha=0.6)
    
    # --- Bottom Plot: Accuracy vs Confidence ---
    # 1. Perfect calibration line
    ax_bottom.plot([0, 1], [0, 1], color='gray', linewidth=2, zorder=1)
    
    # 2. Accuracy bars (Solid Blue)
    ax_bottom.bar(centers, accuracies, width=widths, color='royalblue', edgecolor='black', zorder=2)
    
    # 3. Confidence gap bars (Red hollow boxes)
    ax_bottom.bar(centers, confidences, width=widths, color='none', edgecolor='lightcoral', linewidth=1.5, zorder=3)
    
    ax_bottom.set_xlim([0, 1])
    ax_bottom.set_ylim([0, 1])
    ax_bottom.set_xlabel("Confidence\n(unscaled)", fontsize=20)
    ax_bottom.grid(True, linestyle='--', alpha=0.6)

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Directories for your models
    DIR_MODEL_1 = "./final/model_lr5e-05_clip10.0_seed21"
    DIR_MODEL_2 = "./final/model_lr1e-05_clip50.0_seed21"
    DIR_MODEL_3 = "./final/model_lr1e-06_clip0.0_seed21"
    
    labels = [
        get_model_label(DIR_MODEL_1, "w1"),
        get_model_label(DIR_MODEL_2, "w2"),
        get_model_label(DIR_MODEL_3, "w3")
    ]
    
    task_name = "SNLI"
    model_name = "roberta-large"
    num_labels = num_labels_mapping.get(task_name, 3) 

    print("Setting up Native DPZero Architecture & REAL SNLI Data...")
    model_args = ModelArguments(model_name_or_path=model_name, few_shot_type="prompt")
    data_args = DynamicDataTrainingArguments(
        task_name=task_name, data_dir=f"data/original/{task_name}", 
        max_seq_length=256, prompt=True, template="*cls**sent-_0*?*mask*,*+sentl_1**sep+*",
        mapping="{'contradiction':'No','entailment':'Yes','neutral':'Maybe'}", num_k=16
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.model_type = "roberta"
    
    # Evaluate Calibration on the TEST set
    test_dataset = FewShotDataset(data_args, tokenizer=tokenizer, mode="test", use_demo=False)
    collator = MyDataCollatorWithPadding(tokenizer)
    test_dataloader = DataLoader(test_dataset, batch_size=16, collate_fn=collator, shuffle=False)
    
    config = RobertaConfig.from_pretrained(model_name, num_labels=num_labels, finetuning_task=task_name, dpzero=True, num_k=data_args.num_k)
    model = MODEL_TYPES[config.model_type].from_pretrained(model_name, config=config).to(device)
    
    class MockModelArgs(ModelArguments):
        def __getattr__(self, name): return False
            
    model.model_args = MockModelArgs(model_name_or_path=model_name)
    model.data_args = data_args
    model.tokenizer = tokenizer
    model.label_word_list = torch.tensor(test_dataset.label_word_list).long().to(device)

    print("Loading Checkpoint Weights...")
    sd1 = load_weights(DIR_MODEL_1, device)
    sd2 = load_weights(DIR_MODEL_2, device)
    sd3 = load_weights(DIR_MODEL_3, device)
    
    # =========================================================================
    # PART 1: GATHER SNLI LOGITS FOR ALL MODELS
    # =========================================================================
    print("Extracting Logits for Model 1...")
    model.load_state_dict(sd1, strict=False)
    logits_m1, true_labels = get_all_logits(model, test_dataloader, device)
    
    print("Extracting Logits for Model 2...")
    model.load_state_dict(sd2, strict=False)
    logits_m2, _ = get_all_logits(model, test_dataloader, device)
    
    print("Extracting Logits for Model 3...")
    model.load_state_dict(sd3, strict=False)
    logits_m3, _ = get_all_logits(model, test_dataloader, device)

    # =========================================================================
    # PART 2: COMPUTE CALIBRATION AND ECE
    # =========================================================================
    ece_m1, stats_m1 = compute_calibration(logits_m1, true_labels, num_bins=15)
    ece_m2, stats_m2 = compute_calibration(logits_m2, true_labels, num_bins=15)
    ece_m3, stats_m3 = compute_calibration(logits_m3, true_labels, num_bins=15)

    # =========================================================================
    # PART 3: GENERATE THE RELIABILITY DIAGRAMS (1x3 Grid)
    # =========================================================================
    print("Generating Reliability Diagrams...")
    
    # 2 Rows (Histogram, Reliability), 3 Columns (Models)
    fig, axes = plt.subplots(2, 3, figsize=(12, 6), gridspec_kw={'height_ratios': [1, 2]}, sharex=True, sharey='row')
    
    # Plot Model 1
    plot_reliability_diagram(axes[0, 0], axes[1, 0], stats_m1, labels[0], ece_m1)
    # Plot Model 2
    plot_reliability_diagram(axes[0, 1], axes[1, 1], stats_m2, labels[1], ece_m2)
    # Plot Model 3
    plot_reliability_diagram(axes[0, 2], axes[1, 2], stats_m3, labels[2], ece_m3)
    
    # Setup axis labels for the leftmost column only
    axes[0, 0].set_ylabel("Sample frac.", fontsize=24)
    axes[1, 0].set_ylabel("Accuracy", fontsize=24)
    
    plt.tight_layout()
    output_filename = "snli_calibration_comparison_2.png"
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    print(f"Done! Calibration diagrams saved to {output_filename}")

if __name__ == "__main__":
    main()