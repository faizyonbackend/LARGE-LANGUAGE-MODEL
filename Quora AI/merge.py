# merge_lora_to_full_model.py
import os
import torch
from peft import PeftModel, LoraConfig, get_peft_model
from main import GPTModel, load_weights_into_gpt
from gpt_download3 import download_and_load_gpt2

# ------------------- CONFIG -------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CKPT_DIR = "./checkpoints"         # folder containing LoRA checkpoints
OUTPUT_PATH = "./gpt2-medium-lora-merged.pth"

# ------------------- BASE MODEL -------------------
BASE_CONFIG = {
    "vocab_size": 50257,
    "context_length": 1024,  # match your training setup
    "drop_rate": 0.0,
    "qkv_bias": True,
    "emb_dim": 1024,
    "n_layers": 24,
    "n_heads": 16,
}

print("Loading GPT-2 medium base model...")
settings, params = download_and_load_gpt2("355M", "gpt2")
base_model = GPTModel(BASE_CONFIG)
load_weights_into_gpt(base_model, params)
base_model.to(DEVICE)

# ------------------- LoRA CONFIG -------------------
lora_cfg = LoraConfig(
    r=4,
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none",
    target_modules=["c_attn", "c_proj"],  # ensure these exist in your GPTModel
    task_type="CAUSAL_LM",
)

# ------------------- MERGE PROCESS -------------------
# Find the most recent LoRA checkpoint
lora_ckpts = [f for f in os.listdir(CKPT_DIR) if f.endswith(".pth")]
if not lora_ckpts:
    raise FileNotFoundError(f"No LoRA checkpoints found in {CKPT_DIR}")
last_ckpt = sorted(lora_ckpts)[-1]
lora_path = os.path.join(CKPT_DIR, last_ckpt)
print(f"Loading LoRA adapters from: {lora_path}")

# Attach LoRA adapters and load weights
peft_model = get_peft_model(base_model, lora_cfg)
peft_model = PeftModel.from_pretrained(peft_model, lora_path)
peft_model.to(DEVICE)

# Merge LoRA weights into base model
print("Merging LoRA weights into base GPT-2 model...")
merged_model = peft_model.merge_and_unload()  # merges and detaches adapters

# ------------------- SAVE FULL MODEL -------------------
torch.save(merged_model.state_dict(), OUTPUT_PATH)
print(f"✅ Full fine-tuned GPT-2 model (merged) saved at: {OUTPUT_PATH}")
