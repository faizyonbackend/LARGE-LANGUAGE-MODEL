# train_gpt2_medium_final.py
import os
import re
import time
import json
import ssl
import urllib.request
from functools import partial

import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

import tiktoken
from typing import List

# --- IMPORTS FROM YOUR CODEBASE (expect these to exist like in your earlier script) ---
from main import (
    evaluate_model,
    generate_and_print_sample,
    token_ids_to_text,
    text_to_token_ids,
    generate,
    GPTModel,
    load_weights_into_gpt,
)
from gpt_download3 import download_and_load_gpt2

# Optional: LoRA (PEFT)
USE_LORA = False  # set True to use LoRA (requires 'peft' installed)
try:
    if USE_LORA:
        from peft import LoraConfig, get_peft_model
except Exception:
    USE_LORA = False

# ------------------- Config -------------------
SEED = 123
torch.manual_seed(SEED)

CHOOSE_MODEL = "gpt2-medium (355M)"  # must match keys in model_configs below
MODELS_DIR = "gpt2"
CHECKPOINT_DIR = "./checkpoints"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

BASE_CONFIG = {
    "vocab_size": 50257,
    "context_length": 1024,
    "drop_rate": 0.0,
    "qkv_bias": True,
}

model_configs = {
    "gpt2-small (124M)": {"emb_dim": 768, "n_layers": 12, "n_heads": 12},
    "gpt2-medium (355M)": {"emb_dim": 1024, "n_layers": 24, "n_heads": 16},
    "gpt2-large (774M)": {"emb_dim": 1280, "n_layers": 36, "n_heads": 20},
    "gpt2-xl (1558M)": {"emb_dim": 1600, "n_layers": 48, "n_heads": 25},
}

BASE_CONFIG.update(model_configs[CHOOSE_MODEL])

# Hyperparameters
BATCH_SIZE = 4                 # reduce if VRAM limited
NUM_EPOCHS = 3
LEARNING_RATE = 5e-05
WEIGHT_DECAY = 0.01
GRAD_CLIP_NORM = 1.0
EVAL_FREQ_STEPS = 200          # run eval every N steps (set to small during testing)
EVAL_ITER = 50
SAVE_EVERY_EPOCH = True
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ALLOW_MAX_LENGTH = 1024
PAD_TOKEN_ID = 50256           # GPT-2 endoftext
IGNORE_INDEX = -100

# Dataset URL (Raschka example)
DATA_FILE = "instruction-data.json"
DATA_URL = (
    "https://raw.githubusercontent.com/rasbt/LLMs-from-scratch"
    "/main/ch07/01_main-chapter-code/instruction-data.json"
)

# ------------------- Utilities -------------------
def download_and_load_file(file_path: str, url: str):
    ssl_context = ssl.create_default_context()
    ssl_context.check_hostname = False
    ssl_context.verify_mode = ssl.CERT_NONE

    if not os.path.exists(file_path):
        with urllib.request.urlopen(url, context=ssl_context) as response:
            text_data = response.read().decode("utf-8")
        with open(file_path, "w", encoding="utf-8") as file:
            file.write(text_data)

    with open(file_path, "r", encoding="utf-8") as file:
        data = json.load(file)

    return data

def format_input(entry: dict) -> str:
    instruction_text = (
        f"Below is an instruction that describes a task. "
        f"Write a response that appropriately completes the request."
        f"\n\n### Instruction:\n{entry['instruction']}"
    )
    input_text = f"\n\n### Input:\n{entry['input']}" if entry["input"] else ""
    return instruction_text + input_text

# ------------------- Dataset -------------------
class InstructionDataset(Dataset):
    def __init__(self, data: List[dict], tokenizer):
        self.data = data
        self.tokenizer = tokenizer
        self.encoded_texts = []
        for entry in data:
            instruction_plus_input = format_input(entry)
            response_text = f"\n\n### Response:\n{entry['output']}"
            full_text = instruction_plus_input + response_text
            self.encoded_texts.append(self.tokenizer.encode(full_text))

    def __getitem__(self, index):
        return self.encoded_texts[index]

    def __len__(self):
        return len(self.data)

# ------------------- Collate -------------------
def custom_collate_fn(batch, pad_token_id=PAD_TOKEN_ID, ignore_index=IGNORE_INDEX,
                      allowed_max_length=None, device="cpu"):
    batch_max_length = max(len(item) + 1 for item in batch)
    inputs_lst, targets_lst = [], []
    for item in batch:
        new_item = list(item)[:]  # copy
        new_item += [pad_token_id]
        padded = new_item + [pad_token_id] * (batch_max_length - len(new_item))
        inputs = torch.tensor(padded[:-1], dtype=torch.long)
        targets = torch.tensor(padded[1:], dtype=torch.long)
        # mask targets: keep first padding occurrence, replace rest with ignore_index
        mask = targets == pad_token_id
        if mask.any():
            idxs = torch.nonzero(mask).squeeze()
            if idxs.dim() == 0:
                idxs = idxs.unsqueeze(0)
            if idxs.numel() > 1:
                targets[idxs[1:]] = ignore_index
        if allowed_max_length is not None:
            inputs = inputs[:allowed_max_length]
            targets = targets[:allowed_max_length]
        inputs_lst.append(inputs)
        targets_lst.append(targets)
    inputs_tensor = torch.stack(inputs_lst).to(device)
    targets_tensor = torch.stack(targets_lst).to(device)
    return inputs_tensor, targets_tensor

# ------------------- Load dataset -------------------
data = download_and_load_file(DATA_FILE, DATA_URL)
train_portion = int(len(data) * 0.85)
test_portion = int(len(data) * 0.1)
val_portion = len(data) - train_portion - test_portion
train_data = data[:train_portion]
test_data = data[train_portion:train_portion + test_portion]
val_data = data[train_portion + test_portion:]

tokenizer = tiktoken.get_encoding("gpt2")  # same as earlier

# Dataloaders
num_workers = 0
customized_collate = partial(custom_collate_fn, device=DEVICE, allowed_max_length=ALLOW_MAX_LENGTH)

train_dataset = InstructionDataset(train_data, tokenizer)
val_dataset = InstructionDataset(val_data, tokenizer)
test_dataset = InstructionDataset(test_data, tokenizer)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                          collate_fn=customized_collate, drop_last=True, num_workers=num_workers)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False,
                        collate_fn=customized_collate, drop_last=False, num_workers=num_workers)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False,
                         collate_fn=customized_collate, drop_last=False, num_workers=num_workers)

# ------------------- Model init & load weights -------------------
# Download base params (if needed) and load pretrained weights into model structure
model_size = CHOOSE_MODEL.split(" ")[-1].lstrip("(").rstrip(")")
settings, params = download_and_load_gpt2(model_size=model_size, models_dir=MODELS_DIR)

model = GPTModel(BASE_CONFIG)
# First load pretrained weights into model (base)
load_weights_into_gpt(model, params)

# Optional: if you have an existing fine-tuned checkpoint and want to resume, you'll load it later.

# Move model to device
model.to(DEVICE)
model.eval()

# Optional LoRA wrap (if USE_LORA True and peft available)
if USE_LORA:
    # NOTE: target_modules must match module names inside GPTModel.
    # Adjust target_modules to actual attribute names (example placeholders).
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["c_attn", "c_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, lora_config)
    model.to(DEVICE)

# ------------------- Loss helpers -------------------
def calc_loss_batch(input_batch, target_batch, model, device):
    input_batch = input_batch.to(device)
    target_batch = target_batch.to(device)
    logits = model(input_batch)  # expecting shape (B, T, V)
    V = logits.size(-1)
    loss = F.cross_entropy(logits.view(-1, V), target_batch.view(-1), ignore_index=IGNORE_INDEX)
    return loss

def calc_loss_loader(data_loader, model, device, num_batches=None):
    model.eval()
    total_loss = 0.0
    count = 0
    if len(data_loader) == 0:
        return float("nan")
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if num_batches is not None and i >= num_batches:
            break
        with torch.no_grad():
            loss = calc_loss_batch(input_batch, target_batch, model, device)
        total_loss += loss.item()
        count += 1
    return total_loss / count if count > 0 else float("nan")

# ------------------- Training loop (resumable) -------------------
def save_checkpoint(state: dict, filename: str):
    torch.save(state, filename)

def load_checkpoint_if_exists(model, optimizer, checkpoint_path):
    if checkpoint_path and os.path.exists(checkpoint_path):
        print(f"Resuming from checkpoint: {checkpoint_path}")
        chk = torch.load(checkpoint_path, map_location=DEVICE)
        model.load_state_dict(chk["model_state_dict"])
        optimizer.load_state_dict(chk["optimizer_state_dict"])
        start_epoch = chk.get("epoch", 0) + 1
        global_step = chk.get("global_step", 0)
        return start_epoch, global_step
    return 0, 0

def train(model, train_loader, val_loader, optimizer, device, num_epochs,
          eval_freq_steps, eval_iter, start_context, tokenizer, resume_checkpoint=None):
    model.train()
    scaler = torch.cuda.amp.GradScaler() if DEVICE.type == "cuda" else None

    start_epoch, global_step = load_checkpoint_if_exists(model, optimizer, resume_checkpoint)
    tokens_seen = 0
    train_losses, val_losses, track_tokens_seen = [], [], []

    for epoch in range(start_epoch, num_epochs):
        model.train()
        epoch_start = time.time()
        for batch_idx, (input_batch, target_batch) in enumerate(train_loader):
            optimizer.zero_grad()
            if scaler is not None:
                with torch.cuda.amp.autocast():
                    loss = calc_loss_batch(input_batch, target_batch, model, device)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss = calc_loss_batch(input_batch, target_batch, model, device)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
                optimizer.step()

            tokens_seen += input_batch.numel()
            global_step += 1

            if global_step % eval_freq_steps == 0:
                # run quick eval
                train_loss = calc_loss_loader(train_loader, model, device, num_batches=eval_iter)
                val_loss = calc_loss_loader(val_loader, model, device, num_batches=eval_iter)
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                track_tokens_seen.append(tokens_seen)
                print(f"[Epoch {epoch+1}] Step {global_step}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")

        # end epoch operations
        epoch_time = (time.time() - epoch_start) / 60.0
        print(f"Epoch {epoch+1} finished in {epoch_time:.2f} minutes.")

        # generate sample
        generate_and_print_sample(model, tokenizer, device, start_context)

        # save checkpoint (model + optimizer state)
        ckpt_path = os.path.join(CHECKPOINT_DIR, f"checkpoint_epoch{epoch+1}.pth")
        save_checkpoint({
            "epoch": epoch,
            "global_step": global_step,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict()
        }, ckpt_path)
        print(f"Saved checkpoint: {ckpt_path}")

    return train_losses, val_losses, track_tokens_seen

# ------------------- Prepare optimizer -------------------
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

# Optionally resume from a specific checkpoint file by setting resume_checkpoint variable
resume_checkpoint = None  # e.g., "./checkpoints/checkpoint_epoch1.pth"

# ------------------- Run training -------------------
start_context = format_input(val_data[0])
start_time = time.time()

train_losses, val_losses, tokens_seen = train(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    optimizer=optimizer,
    device=DEVICE,
    num_epochs=NUM_EPOCHS,
    eval_freq_steps=EVAL_FREQ_STEPS,
    eval_iter=EVAL_ITER,
    start_context=start_context,
    tokenizer=tokenizer,
    resume_checkpoint=resume_checkpoint
)

end_time = time.time()
print(f"Training completed in {(end_time - start_time) / 60.0:.2f} minutes.")

# ------------------- Save final model -------------------
safe_name = re.sub(r"[ ()]", "", CHOOSE_MODEL)
final_path = f"{safe_name}-sft-final.pth"
torch.save(model.state_dict(), final_path)
print(f"Final state_dict saved to: {final_path}")

# ------------------- Quick test generation on a few test examples -------------------
model.eval()
with torch.no_grad():
    for entry in test_data[:3]:
        input_text = format_input(entry)
        idxs = text_to_token_ids(input_text, tokenizer)
        idxs = idxs.to(DEVICE)
        token_ids = generate(
            model=model,
            idx=idxs,
            max_new_tokens=256,
            context_size=BASE_CONFIG["context_length"],
            eos_id=PAD_TOKEN_ID
        )
        generated_text = token_ids_to_text(token_ids, tokenizer)
        response_text = re.sub(r"### Response:\s*", "", generated_text[len(input_text):]).strip()
        print("PROMPT:\n", input_text)
        print("\nTARGET:\n", entry["output"])
        print("\nMODEL:\n", response_text)
        print("----------------------------------------------------")
