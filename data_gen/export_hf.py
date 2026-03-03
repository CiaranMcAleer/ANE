#!/usr/bin/env python3
"""
export_hf.py — Export ANE ER-extraction checkpoint to HuggingFace format.

Reads:   ../training/er_ckpt.bin  (custom binary from er_train.m)
Writes:  ./hf_export/
            config.json
            model.safetensors   (weights only, no optimizer state)
            generation_config.json
            tokenizer_config.json
            special_tokens_map.json
            README.md           (model card)

The exported model loads directly with:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    model = AutoModelForCausalLM.from_pretrained("./hf_export")
    tok   = AutoTokenizer.from_pretrained("./hf_export")

Usage:
    uv run python3 export_hf.py
    uv run python3 export_hf.py --checkpoint ../training/er_ckpt.bin
    uv run python3 export_hf.py --fp16          # export in float16 (~210MB)
    uv run python3 export_hf.py --output ./my_model
"""

import argparse
import json
import struct
import sys
from pathlib import Path

import numpy as np
from safetensors.numpy import save_file

# ── Model constants (must match er_config.h / stories_config.h) ──────────────
DIM    = 768
HIDDEN = 2048
HEADS  = 12
NLAYERS= 12
VOCAB  = 32000
HD     = DIM // HEADS   # 64

WQ_SZ  = DIM * DIM      # 589_824
WO_SZ  = DIM * DIM
W1_SZ  = HIDDEN * DIM   # 1_572_864
W2_SZ  = DIM * HIDDEN
W3_SZ  = HIDDEN * DIM

ER_MAGIC   = 0x45524348  # "ERCH" — er_train.m
STOR_MAGIC = 0x424C5A54  # "BLZT" — train_large.m

# ── Checkpoint reader ─────────────────────────────────────────────────────────

def _skip(f, n_floats):
    f.read(n_floats * 4)

def _read_f32(f, n) -> np.ndarray:
    return np.frombuffer(f.read(n * 4), dtype=np.float32).copy()

def read_checkpoint(path: Path) -> tuple[dict[str, np.ndarray], dict]:
    """
    Read er_ckpt.bin (or train_large ckpt).
    Returns (weights_dict, metadata).

    Weight layout on disk per layer:
        Wq Wk Wv Wo W1 W2 W3 rms_att rms_ffn   ← actual weights
        Wq_m Wq_v  Wk_m Wk_v  Wv_m Wv_v        ← Adam m/v (skipped)
        Wo_m Wo_v  W1_m W1_v  W2_m W2_v
        W3_m W3_v  rms_att_m rms_att_v  rms_ffn_m rms_ffn_v
    Then: rms_final rms_final_m rms_final_v embed embed_m embed_v
    """
    with open(path, "rb") as f:

        # ── Header ────────────────────────────────────────────────────────────
        magic, version = struct.unpack("<II", f.read(8))

        if magic == ER_MAGIC:
            # ErCkptHdr: step total_steps n_layers vocab_size dim seq_len  (6×i32)
            #            lr loss  (2×f32)   cum_compile cum_train cum_wall  (3×f64)
            #            cum_steps cum_batches adam_t  (3×i32)  pad[3]  (3×i32)
            step, total_steps, n_layers, vocab_size, dim, seq_len = \
                struct.unpack("<iiiiii", f.read(24))
            lr, loss = struct.unpack("<ff", f.read(8))
            f.read(24)   # 3 doubles
            f.read(24)   # 6 ints (cum_steps…pad[3])
            fmt = "ER"

        elif magic == STOR_MAGIC:
            # CkptHdr (train_large.m): step total_steps n_layers vocab_size
            #         dim hidden_dim n_heads seq_len  (6 more ints = 8 total)
            #         lr loss  cum_compile cum_train cum_wall  cum_steps cum_batches adam_t  pad[3]
            step, total_steps, n_layers, vocab_size, dim, hidden_dim, n_heads, seq_len = \
                struct.unpack("<iiiiiiii", f.read(32))
            lr, loss = struct.unpack("<ff", f.read(8))
            f.read(24)   # 3 doubles
            f.read(24)   # 6 ints
            fmt = "STORIES"
        else:
            raise ValueError(f"Unknown checkpoint magic: {magic:#010x}")

        meta = {"step": step, "loss": loss, "lr": lr, "format": fmt,
                "n_layers": n_layers, "dim": dim, "vocab": vocab_size}
        print(f"Checkpoint: {fmt} format  step={step}  loss={loss:.4f}  lr={lr:.2e}")

        # ── Layer weights ─────────────────────────────────────────────────────
        tensors: dict[str, np.ndarray] = {}

        for L in range(n_layers):
            p = f"model.layers.{L}"

            # --- Weights ---
            wq      = _read_f32(f, WQ_SZ).reshape(DIM, DIM)
            wk      = _read_f32(f, WQ_SZ).reshape(DIM, DIM)
            wv      = _read_f32(f, WQ_SZ).reshape(DIM, DIM)
            wo      = _read_f32(f, WO_SZ).reshape(DIM, DIM)
            w1      = _read_f32(f, W1_SZ).reshape(HIDDEN, DIM)
            w2      = _read_f32(f, W2_SZ).reshape(DIM, HIDDEN)
            w3      = _read_f32(f, W3_SZ).reshape(HIDDEN, DIM)
            rms_att = _read_f32(f, DIM)
            rms_ffn = _read_f32(f, DIM)

            # --- Skip Adam state (m+v for each of the 9 weight arrays) ---
            for sz in [WQ_SZ, WQ_SZ, WQ_SZ, WO_SZ, W1_SZ, W2_SZ, W3_SZ, DIM, DIM]:
                _skip(f, sz * 2)  # m and v

            # --- Map to HuggingFace tensor names ---
            # Our layout:  output = W @ X  (W is [out, in], X is [in, seq])
            # HF layout:   output = X @ W.T (W is [out, in])
            # These are identical — no transposition needed.
            tensors[f"{p}.self_attn.q_proj.weight"] = wq
            tensors[f"{p}.self_attn.k_proj.weight"] = wk
            tensors[f"{p}.self_attn.v_proj.weight"] = wv
            tensors[f"{p}.self_attn.o_proj.weight"] = wo
            tensors[f"{p}.mlp.gate_proj.weight"]    = w1   # W1 = gate (SiLU branch)
            tensors[f"{p}.mlp.down_proj.weight"]    = w2   # W2 = down projection
            tensors[f"{p}.mlp.up_proj.weight"]      = w3   # W3 = up (multiply branch)
            tensors[f"{p}.input_layernorm.weight"]          = rms_att
            tensors[f"{p}.post_attention_layernorm.weight"] = rms_ffn

        # ── Final norm + embedding ────────────────────────────────────────────
        rms_final = _read_f32(f, DIM)
        _skip(f, DIM * 2)   # rms_final Adam m/v

        embed = _read_f32(f, VOCAB * DIM).reshape(VOCAB, DIM)
        # aembed m/v remain in file but we don't need them

        tensors["model.norm.weight"]        = rms_final
        tensors["model.embed_tokens.weight"]= embed
        tensors["lm_head.weight"]           = embed   # tied (shared pointer is fine in safetensors)

    return tensors, meta


# ── Config generation ─────────────────────────────────────────────────────────

def make_config(meta: dict) -> dict:
    return {
        "architectures":           ["LlamaForCausalLM"],
        "model_type":              "llama",
        "bos_token_id":            1,
        "eos_token_id":            2,
        "pad_token_id":            0,
        "hidden_size":             DIM,
        "intermediate_size":       HIDDEN,
        "num_hidden_layers":       NLAYERS,
        "num_attention_heads":     HEADS,
        "num_key_value_heads":     HEADS,   # MHA (not GQA)
        "head_dim":                HD,
        "hidden_act":              "silu",
        "max_position_embeddings": 512,
        "vocab_size":              VOCAB,
        "rms_norm_eps":            1e-5,
        # No RoPE was applied during ANE training (the MIL attention kernels go
        # directly from Q/K projection to Q@K^T without any rotation).
        # Setting rope_theta=1e10 makes rotation angles ≈ 0 for all sequence
        # positions ≤ 512, effectively reproducing the no-RoPE training condition.
        "rope_theta":              1e10,
        "rope_scaling":            None,
        "tie_word_embeddings":     True,
        "torch_dtype":             "float32",
        "transformers_version":    "4.40.0",
        "use_cache":               True,
        "_training_step":          meta["step"],
        "_training_loss":          round(float(meta["loss"]), 4),
    }


def make_generation_config() -> dict:
    return {
        "bos_token_id": 1,
        "eos_token_id": 2,
        "max_new_tokens": 256,
        "do_sample": False,
        "transformers_version": "4.40.0",
    }


def make_tokenizer_config() -> dict:
    return {
        "bos_token": "<s>",
        "eos_token": "</s>",
        "model_max_length": 512,
        "tokenizer_class": "LlamaTokenizer",
        "legacy": False,
        # Point to the community LLaMA tokenizer — downloads automatically
        # when loading via: AutoTokenizer.from_pretrained("./hf_export")
        # (tokenizer.model is ~488KB, no model weights downloaded)
        "tokenizer_model": "huggyllama/llama-7b",
    }


def make_special_tokens_map() -> dict:
    return {
        "bos_token": "<s>",
        "eos_token": "</s>",
        "unk_token": "<unk>",
    }


MODEL_CARD_TEMPLATE = """\
---
language: en
license: mit
tags:
  - entity-extraction
  - relation-extraction
  - information-extraction
  - llama
  - ane
---

# ER Extraction — ANE-Trained LLaMA (110M)

Entity and relationship extraction model trained directly on Apple's Neural Engine
via reverse-engineered private APIs. Architecture is LLaMA-style (110M parameters).

## Model details

| | |
|---|---|
| Architecture | LLaMA (decoder-only transformer) |
| Parameters | 109.5M |
| Layers | 12 |
| Hidden size | 768 |
| FFN size | 2048 |
| Attention heads | 12 |
| Vocabulary | 32,000 (LLaMA SentencePiece) |
| Max sequence | 512 tokens |
| Trained on | Apple M4 Neural Engine |

## Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("path/to/this/model")
tok   = AutoTokenizer.from_pretrained("huggyllama/llama-7b")

text = "Alice is a senior engineer in the infrastructure team, reporting to Bob."
prompt = f"Extract entities and relations:\\n{text}"

inputs = tok(prompt, return_tensors="pt")
out    = model.generate(**inputs, max_new_tokens=200, do_sample=False)
print(tok.decode(out[0], skip_special_tokens=True))
```

Expected output format:
```
ENTITY: Alice | Person
ENTITY: Bob | Person
ENTITY: infrastructure team | Department
RELATION: Alice | WORKS_IN | infrastructure team
RELATION: Alice | REPORTS_TO | Bob
```

## Training

Trained from random initialisation using a custom ANE training loop
([github.com/m0at/ANE](https://github.com/m0at/ANE)). Synthetic training data
was generated by Qwen2.5-7B (Apache 2.0) via Ollama across 20 diverse domains
including healthcare, legal, finance, and software.

## Technical note on positional encoding

This model was trained **without positional encoding** — the ANE attention kernels
compute Q @ K^T directly without RoPE rotation. The exported checkpoint sets
`rope_theta=1e10`, which makes rotation angles ≈ 0 for all positions ≤ 512,
reproducing the training condition when loaded via standard `transformers`.

## License

MIT
"""


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Export ANE checkpoint to HuggingFace format")
    parser.add_argument("--checkpoint", default="../training/er_ckpt.bin",
                        help="Path to .bin checkpoint")
    parser.add_argument("--output",     default="./hf_export",
                        help="Output directory")
    parser.add_argument("--fp16",       action="store_true",
                        help="Export weights in float16 (halves file size)")
    args = parser.parse_args()

    ckpt_path   = Path(args.checkpoint)
    output_path = Path(__file__).parent / args.output

    if not ckpt_path.exists():
        print(f"ERROR: checkpoint not found: {ckpt_path}", file=sys.stderr)
        print("Run ./er_train first to produce er_ckpt.bin", file=sys.stderr)
        sys.exit(1)

    output_path.mkdir(parents=True, exist_ok=True)
    print(f"Reading checkpoint: {ckpt_path}")

    tensors, meta = read_checkpoint(ckpt_path)

    dtype = np.float16 if args.fp16 else np.float32
    dtype_name = "float16" if args.fp16 else "float32"
    print(f"Exporting {len(tensors)} tensors as {dtype_name}...")

    # Cast if needed
    if args.fp16:
        tensors = {k: v.astype(np.float16) for k, v in tensors.items()}

    # safetensors requires contiguous arrays
    tensors = {k: np.ascontiguousarray(v) for k, v in tensors.items()}

    safetensors_path = output_path / "model.safetensors"
    save_file(tensors, str(safetensors_path))

    total_mb = safetensors_path.stat().st_size / 1e6
    print(f"  model.safetensors: {total_mb:.0f} MB")

    # Config files
    cfg = make_config(meta)
    cfg["torch_dtype"] = dtype_name

    (output_path / "config.json").write_text(json.dumps(cfg, indent=2))
    (output_path / "generation_config.json").write_text(
        json.dumps(make_generation_config(), indent=2))
    (output_path / "tokenizer_config.json").write_text(
        json.dumps(make_tokenizer_config(), indent=2))
    (output_path / "special_tokens_map.json").write_text(
        json.dumps(make_special_tokens_map(), indent=2))
    (output_path / "README.md").write_text(MODEL_CARD_TEMPLATE)

    print(f"\nExported to: {output_path}/")
    for f in sorted(output_path.iterdir()):
        print(f"  {f.name:35s} {f.stat().st_size / 1e6:.2f} MB")

    print(f"""
Next steps:
  # Test locally
  uv run python3 -c "
from transformers import AutoModelForCausalLM, AutoTokenizer
m = AutoModelForCausalLM.from_pretrained('{output_path}')
t = AutoTokenizer.from_pretrained('huggyllama/llama-7b')
print(m)
"

  # Upload to HuggingFace Hub
  huggingface-cli login
  huggingface-cli upload YOUR_USERNAME/er-llama-110m-ane {output_path}

  # Convert to GGUF (for llama.cpp, runs on any hardware without Python)
  git clone https://github.com/ggerganov/llama.cpp
  pip install -r llama.cpp/requirements/requirements-convert_hf_to_gguf.txt
  python llama.cpp/convert_hf_to_gguf.py {output_path} --outtype f16
  # Then quantize: ./llama.cpp/llama-quantize model-f16.gguf model-q4_k_m.gguf Q4_K_M
""")


if __name__ == "__main__":
    main()
