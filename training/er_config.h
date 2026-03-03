// er_config.h — Config for ER extraction fine-tuning
// Same model architecture as stories_config.h (dim=768, 12 layers, vocab=32k)
// but SEQ=512 to fit instruction+output pairs, plus ER-specific data structures.
#pragma once

// Pull in base config first (defines SEQ=256 and everything else), then override.
#include "stories_config.h"
#undef SEQ
#define SEQ 512  // Longer context: instruction (≤256) + entity/relation output (≤200)
#undef ACCUM_STEPS
#define ACCUM_STEPS 50  // 50 steps/batch → ~100 exec() restarts for 5000 steps (vs 500)

// ── ER dataset format ────────────────────────────────────────────────────────
// Binary file written by data_gen/tokenize_er.py
//   Header:  [magic: u32][n_examples: u32]
//   Records: [input_len: u16][total_len: u16][tokens: u16 × total_len] ...
//
// input_len  = # instruction+context tokens  (no loss computed here)
// total_len  = input_len + output_len
// Loss is computed at positions [input_len-1 .. total_len-2]  (predict output tokens)
#define ER_MAGIC 0x45525F31u  // "ER_1"

typedef struct {
    uint16_t input_len;   // length of input (instruction + text) portion
    uint16_t total_len;   // input_len + output_len
    uint16_t *tokens;     // [total_len] token ids, heap-allocated
} ErExample;

typedef struct {
    ErExample *examples;  // heap array [n]
    int        n;
} ErDataset;

// ── Dataset I/O ──────────────────────────────────────────────────────────────

static ErDataset er_dataset_load(const char *path) {
    ErDataset ds = {NULL, 0};
    FILE *f = fopen(path, "rb");
    if (!f) { printf("Cannot open %s\n", path); return ds; }

    uint32_t magic, n;
    fread(&magic, 4, 1, f);
    fread(&n,     4, 1, f);
    if (magic != ER_MAGIC) {
        printf("Bad magic in %s: expected 0x%08X got 0x%08X\n", path, ER_MAGIC, magic);
        fclose(f); return ds;
    }

    ErExample *examples = (ErExample*)malloc(n * sizeof(ErExample));
    for (uint32_t i = 0; i < n; i++) {
        uint16_t input_len, total_len;
        fread(&input_len, 2, 1, f);
        fread(&total_len, 2, 1, f);
        examples[i].input_len = input_len;
        examples[i].total_len = total_len;
        examples[i].tokens    = (uint16_t*)malloc(total_len * 2);
        fread(examples[i].tokens, 2, total_len, f);
    }
    fclose(f);

    ds.examples = examples;
    ds.n        = (int)n;
    printf("Loaded ER dataset: %d examples from %s\n", ds.n, path);
    return ds;
}

static void er_dataset_free(ErDataset *ds) {
    for (int i = 0; i < ds->n; i++) free(ds->examples[i].tokens);
    free(ds->examples);
    ds->examples = NULL; ds->n = 0;
}

// Fill a SEQ-length token buffer from one example, pad remainder with EOS (2).
// Returns loss_start: first position where loss is computed.
// Returns loss_end:   last position where loss is computed (inclusive).
//
//   seq_in[t]  = token fed to model at step t  (tokens[0..total-2], then pad)
//   seq_tgt[t] = target token at step t         (tokens[1..total-1], then pad)
//   loss on positions [loss_start .. loss_end] only
static int er_fill_buffers(const ErExample *ex,
                            uint16_t *seq_in,   // [SEQ] model input
                            uint16_t *seq_tgt,  // [SEQ] target labels
                            int *loss_end_out)
{
    int total = ex->total_len;
    int eos   = 2;  // LLaMA EOS token id

    // seq_in = tokens[0..total-2], padded with EOS
    for (int t = 0; t < SEQ; t++) {
        seq_in[t]  = (t   < total - 1) ? ex->tokens[t]     : (uint16_t)eos;
        seq_tgt[t] = (t+1 < total)     ? ex->tokens[t + 1] : (uint16_t)eos;
    }

    // Loss starts where output begins: position input_len-1 predicts output_tok_0
    int loss_start = ex->input_len - 1;
    int loss_end   = total - 2;  // last output token target position
    if (loss_end   >= SEQ) loss_end = SEQ - 1;

    *loss_end_out = loss_end;
    return loss_start;
}

// ── ER checkpoint header ─────────────────────────────────────────────────────

typedef struct {
    uint32_t magic;        // 0x45524348 "ERCH"
    uint32_t version;      // 1
    int32_t  step;
    int32_t  total_steps;
    int32_t  n_layers;
    int32_t  vocab_size;
    int32_t  dim;
    int32_t  seq_len;
    float    lr;
    float    loss;
    double   cum_compile;
    double   cum_train;
    double   cum_wall;
    int32_t  cum_steps;
    int32_t  cum_batches;
    int32_t  adam_t;
    int32_t  pad[3];
} ErCkptHdr;

#define ER_CKPT_MAGIC 0x45524348u

// ── Masked cross-entropy ─────────────────────────────────────────────────────
// Like cross_entropy_loss() in stories_cpu_ops.h but only computes loss and
// sets gradients at positions [loss_start .. loss_end] (inclusive).
// Positions outside that range get dlogits=0.
// Returns mean CE loss over the output positions only.
static float masked_cross_entropy_loss(float *dlogits, const float *logits,
                                        const uint16_t *targets,
                                        int V, int S,
                                        int loss_start, int loss_end)
{
    float *buf = (float*)malloc(S * V * 4);
    vDSP_mtrans(logits, 1, buf, 1, (vDSP_Length)S, (vDSP_Length)V);

    int n_loss = loss_end - loss_start + 1;
    float total_loss = 0;
    float invN = 1.0f / n_loss;

    for (int t = 0; t < S; t++) {
        float *row = buf + t * V;
        float maxv; vDSP_maxv(row, 1, &maxv, (vDSP_Length)V);
        float neg_max = -maxv; vDSP_vsadd(row, 1, &neg_max, row, 1, (vDSP_Length)V);
        int n = V; vvexpf(row, row, &n);
        float sum; vDSP_sve(row, 1, &sum, (vDSP_Length)V);
        float inv_sum = 1.0f / sum; vDSP_vsmul(row, 1, &inv_sum, row, 1, (vDSP_Length)V);

        if (t >= loss_start && t <= loss_end) {
            int tgt = targets[t];
            total_loss -= logf(row[tgt] + 1e-10f);
            row[tgt] -= 1.0f;
            vDSP_vsmul(row, 1, &invN, row, 1, (vDSP_Length)V);
        } else {
            // Zero out gradient for masked positions
            memset(row, 0, V * 4);
        }
    }

    vDSP_mtrans(buf, 1, dlogits, 1, (vDSP_Length)V, (vDSP_Length)S);
    free(buf);
    return total_loss / n_loss;
}

