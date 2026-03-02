// er_train.m — Fine-tune Stories110M on ER extraction task (ANE)
//
// Input:  training/er_train.bin   (from data_gen/tokenize_er.py)
// Config: SEQ=512 (fits instruction + entity/relation output)
// Loss:   cross-entropy on output tokens only (input positions masked)
//
// Build:
//   xcrun clang -O2 -Wall -Wno-deprecated-declarations -fobjc-arc \
//     -o er_train er_train.m \
//     -framework Foundation -framework CoreML -framework IOSurface \
//     -framework Accelerate -ldl
//
// Run:
//   ./er_train
//   ./er_train --resume
//   ./er_train --steps 5000 --lr 1e-4

#include "er_config.h"
#include "stories_mil.h"
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunused-function"
#include "stories_cpu_ops.h"   // rmsnorm, adam_update, embed_lookup/backward
#pragma clang diagnostic pop

#define CKPT_PATH "er_ckpt.bin"
#define DATA_PATH "er_train.bin"

// ── Kernel compile helpers (identical to train_large.m) ──────────────────────

static bool compile_layer_kernels(LayerKernels *lk, LayerWeights *w) {
    lk->fwdAttn = compile_kern_mil_w(gen_sdpa_fwd_taps(), (@{
        @"@model_path/weights/rms1.bin": @{@"offset":@0, @"data":build_blob(w->rms_att,1,DIM)},
        @"@model_path/weights/wq.bin":   @{@"offset":@0, @"data":build_blob(w->Wq,DIM,DIM)},
        @"@model_path/weights/wk.bin":   @{@"offset":@0, @"data":build_blob(w->Wk,DIM,DIM)},
        @"@model_path/weights/wv.bin":   @{@"offset":@0, @"data":build_blob(w->Wv,DIM,DIM)},
        @"@model_path/weights/wo.bin":   @{@"offset":@0, @"data":build_blob(w->Wo,DIM,DIM)},
        @"@model_path/weights/mask.bin": @{@"offset":@0, @"data":get_mask_blob()},
    }), DIM*SEQ*2, 6*DIM*SEQ*2);

    lk->fwdFFN = compile_kern_mil_w(gen_ffn_fwd_taps(), (@{
        @"@model_path/weights/rms2.bin": @{@"offset":@0, @"data":build_blob(w->rms_ffn,1,DIM)},
        @"@model_path/weights/w1.bin":   @{@"offset":@0, @"data":build_blob(w->W1,HIDDEN,DIM)},
        @"@model_path/weights/w3.bin":   @{@"offset":@0, @"data":build_blob(w->W3,HIDDEN,DIM)},
        @"@model_path/weights/w2.bin":   @{@"offset":@0, @"data":build_blob(w->W2,DIM,HIDDEN)},
    }), DIM*SEQ*2, (2*DIM+3*HIDDEN)*SEQ*2);

    lk->ffnBwd = compile_kern_mil_w(gen_ffn_bwd(), (@{
        @"@model_path/weights/w2t.bin": @{@"offset":@0, @"data":build_blob_t(w->W2,DIM,HIDDEN)},
        @"@model_path/weights/w1t.bin": @{@"offset":@0, @"data":build_blob_t(w->W1,HIDDEN,DIM)},
        @"@model_path/weights/w3t.bin": @{@"offset":@0, @"data":build_blob_t(w->W3,HIDDEN,DIM)},
    }), (DIM+2*HIDDEN)*SEQ*2, (DIM+2*HIDDEN)*SEQ*2);

    lk->sdpaBwd1 = compile_kern_mil_w(gen_sdpa_bwd1(), (@{
        @"@model_path/weights/mask.bin": @{@"offset":@0, @"data":get_mask_blob()},
        @"@model_path/weights/wot.bin":  @{@"offset":@0, @"data":build_blob_t(w->Wo,DIM,DIM)},
    }), 4*DIM*SEQ*2, (DIM+2*SCORE_CH)*SEQ*2);

    lk->qkvBwd = compile_kern_mil_w(gen_qkvb(), (@{
        @"@model_path/weights/wqt.bin": @{@"offset":@0, @"data":build_blob_t(w->Wq,DIM,DIM)},
        @"@model_path/weights/wkt.bin": @{@"offset":@0, @"data":build_blob_t(w->Wk,DIM,DIM)},
        @"@model_path/weights/wvt.bin": @{@"offset":@0, @"data":build_blob_t(w->Wv,DIM,DIM)},
    }), 3*DIM*SEQ*2, DIM*SEQ*2);

    return lk->fwdAttn && lk->fwdFFN && lk->ffnBwd && lk->sdpaBwd1 && lk->qkvBwd;
}

static Kern *compile_sdpa_bwd2_kern(void) {
    return compile_kern_mil_w(gen_sdpa_bwd2(), @{},
        (2*SCORE_CH+2*DIM)*SEQ*2, 2*DIM*SEQ*2);
}

static void free_layer_kernels(LayerKernels *lk) {
    free_kern(lk->fwdAttn); free_kern(lk->fwdFFN); free_kern(lk->ffnBwd);
    free_kern(lk->sdpaBwd1); free_kern(lk->qkvBwd);
    lk->fwdAttn = lk->fwdFFN = lk->ffnBwd = lk->sdpaBwd1 = lk->qkvBwd = NULL;
}

// ── Checkpoint ───────────────────────────────────────────────────────────────

static void save_checkpoint(const char *path, int step, int total_steps,
                             float lr, float loss, double cc, double ct,
                             double cw, int cs, int cb, int adam_t,
                             LayerWeights *lw, LayerAdam *la,
                             float *rms_final, AdamState *arms_final,
                             float *embed, AdamState *aembed)
{
    FILE *f = fopen(path, "wb");
    ErCkptHdr h = {0};
    h.magic = ER_CKPT_MAGIC; h.version = 1;
    h.step = step; h.total_steps = total_steps;
    h.n_layers = NLAYERS; h.vocab_size = VOCAB;
    h.dim = DIM; h.seq_len = SEQ;
    h.lr = lr; h.loss = loss;
    h.cum_compile = cc; h.cum_train = ct; h.cum_wall = cw;
    h.cum_steps = cs; h.cum_batches = cb; h.adam_t = adam_t;
    fwrite(&h, sizeof(h), 1, f);
    for (int L = 0; L < NLAYERS; L++) {
        fwrite(lw[L].Wq,4,WQ_SZ,f); fwrite(lw[L].Wk,4,WQ_SZ,f);
        fwrite(lw[L].Wv,4,WQ_SZ,f); fwrite(lw[L].Wo,4,WO_SZ,f);
        fwrite(lw[L].W1,4,W1_SZ,f); fwrite(lw[L].W2,4,W2_SZ,f);
        fwrite(lw[L].W3,4,W3_SZ,f);
        fwrite(lw[L].rms_att,4,DIM,f); fwrite(lw[L].rms_ffn,4,DIM,f);
        fwrite(la[L].Wq.m,4,WQ_SZ,f); fwrite(la[L].Wq.v,4,WQ_SZ,f);
        fwrite(la[L].Wk.m,4,WQ_SZ,f); fwrite(la[L].Wk.v,4,WQ_SZ,f);
        fwrite(la[L].Wv.m,4,WQ_SZ,f); fwrite(la[L].Wv.v,4,WQ_SZ,f);
        fwrite(la[L].Wo.m,4,WO_SZ,f); fwrite(la[L].Wo.v,4,WO_SZ,f);
        fwrite(la[L].W1.m,4,W1_SZ,f); fwrite(la[L].W1.v,4,W1_SZ,f);
        fwrite(la[L].W2.m,4,W2_SZ,f); fwrite(la[L].W2.v,4,W2_SZ,f);
        fwrite(la[L].W3.m,4,W3_SZ,f); fwrite(la[L].W3.v,4,W3_SZ,f);
        fwrite(la[L].rms_att.m,4,DIM,f); fwrite(la[L].rms_att.v,4,DIM,f);
        fwrite(la[L].rms_ffn.m,4,DIM,f); fwrite(la[L].rms_ffn.v,4,DIM,f);
    }
    fwrite(rms_final,4,DIM,f);
    fwrite(arms_final->m,4,DIM,f); fwrite(arms_final->v,4,DIM,f);
    fwrite(embed,4,(size_t)VOCAB*DIM,f);
    fwrite(aembed->m,4,(size_t)VOCAB*DIM,f); fwrite(aembed->v,4,(size_t)VOCAB*DIM,f);
    fclose(f);
}

static bool load_checkpoint(const char *path, int *step, int *total_steps,
                              float *lr, float *loss, double *cc, double *ct,
                              double *cw, int *cs, int *cb, int *adam_t,
                              LayerWeights *lw, LayerAdam *la,
                              float *rms_final, AdamState *arms_final,
                              float *embed, AdamState *aembed)
{
    FILE *f = fopen(path, "rb");
    if (!f) return false;
    ErCkptHdr h;
    fread(&h, sizeof(h), 1, f);
    if (h.magic != ER_CKPT_MAGIC || h.version != 1) { fclose(f); return false; }
    *step = h.step; *total_steps = h.total_steps; *lr = h.lr; *loss = h.loss;
    *cc = h.cum_compile; *ct = h.cum_train; *cw = h.cum_wall;
    *cs = h.cum_steps; *cb = h.cum_batches; *adam_t = h.adam_t;
    for (int L = 0; L < NLAYERS; L++) {
        fread(lw[L].Wq,4,WQ_SZ,f); fread(lw[L].Wk,4,WQ_SZ,f);
        fread(lw[L].Wv,4,WQ_SZ,f); fread(lw[L].Wo,4,WO_SZ,f);
        fread(lw[L].W1,4,W1_SZ,f); fread(lw[L].W2,4,W2_SZ,f);
        fread(lw[L].W3,4,W3_SZ,f);
        fread(lw[L].rms_att,4,DIM,f); fread(lw[L].rms_ffn,4,DIM,f);
        fread(la[L].Wq.m,4,WQ_SZ,f); fread(la[L].Wq.v,4,WQ_SZ,f);
        fread(la[L].Wk.m,4,WQ_SZ,f); fread(la[L].Wk.v,4,WQ_SZ,f);
        fread(la[L].Wv.m,4,WQ_SZ,f); fread(la[L].Wv.v,4,WQ_SZ,f);
        fread(la[L].Wo.m,4,WO_SZ,f); fread(la[L].Wo.v,4,WO_SZ,f);
        fread(la[L].W1.m,4,W1_SZ,f); fread(la[L].W1.v,4,W1_SZ,f);
        fread(la[L].W2.m,4,W2_SZ,f); fread(la[L].W2.v,4,W2_SZ,f);
        fread(la[L].W3.m,4,W3_SZ,f); fread(la[L].W3.v,4,W3_SZ,f);
        fread(la[L].rms_att.m,4,DIM,f); fread(la[L].rms_att.v,4,DIM,f);
        fread(la[L].rms_ffn.m,4,DIM,f); fread(la[L].rms_ffn.v,4,DIM,f);
    }
    fread(rms_final,4,DIM,f);
    fread(arms_final->m,4,DIM,f); fread(arms_final->v,4,DIM,f);
    fread(embed,4,(size_t)VOCAB*DIM,f);
    fread(aembed->m,4,(size_t)VOCAB*DIM,f); fread(aembed->v,4,(size_t)VOCAB*DIM,f);
    fclose(f);
    return true;
}

// ── Main ─────────────────────────────────────────────────────────────────────

int main(int argc, char *argv[]) {
    @autoreleasepool {
    setbuf(stdout, NULL);
    ane_init();
    mach_timebase_info(&g_tb);

    int total_steps = 5000;
    float lr = 1e-4f;   // lower than stories (3e-4) since task is narrower
    float adam_b1=0.9f, adam_b2=0.999f, adam_eps=1e-8f;
    int adam_t = 0, start_step = 0;
    bool do_resume = false;

    for (int i = 1; i < argc; i++) {
        if      (strcmp(argv[i], "--resume") == 0) do_resume = true;
        else if (strcmp(argv[i], "--steps") == 0  && i+1<argc) total_steps = atoi(argv[++i]);
        else if (strcmp(argv[i], "--lr")    == 0  && i+1<argc) lr          = atof(argv[++i]);
    }

    // ── Allocate model state ─────────────────────────────────────────────────
    LayerWeights lw[NLAYERS];
    LayerAdam    la[NLAYERS];
    LayerActs    acts[NLAYERS];
    LayerGrads   grads[NLAYERS];
    LayerKernels kern[NLAYERS];
    for (int L = 0; L < NLAYERS; L++) {
        lw[L]   = layer_weights_alloc();
        la[L]   = layer_adam_alloc();
        acts[L] = layer_acts_alloc();
        grads[L]= layer_grads_alloc();
        memset(&kern[L], 0, sizeof(LayerKernels));
    }
    float *rms_final = (float*)malloc(DIM*4);
    float *embed     = (float*)malloc((size_t)VOCAB*DIM*4);
    float *grms_final= (float*)calloc(DIM, 4);
    float *gembed    = (float*)calloc((size_t)VOCAB*DIM, 4);
    AdamState arms_final = adam_alloc(DIM);
    AdamState aembed     = adam_alloc((size_t)VOCAB*DIM);

    double cum_compile=0, cum_train=0, cum_wall=0;
    int    cum_steps=0,   cum_batches=0;
    float  resume_loss = 0;
    bool   resuming = false;

    if (do_resume) {
        resuming = load_checkpoint(CKPT_PATH, &start_step, &total_steps, &lr, &resume_loss,
                       &cum_compile, &cum_train, &cum_wall, &cum_steps, &cum_batches, &adam_t,
                       lw, la, rms_final, &arms_final, embed, &aembed);
        if (resuming) printf("[RESUMED step=%d loss=%.4f lr=%.2e]\n", start_step, resume_loss, lr);
    }

    if (!resuming) {
        printf("=== ER Extraction Fine-tuning (ANE) ===\n");
        printf("dim=%d hidden=%d heads=%d seq=%d vocab=%d layers=%d\n",
               DIM, HIDDEN, HEADS, SEQ, VOCAB, NLAYERS);
        // Random init (Xavier-ish)
        srand48(42);
        float scale_d = 1.0f/sqrtf(DIM), scale_h = 1.0f/sqrtf(HIDDEN);
        for (int L = 0; L < NLAYERS; L++) {
            for(size_t i=0;i<WQ_SZ;i++){lw[L].Wq[i]=scale_d*(2*drand48()-1); lw[L].Wk[i]=scale_d*(2*drand48()-1);}
            for(size_t i=0;i<WQ_SZ;i++){lw[L].Wv[i]=scale_d*(2*drand48()-1); lw[L].Wo[i]=scale_d*(2*drand48()-1);}
            for(size_t i=0;i<W1_SZ;i++) lw[L].W1[i]=scale_h*(2*drand48()-1);
            for(size_t i=0;i<W2_SZ;i++) lw[L].W2[i]=scale_d*(2*drand48()-1);
            for(size_t i=0;i<W3_SZ;i++) lw[L].W3[i]=scale_h*(2*drand48()-1);
            for(int i=0;i<DIM;i++){lw[L].rms_att[i]=1.0f; lw[L].rms_ffn[i]=1.0f;}
        }
        for(int i=0;i<DIM;i++) rms_final[i]=1.0f;
        float escale = 0.02f;
        for(size_t i=0;i<(size_t)VOCAB*DIM;i++) embed[i]=escale*(2*drand48()-1);

        size_t total_params = (size_t)NLAYERS*LAYER_PARAMS + DIM + (size_t)VOCAB*DIM;
        printf("Params:  %.2fM  |  LR=%.2e  |  Steps=%d\n",
               total_params/1e6, lr, total_steps);
        printf("Kernels: %d (%d weight-bearing + %d static)\n",
               TOTAL_WEIGHT_KERNELS+NLAYERS, TOTAL_WEIGHT_KERNELS, NLAYERS);
        printf("Accum:   %d steps per recompile\n\n", ACCUM_STEPS);
    }

    // ── Load ER dataset ──────────────────────────────────────────────────────
    ErDataset ds = er_dataset_load(DATA_PATH);
    if (ds.n == 0) { printf("ERROR: no examples loaded from %s\n", DATA_PATH); return 1; }

    // Stats
    int min_in=65535, max_in=0, min_out=65535, max_out=0, skipped=0;
    for (int i = 0; i < ds.n; i++) {
        int il = ds.examples[i].input_len;
        int ol = ds.examples[i].total_len - il;
        if (il < min_in)  min_in  = il;
        if (il > max_in)  max_in  = il;
        if (ol < min_out) min_out = ol;
        if (ol > max_out) max_out = ol;
        if (ds.examples[i].total_len > SEQ) skipped++;
    }
    printf("  input_len:  [%d .. %d]  output_len: [%d .. %d]  "
           "exceeds_seq: %d\n\n", min_in, max_in, min_out, max_out, skipped);

    // Per-step buffers
    uint16_t *seq_in  = (uint16_t*)malloc(SEQ*2);
    uint16_t *seq_tgt = (uint16_t*)malloc(SEQ*2);
    float    *dy      = (float*)malloc(SEQ*DIM*4);
    float    *dffn    = (float*)malloc(SEQ*DIM*4);
    float    *dh1     = (float*)malloc(SEQ*HIDDEN*4);
    float    *dh3     = (float*)malloc(SEQ*HIDDEN*4);
    float    *dx_ffn  = (float*)malloc(SEQ*DIM*4);
    float    *dx2     = (float*)malloc(SEQ*DIM*4);
    float    *do_buf  = (float*)malloc(SEQ*DIM*4);
    float    *dq      = (float*)malloc(SEQ*DIM*4);
    float    *dk      = (float*)malloc(SEQ*DIM*4);
    float    *dv      = (float*)malloc(SEQ*DIM*4);
    float    *dx_attn = (float*)malloc(SEQ*DIM*4);
    float    *x_cur   = (float*)malloc(SEQ*DIM*4);
    float    *x_final = (float*)malloc(SEQ*DIM*4);
    float    *logits  = (float*)malloc((size_t)SEQ*VOCAB*4);
    float    *dlogits = (float*)malloc((size_t)SEQ*VOCAB*4);

    // Static sdpaBwd2 kernels
    Kern *sdpaBwd2[NLAYERS];
    for (int L = 0; L < NLAYERS; L++) {
        sdpaBwd2[L] = compile_sdpa_bwd2_kern();
        if (!sdpaBwd2[L]) { printf("sdpaBwd2 compile failed\n"); return 1; }
    }

    dispatch_queue_t dw_q   = dispatch_queue_create("dw_cblas", DISPATCH_QUEUE_SERIAL);
    dispatch_group_t dw_grp = dispatch_group_create();

    float  last_loss = 999.0f;
    double total_compile_ms=0, total_train_ms=0;
    int    total_steps_done=0, total_batches=0;
    uint64_t t_wall_start = mach_absolute_time();
    srand48(42 + start_step);

    int step = start_step;
    while (step < total_steps) {

        // ── Compile budget check + exec() restart ────────────────────────────
        if (g_compile_count + TOTAL_WEIGHT_KERNELS > MAX_COMPILES) {
            for (int L=0;L<NLAYERS;L++) { free_layer_kernels(&kern[L]); free_kern(sdpaBwd2[L]); }
            double wall = tb_ms(mach_absolute_time() - t_wall_start);
            save_checkpoint(CKPT_PATH, step, total_steps, lr, last_loss,
                total_compile_ms+cum_compile, total_train_ms+cum_train, wall+cum_wall,
                total_steps_done+cum_steps, total_batches+cum_batches, adam_t,
                lw, la, rms_final, &arms_final, embed, &aembed);
            printf("[exec() restart step=%d compiles=%d loss=%.4f]\n",
                   step, g_compile_count, last_loss);
            fflush(stdout);
            execl(argv[0], argv[0], "--resume", NULL);
            perror("execl"); return 1;
        }

        // ── Compile all layers ───────────────────────────────────────────────
        uint64_t tc = mach_absolute_time();
        for (int L = 0; L < NLAYERS; L++) free_layer_kernels(&kern[L]);

        bool compile_ok = true;
        for (int L = 0; L < NLAYERS; L++) {
            printf("  Compiling layer %d/%d... (%d compiles)\r", L+1, NLAYERS, g_compile_count);
            fflush(stdout);
            if (!compile_layer_kernels(&kern[L], &lw[L])) {
                printf("\nCompile failed at layer %d\n", L);
                compile_ok = false; break;
            }
        }
        if (!compile_ok) { g_compile_count = MAX_COMPILES; continue; }

        for (int L = 0; L < NLAYERS; L++) {
            if (!sdpaBwd2[L]) {
                sdpaBwd2[L] = compile_sdpa_bwd2_kern();
                if (!sdpaBwd2[L]) { printf("sdpaBwd2 recompile failed\n"); return 1; }
            }
        }

        double cms = tb_ms(mach_absolute_time() - tc);
        total_compile_ms += cms;
        printf("  Compiled %d kernels in %.0fms                    \n", TOTAL_WEIGHT_KERNELS, cms);

        for (int L = 0; L < NLAYERS; L++) layer_grads_zero(&grads[L]);
        memset(grms_final, 0, DIM*4);
        memset(gembed, 0, (size_t)VOCAB*DIM*4);

        int    steps_batch = 0;
        uint64_t tt = mach_absolute_time();
        double t_ane=0, t_io=0, t_elem=0, t_rms=0, t_cblas_wait=0, t_cls=0;

        for (int a = 0; a < ACCUM_STEPS && step < total_steps; a++, step++) {
            uint64_t t0, t1;

            // ── Sample a random example ──────────────────────────────────────
            // Skip examples whose total_len > SEQ (tokenizer truncates most, but guard here)
            ErExample *ex;
            int tries = 0;
            do {
                int idx = (int)(drand48() * ds.n);
                ex = &ds.examples[idx];
                tries++;
            } while (ex->total_len > SEQ && tries < 100);
            if (ex->total_len > SEQ) { step--; continue; }  // pathological: skip

            int loss_end, loss_start = er_fill_buffers(ex, seq_in, seq_tgt, &loss_end);

            // ── Embedding lookup → x_cur [DIM, SEQ] ─────────────────────────
            t0 = mach_absolute_time();
            embed_lookup(x_cur, embed, seq_in, DIM, SEQ);
            t1 = mach_absolute_time(); t_elem += tb_ms(t1-t0);

            // ── FORWARD (12 layers) ──────────────────────────────────────────
            for (int L = 0; L < NLAYERS; L++) {
                LayerActs *ac = &acts[L];
                memcpy(ac->layer_in, x_cur, SEQ*DIM*4);

                t0 = mach_absolute_time();
                dispatch_group_wait(dw_grp, DISPATCH_TIME_FOREVER);
                t1 = mach_absolute_time(); t_cblas_wait += tb_ms(t1-t0); t0 = t1;

                io_write_fp16(kern[L].fwdAttn->ioIn, x_cur, DIM, SEQ);
                t1 = mach_absolute_time(); t_io += tb_ms(t1-t0); t0 = t1;
                ane_eval(kern[L].fwdAttn);
                t1 = mach_absolute_time(); t_ane += tb_ms(t1-t0); t0 = t1;

                io_read_fp16(kern[L].fwdAttn->ioOut, ac->o_out,    0,     DIM, SEQ);
                io_read_fp16(kern[L].fwdAttn->ioOut, ac->attn_out, 4*DIM, DIM, SEQ);
                io_read_fp16(kern[L].fwdAttn->ioOut, ac->xnorm,    5*DIM, DIM, SEQ);
                t1 = mach_absolute_time(); t_io += tb_ms(t1-t0); t0 = t1;

                vDSP_vadd(x_cur, 1, ac->o_out, 1, ac->x2, 1, (vDSP_Length)(SEQ*DIM));
                t1 = mach_absolute_time(); t_elem += tb_ms(t1-t0); t0 = t1;

                io_write_fp16(kern[L].fwdFFN->ioIn, ac->x2, DIM, SEQ);
                t1 = mach_absolute_time(); t_io += tb_ms(t1-t0); t0 = t1;
                ane_eval(kern[L].fwdFFN);
                t1 = mach_absolute_time(); t_ane += tb_ms(t1-t0); t0 = t1;

                io_read_fp16(kern[L].fwdFFN->ioOut, ac->ffn_out,  0,            DIM,    SEQ);
                io_read_fp16(kern[L].fwdFFN->ioOut, ac->h1,       DIM,          HIDDEN, SEQ);
                io_read_fp16(kern[L].fwdFFN->ioOut, ac->h3,       DIM+HIDDEN,   HIDDEN, SEQ);
                io_read_fp16(kern[L].fwdFFN->ioOut, ac->silu_out, DIM+2*HIDDEN, HIDDEN, SEQ);
                io_read_fp16(kern[L].fwdFFN->ioOut, ac->x2norm,   DIM+3*HIDDEN, DIM,    SEQ);
                t1 = mach_absolute_time(); t_io += tb_ms(t1-t0); t0 = t1;

                vDSP_vadd(ac->x2, 1, ac->ffn_out, 1, x_cur, 1, (vDSP_Length)(SEQ*DIM));
                t1 = mach_absolute_time(); t_elem += tb_ms(t1-t0);
            }

            // ── Final RMSNorm + Classifier ───────────────────────────────────
            t0 = mach_absolute_time();
            rmsnorm(x_final, x_cur, rms_final, DIM, SEQ);
            t1 = mach_absolute_time(); t_rms += tb_ms(t1-t0); t0 = t1;

            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                        VOCAB, SEQ, DIM, 1.0f,
                        embed, DIM, x_final, SEQ, 0.0f, logits, SEQ);
            t1 = mach_absolute_time(); t_cls += tb_ms(t1-t0); t0 = t1;

            // ── Masked cross-entropy loss (output tokens only) ────────────────
            float loss = masked_cross_entropy_loss(
                dlogits, logits, seq_tgt, VOCAB, SEQ, loss_start, loss_end);
            last_loss = loss;
            t1 = mach_absolute_time(); t_elem += tb_ms(t1-t0); t0 = t1;

            // ── BACKWARD ─────────────────────────────────────────────────────
            // dlogits is already zeroed for input positions by masked_cross_entropy_loss

            // Classifier backward: dy = embed^T @ dlogits
            cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                        DIM, SEQ, VOCAB, 1.0f,
                        embed, DIM, dlogits, SEQ, 0.0f, dy, SEQ);

            // gembed += dlogits @ x_final^T  (async)
            dispatch_group_async(dw_grp, dw_q, ^{
                cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                            VOCAB, DIM, SEQ, 1.0f,
                            dlogits, SEQ, x_final, SEQ, 1.0f, gembed, DIM);
            });

            // Final RMSNorm backward
            float *dx_rms_final = (float*)calloc(SEQ*DIM, 4);
            rmsnorm_bwd(dx_rms_final, grms_final, dy, x_cur, rms_final, DIM, SEQ);
            memcpy(dy, dx_rms_final, SEQ*DIM*4);
            free(dx_rms_final);

            // ── Layer backward (reverse order) ───────────────────────────────
            for (int L = NLAYERS-1; L >= 0; L--) {
                LayerActs  *ac = &acts[L];
                LayerGrads *gr = &grads[L];

                memcpy(dffn, dy, SEQ*DIM*4);

                // FFN backward
                io_write_fp16_at(kern[L].ffnBwd->ioIn, 0, dffn, DIM, SEQ);
                io_copy(kern[L].ffnBwd->ioIn, DIM, kern[L].fwdFFN->ioOut, DIM, 2*HIDDEN, SEQ);
                ane_eval(kern[L].ffnBwd);
                io_read_fp16(kern[L].ffnBwd->ioOut, dx_ffn, 0,          DIM,    SEQ);
                io_read_fp16(kern[L].ffnBwd->ioOut, dh1,    DIM,        HIDDEN, SEQ);
                io_read_fp16(kern[L].ffnBwd->ioOut, dh3,    DIM+HIDDEN, HIDDEN, SEQ);

                float *capt_dffn = (float*)malloc(SEQ*DIM*4);    memcpy(capt_dffn, dffn, SEQ*DIM*4);
                float *capt_silu = (float*)malloc(SEQ*HIDDEN*4); memcpy(capt_silu, ac->silu_out, SEQ*HIDDEN*4);
                float *capt_dh1  = (float*)malloc(SEQ*HIDDEN*4); memcpy(capt_dh1,  dh1,  SEQ*HIDDEN*4);
                float *capt_dh3  = (float*)malloc(SEQ*HIDDEN*4); memcpy(capt_dh3,  dh3,  SEQ*HIDDEN*4);
                float *capt_x2n  = (float*)malloc(SEQ*DIM*4);    memcpy(capt_x2n,  ac->x2norm, SEQ*DIM*4);
                dispatch_group_async(dw_grp, dw_q, ^{
                    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, DIM, HIDDEN, SEQ,
                                1.0f, capt_dffn, SEQ, capt_silu, SEQ, 1.0f, gr->W2, HIDDEN);
                    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, HIDDEN, DIM, SEQ,
                                1.0f, capt_dh1, SEQ, capt_x2n, SEQ, 1.0f, gr->W1, DIM);
                    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, HIDDEN, DIM, SEQ,
                                1.0f, capt_dh3, SEQ, capt_x2n, SEQ, 1.0f, gr->W3, DIM);
                    free(capt_dffn); free(capt_silu); free(capt_dh1); free(capt_dh3); free(capt_x2n);
                });

                memset(dx2, 0, SEQ*DIM*4);
                rmsnorm_bwd(dx2, gr->rms_ffn, dx_ffn, ac->x2, lw[L].rms_ffn, DIM, SEQ);
                for(int i = 0; i < SEQ*DIM; i++) dx2[i] += dy[i];

                memcpy(do_buf, dx2, SEQ*DIM*4);
                float *capt_do   = (float*)malloc(SEQ*DIM*4); memcpy(capt_do,   do_buf,       SEQ*DIM*4);
                float *capt_attn = (float*)malloc(SEQ*DIM*4); memcpy(capt_attn, ac->attn_out, SEQ*DIM*4);
                dispatch_group_async(dw_grp, dw_q, ^{
                    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, DIM, DIM, SEQ,
                                1.0f, capt_do, SEQ, capt_attn, SEQ, 1.0f, gr->Wo, DIM);
                    free(capt_do); free(capt_attn);
                });

                io_copy(kern[L].sdpaBwd1->ioIn, 0, kern[L].fwdAttn->ioOut, DIM, 3*DIM, SEQ);
                io_write_fp16_at(kern[L].sdpaBwd1->ioIn, 3*DIM, dx2, DIM, SEQ);
                ane_eval(kern[L].sdpaBwd1);
                io_copy(sdpaBwd2[L]->ioIn, 0,        kern[L].sdpaBwd1->ioOut, DIM,   2*SCORE_CH, SEQ);
                io_copy(sdpaBwd2[L]->ioIn, 2*SCORE_CH, kern[L].fwdAttn->ioOut, DIM,  2*DIM,      SEQ);
                ane_eval(sdpaBwd2[L]);

                io_read_fp16(sdpaBwd2[L]->ioOut,       dq, 0,   DIM, SEQ);
                io_read_fp16(sdpaBwd2[L]->ioOut,       dk, DIM, DIM, SEQ);
                io_read_fp16(kern[L].sdpaBwd1->ioOut,  dv, 0,   DIM, SEQ);

                float *capt_dq = (float*)malloc(SEQ*DIM*4); memcpy(capt_dq, dq, SEQ*DIM*4);
                float *capt_dk = (float*)malloc(SEQ*DIM*4); memcpy(capt_dk, dk, SEQ*DIM*4);
                float *capt_dv = (float*)malloc(SEQ*DIM*4); memcpy(capt_dv, dv, SEQ*DIM*4);
                float *capt_xn = (float*)malloc(SEQ*DIM*4); memcpy(capt_xn, ac->xnorm, SEQ*DIM*4);
                dispatch_group_async(dw_grp, dw_q, ^{
                    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, DIM, DIM, SEQ,
                                1.0f, capt_dq, SEQ, capt_xn, SEQ, 1.0f, gr->Wq, DIM);
                    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, DIM, DIM, SEQ,
                                1.0f, capt_dk, SEQ, capt_xn, SEQ, 1.0f, gr->Wk, DIM);
                    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, DIM, DIM, SEQ,
                                1.0f, capt_dv, SEQ, capt_xn, SEQ, 1.0f, gr->Wv, DIM);
                    free(capt_dq); free(capt_dk); free(capt_dv); free(capt_xn);
                });

                io_copy(kern[L].qkvBwd->ioIn, 0,     sdpaBwd2[L]->ioOut,      0,   2*DIM, SEQ);
                io_copy(kern[L].qkvBwd->ioIn, 2*DIM, kern[L].sdpaBwd1->ioOut, 0,   DIM,   SEQ);
                ane_eval(kern[L].qkvBwd);
                io_read_fp16(kern[L].qkvBwd->ioOut, dx_attn, 0, DIM, SEQ);

                float *dx_rms1 = (float*)calloc(SEQ*DIM, 4);
                rmsnorm_bwd(dx_rms1, gr->rms_att, dx_attn, ac->layer_in, lw[L].rms_att, DIM, SEQ);
                for(int i = 0; i < SEQ*DIM; i++) dy[i] = dx_rms1[i] + dx2[i];
                free(dx_rms1);
            }

            // ── Embedding backward ───────────────────────────────────────────
            dispatch_group_wait(dw_grp, DISPATCH_TIME_FOREVER);
            embed_backward(gembed, dy, seq_in, DIM, SEQ);

            steps_batch++;
            if (step % 50 == 0 || step == start_step)
                printf("step %-5d loss=%.4f  (output positions %d-%d)\n",
                       step, loss, loss_start, loss_end);

            fprintf(stderr,
                "{\"type\":\"step\",\"step\":%d,\"loss\":%.6f,"
                "\"loss_start\":%d,\"loss_end\":%d,"
                "\"t_ane\":%.3f,\"t_io\":%.3f,\"t_cls\":%.3f,"
                "\"t_elem\":%.3f,\"t_rms\":%.3f,\"t_cblas_wait\":%.3f}\n",
                step, loss, loss_start, loss_end,
                t_ane/steps_batch, t_io/steps_batch, t_cls/steps_batch,
                t_elem/steps_batch, t_rms/steps_batch, t_cblas_wait/steps_batch);
        }

        double tms = tb_ms(mach_absolute_time() - tt);
        total_train_ms   += tms;
        total_steps_done += steps_batch;
        total_batches++;

        dispatch_group_wait(dw_grp, DISPATCH_TIME_FOREVER);

        // ── Adam update ──────────────────────────────────────────────────────
        float gsc = 1.0f / steps_batch;
        adam_t++;
        for (int L = 0; L < NLAYERS; L++) {
            LayerGrads *g = &grads[L];
            for(size_t i=0;i<WQ_SZ;i++){g->Wq[i]*=gsc; g->Wk[i]*=gsc; g->Wv[i]*=gsc; g->Wo[i]*=gsc;}
            for(size_t i=0;i<W1_SZ;i++) g->W1[i]*=gsc;
            for(size_t i=0;i<W2_SZ;i++) g->W2[i]*=gsc;
            for(size_t i=0;i<W3_SZ;i++) g->W3[i]*=gsc;
            for(int i=0;i<DIM;i++){g->rms_att[i]*=gsc; g->rms_ffn[i]*=gsc;}
            adam_update(lw[L].Wq,      g->Wq,      &la[L].Wq,      adam_t, lr, adam_b1, adam_b2, adam_eps);
            adam_update(lw[L].Wk,      g->Wk,      &la[L].Wk,      adam_t, lr, adam_b1, adam_b2, adam_eps);
            adam_update(lw[L].Wv,      g->Wv,      &la[L].Wv,      adam_t, lr, adam_b1, adam_b2, adam_eps);
            adam_update(lw[L].Wo,      g->Wo,      &la[L].Wo,      adam_t, lr, adam_b1, adam_b2, adam_eps);
            adam_update(lw[L].W1,      g->W1,      &la[L].W1,      adam_t, lr, adam_b1, adam_b2, adam_eps);
            adam_update(lw[L].W2,      g->W2,      &la[L].W2,      adam_t, lr, adam_b1, adam_b2, adam_eps);
            adam_update(lw[L].W3,      g->W3,      &la[L].W3,      adam_t, lr, adam_b1, adam_b2, adam_eps);
            adam_update(lw[L].rms_att, g->rms_att, &la[L].rms_att, adam_t, lr, adam_b1, adam_b2, adam_eps);
            adam_update(lw[L].rms_ffn, g->rms_ffn, &la[L].rms_ffn, adam_t, lr, adam_b1, adam_b2, adam_eps);
        }
        for(int i=0;i<DIM;i++) grms_final[i]*=gsc;
        adam_update(rms_final, grms_final, &arms_final, adam_t, lr, adam_b1, adam_b2, adam_eps);
        for(size_t i=0;i<(size_t)VOCAB*DIM;i++) gembed[i]*=gsc;
        adam_update(embed, gembed, &aembed, adam_t, lr, adam_b1, adam_b2, adam_eps);

        printf("  [batch %d: compile=%.0fms train=%.1fms (%.1fms/step) compiles=%d]\n",
               steps_batch, cms, tms, tms/steps_batch, g_compile_count);

        fprintf(stderr,
            "{\"type\":\"batch\",\"batch\":%d,\"compile_ms\":%.1f,"
            "\"train_ms\":%.1f,\"ms_per_step\":%.1f}\n",
            steps_batch, cms, tms, tms/steps_batch);
    }

    // ── Final report ─────────────────────────────────────────────────────────
    double wall = tb_ms(mach_absolute_time() - t_wall_start);
    total_compile_ms += cum_compile; total_train_ms += cum_train;
    wall += cum_wall; total_steps_done += cum_steps; total_batches += cum_batches;

    printf("\n=== Training Complete ===\n");
    printf("Steps:        %d  |  Final loss: %.4f\n", total_steps_done, last_loss);
    printf("Wall:         %.1f s\n",     wall/1000);
    printf("Compile time: %.1f s (%.1f%%)\n", total_compile_ms/1000, 100*total_compile_ms/wall);
    printf("Train time:   %.1f s (%.1f%%)\n", total_train_ms/1000,   100*total_train_ms/wall);
    printf("Avg step:     %.1f ms\n",    total_train_ms/total_steps_done);

    save_checkpoint(CKPT_PATH, total_steps, total_steps, lr, last_loss,
        total_compile_ms, total_train_ms, wall,
        total_steps_done, total_batches, adam_t,
        lw, la, rms_final, &arms_final, embed, &aembed);
    printf("Checkpoint saved: %s\n", CKPT_PATH);

    // Cleanup
    er_dataset_free(&ds);
    for (int L=0;L<NLAYERS;L++) {
        free_layer_kernels(&kern[L]); free_kern(sdpaBwd2[L]);
        layer_weights_free(&lw[L]); layer_adam_free(&la[L]);
        layer_acts_free(&acts[L]);   layer_grads_free(&grads[L]);
    }
    free(rms_final); free(embed); free(grms_final); free(gembed);
    adam_free(&arms_final); adam_free(&aembed);
    free(seq_in); free(seq_tgt); free(dy); free(dffn); free(dh1); free(dh3);
    free(dx_ffn); free(dx2); free(do_buf); free(dq); free(dk); free(dv);
    free(dx_attn); free(x_cur); free(x_final); free(logits); free(dlogits);
    } // @autoreleasepool
    return 0;
}
