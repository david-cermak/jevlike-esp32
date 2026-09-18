#include "jevlike_scorer.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

#define JEV_LN_EPS 1e-5f
#define JEV_MAGIC "JEVLIKE1"

static uint32_t u32le(const uint8_t *p)
{
    return (uint32_t)p[0] | ((uint32_t)p[1] << 8) | ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
}

static void layernorm(const float *x, int n, int width, const float *w, const float *b, float *y)
{
    for (int i = 0; i < n; i++) {
        const float *xi = x + (size_t)i * width;
        float mean = 0.0f;
        for (int d = 0; d < width; d++) {
            mean += xi[d];
        }
        mean /= (float)width;
        float var = 0.0f;
        for (int d = 0; d < width; d++) {
            float z = xi[d] - mean;
            var += z * z;
        }
        var /= (float)width;
        float inv = 1.0f / sqrtf(var + JEV_LN_EPS);
        float *yi = y + (size_t)i * width;
        for (int d = 0; d < width; d++) {
            yi[d] = (xi[d] - mean) * inv * w[d] + b[d];
        }
    }
}

static void linear(const float *x, int n, int in_dim, const float *weight, int out_dim, float *y)
{
    for (int i = 0; i < n; i++) {
        const float *xi = x + (size_t)i * in_dim;
        for (int o = 0; o < out_dim; o++) {
            const float *row = weight + (size_t)o * in_dim;
            float sum = 0.0f;
            for (int k = 0; k < in_dim; k++) {
                sum += xi[k] * row[k];
            }
            y[(size_t)i * out_dim + o] = sum;
        }
    }
}

static int encode(const char *text, int max_tokens, uint16_t *ids)
{
    int n = 0;
    for (const unsigned char *p = (const unsigned char *)text; *p && n < max_tokens; p++, n++) {
        ids[n] = (uint16_t)(*p + 1);
    }
    return n;
}

static void embed_mean(const JevModel *m, const uint16_t *ids, int n, float *out)
{
    memset(out, 0, (size_t)m->width * sizeof(float));
    if (n <= 0) {
        return;
    }
    for (int t = 0; t < n; t++) {
        const float *row = m->embedding + (size_t)ids[t] * m->width;
        for (int d = 0; d < m->width; d++) {
            out[d] += row[d];
        }
    }
    float inv = 1.0f / (float)n;
    for (int d = 0; d < m->width; d++) {
        out[d] *= inv;
    }
}

int jev_load(JevModel *model, const void *bytes, size_t nbytes)
{
    memset(model, 0, sizeof(*model));
    if (nbytes < 32 || memcmp(bytes, JEV_MAGIC, 8) != 0) {
        return -1;
    }
    const uint8_t *p = (const uint8_t *)bytes;
    model->width = (int)u32le(p + 8);
    model->rank = (int)u32le(p + 12);
    model->context_tokens = (int)u32le(p + 16);
    model->option_tokens = (int)u32le(p + 20);
    model->vocab = (int)u32le(p + 24);
    model->n_options = (int)u32le(p + 28);
    if (model->width <= 0 || model->rank <= 0 || model->n_options < 2 ||
        model->n_options > JEV_MAX_OPTIONS || model->vocab <= 0) {
        return -1;
    }

    size_t floats = (size_t)model->vocab * model->width
        + (size_t)model->context_tokens * model->width
        + (size_t)model->width * 4
        + (size_t)model->rank * model->width * 3;
    size_t need = 32 + floats * sizeof(float);
    if (nbytes < need) {
        return -1;
    }
    const float *f = (const float *)(p + 32);
    model->embedding = f;
    f += (size_t)model->vocab * model->width;
    model->position = f;
    f += (size_t)model->context_tokens * model->width;
    model->context_norm_w = f;
    f += model->width;
    model->context_norm_b = f;
    f += model->width;
    model->option_norm_w = f;
    f += model->width;
    model->option_norm_b = f;
    f += model->width;
    model->query = f;
    f += (size_t)model->rank * model->width;
    model->key = f;
    f += (size_t)model->rank * model->width;
    model->value = f;

    const char *s = (const char *)bytes + need;
    size_t left = nbytes - need;
    for (int i = 0; i < model->n_options; i++) {
        if (left == 0) {
            return -1;
        }
        model->options[i] = s;
        size_t n = 0;
        while (n < left && s[n] != 0) {
            n++;
        }
        if (n == left) {
            return -1;
        }
        s += n + 1;
        left -= n + 1;
    }

    model->work_floats = (size_t)model->context_tokens * model->width * 3
        + (size_t)model->n_options * model->width * 3
        + (size_t)model->n_options * model->context_tokens
        + (size_t)model->context_tokens
        + 16;
    model->work = (float *)malloc(model->work_floats * sizeof(float));
    return model->work ? 0 : -1;
}

void jev_unload(JevModel *model)
{
    free(model->work);
    memset(model, 0, sizeof(*model));
}

int jev_score(JevModel *model, const char *text, float *probs)
{
    const int W = model->width, R = model->rank, N = model->n_options;
    uint16_t ctx_ids[192];
    uint16_t opt_ids[32];
    if (model->context_tokens > 192 || model->option_tokens > 32) {
        return -1;
    }
    int L = encode(text ? text : "", model->context_tokens, ctx_ids);
    if (L <= 0) {
        L = 1;
        ctx_ids[0] = 0;
    }

    float *ctx = model->work;
    float *opt = ctx + (size_t)L * W;
    float *q = opt + (size_t)N * W;
    float *k = q + (size_t)N * R;
    float *v = k + (size_t)L * R;
    float *scores = v + (size_t)L * R;

    for (int t = 0; t < L; t++) {
        const float *emb = model->embedding + (size_t)ctx_ids[t] * W;
        const float *pos = model->position + (size_t)t * W;
        float *row = ctx + (size_t)t * W;
        for (int d = 0; d < W; d++) {
            row[d] = emb[d] + pos[d];
        }
    }
    layernorm(ctx, L, W, model->context_norm_w, model->context_norm_b, ctx);

    for (int n = 0; n < N; n++) {
        int olen = encode(model->options[n], model->option_tokens, opt_ids);
        embed_mean(model, opt_ids, olen, opt + (size_t)n * W);
    }
    layernorm(opt, N, W, model->option_norm_w, model->option_norm_b, opt);

    linear(opt, N, W, model->query, R, q);
    linear(ctx, L, W, model->key, R, k);
    linear(ctx, L, W, model->value, R, v);

    float scale = 1.0f / sqrtf((float)R);
    for (int n = 0; n < N; n++) {
        const float *qn = q + (size_t)n * R;
        float *sn = scores + (size_t)n * L;
        float max_s = -1e30f;
        for (int t = 0; t < L; t++) {
            const float *kt = k + (size_t)t * R;
            float s = 0.0f;
            for (int d = 0; d < R; d++) {
                s += qn[d] * kt[d];
            }
            s *= scale;
            sn[t] = s;
            if (s > max_s) {
                max_s = s;
            }
        }
        float sum = 0.0f;
        for (int t = 0; t < L; t++) {
            sn[t] = expf(sn[t] - max_s);
            sum += sn[t];
        }
        float inv = 1.0f / sum;
        for (int t = 0; t < L; t++) {
            sn[t] *= inv;
        }
        float logit = 0.0f;
        for (int d = 0; d < R; d++) {
            float att = 0.0f;
            for (int t = 0; t < L; t++) {
                att += sn[t] * v[(size_t)t * R + d];
            }
            logit += qn[d] * att;
        }
        probs[n] = logit * scale;
    }

    float max_l = probs[0];
    for (int n = 1; n < N; n++) {
        if (probs[n] > max_l) {
            max_l = probs[n];
        }
    }
    float sum = 0.0f;
    for (int n = 0; n < N; n++) {
        probs[n] = expf(probs[n] - max_l);
        sum += probs[n];
    }
    for (int n = 0; n < N; n++) {
        probs[n] /= sum;
    }
    return N;
}
