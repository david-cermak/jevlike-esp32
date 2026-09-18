#pragma once

#include <stddef.h>
#include <stdint.h>

enum { JEV_MAX_OPTIONS = 4 };

typedef struct {
    int width;
    int rank;
    int context_tokens;
    int option_tokens;
    int vocab;
    int n_options;
    const float *embedding;
    const float *position;
    const float *context_norm_w;
    const float *context_norm_b;
    const float *option_norm_w;
    const float *option_norm_b;
    const float *query;
    const float *key;
    const float *value;
    const char *options[JEV_MAX_OPTIONS];
    float *work;
    size_t work_floats;
} JevModel;

int jev_load(JevModel *model, const void *bytes, size_t nbytes);
void jev_unload(JevModel *model);
int jev_score(JevModel *model, const char *text, float *probs);
