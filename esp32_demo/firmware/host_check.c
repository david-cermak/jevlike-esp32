#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "jevlike_scorer.h"

static int load_file(const char *path, uint8_t **out, size_t *nbytes)
{
    FILE *f = fopen(path, "rb");
    if (!f) {
        perror(path);
        return -1;
    }
    if (fseek(f, 0, SEEK_END) != 0) {
        fclose(f);
        return -1;
    }
    long n = ftell(f);
    if (n <= 0) {
        fclose(f);
        return -1;
    }
    rewind(f);
    uint8_t *buf = (uint8_t *)malloc((size_t)n);
    if (!buf || fread(buf, 1, (size_t)n, f) != (size_t)n) {
        free(buf);
        fclose(f);
        return -1;
    }
    fclose(f);
    *out = buf;
    *nbytes = (size_t)n;
    return 0;
}

int main(int argc, char **argv)
{
    const char *weights = argc > 1 ? argv[1] : "main/weights.bin";
    uint8_t *bytes = NULL;
    size_t nbytes = 0;
    if (load_file(weights, &bytes, &nbytes) != 0) {
        return 1;
    }
    JevModel model;
    if (jev_load(&model, bytes, nbytes) != 0) {
        fprintf(stderr, "jev_load failed\n");
        free(bytes);
        return 1;
    }
    for (int i = 2; i < argc; i++) {
        float probs[JEV_MAX_OPTIONS];
        int n = jev_score(&model, argv[i], probs);
        printf("%s", argv[i]);
        for (int k = 0; k < n; k++) {
            printf("\t%s=%.8f", model.options[k], (double)probs[k]);
        }
        printf("\n");
    }
    jev_unload(&model);
    free(bytes);
    return 0;
}
