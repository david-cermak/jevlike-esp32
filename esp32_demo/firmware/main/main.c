#include <ctype.h>
#include <stdio.h>
#include <string.h>

#include "esp_log.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "jevlike_scorer.h"

static const char *TAG = "jevlike";
static const float k_confidence = 0.45f;

extern const uint8_t weights_bin_start[] asm("_binary_weights_bin_start");
extern const uint8_t weights_bin_end[] asm("_binary_weights_bin_end");

static int contains_word(const char *text, const char *needle)
{
    const char *found = strstr(text, needle);
    return found != NULL;
}

static void lower_copy(const char *src, char *dst, size_t n)
{
    size_t i = 0;
    for (; src[i] && i + 1 < n; i++) {
        dst[i] = (char)tolower((unsigned char)src[i]);
    }
    dst[i] = 0;
}

static void local_action(const char *text)
{
    char lowered[256];
    lower_copy(text, lowered, sizeof(lowered));
    printf("[ESP32] local command: '%s'\n", text);
    if (contains_word(lowered, "on") && !contains_word(lowered, "off")) {
        printf("[ESP32] GPIO -> LIGHT ON\n");
    } else if (contains_word(lowered, "off")) {
        printf("[ESP32] GPIO -> LIGHT OFF\n");
    } else {
        printf("[ESP32] GPIO/MQTT -> execute command\n");
    }
}

static void route(JevModel *model, const char *text)
{
    float probs[JEV_MAX_OPTIONS];
    int n = jev_score(model, text, probs);
    if (n < 0) {
        printf("[EDGE] scorer failed\n");
        return;
    }
    int best = 0;
    for (int i = 1; i < n; i++) {
        if (probs[i] > probs[best]) {
            best = i;
        }
    }
    printf("[\n");
    for (int i = 0; i < n; i++) {
        printf("  {\"option\": \"%s\", \"probability\": %.8f}%s\n",
               model->options[i], (double)probs[i], i + 1 < n ? "," : "");
    }
    printf("]\n");
    if (probs[best] < k_confidence) {
        printf("[EDGE] low confidence (%.3f) -> forward to cloud LLM\n", (double)probs[best]);
        return;
    }
    const char *option = model->options[best];
    if (strcmp(option, "weather") == 0) {
        printf("[EDGE] weather -> local weather client / API\n");
    } else if (strcmp(option, "command") == 0) {
        local_action(text);
    } else {
        printf("[EDGE] complex -> forward to cloud LLM\n");
    }
}

static void strip_line(char *line)
{
    size_t n = strlen(line);
    while (n > 0 && (line[n - 1] == '\n' || line[n - 1] == '\r')) {
        line[--n] = 0;
    }
}

void app_main(void)
{
    JevModel model;
    size_t nbytes = (size_t)(weights_bin_end - weights_bin_start);
    if (jev_load(&model, weights_bin_start, nbytes) != 0) {
        ESP_LOGE(TAG, "failed to load scorer weights (%u bytes)", (unsigned)nbytes);
        return;
    }

    printf("\nFake ESP32 online.\n\n");
    const char *demos[] = {
        "turn on the kitchen light",
        "what is the weather tomorrow?",
        "explain how TLS 1.3 works",
    };
    for (size_t i = 0; i < sizeof(demos) / sizeof(demos[0]); i++) {
        printf("> %s\n", demos[i]);
        route(&model, demos[i]);
        printf("\n");
    }

    printf("Type ASCII commands (UART). Examples:\n"
           "  turn on the kitchen light\n"
           "  what is the weather tomorrow?\n"
           "  explain how TLS 1.3 works\n\n");

    char line[256];
    while (1) {
        printf("> ");
        fflush(stdout);
        if (fgets(line, sizeof(line), stdin) == NULL) {
            vTaskDelay(pdMS_TO_TICKS(50));
            continue;
        }
        strip_line(line);
        if (line[0] == 0) {
            continue;
        }
        route(&model, line);
    }
}
