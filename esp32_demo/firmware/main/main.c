#include <ctype.h>
#include <stdio.h>
#include <string.h>

#include "esp_console.h"
#include "esp_log.h"
#include "jevlike_scorer.h"

static const char *TAG = "jevlike";
static const float k_confidence = 0.45f;

extern const uint8_t weights_bin_start[] asm("_binary_weights_bin_start");
extern const uint8_t weights_bin_end[] asm("_binary_weights_bin_end");

static JevModel s_model;

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

/* Join argv[1..] into one string and score/route it. */
static int cmd_ask(int argc, char **argv)
{
    if (argc < 2) {
        printf("Usage: ask <text>\n");
        return 1;
    }
    char buf[256];
    size_t pos = 0;
    for (int i = 1; i < argc; i++) {
        size_t n = strlen(argv[i]);
        if (pos + n + (i > 1 ? 1 : 0) >= sizeof(buf)) {
            printf("ask: text too long\n");
            return 1;
        }
        if (i > 1) {
            buf[pos++] = ' ';
        }
        memcpy(buf + pos, argv[i], n);
        pos += n;
    }
    buf[pos] = 0;
    route(&s_model, buf);
    return 0;
}

static void start_console(void)
{
    esp_console_repl_t *repl = NULL;
    esp_console_repl_config_t repl_config = ESP_CONSOLE_REPL_CONFIG_DEFAULT();
    repl_config.prompt = ">";
    repl_config.max_cmdline_length = 256;
    /* Scorer needs more stack than the REPL default (4096). */
    repl_config.task_stack_size = 16384;

    const esp_console_cmd_t ask = {
        .command = "ask",
        .help = "Score and route a natural-language request",
        .hint = "<text>",
        .func = &cmd_ask,
    };
    ESP_ERROR_CHECK(esp_console_cmd_register(&ask));

    esp_console_dev_uart_config_t hw_config = ESP_CONSOLE_DEV_UART_CONFIG_DEFAULT();
    ESP_ERROR_CHECK(esp_console_new_repl_uart(&hw_config, &repl_config, &repl));
    ESP_ERROR_CHECK(esp_console_start_repl(repl));
}

void app_main(void)
{
    size_t nbytes = (size_t)(weights_bin_end - weights_bin_start);
    if (jev_load(&s_model, weights_bin_start, nbytes) != 0) {
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
        route(&s_model, demos[i]);
        printf("\n");
    }

    printf("Type: ask <text>  (linenoise/UART console, Enter to submit)\n"
           "  ask turn on the kitchen light\n"
           "  ask what is the weather tomorrow?\n"
           "  ask explain how TLS 1.3 works\n\n");

    start_console();
}
