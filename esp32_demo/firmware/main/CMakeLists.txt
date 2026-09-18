idf_component_register(
    SRCS "main.c" "jevlike_scorer.c"
    INCLUDE_DIRS "."
    EMBED_FILES "weights.bin"
)
