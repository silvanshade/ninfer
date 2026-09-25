/**
 * @file capi.h
 * @brief C linkage for the subset of Engine a foreign scheduler needs.
 *
 * The C++ surface in `ninfer/engine.h` carries `std::` containers, pimpl handles and exceptions,
 * none of which crosses a C ABI. This facade exposes the narrow path a host outside C++ drives:
 * open an artifact, turn text into tokens, generate greedily from tokens, and turn tokens back
 * into bytes.
 *
 * Three rules hold at every entry point. No exception escapes: a failure becomes a status and,
 * when the caller supplies a buffer, a NUL-terminated message. No ownership crosses except the
 * opaque engine handle, which `ninfer_engine_close` releases. Every array is caller-allocated,
 * and a call that would overflow one reports the required length instead of writing past it.
 */

#ifndef NINFER_CAPI_H
#define NINFER_CAPI_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/** Marks the five entry points as exported; everything else in the object stays hidden. */
#if defined(_WIN32)
#define NINFER_CAPI __declspec(dllexport)
#else
#define NINFER_CAPI __attribute__((visibility("default")))
#endif

/** Outcome of a facade call. Every entry point returns one of these and never throws. */
typedef enum ninfer_status {
    /** The call completed. */
    NINFER_OK = 0,
    /** A required pointer was null, or a numeric argument was outside its domain. */
    NINFER_INVALID_ARGUMENT = 1,
    /** The artifact, its directory, or a required resource was not found or not readable. */
    NINFER_NOT_FOUND = 2,
    /** A caller-allocated array was too small; the required length is reported through its count. */
    NINFER_BUFFER_TOO_SMALL = 3,
    /** The engine rejected the request or failed during execution. */
    NINFER_RUNTIME_ERROR = 4,
    /** A failure that carried no message the facade could interpret. */
    NINFER_UNKNOWN_ERROR = 5
} ninfer_status;

/** An opened engine. Created by ninfer_engine_open, released by ninfer_engine_close. */
typedef struct ninfer_engine ninfer_engine;

/**
 * Options for opening an engine.
 *
 * The fields are the ones a foreign scheduler must set; every other EngineOptions field keeps its
 * C++ default. Zero-initializing this struct and setting `artifact_path` is a valid call:
 * `device` 0, `max_context` 0 and `max_concurrency` 0 each select the C++ default rather than an
 * empty value, because a zero context or concurrency has no meaning.
 */
typedef struct ninfer_engine_options {
    /** NUL-terminated path to the artifact entry file. Required. */
    const char* artifact_path;
    /** NUL-terminated chat-template path, or null to use the artifact's own. */
    const char* chat_template_path;
    /** CUDA device ordinal. */
    int32_t device;
    /** Logical ceiling of one request, in tokens. Zero keeps the engine default. */
    uint32_t max_context;
    /** KV capacity in tokens. Zero follows `max_context`. */
    uint32_t max_kv_tokens;
    /** Concurrent requests admitted. Zero keeps the engine default. */
    uint32_t max_concurrency;
    /** Nonzero enables CUDA graph capture. Zero disables it. */
    uint8_t use_cuda_graph;
} ninfer_engine_options;

/**
 * Opens an engine over one artifact.
 *
 * @param options Required. `options->artifact_path` must be non-null.
 * @param out Required. Receives the handle on success and null on failure.
 * @param error Optional buffer for a NUL-terminated failure message; may be null.
 * @param error_bytes Capacity of `error`; a message longer than this is truncated.
 * @return NINFER_OK, or NINFER_INVALID_ARGUMENT, NINFER_NOT_FOUND, NINFER_RUNTIME_ERROR.
 */
NINFER_CAPI ninfer_status ninfer_engine_open(const ninfer_engine_options* options, ninfer_engine** out,
                                 char* error, size_t error_bytes);

/** Releases an engine. Passing null is a no-op, so this is safe on a failed open. */
NINFER_CAPI void ninfer_engine_close(ninfer_engine* engine);

/**
 * Encodes raw text with the artifact's tokenizer. No chat template or special token is added.
 *
 * @param out_tokens Caller-allocated array of `capacity` entries; may be null when `capacity` is 0.
 * @param out_count Required. Receives the token count produced, whether or not it fit.
 * @return NINFER_BUFFER_TOO_SMALL when `capacity` is below `*out_count`; nothing is written then.
 */
NINFER_CAPI ninfer_status ninfer_tokenize(ninfer_engine* engine, const char* text, int32_t* out_tokens,
                              size_t capacity, size_t* out_count, char* error, size_t error_bytes);

/**
 * Generates greedily from a token prefix: temperature 0, no penalties, model stop tokens honoured.
 *
 * Determinism is the engine's stated contract, so the same artifact, prefix and `max_new_tokens`
 * produce the same ids — which is what makes this callable an oracle for another implementation.
 *
 * @param tokens Required prefix of `token_count` ids; `token_count` must be positive.
 * @param max_new_tokens Positive number of tokens to generate.
 * @param out_tokens Caller-allocated array of `capacity` entries; may be null when `capacity` is 0.
 * @param out_count Required. Receives the number generated, whether or not it fit.
 * @return NINFER_BUFFER_TOO_SMALL when `capacity` is below `*out_count`; nothing is written then.
 */
NINFER_CAPI ninfer_status ninfer_generate_greedy(ninfer_engine* engine, const int32_t* tokens,
                                     size_t token_count, uint32_t max_new_tokens,
                                     int32_t* out_tokens, size_t capacity, size_t* out_count,
                                     char* error, size_t error_bytes);

/**
 * Renders token ids as bytes with the artifact's tokenizer: each id's bytes concatenated in
 * order, special tokens included and no stop token trimmed.
 *
 * The bytes are not UTF-8 validated and not NUL-terminated: a generation budget can end inside a
 * multi-byte character, and an id's bytes may contain a zero byte, so the caller receives exactly
 * what the ids encode and the length is the only terminator.
 *
 * @param tokens Ids to render; may be null when `token_count` is 0, which renders zero bytes.
 * @param out_bytes Caller-allocated array of `capacity` bytes; may be null when `capacity` is 0.
 * @param out_length Required. Receives the byte length produced, whether or not it fit.
 * @return NINFER_INVALID_ARGUMENT for an id outside the vocabulary; NINFER_BUFFER_TOO_SMALL when
 *         `capacity` is below `*out_length`, and nothing is written then.
 */
NINFER_CAPI ninfer_status ninfer_detokenize(ninfer_engine* engine, const int32_t* tokens,
                                            size_t token_count, char* out_bytes, size_t capacity,
                                            size_t* out_length, char* error, size_t error_bytes);

#ifdef __cplusplus
} // extern "C"
#endif

#endif // NINFER_CAPI_H
