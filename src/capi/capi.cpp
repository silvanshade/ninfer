#include "ninfer/capi.h"

#include "ninfer/engine.h"
#include "ninfer/types.h"

#include <cstring>
#include <exception>
#include <filesystem>
#include <new>
#include <stdexcept>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

namespace {

struct EngineHolder {
    explicit EngineHolder(ninfer::EngineOptions options) : engine(std::move(options)) {}
    ninfer::Engine engine;
};

void write_message(char* error, std::size_t error_bytes, const char* text) {
    if (error == nullptr || error_bytes == 0) { return; }
    const auto length = std::strlen(text);
    const auto copied = length < error_bytes - 1 ? length : error_bytes - 1;
    std::memcpy(error, text, copied);
    error[copied] = '\0';
}

// One translation of a thrown failure into a status plus a message, shared by every entry point:
// the boundary is the only place that knows both a C++ exception and a C status exist.
template <typename Body> ninfer_status guarded(char* error, std::size_t error_bytes, Body&& body) {
    try {
        return std::forward<Body>(body)();
    } catch (const std::filesystem::filesystem_error& failure) {
        write_message(error, error_bytes, failure.what());
        return NINFER_NOT_FOUND;
    } catch (const std::system_error& failure) {
        write_message(error, error_bytes, failure.what());
        return failure.code() == std::errc::no_such_file_or_directory ? NINFER_NOT_FOUND
                                                                      : NINFER_RUNTIME_ERROR;
    } catch (const std::invalid_argument& failure) {
        write_message(error, error_bytes, failure.what());
        return NINFER_INVALID_ARGUMENT;
    } catch (const std::bad_alloc& failure) {
        write_message(error, error_bytes, failure.what());
        return NINFER_RUNTIME_ERROR;
    } catch (const std::exception& failure) {
        write_message(error, error_bytes, failure.what());
        return NINFER_RUNTIME_ERROR;
    } catch (...) {
        write_message(error, error_bytes, "ninfer: unknown failure");
        return NINFER_UNKNOWN_ERROR;
    }
}

// Copies a produced id sequence out under the caller's capacity, reporting the length either way.
ninfer_status deliver(const std::vector<ninfer::TokenId>& produced, std::int32_t* out_tokens,
                      std::size_t capacity, std::size_t* out_count) {
    *out_count = produced.size();
    if (produced.size() > capacity) { return NINFER_BUFFER_TOO_SMALL; }
    if (!produced.empty()) {
        std::memcpy(out_tokens, produced.data(), produced.size() * sizeof(std::int32_t));
    }
    return NINFER_OK;
}

} // namespace

extern "C" {

ninfer_status ninfer_engine_open(const ninfer_engine_options* options, ninfer_engine** out,
                                 char* error, std::size_t error_bytes) {
    if (out == nullptr) { return NINFER_INVALID_ARGUMENT; }
    *out = nullptr;
    if (options == nullptr || options->artifact_path == nullptr) {
        write_message(error, error_bytes, "ninfer: artifact_path is required");
        return NINFER_INVALID_ARGUMENT;
    }
    return guarded(error, error_bytes, [&]() -> ninfer_status {
        ninfer::EngineOptions settings;
        settings.artifact_path = std::filesystem::path{options->artifact_path};
        if (options->chat_template_path != nullptr) {
            settings.chat_template_path = std::filesystem::path{options->chat_template_path};
        }
        settings.device = options->device;
        if (options->max_context != 0) { settings.max_context = options->max_context; }
        const auto kv_tokens =
            options->max_kv_tokens != 0 ? options->max_kv_tokens : settings.max_context;
        settings.kv_capacity = ninfer::KvCapacityPolicy::explicit_capacity(kv_tokens);
        if (options->max_concurrency != 0) { settings.max_concurrency = options->max_concurrency; }
        settings.use_cuda_graph = options->use_cuda_graph != 0;

        auto holder = std::make_unique<EngineHolder>(std::move(settings));
        *out        = reinterpret_cast<ninfer_engine*>(holder.release());
        return NINFER_OK;
    });
}

void ninfer_engine_close(ninfer_engine* engine) {
    delete reinterpret_cast<EngineHolder*>(engine);
}

ninfer_status ninfer_tokenize(ninfer_engine* engine, const char* text, std::int32_t* out_tokens,
                              std::size_t capacity, std::size_t* out_count, char* error,
                              std::size_t error_bytes) {
    if (engine == nullptr || text == nullptr || out_count == nullptr ||
        (out_tokens == nullptr && capacity != 0)) {
        write_message(error, error_bytes, "ninfer: engine, text and out_count are required");
        return NINFER_INVALID_ARGUMENT;
    }
    auto* holder = reinterpret_cast<EngineHolder*>(engine);
    return guarded(error, error_bytes, [&]() -> ninfer_status {
        return deliver(holder->engine.tokenize_text(text), out_tokens, capacity, out_count);
    });
}

ninfer_status ninfer_generate_greedy(ninfer_engine* engine, const std::int32_t* tokens,
                                     std::size_t token_count, std::uint32_t max_new_tokens,
                                     std::int32_t* out_tokens, std::size_t capacity,
                                     std::size_t* out_count, char* error,
                                     std::size_t error_bytes) {
    if (engine == nullptr || tokens == nullptr || out_count == nullptr || token_count == 0 ||
        max_new_tokens == 0 || (out_tokens == nullptr && capacity != 0)) {
        write_message(error, error_bytes,
                      "ninfer: engine, a nonempty prefix, a positive budget and out_count are "
                      "required");
        return NINFER_INVALID_ARGUMENT;
    }
    auto* holder = reinterpret_cast<EngineHolder*>(engine);
    return guarded(error, error_bytes, [&]() -> ninfer_status {
        auto prompt = holder->engine.prepare_tokens(
            std::vector<ninfer::TokenId>(tokens, tokens + token_count));

        ninfer::RequestOptions request;
        request.execution.requested_output_tokens = max_new_tokens;
        // Greedy is the oracle contract: argmax, no penalties, a fixed seed that cannot matter.
        request.execution.sampling.temperature       = 0.0F;
        request.execution.sampling.top_k             = 0;
        request.execution.sampling.top_p             = 1.0F;
        request.execution.sampling.min_p             = 0.0F;
        request.execution.sampling.presence_penalty  = 0.0F;
        request.execution.sampling.frequency_penalty = 0.0F;
        request.execution.sampling.seed              = 0;

        auto result = holder->engine.generate(std::move(prompt), std::move(request));
        return deliver(result.generated_token_ids, out_tokens, capacity, out_count);
    });
}

} // extern "C"
