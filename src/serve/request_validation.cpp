#include "serve/request_validation.h"

#include <cctype>
#include <cmath>
#include <cstdint>
#include <limits>
#include <utility>

namespace ninfer::serve {

[[noreturn]] void bad_request(std::string message, std::string param, std::string code) {
    ApiError error;
    error.status  = 400;
    error.type    = "invalid_request_error";
    error.message = std::move(message);
    error.param   = std::move(param);
    error.code    = std::move(code);
    throw ApiException(std::move(error));
}

std::optional<int> optional_int(const RequestJson& object, const char* key) {
    if (!object.contains(key) || object.at(key).is_null()) { return std::nullopt; }
    const RequestJson& value = object.at(key);
    if (!value.is_number_integer()) { bad_request(std::string(key) + " must be an integer", key); }
    if (value.is_number_unsigned()) {
        const std::uint64_t converted = value.get<std::uint64_t>();
        if (converted > static_cast<std::uint64_t>(std::numeric_limits<int>::max())) {
            bad_request(std::string(key) + " is out of range", key);
        }
        return static_cast<int>(converted);
    }
    const std::int64_t converted = value.get<std::int64_t>();
    if (converted < std::numeric_limits<int>::min() ||
        converted > std::numeric_limits<int>::max()) {
        bad_request(std::string(key) + " is out of range", key);
    }
    return static_cast<int>(converted);
}

std::optional<double> optional_number(const RequestJson& object, const char* key) {
    if (!object.contains(key) || object.at(key).is_null()) { return std::nullopt; }
    if (!object.at(key).is_number()) { bad_request(std::string(key) + " must be a number", key); }
    const double value = object.at(key).get<double>();
    if (!std::isfinite(value)) { bad_request(std::string(key) + " must be finite", key); }
    return value;
}

bool optional_bool(const RequestJson& object, const char* key, bool fallback) {
    if (!object.contains(key) || object.at(key).is_null()) { return fallback; }
    if (!object.at(key).is_boolean()) { bad_request(std::string(key) + " must be a boolean", key); }
    return object.at(key).get<bool>();
}

// Parse phase overrides identically across serving protocols.
// # Specification
// - requires: body is a parsed request object.
// - ensures: omission/null preserves defaults; explicit zero and integer seed bits are retained.
// - fails: ApiException with status 400 for non-object, wrong scalar types or out-of-range fields.
// - panics: none.
SamplingParams parse_post_thinking_sampling(const RequestJson& body) {
    SamplingParams result;
    if (!body.contains("post_thinking") || body.at("post_thinking").is_null()) { return result; }
    const auto& object = body.at("post_thinking");
    if (!object.is_object()) { bad_request("post_thinking must be an object", "post_thinking"); }
    const auto number = [&](const char* key, double minimum, double maximum) {
        const auto value = optional_number(object, key);
        if (value && (*value < minimum || *value > maximum)) {
            bad_request(std::string("post_thinking.") + key + " is out of range",
                        std::string("post_thinking.") + key);
        }
        return value;
    };
    result.temperature       = number("temperature", 0.0, 2.0);
    result.top_p             = number("top_p", 0.0, 1.0);
    result.min_p             = number("min_p", 0.0, 1.0);
    result.presence_penalty  = number("presence_penalty", -2.0, 2.0);
    result.frequency_penalty = number("frequency_penalty", -2.0, 2.0);
    result.top_k             = optional_int(object, "top_k");
    if (result.top_k && (*result.top_k < 0 || *result.top_k > 20)) {
        bad_request("post_thinking.top_k must be in [0,20]", "post_thinking.top_k");
    }
    if (object.contains("seed") && !object.at("seed").is_null()) {
        const auto& seed = object.at("seed");
        if (!seed.is_number_integer()) {
            bad_request("post_thinking.seed must be an integer", "post_thinking.seed");
        }
        result.seed = seed.is_number_unsigned()
                          ? seed.get<std::uint64_t>()
                          : static_cast<std::uint64_t>(seed.get<std::int64_t>());
    }
    return result;
}

bool valid_tool_name(std::string_view name, std::size_t maximum_length) noexcept {
    if (name.empty() || name.size() > maximum_length) { return false; }
    for (const unsigned char character : name) {
        if (std::isalnum(character) == 0 && character != '_' && character != '-') { return false; }
    }
    return true;
}

} // namespace ninfer::serve
