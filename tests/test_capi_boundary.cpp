// The C boundary's own contract, exercised through the header a foreign host includes. Nothing
// here opens a device: these are the promises that hold before any artifact does.

#include "ninfer/capi.h"

#include <cstdint>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

namespace {

int check(bool condition, const char* message) {
    if (condition) { return 0; }
    std::cerr << message << '\n';
    return 1;
}

} // namespace

int main() {
    int failures = 0;

    // A missing artifact reports a status and a message rather than unwinding into the caller.
    ninfer_engine_options options{};
    options.artifact_path = "/nonexistent/path/to/an/artifact.ninfer";
    ninfer_engine* engine = nullptr;
    std::vector<char> message(256, '\xff');
    const ninfer_status missing =
        ninfer_engine_open(&options, &engine, message.data(), message.size());
    failures += check(missing != NINFER_OK, "opening a missing artifact reported success");
    failures += check(engine == nullptr, "a failed open still handed back a handle");
    failures += check(std::strlen(message.data()) > 0 && std::strlen(message.data()) < message.size(),
                      "a failed open left no NUL-terminated message");

    // Closing what a failed open produced is defined, so a caller needs no null test of its own.
    ninfer_engine_close(engine);
    ninfer_engine_close(nullptr);

    // A null artifact path is the caller's error, and it is distinguished from a missing file.
    ninfer_engine_options empty{};
    ninfer_engine* unused = nullptr;
    failures += check(ninfer_engine_open(&empty, &unused, nullptr, 0) == NINFER_INVALID_ARGUMENT,
                      "a null artifact_path was not rejected as an invalid argument");
    failures += check(ninfer_engine_open(&options, nullptr, nullptr, 0) == NINFER_INVALID_ARGUMENT,
                      "a null out parameter was not rejected");

    // A one-byte buffer still comes back NUL-terminated: truncation never runs off the end.
    char single = '\xff';
    ninfer_engine* discarded = nullptr;
    ninfer_engine_open(&options, &discarded, &single, 1);
    failures += check(single == '\0', "a one-byte error buffer was not NUL-terminated");
    ninfer_engine_close(discarded);

    // Request-side argument rules hold without an engine, because the boundary checks them first.
    std::size_t count = 42;
    std::int32_t token = 0;
    failures += check(ninfer_tokenize(nullptr, "hello", &token, 1, &count, nullptr, 0) ==
                          NINFER_INVALID_ARGUMENT,
                      "tokenize accepted a null engine");
    failures += check(ninfer_generate_greedy(nullptr, &token, 1, 1, &token, 1, &count, nullptr,
                                             0) == NINFER_INVALID_ARGUMENT,
                      "generate accepted a null engine");
    failures += check(ninfer_generate_greedy(nullptr, &token, 0, 1, &token, 1, &count, nullptr,
                                             0) == NINFER_INVALID_ARGUMENT,
                      "generate accepted an empty prefix");
    failures += check(ninfer_generate_greedy(nullptr, &token, 1, 0, &token, 1, &count, nullptr,
                                             0) == NINFER_INVALID_ARGUMENT,
                      "generate accepted a zero token budget");

    // Detokenize holds the same boundary: a null engine is the caller's error, reported with a
    // NUL-terminated message rather than dereferenced.
    char rendered = '\xff';
    std::vector<char> detokenize_message(256, '\xff');
    failures +=
        check(ninfer_detokenize(nullptr, &token, 1, &rendered, 1, &count, detokenize_message.data(),
                                detokenize_message.size()) == NINFER_INVALID_ARGUMENT,
              "detokenize accepted a null engine");
    failures += check(std::strlen(detokenize_message.data()) > 0 &&
                          std::strlen(detokenize_message.data()) < detokenize_message.size(),
                      "a rejected detokenize left no NUL-terminated message");
    failures += check(rendered == '\xff', "a rejected detokenize wrote into the output bytes");

    return failures == 0 ? 0 : 1;
}
