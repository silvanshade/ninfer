#pragma once

#include "ninfer/types.h"

#include <cstdint>
#include <optional>
#include <span>

namespace ninfer {

// One round of a request's output, offered to the request's controller before the Engine decides
// how much of it to commit.
struct RoundOffer {
    // Tokens the target licensed this round, in order. A decode round of a speculative Engine
    // licenses its accepted drafts followed by the correction or bonus token; an ordinary decode
    // round and prefill finalization license one token.
    std::span<const TokenId> licensed;
    // True for a decode round; false for the token prefill finalization produced.
    bool decode_round = false;
};

// A controller's answer for one round. An empty limit leaves the round to the output policy. A
// limit below the tokens the request may still produce becomes the request's remaining output
// budget for this round: the Engine commits at most that many licensed tokens and finishes the
// request with FinishReason::OutputLimit when the limit binds. A limit of zero is invalid and
// fails the request, because a committed round always keeps at least one token.
struct RoundVerdict {
    std::optional<std::uint32_t> limit;
};

// Per-request observer and limiter of committed rounds, supplied to Engine::submit. The Engine
// holds it for the request's lifetime and calls it on its worker thread, once per round, before
// the output policy previews that round, so the controller sees every licensed token in order.
// review() runs on the Engine's critical path: it must return promptly, must not call back into
// the Engine, and must not throw.
class RoundController {
public:
    RoundController() noexcept                         = default;
    RoundController(const RoundController&)            = delete;
    RoundController& operator=(const RoundController&) = delete;
    RoundController(RoundController&&)                 = delete;
    RoundController& operator=(RoundController&&)      = delete;
    virtual ~RoundController()                         = default;

    [[nodiscard]] virtual RoundVerdict review(const RoundOffer& offer) noexcept = 0;
};

} // namespace ninfer
