#include "ninfer/engine.h"
#include "ninfer/round_control.h"

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
void require(bool condition, const char* message) {
    if (!condition) { throw std::runtime_error(message); }
}

ninfer::RequestOptions request(std::uint32_t outputs) {
    ninfer::RequestOptions options;
    options.execution.requested_output_tokens = outputs;
    options.execution.sampling.temperature    = 0.0F;
    options.execution.allow_prefix_reuse      = false;
    options.stop.include_model_defaults       = false;
    return options;
}

// Records every offered round and optionally stops the request once a token total is reached.
class Recorder final : public ninfer::RoundController {
public:
    explicit Recorder(std::optional<std::uint32_t> stop_after) : stop_after_(stop_after) {}

    ninfer::RoundVerdict review(const ninfer::RoundOffer& offer) noexcept override {
        rounds.push_back(offer.licensed.size());
        decode_rounds += offer.decode_round ? 1U : 0U;
        licensed.insert(licensed.end(), offer.licensed.begin(), offer.licensed.end());
        if (!stop_after_) { return {}; }
        const std::uint32_t remaining = *stop_after_ - committed_;
        const auto offered            = static_cast<std::uint32_t>(offer.licensed.size());
        committed_ += offered < remaining ? offered : remaining;
        return {.limit = remaining};
    }

    std::vector<std::size_t> rounds;
    std::uint32_t decode_rounds = 0;
    std::vector<ninfer::TokenId> licensed;

private:
    std::optional<std::uint32_t> stop_after_;
    std::uint32_t committed_ = 0;
};
} // namespace

// Arguments: K, then the CUDA Graph switch. The artifact must carry a DFlash2 companion.
int main(int argc, char** argv) {
    const char* artifact = std::getenv("NINFER_TEST_ARTIFACT");
    if (!artifact || !*artifact) {
        std::cout << "skip: NINFER_TEST_ARTIFACT is not set\n";
        return 77;
    }
    try {
        const auto k     = argc > 1 ? static_cast<unsigned>(std::stoul(argv[1])) : 7U;
        const bool graph = argc > 2 ? std::stoi(argv[2]) != 0 : true;
        ninfer::EngineOptions options;
        options.artifact_path             = artifact;
        options.max_context               = 2304;
        options.kv_capacity               = ninfer::KvCapacityPolicy::explicit_capacity(2304);
        options.prefill_chunk             = 2304;
        options.max_concurrency           = 1;
        options.use_cuda_graph            = graph;
        options.speculative.backend       = ninfer::SpeculativeBackend::DFlash2;
        options.speculative.draft_tokens  = k;
        options.speculative.proposal_head = ninfer::ProposalHead::Optimized;
        ninfer::Engine engine(options);
        const auto prompt = engine.tokenize_text("Count from one to twenty: one, two, three,");

        const auto reference = engine.generate(engine.prepare_tokens(prompt), request(32));
        require(reference.generated_token_ids.size() == 32 &&
                    reference.speculative.accepted_tokens != 0,
                "reference DFlash2 run did not accept any proposal");

        // An observing controller changes nothing and sees every round in order.
        auto observer = std::make_shared<Recorder>(std::nullopt);
        const auto observed =
            engine.submit(engine.prepare_tokens(prompt), request(32),
                          ninfer::OutputConsumerMode::Aggregate, {}, {}, observer)
                .wait();
        require(observed.generated_token_ids == reference.generated_token_ids,
                "an observing controller changed the generated ids");
        require(observer->licensed.size() >= observed.generated_token_ids.size() &&
                    std::equal(observed.generated_token_ids.begin(),
                               observed.generated_token_ids.end(), observer->licensed.begin()),
                "the controller did not see the committed tokens in order");
        require(observer->decode_rounds == observed.speculative.rounds +
                                               observed.speculative.fallback_steps,
                "the controller did not see every decode round");
        bool multi_token_round = false;
        for (const auto size : observer->rounds) { multi_token_round |= size > 1; }
        require(multi_token_round, "no round offered more than one licensed token");

        // A limit inside a licensed block stops the request exactly there.
        const std::uint32_t stop_after = 5;
        auto limiter                   = std::make_shared<Recorder>(stop_after);
        const auto limited =
            engine.submit(engine.prepare_tokens(prompt), request(32),
                          ninfer::OutputConsumerMode::Aggregate, {}, {}, limiter)
                .wait();
        require(limited.finish_reason == ninfer::FinishReason::OutputLimit &&
                    limited.generated_token_ids.size() == stop_after,
                "a controller limit did not end the request at the limit");
        require(std::equal(limited.generated_token_ids.begin(), limited.generated_token_ids.end(),
                           reference.generated_token_ids.begin()),
                "a controller limit changed the committed prefix");

        std::cout << "ok K=" << k << " graph=" << graph << " rounds=" << observer->rounds.size()
                  << " accepted=" << observed.speculative.accepted_tokens << "/"
                  << observed.speculative.drafted_tokens << '\n';
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
