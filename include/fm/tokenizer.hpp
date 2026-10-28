#pragma once

#include "fm/common.hpp"
#include "fm/gguf.hpp"

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace fm {

// Real tokenizers for real checkpoints.
//
// The reference adapter shipped with a raw-byte tokenizer, which is enough for a
// synthetic fixture and useless for a published model: every vendor checkpoint
// carries either SentencePiece-derived pieces (`tokenizer.ggml.model = "llama"`)
// or a GPT-2 style BPE table (`"gpt2"`). Without one, generation produces ids
// that cannot be turned back into text, which is how this gap showed up: the
// model computed correct logits and then failed at detokenisation.
class Tokenizer {
public:
    enum class Kind { Byte, SentencePiece, Bpe };

    // Builds from GGUF tokenizer metadata. `vocab` is the declared vocabulary
    // size and is validated against the token table rather than trusted.
    static Tokenizer from_gguf(const GGUFDocument &doc, uint32_t vocab);

    Kind kind() const { return kind_; }
    const std::string &kind_name() const { return kind_name_; }

    // Encodes text into token ids. SentencePiece greedy longest-match with
    // byte fallback, which is deterministic and matches the piece table.
    std::vector<int32_t> encode(const std::string &text, uint32_t bos_id) const;

    // Decodes ids back to text. Special tokens are skipped; byte-fallback
    // pieces are emitted as raw bytes.
    std::string decode(const std::vector<int32_t> &ids, uint32_t bos_id,
                       uint32_t eos_id) const;

    // True only when every one of the 256 bytes can be represented, so encoding
    // arbitrary input cannot fail. A missing byte and a missing merge list are
    // different gaps and both are reported honestly.
    bool complete() const { return byte_coverage_complete_ && !(kind_ == Kind::Bpe && merges_.empty()); }

private:
    Kind kind_ = Kind::Byte;
    std::string kind_name_ = "byte";
    uint32_t vocab_ = 0;
    std::vector<std::string> pieces_;
    std::vector<float> scores_;
    // First byte of each piece -> candidate token ids, so the greedy match does
    // not scan the whole vocabulary for every position.
    std::unordered_map<unsigned char, std::vector<int32_t>> by_first_byte_;
    std::unordered_map<std::string, int32_t> piece_to_id_;
    std::vector<std::pair<std::string, std::string>> merges_;
    std::unordered_map<std::string, int32_t> merge_rank_;
    // Ids of the <0xNN> byte-fallback pieces, or -1 when absent.
    std::array<int32_t, 256> byte_ids_{};
    uint32_t max_piece_bytes_ = 0;
    bool has_byte_fallback_ = false;
    // Which raw bytes a piece or byte-fallback entry can represent.
    std::array<bool, 256> byte_covered_{};
    bool byte_coverage_complete_ = false;
};

} // namespace fm
