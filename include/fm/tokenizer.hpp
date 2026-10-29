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

    // Pre-tokenisers implemented here. Byte-level BPE is not just "apply the
    // merges": the text is first cut into words by a family-specific rule, and
    // merges never cross a word boundary. Two families cover the llama-3 /
    // Qwen and GPT-2 / StarCoder lineages; anything else is refused by name
    // rather than segmented by the wrong rule.
    enum class PreTokenizer {
        Unspecified, // BPE file without `tokenizer.ggml.pre`
        Gpt2,        // 's|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)
        Gpt2Digits,  // the same, with each digit split off first (StarCoder family)
        Llama3,      // contractions, [^\r\n\p{L}\p{N}]?\p{L}+, \p{N}{1,3}, punctuation runs
        Qwen2,       // as llama-3 but a single digit per token
        Unsupported, // recognised but not implemented: refuse instead of guessing
    };

    // Builds from GGUF tokenizer metadata. `vocab` is the declared vocabulary
    // size and is validated against the token table rather than trusted.
    static Tokenizer from_gguf(const GGUFDocument &doc, uint32_t vocab);

    Kind kind() const { return kind_; }
    const std::string &kind_name() const { return kind_name_; }

    // Maps a `tokenizer.ggml.pre` value to the splitting rule it selects,
    // without building a tokenizer. The support probe uses it to report "this
    // file asks for a rule that is not implemented".
    static PreTokenizer pre_tokenizer_from_name(const std::string &name);
    static const char *pre_tokenizer_family_name(PreTokenizer family);

    // The pre-tokeniser actually in force, and the value the file declared.
    // Reported separately so the probe can say "this file asks for X, which is
    // not implemented" instead of silently segmenting with Y.
    PreTokenizer pre_tokenizer() const { return pre_; }
    const std::string &pre_tokenizer_name() const { return pre_name_; }
    const char *pre_tokenizer_family() const;

    // Encodes text into token ids.
    //
    //   * SentencePiece: the piece set's own scores, applied by the reference's
    //     bigram merge (see `encode_sentencepiece`), not greedy longest-match.
    //   * BPE: the family pre-tokeniser cuts words, then merges are applied in
    //     rank order, exactly as the reference tokenizer does. Byte fallback
    //     covers any byte the vocabulary cannot express as a piece.
    // `add_bos = false` encodes raw text without the leading BOS, which is what
    // reference parity vectors and chat-template rendering both need.
    std::vector<int32_t> encode(const std::string &text, uint32_t bos_id,
                                bool add_bos = true) const;

    // The literal piece text for an id, or an empty string when out of range.
    // Chat templates splice `bos_token`/`eos_token` into the prompt as text, so
    // the engine needs the piece itself rather than its id.
    std::string piece(uint32_t id) const;

    // True when the vocabulary contains this exact piece. Used to pick a chat
    // template from evidence (does the vocab carry `<|im_start|>`?) instead of
    // from the architecture name.
    bool contains_piece(const std::string &piece) const;

    // Decodes ids back to text. Special tokens are skipped; byte-fallback
    // pieces are emitted as raw bytes.
    std::string decode(const std::vector<int32_t> &ids, uint32_t bos_id,
                       uint32_t eos_id) const;

    // True only when every one of the 256 bytes can be represented, so encoding
    // arbitrary input cannot fail. A missing byte and a missing merge list are
    // different gaps and both are reported honestly.
    bool complete() const {
        if (!byte_coverage_complete_) return false;
        if (kind_ != Kind::Bpe) return true;
        if (merges_.empty()) return false;
        return pre_ != PreTokenizer::Unspecified && pre_ != PreTokenizer::Unsupported;
    }

private:
    Kind kind_ = Kind::Byte;
    std::string kind_name_ = "byte";
    uint32_t vocab_ = 0;
    std::vector<std::string> pieces_;
    std::vector<float> scores_;
    std::unordered_map<std::string, int32_t> piece_to_id_;
    std::vector<std::pair<std::string, std::string>> merges_;
    // "left\x1fright" -> rank. A std::string key rather than a pair key so the
    // hot path allocates nothing per lookup beyond the key itself.
    std::unordered_map<std::string, int32_t> merge_rank_;
    // BPE encoding: pre-tokenise, byte-encode, merge, look up.
    std::vector<int32_t> encode_bpe(const std::string &text, uint32_t bos_id, bool add_bos) const;
    // SentencePiece encoding: escape whitespace, then repeatedly merge the
    // adjacent pair whose concatenation has the highest piece score. That is the
    // reference's segmentation for a unigram/`llama` vocabulary; greedy
    // longest-match disagrees on ordinary text because a long low-score piece
    // beats the two high-score pieces the model was trained on.
    std::vector<int32_t> encode_sentencepiece(const std::string &text, uint32_t bos_id,
                                              bool add_bos) const;
    // The vocabulary holds SentencePiece pieces with the space marker, so
    // lookups happen in the escaped domain. `mark_spaces`/`unmark_spaces` are the
    // two directions; encoding works with raw spaces because every piece is
    // normalised on load.
    std::string normalize_spm(const std::string &text) const;
    // `tokenizer.ggml.add_space_prefix`, true for SentencePiece vocabularies.
    bool spm_add_space_prefix_ = true;
    // Merges one pre-tokenised word into pieces, given its byte-encoded form.
    std::vector<std::string> merge_word(const std::string &word) const;

    PreTokenizer pre_ = PreTokenizer::Unspecified;
    std::string pre_name_;
    // BPE: emit a whole-word token without merging when the vocabulary has it.
    bool ignore_merges_ = false;
    // Byte-level alphabet: one UTF-8 string per byte, used to turn raw bytes
    // into the characters a BPE vocabulary stores.
    std::array<std::string, 256> byte_text_{};
    // Ids of the <0xNN> byte-fallback pieces, or -1 when absent.
    std::array<int32_t, 256> byte_ids_{};
    uint32_t max_piece_bytes_ = 0;
    bool has_byte_fallback_ = false;
    // Which raw bytes a piece or byte-fallback entry can represent.
    std::array<bool, 256> byte_covered_{};
    bool byte_coverage_complete_ = false;
};

} // namespace fm
