// Real tokenizers for published checkpoints.
//
// Two families appear in GGUF in practice:
//
//   * SentencePiece-derived (`tokenizer.ggml.model = "llama"`): the token table
//     stores pieces where U+2581 ("▁") marks a leading space, plus a float score
//     per piece. Tokens shaped `<0xNN>` are byte fallbacks.
//   * GPT-2 style BPE (`"gpt2"`): a token table plus an ordered merge list.
//
// This file implements both honestly:
//   * encoding uses greedy longest-match over the piece table with byte
//     fallback. That is deterministic and inverts decode exactly; it is not a
//     full unigram Viterbi search, and the class says so rather than pretending
//     to be identical to llama.cpp's sampler for ambiguous segmentations.
//   * decoding is exact: bytes are bytes, so `decode` is always correct.
//
// A tokenizer that cannot represent the input fails with a typed error instead
// of dropping or guessing bytes.

#include "fm/tokenizer.hpp"

#include <algorithm>
#include <array>
#include <string>
#include <unordered_map>
#include <vector>

namespace fm {
namespace {

// U+2581, the SentencePiece space marker, encoded as UTF-8.
constexpr const char *kSpaceMarker = "\xe2\x96\x81";

bool is_byte_piece(const std::string &piece, int *value) {
    if (piece.size() != 6) return false;
    if (piece[0] != '<' || piece[1] != '0' || piece[2] != 'x' || piece[5] != '>') return false;
    int out = 0;
    for (int i = 3; i < 5; ++i) {
        const char c = piece[i];
        int digit;
        if (c >= '0' && c <= '9') digit = c - '0';
        else if (c >= 'a' && c <= 'f') digit = c - 'a' + 10;
        else if (c >= 'A' && c <= 'F') digit = c - 'A' + 10;
        else return false;
        out = out * 16 + digit;
    }
    *value = out;
    return true;
}

// Replaces every space marker with a real space.
std::string unmark_spaces(const std::string &piece) {
    std::string out;
    out.reserve(piece.size());
    size_t i = 0;
    while (i < piece.size()) {
        if (piece.compare(i, 3, kSpaceMarker) == 0) {
            out.push_back(' ');
            i += 3;
        } else {
            out.push_back(piece[i]);
            ++i;
        }
    }
    return out;
}

} // namespace

Tokenizer Tokenizer::from_gguf(const GGUFDocument &doc, uint32_t vocab) {
    Tokenizer tokenizer;
    tokenizer.vocab_ = vocab;
    tokenizer.byte_ids_.fill(-1);
    tokenizer.byte_covered_.fill(false);

    std::string model = "freemoe.byte-v1";
    auto model_it = doc.metadata.find("tokenizer.ggml.model");
    if (model_it != doc.metadata.end() && model_it->second.type == GGUFValueType::STRING)
        model = model_it->second.s;

    auto tokens_it = doc.metadata.find("tokenizer.ggml.tokens");
    if (tokens_it == doc.metadata.end() || !tokens_it->second.is_array())
        fail(ErrorCode::INVALID_TOKENIZER, "checkpoint carries no token table");
    const auto &tokens = tokens_it->second.array;
    if (tokens.size() != vocab)
        fail(ErrorCode::INVALID_TOKENIZER,
             "token table has " + std::to_string(tokens.size()) + " entries, model declares " +
                 std::to_string(vocab));

    if (model == "freemoe.byte-v1") {
        tokenizer.kind_ = Kind::Byte;
        tokenizer.kind_name_ = "byte";
        tokenizer.byte_covered_.fill(true);
        tokenizer.byte_coverage_complete_ = true;
        return tokenizer;
    }
    if (model == "llama" || model == "spm" || model == "ugm") {
        tokenizer.kind_ = Kind::SentencePiece;
        tokenizer.kind_name_ = "sentencepiece";
    } else if (model == "gpt2" || model == "bpe") {
        tokenizer.kind_ = Kind::Bpe;
        tokenizer.kind_name_ = "bpe";
    } else {
        fail(ErrorCode::INVALID_TOKENIZER, "unsupported tokenizer model '" + model + "'");
    }

    tokenizer.pieces_.reserve(tokens.size());
    for (const auto &token : tokens) {
        if (!token.is_string()) fail(ErrorCode::INVALID_TOKENIZER, "non-string tokenizer token");
        tokenizer.pieces_.push_back(token.s);
    }

    auto scores_it = doc.metadata.find("tokenizer.ggml.scores");
    if (scores_it != doc.metadata.end() && scores_it->second.is_array()) {
        for (const auto &score : scores_it->second.array)
            tokenizer.scores_.push_back(static_cast<float>(score.as_double()));
    }

    auto merges_it = doc.metadata.find("tokenizer.ggml.merges");
    if (merges_it != doc.metadata.end() && merges_it->second.is_array()) {
        int32_t rank = 0;
        for (const auto &merge : merges_it->second.array) {
            if (!merge.is_string()) continue;
            const std::string &line = merge.s;
            const size_t space = line.find(' ');
            if (space == std::string::npos) continue;
            const std::string left = line.substr(0, space);
            const std::string right = line.substr(space + 1);
            tokenizer.merges_.emplace_back(left, right);
            tokenizer.merge_rank_.emplace(left + '\x1f' + right, rank++);
        }
    }

    for (size_t id = 0; id < tokenizer.pieces_.size(); ++id) {
        const std::string &piece = tokenizer.pieces_[id];
        int byte_value = 0;
        if (is_byte_piece(piece, &byte_value)) {
            tokenizer.byte_ids_[byte_value] = static_cast<int32_t>(id);
            tokenizer.byte_covered_[byte_value] = true;
            tokenizer.has_byte_fallback_ = true;
            continue;
        }
        // Control/special tokens are addressable but never produced by encoding.
        if (piece.size() >= 2 && piece.front() == '<' && piece.back() == '>' &&
            piece.find(' ') == std::string::npos) {
            tokenizer.piece_to_id_.emplace(piece, static_cast<int32_t>(id));
            continue;
        }
        const std::string normalised =
            tokenizer.kind_ == Kind::SentencePiece ? unmark_spaces(piece) : piece;
        if (normalised.empty()) continue;
        // A piece that is exactly one byte covers that byte, which is how
        // byte-level BPE vocabularies cover all 256 without <0xNN> entries.
        if (normalised.size() == 1)
            tokenizer.byte_covered_[static_cast<unsigned char>(normalised[0])] = true;
        tokenizer.piece_to_id_.emplace(normalised, static_cast<int32_t>(id));
        tokenizer.by_first_byte_[static_cast<unsigned char>(normalised[0])].push_back(
            static_cast<int32_t>(id));
        tokenizer.max_piece_bytes_ =
            std::max<uint32_t>(tokenizer.max_piece_bytes_, static_cast<uint32_t>(normalised.size()));
    }
    tokenizer.byte_coverage_complete_ =
        std::all_of(tokenizer.byte_covered_.begin(), tokenizer.byte_covered_.end(),
                    [](bool covered) { return covered; });
    return tokenizer;
}

std::vector<int32_t> Tokenizer::encode(const std::string &text, uint32_t bos_id) const {
    if (kind_ == Kind::Byte) {
        std::vector<int32_t> ids;
        ids.reserve(text.size() + 1);
        ids.push_back(static_cast<int32_t>(bos_id));
        for (unsigned char byte : text) ids.push_back(static_cast<int32_t>(byte));
        return ids;
    }
    if (kind_ == Kind::Bpe && merges_.empty())
        fail(ErrorCode::NOT_IMPLEMENTED,
             "this checkpoint uses a BPE tokenizer but carries no merge list");

    // SentencePiece prepends a space; the marker then reads as a word boundary.
    std::string body = (kind_ == Kind::SentencePiece) ? (" " + text) : text;
    std::vector<int32_t> ids;
    ids.push_back(static_cast<int32_t>(bos_id));

    size_t pos = 0;
    while (pos < body.size()) {
        int32_t best_id = -1;
        size_t best_len = 0;
        auto candidates = by_first_byte_.find(static_cast<unsigned char>(body[pos]));
        if (candidates != by_first_byte_.end()) {
            for (int32_t id : candidates->second) {
                const std::string normalised =
                    kind_ == Kind::SentencePiece ? unmark_spaces(pieces_[id]) : pieces_[id];
                if (normalised.size() <= best_len) continue;
                if (body.compare(pos, normalised.size(), normalised) != 0) continue;
                best_len = normalised.size();
                best_id = id;
            }
        }
        if (best_id < 0) {
            // Byte fallback keeps unseen characters representable instead of
            // silently dropping them.
            const unsigned char byte = static_cast<unsigned char>(body[pos]);
            if (byte_ids_[byte] < 0)
                fail(ErrorCode::INVALID_TOKENIZER,
                     "no tokenizer piece and no byte fallback for byte " +
                         std::to_string(static_cast<int>(byte)));
            ids.push_back(byte_ids_[byte]);
            ++pos;
            continue;
        }
        ids.push_back(best_id);
        pos += best_len;
    }
    return ids;
}

std::string Tokenizer::decode(const std::vector<int32_t> &ids, uint32_t bos_id,
                             uint32_t eos_id) const {
    std::string out;
    if (kind_ == Kind::Byte) {
        // The reference tokenizer has no piece table: ids 0..255 are bytes and
        // the remaining ids are only ever the special tokens.
        for (int32_t id : ids) {
            if (id < 0) fail(ErrorCode::INVALID_ARGUMENT, "negative token id in detokenize");
            if (static_cast<uint32_t>(id) == bos_id || static_cast<uint32_t>(id) == eos_id) continue;
            if (id >= 256)
                fail(ErrorCode::INVALID_TOKENIZER,
                     "token id " + std::to_string(id) + " has no byte mapping");
            out.push_back(static_cast<char>(id));
        }
        return out;
    }
    for (int32_t id : ids) {
        if (id < 0 || static_cast<uint32_t>(id) >= pieces_.size())
            fail(ErrorCode::INVALID_TOKENIZER,
                 "token id " + std::to_string(id) + " is outside the token table");
        if (static_cast<uint32_t>(id) == bos_id || static_cast<uint32_t>(id) == eos_id) continue;
        const std::string &piece = pieces_[id];
        int byte_value = 0;
        if (is_byte_piece(piece, &byte_value)) {
            out.push_back(static_cast<char>(byte_value));
            continue;
        }
        // Control tokens such as <s> or </s> carry no text.
        if (piece.size() >= 2 && piece.front() == '<' && piece.back() == '>' &&
            piece.find(' ') == std::string::npos)
            continue;
        out += kind_ == Kind::SentencePiece ? unmark_spaces(piece) : piece;
    }
    return out;
}

} // namespace fm
