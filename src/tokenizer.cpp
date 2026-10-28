// Real tokenizers for published checkpoints.
//
// Two families appear in GGUF in practice:
//
//   * SentencePiece-derived (`tokenizer.ggml.model = "llama"`): the token table
//     stores pieces where U+2581 ("▁") marks a leading space, plus a float score
//     per piece. Tokens shaped `<0xNN>` are byte fallbacks.
//   * GPT-2 style BPE (`"gpt2"`): a token table plus an ordered merge list.
//
// SentencePiece encoding is greedy longest-match over the piece table with
// byte fallback. That is deterministic and inverts decode exactly; it is not a
// full unigram Viterbi search, and the class says so rather than pretending to
// be identical to llama.cpp's sampler for ambiguous segmentations.
//
// BPE encoding is the real algorithm: a family-specific pre-tokeniser cuts the
// text into words, every byte is mapped through the byte-level alphabet, and
// merges are applied lowest-rank-first. Max-match over the piece table — what
// this file used to do — produces different ids on ordinary text for every
// llama-3, Qwen or GPT-2 checkpoint, so a model could compute perfect logits
// and still answer wrongly. The two pre-tokeniser families implemented here are
// the ones covering those vocabularies; a file asking for a different family is
// refused by name rather than segmented by the wrong rule.
//
// Decoding is exact in both: bytes are bytes, so `decode` is always correct.
//
// A tokenizer that cannot represent the input fails with a typed error instead
// of dropping or guessing bytes.

#include "fm/tokenizer.hpp"

#include "fm/unicode.hpp"

#include <algorithm>
#include <array>
#include <climits>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace fm {
namespace {

// U+2581, the SentencePiece space marker, encoded as UTF-8.
constexpr const char *kSpaceMarker = "\xe2\x96\x81";

// GGUF stores the pre-tokeniser as a name (`tokenizer.ggml.pre`). The names map
// onto the splitter families implemented here; a name that is recognised but
// not implemented maps to Unsupported so the caller refuses instead of
// segmenting by a rule the checkpoint did not ask for.
Tokenizer::PreTokenizer lookup_pre_tokenizer(const std::string &name) {
    using Pre = Tokenizer::PreTokenizer;
    static const std::pair<const char *, Pre> table[] = {
        // llama-3 and the vocabularies that share its splitting rule
        {"llama3", Pre::Llama3},       {"llama-v3", Pre::Llama3},
        {"llama-bpe", Pre::Llama3},    {"chatglm4", Pre::Llama3},
        {"dbrx", Pre::Llama3},         {"smaug", Pre::Llama3},
        {"falcon3", Pre::Llama3},      {"falcon-h1", Pre::Llama3},
        {"midm-2.0", Pre::Llama3},     {"jina-v5-nano", Pre::Llama3},
        {"lfm2", Pre::Llama3},
        // one digit at a time, letters glued to a preceding non-space
        {"qwen2", Pre::Qwen2},         {"deepseek-r1-qwen", Pre::Qwen2},
        {"stablelm2", Pre::Qwen2},     {"hunyuan", Pre::Qwen2},
        {"solar-open", Pre::Qwen2},    {"kormo", Pre::Qwen2},
        {"f2llmv2", Pre::Qwen2},
        // GPT-2 splitting rule, whole digit runs
        {"gpt-2", Pre::Gpt2},          {"phi-2", Pre::Gpt2},
        {"mpt", Pre::Gpt2},            {"olmo", Pre::Gpt2},
        {"jais", Pre::Gpt2},           {"trillion", Pre::Gpt2},
        {"granite-docling", Pre::Gpt2}, {"gigachat", Pre::Gpt2},
        {"mellum", Pre::Gpt2},         {"modern-bert", Pre::Gpt2},
        {"jina-es", Pre::Gpt2},        {"jina-de", Pre::Gpt2},
        {"jina-v2-es", Pre::Gpt2},     {"jina-v2-de", Pre::Gpt2},
        {"a.x-4.0", Pre::Gpt2},
        // GPT-2 splitting rule with every digit split off on its own
        {"starcoder", Pre::Gpt2Digits}, {"refact", Pre::Gpt2Digits},
        {"command-r", Pre::Gpt2Digits}, {"smollm", Pre::Gpt2Digits},
        {"codeshell", Pre::Gpt2Digits}, {"exaone", Pre::Gpt2Digits},
        {"minerva", Pre::Gpt2Digits},   {"mellum2", Pre::Gpt2Digits},
        // Recognised vocabularies whose splitting rule is not implemented here.
        {"deepseek-llm", Pre::Unsupported},   {"deepseek-coder", Pre::Unsupported},
        {"deepseek-v3", Pre::Unsupported},    {"falcon", Pre::Unsupported},
        {"minicpm5", Pre::Unsupported},       {"gemma4", Pre::Unsupported},
        {"mmbert", Pre::Unsupported},         {"tekken", Pre::Unsupported},
        {"youtu", Pre::Unsupported},          {"spark2_5", Pre::Unsupported},
        {"kimi-k2", Pre::Unsupported},        {"gorilla", Pre::Unsupported},
        {"whitespace", Pre::Unsupported},     {"sarvam-moe", Pre::Unsupported},
    };
    if (name.empty()) return Pre::Unspecified;
    for (const auto &[key, value] : table) {
        if (name == key) return value;
    }
    return Pre::Unsupported;
}


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

// Where a word starts and ends, in code points.
struct WordSpan {
    size_t begin = 0;
    size_t end = 0;
};

// Sentinel for "outside the text", matching the reference splitter's
// out-of-range code point: it carries no Unicode flags, so every class test
// against it fails and every run stops there.
constexpr uint32_t kOutOfRange = 0xFFFFFFFFu;

// The GPT-2 pre-tokeniser:
//
//   's|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)
//
// with an optional rule that splits each digit off on its own (the StarCoder
// and SmolLM lineages add `\p{N}` ahead of the rest).
std::vector<WordSpan> split_gpt2(const std::vector<uint32_t> &cpts, bool split_digits) {
    const size_t size = cpts.size();
    auto at = [&](size_t pos) -> uint32_t { return pos < size ? cpts[pos] : kOutOfRange; };
    auto letter = [&](size_t pos) { return unicode::is_letter(at(pos)); };
    auto number = [&](size_t pos) { return unicode::is_number(at(pos)); };
    auto space = [&](size_t pos) { return unicode::is_space(at(pos)); };
    auto other = [&](size_t pos) {
        const uint32_t cpt = at(pos);
        if (cpt == kOutOfRange) return false;
        return !unicode::is_space(cpt) && !unicode::is_letter(cpt) && !unicode::is_number(cpt);
    };

    std::vector<WordSpan> words;
    size_t start = 0;
    size_t pos = 0;
    auto emit = [&](size_t end) {
        if (end > start) words.push_back({start, end});
        start = end;
    };

    while (pos < size) {
        const uint32_t cpt = at(pos);

        // Contractions. The reference table is case-sensitive here because the
        // GGUF pre-tokenizer name selects that variant.
        if (cpt == '\'' && pos + 1 < size) {
            const uint32_t next = unicode::ascii_lower(at(pos + 1));
            if (next == 's' || next == 't' || next == 'm' || next == 'd') {
                pos += 2;
                emit(pos);
                continue;
            }
            if (pos + 2 < size) {
                const uint32_t next_next = unicode::ascii_lower(at(pos + 2));
                if ((next == 'r' && next_next == 'e') || (next == 'v' && next_next == 'e') ||
                    (next == 'l' && next_next == 'l')) {
                    pos += 3;
                    emit(pos);
                    continue;
                }
            }
        }

        // The digit rule is the first pattern, so it matches at the first digit
        // even when that digit is preceded by a space: the space then falls out
        // as its own word instead of joining " ?\p{N}+".
        if (split_digits && cpt == ' ' && number(pos + 1)) {
            emit(++pos);
            continue;
        }
        if (split_digits && number(pos)) {
            emit(++pos);
            continue;
        }

        // " ?\p{L}+" and " ?\p{N}+" and " ?[^\s\p{L}\p{N}]+": the optional
        // leading space belongs to the word it precedes.
        const bool after_space = cpt == ' ';
        if (after_space ? letter(pos + 1) : letter(pos)) {
            pos += after_space ? 1 : 0;
            while (letter(pos)) ++pos;
            emit(pos);
            continue;
        }
        if (after_space ? number(pos + 1) : number(pos)) {
            pos += after_space ? 1 : 0;
            while (number(pos)) ++pos;
            emit(pos);
            continue;
        }
        if (after_space ? other(pos + 1) : other(pos)) {
            pos += after_space ? 1 : 0;
            while (other(pos)) ++pos;
            emit(pos);
            continue;
        }

        size_t spaces = 0;
        while (space(pos + spaces)) ++spaces;
        // "\s+(?!\S)": trailing whitespace at the end of the text is one word,
        // while leading whitespace leaves its last space to the next word.
        if (spaces > 1 && at(pos + spaces) != kOutOfRange) {
            pos += spaces - 1;
            emit(pos);
            continue;
        }
        if (spaces > 0) {
            pos += spaces;
            emit(pos);
            continue;
        }
        emit(++pos);
    }
    return words;
}

// The llama-3 pre-tokeniser (`llama-bpe`), and with `max_digits == 1` the Qwen2
// variant:
//
//   (?:'[sS]|...)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,max}| ?[^\s\p{L}\p{N}]+[\r\n]*|
//   \s*[\r\n]+|\s+(?!\S)|\s+
std::vector<WordSpan> split_llama3(const std::vector<uint32_t> &cpts, size_t max_digits) {
    const size_t size = cpts.size();
    auto at = [&](size_t pos) -> uint32_t { return pos < size ? cpts[pos] : kOutOfRange; };
    auto letter = [&](size_t pos) { return unicode::is_letter(at(pos)); };
    auto number = [&](size_t pos) { return unicode::is_number(at(pos)); };
    auto space = [&](size_t pos) { return unicode::is_space(at(pos)); };
    auto other = [&](size_t pos) {
        const uint32_t cpt = at(pos);
        if (cpt == kOutOfRange) return false;
        return !unicode::is_space(cpt) && !unicode::is_letter(cpt) && !unicode::is_number(cpt);
    };

    std::vector<WordSpan> words;
    size_t start = 0;
    size_t pos = 0;
    auto emit = [&](size_t end) {
        if (end > start) words.push_back({start, end});
        start = end;
    };

    while (pos < size) {
        const uint32_t cpt = at(pos);

        // Case-insensitive contractions: "'S" and "'S" are both a word.
        if (cpt == '\'' && pos + 1 < size) {
            const uint32_t next = unicode::ascii_lower(at(pos + 1));
            if (next == 's' || next == 't' || next == 'm' || next == 'd') {
                pos += 2;
                emit(pos);
                continue;
            }
            if (pos + 2 < size) {
                const uint32_t next_next = unicode::ascii_lower(at(pos + 2));
                if ((next == 'r' && next_next == 'e') || (next == 'v' && next_next == 'e') ||
                    (next == 'l' && next_next == 'l')) {
                    pos += 3;
                    emit(pos);
                    continue;
                }
            }
        }

        // "[^\r\n\p{L}\p{N}]?\p{L}+": one character that is not CR, LF, a
        // letter or a number may be glued to the letter run, so " hello" and
        // "'hello" are single words.
        if (cpt != '\r' && cpt != '\n' && !number(pos) && (letter(pos) || letter(pos + 1))) {
            ++pos;
            while (letter(pos)) ++pos;
            emit(pos);
            continue;
        }

        // "\p{N}{1,max}": digits are grouped, three at a time for llama-3.
        if (number(pos)) {
            size_t run = 0;
            while (number(pos)) {
                ++pos;
                if (++run == max_digits) {
                    emit(pos);
                    run = 0;
                }
            }
            emit(pos);
            continue;
        }

        // " ?[^\s\p{L}\p{N}]+[\r\n]*": punctuation, optionally led by one
        // space, absorbing any line breaks that follow it.
        const bool after_space = cpt == ' ';
        if (after_space ? other(pos + 1) : other(pos)) {
            pos += after_space ? 1 : 0;
            while (other(pos)) ++pos;
            while (at(pos) == '\r' || at(pos) == '\n') ++pos;
            emit(pos);
            continue;
        }

        size_t spaces = 0;
        size_t last_break = 0;
        while (space(pos + spaces)) {
            const uint32_t cpt2 = at(pos + spaces);
            if (cpt2 == '\r' || cpt2 == '\n') last_break = pos + spaces + 1;
            ++spaces;
        }
        // "\s*[\r\n]+": a run of whitespace that contains a line break.
        if (last_break > 0) {
            pos = last_break;
            emit(pos);
            continue;
        }
        if (spaces > 1 && at(pos + spaces) != kOutOfRange) {
            pos += spaces - 1;
            emit(pos);
            continue;
        }
        if (spaces > 0) {
            pos += spaces;
            emit(pos);
            continue;
        }
        emit(++pos);
    }
    return words;
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

Tokenizer::PreTokenizer Tokenizer::pre_tokenizer_from_name(const std::string &name) {
    return lookup_pre_tokenizer(name);
}

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

    auto pre_it = doc.metadata.find("tokenizer.ggml.pre");
    if (pre_it != doc.metadata.end() && pre_it->second.type == GGUFValueType::STRING)
        tokenizer.pre_name_ = pre_it->second.s;
    tokenizer.pre_ = lookup_pre_tokenizer(tokenizer.pre_name_);

    // `ignore_merges` means "if the whole word is a vocabulary piece, emit it
    // without merging". The reference does not read this from the file: it is
    // implied by the pre-tokenizer family, and llama-3 vocabularies depend on it
    // (multi-byte Vietnamese and CJK pieces are only reachable this way).
    tokenizer.ignore_merges_ = tokenizer.pre_ == PreTokenizer::Llama3;

    for (int byte = 0; byte < 256; ++byte)
        tokenizer.byte_text_[byte] = unicode::byte_to_utf8(static_cast<unsigned char>(byte));

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
            // The separator is searched from the second byte, as the reference
            // does: a merge whose left half is one character long is the
            // narrowest legal entry, and skipping byte 0 cannot mis-split one.
            const size_t space = line.empty() ? std::string::npos : line.find(' ', 1);
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
    // For BPE the vocabulary represents a byte with a *character*, not with a
    // one-byte piece: 0x20 is stored as U+0120 and 0xC3 as U+00C3. Judging
    // coverage by piece length alone would call every real GPT-2 vocabulary
    // incomplete, so each byte is looked up in the alphabet the file defines.
    if (tokenizer.kind_ == Kind::Bpe) {
        for (int byte = 0; byte < 256; ++byte) {
            if (tokenizer.byte_covered_[byte]) continue;
            if (tokenizer.piece_to_id_.count(tokenizer.byte_text_[byte]) != 0)
                tokenizer.byte_covered_[byte] = true;
        }
    }
    tokenizer.byte_coverage_complete_ =
        std::all_of(tokenizer.byte_covered_.begin(), tokenizer.byte_covered_.end(),
                    [](bool covered) { return covered; });
    return tokenizer;
}

bool Tokenizer::contains_piece(const std::string &piece) const {
    return piece_to_id_.count(piece) != 0;
}

std::string Tokenizer::piece(uint32_t id) const {
    if (id >= pieces_.size()) return std::string();
    return pieces_[id];
}

const char *Tokenizer::pre_tokenizer_family() const { return pre_tokenizer_family_name(pre_); }

const char *Tokenizer::pre_tokenizer_family_name(PreTokenizer family) {
    switch (family) {
        case PreTokenizer::Gpt2: return "gpt2";
        case PreTokenizer::Gpt2Digits: return "gpt2-digits";
        case PreTokenizer::Llama3: return "llama3";
        case PreTokenizer::Qwen2: return "qwen2";
        case PreTokenizer::Unspecified: return "unspecified";
        case PreTokenizer::Unsupported: return "unsupported";
    }
    return "unspecified";
}

// Merges one byte-encoded word. Two symbols merge when their concatenation has
// a rank, lowest rank first; ties go to the leftmost pair, which is what the
// reference's priority queue does. Words are short, so the quadratic scan is
// cheaper than maintaining a queue and cannot disagree with itself.
std::vector<std::string> Tokenizer::merge_word(const std::string &word) const {
    std::vector<std::string> symbols;
    if (ignore_merges_ && piece_to_id_.count(word) != 0) {
        symbols.push_back(word);
        return symbols;
    }
    // One symbol per byte-encoded character, not per byte: the alphabet maps
    // most bytes to multi-byte characters, so splitting at bytes would cut
    // characters in half.
    for (uint32_t cpt : unicode::from_utf8(word)) symbols.push_back(unicode::to_utf8(cpt));

    std::string key;
    while (symbols.size() > 1) {
        int32_t best_rank = INT_MAX;
        size_t best = 0;
        for (size_t i = 1; i < symbols.size(); ++i) {
            key.assign(symbols[i - 1]);
            key.push_back('\x1f');
            key.append(symbols[i]);
            auto it = merge_rank_.find(key);
            if (it != merge_rank_.end() && it->second < best_rank) {
                best_rank = it->second;
                best = i;
            }
        }
        if (best_rank == INT_MAX) break;
        symbols[best - 1] += symbols[best];
        symbols.erase(symbols.begin() + static_cast<long>(best));
    }
    return symbols;
}

std::vector<int32_t> Tokenizer::encode_bpe(const std::string &text, uint32_t bos_id,
                                           bool add_bos) const {
    if (merges_.empty())
        fail(ErrorCode::NOT_IMPLEMENTED,
             "this checkpoint uses a BPE tokenizer but carries no merge list");
    if (pre_ == PreTokenizer::Unspecified)
        fail(ErrorCode::NOT_IMPLEMENTED,
             "a BPE checkpoint needs `tokenizer.ggml.pre` to say how to split words; refusing "
             "to guess, because the wrong rule changes every token id");
    if (pre_ == PreTokenizer::Unsupported)
        fail(ErrorCode::NOT_IMPLEMENTED,
             "pre-tokenizer '" + pre_name_ +
                 "' is not implemented; implemented families are gpt2, gpt2-digits, llama3 and "
                 "qwen2");

    const std::vector<uint32_t> cpts = unicode::from_utf8(text);
    std::vector<WordSpan> words;
    switch (pre_) {
        case PreTokenizer::Gpt2: words = split_gpt2(cpts, false); break;
        case PreTokenizer::Gpt2Digits: words = split_gpt2(cpts, true); break;
        case PreTokenizer::Llama3: words = split_llama3(cpts, 3); break;
        case PreTokenizer::Qwen2: words = split_llama3(cpts, 1); break;
        default: break;
    }

    std::vector<int32_t> ids;
    if (add_bos) ids.push_back(static_cast<int32_t>(bos_id));
    for (const WordSpan &word : words) {
        // Split on characters, then byte-encode: the alphabet is defined over
        // bytes, so the word's UTF-8 bytes are what merges operate on.
        std::string raw;
        for (size_t i = word.begin; i < word.end; ++i) raw += unicode::to_utf8(cpts[i]);
        std::string encoded;
        encoded.reserve(raw.size() * 2);
        for (unsigned char byte : raw) encoded += byte_text_[byte];
        if (encoded.empty()) continue;
        for (const std::string &piece : merge_word(encoded)) {
            auto it = piece_to_id_.find(piece);
            if (it != piece_to_id_.end()) {
                ids.push_back(it->second);
                continue;
            }
            // The vocabulary has no such piece: fall back to the bytes, so the
            // text stays representable instead of being dropped.
            for (uint32_t cpt : unicode::from_utf8(piece)) {
                const int byte = unicode::utf8_to_byte(cpt);
                if (byte < 0 || byte_ids_[byte] < 0)
                    fail(ErrorCode::INVALID_TOKENIZER,
                         "BPE piece '" + piece + "' is not in the vocabulary and has no byte "
                         "fallback token");
                ids.push_back(byte_ids_[byte]);
            }
        }
    }
    return ids;
}

std::vector<int32_t> Tokenizer::encode(const std::string &text, uint32_t bos_id,
                                       bool add_bos) const {
    if (kind_ == Kind::Byte) {
        std::vector<int32_t> ids;
        ids.reserve(text.size() + 1);
        if (add_bos) ids.push_back(static_cast<int32_t>(bos_id));
        for (unsigned char byte : text) ids.push_back(static_cast<int32_t>(byte));
        return ids;
    }
    if (kind_ == Kind::Bpe) return encode_bpe(text, bos_id, add_bos);

    // SentencePiece prepends a space; the marker then reads as a word boundary.
    std::string body = (kind_ == Kind::SentencePiece) ? (" " + text) : text;
    std::vector<int32_t> ids;
    if (add_bos) ids.push_back(static_cast<int32_t>(bos_id));

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
        if (kind_ == Kind::SentencePiece) {
            out += unmark_spaces(piece);
            continue;
        }
        // BPE pieces are stored in the byte-level alphabet, so each character
        // has to be mapped back to the byte it stands for. A character outside
        // the alphabet is literal text some vocabularies add (e.g. a lone "é"),
        // and is emitted as itself.
        for (uint32_t cpt : unicode::from_utf8(piece)) {
            const int byte = unicode::utf8_to_byte(cpt);
            if (byte >= 0) out.push_back(static_cast<char>(byte));
            else out += unicode::to_utf8(cpt);
        }
    }
    return out;
}

} // namespace fm
