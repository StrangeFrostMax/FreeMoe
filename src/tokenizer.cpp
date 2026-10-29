// Real tokenizers for published checkpoints.
//
// Two families appear in GGUF in practice:
//
//   * SentencePiece-derived (`tokenizer.ggml.model = "llama"`): the token table
//     stores pieces where U+2581 ("▁") marks a leading space, plus a float score
//     per piece. Tokens shaped `<0xNN>` are byte fallbacks.
//   * GPT-2 style BPE (`"gpt2"`): a token table plus an ordered merge list.
//
// SentencePiece encoding uses the piece scores: the reference merges the
// adjacent pair with the best piece score until nothing more merges. Greedy
// longest-match over the same table produces different ids on ordinary text,
// because a long low-score piece can beat the two high-score pieces the model
// was trained on.
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

#include <queue>
#include <map>

#include "fm/unicode.hpp"

#include <algorithm>
#include <array>
#include <climits>
#include <functional>
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
        // deepseek-llm / -coder / -v3: their own splitting rules, each
        // transcribed from the reference table and checked against its
        // published test vectors.
        {"deepseek-llm", Pre::DeepseekLlm},   {"deepseek-coder", Pre::DeepseekCoder},
        {"deepseek-v3", Pre::DeepseekV3},     {"hunyuan-dense", Pre::DeepseekV3},
        {"joyai-llm", Pre::DeepseekV3},       {"hy_v4", Pre::DeepseekV3},
        // falcon: punctuation runs first, then the GPT-2 rule.
        {"falcon", Pre::Falcon},
        // tekken: Mistral 3 / Ministral, case-aware letter runs.
        {"tekken", Pre::Tekken},
        // qwen35: llama-3's rule with marks counted as letters, one digit each.
        {"qwen35", Pre::Qwen35},
        // Recognised vocabularies whose splitting rule is not implemented here.
        {"minicpm5", Pre::Unsupported},       {"gemma4", Pre::Unsupported},
        {"mmbert", Pre::Unsupported},
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

// The reference splitter applies `regex_exprs` **one after another**, each pass
// subdividing the words the previous pass produced (`unicode_regex_split` in
// llama.cpp/src/unicode.cpp). A word that a pattern does not match is left
// whole, which is why a run of spaces survives as one piece: no pattern in the
// deepseek tables matches a bare space.
//
// A "matcher" returns the length of the longest match at a position, or 0 for no
// match. Applying one matcher to a word means: emit every match, and emit the
// gaps between matches as their own pieces.
using Matcher = std::function<size_t(const std::vector<uint32_t> &, size_t)>;

std::vector<WordSpan> apply_matcher(const std::vector<WordSpan> &words,
                                    const std::vector<uint32_t> &cpts, const Matcher &match) {
    std::vector<WordSpan> out;
    for (const WordSpan &w : words) {
        size_t pos = w.begin;
        size_t run_start = w.begin;
        auto flush = [&](size_t end) {
            if (end > run_start) out.push_back({run_start, end});
        };
        while (pos < w.end) {
            const size_t len = match(cpts, pos);
            if (len == 0 || pos + len > w.end) {
                ++pos;
                continue;
            }
            flush(pos);
            out.push_back({pos, pos + len});
            pos += len;
            run_start = pos;
        }
        flush(w.end);
    }
    return out;
}

std::vector<WordSpan> split_by_patterns(const std::vector<uint32_t> &cpts,
                                        const std::vector<Matcher> &patterns) {
    std::vector<WordSpan> words;
    if (!cpts.empty()) words.push_back({0, cpts.size()});
    for (const Matcher &m : patterns) words = apply_matcher(words, cpts, m);
    return words;
}

// The matchers the deepseek tables are built from. Each one is a single
// alternative of the reference's table, applied as its own pass.
namespace match {

inline bool at_letter(const std::vector<uint32_t> &c, size_t p) {
    return p < c.size() && unicode::is_letter(c[p]);
}
inline bool at_number(const std::vector<uint32_t> &c, size_t p) {
    return p < c.size() && unicode::is_number(c[p]);
}
inline bool at_space(const std::vector<uint32_t> &c, size_t p) {
    return p < c.size() && unicode::is_space(c[p]);
}
inline bool at_class(const std::vector<uint32_t> &c, size_t p, uint32_t bit) {
    return p < c.size() && (unicode::classes(c[p]) & bit) != 0;
}
inline bool at_newline(const std::vector<uint32_t> &c, size_t p) {
    return p < c.size() && (c[p] == '\r' || c[p] == '\n');
}

// "[\r\n]"
inline size_t newline(const std::vector<uint32_t> &c, size_t p) { return at_newline(c, p) ? 1 : 0; }

// "\s?[X]+" with X = letters, punctuation or "letters or marks".
inline size_t lead_run(const std::vector<uint32_t> &c, size_t p, uint32_t bits) {
    size_t start = p;
    if (at_space(c, p)) ++start;
    if (start >= c.size() || (unicode::classes(c[start]) & bits) == 0) return 0;
    while (start < c.size() && (unicode::classes(c[start]) & bits) != 0) ++start;
    return start - p;
}

// "\s+$": trailing whitespace that reaches the end of the text.
inline size_t trailing_space(const std::vector<uint32_t> &c, size_t p) {
    if (!at_space(c, p)) return 0;
    size_t end = p;
    while (end < c.size() && at_space(c, end)) ++end;
    // The reference anchors on the end of the *word*, which here is the whole
    // remaining text, so a run that stops at a non-space is not this pattern.
    return end == c.size() ? end - p : 0;
}

// "\p{N}+" and "\p{N}" and "\p{N}{1,n}".
inline size_t digit_run(const std::vector<uint32_t> &c, size_t p, size_t max_digits) {
    if (!at_number(c, p)) return 0;
    size_t n = 0;
    while (p + n < c.size() && at_number(c, p + n) && n < max_digits) ++n;
    return n;
}

}  // namespace match

// deepseek-llm (`LLAMA_VOCAB_PRE_TYPE_DEEPSEEK_LLM`):
//
//   [\r\n] | \s?[letters]+ | \s?[punct]+ | \s+$ | [CJK]+ | \p{N}+
//
// The CJK alternative is redundant after the letter pass, which runs first and
// claims every CJK code point (they are `\p{L}`) with the same boundaries.
std::vector<WordSpan> split_deepseek_llm(const std::vector<uint32_t> &cpts) {
    using namespace match;
    return split_by_patterns(cpts, {
        newline,
        [](const std::vector<uint32_t> &c, size_t p) { return lead_run(c, p, unicode::kLetter); },
        [](const std::vector<uint32_t> &c, size_t p) { return lead_run(c, p, unicode::kPunctuation); },
        trailing_space,
        [](const std::vector<uint32_t> &c, size_t p) { return digit_run(c, p, SIZE_MAX); },
    });
}

// deepseek-coder (`LLAMA_VOCAB_PRE_TYPE_DEEPSEEK_CODER`): the same table with
// one digit per token instead of digit runs.
std::vector<WordSpan> split_deepseek_coder(const std::vector<uint32_t> &cpts) {
    using namespace match;
    return split_by_patterns(cpts, {
        newline,
        [](const std::vector<uint32_t> &c, size_t p) { return lead_run(c, p, unicode::kLetter); },
        [](const std::vector<uint32_t> &c, size_t p) { return lead_run(c, p, unicode::kPunctuation); },
        [](const std::vector<uint32_t> &c, size_t p) { return digit_run(c, p, 1); },
    });
}

// deepseek-v3 (`LLAMA_VOCAB_PRE_TYPE_DEEPSEEK3_LLM`, shared with hunyuan-dense,
// joyai and hy_v4):
//
//   \p{N}{1,3} | [CJK kana]+ | [!-~][A-Za-z]+ | [^\r\n\p{L}\p{P}\p{S}]?[\p{L}\p{M}]+ |
//   ?[\p{P}\p{S}]+[\r\n]* | \s*[\r\n]+ | \s+(?!\S) | \s+
std::vector<WordSpan> split_deepseek_v3(const std::vector<uint32_t> &cpts) {
    using namespace match;
    constexpr uint32_t kLetterMark = unicode::kLetter | unicode::kMark;
    constexpr uint32_t kPunctSymbol = unicode::kPunctuation | unicode::kSymbol;
    return split_by_patterns(cpts, {
        [](const std::vector<uint32_t> &c, size_t p) { return digit_run(c, p, 3); },
        // "[!-~][A-Za-z]+": one ASCII punctuation or symbol, then letters.
        [](const std::vector<uint32_t> &c, size_t p) {
            if (p + 1 >= c.size()) return size_t{0};
            const uint32_t ch = c[p];
            if (ch < 0x21 || ch > 0x7e) return size_t{0};
            if (unicode::is_letter(ch) || unicode::is_number(ch)) return size_t{0};
            if (!at_letter(c, p + 1)) return size_t{0};
            size_t end = p + 1;
            while (end < c.size() && unicode::is_letter(c[end])) ++end;
            return end - p;
        },
        // "[^\r\n\p{L}\p{P}\p{S}]?[\p{L}\p{M}]+"
        [](const std::vector<uint32_t> &c, size_t p) {
            size_t start = p;
            if (start < c.size() && !at_newline(c, start) &&
                (unicode::classes(c[start]) & (kLetterMark | kPunctSymbol)) == 0)
                ++start;
            if (start >= c.size() || (unicode::classes(c[start]) & kLetterMark) == 0) return size_t{0};
            while (start < c.size() && (unicode::classes(c[start]) & kLetterMark) != 0) ++start;
            return start - p;
        },
        // " ?[\p{P}\p{S}]+[\r\n]*"
        [](const std::vector<uint32_t> &c, size_t p) {
            size_t start = p;
            if (start < c.size() && c[start] == ' ') ++start;
            if (start >= c.size() || (unicode::classes(c[start]) & kPunctSymbol) == 0) return size_t{0};
            while (start < c.size() && (unicode::classes(c[start]) & kPunctSymbol) != 0) ++start;
            while (start < c.size() && at_newline(c, start)) ++start;
            return start - p;
        },
        // "\s*[\r\n]+"
        [](const std::vector<uint32_t> &c, size_t p) {
            size_t start = p;
            while (start < c.size() && at_space(c, start) && !at_newline(c, start)) ++start;
            if (start >= c.size() || !at_newline(c, start)) return size_t{0};
            while (start < c.size() && at_newline(c, start)) ++start;
            return start - p;
        },
        // "\s+(?!\S)": a whitespace run not followed by a non-space.
        [](const std::vector<uint32_t> &c, size_t p) {
            if (!at_space(c, p)) return size_t{0};
            size_t end = p;
            while (end < c.size() && at_space(c, end)) ++end;
            return end == c.size() ? end - p : size_t{0};
        },
        // "\s+"
        [](const std::vector<uint32_t> &c, size_t p) {
            if (!at_space(c, p)) return size_t{0};
            size_t end = p;
            while (end < c.size() && at_space(c, end)) ++end;
            return end - p;
        },
    });
}

// falcon (`LLAMA_VOCAB_PRE_TYPE_FALCON`):
//
//   [\p{P}$+<=>^~|`]+ | 's|'t|...| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S) |
//   [0-9][0-9][0-9]
//
// Applied as sequential passes, like the deepseek tables. The punctuation-run
// pass comes first and has no leading space, so it splits a run of punctuation
// before the GPT-2-shaped pass can glue a space to it.
std::vector<WordSpan> split_falcon(const std::vector<uint32_t> &cpts) {
    using namespace match;
    auto punct_plus = [](const std::vector<uint32_t> &c, size_t p) {
        const uint32_t ch = p < c.size() ? c[p] : ~0u;
        if ((unicode::classes(ch) & unicode::kPunctuation) != 0) return true;
        return ch == '$' || ch == '+' || ch == '<' || ch == '=' || ch == '>' || ch == '^' ||
               ch == '~' || ch == '|' || ch == '`';
    };
    return split_by_patterns(cpts, {
        // "[\p{P}$+<=>^~|`]+"
        [punct_plus](const std::vector<uint32_t> &c, size_t p) {
            if (!punct_plus(c, p)) return size_t{0};
            size_t end = p;
            while (end < c.size() && punct_plus(c, end)) ++end;
            return end - p;
        },
        // The contractions, case-insensitive as in the reference table.
        [](const std::vector<uint32_t> &c, size_t p) {
            if (p >= c.size() || c[p] != '\'') return size_t{0};
            auto lower = [&](size_t i) {
                return i < c.size() ? static_cast<uint32_t>(unicode::ascii_lower(c[i])) : ~0u;
            };
            const uint32_t a = lower(p + 1), b = lower(p + 2);
            if (a == 's' || a == 't' || a == 'm' || a == 'd') return size_t{2};
            if ((a == 'r' && b == 'e') || (a == 'v' && b == 'e') || (a == 'l' && b == 'l'))
                return size_t{3};
            return size_t{0};
        },
        // " ?\p{L}+", " ?\p{N}+", " ?[^\s\p{L}\p{N}]+"
        [](const std::vector<uint32_t> &c, size_t p) { return lead_run(c, p, unicode::kLetter); },
        [](const std::vector<uint32_t> &c, size_t p) { return lead_run(c, p, unicode::kNumber); },
        [](const std::vector<uint32_t> &c, size_t p) {
            constexpr uint32_t kOther = ~(unicode::kSpace | unicode::kLetter | unicode::kNumber);
            if (p >= c.size()) return size_t{0};
            size_t start = p;
            if (c[start] == ' ') ++start;
            if (start >= c.size() || (unicode::classes(c[start]) & kOther) == 0) return size_t{0};
            while (start < c.size() && (unicode::classes(c[start]) & kOther) != 0) ++start;
            return start - p;
        },
        // "\s+(?!\S)"
        [](const std::vector<uint32_t> &c, size_t p) {
            if (!at_space(c, p)) return size_t{0};
            size_t end = p;
            while (end < c.size() && at_space(c, end)) ++end;
            if (end < c.size()) return size_t{0};  // followed by a non-space: no match
            return end - p;
        },
        // "[0-9][0-9][0-9]"
        [](const std::vector<uint32_t> &c, size_t p) {
            for (size_t k = 0; k < 3; ++k)
                if (p + k >= c.size() || c[p + k] < '0' || c[p + k] > '9') return size_t{0};
            return size_t{3};
        },
    });
}

// tekken (Mistral 3 / Ministral):
//
//   [^\r\n\p{L}\p{N}]?((?=[\p{L}\p{M}])([^a-z]))*((?=[\p{L}\p{M}])([^A-Z]))+ |
//   [^\r\n\p{L}\p{N}]?((?=[\p{L}\p{M}])([^a-z]))+((?=[\p{L}\p{M}])([^A-Z]))* |
//   \p{N} | ?[^\s\p{L}\p{N}]+[\r\n/]* | \s*[\r\n]+ | \s+(?!\S) | \s+
//
// The two lookahead forms together mean: an optional leading non-letter, then a
// run of letters/marks that starts with an uppercase (first form) or continues
// as the case run that the leading character began (second form). In practice
// this cuts a letter run at an upper/lower case boundary, keeping the run of one
// case together.
std::vector<WordSpan> split_tekken(const std::vector<uint32_t> &cpts) {
    const size_t size = cpts.size();
    auto at = [&](size_t pos) -> uint32_t { return pos < size ? cpts[pos] : kOutOfRange; };
    auto letter_mark = [&](size_t pos) {
        const uint32_t cpt = at(pos);
        if (cpt == kOutOfRange) return false;
        return (unicode::classes(cpt) & (unicode::kLetter | unicode::kMark)) != 0;
    };
    auto number = [&](size_t pos) { return unicode::is_number(at(pos)); };
    auto space = [&](size_t pos) { return unicode::is_space(at(pos)); };
    auto lead_ok = [&](size_t pos) {
        const uint32_t c = at(pos);
        if (c == kOutOfRange || c == '\r' || c == '\n') return false;
        return !unicode::is_letter(c) && !unicode::is_number(c);
    };

    std::vector<WordSpan> words;
    size_t start = 0, pos = 0;
    auto emit = [&](size_t end) {
        if (end > start) words.push_back({start, end});
        start = end;
    };

    while (pos < size) {
        // \p{N} one at a time.
        if (number(pos)) {
            ++pos;
            emit(pos);
            continue;
        }
        // The two letter-run forms: an optional lead, then a maximal run of
        // letters/marks.
        {
            const bool lead = lead_ok(pos) && letter_mark(pos + 1);
            const size_t body = lead ? pos + 1 : pos;
            if (letter_mark(body)) {
                pos = body;
                while (letter_mark(pos)) ++pos;
                emit(pos);
                continue;
            }
        }
        // " ?[^\s\p{L}\p{N}]+[\r\n/]*"
        auto other = [&](size_t p) {
            const uint32_t c = at(p);
            if (c == kOutOfRange) return false;
            return !unicode::is_space(c) && !unicode::is_letter(c) && !unicode::is_number(c);
        };
        {
            const bool lead = at(pos) == ' ';
            if (other(lead ? pos + 1 : pos)) {
                pos += lead ? 1 : 0;
                while (other(pos)) ++pos;
                while (at(pos) == '\r' || at(pos) == '\n' || at(pos) == '/') ++pos;
                emit(pos);
                continue;
            }
        }
        // "\s*[\r\n]+"
        {
            size_t run = pos;
            while (space(run) && at(run) != '\r' && at(run) != '\n') ++run;
            if (at(run) == '\r' || at(run) == '\n') {
                while (at(run) == '\r' || at(run) == '\n') ++run;
                pos = run;
                emit(pos);
                continue;
            }
        }
        // "\s+(?!\S)" and "\s+"
        size_t run = 0;
        while (space(pos + run)) ++run;
        if (run > 1 && at(pos + run) != kOutOfRange) {
            pos += run - 1;
            emit(pos);
            continue;
        }
        if (run > 0) {
            pos += run;
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

    // SentencePiece normalization options. The defaults are the reference's for
    // `LLAMA_VOCAB_TYPE_SPM` (prefix a space, keep runs as written); a file may
    // override either.
    auto bool_key = [&doc](const char *key, bool fallback) {
        auto it = doc.metadata.find(key);
        if (it == doc.metadata.end() || it->second.type != GGUFValueType::BOOL) return fallback;
        return it->second.b;
    };
    tokenizer.spm_add_space_prefix_ = bool_key("tokenizer.ggml.add_space_prefix", true);

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
        case PreTokenizer::Qwen35: return "qwen35";
        case PreTokenizer::DeepseekLlm: return "deepseek-llm";
        case PreTokenizer::DeepseekCoder: return "deepseek-coder";
        case PreTokenizer::DeepseekV3: return "deepseek-v3";
        case PreTokenizer::Falcon: return "falcon";
        case PreTokenizer::Tekken: return "tekken";
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
        case PreTokenizer::Qwen35: words = split_llama3(cpts, 1); break;
        case PreTokenizer::DeepseekLlm: words = split_deepseek_llm(cpts); break;
        case PreTokenizer::DeepseekCoder: words = split_deepseek_coder(cpts); break;
        case PreTokenizer::DeepseekV3: words = split_deepseek_v3(cpts); break;
        case PreTokenizer::Falcon: words = split_falcon(cpts); break;
        case PreTokenizer::Tekken: words = split_tekken(cpts); break;
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
    return encode_sentencepiece(text, bos_id, add_bos);
}

std::string Tokenizer::normalize_spm(const std::string &text) const {
    // The reference's SentencePiece path: prepend a space (unconditionally -- the
    // dummy prefix is what makes a word's first piece carry the boundary) and
    // then escape every space 1:1 as U+2581. We keep raw spaces, which is the
    // same string with a different marker, because pieces are normalised that
    // way on load.
    //
    // Two details that look like nitpicks and are not:
    //  * the prefix is added even when the text already starts with a space, so
    //    " hello" escapes to three markers' worth of text, not two;
    //  * spaces are not collapsed. `remove_extra_whitespaces` never reaches this
    //    path upstream (only the fragment normalizer uses it), so honouring it
    //    here would disagree with the reference on inputs with runs of spaces.
    if (!spm_add_space_prefix_ || text.empty()) return text;
    return " " + text;
}

std::vector<int32_t> Tokenizer::encode_sentencepiece(const std::string &text, uint32_t bos_id,
                                                     bool add_bos) const {
    // The reference's `llm_tokenizer_spm_session`: every UTF-8 character starts as
    // its own symbol, then the highest-scoring adjacent pair whose concatenation
    // is a vocabulary piece is merged, repeatedly. Score is the *piece's* score,
    // so a long low-score piece loses to two high-score pieces -- which is the
    // whole difference from longest-match.
    struct Symbol {
        size_t offset = 0;
        size_t length = 0;
        int32_t prev = -1;
        int32_t next = -1;
    };
    struct Bigram {
        int32_t left = -1;
        int32_t right = -1;
        float score = 0.0F;
        size_t size = 0;
        // priority_queue is a max-heap under this order: highest score first, and
        // on a tie the leftmost pair, matching the reference's comparator.
        bool operator<(const Bigram &other) const {
            return score < other.score || (score == other.score && left > other.left);
        }
    };

    const std::string body = normalize_spm(text);
    std::vector<int32_t> ids;
    if (add_bos) ids.push_back(static_cast<int32_t>(bos_id));
    if (body.empty()) return ids;

    std::vector<Symbol> symbols;
    symbols.reserve(body.size());
    for (size_t offs = 0; offs < body.size();) {
        const unsigned char lead = static_cast<unsigned char>(body[offs]);
        size_t len = 1;
        if ((lead & 0xE0) == 0xC0) len = 2;
        else if ((lead & 0xF0) == 0xE0) len = 3;
        else if ((lead & 0xF8) == 0xF0) len = 4;
        len = std::min(len, body.size() - offs);
        Symbol sym;
        sym.offset = offs;
        sym.length = len;
        sym.prev = static_cast<int32_t>(symbols.size()) - 1;
        sym.next = offs + len == body.size() ? -1 : static_cast<int32_t>(symbols.size()) + 1;
        symbols.push_back(sym);
        offs += len;
    }

    std::priority_queue<Bigram> queue;
    // Merged text -> the two symbols it came from, so a symbol that is not itself
    // a piece can still be decomposed instead of falling back to bytes.
    std::map<std::string, std::pair<int32_t, int32_t>> history;

    auto try_add = [&](int32_t left, int32_t right) {
        if (left < 0 || right < 0) return;
        const std::string joined = body.substr(symbols[left].offset,
                                               symbols[left].length + symbols[right].length);
        auto it = piece_to_id_.find(joined);
        if (it == piece_to_id_.end()) return;
        if (static_cast<size_t>(it->second) >= scores_.size()) return;
        Bigram bigram;
        bigram.left = left;
        bigram.right = right;
        bigram.score = scores_[static_cast<size_t>(it->second)];
        bigram.size = joined.size();
        queue.push(bigram);
        history[joined] = {left, right};
    };

    for (size_t i = 1; i < symbols.size(); ++i)
        try_add(static_cast<int32_t>(i) - 1, static_cast<int32_t>(i));

    while (!queue.empty()) {
        const Bigram bigram = queue.top();
        queue.pop();
        Symbol &left = symbols[static_cast<size_t>(bigram.left)];
        Symbol &right = symbols[static_cast<size_t>(bigram.right)];
        // One of the two may have been merged into a neighbour since this entry
        // was queued; the reference skips those rather than re-deriving them.
        if (left.length == 0 || right.length == 0 || left.length + right.length != bigram.size)
            continue;
        left.length += right.length;
        right.length = 0;
        left.next = right.next;
        if (right.next >= 0) symbols[static_cast<size_t>(right.next)].prev = bigram.left;
        try_add(left.prev, bigram.left);
        try_add(bigram.left, left.next);
    }

    // Emit: a merged symbol is a piece by construction; anything left is either a
    // single character that is a piece, a decomposable merge, or raw bytes.
    std::vector<int32_t> order;
    for (int32_t i = symbols.empty() ? -1 : 0; i != -1; i = symbols[static_cast<size_t>(i)].next)
        order.push_back(i);
    std::vector<const Symbol *> todo;
    for (int32_t index : order) todo.push_back(&symbols[static_cast<size_t>(index)]);
    std::reverse(todo.begin(), todo.end());  // pop order == left-to-right
    while (!todo.empty()) {
        const Symbol &cur = *todo.back();
        todo.pop_back();
        const std::string text = body.substr(cur.offset, cur.length);
        auto it = piece_to_id_.find(text);
        if (it != piece_to_id_.end()) {
            ids.push_back(it->second);
            continue;
        }
        auto hist = history.find(text);
        if (hist != history.end()) {
            // Push right then left so the stack pops left-to-right.
            todo.push_back(&symbols[static_cast<size_t>(hist->second.second)]);
            todo.push_back(&symbols[static_cast<size_t>(hist->second.first)]);
            continue;
        }
        if (!has_byte_fallback_)
            fail(ErrorCode::INVALID_TOKENIZER,
                 "SentencePiece segment '" + text +
                     "' is not a piece and the vocabulary has no <0xNN> byte fallback");
        for (char c : text) {
            const int32_t id = byte_ids_[static_cast<unsigned char>(c)];
            if (id < 0)
                fail(ErrorCode::INVALID_TOKENIZER,
                     "no byte-fallback token for byte " +
                         std::to_string(static_cast<int>(static_cast<unsigned char>(c))));
            ids.push_back(id);
        }
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
