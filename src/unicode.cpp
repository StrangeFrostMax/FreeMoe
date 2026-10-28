// Unicode classification, UTF-8 and the GPT-2 byte-level alphabet.
//
// The BPE tokenizer needs all three and none of them can be approximated
// without changing token ids:
//
//   * the pre-tokeniser decides where a word boundary is, using Unicode
//     general categories (`\p{L}`, `\p{N}`, `\s`);
//   * the merge phase works on the byte-level alphabet, so bytes and "text"
//     are two different encodings of the same thing;
//   * detokenisation has to reverse the byte-level alphabet exactly.
//
// The category tables are generated (`tools/gen_unicode_tables.py`) rather than
// written by hand, and `tests` checks the generated classifiers against the
// same Python source of truth.

#include "fm/unicode.hpp"

#include <algorithm>
#include <array>
#include <cstdio>

namespace fm::unicode {
namespace {

struct RuneRange {
    uint32_t first;
    uint32_t last;
};

#include "unicode_tables.inc"

template <size_t N>
bool in_ranges(const RuneRange (&ranges)[N], uint32_t cpt) {
    // Tables are sorted and non-overlapping, so a binary search over ranges is
    // exact: find the first range whose end is at or after `cpt`.
    size_t lo = 0;
    size_t hi = N;
    while (lo < hi) {
        const size_t mid = lo + (hi - lo) / 2;
        if (ranges[mid].last < cpt) lo = mid + 1;
        else hi = mid;
    }
    return lo < N && ranges[lo].first <= cpt;
}

// GPT-2's bytes-to-unicode table: printable ASCII and the Latin-1 letters are
// mapped to themselves, everything else to U+0100 + n in byte order. Building
// it at startup keeps the two directions consistent by construction.
struct ByteAlphabet {
    std::array<std::string, 256> to_text;
    std::array<int, 0x200> to_byte;

    ByteAlphabet() {
        to_byte.fill(-1);
        std::array<bool, 256> direct{};
        for (int ch = 0x21; ch <= 0x7E; ++ch) direct[ch] = true;
        for (int ch = 0xA1; ch <= 0xAC; ++ch) direct[ch] = true;
        for (int ch = 0xAE; ch <= 0xFF; ++ch) direct[ch] = true;
        uint32_t next = 256;
        for (int ch = 0; ch < 256; ++ch) {
            const uint32_t cpt = direct[ch] ? static_cast<uint32_t>(ch) : next++;
            to_text[ch] = to_utf8(cpt);
            if (cpt < to_byte.size()) to_byte[cpt] = ch;
        }
    }
};

const ByteAlphabet &alphabet() {
    static const ByteAlphabet table;
    return table;
}

} // namespace

uint32_t classes(uint32_t cpt) {
    uint32_t out = kOther;
    if (in_ranges(kRangesLetter, cpt)) out |= kLetter;
    if (in_ranges(kRangesNumber, cpt)) out |= kNumber;
    if (in_ranges(kRangesPunctuation, cpt)) out |= kPunctuation;
    if (in_ranges(kRangesMark, cpt)) out |= kMark;
    if (in_ranges(kRangesSymbol, cpt)) out |= kSymbol;
    if (in_ranges(kRangesSpace, cpt)) out |= kSpace;
    return out;
}

std::vector<uint32_t> from_utf8(const std::string &text) {
    std::vector<uint32_t> out;
    out.reserve(text.size());
    size_t i = 0;
    while (i < text.size()) {
        const unsigned char lead = static_cast<unsigned char>(text[i]);
        size_t extra = 0;
        uint32_t cpt = 0;
        if (lead < 0x80) {
            out.push_back(lead);
            ++i;
            continue;
        } else if ((lead & 0xE0) == 0xC0) {
            extra = 1;
            cpt = lead & 0x1Fu;
        } else if ((lead & 0xF0) == 0xE0) {
            extra = 2;
            cpt = lead & 0x0Fu;
        } else if ((lead & 0xF8) == 0xF0) {
            extra = 3;
            cpt = lead & 0x07u;
        } else {
            out.push_back(0xFFFD);
            ++i;
            continue;
        }
        if (i + extra >= text.size()) {
            out.push_back(0xFFFD);
            ++i;
            continue;
        }
        bool ok = true;
        for (size_t k = 1; k <= extra; ++k) {
            const unsigned char cont = static_cast<unsigned char>(text[i + k]);
            if ((cont & 0xC0) != 0x80) {
                ok = false;
                break;
            }
            cpt = (cpt << 6) | (cont & 0x3Fu);
        }
        if (!ok) {
            out.push_back(0xFFFD);
            ++i;
            continue;
        }
        out.push_back(cpt);
        i += extra + 1;
    }
    return out;
}

std::string to_utf8(uint32_t cpt) {
    std::string out;
    if (cpt < 0x80) {
        out.push_back(static_cast<char>(cpt));
    } else if (cpt < 0x800) {
        out.push_back(static_cast<char>(0xC0 | (cpt >> 6)));
        out.push_back(static_cast<char>(0x80 | (cpt & 0x3F)));
    } else if (cpt < 0x10000) {
        out.push_back(static_cast<char>(0xE0 | (cpt >> 12)));
        out.push_back(static_cast<char>(0x80 | ((cpt >> 6) & 0x3F)));
        out.push_back(static_cast<char>(0x80 | (cpt & 0x3F)));
    } else {
        out.push_back(static_cast<char>(0xF0 | (cpt >> 18)));
        out.push_back(static_cast<char>(0x80 | ((cpt >> 12) & 0x3F)));
        out.push_back(static_cast<char>(0x80 | ((cpt >> 6) & 0x3F)));
        out.push_back(static_cast<char>(0x80 | (cpt & 0x3F)));
    }
    return out;
}

uint32_t ascii_lower(uint32_t cpt) {
    return (cpt >= 'A' && cpt <= 'Z') ? cpt + ('a' - 'A') : cpt;
}

const std::string &byte_to_utf8(unsigned char byte) { return alphabet().to_text[byte]; }

int utf8_to_byte(uint32_t cpt) {
    const ByteAlphabet &table = alphabet();
    return cpt < table.to_byte.size() ? table.to_byte[cpt] : -1;
}

} // namespace fm::unicode
