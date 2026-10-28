#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace fm::unicode {

// The Unicode classes the BPE pre-tokenisers test. The classes come from
// generated tables (`src/unicode_tables.inc`, produced by
// `tools/gen_unicode_tables.py`), so they are data, not hand-written guesses:
// a wrong class silently moves a word boundary and yields a different token id.
enum : uint32_t {
    kOther = 0,
    kLetter = 1u << 0,
    kNumber = 1u << 1,
    kPunctuation = 1u << 2,
    kMark = 1u << 3,
    kSymbol = 1u << 4,
    kSpace = 1u << 5,
};

// Bitmask of the classes above for one code point.
uint32_t classes(uint32_t cpt);
inline bool is_letter(uint32_t cpt) { return (classes(cpt) & kLetter) != 0; }
inline bool is_number(uint32_t cpt) { return (classes(cpt) & kNumber) != 0; }
inline bool is_space(uint32_t cpt) { return (classes(cpt) & kSpace) != 0; }

// Decodes UTF-8 into code points. Invalid bytes become U+FFFD and consume one
// byte, which is what llama.cpp's tokenizer does; the alternative (failing)
// would make a single bad byte in a prompt reject the whole prompt, and the
// alternative that matters more — reinterpreting bytes as Latin-1 — would
// tokenize differently from every other runtime.
std::vector<uint32_t> from_utf8(const std::string &text);

// Appends the UTF-8 encoding of one code point.
std::string to_utf8(uint32_t cpt);

// ASCII-only lowercase. The only caller is contraction matching ("'S" -> "'s"),
// whose alphabet is ASCII, so a full Unicode case table would be dead weight.
uint32_t ascii_lower(uint32_t cpt);

// GPT-2 byte-level encoding: every one of the 256 bytes is mapped to a
// printable code point so a BPE vocabulary can store arbitrary bytes as text.
// Reverse lookup returns -1 when the code point is not part of the mapping.
const std::string &byte_to_utf8(unsigned char byte);
int utf8_to_byte(uint32_t cpt);

} // namespace fm::unicode
