#include "fm/gguf.hpp"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <random>
#include <string>
#include <thread>
#include <vector>

int main(int argc, char **argv) {
    const uint32_t seconds = argc > 1 ? static_cast<uint32_t>(std::strtoul(argv[1], nullptr, 10)) : 2;
    const uint64_t seed = argc > 2 ? std::strtoull(argv[2], nullptr, 0) : 0xFEEE0Eull;
    std::vector<uint8_t> corpus = fm::read_binary_file("tests/fixtures/dense-fm.gguf");
    // Reuse one allocation. Creating/freeing a fresh 60 KiB vector every
    // iteration makes ASan's quarantine grow faster than it can recycle.
    std::vector<uint8_t> input = corpus;
    std::mt19937_64 rng(seed);
    auto start = std::chrono::steady_clock::now();
    uint64_t iterations = 0;
    uint64_t typed_errors = 0;
    uint64_t accepted = 0;

    while (std::chrono::duration_cast<std::chrono::seconds>(
               std::chrono::steady_clock::now() - start).count() < seconds) {
        input = corpus;
        uint32_t mutations = 1 + static_cast<uint32_t>(rng() % 8);
        for (uint32_t i = 0; i < mutations && !input.empty(); ++i) {
            size_t position = static_cast<size_t>(rng() % input.size());
            input[position] = static_cast<uint8_t>(rng() & 0xff);
        }
        if ((rng() & 3) == 0 && !input.empty())
            input.resize(static_cast<size_t>(rng() % input.size()));
        if ((rng() & 7) == 0 && input.size() >= 24) {
            uint64_t bad_count = 0x8000000000000000ull | (rng() & 0xffff);
            for (int b = 0; b < 8; ++b)
                input[8 + b] = static_cast<uint8_t>(bad_count >> (8 * b));
        }
        ++iterations;
        // Cap allocation churn at roughly 1000 cases/second. ASan's allocator
        // and quarantine can otherwise become the bottleneck on a fast CPU.
        std::this_thread::sleep_until(start + std::chrono::milliseconds(iterations));
        if (std::getenv("FM_FUZZ_DEBUG")) {
            std::cerr << "iteration=" << iterations << "\n";
            std::ofstream dump("/tmp/freemoe-fuzz-current.gguf", std::ios::binary | std::ios::trunc);
            dump.write(reinterpret_cast<const char *>(input.data()),
                       static_cast<std::streamsize>(input.size()));
        }
        try {
            auto doc = fm::parse_gguf(input.data(), input.size());
            ++accepted;
            // Exercise the adapter too, but not on every parsed mutation.
            if (iterations % 8 == 0) {
                try {
                    (void)fm::build_manifest(doc, "fuzz-input.gguf");
                } catch (const fm::Error &) {
                    ++typed_errors;
                }
            }
        } catch (const fm::Error &) {
            ++typed_errors;
        } catch (const std::exception &e) {
            std::cerr << "unexpected exception: " << e.what() << "\n";
            return 2;
        }
    }
    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - start).count();
    std::cout << "{\"iterations\":" << iterations
              << ",\"typed_errors\":" << typed_errors
              << ",\"accepted\":" << accepted
              << ",\"elapsed_ms\":" << elapsed
              << ",\"seed\":" << seed
              << ",\"sanitizers\":\"address,undefined\"}\n";
    return 0;
}
