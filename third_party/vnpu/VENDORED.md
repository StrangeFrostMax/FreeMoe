# Vendored Virtual NPU

Upstream: https://github.com/StrangeFrostMax/npuxl-improved-rinai
Pinned commit: `4298c454aa56b90dac240e81f80ed4467221c1dd`
License: MIT (see `LICENSE` and `../../LICENSES/VNPU-MIT.txt`).

Only source, public headers, licence text and the upstream selftest/fuzz sources
are vendored. Prebuilt `.so`/`.a`, release tarballs, archive zips, legacy
executables and benchmark outputs are intentionally excluded. FreeMoe builds
the library from this source.

## Local patches

See `PATCHES_FREEMOE.md`. Every patch is covered by a test in the top-level
FreeMoe test suite.
