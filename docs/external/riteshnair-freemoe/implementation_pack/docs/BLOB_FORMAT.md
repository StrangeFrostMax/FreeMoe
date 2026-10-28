# Native Blob Format Summary
A blob is a binary file containing actual tensor bytes. Proposed regions: versioned header, tensor directory, payload, integrity metadata. A manifest maps logical tensor IDs to blob ranges and generations. One tensor may span blobs; one blob may contain several tensors; experts are logical units, not file boundaries.

Use fixed-width fields and explicit byte order. Never write native C++ structs directly. Validate offsets and shape products for overflow. Keep source and published versions immutable. Use staging writes, checksums, journal/manifest generations, and recovery. Begin with tiny files and F32 tests; add quantised encodings only when their exact block layout is implemented.
