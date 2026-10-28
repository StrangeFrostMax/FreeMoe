# FreeMoe correctness prototype - reproducible GNU Make build.
# Primary one-command build/test: `make -j2 test`

CXX ?= g++
BUILD := build
CPPFLAGS ?= -Iinclude -Ithird_party/vnpu/include -Ithird_party/vnpu/src
CXXFLAGS ?= -std=c++17 -O2 -pthread -ffp-contract=fast \
            -Wall -Wextra -Werror -Wno-unused-parameter \
            -Wno-missing-field-initializers
CXXFLAGS += -MMD -MP
LDFLAGS ?= -pthread
FUZZ_FLAGS := -std=c++17 -O1 -g -pthread -fno-omit-frame-pointer \
              -fsanitize=address,undefined -Wall -Wextra -Werror

OUR_LIB_SRCS := \
  src/common.cpp \
  src/gguf.cpp \
  src/io.cpp \
  src/backend.cpp \
  src/kernels.cpp \
  src/quant.cpp \
  src/support.cpp \
  src/tokenizer.cpp \
  src/residency.cpp \
  src/engine.cpp
APP_MAIN := src/main.cpp
TEST_MAIN := tests/unit/tests.cpp
FUZZ_MAIN := tests/fuzz_gguf.cpp
VNPU_SRCS := $(shell find third_party/vnpu/src -name '*.cpp' | sort)

OUR_OBJS := $(patsubst src/%.cpp,$(BUILD)/obj/freemoe/%.o,$(OUR_LIB_SRCS))
MAIN_OBJ := $(BUILD)/obj/freemoe/main.o
TEST_OBJ := $(BUILD)/obj/tests.o
VNPU_OBJS := $(patsubst third_party/vnpu/src/%.cpp,$(BUILD)/obj/vnpu/%.o,$(VNPU_SRCS))
SELFTEST_OBJS := $(BUILD)/obj/vnpu-selftest-main.o $(BUILD)/obj/vnpu-selftest.o

FUZZ_SECONDS ?= 72
DEPS := $(OUR_OBJS:.o=.d) $(MAIN_OBJ:.o=.d) $(TEST_OBJ:.o=.d) \
        $(VNPU_OBJS:.o=.d) $(SELFTEST_OBJS:.o=.d)

.PHONY: all test zen3-fallback fuzz soak bench benchmark fixtures clean distclean help

all: $(BUILD)/freemoe

$(BUILD)/freemoe: $(OUR_OBJS) $(MAIN_OBJ) $(VNPU_OBJS)
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) $^ -o $@ $(LDFLAGS)

$(BUILD)/tests: $(OUR_OBJS) $(TEST_OBJ) $(VNPU_OBJS)
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) $^ -o $@ $(LDFLAGS)

$(BUILD)/vnpu-selftest: $(SELFTEST_OBJS) $(VNPU_OBJS)
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) $^ -o $@ $(LDFLAGS)

$(BUILD)/fuzz_gguf: $(FUZZ_MAIN) src/common.cpp src/gguf.cpp src/quant.cpp src/kernels.cpp src/backend.cpp $(VNPU_SRCS)
	@mkdir -p $(dir $@)
	$(CXX) $(FUZZ_FLAGS) $(CPPFLAGS) $^ -o $@ $(LDFLAGS)

$(BUILD)/obj/freemoe/%.o: src/%.cpp
	@mkdir -p $(dir $@)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -c $< -o $@

$(BUILD)/obj/vnpu/%.o: third_party/vnpu/src/%.cpp
	@mkdir -p $(dir $@)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -c $< -o $@

$(BUILD)/obj/tests.o: $(TEST_MAIN)
	@mkdir -p $(dir $@)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -c $< -o $@

$(BUILD)/obj/vnpu-selftest-main.o: third_party/vnpu/tools/selftest_main.cpp
	@mkdir -p $(dir $@)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -c $< -o $@

$(BUILD)/obj/vnpu-selftest.o: third_party/vnpu/tools/selftest.cpp
	@mkdir -p $(dir $@)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -c $< -o $@

test: all $(BUILD)/tests $(BUILD)/vnpu-selftest $(BUILD)/fuzz_gguf
	@mkdir -p $(BUILD)/test-tmp
	./$(BUILD)/tests
	VNPU_FORCE_ISA=avx2_dual ./$(BUILD)/vnpu-selftest
	ASAN_OPTIONS=quarantine_size_mb=16:detect_leaks=1:allocator_may_return_null=1 UBSAN_OPTIONS=halt_on_error=1 ./$(BUILD)/fuzz_gguf 2 0xFEEE0E > $(BUILD)/test-tmp/fuzz-quick.json
	python3 tools/trace_replay.py --trace $(BUILD)/test-tmp/replay-trace.jsonl --metrics $(BUILD)/test-tmp/replay-metrics.json --check
	python3 tools/check_trace.py $(BUILD)/test-tmp/replay-trace.jsonl $(BUILD)/test-tmp/prefetch-trace.jsonl $(BUILD)/test-tmp/cachehit-trace.jsonl
	python3 scripts/check_architecture.py
	python3 scripts/check_dependencies.py

zen3-fallback: all $(BUILD)/tests $(BUILD)/vnpu-selftest
	python3 scripts/zen3_fallback_check.py

fuzz: $(BUILD)/fuzz_gguf
	@mkdir -p $(BUILD)/test-tmp
	ASAN_OPTIONS=quarantine_size_mb=16:detect_leaks=1:allocator_may_return_null=1 UBSAN_OPTIONS=halt_on_error=1 ./$(BUILD)/fuzz_gguf $(FUZZ_SECONDS) 0xFEEE0E | tee $(BUILD)/test-tmp/fuzz-soak.json

soak: all
	bash scripts/soak.sh

bench: all
	@mkdir -p docs/evidence
	$(BUILD)/freemoe bench --iterations 20 --out docs/evidence/kernels_bench.json > /dev/null
	@python3 -c "import json;d=json.load(open('docs/evidence/kernels_bench.json'));print('kernel bench: isa=%s threads=%d rows=%d' % (d['resolved_isa'], d['resolved_threads'], len(d['results'])))"

benchmark: all
	bash benchmarks/run_repeats.sh configs/benchmark.env

fixtures:
	python3 tools/make_fixtures.py --all
	@echo "Fixtures regenerated. Commit them and record the new D5 pin in docs/DECISIONS.md before release."

clean:
	rm -rf $(BUILD)

distclean: clean
	rm -rf results traces

help:
	@echo "Targets: all, test, zen3-fallback, fuzz FUZZ_SECONDS=N, soak, benchmark, fixtures, clean, distclean"

-include $(DEPS)
