cmake_minimum_required(VERSION 3.16)
project(freemoe VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

find_package(Threads REQUIRED)
file(GLOB_RECURSE VNPU_SOURCES CONFIGURE_DEPENDS third_party/vnpu/src/*.cpp)
set(FREEMOE_LIB_SOURCES
  src/common.cpp
  src/gguf.cpp
  src/io.cpp
  src/backend.cpp
  src/kernels.cpp
  src/quant.cpp
  src/support.cpp
  src/tokenizer.cpp
  src/residency.cpp
  src/engine.cpp)
set(FREEMOE_INCLUDE_DIRS
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${CMAKE_CURRENT_SOURCE_DIR}/third_party/vnpu/include
  ${CMAKE_CURRENT_SOURCE_DIR}/third_party/vnpu/src)

function(freemoe_warnings target)
  if(MSVC)
    target_compile_options(${target} PRIVATE /W4 /WX)
  else()
    target_compile_options(${target} PRIVATE
      -Wall -Wextra -Werror -Wno-unused-parameter
      -Wno-missing-field-initializers -ffp-contract=fast)
  endif()
endfunction()

add_executable(freemoe src/main.cpp ${FREEMOE_LIB_SOURCES} ${VNPU_SOURCES})
target_include_directories(freemoe PRIVATE ${FREEMOE_INCLUDE_DIRS})
target_link_libraries(freemoe PRIVATE Threads::Threads)
freemoe_warnings(freemoe)

add_executable(freemoe-tests tests/unit/tests.cpp ${FREEMOE_LIB_SOURCES} ${VNPU_SOURCES})
target_include_directories(freemoe-tests PRIVATE ${FREEMOE_INCLUDE_DIRS})
target_link_libraries(freemoe-tests PRIVATE Threads::Threads)
freemoe_warnings(freemoe-tests)

add_executable(vnpu-selftest
  third_party/vnpu/tools/selftest_main.cpp
  third_party/vnpu/tools/selftest.cpp
  ${VNPU_SOURCES})
target_include_directories(vnpu-selftest PRIVATE ${FREEMOE_INCLUDE_DIRS})
target_link_libraries(vnpu-selftest PRIVATE Threads::Threads)
freemoe_warnings(vnpu-selftest)

add_executable(freemoe-fuzz tests/fuzz_gguf.cpp src/common.cpp src/gguf.cpp)
target_include_directories(freemoe-fuzz PRIVATE ${FREEMOE_INCLUDE_DIRS})
target_link_libraries(freemoe-fuzz PRIVATE Threads::Threads)
if(MSVC)
  target_compile_options(freemoe-fuzz PRIVATE /W4 /WX /Od)
else()
  target_compile_options(freemoe-fuzz PRIVATE
    -Wall -Wextra -Werror -O1 -g -fno-omit-frame-pointer
    -fsanitize=address,undefined)
  target_link_options(freemoe-fuzz PRIVATE -fsanitize=address,undefined)
endif()

enable_testing()
add_test(NAME freemoe_unit COMMAND freemoe-tests)
add_test(NAME vnpu_selftest COMMAND vnpu-selftest)
set_tests_properties(vnpu_selftest PROPERTIES ENVIRONMENT "VNPU_FORCE_ISA=avx2_dual")
add_test(NAME fuzz_quick COMMAND freemoe-fuzz 2 0xFEEE0E)
set_tests_properties(fuzz_quick PROPERTIES ENVIRONMENT
  "ASAN_OPTIONS=quarantine_size_mb=16:detect_leaks=1:allocator_may_return_null=1;UBSAN_OPTIONS=halt_on_error=1")
set_tests_properties(freemoe_unit vnpu_selftest fuzz_quick PROPERTIES
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
add_test(NAME trace_replay
  COMMAND python3 ${CMAKE_CURRENT_SOURCE_DIR}/tools/trace_replay.py
          --trace ${CMAKE_CURRENT_SOURCE_DIR}/build/test-tmp/replay-trace.jsonl
          --metrics ${CMAKE_CURRENT_SOURCE_DIR}/build/test-tmp/replay-metrics.json --check)
add_test(NAME trace_event_types
  COMMAND python3 ${CMAKE_CURRENT_SOURCE_DIR}/tools/check_trace.py
          ${CMAKE_CURRENT_SOURCE_DIR}/build/test-tmp/replay-trace.jsonl
          ${CMAKE_CURRENT_SOURCE_DIR}/build/test-tmp/prefetch-trace.jsonl
          ${CMAKE_CURRENT_SOURCE_DIR}/build/test-tmp/cachehit-trace.jsonl)
set_tests_properties(trace_replay trace_event_types PROPERTIES DEPENDS freemoe_unit)

add_custom_target(fixtures
  COMMAND ${CMAKE_COMMAND} -E echo "Regenerating deterministic fixtures"
  COMMAND python3 ${CMAKE_CURRENT_SOURCE_DIR}/tools/make_fixtures.py --all
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
  COMMENT "Regenerate GGUF fixtures and reference outputs")
