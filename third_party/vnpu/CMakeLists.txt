cmake_minimum_required(VERSION 3.16)
project(vnpu VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

find_package(Threads REQUIRED)

option(VNPU_BUILD_SHARED "Build shared library" ON)
option(VNPU_BUILD_STATIC "Build static library" ON)
option(VNPU_BUILD_TOOLS "Build CLI, node and tests" ON)
option(VNPU_BUILD_TESTS "Build selftest executable" ON)

set(VNPU_SOURCES
  src/core/runtime.cpp
  src/core/scheduler.cpp
  src/core/autotune.cpp
  src/core/backend.cpp
  src/kernels/kernels.cpp
  src/kernels/llm.cpp
  src/cluster/net.cpp
  src/cluster/node.cpp
  src/cluster/cluster.cpp
  src/compat/compat.cpp
)

set(VNPU_INCLUDE_DIRS
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${CMAKE_CURRENT_SOURCE_DIR}/src
)

function(vnpu_set_warnings target)
  if(MSVC)
    target_compile_options(${target} PRIVATE /W4 /O2 /DNDEBUG)
  else()
    target_compile_options(${target} PRIVATE
      -Wall -Wextra -Wno-unused-parameter
      -Wno-missing-field-initializers
      -O3 -fno-math-errno -ffp-contract=fast -funroll-loops)
  endif()
endfunction()

if(VNPU_BUILD_SHARED)
  add_library(vnpu_shared SHARED ${VNPU_SOURCES})
  target_include_directories(vnpu_shared PUBLIC ${VNPU_INCLUDE_DIRS})
  target_compile_definitions(vnpu_shared PRIVATE VNPU_BUILD_SHARED)
  target_link_libraries(vnpu_shared PRIVATE Threads::Threads)
  if(WIN32)
    target_link_libraries(vnpu_shared PRIVATE ws2_32)
  endif()
  set_target_properties(vnpu_shared PROPERTIES
    OUTPUT_NAME vnpu
    VERSION ${PROJECT_VERSION}
    SOVERSION 1
    CXX_VISIBILITY_PRESET hidden
    VISIBILITY_INLINES_HIDDEN ON)
  vnpu_set_warnings(vnpu_shared)
endif()

if(VNPU_BUILD_STATIC)
  add_library(vnpu_static STATIC ${VNPU_SOURCES})
  target_include_directories(vnpu_static PUBLIC ${VNPU_INCLUDE_DIRS})
  target_link_libraries(vnpu_static PRIVATE Threads::Threads)
  if(WIN32)
    target_link_libraries(vnpu_static PRIVATE ws2_32)
  endif()
  set_target_properties(vnpu_static PROPERTIES OUTPUT_NAME vnpu)
  vnpu_set_warnings(vnpu_static)
endif()

if(VNPU_BUILD_TOOLS)
  if(TARGET vnpu_shared)
    set(VNPU_LINK_TARGET vnpu_shared)
  else()
    set(VNPU_LINK_TARGET vnpu_static)
  endif()

  add_executable(vnpu-cli
    tools/vnpu_cli.cpp
    tools/selftest.cpp
    tools/distributed_demo.cpp
    tools/llm_bench.cpp)
  target_include_directories(vnpu-cli PRIVATE ${VNPU_INCLUDE_DIRS})
  target_link_libraries(vnpu-cli PRIVATE ${VNPU_LINK_TARGET} Threads::Threads)
  vnpu_set_warnings(vnpu-cli)

  add_executable(vnpu-node tools/node_main.cpp)
  target_include_directories(vnpu-node PRIVATE ${VNPU_INCLUDE_DIRS})
  target_link_libraries(vnpu-node PRIVATE ${VNPU_LINK_TARGET} Threads::Threads)
  vnpu_set_warnings(vnpu-node)

  if(VNPU_BUILD_TESTS)
    add_executable(vnpu-selftest
      tools/selftest_main.cpp
      tools/selftest.cpp)
    target_include_directories(vnpu-selftest PRIVATE ${VNPU_INCLUDE_DIRS})
    target_link_libraries(vnpu-selftest PRIVATE ${VNPU_LINK_TARGET} Threads::Threads)
    vnpu_set_warnings(vnpu-selftest)
    enable_testing()
    add_test(NAME vnpu_selftest COMMAND vnpu-selftest)

    add_executable(vnpu-fuzz tests/fuzz_shapes.cpp)
    target_include_directories(vnpu-fuzz PRIVATE ${VNPU_INCLUDE_DIRS})
    target_link_libraries(vnpu-fuzz PRIVATE ${VNPU_LINK_TARGET} Threads::Threads)
    vnpu_set_warnings(vnpu-fuzz)
    add_test(NAME vnpu_fuzz COMMAND vnpu-fuzz 60)
  endif()
endif()

set(VNPU_INSTALL_TARGETS)
if(TARGET vnpu_shared)
  list(APPEND VNPU_INSTALL_TARGETS vnpu_shared)
endif()
if(TARGET vnpu_static)
  list(APPEND VNPU_INSTALL_TARGETS vnpu_static)
endif()
if(VNPU_INSTALL_TARGETS)
  install(TARGETS ${VNPU_INSTALL_TARGETS}
          RUNTIME DESTINATION bin
          LIBRARY DESTINATION lib
          ARCHIVE DESTINATION lib)
endif()
install(DIRECTORY include/ DESTINATION include)
if(VNPU_BUILD_TOOLS)
  install(TARGETS vnpu-cli vnpu-node RUNTIME DESTINATION bin)
endif()
