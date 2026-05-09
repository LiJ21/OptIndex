cmake_minimum_required(VERSION 3.16)
project(FastMM LANGUAGES CXX C)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

include(FetchContent)
FetchContent_Declare(
    googletest
    URL https://github.com/google/googletest/archive/refs/tags/v1.17.0.zip
)
FetchContent_MakeAvailable(googletest)

FetchContent_Declare(
    googlebenchmark
    URL https://github.com/google/benchmark/archive/refs/tags/v1.9.1.zip
)
set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "" FORCE)
set(BENCHMARK_ENABLE_GTEST_TESTS OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googlebenchmark)

FetchContent_Declare(
    hdrhistogram
    GIT_REPOSITORY https://github.com/HdrHistogram/HdrHistogram_c.git
    GIT_TAG 0.11.8
)
set(HDR_HISTOGRAM_BUILD_PROGRAMS OFF CACHE BOOL "" FORCE)
set(HDR_LOG_REQUIRED OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(hdrhistogram)
FetchContent_GetProperties(hdrhistogram)

add_library(FastMM INTERFACE)
target_include_directories(FastMM INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_compile_features(FastMM INTERFACE cxx_std_20)

find_package(Boost REQUIRED COMPONENTS container)

file(GLOB_RECURSE FastMM_TEST_SOURCES CONFIGURE_DEPENDS test/*.cc)

foreach(test_src IN LISTS FastMM_TEST_SOURCES)
    file(RELATIVE_PATH rel_path ${CMAKE_CURRENT_SOURCE_DIR}/test ${test_src})
    string(REPLACE "/" "_" target_name ${rel_path})
    string(REPLACE "\\" "_" target_name ${target_name})
    string(REPLACE ".cc" "" target_name ${target_name})
    set(target_name test_${target_name})

    add_executable(${target_name} ${test_src})
    target_link_libraries(${target_name} PRIVATE FastMM Boost::container GTest::gtest_main)
    target_include_directories(${target_name} PRIVATE ${Boost_INCLUDE_DIRS})
endforeach()

file(GLOB_RECURSE FastMM_BENCH_SOURCES CONFIGURE_DEPENDS benchmarks/*.cc)

foreach(bench_src IN LISTS FastMM_BENCH_SOURCES)
    file(RELATIVE_PATH rel_path ${CMAKE_CURRENT_SOURCE_DIR}/benchmarks ${bench_src})
    string(REPLACE "/" "_" target_name ${rel_path})
    string(REPLACE "\\" "_" target_name ${target_name})
    string(REPLACE ".cc" "" target_name ${target_name})
    set(target_name bench_${target_name})

    add_executable(${target_name} ${bench_src})
    target_link_libraries(${target_name} PRIVATE
        FastMM
        Boost::container
        benchmark::benchmark
        hdr_histogram_static
    )
    target_include_directories(${target_name} PRIVATE
        ${Boost_INCLUDE_DIRS}
        ${hdrhistogram_SOURCE_DIR}/include
    )
    target_compile_options(${target_name} PRIVATE -O3 -march=native -DNDEBUG)
endforeach()
if(DEFINED SET_KN_VALUE)
  target_compile_definitions(bench_particle PRIVATE SET_KN=${SET_KN_VALUE})
  if(TARGET bench_particle_latency)
    target_compile_definitions(bench_particle_latency PRIVATE SET_KN=${SET_KN_VALUE})
  endif()
endif()

if(DEFINED REV_INDEX)
  target_compile_definitions(bench_particle PRIVATE REV_INDEX=${REV_INDEX})
  if(TARGET bench_particle_latency)
    target_compile_definitions(bench_particle_latency PRIVATE REV_INDEX=${REV_INDEX})
  endif()
endif()

if(DEFINED WITH_HASH)
  target_compile_definitions(bench_particle PRIVATE WITH_HASH=${WITH_HASH})
  if(TARGET bench_particle_latency)
    target_compile_definitions(bench_particle_latency PRIVATE WITH_HASH=${WITH_HASH})
  endif()
endif()