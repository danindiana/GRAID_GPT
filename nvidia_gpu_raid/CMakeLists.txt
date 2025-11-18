cmake_minimum_required(VERSION 3.18)
project(NVIDIA_GPU_RAID LANGUAGES CXX CUDA)

# C++17 required for modern features
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)

# Find CUDA
find_package(CUDA REQUIRED)
if(CUDA_FOUND)
    message(STATUS "CUDA found: ${CUDA_VERSION}")
    message(STATUS "CUDA toolkit root: ${CUDA_TOOLKIT_ROOT_DIR}")
endif()

# Options
option(ENABLE_TENSTORRENT "Enable experimental Tenstorrent integration" OFF)
option(BUILD_TESTS "Build unit tests" ON)
option(BUILD_BENCHMARKS "Build performance benchmarks" ON)

# Auto-detect GPU architecture if not specified
if(NOT DEFINED CUDA_ARCH)
    include(FindCUDA/select_compute_arch)
    CUDA_DETECT_INSTALLED_GPUS(DETECTED_GPUS)
    string(REPLACE "." "" CUDA_ARCH_LIST "${DETECTED_GPUS}")
    list(GET CUDA_ARCH_LIST 0 CUDA_ARCH)
    message(STATUS "Auto-detected CUDA architecture: sm_${CUDA_ARCH}")
else()
    message(STATUS "Using specified CUDA architecture: sm_${CUDA_ARCH}")
endif()

# Set CUDA architecture
set(CMAKE_CUDA_ARCHITECTURES ${CUDA_ARCH})

# CUDA compilation flags
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode arch=compute_${CUDA_ARCH},code=sm_${CUDA_ARCH}")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --use_fast_math")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr")
set(CMAKE_CUDA_FLAGS_RELEASE "-O3 -DNDEBUG")
set(CMAKE_CUDA_FLAGS_DEBUG "-g -G -O0")

# Include directories
include_directories(
    ${CMAKE_SOURCE_DIR}/include
    ${CUDA_INCLUDE_DIRS}
)

# CUDA kernel libraries
add_library(raid5_kernels STATIC
    kernels/raid5_xor.cu
)

add_library(raid6_kernels STATIC
    kernels/raid6_rs.cu
)

add_library(gf_kernels STATIC
    kernels/galois_field.cu
)

add_library(reconstruction_kernels STATIC
    kernels/reconstruction.cu
)

# Set properties for kernel libraries
set_target_properties(raid5_kernels raid6_kernels gf_kernels reconstruction_kernels
    PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON
)

# Main GPU RAID library
add_library(gpu_raid SHARED
    src/gpu_raid_manager.cpp
    src/device_manager.cpp
    src/memory_pool.cpp
    src/stripe_buffer.cpp
    src/performance_monitor.cpp
)

target_link_libraries(gpu_raid
    raid5_kernels
    raid6_kernels
    gf_kernels
    reconstruction_kernels
    ${CUDA_LIBRARIES}
)

# Tenstorrent integration (optional)
if(ENABLE_TENSTORRENT)
    message(WARNING "Tenstorrent integration is EXPERIMENTAL and uses DEPRECATED SDK")

    # Try to find Tenstorrent SDK (likely to fail)
    find_path(TT_INCLUDE_DIR
        NAMES tt_device.h
        PATHS /opt/tenstorrent/include
        NO_DEFAULT_PATH
    )

    find_library(TT_LIBRARY
        NAMES tt_runtime
        PATHS /opt/tenstorrent/lib
        NO_DEFAULT_PATH
    )

    if(TT_INCLUDE_DIR AND TT_LIBRARY)
        message(STATUS "Tenstorrent SDK found (deprecated version)")
        add_library(tt_compat STATIC
            tenstorrent_integration/tt_tensix_compat.cpp
        )
        target_include_directories(tt_compat PRIVATE ${TT_INCLUDE_DIR})
        target_link_libraries(tt_compat ${TT_LIBRARY})
        target_compile_definitions(tt_compat PRIVATE ENABLE_TENSTORRENT)

        target_link_libraries(gpu_raid tt_compat)
        target_compile_definitions(gpu_raid PRIVATE ENABLE_TENSTORRENT)
    else()
        message(WARNING "Tenstorrent SDK not found, disabling integration")
        set(ENABLE_TENSTORRENT OFF)
    endif()
endif()

# Command-line tool
add_executable(gpu_raid_cli
    tools/gpu_raid_cli.cpp
)

target_link_libraries(gpu_raid_cli gpu_raid)

# Tests
if(BUILD_TESTS)
    enable_testing()

    # Unit tests
    add_executable(test_xor_kernels
        tests/unit_tests/test_xor_kernels.cu
    )
    target_link_libraries(test_xor_kernels raid5_kernels ${CUDA_LIBRARIES})
    add_test(NAME test_xor_kernels COMMAND test_xor_kernels)

    add_executable(test_gf_arithmetic
        tests/unit_tests/test_gf_arithmetic.cu
    )
    target_link_libraries(test_gf_arithmetic gf_kernels ${CUDA_LIBRARIES})
    add_test(NAME test_gf_arithmetic COMMAND test_gf_arithmetic)

    add_executable(test_raid6_encoding
        tests/unit_tests/test_raid6_encoding.cu
    )
    target_link_libraries(test_raid6_encoding raid6_kernels gf_kernels ${CUDA_LIBRARIES})
    add_test(NAME test_raid6_encoding COMMAND test_raid6_encoding)

    add_executable(test_reconstruction
        tests/unit_tests/test_reconstruction.cu
    )
    target_link_libraries(test_reconstruction reconstruction_kernels gf_kernels ${CUDA_LIBRARIES})
    add_test(NAME test_reconstruction COMMAND test_reconstruction)
endif()

# Benchmarks
if(BUILD_BENCHMARKS)
    add_executable(bench_xor_throughput
        tests/benchmarks/bench_xor_throughput.cu
    )
    target_link_libraries(bench_xor_throughput raid5_kernels ${CUDA_LIBRARIES})

    add_executable(bench_raid6_throughput
        tests/benchmarks/bench_raid6_throughput.cu
    )
    target_link_libraries(bench_raid6_throughput raid6_kernels gf_kernels ${CUDA_LIBRARIES})

    add_executable(bench_rebuild_speed
        tests/benchmarks/bench_rebuild_speed.cu
    )
    target_link_libraries(bench_rebuild_speed reconstruction_kernels gf_kernels ${CUDA_LIBRARIES})
endif()

# Installation
install(TARGETS gpu_raid gpu_raid_cli
    LIBRARY DESTINATION lib
    RUNTIME DESTINATION bin
)

install(DIRECTORY include/
    DESTINATION include/gpu_raid
)

install(FILES
    config/rtx3080_config.json
    config/rtx3060_config.json
    config/quadro4000_config.json
    DESTINATION etc/gpu_raid
)

# Print configuration summary
message(STATUS "")
message(STATUS "========================================")
message(STATUS "NVIDIA GPU RAID Configuration Summary")
message(STATUS "========================================")
message(STATUS "CUDA Version: ${CUDA_VERSION}")
message(STATUS "CUDA Architecture: sm_${CUDA_ARCH}")
message(STATUS "Tenstorrent Support: ${ENABLE_TENSTORRENT}")
message(STATUS "Build Tests: ${BUILD_TESTS}")
message(STATUS "Build Benchmarks: ${BUILD_BENCHMARKS}")
message(STATUS "========================================")
message(STATUS "")
