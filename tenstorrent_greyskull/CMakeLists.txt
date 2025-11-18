cmake_minimum_required(VERSION 3.16)
project(TenstorrentGreyskullGRAID VERSION 0.1.0 LANGUAGES CXX)

# C++ Standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Build type
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Compiler flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Wpedantic")
set(CMAKE_CXX_FLAGS_DEBUG "-g -O0")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native -DNDEBUG")

# Project information
message(STATUS "")
message(STATUS "==============================================")
message(STATUS "  Tenstorrent Grayskull GRAID")
message(STATUS "  GPU-Accelerated RAID with Reed-Solomon")
message(STATUS "==============================================")
message(STATUS "  Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "  C++ Standard: C++${CMAKE_CXX_STANDARD}")
message(STATUS "  Compiler: ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}")
message(STATUS "==============================================")
message(STATUS "")

# Find TT-Metalium (if installed)
# Note: Grayskull support discontinued, using v0.55 if available
find_path(TT_METAL_INCLUDE_DIR
    NAMES tt_metal/host_api.hpp
    PATHS
        $ENV{TT_METAL_HOME}/tt_metal
        /usr/local/include
        /usr/include
)

if(TT_METAL_INCLUDE_DIR)
    message(STATUS "Found TT-Metal includes: ${TT_METAL_INCLUDE_DIR}")
    set(HAVE_TT_METAL ON)
else()
    message(WARNING "TT-Metal not found. Building headers-only library.")
    message(WARNING "Install TT-Metalium v0.55 for full functionality.")
    set(HAVE_TT_METAL OFF)
endif()

# Include directories
include_directories(
    ${PROJECT_SOURCE_DIR}/include
)

if(HAVE_TT_METAL)
    include_directories(${TT_METAL_INCLUDE_DIR})
endif()

# Source files
set(GRAID_SOURCES
    src/tt_graid_device.cpp
    src/tt_graid_memory.cpp
    src/tt_graid_reed_solomon.cpp
)

# Create library (header-only if TT-Metal not available)
if(HAVE_TT_METAL)
    # Shared library
    add_library(tt_graid SHARED ${GRAID_SOURCES})
    target_include_directories(tt_graid PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
    )

    # Link TT-Metal libraries
    find_library(TT_METAL_LIB
        NAMES tt_metal
        PATHS
            $ENV{TT_METAL_HOME}/build/lib
            /usr/local/lib
            /usr/lib
    )

    if(TT_METAL_LIB)
        target_link_libraries(tt_graid PUBLIC ${TT_METAL_LIB})
        message(STATUS "Linked TT-Metal library: ${TT_METAL_LIB}")
    else()
        message(WARNING "TT-Metal library not found")
    endif()

    set_target_properties(tt_graid PROPERTIES
        VERSION ${PROJECT_VERSION}
        SOVERSION 0
    )
else()
    # Header-only interface library
    add_library(tt_graid INTERFACE)
    target_include_directories(tt_graid INTERFACE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
    )
endif()

# Examples
option(BUILD_EXAMPLES "Build example programs" ON)

if(BUILD_EXAMPLES AND HAVE_TT_METAL)
    message(STATUS "Building examples (TT-Metal required)")

    # Example 1: Device Enumeration
    add_executable(01_device_enumeration
        examples/01_device_enumeration.cpp
    )
    target_link_libraries(01_device_enumeration PRIVATE tt_graid)

    # Example 2: Memory Operations
    add_executable(02_memory_operations
        examples/02_memory_operations.cpp
    )
    target_link_libraries(02_memory_operations PRIVATE tt_graid)

    # Example 3: Reed-Solomon RAID
    add_executable(03_reed_solomon_raid
        examples/03_reed_solomon_raid.cpp
    )
    target_link_libraries(03_reed_solomon_raid PRIVATE tt_graid)

    # Install examples
    install(TARGETS
        01_device_enumeration
        02_memory_operations
        03_reed_solomon_raid
        RUNTIME DESTINATION bin/examples
    )
endif()

# Installation
if(HAVE_TT_METAL)
    install(TARGETS tt_graid
        EXPORT TT_GRAID_Targets
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib
        RUNTIME DESTINATION bin
        INCLUDES DESTINATION include
    )
else()
    install(TARGETS tt_graid
        EXPORT TT_GRAID_Targets
    )
endif()

install(DIRECTORY include/
    DESTINATION include
    FILES_MATCHING PATTERN "*.hpp"
)

install(FILES README.md
    DESTINATION share/doc/tt_graid
)

# Export targets
install(EXPORT TT_GRAID_Targets
    FILE TT_GRAID_Targets.cmake
    NAMESPACE TT_GRAID::
    DESTINATION lib/cmake/TT_GRAID
)

# Package configuration
include(CMakePackageConfigHelpers)

configure_package_config_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/Config.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/TT_GRAID_Config.cmake
    INSTALL_DESTINATION lib/cmake/TT_GRAID
)

write_basic_package_version_file(
    ${CMAKE_CURRENT_BINARY_DIR}/TT_GRAID_ConfigVersion.cmake
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion
)

install(FILES
    ${CMAKE_CURRENT_BINARY_DIR}/TT_GRAID_Config.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/TT_GRAID_ConfigVersion.cmake
    DESTINATION lib/cmake/TT_GRAID
)

# Print summary
message(STATUS "")
message(STATUS "==============================================")
message(STATUS "  Configuration Summary")
message(STATUS "==============================================")
message(STATUS "  TT-Metal Support:  ${HAVE_TT_METAL}")
message(STATUS "  Build Examples:    ${BUILD_EXAMPLES}")
message(STATUS "  Install Prefix:    ${CMAKE_INSTALL_PREFIX}")
message(STATUS "==============================================")
message(STATUS "")
message(STATUS "Build commands:")
message(STATUS "  mkdir build && cd build")
message(STATUS "  cmake ..")
message(STATUS "  make -j$(nproc)")
message(STATUS "")
message(STATUS "Run examples:")
message(STATUS "  ./01_device_enumeration")
message(STATUS "  ./02_memory_operations [size_mb]")
message(STATUS "  ./03_reed_solomon_raid [k] [m] [block_kb]")
message(STATUS "")
