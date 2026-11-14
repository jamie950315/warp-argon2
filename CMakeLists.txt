cmake_minimum_required(VERSION 3.22)
project(warp_argon2 VERSION 0.1.0 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(Threads REQUIRED)

# Minimal vendored phc-winner-argon2 reference library for RFC 9106 KATs,
# arbitrary-input H0 compatibility, and host CPU verification.
add_library(argon2_ref STATIC
    third_party/argon2/src/argon2.c
    third_party/argon2/src/core.c
    third_party/argon2/src/encoding.c
    third_party/argon2/src/ref.c
    third_party/argon2/src/thread.c
    third_party/argon2/src/blake2/blake2b.c
)
target_include_directories(argon2_ref PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/argon2/include
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/argon2/src
)
target_compile_definitions(argon2_ref PRIVATE ARGON2_NO_THREADS)
if(NOT MSVC)
    target_compile_options(argon2_ref PRIVATE -O3)
endif()

if(APPLE)
    option(WARP_ARGON2_ENABLE_METAL "Enable Apple Silicon Metal backend" ON)
    option(WARP_ARGON2_ENABLE_CUDA "Enable NVIDIA CUDA backend" OFF)
else()
    option(WARP_ARGON2_ENABLE_METAL "Enable Apple Silicon Metal backend" OFF)
    include(CheckLanguage)
    check_language(CUDA)
    if(CMAKE_CUDA_COMPILER)
        option(WARP_ARGON2_ENABLE_CUDA "Enable NVIDIA CUDA backend" ON)
    else()
        option(WARP_ARGON2_ENABLE_CUDA "Enable NVIDIA CUDA backend" OFF)
    endif()
endif()

set(WARP_ARGON2_SOURCES
    src/cpu_ref.cpp
)

if(WARP_ARGON2_ENABLE_METAL)
    find_program(XCRUN_EXECUTABLE xcrun REQUIRED)
    list(APPEND WARP_ARGON2_SOURCES src/metal/metal_backend.mm)

    set(WARP_METAL_SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/src/metal/argon2id.metal")
    set(WARP_METAL_AIR "${CMAKE_BINARY_DIR}/argon2id.air")
    set(WARP_METAL_LIB "${CMAKE_BINARY_DIR}/warp_argon2.metallib")

    add_custom_command(
        OUTPUT "${WARP_METAL_AIR}"
        COMMAND ${XCRUN_EXECUTABLE} -sdk macosx metal -O3 -c "${WARP_METAL_SOURCE}" -o "${WARP_METAL_AIR}"
        DEPENDS "${WARP_METAL_SOURCE}"
        COMMENT "Compiling Metal compute shader argon2id.metal"
        VERBATIM
    )
    add_custom_command(
        OUTPUT "${WARP_METAL_LIB}"
        COMMAND ${XCRUN_EXECUTABLE} -sdk macosx metallib "${WARP_METAL_AIR}" -o "${WARP_METAL_LIB}"
        DEPENDS "${WARP_METAL_AIR}"
        COMMENT "Linking Metal library warp_argon2.metallib"
        VERBATIM
    )
    add_custom_target(warp_argon2_metallib ALL DEPENDS "${WARP_METAL_LIB}")
endif()

if(WARP_ARGON2_ENABLE_CUDA)
    if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
        set(CMAKE_CUDA_ARCHITECTURES "75;80;86;89;90")
    endif()
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 20)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)

    set(WARP_A2_MIN_BLOCKS_PER_SM "16" CACHE STRING "minBlocksPerSM for fill kernel")
    set(WARP_A2_STAGE_STRIDE "32" CACHE STRING "Shared stage second dimension (32..44)")
    set(WARP_A2_WORKSPACE_LOAD_POLICY "2" CACHE STRING "Workspace load cache policy (0=ldg,1=ca,2=cg,3=default)")
    set(WARP_A2_WORKSPACE_LOAD_IMPL "2" CACHE STRING "Workspace load impl (0=ptx,1=ptx+mem,2=__ld*,3=default)")
    set(WARP_A2_WORKSPACE_VISIBILITY_MODE "1" CACHE STRING "Post-store visibility (0=fence+sync,1=sync,2=none)")
    set(WARP_A2_REGIO_TRANSPOSE_MASK "0" CACHE STRING "REG_IO transpose mask (0..3)")
    set(WARP_A2_DD_PIPELINE_POINT "0" CACHE STRING "DD pipeline issue point (0..3)")
    set(WARP_A2_DD_POST_STORE_SYNC "1" CACHE STRING "DD post-store sync (0/1)")
    option(WARP_CUDA_LINEINFO "Emit -lineinfo for Nsight Compute" ON)
    option(WARP_CUDA_PTXAS_VERBOSE "Pass -Xptxas=-v to inspect registers and spills" ON)

    list(APPEND WARP_ARGON2_SOURCES src/cuda/cuda_backend.cu)
endif()

add_library(warp_argon2 STATIC ${WARP_ARGON2_SOURCES})
target_include_directories(warp_argon2 PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/argon2/include
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/argon2/src
)
target_link_libraries(warp_argon2 PUBLIC argon2_ref Threads::Threads)

if(NOT MSVC)
    target_compile_options(warp_argon2 PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:-O3;-Wall;-Wextra>
    )
endif()

if(WARP_ARGON2_ENABLE_METAL)
    target_compile_definitions(warp_argon2 PUBLIC WARP_ARGON2_HAS_METAL=1)
    target_link_libraries(warp_argon2 PUBLIC "-framework Metal" "-framework Foundation")
    set_source_files_properties(src/metal/metal_backend.mm PROPERTIES
        COMPILE_FLAGS "-fobjc-arc -O3"
    )
else()
    target_compile_definitions(warp_argon2 PUBLIC WARP_ARGON2_HAS_METAL=0)
endif()

if(WARP_ARGON2_ENABLE_CUDA)
    target_compile_definitions(warp_argon2 PUBLIC WARP_ARGON2_HAS_CUDA=1)
    target_compile_definitions(warp_argon2 PRIVATE
        WARP_A2_MIN_BLOCKS_PER_SM=${WARP_A2_MIN_BLOCKS_PER_SM}
        WARP_A2_STAGE_STRIDE=${WARP_A2_STAGE_STRIDE}
        WARP_A2_WORKSPACE_LOAD_POLICY=${WARP_A2_WORKSPACE_LOAD_POLICY}
        WARP_A2_WORKSPACE_LOAD_IMPL=${WARP_A2_WORKSPACE_LOAD_IMPL}
        WARP_A2_WORKSPACE_VISIBILITY_MODE=${WARP_A2_WORKSPACE_VISIBILITY_MODE}
        WARP_A2_REGIO_TRANSPOSE_MASK=${WARP_A2_REGIO_TRANSPOSE_MASK}
        WARP_A2_DD_PIPELINE_POINT=${WARP_A2_DD_PIPELINE_POINT}
        WARP_A2_DD_POST_STORE_SYNC=${WARP_A2_DD_POST_STORE_SYNC}
    )
    target_compile_options(warp_argon2 PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:-Xptxas=-O3>
    )
    if(WARP_CUDA_LINEINFO)
        target_compile_options(warp_argon2 PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>)
    endif()
    if(WARP_CUDA_PTXAS_VERBOSE)
        target_compile_options(warp_argon2 PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xptxas=-v>)
    endif()
    set_target_properties(warp_argon2 PROPERTIES
        CUDA_SEPARABLE_COMPILATION OFF
        CUDA_RESOLVE_DEVICE_SYMBOLS ON
        POSITION_INDEPENDENT_CODE ON
    )
else()
    target_compile_definitions(warp_argon2 PUBLIC WARP_ARGON2_HAS_CUDA=0)
endif()

add_executable(warp-argon2 src/cli_main.cpp)
target_link_libraries(warp-argon2 PRIVATE warp_argon2)
if(WARP_ARGON2_ENABLE_METAL)
    add_dependencies(warp-argon2 warp_argon2_metallib)
endif()

enable_testing()
add_executable(test_correctness tests/test_correctness.cpp)
target_link_libraries(test_correctness PRIVATE warp_argon2)
if(WARP_ARGON2_ENABLE_METAL)
    add_dependencies(test_correctness warp_argon2_metallib)
endif()
add_test(NAME warp_argon2_correctness COMMAND test_correctness)

if(WARP_ARGON2_ENABLE_CUDA)
    add_executable(test_cuda_fill_block tests/test_cuda_fill_block.cu)
    target_link_libraries(test_cuda_fill_block PRIVATE warp_argon2)
    target_compile_definitions(test_cuda_fill_block PRIVATE
        WARP_A2_STAGE_STRIDE=${WARP_A2_STAGE_STRIDE}
        WARP_A2_WORKSPACE_LOAD_POLICY=${WARP_A2_WORKSPACE_LOAD_POLICY}
        WARP_A2_WORKSPACE_LOAD_IMPL=${WARP_A2_WORKSPACE_LOAD_IMPL}
        WARP_A2_WORKSPACE_VISIBILITY_MODE=${WARP_A2_WORKSPACE_VISIBILITY_MODE}
        WARP_A2_REGIO_TRANSPOSE_MASK=${WARP_A2_REGIO_TRANSPOSE_MASK}
    )
    set_target_properties(test_cuda_fill_block PROPERTIES
        CUDA_SEPARABLE_COMPILATION OFF
        CUDA_RESOLVE_DEVICE_SYMBOLS ON
    )
    add_test(NAME cuda_fill_block COMMAND test_cuda_fill_block)
endif()
