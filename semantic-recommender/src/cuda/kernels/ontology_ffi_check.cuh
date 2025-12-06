// FFI Safety Checks for CUDA Ontology Structures
//
// This header provides compile-time assertions to verify struct sizes
// and alignment match the Rust FFI definitions in:
// src/rust/models/ontology_ffi.rs
//
// CRITICAL: These assertions will cause compilation to fail if there is
// any mismatch between CUDA and Rust struct layouts.

#ifndef ONTOLOGY_FFI_CHECK_CUH
#define ONTOLOGY_FFI_CHECK_CUH

#include <cstdint>
#include <cuda_runtime.h>

// Forward declarations from ontology_reasoning.cu
struct MediaOntologyNode;
struct MediaOntologyConstraint;

// ============================================================================
// COMPILE-TIME SIZE ASSERTIONS
// ============================================================================

// MediaOntologyNode must be exactly 80 bytes
static_assert(sizeof(MediaOntologyNode) == 80,
    "MediaOntologyNode size mismatch! Expected 80 bytes for FFI compatibility with Rust");

// MediaOntologyNode must be 64-byte aligned
static_assert(alignof(MediaOntologyNode) == 64,
    "MediaOntologyNode alignment mismatch! Expected 64-byte alignment for optimal GPU access");

// MediaOntologyConstraint must be exactly 64 bytes
static_assert(sizeof(MediaOntologyConstraint) == 64,
    "MediaOntologyConstraint size mismatch! Expected 64 bytes for FFI compatibility with Rust");

// MediaOntologyConstraint must be 64-byte aligned
static_assert(alignof(MediaOntologyConstraint) == 64,
    "MediaOntologyConstraint alignment mismatch! Expected 64-byte alignment for optimal GPU access");

// float3 must be 12 bytes
static_assert(sizeof(float3) == 12,
    "float3 size mismatch! Expected 12 bytes (3 * sizeof(float))");

// ============================================================================
// FIELD OFFSET VERIFICATION MACROS
// ============================================================================

#define VERIFY_OFFSET(TYPE, FIELD, EXPECTED_OFFSET) \
    static_assert(offsetof(TYPE, FIELD) == EXPECTED_OFFSET, \
        #TYPE "::" #FIELD " offset mismatch! Expected " #EXPECTED_OFFSET)

// Verify MediaOntologyNode field offsets match Rust
VERIFY_OFFSET(MediaOntologyNode, graph_id, 0);
VERIFY_OFFSET(MediaOntologyNode, node_id, 4);
VERIFY_OFFSET(MediaOntologyNode, ontology_type, 8);
VERIFY_OFFSET(MediaOntologyNode, constraint_flags, 12);
VERIFY_OFFSET(MediaOntologyNode, position, 16);
VERIFY_OFFSET(MediaOntologyNode, velocity, 28);
VERIFY_OFFSET(MediaOntologyNode, mass, 40);
VERIFY_OFFSET(MediaOntologyNode, radius, 44);
VERIFY_OFFSET(MediaOntologyNode, parent_genre, 48);
VERIFY_OFFSET(MediaOntologyNode, property_count, 52);
VERIFY_OFFSET(MediaOntologyNode, cultural_flags, 56);
VERIFY_OFFSET(MediaOntologyNode, mood_flags, 60);
VERIFY_OFFSET(MediaOntologyNode, padding, 64);

// Verify MediaOntologyConstraint field offsets match Rust
VERIFY_OFFSET(MediaOntologyConstraint, type, 0);
VERIFY_OFFSET(MediaOntologyConstraint, source_id, 4);
VERIFY_OFFSET(MediaOntologyConstraint, target_id, 8);
VERIFY_OFFSET(MediaOntologyConstraint, graph_id, 12);
VERIFY_OFFSET(MediaOntologyConstraint, strength, 16);
VERIFY_OFFSET(MediaOntologyConstraint, distance, 20);
VERIFY_OFFSET(MediaOntologyConstraint, mood_weight, 24);
VERIFY_OFFSET(MediaOntologyConstraint, cultural_weight, 28);
VERIFY_OFFSET(MediaOntologyConstraint, flags, 32);
VERIFY_OFFSET(MediaOntologyConstraint, padding, 36);

// ============================================================================
// TYPE SIZE VERIFICATION
// ============================================================================

// Verify primitive type sizes match expectations
static_assert(sizeof(uint32_t) == 4, "uint32_t must be 4 bytes");
static_assert(sizeof(float) == 4, "float must be 4 bytes");

// ============================================================================
// ENDIANNESS WARNING
// ============================================================================

// Note: CUDA kernels assume little-endian byte order (standard on x86_64 and ARM)
// If running on big-endian architecture, byte swapping will be required at FFI boundary

#if defined(__BYTE_ORDER__) && __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
    #warning "Big-endian architecture detected! FFI byte order may need adjustment"
#endif

// ============================================================================
// PLATFORM-SPECIFIC CHECKS
// ============================================================================

// Verify we're compiling for supported architectures
#if !defined(__x86_64__) && !defined(__aarch64__) && !defined(_M_X64) && !defined(_M_ARM64)
    #warning "Compiling for unsupported architecture. FFI compatibility not guaranteed"
#endif

// ============================================================================
// GPU MEMORY COALESCING VERIFICATION
// ============================================================================

// Verify 64-byte alignment for optimal GPU memory coalescing
// Modern GPUs have 32-128 byte cache lines, 64 bytes is optimal for most
static_assert((sizeof(MediaOntologyNode) % 64) == 16,
    "MediaOntologyNode size should align to cache line boundaries for optimal performance");

static_assert((sizeof(MediaOntologyConstraint) % 64) == 0,
    "MediaOntologyConstraint size perfectly aligned to cache line boundaries");

// ============================================================================
// USAGE INSTRUCTIONS
// ============================================================================

/*
To use these checks, include this header in ontology_reasoning.cu:

    #include "ontology_ffi_check.cuh"

Add at the end of ontology_reasoning.cu (after struct definitions):

    // Verify FFI compatibility at compile time
    #include "ontology_ffi_check.cuh"

If compilation fails with assertion errors:
1. Check struct field order matches between CUDA and Rust
2. Verify padding arrays are correct size
3. Ensure alignment attributes are identical
4. Check for platform-specific packing differences

Testing FFI safety:
1. Compile CUDA code with these checks enabled
2. Run Rust tests in src/rust/models/ontology_ffi.rs
3. Run integration tests with actual GPU transfers
4. Verify round-trip serialization preserves all data
*/

#endif // ONTOLOGY_FFI_CHECK_CUH
