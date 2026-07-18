//******************************************************************************
// ET Vectorized Block Operations Library
// Provides optimized block-level operations using ET hardware vector instructions
//******************************************************************************

#ifndef BLOCK_OPS_H
#    define BLOCK_OPS_H

#    include "math_fp.h"
#    include "quants.h"

#    include <stdint.h>

//******************************************************************************
// Block Dot Product Operations
//******************************************************************************
inline void __attribute__((always_inline)) excl_mode(uint64_t val) {
    __asm__ __volatile__("csrw 0x7d3, %[csr_enc]\n" : : [csr_enc] "r"(val) : "x31");
}

static inline float compute_block_dot_product_q4_0(const block_q4_0 * a_block, const float * b_col_start) {
    // Set mask register to enable all 8 vector elements
    unsigned long temp_mask;
    __asm__ volatile("mova.x.m %0" : "=r"(temp_mask));  // Save current mask
    __asm__ volatile("mov.m.x m0, x0, 0xFF");           // Enable all 8 elements

    // Use f10 as accumulator, init to 0
    __asm__ volatile("fbci.ps f10, 0" ::: "f10");

    static const int32_t gather_pattern[8] = { 0, 1, 2, 3, 4, 5, 6, 7 };
    __asm__ volatile("flw.ps f31, %[gather]\n" : : [gather] "m"(*(const int32_t (*)[8]) gather_pattern) : "f31");

    // Process 32 elements in 2 chunks of 16 elements (8 bytes) each
    for (int chunk = 0; chunk < 2; chunk++) {
        int offset_a      = chunk * 8;
        int offset_b_low  = chunk * 8;       // Activations for lower nibbles
        int offset_b_high = chunk * 8 + 16;  // Activations for upper nibbles (16 elements later)

        __asm__ volatile(
            "fgb.ps f11, f31(%[a_ptr])\n"  // Gather 8 bytes (16 packed q4_0 weights)

            // 1. Extract & Multiply Lower Nibbles
            "fandi.pi f12, f11, 15\n"             // Mask lower 4 bits (x & 0xF)
            "faddi.pi f12, f12, -8\n"             // GGML offset to signed: (x & 0xF) - 8
            "fcvt.ps.pw f12, f12, rne\n"          // Convert INT32 to FP32
            "flw.ps f13, 0(%[b_low])\n"           // Load 8 B values (floats)
            "fmadd.ps f10, f12, f13, f10, rne\n"  // acc += A_low * B_low

            // 2. Extract & Multiply Upper Nibbles
            "fsrli.pi f14, f11, 4\n"              // Shift upper 4 bits down
            "fandi.pi f14, f14, 15\n"             // Mask new lower 4 bits
            "faddi.pi f14, f14, -8\n"             // GGML offset to signed
            "fcvt.ps.pw f14, f14, rne\n"          // Convert INT32 to FP32
            "flw.ps f15, 0(%[b_high])\n"          // Load next 8 B values (floats)
            "fmadd.ps f10, f14, f15, f10, rne\n"  // acc += A_high * B_high
            :
            : [a_ptr] "r"(&a_block->qs[offset_a]), [b_low] "r"(&b_col_start[offset_b_low]),
              [b_high] "r"(&b_col_start[offset_b_high])
            // Note: f10 is explicitly NOT listed in the clobbers here to ensure the compiler
            // preserves the running sum across C loop iterations safely.
            : "f11", "f12", "f13", "f14", "f15");
    }

    // Horizontal sum: reduce f10 into a single scalar
    float final_sum;
    __asm__ __volatile__(
        // Pairwise sum within each 128-bit half
        "fswizz.ps f1, f10, 0xB1 \n\t"  // Swaps: e0<->e1 and e2<->e3
        "fadd.ps   f2, f10, f1, rne \n\t"
        // Complete the sum for each 128-bit half
        "fswizz.ps f3, f2, 0x4E \n\t"  // Swaps: e0,e1 <-> e2,e3
        "fadd.ps   f4, f2, f3, rne \n\t"
        // Sum across the two 128b halfs
        "fmvz.x.ps t0, f4, 4 \n\t"
        "fbcx.ps   f5, t0 \n\t"
        "fadd.ps   %[vout], f4, f5, rne \n\t"
        : [vout] "=f"(final_sum)::"t0", "f1", "f2", "f3", "f4", "f5", "f10");

    // Restore original mask
    __asm__ volatile("mova.m.x %0" ::"r"(temp_mask));

    const float scale = fp16_to_fp32(a_block->d);
    return final_sum * scale;
}

// Compute dot product between dequantized q8_0 block and f32 column vector
// Vectorized: processes 8 elements at a time using ET vector instructions
// Block size: 32 int8 values (QK8_0)
static inline float compute_block_dot_product_q8_0(const block_q8_0 * a_block, const float * b_col_start) {
    // Set mask register to enable all 8 vector elements
    unsigned long temp_mask;
    __asm__ volatile("mova.x.m %0" : "=r"(temp_mask));  // Save current mask
    __asm__ volatile("mov.m.x m0, x0, 0xFF");           // Enable all 8 elements
    __asm__ volatile("fbci.pi f10, 0" ::: "f10");       // Use f10 as accumulator, init to 0

    static const int32_t gather_pattern[8] = { 0, 1, 2, 3, 4, 5, 6, 7 };

    __asm__ volatile("flw.ps f31, %[gather]\n" : : [gather] "m"(*(const int32_t (*)[8]) gather_pattern) : "f31");

    // Process 32 elements in 4 chunks of 8 elements each
    for (int chunk = 0; chunk < 4; chunk++) {
        int offset = chunk << 3;  // chunk * 8

        __asm__ volatile(
            "flw.ps f12, %[b_vec]\n"         // Load 8 B values (floats)
            "fgb.ps f11, f31(%[a_ptr])\n"    // Gather 8 int8 bytes from A using pattern
            "fcvt.ps.pw f11, f11\n"          // Convert int8 vector to float vector
            "fmadd.ps f10, f11, f12, f10\n"  // acc += a_vec * b_vec (8-wide)
            :
            : [a_ptr] "r"(&a_block->qs[offset]), [b_vec] "m"(*(const float (*)[8]) & b_col_start[offset]),
              [scale] "m"(a_block->d)
            : "f10", "f11", "f12");
    }

    // Horizontal sum: reduce f10 into a single scalar
    float final_sum;
    __asm__ __volatile__(
        // Pairwise sum within each 128-bit half
        "fswizz.ps f1, f10, 0xB1 \n\t"  // Swaps: e0<->e1 and e2<->e3
        "fadd.ps   f2, f10, f1, rne \n\t"
        // Complete the sum for each 128-bit half
        "fswizz.ps f3, f2, 0x4E \n\t"  // Swaps: e0,e1 <-> e2,e3
        "fadd.ps   f4, f2, f3, rne \n\t"
        // Sum across the two 128b halfs
        "fmvz.x.ps t0, f4, 4 \n\t"
        "fbcx.ps   f5, t0 \n\t"
        "fadd.ps   %[vout], f4, f5, rne \n\t"
        : [vout] "=f"(final_sum)::"t0", "f10", "f2", "f3", "f4", "f5");

    // Restore original mask
    __asm__ volatile("mova.m.x %0" ::"r"(temp_mask));

    const float scale = fp16_to_fp32(a_block->d);
    return final_sum * scale;
}

//******************************************************************************
// Split-phase Q8_0 dot product API
//
//   q8_dot_begin(st)      — save mask, set mask 0xFF
//   q8_dot_reset()        — zero vector accumulator f20
//   q8_dot_tile(q, b, n)  — accumulate n Q8_0 blocks into f20
//   q8_dot_reduce()       — horizontal sum of f20, return scalar float
//   q8_dot_teardown(st)   — restore original mask
//
// Register contract:
//   f20       — row accumulator (persistent across tiles, reset per row)
//   f31       — gather pattern (reloaded per q8_dot_tile call)
//   f10-f12   — scratch within tile
//   f15       — scale broadcast within tile
//   f1-f5, t0 — scratch within reduce
//******************************************************************************

static inline void __attribute__((always_inline)) q8_dot_reset(void) {
    __asm__ volatile("fbci.pi f20, 0" ::: "f20");
}

// Accumulate n_blocks Q8_0 blocks into f20.
// Uses fg32b.ps (fast gather with scalar pattern) for aligned chunks,
// falls back to fgb.ps for chunks crossing a 32-byte boundary.
static inline void __attribute__((always_inline)) q8_dot_tile(const block_q8_0 * q_row,
                                                              const float *      b_col,
                                                              int64_t            n_blocks) {
    // Use proven block-level dot product (handles mask internally)
    float scalar = 0.0f;
    for (int64_t kb = 0; kb < n_blocks; kb++) {
        scalar += compute_block_dot_product_q8_0(q_row + kb, b_col + (kb << 5));
    }
    // Broadcast scalar/8 into f20 so q8_dot_reduce() gives correct value
    scalar *= 0.125f;
    uint32_t bits;
    __builtin_memcpy(&bits, &scalar, sizeof(bits));
    __asm__ volatile(
        "fbcx.ps f20, %[s]
"
        :
        : [s] "r"(bits)
        : "f20");
}
static inline float compute_row_dot_q8_0(const block_q8_0 * q_row, const float * b_col, int64_t K_blocks) {
    unsigned long saved_mask;
    __asm__ volatile("mova.x.m %0" : "=r"(saved_mask));
    __asm__ volatile("mov.m.x m0, x0, 0xFF");
    q8_dot_reset();
    q8_dot_tile(q_row, b_col, K_blocks);
    float result = q8_dot_reduce();
    __asm__ volatile("mova.m.x %0" ::"r"(saved_mask));
    return result;
}

//******************************************************************************
// Hoisted Q8_0 dot API
//
// q8_dot_begin/end save/restore the vector mask once around a long sequence of
// dot products, so the per-row mask shuffles are hoisted out of the inner
// loops. q8_dot_compute does a full-row dot (no mask handling). The _x2
// variant computes two rows together while reusing each loaded B chunk —
// only safe when both row pointers share the same 32-byte alignment phase
// (i.e. the Q8 row stride is a multiple of 32).
//******************************************************************************

typedef struct {
    unsigned long saved_mask;
} q8_dot_state;

static inline void q8_dot_begin(q8_dot_state * state) {
    __asm__ volatile("mova.x.m %0" : "=r"(state->saved_mask));
    __asm__ volatile("mov.m.x m0, x0, 0xFF");
}

static inline void q8_dot_end(const q8_dot_state * state) {
    __asm__ volatile("mova.m.x %0" ::"r"(state->saved_mask));
}

// Equivalent to q8_dot_reset+tile+reduce, without touching the mask register.
// Caller is responsible for q8_dot_begin/end around the surrounding loop.
static inline float q8_dot_compute(const block_q8_0 * q_row, const float * b_col, int64_t K_blocks) {
    q8_dot_reset();
    q8_dot_tile(q_row, b_col, K_blocks);
    return q8_dot_reduce();
}

// Compute two row dots together while reusing the same loaded B chunks.
//
// Safe when every row starts at the same 32-byte offset, i.e. the Q8 row stride
// is a multiple of 32. In that case the gather/alignment pattern is the same
// for both rows at a given `kb`, so one set of B vector loads feeds both row
// accumulators.
static inline void q8_dot_compute_x2_aligned(const block_q8_0 * q_row0,
                                             const block_q8_0 * q_row1,
                                             const float *      b_col,
                                             int64_t            K_blocks,
                                             float *            out0,
                                             float *            out1) {
    float s0 = 0.0f, s1 = 0.0f;
    for (int64_t kb = 0; kb < K_blocks; kb++) {
        s0 += compute_block_dot_product_q8_0(q_row0 + kb, b_col + (kb << 5));
        s1 += compute_block_dot_product_q8_0(q_row1 + kb, b_col + (kb << 5));
    }
    *out0 = s0;
    *out1 = s1;
}

