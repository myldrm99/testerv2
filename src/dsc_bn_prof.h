// #pragma once
// #include <stdint.h>

// #ifdef __cplusplus
// extern "C" {
// #endif

// // Stages inside one bottleneck block
// typedef enum {
//   BN_EX_SETUP = 0, BN_EX_MAC, BN_EX_STORE,
//   BN_DW_SETUP, BN_DW_MAC, BN_DW_STORE,
//   BN_PR_SETUP, BN_PR_MAC, BN_PR_STORE,
//   BN_STAGE_COUNT
// } bn_stage_t;

// // Expected next layer in the 3-layer bottleneck
// typedef enum { BN_EXPECT_EX=0, BN_EXPECT_DW, BN_EXPECT_PR } bn_expect_t;

// #define BN_MAX_BLOCKS 64

// void bn_prof_reset(void);
// void bn_prof_begin_block(void);
// void bn_prof_add(bn_stage_t st, uint64_t cycles);
// void bn_prof_finish_block(void);
// void bn_prof_dump_and_reset(void);

// bn_expect_t bn_prof_expect(void);
// void bn_prof_set_expect(bn_expect_t e);

// // 64-bit cycle counter (safe on rv32)
// static inline uint64_t rdcycle() {
//   uint32_t lo, hi, hi2;
//   do {
//     asm volatile("rdcycleh %0" : "=r"(hi));
//     asm volatile("rdcycle %0"  : "=r"(lo));
//     asm volatile("rdcycleh %0" : "=r"(hi2));
//   } while (hi != hi2);
//   return ((uint64_t)hi << 32) | lo;
// }

// #ifdef __cplusplus
// }
// #endif

#pragma once
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Stages inside one bottleneck block
// NOTE: we now use *_SETUP fields to mean POSTPROC (bias+requant+clamp).
typedef enum {
  BN_EX_SETUP = 0, BN_EX_MAC, BN_EX_STORE,
  BN_DW_SETUP, BN_DW_MAC, BN_DW_STORE,
  BN_PR_SETUP, BN_PR_MAC, BN_PR_STORE,
  BN_STAGE_COUNT
} bn_stage_t;

// Expected next layer in the 3-layer bottleneck
typedef enum { BN_EXPECT_EX=0, BN_EXPECT_DW, BN_EXPECT_PR } bn_expect_t;

#define BN_MAX_BLOCKS 64

// Optional metadata per bottleneck (for the dump)
typedef struct {
  int if_h, if_w, if_c;   // IFMAP dims entering EX
  int ex_out_c;           // EX 1x1 output channels
  int valid;              // whether meta was set for this block
} bn_meta_t;

void bn_prof_reset(void);
void bn_prof_begin_block(void);
void bn_prof_add(bn_stage_t st, uint64_t cycles);
void bn_prof_finish_block(void);
void bn_prof_dump_and_reset(void);

bn_expect_t bn_prof_expect(void);
void bn_prof_set_expect(bn_expect_t e);

// Set metadata for current block (safe to call multiple times in EX)
void bn_prof_set_meta(int if_h, int if_w, int if_c, int ex_out_c);

// 64-bit cycle counter (safe on rv32)
static inline uint64_t rdcycle() {
  uint32_t lo, hi, hi2;
  do {
    asm volatile("rdcycleh %0" : "=r"(hi));
    asm volatile("rdcycle %0"  : "=r"(lo));
    asm volatile("rdcycleh %0" : "=r"(hi2));
  } while (hi != hi2);
  return ((uint64_t)hi << 32) | lo;
}

#ifdef __cplusplus
}
#endif
