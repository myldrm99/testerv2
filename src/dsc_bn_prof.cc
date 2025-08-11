// #include "dsc_bn_prof.h"
// #include <cstdio>
// #include <cstring>
// #include <inttypes.h>

// static uint64_t acc[BN_MAX_BLOCKS][BN_STAGE_COUNT];
// static int bn_idx = -1;
// static bn_expect_t expect_stage = BN_EXPECT_EX;

// void bn_prof_reset(void) {
//   std::memset(acc, 0, sizeof(acc));
//   bn_idx = -1;
//   expect_stage = BN_EXPECT_EX;
// }

// void bn_prof_begin_block(void) {
//   if (bn_idx + 1 < BN_MAX_BLOCKS) {
//     bn_idx++;
//     std::memset(acc[bn_idx], 0, sizeof(acc[bn_idx]));
//   }
//   expect_stage = BN_EXPECT_EX;
// }

// void bn_prof_add(bn_stage_t st, uint64_t cycles) {
//   if (bn_idx >= 0 && bn_idx < BN_MAX_BLOCKS && st >= 0 && st < BN_STAGE_COUNT) {
//     acc[bn_idx][st] += cycles;
//   }
// }

// void bn_prof_finish_block(void) {
//   expect_stage = BN_EXPECT_EX;
// }

// bn_expect_t bn_prof_expect(void) { return expect_stage; }
// void bn_prof_set_expect(bn_expect_t e) { expect_stage = e; }

// void bn_prof_dump_and_reset(void) {
//   for (int i = 0; i <= bn_idx; ++i) {
//     uint64_t ex = acc[i][BN_EX_SETUP] + acc[i][BN_EX_MAC] + acc[i][BN_EX_STORE];
//     uint64_t dw = acc[i][BN_DW_SETUP] + acc[i][BN_DW_MAC] + acc[i][BN_DW_STORE];
//     uint64_t pr = acc[i][BN_PR_SETUP] + acc[i][BN_PR_MAC] + acc[i][BN_PR_STORE];
//     uint64_t tot = ex + dw + pr;
//     std::printf("\n[Bottleneck %2d] total=%" PRIu64 "\n", i, tot);
//     std::printf("  EX: setup=%" PRIu64 ", mac=%" PRIu64 ", store=%" PRIu64 ", sum=%" PRIu64 "\n",
//                 acc[i][BN_EX_SETUP], acc[i][BN_EX_MAC], acc[i][BN_EX_STORE], ex);
//     std::printf("  DW: setup=%" PRIu64 ", mac=%" PRIu64 ", store=%" PRIu64 ", sum=%" PRIu64 "\n",
//                 acc[i][BN_DW_SETUP], acc[i][BN_DW_MAC], acc[i][BN_DW_STORE], dw);
//     std::printf("  PR: setup=%" PRIu64 ", mac=%" PRIu64 ", store=%" PRIu64 ", sum=%" PRIu64 "\n",
//                 acc[i][BN_PR_SETUP], acc[i][BN_PR_MAC], acc[i][BN_PR_STORE], pr);
//   }
//   bn_prof_reset();
// }

#include "dsc_bn_prof.h"
#include <cstdio>
#include <cstring>
#include <inttypes.h>

static uint64_t acc[BN_MAX_BLOCKS][BN_STAGE_COUNT];
static bn_meta_t meta[BN_MAX_BLOCKS];
static int bn_idx = -1;
static bn_expect_t expect_stage = BN_EXPECT_EX;

void bn_prof_reset(void) {
  std::memset(acc, 0, sizeof(acc));
  std::memset(meta, 0, sizeof(meta));
  bn_idx = -1;
  expect_stage = BN_EXPECT_EX;
}

void bn_prof_begin_block(void) {
  if (bn_idx + 1 < BN_MAX_BLOCKS) {
    bn_idx++;
    std::memset(acc[bn_idx], 0, sizeof(acc[bn_idx]));
    std::memset(&meta[bn_idx], 0, sizeof(meta[bn_idx]));
  }
  expect_stage = BN_EXPECT_EX;
}

void bn_prof_add(bn_stage_t st, uint64_t cycles) {
  if (bn_idx >= 0 && bn_idx < BN_MAX_BLOCKS && st >= 0 && st < BN_STAGE_COUNT) {
    acc[bn_idx][st] += cycles;
  }
}

void bn_prof_finish_block(void) { expect_stage = BN_EXPECT_EX; }

bn_expect_t bn_prof_expect(void) { return expect_stage; }
void bn_prof_set_expect(bn_expect_t e) { expect_stage = e; }

void bn_prof_set_meta(int if_h, int if_w, int if_c, int ex_out_c) {
  if (bn_idx >= 0 && bn_idx < BN_MAX_BLOCKS) {
    meta[bn_idx].if_h = if_h;
    meta[bn_idx].if_w = if_w;
    meta[bn_idx].if_c = if_c;
    meta[bn_idx].ex_out_c = ex_out_c;
    meta[bn_idx].valid = 1;
  }
}

void bn_prof_dump_and_reset(void) {
  for (int i = 0; i <= bn_idx; ++i) {
    uint64_t ex = acc[i][BN_EX_SETUP] + acc[i][BN_EX_MAC] + acc[i][BN_EX_STORE];
    uint64_t dw = acc[i][BN_DW_SETUP] + acc[i][BN_DW_MAC] + acc[i][BN_DW_STORE];
    uint64_t pr = acc[i][BN_PR_SETUP] + acc[i][BN_PR_MAC] + acc[i][BN_PR_STORE];
    uint64_t tot = ex + dw + pr;
    std::printf("\n[Bottleneck %2d] total=%" PRIu64, i, tot);
    if (meta[i].valid) {
      std::printf("  IFMAP=%dx%dx%d  EX_outC=%d",
                  meta[i].if_h, meta[i].if_w, meta[i].if_c, meta[i].ex_out_c);
    }
    std::printf("\n");
    // NOTE: *_SETUP reports POSTPROC (bias+requant+clamp); *_STORE is final write
    std::printf("  EX: post=%" PRIu64 ", mac=%" PRIu64 ", write=%" PRIu64 ", sum=%" PRIu64 "\n",
                acc[i][BN_EX_SETUP], acc[i][BN_EX_MAC], acc[i][BN_EX_STORE], ex);
    std::printf("  DW: post=%" PRIu64 ", mac=%" PRIu64 ", write=%" PRIu64 ", sum=%" PRIu64 "\n",
                acc[i][BN_DW_SETUP], acc[i][BN_DW_MAC], acc[i][BN_DW_STORE], dw);
    std::printf("  PR: post=%" PRIu64 ", mac=%" PRIu64 ", write=%" PRIu64 ", sum=%" PRIu64 "\n",
                acc[i][BN_PR_SETUP], acc[i][BN_PR_MAC], acc[i][BN_PR_STORE], pr);
  }
  bn_prof_reset();
}
