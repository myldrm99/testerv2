// /* Copyright 2019 The TensorFlow Authors.

// Licensed under the Apache License, Version 2.0 (the "License");
// ... (license unchanged)
// ==============================================================================*/
// #ifndef TENSORFLOW_LITE_KERNELS_INTERNAL_REFERENCE_INTEGER_OPS_CONV_H_
// #define TENSORFLOW_LITE_KERNELS_INTERNAL_REFERENCE_INTEGER_OPS_CONV_H_

// #include <algorithm>

// #include "tensorflow/lite/kernels/internal/common.h"
// #include "tensorflow/lite/kernels/internal/portable_tensor_utils.h"

// #ifdef BASELINE_BN_PROF
// #include "dsc_bn_prof.h"
// #endif

// namespace tflite {
// namespace reference_integer_ops {

// // Fixed-point per-channel-quantization convolution reference kernel. (int8)
// inline void ConvPerChannel(
//     const ConvParams& params, const int32_t* output_multiplier,
//     const int32_t* output_shift, const RuntimeShape& input_shape,
//     const int8_t* input_data, const RuntimeShape& filter_shape,
//     const int8_t* filter_data, const RuntimeShape& bias_shape,
//     const int32_t* bias_data, const RuntimeShape& output_shape,
//     int8_t* output_data) {
//   // Get parameters.
//   const int32_t input_offset = params.input_offset;
//   const int stride_width = params.stride_width;
//   const int stride_height = params.stride_height;
//   const int dilation_width_factor = params.dilation_width_factor;
//   const int dilation_height_factor = params.dilation_height_factor;
//   const int pad_width = params.padding_values.width;
//   const int pad_height = params.padding_values.height;
//   const int32_t output_offset = params.output_offset;

//   // Set min and max value of the output.
//   const int32_t output_activation_min = params.quantized_activation_min;
//   const int32_t output_activation_max = params.quantized_activation_max;

//   // Consistency checks.
//   TFLITE_DCHECK_LE(output_activation_min, output_activation_max);
//   TFLITE_DCHECK_EQ(input_shape.DimensionsCount(), 4);
//   TFLITE_DCHECK_EQ(filter_shape.DimensionsCount(), 4);
//   TFLITE_DCHECK_EQ(output_shape.DimensionsCount(), 4);
//   const int batches = MatchingDim(input_shape, 0, output_shape, 0);
//   const int input_depth = input_shape.Dims(3);
//   const int output_depth = MatchingDim(filter_shape, 0, output_shape, 3);
//   if (bias_data) {
//     TFLITE_DCHECK_EQ(bias_shape.FlatSize(), output_depth);
//   }

//   // Dimensions.
//   const int input_height = input_shape.Dims(1);
//   const int input_width = input_shape.Dims(2);
//   const int filter_height = filter_shape.Dims(1);
//   const int filter_width = filter_shape.Dims(2);
//   const int filter_input_depth = filter_shape.Dims(3);
//   const int groups = input_depth / filter_input_depth;
//   TFLITE_DCHECK_EQ(input_depth % filter_input_depth, 0);
//   const int filters_per_group = output_depth / groups;
//   const int output_height = output_shape.Dims(1);
//   const int output_width = output_shape.Dims(2);

// #ifdef BASELINE_BN_PROF
//   const bool is_1x1_kernel = (filter_height == 1 && filter_width == 1);
//   const bool is_ex = is_1x1_kernel && (output_depth > input_depth);
//   const bool is_pr = is_1x1_kernel && (output_depth <= input_depth);
//   bool finish_this_block_at_end = false;

//   if (is_ex && bn_prof_expect() == BN_EXPECT_EX) {
//     bn_prof_begin_block();
//     bn_prof_set_expect(BN_EXPECT_DW);
//   } else if (is_pr && bn_prof_expect() == BN_EXPECT_PR) {
//     finish_this_block_at_end = true;
//   }
// #endif

//   for (int batch = 0; batch < batches; ++batch) {
//     for (int out_y = 0; out_y < output_height; ++out_y) {
//       const int in_y_origin = (out_y * stride_height) - pad_height;
//       for (int out_x = 0; out_x < output_width; ++out_x) {
//         const int in_x_origin = (out_x * stride_width) - pad_width;
//         for (int out_channel = 0; out_channel < output_depth; ++out_channel) {
//           auto group = out_channel / filters_per_group;

// #ifdef BASELINE_BN_PROF
//           uint64_t __tmac0 = 0;
//           if (is_1x1_kernel) __tmac0 = rdcycle();
// #endif

//           int32_t acc = 0;
//           for (int filter_y = 0; filter_y < filter_height; ++filter_y) {
//             const int in_y = in_y_origin + dilation_height_factor * filter_y;
//             for (int filter_x = 0; filter_x < filter_width; ++filter_x) {
//               const int in_x = in_x_origin + dilation_width_factor * filter_x;

//               // Zero padding by omitting the areas outside the image.
//               const bool is_point_inside_image =
//                   (in_x >= 0) && (in_x < input_width) &&
//                   (in_y >= 0) && (in_y < input_height);
//               if (!is_point_inside_image) continue;

//               for (int in_channel = 0; in_channel < filter_input_depth;
//                    ++in_channel) {
//                 int32_t input_val = input_data[Offset(
//                     input_shape, batch, in_y, in_x,
//                     in_channel + group * filter_input_depth)];
//                 int32_t filter_val = filter_data[Offset(
//                     filter_shape, out_channel, filter_y, filter_x, in_channel)];
//                 acc += filter_val * (input_val + input_offset);
//               }
//             }
//           }

// #ifdef BASELINE_BN_PROF
//           if (is_1x1_kernel) {
//             bn_prof_add(is_ex ? BN_EX_MAC : BN_PR_MAC, rdcycle() - __tmac0);
//           }
//           uint64_t __tstore0 = 0;
//           if (is_1x1_kernel) __tstore0 = rdcycle();
// #endif

//           if (bias_data) {
//             acc += bias_data[out_channel];
//           }
//           acc = MultiplyByQuantizedMultiplier(
//               acc, output_multiplier[out_channel], output_shift[out_channel]);
//           acc += output_offset;
//           acc = std::max(acc, output_activation_min);
//           acc = std::min(acc, output_activation_max);
//           output_data[Offset(output_shape, batch, out_y, out_x, out_channel)] =
//               static_cast<int8_t>(acc);

// #ifdef BASELINE_BN_PROF
//           if (is_1x1_kernel) {
//             bn_prof_add(is_ex ? BN_EX_STORE : BN_PR_STORE, rdcycle() - __tstore0);
//           }
// #endif
//         }
//       }
//     }
//   }

// #ifdef BASELINE_BN_PROF
//   if (finish_this_block_at_end) {
//     bn_prof_set_expect(BN_EXPECT_EX);
//     bn_prof_finish_block();
//     bn_prof_dump_and_reset();
//   }
// #endif
// }

// // Unpack int4 weights then call the int8 ConvPerChannel.
// inline void ConvPerChannelWithPackedInt4Weights(
//     const ConvParams& params, const int32_t* output_multiplier,
//     const int32_t* output_shift, const RuntimeShape& input_shape,
//     const int8_t* input_data, const RuntimeShape& filter_shape,
//     const int8_t* filter_input, int8_t* unpacked_filter_data,
//     const RuntimeShape& bias_shape, const int32_t* bias_data,
//     const RuntimeShape& output_shape, int8_t* output_data) {
//   TFLITE_DCHECK(unpacked_filter_data != nullptr);
//   tflite::tensor_utils::UnpackDenseInt4IntoInt8(
//       filter_input, filter_shape.FlatSize(), unpacked_filter_data);
//   ConvPerChannel(params, output_multiplier, output_shift, input_shape,
//                  input_data, filter_shape, unpacked_filter_data, bias_shape,
//                  bias_data, output_shape, output_data);
// }

// // Fixed-point per-channel-quantization convolution reference kernel.
// // 16-bit data and 8-bit filter
// template <typename AccumScalar>
// inline void ConvPerChannel(
//     const ConvParams& params, const int32_t* output_multiplier,
//     const int32_t* output_shift, const RuntimeShape& input_shape,
//     const int16_t* input_data, const RuntimeShape& filter_shape,
//     const int8_t* filter_data, const RuntimeShape& bias_shape,
//     const AccumScalar* bias_data, const RuntimeShape& output_shape,
//     int16_t* output_data) {
//   const int stride_width = params.stride_width;
//   const int stride_height = params.stride_height;
//   const int dilation_width_factor = params.dilation_width_factor;
//   const int dilation_height_factor = params.dilation_height_factor;
//   const int pad_width = params.padding_values.width;
//   const int pad_height = params.padding_values.height;

//   const int32_t output_activation_min = params.quantized_activation_min;
//   const int32_t output_activation_max = params.quantized_activation_max;

//   TFLITE_DCHECK_LE(output_activation_min, output_activation_max);
//   TFLITE_DCHECK_EQ(input_shape.DimensionsCount(), 4);
//   TFLITE_DCHECK_EQ(filter_shape.DimensionsCount(), 4);
//   TFLITE_DCHECK_EQ(output_shape.DimensionsCount(), 4);
//   const int batches = MatchingDim(input_shape, 0, output_shape, 0);
//   const int input_depth = input_shape.Dims(3);
//   const int output_depth = MatchingDim(filter_shape, 0, output_shape, 3);
//   if (bias_data) {
//     TFLITE_DCHECK_EQ(bias_shape.FlatSize(), output_depth);
//   }

//   const int input_height = input_shape.Dims(1);
//   const int input_width = input_shape.Dims(2);
//   const int filter_height = filter_shape.Dims(1);
//   const int filter_width = filter_shape.Dims(2);
//   const int filter_input_depth = filter_shape.Dims(3);
//   const int groups = input_depth / filter_input_depth;
//   TFLITE_DCHECK_EQ(input_depth % filter_input_depth, 0);
//   const int filters_per_group = output_depth / groups;
//   const int output_height = output_shape.Dims(1);
//   const int output_width = output_shape.Dims(2);

//   for (int batch = 0; batch < batches; ++batch) {
//     for (int out_y = 0; out_y < output_height; ++out_y) {
//       const int in_y_origin = (out_y * stride_height) - pad_height;
//       for (int out_x = 0; out_x < output_width; ++out_x) {
//         const int in_x_origin = (out_x * stride_width) - pad_width;
//         for (int out_channel = 0; out_channel < output_depth; ++out_channel) {
//           auto group = out_channel / filters_per_group;
//           AccumScalar acc = 0;
//           for (int filter_y = 0; filter_y < filter_height; ++filter_y) {
//             const int in_y = in_y_origin + dilation_height_factor * filter_y;
//             for (int filter_x = 0; filter_x < filter_width; ++filter_x) {
//               const int in_x = in_x_origin + dilation_width_factor * filter_x;
//               const bool is_point_inside_image =
//                   (in_x >= 0) && (in_x < input_width) &&
//                   (in_y >= 0) && (in_y < input_height);
//               if (!is_point_inside_image) continue;

//               for (int in_channel = 0; in_channel < filter_input_depth;
//                    ++in_channel) {
//                 int32_t input_val =
//                     input_data[Offset(input_shape, batch, in_y, in_x,
//                                       in_channel + group * filter_input_depth)];
//                 int32_t filter_val = filter_data[Offset(
//                     filter_shape, out_channel, filter_y, filter_x, in_channel)];
//                 acc += filter_val * input_val;
//               }
//             }
//           }
//           if (bias_data) {
//             acc += bias_data[out_channel];
//           }
//           int32_t scaled_acc = MultiplyByQuantizedMultiplier(
//               acc, output_multiplier[out_channel], output_shift[out_channel]);
//           scaled_acc = std::max(scaled_acc, output_activation_min);
//           scaled_acc = std::min(scaled_acc, output_activation_max);
//           output_data[Offset(output_shape, batch, out_y, out_x, out_channel)] =
//               static_cast<int16_t>(scaled_acc);
//         }
//       }
//     }
//   }
// }

// }  // namespace reference_integer_ops
// }  // namespace tflite

// #endif  // TENSORFLOW_LITE_KERNELS_INTERNAL_REFERENCE_INTEGER_OPS_CONV_H_

/* Copyright 2019 The TensorFlow Authors.

Licensed under the Apache License, Version 2.0 (the "License");
... (license unchanged)
==============================================================================*/
#ifndef TENSORFLOW_LITE_KERNELS_INTERNAL_REFERENCE_INTEGER_OPS_CONV_H_
#define TENSORFLOW_LITE_KERNELS_INTERNAL_REFERENCE_INTEGER_OPS_CONV_H_

#include <algorithm>

#include "tensorflow/lite/kernels/internal/common.h"
#include "tensorflow/lite/kernels/internal/portable_tensor_utils.h"

#ifdef BASELINE_BN_PROF
#include "dsc_bn_prof.h"
#endif

namespace tflite {
namespace reference_integer_ops {

// Fixed-point per-channel-quantization convolution reference kernel. (int8)
inline void ConvPerChannel(
    const ConvParams& params, const int32_t* output_multiplier,
    const int32_t* output_shift, const RuntimeShape& input_shape,
    const int8_t* input_data, const RuntimeShape& filter_shape,
    const int8_t* filter_data, const RuntimeShape& bias_shape,
    const int32_t* bias_data, const RuntimeShape& output_shape,
    int8_t* output_data) {
  // Get parameters.
  const int32_t input_offset = params.input_offset;
  const int stride_width = params.stride_width;
  const int stride_height = params.stride_height;
  const int dilation_width_factor = params.dilation_width_factor;
  const int dilation_height_factor = params.dilation_height_factor;
  const int pad_width = params.padding_values.width;
  const int pad_height = params.padding_values.height;
  const int32_t output_offset = params.output_offset;

  // Set min and max value of the output.
  const int32_t output_activation_min = params.quantized_activation_min;
  const int32_t output_activation_max = params.quantized_activation_max;

  // Consistency checks.
  TFLITE_DCHECK_LE(output_activation_min, output_activation_max);
  TFLITE_DCHECK_EQ(input_shape.DimensionsCount(), 4);
  TFLITE_DCHECK_EQ(filter_shape.DimensionsCount(), 4);
  TFLITE_DCHECK_EQ(output_shape.DimensionsCount(), 4);
  const int batches = MatchingDim(input_shape, 0, output_shape, 0);
  const int input_depth = input_shape.Dims(3);
  const int output_depth = MatchingDim(filter_shape, 0, output_shape, 3);
  if (bias_data) {
    TFLITE_DCHECK_EQ(bias_shape.FlatSize(), output_depth);
  }

  // Dimensions.
  const int input_height = input_shape.Dims(1);
  const int input_width = input_shape.Dims(2);
  const int filter_height = filter_shape.Dims(1);
  const int filter_width = filter_shape.Dims(2);
  const int filter_input_depth = filter_shape.Dims(3);
  const int groups = input_depth / filter_input_depth;
  TFLITE_DCHECK_EQ(input_depth % filter_input_depth, 0);
  const int filters_per_group = output_depth / groups;
  const int output_height = output_shape.Dims(1);
  const int output_width = output_shape.Dims(2);

#ifdef BASELINE_BN_PROF
  const bool is_1x1_kernel = (filter_height == 1 && filter_width == 1);
  const bool is_ex = is_1x1_kernel && (output_depth > input_depth);
  const bool is_pr = is_1x1_kernel && (output_depth <= input_depth);
  bool finish_this_block_at_end = false;

  if (is_ex && bn_prof_expect() == BN_EXPECT_EX) {
    bn_prof_begin_block();
    // Attach metadata for this bottleneck:
    bn_prof_set_meta(input_height, input_width, input_depth, output_depth);
    bn_prof_set_expect(BN_EXPECT_DW);
  } else if (is_pr && bn_prof_expect() == BN_EXPECT_PR) {
    // finishing a bottleneck after PR
    finish_this_block_at_end = true;
  }
#endif

  for (int batch = 0; batch < batches; ++batch) {
    for (int out_y = 0; out_y < output_height; ++out_y) {
      const int in_y_origin = (out_y * stride_height) - pad_height;
      for (int out_x = 0; out_x < output_width; ++out_x) {
        const int in_x_origin = (out_x * stride_width) - pad_width;
        for (int out_channel = 0; out_channel < output_depth; ++out_channel) {
          auto group = out_channel / filters_per_group;

#ifdef BASELINE_BN_PROF
          uint64_t __tmac0 = 0;
          if (is_1x1_kernel) __tmac0 = rdcycle();
#endif

          int32_t acc = 0;
          // ----- MAC (unchanged) -----
          for (int filter_y = 0; filter_y < filter_height; ++filter_y) {
            const int in_y = in_y_origin + dilation_height_factor * filter_y;
            for (int filter_x = 0; filter_x < filter_width; ++filter_x) {
              const int in_x = in_x_origin + dilation_width_factor * filter_x;

              // Zero padding by omitting the areas outside the image.
              const bool is_point_inside_image =
                  (in_x >= 0) && (in_x < input_width) &&
                  (in_y >= 0) && (in_y < input_height);
              if (!is_point_inside_image) continue;

              for (int in_channel = 0; in_channel < filter_input_depth;
                   ++in_channel) {
                int32_t input_val = input_data[Offset(
                    input_shape, batch, in_y, in_x,
                    in_channel + group * filter_input_depth)];
                int32_t filter_val = filter_data[Offset(
                    filter_shape, out_channel, filter_y, filter_x, in_channel)];
                acc += filter_val * (input_val + input_offset);
              }
            }
          }

#ifdef BASELINE_BN_PROF
          if (is_1x1_kernel) {
            bn_prof_add(is_ex ? BN_EX_MAC : BN_PR_MAC, rdcycle() - __tmac0);
          }
          // POSTPROC (bias + requant + clamp)
          uint64_t __tpost0 = 0;
          if (is_1x1_kernel) __tpost0 = rdcycle();
#endif

          // ----- POST (bias + requant + clamp) -----
          if (bias_data) {
            acc += bias_data[out_channel];
          }
          acc = MultiplyByQuantizedMultiplier(
              acc, output_multiplier[out_channel], output_shift[out_channel]);
          acc += output_offset;
          acc = std::max(acc, output_activation_min);
          acc = std::min(acc, output_activation_max);

#ifdef BASELINE_BN_PROF
          if (is_1x1_kernel) {
            bn_prof_add(is_ex ? BN_EX_SETUP : BN_PR_SETUP, rdcycle() - __tpost0);
          }
          // WRITE
          uint64_t __twrite0 = 0;
          if (is_1x1_kernel) __twrite0 = rdcycle();
#endif

          // ----- WRITE out -----
          output_data[Offset(output_shape, batch, out_y, out_x, out_channel)] =
              static_cast<int8_t>(acc);

#ifdef BASELINE_BN_PROF
          if (is_1x1_kernel) {
            bn_prof_add(is_ex ? BN_EX_STORE : BN_PR_STORE, rdcycle() - __twrite0);
          }
#endif
        }
      }
    }
  }

#ifdef BASELINE_BN_PROF
  if (finish_this_block_at_end) {
    bn_prof_set_expect(BN_EXPECT_EX);
    bn_prof_finish_block();
    bn_prof_dump_and_reset();
  }
#endif
}

// Unpack int4 weights then call the int8 ConvPerChannel.
inline void ConvPerChannelWithPackedInt4Weights(
    const ConvParams& params, const int32_t* output_multiplier,
    const int32_t* output_shift, const RuntimeShape& input_shape,
    const int8_t* input_data, const RuntimeShape& filter_shape,
    const int8_t* filter_input, int8_t* unpacked_filter_data,
    const RuntimeShape& bias_shape, const int32_t* bias_data,
    const RuntimeShape& output_shape, int8_t* output_data) {
  TFLITE_DCHECK(unpacked_filter_data != nullptr);
  tflite::tensor_utils::UnpackDenseInt4IntoInt8(
      filter_input, filter_shape.FlatSize(), unpacked_filter_data);
  ConvPerChannel(params, output_multiplier, output_shift, input_shape,
                 input_data, filter_shape, unpacked_filter_data, bias_shape,
                 bias_data, output_shape, output_data);
}

// Fixed-point per-channel-quantization convolution reference kernel.
// 16-bit data and 8-bit filter
template <typename AccumScalar>
inline void ConvPerChannel(
    const ConvParams& params, const int32_t* output_multiplier,
    const int32_t* output_shift, const RuntimeShape& input_shape,
    const int16_t* input_data, const RuntimeShape& filter_shape,
    const int8_t* filter_data, const RuntimeShape& bias_shape,
    const AccumScalar* bias_data, const RuntimeShape& output_shape,
    int16_t* output_data) {
  const int stride_width = params.stride_width;
  const int stride_height = params.stride_height;
  const int dilation_width_factor = params.dilation_width_factor;
  const int dilation_height_factor = params.dilation_height_factor;
  const int pad_width = params.padding_values.width;
  const int pad_height = params.padding_values.height;

  const int32_t output_activation_min = params.quantized_activation_min;
  const int32_t output_activation_max = params.quantized_activation_max;

  TFLITE_DCHECK_LE(output_activation_min, output_activation_max);
  TFLITE_DCHECK_EQ(input_shape.DimensionsCount(), 4);
  TFLITE_DCHECK_EQ(filter_shape.DimensionsCount(), 4);
  TFLITE_DCHECK_EQ(output_shape.DimensionsCount(), 4);
  const int batches = MatchingDim(input_shape, 0, output_shape, 0);
  const int input_depth = input_shape.Dims(3);
  const int output_depth = MatchingDim(filter_shape, 0, output_shape, 3);
  if (bias_data) {
    TFLITE_DCHECK_EQ(bias_shape.FlatSize(), output_depth);
  }

  const int input_height = input_shape.Dims(1);
  const int input_width = input_shape.Dims(2);
  const int filter_height = filter_shape.Dims(1);
  const int filter_width = filter_shape.Dims(2);
  const int filter_input_depth = filter_shape.Dims(3);
  const int groups = input_depth / filter_input_depth;
  TFLITE_DCHECK_EQ(input_depth % filter_input_depth, 0);
  const int filters_per_group = output_depth / groups;
  const int output_height = output_shape.Dims(1);
  const int output_width = output_shape.Dims(2);

  for (int batch = 0; batch < batches; ++batch) {
    for (int out_y = 0; out_y < output_height; ++out_y) {
      const int in_y_origin = (out_y * stride_height) - pad_height;
      for (int out_x = 0; out_x < output_width; ++out_x) {
        const int in_x_origin = (out_x * stride_width) - pad_width;
        for (int out_channel = 0; out_channel < output_depth; ++out_channel) {
          auto group = out_channel / filters_per_group;
          AccumScalar acc = 0;
          for (int filter_y = 0; filter_y < filter_height; ++filter_y) {
            const int in_y = in_y_origin + dilation_height_factor * filter_y;
            for (int filter_x = 0; filter_x < filter_width; ++filter_x) {
              const int in_x = in_x_origin + dilation_width_factor * filter_x;
              const bool is_point_inside_image =
                  (in_x >= 0) && (in_x < input_width) &&
                  (in_y >= 0) && (in_y < input_height);
              if (!is_point_inside_image) continue;

              for (int in_channel = 0; in_channel < filter_input_depth;
                   ++in_channel) {
                int32_t input_val =
                    input_data[Offset(input_shape, batch, in_y, in_x,
                                      in_channel + group * filter_input_depth)];
                int32_t filter_val = filter_data[Offset(
                    filter_shape, out_channel, filter_y, filter_x, in_channel)];
                acc += filter_val * input_val;
              }
            }
          }
          if (bias_data) {
            acc += bias_data[out_channel];
          }
          int32_t scaled_acc = MultiplyByQuantizedMultiplier(
              acc, output_multiplier[out_channel], output_shift[out_channel]);
          scaled_acc = std::max(scaled_acc, output_activation_min);
          scaled_acc = std::min(scaled_acc, output_activation_max);
          output_data[Offset(output_shape, batch, out_y, out_x, out_channel)] =
              static_cast<int16_t>(scaled_acc);
        }
      }
    }
  }
}

}  // namespace reference_integer_ops
}  // namespace tflite

#endif  // TENSORFLOW_LITE_KERNELS_INTERNAL_REFERENCE_INTEGER_OPS_CONV_H_
