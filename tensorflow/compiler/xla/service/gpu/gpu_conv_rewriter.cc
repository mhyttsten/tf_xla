#include <iostream>
#include <fstream>
#include <thread>
#include <chrono>
#include <string>
#include <cstdlib>
#include <sstream>
#include <string>
#include <vector>
#include <stdlib.h>
#include <unistd.h>
class MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPSgpu_conv_rewriterDTcc {
public:
   std::string _s;
   int _indent = 0;
   std::string _functionName;
   bool _isFile = false;
   std::string _fileName;
   std::string _envMHIndent;
   int _lineNumber;
   bool _filtered = false;
   bool _otherThread = false;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPSgpu_conv_rewriterDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
      _functionName = functionName;
      _lineNumber = lineNumber;

      // Check if tracing is enabled
      const char* env_path = std::getenv("PATH");
      if (env_path != nullptr && std::string(env_path).find("MHTRACER_ENABLE") == std::string::npos) {
         return;
      }
      // Should we trace of filter?
      const char* env_filter = std::getenv("MHTRACER_FILTER");
      if (env_filter != nullptr) {
         std::string sfilter = std::string(env_filter);
         std::string sLineNumber = std::to_string(lineNumber);
         while (true) {
            std::size_t ioE = sfilter.find(";");
            if (sfilter.size() == 0) {
               break;
            }
            std::string cfs = sfilter.substr(0, ioE);
            std::size_t ioFileName = cfs.find("|");
            std::string fFileName  = cfs.substr(0, ioFileName);
            std::size_t ioFunctionName = cfs.find("|", ioFileName+1);
            std::string fFunctionName  = cfs.substr(ioFileName+1, ioFunctionName-ioFileName-1);
            std::string fLineNumber    = cfs.substr(ioFunctionName+1, cfs.size()-ioFunctionName-1);

            if (  (fFileName == "*" || fFileName == fileName)
               && (fFunctionName == "*" || fFunctionName == functionName)
               && (fLineNumber == "*" || fLineNumber == sLineNumber)) {
              _filtered = true;
               return;
            }

            if (ioE == std::string::npos) {
               sfilter = "";
            } else {
               sfilter = sfilter.substr(ioE+1, sfilter.size()-ioE-1);
            }
         }
      }

      // Create log string
      std::string ostr;

      // Assign indent spaces (tied to PID and TID)
      pid_t pid = getpid();
      std::thread::id tid = std::this_thread::get_id();
      std::stringstream pid_dash_tid_ss;
      pid_dash_tid_ss << pid << "-" << tid;
      std::string pid_dash_tid_str = pid_dash_tid_ss.str();
      _envMHIndent = "MHTRACER_INDENT_";
      char* env_indent = std::getenv(_envMHIndent.c_str());
      if (env_indent != nullptr) {
         _indent = std::stoi(std::string(env_indent));
      }
      _s.assign(_indent, ' ');

      // Check that reporting matches pid/tid
      const char* env_pid_dash_tid = std::getenv("MHTRACER_PID_DASH_TID");
      if (env_pid_dash_tid != nullptr) {
         std::string env_pid_dash_tid_str(env_pid_dash_tid);
         if (env_pid_dash_tid_str != pid_dash_tid_str) {
            _otherThread = true;
         }
      }
      else {  // PID-THREAD not set, set it for the first time (starter thread)
         setenv("MHTRACER_PID_DASH_TID", pid_dash_tid_str.c_str(), 1);
      }

      std::string paramStr;
      for (int i=0; i < params.size(); i++) {
         auto e = params[i];
         while (e.find("\n") != std::string::npos) {
            size_t pos = e.find("\n");
            e = e.erase(pos, 1);
            e = e.insert(pos, "<NL>");
         }
         while (e.find("[") != std::string::npos) {
            size_t pos = e.find("[");
            e = e.erase(pos, 1);
            e = e.insert(pos, "<LB>");
         }
         while (e.find("]") != std::string::npos) {
            size_t pos = e.find("]");
            e = e.erase(pos, 1);
            e = e.insert(pos, "<RB>");
         }
         paramStr += e;
         if ((i+1) < params.size()) {
            paramStr += ", ";
         }
      }

      const char* env_dont_print_pid_dash_tid = std::getenv("MHTRACER_DONT_PRINT_PID_DASH_TID");
      if (env_dont_print_pid_dash_tid != nullptr) {
         pid_dash_tid_str = "";
      }
      if (_otherThread) {
         functionName = "MHOT_" + functionName;
      }
      ostr += _s + functionName + 
         + " [1]"
         + " [" + prefix + "]"
         + " [" + paramStr + "]"
         + " [" + pid_dash_tid_str + " "
         +    std::to_string(lineNumber)
         +    " @ " + fileName + "]\n";

      // Log to file
      if (env_path != nullptr && std::string(env_path).find("MHTRACER_USEFILE") != std::string::npos) {
         _isFile = true;
         _fileName = "/tmp/mhtracer_" + pid_dash_tid_str + ".log";
         std::ofstream os;
         os.open(_fileName, std::ofstream::out | std::ofstream::app);
         os << ostr << "";
         os.close();
      }
      // Log to stdout
      else {
         std::cout << ostr << "";
      }

      // Increment indent spaces
      if (_otherThread) {
         return;
      }
      _indent += 3;
      setenv(_envMHIndent.c_str(), std::to_string(_indent).c_str(), 1);
   }
   ~MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPSgpu_conv_rewriterDTcc() {
      // Check if tracing is enabled
      char* env_path = std::getenv("PATH");
      if (env_path != nullptr && std::string(env_path).find("MHTRACER_ENABLE") == std::string::npos) {
         return;
      }

      // Don't update indent if tracing was filtered or from another thread
      if (_filtered || _otherThread) {
         return;
      }

      _indent -= 3;
      setenv(_envMHIndent.c_str(), std::to_string(_indent).c_str(), 1);
   }
};

/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "tensorflow/compiler/xla/service/gpu/gpu_conv_rewriter.h"

#include <cstdlib>
#include <numeric>
#include <string>
#include <utility>
#include <vector>

#include "tensorflow/compiler/xla/literal.h"
#include "tensorflow/compiler/xla/permutation_util.h"
#include "tensorflow/compiler/xla/service/dfs_hlo_visitor_with_default.h"
#include "tensorflow/compiler/xla/service/gpu/backend_configs.pb.h"
#include "tensorflow/compiler/xla/service/gpu/cublas_cudnn.h"
#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/compiler/xla/window_util.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/logging.h"

namespace xla {
namespace gpu {

namespace conv_matchers {

bool CanImplementAsGpuForwardConv(HloInstruction* conv) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPSgpu_conv_rewriterDTcc mht_0(mht_0_v, 211, "", "./tensorflow/compiler/xla/service/gpu/gpu_conv_rewriter.cc", "CanImplementAsGpuForwardConv");

  const ConvolutionDimensionNumbers& dnums =
      conv->convolution_dimension_numbers();
  if (dnums.input_spatial_dimensions_size() > 3) {
    return false;
  }

  // CuDNN does not accept zero-element arguments
  if (ShapeUtil::IsZeroElementArray(conv->operand(0)->shape()) ||
      ShapeUtil::IsZeroElementArray(conv->operand(1)->shape())) {
    return false;
  }

  // CuDNN can perform either cross correlation (no reversal),
  // or convolution (all dimensions reversed).
  if (dnums.input_spatial_dimensions_size() == 2
          ? !window_util::AllOrNoneReversed(conv->window())
          : window_util::HasWindowReversal(conv->window())) {
    return false;
  }
  return true;
}

// Try to match a backward filter pattern that contains "conv".
// Precondition: "conv" is a kConvolution.
std::tuple<bool, Window, ConvolutionDimensionNumbers, HloInstruction*>
MatchBackwardFilter(HloInstruction* conv) {
  VLOG(2) << "Trying to match convolution backward filter.";
  const auto no_match_result =
      std::make_tuple(false, Window(), ConvolutionDimensionNumbers(), nullptr);

  if (conv->feature_group_count() > 1) {
    VLOG(1) << conv->ToString()
            << " is a forward convolution. All grouped backward filters are "
               "mapped to batch grouped convolutions in tf2xla bridge. Hence "
               "backward filter "
               "convolutions cannot have feature groups greater than 1 at this "
               "point. No need to fold to backward filter.";
    return no_match_result;
  }

  // Step 1: match the instruction pattern without considering the paddings and
  // dimension numbers just yet. We may need some generic pattern matcher
  // similar to third_party/llvm/llvm/include/llvm/IR/PatternMatch.h
  //
  // Backward filter convolution is implemented in XLA as the forward
  // convolution of padded activations and dilated gradients. Padding on
  // activations and dilation on gradients are specified in the "window" field
  // of the forward convolution.
  //
  //        activations  gradients
  //              \         /
  //               v       v
  //              Convolution
  //                 conv
  CHECK_EQ(HloOpcode::kConvolution, conv->opcode());

  // Step 2: match paddings and dimension numbers of the forward convolution.
  const ConvolutionDimensionNumbers& conv_dnums =
      conv->convolution_dimension_numbers();
  auto input_batch_dim = conv_dnums.input_batch_dimension();
  auto input_feature_dim = conv_dnums.input_feature_dimension();
  auto input_spatial_dims = conv_dnums.input_spatial_dimensions();
  auto kernel_input_feature_dim = conv_dnums.kernel_input_feature_dimension();
  auto kernel_output_feature_dim = conv_dnums.kernel_output_feature_dimension();
  auto kernel_spatial_dims = conv_dnums.kernel_spatial_dimensions();
  auto output_batch_dim = conv_dnums.output_batch_dimension();
  auto output_feature_dim = conv_dnums.output_feature_dimension();
  auto output_spatial_dims = conv_dnums.output_spatial_dimensions();
  for (const WindowDimension& window_dim : conv->window().dimensions()) {
    if (window_dim.stride() != 1) {
      VLOG(1) << "Forward convolution's window "
              << conv->window().ShortDebugString()
              << " should have stride of 1.";
      return no_match_result;
    }
    if (window_dim.base_dilation() != 1) {
      VLOG(1) << "Forward convolution's window "
              << conv->window().ShortDebugString()
              << " should have no base (LHS) dilation.";
      return no_match_result;
    }
    if (window_dim.padding_low() < 0) {
      VLOG(1) << "Padding low should be non-negative.";
      return no_match_result;
    }
    if (window_dim.window_reversal()) {
      VLOG(1) << "Window reversal field not supported";
      return no_match_result;
    }
    // Padding high will be checked in Step 3.
  }
  // Mathematically, there is no difference between convolution forward vs
  // backward filter. A backward filter:
  //   [N, O, H+h-1, W+w-1] x [N, C, H, W] -> [O, C, h, w]
  // Can be treated as a forward convolution with `N` treated as the new
  // contracting (feature) dimension, `O` treated as the new batch dimension,
  // and `C` treated as the new output feature dimension. The only difference is
  // layouts and performance.
  //
  // Since there is no way to precisely tell whether we want a foward conv or
  // backward filter conv, we have to rely on heuristics. Empirically forward
  // convolutions have very small kernel dimensions, while in the backward pass
  // "kernel dimensions" are large. If kernel dimensions are smaller than the
  // output dimensions, return foward conv; otherwise proceed with backward
  // filter conv.
  if ((kernel_spatial_dims.empty() ||
       conv->operand(1)->shape().dimensions(kernel_spatial_dims[0]) <=
           conv->shape().dimensions(output_spatial_dims[0])) &&
      !window_util::HasWindowDilation(conv->window())) {
    VLOG(1) << conv->ToString()
            << " is a regular forward convolution. No need "
               "to fold it to a backward filter convolution....";
    return no_match_result;
  }

  // Step 3: fuse the matched HLOs into a backward convolution instruction.
  //
  // Compute the window of the backward convolution.
  Window backward_conv_window;
  for (int i = 0; i < input_spatial_dims.size(); ++i) {
    WindowDimension* dim = backward_conv_window.add_dimensions();
    // The window size of the backward convolution equals the output size of the
    // forward convolution.
    int64_t filter_size = conv->shape().dimensions(output_spatial_dims[i]);
    dim->set_size(filter_size);
    // The window stride equals the window dilation of the forward convolution.
    dim->set_stride(conv->window().dimensions(i).window_dilation());
    // The window's low padding is the same as the low padding of the
    // activations.
    dim->set_padding_low(conv->window().dimensions(i).padding_low());
    dim->set_base_dilation(1);
    dim->set_window_dilation(1);

    int64_t input_size =
        conv->operand(0)->shape().dimensions(input_spatial_dims[i]);
    int64_t output_size = conv->window().dimensions(i).size();
    // Compute the range of the amount of valid high padding. We first compute
    // min_padding_high, the amount of padding on the right/bottom to ensure the
    // last patch ends at the border, i.e.,
    //
    //   input_size + dim->padding_low() + min_padding_high
    //     = (output_size - 1) * stride + filter_size
    //
    // Because convolution ignores trailing incomplete windows, any amount of
    // padding high from min_padding_high to min_padding_high+stride-1
    // (max_padding_high) has the same effect.
    int64_t padded_input_size = filter_size + (output_size - 1) * dim->stride();
    int64_t min_padding_high =
        padded_input_size - input_size - dim->padding_low();
    int64_t max_padding_high = min_padding_high + dim->stride() - 1;
    CHECK_GE(dim->padding_low(), 0);
    // In practice, since cuDNN convolution only supports even padding, we make
    // the amount of high padding the same as the amount of low padding as long
    // as it is between min_padding_high and max_padding_high. If it is not in
    // that range, we pick the one that's closest to dim->padding_low() and let
    // GpuConvPaddingLegalization canonicalize the resultant backward
    // convolution later. Picking the closest one minimizes the cost of the kPad
    // instruction to be inserted by GpuConvPaddingLegalization.
    if (dim->padding_low() >= min_padding_high &&
        dim->padding_low() <= max_padding_high) {
      dim->set_padding_high(dim->padding_low());
    } else {
      if (dim->padding_low() < min_padding_high) {
        dim->set_padding_high(min_padding_high);
      } else {
        dim->set_padding_high(max_padding_high);
      }
    }
    if (dim->padding_high() < 0) {
      LOG(WARNING)
          << "Fusing this pattern to backward filter convolution would cause "
             "negative padding ("
          << dim->padding_high()
          << ") on right/bottom of the weight gradients, which is not "
             "supported by GpuConvPaddingLegalization (b/32744257). "
             "Falling back to "
             "unfused convolution for instruction: "
          << conv->ToString();
      return no_match_result;
    }
  }

  // Restore the dimension numbers of the backward convolution from the forward
  // convolution. The two activation dimensions are reversed (batch and
  // feature).
  ConvolutionDimensionNumbers backward_conv_dnums;
  backward_conv_dnums.set_input_batch_dimension(input_feature_dim);
  backward_conv_dnums.set_input_feature_dimension(input_batch_dim);
  for (int i = 0; i < input_spatial_dims.size(); ++i) {
    backward_conv_dnums.add_input_spatial_dimensions(input_spatial_dims[i]);
  }
  backward_conv_dnums.set_output_batch_dimension(kernel_input_feature_dim);
  backward_conv_dnums.set_output_feature_dimension(kernel_output_feature_dim);
  for (int i = 0; i < kernel_spatial_dims.size(); ++i) {
    backward_conv_dnums.add_output_spatial_dimensions(kernel_spatial_dims[i]);
  }
  // The dimension numbering of the output of the forward convolution (before
  // transposition) is the same as that of the activations (according to the
  // semantics of kConvolution). The batch dimension of the activations should
  // be treated as the input feature dimension, and the feature dimension should
  // be treated as the output feature.
  backward_conv_dnums.set_kernel_input_feature_dimension(output_batch_dim);
  backward_conv_dnums.set_kernel_output_feature_dimension(output_feature_dim);
  for (int i = 0; i < output_spatial_dims.size(); ++i) {
    backward_conv_dnums.add_kernel_spatial_dimensions(output_spatial_dims[i]);
  }

  HloInstruction* lhs = conv->mutable_operand(0);
  return std::make_tuple(true, backward_conv_window, backward_conv_dnums, lhs);
}

// Try to match a backward input pattern that contains "conv".
// Precondition: "conv" is a kConvolution.
std::tuple<bool, Window, ConvolutionDimensionNumbers, HloInstruction*>
MatchBackwardInput(HloInstruction* conv) {
  VLOG(2) << "Trying to match convolution backward input.";
  const auto no_match_result =
      std::make_tuple(false, Window(), ConvolutionDimensionNumbers(), nullptr);

  // TODO(timshen) Theoretically cuDNN supports grouped convolutions also
  // for the backward input convolution, but based on the cudnn's current state
  // there is not much performance improvement when using the
  // cudnn backward input API for grouped conv.
  // This needs to be re-evaluated for future cuDNN versions.
  // Note that we already have the necessary code down below, the only thing to
  // enable it is to remove the following early return.
  if (conv->feature_group_count() > 1) {
    return no_match_result;
  }

  // Match instruction pattern.
  CHECK_EQ(HloOpcode::kConvolution, conv->opcode());
  HloInstruction* reverse_filter = conv->mutable_operand(1);
  ConvolutionDimensionNumbers dnums = conv->convolution_dimension_numbers();

  // Match BackwardInput for a depthwise convolution and thunk it to forward
  // convolution Output feature dimension and input feature dimension has been
  // swapped in the bridge. Hence to get the actual input features we need to
  // query the output feature dimension
  auto kernel_out_feature_dim = dnums.kernel_output_feature_dimension();
  auto kernel_out_features =
      reverse_filter->shape().dimensions(kernel_out_feature_dim);

  // For a depthwise convolution, the input features must be equal to the
  // feature_group_count. We can leverage this property to match a depthwise
  // convolution and thunk it to forward conv
  if (conv->feature_group_count() > 1 &&
      kernel_out_features == conv->feature_group_count()) {
    return no_match_result;
  }

  // We pattern-match to a backwards input conv if:
  //
  //  - all spatial dims of the filter are reversed
  //
  // OR
  //
  //  - filter is 1x1 or a constant AND
  //  - conv has base dilation (otherwise this is just a regular forward conv).
  //
  // The final criterion above is just for canonicalization; cudnn seems to run
  // just as fast if we canonicalize 1x1/constant filters without base dilation
  // to forward or backward convs.  We canonicalize to forward conv because (a)
  // it's more natural (constant filters usually show up when doing inference,
  // and having backwards convolutions in inference graphs would be weird), and
  // (b) cudnn has special fusions for forward conv plus bias and activation,
  // and we want to pattern-match to that after running this pass.
  bool is_reversed_filter =
      reverse_filter->opcode() == HloOpcode::kReverse &&
      absl::c_is_permutation(dnums.kernel_spatial_dimensions(),
                             reverse_filter->dimensions());
  bool is_1x1_filter =
      absl::c_all_of(conv->window().dimensions(),
                     [](const WindowDimension& d) { return d.size() == 1; });
  if (!is_reversed_filter &&
      !(window_util::HasBaseDilation(conv->window()) &&
        (reverse_filter->IsConstant() || is_1x1_filter))) {
    VLOG(1) << "Can't match to backwards convolution. Either filter is not "
               "kReverse, or it's not a base-dilated conv with a 1x1 or "
               "constant filter.";
    return no_match_result;
  }

  // Match padding and dilation of the forward convolution.
  for (const WindowDimension& window_dim : conv->window().dimensions()) {
    if (window_dim.stride() != 1) {
      VLOG(1) << "Forward convolution's window "
              << conv->window().ShortDebugString()
              << " should have stride of 1.";
      return no_match_result;
    }
    if (window_dim.window_dilation() != 1) {
      VLOG(1) << "Forward convolution's window "
              << conv->window().ShortDebugString()
              << " should have no window dilation.";
      return no_match_result;
    }
    if (window_dim.window_reversal()) {
      VLOG(1) << "Window reversal field not supported";
      return no_match_result;
    }
  }

  const auto& input_spatial_dims = dnums.input_spatial_dimensions();
  const auto& output_spatial_dims = dnums.output_spatial_dimensions();
  CHECK_EQ(conv->window().dimensions().size(), input_spatial_dims.size());
  CHECK_EQ(output_spatial_dims.size(), input_spatial_dims.size());

  const Window& old_window = conv->window();
  Window new_window = old_window;
  for (size_t i = 0; i < input_spatial_dims.size(); ++i) {
    // Restore backward convolution's padding config from the matched pattern.
    // See the comment in tensorflow/core/kernels/conv_grad_ops.h for how we
    // convert backward input convolution to a variant of forward convolution.
    //
    // The stride of the backward convolution
    // = the base dilation factor of the forward convolution
    auto dim = new_window.mutable_dimensions(i);
    dim->set_stride(old_window.dimensions(i).base_dilation());
    dim->set_base_dilation(1);

    // The low padding = kernel_size - 1 - low padding on the gradients
    // Make sure the low padding is not negative.
    auto kernel_size = old_window.dimensions(i).size();
    auto backward_padding_low =
        kernel_size - 1 - old_window.dimensions(i).padding_low();
    if (backward_padding_low < 0) {
      LOG(WARNING)
          << "The low padding of the backward convolution would be negative ("
          << backward_padding_low
          << "), which isn't supported by GpuConvPaddingLegalization "
             "for now (b/32744257).";
      return no_match_result;
    }
    dim->set_padding_low(backward_padding_low);

    // Compute the range of the amount of padding on the right/bottom of the
    // activations. XLA's convolution requires all patches to be within the
    // padded base. This gives us flexiblity to choose the amount of high
    // padding from a set of values without changing the result of the backward
    // convolution. The minimum amount (min_padding_high) makes the last patch
    // end at the border. The maximum amount (max_padding_high) equals
    // min_padding_high+stride-1 -- max_padding_high+1 would cause the output
    // size to change.
    auto unpadded_input_size = conv->shape().dimensions(output_spatial_dims[i]);
    auto output_size =
        conv->operand(0)->shape().dimensions(input_spatial_dims[i]);
    auto padded_input_size = kernel_size + dim->stride() * (output_size - 1);
    auto total_pad_size = padded_input_size - unpadded_input_size;
    auto min_padding_high = total_pad_size - backward_padding_low;
    auto max_padding_high = min_padding_high + dim->stride() - 1;

    if (backward_padding_low >= min_padding_high &&
        backward_padding_low <= max_padding_high) {
      // In the best case (most likely), if backward_padding_low is in the range
      // of the amounts of valid high padding, we choose backward_padding_low
      // because cuDNN supports even padding only.
      dim->set_padding_high(backward_padding_low);
    } else {
      // Otherwise, we choose the amount that's closest to backward_padding_low,
      // and GpuConvPaddingLegalization will later insert kSlice
      // instructions to enforce even padding.
      //
      // For example, consider the backward convolution pattern
      //
      //   ab     xy
      //   | pad  | reverse
      //  .a.b    yx
      //     \   /
      //      ABC
      //
      // The amount of low padding on activations (in backward convolution) is
      //   backward_padding_low = kernel_size - 1 - forward_padding_low
      //                        = 2 - 1 - 1 = 0
      //
      // The amount of padding high must be between 1 and 2, in order to make
      // Conv(ABC, xy, stride=2) produce exactly 2 elements (ab). 0 is not in
      // the range of [1,2], so we pick the closest valid amount of padding
      // high, which is 1 in this case. Therefore, we fuse the above pattern to
      //
      //   ABC = BackwardInputConv(ab, xy, stride=2, padding_high=1)
      if (backward_padding_low < min_padding_high) {
        dim->set_padding_high(min_padding_high);
      } else {
        dim->set_padding_high(max_padding_high);
      }
    }
    // GpuConvPaddingLegalization doesn't handle backward input
    // convolution with negative padding for now. So fall back to unfused
    // convolution in case of negative padding. For example,
    //   ABCD = Conv(abc, reverse(xy), padding_high=2)
    // could be fused to
    //   ABCD = BackwardInputConv(abc, xy, padding_low=1, padding_high=-1)
    // with positive padding low but negative padding high.
    if (dim->padding_high() < 0) {
      LOG(WARNING) << "Fusing this pattern to backward convolution would cause "
                      "negative padding ("
                   << dim->padding_high()
                   << ") on right/bottom of the activations, which is not "
                      "supported by GpuConvPaddingLegalization (b/32744257). "
                      "Falling back to unfused convolution for instruction: "
                   << conv->ToString();
      return no_match_result;
    }
  }

  // OK, it's a match! Switch the input feature dimension with the output
  // feature dimension. Also switch the output with the input. This is the way
  // cuDNN expects it to be.
  auto conv_dnums = conv->convolution_dimension_numbers();
  dnums.set_kernel_input_feature_dimension(
      conv_dnums.kernel_output_feature_dimension());
  dnums.set_kernel_output_feature_dimension(
      conv_dnums.kernel_input_feature_dimension());
  for (int i = 0; i < input_spatial_dims.size(); ++i) {
    dnums.set_input_spatial_dimensions(i,
                                       conv_dnums.output_spatial_dimensions(i));
    dnums.set_output_spatial_dimensions(i,
                                        conv_dnums.input_spatial_dimensions(i));
  }
  dnums.set_input_feature_dimension(conv_dnums.output_feature_dimension());
  dnums.set_input_batch_dimension(conv_dnums.output_batch_dimension());
  dnums.set_output_feature_dimension(conv_dnums.input_feature_dimension());
  dnums.set_output_batch_dimension(conv_dnums.input_batch_dimension());

  // If we matched against a constant, we need to add a reverse op that can be
  // subsumed by the cuDNN call. algebraic-simplifier will later remove any
  // unnecessary reverses.
  if (reverse_filter->opcode() != HloOpcode::kReverse &&
      reverse_filter->IsConstant()) {
    // Create a double-reverse, which is a nop.
    HloComputation* c = conv->parent();
    reverse_filter = c->AddInstruction(
        HloInstruction::CreateReverse(reverse_filter->shape(), reverse_filter,
                                      dnums.kernel_spatial_dimensions()));
    reverse_filter = c->AddInstruction(
        HloInstruction::CreateReverse(reverse_filter->shape(), reverse_filter,
                                      dnums.kernel_spatial_dimensions()));
    TF_CHECK_OK(conv->ReplaceOperandWith(/*operand_num=*/1, reverse_filter));
  }

  // Calculate the 'rhs' that goes into the backward input convolution.
  HloInstruction* rhs = reverse_filter;
  // One reverse is subsumed by the cuDNN call.
  if (rhs->opcode() == HloOpcode::kReverse) {
    rhs = rhs->mutable_operand(0);
  }
  if (conv->feature_group_count() == 1) {
    return std::make_tuple(true, new_window, dnums, rhs);
  }

  // Handle grouped convolutions. Because we swapped the input feature dimension
  // with the output feature dimension, we need to also reshape the kernel so
  // that the 'feature_group_count' parameter still makes sense. The
  // 'feature_group_count' parameter essentially specifies how often the
  // 'kernel_input_feature_dimension' is repeated. So when we swap these
  // dimensions, we need to divide the new 'kernel_input_feature_dimension' by
  // 'feature_group_count' and multiply the new
  // 'kernel_output_feature_dimension' by 'feature_group_count'.
  int64_t input_feature_dimension = dnums.kernel_input_feature_dimension();
  int64_t output_feature_dimension = dnums.kernel_output_feature_dimension();
  // The following code assumes that input_feature_dimension and
  // output_feature_dimension are adjacent.
  if (std::abs(input_feature_dimension - output_feature_dimension) != 1) {
    return no_match_result;
  }

  int64_t input_features = rhs->shape().dimensions(input_feature_dimension);
  int64_t output_features = rhs->shape().dimensions(output_feature_dimension);

  // Reshape [H, W, ..., in_depth, out_depth / G] -> [H, W, ..., G, in_depth/G,
  // out_depth / G]
  std::vector<int64_t> reshape_dims = SpanToVector(rhs->shape().dimensions());
  auto num_groups = conv->feature_group_count();
  CHECK_EQ(input_features % num_groups, 0)
      << "Input feature count should be an exact multiple of feature group "
         "count";
  reshape_dims[input_feature_dimension] =
      reshape_dims[input_feature_dimension] / num_groups;
  reshape_dims.insert(reshape_dims.begin() + input_feature_dimension,
                      num_groups);

  HloComputation* c = conv->parent();
  rhs = c->AddInstruction(HloInstruction::CreateReshape(
      ShapeUtil::MakeShape(rhs->shape().element_type(), reshape_dims), rhs));

  // Transpose [H, W, ..., G, in_depth/G, out_depth / G] -> [H, W, ...,
  // in_depth/G, G, out_depth / G]
  std::vector<int64_t> transpose_dims(rhs->shape().dimensions_size());
  std::iota(transpose_dims.begin(), transpose_dims.end(), 0);
  transpose_dims.erase(transpose_dims.begin() + input_feature_dimension);
  transpose_dims.insert(transpose_dims.begin() + output_feature_dimension,
                        input_feature_dimension);
  std::vector<int64_t> transpose_reshape_dims =
      SpanToVector(rhs->shape().dimensions());
  transpose_reshape_dims.erase(transpose_reshape_dims.begin() +
                               input_feature_dimension);
  transpose_reshape_dims.insert(
      transpose_reshape_dims.begin() + output_feature_dimension, num_groups);
  rhs = c->AddInstruction(HloInstruction::CreateTranspose(
      ShapeUtil::MakeShape(rhs->shape().element_type(), transpose_reshape_dims),
      rhs, transpose_dims));

  // Reshape [H, W, ..., in_depth/G, G, out_depth / G] -> [H, W, ...,
  // in_depth/G, out_depth]
  Shape new_shape = rhs->shape();
  new_shape.DeleteDimension(output_feature_dimension);
  new_shape.set_dimensions(output_feature_dimension,
                           output_features * num_groups);
  rhs = c->AddInstruction(HloInstruction::CreateReshape(new_shape, rhs));
  return std::make_tuple(true, new_window, dnums, rhs);
}

}  // namespace conv_matchers

namespace {

HloInstruction* CreateGpuConv(absl::string_view call_target, const Shape& shape,
                              HloInstruction* lhs, HloInstruction* rhs,
                              const Window& window,
                              const ConvolutionDimensionNumbers& dnums,
                              int64_t feature_group_count,
                              const OpMetadata& metadata) {
   std::vector<std::string> mht_1_v;
   mht_1_v.push_back("call_target: \"" + std::string(call_target.data(), call_target.size()) + "\"");
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPSgpu_conv_rewriterDTcc mht_1(mht_1_v, 738, "", "./tensorflow/compiler/xla/service/gpu/gpu_conv_rewriter.cc", "CreateGpuConv");

  HloComputation* computation = lhs->parent();

  // This call returns a tuple of (conv_result, scratch_memory), where
  // conv_result is the actual result of the convolution, and scratch_memory is
  // temporary memory used by cudnn.
  //
  // At the moment, we don't know how much scratch memory this conv is going to
  // use, so we put u8[0] in this place.  Later on another pass will choose
  // which conv algorithm to use, and at that point we'll modify the shape of
  // this second tuple element.
  Shape call_shape =
      ShapeUtil::MakeTupleShape({shape, ShapeUtil::MakeShape(U8, {0})});

  HloInstruction* custom_call = computation->AddInstruction(
      HloInstruction::CreateCustomCall(call_shape, {lhs, rhs}, call_target));
  custom_call->set_window(window);
  custom_call->set_convolution_dimension_numbers(dnums);
  custom_call->set_feature_group_count(feature_group_count);
  custom_call->set_metadata(metadata);

  // Give the customcall a user-friendly name.
  absl::optional<std::string> name;
  if (call_target == kCudnnConvForwardCallTarget) {
    name = "cudnn-conv";
  } else if (call_target == kCudnnConvBackwardInputCallTarget) {
    name = "cudnn-conv-bw-input";
  } else if (call_target == kCudnnConvBackwardFilterCallTarget) {
    name = "cudnn-conv-bw-filter";
  } else if (call_target == kCudnnConvBiasActivationForwardCallTarget) {
    name = "cudnn-conv-bias-activation";
  }
  if (name.has_value()) {
    computation->parent()->SetAndUniquifyInstrName(custom_call, *name);
  }

  return custom_call;
}

HloInstruction* ConvertBatchGroupedToFeatureGroupedConvolution(
    HloInstruction* conv) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPSgpu_conv_rewriterDTcc mht_2(mht_2_v, 781, "", "./tensorflow/compiler/xla/service/gpu/gpu_conv_rewriter.cc", "ConvertBatchGroupedToFeatureGroupedConvolution");

  CHECK_EQ(conv->feature_group_count(), 1);
  int64_t num_groups = conv->batch_group_count();
  auto dim_numbers = conv->convolution_dimension_numbers();
  auto lhs = conv->mutable_operand(0);
  auto rhs = conv->mutable_operand(1);

  int64_t input_batch_dimension = dim_numbers.input_batch_dimension();

  Shape output_shape = conv->shape();
  int64_t input_feature_dimension = dim_numbers.input_feature_dimension();
  int64_t input_feature = lhs->shape().dimensions(input_feature_dimension);

  HloComputation* computation = lhs->parent();
  auto add = [&](std::unique_ptr<HloInstruction> inst) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPSgpu_conv_rewriterDTcc mht_3(mht_3_v, 798, "", "./tensorflow/compiler/xla/service/gpu/gpu_conv_rewriter.cc", "lambda");

    return computation->AddInstruction(std::move(inst));
  };
  // Reshape batch_dim N -> [G, N/G]
  std::vector<int64_t> reshape_dims = SpanToVector(lhs->shape().dimensions());
  reshape_dims[input_batch_dimension] =
      reshape_dims[input_batch_dimension] / num_groups;
  reshape_dims.insert(reshape_dims.begin() + input_batch_dimension, num_groups);
  lhs = add(HloInstruction::CreateReshape(
      ShapeUtil::MakeShape(lhs->shape().element_type(), reshape_dims), lhs));

  // Transpose G to the axis before C, For eg: [G, N/G, H, W, C ] -> [N/G, H,
  // W, G, C]
  std::vector<int64_t> transpose_dims(lhs->shape().dimensions_size());
  std::iota(transpose_dims.begin(), transpose_dims.end(), 0);
  transpose_dims.erase(transpose_dims.begin() + input_batch_dimension);
  transpose_dims.insert(transpose_dims.begin() + input_feature_dimension,
                        input_batch_dimension);
  std::vector<int64_t> transpose_reshape_dims =
      ComposePermutations(lhs->shape().dimensions(), transpose_dims);
  lhs = add(HloInstruction::CreateTranspose(
      ShapeUtil::MakeShape(lhs->shape().element_type(), transpose_reshape_dims),
      lhs, transpose_dims));

  // Merge [G,C] -> [C*G]
  Shape new_shape = lhs->shape();
  new_shape.DeleteDimension(input_feature_dimension);
  new_shape.set_dimensions(input_feature_dimension, input_feature * num_groups);
  lhs = add(HloInstruction::CreateReshape(new_shape, lhs));

  std::vector<HloInstruction*> new_operands = {lhs, rhs};
  auto new_conv = conv->CloneWithNewOperands(output_shape, new_operands);
  new_conv->set_feature_group_count(num_groups);
  new_conv->set_batch_group_count(1);
  new_conv->set_convolution_dimension_numbers(dim_numbers);
  return computation->AddInstruction(std::move(new_conv));
}

CudnnConvBackendConfig GetDefaultBackendConfig() {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPSgpu_conv_rewriterDTcc mht_4(mht_4_v, 839, "", "./tensorflow/compiler/xla/service/gpu/gpu_conv_rewriter.cc", "GetDefaultBackendConfig");

  CudnnConvBackendConfig config;
  config.set_conv_result_scale(1);
  return config;
}

// Helper function to create a custom_call instruction to replace the given
// conv instruction
static StatusOr<HloInstruction*> CreateCustomCallHelper(HloInstruction* conv) {
  bool match;
  Window window;
  ConvolutionDimensionNumbers dnums;
  HloInstruction* rhs;
  HloInstruction* lhs;

  std::tie(match, window, dnums, rhs) = conv_matchers::MatchBackwardInput(conv);
  if (match) {
    return CreateGpuConv(kCudnnConvBackwardInputCallTarget, conv->shape(),
                         conv->mutable_operand(0), rhs, window, dnums,
                         conv->feature_group_count(), conv->metadata());
  }

  std::tie(match, window, dnums, lhs) =
      conv_matchers::MatchBackwardFilter(conv);
  if (match) {
    return CreateGpuConv(kCudnnConvBackwardFilterCallTarget, conv->shape(), lhs,
                         conv->mutable_operand(1), window, dnums,
                         conv->batch_group_count(), conv->metadata());
  }

  // If all else fails, try a forward convolution.
  if (conv_matchers::CanImplementAsGpuForwardConv(conv)) {
    if (conv->batch_group_count() > 1) {
      conv = ConvertBatchGroupedToFeatureGroupedConvolution(conv);
    }

    return CreateGpuConv(kCudnnConvForwardCallTarget, conv->shape(),
                         conv->mutable_operand(0), conv->mutable_operand(1),
                         conv->window(), conv->convolution_dimension_numbers(),
                         conv->feature_group_count(), conv->metadata());
  }

  return nullptr;
}

// Tries to rewrite a single convolution into a call to cudnn/miopen.
StatusOr<bool> RunOnInstruction(HloInstruction* conv) {
  CHECK_EQ(conv->opcode(), HloOpcode::kConvolution);

  TF_ASSIGN_OR_RETURN(HloInstruction * custom_call,
                      CreateCustomCallHelper(conv));
  if (custom_call == nullptr) {
    return false;
  }

  TF_RETURN_IF_ERROR(
      custom_call->set_backend_config(GetDefaultBackendConfig()));

  VLOG(1) << "Replacing convolution " << conv->ToString() << " with "
          << custom_call->ToString();

  // The CustomCall returns a tuple (conv_result, scratch_memory).  Extract
  // out the conv result and replace `conv` with it.
  TF_RETURN_IF_ERROR(conv->parent()->ReplaceWithNewInstruction(
      conv,
      HloInstruction::CreateGetTupleElement(conv->shape(), custom_call, 0)));
  return true;
}

// Rewrites the convolutions in the given computation into calls to
// cudnn/miopen.
// Returns true if it made any changes.
StatusOr<bool> RunOnComputation(HloComputation* computation) {
  std::vector<HloInstruction*> convs;
  for (auto* hlo : computation->instructions()) {
    if (hlo->opcode() == HloOpcode::kConvolution) {
      convs.push_back(hlo);
    }
  }

  bool changed = false;
  for (HloInstruction* conv : convs) {
    TF_ASSIGN_OR_RETURN(bool result, RunOnInstruction(conv));
    changed |= result;
  }
  return changed;
}
}  // namespace

StatusOr<bool> GpuConvRewriter::Run(HloModule* module) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPSgpu_conv_rewriterDTcc mht_5(mht_5_v, 931, "", "./tensorflow/compiler/xla/service/gpu/gpu_conv_rewriter.cc", "GpuConvRewriter::Run");

  XLA_VLOG_LINES(2, "GpuConvRewriter::Run(), before:\n" + module->ToString());
  bool changed = false;
  for (HloComputation* computation : module->MakeNonfusionComputations()) {
    TF_ASSIGN_OR_RETURN(bool result, RunOnComputation(computation));
    changed |= result;
  }
  XLA_VLOG_LINES(2, "GpuConvRewriter::Run(), after:\n" + module->ToString());
  return changed;
}

}  // namespace gpu
}  // namespace xla
