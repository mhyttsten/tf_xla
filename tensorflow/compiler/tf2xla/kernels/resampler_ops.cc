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
class MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSresampler_opsDTcc {
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
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSresampler_opsDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSresampler_opsDTcc() {
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

#include "tensorflow/compiler/tf2xla/kernels/resampler_ops.h"

#include <numeric>
#include <vector>

#include "tensorflow/compiler/tf2xla/shape_util.h"
#include "tensorflow/compiler/tf2xla/type_util.h"
#include "tensorflow/compiler/tf2xla/xla_helpers.h"
#include "tensorflow/compiler/tf2xla/xla_op_kernel.h"
#include "tensorflow/compiler/tf2xla/xla_op_registry.h"
#include "tensorflow/compiler/xla/array4d.h"
#include "tensorflow/compiler/xla/client/lib/arithmetic.h"
#include "tensorflow/compiler/xla/client/lib/constants.h"
#include "tensorflow/compiler/xla/client/xla_builder.h"
#include "tensorflow/compiler/xla/literal.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/framework/kernel_def_builder.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/gtl/inlined_vector.h"
#include "tensorflow/core/lib/math/math_util.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {
namespace {

using xla::XlaOp;

// Calculates the bilinear weight tensor, given basis ratio (px, py) of the
// sampling position:
//    W = [(1-px)*(1-py), px*(1-py), (1-px)*py, px*py]
// 'ratio' tensor has dimensions [batch, dim_0, ...dim_n, 2].
//
// The returned tensor has dimensions [batch, dim_0, ... dim_n, 4].
XlaOp BilinearWeights(XlaOpKernelContext* ctx, XlaOp ratio,
                      const TensorShape warp_shape,
                      xla::PrimitiveType xla_type) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSresampler_opsDTcc mht_0(mht_0_v, 225, "", "./tensorflow/compiler/tf2xla/kernels/resampler_ops.cc", "BilinearWeights");

  auto first_term = xla::ConstantR2<float>(
      ctx->builder(), {{1.0, 1.0}, {0.0, 1.0}, {1.0, 0.0}, {0.0, 0.0}});
  first_term = xla::ConvertElementType(first_term, xla_type);

  auto warp_dims = warp_shape.dim_sizes();
  std::vector<int64_t> broadcast_dims(warp_dims.begin(), warp_dims.end() - 1);
  broadcast_dims.push_back(4);
  broadcast_dims.push_back(2);

  const int64_t broadcast_dims_size = broadcast_dims.size();

  std::vector<int64_t> last_two_dims_indices = {(broadcast_dims_size - 2),
                                                (broadcast_dims_size - 1)};

  auto broadcast_first_term =
      xla::BroadcastInDim(first_term, broadcast_dims, last_two_dims_indices);

  // Ratio is of the same dimension as warp, which is [batch, dim_0,... dim_n,
  // 2], we broadcast ratio tensor to 'broadcast_dim' by keeping the
  // [batch, dim_0,...dim_n] dimensions and the [2] dimension as the last
  // dimension.
  std::vector<int64_t> ratio_broadcast_indices(broadcast_dims.size());
  std::iota(ratio_broadcast_indices.begin(), ratio_broadcast_indices.end(), 0);
  ratio_broadcast_indices.erase(ratio_broadcast_indices.end() - 2);

  auto broadcast_ratio =
      xla::BroadcastInDim(ratio, broadcast_dims, ratio_broadcast_indices);

  auto first_term_subtract_weights = broadcast_first_term - broadcast_ratio;

  // Now we have [(1-px, 1-py), (-px, 1-py), (1-px, -py), (px, py)], need to
  // flip the signs of the second and the third term.
  auto sign_change = xla::ConstantR2<float>(
      ctx->builder(), {{1.0, 1.0}, {-1.0, 1.0}, {-1.0, 1.0}, {1.0, 1.0}});
  sign_change = xla::ConvertElementType(sign_change, xla_type);

  auto broadcast_sign_change =
      xla::BroadcastInDim(sign_change, broadcast_dims, last_two_dims_indices);

  auto flipped = first_term_subtract_weights * broadcast_sign_change;

  // Build up the final bilinear weight tensor by multiply reduction, which
  // gives:
  //    [(1-px)*(1-py), px*(1-py), (1-px)*py, px*py]
  // for each 4 neighboring pixels where px and py are the weight of the target
  // pixel we are sampling from.
  return xla::Reduce(
      flipped, xla::One(ctx->builder(), xla_type),
      xla::CreateScalarMultiplyComputation(xla_type, ctx->builder()),
      {broadcast_dims_size - 1});
}

// Concatenates the batch indices to the (x, y) coordinate indices.
// This is done by first creating an Iota tensor that represents the current
// batch it is in, then concatenate with the givin (coordinate) indices.
//
// The resulting tensor has dimension (batch, dim_0, ... dim_n, 3) where
// the last dimension of size 3 in turn is [batch_number, x, y].
// The [batch_number, x, y] dimension is needed because the indices
// [x,y] alone cannot allow the xla::Gather operation to gather from the input
// data, which is of dimension [batch, height(y), width(x), channel] with
// 'batch' being the first dimension.
XlaOp ConcatenateIota(xla::XlaBuilder* b, XlaOp indices,
                      const TensorShape& warp_shape) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSresampler_opsDTcc mht_1(mht_1_v, 292, "", "./tensorflow/compiler/tf2xla/kernels/resampler_ops.cc", "ConcatenateIota");

  // We need to create an iota tensor with the same batch dimension.
  std::vector<int64_t> dimensions;
  dimensions.reserve(warp_shape.dims());
  for (auto dim : warp_shape) {
    dimensions.push_back(dim.size);
  }
  // Except the last dimension, which is of size 1.
  dimensions.back() = 1;

  auto batch_indices =
      xla::Iota(b, xla::ShapeUtil::MakeShape(xla::S32, dimensions),
                /*iota_dimension=*/0);

  return xla::ConcatInDim(b, {batch_indices, indices}, dimensions.size() - 1);
}

// Gathers the 2x2 neighbors of the input starting_indices, and return a
// tensor of dimension [batch, dim_0, ... dim_n, 4, data_channels].
// 'gather_indices' is of dimension [batch, dim_0, ..., dim_n, 3] where the last
// dimension of size 3 is (batch_no, x, y).
XlaOp Gather2by2Neighbors(xla::XlaBuilder* b, XlaOp data, XlaOp gather_indices,
                          int64_t data_channels, int warp_dims) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSresampler_opsDTcc mht_2(mht_2_v, 317, "", "./tensorflow/compiler/tf2xla/kernels/resampler_ops.cc", "Gather2by2Neighbors");

  xla::GatherDimensionNumbers gather_dim_numbers;
  const int64_t neighbor_data_dimensions = warp_dims + 2;
  // Since the Gather output dimensions are [batch, dim_0, ... dim_n, 2, 2,
  // data_channels], the offset dimensions for Gather is the last 3 dimensions.
  gather_dim_numbers.add_offset_dims(neighbor_data_dimensions - 3);
  gather_dim_numbers.add_offset_dims(neighbor_data_dimensions - 2);
  gather_dim_numbers.add_offset_dims(neighbor_data_dimensions - 1);
  // The last dimension of 'gather_indices' is the starting indices for gather.
  gather_dim_numbers.set_index_vector_dim(warp_dims - 1);
  gather_dim_numbers.add_collapsed_slice_dims(0);
  gather_dim_numbers.add_start_index_map(0);
  // Since input is of dimension [batch, height(y), width(x), channel], and warp
  // is of dimension [batch, x, y], the ordering of x, y here needs to be
  // swapped when gathering.
  gather_dim_numbers.add_start_index_map(2);
  gather_dim_numbers.add_start_index_map(1);
  // Data dimensions are [batch, x, y, channel].
  // Output dimensions are [batch, dim_0, ... dim_n, 2, 2, data_channels].
  auto neighbors_data = xla::Gather(data, gather_indices, gather_dim_numbers,
                                    /*slice_sizes=*/{1, 2, 2, data_channels});
  // Collapse the ...,2,2,... dimensions into ...,4,...
  return xla::Collapse(neighbors_data, {warp_dims - 1, warp_dims});
}

// Scatter 'updates' tensor to 'grad_data' based on 'indices'. Returns the
// resulting tensor of dimension: [batch, dim_0, ...dim_n, 2, 2, data_channels].
// This function can also be seen as the inverse of 'Gather2by2Neighbors'.
XlaOp ScatterToGradData(XlaOpKernelContext* ctx, XlaOp grad_data, XlaOp indices,
                        XlaOp updates, int64_t warp_dims,
                        xla::PrimitiveType xla_type) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSresampler_opsDTcc mht_3(mht_3_v, 350, "", "./tensorflow/compiler/tf2xla/kernels/resampler_ops.cc", "ScatterToGradData");

  xla::ScatterDimensionNumbers scatter_dim_numbers;
  const int64_t neighbor_data_dimensions = warp_dims + 2;
  // Since the Scatter output dimensions are [batch, dim_0, ... dim_n, 2, 2,
  // data_channels], the update window dimensions is the last 3 dimensions.
  scatter_dim_numbers.add_update_window_dims(neighbor_data_dimensions - 3);
  scatter_dim_numbers.add_update_window_dims(neighbor_data_dimensions - 2);
  scatter_dim_numbers.add_update_window_dims(neighbor_data_dimensions - 1);
  scatter_dim_numbers.set_index_vector_dim(warp_dims - 1);

  scatter_dim_numbers.add_inserted_window_dims(0);
  scatter_dim_numbers.add_scatter_dims_to_operand_dims(0);
  // Since input is of dimension [batch, height(y), width(x), channel], and warp
  // is of dimension [batch, x, y], the ordering of x, y here needs to be
  // swapped when scattering.
  scatter_dim_numbers.add_scatter_dims_to_operand_dims(2);
  scatter_dim_numbers.add_scatter_dims_to_operand_dims(1);

  return xla::Scatter(grad_data, indices, updates,
                      xla::CreateScalarAddComputation(xla_type, ctx->builder()),
                      scatter_dim_numbers);
}

// Bounds samples to 0 if the warp image indices are out of the (-1, image_size)
// bound.
// The resulting dimension is given by 'result_dims'.
XlaOp BoundSamples(XlaOpKernelContext* ctx, XlaOp warp,
                   xla::PrimitiveType warp_type, TensorShape warp_shape,
                   std::vector<int64_t> result_dims,
                   std::vector<int64_t> broadcasted_dims, int64_t last_warp_dim,
                   xla::Shape data_shape, XlaOp sample) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSresampler_opsDTcc mht_4(mht_4_v, 383, "", "./tensorflow/compiler/tf2xla/kernels/resampler_ops.cc", "BoundSamples");

  auto is_gt_minus_one =
      xla::Gt(warp,
              xla::ConvertElementType(
                  xla::ConstantR1<float>(ctx->builder(), {-1, -1}), warp_type),
              /*broadcast_dimensions=*/{warp_shape.dims() - 1});
  auto is_lt_image_size = xla::Lt(
      warp,
      xla::ConvertElementType(
          xla::ConstantR1<float>(
              ctx->builder(),
              {/*width=*/static_cast<float>(data_shape.dimensions(2)),
               /*height=*/static_cast<float>(data_shape.dimensions(1))}),
          warp_type),
      /*broadcast_dimensions=*/{warp_shape.dims() - 1});

  auto is_in_bound_padded_x_y = xla::And(is_gt_minus_one, is_lt_image_size);
  // Reduce along last dimension. The resulting dimension is:
  // [batch, dim_0, ...dim_n].
  auto is_in_bound = xla::Reduce(
      is_in_bound_padded_x_y, xla::ConstantR0<bool>(ctx->builder(), true),
      xla::CreateScalarAndComputation(xla::PrimitiveType::PRED, ctx->builder()),
      {last_warp_dim});

  // Broadcast 'is_in_bound' to the same dimension as 'result_dims'.
  auto broadcasted_is_in_bound =
      xla::BroadcastInDim(is_in_bound, result_dims, broadcasted_dims);

  // Set out of bound samples to zero.
  auto zeros =
      xla::Broadcast(xla::Zero(ctx->builder(), warp_type), result_dims);
  return xla::Select(broadcasted_is_in_bound, sample, zeros);
}

// Build computation the backprop into input 'data'.
// Where input:
// grad_output is of dimension [batch, dim_0, ...dim_n, channel]
// ratio is of dimension [batch, dim_0, ...dim_n, 2]
// gather_indices is of dimension [batch, dim_0, ...dim_n, 3]
// data_shape is of dimension [batch, x(width), y(height), channel]
//
// Output:
// scatter-add to each 2x2 grad_data neighbor:
//  grad_data[fx, fy, chan] += output_grad * dx * dy
//  grad_data[cx, fy, chan] += output_grad * (1 - dx) * dy
//  grad_data[fx, cy, chan] += output_grad * dx * (1 - dy)
//  grad_data[cx, cy, chan] += output_grad * (1 - dx) * (1 - dy)
// where (dx, dy) is (1 - ratio). If (dx, dy) is out of bound, then the their
// contribution is 0 to 'grad_data'.
XlaOp CalculateGradData(XlaOpKernelContext* ctx, XlaOp grad_output, XlaOp ratio,
                        XlaOp gather_indices, XlaOp warp,
                        xla::PrimitiveType warp_type, TensorShape warp_shape,
                        int64_t last_warp_dim, int64_t data_channels,
                        xla::Shape data_shape) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSresampler_opsDTcc mht_5(mht_5_v, 439, "", "./tensorflow/compiler/tf2xla/kernels/resampler_ops.cc", "CalculateGradData");

  // Weights tensor has dimension [batch, dim_0, ... dim_n, 4].
  auto weights = BilinearWeights(ctx, ratio, warp_shape, warp_type);

  auto warp_dims = warp_shape.dim_sizes();
  std::vector<int64_t> warp_dims_without_last_dims(warp_dims.begin(),
                                                   warp_dims.end() - 1);

  std::vector<int64_t> reshaped_weights_dims = warp_dims_without_last_dims;
  // Reshape the last dimension of size 4 to two dimensions [2, 2].
  reshaped_weights_dims.push_back(2);
  reshaped_weights_dims.push_back(2);
  std::vector<int64_t> reshape_dims(warp_shape.dims());
  std::iota(reshape_dims.begin(), reshape_dims.end(), 0);
  // The dimension is [batch, dim_0,..., dim_n, 2, 2].
  auto reshaped_weights = xla::Reshape(weights, /*dimensions=*/reshape_dims,
                                       /*new_sizes=*/reshaped_weights_dims);

  std::vector<int64_t> weights_with_channels_dims = reshaped_weights_dims;
  weights_with_channels_dims.push_back(data_channels);
  std::vector<int64_t> reshaped_weights_indices(reshaped_weights_dims.size());
  std::iota(reshaped_weights_indices.begin(), reshaped_weights_indices.end(),
            0);

  // Set out of bound weights to 0.
  // The dimension of the reshaped_weight: [batch, dim_0, ...dim_n, 2, 2].
  std::vector<int64_t> reshaped_result_dims(warp_dims.begin(),
                                            warp_dims.end() - 1);
  reshaped_result_dims.push_back(2);
  reshaped_result_dims.push_back(2);
  std::vector<int64_t> broadcasted_dims(warp_dims.size() - 1);
  std::iota(broadcasted_dims.begin(), broadcasted_dims.end(), 0);
  reshaped_weights = BoundSamples(ctx, warp, warp_type, warp_shape,
                                  reshaped_result_dims, broadcasted_dims,
                                  last_warp_dim, data_shape, reshaped_weights);

  // The dimension is [batch, dim_0, ..., dim_n, 2, 2, data_channel].
  auto broadcast_reshaped_weights = xla::BroadcastInDim(
      reshaped_weights, weights_with_channels_dims, reshaped_weights_indices);

  std::vector<int64_t> grad_output_indices(warp_dims_without_last_dims.size());
  std::iota(grad_output_indices.begin(), grad_output_indices.end(), 0);
  grad_output_indices.push_back(weights_with_channels_dims.size() - 1);
  XlaOp broadcast_grad_output = xla::BroadcastInDim(
      grad_output, weights_with_channels_dims, grad_output_indices);

  auto grad_output_multiply_weights =
      broadcast_grad_output * broadcast_reshaped_weights;

  auto grad_data = xla::ConstantLiteral(
      ctx->builder(), xla::Literal::CreateFromShape(data_shape));

  // Pad grad data then slice it back.
  //
  // After left and right column 0-padding, the new dimension of padded data
  // will be [batch, x+2, y+2, channel].
  auto padded_grad_data =
      xla::Pad(grad_data, xla::Zero(ctx->builder(), warp_type),
               xla::MakeEdgePaddingConfig({{0, 0}, {1, 1}, {1, 1}, {0, 0}}));

  auto shifting_value = xla::ConstantR1<int32>(
      ctx->builder(), {/*batch=*/0, /*x(width)=*/1, /*y(height)=*/1});
  auto shifted_gather_indices =
      xla::Add(gather_indices, shifting_value, {last_warp_dim});

  auto updated_grad_data = ScatterToGradData(
      ctx, padded_grad_data, shifted_gather_indices,
      grad_output_multiply_weights, warp_shape.dims(), warp_type);

  const int64_t batch_size = data_shape.dimensions(0);
  const int64_t width = data_shape.dimensions(1);
  const int64_t height = data_shape.dimensions(2);
  // Slice out the result accounting for the padding.
  return xla::Slice(
      updated_grad_data, /*start_indices=*/{0, 1, 1, 0},
      /*limit_indices=*/{batch_size, width + 1, height + 1, data_channels},
      /*strides=*/{1, 1, 1, 1});
}

// Build computation for the backprop into input 'warp'.
// Where input:
//  warp is of dimension [batch, dim_0, ...dim_n, 2]
//  grad_output is of dimension [batch, dim_0, ...dim_n, channel]
//  ratio is of dimension [batch, dim_0, ...dim_n, 2]
//  gather_indices is of dimension [batch, dim_0, ...dim_n, 3] where the last
//  dimension of size 3 is for {batch, x(width), y(height)}.
//  data is of dimension [batch, x, y, channel]
//
// Output (simplified by ignoring the batch dimensions):
// Since the forward path has:
//    output = dot(weights * neighbors)
// The backprop into warp will therefore be:
//    grad_warp = output_grad * d_output / d_warp
//              = output_grad * (d_weights / d_warp * neighbors + d_neighbors /
//              d_warp * weight)
// Where:
//    d_weights / d_warp_x = [-(1 - py), (1 - py), -py, py]
//    d_weights / d_warp_y = [-(1 - px), -px, (1-px), px]
// and
//    d_neighbors / d_warp_x = 0
//
// Therefore:
//    grad_warp_x = py * (img_cxcy - img_fxcy) + (1-py) * (img_cxfy-img_fxfy)
//    grad_warp_y = px * (img_cxcy - img_cxfy) + (1-px) * (img_fxcy-img_fxfy)
//
// where (px, py) is warp, (fx, fy) is the top left corner and (cx, cy) is the
// bottom right corner in a 2x2 neighborhood.
XlaOp CalculateGradWarp(XlaOpKernelContext* ctx, XlaOp grad_output, XlaOp ratio,
                        XlaOp gather_indices, XlaOp data,
                        TensorShape warp_shape, int64_t data_channels,
                        xla::PrimitiveType data_type, xla::Shape data_shape) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSresampler_opsDTcc mht_6(mht_6_v, 552, "", "./tensorflow/compiler/tf2xla/kernels/resampler_ops.cc", "CalculateGradWarp");

  auto warp_dims = warp_shape.dim_sizes();
  std::vector<int64_t> warp_dims_without_last_dims(warp_dims.begin(),
                                                   warp_dims.end() - 1);

  // With dimension [batch, dim_0, ...dim_n, 4]
  std::vector<int64_t> neighbor_broadcast_dims = warp_dims_without_last_dims;
  neighbor_broadcast_dims.push_back(4);

  // With dimension [batch, dim_0, ...dim_n, 4]
  auto neighbor_broadcast_shape =
      xla::ShapeUtil::MakeShape(data_type, neighbor_broadcast_dims);

  const int64_t last_warp_dim = warp_shape.dims() - 1;

  // Pad data with 0, before gathering such that 0 will be returned for samples
  // in the range of (-1, 0) or (image_dimension-1, image_dimension).
  // After left and right column 0-padding, the new dimension of padded data
  // will be [batch, x+2, y+2, channel].
  auto padded_data =
      xla::Pad(data, xla::Zero(ctx->builder(), data_type),
               xla::MakeEdgePaddingConfig({{0, 0}, {1, 1}, {1, 1}, {0, 0}}));

  auto shifting_value = xla::ConstantR1<int32>(
      ctx->builder(), {/*batch=*/0, /*x(width)=*/1, /*y(height)=*/1});
  auto shifted_gather_indices =
      xla::Add(gather_indices, shifting_value, {last_warp_dim});

  // The dimension is [batch, dim_0, ... dim_n, 4, data_channels]
  auto neighbors_data =
      Gather2by2Neighbors(ctx->builder(), padded_data, shifted_gather_indices,
                          data_channels, warp_shape.dims());

  // Since we will be creating the dot product of:
  //  lhs: [batch, dim_0, ...dim_n, 4]
  // and
  //  rhs: [batch, dim_0, ...dim_n, 4, data_channels]
  // we choose the last dimension of lhs and the second last dimension of rhs,
  // with size 4, as the contracting dimension.
  xla::DotDimensionNumbers dot_dims;
  for (int i = 0; i < warp_shape.dims() - 1; ++i) {
    dot_dims.add_lhs_batch_dimensions(i);
    dot_dims.add_rhs_batch_dimensions(i);
  }
  dot_dims.add_lhs_contracting_dimensions(warp_shape.dims() - 1);
  dot_dims.add_rhs_contracting_dimensions(warp_shape.dims() - 1);

  // img_cxcy - img_fxcy
  auto bottom_right_minus_bottom_left = xla::DotGeneral(
      xla::BroadcastInDim(
          xla::ConvertElementType(
              xla::ConstantR1<float>(ctx->builder(), {0, 0, -1, 1}), data_type),
          neighbor_broadcast_dims, {last_warp_dim}),
      neighbors_data, dot_dims, /*precision_config=*/nullptr);

  // img_cxfy - img_fxfy
  auto top_right_minus_top_left = xla::DotGeneral(
      xla::BroadcastInDim(
          xla::ConvertElementType(
              xla::ConstantR1<float>(ctx->builder(), {-1, 1, 0, 0}), data_type),
          neighbor_broadcast_dims, {last_warp_dim}),
      neighbors_data, dot_dims, /*precision_config=*/nullptr);

  // img_cxcy - img_cxfy
  auto bottom_right_minus_top_right = xla::DotGeneral(
      xla::BroadcastInDim(
          xla::ConvertElementType(
              xla::ConstantR1<float>(ctx->builder(), {0, -1, 0, 1}), data_type),
          neighbor_broadcast_dims, {last_warp_dim}),
      neighbors_data, dot_dims, /*precision_config=*/nullptr);

  // img_fxcy - img_fxfy
  auto bottom_left_minus_top_left = xla::DotGeneral(
      xla::BroadcastInDim(
          xla::ConvertElementType(
              xla::ConstantR1<float>(ctx->builder(), {-1, 0, 1, 0}), data_type),
          neighbor_broadcast_dims, {last_warp_dim}),
      neighbors_data, dot_dims, /*precision_config=*/nullptr);

  // Slice out x and y.
  auto weight_x = xla::SliceInDim(ratio, /*start_index=*/0, /*limit_index=*/1,
                                  /*stride=*/1, /*dimno=*/last_warp_dim);
  auto weight_y = xla::SliceInDim(ratio, /*start_index=*/1, /*limit_index=*/2,
                                  /*stride=*/1, /*dimno=*/last_warp_dim);

  // Build 1 - y and 1 - x.
  auto one_minus_y = xla::One(ctx->builder(), data_type) - weight_y;
  auto one_minus_x = xla::One(ctx->builder(), data_type) - weight_x;

  auto x_before_reduce =
      grad_output * weight_y * bottom_right_minus_bottom_left +
      one_minus_y * top_right_minus_top_left;

  std::vector<int64_t> reshaped_sizes = warp_dims_without_last_dims;
  reshaped_sizes.push_back(1);

  std::vector<int64_t> reshaped_dims(warp_dims_without_last_dims.size());
  std::iota(reshaped_dims.begin(), reshaped_dims.end(), 0);

  // Reduce-add along the channel dimension.
  auto x_result =
      xla::Reduce(x_before_reduce, xla::Zero(ctx->builder(), data_type),
                  xla::CreateScalarAddComputation(data_type, ctx->builder()),
                  {last_warp_dim});
  // Reshape before concatenating with y values.
  XlaOp reshaped_x = xla::Reshape(x_result, reshaped_dims, reshaped_sizes);

  auto y_before_reduce = grad_output * weight_x * bottom_right_minus_top_right +
                         one_minus_x * bottom_left_minus_top_left;
  // Reduce-add along the channel dimension.
  auto y_result =
      xla::Reduce(y_before_reduce, xla::Zero(ctx->builder(), data_type),

                  xla::CreateScalarAddComputation(data_type, ctx->builder()),
                  {last_warp_dim});
  XlaOp reshaped_y = xla::Reshape(y_result, reshaped_dims, reshaped_sizes);

  return xla::ConcatInDim(ctx->builder(), {reshaped_x, reshaped_y},
                          last_warp_dim);
}
}  // namespace

ResamplerOp::ResamplerOp(OpKernelConstruction* ctx) : XlaOpKernel(ctx) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSresampler_opsDTcc mht_7(mht_7_v, 677, "", "./tensorflow/compiler/tf2xla/kernels/resampler_ops.cc", "ResamplerOp::ResamplerOp");
}

void ResamplerOp::Compile(XlaOpKernelContext* ctx) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSresampler_opsDTcc mht_8(mht_8_v, 682, "", "./tensorflow/compiler/tf2xla/kernels/resampler_ops.cc", "ResamplerOp::Compile");

  TensorShape data_shape = ctx->InputShape("data");
  OP_REQUIRES(ctx, data_shape.dims() == 4,
              errors::InvalidArgument("data must be 4-dimensional",
                                      data_shape.DebugString()));
  const int64_t data_channels = data_shape.dim_size(3);
  xla::PrimitiveType data_type = ctx->input_xla_type(0);

  TensorShape warp_shape = ctx->InputShape("warp");
  OP_REQUIRES(ctx, warp_shape.dims() >= 2,
              errors::InvalidArgument("warp must be at least 2-dimensional",
                                      warp_shape.DebugString()));
  for (int size : warp_shape.dim_sizes()) {
    OP_REQUIRES(ctx, size > 0,
                errors::InvalidArgument("warp sizes must be positive, got [",
                                        size, "]"));
  }
  const int64_t last_warp_dim = warp_shape.dims() - 1;
  // Last dimension of warp shape must be of size 2.
  OP_REQUIRES(ctx, warp_shape.dim_size(last_warp_dim) == 2,
              errors::InvalidArgument(
                  "the last dimension of warp must be exactly size 2."));
  xla::PrimitiveType warp_type = ctx->input_xla_type(1);

  XlaOp data = ctx->Input("data");
  XlaOp warp = ctx->Input("warp");

  // Find the coordinates of the top left corner for the 2x2 region to be
  // sampled from. The dimensions are [batch, dim_0, ... dim_n, 2] where the
  // last dimension of size 2 in turn is [x, y].
  XlaOp top_left = xla::ConvertElementType(warp, xla::S32);

  auto gather_indices = ConcatenateIota(ctx->builder(), top_left, warp_shape);

  // The dimension is [batch, dim_0, ... dim_n, 4, data_channels]
  auto neighbors_data = Gather2by2Neighbors(
      ctx->builder(), data, gather_indices, data_channels, warp_shape.dims());

  // Dimensions are [batch, dim_0, ... dim_n, 2].
  XlaOp ratio = warp - xla::ConvertElementType(top_left, data_type);

  // Obtain the bilinear blending weights, the dimension is [batch, dim_0,
  // ...dim_n, 4].
  auto weights = BilinearWeights(ctx, ratio, warp_shape, data_type);

  // Since we will be creating the dot product of:
  //  lhs: [batch, dim_0, ...dim_n, 4]
  // and
  //  rhs: [batch, dim_0, ...dim_n, 4, data_channels]
  // we choose the last dimension of lhs and the second last dimension of rhs,
  // with size 4, as the contracting dimension.
  xla::DotDimensionNumbers dot_dims;
  for (int i = 0; i < warp_shape.dims() - 1; ++i) {
    dot_dims.add_lhs_batch_dimensions(i);
    dot_dims.add_rhs_batch_dimensions(i);
  }
  dot_dims.add_lhs_contracting_dimensions(warp_shape.dims() - 1);
  dot_dims.add_rhs_contracting_dimensions(warp_shape.dims() - 1);

  // The dimension is [batch, dim_0, ...dim_n, data_channels].
  auto blended_pixels = xla::DotGeneral(weights, neighbors_data, dot_dims,
                                        /*precision_config=*/nullptr);

  // Handle out of boundary cases by constructing a predicate mask array based
  // on the in-bound condition, and output 0 for the blended pixel value if
  // out-bound. The dimension is the same as top_left: [batch, dim_0,
  // ...dim_n, 2] where the last dimension of size 2 is the [x, y] coordinate.

  auto is_ge_zero = xla::Ge(warp, xla::ZerosLike(warp));

  auto is_lt_image_size = xla::Lt(
      warp,
      xla::ConvertElementType(
          xla::ConstantR1<float>(
              ctx->builder(),
              {/*width=*/static_cast<float>(data_shape.dim_size(2) - 1),
               /*height=*/static_cast<float>(data_shape.dim_size(1) - 1)}),
          warp_type),
      /*broadcast_dimensions=*/{warp_shape.dims() - 1});

  auto is_in_bound_x_y = xla::And(is_ge_zero, is_lt_image_size);
  // Reduce along last dimension. The resulting dimension is:
  // [batch, dim_0, ...dim_n].
  auto is_in_bound = xla::Reduce(
      is_in_bound_x_y, xla::ConstantR0<bool>(ctx->builder(), true),
      xla::CreateScalarAndComputation(xla::PrimitiveType::PRED, ctx->builder()),
      {last_warp_dim});

  // Broadcast 'is_in_bound' to the same dimension as 'blended_pixels', which
  // is the dimension of the result:
  //  [batch, dim_0, ...dim_n, data_channels].
  auto warp_dims = warp_shape.dim_sizes();
  std::vector<int64_t> result_dims(warp_dims.begin(), warp_dims.end() - 1);
  result_dims.push_back(data_channels);

  std::vector<int64_t> broadcasted_dims(warp_dims.size() - 1);
  std::iota(broadcasted_dims.begin(), broadcasted_dims.end(), 0);
  auto broadcasted_is_in_bound =
      xla::BroadcastInDim(is_in_bound, result_dims, broadcasted_dims);

  // Set out of bound samples to zero.
  auto zeros =
      xla::Broadcast(xla::Zero(ctx->builder(), data_type), result_dims);
  auto result = xla::Select(broadcasted_is_in_bound, blended_pixels, zeros);

  ctx->SetOutput(0, result);
}

REGISTER_XLA_OP(Name("Resampler"), ResamplerOp);

ResamplerGradOp::ResamplerGradOp(OpKernelConstruction* ctx) : XlaOpKernel(ctx) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSresampler_opsDTcc mht_9(mht_9_v, 795, "", "./tensorflow/compiler/tf2xla/kernels/resampler_ops.cc", "ResamplerGradOp::ResamplerGradOp");

  DataType output_dtype;
  OP_REQUIRES_OK(ctx, ctx->GetAttr("T", &output_dtype));
}

  // TODO(b/112295522): note that sampling from image boundary is not currently
  // being handled properly.
void ResamplerGradOp::Compile(XlaOpKernelContext* ctx) {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSresampler_opsDTcc mht_10(mht_10_v, 805, "", "./tensorflow/compiler/tf2xla/kernels/resampler_ops.cc", "ResamplerGradOp::Compile");

  TensorShape data_shape_tf = ctx->InputShape("data");
  OP_REQUIRES(ctx, data_shape_tf.dims() == 4,
              errors::InvalidArgument("data must be 4-dimensional",
                                      data_shape_tf.DebugString()));
  const int64_t data_channels = data_shape_tf.dim_size(3);
  xla::PrimitiveType data_type = ctx->input_xla_type(0);

  TensorShape warp_shape = ctx->InputShape("warp");
  OP_REQUIRES(ctx, warp_shape.dims() >= 2,
              errors::InvalidArgument("warp must be at least 2-dimensional",
                                      warp_shape.DebugString()));
  for (int size : warp_shape.dim_sizes()) {
    OP_REQUIRES(ctx, size > 0,
                errors::InvalidArgument("warp sizes must be positive, got [",
                                        size, "]"));
  }
  // Last dimension of warp shape must be of size 2.
  const int64_t last_warp_dim = warp_shape.dims() - 1;
  OP_REQUIRES(ctx, warp_shape.dim_size(last_warp_dim) == 2,
              errors::InvalidArgument(
                  "the last dimension of warp must be exactly size 2."));
  xla::PrimitiveType warp_type = ctx->input_xla_type(1);

  TensorShape output_grad_shape = ctx->InputShape("grad_output");
  OP_REQUIRES(
      ctx, output_grad_shape.dims() >= 2,
      errors::InvalidArgument("output_grad must be at least 2-dimensional",
                              output_grad_shape.DebugString()));

  // Dimensions are [batch, x, y, channel].
  XlaOp data = ctx->Input("data");
  xla::Shape data_shape = TensorShapeToXLAShape(data_type, data_shape_tf);

  // Dimensions are [batch, dim_0, ...dim_n, 2].
  XlaOp warp = ctx->Input("warp");
  // Dimensions are [batch, dim_0, ...dim_n, channel].
  XlaOp grad_output = ctx->Input("grad_output");

  // Find the top left corner coordinate for the region to be sampled from.
  // The dimensions are [batch, dim_0, ... dim_n, 2] where the last dimension
  // of size 2 in turn is [x, y].
  XlaOp top_left = xla::ConvertElementType(xla::Floor(warp), xla::S32);

  // Dimensions are [batch, dim_0, ... dim_n, 2].
  XlaOp ratio = warp - xla::ConvertElementType(top_left, warp_type);

  // Indices for gathering neighboring pixels.
  auto gather_indices = ConcatenateIota(ctx->builder(), top_left, warp_shape);

  auto grad_data = CalculateGradData(ctx, grad_output, ratio, gather_indices,
                                     warp, warp_type, warp_shape, last_warp_dim,
                                     data_channels, data_shape);

  auto grad_warp =
      CalculateGradWarp(ctx, grad_output, ratio, gather_indices, data,
                        warp_shape, data_channels, data_type, data_shape);
  auto warp_dims = warp_shape.dim_sizes();
  std::vector<int64_t> result_dims(warp_dims.begin(), warp_dims.end() - 1);
  result_dims.push_back(2);
  std::vector<int64_t> broadcasted_dims(warp_dims.size() - 1);
  std::iota(broadcasted_dims.begin(), broadcasted_dims.end(), 0);
  auto grad_warp_bounded =
      BoundSamples(ctx, warp, warp_type, warp_shape, result_dims,
                   broadcasted_dims, last_warp_dim, data_shape, grad_warp);

  ctx->SetOutput(0, grad_data);
  ctx->SetOutput(1, grad_warp_bounded);
}

REGISTER_XLA_OP(Name("ResamplerGrad"), ResamplerGradOp);

}  // namespace tensorflow
