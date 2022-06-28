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
class MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSsharding_util_opsDTcc {
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
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSsharding_util_opsDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSsharding_util_opsDTcc() {
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

/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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

#include <vector>

#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "tensorflow/compiler/tf2xla/type_util.h"
#include "tensorflow/compiler/tf2xla/xla_op_kernel.h"
#include "tensorflow/compiler/tf2xla/xla_op_registry.h"
#include "tensorflow/compiler/xla/client/lib/constants.h"
#include "tensorflow/compiler/xla/client/xla_builder.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/platform/statusor.h"

namespace tensorflow {
namespace {

constexpr absl::string_view kNumSplitsAttrName = "num_splits";
constexpr absl::string_view kNumConcatsAttrName = "num_concats";

template <bool Split>
Status GetAndValidateAttributes(OpKernelConstruction* ctx,
                                std::vector<int64_t>& num_partitions,
                                int& num_slices, std::vector<int64_t>& paddings,
                                bool& has_paddings) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSsharding_util_opsDTcc mht_0(mht_0_v, 213, "", "./tensorflow/compiler/tf2xla/kernels/sharding_util_ops.cc", "GetAndValidateAttributes");

  absl::string_view num_partitions_attr_name =
      Split ? kNumSplitsAttrName : kNumConcatsAttrName;
  TF_RETURN_IF_ERROR(ctx->GetAttr(num_partitions_attr_name, &num_partitions));

  int num_dims_to_split = 0;
  for (int i = 0, e = num_partitions.size(); i < e; ++i) {
    const auto& split = num_partitions[i];
    if (split <= 0) {
      return errors::InvalidArgument("'", num_partitions_attr_name,
                                     "' at index ", i,
                                     " must be positive, but got ", split, ".");
    }
    if (split > 1) {
      ++num_dims_to_split;
    }
    num_slices *= split;
  }

  int n;
  TF_RETURN_IF_ERROR(ctx->GetAttr("N", &n));
  if (n != num_slices) {
    return errors::InvalidArgument(
        "'N' must match number of slices ", num_slices, " from '",
        num_partitions_attr_name, "', but got ", n, ".");
  }

  TF_RETURN_IF_ERROR(ctx->GetAttr("paddings", &paddings));
  const int expected_rank = num_partitions.size();
  if (!paddings.empty()) {
    if (paddings.size() != expected_rank) {
      return errors::InvalidArgument(
          "'paddings' length must match '", num_partitions_attr_name,
          "' length ", expected_rank, ", but got ", paddings.size(), ".");
    }

    for (int dim = 0; dim < expected_rank; ++dim) {
      if (paddings[dim] < 0) {
        return errors::InvalidArgument(
            "'padding' must be all non-negative, but got ", paddings[dim],
            " at index ", dim, ".");
      }
      if (paddings[dim] > 0) {
        has_paddings = true;
      }
    }
  } else {
    paddings.assign(expected_rank, 0);
  }

  return Status::OK();
}

std::vector<int64_t> GetSliceIndices(absl::Span<const int64> num_partitions,
                                     absl::Span<const int64> slice_shape,
                                     const int index) {
  DCHECK_EQ(num_partitions.size(), slice_shape.size());

  std::vector<int64_t> slice_indices(num_partitions.size());

  if (num_partitions.empty()) {
    return slice_indices;
  }

  auto divisor = [&](const int dim) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSsharding_util_opsDTcc mht_1(mht_1_v, 280, "", "./tensorflow/compiler/tf2xla/kernels/sharding_util_ops.cc", "lambda");

    int divisor = 1;
    for (int i = num_partitions.size() - 1; i > dim; --i) {
      divisor *= num_partitions[i];
    }
    return divisor;
  };

  for (int dim = num_partitions.size() - 1; dim > 0; --dim) {
    slice_indices[dim] =
        ((index / divisor(dim)) % num_partitions[dim]) * slice_shape[dim];
  }
  slice_indices[0] = (index / divisor(0)) * slice_shape[0];

  return slice_indices;
}

constexpr absl::string_view kTensorName = "'input' tensor";
constexpr absl::string_view kResourceName = "'resource' variable tensor";

template <bool Resource>
class XlaSplitNDBaseOp : public XlaOpKernel {
 public:
  explicit XlaSplitNDBaseOp(OpKernelConstruction* ctx) : XlaOpKernel(ctx) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSsharding_util_opsDTcc mht_2(mht_2_v, 306, "", "./tensorflow/compiler/tf2xla/kernels/sharding_util_ops.cc", "XlaSplitNDBaseOp");

    OP_REQUIRES_OK(ctx,
                   GetAndValidateAttributes<true>(ctx, num_splits_, num_slices_,
                                                  paddings_, has_paddings_));
  }

 protected:
  Status CompileInternal(XlaOpKernelContext* ctx, const xla::XlaOp input,
                         const TensorShape& input_shape,
                         const DataType input_dtype) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSsharding_util_opsDTcc mht_3(mht_3_v, 318, "", "./tensorflow/compiler/tf2xla/kernels/sharding_util_ops.cc", "CompileInternal");

    xla::PrimitiveType type;
    TF_RETURN_IF_ERROR(DataTypeToPrimitiveType(input_dtype, &type));

    absl::string_view input_name = Resource ? kResourceName : kTensorName;
    const int rank = input_shape.dims();

    if (rank != num_splits_.size()) {
      return errors::InvalidArgument(
          input_name, " rank must be the same as 'num_splits' length ",
          num_splits_.size(), ", but got rank ", rank, ".");
    }

    for (int dim = 0; dim < rank; ++dim) {
      if ((input_shape.dim_size(dim) + paddings_[dim]) % num_splits_[dim] !=
          0) {
        return errors::InvalidArgument(
            input_name, " shape dimension ", dim, " (",
            input_shape.dim_size(dim), ") with padding ", paddings_[dim],
            " must be evenly divisible by 'num_splits' ", num_splits_[dim],
            ".");
      }
    }

    if (num_slices_ == 1 && has_paddings_) {
      xla::PaddingConfig padding_config;
      for (int dim = 0; dim < rank; ++dim) {
        auto* padding_dim = padding_config.add_dimensions();
        padding_dim->set_edge_padding_low(0);
        padding_dim->set_edge_padding_high(paddings_[dim]);
        padding_dim->set_interior_padding(0);
      }
      ctx->SetOutput(
          /*index=*/0,
          xla::Pad(input,
                   xla::ConstantR0WithType(ctx->builder(), type, /*value=*/0),
                   padding_config));
      return Status::OK();
    } else if (num_slices_ == 1) {
      ctx->SetOutput(/*index=*/0, input);
      return Status::OK();
    }

    // Slice shape with optional padding.
    std::vector<int64_t> slice_shape(rank);
    for (int dim = 0; dim < rank; ++dim) {
      slice_shape[dim] =
          (input_shape.dim_size(dim) + paddings_[dim]) / num_splits_[dim];
    }

    const std::vector<int64_t> slice_strides(rank, 1);

    for (int i = 0; i < num_slices_; ++i) {
      int num_complete_pad_dims = 0;
      int num_partial_pad_dims = 0;
      std::vector<int64_t> slice_start_indices =
          GetSliceIndices(num_splits_, slice_shape, i);
      std::vector<int64_t> slice_limit_indices(slice_shape.size());
      xla::PaddingConfig slice_padding_config;
      for (int dim = 0; dim < rank; ++dim) {
        auto* padding_dim = slice_padding_config.add_dimensions();
        padding_dim->set_edge_padding_low(0);
        padding_dim->set_edge_padding_high(0);
        padding_dim->set_interior_padding(0);
      }

      // Calculate paddings necessary for slice instead of padding input and
      // slicing subsequently to reduce temporary memory allocation.
      for (int dim = 0; dim < rank; ++dim) {
        const int64 dim_size = input_shape.dim_size(dim);
        if (slice_start_indices[dim] >= dim_size) {
          // Complete padding.
          slice_start_indices[dim] = dim_size;
          slice_limit_indices[dim] = dim_size;
          slice_padding_config.mutable_dimensions(dim)->set_edge_padding_high(
              slice_shape[dim]);
          ++num_complete_pad_dims;
        } else if (slice_start_indices[dim] + slice_shape[dim] > dim_size) {
          // Partial padding.
          slice_limit_indices[dim] = dim_size;
          slice_padding_config.mutable_dimensions(dim)->set_edge_padding_high(
              slice_start_indices[dim] + slice_shape[dim] - dim_size);
          ++num_partial_pad_dims;
        } else {
          slice_limit_indices[dim] =
              slice_start_indices[dim] + slice_shape[dim];
        }
      }

      if (num_complete_pad_dims == rank) {
        ctx->SetOutput(i, xla::Broadcast(xla::ConstantR0WithType(
                                             ctx->builder(), type, /*value=*/0),
                                         slice_shape));
      } else if (num_complete_pad_dims > 0 || num_partial_pad_dims > 0) {
        ctx->SetOutput(
            i,
            xla::Pad(xla::Slice(input, slice_start_indices, slice_limit_indices,
                                slice_strides),
                     xla::ConstantR0WithType(ctx->builder(), type, /*value=*/0),
                     slice_padding_config));
      } else {
        ctx->SetOutput(i, xla::Slice(input, slice_start_indices,
                                     slice_limit_indices, slice_strides));
      }
    }
    return Status::OK();
  }

 private:
  std::vector<int64_t> num_splits_;
  int num_slices_ = 1;
  std::vector<int64_t> paddings_;
  bool has_paddings_ = false;
};

class XlaSplitNDOp : public XlaSplitNDBaseOp<false> {
 public:
  explicit XlaSplitNDOp(OpKernelConstruction* ctx)
      : XlaSplitNDBaseOp<false>(ctx) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSsharding_util_opsDTcc mht_4(mht_4_v, 439, "", "./tensorflow/compiler/tf2xla/kernels/sharding_util_ops.cc", "XlaSplitNDOp");
}

  void Compile(XlaOpKernelContext* ctx) override {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSsharding_util_opsDTcc mht_5(mht_5_v, 444, "", "./tensorflow/compiler/tf2xla/kernels/sharding_util_ops.cc", "Compile");

    OP_REQUIRES_OK(ctx,
                   this->CompileInternal(ctx, ctx->Input(0), ctx->InputShape(0),
                                         ctx->input_type(0)));
  }
};

REGISTER_XLA_OP(Name("XlaSplitND"), XlaSplitNDOp);

class ReadVariableXlaSplitNDOp : public XlaSplitNDBaseOp<true> {
 public:
  explicit ReadVariableXlaSplitNDOp(OpKernelConstruction* ctx)
      : XlaSplitNDBaseOp<true>(ctx) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSsharding_util_opsDTcc mht_6(mht_6_v, 459, "", "./tensorflow/compiler/tf2xla/kernels/sharding_util_ops.cc", "ReadVariableXlaSplitNDOp");

    OP_REQUIRES_OK(ctx, ctx->GetAttr("T", &dtype_));
  }

  void Compile(XlaOpKernelContext* ctx) override {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSsharding_util_opsDTcc mht_7(mht_7_v, 466, "", "./tensorflow/compiler/tf2xla/kernels/sharding_util_ops.cc", "Compile");

    DataType variable_input_dtype;
    TensorShape variable_input_shape;
    OP_REQUIRES_OK(
        ctx, ctx->GetVariableTypeAndShape(/*index=*/0, &variable_input_dtype,
                                          &variable_input_shape));
    OP_REQUIRES(
        ctx, variable_input_dtype == dtype_,
        errors::InvalidArgument("'T' must match 'resource' variable dtype ",
                                DataTypeString(variable_input_dtype),
                                ", but got ", dtype_));

    xla::XlaOp handle;
    OP_REQUIRES_OK(ctx, ctx->ReadVariableInput(/*index=*/0, dtype_,
                                               /*shape=*/nullptr, &handle));

    OP_REQUIRES_OK(
        ctx, this->CompileInternal(ctx, handle, variable_input_shape, dtype_));
  }

 private:
  DataType dtype_;
};

REGISTER_XLA_OP(Name("ReadVariableXlaSplitND"), ReadVariableXlaSplitNDOp);

class XlaConcatNDBaseOp : public XlaOpKernel {
 public:
  explicit XlaConcatNDBaseOp(OpKernelConstruction* ctx) : XlaOpKernel(ctx) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSsharding_util_opsDTcc mht_8(mht_8_v, 497, "", "./tensorflow/compiler/tf2xla/kernels/sharding_util_ops.cc", "XlaConcatNDBaseOp");

    OP_REQUIRES_OK(
        ctx, GetAndValidateAttributes<false>(ctx, num_concats_, num_slices_,
                                             paddings_, has_paddings_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("T", &dtype_));
  }

 protected:
  StatusOr<xla::XlaOp> CompileInternal(XlaOpKernelContext* ctx) {
    xla::PrimitiveType type;
    TF_RETURN_IF_ERROR(DataTypeToPrimitiveType(dtype_, &type));

    std::vector<xla::XlaOp> input_handles;
    std::vector<TensorShape> input_shapes;
    std::vector<int64_t> output_shape;
    TF_RETURN_IF_ERROR(GetInputsAndOutputShape(ctx, input_handles, input_shapes,
                                               output_shape));

    const int rank = output_shape.size();

    if (num_slices_ == 1 && has_paddings_) {
      return xla::Slice(input_handles[0],
                        /*start_indices=*/std::vector<int64_t>(rank, 0),
                        /*limit_indices=*/output_shape,
                        /*strides=*/std::vector<int64_t>(rank, 1));
    } else if (num_slices_ == 1) {
      return input_handles[0];
    }

    auto slice_shape = input_shapes[0].dim_sizes();
    xla::XlaOp output = xla::Broadcast(
        xla::ConstantR0WithType(ctx->builder(), type, /*value=*/0),
        output_shape);
    const std::vector<int64_t> input_slice_start_indices(rank, 0);
    const std::vector<int64_t> slice_strides(rank, 1);

    for (int i = 0; i < num_slices_; ++i) {
      std::vector<int64_t> slice_start_indices =
          GetSliceIndices(num_concats_, slice_shape, i);

      int num_complete_pad_dims = 0;
      int num_partial_pad_dims = 0;
      std::vector<int64_t> slice_limit_indices(rank);

      // Calculate paddings necessary to strip from slice.
      for (int dim = 0; dim < rank; ++dim) {
        const int64_t dim_size = output_shape[dim];
        if (slice_start_indices[dim] >= dim_size) {
          // Complete padding.
          slice_start_indices[dim] = dim_size;
          slice_limit_indices[dim] = dim_size;
          ++num_complete_pad_dims;
        } else if (slice_start_indices[dim] + slice_shape[dim] > dim_size) {
          // Partial padding.
          slice_limit_indices[dim] = dim_size;
          ++num_partial_pad_dims;
        } else {
          slice_limit_indices[dim] =
              slice_start_indices[dim] + slice_shape[dim];
        }
      }

      if (num_complete_pad_dims == rank) {
        continue;
      }

      xla::XlaOp input_slice = input_handles[i];
      if (num_complete_pad_dims > 0 || num_partial_pad_dims > 0) {
        std::vector<int64_t> input_slice_limit_indices(rank);
        for (int dim = 0; dim < rank; ++dim) {
          input_slice_limit_indices[dim] =
              slice_limit_indices[dim] - slice_start_indices[dim];
        }
        input_slice = xla::Slice(input_slice, input_slice_start_indices,
                                 input_slice_limit_indices, slice_strides);
      }

      std::vector<xla::XlaOp> update_slice_start_indices;
      update_slice_start_indices.reserve(rank);
      for (int64 start_index : slice_start_indices) {
        update_slice_start_indices.push_back(
            xla::ConstantR0<int32>(ctx->builder(), start_index));
      }
      output = xla::DynamicUpdateSlice(output, input_slice,
                                       update_slice_start_indices);
    }

    return output;
  }

  DataType dtype_;

 private:
  Status GetInputsAndOutputShape(XlaOpKernelContext* ctx,
                                 std::vector<xla::XlaOp>& input_handles,
                                 std::vector<TensorShape>& input_shapes,
                                 std::vector<int64_t>& output_shape) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSsharding_util_opsDTcc mht_9(mht_9_v, 596, "", "./tensorflow/compiler/tf2xla/kernels/sharding_util_ops.cc", "GetInputsAndOutputShape");

    TF_RETURN_IF_ERROR(ctx->InputList("inputs", &input_handles, &input_shapes));

    const TensorShape& slice_shape = input_shapes[0];
    if (slice_shape.dims() != num_concats_.size()) {
      return errors::InvalidArgument(
          "'inputs' rank must be the same as 'num_concats' length ",
          num_concats_.size(), ", but got rank ", slice_shape.dims(), ".");
    }
    for (int i = 1; i < num_slices_; ++i) {
      const TensorShape& slice_shape_i = input_shapes[i];
      if (slice_shape != slice_shape_i) {
        return errors::InvalidArgument(
            "'inputs' must all have the same expected shape ", slice_shape,
            ", but got ", slice_shape_i, " at index ", i, ".");
      }
    }

    const int rank = input_shapes[0].dims();
    for (int dim = 0; dim < rank; ++dim) {
      const int max_dim_size = slice_shape.dim_size(dim) * num_concats_[dim];
      if (paddings_[dim] > max_dim_size) {
        return errors::InvalidArgument(
            "'paddings' must not exceed expected output shape dimension ",
            max_dim_size, " at index ", dim, ", but got ", paddings_[dim], ".");
      }
      output_shape.push_back(max_dim_size - paddings_[dim]);
    }

    return Status::OK();
  }

  std::vector<int64_t> num_concats_;
  int num_slices_ = 1;
  std::vector<int64_t> paddings_;
  bool has_paddings_ = false;
};

class XlaConcatNDOp : public XlaConcatNDBaseOp {
 public:
  explicit XlaConcatNDOp(OpKernelConstruction* ctx) : XlaConcatNDBaseOp(ctx) {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSsharding_util_opsDTcc mht_10(mht_10_v, 639, "", "./tensorflow/compiler/tf2xla/kernels/sharding_util_ops.cc", "XlaConcatNDOp");
}

  void Compile(XlaOpKernelContext* ctx) override {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSsharding_util_opsDTcc mht_11(mht_11_v, 644, "", "./tensorflow/compiler/tf2xla/kernels/sharding_util_ops.cc", "Compile");

    auto output_or = this->CompileInternal(ctx);
    OP_REQUIRES_OK(ctx, output_or.status());
    ctx->SetOutput(/*index=*/0, output_or.ValueOrDie());
  }
};

REGISTER_XLA_OP(Name("XlaConcatND"), XlaConcatNDOp);

class AssignVariableXlaConcatNDOp : public XlaConcatNDBaseOp {
 public:
  explicit AssignVariableXlaConcatNDOp(OpKernelConstruction* ctx)
      : XlaConcatNDBaseOp(ctx) {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSsharding_util_opsDTcc mht_12(mht_12_v, 659, "", "./tensorflow/compiler/tf2xla/kernels/sharding_util_ops.cc", "AssignVariableXlaConcatNDOp");
}

  void Compile(XlaOpKernelContext* ctx) override {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSsharding_util_opsDTcc mht_13(mht_13_v, 664, "", "./tensorflow/compiler/tf2xla/kernels/sharding_util_ops.cc", "Compile");

    auto output_or = this->CompileInternal(ctx);
    OP_REQUIRES_OK(ctx, output_or.status());
    OP_REQUIRES_OK(ctx, ctx->AssignVariable("resource", dtype_,
                                            output_or.ConsumeValueOrDie()));
  }
};

REGISTER_XLA_OP(Name("AssignVariableXlaConcatND"), AssignVariableXlaConcatNDOp);

}  // namespace
}  // namespace tensorflow
