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
class MHTracer_DTPStensorflowPScorePStpuPSkernelsPSsharding_util_opsDTcc {
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
   MHTracer_DTPStensorflowPScorePStpuPSkernelsPSsharding_util_opsDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePStpuPSkernelsPSsharding_util_opsDTcc() {
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

#define EIGEN_USE_THREADS

#include <functional>
#include <vector>

#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/kernel_def.pb.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/op_requires.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/framework/resource_var.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/refcount.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/platform/statusor.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {
namespace {

constexpr absl::string_view kNumSplitsAttrName = "num_splits";
constexpr absl::string_view kNumConcatsAttrName = "num_concats";

template <bool Split>
Status GetAndValidateAttributes(OpKernelConstruction* ctx,
                                std::vector<int32>& num_partitions,
                                int& num_slices, std::vector<int32>& paddings,
                                bool& has_paddings) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePStpuPSkernelsPSsharding_util_opsDTcc mht_0(mht_0_v, 221, "", "./tensorflow/core/tpu/kernels/sharding_util_ops.cc", "GetAndValidateAttributes");

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

absl::string_view kHandle = "handle";
absl::string_view kTensor = "tensor";

template <bool Handle>
Status CreateResourceInvalidDTypeError(const ResourceHandle& handle,
                                       DataType actual_dtype,
                                       DataType expected_dtype) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePStpuPSkernelsPSsharding_util_opsDTcc mht_1(mht_1_v, 283, "", "./tensorflow/core/tpu/kernels/sharding_util_ops.cc", "CreateResourceInvalidDTypeError");

  absl::string_view resource_component = Handle ? kHandle : kTensor;
  return errors::InvalidArgument(
      "'T' must match 'resource' variable ", resource_component, " ('",
      handle.name(), "') container ('", handle.container(), "') dtype ",
      DataTypeString(actual_dtype), ", but got ",
      DataTypeString(expected_dtype), ".");
}

// Converts flatten index to start indices (subscript scaled with slice shape)
// for determining where to start a slice in the input tensor.
template <int Rank>
Eigen::DSizes<Eigen::DenseIndex, Rank> GetSliceIndices(
    absl::Span<const int32> num_partitions,
    const Eigen::DSizes<Eigen::DenseIndex, Rank>& slice_shape,
    const int index) {
  return Eigen::DSizes<Eigen::DenseIndex, Rank>();
}

template <>
Eigen::DSizes<Eigen::DenseIndex, 1> GetSliceIndices(
    absl::Span<const int32> num_partitions,
    const Eigen::DSizes<Eigen::DenseIndex, 1>& slice_shape, const int index) {
  Eigen::DSizes<Eigen::DenseIndex, 1> subscript;
  subscript[0] = index * slice_shape[0];
  return subscript;
}

template <>
Eigen::DSizes<Eigen::DenseIndex, 2> GetSliceIndices(
    absl::Span<const int32> num_partitions,
    const Eigen::DSizes<Eigen::DenseIndex, 2>& slice_shape, const int index) {
  Eigen::DSizes<Eigen::DenseIndex, 2> subscript;
  subscript[1] = (index % num_partitions[1]) * slice_shape[1];
  subscript[0] = (index / num_partitions[1]) * slice_shape[0];
  return subscript;
}

template <>
Eigen::DSizes<Eigen::DenseIndex, 3> GetSliceIndices(
    absl::Span<const int32> num_partitions,
    const Eigen::DSizes<Eigen::DenseIndex, 3>& slice_shape, const int index) {
  Eigen::DSizes<Eigen::DenseIndex, 3> subscript;
  subscript[2] = (index % num_partitions[2]) * slice_shape[2];
  subscript[1] =
      ((index / num_partitions[2]) % num_partitions[1]) * slice_shape[1];
  subscript[0] =
      (index / (num_partitions[2] * num_partitions[1])) * slice_shape[0];
  return subscript;
}

template <>
Eigen::DSizes<Eigen::DenseIndex, 4> GetSliceIndices(
    absl::Span<const int32> num_partitions,
    const Eigen::DSizes<Eigen::DenseIndex, 4>& slice_shape, const int index) {
  Eigen::DSizes<Eigen::DenseIndex, 4> subscript;
  subscript[3] = (index % num_partitions[3]) * slice_shape[3];
  subscript[2] =
      ((index / num_partitions[3]) % num_partitions[2]) * slice_shape[2];
  subscript[1] =
      ((index / (num_partitions[3] * num_partitions[2])) % num_partitions[1]) *
      slice_shape[1];
  subscript[0] =
      (index / (num_partitions[3] * num_partitions[2] * num_partitions[1])) *
      slice_shape[0];
  return subscript;
}

template <>
Eigen::DSizes<Eigen::DenseIndex, 5> GetSliceIndices(
    absl::Span<const int32> num_partitions,
    const Eigen::DSizes<Eigen::DenseIndex, 5>& slice_shape, const int index) {
  Eigen::DSizes<Eigen::DenseIndex, 5> subscript;
  subscript[4] = (index % num_partitions[4]) * slice_shape[4];
  subscript[3] =
      ((index / num_partitions[4]) % num_partitions[3]) * slice_shape[3];
  subscript[2] =
      ((index / (num_partitions[4] * num_partitions[3])) % num_partitions[2]) *
      slice_shape[2];
  subscript[1] =
      ((index / (num_partitions[4] * num_partitions[3] * num_partitions[2])) %
       num_partitions[1]) *
      slice_shape[1];
  subscript[0] = (index / (num_partitions[4] * num_partitions[3] *
                           num_partitions[2] * num_partitions[1])) *
                 slice_shape[0];
  return subscript;
}

template <>
Eigen::DSizes<Eigen::DenseIndex, 6> GetSliceIndices(
    absl::Span<const int32> num_partitions,
    const Eigen::DSizes<Eigen::DenseIndex, 6>& slice_shape, const int index) {
  Eigen::DSizes<Eigen::DenseIndex, 6> subscript;
  subscript[5] = (index % num_partitions[5]) * slice_shape[5];
  subscript[4] =
      ((index / num_partitions[5]) % num_partitions[4]) * slice_shape[4];
  subscript[3] =
      ((index / (num_partitions[5] * num_partitions[4])) % num_partitions[3]) *
      slice_shape[3];
  subscript[2] =
      ((index / (num_partitions[5] * num_partitions[4] * num_partitions[3])) %
       num_partitions[2]) *
      slice_shape[2];
  subscript[1] = ((index / (num_partitions[5] * num_partitions[4] *
                            num_partitions[3] * num_partitions[2])) %
                  num_partitions[1]) *
                 slice_shape[1];
  subscript[0] =
      (index / (num_partitions[5] * num_partitions[4] * num_partitions[3] *
                num_partitions[2] * num_partitions[1])) *
      slice_shape[0];
  return subscript;
}

template <>
Eigen::DSizes<Eigen::DenseIndex, 7> GetSliceIndices(
    absl::Span<const int32> num_partitions,
    const Eigen::DSizes<Eigen::DenseIndex, 7>& slice_shape, const int index) {
  Eigen::DSizes<Eigen::DenseIndex, 7> subscript;
  subscript[6] = (index % num_partitions[6]) * slice_shape[6];
  subscript[5] =
      ((index / num_partitions[6]) % num_partitions[5]) * slice_shape[5];
  subscript[4] =
      ((index / (num_partitions[6] * num_partitions[5])) % num_partitions[4]) *
      slice_shape[4];
  subscript[3] =
      ((index / (num_partitions[6] * num_partitions[5] * num_partitions[4])) %
       num_partitions[3]) *
      slice_shape[3];
  subscript[2] = ((index / (num_partitions[6] * num_partitions[5] *
                            num_partitions[4] * num_partitions[3])) %
                  num_partitions[2]) *
                 slice_shape[2];
  subscript[1] =
      ((index / (num_partitions[6] * num_partitions[5] * num_partitions[4] *
                 num_partitions[3] * num_partitions[2])) %
       num_partitions[1]) *
      slice_shape[1];
  subscript[0] =
      (index / (num_partitions[6] * num_partitions[5] * num_partitions[4] *
                num_partitions[3] * num_partitions[2] * num_partitions[1])) *
      slice_shape[0];
  return subscript;
}

template <>
Eigen::DSizes<Eigen::DenseIndex, 8> GetSliceIndices(
    absl::Span<const int32> num_partitions,
    const Eigen::DSizes<Eigen::DenseIndex, 8>& slice_shape, const int index) {
  Eigen::DSizes<Eigen::DenseIndex, 8> subscript;
  subscript[7] = (index % num_partitions[7]) * slice_shape[7];
  subscript[6] =
      ((index / num_partitions[7]) % num_partitions[6]) * slice_shape[6];
  subscript[5] =
      ((index / (num_partitions[7] * num_partitions[6])) % num_partitions[5]) *
      slice_shape[5];
  subscript[4] =
      ((index / (num_partitions[7] * num_partitions[6] * num_partitions[5])) %
       num_partitions[4]) *
      slice_shape[4];
  subscript[3] = ((index / (num_partitions[7] * num_partitions[6] *
                            num_partitions[5] * num_partitions[4])) %
                  num_partitions[3]) *
                 slice_shape[3];
  subscript[2] =
      ((index / (num_partitions[7] * num_partitions[6] * num_partitions[5] *
                 num_partitions[4] * num_partitions[3])) %
       num_partitions[2]) *
      slice_shape[2];
  subscript[1] =
      ((index / (num_partitions[7] * num_partitions[6] * num_partitions[5] *
                 num_partitions[4] * num_partitions[3] * num_partitions[2])) %
       num_partitions[1]) *
      slice_shape[1];
  subscript[0] =
      (index / (num_partitions[7] * num_partitions[6] * num_partitions[5] *
                num_partitions[4] * num_partitions[3] * num_partitions[2] *
                num_partitions[1])) *
      slice_shape[0];
  return subscript;
}

constexpr absl::string_view kTensorName = "'input' tensor";
constexpr absl::string_view kResourceName = "'resource' variable tensor";

template <typename Device, typename T, bool Resource>
class XlaSplitNDBaseOp : public OpKernel {
 public:
  explicit XlaSplitNDBaseOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePStpuPSkernelsPSsharding_util_opsDTcc mht_2(mht_2_v, 475, "", "./tensorflow/core/tpu/kernels/sharding_util_ops.cc", "XlaSplitNDBaseOp");

    OP_REQUIRES_OK(ctx,
                   GetAndValidateAttributes<true>(ctx, num_splits_, num_slices_,
                                                  paddings_, has_paddings_));
  }

 protected:
  void ComputeInternal(
      OpKernelContext* ctx,
      const std::function<Status(const Tensor&)>& assign_or_copy_value_fn,
      const Tensor* input) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePStpuPSkernelsPSsharding_util_opsDTcc mht_3(mht_3_v, 488, "", "./tensorflow/core/tpu/kernels/sharding_util_ops.cc", "ComputeInternal");

    absl::string_view input_name = Resource ? kResourceName : kTensorName;
    const int rank = input->shape().dims();

    OP_REQUIRES(ctx, rank > 0 && rank <= 8,
                errors::InvalidArgument(
                    input_name, " must have rank in range (0, 8], but got ",
                    rank, "."));
    OP_REQUIRES(
        ctx, rank == num_splits_.size(),
        errors::InvalidArgument(
            input_name, " rank must be the same as 'num_splits' length ",
            num_splits_.size(), ", but got rank ", rank, "."));

    for (int dim = 0; dim < rank; ++dim) {
      OP_REQUIRES(
          ctx,
          (input->shape().dim_size(dim) + paddings_[dim]) % num_splits_[dim] ==
              0,
          errors::InvalidArgument(input_name, " shape dimension ", dim, " (",
                                  input->shape().dim_size(dim),
                                  ") with padding ", paddings_[dim],
                                  " must be evenly divisible by 'num_splits' ",
                                  num_splits_[dim], "."));
    }

    if (has_paddings_) {
      if (rank == 1) {
        SliceAndPad<1>(ctx, input);
      } else if (rank == 2) {
        SliceAndPad<2>(ctx, input);
      } else if (rank == 3) {
        SliceAndPad<3>(ctx, input);
      } else if (rank == 4) {
        SliceAndPad<4>(ctx, input);
      } else if (rank == 5) {
        SliceAndPad<5>(ctx, input);
      } else if (rank == 6) {
        SliceAndPad<6>(ctx, input);
      } else if (rank == 7) {
        SliceAndPad<7>(ctx, input);
      } else if (rank == 8) {
        SliceAndPad<8>(ctx, input);
      }
      return;
    }

    if (rank == 1) {
      Slice<1>(ctx, assign_or_copy_value_fn, input);
    } else if (rank == 2) {
      Slice<2>(ctx, assign_or_copy_value_fn, input);
    } else if (rank == 3) {
      Slice<3>(ctx, assign_or_copy_value_fn, input);
    } else if (rank == 4) {
      Slice<4>(ctx, assign_or_copy_value_fn, input);
    } else if (rank == 5) {
      Slice<5>(ctx, assign_or_copy_value_fn, input);
    } else if (rank == 6) {
      Slice<6>(ctx, assign_or_copy_value_fn, input);
    } else if (rank == 7) {
      Slice<7>(ctx, assign_or_copy_value_fn, input);
    } else if (rank == 8) {
      Slice<8>(ctx, assign_or_copy_value_fn, input);
    }
  }

 private:
  template <int Rank>
  void SliceAndPad(OpKernelContext* ctx, const Tensor* input) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePStpuPSkernelsPSsharding_util_opsDTcc mht_4(mht_4_v, 559, "", "./tensorflow/core/tpu/kernels/sharding_util_ops.cc", "SliceAndPad");

    const auto& shape = input->shape().dim_sizes();
    const Device& device = ctx->eigen_device<Device>();
    if (num_slices_ == 1) {
      Eigen::array<Eigen::IndexPair<int64_t>, Rank> tensor_paddings;
      TensorShape output_shape;
      for (int i = 0; i < Rank; ++i) {
        tensor_paddings[i] = {0, paddings_[i]};
        output_shape.AddDim(shape[i] + paddings_[i]);
      }
      Tensor* output = nullptr;
      OP_REQUIRES_OK(ctx,
                     ctx->allocate_output(/*index=*/0, output_shape, &output));
      output->tensor<T, Rank>().device(device) =
          input->tensor<T, Rank>().pad(tensor_paddings, T());
      return;
    }

    // Slice shape with optional padding.
    TensorShape output_slice_shape;
    for (int dim = 0; dim < Rank; ++dim) {
      output_slice_shape.AddDim((shape[dim] + paddings_[dim]) /
                                num_splits_[dim]);
    }
    const Eigen::DSizes<Eigen::DenseIndex, Rank> output_slice_shape_dsizes =
        output_slice_shape.AsEigenDSizes<Rank>();

    for (int i = 0; i < num_slices_; ++i) {
      Tensor* output_slice = nullptr;
      OP_REQUIRES_OK(ctx, ctx->allocate_output(
                              /*index=*/i, output_slice_shape, &output_slice));

      int num_complete_pad_dims = 0;
      int num_partial_pad_dims = 0;
      TensorShape non_padded_slice_shape;
      Eigen::array<Eigen::IndexPair<int64_t>, Rank> slice_paddings;
      Eigen::DSizes<Eigen::DenseIndex, Rank> slice_indices =
          GetSliceIndices<Rank>(num_splits_, output_slice_shape_dsizes, i);

      // Calculate paddings necessary for slice instead of padding input and
      // slicing subsequently to reduce temporary memory allocation.
      for (int dim = 0; dim < Rank; ++dim) {
        const int64_t dim_size = shape[dim];
        if (slice_indices[dim] >= dim_size) {
          // Complete padding.
          slice_indices[dim] = dim_size;
          non_padded_slice_shape.AddDim(0);
          slice_paddings[dim] = {0, output_slice_shape_dsizes[dim]};
          ++num_complete_pad_dims;
        } else if (slice_indices[dim] + output_slice_shape_dsizes[dim] >
                   dim_size) {
          // Partial padding.
          non_padded_slice_shape.AddDim(dim_size - slice_indices[dim]);
          slice_paddings[dim] = {0, output_slice_shape_dsizes[dim] -
                                        non_padded_slice_shape.dim_size(dim)};
          ++num_partial_pad_dims;
        } else {
          non_padded_slice_shape.AddDim(output_slice_shape_dsizes[dim]);
        }
      }

      if (num_complete_pad_dims == Rank) {
        output_slice->flat<T>().device(device) =
            output_slice->flat<T>().constant(T());
      } else if (num_complete_pad_dims > 0 || num_partial_pad_dims > 0) {
        output_slice->flat<T>().device(device) =
            output_slice->flat<T>().constant(T());
        Eigen::DSizes<Eigen::DenseIndex, Rank> non_padded_slice_shape_dsizes =
            non_padded_slice_shape.AsEigenDSizes<Rank>();
        output_slice->tensor<T, Rank>()
            .slice(Eigen::DSizes<Eigen::DenseIndex, Rank>(),
                   non_padded_slice_shape_dsizes)
            .device(device) = input->tensor<T, Rank>().slice(
            slice_indices, non_padded_slice_shape_dsizes);
      } else {
        output_slice->tensor<T, Rank>().device(device) =
            input->tensor<T, Rank>().slice(slice_indices,
                                           output_slice_shape_dsizes);
      }
    }
  }

  template <int Rank>
  void Slice(
      OpKernelContext* ctx,
      const std::function<Status(const Tensor&)>& assign_or_copy_value_fn,
      const Tensor* input) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePStpuPSkernelsPSsharding_util_opsDTcc mht_5(mht_5_v, 648, "", "./tensorflow/core/tpu/kernels/sharding_util_ops.cc", "Slice");

    if (num_slices_ == 1) {
      OP_REQUIRES_OK(ctx, assign_or_copy_value_fn(*input));
      return;
    }

    const auto& shape = input->shape().dim_sizes();
    const Device& device = ctx->eigen_device<Device>();

    TensorShape output_slice_shape;
    for (int dim = 0; dim < Rank; ++dim) {
      output_slice_shape.AddDim(shape[dim] / num_splits_[dim]);
    }
    const Eigen::DSizes<Eigen::DenseIndex, Rank> output_slice_shape_dsizes =
        output_slice_shape.AsEigenDSizes<Rank>();

    for (int i = 0; i < num_slices_; ++i) {
      Tensor* output_slice = nullptr;
      OP_REQUIRES_OK(ctx, ctx->allocate_output(
                              /*index=*/i, output_slice_shape, &output_slice));
      Eigen::DSizes<Eigen::DenseIndex, Rank> slice_indices =
          GetSliceIndices<Rank>(num_splits_, output_slice_shape_dsizes, i);
      output_slice->tensor<T, Rank>().device(device) =
          input->tensor<T, Rank>().slice(slice_indices,
                                         output_slice_shape_dsizes);
    }
  }

  std::vector<int32> num_splits_;
  int num_slices_ = 1;
  std::vector<int32> paddings_;
  bool has_paddings_ = false;
};

template <typename Device, typename T>
class XlaSplitNDOp : public XlaSplitNDBaseOp<Device, T, false> {
 public:
  explicit XlaSplitNDOp(OpKernelConstruction* ctx)
      : XlaSplitNDBaseOp<Device, T, false>(ctx) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePStpuPSkernelsPSsharding_util_opsDTcc mht_6(mht_6_v, 689, "", "./tensorflow/core/tpu/kernels/sharding_util_ops.cc", "XlaSplitNDOp");
}

  void Compute(OpKernelContext* ctx) override {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePStpuPSkernelsPSsharding_util_opsDTcc mht_7(mht_7_v, 694, "", "./tensorflow/core/tpu/kernels/sharding_util_ops.cc", "Compute");

    const Tensor& input = ctx->input(0);

    auto assign_or_copy_value_fn = [&ctx](const Tensor& input) -> Status {
      ctx->set_output(/*index=*/0, input);
      return Status::OK();
    };

    this->ComputeInternal(ctx, assign_or_copy_value_fn, &input);
  }
};

template <typename Device, typename T>
class ReadVariableXlaSplitNDOp : public XlaSplitNDBaseOp<Device, T, true> {
 public:
  explicit ReadVariableXlaSplitNDOp(OpKernelConstruction* ctx)
      : XlaSplitNDBaseOp<Device, T, true>(ctx) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScorePStpuPSkernelsPSsharding_util_opsDTcc mht_8(mht_8_v, 713, "", "./tensorflow/core/tpu/kernels/sharding_util_ops.cc", "ReadVariableXlaSplitNDOp");

    OP_REQUIRES_OK(ctx, ctx->GetAttr("T", &dtype_));
  }

  void Compute(OpKernelContext* ctx) override {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScorePStpuPSkernelsPSsharding_util_opsDTcc mht_9(mht_9_v, 720, "", "./tensorflow/core/tpu/kernels/sharding_util_ops.cc", "Compute");

    core::RefCountPtr<Var> variable;
    const ResourceHandle& handle = HandleFromInput(ctx, 0);
    const Status status = LookupResource(ctx, handle, &variable);
    OP_REQUIRES(
        ctx, status.ok(),
        errors::InvalidArgument("'resource' variable handle ('", handle.name(),
                                "') container ('", handle.container(),
                                "') cannot be found."));

    tf_shared_lock ml(*variable->mu());
    const Tensor* input = variable->tensor();
    OP_REQUIRES(
        ctx, input->dtype() == dtype_,
        CreateResourceInvalidDTypeError<false>(handle, input->dtype(), dtype_));

    auto assign_or_copy_value_fn = [&ctx,
                                    &variable](const Tensor& input) -> Status {
      if (variable->copy_on_read_mode.load()) {
        Tensor* output;
        TF_RETURN_IF_ERROR(
            ctx->allocate_output(/*index=*/0, input.shape(), &output));
        output->flat<T>().device(ctx->eigen_device<Device>()) = input.flat<T>();
      } else {
        ctx->set_output(/*index=*/0, input);
      }
      return Status::OK();
    };

    this->ComputeInternal(ctx, assign_or_copy_value_fn, input);
  }

 private:
  DataType dtype_;
};

#define REGISTER_XLA_SPLIT_ND(type)                                    \
  REGISTER_KERNEL_BUILDER(                                             \
      Name("XlaSplitND").Device(DEVICE_CPU).TypeConstraint<type>("T"), \
      XlaSplitNDOp<Eigen::ThreadPoolDevice, type>)

TF_CALL_POD_TYPES(REGISTER_XLA_SPLIT_ND);
TF_CALL_QUANTIZED_TYPES(REGISTER_XLA_SPLIT_ND);
#undef REGISTER_XLA_SPLIT_ND

#define REGISTER_READ_VARIABLE_XLA_SPLIT_ND(type) \
  REGISTER_KERNEL_BUILDER(                        \
      Name("ReadVariableXlaSplitND")              \
          .Device(DEVICE_CPU)                     \
          .TypeConstraint<type>("T"),             \
      ReadVariableXlaSplitNDOp<Eigen::ThreadPoolDevice, type>)

TF_CALL_POD_TYPES(REGISTER_READ_VARIABLE_XLA_SPLIT_ND);
TF_CALL_QUANTIZED_TYPES(REGISTER_READ_VARIABLE_XLA_SPLIT_ND);
#undef REGISTER_READ_VARIABLE_XLA_SPLIT_ND

template <typename Device, typename T, bool Resource>
class XlaConcatNDBaseOp : public OpKernel {
 public:
  explicit XlaConcatNDBaseOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScorePStpuPSkernelsPSsharding_util_opsDTcc mht_10(mht_10_v, 782, "", "./tensorflow/core/tpu/kernels/sharding_util_ops.cc", "XlaConcatNDBaseOp");

    OP_REQUIRES_OK(
        ctx, GetAndValidateAttributes<false>(ctx, num_concats_, num_slices_,
                                             paddings_, has_paddings_));
  }

 protected:
  Status GetInputsAndOutputShape(OpKernelContext* ctx, OpInputList& inputs,
                                 TensorShape& output_shape) {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScorePStpuPSkernelsPSsharding_util_opsDTcc mht_11(mht_11_v, 793, "", "./tensorflow/core/tpu/kernels/sharding_util_ops.cc", "GetInputsAndOutputShape");

    TF_RETURN_IF_ERROR(ctx->input_list("inputs", &inputs));
    DCHECK_EQ(inputs.size(), num_slices_);

    const TensorShape& slice_shape = inputs[0].shape();
    if (slice_shape.dims() != num_concats_.size()) {
      return errors::InvalidArgument(
          "'inputs' rank must be the same as 'num_concats' length ",
          num_concats_.size(), ", but got rank ", slice_shape.dims(), ".");
    }
    for (int i = 1; i < num_slices_; ++i) {
      const TensorShape& slice_shape_i = inputs[i].shape();
      if (slice_shape != slice_shape_i) {
        return errors::InvalidArgument(
            "'inputs' must all have the same expected shape ", slice_shape,
            ", but got ", slice_shape_i, " at index ", i, ".");
      }
    }

    for (int i = 0, e = num_concats_.size(); i < e; ++i) {
      const int max_dim_size = slice_shape.dim_size(i) * num_concats_[i];
      if (paddings_[i] > max_dim_size) {
        return errors::InvalidArgument(
            "'paddings' must not exceed expected output shape dimension ",
            max_dim_size, " at index ", i, ", but got ", paddings_[i], ".");
      }
      output_shape.AddDim(max_dim_size - paddings_[i]);
    }

    return Status::OK();
  }

  void ComputeInternal(
      OpKernelContext* ctx, const OpInputList& inputs,
      const std::function<Status(const Tensor&)>& assign_or_copy_value_fn,
      const std::function<StatusOr<Tensor*>()>& get_output_fn) {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScorePStpuPSkernelsPSsharding_util_opsDTcc mht_12(mht_12_v, 831, "", "./tensorflow/core/tpu/kernels/sharding_util_ops.cc", "ComputeInternal");

    const int rank = inputs[0].shape().dims();

    OP_REQUIRES(ctx, rank > 0 && rank <= 8,
                errors::InvalidArgument(
                    "'inputs' tensors must have rank in range (0, 8], but got ",
                    rank, "."));

    if (has_paddings_) {
      if (rank == 1) {
        UnpadAndAssign<1>(ctx, inputs, get_output_fn);
      } else if (rank == 2) {
        UnpadAndAssign<2>(ctx, inputs, get_output_fn);
      } else if (rank == 3) {
        UnpadAndAssign<3>(ctx, inputs, get_output_fn);
      } else if (rank == 4) {
        UnpadAndAssign<4>(ctx, inputs, get_output_fn);
      } else if (rank == 5) {
        UnpadAndAssign<5>(ctx, inputs, get_output_fn);
      } else if (rank == 6) {
        UnpadAndAssign<6>(ctx, inputs, get_output_fn);
      } else if (rank == 7) {
        UnpadAndAssign<7>(ctx, inputs, get_output_fn);
      } else if (rank == 8) {
        UnpadAndAssign<8>(ctx, inputs, get_output_fn);
      }
      return;
    }

    if (rank == 1) {
      Assign<1>(ctx, inputs, assign_or_copy_value_fn, get_output_fn);
    } else if (rank == 2) {
      Assign<2>(ctx, inputs, assign_or_copy_value_fn, get_output_fn);
    } else if (rank == 3) {
      Assign<3>(ctx, inputs, assign_or_copy_value_fn, get_output_fn);
    } else if (rank == 4) {
      Assign<4>(ctx, inputs, assign_or_copy_value_fn, get_output_fn);
    } else if (rank == 5) {
      Assign<5>(ctx, inputs, assign_or_copy_value_fn, get_output_fn);
    } else if (rank == 6) {
      Assign<6>(ctx, inputs, assign_or_copy_value_fn, get_output_fn);
    } else if (rank == 7) {
      Assign<7>(ctx, inputs, assign_or_copy_value_fn, get_output_fn);
    } else if (rank == 8) {
      Assign<8>(ctx, inputs, assign_or_copy_value_fn, get_output_fn);
    }
  }

 private:
  template <int Rank>
  void UnpadAndAssign(OpKernelContext* ctx, const OpInputList& inputs,
                      const std::function<StatusOr<Tensor*>()>& get_output_fn) {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPScorePStpuPSkernelsPSsharding_util_opsDTcc mht_13(mht_13_v, 885, "", "./tensorflow/core/tpu/kernels/sharding_util_ops.cc", "UnpadAndAssign");

    auto status_or_output = get_output_fn();
    OP_REQUIRES_OK(ctx, status_or_output.status());
    Tensor* output = status_or_output.ConsumeValueOrDie();

    const Device& device = ctx->eigen_device<Device>();
    if (num_slices_ == 1) {
      output->tensor<T, Rank>().device(device) =
          inputs[0].tensor<T, Rank>().slice(
              Eigen::DSizes<Eigen::DenseIndex, Rank>(),
              output->shape().AsEigenDSizes<Rank>());
      return;
    }

    Eigen::DSizes<Eigen::DenseIndex, Rank> slice_shape_dsizes =
        inputs[0].shape().AsEigenDSizes<Rank>();
    for (int i = 0; i < num_slices_; ++i) {
      Eigen::DSizes<Eigen::DenseIndex, Rank> slice_indices =
          GetSliceIndices<Rank>(num_concats_, slice_shape_dsizes, i);

      int num_complete_pad_dims = 0;
      int num_partial_pad_dims = 0;
      TensorShape non_padded_slice_shape;
      // Calculate paddings necessary to strip from slice.
      for (int dim = 0; dim < Rank; ++dim) {
        const int64_t dim_size = output->shape().dim_size(dim);
        if (slice_indices[dim] >= dim_size) {
          // Complete padding.
          slice_indices[dim] = dim_size;
          non_padded_slice_shape.AddDim(0);
          ++num_complete_pad_dims;
        } else if (slice_indices[dim] + slice_shape_dsizes[dim] > dim_size) {
          // Partial padding.
          non_padded_slice_shape.AddDim(dim_size - slice_indices[dim]);
          ++num_partial_pad_dims;
        } else {
          non_padded_slice_shape.AddDim(slice_shape_dsizes[dim]);
        }
      }

      if (num_complete_pad_dims == Rank) {
        continue;
      } else if (num_complete_pad_dims > 0 || num_partial_pad_dims > 0) {
        Eigen::DSizes<Eigen::DenseIndex, Rank> non_padded_slice_shape_dsizes =
            non_padded_slice_shape.AsEigenDSizes<Rank>();
        output->tensor<T, Rank>()
            .slice(slice_indices, non_padded_slice_shape_dsizes)
            .device(device) = inputs[i].tensor<T, Rank>().slice(
            Eigen::DSizes<Eigen::DenseIndex, Rank>(),
            non_padded_slice_shape_dsizes);
      } else {
        output->tensor<T, Rank>()
            .slice(slice_indices, slice_shape_dsizes)
            .device(device) = inputs[i].tensor<T, Rank>();
      }
    }
  }

  template <int Rank>
  void Assign(
      OpKernelContext* ctx, const OpInputList& inputs,
      const std::function<Status(const Tensor&)>& assign_or_copy_value_fn,
      const std::function<StatusOr<Tensor*>()>& get_output_fn) {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPScorePStpuPSkernelsPSsharding_util_opsDTcc mht_14(mht_14_v, 950, "", "./tensorflow/core/tpu/kernels/sharding_util_ops.cc", "Assign");

    const Device& device = ctx->eigen_device<Device>();
    if (num_slices_ == 1) {
      OP_REQUIRES_OK(ctx, assign_or_copy_value_fn(inputs[0]));
      return;
    }

    auto status_or_output = get_output_fn();
    OP_REQUIRES_OK(ctx, status_or_output.status());
    Tensor* output = status_or_output.ConsumeValueOrDie();

    Eigen::DSizes<Eigen::DenseIndex, Rank> slice_shape_dsizes =
        inputs[0].shape().AsEigenDSizes<Rank>();
    for (int i = 0; i < num_slices_; ++i) {
      Eigen::DSizes<Eigen::DenseIndex, Rank> slice_indices =
          GetSliceIndices<Rank>(num_concats_, slice_shape_dsizes, i);
      output->tensor<T, Rank>()
          .slice(slice_indices, slice_shape_dsizes)
          .device(device) = inputs[i].tensor<T, Rank>();
    }
  }

  std::vector<int32> num_concats_;
  int num_slices_ = 1;
  std::vector<int32> paddings_;
  bool has_paddings_ = false;
};

template <typename Device, typename T>
class XlaConcatNDOp : public XlaConcatNDBaseOp<Device, T, false> {
 public:
  explicit XlaConcatNDOp(OpKernelConstruction* ctx)
      : XlaConcatNDBaseOp<Device, T, false>(ctx) {
   std::vector<std::string> mht_15_v;
   MHTracer_DTPStensorflowPScorePStpuPSkernelsPSsharding_util_opsDTcc mht_15(mht_15_v, 985, "", "./tensorflow/core/tpu/kernels/sharding_util_ops.cc", "XlaConcatNDOp");
}

  void Compute(OpKernelContext* ctx) override {
   std::vector<std::string> mht_16_v;
   MHTracer_DTPStensorflowPScorePStpuPSkernelsPSsharding_util_opsDTcc mht_16(mht_16_v, 990, "", "./tensorflow/core/tpu/kernels/sharding_util_ops.cc", "Compute");

    OpInputList inputs;
    TensorShape output_shape;
    OP_REQUIRES_OK(ctx,
                   this->GetInputsAndOutputShape(ctx, inputs, output_shape));

    auto assign_or_copy_value_fn = [&ctx](const Tensor& input) -> Status {
      ctx->set_output(/*index=*/0, input);
      return Status::OK();
    };

    auto get_output_fn = [&ctx, &output_shape]() -> StatusOr<Tensor*> {
      Tensor* output = nullptr;
      TF_RETURN_IF_ERROR(
          ctx->allocate_output(/*index=*/0, output_shape, &output));
      return output;
    };
    this->ComputeInternal(ctx, inputs, assign_or_copy_value_fn, get_output_fn);
  }
};

template <typename Device, typename T>
class AssignVariableXlaConcatNDOp : public XlaConcatNDBaseOp<Device, T, true> {
 public:
  explicit AssignVariableXlaConcatNDOp(OpKernelConstruction* ctx)
      : XlaConcatNDBaseOp<Device, T, true>(ctx) {
   std::vector<std::string> mht_17_v;
   MHTracer_DTPStensorflowPScorePStpuPSkernelsPSsharding_util_opsDTcc mht_17(mht_17_v, 1018, "", "./tensorflow/core/tpu/kernels/sharding_util_ops.cc", "AssignVariableXlaConcatNDOp");

    OP_REQUIRES_OK(ctx, ctx->GetAttr("T", &dtype_));
  }

  void Compute(OpKernelContext* ctx) override {
   std::vector<std::string> mht_18_v;
   MHTracer_DTPStensorflowPScorePStpuPSkernelsPSsharding_util_opsDTcc mht_18(mht_18_v, 1025, "", "./tensorflow/core/tpu/kernels/sharding_util_ops.cc", "Compute");

    OpInputList inputs;
    TensorShape output_shape;
    OP_REQUIRES_OK(ctx,
                   this->GetInputsAndOutputShape(ctx, inputs, output_shape));

    core::RefCountPtr<Var> variable;
    const ResourceHandle& handle = HandleFromInput(ctx, 0);
    if (handle.dtypes_and_shapes().size() == 1) {
      const DtypeAndPartialTensorShape dtype_and_shape =
          handle.dtypes_and_shapes().front();
      OP_REQUIRES(ctx, dtype_and_shape.dtype == dtype_,
                  CreateResourceInvalidDTypeError<true>(
                      handle, dtype_and_shape.dtype, dtype_));
      OP_REQUIRES(ctx, dtype_and_shape.shape.IsCompatibleWith(output_shape),
                  errors::InvalidArgument(
                      "'resource' variable handle ('", handle.name(),
                      "') container ('", handle.container(),
                      "') shape must be compatible with expected shape ",
                      output_shape, ", but got ", dtype_and_shape.shape, "."));
    }
    OP_REQUIRES_OK(ctx, LookupOrCreateResource<Var>(ctx, handle, &variable,
                                                    [this](Var** ptr) {
                                                      *ptr = new Var(dtype_);
                                                      return Status::OK();
                                                    }));
    mutex_lock ml(*variable->mu());

    OP_REQUIRES(ctx, variable->tensor()->dtype() == dtype_,
                CreateResourceInvalidDTypeError<false>(
                    handle, variable->tensor()->dtype(), dtype_));

    auto assign_or_copy_value_fn = [this, &ctx, &output_shape,
                                    &variable](const Tensor& input) -> Status {
      if (variable->copy_on_read_mode.load()) {
        TF_RETURN_IF_ERROR(
            ctx->allocate_temp(dtype_, output_shape, variable->tensor()));
        variable->tensor()->flat<T>().device(ctx->eigen_device<Device>()) =
            input.flat<T>();
      } else {
        *variable->tensor() = input;
      }
      return Status::OK();
    };

    auto get_output_fn = [this, &ctx, &output_shape,
                          &variable]() -> StatusOr<Tensor*> {
      if (variable->copy_on_read_mode.load() ||
          !variable->tensor()->RefCountIsOne() ||
          !variable->tensor()->shape().IsSameSize(output_shape)) {
        TF_RETURN_IF_ERROR(
            ctx->allocate_temp(dtype_, output_shape, variable->tensor()));
      }
      return variable->tensor();
    };

    this->ComputeInternal(ctx, inputs, assign_or_copy_value_fn, get_output_fn);
    variable->is_initialized = true;
  }

  DataType dtype_;
};

#define REGISTER_XLA_CONCAT_ND(type)                                    \
  REGISTER_KERNEL_BUILDER(                                              \
      Name("XlaConcatND").Device(DEVICE_CPU).TypeConstraint<type>("T"), \
      XlaConcatNDOp<Eigen::ThreadPoolDevice, type>)

TF_CALL_POD_TYPES(REGISTER_XLA_CONCAT_ND);
TF_CALL_QUANTIZED_TYPES(REGISTER_XLA_CONCAT_ND);
#undef REGISTER_XLA_CONCAT_ND

#define REGISTER_ASSIGN_VARIABLE_XLA_CONCAT_ND(type) \
  REGISTER_KERNEL_BUILDER(                           \
      Name("AssignVariableXlaConcatND")              \
          .Device(DEVICE_CPU)                        \
          .TypeConstraint<type>("T"),                \
      AssignVariableXlaConcatNDOp<Eigen::ThreadPoolDevice, type>)

TF_CALL_POD_TYPES(REGISTER_ASSIGN_VARIABLE_XLA_CONCAT_ND);
TF_CALL_QUANTIZED_TYPES(REGISTER_ASSIGN_VARIABLE_XLA_CONCAT_ND);
#undef REGISTER_ASSIGN_VARIABLE_XLA_CONCAT_ND

}  // anonymous namespace
}  // namespace tensorflow
