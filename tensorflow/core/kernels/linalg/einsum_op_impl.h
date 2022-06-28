/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_CORE_KERNELS_LINALG_EINSUM_OP_IMPL_H_
#define TENSORFLOW_CORE_KERNELS_LINALG_EINSUM_OP_IMPL_H_
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
class MHTracer_DTPStensorflowPScorePSkernelsPSlinalgPSeinsum_op_implDTh {
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
   MHTracer_DTPStensorflowPScorePSkernelsPSlinalgPSeinsum_op_implDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSkernelsPSlinalgPSeinsum_op_implDTh() {
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


#define EIGEN_USE_THREADS
#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
#define EIGEN_USE_GPU
#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM

#include "absl/container/flat_hash_map.h"
#include "absl/strings/str_split.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/kernel_def_builder.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/kernels/fill_functor.h"
#include "tensorflow/core/kernels/linalg/einsum_op.h"
#include "tensorflow/core/kernels/matmul_op_impl.h"
#include "tensorflow/core/kernels/reduction_ops_common.h"
#include "tensorflow/core/kernels/transpose_functor.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/gtl/inlined_vector.h"
#include "tensorflow/core/lib/math/math_util.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/profiler/lib/traceme.h"
#include "tensorflow/core/util/einsum_op_util.h"

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
#include "tensorflow/core/kernels/reduction_ops_common_gpu.h"
#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM

namespace tensorflow {

using CPUDevice = Eigen::ThreadPoolDevice;
using GPUDevice = Eigen::GpuDevice;

using ShapeVec = gtl::InlinedVector<int64_t, 8>;
using Labels = gtl::InlinedVector<int, 8>;
using OperandLabels = gtl::InlinedVector<Labels, 2>;
using LabelCounts = gtl::InlinedVector<int, 8>;
using OperandLabelCounts = gtl::InlinedVector<LabelCounts, 2>;
using LabelToDimSizes = gtl::InlinedVector<int64_t, 8>;

struct EinsumHelper {
  // Insert new (unnamed) broadcasting labels at the location of ellipsis.
  static void InsertBroadcastLabels(int num_bcast_dims, int num_named_labels,
                                    int ellipsis_axis, Labels* labels,
                                    LabelCounts* label_counts) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSlinalgPSeinsum_op_implDTh mht_0(mht_0_v, 235, "", "./tensorflow/core/kernels/linalg/einsum_op_impl.h", "InsertBroadcastLabels");

    labels->erase(labels->begin() + ellipsis_axis);
    labels->insert(labels->begin() + ellipsis_axis, num_bcast_dims, 0);
    std::iota(labels->begin() + ellipsis_axis,
              labels->begin() + ellipsis_axis + num_bcast_dims,
              num_named_labels);
    // Increment label counts. Since these are new labels, the count is set
    // to 1.
    label_counts->resize(num_named_labels + num_bcast_dims, 1);
  }

  // Record and validate the label to dimension mapping. Must be a named
  // (non-broadcasting) label as broadcasting labels don't have a fixed
  // dimension.
  static Status RecordLabelToDimension(const int label, const int axis,
                                       const Tensor& input,
                                       LabelToDimSizes* label_to_dim_sizes) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSlinalgPSeinsum_op_implDTh mht_1(mht_1_v, 254, "", "./tensorflow/core/kernels/linalg/einsum_op_impl.h", "RecordLabelToDimension");

    const int64_t input_dim = input.dim_size(axis);
    // We know that label_to_dim_sizes has the size to accommodate named labels.
    if (label_to_dim_sizes->at(label) != 0 &&
        label_to_dim_sizes->at(label) != input_dim) {
      return errors::InvalidArgument(
          "Expected dimension ", label_to_dim_sizes->at(label), " at axis ",
          axis, " of the input shaped ", input.shape().DebugString(),
          " but got dimension ", input_dim);
    }
    (*label_to_dim_sizes)[label] = input_dim;
    return Status::OK();
  }

  // Validate input dimensions and populate unnamed labels and their label
  // counts.
  static Status ProcessDimensions(
      const OpInputList& inputs,
      const gtl::InlinedVector<bool, 2>& input_has_ellipsis,
      const bool output_has_ellipsis, OperandLabels* input_labels,
      Labels* output_labels, std::vector<EinsumDimensionType>* label_types,
      OperandLabelCounts* input_label_counts, LabelCounts* output_label_counts,
      LabelToDimSizes* label_to_dim_sizes) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSlinalgPSeinsum_op_implDTh mht_2(mht_2_v, 279, "", "./tensorflow/core/kernels/linalg/einsum_op_impl.h", "ProcessDimensions");

    if (inputs.size() != input_labels->size()) {
      return errors::InvalidArgument("Expected ", input_labels->size(),
                                     " inputs but got: ", inputs.size());
    }
    const int num_inputs = inputs.size();

    // We infer the number of broadcasting dimensions by taking the maximum rank
    // among the broadcasting subshapes of the input.
    int max_bcast_dims = 0;
    const int num_named_labels = label_types->size();
    label_to_dim_sizes->resize(num_named_labels);
    for (int i = 0; i < num_inputs; ++i) {
      Labels* labels = &(*input_labels)[i];

      if (!input_has_ellipsis[i]) {
        if (inputs[i].dims() != labels->size()) {
          return errors::InvalidArgument("Expected input ", i, " to have rank ",
                                         labels->size(),
                                         " but got: ", inputs[i].dims());
        }
        for (int label_idx = 0; label_idx < labels->size(); ++label_idx) {
          const int label = (*labels)[label_idx];
          TF_RETURN_IF_ERROR(RecordLabelToDimension(label, label_idx, inputs[i],
                                                    label_to_dim_sizes));
        }
        continue;
      }

      // Input has an ellipsis.
      if (inputs[i].dims() + 1 < labels->size()) {
        return errors::InvalidArgument(
            "Expected input ", i, " to have rank at least ", labels->size() - 1,
            " but got: ", inputs[i].dims());
      }
      int ellipsis_axis = -1;
      const int num_bcast_dims = inputs[i].dims() - labels->size() + 1;
      for (int label_idx = 0; label_idx < labels->size(); ++label_idx) {
        const int label = (*labels)[label_idx];
        if (label == kEllipsisLabel) {
          ellipsis_axis = label_idx;
          continue;
        }
        // Current label is not an ellipsis.
        const int axis =
            label_idx + (ellipsis_axis == -1 ? 0 : num_bcast_dims - 1);
        TF_RETURN_IF_ERROR(
            RecordLabelToDimension(label, axis, inputs[i], label_to_dim_sizes));
      }
      // Found an ellipsis. Replace 'kEllipsisLabel' with broadcasting
      // dimensions.
      if (ellipsis_axis != -1) {
        InsertBroadcastLabels(num_bcast_dims, num_named_labels, ellipsis_axis,
                              labels, &input_label_counts->at(i));
        max_bcast_dims = std::max(max_bcast_dims, num_bcast_dims);
      }
    }
    if (!absl::c_linear_search(input_has_ellipsis, true) &&
        !output_has_ellipsis) {
      return Status::OK();
    }
    // Insert broadcasting dimensions in the output labels.
    auto it =
        std::find(output_labels->begin(), output_labels->end(), kEllipsisLabel);
    if (it != output_labels->end()) {
      const int ellipsis_axis = it - output_labels->begin();
      InsertBroadcastLabels(max_bcast_dims, num_named_labels, ellipsis_axis,
                            output_labels, output_label_counts);
    } else if (max_bcast_dims > 0) {
      return errors::InvalidArgument(
          "Output contains ", max_bcast_dims,
          " broadcasting dimension(s) but no ellipsis "
          "(...) was found in the output subscripts.");
    }
    // Populate EinsumDimensionType for the new broadcasting labels.
    label_types->resize(num_named_labels + max_bcast_dims,
                        EinsumDimensionType::kBroadcasting);
    return Status::OK();
  }

  // Permutes the labels according to the given permutation.
  static void PermuteLabels(const std::vector<int>& permutation,
                            Labels* labels) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSlinalgPSeinsum_op_implDTh mht_3(mht_3_v, 364, "", "./tensorflow/core/kernels/linalg/einsum_op_impl.h", "PermuteLabels");

    Labels permuted_labels(labels->size());
    for (int i = 0; i < labels->size(); ++i) {
      permuted_labels[i] = (*labels)[permutation[i]];
    }
    labels->swap(permuted_labels);
  }

  // Returns a reshaped input Tensor. The underlying buffer is not copied.
  static Status CopyFrom(const Tensor& input, const TensorShape& shape,
                         Tensor* output) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSlinalgPSeinsum_op_implDTh mht_4(mht_4_v, 377, "", "./tensorflow/core/kernels/linalg/einsum_op_impl.h", "CopyFrom");

    if (output->CopyFrom(input, shape)) return Status::OK();
    return errors::Internal(
        "Encountered error while reshaping a Tensor of shape ",
        input.shape().DebugString(), " to shape ", shape.DebugString());
  }

  // Returns whether transposing would be a no-op; whether input has rank < 2 or
  // the permutation is the identity permutation.
  static bool ShouldTranspose(const TensorShape& input_shape,
                              const std::vector<int>& permutation) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSlinalgPSeinsum_op_implDTh mht_5(mht_5_v, 390, "", "./tensorflow/core/kernels/linalg/einsum_op_impl.h", "ShouldTranspose");

    if (input_shape.dims() < 2) return false;
    for (int i = 0; i < permutation.size(); ++i) {
      if (permutation[i] != i) return true;
    }
    return false;
  }

  // Transpose the input given a permutation. Returns a reference to the input
  // if transposing is not necessary.
  template <typename Device, typename T>
  static Status TransposeOperand(OpKernelContext* ctx, const Tensor& input,
                                 const std::vector<int>& permutation,
                                 Tensor* output) {
    if (!ShouldTranspose(input.shape(), permutation)) {
      return CopyFrom(input, input.shape(), output);
    }
    TensorShape transposed_shape;
    for (int i = 0; i < input.dims(); ++i) {
      transposed_shape.AddDim(input.dim_size(permutation[i]));
    }
    // For empty Tensors, just change the shape. E.g. we may need to transpose
    // from shape [1, 0, 5] to [5, 1, 0].
    if (input.NumElements() == 0) {
      return CopyFrom(input, transposed_shape, output);
    }
    TF_RETURN_IF_ERROR(
        ctx->allocate_temp(DataTypeToEnum<T>::value, transposed_shape, output));
    const Device& device = ctx->eigen_device<Device>();
    TF_RETURN_IF_ERROR(DoTranspose(device, input, permutation, output));
    return Status::OK();
  }

  // If there are repeated labels in either the input or output, then this
  // strides the input (e.g. iii->i) or inflates it (e.g. i->iii), respectively.
  template <typename Device, typename T>
  static Status StrideOrInflate(OpKernelContext* ctx, const Tensor& input,
                                const Labels& labels,
                                const LabelCounts& label_counts,
                                const bool should_inflate, Tensor* output) {
    // Return early if there are no repeated indices.
    if (absl::c_all_of(label_counts, [](int c) { return c <= 1; })) {
      return CopyFrom(input, input.shape(), output);
    }
    // We reshape so that each repeated label is compressed to one dimension.
    // E.g. For iiij -> ij, The shape [3, 3, 3, 5] would be compressed to [27,
    // 5]. Striding appropriately (in this case with strides 14 (=1+3+9) and 1)
    // recovers the generalized diagonal of shape [3, 5].
    ShapeVec reshape;
    ShapeVec strides;
    // Strided and inflated shapes correspond to input and output shapes,
    // respectively, should_inflate is true (vice-versa if should_inflate is
    // false). E.g. they are [3, 5] and [3, 3, 3, 5] in the above example.
    ShapeVec strided_shape;
    ShapeVec inflated_shape;
    for (int label : labels) {
      const int count = label_counts[label];
      const int current_axis =
          should_inflate ? strided_shape.size() : inflated_shape.size();
      const int64_t dim = input.dim_size(current_axis);
      strided_shape.push_back(dim);
      inflated_shape.insert(inflated_shape.end(), count, dim);
      const int64_t reshape_dim = MathUtil::IPow(dim, count);
      reshape.push_back(reshape_dim);
      // While taking the d-diagonal in a rank k Tensor, we take d
      // equally-spaced elements including the first and last element. Then, (k
      // - 1) * stride = d^k - 1, or, stride = (d^k - 1)/(d - 1).
      const int64_t stride =
          (dim > 1 && count > 1) ? (reshape_dim - 1) / (dim - 1) : 1;
      strides.push_back(stride);
    }

    TensorShape output_shape =
        TensorShape(should_inflate ? inflated_shape : strided_shape);
    TF_RETURN_IF_ERROR(
        ctx->allocate_temp(DataTypeToEnum<T>::value, output_shape, output));
    const Device& device = ctx->eigen_device<Device>();
    switch (reshape.size()) {
#define NDIMS_CASE(N)                                                 \
  case N: {                                                           \
    if (should_inflate) {                                             \
      auto output_map = output->shaped<T, N>(reshape);                \
      auto input_map = input.shaped<T, N>(strided_shape);             \
      functor::InflateFunctor<Device, T, N>()(                        \
          device, input_map, TensorShape(strides).AsEigenDSizes<N>(), \
          output_map);                                                \
    } else {                                                          \
      auto input_map = input.shaped<T, N>(reshape);                   \
      auto output_map = output->shaped<T, N>(strided_shape);          \
      functor::StrideFunctor<Device, T, N>()(                         \
          device, input_map, TensorShape(strides).AsEigenDSizes<N>(), \
          output_map);                                                \
    }                                                                 \
  } break;
      NDIMS_CASE(1);
      NDIMS_CASE(2);
      NDIMS_CASE(3);
      NDIMS_CASE(4);
      NDIMS_CASE(5);
      NDIMS_CASE(6);
      default:
        return errors::Unimplemented(
            "Unsupported rank: ", reshape.size(),
            " while handling repeated indices. Up to rank 6 is supported.");
#undef NDIMS_CASE
    }
    return Status::OK();
  }

  // Returns true if the input dimensions are already sorted in the order
  // [batch, contract, free, reduce]. Used to implement an optimization to avoid
  // an extra transpose and instead uses (adj_x and adj_y) in BatchMatMul.
  static bool ShouldSwapFreeAndContract(
      const Labels& labels,
      const std::vector<EinsumDimensionType>& label_types) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSlinalgPSeinsum_op_implDTh mht_6(mht_6_v, 507, "", "./tensorflow/core/kernels/linalg/einsum_op_impl.h", "ShouldSwapFreeAndContract");

    // Check that ordering is according to dimension type, with the role of
    // free and contract dimensions swapped.
    gtl::InlinedVector<int, 5> remap = {0, 1, 3, 2, 4};
    for (int i = 0; i + 1 < labels.size(); ++i) {
      const int dimtype_a = remap[label_types[labels[i]]];
      const int dimtype_b = remap[label_types[labels[i + 1]]];
      if (dimtype_a > dimtype_b ||
          (dimtype_a == dimtype_b && labels[i] > labels[i + 1])) {
        return false;
      }
    }
    return true;
  }

  template <typename Device, typename T>
  static Status ReduceOperand(
      OpKernelContext* ctx, const Tensor& input,
      const std::vector<EinsumDimensionType>& label_types,
      const LabelCounts& label_counts, Labels* labels, Labels* free_labels,
      bool* swap_free_and_contract, Tensor* output) {
    // Find the permutation to transpose the input dimensions in the order of
    // EinsumDimensionType; i.e. batch, free, contract and reduce dimensions.
    // This makes it more convenient to invoke Reduce/Contract operations.
    std::vector<int> permutation(input.dims());
    absl::c_iota(permutation, 0);
    Tensor input_transposed;
    // Check if we can avoid the transpose. We need to flip the adj_x (or adj_y)
    // flag during BatchMatMul. This is an extra optimization not necessary for
    // correctness.
    if (ShouldSwapFreeAndContract(*labels, label_types)) {
      *swap_free_and_contract = true;
    } else {
      absl::c_sort(permutation, [&](int i, int j) {
        int label_i = (*labels)[i];
        int label_j = (*labels)[j];
        return std::tie(label_types[label_i], label_i) <
               std::tie(label_types[label_j], label_j);
      });
    }
    // Transpose the input so that EinsumDimensionTypes are in order.
    TF_RETURN_IF_ERROR(TransposeOperand<Device, T>(ctx, input, permutation,
                                                   &input_transposed));
    PermuteLabels(permutation, labels);

    // Take the generalized diagonal for dimensions with repeated axis labels.
    Tensor input_deduped;
    labels->erase(std::unique(labels->begin(), labels->end()), labels->end());
    TF_RETURN_IF_ERROR(
        StrideOrInflate<Device, T>(ctx, input_transposed, *labels, label_counts,
                                   false /* should_inflate */, &input_deduped));

    // Reshape denotes the rank-5 shape [broadcast, batch, free, contract,
    // reduce] where we've compacted the dimensions of each EinsumDimensionType.
    gtl::InlinedVector<int64_t, 5> reshape(5, 1);
    // The output shape is [batch shape] + [free size, contract size]
    // That is, the batch shape is preserved (for broadcasting while
    // contracting) while the free dims and contract dims are compressed to one
    // dimension each.
    TensorShape output_shape;
    for (int label_idx = 0; label_idx < labels->size(); ++label_idx) {
      const int label = labels->at(label_idx);
      int64_t dim = input_deduped.dim_size(label_idx);
      if (label_types[label] == EinsumDimensionType::kBroadcasting ||
          label_types[label] == EinsumDimensionType::kBatch) {
        output_shape.AddDim(dim);
      } else if (label_types[label] == EinsumDimensionType::kFree) {
        free_labels->push_back(label);
      }
      reshape[label_types[label]] *= dim;
    }
    if (*swap_free_and_contract)
      std::swap(reshape[EinsumDimensionType::kFree],
                reshape[EinsumDimensionType::kContract]);
    output_shape.AddDim(reshape[EinsumDimensionType::kFree]);
    output_shape.AddDim(reshape[EinsumDimensionType::kContract]);

    if (reshape[EinsumDimensionType::kReduce] ==
        1) {  // No need to actually reduce.
      return CopyFrom(input_deduped, output_shape, output);
    }
    TF_RETURN_IF_ERROR(
        ctx->allocate_temp(DataTypeToEnum<T>::value, output_shape, output));
    using Reducer = Eigen::internal::SumReducer<T>;
    using Index = typename TTypes<T>::Tensor::Index;
    // Reduce along the last axis (i.e axis 1) of the rank-2 Tensor.
    const int64_t output_size = reshape[kBroadcasting] * reshape[kBatch] *
                                reshape[kFree] * reshape[kContract];
    functor::ReduceFunctor<Device, Reducer>::Reduce(
        ctx, output->shaped<T, 1>({output_size}),
        const_cast<const Tensor&>(input_deduped)
            .shaped<T, 2>({output_size, reshape[kReduce]}),
        Eigen::array<Index, 1>({1}), Reducer());
    return Status::OK();
  }

  // Reshapes a Tensor of shape [b0,b1...bk,N,M] to [prod(b0,b1...bk),N,M].
  static Status ReshapeToRank3(const Tensor& input, int batch_size,
                               Tensor* output) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSlinalgPSeinsum_op_implDTh mht_7(mht_7_v, 608, "", "./tensorflow/core/kernels/linalg/einsum_op_impl.h", "ReshapeToRank3");

    const int rank = input.dims();
    TensorShape output_shape = {batch_size, input.dim_size(rank - 2),
                                input.dim_size(rank - 1)};
    return CopyFrom(input, output_shape, output);
  }

  // Contracts the inputs along the last axis (or the second last if the
  // corresponding value of swap_free_and_contract is true). The batch
  // dimensions are broadcast to the output shape.
  // TODO(anudhyan): BatchMatMul might devolve into a component-wise
  // multiplication when the matrix shape is [1,1]; in this case BatchMatMul
  // functor would be very inefficient. The functor should detect if this is the
  // case and perform componentwise multiplication functor instead.
  template <typename Device, typename T>
  static Status ContractOperands(OpKernelContext* ctx,
                                 absl::Span<const Tensor> inputs,
                                 absl::Span<const bool> swap_free_and_contract,
                                 Tensor* output) {
    if (inputs.size() == 1)
      return CopyFrom(inputs[0], inputs[0].shape(), output);
    MatMulBCast bcast(inputs[0].shape().dim_sizes(),
                      inputs[1].shape().dim_sizes());
    if (!bcast.IsValid()) {
      return errors::InvalidArgument(
          "Invalid broadcasting dimensions: ", inputs[0].shape().DebugString(),
          " vs. ", inputs[1].shape().DebugString());
    }
    Tensor lhs;
    TF_RETURN_IF_ERROR(ReshapeToRank3(inputs[0], bcast.x_batch_size(), &lhs));
    Tensor rhs;
    TF_RETURN_IF_ERROR(ReshapeToRank3(inputs[1], bcast.y_batch_size(), &rhs));
    TensorShape output_shape = bcast.output_batch_shape();
    for (int i = 0; i < inputs.size(); ++i) {
      const int64_t free_axis =
          inputs[i].dims() - (swap_free_and_contract[i] ? 1 : 2);
      output_shape.AddDim(inputs[i].dim_size(free_axis));
    }
    bool trans_x = swap_free_and_contract[0];
    bool trans_y = !swap_free_and_contract[1];
    TF_RETURN_IF_ERROR(
        ctx->allocate_temp(DataTypeToEnum<T>::value, output_shape, output));
    if (lhs.NumElements() == 0 || rhs.NumElements() == 0) {
      functor::SetZeroFunctor<Device, T> set_zero;
      set_zero(ctx->eigen_device<Device>(), output->flat<T>());
      return Status::OK();
    }
    Tensor output_reshaped;
    TF_RETURN_IF_ERROR(
        ReshapeToRank3(*output, bcast.output_batch_size(), &output_reshaped));
    LaunchBatchMatMul<Device, T>::Launch(ctx, lhs, rhs, /*adj_x=*/false,
                                         /*adj_y=*/false, trans_x, trans_y,
                                         bcast, &output_reshaped);
    return Status::OK();
  }
};

template <typename Device, typename T>
class EinsumOp : public OpKernel {
 public:
  explicit EinsumOp(OpKernelConstruction* c) : OpKernel(c) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSlinalgPSeinsum_op_implDTh mht_8(mht_8_v, 671, "", "./tensorflow/core/kernels/linalg/einsum_op_impl.h", "EinsumOp");

    OP_REQUIRES_OK(c, c->GetAttr("equation", &equation_));
    OP_REQUIRES_OK(
        c, ParseEinsumEquation(equation_, &input_labels_, &output_labels_,
                               &label_types_, &input_label_counts_,
                               &output_label_counts_, &input_has_ellipsis_,
                               &output_has_ellipsis_));
  }

  void Compute(OpKernelContext* ctx) override {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSlinalgPSeinsum_op_implDTh mht_9(mht_9_v, 683, "", "./tensorflow/core/kernels/linalg/einsum_op_impl.h", "Compute");

    OpInputList inputs;
    OP_REQUIRES_OK(ctx, ctx->input_list("inputs", &inputs));

    OperandLabels input_labels(input_labels_);
    Labels output_labels(output_labels_);
    std::vector<EinsumDimensionType> label_types(label_types_);
    OperandLabelCounts input_label_counts(input_label_counts_);
    LabelCounts output_label_counts(output_label_counts_);
    LabelToDimSizes label_to_dim_sizes;

    OP_REQUIRES_OK(ctx, EinsumHelper::ProcessDimensions(
                            inputs, input_has_ellipsis_, output_has_ellipsis_,
                            &input_labels, &output_labels, &label_types,
                            &input_label_counts, &output_label_counts,
                            &label_to_dim_sizes));

    // The reduction phase (a) sums across reduction dimensions, (b) takes
    // generalized diagonals, and (c) reshapes it into shape
    //   [(broadcasting) batch shape] + [F,C]
    // where F and C denote the total (compacted) size of free and contract
    // dimensions, respectively.
    const int num_inputs = inputs.size();
    OperandLabels free_labels(num_inputs);
    gtl::InlinedVector<Tensor, 2> inputs_reduced(num_inputs);
    gtl::InlinedVector<bool, 2> swap_free_and_contract(num_inputs);
    for (int i = 0; i < num_inputs; ++i) {
      OP_REQUIRES_OK(ctx,
                     EinsumHelper::ReduceOperand<Device, T>(
                         ctx, inputs[i], label_types, input_label_counts[i],
                         &input_labels[i], &free_labels[i],
                         &swap_free_and_contract[i], &inputs_reduced[i]));
    }

    // After reduction, the inputs should be reshaped to Tensors suitable for
    // contraction. If num_inputs is 1, the reduced input is simply forwarded to
    // the output.
    Tensor contraction_output_reshaped;
    OP_REQUIRES_OK(ctx, EinsumHelper::ContractOperands<Device, T>(
                            ctx, inputs_reduced, swap_free_and_contract,
                            &contraction_output_reshaped));

    // Copy the batch labels from the contraction output. Recover the batch
    // shape, which may have been broadcasted.
    TensorShape result_shape = contraction_output_reshaped.shape();
    result_shape.RemoveLastDims(2);

    int num_labels = label_types.size();
    Labels result_labels;
    // All batch dimensions should be present in the contracted result. First
    // the broadcasting dimensions, then the named batch dimensions.
    for (int label = 0; label < num_labels; ++label) {
      if (label_types[label] == EinsumDimensionType::kBroadcasting)
        result_labels.push_back(label);
    }
    for (int label = 0; label < num_labels; ++label) {
      if (label_types[label] == EinsumDimensionType::kBatch)
        result_labels.push_back(label);
    }
    for (int i = 0; i < num_inputs; ++i) {
      for (int label : free_labels[i]) {
        result_labels.push_back(label);
        result_shape.AddDim(label_to_dim_sizes[label]);
      }
    }

    // Reshape the contraction (or reduction) result to its expanded shape:
    // [(broadcasted) batch shape] + [free shape 0] + [free shape 1].
    Tensor contraction_output;
    OP_REQUIRES_OK(
        ctx, EinsumHelper::CopyFrom(contraction_output_reshaped, result_shape,
                                    &contraction_output));

    // Inflate the output if necessary. (E.g. for the equation 'i->iii' which
    // may arise while computing gradient of a regular Einsum).
    // TODO(anudhyan): It's possible that Eigen's contract and inflate can be
    // chained here to avoid materializing an intermediate.
    Tensor output_inflated;
    OP_REQUIRES_OK(
        ctx, EinsumHelper::StrideOrInflate<Device, T>(
                 ctx, contraction_output, result_labels, output_label_counts,
                 true /* should_inflate */, &output_inflated));
    if (output_inflated.dims() > contraction_output.dims()) {
      // We inflated the output. Modify result labels accordingly.
      Labels inflated_labels;
      for (int label : result_labels) {
        inflated_labels.insert(inflated_labels.end(),
                               output_label_counts[label], label);
      }
      result_labels.swap(inflated_labels);
    }
    // Find the permutation to map the result labels to the output labels. Note
    // that both the result and the final output may have the repeated labels,
    // in which case the permutation preserves the left-to-right ordering.
    // E.g. if result labels are [0, 0, 1] and output is [0, l, 0] then the
    // permutation should be [0, 2, 1]. We also use the fact that repeated
    // labels in the result are adjacent to each other.
    std::vector<int> output_permutation(output_labels.size());
    std::vector<int> label_to_position(num_labels, -1);
    for (int i = 0; i < result_labels.size(); ++i) {
      // Remember the position of only the leftmost result label.
      if (label_to_position[result_labels[i]] == -1) {
        label_to_position[result_labels[i]] = i;
      }
    }
    for (int i = 0; i < output_labels.size(); ++i) {
      output_permutation[i] = label_to_position[output_labels[i]];
      // We have found the leftmost occurrence. The next one would be adjacent.
      label_to_position[output_labels[i]] += 1;
    }
    Tensor output;
    OP_REQUIRES_OK(ctx, EinsumHelper::TransposeOperand<Device, T>(
                            ctx, output_inflated, output_permutation, &output));
    ctx->set_output(0, output);
  }

  string TraceString(const OpKernelContext& ctx, bool verbose) const override {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSlinalgPSeinsum_op_implDTh mht_10(mht_10_v, 802, "", "./tensorflow/core/kernels/linalg/einsum_op_impl.h", "TraceString");

    string op = profiler::TraceMeOp(name_view(), type_string_view());
    string equation = strings::StrCat("(", equation_, ")");
    if (verbose) {
      string shape = ShapeTraceString(ctx);
      if (!shape.empty()) {
        return profiler::TraceMeEncode(
            std::move(op), {{"equation", equation}, {"shape", shape}});
      }
    }
    return profiler::TraceMeEncode(std::move(op), {{"equation", equation}});
  }

 private:
  string equation_;
  OperandLabels input_labels_;
  Labels output_labels_;
  std::vector<EinsumDimensionType> label_types_;
  OperandLabelCounts input_label_counts_;
  LabelCounts output_label_counts_;
  gtl::InlinedVector<bool, 2> input_has_ellipsis_;
  bool output_has_ellipsis_ = false;
};

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
// Forward declarations of the functor specializations for GPU.
namespace functor {
#define DECLARE_GPU_SPEC(T, N)                                      \
  template <>                                                       \
  void StrideFunctor<GPUDevice, T, N>::operator()(                  \
      const GPUDevice& d, typename TTypes<T, N>::ConstTensor input, \
      const Eigen::DSizes<Eigen::DenseIndex, N>& strides,           \
      typename TTypes<T, N>::Tensor output);                        \
  extern template struct StrideFunctor<GPUDevice, T, N>;            \
  template <>                                                       \
  void InflateFunctor<GPUDevice, T, N>::operator()(                 \
      const GPUDevice& d, typename TTypes<T, N>::ConstTensor input, \
      const Eigen::DSizes<Eigen::DenseIndex, N>& strides,           \
      typename TTypes<T, N>::Tensor output);                        \
  extern template struct InflateFunctor<GPUDevice, T, N>;

#define DECLARE_GPU_SPECS(T) \
  DECLARE_GPU_SPEC(T, 1);    \
  DECLARE_GPU_SPEC(T, 2);    \
  DECLARE_GPU_SPEC(T, 3);    \
  DECLARE_GPU_SPEC(T, 4);    \
  DECLARE_GPU_SPEC(T, 5);    \
  DECLARE_GPU_SPEC(T, 6);

DECLARE_GPU_SPECS(Eigen::half);
DECLARE_GPU_SPECS(double);
DECLARE_GPU_SPECS(float);
// TODO(rocm): Enable once complex types are supported.
#if GOOGLE_CUDA
DECLARE_GPU_SPECS(complex64);
DECLARE_GPU_SPECS(complex128);
#endif
#undef DECLARE_GPU_SPEC
#undef DECLARE_GPU_SPECS
}  // namespace functor
#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_KERNELS_LINALG_EINSUM_OP_IMPL_H_
