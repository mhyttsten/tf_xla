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
class MHTracer_DTPStensorflowPScorePSkernelsPSset_kernelsDTcc {
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
   MHTracer_DTPStensorflowPScorePSkernelsPSset_kernelsDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSkernelsPSset_kernelsDTcc() {
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

/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

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

// Ops for operating with sets. They are not checked in
// to TensorFlow because we would first like to demonstrate successful
// end-to-end use of these ops in eval and polish the api a bit like taking two
// SparseTensor rather than on edense and one sparse.

#define EIGEN_USE_THREADS

#include <algorithm>
#include <numeric>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/btree_set.h"
#include "absl/container/flat_hash_set.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_util.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/util/sparse/sparse_tensor.h"

namespace tensorflow {

using ShapeArray = sparse::SparseTensor::ShapeArray;
using VarDimArray = sparse::SparseTensor::VarDimArray;

// Validate rank >= 2.
void CheckRankAtLeast2(OpKernelContext* ctx, const TensorShape& shape) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSset_kernelsDTcc mht_0(mht_0_v, 217, "", "./tensorflow/core/kernels/set_kernels.cc", "CheckRankAtLeast2");

  const auto rank = shape.dims();
  OP_REQUIRES(ctx, rank >= 2,
              errors::InvalidArgument("Invalid rank ", rank, "."));
}

// Return group shape, which is the 1st n-1 dimensions of shape.
Status GroupShape(const VarDimArray& input_shape, ShapeArray* grouped_shape) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSset_kernelsDTcc mht_1(mht_1_v, 227, "", "./tensorflow/core/kernels/set_kernels.cc", "GroupShape");

  if (input_shape.size() < 2) {
    // TODO(irving): Why can't 2 be 1 here?
    return errors::InvalidArgument("Shape [", absl::StrJoin(input_shape, ","),
                                   "] has rank ", input_shape.size(), " < 2");
  }
  // grouped_shape is input_shape[:-1]
  *grouped_shape = ShapeArray(input_shape.begin(), input_shape.end() - 1);
  return Status::OK();
}

// Build `SparseTensor` from indices, values, and shape in inputs
// [base_index, base_index + 3), and validate its rank and indices.
Status SparseTensorFromContext(OpKernelContext* ctx, const int32_t base_index,
                               const bool validate_indices,
                               sparse::SparseTensor* tensor) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSset_kernelsDTcc mht_2(mht_2_v, 245, "", "./tensorflow/core/kernels/set_kernels.cc", "SparseTensorFromContext");

  // Assume row-major order.
  TensorShape shape;
  TF_RETURN_IF_ERROR(TensorShape::BuildTensorShape(
      ctx->input(base_index + 2).vec<int64_t>(), &shape));
  CheckRankAtLeast2(ctx, shape);
  std::vector<int64_t> order(shape.dims());
  std::iota(order.begin(), order.end(), 0);

  Status status = sparse::SparseTensor::Create(
      ctx->input(base_index), ctx->input(base_index + 1), shape, order, tensor);

  if (!validate_indices || !status.ok()) return status;
  return tensor->IndicesValid();
}

// TODO(ptucker): CheckGroup is just a sanity check on the result of
// SparseTensor.group, consider removing.
// `sparse_tensor_shape` is the shape of the `SparseTensor` from which group
// was created, and is used to sanity check the indices in `group'.
template <typename T>
void CheckGroup(OpKernelContext* ctx, const sparse::Group& group,
                const VarDimArray& sparse_tensor_shape) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSset_kernelsDTcc mht_3(mht_3_v, 270, "", "./tensorflow/core/kernels/set_kernels.cc", "CheckGroup");

  const auto& indices = group.indices();
  const auto& values = group.values<T>();

  // Sanity check: group is non-empty, and indices and values are same size.
  const auto num_values = values.dimension(0);
  OP_REQUIRES(ctx, indices.size() > 0, errors::Internal("Empty group."));
  OP_REQUIRES(
      ctx, indices.dimension(0) == num_values,
      errors::Internal("shape[0] of group indices ", indices.dimension(0),
                       " != values ", num_values, "."));

  // Sanity check: valid indices.
  const auto group_rank = indices.dimension(1);
  const auto expected_rank = sparse_tensor_shape.size();
  OP_REQUIRES(ctx, expected_rank == group_rank,
              errors::Internal("Rank expected ", expected_rank, ", got ",
                               group_rank, "."));
  for (int32_t j = 0; j < expected_rank; ++j) {
    const auto dim_size = sparse_tensor_shape[j];
    OP_REQUIRES(
        ctx, dim_size > 0,
        errors::Internal("Invalid dim_size[", j, "] = ", dim_size, "."));
    for (int64_t i = 0; i < num_values; ++i) {
      const auto index = indices(i, j);
      OP_REQUIRES(ctx, dim_size > index,
                  errors::Internal("indices[", i, ", ", j, "] expected < ",
                                   dim_size, ", got ", index, "."));
    }
  }
}

// This lets us calculate the row-major index into flattened output.
const ShapeArray Strides(const VarDimArray& shape) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSset_kernelsDTcc mht_4(mht_4_v, 306, "", "./tensorflow/core/kernels/set_kernels.cc", "Strides");

  ShapeArray result(shape.size());
  int64_t product = 1;
  for (int i = shape.size() - 1; i >= 0; --i) {
    result[i] = product;
    product *= shape[i];
  }
  return result;
}

// TODO(ptucker): If memory becomes an issue, consider a 2-pass approach to
// eliminate the intermediate `values` data structure - iterate once to
// determine `num_values`, allocate output tensors, then write results directly
// to output tensors.

// TODO(ptucker): Consider sharding work across multiple threads. See
// SparseCrossOp for an example.

// Output `SparseTensor` of shape `output_shape`. `sets` contains pairs of
// group indices (i.e., values for all but the last dimension of `output_shape`)
// and set values, each of which will occupy the last dimension of
// `output_shape`. `sets` should be sorted in ascending order by group indices.
template <typename T>
void OutputSparseTensor(
    OpKernelContext* ctx, const TensorShape& output_shape,
    const int64_t num_values,
    const std::vector<std::pair<std::vector<int64_t>, absl::btree_set<T>>>&
        sets) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSset_kernelsDTcc mht_5(mht_5_v, 336, "", "./tensorflow/core/kernels/set_kernels.cc", "OutputSparseTensor");

  // Allocate 3 output tensors for sparse data.
  Tensor *out_indices_t, *out_values_t, *out_shape_t;
  OP_REQUIRES_OK(ctx, ctx->allocate_output(
                          0, TensorShape({num_values, output_shape.dims()}),
                          &out_indices_t));
  OP_REQUIRES_OK(
      ctx, ctx->allocate_output(1, TensorShape({num_values}), &out_values_t));
  OP_REQUIRES_OK(ctx, ctx->allocate_output(
                          2, TensorShape({output_shape.dims()}), &out_shape_t));
  auto out_indices_mat = out_indices_t->matrix<int64_t>();
  auto out_values_flat = out_values_t->vec<T>();

  // For each set, write its indices and values to output tensors.
  int64_t value_index = 0;
  for (auto it = sets.begin(); it != sets.end(); ++it) {
    const auto& group_indices = it->first;
    OP_REQUIRES(
        ctx, group_indices.size() == output_shape.dims() - 1,
        errors::Internal("Invalid number of indices ", group_indices.size(),
                         ", expected ", output_shape.dims() - 1, "."));
    const auto& set = it->second;

    // For each set item, write its indices and value to output tensors.
    int64_t group_value_index = 0;
    for (auto value = set.begin(); value != set.end();
         ++value, ++value_index, ++group_value_index) {
      // First n-1 dimensions are the group, last dimension is the position in
      // the set.
      for (int32_t i = 0; i < group_indices.size(); ++i) {
        out_indices_mat(value_index, i) = group_indices[i];
      }
      out_indices_mat(value_index, group_indices.size()) = group_value_index;

      out_values_flat(value_index) = *value;
    }
  }

  // Write output shape.
  auto out_shape_flat = out_shape_t->vec<int64_t>();
  for (int32_t i = 0; i < output_shape.dims(); ++i) {
    out_shape_flat(i) = output_shape.dim_size(i);
  }
}

bool ValidateIndicesFromContext(OpKernelConstruction* ctx) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSset_kernelsDTcc mht_6(mht_6_v, 384, "", "./tensorflow/core/kernels/set_kernels.cc", "ValidateIndicesFromContext");

  bool result;
  if (ctx->GetAttr("validate_indices", &result).ok()) {
    return result;
  }
  return true;
}

// Populate `result` set from group in `tensor`. "Group" is defined by
// `group_indices`, which are values for the first n-1 dimensions of
// `input_tensor`. `input_strides` is provided to avoid recalculating it
// multiple times, and is used to calculate the flat index into `input_tensor`
// values.
template <typename T>
void PopulateFromDenseGroup(OpKernelContext* ctx, const Tensor& input_tensor,
                            const VarDimArray& input_strides,
                            const std::vector<int64_t>& group_indices,
                            absl::flat_hash_set<T>* result) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSset_kernelsDTcc mht_7(mht_7_v, 404, "", "./tensorflow/core/kernels/set_kernels.cc", "PopulateFromDenseGroup");

  OP_REQUIRES(ctx, group_indices.size() == input_strides.size() - 1,
              errors::Internal("group_indices.size ", group_indices.size(),
                               ", !=  input_strides.size-1 ",
                               input_strides.size() - 1, "."));
  result->clear();
  auto input_flat = input_tensor.flat<T>();
  const auto start = std::inner_product(
      group_indices.begin(), group_indices.end(), input_strides.begin(), 0LL);
  const TensorShape& input_shape = input_tensor.shape();
  const auto end = start + input_shape.dim_size(input_shape.dims() - 1);
  for (int64_t i = start; i < end; ++i) {
    result->insert(input_flat(i));
  }
}

// Populate `result` set from `group`. `sparse_tensor_shape` is the shape of the
// `SparseTensor` from which group was created, and is used to sanity check the
// indices in `group'.
template <typename T>
void PopulateFromSparseGroup(OpKernelContext* ctx, const sparse::Group& group,
                             const VarDimArray& sparse_tensor_shape,
                             absl::flat_hash_set<T>* result) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSset_kernelsDTcc mht_8(mht_8_v, 429, "", "./tensorflow/core/kernels/set_kernels.cc", "PopulateFromSparseGroup");

  CheckGroup<T>(ctx, group, sparse_tensor_shape);
  result->clear();
  const auto& group_values = group.values<T>();
  for (int64_t i = 0; i < group_values.size(); ++i) {
    result->insert(group_values(i));
  }
}

template <typename T>
class SetSizeOp : public OpKernel {
 public:
  explicit SetSizeOp(OpKernelConstruction* ctx)
      : OpKernel(ctx), validate_indices_(ValidateIndicesFromContext(ctx)) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSset_kernelsDTcc mht_9(mht_9_v, 445, "", "./tensorflow/core/kernels/set_kernels.cc", "SetSizeOp");
}

  void Compute(OpKernelContext* ctx) override;

 private:
  const bool validate_indices_;
};

template <typename T>
void SetSizeOp<T>::Compute(OpKernelContext* ctx) {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSset_kernelsDTcc mht_10(mht_10_v, 457, "", "./tensorflow/core/kernels/set_kernels.cc", "SetSizeOp<T>::Compute");

  sparse::SparseTensor set_st;
  OP_REQUIRES_OK(ctx,
                 SparseTensorFromContext(ctx, 0, validate_indices_, &set_st));

  // Output shape is same as input except for last dimension, which reduces
  // to the set size of values along that dimension.
  ShapeArray output_shape;
  OP_REQUIRES_OK(ctx, GroupShape(set_st.shape(), &output_shape));
  const auto output_strides = Strides(output_shape);

  TensorShape output_shape_ts;
  OP_REQUIRES_OK(ctx,
                 TensorShapeUtils::MakeShape(output_shape, &output_shape_ts));
  Tensor* out_t;
  OP_REQUIRES_OK(ctx, ctx->allocate_output(0, output_shape_ts, &out_t));
  auto out = out_t->flat<int32>();
  out.device(ctx->eigen_cpu_device()) = out.constant(static_cast<int32>(0.0));

  // Group by all but last dimension, create a set of group values, and add set
  // size to output.
  VarDimArray group_ix = set_st.order().subspan(0, set_st.order().size() - 1);
  absl::flat_hash_set<T> group_set;
  for (const auto& group : set_st.group(group_ix)) {
    PopulateFromSparseGroup<T>(ctx, group, set_st.shape(), &group_set);

    const auto group_key = group.group();
    const auto output_index = std::inner_product(
        group_key.begin(), group_key.end(), output_strides.begin(), 0LL);
    out(output_index) = group_set.size();
  }
}

#define _SET_SIZE_REGISTER_KERNEL_BUILDER(T)                     \
  REGISTER_KERNEL_BUILDER(                                       \
      Name("SetSize").Device(DEVICE_CPU).TypeConstraint<T>("T"), \
      SetSizeOp<T>);
_SET_SIZE_REGISTER_KERNEL_BUILDER(int8);
_SET_SIZE_REGISTER_KERNEL_BUILDER(int16);
_SET_SIZE_REGISTER_KERNEL_BUILDER(int32);
_SET_SIZE_REGISTER_KERNEL_BUILDER(int64_t);
_SET_SIZE_REGISTER_KERNEL_BUILDER(uint8);
_SET_SIZE_REGISTER_KERNEL_BUILDER(uint16);
_SET_SIZE_REGISTER_KERNEL_BUILDER(tstring);
#undef _SET_SIZE_REGISTER_KERNEL_BUILDER

enum InputTypes {
  DENSE_DENSE = 0,
  DENSE_SPARSE = 1,
  SPARSE_SPARSE = 2,
};

enum SetOperation { A_MINUS_B = 0, B_MINUS_A = 1, INTERSECTION = 2, UNION = 3 };

SetOperation SetOperationFromContext(OpKernelConstruction* ctx) {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSset_kernelsDTcc mht_11(mht_11_v, 514, "", "./tensorflow/core/kernels/set_kernels.cc", "SetOperationFromContext");

  string set_operation_str;
  if (!ctx->GetAttr("set_operation", &set_operation_str).ok()) {
    ctx->CtxFailure(errors::InvalidArgument("Missing set_operation."));
  } else {
    std::transform(set_operation_str.begin(), set_operation_str.end(),
                   set_operation_str.begin(), ::tolower);
    if ("a-b" == set_operation_str) {
      return A_MINUS_B;
    }
    if ("b-a" == set_operation_str) {
      return B_MINUS_A;
    }
    if ("intersection" == set_operation_str) {
      return INTERSECTION;
    }
    if ("union" != set_operation_str) {
      ctx->CtxFailure(errors::InvalidArgument("Invalid set_operation ",
                                              set_operation_str, "."));
    }
  }
  // NOTE: This is not the default, this function fails if no 'set_operation'
  // attribute is provided.
  return UNION;
}

// Abstract base class for performing set operations across the last dimension
// of 2 input tensors.
template <typename T>
class SetOperationOp : public OpKernel {
 public:
  SetOperationOp(OpKernelConstruction* ctx, InputTypes input_types)
      : OpKernel(ctx),
        set_operation_(SetOperationFromContext(ctx)),
        validate_indices_(ValidateIndicesFromContext(ctx)),
        input_types_(input_types) {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSset_kernelsDTcc mht_12(mht_12_v, 552, "", "./tensorflow/core/kernels/set_kernels.cc", "SetOperationOp");
}

  void Compute(OpKernelContext* ctx) override;

 private:
  void ApplySetOperation(const absl::flat_hash_set<T>& set1,
                         const absl::flat_hash_set<T>& set2,
                         absl::btree_set<T>* result) const;
  void ComputeDenseToDense(OpKernelContext* ctx) const;
  void ComputeDenseToSparse(OpKernelContext* ctx) const;
  void ComputeSparseToSparse(OpKernelContext* ctx) const;
  const SetOperation set_operation_;
  const bool validate_indices_;
  const InputTypes input_types_;
};

template <typename T>
void SetDifference(const absl::flat_hash_set<T>& set1,
                   const absl::flat_hash_set<T>& set2,
                   absl::btree_set<T>* result) {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSset_kernelsDTcc mht_13(mht_13_v, 574, "", "./tensorflow/core/kernels/set_kernels.cc", "SetDifference");

  for (const T& elem : set1) {
    if (!set2.contains(elem)) result->insert(elem);
  }
}

template <typename T>
void SetIntersection(const absl::flat_hash_set<T>& set1,
                     const absl::flat_hash_set<T>& set2,
                     absl::btree_set<T>* result) {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSset_kernelsDTcc mht_14(mht_14_v, 586, "", "./tensorflow/core/kernels/set_kernels.cc", "SetIntersection");

  if (set1.size() <= set2.size()) {
    for (const T& elem : set1) {
      if (set2.contains(elem)) result->insert(elem);
    }
  } else {
    for (const T& elem : set2) {
      if (set1.contains(elem)) result->insert(elem);
    }
  }
}

template <typename T>
void SetUnion(const absl::flat_hash_set<T>& set1,
              const absl::flat_hash_set<T>& set2, absl::btree_set<T>* result) {
   std::vector<std::string> mht_15_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSset_kernelsDTcc mht_15(mht_15_v, 603, "", "./tensorflow/core/kernels/set_kernels.cc", "SetUnion");

  result->insert(set1.begin(), set1.end());
  result->insert(set2.begin(), set2.end());
}

template <typename T>
void SetOperationOp<T>::ApplySetOperation(const absl::flat_hash_set<T>& set1,
                                          const absl::flat_hash_set<T>& set2,
                                          absl::btree_set<T>* result) const {
   std::vector<std::string> mht_16_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSset_kernelsDTcc mht_16(mht_16_v, 614, "", "./tensorflow/core/kernels/set_kernels.cc", "SetOperationOp<T>::ApplySetOperation");

  switch (set_operation_) {
    case A_MINUS_B:
      SetDifference<T>(set1, set2, result);
      break;
    case B_MINUS_A:
      SetDifference<T>(set2, set1, result);
      break;
    case INTERSECTION:
      SetIntersection<T>(set1, set2, result);
      break;
    case UNION:
      SetUnion<T>(set1, set2, result);
      break;
  }
}

// Validate shapes have the same dimensions.
Status CheckShapesMatch(VarDimArray shape1, VarDimArray shape2) {
   std::vector<std::string> mht_17_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSset_kernelsDTcc mht_17(mht_17_v, 635, "", "./tensorflow/core/kernels/set_kernels.cc", "CheckShapesMatch");

  if (shape1 != shape2) {
    return errors::InvalidArgument("Mismatched shapes [",
                                   absl::StrJoin(shape1, ","), "] vs [",
                                   absl::StrJoin(shape2, ","), "]");
  }
  return Status::OK();
}

// Validate ranks are the same, and all but last dimension are the same.
// Return GroupShape.
Status GroupShapeFromInputs(VarDimArray shape1, VarDimArray shape2,
                            ShapeArray* group_shape) {
   std::vector<std::string> mht_18_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSset_kernelsDTcc mht_18(mht_18_v, 650, "", "./tensorflow/core/kernels/set_kernels.cc", "GroupShapeFromInputs");

  ShapeArray group_shape_1;
  TF_RETURN_IF_ERROR(GroupShape(shape1, &group_shape_1));
  ShapeArray group_shape_2;
  TF_RETURN_IF_ERROR(GroupShape(shape2, &group_shape_2));
  TF_RETURN_IF_ERROR(CheckShapesMatch(group_shape_1, group_shape_2));
  *group_shape = group_shape_1;
  return Status::OK();
}

// Split `flat_group_index` into separate dimensions based on `group_shape`.
void PopulateGroupIndices(const int64_t flat_group_index,
                          VarDimArray group_shape,
                          std::vector<int64_t>* group_indices) {
   std::vector<std::string> mht_19_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSset_kernelsDTcc mht_19(mht_19_v, 666, "", "./tensorflow/core/kernels/set_kernels.cc", "PopulateGroupIndices");

  group_indices->clear();
  int64_t running_flat_group_index = flat_group_index;
  for (int group_dim_index = group_shape.size() - 1; group_dim_index >= 0;
       --group_dim_index) {
    const auto group_dim = group_shape[group_dim_index];
    group_indices->insert(group_indices->begin(),
                          running_flat_group_index % group_dim);
    running_flat_group_index /= group_dim;
  }
}

ShapeArray TensorShapeToArray(const TensorShape& t) {
   std::vector<std::string> mht_20_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSset_kernelsDTcc mht_20(mht_20_v, 681, "", "./tensorflow/core/kernels/set_kernels.cc", "TensorShapeToArray");

  ShapeArray vec(t.dims());
  for (int i = 0; i < t.dims(); ++i) vec[i] = t.dim_size(i);
  return vec;
}

// `ctx` contains set1 and set2 dense tensors.
// Iterate over groups in set1 and set2, applying `ApplySetOperation` to each,
// and outputting the result `SparseTensor`. A "group" is a collection of values
// with the same first n-1 dimensions in set1 and set2.
template <typename T>
void SetOperationOp<T>::ComputeDenseToDense(OpKernelContext* ctx) const {
   std::vector<std::string> mht_21_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSset_kernelsDTcc mht_21(mht_21_v, 695, "", "./tensorflow/core/kernels/set_kernels.cc", "SetOperationOp<T>::ComputeDenseToDense");

  const Tensor& set1_t = ctx->input(0);
  const Tensor& set2_t = ctx->input(1);
  // The following should stay in sync with `_dense_to_dense_shape` shape
  // assertions in python/ops/set_ops.py, and `SetShapeFn` for
  // `DenseToDenseSetOperation` in ops/set_ops.cc.
  ShapeArray group_shape;
  const auto shape1 = TensorShapeToArray(set1_t.shape());
  const auto shape2 = TensorShapeToArray(set2_t.shape());
  OP_REQUIRES_OK(ctx, GroupShapeFromInputs(shape1, shape2, &group_shape));

  const auto set1_strides = Strides(shape1);
  const auto set2_strides = Strides(shape2);

  std::vector<std::pair<std::vector<int64_t>, absl::btree_set<T>>> group_sets;
  int64_t num_result_values = 0;
  int64_t max_set_size = 0;

  absl::flat_hash_set<T> set1_group_set;
  absl::flat_hash_set<T> set2_group_set;
  std::vector<int64_t> group_indices;
  int64_t num_elements;
  OP_REQUIRES_OK(ctx,
                 TensorShapeUtils::NumElements(group_shape, &num_elements));
  for (int64_t flat_group_index = 0; flat_group_index < num_elements;
       ++flat_group_index) {
    PopulateGroupIndices(flat_group_index, group_shape, &group_indices);
    PopulateFromDenseGroup<T>(ctx, set1_t, set1_strides, group_indices,
                              &set1_group_set);
    PopulateFromDenseGroup<T>(ctx, set2_t, set2_strides, group_indices,
                              &set2_group_set);

    absl::btree_set<T> group_set;
    ApplySetOperation(set1_group_set, set2_group_set, &group_set);
    if (!group_set.empty()) {
      const auto set_size = group_set.size();
      if (set_size > max_set_size) {
        max_set_size = set_size;
      }
      num_result_values += set_size;
      group_sets.push_back({group_indices, std::move(group_set)});
    }
  }

  TensorShape output_shape;
  OP_REQUIRES_OK(ctx, TensorShapeUtils::MakeShape(group_shape, &output_shape));
  output_shape.AddDim(max_set_size);
  OutputSparseTensor<T>(ctx, output_shape, num_result_values, group_sets);
}

// `ctx` contains dense set1 and sparse set2 tensors.
// Iterate over groups in set1 and set2, applying `ApplySetOperation` to each,
// and outputing the result `SparseTensor`. A "group" is a collection of values
// with the same first n-1 dimensions in set1 and set2.
template <typename T>
void SetOperationOp<T>::ComputeDenseToSparse(OpKernelContext* ctx) const {
   std::vector<std::string> mht_22_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSset_kernelsDTcc mht_22(mht_22_v, 753, "", "./tensorflow/core/kernels/set_kernels.cc", "SetOperationOp<T>::ComputeDenseToSparse");

  const Tensor& set1_t = ctx->input(0);
  sparse::SparseTensor set2_st;
  OP_REQUIRES_OK(ctx,
                 SparseTensorFromContext(ctx, 1, validate_indices_, &set2_st));
  // The following should stay in sync with `_dense_to_sparse_shape` shape
  // assertions in python/ops/set_ops.py, and `SetShapeFn` for
  // `DenseToSparseSetOperation` in ops/set_ops.cc.
  ShapeArray group_shape;
  OP_REQUIRES_OK(ctx, GroupShapeFromInputs(TensorShapeToArray(set1_t.shape()),
                                           set2_st.shape(), &group_shape));

  const ShapeArray set1_strides = Strides(TensorShapeToArray(set1_t.shape()));

  std::vector<std::pair<std::vector<int64_t>, absl::btree_set<T>>> group_sets;
  int64_t num_result_values = 0;
  int64_t max_set_size = 0;

  absl::flat_hash_set<T> set1_group_set;
  absl::flat_hash_set<T> set2_group_set;
  auto set2_grouper =
      set2_st.group(set2_st.order().subspan(0, set2_st.order().size() - 1));
  auto set2_group_it = set2_grouper.begin();
  std::vector<int64_t> group_indices;
  int64_t num_elements;
  OP_REQUIRES_OK(ctx,
                 TensorShapeUtils::NumElements(group_shape, &num_elements));
  for (int64_t flat_group_index = 0; flat_group_index < num_elements;
       ++flat_group_index) {
    PopulateGroupIndices(flat_group_index, group_shape, &group_indices);

    // Get values from set1.
    PopulateFromDenseGroup<T>(ctx, set1_t, set1_strides, group_indices,
                              &set1_group_set);

    // Get values from set2, if applicable.
    set2_group_set.clear();
    if (set2_group_it != set2_grouper.end()) {
      const auto& group = *set2_group_it;
      const auto set2_group_indices = group.group();
      OP_REQUIRES(
          ctx, set2_group_indices.size() == group_indices.size(),
          errors::InvalidArgument("Invalid number of group indices ",
                                  set2_group_indices.size(), ", expected ",
                                  group_indices.size(), "."));
      bool group_match = true;
      for (int32_t i = 0; group_match && (i < set2_group_indices.size()); ++i) {
        if (set2_group_indices[i] != group_indices[i]) {
          group_match = false;
        }
      }
      if (group_match) {
        PopulateFromSparseGroup<T>(ctx, group, set2_st.shape(),
                                   &set2_group_set);
        ++set2_group_it;
      }
    }

    absl::btree_set<T> group_set;
    ApplySetOperation(set1_group_set, set2_group_set, &group_set);
    if (!group_set.empty()) {
      const auto set_size = group_set.size();
      if (set_size > max_set_size) {
        max_set_size = set_size;
      }
      num_result_values += set_size;
      group_sets.push_back({group_indices, std::move(group_set)});
    }
  }

  TensorShape output_shape;
  OP_REQUIRES_OK(ctx, TensorShapeUtils::MakeShape(group_shape, &output_shape));
  output_shape.AddDim(max_set_size);
  OutputSparseTensor<T>(ctx, output_shape, num_result_values, group_sets);
}

// This is used to determine which group iterator is less than the other, based
// on row-major ordering of indices.
// An empty index list indicates end of iteration, which is interpreted as "max"
// for the purposes of comparison; i.e., non-empty < empty.
// Return 0 if both groups are empty, or both non-empty with the same values.
// Return <0 if set1 <= set2, or set2 is empty.
// Return >0 if set2 <= set1, or set1 is empty.
void CompareGroups(OpKernelContext* ctx,
                   const std::vector<int64_t>& set1_group_indices,
                   const std::vector<int64_t>& set2_group_indices,
                   int64_t* result) {
   std::vector<std::string> mht_23_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSset_kernelsDTcc mht_23(mht_23_v, 842, "", "./tensorflow/core/kernels/set_kernels.cc", "CompareGroups");

  if (set1_group_indices.empty()) {
    *result = set2_group_indices.empty() ? 0 : 1;
    return;
  }
  if (set2_group_indices.empty()) {
    *result = set1_group_indices.empty() ? 0 : -1;
    return;
  }
  OP_REQUIRES(ctx, set1_group_indices.size() == set2_group_indices.size(),
              errors::InvalidArgument("Mismatched group dims ",
                                      set1_group_indices.size(), " vs ",
                                      set2_group_indices.size(), "."));
  for (int32_t i = 0; i < set1_group_indices.size(); ++i) {
    *result = set1_group_indices[i] - set2_group_indices[i];
    if (*result != 0) {
      return;
    }
  }
}

// `ctx` contains set1 and set2 sparse tensors.
// Iterate over groups in set1 and set2, applying `ApplySetOperation` to each,
// and outputing the result `SparseTensor`. A "group" is a collection of values
// with the same first n-1 dimensions in set1 and set2.
template <typename T>
void SetOperationOp<T>::ComputeSparseToSparse(OpKernelContext* ctx) const {
   std::vector<std::string> mht_24_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSset_kernelsDTcc mht_24(mht_24_v, 871, "", "./tensorflow/core/kernels/set_kernels.cc", "SetOperationOp<T>::ComputeSparseToSparse");

  sparse::SparseTensor set1_st;
  OP_REQUIRES_OK(ctx,
                 SparseTensorFromContext(ctx, 0, validate_indices_, &set1_st));

  sparse::SparseTensor set2_st;
  OP_REQUIRES_OK(ctx,
                 SparseTensorFromContext(ctx, 3, validate_indices_, &set2_st));

  // The following should stay in sync with `_sparse_to_sparse_shape` shape
  // assertions in python/ops/set_ops.py, and `SetShapeFn` for
  // `SparseToSparseSetOperation` in ops/set_ops.cc.
  ShapeArray group_shape;
  OP_REQUIRES_OK(ctx, GroupShapeFromInputs(set1_st.shape(), set2_st.shape(),
                                           &group_shape));

  std::vector<std::pair<std::vector<int64_t>, absl::btree_set<T>>> group_sets;
  int64_t num_result_values = 0;
  int64_t max_set_size = 0;

  absl::flat_hash_set<T> set1_group_set;
  absl::flat_hash_set<T> set2_group_set;
  auto set1_grouper =
      set1_st.group(set1_st.order().subspan(0, set1_st.order().size() - 1));
  auto set1_group_it = set1_grouper.begin();
  auto set2_grouper =
      set2_st.group(set2_st.order().subspan(0, set2_st.order().size() - 1));
  auto set2_group_it = set2_grouper.begin();

  // Empty indices vector represents iteration end in `CompareGroups`.
  const std::vector<int64_t> group_iter_end;
  // Group by rows, and iterate over rows of both sets in parallel, creating a
  // set for each row.
  while ((set1_group_it != set1_grouper.end()) ||
         (set2_group_it != set2_grouper.end())) {
    const std::vector<int64_t>& set1_group_indices =
        (set1_group_it == set1_grouper.end()) ? group_iter_end
                                              : (*set1_group_it).group();
    const std::vector<int64_t>& set2_group_indices =
        (set2_group_it == set2_grouper.end()) ? group_iter_end
                                              : (*set2_group_it).group();

    int64_t compare_groups;
    CompareGroups(ctx, set1_group_indices, set2_group_indices, &compare_groups);
    const std::vector<int64_t>* group_indices = nullptr;

    // Get values from set1, if applicable.
    set1_group_set.clear();
    if (compare_groups <= 0) {
      PopulateFromSparseGroup<T>(ctx, *set1_group_it, set1_st.shape(),
                                 &set1_group_set);
      ++set1_group_it;
      group_indices = &set1_group_indices;
    }

    // Get values from set2, if applicable.
    set2_group_set.clear();
    if (compare_groups >= 0) {
      PopulateFromSparseGroup<T>(ctx, *set2_group_it, set2_st.shape(),
                                 &set2_group_set);
      ++set2_group_it;
      group_indices = &set2_group_indices;
    }

    absl::btree_set<T> group_set;
    ApplySetOperation(set1_group_set, set2_group_set, &group_set);
    if (!group_set.empty()) {
      const auto set_size = group_set.size();
      if (set_size > max_set_size) {
        max_set_size = set_size;
      }
      num_result_values += set_size;
      group_sets.push_back({*group_indices, std::move(group_set)});
    }
  }

  TensorShape output_shape;
  OP_REQUIRES_OK(ctx, TensorShapeUtils::MakeShape(group_shape, &output_shape));
  output_shape.AddDim(max_set_size);
  OutputSparseTensor<T>(ctx, output_shape, num_result_values, group_sets);
}

// Given set1 of shape [b, n1] and data_2 of shape [b, n2], populate result
// sparse tensor with [b, n3] values, where each row `i` contains the result of
// the set operation on elements from set1[i] and set2[i]. `n3` is the number
// of elements in that result row.
template <typename T>
void SetOperationOp<T>::Compute(OpKernelContext* ctx) {
   std::vector<std::string> mht_25_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSset_kernelsDTcc mht_25(mht_25_v, 961, "", "./tensorflow/core/kernels/set_kernels.cc", "SetOperationOp<T>::Compute");

  switch (input_types_) {
    case DENSE_DENSE:
      ComputeDenseToDense(ctx);
      break;
    case DENSE_SPARSE:
      ComputeDenseToSparse(ctx);
      break;
    case SPARSE_SPARSE:
      ComputeSparseToSparse(ctx);
      break;
  }
}

template <typename T>
class DenseToDenseSetOperationOp : public SetOperationOp<T> {
 public:
  explicit DenseToDenseSetOperationOp(OpKernelConstruction* ctx)
      : SetOperationOp<T>(ctx, DENSE_DENSE) {
   std::vector<std::string> mht_26_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSset_kernelsDTcc mht_26(mht_26_v, 982, "", "./tensorflow/core/kernels/set_kernels.cc", "DenseToDenseSetOperationOp");
}
};

#define _DENSE_TO_DENSE_SET_OPERATION_REGISTER_KERNEL_BUILDER(T) \
  REGISTER_KERNEL_BUILDER(Name("DenseToDenseSetOperation")       \
                              .Device(DEVICE_CPU)                \
                              .TypeConstraint<T>("T"),           \
                          DenseToDenseSetOperationOp<T>);
_DENSE_TO_DENSE_SET_OPERATION_REGISTER_KERNEL_BUILDER(int8);
_DENSE_TO_DENSE_SET_OPERATION_REGISTER_KERNEL_BUILDER(int16);
_DENSE_TO_DENSE_SET_OPERATION_REGISTER_KERNEL_BUILDER(int32);
_DENSE_TO_DENSE_SET_OPERATION_REGISTER_KERNEL_BUILDER(int64_t);
_DENSE_TO_DENSE_SET_OPERATION_REGISTER_KERNEL_BUILDER(uint8);
_DENSE_TO_DENSE_SET_OPERATION_REGISTER_KERNEL_BUILDER(uint16);
_DENSE_TO_DENSE_SET_OPERATION_REGISTER_KERNEL_BUILDER(tstring);
#undef _DENSE_TO_DENSE_SET_OPERATION_REGISTER_KERNEL_BUILDER

template <typename T>
class DenseToSparseSetOperationOp : public SetOperationOp<T> {
 public:
  explicit DenseToSparseSetOperationOp(OpKernelConstruction* ctx)
      : SetOperationOp<T>(ctx, DENSE_SPARSE) {
   std::vector<std::string> mht_27_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSset_kernelsDTcc mht_27(mht_27_v, 1006, "", "./tensorflow/core/kernels/set_kernels.cc", "DenseToSparseSetOperationOp");
}
};

#define _DENSE_TO_SPARSE_SET_OPERATION_REGISTER_KERNEL_BUILDER(T) \
  REGISTER_KERNEL_BUILDER(Name("DenseToSparseSetOperation")       \
                              .Device(DEVICE_CPU)                 \
                              .TypeConstraint<T>("T"),            \
                          DenseToSparseSetOperationOp<T>);
_DENSE_TO_SPARSE_SET_OPERATION_REGISTER_KERNEL_BUILDER(int8);
_DENSE_TO_SPARSE_SET_OPERATION_REGISTER_KERNEL_BUILDER(int16);
_DENSE_TO_SPARSE_SET_OPERATION_REGISTER_KERNEL_BUILDER(int32);
_DENSE_TO_SPARSE_SET_OPERATION_REGISTER_KERNEL_BUILDER(int64_t);
_DENSE_TO_SPARSE_SET_OPERATION_REGISTER_KERNEL_BUILDER(uint8);
_DENSE_TO_SPARSE_SET_OPERATION_REGISTER_KERNEL_BUILDER(uint16);
_DENSE_TO_SPARSE_SET_OPERATION_REGISTER_KERNEL_BUILDER(tstring);
#undef _DENSE_TO_SPARSE_SET_OPERATION_REGISTER_KERNEL_BUILDER

template <typename T>
class SparseToSparseSetOperationOp : public SetOperationOp<T> {
 public:
  explicit SparseToSparseSetOperationOp(OpKernelConstruction* ctx)
      : SetOperationOp<T>(ctx, SPARSE_SPARSE) {
   std::vector<std::string> mht_28_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSset_kernelsDTcc mht_28(mht_28_v, 1030, "", "./tensorflow/core/kernels/set_kernels.cc", "SparseToSparseSetOperationOp");
}
};

#define _SPARSE_TO_SPARSE_SET_OPERATION_REGISTER_KERNEL_BUILDER(T) \
  REGISTER_KERNEL_BUILDER(Name("SparseToSparseSetOperation")       \
                              .Device(DEVICE_CPU)                  \
                              .TypeConstraint<T>("T"),             \
                          SparseToSparseSetOperationOp<T>);
_SPARSE_TO_SPARSE_SET_OPERATION_REGISTER_KERNEL_BUILDER(int8);
_SPARSE_TO_SPARSE_SET_OPERATION_REGISTER_KERNEL_BUILDER(int16);
_SPARSE_TO_SPARSE_SET_OPERATION_REGISTER_KERNEL_BUILDER(int32);
_SPARSE_TO_SPARSE_SET_OPERATION_REGISTER_KERNEL_BUILDER(int64_t);
_SPARSE_TO_SPARSE_SET_OPERATION_REGISTER_KERNEL_BUILDER(uint8);
_SPARSE_TO_SPARSE_SET_OPERATION_REGISTER_KERNEL_BUILDER(uint16);
_SPARSE_TO_SPARSE_SET_OPERATION_REGISTER_KERNEL_BUILDER(tstring);
#undef _SPARSE_TO_SPARSE_SET_OPERATION_REGISTER_KERNEL_BUILDER

}  // namespace tensorflow
