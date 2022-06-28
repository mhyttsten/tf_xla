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
class MHTracer_DTPStensorflowPScorePSkernelsPSserialize_sparse_opDTcc {
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
   MHTracer_DTPStensorflowPScorePSkernelsPSserialize_sparse_opDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSkernelsPSserialize_sparse_opDTcc() {
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

/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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

#include <algorithm>
#include <numeric>
#include <unordered_map>
#include <utility>
#include <vector>

#include "tensorflow/core/common_runtime/dma_helper.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/op_requires.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor.pb.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/tensor_util.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/framework/variant.h"
#include "tensorflow/core/framework/variant_encode_decode.h"
#include "tensorflow/core/kernels/reshape_util.h"
#include "tensorflow/core/lib/gtl/inlined_vector.h"
#include "tensorflow/core/lib/gtl/optional.h"
#include "tensorflow/core/util/sparse/group_iterator.h"
#include "tensorflow/core/util/sparse/sparse_tensor.h"

namespace tensorflow {

namespace {

using sparse::SparseTensor;

template <typename T>
class SerializeSparseOp : public OpKernel {
 public:
  explicit SerializeSparseOp(OpKernelConstruction* context)
      : OpKernel(context) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSserialize_sparse_opDTcc mht_0(mht_0_v, 220, "", "./tensorflow/core/kernels/serialize_sparse_op.cc", "SerializeSparseOp");
}

  bool IsExpensive() override;

  Status Initialize(Tensor* result);
  Status Serialize(const Tensor& input, T* result);

  void Compute(OpKernelContext* context) override {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSserialize_sparse_opDTcc mht_1(mht_1_v, 230, "", "./tensorflow/core/kernels/serialize_sparse_op.cc", "Compute");

    const Tensor* input_indices;
    const Tensor* input_values;
    const Tensor* input_shape;

    OP_REQUIRES_OK(context, context->input("sparse_indices", &input_indices));
    OP_REQUIRES_OK(context, context->input("sparse_values", &input_values));
    OP_REQUIRES_OK(context, context->input("sparse_shape", &input_shape));
    OP_REQUIRES(context, TensorShapeUtils::IsMatrix(input_indices->shape()),
                errors::InvalidArgument(
                    "Input indices should be a matrix but received shape ",
                    input_indices->shape().DebugString()));

    OP_REQUIRES(context, TensorShapeUtils::IsVector(input_values->shape()),
                errors::InvalidArgument(
                    "Input values should be a vector but received shape ",
                    input_values->shape().DebugString()));

    OP_REQUIRES(context, TensorShapeUtils::IsVector(input_shape->shape()),
                errors::InvalidArgument(
                    "Input shape should be a vector but received shape ",
                    input_shape->shape().DebugString()));

    Tensor serialized_sparse;
    OP_REQUIRES_OK(context, Initialize(&serialized_sparse));

    auto serialized_sparse_t = serialized_sparse.vec<T>();
    OP_REQUIRES_OK(context, Serialize(*input_indices, &serialized_sparse_t(0)));
    OP_REQUIRES_OK(context, Serialize(*input_values, &serialized_sparse_t(1)));
    OP_REQUIRES_OK(context, Serialize(*input_shape, &serialized_sparse_t(2)));

    context->set_output(0, serialized_sparse);
  }
};

// NOTE(mrry): We specialize the IsExpensive() method differently for
// the string and variant cases, because (i) the string version
// actually performs memory copies as part of its serialization (and
// is hence potentially expensive), and (ii) the variant version
// performs O(1) shallow copies (and hence is much cheaper than
// dispatching to another thread would be).
template <>
bool SerializeSparseOp<tstring>::IsExpensive() {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSserialize_sparse_opDTcc mht_2(mht_2_v, 275, "", "./tensorflow/core/kernels/serialize_sparse_op.cc", "SerializeSparseOp<tstring>::IsExpensive");

  return true;
}
template <>
bool SerializeSparseOp<Variant>::IsExpensive() {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSserialize_sparse_opDTcc mht_3(mht_3_v, 282, "", "./tensorflow/core/kernels/serialize_sparse_op.cc", "SerializeSparseOp<Variant>::IsExpensive");

  return false;
}

template <>
Status SerializeSparseOp<tstring>::Initialize(Tensor* result) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSserialize_sparse_opDTcc mht_4(mht_4_v, 290, "", "./tensorflow/core/kernels/serialize_sparse_op.cc", "SerializeSparseOp<tstring>::Initialize");

  *result = Tensor(DT_STRING, TensorShape({3}));
  return Status::OK();
}

template <>
Status SerializeSparseOp<tstring>::Serialize(const Tensor& input,
                                             tstring* result) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSserialize_sparse_opDTcc mht_5(mht_5_v, 300, "", "./tensorflow/core/kernels/serialize_sparse_op.cc", "SerializeSparseOp<tstring>::Serialize");

  TensorProto proto;
  input.AsProtoTensorContent(&proto);
  *result = proto.SerializeAsString();
  return Status::OK();
}

REGISTER_KERNEL_BUILDER(Name("SerializeSparse")
                            .Device(DEVICE_CPU)
                            .TypeConstraint<tstring>("out_type"),
                        SerializeSparseOp<tstring>);

template <>
Status SerializeSparseOp<Variant>::Initialize(Tensor* result) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSserialize_sparse_opDTcc mht_6(mht_6_v, 316, "", "./tensorflow/core/kernels/serialize_sparse_op.cc", "SerializeSparseOp<Variant>::Initialize");

  *result = Tensor(DT_VARIANT, TensorShape({3}));
  return Status::OK();
}

template <>
Status SerializeSparseOp<Variant>::Serialize(const Tensor& input,
                                             Variant* result) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSserialize_sparse_opDTcc mht_7(mht_7_v, 326, "", "./tensorflow/core/kernels/serialize_sparse_op.cc", "SerializeSparseOp<Variant>::Serialize");

  *result = input;
  return Status::OK();
}

REGISTER_KERNEL_BUILDER(Name("SerializeSparse")
                            .Device(DEVICE_CPU)
                            .TypeConstraint<Variant>("out_type"),
                        SerializeSparseOp<Variant>);

template <typename T, typename U>
struct SerializeGroups {};

template <typename T>
struct SerializeGroups<T, tstring> {
  Status operator()(sparse::GroupIterable* minibatch,
                    const Tensor& output_shape, int64_t N, int rank,
                    Tensor* serialized_sparse) {
    auto serialized_sparse_t = serialized_sparse->matrix<tstring>();

    int64_t last_nonempty_group = -1;

    auto serialize = [](const Tensor& input, tstring* result) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSserialize_sparse_opDTcc mht_8(mht_8_v, 351, "", "./tensorflow/core/kernels/serialize_sparse_op.cc", "lambda");

      TensorProto proto;
      input.AsProtoTensorContent(&proto);
      *result = proto.SerializeAsString();
    };

    tstring serialized_shape;
    serialize(output_shape, &serialized_shape);

    auto serialize_empty_element = [&](int64_t b) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSserialize_sparse_opDTcc mht_9(mht_9_v, 363, "", "./tensorflow/core/kernels/serialize_sparse_op.cc", "lambda");

      serialize(Tensor(DT_INT64, {0, rank - 1}), &serialized_sparse_t(b, 0));
      serialize(Tensor(DataTypeToEnum<T>::value, {0}),
                &serialized_sparse_t(b, 1));
      serialized_sparse_t(b, 2) = serialized_shape;
    };

    for (const auto& subset : *minibatch) {
      const int64_t b = subset.group_at(0);
      if (b < 0 || b >= N) {
        return errors::InvalidArgument(
            "Received unexpected column 0 value in input SparseTensor: ", b,
            " < 0 or >= N (= ", N, ")");
      }

      // GroupIterable generates only the non-empty groups of rows, so we must
      // generate empty outputs for any empty rows since the last non-empty
      // group that was generated.
      for (int64_t empty_b = last_nonempty_group + 1; empty_b < b; ++empty_b) {
        serialize_empty_element(empty_b);
      }

      last_nonempty_group = b;

      const auto indices = subset.indices();
      const auto values = subset.values<T>();
      const int64_t num_entries = values.size();

      Tensor output_indices = Tensor(DT_INT64, {num_entries, rank - 1});
      Tensor output_values = Tensor(DataTypeToEnum<T>::value, {num_entries});

      auto output_indices_t = output_indices.matrix<int64_t>();
      auto output_values_t = output_values.vec<T>();

      for (int i = 0; i < num_entries; ++i) {
        for (int d = 1; d < rank; ++d) {
          output_indices_t(i, d - 1) = indices(i, d);
        }
        output_values_t(i) = values(i);
      }

      serialize(output_indices, &serialized_sparse_t(b, 0));
      serialize(output_values, &serialized_sparse_t(b, 1));
      serialized_sparse_t(b, 2) = serialized_shape;
    }

    for (int64_t empty_b = last_nonempty_group + 1; empty_b < N; ++empty_b) {
      serialize_empty_element(empty_b);
    }

    return Status::OK();
  }
};

template <typename T>
void CopyValues(const T* src, T* dest, int64_t num_values) {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSserialize_sparse_opDTcc mht_10(mht_10_v, 421, "", "./tensorflow/core/kernels/serialize_sparse_op.cc", "CopyValues");

  static_assert(is_simple_type<T>::value, "Memcpy requires a simple type.");
  memcpy(dest, src, num_values * sizeof(T));
}

template <>
void CopyValues<tstring>(const tstring* src, tstring* dest,
                         int64_t num_values) {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSserialize_sparse_opDTcc mht_11(mht_11_v, 431, "", "./tensorflow/core/kernels/serialize_sparse_op.cc", "CopyValues<tstring>");

  std::copy_n(src, num_values, dest);
}

template <>
void CopyValues<Variant>(const Variant* src, Variant* dest,
                         int64_t num_values) {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSserialize_sparse_opDTcc mht_12(mht_12_v, 440, "", "./tensorflow/core/kernels/serialize_sparse_op.cc", "CopyValues<Variant>");

  std::copy_n(src, num_values, dest);
}

template <>
void CopyValues<ResourceHandle>(const ResourceHandle* src, ResourceHandle* dest,
                                int64_t num_values) {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSserialize_sparse_opDTcc mht_13(mht_13_v, 449, "", "./tensorflow/core/kernels/serialize_sparse_op.cc", "CopyValues<ResourceHandle>");

  std::copy_n(src, num_values, dest);
}

template <>
void CopyValues<Eigen::half>(const Eigen::half* src, Eigen::half* dest,
                             int64_t num_values) {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSserialize_sparse_opDTcc mht_14(mht_14_v, 458, "", "./tensorflow/core/kernels/serialize_sparse_op.cc", "CopyValues<Eigen::half>");

  return CopyValues(reinterpret_cast<const char*>(src),
                    reinterpret_cast<char*>(dest),
                    num_values * sizeof(Eigen::half));
}

template <typename T>
struct SerializeGroups<T, Variant> {
  Status operator()(sparse::GroupIterable* minibatch,
                    const Tensor& output_shape, int64_t N, int rank,
                    Tensor* serialized_sparse) {
    auto serialized_sparse_t = serialized_sparse->template matrix<Variant>();

    int64_t last_nonempty_group = -1;

    // The "DataTypeToEnum<T>::value" member is static and defined but not
    // declared.  This leads to linker errors when a "DataTypeToEnum<T>::value"
    // reference is passed to a routine. Creating a local variable here to
    // workaround the linker errors.
    DataType T_type = DataTypeToEnum<T>::value;

    auto serialize_empty_element = [&](int64_t b) {
   std::vector<std::string> mht_15_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSserialize_sparse_opDTcc mht_15(mht_15_v, 482, "", "./tensorflow/core/kernels/serialize_sparse_op.cc", "lambda");

      serialized_sparse_t(b, 0).emplace<Tensor>(DT_INT64,
                                                TensorShape({0, rank - 1}));
      serialized_sparse_t(b, 1).emplace<Tensor>(T_type, TensorShape({0}));
      serialized_sparse_t(b, 2).emplace<Tensor>(output_shape);
    };

    for (const auto& subset : *minibatch) {
      const int64_t b = subset.group_at(0);
      if (b < 0 || b >= N) {
        return errors::InvalidArgument(
            "Received unexpected column 0 value in input SparseTensor: ", b,
            " < 0 or >= N (= ", N, ")");
      }

      // GroupIterable generates only the non-empty groups of rows, so we must
      // generate empty outputs for any empty rows since the last non-empty
      // group that was generated.
      for (int64_t empty_b = last_nonempty_group + 1; empty_b < b; ++empty_b) {
        serialize_empty_element(empty_b);
      }

      last_nonempty_group = b;

      const auto indices = subset.indices();
      const auto values = subset.values<T>();
      const int64_t num_entries = values.size();

      Tensor& output_indices = serialized_sparse_t(b, 0).emplace<Tensor>(
          DT_INT64, TensorShape({num_entries, rank - 1}));
      Tensor& output_values = serialized_sparse_t(b, 1).emplace<Tensor>(
          T_type, TensorShape({num_entries}));

      int64_t* output_indices_ptr =
          static_cast<int64_t*>(DMAHelper::base(&output_indices));
      const int64_t* indices_ptr = indices.data();

      T* output_values_ptr = static_cast<T*>(DMAHelper::base(&output_values));
      const T* values_ptr = values.data();

      // TODO(mrry): Consider adding a template-based specialization for higher
      // ranks.
      if (rank == 2) {
        for (int i = 0; i < num_entries; ++i) {
          output_indices_ptr[i] = indices_ptr[(2 * i) + 1];
        }
      } else {
        for (int i = 0; i < num_entries; ++i) {
          // Skip the first index in each row.
          ++indices_ptr;
          for (int d = 1; d < rank; ++d) {
            *output_indices_ptr++ = *indices_ptr++;
          }
        }
      }

      CopyValues(values_ptr, output_values_ptr, num_entries);
      serialized_sparse_t(b, 2).emplace<Tensor>(output_shape);
    }

    for (int64_t empty_b = last_nonempty_group + 1; empty_b < N; ++empty_b) {
      serialize_empty_element(empty_b);
    }

    return Status::OK();
  }
};

template <typename T, typename U>
class SerializeManySparseOp : public OpKernel {
 public:
  explicit SerializeManySparseOp(OpKernelConstruction* context)
      : OpKernel(context) {
   std::vector<std::string> mht_16_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSserialize_sparse_opDTcc mht_16(mht_16_v, 557, "", "./tensorflow/core/kernels/serialize_sparse_op.cc", "SerializeManySparseOp");
}

  void Compute(OpKernelContext* context) override {
   std::vector<std::string> mht_17_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSserialize_sparse_opDTcc mht_17(mht_17_v, 562, "", "./tensorflow/core/kernels/serialize_sparse_op.cc", "Compute");

    const Tensor* input_indices;
    const Tensor* input_values;
    const Tensor* input_shape;
    OP_REQUIRES_OK(context, context->input("sparse_indices", &input_indices));
    OP_REQUIRES_OK(context, context->input("sparse_values", &input_values));
    OP_REQUIRES_OK(context, context->input("sparse_shape", &input_shape));
    OP_REQUIRES(context, TensorShapeUtils::IsMatrix(input_indices->shape()),
                errors::InvalidArgument(
                    "Input indices should be a matrix but received shape ",
                    input_indices->shape().DebugString()));

    OP_REQUIRES(context, TensorShapeUtils::IsVector(input_values->shape()),
                errors::InvalidArgument(
                    "Input values should be a vector but received shape ",
                    input_values->shape().DebugString()));

    OP_REQUIRES(context, TensorShapeUtils::IsVector(input_shape->shape()),
                errors::InvalidArgument(
                    "Input shape should be a vector but received shape ",
                    input_shape->shape().DebugString()));

    int rank = input_shape->NumElements();

    OP_REQUIRES(
        context, rank > 1,
        errors::InvalidArgument(
            "Rank of input SparseTensor should be > 1, but saw rank: ", rank));

    TensorShape tensor_input_shape;
    OP_REQUIRES_OK(context,
                   TensorShape::BuildTensorShape(input_shape->vec<int64_t>(),
                                                 &tensor_input_shape));
    gtl::InlinedVector<int64_t, 8> std_order(rank);
    std::iota(std_order.begin(), std_order.end(), 0);
    SparseTensor input_st;
    OP_REQUIRES_OK(context, SparseTensor::Create(*input_indices, *input_values,
                                                 tensor_input_shape, std_order,
                                                 &input_st));

    auto input_shape_t = input_shape->vec<int64_t>();
    const int64_t N = input_shape_t(0);

    Tensor* serialized_sparse;
    OP_REQUIRES_OK(context,
                   context->allocate_output(0, {N, 3}, &serialized_sparse));

    OP_REQUIRES_OK(context, input_st.IndicesValid());

    Tensor output_shape(DT_INT64, {rank - 1});
    auto output_shape_t = output_shape.vec<int64_t>();
    for (int d = 1; d < rank; d++) output_shape_t(d - 1) = input_shape_t(d);

    // Get groups by minibatch dimension
    sparse::GroupIterable minibatch = input_st.group({0});

    OP_REQUIRES_OK(context, SerializeGroups<T, U>()(&minibatch, output_shape, N,
                                                    rank, serialized_sparse));
  }
};

#define REGISTER_KERNELS(type)                                      \
  REGISTER_KERNEL_BUILDER(Name("SerializeManySparse")               \
                              .Device(DEVICE_CPU)                   \
                              .TypeConstraint<type>("T")            \
                              .TypeConstraint<tstring>("out_type"), \
                          SerializeManySparseOp<type, tstring>)

TF_CALL_ALL_TYPES(REGISTER_KERNELS);
#undef REGISTER_KERNELS


#define REGISTER_KERNELS(type)                                      \
  REGISTER_KERNEL_BUILDER(Name("SerializeManySparse")               \
                              .Device(DEVICE_CPU)                   \
                              .TypeConstraint<type>("T")            \
                              .TypeConstraint<Variant>("out_type"), \
                          SerializeManySparseOp<type, Variant>)

TF_CALL_ALL_TYPES(REGISTER_KERNELS);
#undef REGISTER_KERNELS

}  // namespace

}  // namespace tensorflow
