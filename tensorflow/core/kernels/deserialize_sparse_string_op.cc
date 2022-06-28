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
class MHTracer_DTPStensorflowPScorePSkernelsPSdeserialize_sparse_string_opDTcc {
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
   MHTracer_DTPStensorflowPScorePSkernelsPSdeserialize_sparse_string_opDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSkernelsPSdeserialize_sparse_string_opDTcc() {
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
#include <utility>
#include <vector>

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor.pb.h"
#include "tensorflow/core/framework/tensor_util.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/framework/variant.h"
#include "tensorflow/core/framework/variant_encode_decode.h"
#include "tensorflow/core/kernels/reshape_util.h"
#include "tensorflow/core/lib/gtl/inlined_vector.h"
#include "tensorflow/core/lib/gtl/optional.h"
#include "tensorflow/core/util/sparse/sparse_tensor.h"

namespace tensorflow {

using CPUDevice = Eigen::ThreadPoolDevice;

namespace {

using sparse::SparseTensor;

class DeserializeSparseOp : public OpKernel {
 public:
  explicit DeserializeSparseOp(OpKernelConstruction* context)
      : OpKernel(context) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdeserialize_sparse_string_opDTcc mht_0(mht_0_v, 216, "", "./tensorflow/core/kernels/deserialize_sparse_string_op.cc", "DeserializeSparseOp");

    OP_REQUIRES_OK(context, context->GetAttr("dtype", &dtype_));
  }

  void Compute(OpKernelContext* context) override {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdeserialize_sparse_string_opDTcc mht_1(mht_1_v, 223, "", "./tensorflow/core/kernels/deserialize_sparse_string_op.cc", "Compute");

    const Tensor& serialized_sparse = context->input(0);
    const int ndims = serialized_sparse.shape().dims();

    OP_REQUIRES(
        context, ndims > 0,
        errors::InvalidArgument("Serialized sparse should have non-zero rank ",
                                serialized_sparse.shape().DebugString()));

    OP_REQUIRES(context, serialized_sparse.shape().dim_size(ndims - 1) == 3,
                errors::InvalidArgument(
                    "Serialized sparse should have 3 as the last dimension ",
                    serialized_sparse.shape().DebugString()));

    int num_sparse_tensors = 1;
    for (int i = 0; i < ndims - 1; ++i) {
      num_sparse_tensors *= serialized_sparse.shape().dim_size(i);
    }

    OP_REQUIRES(
        context, num_sparse_tensors > 0,
        errors::InvalidArgument(
            "Serialized sparse should have at least 1 serialized tensor, "
            "but has a zero dimension ",
            serialized_sparse.shape().DebugString()));

    if (num_sparse_tensors == 1 && ndims == 1) {
      // Special case with a single sparse tensor. We can avoid data
      // motion in the Concat and Reshape.
      const auto& serialized_sparse_t = serialized_sparse.vec<tstring>();

      Tensor output_indices;
      Tensor output_values;
      Tensor output_shape;
      OP_REQUIRES_OK(context,
                     this->GetAndValidateSparseTensor(
                         serialized_sparse_t(0), serialized_sparse_t(1),
                         serialized_sparse_t(2), dtype_, 0 /* index */,
                         &output_indices, &output_values, &output_shape));
      context->set_output(0, output_indices);
      context->set_output(1, output_values);
      context->set_output(2, output_shape);
      return;
    }

    std::vector<Tensor> indices;
    std::vector<Tensor> values;
    TensorShape shape;
    indices.reserve(num_sparse_tensors);
    values.reserve(num_sparse_tensors);

    const auto& serialized_sparse_t =
        serialized_sparse.flat_inner_dims<tstring, 2>();
    for (int i = 0; i < num_sparse_tensors; ++i) {
      Tensor output_indices;
      Tensor output_values;
      Tensor output_shape;
      OP_REQUIRES_OK(context,
                     this->GetAndValidateSparseTensor(
                         serialized_sparse_t(i, 0), serialized_sparse_t(i, 1),
                         serialized_sparse_t(i, 2), dtype_, i, &output_indices,
                         &output_values, &output_shape));
      int64_t num_entries = output_indices.dim_size(0);
      int rank = output_indices.dim_size(1);

      // Now we expand each SparseTensors' indices and shape by
      // prefixing a dimension
      Tensor expanded_indices(DT_INT64, TensorShape({num_entries, 1 + rank}));
      const auto& output_indices_t = output_indices.matrix<int64_t>();
      auto expanded_indices_t = expanded_indices.matrix<int64_t>();
      expanded_indices_t.chip<1>(0).setZero();
      if (rank > 0) {
        Eigen::DSizes<Eigen::DenseIndex, 2> indices_start(0, 1);
        Eigen::DSizes<Eigen::DenseIndex, 2> indices_sizes(num_entries, rank);
        expanded_indices_t.slice(indices_start, indices_sizes) =
            output_indices_t;
      }
      Tensor expanded_shape(DT_INT64, TensorShape({1 + rank}));
      const auto& output_shape_t = output_shape.vec<int64_t>();
      auto expanded_shape_t = expanded_shape.vec<int64_t>();
      expanded_shape_t(0) = 1;
      std::copy_n(&output_shape_t(0), rank, &expanded_shape_t(1));

      TensorShape expanded_tensor_shape(expanded_shape.vec<int64_t>());

      indices.push_back(expanded_indices);
      values.push_back(output_values);
      if (i == 0) {
        shape = expanded_tensor_shape;
      } else {
        OP_REQUIRES(
            context, shape.dims() == expanded_tensor_shape.dims(),
            errors::InvalidArgument(
                "Inconsistent shape across SparseTensors: rank prior to "
                "SparseTensor[",
                i, "] was: ", shape.dims() - 1, " but rank of SparseTensor[", i,
                "] is: ", expanded_tensor_shape.dims() - 1));
        for (int j = 1; j < shape.dims(); ++j) {
          // NOTE(mrry): For compatibility with the implementations of
          // DeserializeManySparse, and many ops that generate
          // SparseTensors to batch that do not have a fixed
          // dense_shape (e.g. `tf.parse_single_example()`), we
          // compute the maximum in each dimension to find the
          // smallest dense_shape that bounds all of the input
          // SparseTensors.
          shape.set_dim(j, std::max(shape.dim_size(j),
                                    expanded_tensor_shape.dim_size(j)));
        }
      }
    }

    // Dimension 0 is the primary dimension.
    int rank = shape.dims();
    gtl::InlinedVector<int64_t, 8> std_order(rank);
    std::iota(std_order.begin(), std_order.end(), 0);

    std::vector<SparseTensor> tensors;
    tensors.reserve(num_sparse_tensors);
    for (int i = 0; i < num_sparse_tensors; ++i) {
      SparseTensor tensor;
      OP_REQUIRES_OK(context, SparseTensor::Create(indices[i], values[i], shape,
                                                   std_order, &tensor));
      tensors.push_back(std::move(tensor));
    }

    gtl::optional<SparseTensor> maybe_output;
#define HANDLE_TYPE(T)                               \
  case DataTypeToEnum<T>::value: {                   \
    maybe_output = SparseTensor::Concat<T>(tensors); \
    break;                                           \
  }

    switch (dtype_) {
      TF_CALL_ALL_TYPES(HANDLE_TYPE);
      TF_CALL_QUANTIZED_TYPES(HANDLE_TYPE);
#undef HANDLE_TYPE
      default:
        OP_REQUIRES(context, false,
                    errors::Unimplemented(
                        "DeserializeSparse Unhandled data type: ", dtype_));
    }
    DCHECK(maybe_output);
    SparseTensor& output = maybe_output.value();

    // Compute the input shape for the reshape operation.
    Tensor input_shape(DT_INT64, TensorShape({output.dims()}));
    std::copy_n(output.shape().data(), output.dims(),
                input_shape.vec<int64_t>().data());

    // Compute the target shape for the reshape operation.
    Tensor target_shape(DT_INT64, TensorShape({ndims + output.dims() - 2}));
    for (int i = 0; i < ndims - 1; ++i) {
      target_shape.vec<int64_t>()(i) = serialized_sparse.shape().dim_size(i);
    }
    for (int i = 0; i < output.dims() - 1; ++i) {
      target_shape.vec<int64_t>()(i + ndims - 1) = output.shape().data()[i + 1];
    }

    ReshapeSparseTensor<CPUDevice>(context, output.indices(), input_shape,
                                   target_shape, 0 /* output indices index */,
                                   2 /* output shape index */);
    context->set_output(1, output.values());
  }

 private:
  Status Deserialize(const tstring& serialized, Tensor* result) {
   std::vector<std::string> mht_2_v;
   mht_2_v.push_back("serialized: \"" + (std::string)serialized + "\"");
   MHTracer_DTPStensorflowPScorePSkernelsPSdeserialize_sparse_string_opDTcc mht_2(mht_2_v, 392, "", "./tensorflow/core/kernels/deserialize_sparse_string_op.cc", "Deserialize");

    TensorProto proto;
    if (!ParseProtoUnlimited(&proto, serialized)) {
      return errors::InvalidArgument("Could not parse serialized proto");
    }
    Tensor tensor;
    if (!tensor.FromProto(proto)) {
      return errors::InvalidArgument("Could not construct tensor from proto");
    }
    *result = tensor;
    return Status::OK();
  }

  Status GetAndValidateSparseTensor(
      const tstring& serialized_indices, const tstring& serialized_values,
      const tstring& serialized_shape, DataType values_dtype, int index,
      Tensor* output_indices, Tensor* output_values, Tensor* output_shape) {
   std::vector<std::string> mht_3_v;
   mht_3_v.push_back("serialized_indices: \"" + (std::string)serialized_indices + "\"");
   mht_3_v.push_back("serialized_values: \"" + (std::string)serialized_values + "\"");
   mht_3_v.push_back("serialized_shape: \"" + (std::string)serialized_shape + "\"");
   MHTracer_DTPStensorflowPScorePSkernelsPSdeserialize_sparse_string_opDTcc mht_3(mht_3_v, 414, "", "./tensorflow/core/kernels/deserialize_sparse_string_op.cc", "GetAndValidateSparseTensor");

    // Deserialize and validate the indices.
    TF_RETURN_IF_ERROR(this->Deserialize(serialized_indices, output_indices));
    if (!TensorShapeUtils::IsMatrix(output_indices->shape())) {
      return errors::InvalidArgument(
          "Expected serialized_sparse[", index,
          ", 0] to represent an index matrix but received shape ",
          output_indices->shape().DebugString());
    }
    int64_t num_entries = output_indices->dim_size(0);
    int rank = output_indices->dim_size(1);

    // Deserialize and validate the values.
    TF_RETURN_IF_ERROR(this->Deserialize(serialized_values, output_values));
    if (!TensorShapeUtils::IsVector(output_values->shape())) {
      return errors::InvalidArgument(
          "Expected serialized_sparse[", index,
          ", 1] to represent a values vector but received shape ",
          output_values->shape().DebugString());
    }
    if (values_dtype != output_values->dtype()) {
      return errors::InvalidArgument(
          "Requested SparseTensor of type ", DataTypeString(values_dtype),
          " but SparseTensor[", index,
          "].values.dtype() == ", DataTypeString(output_values->dtype()));
    }
    if (num_entries != output_values->dim_size(0)) {
      return errors::InvalidArgument(
          "Expected row counts of SparseTensor[", index,
          "].indices and SparseTensor[", index,
          "].values to match but they do not: ", num_entries, " vs. ",
          output_values->dim_size(0));
    }

    // Deserialize and validate the shape.
    TF_RETURN_IF_ERROR(this->Deserialize(serialized_shape, output_shape));
    if (!TensorShapeUtils::IsVector(output_shape->shape())) {
      return errors::InvalidArgument(
          "Expected serialized_sparse[", index,
          ", 1] to be a shape vector but its shape is ",
          output_shape->shape().DebugString());
    }
    if (rank != output_shape->dim_size(0)) {
      return errors::InvalidArgument("Expected column counts of SparseTensor[",
                                     index,
                                     "].indices to match size of SparseTensor[",
                                     index, "].shape but they do not: ", rank,
                                     " vs. ", output_shape->dim_size(0));
    }
    return Status::OK();
  }

  DataType dtype_;
};

REGISTER_KERNEL_BUILDER(Name("DeserializeSparse")
                            .Device(DEVICE_CPU)
                            .TypeConstraint<tstring>("Tserialized"),
                        DeserializeSparseOp)

REGISTER_KERNEL_BUILDER(Name("DeserializeManySparse").Device(DEVICE_CPU),
                        DeserializeSparseOp)

}  // namespace

}  // namespace tensorflow
