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
class MHTracer_DTPStensorflowPScorePSkernelsPSragged_tensor_from_variant_opDTcc {
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
   MHTracer_DTPStensorflowPScorePSkernelsPSragged_tensor_from_variant_opDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSkernelsPSragged_tensor_from_variant_opDTcc() {
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
#include <utility>
#include <vector>

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/variant.h"
#include "tensorflow/core/framework/variant_encode_decode.h"
#include "tensorflow/core/kernels/ragged_tensor_variant.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"

namespace tensorflow {
namespace {

Status RaggedComponentsFromVariant(
    const Tensor& encoded_variant, int ragged_rank, DataType value_dtype,
    DataType split_dtype, std::vector<RaggedTensorVariant>* decoded_ragged) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSragged_tensor_from_variant_opDTcc mht_0(mht_0_v, 201, "", "./tensorflow/core/kernels/ragged_tensor_from_variant_op.cc", "RaggedComponentsFromVariant");

  const auto& flat_variants = encoded_variant.flat<Variant>();
  decoded_ragged->reserve(flat_variants.size());

  for (int i = 0; i < flat_variants.size(); i++) {
    const auto& flat_variant = flat_variants(i);
    const RaggedTensorVariant* decoded =
        flat_variant.get<RaggedTensorVariant>();
    if (decoded == nullptr) {
      return errors::InvalidArgument(
          "Input Variant element at index ", i,
          " doesn't hold a RaggedTensorVariant: ", flat_variant.DebugString());
    }
    decoded_ragged->push_back(*decoded);
    decoded = &decoded_ragged->back();
    // Check ragged rank & types
    if (decoded->ragged_rank() != ragged_rank) {
      return errors::InvalidArgument(
          "Encoded input RaggedTensorVariant has ragged_rank=",
          decoded->ragged_rank(), ".  Expected ragged_rank=", ragged_rank, ".");
    }
    if (decoded->values().dtype() != value_dtype) {
      return errors::InvalidArgument(
          "Expected values Tensor dtype: ", DataTypeString(value_dtype),
          ", found: ", DataTypeString(decoded->values().dtype()));
    }
    if (decoded->values().dims() < 1) {
      return errors::InvalidArgument(
          "Ragged values must have rank >= 1; encoded scalar element at index ",
          i, " has values Tensor: ", decoded->values().DebugString());
    }
    for (const auto& splits : decoded->nested_splits()) {
      if (splits.dtype() != split_dtype) {
        return errors::InvalidArgument(
            "Expected row_splits Tensor dtype: ", DataTypeString(split_dtype),
            ", found: ", DataTypeString(splits.dtype()));
      }
      if (splits.dims() != 1) {
        return errors::InvalidArgument(
            "Ragged splits must have rank 1; encoded scalar element at index ",
            i, " has splits Tensor ", splits.DebugString());
      }
    }
  }
  return Status::OK();
}

template <typename VALUE_TYPE, typename SPLIT_TYPE>
Status NestedStackRaggedTensors(
    const std::vector<RaggedTensorVariant>& ragged_components,
    const std::vector<int>& nested_dim_sizes, const int input_ragged_rank,
    const int output_ragged_rank, RaggedTensorVariant* output_ragged) {
  output_ragged->mutable_nested_splits()->reserve(output_ragged_rank);
  const int dims = nested_dim_sizes.size();

  // Populate first `dims - 1` splits.
  for (int i = 0; i < dims - 1; i++) {
    int dims_splits_size = nested_dim_sizes[i] + 1;
    output_ragged->append_splits(Tensor(DataTypeToEnum<SPLIT_TYPE>::value,
                                        TensorShape({dims_splits_size})));
    auto splits_vec = output_ragged->mutable_splits(i)->vec<SPLIT_TYPE>();
    int split_diff = nested_dim_sizes[i + 1];
    for (int j = 0; j < dims_splits_size; j++) {
      splits_vec(j) = j * split_diff;
    }
  }

  // Populate `dims`-th split.
  int splits_size = ragged_components.size() + 1;
  output_ragged->append_splits(
      Tensor(DataTypeToEnum<SPLIT_TYPE>::value, TensorShape({splits_size})));
  auto dims_splits_vec =
      output_ragged->mutable_splits(dims - 1)->vec<SPLIT_TYPE>();
  dims_splits_vec(0) = 0;
  for (int i = 0; i < ragged_components.size(); i++) {
    int split_val = ragged_components[i].values().shape().dim_size(0);
    if (input_ragged_rank != 0 && ragged_components[i].ragged_rank() > 0) {
      split_val = ragged_components[i].splits(0).NumElements() - 1;
    }
    dims_splits_vec(i + 1) = dims_splits_vec(i) + split_val;
  }

  // Populate last `input_ragged_rank` splits.
  for (int i = 0; i < input_ragged_rank; i++) {
    int split_index = dims + i;
    int split_size = 1;
    for (int j = 0; j < ragged_components.size(); j++) {
      if (!ragged_components[j].nested_splits().empty()) {
        split_size += ragged_components[j].splits(i).NumElements() - 1;
      }
    }
    output_ragged->append_splits(
        Tensor(DataTypeToEnum<SPLIT_TYPE>::value, TensorShape({split_size})));
    auto splits_vec =
        output_ragged->mutable_splits(split_index)->vec<SPLIT_TYPE>();
    splits_vec(0) = 0;
    SPLIT_TYPE last_split_value = 0;
    int index = 1;
    for (int j = 0; j < ragged_components.size(); j++) {
      if (ragged_components[j].nested_splits().empty()) {
        // Corner case: empty row. e.g [ [[x], [x]], [] ]
        continue;
      }
      auto component_splits_vec =
          ragged_components[j].splits(i).vec<SPLIT_TYPE>();
      for (int k = 1; k < component_splits_vec.size(); k++, index++) {
        splits_vec(index) = component_splits_vec(k) + last_split_value;
      }
      last_split_value = splits_vec(index - 1);
    }
  }

  // If the variant tensor input is empty, then we have no way to determine
  // the correct shape for the dense_values.  (It must have rank>=1, and its
  // outer dimension must be 0, but we don't know its shape beyond that.)
  // For now, we just use a shape of `[0]` in this case.
  // TODO(edloper): Update this op with an attribute containing information
  // about dense_values shape.  If it's `None`, then we'll probably still have
  // to use shape=[0] here, but if we have more info, then we can use it.
  // E.g., in map_fn, we may have shape info from the RaggedTensorSpec.
  TensorShape component_values_shape;
  if (ragged_components.empty()) {
    component_values_shape = TensorShape({0});
  } else {
    component_values_shape = ragged_components[0].values().shape();
  }

  // Populate values.
  int values_size = component_values_shape.dim_size(0);
  for (int i = 1; i < ragged_components.size(); i++) {
    if (ragged_components[i].values().dims() != component_values_shape.dims()) {
      return errors::InvalidArgument(
          "Rank of values must match for all "
          "components; values shape at index 0: ",
          component_values_shape.DebugString(), ", values shape at index ", i,
          ": ", ragged_components[i].values().shape().DebugString());
    }
    values_size += ragged_components[i].values().shape().dim_size(0);
  }
  component_values_shape.set_dim(0, values_size);
  output_ragged->set_values(
      Tensor(DataTypeToEnum<VALUE_TYPE>::value, component_values_shape));
  auto output_values_flat =
      output_ragged->mutable_values()->flat_outer_dims<VALUE_TYPE, 2>();
  int values_index = 0;

  TensorShape expected_value_shape = component_values_shape;
  expected_value_shape.RemoveDim(0);

  for (int i = 0; i < ragged_components.size(); i++) {
    // Check that the flat_values tensor shape is compatible.
    TensorShape value_shape = ragged_components[i].values().shape();
    value_shape.RemoveDim(0);
    if (value_shape != expected_value_shape) {
      return errors::InvalidArgument(
          "All flat_values must have compatible shapes.  Shape at index 0: ",
          expected_value_shape, ".  Shape at index ", i, ": ", value_shape,
          ".  If you are using tf.map_fn, then you may need to specify an "
          "explicit fn_output_signature with appropriate ragged_rank, and/or "
          "convert output tensors to RaggedTensors.");
    }

    auto component_values_flat =
        ragged_components[i].values().flat_outer_dims<VALUE_TYPE, 2>();
    int num_inner_elements = ragged_components[i].values().NumElements();
    if (ragged_components[i].values().dim_size(0) > 0) {
      num_inner_elements /= ragged_components[i].values().dim_size(0);
    }
    for (int j = 0; j < ragged_components[i].values().dim_size(0);
         j++, values_index++) {
      for (int k = 0; k < num_inner_elements; k++) {
        output_values_flat(values_index, k) = component_values_flat(j, k);
      }
    }
  }
  return Status::OK();
}
}  // namespace

template <typename VALUE_TYPE, typename SPLIT_TYPE>
class RaggedTensorFromVariantOp : public OpKernel {
 public:
  explicit RaggedTensorFromVariantOp(OpKernelConstruction* context)
      : OpKernel(context) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSragged_tensor_from_variant_opDTcc mht_1(mht_1_v, 387, "", "./tensorflow/core/kernels/ragged_tensor_from_variant_op.cc", "RaggedTensorFromVariantOp");

    OP_REQUIRES_OK(context, context->GetAttr("input_ragged_rank",
                                             &input_ragged_rank_attr_));
    OP_REQUIRES_OK(
        context, context->GetAttr("output_ragged_rank", &output_ragged_rank_));
  }

  void Compute(OpKernelContext* context) override {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSragged_tensor_from_variant_opDTcc mht_2(mht_2_v, 397, "", "./tensorflow/core/kernels/ragged_tensor_from_variant_op.cc", "Compute");

    // Read input Tensor.
    const Tensor& encoded_variant = context->input(0);
    auto input_ragged_rank_ = input_ragged_rank_attr_;

    if (input_ragged_rank_ == -1) {  // Infer input_ragged_rank_.
      input_ragged_rank_ = output_ragged_rank_ - encoded_variant.dims();
      OP_REQUIRES(context, input_ragged_rank_ >= 0,
                  errors::InvalidArgument(
                      "Inferred input_ragged_rank (output_ragged_rank - "
                      "encoded_variant.dims()) must be >= 0, found "
                      "output_ragged_rank: ",
                      output_ragged_rank_,
                      ", encoded_variant.dims(): ", encoded_variant.dims(),
                      ", inferred input_ragged_rank: ", input_ragged_rank_));
    }
    OP_REQUIRES(
        context,
        output_ragged_rank_ == encoded_variant.dims() + input_ragged_rank_,
        errors::InvalidArgument(
            "output_ragged_rank must be equal to input_ragged_rank + "
            "encoded_ragged.dims(); output_ragged_rank: ",
            output_ragged_rank_, ", input_ragged_rank: ", input_ragged_rank_,
            ", encoded_variant.dims(): ", encoded_variant.dims(), "."));

    // Decode all variants.
    const auto value_dtype = DataTypeToEnum<VALUE_TYPE>::v();
    const auto split_dtype = DataTypeToEnum<SPLIT_TYPE>::v();
    std::vector<RaggedTensorVariant> decoded_components;
    OP_REQUIRES_OK(context, RaggedComponentsFromVariant(
                                encoded_variant, input_ragged_rank_,
                                value_dtype, split_dtype, &decoded_components));

    // Corner case: input is a scalar.
    if (encoded_variant.dims() == 0) {
      ReturnRaggedTensor(context, decoded_components[0]);
      return;
    }

    // Nested-Stack Ragged components into a batched RaggedTensor.
    std::vector<int> encoded_dim_sizes(encoded_variant.dims(), 0);
    for (int i = 0; i < encoded_variant.dims(); i++) {
      encoded_dim_sizes[i] = encoded_variant.dim_size(i);
    }
    RaggedTensorVariant output_ragged;
    OP_REQUIRES_OK(
        context, NestedStackRaggedTensors<VALUE_TYPE, SPLIT_TYPE>(
                     decoded_components, encoded_dim_sizes, input_ragged_rank_,
                     output_ragged_rank_, &output_ragged));

    // Set output.
    ReturnRaggedTensor(context, output_ragged);
  }

 private:
  int input_ragged_rank_attr_;
  int output_ragged_rank_;

  void ReturnRaggedTensor(OpKernelContext* context,
                          const RaggedTensorVariant& ragged_tensor) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSragged_tensor_from_variant_opDTcc mht_3(mht_3_v, 459, "", "./tensorflow/core/kernels/ragged_tensor_from_variant_op.cc", "ReturnRaggedTensor");

    int ragged_rank = ragged_tensor.ragged_rank();
    OpOutputList splits_out;
    OP_REQUIRES_OK(context,
                   context->output_list("output_nested_splits", &splits_out));
    for (int i = 0; i < ragged_rank; i++) {
      splits_out.set(i, ragged_tensor.splits(i));
    }
    context->set_output(ragged_rank, ragged_tensor.values());
  }
};

#define REGISTER_KERNELS_WITH_SPLIT_TYPE(value_type, split_type)      \
  REGISTER_KERNEL_BUILDER(Name("RaggedTensorFromVariant")             \
                              .Device(DEVICE_CPU)                     \
                              .TypeConstraint<value_type>("Tvalues")  \
                              .TypeConstraint<split_type>("Tsplits"), \
                          RaggedTensorFromVariantOp<value_type, split_type>);
#define REGISTER_KERNELS(value_type)                  \
  REGISTER_KERNELS_WITH_SPLIT_TYPE(value_type, int32) \
  REGISTER_KERNELS_WITH_SPLIT_TYPE(value_type, int64_t)
TF_CALL_POD_TYPES(REGISTER_KERNELS);
TF_CALL_tstring(REGISTER_KERNELS);
TF_CALL_QUANTIZED_TYPES(REGISTER_KERNELS);
TF_CALL_quint16(REGISTER_KERNELS);
TF_CALL_qint16(REGISTER_KERNELS);
#undef REGISTER_KERNELS
#undef REGISTER_KERNELS_WITH_SPLIT_TYPE
}  // namespace tensorflow
