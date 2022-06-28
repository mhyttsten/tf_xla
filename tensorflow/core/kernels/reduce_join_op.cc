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
class MHTracer_DTPStensorflowPScorePSkernelsPSreduce_join_opDTcc {
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
   MHTracer_DTPStensorflowPScorePSkernelsPSreduce_join_opDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSkernelsPSreduce_join_opDTcc() {
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

// See docs in ../ops/string_ops.cc.

#include <string>

#include "tensorflow/core/framework/kernel_def_builder.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/core/stringpiece.h"
#include "tensorflow/core/lib/gtl/inlined_vector.h"
#include "tensorflow/core/lib/strings/str_util.h"

namespace tensorflow {

namespace {

const gtl::InlinedVector<int64_t, 8> GetStrides(const TensorShape& shape) {
  gtl::InlinedVector<int64_t, 8> result(shape.dims());
  int64_t product = 1;
  for (int32_t i = shape.dims() - 1; i >= 0; --i) {
    result[i] = product;
    product *= shape.dim_size(i);
  }
  return result;
}

// Given a linear index to a subset of dimensions, full shape,
// precomputed list of running products of the full shape, and list of
// dimensions in the subset, outputs the linear index to the full shape with
// nonspecified dimensions set to 0.  Dimensions must be ordered from outer-most
// to inner-most with respect to the subset linear index.
inline int64_t LinearSubIndexToFullIndex(
    int64_t output_index, const gtl::InlinedVector<int32, 8>& dim_list,
    const TensorShape& input_shape,
    const gtl::InlinedVector<int64_t, 8>& strides) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSreduce_join_opDTcc mht_0(mht_0_v, 221, "", "./tensorflow/core/kernels/reduce_join_op.cc", "LinearSubIndexToFullIndex");

  int64_t result = 0;
  int64_t quotient = output_index;
  for (int32_t i = dim_list.size() - 1; i >= 0; --i) {
    int32_t dim = dim_list[i];
    int64_t dim_value = quotient % input_shape.dim_size(dim);
    quotient = quotient / input_shape.dim_size(dim);
    result += strides[dim] * dim_value;
  }
  return result;
}

// Computes the number of input elements reduced per output element.
int64_t GetReductionIterSize(
    const gtl::InlinedVector<int32, 8>& reduced_indices,
    const TensorShape& input_shape) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSreduce_join_opDTcc mht_1(mht_1_v, 239, "", "./tensorflow/core/kernels/reduce_join_op.cc", "GetReductionIterSize");

  int64_t result = 1;
  for (int32_t reduce_dim : reduced_indices) {
    result *= input_shape.dim_size(reduce_dim);
  }
  return result;
}

// Computes a list of all true reduced indices, accounting for negative
// indices.
gtl::InlinedVector<int32, 8> GetReducedIndices(const Tensor& reduction_indices,
                                               int32_t input_dims) {
  const auto reduction_indices_flat = reduction_indices.flat<int32>();
  const int32_t reduction_dims = reduction_indices_flat.size();

  gtl::InlinedVector<int32, 8> reduced_indices(reduction_dims);
  for (int32_t i = 0; i < reduction_dims; ++i) {
    reduced_indices[i] = reduction_indices_flat(reduction_dims - i - 1);
    reduced_indices[i] += reduced_indices[i] < 0 ? input_dims : 0;
  }

  return reduced_indices;
}

// Appends all unreduced dimensions to the given vector.
void MakeUnreducedIndices(gtl::InlinedVector<bool, 8> index_is_reduced,
                          int32_t input_dims,
                          gtl::InlinedVector<int32, 8>* unreduced_indices) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSreduce_join_opDTcc mht_2(mht_2_v, 269, "", "./tensorflow/core/kernels/reduce_join_op.cc", "MakeUnreducedIndices");

  for (int32_t index = 0; index < input_dims; ++index) {
    if (!index_is_reduced[index]) unreduced_indices->push_back(index);
  }
}

TensorShape GetOutputShape(gtl::InlinedVector<bool, 8> index_is_reduced,
                           const TensorShape& input_shape, bool keep_dims) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSreduce_join_opDTcc mht_3(mht_3_v, 279, "", "./tensorflow/core/kernels/reduce_join_op.cc", "GetOutputShape");

  TensorShape output_shape;
  for (size_t index = 0; index < index_is_reduced.size(); ++index) {
    if (index_is_reduced[index]) {
      if (keep_dims) output_shape.AddDim(1);
    } else {
      output_shape.AddDim(input_shape.dim_size(index));
    }
  }
  return output_shape;
}

}  // namespace

class ReduceJoinOp : public OpKernel {
 public:
  using OpKernel::OpKernel;

  explicit ReduceJoinOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSreduce_join_opDTcc mht_4(mht_4_v, 300, "", "./tensorflow/core/kernels/reduce_join_op.cc", "ReduceJoinOp");

    OP_REQUIRES_OK(ctx, ctx->GetAttr("keep_dims", &keep_dims_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("separator", &separator_));
  }

  void Compute(OpKernelContext* context) override {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSreduce_join_opDTcc mht_5(mht_5_v, 308, "", "./tensorflow/core/kernels/reduce_join_op.cc", "Compute");

    const Tensor& input = context->input(0);
    const auto input_flat = input.flat<tstring>();
    const TensorShape& input_shape = input.shape();
    const int32_t input_dims = input_shape.dims();

    const Tensor& reduction_indices = context->input(1);
    const auto reduction_indices_flat = reduction_indices.flat<int32>();
    const int32_t reduction_dims = reduction_indices_flat.size();

    gtl::InlinedVector<bool, 8> index_is_reduced(input_dims, false);
    for (int32_t i = 0; i < reduction_dims; i++) {
      int32_t reduce_index = reduction_indices_flat(i);
      const int32_t true_reduce_index =
          reduce_index < 0 ? reduce_index + input_dims : reduce_index;
      OP_REQUIRES(
          context, reduce_index >= -input_dims && reduce_index < input_dims,
          errors::OutOfRange("Invalid reduction dimension ", reduce_index,
                             " for input with ", input_dims, " dimension(s)"));
      OP_REQUIRES(context, !index_is_reduced[true_reduce_index],
                  errors::InvalidArgument("Duplicate reduction dimension ",
                                          reduce_index));
      index_is_reduced[true_reduce_index] = true;
    }

    gtl::InlinedVector<int32, 8> reduced_indices =
        GetReducedIndices(reduction_indices, input_dims);
    gtl::InlinedVector<int32, 8> unreduced_indices;
    MakeUnreducedIndices(index_is_reduced, input_dims, &unreduced_indices);
    const auto strides = GetStrides(input_shape);

    Tensor* output_tensor = nullptr;
    TensorShape output_shape =
        GetOutputShape(index_is_reduced, input_shape, keep_dims_);
    OP_REQUIRES_OK(context, context->allocate_output("output", output_shape,
                                                     &output_tensor));
    auto output_flat = output_tensor->flat<tstring>();

    const int64_t reduction_iter_size =
        GetReductionIterSize(reduced_indices, input_shape);
    gtl::InlinedVector<StringPiece, 8> curr_strings(reduction_iter_size);
    for (int64_t output_index = 0; output_index < output_shape.num_elements();
         ++output_index) {
      int64_t output_full_index = LinearSubIndexToFullIndex(
          output_index, unreduced_indices, input_shape, strides);
      for (int64_t reduction_index = 0; reduction_index < reduction_iter_size;
           ++reduction_index) {
        int64_t reduction_full_index = LinearSubIndexToFullIndex(
            reduction_index, reduced_indices, input_shape, strides);
        curr_strings[reduction_index] =
            input_flat(output_full_index + reduction_full_index);
      }
      output_flat(output_index) = absl::StrJoin(curr_strings, separator_);
    }
  }

 private:
  bool keep_dims_;
  string separator_;
};

REGISTER_KERNEL_BUILDER(Name("ReduceJoin").Device(DEVICE_CPU), ReduceJoinOp);

}  // namespace tensorflow
