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
class MHTracer_DTPStensorflowPSexamplesPScustom_ops_docPSmultiplex_3PSmultiplex_3_kernelDTcc {
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
   MHTracer_DTPStensorflowPSexamplesPScustom_ops_docPSmultiplex_3PSmultiplex_3_kernelDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSexamplesPScustom_ops_docPSmultiplex_3PSmultiplex_3_kernelDTcc() {
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

#include <algorithm>
#include <cstdint>
#include <string>

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/platform/errors.h"

// Please use the appropriate namespace for your project
namespace tensorflow {
namespace custom_op_examples {

using CPUDevice = Eigen::ThreadPoolDevice;
using ::tensorflow::errors::InvalidArgument;

template <typename T>
class MultiplexSparseOp : public OpKernel {
 public:
  explicit MultiplexSparseOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSexamplesPScustom_ops_docPSmultiplex_3PSmultiplex_3_kernelDTcc mht_0(mht_0_v, 203, "", "./tensorflow/examples/custom_ops_doc/multiplex_3/multiplex_3_kernel.cc", "MultiplexSparseOp");
}
  MultiplexSparseOp(const MultiplexSparseOp& other) = delete;
  MultiplexSparseOp& operator=(const MultiplexSparseOp& other) = delete;
  ~MultiplexSparseOp() override = default;

  void Compute(OpKernelContext* ctx) override {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSexamplesPScustom_ops_docPSmultiplex_3PSmultiplex_3_kernelDTcc mht_1(mht_1_v, 211, "", "./tensorflow/examples/custom_ops_doc/multiplex_3/multiplex_3_kernel.cc", "Compute");

    const auto& cond_indices_tensor = ctx->input(0);
    const auto& cond_values_tensor = ctx->input(1);
    const auto& cond_shape_tensor = ctx->input(2);
    const auto& a_indices_tensor = ctx->input(3);
    const auto& a_values_tensor = ctx->input(4);
    const auto& a_shape_tensor = ctx->input(5);
    const auto& b_indices_tensor = ctx->input(6);
    const auto& b_values_tensor = ctx->input(7);
    const auto& b_shape_tensor = ctx->input(8);
    OP_REQUIRES_OK(ctx,
                   ValidateSparseTensor(cond_indices_tensor, cond_values_tensor,
                                        cond_shape_tensor, "cond"));
    OP_REQUIRES_OK(ctx, ValidateSparseTensor(a_indices_tensor, a_values_tensor,
                                             a_shape_tensor, "a"));
    OP_REQUIRES_OK(ctx, ValidateSparseTensor(b_indices_tensor, b_values_tensor,
                                             b_shape_tensor, "b"));
    OP_REQUIRES(
        ctx, cond_shape_tensor.shape() == a_shape_tensor.shape(),
        InvalidArgument("Sparse tensors must be the same shape. cond_shape: ",
                        cond_shape_tensor.shape().DebugString(),
                        " vs a_shape: ", a_shape_tensor.shape().DebugString()));
    OP_REQUIRES(
        ctx, a_shape_tensor.shape() == b_shape_tensor.shape(),
        InvalidArgument("Sparse tensors must be the same shape. a_shape: ",
                        a_shape_tensor.shape().DebugString(),
                        " vs b_shape: ", b_shape_tensor.shape().DebugString()));
    const int rank = a_shape_tensor.dim_size(0);
    OP_REQUIRES(
        ctx, rank == 1,
        InvalidArgument("Sorry, multiplex for sparse tensors only "
                        "supports rank 1 tensors to simplify this example."));
    const int cond_elements = cond_indices_tensor.dim_size(0);
    const int a_elements = a_indices_tensor.dim_size(0);
    const int b_elements = b_indices_tensor.dim_size(0);
    const auto cond_indices = cond_indices_tensor.matrix<int64_t>();
    const auto cond_values = cond_values_tensor.flat<bool>();
    const auto cond_shape = cond_shape_tensor.flat<int64_t>();
    const auto a_indices = a_indices_tensor.matrix<int64_t>();
    const auto a_values = a_values_tensor.flat<T>();
    const auto a_shape = a_shape_tensor.flat<int64_t>();
    const auto b_indices = b_indices_tensor.matrix<int64_t>();
    const auto b_values = b_values_tensor.flat<T>();
    const auto b_shape = b_shape_tensor.flat<int64_t>();
    int cond_index = 0;
    int a_index = 0;
    int b_index = 0;
    // This vector is a list of source tensors (a = true, b = false) and source
    // indices.
    std::vector<std::pair<bool, int>> merged_output;
    merged_output.reserve(std::min(cond_elements, a_elements) + b_elements);
    while (a_index < a_elements || b_index < b_elements) {
      // Determine the whether the current location with values has a value
      // for `a`, for `b` or for both `a` and `b`.
      int64_t cur_row;
      bool is_a_at_cur = false;
      bool is_b_at_cur = false;
      if (a_index < a_elements && b_index < b_elements) {
        const int64_t a_row = a_indices(a_index, 0);
        const int64_t b_row = b_indices(b_index, 0);
        cur_row = std::min(a_row, b_row);
        if (a_row == cur_row) {
          is_a_at_cur = true;
        }
        if (b_row == cur_row) {
          is_b_at_cur = true;
        }
      } else if (a_index < a_elements) {
        cur_row = a_indices(a_index, 0);
        is_a_at_cur = true;
      } else {  // b_index < b_elements
        cur_row = b_indices(b_index, 0);
        is_b_at_cur = true;
      }
      // Deterimine if `cond` has a value at the current location
      bool cond_flag = false;
      while (cond_index < cond_elements) {
        const int64_t cond_row = cond_indices(cond_index, 0);
        if (cond_row > cur_row) {
          break;
        }
        if (cond_row == cur_row) {
          cond_flag = cond_values(cond_index);
          break;
        }
        ++cond_index;
      }
      // Add `a` or `b` to the merged output based on the condition
      if (is_a_at_cur) {
        if (cond_flag) {
          merged_output.emplace_back(true, a_index);
        }
        ++a_index;
      }
      if (is_b_at_cur) {
        if (!cond_flag) {
          merged_output.emplace_back(false, b_index);
        }
        ++b_index;
      }
    }

    // Allocate output tensors.
    Tensor* output_indices_tensor;
    Tensor* output_values_tensor;
    Tensor* output_dense_shape_tensor;
    const int num_values = merged_output.size();
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, TensorShape({num_values, rank}),
                                             &output_indices_tensor));
    OP_REQUIRES_OK(ctx, ctx->allocate_output(1, TensorShape({num_values}),
                                             &output_values_tensor));
    OP_REQUIRES_OK(ctx, ctx->allocate_output(2, TensorShape({rank}),
                                             &output_dense_shape_tensor));
    auto output_indices = output_indices_tensor->matrix<int64_t>();
    auto output_values = output_values_tensor->flat<T>();
    auto output_shape = output_dense_shape_tensor->flat<int64_t>();
    for (int row = 0; row < num_values; ++row) {
      const auto& source_flag = merged_output[row].first;
      const auto& source_row = merged_output[row].second;
      const auto& indices = source_flag ? a_indices : b_indices;
      const auto& values = source_flag ? a_values : b_values;
      for (int column = 0; column < rank; ++column) {
        output_indices(row, column) = indices(source_row, column);
      }
      output_values(row) = values(source_row);
    }
    // Expand the shape of the output sparse tensor so that it is as large
    // as the shape of the largest input in each dimension.
    // An alternative behavoir would be to require that the shapes be the
    // same and implement error checking that all the corresponding values
    // in the shape tensors are the same (e.g.
    // `cond_shape(i) == a_shape(i)` and `a_shape(i) == b_shape(i)` in
    // OP_REQUIRES above and `output_shape(i) = a_shape(i)` here).
    for (int i = 0; i < rank; ++i) {
      output_shape(i) =
          std::max(cond_shape(i), std::max(a_shape(i), b_shape(i)));
    }
  }

 private:
  Status ValidateSparseTensor(const ::tensorflow::Tensor& indices_tensor,
                              const ::tensorflow::Tensor& values_tensor,
                              const ::tensorflow::Tensor& shape_tensor,
                              const string label) {
   std::vector<std::string> mht_2_v;
   mht_2_v.push_back("label: \"" + label + "\"");
   MHTracer_DTPStensorflowPSexamplesPScustom_ops_docPSmultiplex_3PSmultiplex_3_kernelDTcc mht_2(mht_2_v, 358, "", "./tensorflow/examples/custom_ops_doc/multiplex_3/multiplex_3_kernel.cc", "ValidateSparseTensor");

    if (!TensorShapeUtils::IsMatrix(indices_tensor.shape())) {
      return InvalidArgument(
          "Sparse indices for ", label,
          " must be rank 2, not shape: ", indices_tensor.shape().DebugString());
    }
    if (!TensorShapeUtils::IsVector(values_tensor.shape())) {
      return InvalidArgument("Sparse values for ", label,
                             " must be a vector, not shape: ",
                             values_tensor.shape().DebugString());
    }
    if (!TensorShapeUtils::IsVector(shape_tensor.shape())) {
      return InvalidArgument(
          "Sparse shape for ", label,
          " must be a vector, not shape: ", shape_tensor.shape().DebugString());
    }
    if (indices_tensor.dim_size(0) != values_tensor.dim_size(0)) {
      return InvalidArgument("Sparse indices and values for " + label +
                                 " must have the same "
                                 "number of rows. indices: ",
                             indices_tensor.shape().DebugString(),
                             " values: ", values_tensor.shape().DebugString());
    }
    return Status::OK();
  }
};

// To support tensors containing different types (e.g. int32, float), one
// kernel per type is registered and is templatized by the "T" attr value.
// See go/tf-custom-ops-guide
#define REGISTER_KERNELS_CPU(type)                              \
  REGISTER_KERNEL_BUILDER(Name("Examples>MultiplexSparse")      \
                              .Device(::tensorflow::DEVICE_CPU) \
                              .TypeConstraint<type>("T"),       \
                          MultiplexSparseOp<type>)
TF_CALL_ALL_TYPES(REGISTER_KERNELS_CPU);

#undef REGISTER_KERNELS_CPU

}  // namespace custom_op_examples
}  // namespace tensorflow
