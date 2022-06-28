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

#ifndef TENSORFLOW_GOOGLE_EXAMPLES_CUSTOM_OPS_DOC_MULTIPLEX_2_MULTIPLEX_2_KERNEL_H_
#define TENSORFLOW_GOOGLE_EXAMPLES_CUSTOM_OPS_DOC_MULTIPLEX_2_MULTIPLEX_2_KERNEL_H_
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
class MHTracer_DTPStensorflowPSexamplesPScustom_ops_docPSmultiplex_2PSmultiplex_2_kernelDTh {
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
   MHTracer_DTPStensorflowPSexamplesPScustom_ops_docPSmultiplex_2PSmultiplex_2_kernelDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSexamplesPScustom_ops_docPSmultiplex_2PSmultiplex_2_kernelDTh() {
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
#endif

#include <algorithm>
#include <cstdint>
#include <limits>
#include <utility>
#include <vector>

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/platform/errors.h"

// Please use the appropriate namespace for your project
namespace tensorflow {
namespace custom_op_examples {

// Multiple devices (i.e. CPU and GPU) and multiple types for the values inside
// two of the input tensors (e.g. int32, float) are supported by using a
// template where the device is DEVICE and the type is T.
template <typename Device, typename T>
class MultiplexDenseOp : public OpKernel {
 public:
  explicit MultiplexDenseOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSexamplesPScustom_ops_docPSmultiplex_2PSmultiplex_2_kernelDTh mht_0(mht_0_v, 216, "", "./tensorflow/examples/custom_ops_doc/multiplex_2/multiplex_2_kernel.h", "MultiplexDenseOp");
}
  MultiplexDenseOp(const MultiplexDenseOp& other) = delete;
  MultiplexDenseOp& operator=(const MultiplexDenseOp& other) = delete;
  ~MultiplexDenseOp() override = default;

  void Compute(OpKernelContext* ctx) override {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSexamplesPScustom_ops_docPSmultiplex_2PSmultiplex_2_kernelDTh mht_1(mht_1_v, 224, "", "./tensorflow/examples/custom_ops_doc/multiplex_2/multiplex_2_kernel.h", "Compute");

    const auto& cond_tensor = ctx->input(0);
    const auto& a_values_tensor = ctx->input(1);
    const auto& b_values_tensor = ctx->input(2);

    // Allow any shape, but require that a_values, b_values, and cond all
    // have the same shape.
    // Note that ::tensorflow::TensorShapeUtils has some useful functions
    // for checking shapes.
    OP_REQUIRES(ctx, a_values_tensor.shape() == b_values_tensor.shape(),
                ::tensorflow::errors::InvalidArgument(
                    "a and b must have the same shape. "
                    "a shape: ",
                    a_values_tensor.shape().DebugString(),
                    " b shape: ", b_values_tensor.shape().DebugString()));
    OP_REQUIRES(ctx, a_values_tensor.shape() == cond_tensor.shape(),
                ::tensorflow::errors::InvalidArgument(
                    "a and cond must have the same shape. "
                    "a shape: ",
                    a_values_tensor.shape().DebugString(),
                    " cond shape: ", cond_tensor.shape().DebugString()));
    OP_REQUIRES(ctx, a_values_tensor.NumElements() > 0,
                ::tensorflow::errors::InvalidArgument(
                    "Inputs must have at least one element."));

    const auto a_values = a_values_tensor.flat<T>();
    const auto b_values = b_values_tensor.flat<T>();
    const auto cond = cond_tensor.flat<bool>();

    // Create an output tensor
    Tensor* output_tensor = nullptr;
    OP_REQUIRES_OK(
        ctx, ctx->allocate_output(0, a_values_tensor.shape(), &output_tensor));
    auto output = output_tensor->template flat<T>();
    // Here is an example of processing tensors using the Eigen library.
    // This supports both CPU and GPU.
    // For CPU, it supports chunking into blocks and multi-threading.
    // See
    // https://eigen.tuxfamily.org/dox/unsupported/eigen_tensors.html#title55
    output.device(ctx->eigen_device<Device>()) =
        cond.select(a_values, b_values);
  }
};

}  // namespace custom_op_examples
}  // namespace tensorflow

#endif  // TENSORFLOW_GOOGLE_EXAMPLES_CUSTOM_OPS_DOC_MULTIPLEX_2_MULTIPLEX_2_KERNEL_H_
