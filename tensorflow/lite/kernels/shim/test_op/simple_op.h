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
#ifndef TENSORFLOW_LITE_KERNELS_SHIM_TEST_OP_SIMPLE_OP_H_
#define TENSORFLOW_LITE_KERNELS_SHIM_TEST_OP_SIMPLE_OP_H_
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
class MHTracer_DTPStensorflowPSlitePSkernelsPSshimPStest_opPSsimple_opDTh {
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
   MHTracer_DTPStensorflowPSlitePSkernelsPSshimPStest_opPSsimple_opDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePSkernelsPSshimPStest_opPSsimple_opDTh() {
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


#include <algorithm>
#include <cstdint>
#include <string>

#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "tensorflow/lite/kernels/shim/op_kernel.h"
#include "tensorflow/lite/kernels/shim/status_macros.h"
#include "tensorflow/lite/kernels/shim/tensor_view.h"

namespace tflite {
namespace shim {

// A simple operation for demonstration and testing purposes.
// See the kDoc member for documentation.

template <Runtime Rt>
class SimpleOp : public OpKernelShim<SimpleOp, Rt> {
 protected:
  enum Inputs { kInput0 = 0, kInput1 };
  enum Outputs { kOutput0 = 0, kOutput1, kOutput2, kOutput3 };
  int64_t output1_size_;
  std::string output2_suffix_;
  int64_t n_;
  static constexpr int kOutput0Size = 5;
  static const char kOutput1SizeAttr[];

 public:
  using typename OpKernelShim<SimpleOp, Rt>::InitContext;
  using typename OpKernelShim<SimpleOp, Rt>::InvokeContext;
  using typename OpKernelShim<SimpleOp, Rt>::ShapeInferenceContext;

  SimpleOp() = default;
  static const char kOpName[];
  static const char kDoc[];

  // Attributes declaration (syntax: https://www.tensorflow.org/guide/create_op)
  static std::vector<std::string> Attrs() {
    return {absl::StrCat(kOutput1SizeAttr, ": int"), "output2_suffix: string",
            "N: int >= 0"};
  }
  // Input tensors declaration (syntax:
  // https://www.tensorflow.org/guide/create_op)
  static std::vector<std::string> Inputs() {
    return {"in0: string", "in1: N*int64"};
  }
  // Output tensors declaration (syntax:
  // https://www.tensorflow.org/guide/create_op)
  static std::vector<std::string> Outputs() {
    return {"out0: int32", "out1: float", "out2: string", "out3: N*int64"};
  }

  // Initializes the op
  absl::Status Init(InitContext* ctx) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSshimPStest_opPSsimple_opDTh mht_0(mht_0_v, 241, "", "./tensorflow/lite/kernels/shim/test_op/simple_op.h", "Init");

    SH_RETURN_IF_ERROR(ctx->GetAttr(kOutput1SizeAttr, &output1_size_));
    if (output1_size_ < 1) {
      return absl::InternalError(
          absl::StrCat(kOutput1SizeAttr, " should be >= 1"));
    }
    SH_RETURN_IF_ERROR(ctx->GetAttr("N", &n_));
    absl::string_view output2_suffix;
    SH_RETURN_IF_ERROR(ctx->GetAttr("output2_suffix", &output2_suffix));
    output2_suffix_ = std::string(output2_suffix);
    return absl::OkStatus();
  }

  // Runs the operation
  absl::Status Invoke(InvokeContext* ctx) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSshimPStest_opPSsimple_opDTh mht_1(mht_1_v, 258, "", "./tensorflow/lite/kernels/shim/test_op/simple_op.h", "Invoke");

    using std::int32_t;
    // read input
    SH_ASSIGN_OR_RETURN(const auto input_t, ctx->GetInput(kInput0));
    const auto input_str = input_t->template AsScalar<::tensorflow::tstring>();
    // output0 whose size is static
    SH_ASSIGN_OR_RETURN(auto output0_t,
                        ctx->GetOutput(kOutput0, Shape({kOutput0Size})));
    auto output0 = output0_t->template As<int32_t, 1>();
    for (int i = 0; i < output0.Dim(0); ++i) output0(i) = i;
    // output1 whose size is based on the attr
    SH_ASSIGN_OR_RETURN(
        auto output1_t,
        ctx->GetOutput(kOutput1, Shape({static_cast<int>(output1_size_)})));
    auto output1 = output1_t->template As<float, 1>();
    for (int i = 0; i < output1.Dim(0); ++i) output1(i) = 0.5 * i;
    // output2 whose size is based on input
    const int output2_size = input_str.length() + 1;
    SH_ASSIGN_OR_RETURN(auto output2_t,
                        ctx->GetOutput(kOutput2, Shape({output2_size})));
    auto output2 = output2_t->template As<tensorflow::tstring, 1>();
    for (int i = 0; i < output2.Dim(0) - 1; ++i) output2(i) = std::to_string(i);
    output2(output2.Dim(0) - 1) = output2_suffix_;
    // output3 which is a list of length N
    // The values in output3 are element wise equal to input2 + 1.
    if (ctx->NumInputs() < kInput1 + n_) {
      return absl::InternalError(absl::StrCat(
          "out of bounds: num_inputs=", ctx->NumInputs(), " N=", n_));
    }
    if (ctx->NumOutputs() < kOutput3 + n_) {
      return absl::InternalError(absl::StrCat(
          "out of bounds: num_outputs=", ctx->NumOutputs(), " N=", n_));
    }
    for (int i = 0; i < n_; ++i) {
      SH_ASSIGN_OR_RETURN(const auto input_t, ctx->GetInput(kInput1 + i));
      Shape output_shape(input_t->Shape());
      SH_ASSIGN_OR_RETURN(auto output_t,
                          ctx->GetOutput(kOutput3 + i, output_shape));
      const auto input_data = input_t->template Data<int64_t>();
      auto output_buffer = output_t->template Data<int64_t>().data();
      std::copy(input_data.begin(), input_data.end(), output_buffer);
      // Increment the values of the output
      for (auto& v : output_t->template Data<int64_t>()) ++v;
    }
    return absl::OkStatus();
  }

  // Shape inference
  static absl::Status ShapeInference(ShapeInferenceContext* ctx) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSshimPStest_opPSsimple_opDTh mht_2(mht_2_v, 309, "", "./tensorflow/lite/kernels/shim/test_op/simple_op.h", "ShapeInference");

    // outpu0
    SH_RETURN_IF_ERROR(ctx->SetOutputShape(kOutput0, Shape({kOutput0Size})));
    // output1
    SH_RETURN_IF_ERROR(
        ctx->SetOutputShape(kOutput1, Shape({Shape::kUnknownDim})));
    // output2
    const auto input_t_or = ctx->GetInputTensor(kInput0);
    Shape output2_shape;
    if (input_t_or.ok()) {
      const auto& input_t = input_t_or.value();
      const auto input_str =
          input_t->template AsScalar<::tensorflow::tstring>();
      output2_shape = Shape({static_cast<int>(input_str.length() + 1)});
    } else {
      output2_shape = Shape({Shape::kUnknownDim});
    }
    SH_RETURN_IF_ERROR(ctx->SetOutputShape(kOutput2, output2_shape));
    // output3
    for (int i = kOutput3; i < ctx->NumOutputs(); ++i) {
      SH_RETURN_IF_ERROR(ctx->SetOutputShape(kOutput3, Shape()));
    }
    int64_t n;
    SH_RETURN_IF_ERROR(ctx->GetAttr("N", &n));
    if (n + 1 != ctx->NumInputs()) {
      return absl::InternalError(absl::StrCat("n + 1 != num_inputs: ", n + 1,
                                              " != ", ctx->NumInputs()));
    }
    if (n + 3 != ctx->NumOutputs()) {
      return absl::InternalError(absl::StrCat("n + 1 != num_inputs: ", n + 1,
                                              " != ", ctx->NumOutputs()));
    }
    return absl::OkStatus();
  }
};

// Static member definitions.
// These can be inlined once the toolchain is bumped up to C++17

template <Runtime Rt>
const char SimpleOp<Rt>::kOutput1SizeAttr[] = "output1_size";

template <Runtime Rt>
const char SimpleOp<Rt>::kOpName[] = "SimpleOperation";

template <Runtime Rt>
const char SimpleOp<Rt>::kDoc[] = R"doc(
Description:
  Simple example op for testing and demonstration purposes.

Attrs
  output1_size: int - the size of the second output
  output2_suffix: string - the string value to be appended to the end of out2
  N: int - the number of tensors for the second input and last output
Inputs
  in0: str, shape=[] - A scalar input
  in1: int64, list<shape=?> - A list of tensors as input
Outputs
  out0: int, shape=[5] - first output
  out1: float, shape=[?] - second output
  out2: string, shape=[?] - third output
  out3: int64, list<shape=?> - fourth output that is in1 but incremented.
)doc";

}  // namespace shim
}  // namespace tflite

#endif  // TENSORFLOW_LITE_KERNELS_SHIM_TEST_OP_SIMPLE_OP_H_
