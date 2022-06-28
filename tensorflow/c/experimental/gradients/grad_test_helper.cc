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
class MHTracer_DTPStensorflowPScPSexperimentalPSgradientsPSgrad_test_helperDTcc {
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
   MHTracer_DTPStensorflowPScPSexperimentalPSgradientsPSgrad_test_helperDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScPSexperimentalPSgradientsPSgrad_test_helperDTcc() {
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

/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/c/experimental/gradients/grad_test_helper.h"

#include "tensorflow/c/eager/gradient_checker.h"
#include "tensorflow/c/experimental/gradients/tape/tape_context.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {
namespace gradients {
namespace internal {

void CompareNumericalAndAutodiffGradients(
    Model model, Model grad_model, AbstractContext* ctx,
    absl::Span<AbstractTensorHandle* const> inputs, bool use_function,
    double abs_error) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScPSexperimentalPSgradientsPSgrad_test_helperDTcc mht_0(mht_0_v, 197, "", "./tensorflow/c/experimental/gradients/grad_test_helper.cc", "CompareNumericalAndAutodiffGradients");

  auto num_inputs = inputs.size();
  std::vector<AbstractTensorHandle*> outputs(num_inputs);
  auto s = RunModel(grad_model, ctx, inputs, absl::MakeSpan(outputs),
                    /*use_function=*/use_function);
  ASSERT_EQ(errors::OK, s.code()) << s.error_message();

  for (int i = 0; i < num_inputs; ++i) {
    if (!outputs[i]) continue;

    AbstractTensorHandlePtr numerical_grad;
    {
      AbstractTensorHandle* numerical_grad_raw;
      s = CalcNumericalGrad(ctx, model, inputs,
                            /*input_index=*/i, use_function,
                            &numerical_grad_raw);
      ASSERT_EQ(errors::OK, s.code()) << s.error_message();
      numerical_grad.reset(numerical_grad_raw);
    }

    TF_Tensor* numerical_tensor;
    s = GetValue(numerical_grad.get(), &numerical_tensor);
    ASSERT_EQ(errors::OK, s.code()) << s.error_message();
    auto num_elem_numerical = TF_TensorElementCount(numerical_tensor);

    TF_Tensor* analytical_tensor;
    s = GetValue(outputs[i], &analytical_tensor);
    ASSERT_EQ(errors::OK, s.code()) << s.error_message();
    auto num_elem_analytical = TF_TensorElementCount(analytical_tensor);

    ASSERT_EQ(num_elem_numerical, num_elem_analytical);

    float* dnumerical = new float[num_elem_numerical]{0};
    memcpy(&dnumerical[0], TF_TensorData(numerical_tensor),
           TF_TensorByteSize(numerical_tensor));
    float* danalytical = new float[num_elem_analytical]{0};
    memcpy(&danalytical[0], TF_TensorData(analytical_tensor),
           TF_TensorByteSize(analytical_tensor));

    for (int j = 0; j < num_elem_numerical; j++) {
      ASSERT_NEAR(dnumerical[j], danalytical[j], abs_error);
    }
    TF_DeleteTensor(analytical_tensor);
    TF_DeleteTensor(numerical_tensor);
    delete[] danalytical;
    delete[] dnumerical;
    outputs[i]->Unref();
  }
}

void CheckTensorValue(AbstractTensorHandle* t, absl::Span<const float> manuals,
                      absl::Span<const int64_t> dims, double abs_error) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScPSexperimentalPSgradientsPSgrad_test_helperDTcc mht_1(mht_1_v, 251, "", "./tensorflow/c/experimental/gradients/grad_test_helper.cc", "CheckTensorValue");

  TF_Tensor* analytical_tensor;
  auto s = GetValue(t, &analytical_tensor);
  ASSERT_EQ(errors::OK, s.code()) << s.error_message();

  int64_t num_elem_analytical = 1;
  auto num_dims_analytical = TF_NumDims(analytical_tensor);
  ASSERT_EQ(dims.size(), num_dims_analytical);
  for (int j = 0; j < num_dims_analytical; j++) {
    auto dim_analytical = TF_Dim(analytical_tensor, j);
    ASSERT_EQ(dims[j], dim_analytical);
    num_elem_analytical *= dim_analytical;
  }

  float* danalytical = new float[num_elem_analytical]{0};
  memcpy(&danalytical[0], TF_TensorData(analytical_tensor),
         TF_TensorByteSize(analytical_tensor));

  for (int64_t j = 0; j < num_elem_analytical; j++) {
    if (abs_error == 0) {
      ASSERT_EQ(manuals[j], danalytical[j]);
    } else {
      ASSERT_NEAR(manuals[j], danalytical[j], abs_error);
    }
  }

  TF_DeleteTensor(analytical_tensor);
  delete[] danalytical;
}

Model BuildGradModel(Model forward, GradientRegistry registry) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScPSexperimentalPSgradientsPSgrad_test_helperDTcc mht_2(mht_2_v, 284, "", "./tensorflow/c/experimental/gradients/grad_test_helper.cc", "BuildGradModel");

  return [forward_model = std::move(forward),
          grad_registry = std::move(registry)](
             AbstractContext* ctx,
             absl::Span<AbstractTensorHandle* const> inputs,
             absl::Span<AbstractTensorHandle*> outputs) -> Status {
    Tape tape(/*persistent=*/false);
    for (size_t i{}; i < inputs.size(); ++i) {
      tape.Watch(inputs[i]);
    }
    std::vector<AbstractTensorHandle*> temp_outputs(1);
    AbstractContextPtr tape_ctx(new TapeContext(ctx, &tape, grad_registry));
    TF_RETURN_IF_ERROR(
        forward_model(tape_ctx.get(), inputs, absl::MakeSpan(temp_outputs)));

    TF_RETURN_IF_ERROR(tape.ComputeGradient(ctx, /*targets=*/temp_outputs,
                                            /*sources=*/inputs,
                                            /*output_gradients=*/{}, outputs));
    for (auto temp_output : temp_outputs) {
      temp_output->Unref();
    }
    return Status::OK();
  };
}

}  // namespace internal
}  // namespace gradients
}  // namespace tensorflow
