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
class MHTracer_DTPStensorflowPScPSeagerPSgradient_checkerDTcc {
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
   MHTracer_DTPStensorflowPScPSeagerPSgradient_checkerDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScPSeagerPSgradient_checkerDTcc() {
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
#include "tensorflow/c/eager/gradient_checker.h"

#include <memory>

#include "absl/types/span.h"
#include "tensorflow/c/eager/abstract_tensor_handle.h"
#include "tensorflow/c/experimental/ops/math_ops.h"
#include "tensorflow/c/tf_tensor.h"

namespace tensorflow {
namespace gradients {

using namespace std;

// ================== Helper functions =================

// Fills data with values [start,end) with given step size.
void Range(vector<int32_t>* data, int32_t start, int32_t end,
           int32_t step = 1) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScPSeagerPSgradient_checkerDTcc mht_0(mht_0_v, 202, "", "./tensorflow/c/eager/gradient_checker.cc", "Range");

  for (int32_t i = start; i < end; i += step) {
    (*data)[i] = i;
  }
}

// Fills out_dims with the dimensions of the given tensor.
void GetDims(const TF_Tensor* t, int64_t* out_dims) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScPSeagerPSgradient_checkerDTcc mht_1(mht_1_v, 212, "", "./tensorflow/c/eager/gradient_checker.cc", "GetDims");

  int num_dims = TF_NumDims(t);
  for (int i = 0; i < num_dims; i++) {
    out_dims[i] = TF_Dim(t, i);
  }
}

// Runs model as is if output is a scalar,
// else sums the output tensor before returning.
Status RunAndMaybeSum(AbstractContext* ctx, Model forward,
                      absl::Span<AbstractTensorHandle* const> inputs,
                      absl::Span<AbstractTensorHandle*> outputs,
                      bool use_function) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScPSeagerPSgradient_checkerDTcc mht_2(mht_2_v, 227, "", "./tensorflow/c/eager/gradient_checker.cc", "RunAndMaybeSum");

  AbstractTensorHandle* model_outputs[1];

  // Run the model.
  TF_RETURN_IF_ERROR(
      RunModel(forward, ctx, inputs, model_outputs, use_function));
  AbstractTensorHandlePtr model_out(model_outputs[0]);

  TF_Tensor* model_out_tensor;
  TF_RETURN_IF_ERROR(GetValue(model_out.get(), &model_out_tensor));
  int num_dims_out = TF_NumDims(model_out_tensor);
  TF_DeleteTensor(model_out_tensor);

  // If the output is a scalar, then return the scalar output
  if (num_dims_out == 0) {
    outputs[0] = model_out.release();
    return Status::OK();
  }

  // Else, reduce sum the output to get a scalar

  // Will sum all dimensions, so get a Tensor containing [0,...,num_dims_out-1].
  AbstractTensorHandlePtr sum_dims;
  {
    vector<int32_t> vals(num_dims_out);
    int64_t vals_shape[] = {num_dims_out};
    Range(&vals, 0, num_dims_out);
    AbstractTensorHandle* sum_dims_raw = nullptr;
    TF_RETURN_IF_ERROR(TestTensorHandleWithDims<int32_t, TF_INT32>(
        ctx, vals.data(), vals_shape, 1, &sum_dims_raw));
    sum_dims.reset(sum_dims_raw);
  }

  // Reduce sum the output on all dimensions.
  TF_RETURN_IF_ERROR(ops::Sum(ctx, model_out.get(), sum_dims.get(), &outputs[0],
                              /*keep_dims=*/false, "sum_output"));
  return Status::OK();
}
// ========================= End Helper Functions==============================

Status CalcNumericalGrad(AbstractContext* ctx, Model forward,
                         absl::Span<AbstractTensorHandle* const> inputs,
                         int input_index, bool use_function,
                         AbstractTensorHandle** numerical_grad) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScPSeagerPSgradient_checkerDTcc mht_3(mht_3_v, 273, "", "./tensorflow/c/eager/gradient_checker.cc", "CalcNumericalGrad");

  vector<AbstractTensorHandle*> theta_inputs(inputs.size());
  for (int i{}; i < inputs.size(); ++i) {
    theta_inputs[i] = inputs[i];
  }

  AbstractTensorHandle* theta =
      theta_inputs[input_index];  // parameter we are grad checking

  // Convert from AbstractTensor to TF_Tensor.
  TF_Tensor* theta_tensor;
  TF_RETURN_IF_ERROR(GetValue(theta, &theta_tensor));

  // Get number of elements and fill data.
  int num_elems = TF_TensorElementCount(theta_tensor);
  vector<float> theta_data(num_elems);
  memcpy(theta_data.data(), TF_TensorData(theta_tensor),
         TF_TensorByteSize(theta_tensor));

  // Initialize space for the numerical gradient.
  vector<float> dtheta_approx(num_elems);

  // Get theta shape and store in theta_dims.
  int num_dims = TF_NumDims(theta_tensor);
  vector<int64_t> theta_dims(num_dims);
  GetDims(theta_tensor, theta_dims.data());

  // Initialize auxilary data structures.
  vector<float> thetaPlus_data(num_elems);
  vector<float> thetaMinus_data(num_elems);
  AbstractTensorHandle* f_outputs[1];

  // Numerical Grad Check
  for (int i = 0; i < num_elems; i++) {
    // Get relative epsilon value
    float epsilon = theta_data[i] == 0 ? 1e-4 : std::abs(theta_data[i] * 1e-4);
    AbstractTensorHandlePtr two_eps;
    {
      AbstractTensorHandle* two_eps_raw = nullptr;
      TF_RETURN_IF_ERROR(TestScalarTensorHandle<float, TF_FLOAT>(
          ctx, 2 * epsilon, &two_eps_raw));
      two_eps.reset(two_eps_raw);
    }

    // Initialize theta[i] + epsilon.
    memcpy(thetaPlus_data.data(), TF_TensorData(theta_tensor),
           TF_TensorByteSize(theta_tensor));
    thetaPlus_data[i] += epsilon;
    AbstractTensorHandlePtr thetaPlus;
    {
      AbstractTensorHandle* thetaPlus_raw = nullptr;
      TF_RETURN_IF_ERROR(TestTensorHandleWithDims<float, TF_FLOAT>(
          ctx, thetaPlus_data.data(), theta_dims.data(), num_dims,
          &thetaPlus_raw));
      thetaPlus.reset(thetaPlus_raw);
    }

    // Initialize theta[i] - epsilon.
    memcpy(&thetaMinus_data[0], TF_TensorData(theta_tensor),
           TF_TensorByteSize(theta_tensor));
    thetaMinus_data[i] -= epsilon;
    AbstractTensorHandlePtr thetaMinus;
    {
      AbstractTensorHandle* thetaMinus_raw = nullptr;
      TF_RETURN_IF_ERROR(TestTensorHandleWithDims<float, TF_FLOAT>(
          ctx, thetaMinus_data.data(), theta_dims.data(), num_dims,
          &thetaMinus_raw));
      thetaMinus.reset(thetaMinus_raw);
    }

    // Get f(theta + eps):
    theta_inputs[input_index] = thetaPlus.get();
    TF_RETURN_IF_ERROR(
        RunAndMaybeSum(ctx, forward, theta_inputs, f_outputs, use_function));
    AbstractTensorHandlePtr fPlus(f_outputs[0]);

    // Get f(theta - eps):
    theta_inputs[input_index] = thetaMinus.get();
    TF_RETURN_IF_ERROR(
        RunAndMaybeSum(ctx, forward, theta_inputs, f_outputs, use_function));
    AbstractTensorHandlePtr fMinus(f_outputs[0]);

    // Take Difference of both estimates: (f(theta + eps) - f(theta - eps)).
    TF_RETURN_IF_ERROR(
        ops::Sub(ctx, fPlus.get(), fMinus.get(), f_outputs, "sub_top"));
    AbstractTensorHandlePtr fDiff(f_outputs[0]);

    // Calculate using the difference quotient definition:
    // (f(theta + eps) - f(theta - eps)) / (2 * eps).
    TF_RETURN_IF_ERROR(
        ops::Div(ctx, fDiff.get(), two_eps.get(), f_outputs, "diff_quotient"));
    AbstractTensorHandlePtr diff_quotient(f_outputs[0]);

    TF_Tensor* grad_tensor;
    TF_RETURN_IF_ERROR(GetValue(diff_quotient.get(), &grad_tensor));
    float grad_data[1];
    memcpy(&grad_data[0], TF_TensorData(grad_tensor),
           TF_TensorByteSize(grad_tensor));
    TF_DeleteTensor(grad_tensor);
    dtheta_approx[i] = grad_data[0];
  }

  // Populate *numerical_grad with the data from dtheta_approx.
  TF_RETURN_IF_ERROR(TestTensorHandleWithDims<float, TF_FLOAT>(
      ctx, dtheta_approx.data(), theta_dims.data(), num_dims, numerical_grad));
  TF_DeleteTensor(theta_tensor);
  return Status::OK();
}

}  // namespace gradients
}  // namespace tensorflow
