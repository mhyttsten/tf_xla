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
class MHTracer_DTPStensorflowPSlitePSkernelsPScomplex_supportDTcc {
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
   MHTracer_DTPStensorflowPSlitePSkernelsPScomplex_supportDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePSkernelsPScomplex_supportDTcc() {
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

#include <complex>

#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/internal/tensor.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/kernel_util.h"

namespace tflite {
namespace ops {
namespace builtin {
namespace complex {

static const int kInputTensor = 0;
static const int kOutputTensor = 0;

TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPScomplex_supportDTcc mht_0(mht_0_v, 200, "", "./tensorflow/lite/kernels/complex_support.cc", "Prepare");

  TF_LITE_ENSURE_EQ(context, NumInputs(node), 1);
  TF_LITE_ENSURE_EQ(context, NumOutputs(node), 1);

  const TfLiteTensor* input = GetInput(context, node, kInputTensor);

  TF_LITE_ENSURE(context, input->type == kTfLiteComplex64 ||
                              input->type == kTfLiteComplex128);

  TfLiteTensor* output = GetOutput(context, node, kOutputTensor);

  if (input->type == kTfLiteComplex64) {
    TF_LITE_ENSURE_TYPES_EQ(context, output->type, kTfLiteFloat32);
  } else {
    TF_LITE_ENSURE_TYPES_EQ(context, output->type, kTfLiteFloat64);
  }

  TfLiteIntArray* output_shape = TfLiteIntArrayCopy(input->dims);
  return context->ResizeTensor(context, output, output_shape);
}

template <typename T, typename ExtractF>
void ExtractData(const TfLiteTensor* input, ExtractF extract_func,
                 TfLiteTensor* output) {
  const std::complex<T>* input_data = GetTensorData<std::complex<T>>(input);
  T* output_data = GetTensorData<T>(output);
  const int input_size = NumElements(input);
  for (int i = 0; i < input_size; ++i) {
    *output_data++ = extract_func(*input_data++);
  }
}

TfLiteStatus EvalReal(TfLiteContext* context, TfLiteNode* node) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPScomplex_supportDTcc mht_1(mht_1_v, 235, "", "./tensorflow/lite/kernels/complex_support.cc", "EvalReal");

  const TfLiteTensor* input = GetInput(context, node, kInputTensor);

  TfLiteTensor* output = GetOutput(context, node, kOutputTensor);

  switch (input->type) {
    case kTfLiteComplex64: {
      ExtractData<float>(
          input,
          static_cast<float (*)(const std::complex<float>&)>(std::real<float>),
          output);
      break;
    }
    case kTfLiteComplex128: {
      ExtractData<double>(input,
                          static_cast<double (*)(const std::complex<double>&)>(
                              std::real<double>),
                          output);
      break;
    }
    default: {
      TF_LITE_KERNEL_LOG(context,
                         "Unsupported input type, Real op only supports "
                         "complex input, but got: ",
                         TfLiteTypeGetName(input->type));
      return kTfLiteError;
    }
  }

  return kTfLiteOk;
}

TfLiteStatus EvalImag(TfLiteContext* context, TfLiteNode* node) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPScomplex_supportDTcc mht_2(mht_2_v, 270, "", "./tensorflow/lite/kernels/complex_support.cc", "EvalImag");

  const TfLiteTensor* input = GetInput(context, node, kInputTensor);

  TfLiteTensor* output = GetOutput(context, node, kOutputTensor);

  switch (input->type) {
    case kTfLiteComplex64: {
      ExtractData<float>(
          input,
          static_cast<float (*)(const std::complex<float>&)>(std::imag<float>),
          output);
      break;
    }
    case kTfLiteComplex128: {
      ExtractData<double>(input,
                          static_cast<double (*)(const std::complex<double>&)>(
                              std::imag<double>),
                          output);
      break;
    }
    default: {
      TF_LITE_KERNEL_LOG(context,
                         "Unsupported input type, Imag op only supports "
                         "complex input, but got: ",
                         TfLiteTypeGetName(input->type));
      return kTfLiteError;
    }
  }

  return kTfLiteOk;
}

TfLiteStatus EvalAbs(TfLiteContext* context, TfLiteNode* node) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPScomplex_supportDTcc mht_3(mht_3_v, 305, "", "./tensorflow/lite/kernels/complex_support.cc", "EvalAbs");

  const TfLiteTensor* input = GetInput(context, node, kInputTensor);
  TfLiteTensor* output = GetOutput(context, node, kOutputTensor);

  switch (input->type) {
    case kTfLiteComplex64: {
      ExtractData<float>(
          input,
          static_cast<float (*)(const std::complex<float>&)>(std::abs<float>),
          output);
      break;
    }
    case kTfLiteComplex128: {
      ExtractData<double>(input,
                          static_cast<double (*)(const std::complex<double>&)>(
                              std::abs<double>),
                          output);
      break;
    }
    default: {
      TF_LITE_KERNEL_LOG(context,
                         "Unsupported input type, ComplexAbs op only supports "
                         "complex input, but got: ",
                         TfLiteTypeGetName(input->type));
      return kTfLiteError;
    }
  }

  return kTfLiteOk;
}

}  // namespace complex

TfLiteRegistration* Register_REAL() {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPScomplex_supportDTcc mht_4(mht_4_v, 341, "", "./tensorflow/lite/kernels/complex_support.cc", "Register_REAL");

  static TfLiteRegistration r = {/*init=*/nullptr, /*free=*/nullptr,
                                 complex::Prepare, complex::EvalReal};
  return &r;
}

TfLiteRegistration* Register_IMAG() {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPScomplex_supportDTcc mht_5(mht_5_v, 350, "", "./tensorflow/lite/kernels/complex_support.cc", "Register_IMAG");

  static TfLiteRegistration r = {/*init=*/nullptr, /*free=*/nullptr,
                                 complex::Prepare, complex::EvalImag};
  return &r;
}

TfLiteRegistration* Register_COMPLEX_ABS() {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPScomplex_supportDTcc mht_6(mht_6_v, 359, "", "./tensorflow/lite/kernels/complex_support.cc", "Register_COMPLEX_ABS");

  static TfLiteRegistration r = {/*init=*/nullptr, /*free=*/nullptr,
                                 complex::Prepare, complex::EvalAbs};
  return &r;
}

}  // namespace builtin
}  // namespace ops
}  // namespace tflite
