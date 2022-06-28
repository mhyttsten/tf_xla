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
class MHTracer_DTPStensorflowPSlitePSkernelsPScastDTcc {
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
   MHTracer_DTPStensorflowPSlitePSkernelsPScastDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePSkernelsPScastDTcc() {
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

/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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
#include <complex>

#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/internal/optimized/optimized_ops.h"
#include "tensorflow/lite/kernels/internal/tensor.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/kernels/op_macros.h"

namespace tflite {
namespace ops {
namespace builtin {
namespace cast {
constexpr int kInputTensor = 0;
constexpr int kOutputTensor = 0;

TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPScastDTcc mht_0(mht_0_v, 201, "", "./tensorflow/lite/kernels/cast.cc", "Prepare");

  TF_LITE_ENSURE_EQ(context, NumInputs(node), 1);
  TF_LITE_ENSURE_EQ(context, NumOutputs(node), 1);
  const TfLiteTensor* input;
  TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, kInputTensor, &input));
  TfLiteTensor* output;
  TF_LITE_ENSURE_OK(context,
                    GetOutputSafe(context, node, kOutputTensor, &output));

  // TODO(ahentz): these two checks would make the new implementation
  // incompatible with some existing models, where params is not specified. It
  // is OK not to have them because toco would have set input and output types
  // to match the parameters.
  // auto* params = reinterpret_cast<TfLiteCastParams*>(node->builtin_data);
  // TF_LITE_ENSURE_EQ(context, input->type, params->in_data_type);
  // TF_LITE_ENSURE_EQ(context, output->type, params->out_data_type);

  return context->ResizeTensor(context, output,
                               TfLiteIntArrayCopy(input->dims));
}

template <typename FromT, typename ToT>
void copyCast(const FromT* in, ToT* out, int num_elements) {
  std::transform(in, in + num_elements, out,
                 [](FromT a) { return static_cast<ToT>(a); });
}

template <typename ToT>
void copyCast(const std::complex<float>* in, ToT* out, int num_elements) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPScastDTcc mht_1(mht_1_v, 232, "", "./tensorflow/lite/kernels/cast.cc", "copyCast");

  std::transform(in, in + num_elements, out, [](std::complex<float> a) {
    return static_cast<ToT>(std::real(a));
  });
}

template <>
void copyCast(const std::complex<float>* in, std::complex<float>* out,
              int num_elements) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPScastDTcc mht_2(mht_2_v, 243, "", "./tensorflow/lite/kernels/cast.cc", "copyCast");

  std::transform(in, in + num_elements, out,
                 [](std::complex<float> a) { return a; });
}

template <typename FromT>
TfLiteStatus copyToTensor(TfLiteContext* context, const FromT* in,
                          TfLiteTensor* out, int num_elements) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPScastDTcc mht_3(mht_3_v, 253, "", "./tensorflow/lite/kernels/cast.cc", "copyToTensor");

  switch (out->type) {
    case kTfLiteInt64:
      copyCast(in, out->data.i64, num_elements);
      break;
    case kTfLiteInt32:
      copyCast(in, out->data.i32, num_elements);
      break;
    case kTfLiteUInt32:
      copyCast(in, out->data.u32, num_elements);
      break;
    case kTfLiteInt16:
      copyCast(in, out->data.i16, num_elements);
      break;
    case kTfLiteUInt16:
      copyCast(in, out->data.ui16, num_elements);
      break;
    case kTfLiteUInt8:
      copyCast(in, out->data.uint8, num_elements);
      break;
    case kTfLiteInt8:
      copyCast(in, out->data.int8, num_elements);
      break;
    case kTfLiteFloat32:
      copyCast(in, GetTensorData<float>(out), num_elements);
      break;
    case kTfLiteBool:
      copyCast(in, out->data.b, num_elements);
      break;
    case kTfLiteComplex64:
      copyCast(in, reinterpret_cast<std::complex<float>*>(out->data.c64),
               num_elements);
      break;
    default:
      // Unsupported type.
      TF_LITE_UNSUPPORTED_TYPE(context, out->type, "Cast");
  }
  return kTfLiteOk;
}

TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPScastDTcc mht_4(mht_4_v, 296, "", "./tensorflow/lite/kernels/cast.cc", "Eval");

  const TfLiteTensor* input;
  TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, kInputTensor, &input));
  TfLiteTensor* output;
  TF_LITE_ENSURE_OK(context,
                    GetOutputSafe(context, node, kOutputTensor, &output));
  const int num_elements = NumElements(input);
  TF_LITE_ENSURE_EQ(context, num_elements, NumElements(output));
  switch (input->type) {
    case kTfLiteInt64:
      return copyToTensor(context, input->data.i64, output, num_elements);
    case kTfLiteInt32:
      return copyToTensor(context, input->data.i32, output, num_elements);
    case kTfLiteUInt32:
      return copyToTensor(context, input->data.u32, output, num_elements);
    case kTfLiteUInt16:
      return copyToTensor(context, input->data.ui16, output, num_elements);
    case kTfLiteInt16:
      return copyToTensor(context, input->data.i16, output, num_elements);
    case kTfLiteUInt8:
      return copyToTensor(context, input->data.uint8, output, num_elements);
    case kTfLiteInt8:
      return copyToTensor(context, input->data.int8, output, num_elements);
    case kTfLiteFloat32:
      return copyToTensor(context, GetTensorData<float>(input), output,
                          num_elements);
    case kTfLiteBool:
      return copyToTensor(context, input->data.b, output, num_elements);
    case kTfLiteComplex64:
      return copyToTensor(
          context, reinterpret_cast<std::complex<float>*>(input->data.c64),
          output, num_elements);
    default:
      // Unsupported type.
      TF_LITE_UNSUPPORTED_TYPE(context, input->type, "Cast");
  }
}
}  // namespace cast

TfLiteRegistration* Register_CAST() {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPScastDTcc mht_5(mht_5_v, 338, "", "./tensorflow/lite/kernels/cast.cc", "Register_CAST");

  static TfLiteRegistration r = {nullptr, nullptr, cast::Prepare, cast::Eval};
  return &r;
}

}  // namespace builtin
}  // namespace ops
}  // namespace tflite
