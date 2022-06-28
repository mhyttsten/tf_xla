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
class MHTracer_DTPStensorflowPSlitePSkernelsPSatan2DTcc {
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
   MHTracer_DTPStensorflowPSlitePSkernelsPSatan2DTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePSkernelsPSatan2DTcc() {
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

// Copyright 2021 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <cmath>

#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/custom_ops_register.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/kernel_util.h"

namespace tflite {
namespace ops {
namespace custom {
namespace atan2 {

TfLiteStatus EnsureSameShape(
    TfLiteContext* context,
    const TfLiteTensor* a, const TfLiteTensor* b) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSatan2DTcc mht_0(mht_0_v, 198, "", "./tensorflow/lite/kernels/atan2.cc", "EnsureSameShape");

  TF_LITE_ENSURE_EQ(context,
                    tflite::NumDimensions(a),
                    tflite::NumDimensions(b));

  return TfLiteStatus::kTfLiteOk;
}

TfLiteStatus Atan2Prepare(TfLiteContext* context, TfLiteNode* node) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSatan2DTcc mht_1(mht_1_v, 209, "", "./tensorflow/lite/kernels/atan2.cc", "Atan2Prepare");

  TF_LITE_ENSURE_EQ(context, tflite::NumInputs(node), 2);
  TF_LITE_ENSURE_EQ(context, tflite::NumOutputs(node), 1);

  const TfLiteTensor* input_y = tflite::GetInput(context, node, 0);
  const TfLiteTensor* input_x = tflite::GetInput(context, node, 1);
  TfLiteTensor* output = tflite::GetOutput(context, node, 0);

  // Validate size and type constraints
  TF_LITE_ENSURE_OK(context, EnsureSameShape(context, input_y, input_x));
  TF_LITE_ENSURE_TYPES_EQ(context, input_y->type, input_x->type);
  TF_LITE_ENSURE_TYPES_EQ(context, input_y->type, output->type);
  TF_LITE_ENSURE(context,
                 input_y->type == kTfLiteFloat32 ||
                 input_y->type == kTfLiteFloat64);

  TfLiteIntArray* output_shape = TfLiteIntArrayCopy(input_y->dims);

  return context->ResizeTensor(context, output, output_shape);
}

template<typename Float>
TfLiteStatus Atan2(const TfLiteTensor* input_y,
                   const TfLiteTensor* input_x,
                   TfLiteTensor* output) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSatan2DTcc mht_2(mht_2_v, 236, "", "./tensorflow/lite/kernels/atan2.cc", "Atan2");

  const Float* data_y = tflite::GetTensorData<Float>(input_y);
  const Float* data_x = tflite::GetTensorData<Float>(input_x);
  Float* data_output = tflite::GetTensorData<Float>(output);

  const int64_t num_elements = NumElements(input_y);
  for (int64_t i = 0; i < num_elements; ++i) {
    data_output[i] = std::atan2(data_y[i], data_x[i]);
  }

  return TfLiteStatus::kTfLiteOk;
}

TfLiteStatus Atan2Eval(TfLiteContext* context, TfLiteNode* node) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSatan2DTcc mht_3(mht_3_v, 252, "", "./tensorflow/lite/kernels/atan2.cc", "Atan2Eval");

  const TfLiteTensor* input_y = tflite::GetInput(context, node, 0);
  const TfLiteTensor* input_x = tflite::GetInput(context, node, 1);
  TfLiteTensor* output = tflite::GetOutput(context, node, 0);

  switch (output->type) {
    case kTfLiteFloat32:
      TF_LITE_ENSURE_OK(context, Atan2<float>(input_y, input_x, output));
      break;
    case kTfLiteFloat64:
      TF_LITE_ENSURE_OK(context, Atan2<double>(input_y, input_x, output));
      break;
    default:
      TF_LITE_KERNEL_LOG(
          context,
          "Unsupported datatype for atan2 output: %s",
          TfLiteTypeGetName(output->type));
  }

  return TfLiteStatus::kTfLiteOk;
}

}  // namespace atan2

TfLiteRegistration* Register_ATAN2() {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSatan2DTcc mht_4(mht_4_v, 279, "", "./tensorflow/lite/kernels/atan2.cc", "Register_ATAN2");

  static TfLiteRegistration r = {
    nullptr, nullptr, atan2::Atan2Prepare, atan2::Atan2Eval};
  return &r;
}

}  // namespace custom
}  // namespace ops
}  // namespace tflite
