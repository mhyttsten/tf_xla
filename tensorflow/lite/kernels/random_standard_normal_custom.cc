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
class MHTracer_DTPStensorflowPSlitePSkernelsPSrandom_standard_normal_customDTcc {
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
   MHTracer_DTPStensorflowPSlitePSkernelsPSrandom_standard_normal_customDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePSkernelsPSrandom_standard_normal_customDTcc() {
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

#include <cmath>
#include <random>

#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/kernel_util.h"

namespace tflite {
namespace ops {
namespace custom {
namespace random_standard_normal {

struct OpData {
  std::default_random_engine rng;
};

namespace {

constexpr int kShapeTensor = 0;
constexpr int kOutputTensor = 0;

// Draws a sample from standard normal distribution.
template <typename T>
TfLiteStatus RandomStandardNormalSample(std::default_random_engine& rng,
                                        T* output, size_t output_size) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSrandom_standard_normal_customDTcc mht_0(mht_0_v, 209, "", "./tensorflow/lite/kernels/random_standard_normal_custom.cc", "RandomStandardNormalSample");

  std::normal_distribution<T> dist;
  std::generate(output, output + output_size, [&]() { return dist(rng); });

  return kTfLiteOk;
}

}  // namespace

void* Init(TfLiteContext* context, const char* buffer, size_t length) {
   std::vector<std::string> mht_1_v;
   mht_1_v.push_back("buffer: \"" + (buffer == nullptr ? std::string("nullptr") : std::string((char*)buffer)) + "\"");
   MHTracer_DTPStensorflowPSlitePSkernelsPSrandom_standard_normal_customDTcc mht_1(mht_1_v, 222, "", "./tensorflow/lite/kernels/random_standard_normal_custom.cc", "Init");

  return new OpData();
}

void Free(TfLiteContext* context, void* buffer) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSrandom_standard_normal_customDTcc mht_2(mht_2_v, 229, "", "./tensorflow/lite/kernels/random_standard_normal_custom.cc", "Free");

  delete reinterpret_cast<OpData*>(buffer);
}

TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSrandom_standard_normal_customDTcc mht_3(mht_3_v, 236, "", "./tensorflow/lite/kernels/random_standard_normal_custom.cc", "Prepare");

  // The seed/seed2 attributes are not handled in this custom op implementation.
  TF_LITE_ENSURE(context, NumInputs(node) == 1);
  TF_LITE_ENSURE_EQ(context, NumOutputs(node), 1);

  const TfLiteTensor* shape = GetInput(context, node, kShapeTensor);
  TF_LITE_ENSURE_EQ(context, shape->type, kTfLiteInt32);
  TF_LITE_ENSURE_EQ(context, NumDimensions(shape), 1);
  TfLiteTensor* output = GetOutput(context, node, kOutputTensor);
  if (!IsConstantTensor(shape)) {
    SetTensorToDynamic(output);
    return kTfLiteOk;
  }
  TfLiteIntArray* output_shape;
  TF_LITE_ENSURE_OK(context,
                    GetOutputShapeFromInput(context, shape, &output_shape));
  return context->ResizeTensor(context, output, output_shape);
}

TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSrandom_standard_normal_customDTcc mht_4(mht_4_v, 258, "", "./tensorflow/lite/kernels/random_standard_normal_custom.cc", "Eval");

  OpData* params = reinterpret_cast<OpData*>(node->user_data);
  TF_LITE_ENSURE(context, params != nullptr);

  TfLiteTensor* output = GetOutput(context, node, kOutputTensor);
  if (IsDynamicTensor(output)) {
    const TfLiteTensor* shape = GetInput(context, node, kShapeTensor);
    TfLiteIntArray* output_shape;
    TF_LITE_ENSURE_OK(context,
                      GetOutputShapeFromInput(context, shape, &output_shape));
    TF_LITE_ENSURE_OK(context,
                      context->ResizeTensor(context, output, output_shape));
  }
  const size_t output_size = NumElements(output);
  switch (output->type) {
    case kTfLiteFloat32:
      RandomStandardNormalSample<float>(
          params->rng, GetTensorData<float>(output), output_size);
      break;
    case kTfLiteFloat64:
      RandomStandardNormalSample<double>(
          params->rng, GetTensorData<double>(output), output_size);
      break;
    default:
      TF_LITE_KERNEL_LOG(
          context, "Unsupported output datatype for RandomStandardNormal: %s",
          TfLiteTypeGetName(output->type));
      return kTfLiteError;
  }
  return kTfLiteOk;
}

}  // namespace random_standard_normal

TfLiteRegistration* Register_RANDOM_STANDARD_NORMAL() {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSrandom_standard_normal_customDTcc mht_5(mht_5_v, 295, "", "./tensorflow/lite/kernels/random_standard_normal_custom.cc", "Register_RANDOM_STANDARD_NORMAL");

  static TfLiteRegistration r = {
      random_standard_normal::Init, random_standard_normal::Free,
      random_standard_normal::Prepare, random_standard_normal::Eval};
  return &r;
}

}  // namespace custom
}  // namespace ops
}  // namespace tflite
