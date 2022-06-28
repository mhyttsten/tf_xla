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
class MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStestingPSinterpreter_utilsDTcc {
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
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStestingPSinterpreter_utilsDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStestingPSinterpreter_utilsDTcc() {
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

/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/lite/delegates/gpu/common/testing/interpreter_utils.h"

#include <cstring>
#include <memory>
#include <string>
#include <vector>

#include "absl/memory/memory.h"
#include "tensorflow/lite/core/api/op_resolver.h"
#include "tensorflow/lite/delegates/gpu/common/status.h"
#include "tensorflow/lite/delegates/gpu/common/tensor.h"
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model.h"
#include "tensorflow/lite/string_type.h"

namespace tflite {
namespace gpu {
namespace testing {

absl::Status InterpreterInvokeWithOpResolver(
    const ::tflite::Model* model, TfLiteDelegate* delegate,
    const OpResolver& op_resolver, const std::vector<TensorFloat32>& inputs,
    std::vector<TensorFloat32>* outputs) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStestingPSinterpreter_utilsDTcc mht_0(mht_0_v, 208, "", "./tensorflow/lite/delegates/gpu/common/testing/interpreter_utils.cc", "InterpreterInvokeWithOpResolver");

  auto interpreter = absl::make_unique<Interpreter>();
  if (InterpreterBuilder(model, op_resolver)(&interpreter) != kTfLiteOk) {
    return absl::InternalError("Unable to create TfLite InterpreterBuilder");
  }
  if (delegate && interpreter->ModifyGraphWithDelegate(delegate) != kTfLiteOk) {
    return absl::InternalError(
        "Unable to modify TfLite graph with the delegate");
  }
  interpreter->SetNumThreads(1);
  if (interpreter->AllocateTensors() != kTfLiteOk) {
    return absl::InternalError("Unable to allocate TfLite tensors");
  }
  for (int i = 0; i < inputs.size(); ++i) {
    DCHECK_EQ(interpreter->tensor(interpreter->inputs()[i])->type,
              kTfLiteFloat32);
    float* tflite_data =
        interpreter->typed_tensor<float>(interpreter->inputs()[i]);
    DCHECK_EQ(inputs[i].data.size() * sizeof(float),
              interpreter->tensor(interpreter->inputs()[i])->bytes);
    std::memcpy(tflite_data, inputs[i].data.data(),
                inputs[i].data.size() * sizeof(float));
  }
  if (interpreter->Invoke() != kTfLiteOk) {
    return absl::InternalError("Unable to invoke TfLite interpreter");
  }
  if (!outputs || !outputs->empty()) {
    return absl::InternalError("Invalid outputs pointer");
  }
  outputs->reserve(interpreter->outputs().size());
  for (auto t : interpreter->outputs()) {
    const TfLiteTensor* out_tensor = interpreter->tensor(t);
    TensorFloat32 bhwc;
    bhwc.id = t;
    // TODO(impjdi) Relax this condition to arbitrary batch size.
    if (out_tensor->dims->data[0] != 1) {
      return absl::InternalError("Batch dimension is expected to be 1");
    }
    bhwc.shape.b = out_tensor->dims->data[0];
    switch (out_tensor->dims->size) {
      case 2:
        bhwc.shape.h = 1;
        bhwc.shape.w = 1;
        bhwc.shape.c = out_tensor->dims->data[1];
        break;
      case 3:
        bhwc.shape.h = 1;
        bhwc.shape.w = out_tensor->dims->data[1];
        bhwc.shape.c = out_tensor->dims->data[2];
        break;
      case 4:
        bhwc.shape.h = out_tensor->dims->data[1];
        bhwc.shape.w = out_tensor->dims->data[2];
        bhwc.shape.c = out_tensor->dims->data[3];
        break;
      default:
        return absl::InternalError("Unsupported dimensions size " +
                                   std::to_string(out_tensor->dims->size));
    }
    bhwc.data = std::vector<float>(
        out_tensor->data.f,
        out_tensor->data.f + out_tensor->bytes / sizeof(float));
    outputs->push_back(bhwc);
  }
  return absl::OkStatus();
}

absl::Status InterpreterInvoke(const ::tflite::Model* model,
                               TfLiteDelegate* delegate,
                               const std::vector<TensorFloat32>& inputs,
                               std::vector<TensorFloat32>* outputs) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStestingPSinterpreter_utilsDTcc mht_1(mht_1_v, 281, "", "./tensorflow/lite/delegates/gpu/common/testing/interpreter_utils.cc", "InterpreterInvoke");

  ops::builtin::BuiltinOpResolver builtin_op_resolver;
  return InterpreterInvokeWithOpResolver(model, delegate, builtin_op_resolver,
                                         inputs, outputs);
}

}  // namespace testing
}  // namespace gpu
}  // namespace tflite
