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
class MHTracer_DTPStensorflowPSlitePSkernelsPShashtableDTcc {
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
   MHTracer_DTPStensorflowPSlitePSkernelsPShashtableDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePSkernelsPShashtableDTcc() {
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
#include <string>

#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/core/subgraph.h"
#include "tensorflow/lite/experimental/resource/lookup_interfaces.h"
#include "tensorflow/lite/kernels/kernel_util.h"

namespace tflite {
namespace ops {
namespace builtin {
namespace hashtable {

// The current hash table op returns a key of the hash table resource objects,
// shared by the context. Later, this implementation might be updated by sharing
// the actual reference of hash table objects in the tensor buffer.

static constexpr int kResourceHandleTensor = 0;

TfLiteStatus PrepareHashtable(TfLiteContext* context, TfLiteNode* node) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPShashtableDTcc mht_0(mht_0_v, 203, "", "./tensorflow/lite/kernels/hashtable.cc", "PrepareHashtable");

  TF_LITE_ENSURE_EQ(context, NumInputs(node), 0);
  TF_LITE_ENSURE_EQ(context, NumOutputs(node), 1);

  TF_LITE_ENSURE(context, node->builtin_data != nullptr);
  const auto* params =
      reinterpret_cast<const TfLiteHashtableParams*>(node->builtin_data);

  TF_LITE_ENSURE(context, (params->key_dtype == kTfLiteInt64 &&
                           params->value_dtype == kTfLiteString) ||
                              (params->key_dtype == kTfLiteString &&
                               params->value_dtype == kTfLiteInt64));

  TfLiteTensor* resource_handle_tensor;
  TF_LITE_ENSURE_OK(context, GetOutputSafe(context, node, kResourceHandleTensor,
                                           &resource_handle_tensor));
  TF_LITE_ENSURE_EQ(context, resource_handle_tensor->type, kTfLiteResource);
  size_t bytesRequired = sizeof(int32_t);

  // Realloc space for an integer handle value.
  TfLiteTensorRealloc(bytesRequired, resource_handle_tensor);
  resource_handle_tensor->bytes = bytesRequired;

  TfLiteIntArray* outputSize = TfLiteIntArrayCreate(1);
  outputSize->data[0] = 1;
  if (resource_handle_tensor->dims)
    TfLiteIntArrayFree(resource_handle_tensor->dims);
  resource_handle_tensor->dims = outputSize;

  return kTfLiteOk;
}

TfLiteStatus EvalHashtable(TfLiteContext* context, TfLiteNode* node) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPShashtableDTcc mht_1(mht_1_v, 238, "", "./tensorflow/lite/kernels/hashtable.cc", "EvalHashtable");

  TF_LITE_ENSURE(context, node->builtin_data != nullptr);
  const auto* params =
      reinterpret_cast<const TfLiteHashtableParams*>(node->builtin_data);

  const int32_t resource_id = params->table_id;

  TfLiteTensor* resource_handle_tensor;
  TF_LITE_ENSURE_OK(context, GetOutputSafe(context, node, kResourceHandleTensor,
                                           &resource_handle_tensor));
  *resource_handle_tensor->data.i32 = resource_id;

  Subgraph* subgraph = reinterpret_cast<Subgraph*>(context->impl_);
  auto& resources = subgraph->resources();
  resource::CreateHashtableResourceIfNotAvailable(
      &resources, resource_id, params->key_dtype, params->value_dtype);
  return kTfLiteOk;
}

}  // namespace hashtable

TfLiteRegistration* Register_HASHTABLE() {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPShashtableDTcc mht_2(mht_2_v, 262, "", "./tensorflow/lite/kernels/hashtable.cc", "Register_HASHTABLE");

  static TfLiteRegistration r = {nullptr, nullptr, hashtable::PrepareHashtable,
                                 hashtable::EvalHashtable};
  return &r;
}

}  // namespace builtin
}  // namespace ops
}  // namespace tflite
