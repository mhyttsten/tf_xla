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
class MHTracer_DTPStensorflowPSlitePSexperimentalPSresourcePSresource_variableDTcc {
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
   MHTracer_DTPStensorflowPSlitePSexperimentalPSresourcePSresource_variableDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePSexperimentalPSresourcePSresource_variableDTcc() {
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

#include "tensorflow/lite/experimental/resource/resource_variable.h"

#include <cstdlib>
#include <cstring>
#include <map>
#include <memory>

#include "tensorflow/lite/c/c_api_types.h"

namespace tflite {
namespace resource {

ResourceVariable::ResourceVariable() {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePSexperimentalPSresourcePSresource_variableDTcc mht_0(mht_0_v, 197, "", "./tensorflow/lite/experimental/resource/resource_variable.cc", "ResourceVariable::ResourceVariable");

  memset(&tensor_, 0, sizeof(TfLiteTensor));
}

ResourceVariable::ResourceVariable(ResourceVariable&& other) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSlitePSexperimentalPSresourcePSresource_variableDTcc mht_1(mht_1_v, 204, "", "./tensorflow/lite/experimental/resource/resource_variable.cc", "ResourceVariable::ResourceVariable");

  tensor_ = other.tensor_;
  is_initialized_ = other.is_initialized_;

  memset(&other.tensor_, 0, sizeof(TfLiteTensor));
  other.is_initialized_ = false;
}

ResourceVariable::~ResourceVariable() {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSlitePSexperimentalPSresourcePSresource_variableDTcc mht_2(mht_2_v, 215, "", "./tensorflow/lite/experimental/resource/resource_variable.cc", "ResourceVariable::~ResourceVariable");

  if (is_initialized_) {
    free(tensor_.data.raw);
    if (tensor_.dims) {
      TfLiteIntArrayFree(tensor_.dims);
    }
  }
}

TfLiteStatus ResourceVariable::AssignFrom(const TfLiteTensor* tensor) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPSlitePSexperimentalPSresourcePSresource_variableDTcc mht_3(mht_3_v, 227, "", "./tensorflow/lite/experimental/resource/resource_variable.cc", "ResourceVariable::AssignFrom");

  // Save the old allocated resources and attributes that we might use.
  char* old_raw = tensor_.data.raw;
  size_t old_bytes = tensor_.bytes;
  TfLiteIntArray* old_dims = tensor_.dims;

  // Copy primitive parameters.
  memset(&tensor_, 0, sizeof(tensor_));
  tensor_.allocation_type = kTfLiteDynamic;
  tensor_.type = tensor->type;
  tensor_.params = tensor->params;
  tensor_.quantization = tensor->quantization;

  // Copy old shape if possible otherwise create a new one.
  if (TfLiteIntArrayEqual(old_dims, tensor->dims)) {
    tensor_.dims = old_dims;
  } else {
    TfLiteIntArrayFree(old_dims);
    tensor_.dims = TfLiteIntArrayCopy(tensor->dims);
  }

  // Reuse the same buffer if possible otherwise allocate a new one.
  tensor_.data.raw = old_raw;
  if (old_bytes != tensor->bytes) {
    TfLiteTensorRealloc(tensor->bytes, &tensor_);
  } else {
    tensor_.bytes = old_bytes;
  }

  memcpy(tensor_.data.raw, tensor->data.raw, tensor_.bytes);
  is_initialized_ = true;

  return kTfLiteOk;
}

void CreateResourceVariableIfNotAvailable(ResourceMap* resources,
                                          int resource_id) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPSlitePSexperimentalPSresourcePSresource_variableDTcc mht_4(mht_4_v, 266, "", "./tensorflow/lite/experimental/resource/resource_variable.cc", "CreateResourceVariableIfNotAvailable");

  if (resources->count(resource_id) != 0) {
    return;
  }
  resources->emplace(resource_id,
                     std::unique_ptr<ResourceVariable>(new ResourceVariable()));
}

ResourceVariable* GetResourceVariable(ResourceMap* resources, int resource_id) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPSlitePSexperimentalPSresourcePSresource_variableDTcc mht_5(mht_5_v, 277, "", "./tensorflow/lite/experimental/resource/resource_variable.cc", "GetResourceVariable");

  auto it = resources->find(resource_id);
  if (it != resources->end()) {
    return static_cast<ResourceVariable*>(it->second.get());
  }
  return nullptr;
}

bool IsBuiltinResource(const TfLiteTensor* tensor) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPSlitePSexperimentalPSresourcePSresource_variableDTcc mht_6(mht_6_v, 288, "", "./tensorflow/lite/experimental/resource/resource_variable.cc", "IsBuiltinResource");

  return tensor && tensor->type == kTfLiteResource &&
         tensor->delegate == nullptr;
}

}  // namespace resource
}  // namespace tflite
