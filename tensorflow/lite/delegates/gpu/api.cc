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
class MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSapiDTcc {
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
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSapiDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSapiDTcc() {
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

#include "tensorflow/lite/delegates/gpu/api.h"

namespace tflite {
namespace gpu {
namespace {

struct ObjectTypeGetter {
  ObjectType operator()(absl::monostate) const { return ObjectType::UNKNOWN; }
  ObjectType operator()(OpenGlBuffer) const { return ObjectType::OPENGL_SSBO; }
  ObjectType operator()(OpenGlTexture) const {
    return ObjectType::OPENGL_TEXTURE;
  }
  ObjectType operator()(OpenClBuffer) const {
    return ObjectType::OPENCL_BUFFER;
  }
  ObjectType operator()(OpenClTexture) const {
    return ObjectType::OPENCL_TEXTURE;
  }
  ObjectType operator()(VulkanBuffer) const {
    return ObjectType::VULKAN_BUFFER;
  }
  ObjectType operator()(VulkanTexture) const {
    return ObjectType::VULKAN_TEXTURE;
  }
  ObjectType operator()(CpuMemory) const { return ObjectType::CPU_MEMORY; }
};

struct ObjectValidityChecker {
  bool operator()(absl::monostate) const { return false; }
  bool operator()(OpenGlBuffer obj) const { return obj.id != GL_INVALID_INDEX; }
  bool operator()(OpenGlTexture obj) const {
    return obj.id != GL_INVALID_INDEX && obj.format != GL_INVALID_ENUM;
  }
  bool operator()(OpenClBuffer obj) const { return obj.memobj; }
  bool operator()(OpenClTexture obj) const { return obj.memobj; }
  bool operator()(VulkanBuffer obj) const { return obj.memory; }
  bool operator()(VulkanTexture obj) const { return obj.memory; }
  bool operator()(CpuMemory obj) const {
    return obj.data != nullptr && obj.size_bytes > 0 &&
           (data_type == DataType::UNKNOWN ||
            obj.size_bytes % SizeOf(data_type) == 0);
  }
  DataType data_type;
};

}  // namespace

bool IsValid(const ObjectDef& def) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSapiDTcc mht_0(mht_0_v, 232, "", "./tensorflow/lite/delegates/gpu/api.cc", "IsValid");

  return def.data_type != DataType::UNKNOWN &&
         def.data_layout != DataLayout::UNKNOWN &&
         def.object_type != ObjectType::UNKNOWN;
}

ObjectType GetType(const TensorObject& object) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSapiDTcc mht_1(mht_1_v, 241, "", "./tensorflow/lite/delegates/gpu/api.cc", "GetType");

  return absl::visit(ObjectTypeGetter{}, object);
}

bool IsValid(const TensorObjectDef& def) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSapiDTcc mht_2(mht_2_v, 248, "", "./tensorflow/lite/delegates/gpu/api.cc", "IsValid");
 return IsValid(def.object_def); }

bool IsValid(const TensorObjectDef& def, const TensorObject& object) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSapiDTcc mht_3(mht_3_v, 253, "", "./tensorflow/lite/delegates/gpu/api.cc", "IsValid");

  return GetType(object) == def.object_def.object_type &&
         absl::visit(ObjectValidityChecker{def.object_def.data_type}, object);
}

bool IsObjectPresent(ObjectType type, const TensorObject& obj) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSapiDTcc mht_4(mht_4_v, 261, "", "./tensorflow/lite/delegates/gpu/api.cc", "IsObjectPresent");

  switch (type) {
    case ObjectType::CPU_MEMORY:
      return absl::holds_alternative<CpuMemory>(obj);
    case ObjectType::OPENGL_SSBO:
      return absl::holds_alternative<OpenGlBuffer>(obj);
    case ObjectType::OPENGL_TEXTURE:
      return absl::holds_alternative<OpenGlTexture>(obj);
    case ObjectType::OPENCL_BUFFER:
      return absl::holds_alternative<OpenClBuffer>(obj);
    case ObjectType::OPENCL_TEXTURE:
      return absl::holds_alternative<OpenClTexture>(obj);
    case ObjectType::VULKAN_BUFFER:
      return absl::holds_alternative<VulkanBuffer>(obj);
    case ObjectType::VULKAN_TEXTURE:
      return absl::holds_alternative<VulkanTexture>(obj);
    case ObjectType::UNKNOWN:
      return false;
  }
}

bool IsObjectInitialized(const TensorObject& obj) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSapiDTcc mht_5(mht_5_v, 285, "", "./tensorflow/lite/delegates/gpu/api.cc", "IsObjectInitialized");

  return GetType(obj) != ObjectType::UNKNOWN;
}

uint32_t NumElements(const TensorObjectDef& def) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSapiDTcc mht_6(mht_6_v, 292, "", "./tensorflow/lite/delegates/gpu/api.cc", "NumElements");

  const auto& d = def.dimensions;
  switch (def.object_def.data_layout) {
    case DataLayout::BHWC:
      return d.product();
    case DataLayout::HWDC4:
    case DataLayout::HDWC4:
    case DataLayout::DHWC4:
      return d.b * d.h * d.w * AlignByN(d.c, 4);
    case DataLayout::UNKNOWN:
      return 0;
  }
  return 0;
}

int GetPosition(const InferenceOptions& options, InferencePriority p) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSapiDTcc mht_7(mht_7_v, 310, "", "./tensorflow/lite/delegates/gpu/api.cc", "GetPosition");

  if (options.priority1 == p) return 1;
  if (options.priority2 == p) return 2;
  if (options.priority3 == p) return 3;
  return 4;  // least important
}

PriorityImportance GetRelativeImportance(const InferenceOptions& options,
                                         InferencePriority p1,
                                         InferencePriority p2) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSapiDTcc mht_8(mht_8_v, 322, "", "./tensorflow/lite/delegates/gpu/api.cc", "GetRelativeImportance");

  int p1_position = GetPosition(options, p1);
  int p2_position = GetPosition(options, p2);
  if (p1_position == p2_position) return PriorityImportance::UNKNOWN;
  return p1_position < p2_position ? PriorityImportance::HIGHER
                                   : PriorityImportance::LOWER;
}

bool IsValid(const InferenceOptions& options) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSapiDTcc mht_9(mht_9_v, 333, "", "./tensorflow/lite/delegates/gpu/api.cc", "IsValid");

  if (options.usage == InferenceUsage::UNKNOWN) {
    return false;
  }
  if (options.priority1 == InferencePriority::UNKNOWN ||
      options.priority2 == InferencePriority::UNKNOWN ||
      options.priority3 == InferencePriority::UNKNOWN) {
    return false;
  }
  if (options.priority1 == InferencePriority::AUTO) {
    return false;
  }
  if (options.priority2 == InferencePriority::AUTO &&
      options.priority3 != InferencePriority::AUTO) {
    return false;
  }
  if (options.priority1 == options.priority2 ||
      options.priority1 == options.priority3) {
    return false;
  }
  if (options.priority2 == options.priority3 &&
      options.priority2 != InferencePriority::AUTO) {
    return false;
  }
  return true;
}

// Implementation note: this resolution logic is shared between GL and CL
// backends, but they might have own logic. Thus, the function is defined
// here just for code re-use purposes.
void ResolveAutoPriority(InferenceOptions* options) {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSapiDTcc mht_10(mht_10_v, 366, "", "./tensorflow/lite/delegates/gpu/api.cc", "ResolveAutoPriority");

  // priority1 can not be AUTO as it would make options invalid.
  if (options->priority2 == InferencePriority::AUTO) {
    switch (options->priority1) {
      case InferencePriority::MIN_LATENCY:
        options->priority2 = InferencePriority::MIN_MEMORY_USAGE;
        options->priority3 = InferencePriority::MAX_PRECISION;
        return;
      case InferencePriority::MIN_MEMORY_USAGE:
        options->priority2 = InferencePriority::MAX_PRECISION;
        options->priority3 = InferencePriority::MIN_LATENCY;
        return;
      case InferencePriority::MAX_PRECISION:
        options->priority2 = InferencePriority::MIN_LATENCY;
        options->priority3 = InferencePriority::MIN_MEMORY_USAGE;
        return;
      case InferencePriority::UNKNOWN:
      case InferencePriority::AUTO:
        // Invalid and unreachable option.
        return;
    }
  }

  if (options->priority3 == InferencePriority::AUTO) {
    // Simply add missing priority
    if (GetPosition(*options, InferencePriority::MIN_LATENCY) == 4) {
      options->priority3 = InferencePriority::MIN_LATENCY;
    } else if (GetPosition(*options, InferencePriority::MAX_PRECISION) == 4) {
      options->priority3 = InferencePriority::MAX_PRECISION;
    } else if (GetPosition(*options, InferencePriority::MIN_MEMORY_USAGE) ==
               4) {
      options->priority3 = InferencePriority::MIN_MEMORY_USAGE;
    }
  }
}

}  // namespace gpu
}  // namespace tflite
