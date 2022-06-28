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
class MHTracer_DTPStensorflowPScorePScommon_runtimePSpluggable_devicePSpluggable_device_bfc_allocatorDTcc {
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
   MHTracer_DTPStensorflowPScorePScommon_runtimePSpluggable_devicePSpluggable_device_bfc_allocatorDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePScommon_runtimePSpluggable_devicePSpluggable_device_bfc_allocatorDTcc() {
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

/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/common_runtime/pluggable_device/pluggable_device_bfc_allocator.h"

#include "tensorflow/core/lib/strings/strcat.h"

namespace tensorflow {

bool PluggableDeviceBFCAllocator::GetAllowGrowthValue(
    const GPUOptions& gpu_options, bool force_memory_growth_requested) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSpluggable_devicePSpluggable_device_bfc_allocatorDTcc mht_0(mht_0_v, 192, "", "./tensorflow/core/common_runtime/pluggable_device/pluggable_device_bfc_allocator.cc", "PluggableDeviceBFCAllocator::GetAllowGrowthValue");

  const char* force_allow_growth_string =
      std::getenv("TF_FORCE_GPU_ALLOW_GROWTH");
  if (force_allow_growth_string == nullptr) {
    if (force_memory_growth_requested && !gpu_options.allow_growth()) {
      LOG(WARNING) << "Overriding allow_growth setting because "
                      "force_memory_growth was requested by the device.";
      return true;
    }

    return gpu_options.allow_growth();
  }

  if (force_memory_growth_requested) {
    LOG(WARNING) << "Ignoring the value of TF_FORCE_GPU_ALLOW_GROWTH because "
                    "force_memory_growth was requested by the device.";
    return true;
  }

  if (strcmp("false", force_allow_growth_string) == 0) {
    if (gpu_options.allow_growth()) {
      LOG(WARNING)
          << "Overriding allow_growth setting because the"
          << " TF_FORCE_GPU_ALLOW_GROWTH environment variable is set. Original"
          << " config value was " << gpu_options.allow_growth() << ".";
    }
    return false;
  } else if (strcmp("true", force_allow_growth_string) == 0) {
    if (!gpu_options.allow_growth()) {
      LOG(WARNING)
          << "Overriding allow_growth setting because the"
          << " TF_FORCE_GPU_ALLOW_GROWTH environment variable is set. Original"
          << " config value was " << gpu_options.allow_growth() << ".";
    }
    return true;
  }

  LOG(ERROR)
      << "The TF_FORCE_GPU_ALLOW_GROWTH environment variable is set but could"
      << " not be parsed: \"" << force_allow_growth_string << "\". Valid"
      << " values are \"true\" or \"false\". Using original config value"
      << " of " << gpu_options.allow_growth() << ".";
  return gpu_options.allow_growth();
}

bool PluggableDeviceBFCAllocator::GetGarbageCollectionValue() {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSpluggable_devicePSpluggable_device_bfc_allocatorDTcc mht_1(mht_1_v, 240, "", "./tensorflow/core/common_runtime/pluggable_device/pluggable_device_bfc_allocator.cc", "PluggableDeviceBFCAllocator::GetGarbageCollectionValue");

  const char* enable_gpu_garbage_collection =
      std::getenv("TF_ENABLE_GPU_GARBAGE_COLLECTION");
  if (enable_gpu_garbage_collection == nullptr) {
    // By default, turn on the memory garbage collection
    return true;
  }
  if (strcmp("false", enable_gpu_garbage_collection) == 0) {
    return false;
  } else if (strcmp("true", enable_gpu_garbage_collection) == 0) {
    return true;
  }

  LOG(ERROR)
      << "The TF_ENABLE_GPU_GARBAGE_COLLECTION environment variable is set but"
      << " could not be parsed: \"" << enable_gpu_garbage_collection << "\"."
      << " Valid values are \"true\" or \"false\"."
      << " Using the default value \"true\".";
  return true;
}

PluggableDeviceBFCAllocator::PluggableDeviceBFCAllocator(
    DeviceMemAllocator* sub_allocator, size_t total_memory, const string& name,
    bool force_memory_growth_requested)
    : PluggableDeviceBFCAllocator(sub_allocator, total_memory, GPUOptions(),
                                  name, force_memory_growth_requested) {
   std::vector<std::string> mht_2_v;
   mht_2_v.push_back("name: \"" + name + "\"");
   MHTracer_DTPStensorflowPScorePScommon_runtimePSpluggable_devicePSpluggable_device_bfc_allocatorDTcc mht_2(mht_2_v, 269, "", "./tensorflow/core/common_runtime/pluggable_device/pluggable_device_bfc_allocator.cc", "PluggableDeviceBFCAllocator::PluggableDeviceBFCAllocator");
}

PluggableDeviceBFCAllocator::PluggableDeviceBFCAllocator(
    DeviceMemAllocator* sub_allocator, size_t total_memory,
    const GPUOptions& gpu_options, const string& name,
    bool force_memory_growth_requested)
    : BFCAllocator(absl::WrapUnique(sub_allocator), total_memory, name, [&] {
        BFCAllocator::Options o;
        o.allow_growth = PluggableDeviceBFCAllocator::GetAllowGrowthValue(
            gpu_options, force_memory_growth_requested);
        o.garbage_collection =
            PluggableDeviceBFCAllocator::GetGarbageCollectionValue();
        return o;
      }()) {
   std::vector<std::string> mht_3_v;
   mht_3_v.push_back("name: \"" + name + "\"");
   MHTracer_DTPStensorflowPScorePScommon_runtimePSpluggable_devicePSpluggable_device_bfc_allocatorDTcc mht_3(mht_3_v, 286, "", "./tensorflow/core/common_runtime/pluggable_device/pluggable_device_bfc_allocator.cc", "PluggableDeviceBFCAllocator::PluggableDeviceBFCAllocator");
}

}  // namespace tensorflow
