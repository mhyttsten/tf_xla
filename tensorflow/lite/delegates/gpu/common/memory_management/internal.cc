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
class MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPSmemory_managementPSinternalDTcc {
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
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPSmemory_managementPSinternalDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPSmemory_managementPSinternalDTcc() {
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

#include "tensorflow/lite/delegates/gpu/common/memory_management/internal.h"

#include <algorithm>
#include <cstddef>
#include <vector>

#include "tensorflow/lite/delegates/gpu/common/memory_management/types.h"
#include "tensorflow/lite/delegates/gpu/common/types.h"

namespace tflite {
namespace gpu {

bool CompareBySize(const TensorUsageWithIndex<size_t>& first,
                   const TensorUsageWithIndex<size_t>& second) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPSmemory_managementPSinternalDTcc mht_0(mht_0_v, 198, "", "./tensorflow/lite/delegates/gpu/common/memory_management/internal.cc", "CompareBySize");

  return first.usage_record->tensor_size > second.usage_record->tensor_size;
}

bool IsCoveringObject(const uint2& first_object, const uint2& second_object) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPSmemory_managementPSinternalDTcc mht_1(mht_1_v, 205, "", "./tensorflow/lite/delegates/gpu/common/memory_management/internal.cc", "IsCoveringObject");

  return first_object.x >= second_object.x && first_object.y >= second_object.y;
}

bool IsCoveringObject(const uint3& first_object, const uint3& second_object) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPSmemory_managementPSinternalDTcc mht_2(mht_2_v, 212, "", "./tensorflow/lite/delegates/gpu/common/memory_management/internal.cc", "IsCoveringObject");

  return first_object.x >= second_object.x &&
         first_object.y >= second_object.y && first_object.z >= second_object.z;
}

size_t AbsDiffInElements(const size_t first_size, const size_t second_size) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPSmemory_managementPSinternalDTcc mht_3(mht_3_v, 220, "", "./tensorflow/lite/delegates/gpu/common/memory_management/internal.cc", "AbsDiffInElements");

  return first_size >= second_size ? first_size - second_size
                                   : second_size - first_size;
}

size_t AbsDiffInElements(const uint2& first_size, const uint2& second_size) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPSmemory_managementPSinternalDTcc mht_4(mht_4_v, 228, "", "./tensorflow/lite/delegates/gpu/common/memory_management/internal.cc", "AbsDiffInElements");

  const size_t first_elements_cnt = first_size.y * first_size.x;
  const size_t second_elements_cnt = second_size.y * second_size.x;
  return first_elements_cnt >= second_elements_cnt
             ? first_elements_cnt - second_elements_cnt
             : second_elements_cnt - first_elements_cnt;
}

size_t AbsDiffInElements(const uint3& first_size, const uint3& second_size) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPSmemory_managementPSinternalDTcc mht_5(mht_5_v, 239, "", "./tensorflow/lite/delegates/gpu/common/memory_management/internal.cc", "AbsDiffInElements");

  const size_t first_elements_cnt = first_size.z * first_size.y * first_size.x;
  const size_t second_elements_cnt =
      second_size.z * second_size.y * second_size.x;
  return first_elements_cnt >= second_elements_cnt
             ? first_elements_cnt - second_elements_cnt
             : second_elements_cnt - first_elements_cnt;
}

std::vector<TaskProfile> CalculateTaskProfiles(
    const std::vector<TensorUsageRecord<size_t>>& usage_records) {
  TaskId num_tasks = 0;
  for (size_t i = 0; i < usage_records.size(); ++i) {
    num_tasks = std::max(num_tasks, usage_records[i].last_task + 1);
  }
  std::vector<TaskProfile> task_profiles(num_tasks);
  for (size_t rec_id = 0; rec_id < usage_records.size(); ++rec_id) {
    // Each tensor usage record must be added to profile of every task between
    // its first_task and last_task.
    for (TaskId task_id = usage_records[rec_id].first_task;
         task_id <= usage_records[rec_id].last_task; ++task_id) {
      task_profiles[task_id].emplace_back(&usage_records[rec_id], rec_id);
    }
  }
  // Records in each TaskProfile must be sorted in non-increasing order of
  // corresponding tensors sizes.
  for (auto& task_profile : task_profiles) {
    std::stable_sort(task_profile.begin(), task_profile.end(), CompareBySize);
  }
  return task_profiles;
}

std::vector<size_t> CalculatePositionalMaximums(
    const std::vector<TensorUsageRecord<size_t>>& usage_records) {
  std::vector<TaskProfile> task_profiles = CalculateTaskProfiles(usage_records);
  std::vector<size_t> positional_max;
  for (const auto& task_profile : task_profiles) {
    // Update positional_max with values of current TaskProfile.
    size_t i = 0;
    for (; i < task_profile.size() && i < positional_max.size(); ++i) {
      positional_max[i] = std::max(positional_max[i],
                                   task_profile[i].usage_record->tensor_size);
    }
    // If current task_profile has more records, than there are in
    // positional_max, we should append new elements into positional_max.
    for (; i < task_profile.size(); ++i) {
      positional_max.push_back(task_profile[i].usage_record->tensor_size);
    }
  }
  return positional_max;
}

}  // namespace gpu
}  // namespace tflite
