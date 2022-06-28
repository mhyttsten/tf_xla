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

#ifndef TENSORFLOW_LITE_DELEGATES_GPU_COMMON_MEMORY_MANAGEMENT_INTERNAL_H_
#define TENSORFLOW_LITE_DELEGATES_GPU_COMMON_MEMORY_MANAGEMENT_INTERNAL_H_
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
class MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPSmemory_managementPSinternalDTh {
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
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPSmemory_managementPSinternalDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPSmemory_managementPSinternalDTh() {
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


#include <stddef.h>

#include <limits>
#include <vector>

#include "absl/memory/memory.h"
#include "tensorflow/lite/delegates/gpu/common/memory_management/types.h"
#include "tensorflow/lite/delegates/gpu/common/types.h"

namespace tflite {
namespace gpu {

const size_t kNotAssigned = std::numeric_limits<size_t>::max();

// This structure is used to save the initial indices of usage records after
// they are sorted.
template <typename TensorSizeT>
struct TensorUsageWithIndex {
  const TensorUsageRecord<TensorSizeT>* usage_record;
  size_t idx;

  TensorUsageWithIndex(const TensorUsageRecord<TensorSizeT>* usage_record,
                       size_t idx)
      : usage_record(usage_record), idx(idx) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPSmemory_managementPSinternalDTh mht_0(mht_0_v, 211, "", "./tensorflow/lite/delegates/gpu/common/memory_management/internal.h", "TensorUsageWithIndex");
}
};

bool CompareBySize(const TensorUsageWithIndex<size_t>& first,
                   const TensorUsageWithIndex<size_t>& second);

// TaskProfile is a vector with information about all intermediate tensors, that
// should exist in memory during the execution of the task. Elements of the
// vector must be sorted in non-increasing order of corresponding tensors sizes.
using TaskProfile = std::vector<TensorUsageWithIndex<size_t>>;

// Size of object, that covers both input objects (2-dimensional case).
bool IsCoveringObject(const uint2& first_object, const uint2& second_object);

// Size of object, that covers both input objects (3-dimensional case).
bool IsCoveringObject(const uint3& first_object, const uint3& second_object);

// Difference between two objects in elements count (1-dimensional case).
size_t AbsDiffInElements(const size_t first_size, const size_t second_size);

// Difference between two objects in elements count (2-dimensional case).
size_t AbsDiffInElements(const uint2& first_size, const uint2& second_size);

// Difference between two objects in elements count (3-dimensional case).
size_t AbsDiffInElements(const uint3& first_size, const uint3& second_size);

template <typename ObjectSizeT>
struct PoolRecord {
  PoolRecord(ObjectSizeT size, size_t obj_id)
      : object_size(size), object_id(obj_id) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPSmemory_managementPSinternalDTh mht_1(mht_1_v, 243, "", "./tensorflow/lite/delegates/gpu/common/memory_management/internal.h", "PoolRecord");
}

  // Objects in pool are ordered by size.
  bool operator<(const PoolRecord& other) const {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPSmemory_managementPSinternalDTh mht_2(mht_2_v, 249, "", "./tensorflow/lite/delegates/gpu/common/memory_management/internal.h", "operator<");

    return (object_size < other.object_size) ||
           (object_size == other.object_size && object_id < other.object_id);
  }

  ObjectSizeT object_size;
  size_t object_id;
};

struct QueueRecord {
  QueueRecord(TaskId task_id, size_t obj_id)
      : last_task(task_id), object_id(obj_id) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPSmemory_managementPSinternalDTh mht_3(mht_3_v, 263, "", "./tensorflow/lite/delegates/gpu/common/memory_management/internal.h", "QueueRecord");
}

  // Objects in queue are ordered by last_task.
  bool operator<(const QueueRecord& other) const {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPSmemory_managementPSinternalDTh mht_4(mht_4_v, 269, "", "./tensorflow/lite/delegates/gpu/common/memory_management/internal.h", "operator<");

    return (last_task > other.last_task) ||
           (last_task == other.last_task && object_id > other.object_id);
  }

  // Last task, where shared object is used.
  TaskId last_task;
  size_t object_id;
};

// Returns a vector that contains TaskProfile for each task.
std::vector<TaskProfile> CalculateTaskProfiles(
    const std::vector<TensorUsageRecord<size_t>>& usage_records);

// Iterates over all task profiles to calculate maximum at each position.
std::vector<size_t> CalculatePositionalMaximums(
    const std::vector<TensorUsageRecord<size_t>>& usage_records);

}  // namespace gpu
}  // namespace tflite

#endif  // TENSORFLOW_LITE_DELEGATES_GPU_COMMON_MEMORY_MANAGEMENT_INTERNAL_H_
