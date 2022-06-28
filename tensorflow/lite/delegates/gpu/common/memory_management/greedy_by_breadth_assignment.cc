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
class MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPSmemory_managementPSgreedy_by_breadth_assignmentDTcc {
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
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPSmemory_managementPSgreedy_by_breadth_assignmentDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPSmemory_managementPSgreedy_by_breadth_assignmentDTcc() {
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

#include "tensorflow/lite/delegates/gpu/common/memory_management/greedy_by_breadth_assignment.h"

#include <algorithm>
#include <cstddef>
#include <set>
#include <vector>

#include "absl/status/status.h"
#include "tensorflow/lite/delegates/gpu/common/memory_management/internal.h"
#include "tensorflow/lite/delegates/gpu/common/memory_management/types.h"

namespace tflite {
namespace gpu {
namespace {

// Set of usage records for all tensors assigned to the shared object, ordered
// by first_task.
using SharedObjectSchedule = std::set<TensorUsageRecord<size_t>>;

struct TaskBreadthWithId {
  size_t breadth;
  TaskId task_id;

  TaskBreadthWithId(size_t breadth, size_t task_id)
      : breadth(breadth), task_id(task_id) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPSmemory_managementPSgreedy_by_breadth_assignmentDTcc mht_0(mht_0_v, 209, "", "./tensorflow/lite/delegates/gpu/common/memory_management/greedy_by_breadth_assignment.cc", "TaskBreadthWithId");
}

  // Default order of TaskBreadthWithId is increasing order of their breadth.
  bool operator<(const TaskBreadthWithId& other) const {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPSmemory_managementPSgreedy_by_breadth_assignmentDTcc mht_1(mht_1_v, 215, "", "./tensorflow/lite/delegates/gpu/common/memory_management/greedy_by_breadth_assignment.cc", "operator<");

    return breadth < other.breadth;
  }
};

}  // namespace

absl::Status GreedyByBreadthAssignment(
    const std::vector<TensorUsageRecord<size_t>>& usage_records,
    ObjectsAssignment<size_t>* assignment) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPSmemory_managementPSgreedy_by_breadth_assignmentDTcc mht_2(mht_2_v, 227, "", "./tensorflow/lite/delegates/gpu/common/memory_management/greedy_by_breadth_assignment.cc", "GreedyByBreadthAssignment");

  std::vector<TaskProfile> task_profiles = CalculateTaskProfiles(usage_records);

  // Task breadth is a sum of sizes of all tensors in its TaskProfile
  std::vector<TaskBreadthWithId> task_breadth;
  for (size_t task_id = 0; task_id < task_profiles.size(); ++task_id) {
    size_t breadth = 0;
    for (const auto& tensor_info : task_profiles[task_id]) {
      breadth += tensor_info.usage_record->tensor_size;
    }
    task_breadth.emplace_back(breadth, task_id);
  }

  assignment->object_sizes.clear();
  assignment->object_ids.assign(usage_records.size(), kNotAssigned);
  std::vector<SharedObjectSchedule> obj_schedules;

  // Iterate through all tasks in non-increasing order of their breadth.
  std::stable_sort(task_breadth.rbegin(), task_breadth.rend());
  for (const auto& task : task_breadth) {
    // Iterate through all tensors, that must be allocated during the execution
    // of task, in non-increasing order of their tensor_size.
    for (const auto& tensor_info : task_profiles[task.task_id]) {
      if (assignment->object_ids[tensor_info.idx] != kNotAssigned) {
        continue;
      }
      const auto& rec = *tensor_info.usage_record;
      const size_t num_objects = obj_schedules.size();
      size_t best_object = num_objects;
      for (size_t obj_id = 0; obj_id < num_objects; ++obj_id) {
        // If size of current_object is worse than size of best found before, we
        // can skip it.
        if (best_object != num_objects) {
          const size_t best_size = assignment->object_sizes[best_object];
          const size_t cur_size = assignment->object_sizes[obj_id];
          if (best_size < rec.tensor_size) {
            if (cur_size <= best_size) {
              // best_size is smaller than tensor_size, but cur_size is even
              // smaller.
              continue;
            }
          } else if (cur_size < rec.tensor_size || cur_size >= best_size) {
            // best_size is larger or equal to tensor_size, and cur_size is
            // either smaller than tensor_size, or too large.
            continue;
          }
        }
        const auto& schedule = obj_schedules[obj_id];
        auto it = schedule.lower_bound(rec);
        bool update_best_object = true;
        if (it != schedule.end() && it->first_task <= rec.last_task) {
          // Some tensor, which usage interval intersects with current, already
          // assigned to this object.
          update_best_object = false;
        }
        if (update_best_object && it != schedule.begin()) {
          it--;
          if (it->last_task >= rec.first_task) {
            // Some tensor, which usage interval intersects with current,
            // already assigned to this object.
            update_best_object = false;
          }
        }
        if (update_best_object) {
          best_object = obj_id;
        }
      }
      if (best_object == num_objects) {
        // Create new shared object and assign current tensor to it.
        obj_schedules.push_back({rec});
        assignment->object_sizes.push_back(rec.tensor_size);
      } else {
        // Assign current tensor to best_object.
        obj_schedules[best_object].insert(rec);
        // Size of best_object can be increased, if it is smaller than
        // tensor_size.
        assignment->object_sizes[best_object] =
            std::max(assignment->object_sizes[best_object], rec.tensor_size);
      }
      assignment->object_ids[tensor_info.idx] = best_object;
    }
  }
  // In the end all tensors must be assigned to some objects.
  for (const auto& obj_id : assignment->object_ids) {
    if (obj_id == kNotAssigned) {
      return absl::InternalError("Error while calculating the assignment.");
    }
  }
  return absl::OkStatus();
}

}  // namespace gpu
}  // namespace tflite
