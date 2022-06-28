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

#ifndef TENSORFLOW_LITE_DELEGATES_GPU_COMMON_MEMORY_MANAGEMENT_GREEDY_IN_ORDER_ASSIGNMENT_H_
#define TENSORFLOW_LITE_DELEGATES_GPU_COMMON_MEMORY_MANAGEMENT_GREEDY_IN_ORDER_ASSIGNMENT_H_
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
class MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPSmemory_managementPSgreedy_in_order_assignmentDTh {
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
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPSmemory_managementPSgreedy_in_order_assignmentDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPSmemory_managementPSgreedy_in_order_assignmentDTh() {
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

#include <algorithm>
#include <cstddef>
#include <iterator>
#include <list>
#include <queue>
#include <set>
#include <vector>

#include "tensorflow/lite/delegates/gpu/common/memory_management/internal.h"
#include "tensorflow/lite/delegates/gpu/common/memory_management/types.h"
#include "tensorflow/lite/delegates/gpu/common/status.h"

namespace tflite {
namespace gpu {

// Implements memory management with a greedy algorithm.
//
// The problem of memory management is NP-complete. This implements a
// greedy algorithm that approximates an optimal solution with following
// heuristic:
//
//   1. Iterates through all tensor usage records and for every object
//   reference
//      assigns shared object from the pool. When object reference is used
//      for the last time, corresponding shared object is returned back to
//      the pool.
//
//   2. Shared object pool grows when there are no free shared object
//      available.
//
//   3. Shared object size may increase when tensor requests larger size.
template <typename TensorSizeT>
absl::Status GreedyInOrderAssignment(
    const std::vector<TensorUsageRecord<TensorSizeT>>& usage_records,
    ObjectsAssignment<TensorSizeT>* assignment,
    const UsageGraph* reallocation_graph = nullptr) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPSmemory_managementPSgreedy_in_order_assignmentDTh mht_0(mht_0_v, 225, "", "./tensorflow/lite/delegates/gpu/common/memory_management/greedy_in_order_assignment.h", "GreedyInOrderAssignment");

  std::vector<size_t> last_assigned_tensor;
  size_t num_records = usage_records.size();
  assignment->object_sizes.clear();
  assignment->object_ids.assign(num_records, kNotAssigned);

  // Pool of free shared objects is ordered by object size, because we perform
  // lower_bound search in it.
  std::set<PoolRecord<TensorSizeT>> pool;
  // Queue of shared objects in use, ordered by their last_task.
  std::priority_queue<QueueRecord> objects_in_use;
  for (size_t i = 0; i < num_records; i++) {
    // Pop from the queue and add to the pool all objects that are no longer
    // in use at the time of execution of the first_task of i-th intermediate
    // tensor.
    while (!objects_in_use.empty() &&
           objects_in_use.top().last_task < usage_records[i].first_task) {
      auto object_id = objects_in_use.top().object_id;
      pool.insert({assignment->object_sizes[object_id], object_id});
      objects_in_use.pop();
    }
    TensorSizeT tensor_size = usage_records[i].tensor_size;
    auto best_it = pool.end();
    size_t best_size_diff = 0;
    if (reallocation_graph) {
      for (auto pool_it = pool.begin(); pool_it != pool.end(); ++pool_it) {
        size_t size_diff = AbsDiffInElements(pool_it->object_size, tensor_size);
        if (best_it == pool.end() || size_diff < best_size_diff) {
          const std::vector<size_t>& realloc_options =
              (*reallocation_graph)[last_assigned_tensor[pool_it->object_id]];
          size_t pos = std::lower_bound(realloc_options.begin(),
                                        realloc_options.end(), i) -
                       realloc_options.begin();
          if (pos != realloc_options.size() && realloc_options[pos] == i) {
            // We found, that memory of tensor, that was last assigned to
            // object pool_it->object_id, can be reused for tensor i.
            best_size_diff = size_diff;
            best_it = pool_it;
          }
        }
      }
    } else if (!pool.empty()) {
      // Find shared object from pool, that will waste the least possible
      // amount of memory when reused for current tensor.
      auto pool_it = pool.lower_bound({tensor_size, 0});
      TensorSizeT size_diff = 0;
      if (pool_it != pool.end()) {
        // Try smallest shared object from pool with size >= tensor_size.
        size_diff = pool_it->object_size - tensor_size;
        best_it = pool_it;
      }
      if (pool_it != pool.begin()) {
        // Try largest shared object from pool with size < tensor_size.
        pool_it--;
        if (best_it == pool.end() ||
            tensor_size - pool_it->object_size < size_diff) {
          size_diff = tensor_size - pool_it->object_size;
          best_it = pool_it;
        }
      }
      // best_it can't be equal to pool.end(), because pool is not empty
      if (best_it == pool.end()) {
        return absl::InternalError(
            "No shared object is found in non-empty pool in "
            "GreedyInOrderAssignment.");
      }
    }
    if (best_it == pool.end()) {
      // No free shared object, creating a new one, assign i-th tensor to
      // it and add to the queue of objects in use.
      assignment->object_ids[i] = assignment->object_sizes.size();
      assignment->object_sizes.push_back(tensor_size);
      last_assigned_tensor.push_back(i);
      objects_in_use.push(
          {usage_records[i].last_task, assignment->object_ids[i]});
    } else {
      size_t shared_id = best_it->object_id;
      pool.erase(best_it);
      assignment->object_ids[i] = shared_id;
      assignment->object_sizes[shared_id] =
          std::max(assignment->object_sizes[shared_id], tensor_size);
      last_assigned_tensor[shared_id] = i;
      objects_in_use.push(
          {usage_records[i].last_task, assignment->object_ids[i]});
    }
  }
  return absl::OkStatus();
}

// The same algorithm as above, but for multidimensional case. The only
// difference is that shared object dimensions can't be increased to be reused
// for tensor, that is larger (at least by one dimension).
template <typename TensorSizeT>
absl::Status GreedyInOrderAssignmentMultidimensional(
    const std::vector<TensorUsageRecord<TensorSizeT>>& usage_records,
    ObjectsAssignment<TensorSizeT>* assignment) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPSmemory_managementPSgreedy_in_order_assignmentDTh mht_1(mht_1_v, 323, "", "./tensorflow/lite/delegates/gpu/common/memory_management/greedy_in_order_assignment.h", "GreedyInOrderAssignmentMultidimensional");

  size_t num_records = usage_records.size();
  assignment->object_sizes.clear();
  assignment->object_ids.assign(num_records, kNotAssigned);

  // Pool of free shared objects is unordered in multidimensional version of the
  // algorithm.
  std::list<size_t> pool;
  // Queue of shared objects in use, ordered by their last_task.
  std::priority_queue<QueueRecord> objects_in_use;
  for (size_t i = 0; i < num_records; i++) {
    // Pop from the queue and add to the pool all objects that are no longer
    // in use at the time of execution of the first_task of i-th intermediate
    // tensor.
    while (!objects_in_use.empty() &&
           objects_in_use.top().last_task < usage_records[i].first_task) {
      auto object_id = objects_in_use.top().object_id;
      pool.push_back(object_id);
      objects_in_use.pop();
    }
    const TensorSizeT& tensor_size = usage_records[i].tensor_size;
    auto best_it = pool.end();
    size_t best_size_diff = 0;
    // Find shared object from pool, that will waste the least possible
    // amount of memory when reused for current tensor.
    for (auto pool_it = pool.begin(); pool_it != pool.end(); ++pool_it) {
      // Needed size of shared object to cover current tensor and all previous
      // tensors assigned to it.
      const TensorSizeT& shared_object_size =
          assignment->object_sizes[*pool_it];
      if (IsCoveringObject(shared_object_size, tensor_size)) {
        // Prefer shared object that will waste less memory.
        size_t size_diff = AbsDiffInElements(shared_object_size, tensor_size);
        if (best_it == pool.end() || size_diff < best_size_diff) {
          best_it = pool_it;
          best_size_diff = size_diff;
        }
      }
    }
    if (best_it == pool.end()) {
      // No free suitable shared object, creating a new one, assign i-th tensor
      // to it and add to the queue of objects in use.
      assignment->object_ids[i] = assignment->object_sizes.size();
      assignment->object_sizes.push_back(tensor_size);
      objects_in_use.push(
          {usage_records[i].last_task, assignment->object_ids[i]});
    } else {
      size_t shared_id = *best_it;
      pool.erase(best_it);
      assignment->object_ids[i] = shared_id;
      objects_in_use.push(
          {usage_records[i].last_task, assignment->object_ids[i]});
    }
  }
  return absl::OkStatus();
}

}  // namespace gpu
}  // namespace tflite

#endif  // TENSORFLOW_LITE_DELEGATES_GPU_COMMON_MEMORY_MANAGEMENT_GREEDY_IN_ORDER_ASSIGNMENT_H_
