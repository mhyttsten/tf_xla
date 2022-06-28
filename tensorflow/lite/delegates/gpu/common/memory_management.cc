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
class MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPSmemory_managementDTcc {
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
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPSmemory_managementDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPSmemory_managementDTcc() {
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

#include "tensorflow/lite/delegates/gpu/common/memory_management.h"

#include <cstddef>
#include <numeric>
#include <utility>
#include <vector>

#include "tensorflow/lite/delegates/gpu/common/memory_management/equality_assignment.h"
#include "tensorflow/lite/delegates/gpu/common/memory_management/greedy_by_breadth_assignment.h"
#include "tensorflow/lite/delegates/gpu/common/memory_management/greedy_by_size_assignment.h"
#include "tensorflow/lite/delegates/gpu/common/memory_management/greedy_in_order_assignment.h"
#include "tensorflow/lite/delegates/gpu/common/memory_management/min_cost_flow_assignment.h"
#include "tensorflow/lite/delegates/gpu/common/memory_management/naive_assignment.h"
#include "tensorflow/lite/delegates/gpu/common/memory_management/types.h"
#include "tensorflow/lite/delegates/gpu/common/shape.h"
#include "tensorflow/lite/delegates/gpu/common/status.h"
#include "tensorflow/lite/delegates/gpu/common/types.h"

namespace tflite {
namespace gpu {
namespace {

size_t TotalSize(const ObjectsAssignment<size_t>& assignment) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPSmemory_managementDTcc mht_0(mht_0_v, 207, "", "./tensorflow/lite/delegates/gpu/common/memory_management.cc", "TotalSize");

  return std::accumulate(assignment.object_sizes.begin(),
                         assignment.object_sizes.end(), static_cast<size_t>(0));
}

}  // namespace

OffsetsAssignment ObjectsToOffsets(
    const ObjectsAssignment<size_t>& obj_assignment) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPSmemory_managementDTcc mht_1(mht_1_v, 218, "", "./tensorflow/lite/delegates/gpu/common/memory_management.cc", "ObjectsToOffsets");

  size_t num_tensors = obj_assignment.object_ids.size();
  size_t num_objects = obj_assignment.object_sizes.size();
  OffsetsAssignment result = {/*offsets=*/std::vector<size_t>(num_tensors),
                              /*total_size=*/0};
  std::vector<size_t> ids_to_offset(num_objects);
  for (size_t i = 0; i < num_objects; ++i) {
    ids_to_offset[i] = result.total_size;
    result.total_size += obj_assignment.object_sizes[i];
  }
  for (size_t i = 0; i < num_tensors; ++i) {
    result.offsets[i] = ids_to_offset[obj_assignment.object_ids[i]];
  }
  return result;
}

absl::Status BestGreedy(
    const std::vector<TensorUsageRecord<size_t>>& usage_records,
    ObjectsAssignment<size_t>* assignment) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPSmemory_managementDTcc mht_2(mht_2_v, 239, "", "./tensorflow/lite/delegates/gpu/common/memory_management.cc", "BestGreedy");

  RETURN_IF_ERROR(
      GreedyBySizeDistPriorityAssignment(usage_records, assignment));
  ObjectsAssignment<size_t> assignment_by_breadth;
  if (GreedyByBreadthAssignment(usage_records, &assignment_by_breadth).ok() &&
      TotalSize(assignment_by_breadth) < TotalSize(*assignment)) {
    std::swap(*assignment, assignment_by_breadth);
  }
  return absl::OkStatus();
}

template <>
absl::Status AssignObjectsToTensors(
    const std::vector<TensorUsageRecord<size_t>>& usage_records,
    MemoryStrategy strategy, ObjectsAssignment<size_t>* assignment,
    const UsageGraph* reallocation_graph) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPSmemory_managementDTcc mht_3(mht_3_v, 257, "", "./tensorflow/lite/delegates/gpu/common/memory_management.cc", "AssignObjectsToTensors");

  switch (strategy) {
    case MemoryStrategy::NAIVE:
      return NaiveAssignment(usage_records, assignment);
    case MemoryStrategy::EQUALITY:
      return EqualityAssignmentWithHash(usage_records, assignment);
    case MemoryStrategy::GREEDY_IN_ORDER:
      return GreedyInOrderAssignment(usage_records, assignment,
                                     reallocation_graph);
    case MemoryStrategy::GREEDY_BY_BREADTH:
      return GreedyByBreadthAssignment(usage_records, assignment);
    case MemoryStrategy::GREEDY_BY_SIZE:
      return GreedyBySizeDistPriorityAssignment(usage_records, assignment);
    case MemoryStrategy::GREEDY_BEST:
      return BestGreedy(usage_records, assignment);
    case MemoryStrategy::MINCOSTFLOW:
      return MinCostFlowAssignment(usage_records, assignment);
    default:
      return absl::InternalError(
          "MemoryStrategy is not supported with current tensor size type.");
  }
  return absl::OkStatus();
}

template <>
absl::Status AssignObjectsToTensors(
    const std::vector<TensorUsageRecord<BHWC>>& usage_records,
    MemoryStrategy strategy, ObjectsAssignment<BHWC>* assignment,
    const UsageGraph* reallocation_graph) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPSmemory_managementDTcc mht_4(mht_4_v, 288, "", "./tensorflow/lite/delegates/gpu/common/memory_management.cc", "AssignObjectsToTensors");

  switch (strategy) {
    case MemoryStrategy::NAIVE:
      return NaiveAssignment(usage_records, assignment);
    case MemoryStrategy::EQUALITY:
      return EqualityAssignmentWithHash(usage_records, assignment);
    default:
      return absl::InternalError(
          "MemoryStrategy is not supported with current tensor size type.");
  }
  return absl::OkStatus();
}

template <>
absl::Status AssignObjectsToTensors(
    const std::vector<TensorUsageRecord<uint2>>& usage_records,
    MemoryStrategy strategy, ObjectsAssignment<uint2>* assignment,
    const UsageGraph* reallocation_graph) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPSmemory_managementDTcc mht_5(mht_5_v, 308, "", "./tensorflow/lite/delegates/gpu/common/memory_management.cc", "AssignObjectsToTensors");

  switch (strategy) {
    case MemoryStrategy::NAIVE:
      return NaiveAssignment(usage_records, assignment);
    case MemoryStrategy::EQUALITY:
      return EqualityAssignment(usage_records, assignment);
    case MemoryStrategy::GREEDY_IN_ORDER:
      return GreedyInOrderAssignmentMultidimensional(usage_records, assignment);
    default:
      return absl::InternalError(
          "MemoryStrategy is not supported with current tensor size type.");
  }
  return absl::OkStatus();
}

template <>
absl::Status AssignObjectsToTensors(
    const std::vector<TensorUsageRecord<uint3>>& usage_records,
    MemoryStrategy strategy, ObjectsAssignment<uint3>* assignment,
    const UsageGraph* reallocation_graph) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPSmemory_managementDTcc mht_6(mht_6_v, 330, "", "./tensorflow/lite/delegates/gpu/common/memory_management.cc", "AssignObjectsToTensors");

  switch (strategy) {
    case MemoryStrategy::NAIVE:
      return NaiveAssignment(usage_records, assignment);
    case MemoryStrategy::EQUALITY:
      return EqualityAssignment(usage_records, assignment);
    case MemoryStrategy::GREEDY_IN_ORDER:
      return GreedyInOrderAssignmentMultidimensional(usage_records, assignment);
    default:
      return absl::InternalError(
          "MemoryStrategy is not supported with current tensor size type.");
  }
  return absl::OkStatus();
}

absl::Status AssignOffsetsToTensors(
    const std::vector<TensorUsageRecord<size_t>>& usage_records,
    const MemoryStrategy& strategy, OffsetsAssignment* assignment,
    size_t base_addr_align_bytes, const UsageGraph* reallocation_graph) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPSmemory_managementDTcc mht_7(mht_7_v, 351, "", "./tensorflow/lite/delegates/gpu/common/memory_management.cc", "AssignOffsetsToTensors");

  if (strategy == MemoryStrategy::GREEDY_BY_SIZE) {
    return GreedyBySizeAssignment(usage_records, base_addr_align_bytes,
                                  assignment);
  }
  ObjectsAssignment<size_t> objects_assignment;
  RETURN_IF_ERROR(AssignObjectsToTensors(
      usage_records, strategy, &objects_assignment, reallocation_graph));
  *assignment = ObjectsToOffsets(objects_assignment);
  return absl::OkStatus();
}

}  // namespace gpu
}  // namespace tflite
