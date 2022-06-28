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
class MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPSthunk_scheduleDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPSthunk_scheduleDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPSthunk_scheduleDTcc() {
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

/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/service/gpu/thunk_schedule.h"

#include <algorithm>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_map.h"
#include "absl/strings/str_format.h"
#include "tensorflow/compiler/xla/array2d.h"
#include "tensorflow/compiler/xla/map_util.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/core/lib/gtl/map_util.h"

namespace xla {
namespace gpu {

void ThunkSchedule::AddDependenciesOnTransitiveOperands(
    const Thunk& thunk, const HloInstruction& operand,
    const absl::flat_hash_map<const HloInstruction*, Thunk*>& hlo_to_thunk) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPSthunk_scheduleDTcc mht_0(mht_0_v, 202, "", "./tensorflow/compiler/xla/service/gpu/thunk_schedule.cc", "ThunkSchedule::AddDependenciesOnTransitiveOperands");

  if (hlo_to_thunk.contains(&operand)) {
    // If `operand` is mapped to a thunk, adds `operand` to `thunk`'s dependency
    // list if `operand` is assigned to a different stream. As an optimization,
    // we skip `operand`'s operands because `operand` depends on them already.
    if (stream_assignment_->StreamNumberForHlo(operand) !=
        stream_assignment_->StreamNumberForHlo(*thunk_to_hlo_.at(&thunk))) {
      depends_on_[&thunk].push_back(FindOrDie(hlo_to_thunk, &operand));
    }
  } else {
    // If `operand` doesn't need a thunk (e.g. bitcast), continue with its
    // operands.
    for (const auto* operand_of_operand : operand.operands()) {
      AddDependenciesOnTransitiveOperands(thunk, *operand_of_operand,
                                          hlo_to_thunk);
    }
  }
}

ThunkSchedule::ThunkSchedule(
    std::unique_ptr<ThunkSequence> thunks,
    std::unique_ptr<StreamAssignment> stream_assignment,
    absl::flat_hash_map<const Thunk*, const HloInstruction*> thunk_to_hlo)
    : thunks_(std::move(thunks)),
      stream_assignment_(std::move(stream_assignment)),
      thunk_to_hlo_(std::move(thunk_to_hlo)) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPSthunk_scheduleDTcc mht_1(mht_1_v, 230, "", "./tensorflow/compiler/xla/service/gpu/thunk_schedule.cc", "ThunkSchedule::ThunkSchedule");

  absl::flat_hash_map<const HloInstruction*, Thunk*> hlo_to_thunk;
  for (const std::unique_ptr<Thunk>& thunk : TotalOrder()) {
    InsertOrDie(&hlo_to_thunk, thunk_to_hlo_.at(thunk.get()), thunk.get());
  }

  for (const std::unique_ptr<Thunk>& thunk : TotalOrder()) {
    const auto* dst = thunk_to_hlo_.at(thunk);
    CHECK(stream_assignment_->HasStreamAssigned(*dst));
    for (const auto* src : dst->operands()) {
      AddDependenciesOnTransitiveOperands(*thunk, *src, hlo_to_thunk);
    }
  }

  RemoveRedundantDependencyEdges();

  // Compute `depended_by_`, the inverse of `depends_on_`.
  for (const auto& dependency : depends_on_) {
    for (const auto* depended : dependency.second) {
      depended_by_.insert(depended);
    }
  }
}

ThunkSchedule::ThunkSchedule(std::unique_ptr<ThunkSequence> thunks)
    : thunks_(std::move(thunks)) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPSthunk_scheduleDTcc mht_2(mht_2_v, 258, "", "./tensorflow/compiler/xla/service/gpu/thunk_schedule.cc", "ThunkSchedule::ThunkSchedule");
}

void ThunkSchedule::RemoveRedundantDependencyEdges() {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPSthunk_scheduleDTcc mht_3(mht_3_v, 263, "", "./tensorflow/compiler/xla/service/gpu/thunk_schedule.cc", "ThunkSchedule::RemoveRedundantDependencyEdges");

  absl::flat_hash_map<const Thunk*, int> thunk_to_total_order;
  for (int i = 0; i < thunks_->size(); ++i) {
    InsertOrDie(&thunk_to_total_order, thunks_->at(i).get(), i);
  }

  int stream_count = stream_assignment_->StreamCount();
  // S1  S2
  //
  // T1<----+
  //        |
  // T3<--+ |
  //      | | depends on
  //     T4 |
  //        |
  //     T2-+
  //
  // Suppose thunk T1 and T3 are scheduled on stream S1, and T2 and T4 are on
  // stream S2. If T2 depends on T1 and T4 depends on T3, and
  // order(T1)<order(T3)<order(T4)<order(T2), the dependency of T2 on T1 is
  // redundant.
  //
  // To efficiently detect such redundancy, we leverage array `last_dependency`.
  // last_dependency[S1][S2] indicates the last thunk (with the maximum order
  // number) on stream S2 that thunks on S1 depends on. Therefore, if a future
  // S1 thunk depends on a S2 thunk ordered <=last_dependency[S1][S2], that is a
  // redundant dependency edge.
  Array2D<int> last_dependency(stream_count, stream_count, -1);
  for (const std::unique_ptr<Thunk>& dst_thunk : TotalOrder()) {
    const Thunk* dst = dst_thunk.get();
    if (!depends_on_.contains(dst)) {
      continue;
    }

    int dst_stream =
        stream_assignment_->StreamNumberForHlo(*thunk_to_hlo_.at(dst));
    std::list<const Thunk*>& sources = FindOrDie(depends_on_, dst);
    for (auto iter = sources.begin(); iter != sources.end();) {
      const Thunk* src = *iter;
      // `dst` depends on `src`.
      int src_stream =
          stream_assignment_->StreamNumberForHlo(*thunk_to_hlo_.at(src));
      int src_order = FindOrDie(thunk_to_total_order, src);
      if (src_order <= last_dependency(dst_stream, src_stream)) {
        iter = sources.erase(iter);
      } else {
        last_dependency(dst_stream, src_stream) = src_order;
        ++iter;
      }
    }
    if (sources.empty()) {
      depends_on_.erase(dst);
    }
  }
}

const std::list<const Thunk*>& ThunkSchedule::DependsOn(
    const Thunk* thunk) const {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPSthunk_scheduleDTcc mht_4(mht_4_v, 323, "", "./tensorflow/compiler/xla/service/gpu/thunk_schedule.cc", "ThunkSchedule::DependsOn");

  if (depends_on_.contains(thunk)) {
    return FindOrDie(depends_on_, thunk);
  } else {
    return empty_thunk_list_;
  }
}

std::string ThunkSchedule::ToString() const {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPSthunk_scheduleDTcc mht_5(mht_5_v, 334, "", "./tensorflow/compiler/xla/service/gpu/thunk_schedule.cc", "ThunkSchedule::ToString");

  if (thunks_->empty()) {
    return "No thunks.";
  }

  auto get_thunk_annotation = [&](const Thunk* thunk) -> std::string {
    auto iter = thunk_to_hlo_.find(thunk);
    if (iter != thunk_to_hlo_.end() && iter->second != nullptr) {
      return iter->second->ToString();
    } else {
      return "(no HloInstruction)";
    }
  };

  std::string result = "Total order:\n";
  absl::StrAppend(&result, thunks_->ToString(0, get_thunk_annotation));
  absl::StrAppend(&result, "\nDependencies:\n");
  for (const auto& entry : depends_on_) {
    const Thunk* dependent = entry.first;
    for (const Thunk* dependency : entry.second) {
      auto dependent_iter = thunk_to_hlo_.find(dependent);
      auto dependency_iter = thunk_to_hlo_.find(dependency);
      if (dependent_iter != thunk_to_hlo_.end() &&
          dependency_iter != thunk_to_hlo_.end()) {
        absl::StrAppend(&result, "\t", dependent_iter->second->name(),
                        " depends on ", dependency_iter->second->name(), "\n");
      }
    }
  }
  return result;
}

}  // namespace gpu
}  // namespace xla
