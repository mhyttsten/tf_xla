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
class MHTracer_DTPStensorflowPScorePSprofilerPSutilsPSstep_intersectionDTcc {
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
   MHTracer_DTPStensorflowPScorePSprofilerPSutilsPSstep_intersectionDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSprofilerPSutilsPSstep_intersectionDTcc() {
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

/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/core/profiler/utils/step_intersection.h"

#include "tensorflow/core/lib/gtl/map_util.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/profiler/utils/timespan.h"

namespace tensorflow {
namespace profiler {

namespace {

// Returns the timespan in this step (across all cores).
Timespan StepTimespan(const PerCoreStepInfo& percore_stepinfo) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSutilsPSstep_intersectionDTcc mht_0(mht_0_v, 196, "", "./tensorflow/core/profiler/utils/step_intersection.cc", "StepTimespan");

  uint64 min_ps = kuint64max;
  uint64 max_ps = 0;
  for (const auto& core_stepinfo : percore_stepinfo.step_info_per_core()) {
    const auto& stepinfo = core_stepinfo.second;
    uint64 begin_ps = stepinfo.begin_ps();
    uint64 end_ps = begin_ps + stepinfo.duration_ps();
    min_ps = std::min(min_ps, begin_ps);
    max_ps = std::max(max_ps, end_ps);
  }
  return (min_ps < max_ps) ? Timespan::FromEndPoints(min_ps, max_ps)
                           : Timespan();
}

// Returns the timespan across all steps in the given step_db.
Timespan AllStepsTimespan(const StepDatabaseResult& step_db) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSutilsPSstep_intersectionDTcc mht_1(mht_1_v, 214, "", "./tensorflow/core/profiler/utils/step_intersection.cc", "AllStepsTimespan");

  uint64 min_ps = kuint64max;
  uint64 max_ps = 0;
  for (const auto& step : step_db.step_sequence()) {
    Timespan timespan = StepTimespan(step);
    uint64 begin_ps = timespan.begin_ps();
    uint64 end_ps = timespan.end_ps();
    min_ps = std::min(min_ps, begin_ps);
    max_ps = std::max(max_ps, end_ps);
  }
  return (min_ps < max_ps) ? Timespan::FromEndPoints(min_ps, max_ps)
                           : Timespan();
}

struct AlignmentInfo {
  StepsAlignment alignment;
  double similarity;
};

// Computes the similarity between the given two steps. The closer their
// timespans are, the larger is the similarity.
double StepSimilarity(const PerCoreStepInfo& subordinate_step,
                      const PerCoreStepInfo& chief_step) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSutilsPSstep_intersectionDTcc mht_2(mht_2_v, 239, "", "./tensorflow/core/profiler/utils/step_intersection.cc", "StepSimilarity");

  Timespan subordinate_timespan = StepTimespan(subordinate_step);
  Timespan chief_timespan = StepTimespan(chief_step);
  return chief_timespan.OverlappedDurationPs(subordinate_timespan);
}

// If the subordinate steps and the chief steps are aligned at the given anchor
// points (i.e. at the subordinate_anchor step on the subordinate sequence, at
// the chief_anchor step on the chief sequence), returns the corresponding
// AlignmentInfo.
AlignmentInfo ComputeAlignmentInfo(const StepDatabaseResult& subordinate,
                                   uint32 subordinate_anchor,
                                   const StepDatabaseResult& chief,
                                   uint32 chief_anchor) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSutilsPSstep_intersectionDTcc mht_3(mht_3_v, 255, "", "./tensorflow/core/profiler/utils/step_intersection.cc", "ComputeAlignmentInfo");

  // Assumes that the step at subordinate_anchor on the subordinate sequence is
  // aligned with the step at the chief_anchor on the chief sequence. Then the
  // number of steps before the anchor is the minimum of the number of steps
  // before the anchor in the subordinate and that before the anchor in the
  // chief. Similarly, the number of steps after the anchor is the minimum of
  // the number of steps after the anchor in the subordinate and that after the
  // anchor in the chief.
  uint32 pre_anchor_steps = std::min(subordinate_anchor, chief_anchor);
  uint32 post_anchor_steps =
      std::min(subordinate.step_sequence_size() - subordinate_anchor,
               chief.step_sequence_size() - chief_anchor);
  // total number of steps aligned = pre_anchor_steps + post_anchor_steps.
  uint32 alignment_steps = pre_anchor_steps + post_anchor_steps;

  double similarity = 0;
  // Where the aligned steps begin on the subordinate sequence.
  uint32 begin_subordinate_idx = subordinate_anchor - pre_anchor_steps;
  // Where the aligned steps begin on the chief sequence.
  uint32 begin_chief_idx = chief_anchor - pre_anchor_steps;

  for (uint32 i = 0; i < alignment_steps; i++) {
    // Accumulates the similarity at each step.
    similarity +=
        StepSimilarity(subordinate.step_sequence(begin_subordinate_idx + i),
                       chief.step_sequence(begin_chief_idx + i));
  }
  StepsAlignment alignment = {begin_subordinate_idx, begin_chief_idx,
                              alignment_steps};
  return {alignment, similarity};
}

// Returns the best alignment for aligning subordinate against chief.
StepsAlignment FindStepsAlignment(const StepDatabaseResult& subordinate,
                                  const StepDatabaseResult& chief) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSutilsPSstep_intersectionDTcc mht_4(mht_4_v, 292, "", "./tensorflow/core/profiler/utils/step_intersection.cc", "FindStepsAlignment");

  double max_similarity = -1;
  StepsAlignment alignment = {0, 0, 0};
  if (subordinate.step_sequence_size() == 0 || chief.step_sequence_size() == 0)
    return alignment;
  for (auto c = 0; c < chief.step_sequence_size(); c++) {
    AlignmentInfo info =
        ComputeAlignmentInfo(subordinate, /*subordinate_anchor=*/0, chief, c);
    if (info.similarity <= max_similarity) continue;
    max_similarity = info.similarity;
    alignment = info.alignment;
  }
  for (auto s = 1; s < subordinate.step_sequence_size(); s++) {
    // s starts at 1 instead of 0, because the loop above already considers
    // (s=0, c=0).
    AlignmentInfo info =
        ComputeAlignmentInfo(subordinate, s, chief, /*chief_anchor=*/0);
    if (info.similarity <= max_similarity) continue;
    max_similarity = info.similarity;
    alignment = info.alignment;
  }
  return alignment;
}

std::string StringStepsAlignment(const StepsAlignment& alignment) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSutilsPSstep_intersectionDTcc mht_5(mht_5_v, 319, "", "./tensorflow/core/profiler/utils/step_intersection.cc", "StringStepsAlignment");

  return absl::StrCat(
      "[begin_subordinate_idx: ", alignment.begin_subordinate_idx,
      ", begin_chief_idx: ", alignment.begin_chief_idx,
      ", num_steps: ", alignment.num_steps, "]");
}

std::string StringDstStepNumbers(const std::vector<uint32>& step_numbers) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSutilsPSstep_intersectionDTcc mht_6(mht_6_v, 329, "", "./tensorflow/core/profiler/utils/step_intersection.cc", "StringDstStepNumbers");

  std::string str;
  absl::StrAppend(&str, "[");
  for (auto i = 0; i < step_numbers.size(); i++) {
    if (i > 0) absl::StrAppend(&str, ", ");
    absl::StrAppend(&str, step_numbers[i]);
  }
  absl::StrAppend(&str, "]");
  return str;
}

std::string StringSrcToDstIndexMap(uint32 src_first_step_idx,
                                   uint32 num_steps) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSutilsPSstep_intersectionDTcc mht_7(mht_7_v, 344, "", "./tensorflow/core/profiler/utils/step_intersection.cc", "StringSrcToDstIndexMap");

  std::string str;
  absl::StrAppend(&str, "[");
  for (auto i = 0; i < num_steps; i++) {
    if (i > 0) absl::StrAppend(&str, ", ");
    absl::StrAppend(&str, src_first_step_idx + i, ":", i);
  }
  absl::StrAppend(&str, "]");
  return str;
}

}  // namespace

StepIntersection::StepIntersection(
    uint32 max_steps,
    const absl::flat_hash_map<uint32, const StepDatabaseResult*>&
        perhost_stepdb) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSutilsPSstep_intersectionDTcc mht_8(mht_8_v, 363, "", "./tensorflow/core/profiler/utils/step_intersection.cc", "StepIntersection::StepIntersection");

  empty_intersect_ = false;

  // Figures out the host with the shortest timespan among their steps (called
  // this host the "chief").
  chief_host_id_ = kuint32max;
  uint64 min_duration_ps = kuint64max;
  const StepDatabaseResult* chief_step_db = nullptr;
  for (const auto& hostid_stepdb : perhost_stepdb) {
    auto host_id = hostid_stepdb.first;
    const auto& step_db = hostid_stepdb.second;
    Timespan timespan = AllStepsTimespan(*step_db);
    if (timespan.duration_ps() < min_duration_ps) {
      chief_host_id_ = host_id;
      chief_step_db = step_db;
      min_duration_ps = timespan.duration_ps();
    }
  }
  if (chief_host_id_ == kuint32max) {
    // There is no step at all on any host.
    steps_dropped_ = 0;
    begin_chief_idx_ = 0;
    end_chief_idx_ = 0;
    return;
  }

  uint32 max_begin_chief_idx = 0;
  uint32 min_end_chief_idx = kuint32max;
  // Aligns the steps in all hosts with those in the chief.
  for (const auto& hostid_stepdb : perhost_stepdb) {
    auto host_id = hostid_stepdb.first;
    const auto& step_db = hostid_stepdb.second;
    if (host_id == chief_host_id_) {
      // Simply aligns with itself.
      perhost_alignment_[host_id] = {
          /*begin_subordinate_idx=*/0, /*begin_chief_idx=*/0,
          static_cast<uint32>(step_db->step_sequence_size())};
    } else {
      perhost_alignment_[host_id] =
          FindStepsAlignment(*step_db, *chief_step_db);
    }
    // Intersects this host's alignment with other hosts' alignments.
    uint32 host_begin_chief_idx = perhost_alignment_[host_id].begin_chief_idx;
    max_begin_chief_idx = std::max(max_begin_chief_idx, host_begin_chief_idx);
    uint32 host_end_chief_idx = perhost_alignment_[host_id].begin_chief_idx +
                                perhost_alignment_[host_id].num_steps;
    min_end_chief_idx = std::min(min_end_chief_idx, host_end_chief_idx);
  }
  if (max_begin_chief_idx > min_end_chief_idx) {
    // The intersection is empty.
    steps_dropped_ = 0;
    begin_chief_idx_ = 0;
    end_chief_idx_ = 0;
    empty_intersect_ = true;
    return;
  }

  begin_chief_idx_ = max_begin_chief_idx;

  // Takes max_steps into account.
  uint32 num_steps = min_end_chief_idx - max_begin_chief_idx;
  if (num_steps > max_steps) {
    steps_dropped_ = num_steps - max_steps;
    // TODO(ckluk): Drops from both ends to avoid incomplete steps at the
    // beginning and end of the profile.
    end_chief_idx_ = max_begin_chief_idx + max_steps;
  } else {
    steps_dropped_ = 0;
    end_chief_idx_ = min_end_chief_idx;
  }
}

std::vector<uint32> StepIntersection::DstStepNumbers() const {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSutilsPSstep_intersectionDTcc mht_9(mht_9_v, 438, "", "./tensorflow/core/profiler/utils/step_intersection.cc", "StepIntersection::DstStepNumbers");

  // TODO(ckluk): Honors training-loop boundaries (if more than one loop
  // sampled).
  std::vector<uint32> result;
  result.reserve(NumSteps());
  for (uint32 i = 0; i < NumSteps(); i++) {
    result.push_back(i);
  }
  return result;
}

uint32 StepIntersection::FirstStepIndex(uint32 host_id) const {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSutilsPSstep_intersectionDTcc mht_10(mht_10_v, 452, "", "./tensorflow/core/profiler/utils/step_intersection.cc", "StepIntersection::FirstStepIndex");

  const auto* alignment = gtl::FindOrNull(perhost_alignment_, host_id);
  if (alignment == nullptr) return 0;
  DCHECK(alignment->begin_chief_idx <= begin_chief_idx_);
  uint32 shift = begin_chief_idx_ - alignment->begin_chief_idx;
  uint32 begin_subordinate_idx = alignment->begin_subordinate_idx + shift;
  return begin_subordinate_idx;
}

std::string StepIntersection::DebugString() const {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSutilsPSstep_intersectionDTcc mht_11(mht_11_v, 464, "", "./tensorflow/core/profiler/utils/step_intersection.cc", "StepIntersection::DebugString");

  std::string str;
  absl::StrAppend(&str, "chief host id_: ", chief_host_id_, "\n");
  absl::StrAppend(&str, "begin_chief_idx_: ", begin_chief_idx_,
                  ", num_steps: ", NumSteps(), "\n");
  absl::StrAppend(
      &str, "DstStepNumbers(): ", StringDstStepNumbers(DstStepNumbers()), "\n");

  std::vector<uint32> host_ids;
  host_ids.reserve(perhost_alignment_.size());
  for (const auto& hostid_alignment : perhost_alignment_) {
    auto host_id = hostid_alignment.first;
    host_ids.push_back(host_id);
  }
  absl::c_sort(host_ids);

  absl::StrAppend(&str, "perhost_alignment:\n");
  for (const auto host_id : host_ids) {
    const auto* ptr = gtl::FindOrNull(perhost_alignment_, host_id);
    if (ptr == nullptr) continue;
    absl::StrAppend(&str, "host: ", host_id,
                    ", step-alignment: ", StringStepsAlignment(*ptr), "\n");
  }
  absl::StrAppend(&str, "SrcToDstIndexMap():\n");
  for (const auto host_id : host_ids) {
    absl::StrAppend(&str, "host: ", host_id, ", src-to-dst-index-map: ",
                    StringSrcToDstIndexMap(FirstStepIndex(host_id), NumSteps()),
                    "\n");
  }
  return str;
}

}  // namespace profiler
}  // namespace tensorflow
