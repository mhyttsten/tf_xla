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
class MHTracer_DTPStensorflowPScorePSprofilerPSconvertPSstep_events_to_steps_dbDTcc {
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
   MHTracer_DTPStensorflowPScorePSprofilerPSconvertPSstep_events_to_steps_dbDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSprofilerPSconvertPSstep_events_to_steps_dbDTcc() {
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
#include "tensorflow/core/profiler/convert/step_events_to_steps_db.h"

#include <sstream>
#include <utility>
#include <vector>

#include "google/protobuf/any.pb.h"
#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_map.h"
#include "tensorflow/core/lib/gtl/map_util.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/profiler/protobuf/steps_db.pb.h"
#include "tensorflow/core/profiler/utils/event_span.h"
#include "tensorflow/core/profiler/utils/timespan.h"

namespace tensorflow {
namespace profiler {

// Local core id should start from 1.
const uint32 kDefaultGpuLocalCoreId = 1;

namespace {

// Converts from StepDetails to StepInfoResult.
StepInfoResult ConvertStepDetailsToStepInfo(bool has_device, int64_t step_num,
                                            const StepDetails& step_details) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSconvertPSstep_events_to_steps_dbDTcc mht_0(mht_0_v, 210, "", "./tensorflow/core/profiler/convert/step_events_to_steps_db.cc", "ConvertStepDetailsToStepInfo");

  GenericStepBreakdown generic;
  Timespan step_time = step_details.StepTime();
  auto& type_ps = *(generic.mutable_type_ps());
  uint64 total_event_duration = 0;
  for (const auto& event : step_details.Events()) {
    // Ignore event duration outside the step marker.
    uint64 event_duration = step_time.OverlappedDurationPs(event.span);
    type_ps[event.type] += event_duration;
    total_event_duration += event_duration;
  }
  if (total_event_duration < step_time.duration_ps()) {
    // Some time in the step is not associated with any event. Classify them as
    // "unknown time".
    type_ps[UNKNOWN_TIME] += step_time.duration_ps() - total_event_duration;
  }
  // Determines if this particular step is a well-formed one.
  bool well_formed_step = has_device ? (type_ps.contains(DEVICE_COMPUTE_16) ||
                                        type_ps.contains(DEVICE_COMPUTE_32))
                                     : type_ps.contains(HOST_COMPUTE);
  StepInfoResult step_info;
  step_info.mutable_step_breakdown()->PackFrom(generic);
  if (well_formed_step) {
    step_info.set_step_num(step_num);
    step_info.set_step_name(step_details.StepName());
    step_info.set_begin_ps(step_time.begin_ps());
    step_info.set_duration_ps(step_time.duration_ps());
  } else {
    // For a non-well-formed step, sets its duration to 0 so that it will be
    // ignored by the caller of this function.
    step_info.set_duration_ps(0);
  }
  return step_info;
}

string DebugGenericStepBreakdown(const GenericStepBreakdown& generic) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSconvertPSstep_events_to_steps_dbDTcc mht_1(mht_1_v, 248, "", "./tensorflow/core/profiler/convert/step_events_to_steps_db.cc", "DebugGenericStepBreakdown");

  std::ostringstream out;
  uint64 total_ps = 0;
  const auto& type_ps_map = generic.type_ps();
  for (const auto& type_ps : type_ps_map) {
    total_ps += type_ps.second;
  }
  out << "Total ps = " << total_ps << std::endl;
  for (int type = LAST_EVENT_TYPE; type >= 0; --type) {
    const auto* ps = gtl::FindOrNull(type_ps_map, type);
    if (ps == nullptr) continue;
    double percent = (*ps * 100.0) / total_ps;
    auto event_type = static_cast<EventType>(type);
    out << PrintEventType(event_type) << ": " << percent << "%"
        << ", ps = " << *ps << std::endl;
  }
  return out.str();
}

string DebugStepInfo(const StepInfoResult& step_info) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSconvertPSstep_events_to_steps_dbDTcc mht_2(mht_2_v, 270, "", "./tensorflow/core/profiler/convert/step_events_to_steps_db.cc", "DebugStepInfo");

  std::ostringstream out;
  out << "step_num=" << step_info.step_num()
      << ", duration_ps=" << step_info.duration_ps()
      << ", begin_ps=" << step_info.begin_ps() << std::endl;
  GenericStepBreakdown generic;
  if (step_info.step_breakdown().UnpackTo(&generic)) {
    out << "Generic step breakdown:" << std::endl;
    out << DebugGenericStepBreakdown(generic) << std::endl;
  } else {
    out << step_info.step_breakdown().DebugString() << std::endl;
  }
  return out.str();
}

}  // namespace

StepDatabaseResult ConvertStepEventsToStepDb(
    bool has_device, bool maybe_drop_incomplete_steps,
    const StepEvents& nonoverlapped_step_events) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSconvertPSstep_events_to_steps_dbDTcc mht_3(mht_3_v, 292, "", "./tensorflow/core/profiler/convert/step_events_to_steps_db.cc", "ConvertStepEventsToStepDb");

  StepDatabaseResult step_db;
  // Gets sorted step numbers.
  std::vector<int64_t> step_numbers;
  step_numbers.reserve(nonoverlapped_step_events.size());
  for (const auto& step_events : nonoverlapped_step_events) {
    step_numbers.push_back(step_events.first);
  }
  absl::c_sort(step_numbers);
  for (const auto& step : step_numbers) {
    const auto* step_details = gtl::FindOrNull(nonoverlapped_step_events, step);
    if (step_details == nullptr) continue;
    StepInfoResult step_info =
        ConvertStepDetailsToStepInfo(has_device, step, *step_details);
    if (step_info.duration_ps() == 0)
      continue;  // Do not include non-well-formed steps.
    PerCoreStepInfo per_core_step_info;
    per_core_step_info.set_step_num(step);
    // When we generated StepEvents, we already put events from all device
    // cores and cpu threads on this host into a single event stream, therefore
    // we can't separate them anymore. Simply assigns all events to Core-0.
    (*per_core_step_info.mutable_step_info_per_core())[kDefaultGpuLocalCoreId] =
        std::move(step_info);
    VLOG(2) << std::endl
            << "step_id: " << step << ", step_info:" << std::endl
            << DebugStepInfo((
                   *per_core_step_info
                        .mutable_step_info_per_core())[kDefaultGpuLocalCoreId]);
    // Populates the collective ops information.
    auto& collectives = *per_core_step_info.mutable_all_reduce_db_per_core();
    for (const auto& it : step_details->Collectives()) {
      collectives[it.first] = it.second;
    }
    // Populates the device transfer stats for this step.
    auto& device_memory_transfers =
        *per_core_step_info.mutable_device_memory_transfers();
    for (const auto& dma : step_details->DeviceMemoryTransfers()) {
      *device_memory_transfers.Add() = dma;
    }
    // The remaining fields in PerCoreStepInfo are not filled.
    *step_db.add_step_sequence() = per_core_step_info;
  }

  // If we are using sampling mode and we get enough steps, we would like to
  // drop the incomplete steps at the beginning and the end.
  // (Sometimes CUTPI instrumentation will prolong the first step too).
  int kDropIncomplteteStepThreshold = 5;
  if (maybe_drop_incomplete_steps &&
      step_db.step_sequence_size() > kDropIncomplteteStepThreshold) {
    step_db.mutable_step_sequence()->erase(
        step_db.mutable_step_sequence()->begin());
    step_db.mutable_step_sequence()->RemoveLast();
  }
  return step_db;
}

}  // namespace profiler
}  // namespace tensorflow
