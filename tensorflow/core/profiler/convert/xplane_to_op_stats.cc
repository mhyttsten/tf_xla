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
class MHTracer_DTPStensorflowPScorePSprofilerPSconvertPSxplane_to_op_statsDTcc {
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
   MHTracer_DTPStensorflowPScorePSprofilerPSconvertPSxplane_to_op_statsDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSprofilerPSconvertPSxplane_to_op_statsDTcc() {
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

#include "tensorflow/core/profiler/convert/xplane_to_op_stats.h"

#include <string>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/profiler/convert/op_metrics_db_combiner.h"
#include "tensorflow/core/profiler/convert/op_stats_combiner.h"
#include "tensorflow/core/profiler/convert/step_events_to_steps_db.h"
#include "tensorflow/core/profiler/convert/xplane_to_kernel_stats_db.h"
#include "tensorflow/core/profiler/convert/xplane_to_op_metrics_db.h"
#include "tensorflow/core/profiler/convert/xplane_to_step_events.h"
#include "tensorflow/core/profiler/convert/xplane_to_tf_functions.h"
#include "tensorflow/core/profiler/protobuf/diagnostics.pb.h"
#include "tensorflow/core/profiler/protobuf/kernel_stats.pb.h"
#include "tensorflow/core/profiler/protobuf/op_metrics.pb.h"
#include "tensorflow/core/profiler/protobuf/op_stats.pb.h"
#include "tensorflow/core/profiler/protobuf/steps_db.pb.h"
#include "tensorflow/core/profiler/protobuf/tf_function.pb.h"
#include "tensorflow/core/profiler/protobuf/xplane.pb.h"
#include "tensorflow/core/profiler/utils/device_caps_utils.h"
#include "tensorflow/core/profiler/utils/event_span.h"
#include "tensorflow/core/profiler/utils/hardware_type_utils.h"
#include "tensorflow/core/profiler/utils/kernel_stats_utils.h"
#include "tensorflow/core/profiler/utils/step_intersection.h"
#include "tensorflow/core/profiler/utils/tf_op_utils.h"
#include "tensorflow/core/profiler/utils/tf_xplane_visitor.h"
#include "tensorflow/core/profiler/utils/xplane_schema.h"
#include "tensorflow/core/profiler/utils/xplane_utils.h"
#include "tensorflow/core/profiler/utils/xplane_visitor.h"

namespace tensorflow {
namespace profiler {

PerfEnv MakePerfEnv(double peak_tera_flops_per_second,
                    double peak_hbm_bw_giga_bytes_per_second) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSconvertPSxplane_to_op_statsDTcc mht_0(mht_0_v, 223, "", "./tensorflow/core/profiler/convert/xplane_to_op_stats.cc", "MakePerfEnv");

  PerfEnv result;
  result.set_peak_tera_flops_per_second(peak_tera_flops_per_second);
  result.set_peak_hbm_bw_giga_bytes_per_second(
      peak_hbm_bw_giga_bytes_per_second);
  result.set_ridge_point(peak_tera_flops_per_second * 1000 /
                         peak_hbm_bw_giga_bytes_per_second);
  return result;
}

PerfEnv GetPerfEnvFromXPlane(const XPlane& device_plane) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSconvertPSxplane_to_op_statsDTcc mht_1(mht_1_v, 236, "", "./tensorflow/core/profiler/convert/xplane_to_op_stats.cc", "GetPerfEnvFromXPlane");

  DeviceCapabilities cap = GetDeviceCaps(device_plane);
  return MakePerfEnv(GetFlopMaxThroughputPerSM(cap) / 1000 * cap.num_cores(),
                     cap.memory_bandwidth() / 1e9);
}

namespace {

void SetRunEnvironment(const XSpace& space, int32_t accelerator_count,
                       RunEnvironment* env) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSconvertPSxplane_to_op_statsDTcc mht_2(mht_2_v, 248, "", "./tensorflow/core/profiler/convert/xplane_to_op_stats.cc", "SetRunEnvironment");

  // Currently, we only support profiling one host and one program.
  env->set_host_count(1);
  env->set_task_count(1);
  for (const auto& hostname : space.hostnames()) {
    std::vector<std::string> hostname_split = absl::StrSplit(hostname, ':');
    (*env->mutable_hostnames())[hostname_split[0]] = true;
  }
  env->set_device_type(accelerator_count > 0 ? "GPU" : "CPU");
  env->set_device_core_count(accelerator_count);
}

}  // namespace

void PropagateXSpaceDiagnosticsToOpStats(const XSpace& space,
                                         OpStats* op_stats) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSconvertPSxplane_to_op_statsDTcc mht_3(mht_3_v, 266, "", "./tensorflow/core/profiler/convert/xplane_to_op_stats.cc", "PropagateXSpaceDiagnosticsToOpStats");

  if (!space.errors().empty()) {
    absl::flat_hash_set<std::string> unique_errors;
    unique_errors.insert(space.errors().begin(), space.errors().end());
    *op_stats->mutable_diagnostics()->mutable_errors() = {unique_errors.begin(),
                                                          unique_errors.end()};
  }
  if (!space.warnings().empty()) {
    absl::flat_hash_set<std::string> unique_warnings;
    unique_warnings.insert(space.warnings().begin(), space.warnings().end());
    *op_stats->mutable_diagnostics()->mutable_warnings() = {
        unique_warnings.begin(), unique_warnings.end()};
  }
}

OpStats ConvertXSpaceToOpStats(const XSpace& space,
                               const OpStatsOptions& options) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSconvertPSxplane_to_op_statsDTcc mht_4(mht_4_v, 285, "", "./tensorflow/core/profiler/convert/xplane_to_op_stats.cc", "ConvertXSpaceToOpStats");

  const XPlane* host_plane = FindPlaneWithName(space, kHostThreadsPlaneName);
  std::vector<const XPlane*> device_planes =
      FindPlanesWithPrefix(space, kGpuPlanePrefix);
  OpStats op_stats;
  StepEvents step_events;
  PropagateXSpaceDiagnosticsToOpStats(space, &op_stats);
  // Convert device planes.
  OpMetricsDbCombiner op_metrics_db_combiner(
      op_stats.mutable_device_op_metrics_db());
  SetRunEnvironment(space, device_planes.size(),
                    op_stats.mutable_run_environment());

  KernelReportMap reports;
  absl::string_view gpu_model = "";

  // TODO(b/161942993) parallelize XPlane processing per thread.
  for (const XPlane* device_trace : device_planes) {
    if (options.generate_op_metrics_db) {
      if (!op_stats.has_perf_env()) {
        *op_stats.mutable_perf_env() = GetPerfEnvFromXPlane(*device_trace);
      }
      OpMetricsDb device_op_metrics_db =
          ConvertDeviceTraceXPlaneToOpMetricsDb(*device_trace);
      op_metrics_db_combiner.Combine(device_op_metrics_db);
    }
    if (gpu_model.empty()) {
      gpu_model = GpuModelName(GetDeviceCaps(*device_trace));
    }
    if (options.generate_step_db) {
      StepEvents device_step_events =
          ConvertDeviceTraceXPlaneToStepEvents(*device_trace);
      CombineStepEvents(device_step_events, &step_events);
    }
    if (options.generate_kernel_stats_db) {
      ConvertDeviceTraceXPlaneToKernelReports(*device_trace,
                                              /*on_kernel_fn=*/{}, &reports);
    }
  }

  if (!gpu_model.empty()) {
    // Overwrites the device type with the more specific GPU model name.
    op_stats.mutable_run_environment()->set_device_type(std::string(gpu_model));
  }

  // Combine into reports.
  if (options.generate_kernel_stats_db) {
    CopyTopKDurationKernelReportsToDb(reports,
                                      op_stats.mutable_kernel_stats_db());
  }

  bool has_device = !device_planes.empty();
  // Convert a host plane.
  if (host_plane) {
    if (options.generate_op_metrics_db) {
      *op_stats.mutable_host_op_metrics_db() =
          ConvertHostThreadsXPlaneToOpMetricsDb(*host_plane);
    }
    if (options.generate_step_db) {
      const StepEvents* device_step_events =
          has_device ? &step_events : nullptr;
      StepEvents host_step_events =
          ConvertHostThreadsXPlaneToStepEvents(*host_plane, device_step_events);
      CombineStepEvents(host_step_events, &step_events);
    }
  }
  if (options.generate_step_db) {
    StepEvents nonoverlapped_step_events =
        ToNonOverlappedStepEvents(step_events);
    *op_stats.mutable_step_db() = ConvertStepEventsToStepDb(
        has_device, options.maybe_drop_incomplete_steps,
        nonoverlapped_step_events);
    *op_stats.mutable_device_op_metrics_db()->mutable_precision_stats() =
        ComputePrecisionStats(nonoverlapped_step_events);
  }

  CoreDetails& details =
      (*op_stats.mutable_core_id_to_details())[kDefaultGpuLocalCoreId];
  details.set_hostname(space.hostnames().empty() ? "localhost"
                                                 : space.hostnames(0));
  return op_stats;
}

Status ConvertMultiXSpacesToCombinedOpStats(const std::vector<XSpace>& xspaces,
                                            const OpStatsOptions& options,
                                            OpStats* combined_op_stats) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSconvertPSxplane_to_op_statsDTcc mht_5(mht_5_v, 373, "", "./tensorflow/core/profiler/convert/xplane_to_op_stats.cc", "ConvertMultiXSpacesToCombinedOpStats");

  // A shortcut code path for a single XSpace. There is no need to merge OpStats
  // if there is only a single XSpace.
  if (xspaces.size() == 1) {
    *combined_op_stats = ConvertXSpaceToOpStats(xspaces[0], options);
    return Status::OK();
  }

  // Read multiple XSpaces and convert to multiple OpStats.
  std::vector<OpStats> all_op_stats;
  all_op_stats.reserve(xspaces.size());
  for (const XSpace& xspace : xspaces) {
    all_op_stats.push_back(ConvertXSpaceToOpStats(xspace, options));
  }

  // Combine OpStats.
  std::vector<OpStatsInfo> all_op_stats_info;
  all_op_stats_info.reserve(all_op_stats.size());
  for (int i = 0; i < all_op_stats.size(); i++) {
    all_op_stats_info.emplace_back(
        &all_op_stats[i],
        ParseHardwareType(all_op_stats[i].run_environment().device_type()), i);
  }

  // Do not limit the maximum number of steps during the merge of OpStats.
  StepIntersection step_intersection =
      ComputeStepIntersectionToMergeOpStats(all_op_stats_info, kuint32max);
  CombineAllOpStats(all_op_stats_info, step_intersection, combined_op_stats);

  return Status::OK();
}

}  // namespace profiler
}  // namespace tensorflow
