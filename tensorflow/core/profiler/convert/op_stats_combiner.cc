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
class MHTracer_DTPStensorflowPScorePSprofilerPSconvertPSop_stats_combinerDTcc {
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
   MHTracer_DTPStensorflowPScorePSprofilerPSconvertPSop_stats_combinerDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSprofilerPSconvertPSop_stats_combinerDTcc() {
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

#include "tensorflow/core/profiler/convert/op_stats_combiner.h"

#include "absl/container/flat_hash_map.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/protobuf.h"
#include "tensorflow/core/profiler/convert/op_metrics_db_combiner.h"
#include "tensorflow/core/profiler/convert/xplane_to_tf_functions.h"
#include "tensorflow/core/profiler/protobuf/diagnostics.pb.h"
#include "tensorflow/core/profiler/protobuf/hardware_types.pb.h"
#include "tensorflow/core/profiler/protobuf/kernel_stats.pb.h"
#include "tensorflow/core/profiler/protobuf/op_stats.pb.h"
#include "tensorflow/core/profiler/protobuf/steps_db.pb.h"
#include "tensorflow/core/profiler/utils/hardware_type_utils.h"
#include "tensorflow/core/profiler/utils/kernel_stats_utils.h"
#include "tensorflow/core/profiler/utils/step_intersection.h"

namespace tensorflow {
namespace profiler {

namespace {

// Combines the src PerCoreStepInfo into the dst PerCoreStepInfo.
void CombinePerCoreStepInfo(
    int src_host_id, const PerCoreStepInfo& src, bool use_incomplete_step,
    PerCoreStepInfo* dst,
    OpMetricsDbCombiner* hlo_metrics_db_complete_steps_only_combiner,
    OpMetricsDbCombiner* hlo_metrics_db_per_step_combiner) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSconvertPSop_stats_combinerDTcc mht_0(mht_0_v, 212, "", "./tensorflow/core/profiler/convert/op_stats_combiner.cc", "CombinePerCoreStepInfo");

  CombineCoreIdMap(src_host_id, src.step_info_per_core(),
                   dst->mutable_step_info_per_core());

  // Since we have assigned a new step number to the combined result, update
  // the step number on each core to this new step number.
  uint32 new_step_num = dst->step_num();
  for (auto& percore_stepinfo : *dst->mutable_step_info_per_core()) {
    auto& stepinfo = percore_stepinfo.second;
    stepinfo.set_step_num(new_step_num);
  }

  if (!use_incomplete_step) {
    hlo_metrics_db_complete_steps_only_combiner->Combine(src.hlo_metrics_db());
  }
  hlo_metrics_db_per_step_combiner->Combine(src.hlo_metrics_db());
  CombineCoreIdMap(src_host_id, src.all_reduce_db_per_core(),
                   dst->mutable_all_reduce_db_per_core());
  CombineCoreIdMap(src_host_id, src.core_id_to_replica_id_map(),
                   dst->mutable_core_id_to_replica_id_map());
}

void CombineStepDatabase(
    int src_host_id, const StepIntersection& step_intersection,
    const StepDatabaseResult& src, StepDatabaseResult* dst,
    OpMetricsDbCombiner* hlo_metrics_db_complete_steps_only_combiner,
    std::vector<OpMetricsDbCombiner>* hlo_metrics_db_per_step_combiners) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSconvertPSop_stats_combinerDTcc mht_1(mht_1_v, 241, "", "./tensorflow/core/profiler/convert/op_stats_combiner.cc", "CombineStepDatabase");

  if (src.use_incomplete_step()) dst->set_use_incomplete_step(true);
  uint32 src_first_step_idx = step_intersection.FirstStepIndex(src_host_id);
  for (uint32 i = 0; i < step_intersection.NumSteps(); i++) {
    CombinePerCoreStepInfo(
        src_host_id, src.step_sequence(src_first_step_idx + i),
        src.use_incomplete_step(), dst->mutable_step_sequence(i),
        hlo_metrics_db_complete_steps_only_combiner,
        &(*hlo_metrics_db_per_step_combiners)[i]);
  }
}

void CombineRunEnvironment(const RunEnvironment& src, RunEnvironment* dst) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSconvertPSop_stats_combinerDTcc mht_2(mht_2_v, 256, "", "./tensorflow/core/profiler/convert/op_stats_combiner.cc", "CombineRunEnvironment");

  dst->mutable_hostnames()->insert(src.hostnames().begin(),
                                   src.hostnames().end());
  dst->set_host_count(dst->hostnames_size());
  if (src.device_type() != "CPU") {
    dst->set_device_type(src.device_type());
    // TODO(b/111402648): Batch size may differ per-core. Currently, we report
    // the max batch size. We need to come up with a better measure.
    dst->set_per_core_batch_size(
        std::max(src.per_core_batch_size(), dst->per_core_batch_size()));
    dst->set_device_core_count(src.device_core_count() +
                               dst->device_core_count());
    // Replica count and num cores per replica must be same for all copies.
    dst->set_replica_count(std::max(src.replica_count(), dst->replica_count()));
    dst->set_num_cores_per_replica(
        std::max(src.num_cores_per_replica(), dst->num_cores_per_replica()));
    *dst->mutable_topology() = src.topology();
  } else if (dst->device_type().empty()) {
    dst->set_device_type(src.device_type());
  }
  dst->set_task_count(src.task_count() + dst->task_count());
  (*dst->mutable_host_independent_job_info()) = src.host_independent_job_info();
  for (const auto& job_info : src.host_dependent_job_info()) {
    *(dst->add_host_dependent_job_info()) = job_info;
  }
  dst->set_host_trace_level(src.host_trace_level());
}

// Combines the src PerfEnv into the dst PerfEnv.
void CombinePerfEnv(const PerfEnv& src, PerfEnv* dst) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSconvertPSop_stats_combinerDTcc mht_3(mht_3_v, 288, "", "./tensorflow/core/profiler/convert/op_stats_combiner.cc", "CombinePerfEnv");

  dst->set_peak_tera_flops_per_second(src.peak_tera_flops_per_second());
  dst->set_peak_hbm_bw_giga_bytes_per_second(
      src.peak_hbm_bw_giga_bytes_per_second());
  dst->set_ridge_point(src.ridge_point());
}

// Combines the src Diagnostics into the dst Diagnostics.
void CombineDiagnostics(const Diagnostics& src, Diagnostics* dst) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSconvertPSop_stats_combinerDTcc mht_4(mht_4_v, 299, "", "./tensorflow/core/profiler/convert/op_stats_combiner.cc", "CombineDiagnostics");

  dst->mutable_info()->MergeFrom(src.info());
  dst->mutable_warnings()->MergeFrom(src.warnings());
  dst->mutable_errors()->MergeFrom(src.errors());
}

// Combine the src OpStats into the dst OpStats.
void CombineOpStats(
    bool no_accelerator_in_system, int src_host_id, HardwareType hardware_type,
    const StepIntersection& step_intersection, const OpStats& src, OpStats* dst,
    OpMetricsDbCombiner* host_op_metrics_db_combiner,
    OpMetricsDbCombiner* device_op_metrics_db_combiner,
    OpMetricsDbCombiner* hlo_metrics_db_complete_steps_only_combiner,
    std::vector<OpMetricsDbCombiner>* hlo_metrics_db_per_step_combiners) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSconvertPSop_stats_combinerDTcc mht_5(mht_5_v, 315, "", "./tensorflow/core/profiler/convert/op_stats_combiner.cc", "CombineOpStats");

  // Combine host_metrics_db.
  host_op_metrics_db_combiner->Combine(src.host_op_metrics_db());
  // Combine device_metrics_db.
  device_op_metrics_db_combiner->Combine(src.device_op_metrics_db());

  // Combine step_db.
  if (!IsCoordinator(no_accelerator_in_system, hardware_type)) {
    CombineStepDatabase(src_host_id, step_intersection, src.step_db(),
                        dst->mutable_step_db(),
                        hlo_metrics_db_complete_steps_only_combiner,
                        hlo_metrics_db_per_step_combiners);
  }

  // Combine run environment info.
  CombineRunEnvironment(src.run_environment(), dst->mutable_run_environment());

  // Combine the perf environment info.
  CombinePerfEnv(src.perf_env(), dst->mutable_perf_env());

  // Combine diagnostics.
  CombineDiagnostics(src.diagnostics(), dst->mutable_diagnostics());

  // Combine kernel stats.
  dst->mutable_kernel_stats_db()->mutable_reports()->MergeFrom(
      src.kernel_stats_db().reports());

  // Combine tf-function stats.
  CombineTfFunctionDb(src.tf_function_db(), dst->mutable_tf_function_db());

  // Combine the mapping from core ID to details.
  CombineCoreIdMap(src_host_id, src.core_id_to_details(),
                   dst->mutable_core_id_to_details());
}

}  // namespace

bool IsCoordinator(bool no_accelerator_in_system, HardwareType hardware_type) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSconvertPSop_stats_combinerDTcc mht_6(mht_6_v, 355, "", "./tensorflow/core/profiler/convert/op_stats_combiner.cc", "IsCoordinator");

  // A host is a coordinator if:
  //   (1) The host doesn't have a device, and
  //   (2) The system does use accelerator (if not, it uses CPU only and so this
  //   host should be regarded as a worker as well).
  return !HasDevice(hardware_type) && !no_accelerator_in_system;
}

bool NoAcceleratorInSystem(const std::vector<OpStatsInfo>& all_op_stats_info) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSconvertPSop_stats_combinerDTcc mht_7(mht_7_v, 366, "", "./tensorflow/core/profiler/convert/op_stats_combiner.cc", "NoAcceleratorInSystem");

  for (const auto& op_stats_info : all_op_stats_info) {
    if (HasDevice(op_stats_info.hardware_type)) {
      return false;
    }
  }
  return true;
}

uint32 GlobalCoreId(int host_id, uint32 device_ordinal) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSconvertPSop_stats_combinerDTcc mht_8(mht_8_v, 378, "", "./tensorflow/core/profiler/convert/op_stats_combiner.cc", "GlobalCoreId");

  constexpr uint32 kMaxDevicesPerHost = 1000;  // power-of-10 for debuggability
  return host_id * kMaxDevicesPerHost + device_ordinal;
}

StepIntersection ComputeStepIntersectionToMergeOpStats(
    const std::vector<OpStatsInfo>& all_op_stats_info,
    uint32 max_step_per_host) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSconvertPSop_stats_combinerDTcc mht_9(mht_9_v, 388, "", "./tensorflow/core/profiler/convert/op_stats_combiner.cc", "ComputeStepIntersectionToMergeOpStats");

  bool no_accelerator_in_system = NoAcceleratorInSystem(all_op_stats_info);

  absl::flat_hash_map<uint32, const StepDatabaseResult*> per_host_step_db;
  for (const auto& op_stats_info : all_op_stats_info) {
    if (IsCoordinator(no_accelerator_in_system, op_stats_info.hardware_type))
      continue;
    // Includes only workers in per_host_step_db.
    per_host_step_db[op_stats_info.src_host_id] =
        &op_stats_info.op_stats->step_db();
  }

  return StepIntersection(max_step_per_host, per_host_step_db);
}

void CombineAllOpStats(const std::vector<OpStatsInfo>& all_op_stats_info,
                       const StepIntersection& step_intersection,
                       OpStats* combined_op_stats) {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSconvertPSop_stats_combinerDTcc mht_10(mht_10_v, 408, "", "./tensorflow/core/profiler/convert/op_stats_combiner.cc", "CombineAllOpStats");

  StepDatabaseResult* combined_step_db = combined_op_stats->mutable_step_db();
  // Initialize the StepDatabaseResult field that depends on the number of
  // steps.
  for (uint32 dst_step_num : step_intersection.DstStepNumbers()) {
    combined_step_db->add_step_sequence()->set_step_num(dst_step_num);
  }
  // Record the number of steps that are dropped.
  combined_step_db->set_num_steps_dropped(step_intersection.StepsDropped());

  combined_step_db->set_empty_intersect(step_intersection.EmptyIntersect());

  // Set the default value of per_core_batch_size in <combined_op_stats>
  combined_op_stats->mutable_run_environment()->set_per_core_batch_size(-1);

  // Initialize all the OpMetricsDbCombiners.
  OpMetricsDbCombiner host_op_metrics_db_combiner(
      combined_op_stats->mutable_host_op_metrics_db());
  OpMetricsDbCombiner device_op_metrics_db_combiner(
      combined_op_stats->mutable_device_op_metrics_db());
  OpMetricsDbCombiner hlo_metrics_db_complete_steps_only_combiner(
      combined_op_stats->mutable_hlo_metrics_db_complete_steps_only());
  std::vector<OpMetricsDbCombiner> hlo_metrics_db_per_step_combiners;
  hlo_metrics_db_per_step_combiners.reserve(
      combined_step_db->step_sequence_size());
  for (PerCoreStepInfo& step_info :
       *combined_step_db->mutable_step_sequence()) {
    hlo_metrics_db_per_step_combiners.emplace_back(
        step_info.mutable_hlo_metrics_db());
  }

  bool no_accelerator_in_system = NoAcceleratorInSystem(all_op_stats_info);

  for (const auto& op_stats_info : all_op_stats_info) {
    CombineOpStats(no_accelerator_in_system, op_stats_info.src_host_id,
                   op_stats_info.hardware_type, step_intersection,
                   *op_stats_info.op_stats, combined_op_stats,
                   &host_op_metrics_db_combiner, &device_op_metrics_db_combiner,
                   &hlo_metrics_db_complete_steps_only_combiner,
                   &hlo_metrics_db_per_step_combiners);
  }

  // Sorts all the kernel reports that have been merged by CombineTfOpStats and
  // keeps only the top kernel reports with long kernel duration.
  SortAndKeepTopKDurationKernelReportsInDb(
      combined_op_stats->mutable_kernel_stats_db());
}

}  // namespace profiler
}  // namespace tensorflow
