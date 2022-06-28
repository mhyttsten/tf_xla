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
class MHTracer_DTPStensorflowPScorePSprofilerPSconvertPSop_metrics_db_combinerDTcc {
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
   MHTracer_DTPStensorflowPScorePSprofilerPSconvertPSop_metrics_db_combinerDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSprofilerPSconvertPSop_metrics_db_combinerDTcc() {
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

#include "tensorflow/core/profiler/convert/op_metrics_db_combiner.h"

#include "absl/container/flat_hash_map.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/profiler/protobuf/op_metrics.pb.h"

namespace tensorflow {
namespace profiler {
namespace {

using OperationType = OpMetrics::MemoryAccessed::OperationType;

void CombinePrecisionStats(const PrecisionStats& src, PrecisionStats* dst) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSconvertPSop_metrics_db_combinerDTcc mht_0(mht_0_v, 197, "", "./tensorflow/core/profiler/convert/op_metrics_db_combiner.cc", "CombinePrecisionStats");

  dst->set_compute_16bit_ps(src.compute_16bit_ps() + dst->compute_16bit_ps());
  dst->set_compute_32bit_ps(src.compute_32bit_ps() + dst->compute_32bit_ps());
}

}  // namespace

void CopyOpMetricsMetadata(const OpMetrics& src, OpMetrics* dst) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSconvertPSop_metrics_db_combinerDTcc mht_1(mht_1_v, 207, "", "./tensorflow/core/profiler/convert/op_metrics_db_combiner.cc", "CopyOpMetricsMetadata");

  DCHECK(dst != nullptr);
  DCHECK_EQ(src.hlo_module_id(), dst->hlo_module_id());
  DCHECK_EQ(src.name(), dst->name());
  if (dst->long_name().empty()) {
    dst->set_long_name(src.long_name());
  }
  if (dst->category().empty()) {
    dst->set_category(src.category());
  }
  if (dst->provenance().empty()) {
    dst->set_provenance(src.provenance());
  }
  if (dst->deduplicated_name().empty()) {
    dst->set_deduplicated_name(src.deduplicated_name());
  }
  if (!dst->has_layout() && src.has_layout()) {
    *dst->mutable_layout() = src.layout();
  }
  if (!dst->has_children() && src.has_children()) {
    *dst->mutable_children() = src.children();
  }
}

void CombineOpMetrics(const OpMetrics& src, OpMetrics* dst) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSconvertPSop_metrics_db_combinerDTcc mht_2(mht_2_v, 234, "", "./tensorflow/core/profiler/convert/op_metrics_db_combiner.cc", "CombineOpMetrics");

  DCHECK(dst != nullptr);
  if (dst->occurrences() == 0) {
    dst->set_min_time_ps(src.min_time_ps());
  } else {
    dst->set_min_time_ps(std::min(src.min_time_ps(), dst->min_time_ps()));
  }
  dst->set_is_eager(dst->is_eager() || src.is_eager());
  dst->set_occurrences(src.occurrences() + dst->occurrences());
  dst->set_time_ps(src.time_ps() + dst->time_ps());
  dst->set_self_time_ps(src.self_time_ps() + dst->self_time_ps());
  dst->set_flops(src.flops() + dst->flops());
  dst->set_bytes_accessed(src.bytes_accessed() + dst->bytes_accessed());
  CombineMemoryAccessedBreakdown(src.memory_accessed_breakdown(),
                                 dst->mutable_memory_accessed_breakdown());
  dst->set_dma_stall_ps(src.dma_stall_ps() + dst->dma_stall_ps());
}

void CombineMemoryAccessedBreakdown(
    const protobuf::RepeatedPtrField<OpMetrics_MemoryAccessed>& src,
    protobuf::RepeatedPtrField<OpMetrics_MemoryAccessed>* dst) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSconvertPSop_metrics_db_combinerDTcc mht_3(mht_3_v, 257, "", "./tensorflow/core/profiler/convert/op_metrics_db_combiner.cc", "CombineMemoryAccessedBreakdown");

  if (src.empty()) return;
  absl::flat_hash_map<std::pair<uint64 /*memory_space*/, OperationType>,
                      OpMetrics_MemoryAccessed*>
      dst_memory_accessed_map;
  for (auto& dst_memory_accessed : *dst) {
    dst_memory_accessed_map[{dst_memory_accessed.memory_space(),
                             dst_memory_accessed.operation_type()}] =
        &dst_memory_accessed;
  }
  for (const auto& src_memory_accessed : src) {
    uint64 memory_space = src_memory_accessed.memory_space();
    OperationType operation_type = src_memory_accessed.operation_type();
    auto*& dst_memory_accessed =
        dst_memory_accessed_map[{memory_space, operation_type}];
    if (dst_memory_accessed == nullptr) {
      dst_memory_accessed = dst->Add();
      dst_memory_accessed->set_memory_space(memory_space);
      dst_memory_accessed->set_operation_type(operation_type);
    }
    dst_memory_accessed->set_bytes_accessed(
        src_memory_accessed.bytes_accessed() +
        dst_memory_accessed->bytes_accessed());
  }
}

void OpMetricsDbCombiner::Combine(const OpMetricsDb& src) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSconvertPSop_metrics_db_combinerDTcc mht_4(mht_4_v, 286, "", "./tensorflow/core/profiler/convert/op_metrics_db_combiner.cc", "OpMetricsDbCombiner::Combine");

  OpMetricsDb* dst = db();
  dst->set_total_host_infeed_enq_duration_ps(
      src.total_host_infeed_enq_duration_ps() +
      dst->total_host_infeed_enq_duration_ps());
  dst->set_total_host_infeed_enq_start_timestamp_ps_diff(
      src.total_host_infeed_enq_start_timestamp_ps_diff() +
      dst->total_host_infeed_enq_start_timestamp_ps_diff());
  dst->set_total_time_ps(src.total_time_ps() + dst->total_time_ps());
  dst->set_total_op_time_ps(src.total_op_time_ps() + dst->total_op_time_ps());
  CombinePrecisionStats(src.precision_stats(), dst->mutable_precision_stats());

  for (const auto& src_metrics : src.metrics_db()) {
    auto* dst_metrics = LookupOrInsertNewOpMetrics(src_metrics.hlo_module_id(),
                                                   src_metrics.name());
    CopyOpMetricsMetadata(src_metrics, dst_metrics);
    CombineOpMetrics(src_metrics, dst_metrics);
  }
}

}  // namespace profiler
}  // namespace tensorflow
