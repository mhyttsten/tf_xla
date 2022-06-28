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

#ifndef TENSORFLOW_CORE_PROFILER_UTILS_KERNEL_STATS_UTILS_H_
#define TENSORFLOW_CORE_PROFILER_UTILS_KERNEL_STATS_UTILS_H_
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
class MHTracer_DTPStensorflowPScorePSprofilerPSutilsPSkernel_stats_utilsDTh {
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
   MHTracer_DTPStensorflowPScorePSprofilerPSutilsPSkernel_stats_utilsDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSprofilerPSutilsPSkernel_stats_utilsDTh() {
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


#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/strings/string_view.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/profiler/protobuf/kernel_stats.pb.h"

namespace tensorflow {
namespace profiler {

// Populates kernel launch information from a kKernelDetails XStat.
void ParseKernelLaunchParams(absl::string_view xstat_kernel_details,
                             KernelReport* kernel);

// Returns true if kernel uses TensorCores.
bool IsKernelUsingTensorCore(absl::string_view kernel_name);

// Returns true if operation is eligible to use TensorCores.
bool IsOpTensorCoreEligible(absl::string_view tf_op_name);

// Returns true if Einsum equation is eligible to use TensorCores.
bool IsEinsumTensorCoreEligible(absl::string_view equation);

// Less than comparator for Kernel Reports.
struct KernelReportLessThanComparator {
  bool operator()(const KernelReport& lhs, const KernelReport& rhs) const;
};

// Equal to comparator for Kernel Reports.
struct KernelReportEqualToComparator {
  bool operator()(const KernelReport& lhs, const KernelReport& rhs) const;
};

// Sorts kernel reorts by total duration descendingly.
// Keeps only the top kernel reports with long kernel duration in the given
// KernelStatsDb. Kernel reports with shorter kernel duration are dropped.
void SortAndKeepTopKDurationKernelReportsInDb(KernelStatsDb* kernel_stats_db);

struct KernelReportValue {
  uint64 total_duration_ns = 0;
  uint64 min_duration_ns = 0;
  uint64 max_duration_ns = 0;
  uint64 occurrences = 0;
};

struct KernelKeyWrap {
  const KernelReport* key;
  template <typename H>
  friend H AbslHashValue(H h, KernelKeyWrap wrap) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSutilsPSkernel_stats_utilsDTh mht_0(mht_0_v, 236, "", "./tensorflow/core/profiler/utils/kernel_stats_utils.h", "AbslHashValue");

    // Kernel reports are grouped by these fields, hence they are used as
    // hashing criteria.
    // clang-format off
    return H::combine(
        std::move(h),
        wrap.key->is_kernel_using_tensor_core(),
        wrap.key->is_op_tensor_core_eligible(),
        wrap.key->block_dim(0),
        wrap.key->block_dim(1),
        wrap.key->block_dim(2),
        wrap.key->grid_dim(0),
        wrap.key->grid_dim(1),
        wrap.key->grid_dim(2),
        wrap.key->registers_per_thread(),
        wrap.key->static_shmem_bytes(),
        wrap.key->dynamic_shmem_bytes(),
        wrap.key->name(),
        wrap.key->op_name());
    // clang-format on
  }
};

struct KernelHash {
  size_t operator()(const KernelReport& key) const {
    return absl::Hash<KernelKeyWrap>()(KernelKeyWrap{&key});
  }
};

using KernelReportMap =
    absl::flat_hash_map<KernelReport, KernelReportValue, KernelHash,
                        KernelReportEqualToComparator>;

// Copies the top kernel reports with long kernel duration into the given
// KernelStatsDb.
void CopyTopKDurationKernelReportsToDb(const KernelReportMap& reports,
                                       KernelStatsDb* dst);

// Inserts or aggregates KernelReports into the given KernelReportMap.
void InsertOrUpdateKernelReport(const KernelReport& kernel,
                                const KernelReportValue& value,
                                KernelReportMap* dst);

// Aggregates values from one KernelReportMap into another.
void MergeKernelReports(const KernelReportMap& reports, KernelReportMap* dst);

// Kernel stats aggregated at TF operation level.
struct OpLevelKernelStats {
  // Whether op is eligible to use TensorCore.
  bool is_op_tensor_core_eligible = false;
  // The accumulated duration of all the kernels launched in this op.
  uint64 total_duration_ns = 0;
  // The accumulated duration of all the kernels using TensorCore in this op.
  // If this value is not 0, at least one of the kernels launched by this op
  // is using TensorCore.
  uint64 tensor_core_duration_ns = 0;
};

using KernelStatsByOpName =
    absl::flat_hash_map<absl::string_view, OpLevelKernelStats>;

// Groups KernelReport in <kernel_stats_db> by tensorflow operation name.
KernelStatsByOpName GroupKernelReportsByOpName(
    const KernelStatsDb& kernel_stats_db);

}  // namespace profiler
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_PROFILER_UTILS_KERNEL_STATS_UTILS_H_
