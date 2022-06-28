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

#ifndef TENSORFLOW_CORE_PROFILER_UTILS_OP_METRICS_DB_UTILS_H_
#define TENSORFLOW_CORE_PROFILER_UTILS_OP_METRICS_DB_UTILS_H_
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
class MHTracer_DTPStensorflowPScorePSprofilerPSutilsPSop_metrics_db_utilsDTh {
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
   MHTracer_DTPStensorflowPScorePSprofilerPSutilsPSop_metrics_db_utilsDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSprofilerPSutilsPSop_metrics_db_utilsDTh() {
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


#include <algorithm>
#include <string>

#include "absl/container/flat_hash_map.h"
#include "absl/strings/string_view.h"
#include "absl/types/optional.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/profiler/protobuf/op_metrics.pb.h"

namespace tensorflow {
namespace profiler {

// The name of OpMetrics to represent the idle time.
TF_CONST_INIT extern const absl::string_view kIdle;

// Helps build an op metrics database (borrowed).
// Enables fast lookup of existing ops and prevents the creation of duplicate
// ops. It is the user's responsibility to ensure an op metrics database
// outlives its builder, and that no ops are added to the database outside of
// the builder.
class OpMetricsDbBuilder {
 public:
  // Create with a borrowed op database.
  // REQUIRED: The op database must be empty.
  explicit OpMetricsDbBuilder(OpMetricsDb* db);

 protected:
  // Looks up the given OP name. If it is already in the database,
  // return its OpMetrics; otherwise, insert a new one.
  OpMetrics* LookupOrInsertNewOpMetrics(uint64 hlo_module_id,
                                        absl::string_view name);

  OpMetricsDb* db() {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSutilsPSop_metrics_db_utilsDTh mht_0(mht_0_v, 221, "", "./tensorflow/core/profiler/utils/op_metrics_db_utils.h", "db");
 return db_; }

 private:
  // Map op (hlo_module_id, name) to the corresponding metrics in the op
  // database.
  absl::flat_hash_map<uint64 /*hlo_module_id*/,
                      absl::flat_hash_map<std::string /*name*/, OpMetrics*>>
      op_metrics_map_;

  // The op database.
  OpMetricsDb* db_;
};

// Sets the total time for OpMetricsDb, ensuring idle time is not negative.
inline void SetTotalTimePs(OpMetricsDb& db, uint64_t total_time_ps) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSutilsPSop_metrics_db_utilsDTh mht_1(mht_1_v, 238, "", "./tensorflow/core/profiler/utils/op_metrics_db_utils.h", "SetTotalTimePs");

  db.set_total_time_ps(std::max(db.total_op_time_ps(), total_time_ps));
}

// Returns the total time in OpMetricsDb, optionally excluding the idle time.
inline uint64_t TotalTimePs(const OpMetricsDb& db, bool exclude_idle = false) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSutilsPSop_metrics_db_utilsDTh mht_2(mht_2_v, 246, "", "./tensorflow/core/profiler/utils/op_metrics_db_utils.h", "TotalTimePs");

  return exclude_idle ? db.total_op_time_ps() : db.total_time_ps();
}

// Returns the ratio of time that is idle (no op execution) over total time.
double IdleTimeRatio(const OpMetricsDb& db);

// Returns the idle time in picoseconds.
uint64 IdleTimePs(const OpMetricsDb& db);

// Adds an op representing idle time, i.e., the amount of time spent without any
// op execution.
// REQUIRED: All ops must have been added to the database and the total time
// must have been set.
void AddIdleOp(OpMetricsDb& db);

// Returns true if the given metrics represents idle time.
inline bool IsIdleOp(const OpMetrics& metrics) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSutilsPSop_metrics_db_utilsDTh mht_3(mht_3_v, 266, "", "./tensorflow/core/profiler/utils/op_metrics_db_utils.h", "IsIdleOp");

  return metrics.category() == kIdle;
}

// Returns the ratio of time spent sending data from the host to the device
// relative to the total time the host was active.
absl::optional<double> HostInfeedEnqueueRatio(const OpMetricsDb& db);

// Converts from the device op metrics to Tf-op metrics.
OpMetricsDb CreateTfMetricsDbFromDeviceOpMetricsDb(
    const OpMetricsDb& device_op_metrics_db, bool with_idle = true);

}  // namespace profiler
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_PROFILER_UTILS_OP_METRICS_DB_UTILS_H_
