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

#ifndef TENSORFLOW_CORE_PROFILER_CONVERT_OP_METRICS_TO_RECORD_H_
#define TENSORFLOW_CORE_PROFILER_CONVERT_OP_METRICS_TO_RECORD_H_
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
class MHTracer_DTPStensorflowPScorePSprofilerPSconvertPSop_metrics_to_recordDTh {
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
   MHTracer_DTPStensorflowPScorePSprofilerPSconvertPSop_metrics_to_recordDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSprofilerPSconvertPSop_metrics_to_recordDTh() {
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

#include "tensorflow/core/profiler/protobuf/op_metrics.pb.h"
#include "tensorflow/core/profiler/utils/math_utils.h"

namespace tensorflow {
namespace profiler {

std::vector<const OpMetrics*> SortedOpMetricsDb(const OpMetricsDb& metrics_db,
                                                int max_records = -1);

template <typename Record>
inline void SetExecutionTimes(const OpMetrics& metrics, Record* record) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSconvertPSop_metrics_to_recordDTh mht_0(mht_0_v, 200, "", "./tensorflow/core/profiler/convert/op_metrics_to_record.h", "SetExecutionTimes");

  record->set_occurrences(metrics.occurrences());
  record->set_total_time_in_us(PicoToMicro(metrics.time_ps()));
  record->set_avg_time_in_us(
      SafeDivide(record->total_time_in_us(), metrics.occurrences()));
  record->set_total_self_time_in_us(PicoToMicro(metrics.self_time_ps()));
  record->set_avg_self_time_in_us(
      SafeDivide(record->total_self_time_in_us(), metrics.occurrences()));
}

template <typename Record>
inline void SetTpuUnitFractions(const OpMetrics& metrics, Record* record) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSconvertPSop_metrics_to_recordDTh mht_1(mht_1_v, 214, "", "./tensorflow/core/profiler/convert/op_metrics_to_record.h", "SetTpuUnitFractions");

  record->set_dma_stall_fraction(
      SafeDivide(metrics.dma_stall_ps(), metrics.time_ps()));
}

template <typename Record>
inline void SetRankAndTimeFractions(double total_time_us,
                                    const Record& prev_record, Record* record) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSconvertPSop_metrics_to_recordDTh mht_2(mht_2_v, 224, "", "./tensorflow/core/profiler/convert/op_metrics_to_record.h", "SetRankAndTimeFractions");

  record->set_rank(prev_record.rank() + 1);
  record->set_total_self_time_as_fraction(
      SafeDivide(record->total_self_time_in_us(), total_time_us));
  record->set_cumulative_total_self_time_as_fraction(
      prev_record.cumulative_total_self_time_as_fraction() +
      record->total_self_time_as_fraction());
}

template <typename Record>
inline void SetRankAndDeviceTimeFractions(double total_time_us,
                                          const Record& prev_record,
                                          Record* record) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSconvertPSop_metrics_to_recordDTh mht_3(mht_3_v, 239, "", "./tensorflow/core/profiler/convert/op_metrics_to_record.h", "SetRankAndDeviceTimeFractions");

  record->set_rank(prev_record.rank() + 1);
  record->set_device_total_self_time_as_fraction(
      SafeDivide(record->total_self_time_in_us(), total_time_us));
  record->set_device_cumulative_total_self_time_as_fraction(
      prev_record.device_cumulative_total_self_time_as_fraction() +
      record->device_total_self_time_as_fraction());
}

template <typename Record>
inline void SetRankAndHostTimeFractions(double total_time_us,
                                        const Record& prev_record,
                                        Record* record) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSconvertPSop_metrics_to_recordDTh mht_4(mht_4_v, 254, "", "./tensorflow/core/profiler/convert/op_metrics_to_record.h", "SetRankAndHostTimeFractions");

  record->set_rank(prev_record.rank() + 1);
  record->set_host_total_self_time_as_fraction(
      SafeDivide(record->total_self_time_in_us(), total_time_us));
  record->set_host_cumulative_total_self_time_as_fraction(
      prev_record.host_cumulative_total_self_time_as_fraction() +
      record->host_total_self_time_as_fraction());
}

template <typename Record>
inline void SetRooflineMetrics(const OpMetrics& metrics,
                               double ridge_point_operational_intensity,
                               Record* record) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSconvertPSop_metrics_to_recordDTh mht_5(mht_5_v, 269, "", "./tensorflow/core/profiler/convert/op_metrics_to_record.h", "SetRooflineMetrics");

  using ::tensorflow::profiler::PicoToNano;
  record->set_measured_flop_rate(
      SafeDivide(metrics.flops(), PicoToNano(metrics.time_ps())));
  record->set_measured_memory_bw(
      SafeDivide(metrics.bytes_accessed(), PicoToNano(metrics.time_ps())));
  record->set_operational_intensity(
      SafeDivide(metrics.flops(), metrics.bytes_accessed()));
  record->set_bound_by((metrics.bytes_accessed() != 0)
                           ? ((record->operational_intensity() >=
                               ridge_point_operational_intensity)
                                  ? "Compute"
                                  : "Memory")
                           : ((metrics.flops() != 0) ? "Compute" : "Unknown"));
}

}  // namespace profiler
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_PROFILER_CONVERT_OP_METRICS_TO_RECORD_H_
