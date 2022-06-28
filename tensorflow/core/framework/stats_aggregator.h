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
#ifndef TENSORFLOW_CORE_FRAMEWORK_STATS_AGGREGATOR_H_
#define TENSORFLOW_CORE_FRAMEWORK_STATS_AGGREGATOR_H_
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
class MHTracer_DTPStensorflowPScorePSframeworkPSstats_aggregatorDTh {
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
   MHTracer_DTPStensorflowPScorePSframeworkPSstats_aggregatorDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSframeworkPSstats_aggregatorDTh() {
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


#include <memory>
#include <string>

#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/lib/gtl/array_slice.h"

namespace tensorflow {

class Summary;
class SummaryWriterInterface;
namespace data {

// A `StatsAggregator` accumulates statistics incrementally. A
// `StatsAggregator` can accumulate multiple different statistics, distinguished
// by a string name.
//
// The class currently supports accumulating `Histogram`, `scalar` objects and
// tfstreamz metrics, and we expect to add other methods in future.
//
// NOTE(mrry): `StatsAggregator` is a virtual interface because we anticipate
// that many different implementations will have the same interface. For
// example, we have different implementations in "stats_aggregator_ops.cc" for
// simple in-memory implementation that integrates with the pull-based summary
// API, and for the push-based `SummaryWriterInterface`, and we may add
// implementations that work well with other custom monitoring services.
class StatsAggregator {
 public:
  virtual ~StatsAggregator() {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSstats_aggregatorDTh mht_0(mht_0_v, 214, "", "./tensorflow/core/framework/stats_aggregator.h", "~StatsAggregator");
}

  // Add the given `values` to the histogram with the given `name`. Each
  // element of `values` will be treated as a separate sample in the histogram.
  virtual void AddToHistogram(const string& name,
                              gtl::ArraySlice<double> values,
                              int64_t global_step) = 0;

  // TODO(shivaniagrawal): consistency in double and float usage.
  // Add the given `value` as Scalar with the given `name`.
  virtual void AddScalar(const string& name, float value,
                         int64_t global_step) = 0;

  // Stores a protocol buffer representation of the aggregator state in the
  // given `out_summary`.
  virtual void EncodeToProto(Summary* out_summary) = 0;

  // Sets a `summary_writer` with this stats_aggregator.
  virtual Status SetSummaryWriter(SummaryWriterInterface* summary_writer) = 0;

  // Increment the `label` cell of metrics mapped with `name` by given `value`.
  virtual void IncrementCounter(const string& name, const string& label,
                                int64_t val) = 0;
};

// A `StatsAggregatorResource` wraps a sharable `StatsAggregator` as a resource
// in the TensorFlow resource manager.
//
// NOTE(mrry): This class is separate from `StatsAggregator` in order to
// simplify the memory management of the shared object. Most users of
// `StatsAggregator` interact with a `std::shared_ptr<StatsAggregator>` whereas
// the `ResourceBase` API requires explicit reference counting.
class StatsAggregatorResource : public ResourceBase {
 public:
  // Creates a new resource from the given `stats_aggregator`.
  StatsAggregatorResource(std::unique_ptr<StatsAggregator> stats_aggregator)
      : stats_aggregator_(stats_aggregator.release()) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSstats_aggregatorDTh mht_1(mht_1_v, 253, "", "./tensorflow/core/framework/stats_aggregator.h", "StatsAggregatorResource");
}

  // Returns the wrapped `StatsAggregator`.
  std::shared_ptr<StatsAggregator> stats_aggregator() const {
    return stats_aggregator_;
  }

  string DebugString() const override {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSstats_aggregatorDTh mht_2(mht_2_v, 263, "", "./tensorflow/core/framework/stats_aggregator.h", "DebugString");
 return "StatsAggregatorResource"; }

 private:
  const std::shared_ptr<StatsAggregator> stats_aggregator_;
};

}  // namespace data
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_FRAMEWORK_STATS_AGGREGATOR_H_
