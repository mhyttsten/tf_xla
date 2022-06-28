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

#ifndef TENSORFLOW_CORE_GRAPPLER_COSTS_COST_ESTIMATOR_H_
#define TENSORFLOW_CORE_GRAPPLER_COSTS_COST_ESTIMATOR_H_
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
class MHTracer_DTPStensorflowPScorePSgrapplerPScostsPScost_estimatorDTh {
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
   MHTracer_DTPStensorflowPScorePSgrapplerPScostsPScost_estimatorDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSgrapplerPScostsPScost_estimatorDTh() {
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


#include <cmath>
#include <string>
#include <unordered_map>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/protobuf/config.pb.h"

namespace tensorflow {
class GraphDef;
class CostGraphDef;

namespace grappler {
struct GrapplerItem;

constexpr int64_t kMemoryUnknown = -1ll;
constexpr int64_t kZeroMemory = 0ll;

struct DeviceInfo {
  // Billions of operations executed per second.
  double gigaops;

  // Bandwidth to main memory in GB per second.
  double gb_per_sec;

  // Read bandwidth to intermediate memory in GB per second.
  double intermediate_read_gb_per_sec;

  // Write bandwidth to intermediate memory in GB per second.
  double intermediate_write_gb_per_sec;

  DeviceInfo()
      : gigaops(INFINITY),
        gb_per_sec(INFINITY),
        intermediate_read_gb_per_sec(INFINITY),
        intermediate_write_gb_per_sec(INFINITY) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPScostsPScost_estimatorDTh mht_0(mht_0_v, 224, "", "./tensorflow/core/grappler/costs/cost_estimator.h", "DeviceInfo");
}

  DeviceInfo(const DeviceInfo& input)
      : gigaops(input.gigaops),
        gb_per_sec(input.gb_per_sec),
        intermediate_read_gb_per_sec(input.intermediate_read_gb_per_sec),
        intermediate_write_gb_per_sec(input.intermediate_write_gb_per_sec) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPScostsPScost_estimatorDTh mht_1(mht_1_v, 233, "", "./tensorflow/core/grappler/costs/cost_estimator.h", "DeviceInfo");
}

  DeviceInfo(double gigaops, double gb_per_sec,
             double intermediate_read_gb_per_sec = INFINITY,
             double intermediate_write_gb_per_sec = INFINITY)
      : gigaops(gigaops),
        gb_per_sec(gb_per_sec),
        intermediate_read_gb_per_sec(intermediate_read_gb_per_sec),
        intermediate_write_gb_per_sec(intermediate_write_gb_per_sec) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPScostsPScost_estimatorDTh mht_2(mht_2_v, 244, "", "./tensorflow/core/grappler/costs/cost_estimator.h", "DeviceInfo");
}
};

// Holds the set of things we might want to estimate or measure in Grappler.
// Always produce execution time. Other fields are optional depending on the
// estimator being used.
struct Costs {
  // Returns a Costs structure with default values for all of the fields.
  inline Costs();

  // Builds a Costs structure with all zero values, rather than unknowns.
  static inline Costs ZeroCosts(bool inaccurate = false);

  struct MilliSeconds : std::chrono::milliseconds {
    MilliSeconds() : std::chrono::milliseconds(0) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPScostsPScost_estimatorDTh mht_3(mht_3_v, 261, "", "./tensorflow/core/grappler/costs/cost_estimator.h", "MilliSeconds");
}
    MilliSeconds(double d)
        : std::chrono::milliseconds(static_cast<int64_t>(d)) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPScostsPScost_estimatorDTh mht_4(mht_4_v, 266, "", "./tensorflow/core/grappler/costs/cost_estimator.h", "MilliSeconds");
}
    MilliSeconds(const std::chrono::milliseconds& d)
        : std::chrono::milliseconds(d) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPScostsPScost_estimatorDTh mht_5(mht_5_v, 271, "", "./tensorflow/core/grappler/costs/cost_estimator.h", "MilliSeconds");
}
    MilliSeconds& operator=(const std::chrono::milliseconds& d) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPScostsPScost_estimatorDTh mht_6(mht_6_v, 275, "", "./tensorflow/core/grappler/costs/cost_estimator.h", "=");

      std::chrono::milliseconds::operator=(d);
      return *this;
    }
  };
  struct MicroSeconds : std::chrono::microseconds {
    MicroSeconds() : std::chrono::microseconds(0) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPScostsPScost_estimatorDTh mht_7(mht_7_v, 284, "", "./tensorflow/core/grappler/costs/cost_estimator.h", "MicroSeconds");
}
    MicroSeconds(double d)
        : std::chrono::microseconds(static_cast<int64_t>(d)) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPScostsPScost_estimatorDTh mht_8(mht_8_v, 289, "", "./tensorflow/core/grappler/costs/cost_estimator.h", "MicroSeconds");
}
    MicroSeconds(const std::chrono::microseconds& d)
        : std::chrono::microseconds(d) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPScostsPScost_estimatorDTh mht_9(mht_9_v, 294, "", "./tensorflow/core/grappler/costs/cost_estimator.h", "MicroSeconds");
}
    MicroSeconds& operator=(const std::chrono::microseconds& d) {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPScostsPScost_estimatorDTh mht_10(mht_10_v, 298, "", "./tensorflow/core/grappler/costs/cost_estimator.h", "=");

      std::chrono::microseconds::operator=(d);
      return *this;
    }
    MilliSeconds asMilliSeconds() const {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPScostsPScost_estimatorDTh mht_11(mht_11_v, 305, "", "./tensorflow/core/grappler/costs/cost_estimator.h", "asMilliSeconds");

      return std::chrono::duration_cast<std::chrono::milliseconds>(*this);
    }
  };
  struct NanoSeconds : std::chrono::nanoseconds {
    NanoSeconds() : std::chrono::nanoseconds(0) {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPScostsPScost_estimatorDTh mht_12(mht_12_v, 313, "", "./tensorflow/core/grappler/costs/cost_estimator.h", "NanoSeconds");
}
    NanoSeconds(double d) : std::chrono::nanoseconds(static_cast<int64_t>(d)) {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPScostsPScost_estimatorDTh mht_13(mht_13_v, 317, "", "./tensorflow/core/grappler/costs/cost_estimator.h", "NanoSeconds");
}
    NanoSeconds(const std::chrono::nanoseconds& d)
        : std::chrono::nanoseconds(d) {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPScostsPScost_estimatorDTh mht_14(mht_14_v, 322, "", "./tensorflow/core/grappler/costs/cost_estimator.h", "NanoSeconds");
}
    NanoSeconds& operator=(const std::chrono::nanoseconds& d) {
   std::vector<std::string> mht_15_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPScostsPScost_estimatorDTh mht_15(mht_15_v, 326, "", "./tensorflow/core/grappler/costs/cost_estimator.h", "=");

      std::chrono::nanoseconds::operator=(d);
      return *this;
    }
    MicroSeconds asMicroSeconds() const {
   std::vector<std::string> mht_16_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPScostsPScost_estimatorDTh mht_16(mht_16_v, 333, "", "./tensorflow/core/grappler/costs/cost_estimator.h", "asMicroSeconds");

      return std::chrono::duration_cast<std::chrono::microseconds>(*this);
    }
    MilliSeconds asMilliSeconds() const {
   std::vector<std::string> mht_17_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPScostsPScost_estimatorDTh mht_17(mht_17_v, 339, "", "./tensorflow/core/grappler/costs/cost_estimator.h", "asMilliSeconds");

      return std::chrono::duration_cast<std::chrono::milliseconds>(*this);
    }
    static NanoSeconds infinity() {
   std::vector<std::string> mht_18_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPScostsPScost_estimatorDTh mht_18(mht_18_v, 345, "", "./tensorflow/core/grappler/costs/cost_estimator.h", "infinity");

      return NanoSeconds(std::chrono::nanoseconds::max());
    }
  };

  // We store all our times in nanoseconds. If needs be, we can always switch to
  // picoseconds in the future by updating this typedef.
  typedef NanoSeconds Duration;

  // Overall cost of running the graph; latency.
  Duration execution_time;

  // Computation cost of running the graph.
  Duration compute_time;

  // Memory access cost of running the graph.
  Duration memory_time;

  // Intermediate memory access cost of running the graph
  Duration intermediate_memory_time;
  Duration intermediate_memory_read_time;   // Intermediate memory read cost.
  Duration intermediate_memory_write_time;  // Intermediate memory write cost.

  // This field can be a very pessimistic estimate of the main memory
  // requirements of a graph. For example, it might assume that all activations
  // are live for all of a graph's execution.
  int64_t max_memory;  // Maximum main memory requirement in bytes over all ops.
  int64_t persistent_memory;
  int64_t temporary_memory;

  // Output memory usage per port.
  absl::flat_hash_map<int32_t, int64_t> output_tensor_size_bytes;

  // Track persistent versus temporary memory.
  absl::flat_hash_set<int32_t> persistent_output_ports;

  // These fields are used for TPU-related estimations. They are per-op
  // maximums, so each op is evaluated independently, but we want the maximum of
  // the value over all ops.
  int64_t max_per_op_buffers;    // Sum of all buffers used by the ops.
  int64_t max_per_op_streaming;  // Ignore largest input buffer, assuming it
                                 // streams from main memory.

  // Number of ops included in this Costs in total.
  // Default initialized to be one.
  int64_t num_ops_total = 1;
  // If the time estimation is inaccurate.
  bool inaccurate = false;
  // Number of ops that are estimated with unknown shapes.
  int64_t num_ops_with_unknown_shapes = 0;
  // TODO(pcma): include a counter for total inaccurate ops and counters for
  // other reasons causing the inaccuracy

  // Max possible memory usage per device.
  std::unordered_map<string, uint64> estimated_max_memory_per_device;
};

inline std::ostream& operator<<(std::ostream& os, const Costs::MilliSeconds d) {
   std::vector<std::string> mht_19_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPScostsPScost_estimatorDTh mht_19(mht_19_v, 405, "", "./tensorflow/core/grappler/costs/cost_estimator.h", "operator<<");

  os << d.count() << "ms";
  return os;
}
inline std::ostream& operator<<(std::ostream& os, const Costs::MicroSeconds d) {
   std::vector<std::string> mht_20_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPScostsPScost_estimatorDTh mht_20(mht_20_v, 412, "", "./tensorflow/core/grappler/costs/cost_estimator.h", "operator<<");

  os << d.count() << "us";
  return os;
}
inline std::ostream& operator<<(std::ostream& os, const Costs::NanoSeconds d) {
   std::vector<std::string> mht_21_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPScostsPScost_estimatorDTh mht_21(mht_21_v, 419, "", "./tensorflow/core/grappler/costs/cost_estimator.h", "operator<<");

  os << d.count() << "ns";
  return os;
}

Costs::Costs() {
   std::vector<std::string> mht_22_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPScostsPScost_estimatorDTh mht_22(mht_22_v, 427, "", "./tensorflow/core/grappler/costs/cost_estimator.h", "Costs::Costs");

  execution_time = Duration::zero();
  compute_time = Duration::zero();
  memory_time = Duration::zero();
  intermediate_memory_time = Duration::zero();
  max_memory = kMemoryUnknown;
  persistent_memory = kMemoryUnknown;
  temporary_memory = kMemoryUnknown;
  max_per_op_buffers = kMemoryUnknown;
  max_per_op_streaming = kMemoryUnknown;
}

Costs Costs::ZeroCosts(bool inaccurate) {
   std::vector<std::string> mht_23_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPScostsPScost_estimatorDTh mht_23(mht_23_v, 442, "", "./tensorflow/core/grappler/costs/cost_estimator.h", "Costs::ZeroCosts");

  Costs costs;
  costs.execution_time = Duration::zero();
  costs.compute_time = Duration::zero();
  costs.memory_time = Duration::zero();
  costs.intermediate_memory_time = Duration::zero();
  costs.max_memory = kZeroMemory;
  costs.persistent_memory = kZeroMemory;
  costs.temporary_memory = kZeroMemory;
  costs.max_per_op_buffers = kZeroMemory;
  costs.max_per_op_streaming = kZeroMemory;
  costs.inaccurate = inaccurate;
  return costs;
}

Costs CombineCosts(const Costs& left, const Costs& right);

// Multiplies Costs by a scalar.
// Equivalent to applying CombineCosts "multiplier" times.
Costs MultiplyCosts(const Costs& costs, int multiplier);

// Given a GrapperItem and an optimized implementation of the corresponding
// TensorFlow graph, the CostEstimator attempts to predicts the actual cost of
// running the graph.
class CostEstimator {
 public:
  virtual ~CostEstimator() {
   std::vector<std::string> mht_24_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPScostsPScost_estimatorDTh mht_24(mht_24_v, 471, "", "./tensorflow/core/grappler/costs/cost_estimator.h", "~CostEstimator");
}

  // Initializes the estimator for the specified grappler item.
  // The estimator shouldn't be used if this function returns any status other
  // that OK.
  virtual Status Initialize(const GrapplerItem& item) = 0;

  // Predicts the cost of running the given optimized version of the grappler
  // item.
  // If a RunMetadata is passed, it will be populated with detailed information
  // about the cost of running each operation of the optimized graph.
  // if a double value is passed, it will be set to a value that reflects the
  // overall cost of running the graph (e.g. the latency of the computation).
  // Returns a status that indicate is the performance could be estimated or
  // not.
  virtual Status PredictCosts(const GraphDef& optimized_graph,
                              RunMetadata* run_metadata, Costs* cost) const = 0;
};

}  // end namespace grappler
}  // end namespace tensorflow

#endif  // TENSORFLOW_CORE_GRAPPLER_COSTS_COST_ESTIMATOR_H_
