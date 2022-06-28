/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_CORE_LIB_MONITORING_COUNTER_H_
#define TENSORFLOW_CORE_LIB_MONITORING_COUNTER_H_
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
class MHTracer_DTPStensorflowPScorePSlibPSmonitoringPScounterDTh {
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
   MHTracer_DTPStensorflowPScorePSlibPSmonitoringPScounterDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSlibPSmonitoringPScounterDTh() {
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


// clang-format off
// Required for IS_MOBILE_PLATFORM
#include "tensorflow/core/platform/platform.h"
// clang-format on

// We replace this implementation with a null implementation for mobile
// platforms.
#ifdef IS_MOBILE_PLATFORM

#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {
namespace monitoring {

// CounterCell which has a null implementation.
class CounterCell {
 public:
  CounterCell() {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSlibPSmonitoringPScounterDTh mht_0(mht_0_v, 207, "", "./tensorflow/core/lib/monitoring/counter.h", "CounterCell");
}
  ~CounterCell() {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSlibPSmonitoringPScounterDTh mht_1(mht_1_v, 211, "", "./tensorflow/core/lib/monitoring/counter.h", "~CounterCell");
}

  void IncrementBy(int64 step) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSlibPSmonitoringPScounterDTh mht_2(mht_2_v, 216, "", "./tensorflow/core/lib/monitoring/counter.h", "IncrementBy");
}
  int64 value() const {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSlibPSmonitoringPScounterDTh mht_3(mht_3_v, 220, "", "./tensorflow/core/lib/monitoring/counter.h", "value");
 return 0; }

 private:
  TF_DISALLOW_COPY_AND_ASSIGN(CounterCell);
};

// Counter which has a null implementation.
template <int NumLabels>
class Counter {
 public:
  ~Counter() {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSlibPSmonitoringPScounterDTh mht_4(mht_4_v, 233, "", "./tensorflow/core/lib/monitoring/counter.h", "~Counter");
}

  template <typename... MetricDefArgs>
  static Counter* New(MetricDefArgs&&... metric_def_args) {
    return new Counter<NumLabels>();
  }

  template <typename... Labels>
  CounterCell* GetCell(const Labels&... labels) {
    return &default_counter_cell_;
  }

  Status GetStatus() {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSlibPSmonitoringPScounterDTh mht_5(mht_5_v, 248, "", "./tensorflow/core/lib/monitoring/counter.h", "GetStatus");
 return Status::OK(); }

 private:
  Counter() {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePSlibPSmonitoringPScounterDTh mht_6(mht_6_v, 254, "", "./tensorflow/core/lib/monitoring/counter.h", "Counter");
}

  CounterCell default_counter_cell_;

  TF_DISALLOW_COPY_AND_ASSIGN(Counter);
};

}  // namespace monitoring
}  // namespace tensorflow

#else  // IS_MOBILE_PLATFORM

#include <array>
#include <atomic>
#include <map>

#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/monitoring/collection_registry.h"
#include "tensorflow/core/lib/monitoring/metric_def.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/thread_annotations.h"

namespace tensorflow {
namespace monitoring {

// CounterCell stores each value of an Counter.
//
// A cell can be passed off to a module which may repeatedly update it without
// needing further map-indexing computations. This improves both encapsulation
// (separate modules can own a cell each, without needing to know about the map
// to which both cells belong) and performance (since map indexing and
// associated locking are both avoided).
//
// This class is thread-safe.
class CounterCell {
 public:
  explicit CounterCell(int64_t value) : value_(value) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePSlibPSmonitoringPScounterDTh mht_7(mht_7_v, 295, "", "./tensorflow/core/lib/monitoring/counter.h", "CounterCell");
}
  ~CounterCell() {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScorePSlibPSmonitoringPScounterDTh mht_8(mht_8_v, 299, "", "./tensorflow/core/lib/monitoring/counter.h", "~CounterCell");
}

  // Atomically increments the value by step.
  // REQUIRES: Step be non-negative.
  void IncrementBy(int64_t step);

  // Retrieves the current value.
  int64_t value() const;

 private:
  std::atomic<int64_t> value_;

  TF_DISALLOW_COPY_AND_ASSIGN(CounterCell);
};

// A stateful class for updating a cumulative integer metric.
//
// This class encapsulates a set of values (or a single value for a label-less
// metric). Each value is identified by a tuple of labels. The class allows the
// user to increment each value.
//
// Counter allocates storage and maintains a cell for each value. You can
// retrieve an individual cell using a label-tuple and update it separately.
// This improves performance since operations related to retrieval, like
// map-indexing and locking, are avoided.
//
// This class is thread-safe.
template <int NumLabels>
class Counter {
 public:
  ~Counter() {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScorePSlibPSmonitoringPScounterDTh mht_9(mht_9_v, 332, "", "./tensorflow/core/lib/monitoring/counter.h", "~Counter");

    // Deleted here, before the metric_def is destroyed.
    registration_handle_.reset();
  }

  // Creates the metric based on the metric-definition arguments.
  //
  // Example;
  // auto* counter_with_label = Counter<1>::New("/tensorflow/counter",
  //   "Tensorflow counter", "MyLabelName");
  template <typename... MetricDefArgs>
  static Counter* New(MetricDefArgs&&... metric_def_args);

  // Retrieves the cell for the specified labels, creating it on demand if
  // not already present.
  template <typename... Labels>
  CounterCell* GetCell(const Labels&... labels) TF_LOCKS_EXCLUDED(mu_);

  Status GetStatus() {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScorePSlibPSmonitoringPScounterDTh mht_10(mht_10_v, 353, "", "./tensorflow/core/lib/monitoring/counter.h", "GetStatus");
 return status_; }

 private:
  explicit Counter(
      const MetricDef<MetricKind::kCumulative, int64_t, NumLabels>& metric_def)
      : metric_def_(metric_def),
        registration_handle_(CollectionRegistry::Default()->Register(
            &metric_def_, [&](MetricCollectorGetter getter) {
              auto metric_collector = getter.Get(&metric_def_);

              mutex_lock l(mu_);
              for (const auto& cell : cells_) {
                metric_collector.CollectValue(cell.first, cell.second.value());
              }
            })) {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScorePSlibPSmonitoringPScounterDTh mht_11(mht_11_v, 370, "", "./tensorflow/core/lib/monitoring/counter.h", "Counter");

    if (registration_handle_) {
      status_ = Status::OK();
    } else {
      status_ = Status(tensorflow::error::Code::ALREADY_EXISTS,
                       "Another metric with the same name already exists.");
    }
  }

  mutable mutex mu_;

  Status status_;

  // The metric definition. This will be used to identify the metric when we
  // register it for collection.
  const MetricDef<MetricKind::kCumulative, int64_t, NumLabels> metric_def_;

  std::unique_ptr<CollectionRegistry::RegistrationHandle> registration_handle_;

  using LabelArray = std::array<string, NumLabels>;
  std::map<LabelArray, CounterCell> cells_ TF_GUARDED_BY(mu_);

  TF_DISALLOW_COPY_AND_ASSIGN(Counter);
};

////
//  Implementation details follow. API readers may skip.
////

inline void CounterCell::IncrementBy(const int64_t step) {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScorePSlibPSmonitoringPScounterDTh mht_12(mht_12_v, 402, "", "./tensorflow/core/lib/monitoring/counter.h", "CounterCell::IncrementBy");

  DCHECK_LE(0, step) << "Must not decrement cumulative metrics.";
  value_ += step;
}

inline int64_t CounterCell::value() const {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPScorePSlibPSmonitoringPScounterDTh mht_13(mht_13_v, 410, "", "./tensorflow/core/lib/monitoring/counter.h", "CounterCell::value");
 return value_; }

template <int NumLabels>
template <typename... MetricDefArgs>
Counter<NumLabels>* Counter<NumLabels>::New(
    MetricDefArgs&&... metric_def_args) {
  return new Counter<NumLabels>(
      MetricDef<MetricKind::kCumulative, int64_t, NumLabels>(
          std::forward<MetricDefArgs>(metric_def_args)...));
}

template <int NumLabels>
template <typename... Labels>
CounterCell* Counter<NumLabels>::GetCell(const Labels&... labels)
    TF_LOCKS_EXCLUDED(mu_) {
  // Provides a more informative error message than the one during array
  // construction below.
  static_assert(sizeof...(Labels) == NumLabels,
                "Mismatch between Counter<NumLabels> and number of labels "
                "provided in GetCell(...).");

  const LabelArray& label_array = {{labels...}};
  mutex_lock l(mu_);
  const auto found_it = cells_.find(label_array);
  if (found_it != cells_.end()) {
    return &(found_it->second);
  }
  return &(cells_
               .emplace(std::piecewise_construct,
                        std::forward_as_tuple(label_array),
                        std::forward_as_tuple(0))
               .first->second);
}

}  // namespace monitoring
}  // namespace tensorflow

#endif  // IS_MOBILE_PLATFORM
#endif  // TENSORFLOW_CORE_LIB_MONITORING_COUNTER_H_
