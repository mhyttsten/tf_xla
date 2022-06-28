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

#ifndef TENSORFLOW_CORE_LIB_MONITORING_GAUGE_H_
#define TENSORFLOW_CORE_LIB_MONITORING_GAUGE_H_
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
class MHTracer_DTPStensorflowPScorePSlibPSmonitoringPSgaugeDTh {
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
   MHTracer_DTPStensorflowPScorePSlibPSmonitoringPSgaugeDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSlibPSmonitoringPSgaugeDTh() {
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

#include <functional>
#include <string>

#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {
namespace monitoring {

// GaugeCell which has a null implementation.
template <typename T>
class GaugeCell {
 public:
 public:
  GaugeCell() {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSlibPSmonitoringPSgaugeDTh mht_0(mht_0_v, 212, "", "./tensorflow/core/lib/monitoring/gauge.h", "GaugeCell");
}
  ~GaugeCell() {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSlibPSmonitoringPSgaugeDTh mht_1(mht_1_v, 216, "", "./tensorflow/core/lib/monitoring/gauge.h", "~GaugeCell");
}

  void Set(const T& value) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSlibPSmonitoringPSgaugeDTh mht_2(mht_2_v, 221, "", "./tensorflow/core/lib/monitoring/gauge.h", "Set");
}
  T value() const {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSlibPSmonitoringPSgaugeDTh mht_3(mht_3_v, 225, "", "./tensorflow/core/lib/monitoring/gauge.h", "value");
 return T(); }

 private:
  TF_DISALLOW_COPY_AND_ASSIGN(GaugeCell);
};

// Gauge which has a null implementation.
template <typename ValueType, int NumLabels>
class Gauge {
 public:
  ~Gauge() {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSlibPSmonitoringPSgaugeDTh mht_4(mht_4_v, 238, "", "./tensorflow/core/lib/monitoring/gauge.h", "~Gauge");
}

  template <typename... MetricDefArgs>
  static Gauge* New(MetricDefArgs&&... metric_def_args) {
    static_assert(
        std::is_same<ValueType, int64>::value ||
            std::is_same<ValueType, std::string>::value ||
            std::is_same<ValueType, bool>::value ||
            std::is_same<ValueType, std::function<int64()> >::value ||
            std::is_same<ValueType, std::function<std::string()> >::value ||
            std::is_same<ValueType, std::function<bool()> >::value,
        "Gauge only allows bool, int64, and string types.");
    return new Gauge();
  }

  template <typename... Labels>
  GaugeCell<ValueType>* GetCell(const Labels&... labels) {
    return &default_gauge_cell_;
  }

  Status GetStatus() {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSlibPSmonitoringPSgaugeDTh mht_5(mht_5_v, 261, "", "./tensorflow/core/lib/monitoring/gauge.h", "GetStatus");
 return Status::OK(); }

 private:
  Gauge() {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePSlibPSmonitoringPSgaugeDTh mht_6(mht_6_v, 267, "", "./tensorflow/core/lib/monitoring/gauge.h", "Gauge");
}

  GaugeCell<ValueType> default_gauge_cell_;

  TF_DISALLOW_COPY_AND_ASSIGN(Gauge);
};

}  // namespace monitoring
}  // namespace tensorflow

#else  // IS_MOBILE_PLATFORM

#include <array>
#include <atomic>
#include <functional>
#include <map>
#include <string>

#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/monitoring/collection_registry.h"
#include "tensorflow/core/lib/monitoring/metric_def.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/thread_annotations.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {
namespace monitoring {

// GaugeCell stores each value of a gauge.
//
// A cell can be passed off to a module which may repeatedly update it without
// needing further map-indexing computations. This improves both encapsulation
// (separate modules can own a cell each, without needing to know about the map
// to which both cells belong) and performance (since map indexing and
// associated locking are both avoided).
//
// This class is thread-safe.
template <typename T>
class GaugeCell {
 public:
  explicit GaugeCell(const T& value) : value_(value) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePSlibPSmonitoringPSgaugeDTh mht_7(mht_7_v, 311, "", "./tensorflow/core/lib/monitoring/gauge.h", "GaugeCell");
}
  ~GaugeCell() {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScorePSlibPSmonitoringPSgaugeDTh mht_8(mht_8_v, 315, "", "./tensorflow/core/lib/monitoring/gauge.h", "~GaugeCell");
}

  // Atomically sets the value.
  void Set(const T& value) TF_LOCKS_EXCLUDED(mu_);

  // Retrieves the current value.
  T value() const TF_LOCKS_EXCLUDED(mu_);

 private:
  T value_ TF_GUARDED_BY(mu_);
  mutable mutex mu_;

  TF_DISALLOW_COPY_AND_ASSIGN(GaugeCell);
};

// Explicit specialization of GaugeCell<int64_t>. Compared to the primary
// template, it uses atomic values as opposed to mutex. This class is
// thread-safe.
template <>
class GaugeCell<int64_t> {
 public:
  explicit GaugeCell(int64_t value) : value_(value) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScorePSlibPSmonitoringPSgaugeDTh mht_9(mht_9_v, 339, "", "./tensorflow/core/lib/monitoring/gauge.h", "GaugeCell");
}
  ~GaugeCell() {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScorePSlibPSmonitoringPSgaugeDTh mht_10(mht_10_v, 343, "", "./tensorflow/core/lib/monitoring/gauge.h", "~GaugeCell");
}

  // Atomically sets the value.
  void Set(int64_t value);

  // Retrieves the current value.
  int64_t value() const;

 private:
  std::atomic<int64_t> value_;

  TF_DISALLOW_COPY_AND_ASSIGN(GaugeCell);
};

// Explicit specialization of GaugeCell<bool>. Compared to the primary
// template, it uses atomic values as opposed to mutex. This class is
// thread-safe.
template <>
class GaugeCell<bool> {
 public:
  explicit GaugeCell(bool value) : value_(value) {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScorePSlibPSmonitoringPSgaugeDTh mht_11(mht_11_v, 366, "", "./tensorflow/core/lib/monitoring/gauge.h", "GaugeCell");
}
  ~GaugeCell() {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScorePSlibPSmonitoringPSgaugeDTh mht_12(mht_12_v, 370, "", "./tensorflow/core/lib/monitoring/gauge.h", "~GaugeCell");
}

  // Atomically sets the value.
  void Set(bool value);

  // Retrieves the current value.
  bool value() const;

 private:
  std::atomic<bool> value_;

  TF_DISALLOW_COPY_AND_ASSIGN(GaugeCell);
};

// A stateful class for updating a gauge-like metric. Allowed ValueType are
// int64, string and bool.
//
// This class encapsulates a set of values (or a single value for a label-less
// metric). Each value is identified by a tuple of labels. The class allows the
// user to set each value.
//
// Gauge allocates storage and maintains a cell for each value. You can
// retrieve an individual cell using a label-tuple and update it separately.
// This improves performance since operations related to retrieval, like
// map-indexing and locking, are avoided.
//
// This class is thread-safe.
template <typename ValueType, int NumLabels>
class Gauge {
 public:
  ~Gauge() {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPScorePSlibPSmonitoringPSgaugeDTh mht_13(mht_13_v, 403, "", "./tensorflow/core/lib/monitoring/gauge.h", "~Gauge");

    // Deleted here, before the metric_def is destroyed.
    registration_handle_.reset();
  }

  // Creates the metric based on the metric-definition arguments.
  //
  // Example:
  //
  // auto* string_gauge_with_label = Gauge<string,1>::New(
  //   "/tensorflow/string_gauge_with_label",
  //   "String gauge with one label.", "MyLabelName");
  //
  // auto* integer_gauge = Gauge<int64, 0>::New("/tensorflow/integer_gauge",
  //   "Integer gauge")
  //
  // auto* bool_gauge = Gauge<bool, 0>::New("/tensorflow/bool_gauge",
  //   "Bool gauge")
  template <typename... MetricDefArgs>
  static Gauge* New(MetricDefArgs&&... metric_def_args);

  // Retrieves the cell for the specified labels, creating it on demand if not
  // already present.
  template <typename... Labels>
  GaugeCell<ValueType>* GetCell(const Labels&... labels) TF_LOCKS_EXCLUDED(mu_);

  Status GetStatus() {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPScorePSlibPSmonitoringPSgaugeDTh mht_14(mht_14_v, 432, "", "./tensorflow/core/lib/monitoring/gauge.h", "GetStatus");
 return status_; }

 private:
  explicit Gauge(
      const MetricDef<MetricKind::kGauge, ValueType, NumLabels>& metric_def)
      : metric_def_(metric_def),
        registration_handle_(CollectionRegistry::Default()->Register(
            &metric_def_, [&](MetricCollectorGetter getter) {
              auto metric_collector = getter.Get(&metric_def_);

              mutex_lock l(mu_);
              for (const auto& cell : cells_) {
                metric_collector.CollectValue(cell.first, cell.second.value());
              }
            })) {
   std::vector<std::string> mht_15_v;
   MHTracer_DTPStensorflowPScorePSlibPSmonitoringPSgaugeDTh mht_15(mht_15_v, 449, "", "./tensorflow/core/lib/monitoring/gauge.h", "Gauge");

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
  const MetricDef<MetricKind::kGauge, ValueType, NumLabels> metric_def_;

  std::unique_ptr<CollectionRegistry::RegistrationHandle> registration_handle_;

  using LabelArray = std::array<string, NumLabels>;
  std::map<LabelArray, GaugeCell<ValueType> > cells_ TF_GUARDED_BY(mu_);

  TF_DISALLOW_COPY_AND_ASSIGN(Gauge);
};

////
//  Implementation details follow. API readers may skip.
////
template <typename T>
void GaugeCell<T>::Set(const T& value) {
   std::vector<std::string> mht_16_v;
   MHTracer_DTPStensorflowPScorePSlibPSmonitoringPSgaugeDTh mht_16(mht_16_v, 481, "", "./tensorflow/core/lib/monitoring/gauge.h", "GaugeCell<T>::Set");

  mutex_lock l(mu_);
  value_ = value;
}

template <typename T>
T GaugeCell<T>::value() const {
   std::vector<std::string> mht_17_v;
   MHTracer_DTPStensorflowPScorePSlibPSmonitoringPSgaugeDTh mht_17(mht_17_v, 490, "", "./tensorflow/core/lib/monitoring/gauge.h", "GaugeCell<T>::value");

  mutex_lock l(mu_);
  return value_;
}

inline void GaugeCell<int64_t>::Set(int64_t value) {
   std::vector<std::string> mht_18_v;
   MHTracer_DTPStensorflowPScorePSlibPSmonitoringPSgaugeDTh mht_18(mht_18_v, 498, "", "./tensorflow/core/lib/monitoring/gauge.h", "GaugeCell<int64_t>::Set");
 value_ = value; }

inline int64_t GaugeCell<int64_t>::value() const {
   std::vector<std::string> mht_19_v;
   MHTracer_DTPStensorflowPScorePSlibPSmonitoringPSgaugeDTh mht_19(mht_19_v, 503, "", "./tensorflow/core/lib/monitoring/gauge.h", "GaugeCell<int64_t>::value");
 return value_; }

inline void GaugeCell<bool>::Set(bool value) {
   std::vector<std::string> mht_20_v;
   MHTracer_DTPStensorflowPScorePSlibPSmonitoringPSgaugeDTh mht_20(mht_20_v, 508, "", "./tensorflow/core/lib/monitoring/gauge.h", "GaugeCell<bool>::Set");
 value_ = value; }

inline bool GaugeCell<bool>::value() const {
   std::vector<std::string> mht_21_v;
   MHTracer_DTPStensorflowPScorePSlibPSmonitoringPSgaugeDTh mht_21(mht_21_v, 513, "", "./tensorflow/core/lib/monitoring/gauge.h", "GaugeCell<bool>::value");
 return value_; }

template <typename ValueType, int NumLabels>
template <typename... MetricDefArgs>
Gauge<ValueType, NumLabels>* Gauge<ValueType, NumLabels>::New(
    MetricDefArgs&&... metric_def_args) {
  static_assert(
      std::is_same<ValueType, int64_t>::value ||
          std::is_same<ValueType, std::string>::value ||
          std::is_same<ValueType, bool>::value ||
          std::is_same<ValueType, std::function<int64_t()> >::value ||
          std::is_same<ValueType, std::function<std::string()> >::value ||
          std::is_same<ValueType, std::function<bool()> >::value,
      "Gauge only allows bool, int64, and string types.");
  return new Gauge<ValueType, NumLabels>(
      MetricDef<MetricKind::kGauge, ValueType, NumLabels>(
          std::forward<MetricDefArgs>(metric_def_args)...));
}

template <typename ValueType, int NumLabels>
template <typename... Labels>
GaugeCell<ValueType>* Gauge<ValueType, NumLabels>::GetCell(
    const Labels&... labels) TF_LOCKS_EXCLUDED(mu_) {
  // Provides a more informative error message than the one during array
  // construction below.
  static_assert(
      sizeof...(Labels) == NumLabels,
      "Mismatch between Gauge<ValueType, NumLabels> and number of labels "
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
                        std::forward_as_tuple(ValueType()))
               .first->second);
}

}  // namespace monitoring
}  // namespace tensorflow

#endif  // IS_MOBILE_PLATFORM
#endif  // TENSORFLOW_CORE_LIB_MONITORING_GAUGE_H_
