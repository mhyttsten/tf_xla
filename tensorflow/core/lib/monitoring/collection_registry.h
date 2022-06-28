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

#ifndef TENSORFLOW_CORE_LIB_MONITORING_COLLECTION_REGISTRY_H_
#define TENSORFLOW_CORE_LIB_MONITORING_COLLECTION_REGISTRY_H_
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
class MHTracer_DTPStensorflowPScorePSlibPSmonitoringPScollection_registryDTh {
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
   MHTracer_DTPStensorflowPScorePSlibPSmonitoringPScollection_registryDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSlibPSmonitoringPScollection_registryDTh() {
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

// We use a null implementation for mobile platforms.
#ifdef IS_MOBILE_PLATFORM

#include <functional>
#include <map>
#include <memory>

#include "tensorflow/core/lib/monitoring/metric_def.h"
#include "tensorflow/core/platform/macros.h"

namespace tensorflow {
namespace monitoring {

// MetricCollector which has a null implementation.
template <MetricKind metric_kind, typename Value, int NumLabels>
class MetricCollector {
 public:
  ~MetricCollector() = default;

  void CollectValue(const std::array<std::string, NumLabels>& labels,
                    Value value) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSlibPSmonitoringPScollection_registryDTh mht_0(mht_0_v, 213, "", "./tensorflow/core/lib/monitoring/collection_registry.h", "CollectValue");
}

 private:
  friend class MetricCollectorGetter;

  MetricCollector() {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSlibPSmonitoringPScollection_registryDTh mht_1(mht_1_v, 221, "", "./tensorflow/core/lib/monitoring/collection_registry.h", "MetricCollector");
}
};

// MetricCollectorGetter which has a null implementation.
class MetricCollectorGetter {
 public:
  template <MetricKind metric_kind, typename Value, int NumLabels>
  MetricCollector<metric_kind, Value, NumLabels> Get(
      const MetricDef<metric_kind, Value, NumLabels>* const metric_def) {
    return MetricCollector<metric_kind, Value, NumLabels>();
  }

 private:
  MetricCollectorGetter() {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSlibPSmonitoringPScollection_registryDTh mht_2(mht_2_v, 237, "", "./tensorflow/core/lib/monitoring/collection_registry.h", "MetricCollectorGetter");
}
};

// CollectionRegistry which has a null implementation.
class CollectionRegistry {
 public:
  ~CollectionRegistry() = default;

  static CollectionRegistry* Default() {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSlibPSmonitoringPScollection_registryDTh mht_3(mht_3_v, 248, "", "./tensorflow/core/lib/monitoring/collection_registry.h", "Default");
 return new CollectionRegistry(); }

  using CollectionFunction = std::function<void(MetricCollectorGetter getter)>;

  // RegistrationHandle which has a null implementation.
  class RegistrationHandle {
   public:
    RegistrationHandle() {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSlibPSmonitoringPScollection_registryDTh mht_4(mht_4_v, 258, "", "./tensorflow/core/lib/monitoring/collection_registry.h", "RegistrationHandle");
}

    ~RegistrationHandle() {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSlibPSmonitoringPScollection_registryDTh mht_5(mht_5_v, 263, "", "./tensorflow/core/lib/monitoring/collection_registry.h", "~RegistrationHandle");
}
  };

  std::unique_ptr<RegistrationHandle> Register(
      const AbstractMetricDef* metric_def,
      const CollectionFunction& collection_function) {
    return std::unique_ptr<RegistrationHandle>(new RegistrationHandle());
  }

 private:
  CollectionRegistry() {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePSlibPSmonitoringPScollection_registryDTh mht_6(mht_6_v, 276, "", "./tensorflow/core/lib/monitoring/collection_registry.h", "CollectionRegistry");
}

  TF_DISALLOW_COPY_AND_ASSIGN(CollectionRegistry);
};

}  // namespace monitoring
}  // namespace tensorflow
#else  // !defined(IS_MOBILE_PLATFORM)

#include <functional>
#include <map>
#include <memory>
#include <utility>

#include "tensorflow/core/framework/summary.pb.h"
#include "tensorflow/core/lib/monitoring/collected_metrics.h"
#include "tensorflow/core/lib/monitoring/metric_def.h"
#include "tensorflow/core/lib/monitoring/types.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/stringpiece.h"
#include "tensorflow/core/platform/thread_annotations.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {
namespace monitoring {

namespace test_util {
class CollectionRegistryTestAccess;
}  // namespace test_util

namespace internal {
class Collector;
}  // namespace internal

// Metric implementations would get an instance of this class using the
// MetricCollectorGetter in the collection-function lambda, so that their values
// can be collected.
//
// Read the documentation on CollectionRegistry::Register() for more details.
//
// For example:
//   auto metric_collector = metric_collector_getter->Get(&metric_def);
//   metric_collector.CollectValue(some_labels, some_value);
//   metric_collector.CollectValue(others_labels, other_value);
//
// This class is NOT thread-safe.
template <MetricKind metric_kind, typename Value, int NumLabels>
class MetricCollector {
 public:
  ~MetricCollector() = default;

  // Collects the value with these labels.
  void CollectValue(const std::array<std::string, NumLabels>& labels,
                    Value value);

 private:
  friend class internal::Collector;

  MetricCollector(
      const MetricDef<metric_kind, Value, NumLabels>* const metric_def,
      const uint64 registration_time_millis,
      internal::Collector* const collector, PointSet* const point_set)
      : metric_def_(metric_def),
        registration_time_millis_(registration_time_millis),
        collector_(collector),
        point_set_(point_set) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePSlibPSmonitoringPScollection_registryDTh mht_7(mht_7_v, 347, "", "./tensorflow/core/lib/monitoring/collection_registry.h", "MetricCollector");

    point_set_->metric_name = std::string(metric_def->name());
  }

  const MetricDef<metric_kind, Value, NumLabels>* const metric_def_;
  const uint64 registration_time_millis_;
  internal::Collector* const collector_;
  PointSet* const point_set_;

  // This is made copyable because we can't hand out references of this class
  // from MetricCollectorGetter because this class is templatized, and we need
  // MetricCollectorGetter not to be templatized and hence MetricCollectorGetter
  // can't own an instance of this class.
};

// Returns a MetricCollector with the same template parameters as the
// metric-definition, so that the values of a metric can be collected.
//
// The collection-function defined by a metric takes this as a parameter.
//
// Read the documentation on CollectionRegistry::Register() for more details.
class MetricCollectorGetter {
 public:
  // Returns the MetricCollector with the same template parameters as the
  // metric_def.
  template <MetricKind metric_kind, typename Value, int NumLabels>
  MetricCollector<metric_kind, Value, NumLabels> Get(
      const MetricDef<metric_kind, Value, NumLabels>* const metric_def);

 private:
  friend class internal::Collector;

  MetricCollectorGetter(internal::Collector* const collector,
                        const AbstractMetricDef* const allowed_metric_def,
                        const uint64 registration_time_millis)
      : collector_(collector),
        allowed_metric_def_(allowed_metric_def),
        registration_time_millis_(registration_time_millis) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScorePSlibPSmonitoringPScollection_registryDTh mht_8(mht_8_v, 387, "", "./tensorflow/core/lib/monitoring/collection_registry.h", "MetricCollectorGetter");
}

  internal::Collector* const collector_;
  const AbstractMetricDef* const allowed_metric_def_;
  const uint64 registration_time_millis_;
};

// A collection registry for metrics.
//
// Metrics are registered here so that their state can be collected later and
// exported.
//
// This class is thread-safe.
class CollectionRegistry {
 public:
  ~CollectionRegistry() = default;

  // Returns the default registry for the process.
  //
  // This registry belongs to this library and should never be deleted.
  static CollectionRegistry* Default();

  using CollectionFunction = std::function<void(MetricCollectorGetter getter)>;

  // Registers the metric and the collection-function which can be used to
  // collect its values. Returns a Registration object, which when upon
  // destruction would cause the metric to be unregistered from this registry.
  //
  // IMPORTANT: Delete the handle before the metric-def is deleted.
  //
  // Example usage;
  // CollectionRegistry::Default()->Register(
  //   &metric_def,
  //   [&](MetricCollectorGetter getter) {
  //     auto metric_collector = getter.Get(&metric_def);
  //     for (const auto& cell : cells) {
  //       metric_collector.CollectValue(cell.labels(), cell.value());
  //     }
  //   });
  class RegistrationHandle;
  std::unique_ptr<RegistrationHandle> Register(
      const AbstractMetricDef* metric_def,
      const CollectionFunction& collection_function)
      TF_LOCKS_EXCLUDED(mu_) TF_MUST_USE_RESULT;

  // Options for collecting metrics.
  struct CollectMetricsOptions {
    CollectMetricsOptions() {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScorePSlibPSmonitoringPScollection_registryDTh mht_9(mht_9_v, 437, "", "./tensorflow/core/lib/monitoring/collection_registry.h", "CollectMetricsOptions");
}
    bool collect_metric_descriptors = true;
  };
  // Goes through all the registered metrics, collects their definitions
  // (optionally) and current values and returns them in a standard format.
  std::unique_ptr<CollectedMetrics> CollectMetrics(
      const CollectMetricsOptions& options) const;

 private:
  friend class test_util::CollectionRegistryTestAccess;
  friend class internal::Collector;

  explicit CollectionRegistry(Env* env);

  // Unregisters the metric from this registry. This is private because the
  // public interface provides a Registration handle which automatically calls
  // this upon destruction.
  void Unregister(const AbstractMetricDef* metric_def) TF_LOCKS_EXCLUDED(mu_);

  // TF environment, mainly used for timestamping.
  Env* const env_;

  mutable mutex mu_;

  // Information required for collection.
  struct CollectionInfo {
    const AbstractMetricDef* const metric_def;
    CollectionFunction collection_function;
    uint64 registration_time_millis;
  };
  std::map<StringPiece, CollectionInfo> registry_ TF_GUARDED_BY(mu_);

  TF_DISALLOW_COPY_AND_ASSIGN(CollectionRegistry);
};

////
// Implementation details follow. API readers may skip.
////

class CollectionRegistry::RegistrationHandle {
 public:
  RegistrationHandle(CollectionRegistry* const export_registry,
                     const AbstractMetricDef* const metric_def)
      : export_registry_(export_registry), metric_def_(metric_def) {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScorePSlibPSmonitoringPScollection_registryDTh mht_10(mht_10_v, 483, "", "./tensorflow/core/lib/monitoring/collection_registry.h", "RegistrationHandle");
}

  ~RegistrationHandle() {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScorePSlibPSmonitoringPScollection_registryDTh mht_11(mht_11_v, 488, "", "./tensorflow/core/lib/monitoring/collection_registry.h", "~RegistrationHandle");
 export_registry_->Unregister(metric_def_); }

 private:
  CollectionRegistry* const export_registry_;
  const AbstractMetricDef* const metric_def_;
};

namespace internal {

template <typename Value>
void CollectValue(Value value, Point* point);

template <>
inline void CollectValue(int64_t value, Point* const point) {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScorePSlibPSmonitoringPScollection_registryDTh mht_12(mht_12_v, 504, "", "./tensorflow/core/lib/monitoring/collection_registry.h", "CollectValue");

  point->value_type = ValueType::kInt64;
  point->int64_value = value;
}

template <>
inline void CollectValue(std::function<int64_t()> value_fn,
                         Point* const point) {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPScorePSlibPSmonitoringPScollection_registryDTh mht_13(mht_13_v, 514, "", "./tensorflow/core/lib/monitoring/collection_registry.h", "CollectValue");

  point->value_type = ValueType::kInt64;
  point->int64_value = value_fn();
}

template <>
inline void CollectValue(std::string value, Point* const point) {
   std::vector<std::string> mht_14_v;
   mht_14_v.push_back("value: \"" + value + "\"");
   MHTracer_DTPStensorflowPScorePSlibPSmonitoringPScollection_registryDTh mht_14(mht_14_v, 524, "", "./tensorflow/core/lib/monitoring/collection_registry.h", "CollectValue");

  point->value_type = ValueType::kString;
  point->string_value = std::move(value);
}

template <>
inline void CollectValue(std::function<std::string()> value_fn,
                         Point* const point) {
   std::vector<std::string> mht_15_v;
   MHTracer_DTPStensorflowPScorePSlibPSmonitoringPScollection_registryDTh mht_15(mht_15_v, 534, "", "./tensorflow/core/lib/monitoring/collection_registry.h", "CollectValue");

  point->value_type = ValueType::kString;
  point->string_value = value_fn();
}

template <>
inline void CollectValue(bool value, Point* const point) {
   std::vector<std::string> mht_16_v;
   MHTracer_DTPStensorflowPScorePSlibPSmonitoringPScollection_registryDTh mht_16(mht_16_v, 543, "", "./tensorflow/core/lib/monitoring/collection_registry.h", "CollectValue");

  point->value_type = ValueType::kBool;
  point->bool_value = value;
}

template <>
inline void CollectValue(std::function<bool()> value_fn, Point* const point) {
   std::vector<std::string> mht_17_v;
   MHTracer_DTPStensorflowPScorePSlibPSmonitoringPScollection_registryDTh mht_17(mht_17_v, 552, "", "./tensorflow/core/lib/monitoring/collection_registry.h", "CollectValue");

  point->value_type = ValueType::kBool;
  point->bool_value = value_fn();
}

template <>
inline void CollectValue(HistogramProto value, Point* const point) {
   std::vector<std::string> mht_18_v;
   MHTracer_DTPStensorflowPScorePSlibPSmonitoringPScollection_registryDTh mht_18(mht_18_v, 561, "", "./tensorflow/core/lib/monitoring/collection_registry.h", "CollectValue");

  point->value_type = ValueType::kHistogram;
  // This is inefficient. If and when we hit snags, we can change the API to do
  // this more efficiently.
  point->histogram_value = std::move(value);
}

template <>
inline void CollectValue(Percentiles value, Point* const point) {
   std::vector<std::string> mht_19_v;
   MHTracer_DTPStensorflowPScorePSlibPSmonitoringPScollection_registryDTh mht_19(mht_19_v, 572, "", "./tensorflow/core/lib/monitoring/collection_registry.h", "CollectValue");

  point->value_type = ValueType::kPercentiles;
  point->percentiles_value = std::move(value);
}

// Used by the CollectionRegistry class to collect all the values of all the
// metrics in the registry. This is an implementation detail of the
// CollectionRegistry class, please do not depend on this.
//
// This cannot be a private nested class because we need to forward declare this
// so that the MetricCollector and MetricCollectorGetter classes can be friends
// with it.
//
// This class is thread-safe.
class Collector {
 public:
  explicit Collector(const uint64 collection_time_millis)
      : collected_metrics_(new CollectedMetrics()),
        collection_time_millis_(collection_time_millis) {
   std::vector<std::string> mht_20_v;
   MHTracer_DTPStensorflowPScorePSlibPSmonitoringPScollection_registryDTh mht_20(mht_20_v, 593, "", "./tensorflow/core/lib/monitoring/collection_registry.h", "Collector");
}

  template <MetricKind metric_kind, typename Value, int NumLabels>
  MetricCollector<metric_kind, Value, NumLabels> GetMetricCollector(
      const MetricDef<metric_kind, Value, NumLabels>* const metric_def,
      const uint64 registration_time_millis,
      internal::Collector* const collector) TF_LOCKS_EXCLUDED(mu_) {
    auto* const point_set = [&]() {
   std::vector<std::string> mht_21_v;
   MHTracer_DTPStensorflowPScorePSlibPSmonitoringPScollection_registryDTh mht_21(mht_21_v, 603, "", "./tensorflow/core/lib/monitoring/collection_registry.h", "lambda");

      mutex_lock l(mu_);
      return collected_metrics_->point_set_map
          .insert(std::make_pair(std::string(metric_def->name()),
                                 std::unique_ptr<PointSet>(new PointSet())))
          .first->second.get();
    }();
    return MetricCollector<metric_kind, Value, NumLabels>(
        metric_def, registration_time_millis, collector, point_set);
  }

  uint64 collection_time_millis() const {
   std::vector<std::string> mht_22_v;
   MHTracer_DTPStensorflowPScorePSlibPSmonitoringPScollection_registryDTh mht_22(mht_22_v, 617, "", "./tensorflow/core/lib/monitoring/collection_registry.h", "collection_time_millis");
 return collection_time_millis_; }

  void CollectMetricDescriptor(const AbstractMetricDef* const metric_def)
      TF_LOCKS_EXCLUDED(mu_);

  void CollectMetricValues(
      const CollectionRegistry::CollectionInfo& collection_info);

  std::unique_ptr<CollectedMetrics> ConsumeCollectedMetrics()
      TF_LOCKS_EXCLUDED(mu_);

 private:
  mutable mutex mu_;
  std::unique_ptr<CollectedMetrics> collected_metrics_ TF_GUARDED_BY(mu_);
  const uint64 collection_time_millis_;

  TF_DISALLOW_COPY_AND_ASSIGN(Collector);
};

// Write the timestamps for the point based on the MetricKind.
//
// Gauge metrics will have start and end timestamps set to the collection time.
//
// Cumulative metrics will have the start timestamp set to the time when the
// collection function was registered, while the end timestamp will be set to
// the collection time.
template <MetricKind kind>
void WriteTimestamps(const uint64 registration_time_millis,
                     const uint64 collection_time_millis, Point* const point);

template <>
inline void WriteTimestamps<MetricKind::kGauge>(
    const uint64 registration_time_millis, const uint64 collection_time_millis,
    Point* const point) {
   std::vector<std::string> mht_23_v;
   MHTracer_DTPStensorflowPScorePSlibPSmonitoringPScollection_registryDTh mht_23(mht_23_v, 653, "", "./tensorflow/core/lib/monitoring/collection_registry.h", "WriteTimestamps<MetricKind::kGauge>");

  point->start_timestamp_millis = collection_time_millis;
  point->end_timestamp_millis = collection_time_millis;
}

template <>
inline void WriteTimestamps<MetricKind::kCumulative>(
    const uint64 registration_time_millis, const uint64 collection_time_millis,
    Point* const point) {
   std::vector<std::string> mht_24_v;
   MHTracer_DTPStensorflowPScorePSlibPSmonitoringPScollection_registryDTh mht_24(mht_24_v, 664, "", "./tensorflow/core/lib/monitoring/collection_registry.h", "WriteTimestamps<MetricKind::kCumulative>");

  point->start_timestamp_millis = registration_time_millis;
  // There's a chance that the clock goes backwards on the same machine, so we
  // protect ourselves against that.
  point->end_timestamp_millis =
      registration_time_millis < collection_time_millis
          ? collection_time_millis
          : registration_time_millis;
}

}  // namespace internal

template <MetricKind metric_kind, typename Value, int NumLabels>
void MetricCollector<metric_kind, Value, NumLabels>::CollectValue(
    const std::array<std::string, NumLabels>& labels, Value value) {
  point_set_->points.emplace_back(new Point());
  auto* const point = point_set_->points.back().get();
  const std::vector<std::string> label_descriptions =
      metric_def_->label_descriptions();
  point->labels.reserve(NumLabels);
  for (int i = 0; i < NumLabels; ++i) {
    point->labels.push_back({});
    auto* const label = &point->labels.back();
    label->name = label_descriptions[i];
    label->value = labels[i];
  }
  internal::CollectValue(std::move(value), point);
  internal::WriteTimestamps<metric_kind>(
      registration_time_millis_, collector_->collection_time_millis(), point);
}

template <MetricKind metric_kind, typename Value, int NumLabels>
MetricCollector<metric_kind, Value, NumLabels> MetricCollectorGetter::Get(
    const MetricDef<metric_kind, Value, NumLabels>* const metric_def) {
  if (allowed_metric_def_ != metric_def) {
    LOG(FATAL) << "Expected collection for: " << allowed_metric_def_->name()
               << " but instead got: " << metric_def->name();
  }

  return collector_->GetMetricCollector(metric_def, registration_time_millis_,
                                        collector_);
}

class Exporter {
 public:
  virtual ~Exporter() {
   std::vector<std::string> mht_25_v;
   MHTracer_DTPStensorflowPScorePSlibPSmonitoringPScollection_registryDTh mht_25(mht_25_v, 712, "", "./tensorflow/core/lib/monitoring/collection_registry.h", "~Exporter");
}
  virtual void PeriodicallyExportMetrics() = 0;
  virtual void ExportMetrics() = 0;
};

namespace exporter_registration {

class ExporterRegistration {
 public:
  explicit ExporterRegistration(Exporter* exporter) : exporter_(exporter) {
   std::vector<std::string> mht_26_v;
   MHTracer_DTPStensorflowPScorePSlibPSmonitoringPScollection_registryDTh mht_26(mht_26_v, 724, "", "./tensorflow/core/lib/monitoring/collection_registry.h", "ExporterRegistration");

    exporter_->PeriodicallyExportMetrics();
  }

 private:
  Exporter* exporter_;
};

}  // namespace exporter_registration

#define REGISTER_TF_METRICS_EXPORTER(exporter) \
  REGISTER_TF_METRICS_EXPORTER_UNIQ_HELPER(__COUNTER__, exporter)

#define REGISTER_TF_METRICS_EXPORTER_UNIQ_HELPER(ctr, exporter) \
  REGISTER_TF_METRICS_EXPORTER_UNIQ(ctr, exporter)

#define REGISTER_TF_METRICS_EXPORTER_UNIQ(ctr, exporter)                       \
  static ::tensorflow::monitoring::exporter_registration::ExporterRegistration \
      exporter_registration_##ctr(new exporter())

}  // namespace monitoring
}  // namespace tensorflow

#endif  // IS_MOBILE_PLATFORM
#endif  // TENSORFLOW_CORE_LIB_MONITORING_COLLECTION_REGISTRY_H_
