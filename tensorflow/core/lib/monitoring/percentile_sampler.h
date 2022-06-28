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

#ifndef TENSORFLOW_CORE_LIB_MONITORING_PERCENTILE_SAMPLER_H_
#define TENSORFLOW_CORE_LIB_MONITORING_PERCENTILE_SAMPLER_H_
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
class MHTracer_DTPStensorflowPScorePSlibPSmonitoringPSpercentile_samplerDTh {
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
   MHTracer_DTPStensorflowPScorePSlibPSmonitoringPSpercentile_samplerDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSlibPSmonitoringPSpercentile_samplerDTh() {
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
#include "tensorflow/core/lib/monitoring/collection_registry.h"
#include "tensorflow/core/lib/monitoring/metric_def.h"
#include "tensorflow/core/lib/monitoring/types.h"
#include "tensorflow/core/platform/macros.h"

namespace tensorflow {
namespace monitoring {

class PercentileSamplerCell {
 public:
  void Add(double sample) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSlibPSmonitoringPSpercentile_samplerDTh mht_0(mht_0_v, 208, "", "./tensorflow/core/lib/monitoring/percentile_sampler.h", "Add");
}

  Percentiles value() const {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSlibPSmonitoringPSpercentile_samplerDTh mht_1(mht_1_v, 213, "", "./tensorflow/core/lib/monitoring/percentile_sampler.h", "value");
 return Percentiles(); }
};

template <int NumLabels>
class PercentileSampler {
 public:
  static PercentileSampler* New(
      const MetricDef<MetricKind::kCumulative, Percentiles, NumLabels>&
          metric_def,
      std::vector<double> percentiles, size_t max_samples,
      UnitOfMeasure unit_of_measure);

  template <typename... Labels>
  PercentileSamplerCell* GetCell(const Labels&... labels) {
    return &default_cell_;
  }

  Status GetStatus() {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSlibPSmonitoringPSpercentile_samplerDTh mht_2(mht_2_v, 233, "", "./tensorflow/core/lib/monitoring/percentile_sampler.h", "GetStatus");
 return Status::OK(); }

 private:
  PercentileSamplerCell default_cell_;

  PercentileSampler() = default;

  TF_DISALLOW_COPY_AND_ASSIGN(PercentileSampler);
};

template <int NumLabels>
PercentileSampler<NumLabels>* PercentileSampler<NumLabels>::New(
    const MetricDef<MetricKind::kCumulative, Percentiles, NumLabels>&
    /* metric_def */,
    std::vector<double> /* percentiles */, size_t /* max_samples */,
    UnitOfMeasure /* unit_of_measure */) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSlibPSmonitoringPSpercentile_samplerDTh mht_3(mht_3_v, 251, "", "./tensorflow/core/lib/monitoring/percentile_sampler.h", "PercentileSampler<NumLabels>::New");

  return new PercentileSampler<NumLabels>();
}

}  // namespace monitoring
}  // namespace tensorflow

#else  // IS_MOBILE_PLATFORM

#include <cmath>
#include <map>

#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/monitoring/collection_registry.h"
#include "tensorflow/core/lib/monitoring/metric_def.h"
#include "tensorflow/core/lib/monitoring/types.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/thread_annotations.h"

namespace tensorflow {
namespace monitoring {

// PercentileSamplerCell stores each value of an PercentileSampler.
// The class uses a circular buffer to maintain a window of samples.
//
// This class is thread-safe.
class PercentileSamplerCell {
 public:
  PercentileSamplerCell(UnitOfMeasure unit_of_measure,
                        std::vector<double> percentiles, size_t max_samples)
      : unit_of_measure_(unit_of_measure),
        percentiles_(std::move(percentiles)),
        samples_(max_samples),
        num_samples_(0),
        next_position_(0),
        total_samples_(0),
        accumulator_(0.0) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSlibPSmonitoringPSpercentile_samplerDTh mht_4(mht_4_v, 291, "", "./tensorflow/core/lib/monitoring/percentile_sampler.h", "PercentileSamplerCell");
}

  // Atomically adds a sample.
  void Add(double sample);

  Percentiles value() const;

 private:
  struct Sample {
    bool operator<(const Sample& rhs) const {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSlibPSmonitoringPSpercentile_samplerDTh mht_5(mht_5_v, 303, "", "./tensorflow/core/lib/monitoring/percentile_sampler.h", "operator<");
 return value < rhs.value; }

    uint64 nstime = 0;
    double value = NAN;
  };

  std::vector<Sample> GetSamples(size_t* total_samples,
                                 long double* accumulator) const;

  mutable mutex mu_;
  UnitOfMeasure unit_of_measure_;
  const std::vector<double> percentiles_;
  std::vector<Sample> samples_ TF_GUARDED_BY(mu_);
  size_t num_samples_ TF_GUARDED_BY(mu_);
  size_t next_position_ TF_GUARDED_BY(mu_);
  size_t total_samples_ TF_GUARDED_BY(mu_);
  long double accumulator_ TF_GUARDED_BY(mu_);

  TF_DISALLOW_COPY_AND_ASSIGN(PercentileSamplerCell);
};

// A stateful class for updating a cumulative percentile sampled metric.
//
// This class stores, in each cell, up to max_samples values in a circular
// buffer, and returns the percentiles information as cell value.
//
// PercentileSampler allocates storage and maintains a cell for each value. You
// can retrieve an individual cell using a label-tuple and update it separately.
// This improves performance since operations related to retrieval, like
// map-indexing and locking, are avoided.
//
// This class is thread-safe.
template <int NumLabels>
class PercentileSampler {
 public:
  ~PercentileSampler() {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePSlibPSmonitoringPSpercentile_samplerDTh mht_6(mht_6_v, 341, "", "./tensorflow/core/lib/monitoring/percentile_sampler.h", "~PercentileSampler");

    // Deleted here, before the metric_def is destroyed.
    registration_handle_.reset();
  }

  // Creates the metric based on the metric-definition arguments and buckets.
  //
  // Example;
  // auto* sampler_with_label =
  // PercentileSampler<1>::New({"/tensorflow/sampler",
  //   "Tensorflow sampler", "MyLabelName"}, {10.0, 20.0, 30.0}, 1024,
  //   UnitOfMeasure::kTime);
  static PercentileSampler* New(
      const MetricDef<MetricKind::kCumulative, Percentiles, NumLabels>&
          metric_def,
      std::vector<double> percentiles, size_t max_samples,
      UnitOfMeasure unit_of_measure);

  // Retrieves the cell for the specified labels, creating it on demand if
  // not already present.
  template <typename... Labels>
  PercentileSamplerCell* GetCell(const Labels&... labels)
      TF_LOCKS_EXCLUDED(mu_);

  Status GetStatus() {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePSlibPSmonitoringPSpercentile_samplerDTh mht_7(mht_7_v, 368, "", "./tensorflow/core/lib/monitoring/percentile_sampler.h", "GetStatus");
 return status_; }

 private:
  friend class PercentileSamplerCell;

  PercentileSampler(const MetricDef<MetricKind::kCumulative, Percentiles,
                                    NumLabels>& metric_def,
                    std::vector<double> percentiles, size_t max_samples,
                    UnitOfMeasure unit_of_measure)
      : metric_def_(metric_def),
        unit_of_measure_(unit_of_measure),
        percentiles_(std::move(percentiles)),
        max_samples_(max_samples),
        registration_handle_(CollectionRegistry::Default()->Register(
            &metric_def_, [&](MetricCollectorGetter getter) {
              auto metric_collector = getter.Get(&metric_def_);
              mutex_lock l(mu_);
              for (const auto& cell : cells_) {
                metric_collector.CollectValue(cell.first, cell.second.value());
              }
            })) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScorePSlibPSmonitoringPSpercentile_samplerDTh mht_8(mht_8_v, 391, "", "./tensorflow/core/lib/monitoring/percentile_sampler.h", "PercentileSampler");

    if (registration_handle_) {
      for (size_t i = 0; i < percentiles_.size(); ++i) {
        if (percentiles_[i] < 0.0 || percentiles_[i] > 100.0) {
          status_ = Status(tensorflow::error::Code::INVALID_ARGUMENT,
                           "Percentile values must be in [0, 100] range.");
          break;
        }
        if (i + 1 < percentiles_.size() &&
            percentiles_[i] >= percentiles_[i + 1]) {
          status_ =
              Status(tensorflow::error::Code::INVALID_ARGUMENT,
                     "Percentile values must be in strictly ascending order.");
          break;
        }
      }
    } else {
      status_ = Status(tensorflow::error::Code::ALREADY_EXISTS,
                       "Another metric with the same name already exists.");
    }
  }

  mutable mutex mu_;

  Status status_;

  // The metric definition. This will be used to identify the metric when we
  // register it for collection.
  const MetricDef<MetricKind::kCumulative, Percentiles, NumLabels> metric_def_;

  UnitOfMeasure unit_of_measure_ = UnitOfMeasure::kNumber;

  // The percentiles samples required for this metric.
  const std::vector<double> percentiles_;

  // The maximum size of the samples colected by the PercentileSamplerCell cell.
  const size_t max_samples_ = 0;

  // Registration handle with the CollectionRegistry.
  std::unique_ptr<CollectionRegistry::RegistrationHandle> registration_handle_;

  using LabelArray = std::array<string, NumLabels>;
  // we need a container here that guarantees pointer stability of the value,
  // namely, the pointer of the value should remain valid even after more cells
  // are inserted.
  std::map<LabelArray, PercentileSamplerCell> cells_ TF_GUARDED_BY(mu_);

  TF_DISALLOW_COPY_AND_ASSIGN(PercentileSampler);
};

template <int NumLabels>
PercentileSampler<NumLabels>* PercentileSampler<NumLabels>::New(
    const MetricDef<MetricKind::kCumulative, Percentiles, NumLabels>&
        metric_def,
    std::vector<double> percentiles, size_t max_samples,
    UnitOfMeasure unit_of_measure) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScorePSlibPSmonitoringPSpercentile_samplerDTh mht_9(mht_9_v, 449, "", "./tensorflow/core/lib/monitoring/percentile_sampler.h", "PercentileSampler<NumLabels>::New");

  return new PercentileSampler<NumLabels>(metric_def, std::move(percentiles),
                                          max_samples, unit_of_measure);
}

template <int NumLabels>
template <typename... Labels>
PercentileSamplerCell* PercentileSampler<NumLabels>::GetCell(
    const Labels&... labels) TF_LOCKS_EXCLUDED(mu_) {
  // Provides a more informative error message than the one during array
  // construction below.
  static_assert(
      sizeof...(Labels) == NumLabels,
      "Mismatch between PercentileSampler<NumLabels> and number of labels "
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
                        std::forward_as_tuple(unit_of_measure_, percentiles_,
                                              max_samples_))
               .first->second);
}

}  // namespace monitoring
}  // namespace tensorflow

#endif  // IS_MOBILE_PLATFORM
#endif  // TENSORFLOW_CORE_LIB_MONITORING_PERCENTILE_SAMPLER_H_
