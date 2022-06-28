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

#ifndef TENSORFLOW_CORE_LIB_MONITORING_SAMPLER_H_
#define TENSORFLOW_CORE_LIB_MONITORING_SAMPLER_H_
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
class MHTracer_DTPStensorflowPScorePSlibPSmonitoringPSsamplerDTh {
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
   MHTracer_DTPStensorflowPScorePSlibPSmonitoringPSsamplerDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSlibPSmonitoringPSsamplerDTh() {
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

#include <memory>

#include "tensorflow/core/framework/summary.pb.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/monitoring/metric_def.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {
namespace monitoring {

// SamplerCell which has a null implementation.
class SamplerCell {
 public:
  SamplerCell() {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSlibPSmonitoringPSsamplerDTh mht_0(mht_0_v, 211, "", "./tensorflow/core/lib/monitoring/sampler.h", "SamplerCell");
}
  ~SamplerCell() {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSlibPSmonitoringPSsamplerDTh mht_1(mht_1_v, 215, "", "./tensorflow/core/lib/monitoring/sampler.h", "~SamplerCell");
}

  void Add(double value) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSlibPSmonitoringPSsamplerDTh mht_2(mht_2_v, 220, "", "./tensorflow/core/lib/monitoring/sampler.h", "Add");
}
  HistogramProto value() const {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSlibPSmonitoringPSsamplerDTh mht_3(mht_3_v, 224, "", "./tensorflow/core/lib/monitoring/sampler.h", "value");
 return HistogramProto(); }

 private:
  TF_DISALLOW_COPY_AND_ASSIGN(SamplerCell);
};

// Buckets which has a null implementation.
class Buckets {
 public:
  Buckets() = default;
  ~Buckets() = default;

  static std::unique_ptr<Buckets> Explicit(
      std::initializer_list<double> bucket_limits) {
    return std::unique_ptr<Buckets>(new Buckets());
  }

  static std::unique_ptr<Buckets> Exponential(double scale,
                                              double growth_factor,
                                              int bucket_count) {
    return std::unique_ptr<Buckets>(new Buckets());
  }

  const std::vector<double>& explicit_bounds() const {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSlibPSmonitoringPSsamplerDTh mht_4(mht_4_v, 250, "", "./tensorflow/core/lib/monitoring/sampler.h", "explicit_bounds");

    return explicit_bounds_;
  }

 private:
  std::vector<double> explicit_bounds_;

  TF_DISALLOW_COPY_AND_ASSIGN(Buckets);
};

// Sampler which has a null implementation.
template <int NumLabels>
class Sampler {
 public:
  ~Sampler() {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSlibPSmonitoringPSsamplerDTh mht_5(mht_5_v, 267, "", "./tensorflow/core/lib/monitoring/sampler.h", "~Sampler");
}

  template <typename... MetricDefArgs>
  static Sampler* New(const MetricDef<MetricKind::kCumulative, HistogramProto,
                                      NumLabels>& metric_def,
                      std::unique_ptr<Buckets> buckets) {
    return new Sampler<NumLabels>(std::move(buckets));
  }

  template <typename... Labels>
  SamplerCell* GetCell(const Labels&... labels) {
    return &default_sampler_cell_;
  }

  Status GetStatus() {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePSlibPSmonitoringPSsamplerDTh mht_6(mht_6_v, 284, "", "./tensorflow/core/lib/monitoring/sampler.h", "GetStatus");
 return Status::OK(); }

 private:
  Sampler(std::unique_ptr<Buckets> buckets) : buckets_(std::move(buckets)) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePSlibPSmonitoringPSsamplerDTh mht_7(mht_7_v, 290, "", "./tensorflow/core/lib/monitoring/sampler.h", "Sampler");
}

  SamplerCell default_sampler_cell_;
  std::unique_ptr<Buckets> buckets_;

  TF_DISALLOW_COPY_AND_ASSIGN(Sampler);
};

}  // namespace monitoring
}  // namespace tensorflow

#else  // IS_MOBILE_PLATFORM

#include <float.h>

#include <map>

#include "tensorflow/core/framework/summary.pb.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/histogram/histogram.h"
#include "tensorflow/core/lib/monitoring/collection_registry.h"
#include "tensorflow/core/lib/monitoring/metric_def.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/thread_annotations.h"

namespace tensorflow {
namespace monitoring {

// SamplerCell stores each value of an Sampler.
//
// A cell can be passed off to a module which may repeatedly update it without
// needing further map-indexing computations. This improves both encapsulation
// (separate modules can own a cell each, without needing to know about the map
// to which both cells belong) and performance (since map indexing and
// associated locking are both avoided).
//
// This class is thread-safe.
class SamplerCell {
 public:
  SamplerCell(const std::vector<double>& bucket_limits)
      : histogram_(bucket_limits) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScorePSlibPSmonitoringPSsamplerDTh mht_8(mht_8_v, 334, "", "./tensorflow/core/lib/monitoring/sampler.h", "SamplerCell");
}

  ~SamplerCell() {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScorePSlibPSmonitoringPSsamplerDTh mht_9(mht_9_v, 339, "", "./tensorflow/core/lib/monitoring/sampler.h", "~SamplerCell");
}

  // Atomically adds a sample.
  void Add(double sample);

  // Returns the current histogram value as a proto.
  HistogramProto value() const;

 private:
  histogram::ThreadSafeHistogram histogram_;

  TF_DISALLOW_COPY_AND_ASSIGN(SamplerCell);
};

// Bucketing strategies for the samplers.
//
// We automatically add -DBL_MAX and DBL_MAX to the ranges, so that no sample
// goes out of bounds.
//
// WARNING: If you are changing the interface here, please do change the same in
// mobile_sampler.h.
class Buckets {
 public:
  virtual ~Buckets() = default;

  // Sets up buckets of the form:
  // [-DBL_MAX, ..., scale * growth^i,
  //   scale * growth_factor^(i + 1), ..., DBL_MAX].
  //
  // So for powers of 2 with a bucket count of 10, you would say (1, 2, 10)
  static std::unique_ptr<Buckets> Exponential(double scale,
                                              double growth_factor,
                                              int bucket_count);

  // Sets up buckets of the form:
  // [-DBL_MAX, ..., bucket_limits[i], bucket_limits[i + 1], ..., DBL_MAX].
  static std::unique_ptr<Buckets> Explicit(
      std::initializer_list<double> bucket_limits);

  // This alternative Explicit Buckets factory method is primarily meant to be
  // used by the CLIF layer code paths that are incompatible with
  // initialize_lists.
  static std::unique_ptr<Buckets> Explicit(std::vector<double> bucket_limits);

  virtual const std::vector<double>& explicit_bounds() const = 0;
};

// A stateful class for updating a cumulative histogram metric.
//
// This class encapsulates a set of histograms (or a single histogram for a
// label-less metric) configured with a list of increasing bucket boundaries.
// Each histogram is identified by a tuple of labels. The class allows the
// user to add a sample to each histogram value.
//
// Sampler allocates storage and maintains a cell for each value. You can
// retrieve an individual cell using a label-tuple and update it separately.
// This improves performance since operations related to retrieval, like
// map-indexing and locking, are avoided.
//
// This class is thread-safe.
template <int NumLabels>
class Sampler {
 public:
  ~Sampler() {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScorePSlibPSmonitoringPSsamplerDTh mht_10(mht_10_v, 405, "", "./tensorflow/core/lib/monitoring/sampler.h", "~Sampler");

    // Deleted here, before the metric_def is destroyed.
    registration_handle_.reset();
  }

  // Creates the metric based on the metric-definition arguments and buckets.
  //
  // Example;
  // auto* sampler_with_label = Sampler<1>::New({"/tensorflow/sampler",
  //   "Tensorflow sampler", "MyLabelName"}, {10.0, 20.0, 30.0});
  static Sampler* New(const MetricDef<MetricKind::kCumulative, HistogramProto,
                                      NumLabels>& metric_def,
                      std::unique_ptr<Buckets> buckets);

  // Retrieves the cell for the specified labels, creating it on demand if
  // not already present.
  template <typename... Labels>
  SamplerCell* GetCell(const Labels&... labels) TF_LOCKS_EXCLUDED(mu_);

  Status GetStatus() {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScorePSlibPSmonitoringPSsamplerDTh mht_11(mht_11_v, 427, "", "./tensorflow/core/lib/monitoring/sampler.h", "GetStatus");
 return status_; }

 private:
  friend class SamplerCell;

  Sampler(const MetricDef<MetricKind::kCumulative, HistogramProto, NumLabels>&
              metric_def,
          std::unique_ptr<Buckets> buckets)
      : metric_def_(metric_def),
        buckets_(std::move(buckets)),
        registration_handle_(CollectionRegistry::Default()->Register(
            &metric_def_, [&](MetricCollectorGetter getter) {
              auto metric_collector = getter.Get(&metric_def_);

              mutex_lock l(mu_);
              for (const auto& cell : cells_) {
                metric_collector.CollectValue(cell.first, cell.second.value());
              }
            })) {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScorePSlibPSmonitoringPSsamplerDTh mht_12(mht_12_v, 448, "", "./tensorflow/core/lib/monitoring/sampler.h", "Sampler");

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
  const MetricDef<MetricKind::kCumulative, HistogramProto, NumLabels>
      metric_def_;

  // Bucket limits for the histograms in the cells.
  std::unique_ptr<Buckets> buckets_;

  // Registration handle with the CollectionRegistry.
  std::unique_ptr<CollectionRegistry::RegistrationHandle> registration_handle_;

  using LabelArray = std::array<string, NumLabels>;
  // we need a container here that guarantees pointer stability of the value,
  // namely, the pointer of the value should remain valid even after more cells
  // are inserted.
  std::map<LabelArray, SamplerCell> cells_ TF_GUARDED_BY(mu_);

  TF_DISALLOW_COPY_AND_ASSIGN(Sampler);
};

////
//  Implementation details follow. API readers may skip.
////

inline void SamplerCell::Add(const double sample) {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPScorePSlibPSmonitoringPSsamplerDTh mht_13(mht_13_v, 488, "", "./tensorflow/core/lib/monitoring/sampler.h", "SamplerCell::Add");
 histogram_.Add(sample); }

inline HistogramProto SamplerCell::value() const {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPScorePSlibPSmonitoringPSsamplerDTh mht_14(mht_14_v, 493, "", "./tensorflow/core/lib/monitoring/sampler.h", "SamplerCell::value");

  HistogramProto pb;
  histogram_.EncodeToProto(&pb, true /* preserve_zero_buckets */);
  return pb;
}

template <int NumLabels>
Sampler<NumLabels>* Sampler<NumLabels>::New(
    const MetricDef<MetricKind::kCumulative, HistogramProto, NumLabels>&
        metric_def,
    std::unique_ptr<Buckets> buckets) {
   std::vector<std::string> mht_15_v;
   MHTracer_DTPStensorflowPScorePSlibPSmonitoringPSsamplerDTh mht_15(mht_15_v, 506, "", "./tensorflow/core/lib/monitoring/sampler.h", "Sampler<NumLabels>::New");

  return new Sampler<NumLabels>(metric_def, std::move(buckets));
}

template <int NumLabels>
template <typename... Labels>
SamplerCell* Sampler<NumLabels>::GetCell(const Labels&... labels)
    TF_LOCKS_EXCLUDED(mu_) {
  // Provides a more informative error message than the one during array
  // construction below.
  static_assert(sizeof...(Labels) == NumLabels,
                "Mismatch between Sampler<NumLabels> and number of labels "
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
                        std::forward_as_tuple(buckets_->explicit_bounds()))
               .first->second);
}

}  // namespace monitoring
}  // namespace tensorflow

#endif  // IS_MOBILE_PLATFORM
#endif  // TENSORFLOW_CORE_LIB_MONITORING_SAMPLER_H_
