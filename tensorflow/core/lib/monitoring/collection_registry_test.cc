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
class MHTracer_DTPStensorflowPScorePSlibPSmonitoringPScollection_registry_testDTcc {
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
   MHTracer_DTPStensorflowPScorePSlibPSmonitoringPScollection_registry_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSlibPSmonitoringPScollection_registry_testDTcc() {
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

#include "tensorflow/core/lib/monitoring/collection_registry.h"

#include "tensorflow/core/lib/monitoring/counter.h"
#include "tensorflow/core/lib/monitoring/gauge.h"
#include "tensorflow/core/lib/monitoring/percentile_sampler.h"
#include "tensorflow/core/lib/monitoring/sampler.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/protobuf.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {
namespace monitoring {

using histogram::Histogram;

namespace test_util {

class CollectionRegistryTestAccess {
 public:
  static std::unique_ptr<CollectionRegistry> CreateRegistry(Env* const env) {
    return std::unique_ptr<CollectionRegistry>(new CollectionRegistry(env));
  }
};

}  // namespace test_util

namespace {

void EmptyCollectionFunction(MetricCollectorGetter getter) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSlibPSmonitoringPScollection_registry_testDTcc mht_0(mht_0_v, 213, "", "./tensorflow/core/lib/monitoring/collection_registry_test.cc", "EmptyCollectionFunction");
}

TEST(CollectionRegistryTest, RegistrationUnregistration) {
  auto* collection_registry = CollectionRegistry::Default();
  const MetricDef<MetricKind::kCumulative, int64_t, 0> metric_def0(
      "/tensorflow/metric0", "An example metric with no labels.");
  const MetricDef<MetricKind::kGauge, HistogramProto, 1> metric_def1(
      "/tensorflow/metric1", "An example metric with one label.", "LabelName");

  {
    // Enclosed in a scope so that we unregister before the stack variables
    // above are destroyed.

    std::unique_ptr<CollectionRegistry::RegistrationHandle> handle0 =
        collection_registry->Register(&metric_def0, EmptyCollectionFunction);
    std::unique_ptr<CollectionRegistry::RegistrationHandle> handle1 =
        collection_registry->Register(&metric_def1, EmptyCollectionFunction);

    handle0.reset();

    // Able to register again because it was unregistered earlier.
    handle0 =
        collection_registry->Register(&metric_def0, EmptyCollectionFunction);
  }
}

TEST(CollectionRegistryDeathTest, DuplicateRegistration) {
  auto* collection_registry = CollectionRegistry::Default();
  const MetricDef<MetricKind::kCumulative, int64_t, 0> metric_def(
      "/tensorflow/metric", "An example metric with no labels.");

  auto handle =
      collection_registry->Register(&metric_def, EmptyCollectionFunction);
  auto duplicate_handle =
      collection_registry->Register(&metric_def, EmptyCollectionFunction);
  EXPECT_EQ(duplicate_handle, nullptr);
}

TEST(CollectMetricsTest, Counter) {
  auto counter_with_labels = std::unique_ptr<Counter<2>>(
      Counter<2>::New("/tensorflow/test/counter_with_labels",
                      "Counter with labels.", "MyLabel0", "MyLabel1"));
  auto counter_without_labels = std::unique_ptr<Counter<0>>(Counter<0>::New(
      "/tensorflow/test/counter_without_labels", "Counter without labels."));

  counter_with_labels->GetCell("Label00", "Label10")->IncrementBy(42);
  counter_with_labels->GetCell("Label01", "Label11")->IncrementBy(58);
  counter_without_labels->GetCell()->IncrementBy(7);

  for (const bool collect_metric_descriptors : {true, false}) {
    SCOPED_TRACE(strings::StrCat("collect_metric_descriptors: ",
                                 collect_metric_descriptors));

    auto* collection_registry = CollectionRegistry::Default();
    CollectionRegistry::CollectMetricsOptions options;
    options.collect_metric_descriptors = collect_metric_descriptors;
    const std::unique_ptr<CollectedMetrics> collected_metrics =
        collection_registry->CollectMetrics(options);

    if (collect_metric_descriptors) {
      ASSERT_GE(collected_metrics->metric_descriptor_map.size(), 2);

      const MetricDescriptor& ld = *collected_metrics->metric_descriptor_map.at(
          "/tensorflow/test/counter_with_labels");
      EXPECT_EQ("/tensorflow/test/counter_with_labels", ld.name);
      EXPECT_EQ("Counter with labels.", ld.description);
      ASSERT_EQ(2, ld.label_names.size());
      EXPECT_EQ("MyLabel0", ld.label_names[0]);
      EXPECT_EQ("MyLabel1", ld.label_names[1]);
      EXPECT_EQ(MetricKind::kCumulative, ld.metric_kind);
      EXPECT_EQ(ValueType::kInt64, ld.value_type);

      const MetricDescriptor& ud = *collected_metrics->metric_descriptor_map.at(
          "/tensorflow/test/counter_without_labels");
      EXPECT_EQ("/tensorflow/test/counter_without_labels", ud.name);
      EXPECT_EQ("Counter without labels.", ud.description);
      ASSERT_EQ(0, ud.label_names.size());
      EXPECT_EQ(MetricKind::kCumulative, ud.metric_kind);
      EXPECT_EQ(ValueType::kInt64, ud.value_type);
    } else {
      EXPECT_EQ(0, collected_metrics->metric_descriptor_map.size());
    }

    ASSERT_GE(collected_metrics->point_set_map.size(), 2);

    const PointSet& lps = *collected_metrics->point_set_map.at(
        "/tensorflow/test/counter_with_labels");
    EXPECT_EQ("/tensorflow/test/counter_with_labels", lps.metric_name);
    ASSERT_EQ(2, lps.points.size());
    ASSERT_EQ(2, lps.points[0]->labels.size());
    EXPECT_EQ("MyLabel0", lps.points[0]->labels[0].name);
    EXPECT_EQ("Label00", lps.points[0]->labels[0].value);
    EXPECT_EQ("MyLabel1", lps.points[0]->labels[1].name);
    EXPECT_EQ("Label10", lps.points[0]->labels[1].value);
    EXPECT_EQ(ValueType::kInt64, lps.points[0]->value_type);
    EXPECT_EQ(42, lps.points[0]->int64_value);
    EXPECT_LT(0, lps.points[0]->start_timestamp_millis);
    EXPECT_LT(0, lps.points[0]->end_timestamp_millis);
    EXPECT_GE(lps.points[0]->end_timestamp_millis,
              lps.points[0]->start_timestamp_millis);
    ASSERT_EQ(2, lps.points[1]->labels.size());
    EXPECT_EQ("MyLabel0", lps.points[1]->labels[0].name);
    EXPECT_EQ("Label01", lps.points[1]->labels[0].value);
    EXPECT_EQ("MyLabel1", lps.points[1]->labels[1].name);
    EXPECT_EQ("Label11", lps.points[1]->labels[1].value);
    EXPECT_EQ(ValueType::kInt64, lps.points[1]->value_type);
    EXPECT_EQ(58, lps.points[1]->int64_value);
    EXPECT_LT(0, lps.points[1]->start_timestamp_millis);
    EXPECT_LT(0, lps.points[1]->end_timestamp_millis);
    EXPECT_GE(lps.points[1]->end_timestamp_millis,
              lps.points[1]->start_timestamp_millis);

    const PointSet& ups = *collected_metrics->point_set_map.at(
        "/tensorflow/test/counter_without_labels");
    EXPECT_EQ("/tensorflow/test/counter_without_labels", ups.metric_name);
    ASSERT_EQ(1, ups.points.size());
    EXPECT_EQ(0, ups.points[0]->labels.size());
    EXPECT_EQ(ValueType::kInt64, ups.points[0]->value_type);
    EXPECT_EQ(7, ups.points[0]->int64_value);
    EXPECT_LT(0, ups.points[0]->start_timestamp_millis);
    EXPECT_LT(0, ups.points[0]->end_timestamp_millis);
    EXPECT_GE(ups.points[0]->end_timestamp_millis,
              ups.points[0]->start_timestamp_millis);
  }
}

TEST(CollectMetricsTest, Gauge) {
  auto string_gauge_with_labels =
      std::unique_ptr<Gauge<string, 2>>(Gauge<string, 2>::New(
          "/tensorflow/test/string_gauge_with_labels",
          "String gauge with labels.", "MyLabel0", "MyLabel1"));
  auto inteter_gauge_without_labels = std::unique_ptr<Gauge<int64_t, 0>>(
      Gauge<int64_t, 0>::New("/tensorflow/test/integer_gauge_without_labels",
                             "Integer gauge without labels."));

  string_gauge_with_labels->GetCell("Label00", "Label10")->Set("test1");
  string_gauge_with_labels->GetCell("Label01", "Label11")->Set("test2");
  inteter_gauge_without_labels->GetCell()->Set(7);

  for (const bool collect_metric_descriptors : {true, false}) {
    SCOPED_TRACE(strings::StrCat("collect_metric_descriptors: ",
                                 collect_metric_descriptors));

    auto* collection_registry = CollectionRegistry::Default();
    CollectionRegistry::CollectMetricsOptions options;
    options.collect_metric_descriptors = collect_metric_descriptors;
    const std::unique_ptr<CollectedMetrics> collected_metrics =
        collection_registry->CollectMetrics(options);

    if (collect_metric_descriptors) {
      ASSERT_GE(collected_metrics->metric_descriptor_map.size(), 2);

      const MetricDescriptor& ld = *collected_metrics->metric_descriptor_map.at(
          "/tensorflow/test/string_gauge_with_labels");
      EXPECT_EQ("/tensorflow/test/string_gauge_with_labels", ld.name);
      EXPECT_EQ("String gauge with labels.", ld.description);
      ASSERT_EQ(2, ld.label_names.size());
      EXPECT_EQ("MyLabel0", ld.label_names[0]);
      EXPECT_EQ("MyLabel1", ld.label_names[1]);
      EXPECT_EQ(MetricKind::kGauge, ld.metric_kind);
      EXPECT_EQ(ValueType::kString, ld.value_type);

      const MetricDescriptor& ud = *collected_metrics->metric_descriptor_map.at(
          "/tensorflow/test/integer_gauge_without_labels");
      EXPECT_EQ("/tensorflow/test/integer_gauge_without_labels", ud.name);
      EXPECT_EQ("Integer gauge without labels.", ud.description);
      ASSERT_EQ(0, ud.label_names.size());
      EXPECT_EQ(MetricKind::kGauge, ud.metric_kind);
      EXPECT_EQ(ValueType::kInt64, ud.value_type);
    } else {
      EXPECT_EQ(0, collected_metrics->metric_descriptor_map.size());
    }

    ASSERT_GE(collected_metrics->point_set_map.size(), 2);

    const PointSet& lps = *collected_metrics->point_set_map.at(
        "/tensorflow/test/string_gauge_with_labels");
    EXPECT_EQ("/tensorflow/test/string_gauge_with_labels", lps.metric_name);
    ASSERT_EQ(2, lps.points.size());
    ASSERT_EQ(2, lps.points[0]->labels.size());
    EXPECT_EQ("MyLabel0", lps.points[0]->labels[0].name);
    EXPECT_EQ("Label00", lps.points[0]->labels[0].value);
    EXPECT_EQ("MyLabel1", lps.points[0]->labels[1].name);
    EXPECT_EQ("Label10", lps.points[0]->labels[1].value);
    EXPECT_EQ(ValueType::kString, lps.points[0]->value_type);
    EXPECT_EQ("test1", lps.points[0]->string_value);
    EXPECT_LT(0, lps.points[0]->start_timestamp_millis);
    EXPECT_LT(0, lps.points[0]->end_timestamp_millis);
    EXPECT_GE(lps.points[0]->end_timestamp_millis,
              lps.points[0]->start_timestamp_millis);
    ASSERT_EQ(2, lps.points[1]->labels.size());
    EXPECT_EQ("MyLabel0", lps.points[1]->labels[0].name);
    EXPECT_EQ("Label01", lps.points[1]->labels[0].value);
    EXPECT_EQ("MyLabel1", lps.points[1]->labels[1].name);
    EXPECT_EQ("Label11", lps.points[1]->labels[1].value);
    EXPECT_EQ(ValueType::kString, lps.points[1]->value_type);
    EXPECT_EQ("test2", lps.points[1]->string_value);
    EXPECT_LT(0, lps.points[1]->start_timestamp_millis);
    EXPECT_LT(0, lps.points[1]->end_timestamp_millis);
    EXPECT_GE(lps.points[1]->end_timestamp_millis,
              lps.points[1]->start_timestamp_millis);

    const PointSet& ups = *collected_metrics->point_set_map.at(
        "/tensorflow/test/integer_gauge_without_labels");
    EXPECT_EQ("/tensorflow/test/integer_gauge_without_labels", ups.metric_name);
    ASSERT_EQ(1, ups.points.size());
    EXPECT_EQ(0, ups.points[0]->labels.size());
    EXPECT_EQ(ValueType::kInt64, ups.points[0]->value_type);
    EXPECT_EQ(7, ups.points[0]->int64_value);
    EXPECT_LT(0, ups.points[0]->start_timestamp_millis);
    EXPECT_LT(0, ups.points[0]->end_timestamp_millis);
    EXPECT_GE(ups.points[0]->end_timestamp_millis,
              ups.points[0]->start_timestamp_millis);
  }
}

void EqHistograms(const Histogram& expected,
                  const HistogramProto& actual_proto) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSlibPSmonitoringPScollection_registry_testDTcc mht_1(mht_1_v, 433, "", "./tensorflow/core/lib/monitoring/collection_registry_test.cc", "EqHistograms");

  Histogram actual;
  ASSERT_TRUE(actual.DecodeFromProto(actual_proto));

  EXPECT_EQ(expected.ToString(), actual.ToString());
}

TEST(CollectMetricsTest, Sampler) {
  auto sampler_with_labels = std::unique_ptr<Sampler<2>>(
      Sampler<2>::New({"/tensorflow/test/sampler_with_labels",
                       "Sampler with labels.", "MyLabel0", "MyLabel1"},
                      Buckets::Explicit({1.0, 2.0})));
  auto sampler_without_labels = std::unique_ptr<Sampler<0>>(Sampler<0>::New(
      {"/tensorflow/test/sampler_without_labels", "Sampler without labels."},
      Buckets::Explicit({0.0})));

  Histogram with_labels0({1.0, 2.0, DBL_MAX});
  sampler_with_labels->GetCell("Label00", "Label10")->Add(0.7);
  with_labels0.Add(0.7);

  Histogram with_labels1({1.0, 2.0, DBL_MAX});
  sampler_with_labels->GetCell("Label01", "Label11")->Add(1.5);
  with_labels1.Add(1.5);

  Histogram without_labels({0.0, DBL_MAX});
  sampler_without_labels->GetCell()->Add(0.5);
  without_labels.Add(0.5);

  for (const bool collect_metric_descriptors : {true, false}) {
    SCOPED_TRACE(strings::StrCat("collect_metric_descriptors: ",
                                 collect_metric_descriptors));

    auto* collection_registry = CollectionRegistry::Default();
    CollectionRegistry::CollectMetricsOptions options;
    options.collect_metric_descriptors = collect_metric_descriptors;
    const std::unique_ptr<CollectedMetrics> collected_metrics =
        collection_registry->CollectMetrics(options);

    if (collect_metric_descriptors) {
      ASSERT_GE(collected_metrics->metric_descriptor_map.size(), 2);

      const MetricDescriptor& ld = *collected_metrics->metric_descriptor_map.at(
          "/tensorflow/test/sampler_with_labels");
      EXPECT_EQ("/tensorflow/test/sampler_with_labels", ld.name);
      EXPECT_EQ("Sampler with labels.", ld.description);
      ASSERT_EQ(2, ld.label_names.size());
      EXPECT_EQ("MyLabel0", ld.label_names[0]);
      EXPECT_EQ("MyLabel1", ld.label_names[1]);
      EXPECT_EQ(MetricKind::kCumulative, ld.metric_kind);
      EXPECT_EQ(ValueType::kHistogram, ld.value_type);

      const MetricDescriptor& ud = *collected_metrics->metric_descriptor_map.at(
          "/tensorflow/test/sampler_without_labels");
      EXPECT_EQ("/tensorflow/test/sampler_without_labels", ud.name);
      EXPECT_EQ("Sampler without labels.", ud.description);
      ASSERT_EQ(0, ud.label_names.size());
      EXPECT_EQ(MetricKind::kCumulative, ud.metric_kind);
      EXPECT_EQ(ValueType::kHistogram, ud.value_type);
    } else {
      EXPECT_EQ(0, collected_metrics->metric_descriptor_map.size());
    }

    ASSERT_GE(collected_metrics->point_set_map.size(), 2);

    const PointSet& lps = *collected_metrics->point_set_map.at(
        "/tensorflow/test/sampler_with_labels");
    EXPECT_EQ("/tensorflow/test/sampler_with_labels", lps.metric_name);
    ASSERT_EQ(2, lps.points.size());
    ASSERT_EQ(2, lps.points[0]->labels.size());
    EXPECT_EQ("MyLabel0", lps.points[0]->labels[0].name);
    EXPECT_EQ("Label00", lps.points[0]->labels[0].value);
    EXPECT_EQ("MyLabel1", lps.points[0]->labels[1].name);
    EXPECT_EQ("Label10", lps.points[0]->labels[1].value);
    EXPECT_EQ(ValueType::kHistogram, lps.points[0]->value_type);
    EqHistograms(with_labels0, lps.points[0]->histogram_value);
    EXPECT_LT(0, lps.points[0]->start_timestamp_millis);
    EXPECT_LT(0, lps.points[0]->end_timestamp_millis);
    EXPECT_GE(lps.points[0]->end_timestamp_millis,
              lps.points[0]->start_timestamp_millis);
    ASSERT_EQ(2, lps.points[1]->labels.size());
    EXPECT_EQ("MyLabel0", lps.points[1]->labels[0].name);
    EXPECT_EQ("Label01", lps.points[1]->labels[0].value);
    EXPECT_EQ("MyLabel1", lps.points[1]->labels[1].name);
    EXPECT_EQ("Label11", lps.points[1]->labels[1].value);
    EXPECT_EQ(ValueType::kHistogram, lps.points[1]->value_type);
    EqHistograms(with_labels1, lps.points[1]->histogram_value);
    EXPECT_LT(0, lps.points[1]->start_timestamp_millis);
    EXPECT_LT(0, lps.points[1]->end_timestamp_millis);
    EXPECT_GE(lps.points[1]->end_timestamp_millis,
              lps.points[1]->start_timestamp_millis);

    const PointSet& ups = *collected_metrics->point_set_map.at(
        "/tensorflow/test/sampler_without_labels");
    EXPECT_EQ("/tensorflow/test/sampler_without_labels", ups.metric_name);
    ASSERT_EQ(1, ups.points.size());
    EXPECT_EQ(0, ups.points[0]->labels.size());
    EXPECT_EQ(ValueType::kHistogram, ups.points[0]->value_type);
    EqHistograms(without_labels, ups.points[0]->histogram_value);
    EXPECT_LT(0, ups.points[0]->start_timestamp_millis);
    EXPECT_LT(0, ups.points[0]->end_timestamp_millis);
    EXPECT_GE(ups.points[0]->end_timestamp_millis,
              ups.points[0]->start_timestamp_millis);
  }
}

TEST(CollectMetricsTest, PercentileSampler) {
  auto sampler_with_labels =
      std::unique_ptr<PercentileSampler<2>>(PercentileSampler<2>::New(
          {"/tensorflow/test/pctsampler_with_labels",
           "Percentile sampler with labels.", "MyLabel0", "MyLabel1"},
          {25.0, 50.0, 75.0}, 1024, UnitOfMeasure::kNumber));
  auto sampler_without_labels =
      std::unique_ptr<PercentileSampler<0>>(PercentileSampler<0>::New(
          {"/tensorflow/test/pctsampler_without_labels",
           "Percentile sampler without labels."},
          {25.0, 50.0, 75.0}, 1024, UnitOfMeasure::kNumber));

  sampler_with_labels->GetCell("Label00", "Label10")->Add(0.7);
  sampler_with_labels->GetCell("Label01", "Label11")->Add(1.5);

  sampler_without_labels->GetCell()->Add(0.5);

  for (const bool collect_metric_descriptors : {true, false}) {
    SCOPED_TRACE(strings::StrCat("collect_metric_descriptors: ",
                                 collect_metric_descriptors));

    auto* collection_registry = CollectionRegistry::Default();
    CollectionRegistry::CollectMetricsOptions options;
    options.collect_metric_descriptors = collect_metric_descriptors;
    const std::unique_ptr<CollectedMetrics> collected_metrics =
        collection_registry->CollectMetrics(options);

    if (collect_metric_descriptors) {
      ASSERT_GE(collected_metrics->metric_descriptor_map.size(), 2);

      const MetricDescriptor& ld = *collected_metrics->metric_descriptor_map.at(
          "/tensorflow/test/pctsampler_with_labels");
      EXPECT_EQ("/tensorflow/test/pctsampler_with_labels", ld.name);
      EXPECT_EQ("Percentile sampler with labels.", ld.description);
      ASSERT_EQ(2, ld.label_names.size());
      EXPECT_EQ("MyLabel0", ld.label_names[0]);
      EXPECT_EQ("MyLabel1", ld.label_names[1]);
      EXPECT_EQ(MetricKind::kCumulative, ld.metric_kind);
      EXPECT_EQ(ValueType::kPercentiles, ld.value_type);

      const MetricDescriptor& ud = *collected_metrics->metric_descriptor_map.at(
          "/tensorflow/test/pctsampler_without_labels");
      EXPECT_EQ("/tensorflow/test/pctsampler_without_labels", ud.name);
      EXPECT_EQ("Percentile sampler without labels.", ud.description);
      ASSERT_EQ(0, ud.label_names.size());
      EXPECT_EQ(MetricKind::kCumulative, ud.metric_kind);
      EXPECT_EQ(ValueType::kPercentiles, ud.value_type);
    } else {
      EXPECT_EQ(0, collected_metrics->metric_descriptor_map.size());
    }

    ASSERT_GE(collected_metrics->point_set_map.size(), 2);

    const PointSet& lps = *collected_metrics->point_set_map.at(
        "/tensorflow/test/pctsampler_with_labels");
    EXPECT_EQ("/tensorflow/test/pctsampler_with_labels", lps.metric_name);
    ASSERT_EQ(2, lps.points.size());
    ASSERT_EQ(2, lps.points[0]->labels.size());
    EXPECT_EQ("MyLabel0", lps.points[0]->labels[0].name);
    EXPECT_EQ("Label00", lps.points[0]->labels[0].value);
    EXPECT_EQ("MyLabel1", lps.points[0]->labels[1].name);
    EXPECT_EQ("Label10", lps.points[0]->labels[1].value);
    EXPECT_EQ(ValueType::kPercentiles, lps.points[0]->value_type);

    EXPECT_LT(0, lps.points[0]->start_timestamp_millis);
    EXPECT_LT(0, lps.points[0]->end_timestamp_millis);
    EXPECT_GE(lps.points[0]->end_timestamp_millis,
              lps.points[0]->start_timestamp_millis);
    ASSERT_EQ(2, lps.points[1]->labels.size());
    EXPECT_EQ("MyLabel0", lps.points[1]->labels[0].name);
    EXPECT_EQ("Label01", lps.points[1]->labels[0].value);
    EXPECT_EQ("MyLabel1", lps.points[1]->labels[1].name);
    EXPECT_EQ("Label11", lps.points[1]->labels[1].value);
    EXPECT_EQ(ValueType::kPercentiles, lps.points[1]->value_type);
    EXPECT_LT(0, lps.points[1]->start_timestamp_millis);
    EXPECT_LT(0, lps.points[1]->end_timestamp_millis);
    EXPECT_GE(lps.points[1]->end_timestamp_millis,
              lps.points[1]->start_timestamp_millis);

    const PointSet& ups = *collected_metrics->point_set_map.at(
        "/tensorflow/test/pctsampler_without_labels");
    EXPECT_EQ("/tensorflow/test/pctsampler_without_labels", ups.metric_name);
    ASSERT_EQ(1, ups.points.size());
    EXPECT_EQ(0, ups.points[0]->labels.size());
    EXPECT_EQ(ValueType::kPercentiles, ups.points[0]->value_type);
    EXPECT_LT(0, ups.points[0]->start_timestamp_millis);
    EXPECT_LT(0, ups.points[0]->end_timestamp_millis);
    EXPECT_GE(ups.points[0]->end_timestamp_millis,
              ups.points[0]->start_timestamp_millis);
  }
}

// A FakeClockEnv to manually advance time.
class FakeClockEnv : public EnvWrapper {
 public:
  FakeClockEnv() : EnvWrapper(Env::Default()), current_millis_(0) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSlibPSmonitoringPScollection_registry_testDTcc mht_2(mht_2_v, 636, "", "./tensorflow/core/lib/monitoring/collection_registry_test.cc", "FakeClockEnv");
}

  // Manually advance the current time by 'millis' milliseconds.
  void AdvanceByMillis(const uint64 millis) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSlibPSmonitoringPScollection_registry_testDTcc mht_3(mht_3_v, 642, "", "./tensorflow/core/lib/monitoring/collection_registry_test.cc", "AdvanceByMillis");
 current_millis_ += millis; }

  // Method that this environment specifically overrides.
  uint64 NowMicros() const override {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSlibPSmonitoringPScollection_registry_testDTcc mht_4(mht_4_v, 648, "", "./tensorflow/core/lib/monitoring/collection_registry_test.cc", "NowMicros");
 return current_millis_ * 1000; }

 private:
  uint64 current_millis_;
};

TEST(CollectionRegistryTest, WriteTimestamps) {
  FakeClockEnv fake_clock_env;
  auto collection_registry =
      test_util::CollectionRegistryTestAccess::CreateRegistry(&fake_clock_env);

  fake_clock_env.AdvanceByMillis(25);
  {
    const MetricDef<MetricKind::kCumulative, int64_t, 0> cumulative_metric(
        "/tensorflow/cumulative/metric", "An example metric with no labels.");
    auto handle = collection_registry->Register(
        &cumulative_metric, [&](MetricCollectorGetter getter) {
          auto metric_collector = getter.Get(&cumulative_metric);
          metric_collector.CollectValue({}, 42);
        });
    fake_clock_env.AdvanceByMillis(75);
    const std::unique_ptr<CollectedMetrics> collected_metrics =
        collection_registry->CollectMetrics({});
    const PointSet& point_set =
        *collected_metrics->point_set_map.at("/tensorflow/cumulative/metric");
    ASSERT_EQ(1, point_set.points.size());
    EXPECT_EQ(25, point_set.points[0]->start_timestamp_millis);
    EXPECT_EQ(100, point_set.points[0]->end_timestamp_millis);
  }
  {
    const MetricDef<MetricKind::kGauge, int64_t, 0> gauge_metric(
        "/tensorflow/gauge/metric", "An example metric with no labels.");
    auto handle = collection_registry->Register(
        &gauge_metric, [&](MetricCollectorGetter getter) {
          auto metric_collector = getter.Get(&gauge_metric);
          metric_collector.CollectValue({}, 42);
        });
    fake_clock_env.AdvanceByMillis(75);
    const std::unique_ptr<CollectedMetrics> collected_metrics =
        collection_registry->CollectMetrics({});
    const PointSet& point_set =
        *collected_metrics->point_set_map.at("/tensorflow/gauge/metric");
    ASSERT_EQ(1, point_set.points.size());
    EXPECT_EQ(175, point_set.points[0]->start_timestamp_millis);
    EXPECT_EQ(175, point_set.points[0]->end_timestamp_millis);
  }
}

}  // namespace
}  // namespace monitoring
}  // namespace tensorflow
