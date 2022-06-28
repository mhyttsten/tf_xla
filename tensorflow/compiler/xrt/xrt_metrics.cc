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
class MHTracer_DTPStensorflowPScompilerPSxrtPSxrt_metricsDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSxrtPSxrt_metricsDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSxrtPSxrt_metricsDTcc() {
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

#include "tensorflow/compiler/xrt/xrt_metrics.h"

#include "tensorflow/core/lib/monitoring/collection_registry.h"
#include "tensorflow/core/platform/regexp.h"

namespace tensorflow {
namespace {

static const size_t kMaxSamples = 1024;

std::vector<double> GetDefaultPercentiles() {
  return {25.0, 50.0, 80.0, 90.0, 95.0, 99.0};
}

bool IsSelectedMetric(const xrt::XRTMetricsCollect& metrics,
                      const string& name) {
   std::vector<std::string> mht_0_v;
   mht_0_v.push_back("name: \"" + name + "\"");
   MHTracer_DTPStensorflowPScompilerPSxrtPSxrt_metricsDTcc mht_0(mht_0_v, 201, "", "./tensorflow/compiler/xrt/xrt_metrics.cc", "IsSelectedMetric");

  if (metrics.metrics_regex_size() == 0) {
    return true;
  }
  for (auto& metric_regex : metrics.metrics_regex()) {
    if (RE2::FullMatch(name, metric_regex)) {
      return true;
    }
  }
  return false;
}

void SetUnitOfMeasure(xrt::MetricValues* metrics,
                      monitoring::UnitOfMeasure unit_of_measure) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSxrtPSxrt_metricsDTcc mht_1(mht_1_v, 217, "", "./tensorflow/compiler/xrt/xrt_metrics.cc", "SetUnitOfMeasure");

  switch (unit_of_measure) {
    case monitoring::UnitOfMeasure::kNumber:
      metrics->set_unit_of_measure(xrt::MetricValues::NUMBER);
      break;
    case monitoring::UnitOfMeasure::kTime:
      metrics->set_unit_of_measure(xrt::MetricValues::TIME);
      break;
    case monitoring::UnitOfMeasure::kBytes:
      metrics->set_unit_of_measure(xrt::MetricValues::BYTES);
      break;
  }
}

Status AddMetrics(xrt::MetricsReport* report,
                  const monitoring::PointSet& point_set) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSxrtPSxrt_metricsDTcc mht_2(mht_2_v, 235, "", "./tensorflow/compiler/xrt/xrt_metrics.cc", "AddMetrics");

  for (auto& point : point_set.points) {
    xrt::MetricValues* metrics = report->add_metrics();
    metrics->set_name(point_set.metric_name);
    if (point->value_type == monitoring::ValueType::kPercentiles) {
      xrt::Percentiles* percentiles = metrics->mutable_percentiles_value();
      SetUnitOfMeasure(metrics, point->percentiles_value.unit_of_measure);
      percentiles->set_start_nstime(point->percentiles_value.start_nstime);
      percentiles->set_end_nstime(point->percentiles_value.end_nstime);
      percentiles->set_min_value(point->percentiles_value.min_value);
      percentiles->set_max_value(point->percentiles_value.max_value);
      percentiles->set_mean(point->percentiles_value.mean);
      percentiles->set_stddev(point->percentiles_value.stddev);
      percentiles->set_num_samples(point->percentiles_value.num_samples);
      percentiles->set_total_samples(point->percentiles_value.total_samples);
      percentiles->set_accumulator(point->percentiles_value.accumulator);
      for (auto& pct_point : point->percentiles_value.points) {
        xrt::Percentiles::Point* xpoint = percentiles->add_points();
        xpoint->set_percentile(pct_point.percentile);
        xpoint->set_value(pct_point.value);
      }
    } else if (point->value_type == monitoring::ValueType::kInt64) {
      metrics->set_unit_of_measure(xrt::MetricValues::NUMBER);
      metrics->set_int64_value(point->int64_value);
    }
  }
  return Status::OK();
}

}  // namespace

namespace xrt_metrics {

monitoring::PercentileSamplerCell* GetAllocateCell() {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPSxrtPSxrt_metricsDTcc mht_3(mht_3_v, 271, "", "./tensorflow/compiler/xrt/xrt_metrics.cc", "GetAllocateCell");

  static monitoring::PercentileSamplerCell* cell =
      monitoring::PercentileSampler<0>::New(
          {"/tensorflow/xrt/ops/allocate", "Tracks XRTAllocate times"},
          GetDefaultPercentiles(), kMaxSamples,
          monitoring::UnitOfMeasure::kTime)
          ->GetCell();
  return cell;
}

monitoring::PercentileSamplerCell* GetAllocateUninitializedCell() {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScompilerPSxrtPSxrt_metricsDTcc mht_4(mht_4_v, 284, "", "./tensorflow/compiler/xrt/xrt_metrics.cc", "GetAllocateUninitializedCell");

  static monitoring::PercentileSamplerCell* cell =
      monitoring::PercentileSampler<0>::New(
          {"/tensorflow/xrt/ops/allocate_uninitialized",
           "Tracks XRTAllocateUninitialized times"},
          GetDefaultPercentiles(), kMaxSamples,
          monitoring::UnitOfMeasure::kTime)
          ->GetCell();
  return cell;
}

monitoring::PercentileSamplerCell* GetAllocateFromTensorCell() {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScompilerPSxrtPSxrt_metricsDTcc mht_5(mht_5_v, 298, "", "./tensorflow/compiler/xrt/xrt_metrics.cc", "GetAllocateFromTensorCell");

  static monitoring::PercentileSamplerCell* cell =
      monitoring::PercentileSampler<0>::New(
          {"/tensorflow/xrt/ops/allocate_from_tensor",
           "Tracks XRTAllocateFromTensor times"},
          GetDefaultPercentiles(), kMaxSamples,
          monitoring::UnitOfMeasure::kTime)
          ->GetCell();
  return cell;
}

monitoring::PercentileSamplerCell* GetSubTupleCell() {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScompilerPSxrtPSxrt_metricsDTcc mht_6(mht_6_v, 312, "", "./tensorflow/compiler/xrt/xrt_metrics.cc", "GetSubTupleCell");

  static monitoring::PercentileSamplerCell* cell =
      monitoring::PercentileSampler<0>::New(
          {"/tensorflow/xrt/ops/sub_tuple", "Tracks XRTSubTuple times"},
          GetDefaultPercentiles(), kMaxSamples,
          monitoring::UnitOfMeasure::kTime)
          ->GetCell();
  return cell;
}

monitoring::PercentileSamplerCell* GetMakeTupleCell() {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScompilerPSxrtPSxrt_metricsDTcc mht_7(mht_7_v, 325, "", "./tensorflow/compiler/xrt/xrt_metrics.cc", "GetMakeTupleCell");

  static monitoring::PercentileSamplerCell* cell =
      monitoring::PercentileSampler<0>::New(
          {"/tensorflow/xrt/ops/make_tuple", "Tracks XRTMakeTuple times"},
          GetDefaultPercentiles(), kMaxSamples,
          monitoring::UnitOfMeasure::kTime)
          ->GetCell();
  return cell;
}

monitoring::PercentileSamplerCell* GetReadLiteralCell() {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScompilerPSxrtPSxrt_metricsDTcc mht_8(mht_8_v, 338, "", "./tensorflow/compiler/xrt/xrt_metrics.cc", "GetReadLiteralCell");

  static monitoring::PercentileSamplerCell* cell =
      monitoring::PercentileSampler<0>::New(
          {"/tensorflow/xrt/ops/read_literal", "Tracks XRTReadLiteral times"},
          GetDefaultPercentiles(), kMaxSamples,
          monitoring::UnitOfMeasure::kTime)
          ->GetCell();
  return cell;
}

monitoring::PercentileSamplerCell* GetReadToTensorCell() {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScompilerPSxrtPSxrt_metricsDTcc mht_9(mht_9_v, 351, "", "./tensorflow/compiler/xrt/xrt_metrics.cc", "GetReadToTensorCell");

  static monitoring::PercentileSamplerCell* cell =
      monitoring::PercentileSampler<0>::New(
          {"/tensorflow/xrt/ops/read_tensor", "Tracks XRTReadToTensor times"},
          GetDefaultPercentiles(), kMaxSamples,
          monitoring::UnitOfMeasure::kTime)
          ->GetCell();
  return cell;
}

monitoring::PercentileSamplerCell* GetWriteLiteralCell() {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScompilerPSxrtPSxrt_metricsDTcc mht_10(mht_10_v, 364, "", "./tensorflow/compiler/xrt/xrt_metrics.cc", "GetWriteLiteralCell");

  static monitoring::PercentileSamplerCell* cell =
      monitoring::PercentileSampler<0>::New(
          {"/tensorflow/xrt/ops/write_literal", "Tracks XRTWriteLiteral times"},
          GetDefaultPercentiles(), kMaxSamples,
          monitoring::UnitOfMeasure::kTime)
          ->GetCell();
  return cell;
}

monitoring::PercentileSamplerCell* GetReleaseAllocationCell() {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScompilerPSxrtPSxrt_metricsDTcc mht_11(mht_11_v, 377, "", "./tensorflow/compiler/xrt/xrt_metrics.cc", "GetReleaseAllocationCell");

  static monitoring::PercentileSamplerCell* cell =
      monitoring::PercentileSampler<0>::New(
          {"/tensorflow/xrt/ops/release_allocation",
           "Tracks XRTReleaseAllocation times"},
          GetDefaultPercentiles(), kMaxSamples,
          monitoring::UnitOfMeasure::kTime)
          ->GetCell();
  return cell;
}

monitoring::PercentileSamplerCell* GetReleaseAllAllocationsCell() {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScompilerPSxrtPSxrt_metricsDTcc mht_12(mht_12_v, 391, "", "./tensorflow/compiler/xrt/xrt_metrics.cc", "GetReleaseAllAllocationsCell");

  static monitoring::PercentileSamplerCell* cell =
      monitoring::PercentileSampler<0>::New(
          {"/tensorflow/xrt/ops/release_all_allocations",
           "Tracks XRTReleaseAllAllocations times"},
          GetDefaultPercentiles(), kMaxSamples,
          monitoring::UnitOfMeasure::kTime)
          ->GetCell();
  return cell;
}

monitoring::PercentileSamplerCell* GetCompactAllocationsCell() {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPScompilerPSxrtPSxrt_metricsDTcc mht_13(mht_13_v, 405, "", "./tensorflow/compiler/xrt/xrt_metrics.cc", "GetCompactAllocationsCell");

  static monitoring::PercentileSamplerCell* cell =
      monitoring::PercentileSampler<0>::New(
          {"/tensorflow/xrt/ops/compact_allocations",
           "Tracks XRTCompactAllocations times"},
          GetDefaultPercentiles(), kMaxSamples,
          monitoring::UnitOfMeasure::kTime)
          ->GetCell();
  return cell;
}

monitoring::PercentileSamplerCell* GetCompileCell() {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPScompilerPSxrtPSxrt_metricsDTcc mht_14(mht_14_v, 419, "", "./tensorflow/compiler/xrt/xrt_metrics.cc", "GetCompileCell");

  static monitoring::PercentileSamplerCell* cell =
      monitoring::PercentileSampler<0>::New(
          {"/tensorflow/xrt/ops/compile", "Tracks XRTCompile times"},
          GetDefaultPercentiles(), kMaxSamples,
          monitoring::UnitOfMeasure::kTime)
          ->GetCell();
  return cell;
}

monitoring::PercentileSamplerCell* GetReleaseCompilationCell() {
   std::vector<std::string> mht_15_v;
   MHTracer_DTPStensorflowPScompilerPSxrtPSxrt_metricsDTcc mht_15(mht_15_v, 432, "", "./tensorflow/compiler/xrt/xrt_metrics.cc", "GetReleaseCompilationCell");

  static monitoring::PercentileSamplerCell* cell =
      monitoring::PercentileSampler<0>::New(
          {"/tensorflow/xrt/ops/release_compilation",
           "Tracks XRTReleaseCompilationRef times"},
          GetDefaultPercentiles(), kMaxSamples,
          monitoring::UnitOfMeasure::kTime)
          ->GetCell();
  return cell;
}

monitoring::PercentileSamplerCell* GetExecuteCell() {
   std::vector<std::string> mht_16_v;
   MHTracer_DTPStensorflowPScompilerPSxrtPSxrt_metricsDTcc mht_16(mht_16_v, 446, "", "./tensorflow/compiler/xrt/xrt_metrics.cc", "GetExecuteCell");

  static monitoring::PercentileSamplerCell* cell =
      monitoring::PercentileSampler<0>::New(
          {"/tensorflow/xrt/ops/execute", "Tracks XRTExecute times"},
          GetDefaultPercentiles(), kMaxSamples,
          monitoring::UnitOfMeasure::kTime)
          ->GetCell();
  return cell;
}

monitoring::PercentileSamplerCell* GetExecuteChainedCell() {
   std::vector<std::string> mht_17_v;
   MHTracer_DTPStensorflowPScompilerPSxrtPSxrt_metricsDTcc mht_17(mht_17_v, 459, "", "./tensorflow/compiler/xrt/xrt_metrics.cc", "GetExecuteChainedCell");

  static monitoring::PercentileSamplerCell* cell =
      monitoring::PercentileSampler<0>::New(
          {"/tensorflow/xrt/ops/execute_chained",
           "Tracks XRTExecuteChained times"},
          GetDefaultPercentiles(), kMaxSamples,
          monitoring::UnitOfMeasure::kTime)
          ->GetCell();
  return cell;
}

monitoring::PercentileSamplerCell* GetMemoryCompactCell() {
   std::vector<std::string> mht_18_v;
   MHTracer_DTPStensorflowPScompilerPSxrtPSxrt_metricsDTcc mht_18(mht_18_v, 473, "", "./tensorflow/compiler/xrt/xrt_metrics.cc", "GetMemoryCompactCell");

  static monitoring::PercentileSamplerCell* cell =
      monitoring::PercentileSampler<0>::New(
          {"/tensorflow/xrt/memory_manager/compaction",
           "Tracks XRT memory manager memory compaction times"},
          GetDefaultPercentiles(), kMaxSamples,
          monitoring::UnitOfMeasure::kTime)
          ->GetCell();
  return cell;
}

monitoring::PercentileSamplerCell* GetTryFreeMemoryCell() {
   std::vector<std::string> mht_19_v;
   MHTracer_DTPStensorflowPScompilerPSxrtPSxrt_metricsDTcc mht_19(mht_19_v, 487, "", "./tensorflow/compiler/xrt/xrt_metrics.cc", "GetTryFreeMemoryCell");

  static monitoring::PercentileSamplerCell* cell =
      monitoring::PercentileSampler<0>::New(
          {"/tensorflow/xrt/memory_manager/try_free_memory",
           "Tracks XRT memory manager times in trying to "
           "free memory by swpping device memory to host memory"},
          GetDefaultPercentiles(), kMaxSamples,
          monitoring::UnitOfMeasure::kTime)
          ->GetCell();
  return cell;
}

}  // namespace xrt_metrics

xla::StatusOr<xrt::MetricsReport> CollectMetrics(
    const xrt::XRTMetricsCollect& metrics) {
  auto* collection_registry = monitoring::CollectionRegistry::Default();
  monitoring::CollectionRegistry::CollectMetricsOptions options;
  options.collect_metric_descriptors = false;
  auto collected_metrics = collection_registry->CollectMetrics(options);
  xrt::MetricsReport report;
  for (auto& name_pointset : collected_metrics->point_set_map) {
    if (IsSelectedMetric(metrics, name_pointset.first)) {
      TF_RETURN_IF_ERROR(AddMetrics(&report, *name_pointset.second));
    }
  }
  return std::move(report);
}

}  // namespace tensorflow
