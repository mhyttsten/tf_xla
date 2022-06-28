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
class MHTracer_DTPStensorflowPScorePSlibPSmonitoringPScollection_registryDTcc {
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
   MHTracer_DTPStensorflowPScorePSlibPSmonitoringPScollection_registryDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSlibPSmonitoringPScollection_registryDTcc() {
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

// We replace this implementation with a null implementation for mobile
// platforms.
#ifndef IS_MOBILE_PLATFORM

#include "tensorflow/core/platform/logging.h"

namespace tensorflow {
namespace monitoring {
namespace internal {

void Collector::CollectMetricValues(
    const CollectionRegistry::CollectionInfo& info) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSlibPSmonitoringPScollection_registryDTcc mht_0(mht_0_v, 198, "", "./tensorflow/core/lib/monitoring/collection_registry.cc", "Collector::CollectMetricValues");

  info.collection_function(MetricCollectorGetter(
      this, info.metric_def, info.registration_time_millis));
}

std::unique_ptr<CollectedMetrics> Collector::ConsumeCollectedMetrics() {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSlibPSmonitoringPScollection_registryDTcc mht_1(mht_1_v, 206, "", "./tensorflow/core/lib/monitoring/collection_registry.cc", "Collector::ConsumeCollectedMetrics");

  mutex_lock l(mu_);
  return std::move(collected_metrics_);
}

void Collector::CollectMetricDescriptor(
    const AbstractMetricDef* const metric_def) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSlibPSmonitoringPScollection_registryDTcc mht_2(mht_2_v, 215, "", "./tensorflow/core/lib/monitoring/collection_registry.cc", "Collector::CollectMetricDescriptor");

  auto* const metric_descriptor = [&]() {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSlibPSmonitoringPScollection_registryDTcc mht_3(mht_3_v, 219, "", "./tensorflow/core/lib/monitoring/collection_registry.cc", "lambda");

    mutex_lock l(mu_);
    return collected_metrics_->metric_descriptor_map
        .insert(std::make_pair(
            string(metric_def->name()),
            std::unique_ptr<MetricDescriptor>(new MetricDescriptor())))
        .first->second.get();
  }();
  metric_descriptor->name = string(metric_def->name());
  metric_descriptor->description = string(metric_def->description());

  for (const StringPiece label_name : metric_def->label_descriptions()) {
    metric_descriptor->label_names.emplace_back(label_name);
  }

  metric_descriptor->metric_kind = metric_def->kind();
  metric_descriptor->value_type = metric_def->value_type();
}

}  // namespace internal

// static
CollectionRegistry* CollectionRegistry::Default() {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSlibPSmonitoringPScollection_registryDTcc mht_4(mht_4_v, 244, "", "./tensorflow/core/lib/monitoring/collection_registry.cc", "CollectionRegistry::Default");

  static CollectionRegistry* default_registry =
      new CollectionRegistry(Env::Default());
  return default_registry;
}

CollectionRegistry::CollectionRegistry(Env* const env) : env_(env) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSlibPSmonitoringPScollection_registryDTcc mht_5(mht_5_v, 253, "", "./tensorflow/core/lib/monitoring/collection_registry.cc", "CollectionRegistry::CollectionRegistry");
}

std::unique_ptr<CollectionRegistry::RegistrationHandle>
CollectionRegistry::Register(const AbstractMetricDef* const metric_def,
                             const CollectionFunction& collection_function) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePSlibPSmonitoringPScollection_registryDTcc mht_6(mht_6_v, 260, "", "./tensorflow/core/lib/monitoring/collection_registry.cc", "CollectionRegistry::Register");

  CHECK(collection_function)
      << "Requires collection_function to contain an implementation.";

  mutex_lock l(mu_);

  const auto found_it = registry_.find(metric_def->name());
  if (found_it != registry_.end()) {
    LOG(ERROR) << "Cannot register 2 metrics with the same name: "
               << metric_def->name();
    return nullptr;
  }
  registry_.insert(
      {metric_def->name(),
       {metric_def, collection_function, env_->NowMicros() / 1000}});

  return std::unique_ptr<RegistrationHandle>(
      new RegistrationHandle(this, metric_def));
}

void CollectionRegistry::Unregister(const AbstractMetricDef* const metric_def) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePSlibPSmonitoringPScollection_registryDTcc mht_7(mht_7_v, 283, "", "./tensorflow/core/lib/monitoring/collection_registry.cc", "CollectionRegistry::Unregister");

  mutex_lock l(mu_);
  registry_.erase(metric_def->name());
}

std::unique_ptr<CollectedMetrics> CollectionRegistry::CollectMetrics(
    const CollectMetricsOptions& options) const {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScorePSlibPSmonitoringPScollection_registryDTcc mht_8(mht_8_v, 292, "", "./tensorflow/core/lib/monitoring/collection_registry.cc", "CollectionRegistry::CollectMetrics");

  internal::Collector collector(env_->NowMicros() / 1000);

  mutex_lock l(mu_);
  for (const auto& registration : registry_) {
    if (options.collect_metric_descriptors) {
      collector.CollectMetricDescriptor(registration.second.metric_def);
    }

    collector.CollectMetricValues(registration.second /* collection_info */);
  }
  return collector.ConsumeCollectedMetrics();
}

}  // namespace monitoring
}  // namespace tensorflow

#endif  // IS_MOBILE_PLATFORM
