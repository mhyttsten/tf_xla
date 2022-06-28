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
#ifndef TENSORFLOW_C_EAGER_TFE_MONITORING_INTERNAL_H_
#define TENSORFLOW_C_EAGER_TFE_MONITORING_INTERNAL_H_
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
class MHTracer_DTPStensorflowPScPSeagerPStfe_monitoring_internalDTh {
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
   MHTracer_DTPStensorflowPScPSeagerPStfe_monitoring_internalDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScPSeagerPStfe_monitoring_internalDTh() {
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


#include <functional>
#include <memory>
#include <string>

#include "absl/memory/memory.h"
#include "tensorflow/core/lib/monitoring/counter.h"
#include "tensorflow/core/lib/monitoring/gauge.h"
#include "tensorflow/core/lib/monitoring/sampler.h"
#include "tensorflow/core/platform/types.h"

struct TFE_MonitoringCounterCell {
  tensorflow::monitoring::CounterCell cell;
};

template <int NumLabels>
struct TFE_MonitoringCounter {
  template <typename... LabelDesc>
  TFE_MonitoringCounter(const char* name, const char* description,
                        LabelDesc&&... label) {
    counter = absl::WrapUnique(tensorflow::monitoring::Counter<NumLabels>::New(
        name, description, label...));
  }

  std::unique_ptr<tensorflow::monitoring::Counter<NumLabels>> counter;
};

struct TFE_MonitoringCounter0 : TFE_MonitoringCounter<0> {
  using TFE_MonitoringCounter::TFE_MonitoringCounter;
};
struct TFE_MonitoringCounter1 : TFE_MonitoringCounter<1> {
  using TFE_MonitoringCounter::TFE_MonitoringCounter;
};
struct TFE_MonitoringCounter2 : TFE_MonitoringCounter<2> {
  using TFE_MonitoringCounter::TFE_MonitoringCounter;
};

struct TFE_MonitoringIntGaugeCell {
  tensorflow::monitoring::GaugeCell<int64_t> cell;
};
struct TFE_MonitoringStringGaugeCell {
  tensorflow::monitoring::GaugeCell<tensorflow::string> cell;
};
struct TFE_MonitoringBoolGaugeCell {
  tensorflow::monitoring::GaugeCell<bool> cell;
};

template <typename ValueType, int NumLabels>
struct TFE_MonitoringGauge {
  template <typename... LabelDesc>
  TFE_MonitoringGauge(const char* name, const char* description,
                      LabelDesc&&... label) {
    gauge = absl::WrapUnique(
        tensorflow::monitoring::Gauge<ValueType, NumLabels>::New(
            name, description, label...));
  }

  std::unique_ptr<tensorflow::monitoring::Gauge<ValueType, NumLabels>> gauge;
};

struct TFE_MonitoringIntGauge0 : TFE_MonitoringGauge<int64_t, 0> {
  using TFE_MonitoringGauge::TFE_MonitoringGauge;
};
struct TFE_MonitoringIntGauge1 : TFE_MonitoringGauge<int64_t, 1> {
  using TFE_MonitoringGauge::TFE_MonitoringGauge;
};
struct TFE_MonitoringIntGauge2 : TFE_MonitoringGauge<int64_t, 2> {
  using TFE_MonitoringGauge::TFE_MonitoringGauge;
};

struct TFE_MonitoringStringGauge0 : TFE_MonitoringGauge<tensorflow::string, 0> {
  using TFE_MonitoringGauge::TFE_MonitoringGauge;
};
struct TFE_MonitoringStringGauge1 : TFE_MonitoringGauge<tensorflow::string, 1> {
  using TFE_MonitoringGauge::TFE_MonitoringGauge;
};
struct TFE_MonitoringStringGauge2 : TFE_MonitoringGauge<tensorflow::string, 2> {
  using TFE_MonitoringGauge::TFE_MonitoringGauge;
};
struct TFE_MonitoringStringGauge3 : TFE_MonitoringGauge<tensorflow::string, 3> {
  using TFE_MonitoringGauge::TFE_MonitoringGauge;
};
struct TFE_MonitoringStringGauge4 : TFE_MonitoringGauge<tensorflow::string, 4> {
  using TFE_MonitoringGauge::TFE_MonitoringGauge;
};

struct TFE_MonitoringBoolGauge0 : TFE_MonitoringGauge<bool, 0> {
  using TFE_MonitoringGauge::TFE_MonitoringGauge;
};
struct TFE_MonitoringBoolGauge1 : TFE_MonitoringGauge<bool, 1> {
  using TFE_MonitoringGauge::TFE_MonitoringGauge;
};
struct TFE_MonitoringBoolGauge2 : TFE_MonitoringGauge<bool, 2> {
  using TFE_MonitoringGauge::TFE_MonitoringGauge;
};

struct TFE_MonitoringBuckets {
  explicit TFE_MonitoringBuckets(
      std::function<std::unique_ptr<tensorflow::monitoring::Buckets>(void)>
          fn) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScPSeagerPStfe_monitoring_internalDTh mht_0(mht_0_v, 285, "", "./tensorflow/c/eager/tfe_monitoring_internal.h", "TFE_MonitoringBuckets");

    create_buckets = fn;
  }

  std::function<std::unique_ptr<tensorflow::monitoring::Buckets>(void)>
      create_buckets;
};

struct TFE_MonitoringSamplerCell {
  tensorflow::monitoring::SamplerCell cell;
};

template <int NumLabels>
struct TFE_MonitoringSampler {
  template <typename... LabelDesc>
  TFE_MonitoringSampler(
      const char* name,
      std::unique_ptr<tensorflow::monitoring::Buckets> buckets,
      const char* description, LabelDesc&&... label) {
    sampler = absl::WrapUnique(tensorflow::monitoring::Sampler<NumLabels>::New(
        {name, description, label...}, std::move(buckets)));
  }

  std::unique_ptr<tensorflow::monitoring::Sampler<NumLabels>> sampler;
};

struct TFE_MonitoringSampler0 : TFE_MonitoringSampler<0> {
  using TFE_MonitoringSampler::TFE_MonitoringSampler;
};
struct TFE_MonitoringSampler1 : TFE_MonitoringSampler<1> {
  using TFE_MonitoringSampler::TFE_MonitoringSampler;
};
struct TFE_MonitoringSampler2 : TFE_MonitoringSampler<2> {
  using TFE_MonitoringSampler::TFE_MonitoringSampler;
};

#endif  // TENSORFLOW_C_EAGER_TFE_MONITORING_INTERNAL_H_
