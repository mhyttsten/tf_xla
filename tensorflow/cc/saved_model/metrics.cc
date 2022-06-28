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
class MHTracer_DTPStensorflowPSccPSsaved_modelPSmetricsDTcc {
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
   MHTracer_DTPStensorflowPSccPSsaved_modelPSmetricsDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSccPSsaved_modelPSmetricsDTcc() {
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

/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/cc/saved_model/metrics.h"

#include <string>

#include "tensorflow/core/lib/monitoring/counter.h"
#include "tensorflow/core/lib/monitoring/sampler.h"

namespace tensorflow {
namespace metrics {

namespace {

// Counter that tracks total number and `write_version` of SavedModels written.
auto* saved_model_write_counter = monitoring::Counter<1>::New(
    "/tensorflow/core/saved_model/write/count",
    "The number of SavedModels successfully written.", "write_version");

// Counter that tracks total number and `write_version` of SavedModels read.
auto* saved_model_read_counter = monitoring::Counter<1>::New(
    "/tensorflow/core/saved_model/read/count",
    "The number of SavedModels successfully loaded.", "write_version");

// Counter that tracks number of calls for each SavedModel write API. Summing
// across "api_label" is not expected to equal the ".../write/count" cell value
// because programs can invoke more than one API to save a single SM and
// because the API may error out before successfully writing a SM.
auto* saved_model_write_api = monitoring::Counter<1>::New(
    "/tensorflow/core/saved_model/write/api",
    "The API used to write the SavedModel.", "api_label");

// Counter that tracks number of calls for each SavedModel read API. Summing
// across "api_label" is not expected to equal the ".../read/count" cell value
// because programs can invoke more than one API to load a single SM and
// because the API may error out before successfully reading a SM.
auto* saved_model_read_api = monitoring::Counter<1>::New(
    "/tensorflow/core/saved_model/read/api",
    "The API used to load the SavedModel.", "api_label");

// Distribution of checkpoint write durations.
auto* checkpoint_write_durations = monitoring::Sampler<1>::New(
    {
        "/tensorflow/core/checkpoint/write/write_durations",  // Metric name.
        "Distribution of the wall time duration in microseconds of the "
        "checkpoint write operation.",  // Metric description.
        "api_label"                     // Cell label.
    },
    // Scale of 1000, growth factor of 1.5 with upper bound of ~184 minutes.
    monitoring::Buckets::Exponential(1000, 1.5, 41));

// Distribution of checkpoint read durations.
auto* checkpoint_read_durations = monitoring::Sampler<1>::New(
    {
        "/tensorflow/core/checkpoint/read/read_durations",  // Metric name.
        "Distribution of the wall time duration in microseconds of the "
        "checkpoint read operation.",  // Metric description.
        "api_label"                    // Cell label.
    },
    // Scale of 1000, growth factor of 1.5 with upper bound of ~184 minutes.
    monitoring::Buckets::Exponential(1000, 1.5, 41));

// Counter that accumulates total time elapsed between module import time and
// the last successful Checkpoint write prior to job pre-emption or completion.
auto* checkpoint_training_time_saved = monitoring::Counter<1>::New(
    "/tensorflow/core/checkpoint/write/training_time_saved",
    "Total time in microseconds elapsed between two consecutive write "
    "operations in a single job or between Checkpoint construction and the "
    "first write operation.",
    "api_label");

// Counter that records filesize (MB) of written checkpoint. Contains two cells:
// (api_label, filesize). Cardinality should not be an issue as the filesize
// should be equal among all checkpoints written per job.
auto* checkpoint_size = monitoring::Counter<2>::New(
    "/tensorflow/core/checkpoint/write/checkpoint_size",
    "Size of checkpoint (.index and sharded data files), rounded to the "
    "nearest 100 MB.",
    "api_label", "filesize");

}  // namespace

monitoring::CounterCell& SavedModelWrite(absl::string_view write_version) {
   std::vector<std::string> mht_0_v;
   mht_0_v.push_back("write_version: \"" + std::string(write_version.data(), write_version.size()) + "\"");
   MHTracer_DTPStensorflowPSccPSsaved_modelPSmetricsDTcc mht_0(mht_0_v, 266, "", "./tensorflow/cc/saved_model/metrics.cc", "SavedModelWrite");

  return *saved_model_write_counter->GetCell(std::string(write_version));
}

monitoring::CounterCell& SavedModelRead(absl::string_view write_version) {
   std::vector<std::string> mht_1_v;
   mht_1_v.push_back("write_version: \"" + std::string(write_version.data(), write_version.size()) + "\"");
   MHTracer_DTPStensorflowPSccPSsaved_modelPSmetricsDTcc mht_1(mht_1_v, 274, "", "./tensorflow/cc/saved_model/metrics.cc", "SavedModelRead");

  return *saved_model_read_counter->GetCell(std::string(write_version));
}

monitoring::CounterCell& SavedModelWriteApi(absl::string_view api_label) {
   std::vector<std::string> mht_2_v;
   mht_2_v.push_back("api_label: \"" + std::string(api_label.data(), api_label.size()) + "\"");
   MHTracer_DTPStensorflowPSccPSsaved_modelPSmetricsDTcc mht_2(mht_2_v, 282, "", "./tensorflow/cc/saved_model/metrics.cc", "SavedModelWriteApi");

  return *saved_model_write_api->GetCell(std::string(api_label));
}

monitoring::CounterCell& SavedModelReadApi(absl::string_view api_label) {
   std::vector<std::string> mht_3_v;
   mht_3_v.push_back("api_label: \"" + std::string(api_label.data(), api_label.size()) + "\"");
   MHTracer_DTPStensorflowPSccPSsaved_modelPSmetricsDTcc mht_3(mht_3_v, 290, "", "./tensorflow/cc/saved_model/metrics.cc", "SavedModelReadApi");

  return *saved_model_read_api->GetCell(std::string(api_label));
}

monitoring::SamplerCell& CheckpointReadDuration(absl::string_view api_label) {
   std::vector<std::string> mht_4_v;
   mht_4_v.push_back("api_label: \"" + std::string(api_label.data(), api_label.size()) + "\"");
   MHTracer_DTPStensorflowPSccPSsaved_modelPSmetricsDTcc mht_4(mht_4_v, 298, "", "./tensorflow/cc/saved_model/metrics.cc", "CheckpointReadDuration");

  return *checkpoint_read_durations->GetCell(std::string(api_label));
}

monitoring::SamplerCell& CheckpointWriteDuration(absl::string_view api_label) {
   std::vector<std::string> mht_5_v;
   mht_5_v.push_back("api_label: \"" + std::string(api_label.data(), api_label.size()) + "\"");
   MHTracer_DTPStensorflowPSccPSsaved_modelPSmetricsDTcc mht_5(mht_5_v, 306, "", "./tensorflow/cc/saved_model/metrics.cc", "CheckpointWriteDuration");

  return *checkpoint_write_durations->GetCell(std::string(api_label));
}

monitoring::CounterCell& TrainingTimeSaved(absl::string_view api_label) {
   std::vector<std::string> mht_6_v;
   mht_6_v.push_back("api_label: \"" + std::string(api_label.data(), api_label.size()) + "\"");
   MHTracer_DTPStensorflowPSccPSsaved_modelPSmetricsDTcc mht_6(mht_6_v, 314, "", "./tensorflow/cc/saved_model/metrics.cc", "TrainingTimeSaved");

  return *checkpoint_training_time_saved->GetCell(std::string(api_label));
}

monitoring::CounterCell& CheckpointSize(absl::string_view api_label,
                                        int64_t filesize) {
   std::vector<std::string> mht_7_v;
   mht_7_v.push_back("api_label: \"" + std::string(api_label.data(), api_label.size()) + "\"");
   MHTracer_DTPStensorflowPSccPSsaved_modelPSmetricsDTcc mht_7(mht_7_v, 323, "", "./tensorflow/cc/saved_model/metrics.cc", "CheckpointSize");

  return *checkpoint_size->GetCell(std::string(api_label),
                                   std::to_string(filesize));
}

}  // namespace metrics
}  // namespace tensorflow
