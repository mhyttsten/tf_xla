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
class MHTracer_DTPStensorflowPSpythonPSsaved_modelPSpywrap_saved_model_metricsDTcc {
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
   MHTracer_DTPStensorflowPSpythonPSsaved_modelPSpywrap_saved_model_metricsDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSpythonPSsaved_modelPSpywrap_saved_model_metricsDTcc() {
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

Licensed under the Apache License, Version 2.0 (the "License");;
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "absl/strings/string_view.h"
#include "pybind11/pybind11.h"
#include "tensorflow/cc/saved_model/metrics.h"

namespace tensorflow {
namespace saved_model {
namespace python {

namespace py = pybind11;

void DefineMetricsModule(py::module main_module) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSpythonPSsaved_modelPSpywrap_saved_model_metricsDTcc mht_0(mht_0_v, 195, "", "./tensorflow/python/saved_model/pywrap_saved_model_metrics.cc", "DefineMetricsModule");

  auto m = main_module.def_submodule("metrics");

  m.doc() = "Python bindings for TensorFlow SavedModel and Checkpoint Metrics.";

  m.def(
      "IncrementWrite",
      [](const char* write_version) {
        metrics::SavedModelWrite(write_version).IncrementBy(1);
      },
      py::kw_only(), py::arg("write_version"),
      py::doc("Increment the '/tensorflow/core/saved_model/write/count' "
              "counter."));

  m.def(
      "GetWrite",
      [](const char* write_version) {
        return metrics::SavedModelWrite(write_version).value();
      },
      py::kw_only(), py::arg("write_version"),
      py::doc("Get value of '/tensorflow/core/saved_model/write/count' "
              "counter."));

  m.def(
      "IncrementWriteApi",
      [](const char* api_label) {
        metrics::SavedModelWriteApi(api_label).IncrementBy(1);
      },
      py::doc("Increment the '/tensorflow/core/saved_model/write/api' "
              "counter for API with `api_label`"));

  m.def(
      "GetWriteApi",
      [](const char* api_label) {
        return metrics::SavedModelWriteApi(api_label).value();
      },
      py::doc("Get value of '/tensorflow/core/saved_model/write/api' "
              "counter for `api_label` cell."));

  m.def(
      "IncrementRead",
      [](const char* write_version) {
        metrics::SavedModelRead(write_version).IncrementBy(1);
      },
      py::kw_only(), py::arg("write_version"),
      py::doc("Increment the '/tensorflow/core/saved_model/read/count' "
              "counter after reading a SavedModel with the specifed "
              "`write_version`."));

  m.def(
      "GetRead",
      [](const char* write_version) {
        return metrics::SavedModelRead(write_version).value();
      },
      py::kw_only(), py::arg("write_version"),
      py::doc("Get value of '/tensorflow/core/saved_model/read/count' "
              "counter for SavedModels with the specified `write_version`."));

  m.def(
      "IncrementReadApi",
      [](const char* api_label) {
        metrics::SavedModelReadApi(api_label).IncrementBy(1);
      },
      py::doc("Increment the '/tensorflow/core/saved_model/read/api' "
              "counter for API with `api_label`."));

  m.def(
      "GetReadApi",
      [](const char* api_label) {
        return metrics::SavedModelReadApi(api_label).value();
      },
      py::doc("Get value of '/tensorflow/core/saved_model/read/api' "
              "counter for `api_label` cell."));

  m.def(
      "AddCheckpointReadDuration",
      [](const char* api_label, double microseconds) {
        metrics::CheckpointReadDuration(api_label).Add(microseconds);
      },
      py::kw_only(), py::arg("api_label"), py::arg("microseconds"),
      py::doc("Add `microseconds` to the cell `api_label`for "
              "'/tensorflow/core/checkpoint/read/read_durations'."));

  m.def(
      "GetCheckpointReadDurations",
      [](const char* api_label) {
        // This function is called sparingly in unit tests, so protobuf
        // (de)-serialization round trip is not an issue.
        return py::bytes(metrics::CheckpointReadDuration(api_label)
                             .value()
                             .SerializeAsString());
      },
      py::kw_only(), py::arg("api_label"),
      py::doc("Get serialized HistogramProto of `api_label` cell for "
              "'/tensorflow/core/checkpoint/read/read_durations'."));

  m.def(
      "AddCheckpointWriteDuration",
      [](const char* api_label, double microseconds) {
        metrics::CheckpointWriteDuration(api_label).Add(microseconds);
      },
      py::kw_only(), py::arg("api_label"), py::arg("microseconds"),
      py::doc("Add `microseconds` to the cell `api_label` for "
              "'/tensorflow/core/checkpoint/write/write_durations'."));

  m.def(
      "GetCheckpointWriteDurations",
      [](const char* api_label) {
        // This function is called sparingly, so protobuf (de)-serialization
        // round trip is not an issue.
        return py::bytes(metrics::CheckpointWriteDuration(api_label)
                             .value()
                             .SerializeAsString());
      },
      py::kw_only(), py::arg("api_label"),
      py::doc("Get serialized HistogramProto of `api_label` cell for "
              "'/tensorflow/core/checkpoint/write/write_durations'."));

  m.def(
      "AddTrainingTimeSaved",
      [](const char* api_label, double microseconds) {
        metrics::TrainingTimeSaved(api_label).IncrementBy(microseconds);
      },
      py::kw_only(), py::arg("api_label"), py::arg("microseconds"),
      py::doc("Add `microseconds` to the cell `api_label` for "
              "'/tensorflow/core/checkpoint/write/training_time_saved'."));

  m.def(
      "GetTrainingTimeSaved",
      [](const char* api_label) {
        return metrics::TrainingTimeSaved(api_label).value();
      },
      py::kw_only(), py::arg("api_label"),
      py::doc("Get cell `api_label` for "
              "'/tensorflow/core/checkpoint/write/training_time_saved'."));

  m.def(
      "CalculateFileSize",
      [](const char* filename) {
        Env* env = Env::Default();
        uint64 filesize = 0;
        if (!env->GetFileSize(filename, &filesize).ok()) {
          return (int64_t)-1;
        }
        // Convert to MB.
        int64_t filesize_mb = filesize / 1000;
        // Round to the nearest 100 MB.
        // Smaller multiple.
        int64_t a = (filesize_mb / 100) * 100;
        // Larger multiple.
        int64_t b = a + 100;
        // Return closest of two.
        return (filesize_mb - a > b - filesize_mb) ? b : a;
      },
      py::doc("Calculate filesize (MB) for `filename`, rounding to the nearest "
              "100MB. Returns -1 if `filename` is invalid."));

  m.def(
      "RecordCheckpointSize",
      [](const char* api_label, int64_t filesize) {
        metrics::CheckpointSize(api_label, filesize).IncrementBy(1);
      },
      py::kw_only(), py::arg("api_label"), py::arg("filesize"),
      py::doc("Increment the "
              "'/tensorflow/core/checkpoint/write/checkpoint_size' counter for "
              "cell (api_label, filesize) after writing a checkpoint."));

  m.def(
      "GetCheckpointSize",
      [](const char* api_label, uint64 filesize) {
        return metrics::CheckpointSize(api_label, filesize).value();
      },
      py::kw_only(), py::arg("api_label"), py::arg("filesize"),
      py::doc("Get cell (api_label, filesize) for "
              "'/tensorflow/core/checkpoint/write/checkpoint_size'."));
}

}  // namespace python
}  // namespace saved_model
}  // namespace tensorflow
