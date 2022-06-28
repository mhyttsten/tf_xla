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
class MHTracer_DTPStensorflowPScompilerPSmlirPSquantizationPStensorflowPSpythonPSquantize_model_wrapperDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSmlirPSquantizationPStensorflowPSpythonPSquantize_model_wrapperDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSmlirPSquantizationPStensorflowPSpythonPSquantize_model_wrapperDTcc() {
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

/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/compiler/mlir/quantization/tensorflow/python/quantize_model_wrapper.h"

#include <memory>
#include <string>
#include <string_view>
#include <unordered_set>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "llvm/Support/Debug.h"
#include "pybind11/numpy.h"
#include "pybind11/pybind11.h"
#include "tensorflow/cc/saved_model/loader.h"
#include "tensorflow/compiler/mlir/quantization/tensorflow/calibrator/calibrator_singleton.h"
#include "tensorflow/compiler/mlir/quantization/tensorflow/python/quantize_model.h"
#include "tensorflow/core/common_runtime/graph_constructor.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/platform/statusor.h"
#include "tensorflow/core/platform/stringpiece.h"
#include "tensorflow/lite/python/interpreter_wrapper/python_utils.h"

using tensorflow::FunctionDefLibrary;
using tensorflow::Graph;
using tensorflow::GraphDef;
using tensorflow::ImportGraphDefOptions;
using tensorflow::OpRegistry;

namespace tensorflow {
namespace quantization {

PyObject* QuantizeQATModel(absl::string_view saved_model_path,
                           absl::string_view exported_names_str,
                           absl::string_view tags) {
   std::vector<std::string> mht_0_v;
   mht_0_v.push_back("saved_model_path: \"" + std::string(saved_model_path.data(), saved_model_path.size()) + "\"");
   mht_0_v.push_back("exported_names_str: \"" + std::string(exported_names_str.data(), exported_names_str.size()) + "\"");
   mht_0_v.push_back("tags: \"" + std::string(tags.data(), tags.size()) + "\"");
   MHTracer_DTPStensorflowPScompilerPSmlirPSquantizationPStensorflowPSpythonPSquantize_model_wrapperDTcc mht_0(mht_0_v, 222, "", "./tensorflow/compiler/mlir/quantization/tensorflow/python/quantize_model_wrapper.cc", "QuantizeQATModel");

  absl::StatusOr<tensorflow::GraphDef> graph_def =
      internal::QuantizeQATModel(saved_model_path, exported_names_str, tags);
  if (!graph_def.ok()) {
    PyErr_Format(PyExc_ValueError, "failed to quantize QAT model: %s",
                 std::string(graph_def.status().message()).c_str());
    return nullptr;
  }

  std::string ret_str = graph_def.value().SerializeAsString();

  return tflite::python_utils::ConvertToPyString(ret_str.c_str(),
                                                 ret_str.size());
}

PyObject* QuantizePTQModelPreCalibration(absl::string_view saved_model_path,
                                         absl::string_view exported_names_str,
                                         absl::string_view tags) {
   std::vector<std::string> mht_1_v;
   mht_1_v.push_back("saved_model_path: \"" + std::string(saved_model_path.data(), saved_model_path.size()) + "\"");
   mht_1_v.push_back("exported_names_str: \"" + std::string(exported_names_str.data(), exported_names_str.size()) + "\"");
   mht_1_v.push_back("tags: \"" + std::string(tags.data(), tags.size()) + "\"");
   MHTracer_DTPStensorflowPScompilerPSmlirPSquantizationPStensorflowPSpythonPSquantize_model_wrapperDTcc mht_1(mht_1_v, 245, "", "./tensorflow/compiler/mlir/quantization/tensorflow/python/quantize_model_wrapper.cc", "QuantizePTQModelPreCalibration");

  absl::StatusOr<tensorflow::GraphDef> graph_def =
      internal::QuantizePTQModelPreCalibration(saved_model_path,
                                               exported_names_str, tags);
  if (!graph_def.ok()) {
    PyErr_Format(PyExc_ValueError,
                 "failed to quantize PTQ model at the precalibration stage: %s",
                 std::string(graph_def.status().message()).c_str());
    return nullptr;
  }

  std::string ret_str = graph_def.value().SerializeAsString();

  return tflite::python_utils::ConvertToPyString(ret_str.c_str(),
                                                 ret_str.size());
}

PyObject* QuantizePTQModelPostCalibration(absl::string_view saved_model_path,
                                          absl::string_view exported_names_str,
                                          absl::string_view tags) {
   std::vector<std::string> mht_2_v;
   mht_2_v.push_back("saved_model_path: \"" + std::string(saved_model_path.data(), saved_model_path.size()) + "\"");
   mht_2_v.push_back("exported_names_str: \"" + std::string(exported_names_str.data(), exported_names_str.size()) + "\"");
   mht_2_v.push_back("tags: \"" + std::string(tags.data(), tags.size()) + "\"");
   MHTracer_DTPStensorflowPScompilerPSmlirPSquantizationPStensorflowPSpythonPSquantize_model_wrapperDTcc mht_2(mht_2_v, 270, "", "./tensorflow/compiler/mlir/quantization/tensorflow/python/quantize_model_wrapper.cc", "QuantizePTQModelPostCalibration");

  absl::StatusOr<tensorflow::GraphDef> graph_def =
      internal::QuantizePTQModelPostCalibration(saved_model_path,
                                                exported_names_str, tags);
  if (!graph_def.ok()) {
    PyErr_Format(
        PyExc_ValueError,
        "failed to quantize PTQ model at the postcalibration stage: %s",
        std::string(graph_def.status().message()).c_str());
    return nullptr;
  }

  std::string ret_str = graph_def.value().SerializeAsString();

  return tflite::python_utils::ConvertToPyString(ret_str.c_str(),
                                                 ret_str.size());
}

void ClearCollectedInformationFromCalibrator() {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSquantizationPStensorflowPSpythonPSquantize_model_wrapperDTcc mht_3(mht_3_v, 291, "", "./tensorflow/compiler/mlir/quantization/tensorflow/python/quantize_model_wrapper.cc", "ClearCollectedInformationFromCalibrator");

  calibrator::CalibratorSingleton::ClearCollectedInformation();
}

void ClearDataFromCalibrator(absl::string_view id) {
   std::vector<std::string> mht_4_v;
   mht_4_v.push_back("id: \"" + std::string(id.data(), id.size()) + "\"");
   MHTracer_DTPStensorflowPScompilerPSmlirPSquantizationPStensorflowPSpythonPSquantize_model_wrapperDTcc mht_4(mht_4_v, 299, "", "./tensorflow/compiler/mlir/quantization/tensorflow/python/quantize_model_wrapper.cc", "ClearDataFromCalibrator");

  calibrator::CalibratorSingleton::ClearData(id);
}

float GetMinFromCalibrator(absl::string_view id) {
   std::vector<std::string> mht_5_v;
   mht_5_v.push_back("id: \"" + std::string(id.data(), id.size()) + "\"");
   MHTracer_DTPStensorflowPScompilerPSmlirPSquantizationPStensorflowPSpythonPSquantize_model_wrapperDTcc mht_5(mht_5_v, 307, "", "./tensorflow/compiler/mlir/quantization/tensorflow/python/quantize_model_wrapper.cc", "GetMinFromCalibrator");

  absl::optional<std::pair<float, float>> min_max =
      calibrator::CalibratorSingleton::GetMinMax(id);
  if (!min_max.has_value()) {
    PyErr_Format(PyExc_ValueError, "No calibrated data for '%s'",
                 std::string{id}.c_str());
    throw py::error_already_set();
  }

  return min_max->first;
}

float GetMaxFromCalibrator(absl::string_view id) {
   std::vector<std::string> mht_6_v;
   mht_6_v.push_back("id: \"" + std::string(id.data(), id.size()) + "\"");
   MHTracer_DTPStensorflowPScompilerPSmlirPSquantizationPStensorflowPSpythonPSquantize_model_wrapperDTcc mht_6(mht_6_v, 323, "", "./tensorflow/compiler/mlir/quantization/tensorflow/python/quantize_model_wrapper.cc", "GetMaxFromCalibrator");

  absl::optional<std::pair<float, float>> min_max =
      calibrator::CalibratorSingleton::GetMinMax(id);
  if (!min_max.has_value()) {
    PyErr_Format(PyExc_ValueError, "No calibrated data for '%s'",
                 std::string{id}.c_str());
    throw py::error_already_set();
  }

  return min_max->second;
}

}  // namespace quantization
}  // namespace tensorflow
