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
class MHTracer_DTPStensorflowPSpythonPSframeworkPSpython_api_info_wrapperDTcc {
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
   MHTracer_DTPStensorflowPSpythonPSframeworkPSpython_api_info_wrapperDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSpythonPSframeworkPSpython_api_info_wrapperDTcc() {
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

/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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
// Note: This library is only used by python_api_info_test.  It
// is not meant to be used in other circumstances.

#include "pybind11/pybind11.h"
#include "pybind11/pytypes.h"
#include "pybind11/stl.h"
#include "tensorflow/python/framework/python_api_info.h"

namespace py = pybind11;

namespace tensorflow {
namespace {

void InitializeFromRegisteredOp(PythonAPIInfo* api_info,
                                const std::string& op_name) {
   std::vector<std::string> mht_0_v;
   mht_0_v.push_back("op_name: \"" + op_name + "\"");
   MHTracer_DTPStensorflowPSpythonPSframeworkPSpython_api_info_wrapperDTcc mht_0(mht_0_v, 199, "", "./tensorflow/python/framework/python_api_info_wrapper.cc", "InitializeFromRegisteredOp");

  auto result = api_info->InitializeFromRegisteredOp(op_name);
  if (!result.ok()) {
    PyErr_SetString(PyExc_ValueError, result.ToString().c_str());
    throw py::error_already_set();
  }
}

void InitializeFromParamSpecs(
    PythonAPIInfo* api_info,
    const std::map<std::string, std::string>& input_specs,
    const std::map<std::string, std::string>& attr_specs,
    const std::vector<string>& param_names, py::handle defaults_tuple) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSpythonPSframeworkPSpython_api_info_wrapperDTcc mht_1(mht_1_v, 214, "", "./tensorflow/python/framework/python_api_info_wrapper.cc", "InitializeFromParamSpecs");

  auto result = api_info->InitializeFromParamSpecs(
      input_specs, attr_specs, param_names, defaults_tuple.ptr());
  if (!result.ok()) {
    PyErr_SetString(PyExc_ValueError, result.ToString().c_str());
    throw py::error_already_set();
  }
}

std::string DebugInfo(PythonAPIInfo* api_info) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSpythonPSframeworkPSpython_api_info_wrapperDTcc mht_2(mht_2_v, 226, "", "./tensorflow/python/framework/python_api_info_wrapper.cc", "DebugInfo");
 return api_info->DebugInfo(); }

}  // namespace
}  // namespace tensorflow

using PythonAPIInfo = tensorflow::PythonAPIInfo;
using InferredAttributes = tensorflow::PythonAPIInfo::InferredAttributes;

PYBIND11_MODULE(_pywrap_python_api_info, m) {
  py::class_<PythonAPIInfo>(m, "PythonAPIInfo")
      .def(py::init<const std::string&>())
      .def("InitializeFromRegisteredOp",
           &tensorflow::InitializeFromRegisteredOp)
      .def("InitializeFromParamSpecs", &tensorflow::InitializeFromParamSpecs)
      .def("DebugInfo", &tensorflow::DebugInfo)
      .def("InferredTypeAttrs",
           [](PythonAPIInfo* self) { return self->inferred_type_attrs(); })
      .def("InferredTypeListAttrs",
           [](PythonAPIInfo* self) { return self->inferred_type_list_attrs(); })
      .def("InferredLengthAttrs",
           [](PythonAPIInfo* self) { return self->inferred_length_attrs(); });
  py::class_<InferredAttributes>(m, "InferredAttributes")
      .def_readonly("types", &InferredAttributes::types)
      .def_readonly("type_lists", &InferredAttributes::type_lists)
      .def_readonly("lengths", &InferredAttributes::lengths);
}
