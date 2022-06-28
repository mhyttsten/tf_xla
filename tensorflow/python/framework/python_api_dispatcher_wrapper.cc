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
class MHTracer_DTPStensorflowPSpythonPSframeworkPSpython_api_dispatcher_wrapperDTcc {
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
   MHTracer_DTPStensorflowPSpythonPSframeworkPSpython_api_dispatcher_wrapperDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSpythonPSframeworkPSpython_api_dispatcher_wrapperDTcc() {
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
// Python bindings for tensorflow/python/framework/python_api_dispatcher.h.

#include "pybind11/pybind11.h"
#include "pybind11/pytypes.h"
#include "pybind11/stl.h"
#include "tensorflow/python/framework/python_api_dispatcher.h"
#include "tensorflow/python/lib/core/safe_pyobject_ptr.h"

namespace py = pybind11;

using tensorflow::py_dispatch::PyInstanceChecker;
using tensorflow::py_dispatch::PyListChecker;
using tensorflow::py_dispatch::PySignatureChecker;
using tensorflow::py_dispatch::PythonAPIDispatcher;
using tensorflow::py_dispatch::PyTypeChecker;
using tensorflow::py_dispatch::PyUnionChecker;

namespace {

py::object Dispatch(PythonAPIDispatcher* self, py::handle args,
                    py::handle kwargs) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSpythonPSframeworkPSpython_api_dispatcher_wrapperDTcc mht_0(mht_0_v, 204, "", "./tensorflow/python/framework/python_api_dispatcher_wrapper.cc", "Dispatch");

  auto result = self->Dispatch(args.ptr(), kwargs.ptr());
  if (result == nullptr) {
    throw py::error_already_set();
  } else {
    return py::reinterpret_steal<py::object>(result.release());
  }
}

PythonAPIDispatcher MakePythonAPIDispatcher(
    const std::string& api_name, const std::vector<std::string>& arg_names,
    py::handle defaults) {
   std::vector<std::string> mht_1_v;
   mht_1_v.push_back("api_name: \"" + api_name + "\"");
   MHTracer_DTPStensorflowPSpythonPSframeworkPSpython_api_dispatcher_wrapperDTcc mht_1(mht_1_v, 219, "", "./tensorflow/python/framework/python_api_dispatcher_wrapper.cc", "MakePythonAPIDispatcher");

  std::vector<const char*> name_strs;
  name_strs.reserve(arg_names.size());
  for (const auto& name : arg_names) {
    name_strs.push_back(name.c_str());
  }
  absl::Span<const char*> arg_names_span(name_strs);
  if (defaults.ptr() == Py_None) {
    return PythonAPIDispatcher(api_name, arg_names_span, {});
  } else {
    tensorflow::Safe_PyObjectPtr fast_defaults(
        PySequence_Fast(defaults.ptr(), "defaults is not a sequence"));
    if (!fast_defaults) {
      throw py::error_already_set();
    }
    return PythonAPIDispatcher(
        api_name, arg_names_span,
        absl::MakeSpan(PySequence_Fast_ITEMS(fast_defaults.get()),
                       PySequence_Fast_GET_SIZE(fast_defaults.get())));
  }
}

}  // namespace

PYBIND11_MODULE(_pywrap_python_api_dispatcher, m) {
  py::enum_<PyTypeChecker::MatchType>(m, "MatchType")
      .value("NO_MATCH", PyTypeChecker::MatchType::NO_MATCH)
      .value("MATCH", PyTypeChecker::MatchType::MATCH)
      .value("MATCH_DISPATCHABLE", PyTypeChecker::MatchType::MATCH_DISPATCHABLE)
      .export_values();

  py::class_<PyTypeChecker, std::shared_ptr<PyTypeChecker>>(m, "PyTypeChecker")
      .def("Check", [](PyTypeChecker* self,
                       py::handle value) { return self->Check(value.ptr()); })
      .def("cost", &PyTypeChecker::cost)
      .def("cache_size",
           [](PyTypeChecker* self) {
             return static_cast<PyInstanceChecker*>(self)->cache_size();
           })
      .def("__repr__", [](PyTypeChecker* self) {
        return absl::StrCat("<PyTypeChecker ", self->DebugString(), ">");
      });

  py::class_<PySignatureChecker>(m, "PySignatureChecker")
      .def(py::init<
           std::vector<std::pair<int, std::shared_ptr<PyTypeChecker>>>>())
      .def("CheckCanonicalizedArgs",
           [](PySignatureChecker* self, py::tuple args) {
             tensorflow::Safe_PyObjectPtr seq(PySequence_Fast(args.ptr(), ""));
             PyObject** items = PySequence_Fast_ITEMS(seq.get());
             int n = PySequence_Fast_GET_SIZE(seq.get());
             return self->CheckCanonicalizedArgs(absl::MakeSpan(items, n));
           })
      .def("__repr__", [](PySignatureChecker* self) {
        return absl::StrCat("<PySignatureChecker ", self->DebugString(), ">");
      });

  py::class_<PythonAPIDispatcher>(m, "PythonAPIDispatcher")
      .def(py::init(&MakePythonAPIDispatcher))
      .def("Register",
           [](PythonAPIDispatcher* self, PySignatureChecker signature_checker,
              py::handle func) {
             return self->Register(signature_checker, func.ptr());
           })
      .def("Dispatch", &Dispatch)
      .def("Unregister",
           [](PythonAPIDispatcher* self, py::handle func) {
             return self->Unregister(func.ptr());
           })
      .def("__repr__", &PythonAPIDispatcher::DebugString);

  m.def("MakeInstanceChecker", [](py::args py_classes) {
    std::vector<PyObject*> py_classes_vector;
    py_classes_vector.reserve(py_classes.size());
    for (auto& cls : py_classes) {
      if (!PyType_Check(cls.ptr())) {
        throw py::type_error("`*py_classes` must be a tuple of types.");
      }
      py_classes_vector.push_back(cls.ptr());
    }
    return std::shared_ptr<PyTypeChecker>(
        std::make_shared<PyInstanceChecker>(py_classes_vector));
  });
  m.def("MakeListChecker", [](std::shared_ptr<PyTypeChecker> elt_type) {
    return std::shared_ptr<PyTypeChecker>(
        std::make_shared<PyListChecker>(elt_type));
  });
  m.def("MakeUnionChecker",
        [](const std::vector<std::shared_ptr<PyTypeChecker>>& options) {
          return std::shared_ptr<PyTypeChecker>(
              std::make_shared<PyUnionChecker>(options));
        });
  m.def("register_dispatchable_type", [](py::handle py_class) {
    if (!tensorflow::py_dispatch::RegisterDispatchableType(py_class.ptr())) {
      throw py::error_already_set();
    } else {
      return py_class;
    }
  });
}
