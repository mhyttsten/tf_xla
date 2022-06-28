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
class MHTracer_DTPStensorflowPSpythonPSframeworkPSpython_tensor_converter_wrapperDTcc {
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
   MHTracer_DTPStensorflowPSpythonPSframeworkPSpython_tensor_converter_wrapperDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSpythonPSframeworkPSpython_tensor_converter_wrapperDTcc() {
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
// Note: This library is only used by python_tensor_converter_test.  It is
// not meant to be used in other circumstances.

#include "pybind11/pybind11.h"
#include "pybind11/pytypes.h"
#include "pybind11/stl.h"
#include "tensorflow/python/eager/pywrap_tfe.h"
#include "tensorflow/python/framework/python_tensor_converter.h"

#if PY_MAJOR_VERSION < 3
// Python 2.x:
#define PY_STRING_INTERN_FROM_STRING(x) (PyString_InternFromString(x))
#define PY_INT_AS_LONG(x) (PyInt_AsLong(x))
#define PY_INT_FROM_LONG(x) (PyInt_FromLong(x))
#else
// Python 3.x:
#define PY_INT_AS_LONG(x) (PyLong_AsLong(x))
#define PY_STRING_INTERN_FROM_STRING(x) (PyUnicode_InternFromString(x))
#define PY_INT_FROM_LONG(x) (PyLong_FromLong(x))
#endif

namespace py = pybind11;

namespace tensorflow {
namespace {

Safe_PyObjectPtr GetAttr_ThreadLocalData(PyObject* eager_context) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSpythonPSframeworkPSpython_tensor_converter_wrapperDTcc mht_0(mht_0_v, 210, "", "./tensorflow/python/framework/python_tensor_converter_wrapper.cc", "GetAttr_ThreadLocalData");

  static PyObject* attr = PY_STRING_INTERN_FROM_STRING("_thread_local_data");
  return Safe_PyObjectPtr(PyObject_GetAttr(eager_context, attr));
}

Safe_PyObjectPtr GetAttr_ContextHandle(PyObject* eager_context) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSpythonPSframeworkPSpython_tensor_converter_wrapperDTcc mht_1(mht_1_v, 218, "", "./tensorflow/python/framework/python_tensor_converter_wrapper.cc", "GetAttr_ContextHandle");

  static PyObject* attr = PY_STRING_INTERN_FROM_STRING("_context_handle");
  return Safe_PyObjectPtr(PyObject_GetAttr(eager_context, attr));
}

Safe_PyObjectPtr GetAttr_IsEager(PyObject* tld) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSpythonPSframeworkPSpython_tensor_converter_wrapperDTcc mht_2(mht_2_v, 226, "", "./tensorflow/python/framework/python_tensor_converter_wrapper.cc", "GetAttr_IsEager");

  static PyObject* attr = PY_STRING_INTERN_FROM_STRING("is_eager");
  return Safe_PyObjectPtr(PyObject_GetAttr(tld, attr));
}

Safe_PyObjectPtr GetAttr_DeviceName(PyObject* tld) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPSpythonPSframeworkPSpython_tensor_converter_wrapperDTcc mht_3(mht_3_v, 234, "", "./tensorflow/python/framework/python_tensor_converter_wrapper.cc", "GetAttr_DeviceName");

  static PyObject* attr = PY_STRING_INTERN_FROM_STRING("device_name");
  return Safe_PyObjectPtr(PyObject_GetAttr(tld, attr));
}

Safe_PyObjectPtr GetAttr_TypeEnum(PyObject* dtype) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPSpythonPSframeworkPSpython_tensor_converter_wrapperDTcc mht_4(mht_4_v, 242, "", "./tensorflow/python/framework/python_tensor_converter_wrapper.cc", "GetAttr_TypeEnum");

  static PyObject* attr = PY_STRING_INTERN_FROM_STRING("_type_enum");
  return Safe_PyObjectPtr(PyObject_GetAttr(dtype, attr));
}

PythonTensorConverter MakePythonTensorConverter(py::handle py_eager_context) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPSpythonPSframeworkPSpython_tensor_converter_wrapperDTcc mht_5(mht_5_v, 250, "", "./tensorflow/python/framework/python_tensor_converter_wrapper.cc", "MakePythonTensorConverter");

  Safe_PyObjectPtr tld = GetAttr_ThreadLocalData(py_eager_context.ptr());
  if (!tld) throw py::error_already_set();

  Safe_PyObjectPtr py_is_eager = GetAttr_IsEager(tld.get());
  if (!py_is_eager) throw py::error_already_set();
  bool is_eager = PyObject_IsTrue(py_is_eager.get());

  // Initialize the eager context, if necessary.
  TFE_Context* ctx = nullptr;
  const char* device_name = nullptr;
  if (is_eager) {
    Safe_PyObjectPtr context_handle =
        GetAttr_ContextHandle(py_eager_context.ptr());
    if (!context_handle) throw py::error_already_set();
    if (context_handle.get() == Py_None) {
      throw std::runtime_error("Error retrieving context handle.");
    }
    Safe_PyObjectPtr py_device_name = GetAttr_DeviceName(tld.get());
    if (!py_device_name) {
      throw std::runtime_error("Error retrieving device name.");
    }
    device_name = TFE_GetPythonString(py_device_name.get());
    ctx = reinterpret_cast<TFE_Context*>(
        PyCapsule_GetPointer(context_handle.get(), nullptr));
  }

  return PythonTensorConverter(py_eager_context.ptr(), ctx, device_name);
}

py::handle Convert(tensorflow::PythonTensorConverter* self, py::handle obj,
                   py::handle dtype) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPSpythonPSframeworkPSpython_tensor_converter_wrapperDTcc mht_6(mht_6_v, 284, "", "./tensorflow/python/framework/python_tensor_converter_wrapper.cc", "Convert");

  DataType dtype_enum = static_cast<DataType>(PY_INT_AS_LONG(dtype.ptr()));
  bool used_fallback = false;
  Safe_PyObjectPtr converted =
      self->Convert(obj.ptr(), dtype_enum, &used_fallback);
  if (!converted) throw py::error_already_set();

  PyObject* result = PyTuple_New(3);
  PyTuple_SET_ITEM(result, 0, converted.release());
  PyTuple_SET_ITEM(result, 1, PY_INT_FROM_LONG(dtype_enum));
  PyTuple_SET_ITEM(result, 2, used_fallback ? Py_True : Py_False);
  Py_INCREF(PyTuple_GET_ITEM(result, 1));
  Py_INCREF(PyTuple_GET_ITEM(result, 2));
  return result;
}

}  // namespace
}  // namespace tensorflow

PYBIND11_MODULE(_pywrap_python_tensor_converter, m) {
  py::class_<tensorflow::PythonTensorConverter>(m, "PythonTensorConverter")
      .def(py::init(&tensorflow::MakePythonTensorConverter))
      .def("Convert", tensorflow::Convert);
}
