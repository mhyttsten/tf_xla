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
class MHTracer_DTPStensorflowPSpythonPSframeworkPSpython_tensor_converterDTcc {
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
   MHTracer_DTPStensorflowPSpythonPSframeworkPSpython_tensor_converterDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSpythonPSframeworkPSpython_tensor_converterDTcc() {
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
#include "tensorflow/python/framework/python_tensor_converter.h"

#include "absl/strings/str_cat.h"
#include "tensorflow/python/eager/pywrap_tensor.h"
#include "tensorflow/python/eager/pywrap_tfe.h"
#include "tensorflow/python/util/util.h"

#if PY_MAJOR_VERSION < 3
// Python 2.x:
#define PY_INT_AS_LONG(x) (PyInt_AsLong(x))
#define PY_STRING_INTERN_FROM_STRING(x) (PyString_InternFromString(x))
#else
// Python 3.x:
#define PY_INT_AS_LONG(x) (PyLong_AsLong(x))
#define PY_STRING_INTERN_FROM_STRING(x) (PyUnicode_InternFromString(x))
#endif

namespace tensorflow {
namespace {

// Returns `tensor.dtype._type_enum` as a DataType enum.  Assumes that `tensor`
// is a python `Tensor` object.
//
// On error: sets a python AttributeError exception and returns DT_INVALID.
DataType DataTypeForTensor(PyObject* tensor) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSpythonPSframeworkPSpython_tensor_converterDTcc mht_0(mht_0_v, 208, "", "./tensorflow/python/framework/python_tensor_converter.cc", "DataTypeForTensor");

  static PyObject* dtype_attr = PY_STRING_INTERN_FROM_STRING("dtype");
  static PyObject* type_enum_attr = PY_STRING_INTERN_FROM_STRING("_type_enum");

  Safe_PyObjectPtr py_dtype(PyObject_GetAttr(tensor, dtype_attr));
  if (!py_dtype) return DT_INVALID;

  Safe_PyObjectPtr enum_field(PyObject_GetAttr(py_dtype.get(), type_enum_attr));
  if (!enum_field) return DT_INVALID;

  DataType result = static_cast<DataType>(PY_INT_AS_LONG(enum_field.get()));
  return result;
}

// Check that actual_dtype == expected_dtype.  If not, set an exception and
// return false.  (If expected_dtype is DT_INVALID, then instead simply update
// its value to `actual_dtype` and return true.)
bool CheckDType(DataType actual_dtype, DataType& expected_dtype) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSpythonPSframeworkPSpython_tensor_converterDTcc mht_1(mht_1_v, 228, "", "./tensorflow/python/framework/python_tensor_converter.cc", "CheckDType");

  if (expected_dtype == DT_INVALID) {
    expected_dtype = actual_dtype;  // set output parameter.
  } else if (expected_dtype != actual_dtype) {
    PyErr_SetString(PyExc_TypeError,
                    absl::StrCat("Expected ", DataType_Name(expected_dtype),
                                 " but got ", DataType_Name(actual_dtype))
                        .c_str());
    return false;
  }
  return true;
}

}  // namespace

Safe_PyObjectPtr PythonTensorConverter::Convert(PyObject* src, DataType& dtype,
                                                bool* used_fallback) const {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSpythonPSframeworkPSpython_tensor_converterDTcc mht_2(mht_2_v, 247, "", "./tensorflow/python/framework/python_tensor_converter.cc", "PythonTensorConverter::Convert");

  // First, try converting `src` to a Tensor without calling back into Python.
  if (ctx_) {  // Eager mode
    // TODO(b/164980194): Handle resource variables as well.  (See
    // ConvertToTensor function in pywrap_tfe_src.cc).
    if (EagerTensor_CheckExact(src)) {
      // `src` is already an eager tensor; check its type, and return it as-is.
      if (!CheckDType(PyEagerTensor_Dtype(src), dtype)) return nullptr;
      Py_INCREF(src);
      return Safe_PyObjectPtr(src);
    } else {
      TFE_TensorHandle* handle =
          tensorflow::ConvertToEagerTensor(ctx_, src, dtype, device_name_);
      if (handle) {
        Safe_PyObjectPtr result(EagerTensorFromHandle(handle));
        if (!CheckDType(PyEagerTensor_Dtype(result.get()), dtype)) {
          return nullptr;
        }
        return result;
      } else {
        PyErr_Clear();
      }
    }
  } else {  // Graph mode
    if (swig::IsTensor(src)) {
      DataType src_dtype = DataTypeForTensor(src);
      if (src_dtype == DT_INVALID) return nullptr;
      if (!CheckDType(src_dtype, dtype)) return nullptr;
      Py_INCREF(src);
      return Safe_PyObjectPtr(src);
    }
  }

  // Fallback: use the Python tf.convert_to_tensor function.
  // Currently this is used:
  //
  // * In Eager mode: for anything that's not already an Eager tensor, or
  //   handled by `tensorflow::ConvertToEagerTensor`.  (At time of writing
  //   for this comment, ConvertToEagerTensor handles simple values like ints,
  //   nested lists of simple values, and numpy arrays.)
  // * In graph mode: for anything that's not already a tensor.
  //
  // TODO(b/164980194) Reduce/eliminate cases where fallback is used.
  if (used_fallback) *used_fallback = true;
  static PyObject* convert_to_tensor =
      swig::GetRegisteredPyObject("tf.convert_to_tensor");
  if (!convert_to_tensor) return nullptr;

  Safe_PyObjectPtr args(PyTuple_New(dtype == DT_INVALID ? 1 : 2));
  Safe_PyObjectPtr kwargs(PyDict_New());
  Py_INCREF(src);
  PyTuple_SetItem(args.get(), 0, src);
  if (dtype != DT_INVALID) {
    PyTuple_SetItem(args.get(), 1, PyLong_FromLong(dtype));
  }
  PyDict_SetItemString(kwargs.get(), "ctx", py_eager_context_);
  Safe_PyObjectPtr result(
      PyObject_Call(convert_to_tensor, args.get(), kwargs.get()));
  if (!result) return nullptr;
  dtype = DataTypeForTensor(result.get());  // set output parameter.
  if (dtype == DT_INVALID) return nullptr;
  return result;
}

}  // namespace tensorflow
