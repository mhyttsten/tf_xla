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
class MHTracer_DTPStensorflowPSlitePSpythonPSinterpreter_wrapperPSnumpyDTcc {
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
   MHTracer_DTPStensorflowPSlitePSpythonPSinterpreter_wrapperPSnumpyDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePSpythonPSinterpreter_wrapperPSnumpyDTcc() {
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

#define TFLITE_IMPORT_NUMPY  // See numpy.h for explanation.
#include "tensorflow/lite/python/interpreter_wrapper/numpy.h"

#include <memory>

namespace tflite {
namespace python {

void ImportNumpy() {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePSpythonPSinterpreter_wrapperPSnumpyDTcc mht_0(mht_0_v, 193, "", "./tensorflow/lite/python/interpreter_wrapper/numpy.cc", "ImportNumpy");
 import_array1(); }

}  // namespace python

namespace python_utils {

struct PyObjectDereferencer {
  void operator()(PyObject* py_object) const { Py_DECREF(py_object); }
};
using UniquePyObjectRef = std::unique_ptr<PyObject, PyObjectDereferencer>;

int TfLiteTypeToPyArrayType(TfLiteType tf_lite_type) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSlitePSpythonPSinterpreter_wrapperPSnumpyDTcc mht_1(mht_1_v, 207, "", "./tensorflow/lite/python/interpreter_wrapper/numpy.cc", "TfLiteTypeToPyArrayType");

  switch (tf_lite_type) {
    case kTfLiteFloat32:
      return NPY_FLOAT32;
    case kTfLiteFloat16:
      return NPY_FLOAT16;
    case kTfLiteFloat64:
      return NPY_FLOAT64;
    case kTfLiteInt32:
      return NPY_INT32;
    case kTfLiteUInt32:
      return NPY_UINT32;
    case kTfLiteUInt16:
      return NPY_UINT16;
    case kTfLiteInt16:
      return NPY_INT16;
    case kTfLiteUInt8:
      return NPY_UINT8;
    case kTfLiteInt8:
      return NPY_INT8;
    case kTfLiteInt64:
      return NPY_INT64;
    case kTfLiteUInt64:
      return NPY_UINT64;
    case kTfLiteString:
      return NPY_STRING;
    case kTfLiteBool:
      return NPY_BOOL;
    case kTfLiteComplex64:
      return NPY_COMPLEX64;
    case kTfLiteComplex128:
      return NPY_COMPLEX128;
    case kTfLiteResource:
    case kTfLiteVariant:
      return NPY_OBJECT;
    case kTfLiteNoType:
      return NPY_NOTYPE;
      // Avoid default so compiler errors created when new types are made.
  }
  return NPY_NOTYPE;
}

TfLiteType TfLiteTypeFromPyType(int py_type) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSlitePSpythonPSinterpreter_wrapperPSnumpyDTcc mht_2(mht_2_v, 252, "", "./tensorflow/lite/python/interpreter_wrapper/numpy.cc", "TfLiteTypeFromPyType");

  switch (py_type) {
    case NPY_FLOAT32:
      return kTfLiteFloat32;
    case NPY_FLOAT16:
      return kTfLiteFloat16;
    case NPY_FLOAT64:
      return kTfLiteFloat64;
    case NPY_INT32:
      return kTfLiteInt32;
    case NPY_UINT32:
      return kTfLiteUInt32;
    case NPY_INT16:
      return kTfLiteInt16;
    case NPY_UINT8:
      return kTfLiteUInt8;
    case NPY_INT8:
      return kTfLiteInt8;
    case NPY_INT64:
      return kTfLiteInt64;
    case NPY_UINT64:
      return kTfLiteUInt64;
    case NPY_BOOL:
      return kTfLiteBool;
    case NPY_OBJECT:
    case NPY_STRING:
    case NPY_UNICODE:
      return kTfLiteString;
    case NPY_COMPLEX64:
      return kTfLiteComplex64;
    case NPY_COMPLEX128:
      return kTfLiteComplex128;
  }
  return kTfLiteNoType;
}

TfLiteType TfLiteTypeFromPyArray(PyArrayObject* array) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPSlitePSpythonPSinterpreter_wrapperPSnumpyDTcc mht_3(mht_3_v, 291, "", "./tensorflow/lite/python/interpreter_wrapper/numpy.cc", "TfLiteTypeFromPyArray");

  int pyarray_type = PyArray_TYPE(array);
  return TfLiteTypeFromPyType(pyarray_type);
}

#if PY_VERSION_HEX >= 0x03030000
bool FillStringBufferFromPyUnicode(PyObject* value,
                                   DynamicBuffer* dynamic_buffer) {
  Py_ssize_t len = -1;
  const char* buf = PyUnicode_AsUTF8AndSize(value, &len);
  if (buf == nullptr) {
    PyErr_SetString(PyExc_ValueError, "PyUnicode_AsUTF8AndSize() failed.");
    return false;
  }
  dynamic_buffer->AddString(buf, len);
  return true;
}
#else
bool FillStringBufferFromPyUnicode(PyObject* value,
                                   DynamicBuffer* dynamic_buffer) {
  UniquePyObjectRef utemp(PyUnicode_AsUTF8String(value));
  if (!utemp) {
    PyErr_SetString(PyExc_ValueError, "PyUnicode_AsUTF8String() failed.");
    return false;
  }
  char* buf = nullptr;
  Py_ssize_t len = -1;
  if (PyBytes_AsStringAndSize(utemp.get(), &buf, &len) == -1) {
    PyErr_SetString(PyExc_ValueError, "PyBytes_AsStringAndSize() failed.");
    return false;
  }
  dynamic_buffer->AddString(buf, len);
  return true;
}
#endif

bool FillStringBufferFromPyString(PyObject* value,
                                  DynamicBuffer* dynamic_buffer) {
  if (PyUnicode_Check(value)) {
    return FillStringBufferFromPyUnicode(value, dynamic_buffer);
  }

  char* buf = nullptr;
  Py_ssize_t len = -1;
  if (PyBytes_AsStringAndSize(value, &buf, &len) == -1) {
    PyErr_SetString(PyExc_ValueError, "PyBytes_AsStringAndSize() failed.");
    return false;
  }
  dynamic_buffer->AddString(buf, len);
  return true;
}

bool FillStringBufferWithPyArray(PyObject* value,
                                 DynamicBuffer* dynamic_buffer) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPSlitePSpythonPSinterpreter_wrapperPSnumpyDTcc mht_4(mht_4_v, 347, "", "./tensorflow/lite/python/interpreter_wrapper/numpy.cc", "FillStringBufferWithPyArray");

  PyArrayObject* array = reinterpret_cast<PyArrayObject*>(value);
  switch (PyArray_TYPE(array)) {
    case NPY_OBJECT:
    case NPY_STRING:
    case NPY_UNICODE: {
      if (PyArray_NDIM(array) == 0) {
        dynamic_buffer->AddString(static_cast<char*>(PyArray_DATA(array)),
                                  PyArray_NBYTES(array));
        return true;
      }
      UniquePyObjectRef iter(PyArray_IterNew(value));
      while (PyArray_ITER_NOTDONE(iter.get())) {
        UniquePyObjectRef item(PyArray_GETITEM(
            array, reinterpret_cast<char*>(PyArray_ITER_DATA(iter.get()))));

        if (!FillStringBufferFromPyString(item.get(), dynamic_buffer)) {
          return false;
        }

        PyArray_ITER_NEXT(iter.get());
      }
      return true;
    }
    default:
      break;
  }

  PyErr_Format(PyExc_ValueError,
               "Cannot use numpy array of type %d for string tensor.",
               PyArray_TYPE(array));
  return false;
}

}  // namespace python_utils
}  // namespace tflite
