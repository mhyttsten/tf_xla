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
class MHTracer_DTPStensorflowPSpythonPSlibPScorePSpy_utilDTcc {
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
   MHTracer_DTPStensorflowPSpythonPSlibPScorePSpy_utilDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSpythonPSlibPScorePSpy_utilDTcc() {
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

/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/python/lib/core/py_util.h"

// Place `<locale>` before <Python.h> to avoid build failure in macOS.
#include <locale>

#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/strings/strcat.h"

namespace tensorflow {
namespace {

// py.__class__.__name__
const char* ClassName(PyObject* py) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSpythonPSlibPScorePSpy_utilDTcc mht_0(mht_0_v, 197, "", "./tensorflow/python/lib/core/py_util.cc", "ClassName");

/* PyPy doesn't have a separate C API for old-style classes. */
#if PY_MAJOR_VERSION < 3 && !defined(PYPY_VERSION)
  if (PyClass_Check(py))
    return PyString_AS_STRING(
        CHECK_NOTNULL(reinterpret_cast<PyClassObject*>(py)->cl_name));
  if (PyInstance_Check(py))
    return PyString_AS_STRING(CHECK_NOTNULL(
        reinterpret_cast<PyInstanceObject*>(py)->in_class->cl_name));
#endif
  if (Py_TYPE(py) == &PyType_Type) {
    return reinterpret_cast<PyTypeObject*>(py)->tp_name;
  }
  return Py_TYPE(py)->tp_name;
}

}  // end namespace

// Returns a PyObject containing a string, or null
void TryAppendTraceback(PyObject* ptype, PyObject* pvalue, PyObject* ptraceback,
                        string* out) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSpythonPSlibPScorePSpy_utilDTcc mht_1(mht_1_v, 220, "", "./tensorflow/python/lib/core/py_util.cc", "TryAppendTraceback");

  // The "traceback" module is assumed to be imported already by script_ops.py.
  PyObject* tb_module = PyImport_AddModule("traceback");

  if (!tb_module) {
    return;
  }

  PyObject* format_exception =
      PyObject_GetAttrString(tb_module, "format_exception");

  if (!format_exception) {
    return;
  }

  if (!PyCallable_Check(format_exception)) {
    Py_DECREF(format_exception);
    return;
  }

  PyObject* ret_val = PyObject_CallFunctionObjArgs(format_exception, ptype,
                                                   pvalue, ptraceback, nullptr);
  Py_DECREF(format_exception);

  if (!ret_val) {
    return;
  }

  if (!PyList_Check(ret_val)) {
    Py_DECREF(ret_val);
    return;
  }

  Py_ssize_t n = PyList_GET_SIZE(ret_val);
  for (Py_ssize_t i = 0; i < n; ++i) {
    PyObject* v = PyList_GET_ITEM(ret_val, i);
#if PY_MAJOR_VERSION < 3
    strings::StrAppend(out, PyString_AS_STRING(v), "\n");
#else
    strings::StrAppend(out, PyUnicode_AsUTF8(v), "\n");
#endif
  }

  // Iterate through ret_val.
  Py_DECREF(ret_val);
}

string PyExceptionFetch() {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSpythonPSlibPScorePSpy_utilDTcc mht_2(mht_2_v, 270, "", "./tensorflow/python/lib/core/py_util.cc", "PyExceptionFetch");

  CHECK(PyErr_Occurred())
      << "Must only call PyExceptionFetch after an exception.";
  PyObject* ptype;
  PyObject* pvalue;
  PyObject* ptraceback;
  PyErr_Fetch(&ptype, &pvalue, &ptraceback);
  PyErr_NormalizeException(&ptype, &pvalue, &ptraceback);
  string err = ClassName(ptype);
  if (pvalue) {
    PyObject* str = PyObject_Str(pvalue);

    if (str) {
#if PY_MAJOR_VERSION < 3
      strings::StrAppend(&err, ": ", PyString_AS_STRING(str), "\n");
#else
      strings::StrAppend(&err, ": ", PyUnicode_AsUTF8(str), "\n");
#endif
      Py_DECREF(str);
    } else {
      strings::StrAppend(&err, "(unknown error message)\n");
    }

    TryAppendTraceback(ptype, pvalue, ptraceback, &err);

    Py_DECREF(pvalue);
  }
  Py_DECREF(ptype);
  Py_XDECREF(ptraceback);
  return err;
}

}  // end namespace tensorflow
