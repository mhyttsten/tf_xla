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
class MHTracer_DTPStensorflowPSpythonPSutilPSfast_module_typeDTcc {
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
   MHTracer_DTPStensorflowPSpythonPSutilPSfast_module_typeDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSpythonPSutilPSfast_module_typeDTcc() {
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
// clang-format off
// These headers must be at the top, before including Python.h header
// Otherwise, we get C2039 on MSVC due to 'copysign'
#include "pybind11/complex.h"
#include "pybind11/pybind11.h"
// clang-format on

#include "Python.h"
#include "absl/container/flat_hash_map.h"
#include "tensorflow/core/platform/logging.h"

namespace py = pybind11;
constexpr int PY_MODULE_TYPE_TP_BASIC_SIZE = 56;

struct FastModuleObject {
  // A dummy array that ensures enough size is reserved for FastModuleObject,
  // because it's inherited from PyModuleObject.
  const std::array<char, PY_MODULE_TYPE_TP_BASIC_SIZE> opaque_base_fields;
  // A cache that helps reduce attribute lookup overhead.
  absl::flat_hash_map<PyObject *, PyObject *> attr_map;
  // pointer to the external getattribute function
  PyObject *cb_getattribute = nullptr;
  // pointer to the external getattr function
  PyObject *cb_getattr = nullptr;
  // static PyTypeObject type;

  FastModuleObject() = delete;
  ~FastModuleObject() = delete;
  static FastModuleObject *UncheckedCast(PyObject *obj);
};

static int FastModule_init(FastModuleObject *self, PyObject *args,
                           PyObject *kwds) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSpythonPSutilPSfast_module_typeDTcc mht_0(mht_0_v, 216, "", "./tensorflow/python/util/fast_module_type.cc", "FastModule_init");

  DCHECK_EQ(PY_MODULE_TYPE_TP_BASIC_SIZE, PyModule_Type.tp_basicsize);
  if (PyModule_Type.tp_init(reinterpret_cast<PyObject *>(self), args, kwds) < 0)
    return -1;
  new (&(self->attr_map)) absl::flat_hash_map<PyObject *, PyObject *>();
  return 0;
}

// Parses the input as a callable and checks the result.
static PyObject *ParseFunc(PyObject *args) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSpythonPSutilPSfast_module_typeDTcc mht_1(mht_1_v, 228, "", "./tensorflow/python/util/fast_module_type.cc", "ParseFunc");

  PyObject *func;
  if (!PyArg_ParseTuple(args, "O:set_callback", &func)) return nullptr;
  if (!PyCallable_Check(func)) {
    PyErr_SetString(PyExc_TypeError, "input args must be callable");
    return nullptr;
  }
  Py_INCREF(func);  // Add a reference to new callback
  return func;
}

// Sets the pointer 'cb_getattribute' in the FastModuleObject object
// corresponding to 'self'.
static PyObject *SetGetattributeCallback(PyObject *self, PyObject *args) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSpythonPSutilPSfast_module_typeDTcc mht_2(mht_2_v, 244, "", "./tensorflow/python/util/fast_module_type.cc", "SetGetattributeCallback");

  PyObject *func = ParseFunc(args);
  // Dispose of previous callback
  Py_XDECREF(FastModuleObject::UncheckedCast(self)->cb_getattribute);
  // Remember new callback
  FastModuleObject::UncheckedCast(self)->cb_getattribute = func;
  Py_RETURN_NONE;
}

// Sets the pointer 'cb_getattr' in the FastModuleObject object
// corresponding to 'self'.
static PyObject *SetGetattrCallback(PyObject *self, PyObject *args) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPSpythonPSutilPSfast_module_typeDTcc mht_3(mht_3_v, 258, "", "./tensorflow/python/util/fast_module_type.cc", "SetGetattrCallback");

  PyObject *func = ParseFunc(args);
  // Dispose of previous callback
  Py_XDECREF(FastModuleObject::UncheckedCast(self)->cb_getattr);
  // Remember new callback
  FastModuleObject::UncheckedCast(self)->cb_getattr = func;
  Py_RETURN_NONE;
}

// Inserts or updates a key-value pair in the cache 'attr_map'
// of the FastModuleObject object corresponding to 'self'.
static PyObject *FastDictInsert(FastModuleObject *self, PyObject *args) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPSpythonPSutilPSfast_module_typeDTcc mht_4(mht_4_v, 272, "", "./tensorflow/python/util/fast_module_type.cc", "FastDictInsert");

  PyObject *name, *value;
  if (!PyArg_ParseTuple(args, "OO", &name, &value)) {
    PyErr_SetString(PyExc_TypeError, "_fastdict_insert: incorrect inputs");
    return nullptr;
  }
  auto &attr_map = self->attr_map;
  if (attr_map.find(name) != attr_map.end()) {
    Py_DECREF(name);
    Py_DECREF(value);
  }
  attr_map.insert_or_assign(name, value);
  // Increment the reference count
  Py_INCREF(name);
  Py_INCREF(value);
  // Properly handle returning Py_None
  Py_RETURN_NONE;
}

// Gets a value from a key in the cache 'attr_map'
// of the FastModuleObject object corresponding to 'self'.
static PyObject *FastDictGet(FastModuleObject *self, PyObject *args) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPSpythonPSutilPSfast_module_typeDTcc mht_5(mht_5_v, 296, "", "./tensorflow/python/util/fast_module_type.cc", "FastDictGet");

  PyObject *name;
  if (!PyArg_ParseTuple(args, "O", &name)) {
    PyErr_SetString(PyExc_TypeError, "_fastdict_get: incorrect inputs");
    return nullptr;
  }
  auto &attr_map = self->attr_map;
  auto result = attr_map.find(name);
  if (result != attr_map.end()) {
    PyObject *value = result->second;
    Py_INCREF(value);
    return value;
  }
  // Copied from CPython's moduleobject.c
  PyErr_Format(PyExc_KeyError, "module has no attribute '%U'", name);
  return nullptr;
}

// Returns true if a key exists in the cache 'attr_map'
// of the FastModuleObject object corresponding to 'self',
// otherwise returns false.
static PyObject *FastDictContains(FastModuleObject *self, PyObject *args) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPSpythonPSutilPSfast_module_typeDTcc mht_6(mht_6_v, 320, "", "./tensorflow/python/util/fast_module_type.cc", "FastDictContains");

  PyObject *name;
  if (!PyArg_ParseTuple(args, "O", &name)) {
    PyErr_SetString(PyExc_TypeError, "_fastdict_key_in: incorrect inputs");
    return nullptr;
  }
  const auto &attr_map = self->attr_map;
  const auto result = attr_map.contains(name);
  if (result) {
    // Properly handle returning Py_True
    Py_RETURN_TRUE;
  }
  // Properly handle returning Py_False
  Py_RETURN_FALSE;
}

// Calls a function 'func' with inputs 'self' and 'args'.
static PyObject *CallFunc(FastModuleObject *self, PyObject *args,
                          PyObject *func) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPSpythonPSutilPSfast_module_typeDTcc mht_7(mht_7_v, 341, "", "./tensorflow/python/util/fast_module_type.cc", "CallFunc");

  if (func == nullptr) {
    PyErr_SetString(PyExc_NameError,
                    "Attempting to call a callback that was not defined");
    return nullptr;
  }
  PyObject *name;
  if (!PyArg_ParseTuple(args, "O", &name)) {
    PyErr_SetString(PyExc_TypeError, "CallFunc: incorrect inputs");
    return nullptr;
  }
  PyObject *arglist = Py_BuildValue("(OO)", self, name);
  auto result = PyObject_CallObject(func, arglist);
  Py_DECREF(arglist);
  return result;
}

static PyMethodDef FastModule_methods[] = {
    {"_fastdict_insert", reinterpret_cast<PyCFunction>(FastDictInsert),
     METH_VARARGS, "Registers a method to the fast lookup table."},
    {"_fastdict_get", reinterpret_cast<PyCFunction>(FastDictGet), METH_VARARGS,
     "Gets a method from the fast lookup table."},
    {"_fastdict_key_in", reinterpret_cast<PyCFunction>(FastDictContains),
     METH_VARARGS, "Checks if a method exists in the fast lookup table."},
    {"set_getattribute_callback", SetGetattributeCallback, METH_VARARGS,
     "Defines the callback function to replace __getattribute__"},
    {"set_getattr_callback", SetGetattrCallback, METH_VARARGS,
     "Defines the callback function to replace __getattr__"},
    {nullptr, nullptr, 0, nullptr},
};

// Attempts to get the attribute based on 'name' as the key in cache 'attr_map'
// of the FastModuleObject object corresponding to 'module'.
// If the lookup fails in the cache, either uses
// a user-defined callback 'cb_getattribute'
// or the default 'tp_getattro' function to look for the attribute.
static PyObject *FastTpGetattro(PyObject *module, PyObject *name) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPSpythonPSutilPSfast_module_typeDTcc mht_8(mht_8_v, 380, "", "./tensorflow/python/util/fast_module_type.cc", "FastTpGetattro");

  FastModuleObject *fast_module = FastModuleObject::UncheckedCast(module);
  auto &attr_map = fast_module->attr_map;
  auto it = attr_map.find(name);
  // If the attribute lookup is successful in the cache, directly return it.
  if (it != attr_map.end()) {
    PyObject *value = it->second;
    Py_INCREF(value);
    return value;
  }
  PyObject *arglist = Py_BuildValue("(O)", name);
  PyObject *result;
  // Prefer the customized callback function over the default function.
  if (fast_module->cb_getattribute != nullptr) {
    result = CallFunc(fast_module, arglist, fast_module->cb_getattribute);
  } else {
    result = PyModule_Type.tp_getattro(module, name);
  }
  // Return result if it's found
  if (result != nullptr) {
    return result;
  }
  // If the default lookup fails and an AttributeError is raised,
  // clear the error status before using the __getattr__ callback function.
  auto is_error = PyErr_Occurred();
  if (is_error && PyErr_ExceptionMatches(PyExc_AttributeError) &&
      fast_module->cb_getattr != nullptr) {
    PyErr_Clear();
    return CallFunc(fast_module, arglist, fast_module->cb_getattr);
  }
  // If all options were used up
  return result;
}

// Customized destructor for FastModuleType.tp_dealloc
// In addition to default behavior it also clears up the contents in attr_map.
static void FastModuleObjectDealloc(PyObject *module) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPSpythonPSutilPSfast_module_typeDTcc mht_9(mht_9_v, 419, "", "./tensorflow/python/util/fast_module_type.cc", "FastModuleObjectDealloc");

  auto &attr_map = FastModuleObject::UncheckedCast(module)->attr_map;
  for (auto &it : attr_map) {
    Py_DECREF(it.first);
    Py_DECREF(it.second);
  }
  attr_map.~flat_hash_map<PyObject *, PyObject *>();
  Py_TYPE(module)->tp_free(module);
}

static PyTypeObject FastModuleType = []() {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPSpythonPSutilPSfast_module_typeDTcc mht_10(mht_10_v, 432, "", "./tensorflow/python/util/fast_module_type.cc", "lambda");

  PyTypeObject obj = {PyVarObject_HEAD_INIT(&PyType_Type, 0)};
  obj.tp_name = "fast_module_type.FastModuleType";
  obj.tp_basicsize = sizeof(FastModuleObject);
  obj.tp_itemsize = 0;
  obj.tp_dealloc = FastModuleObjectDealloc;
  obj.tp_getattro = FastTpGetattro;
  obj.tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE;
  obj.tp_doc = "FastModuleType objects";
  obj.tp_methods = FastModule_methods;
  obj.tp_init = reinterpret_cast<initproc>(FastModule_init);
  return obj;
}();

// Returns true if the type of 'obj' or any of its parent class
// is equal to 'target'. Otherwise returns false.
bool IsAnyBaseSameType(const PyObject *obj, const PyTypeObject *target) {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPSpythonPSutilPSfast_module_typeDTcc mht_11(mht_11_v, 451, "", "./tensorflow/python/util/fast_module_type.cc", "IsAnyBaseSameType");

  auto *tp = Py_TYPE(obj);
  while (true) {
    if (tp == target) return true;
    // If the default type is found, there is no need to search further
    if (tp == &PyBaseObject_Type) break;
    tp = tp->tp_base;
  }
  return false;
}

// Casts 'obj' to 'FastModuleObject *'.
// Conducts a check only in non-optimized builds.
FastModuleObject *FastModuleObject::UncheckedCast(PyObject *obj) {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPSpythonPSutilPSfast_module_typeDTcc mht_12(mht_12_v, 467, "", "./tensorflow/python/util/fast_module_type.cc", "FastModuleObject::UncheckedCast");

  DCHECK(IsAnyBaseSameType(obj, &FastModuleType));
  return reinterpret_cast<FastModuleObject *>(obj);
}

PYBIND11_MODULE(fast_module_type, m) {
  FastModuleType.tp_base = &PyModule_Type;
  FastModuleType.tp_setattro = [](PyObject *module, PyObject *name,
                                  PyObject *value) -> int {
    auto &attr_map = FastModuleObject::UncheckedCast(module)->attr_map;
    if (attr_map.find(name) != attr_map.end()) {
      Py_DECREF(name);
      Py_DECREF(value);
    }
    attr_map.insert_or_assign(name, value);
    // Increment the reference count
    Py_INCREF(name);
    Py_INCREF(value);
    PyObject_GenericSetAttr(module, name, value);
    return 0;
  };

  m.doc() = R"pbdoc(
    fast_module_type
    -----
  )pbdoc";
  // Use getter function to hold attributes rather than pybind11's m.attr due to
  // b/145559202.
  m.def(
      "get_fast_module_type_class",
      []() {
        return py::cast<py::object>(
            reinterpret_cast<PyObject *>(&FastModuleType));
      },
      py::return_value_policy::reference);
}
