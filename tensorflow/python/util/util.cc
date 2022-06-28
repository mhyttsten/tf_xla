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
class MHTracer_DTPStensorflowPSpythonPSutilPSutilDTcc {
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
   MHTracer_DTPStensorflowPSpythonPSutilPSutilDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSpythonPSutilPSutilDTcc() {
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

/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/python/util/util.h"

#include <functional>
#include <memory>
#include <unordered_map>
#include <vector>

#include "absl/memory/memory.h"
#include "tensorflow/core/lib/gtl/map_util.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/python/lib/core/safe_pyobject_ptr.h"

namespace tensorflow {
namespace swig {

namespace {
string PyObjectToString(PyObject* o);
}  // namespace

std::unordered_map<string, PyObject*>* RegisteredPyObjectMap() {
  static auto* m = new std::unordered_map<string, PyObject*>();
  return m;
}

PyObject* GetRegisteredPyObject(const string& name) {
   std::vector<std::string> mht_0_v;
   mht_0_v.push_back("name: \"" + name + "\"");
   MHTracer_DTPStensorflowPSpythonPSutilPSutilDTcc mht_0(mht_0_v, 211, "", "./tensorflow/python/util/util.cc", "GetRegisteredPyObject");

  const auto* m = RegisteredPyObjectMap();
  auto it = m->find(name);
  if (it == m->end()) {
    PyErr_SetString(PyExc_TypeError,
                    tensorflow::strings::StrCat("No object with name ", name,
                                                " has been registered.")
                        .c_str());
    return nullptr;
  }
  return it->second;
}

PyObject* RegisterType(PyObject* type_name, PyObject* type) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSpythonPSutilPSutilDTcc mht_1(mht_1_v, 227, "", "./tensorflow/python/util/util.cc", "RegisterType");

  if (!PyType_Check(type)) {
    PyErr_SetString(PyExc_TypeError,
                    tensorflow::strings::StrCat("Expecting a type, got ",
                                                Py_TYPE(type)->tp_name)
                        .c_str());
    return nullptr;
  }
  return RegisterPyObject(type_name, type);
}

PyObject* RegisterPyObject(PyObject* name, PyObject* value) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSpythonPSutilPSutilDTcc mht_2(mht_2_v, 241, "", "./tensorflow/python/util/util.cc", "RegisterPyObject");

  string key;
  if (PyBytes_Check(name)) {
    key = PyBytes_AsString(name);
#if PY_MAJOR_VERSION >= 3
  } else if (PyUnicode_Check(name)) {
    key = PyUnicode_AsUTF8(name);
#endif
  } else {
    PyErr_SetString(PyExc_TypeError, tensorflow::strings::StrCat(
                                         "Expected name to be a str, got",
                                         PyObjectToString(name))
                                         .c_str());
    return nullptr;
  }

  auto* m = RegisteredPyObjectMap();
  if (m->find(key) != m->end()) {
    PyErr_SetString(PyExc_TypeError, tensorflow::strings::StrCat(
                                         "Value already registered for ", key)
                                         .c_str());
    return nullptr;
  }

  Py_INCREF(value);
  m->emplace(key, value);

  Py_RETURN_NONE;
}

namespace {
const int kMaxItemsInCache = 1024;

bool IsString(PyObject* o) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPSpythonPSutilPSutilDTcc mht_3(mht_3_v, 277, "", "./tensorflow/python/util/util.cc", "IsString");

  return PyBytes_Check(o) ||
#if PY_MAJOR_VERSION < 3
         PyString_Check(o) ||
#endif
         PyUnicode_Check(o);
}

// Equivalent to Python's 'o.__class__.__name__'
// Note that '__class__' attribute is set only in new-style classes.
// A lot of tensorflow code uses __class__ without checks, so it seems like
// we only support new-style classes.
StringPiece GetClassName(PyObject* o) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPSpythonPSutilPSutilDTcc mht_4(mht_4_v, 292, "", "./tensorflow/python/util/util.cc", "GetClassName");

  // __class__ is equivalent to type() for new style classes.
  // type() is equivalent to PyObject_Type()
  // (https://docs.python.org/3.5/c-api/object.html#c.PyObject_Type)
  // PyObject_Type() is equivalent to o->ob_type except for Py_INCREF, which
  // we don't need here.
  PyTypeObject* type = o->ob_type;

  // __name__ is the value of `tp_name` after the last '.'
  // (https://docs.python.org/2/c-api/typeobj.html#c.PyTypeObject.tp_name)
  StringPiece name(type->tp_name);
  size_t pos = name.rfind('.');
  if (pos != StringPiece::npos) {
    name.remove_prefix(pos + 1);
  }
  return name;
}

string PyObjectToString(PyObject* o) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPSpythonPSutilPSutilDTcc mht_5(mht_5_v, 313, "", "./tensorflow/python/util/util.cc", "PyObjectToString");

  if (o == nullptr) {
    return "<null object>";
  }
  PyObject* str = PyObject_Str(o);
  if (str) {
#if PY_MAJOR_VERSION < 3
    string s(PyString_AS_STRING(str));
#else
    string s(PyUnicode_AsUTF8(str));
#endif
    Py_DECREF(str);
    return tensorflow::strings::StrCat("type=", GetClassName(o), " str=", s);
  } else {
    return "<failed to execute str() on object>";
  }
}

class CachedTypeCheck {
 public:
  explicit CachedTypeCheck(std::function<int(PyObject*)> ternary_predicate)
      : ternary_predicate_(std::move(ternary_predicate)) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPSpythonPSutilPSutilDTcc mht_6(mht_6_v, 337, "", "./tensorflow/python/util/util.cc", "CachedTypeCheck");
}

  ~CachedTypeCheck() {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPSpythonPSutilPSutilDTcc mht_7(mht_7_v, 342, "", "./tensorflow/python/util/util.cc", "~CachedTypeCheck");

    mutex_lock l(type_to_sequence_map_mu_);
    for (const auto& pair : type_to_sequence_map_) {
      Py_DECREF(pair.first);
    }
  }

  // Caches successful executions of the one-argument (PyObject*) callable
  // "ternary_predicate" based on the type of "o". -1 from the callable
  // indicates an unsuccessful check (not cached), 0 indicates that "o"'s type
  // does not match the predicate, and 1 indicates that it does. Used to avoid
  // calling back into Python for expensive isinstance checks.
  int CachedLookup(PyObject* o) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPSpythonPSutilPSutilDTcc mht_8(mht_8_v, 357, "", "./tensorflow/python/util/util.cc", "CachedLookup");

    // Try not to return to Python - see if the type has already been seen
    // before.

    auto* type = Py_TYPE(o);

    {
      tf_shared_lock l(type_to_sequence_map_mu_);
      auto it = type_to_sequence_map_.find(type);
      if (it != type_to_sequence_map_.end()) {
        return it->second;
      }
    }

    int check_result = ternary_predicate_(o);

    if (check_result == -1) {
      return -1;  // Type check error, not cached.
    }

    // NOTE: This is never decref'd as long as the object lives, which is likely
    // forever, but we don't want the type to get deleted as long as it is in
    // the map. This should not be too much of a leak, as there should only be a
    // relatively small number of types in the map, and an even smaller number
    // that are eligible for decref. As a precaution, we limit the size of the
    // map to 1024.
    {
      mutex_lock l(type_to_sequence_map_mu_);
      if (type_to_sequence_map_.size() < kMaxItemsInCache) {
        Py_INCREF(type);
        auto insert_result = type_to_sequence_map_.insert({type, check_result});
        if (!insert_result.second) {
          // The type was added to the cache by a concurrent thread after we
          // looked it up above.
          Py_DECREF(type);
        }
      }
    }

    return check_result;
  }

 private:
  std::function<int(PyObject*)> ternary_predicate_;
  mutex type_to_sequence_map_mu_;
  std::unordered_map<PyTypeObject*, bool> type_to_sequence_map_
      TF_GUARDED_BY(type_to_sequence_map_mu_);
};

// Returns 1 if 'obj' is an instance of 'type_name'
// Returns 0 otherwise.
// Returns -1 if an error occurred (e.g., if 'type_name' is not registered.)
int IsInstanceOfRegisteredType(PyObject* obj, const char* type_name) {
   std::vector<std::string> mht_9_v;
   mht_9_v.push_back("type_name: \"" + (type_name == nullptr ? std::string("nullptr") : std::string((char*)type_name)) + "\"");
   MHTracer_DTPStensorflowPSpythonPSutilPSutilDTcc mht_9(mht_9_v, 413, "", "./tensorflow/python/util/util.cc", "IsInstanceOfRegisteredType");

  PyObject* type_obj = GetRegisteredPyObject(type_name);
  if (TF_PREDICT_FALSE(type_obj == nullptr)) {
    PyErr_SetString(PyExc_RuntimeError,
                    tensorflow::strings::StrCat(
                        type_name,
                        " type has not been set. "
                        "Please register the type with the identifier \"",
                        type_name, "\" using RegisterType.")
                        .c_str());
    return -1;
  }
  return PyObject_IsInstance(obj, type_obj);
}

// Returns 1 if `o` is considered a mapping for the purposes of Flatten().
// Returns 0 otherwise.
// Returns -1 if an error occurred.
int IsMappingHelper(PyObject* o) {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPSpythonPSutilPSutilDTcc mht_10(mht_10_v, 434, "", "./tensorflow/python/util/util.cc", "IsMappingHelper");

  static auto* const check_cache = new CachedTypeCheck([](PyObject* to_check) {
    return IsInstanceOfRegisteredType(to_check, "Mapping");
  });
  if (PyDict_Check(o)) return true;
  return check_cache->CachedLookup(o);
}

// Returns 1 if `o` is considered a mutable mapping for the purposes of
// Flatten(). Returns 0 otherwise. Returns -1 if an error occurred.
int IsMutableMappingHelper(PyObject* o) {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPSpythonPSutilPSutilDTcc mht_11(mht_11_v, 447, "", "./tensorflow/python/util/util.cc", "IsMutableMappingHelper");

  static auto* const check_cache = new CachedTypeCheck([](PyObject* to_check) {
    return IsInstanceOfRegisteredType(to_check, "MutableMapping");
  });
  if (PyDict_Check(o)) return true;
  return check_cache->CachedLookup(o);
}

// Returns 1 if `o` is considered a mapping view for the purposes of Flatten().
// Returns 0 otherwise.
// Returns -1 if an error occurred.
int IsMappingViewHelper(PyObject* o) {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPSpythonPSutilPSutilDTcc mht_12(mht_12_v, 461, "", "./tensorflow/python/util/util.cc", "IsMappingViewHelper");

  static auto* const check_cache = new CachedTypeCheck([](PyObject* to_check) {
    return IsInstanceOfRegisteredType(to_check, "MappingView");
  });
  return check_cache->CachedLookup(o);
}

// Returns 1 if `o` is considered an object proxy
// Returns 0 otherwise.
// Returns -1 if an error occurred.
int IsObjectProxy(PyObject* o) {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPSpythonPSutilPSutilDTcc mht_13(mht_13_v, 474, "", "./tensorflow/python/util/util.cc", "IsObjectProxy");

  static auto* const check_cache = new CachedTypeCheck([](PyObject* to_check) {
    return IsInstanceOfRegisteredType(to_check, "ObjectProxy");
  });
  return check_cache->CachedLookup(o);
}

// Returns 1 if `o` is an instance of attrs-decorated class.
// Returns 0 otherwise.
int IsAttrsHelper(PyObject* o) {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPSpythonPSutilPSutilDTcc mht_14(mht_14_v, 486, "", "./tensorflow/python/util/util.cc", "IsAttrsHelper");

  static auto* const check_cache = new CachedTypeCheck([](PyObject* to_check) {
    Safe_PyObjectPtr cls(PyObject_GetAttrString(to_check, "__class__"));
    if (cls) {
      return PyObject_HasAttrString(cls.get(), "__attrs_attrs__");
    }

    // PyObject_GetAttrString returns null on error
    PyErr_Clear();
    return 0;
  });
  return check_cache->CachedLookup(o);
}

// Returns 1 if `o` is an object of type IndexedSlices.
// Returns 0 otherwise.
// Returns -1 if an error occurred.
int IsIndexedSlicesHelper(PyObject* o) {
   std::vector<std::string> mht_15_v;
   MHTracer_DTPStensorflowPSpythonPSutilPSutilDTcc mht_15(mht_15_v, 506, "", "./tensorflow/python/util/util.cc", "IsIndexedSlicesHelper");

  static auto* const check_cache = new CachedTypeCheck([](PyObject* to_check) {
    return IsInstanceOfRegisteredType(to_check, "IndexedSlices");
  });
  return check_cache->CachedLookup(o);
}

// Returns 1 if `o` is a Tensor.
// Returns 0 otherwise.
// Returns -1 if an error occurred.
int IsTensorHelper(PyObject* o) {
   std::vector<std::string> mht_16_v;
   MHTracer_DTPStensorflowPSpythonPSutilPSutilDTcc mht_16(mht_16_v, 519, "", "./tensorflow/python/util/util.cc", "IsTensorHelper");

  static auto* const check_cache = new CachedTypeCheck([](PyObject* to_check) {
    return IsInstanceOfRegisteredType(to_check, "Tensor");
  });
  return check_cache->CachedLookup(o);
}

// Returns 1 if `o` is a TensorSpec.
// Returns 0 otherwise.
// Returns -1 if an error occurred.
int IsTensorSpecHelper(PyObject* o) {
   std::vector<std::string> mht_17_v;
   MHTracer_DTPStensorflowPSpythonPSutilPSutilDTcc mht_17(mht_17_v, 532, "", "./tensorflow/python/util/util.cc", "IsTensorSpecHelper");

  static auto* const check_cache = new CachedTypeCheck([](PyObject* to_check) {
    return IsInstanceOfRegisteredType(to_check, "TensorSpec");
  });
  return check_cache->CachedLookup(o);
}

// Returns 1 if `o` is an EagerTensor.
// Returns 0 otherwise.
// Returns -1 if an error occurred.
int IsEagerTensorHelper(PyObject* o) {
   std::vector<std::string> mht_18_v;
   MHTracer_DTPStensorflowPSpythonPSutilPSutilDTcc mht_18(mht_18_v, 545, "", "./tensorflow/python/util/util.cc", "IsEagerTensorHelper");

  static auto* const check_cache = new CachedTypeCheck([](PyObject* to_check) {
    return IsInstanceOfRegisteredType(to_check, "EagerTensor");
  });
  return check_cache->CachedLookup(o);
}

// Returns 1 if `o` is a ResourceVariable.
// Returns 0 otherwise.
// Returns -1 if an error occurred.
int IsResourceVariableHelper(PyObject* o) {
   std::vector<std::string> mht_19_v;
   MHTracer_DTPStensorflowPSpythonPSutilPSutilDTcc mht_19(mht_19_v, 558, "", "./tensorflow/python/util/util.cc", "IsResourceVariableHelper");

  static auto* const check_cache = new CachedTypeCheck([](PyObject* to_check) {
    return IsInstanceOfRegisteredType(to_check, "ResourceVariable");
  });
  return check_cache->CachedLookup(o);
}

// Returns 1 if `o` is a OwnedIterator.
// Returns 0 otherwise.
// Returns -1 if an error occurred.
int IsOwnedIteratorHelper(PyObject* o) {
   std::vector<std::string> mht_20_v;
   MHTracer_DTPStensorflowPSpythonPSutilPSutilDTcc mht_20(mht_20_v, 571, "", "./tensorflow/python/util/util.cc", "IsOwnedIteratorHelper");

  static auto* const check_cache = new CachedTypeCheck([](PyObject* to_check) {
    return IsInstanceOfRegisteredType(to_check, "OwnedIterator");
  });
  return check_cache->CachedLookup(o);
}

// Returns 1 if `o` is a ResourceVariable.
// Returns 0 otherwise.
// Returns -1 if an error occurred.
int IsVariableHelper(PyObject* o) {
   std::vector<std::string> mht_21_v;
   MHTracer_DTPStensorflowPSpythonPSutilPSutilDTcc mht_21(mht_21_v, 584, "", "./tensorflow/python/util/util.cc", "IsVariableHelper");

  static auto* const check_cache = new CachedTypeCheck([](PyObject* to_check) {
    return IsInstanceOfRegisteredType(to_check, "Variable");
  });
  return check_cache->CachedLookup(o);
}

// Returns 1 if `o` is considered a sequence for the purposes of Flatten().
// Returns 0 otherwise.
// Returns -1 if an error occurred.
int IsNestedHelper(PyObject* o) {
   std::vector<std::string> mht_22_v;
   MHTracer_DTPStensorflowPSpythonPSutilPSutilDTcc mht_22(mht_22_v, 597, "", "./tensorflow/python/util/util.cc", "IsNestedHelper");

  // We treat dicts and other mappings as special cases of sequences.
  if (IsMappingHelper(o)) return true;
  if (IsMappingViewHelper(o)) return true;
  if (IsAttrsHelper(o)) return true;

  static auto* const check_cache = new CachedTypeCheck([](PyObject* to_check) {
    int is_instance = IsInstanceOfRegisteredType(to_check, "Sequence");

    // Don't cache a failed is_instance check.
    if (is_instance == -1) return -1;

    return static_cast<int>(is_instance != 0 && !IsString(to_check));
  });
  return check_cache->CachedLookup(o);
}

// Returns 1 if `o`'s class has a `__tf_dispatch__` attribute.
// Returns 0 otherwise.
int IsDispatchableHelper(PyObject* o) {
   std::vector<std::string> mht_23_v;
   MHTracer_DTPStensorflowPSpythonPSutilPSutilDTcc mht_23(mht_23_v, 619, "", "./tensorflow/python/util/util.cc", "IsDispatchableHelper");

  static auto* const check_cache = new CachedTypeCheck([](PyObject* to_check) {
    return PyObject_HasAttrString(
        reinterpret_cast<PyObject*>(to_check->ob_type), "__tf_dispatch__");
  });
  return check_cache->CachedLookup(o);
}

// ValueIterator interface
class ValueIterator {
 public:
  virtual ~ValueIterator() {
   std::vector<std::string> mht_24_v;
   MHTracer_DTPStensorflowPSpythonPSutilPSutilDTcc mht_24(mht_24_v, 633, "", "./tensorflow/python/util/util.cc", "~ValueIterator");
}
  virtual Safe_PyObjectPtr next() = 0;

  bool valid() const {
   std::vector<std::string> mht_25_v;
   MHTracer_DTPStensorflowPSpythonPSutilPSutilDTcc mht_25(mht_25_v, 639, "", "./tensorflow/python/util/util.cc", "valid");
 return is_valid_; }

 protected:
  void invalidate() {
   std::vector<std::string> mht_26_v;
   MHTracer_DTPStensorflowPSpythonPSutilPSutilDTcc mht_26(mht_26_v, 645, "", "./tensorflow/python/util/util.cc", "invalidate");
 is_valid_ = false; }

 private:
  bool is_valid_ = true;
};

using ValueIteratorPtr = std::unique_ptr<ValueIterator>;

// Iterate through dictionaries in a deterministic order by sorting the
// keys. Notice this means that we ignore the original order of
// `OrderedDict` instances. This is intentional, to avoid potential
// bugs caused by mixing ordered and plain dicts (e.g., flattening
// a dict but using a corresponding `OrderedDict` to pack it back).
class DictValueIterator : public ValueIterator {
 public:
  explicit DictValueIterator(PyObject* dict)
      : dict_(dict), keys_(PyDict_Keys(dict)) {
   std::vector<std::string> mht_27_v;
   MHTracer_DTPStensorflowPSpythonPSutilPSutilDTcc mht_27(mht_27_v, 664, "", "./tensorflow/python/util/util.cc", "DictValueIterator");

    if (PyList_Sort(keys_.get()) == -1) {
      invalidate();
    } else {
      iter_.reset(PyObject_GetIter(keys_.get()));
    }
  }

  Safe_PyObjectPtr next() override {
   std::vector<std::string> mht_28_v;
   MHTracer_DTPStensorflowPSpythonPSutilPSutilDTcc mht_28(mht_28_v, 675, "", "./tensorflow/python/util/util.cc", "next");

    Safe_PyObjectPtr result;
    Safe_PyObjectPtr key(PyIter_Next(iter_.get()));
    if (key) {
      // PyDict_GetItem returns a borrowed reference.
      PyObject* elem = PyDict_GetItem(dict_, key.get());
      if (elem) {
        Py_INCREF(elem);
        result.reset(elem);
      } else {
        PyErr_SetString(PyExc_RuntimeError,
                        "Dictionary was modified during iteration over it");
      }
    }
    return result;
  }

 private:
  PyObject* dict_;
  Safe_PyObjectPtr keys_;
  Safe_PyObjectPtr iter_;
};

// Iterate over mapping objects by sorting the keys first
class MappingValueIterator : public ValueIterator {
 public:
  explicit MappingValueIterator(PyObject* mapping)
      : mapping_(mapping), keys_(MappingKeys(mapping)) {
   std::vector<std::string> mht_29_v;
   MHTracer_DTPStensorflowPSpythonPSutilPSutilDTcc mht_29(mht_29_v, 705, "", "./tensorflow/python/util/util.cc", "MappingValueIterator");

    if (!keys_ || PyList_Sort(keys_.get()) == -1) {
      invalidate();
    } else {
      iter_.reset(PyObject_GetIter(keys_.get()));
    }
  }

  Safe_PyObjectPtr next() override {
   std::vector<std::string> mht_30_v;
   MHTracer_DTPStensorflowPSpythonPSutilPSutilDTcc mht_30(mht_30_v, 716, "", "./tensorflow/python/util/util.cc", "next");

    Safe_PyObjectPtr result;
    Safe_PyObjectPtr key(PyIter_Next(iter_.get()));
    if (key) {
      // Unlike PyDict_GetItem, PyObject_GetItem returns a new reference.
      PyObject* elem = PyObject_GetItem(mapping_, key.get());
      if (elem) {
        result.reset(elem);
      } else {
        PyErr_SetString(PyExc_RuntimeError,
                        "Mapping was modified during iteration over it");
      }
    }
    return result;
  }

 private:
  PyObject* mapping_;
  Safe_PyObjectPtr keys_;
  Safe_PyObjectPtr iter_;
};

// Iterate over a sequence, by index.
class SequenceValueIterator : public ValueIterator {
 public:
  explicit SequenceValueIterator(PyObject* iterable)
      : seq_(PySequence_Fast(iterable, "")),
        size_(seq_.get() ? PySequence_Fast_GET_SIZE(seq_.get()) : 0),
        index_(0) {
   std::vector<std::string> mht_31_v;
   MHTracer_DTPStensorflowPSpythonPSutilPSutilDTcc mht_31(mht_31_v, 747, "", "./tensorflow/python/util/util.cc", "SequenceValueIterator");
}

  Safe_PyObjectPtr next() override {
   std::vector<std::string> mht_32_v;
   MHTracer_DTPStensorflowPSpythonPSutilPSutilDTcc mht_32(mht_32_v, 752, "", "./tensorflow/python/util/util.cc", "next");

    Safe_PyObjectPtr result;
    if (index_ < size_) {
      // PySequence_Fast_GET_ITEM returns a borrowed reference.
      PyObject* elem = PySequence_Fast_GET_ITEM(seq_.get(), index_);
      ++index_;
      if (elem) {
        Py_INCREF(elem);
        result.reset(elem);
      }
    }

    return result;
  }

 private:
  Safe_PyObjectPtr seq_;
  const Py_ssize_t size_;
  Py_ssize_t index_;
};

// Iterator that just returns a single python object.
class SingleValueIterator : public ValueIterator {
 public:
  explicit SingleValueIterator(PyObject* x) : x_(x) {
   std::vector<std::string> mht_33_v;
   MHTracer_DTPStensorflowPSpythonPSutilPSutilDTcc mht_33(mht_33_v, 779, "", "./tensorflow/python/util/util.cc", "SingleValueIterator");
 Py_INCREF(x); }

  Safe_PyObjectPtr next() override {
   std::vector<std::string> mht_34_v;
   MHTracer_DTPStensorflowPSpythonPSutilPSutilDTcc mht_34(mht_34_v, 784, "", "./tensorflow/python/util/util.cc", "next");
 return std::move(x_); }

 private:
  Safe_PyObjectPtr x_;
};

// Returns nullptr (to raise an exception) when next() is called.  Caller
// should have already called PyErr_SetString.
class ErrorValueIterator : public ValueIterator {
 public:
  ErrorValueIterator() {
   std::vector<std::string> mht_35_v;
   MHTracer_DTPStensorflowPSpythonPSutilPSutilDTcc mht_35(mht_35_v, 797, "", "./tensorflow/python/util/util.cc", "ErrorValueIterator");
}
  Safe_PyObjectPtr next() override {
   std::vector<std::string> mht_36_v;
   MHTracer_DTPStensorflowPSpythonPSutilPSutilDTcc mht_36(mht_36_v, 801, "", "./tensorflow/python/util/util.cc", "next");
 return nullptr; }
};

class AttrsValueIterator : public ValueIterator {
 public:
  explicit AttrsValueIterator(PyObject* nested) : nested_(nested) {
   std::vector<std::string> mht_37_v;
   MHTracer_DTPStensorflowPSpythonPSutilPSutilDTcc mht_37(mht_37_v, 809, "", "./tensorflow/python/util/util.cc", "AttrsValueIterator");

    Py_INCREF(nested);
    cls_.reset(PyObject_GetAttrString(nested_.get(), "__class__"));
    if (cls_) {
      attrs_.reset(PyObject_GetAttrString(cls_.get(), "__attrs_attrs__"));
      if (attrs_) {
        iter_.reset(PyObject_GetIter(attrs_.get()));
      }
    }
    if (!iter_ || PyErr_Occurred()) invalidate();
  }

  Safe_PyObjectPtr next() override {
   std::vector<std::string> mht_38_v;
   MHTracer_DTPStensorflowPSpythonPSutilPSutilDTcc mht_38(mht_38_v, 824, "", "./tensorflow/python/util/util.cc", "next");

    Safe_PyObjectPtr result;
    Safe_PyObjectPtr item(PyIter_Next(iter_.get()));
    if (item) {
      Safe_PyObjectPtr name(PyObject_GetAttrString(item.get(), "name"));
      result.reset(PyObject_GetAttr(nested_.get(), name.get()));
    }

    return result;
  }

 private:
  Safe_PyObjectPtr nested_;
  Safe_PyObjectPtr cls_;
  Safe_PyObjectPtr attrs_;
  Safe_PyObjectPtr iter_;
};

bool IsSparseTensorValueType(PyObject* o) {
   std::vector<std::string> mht_39_v;
   MHTracer_DTPStensorflowPSpythonPSutilPSutilDTcc mht_39(mht_39_v, 845, "", "./tensorflow/python/util/util.cc", "IsSparseTensorValueType");

  PyObject* sparse_tensor_value_type =
      GetRegisteredPyObject("SparseTensorValue");
  if (TF_PREDICT_FALSE(sparse_tensor_value_type == nullptr)) {
    return false;
  }

  return PyObject_TypeCheck(
             o, reinterpret_cast<PyTypeObject*>(sparse_tensor_value_type)) == 1;
}

// Returns 1 if `o` is an instance of CompositeTensor.
// Returns 0 otherwise.
// Returns -1 if an error occurred.
bool IsCompositeTensorHelper(PyObject* o) {
   std::vector<std::string> mht_40_v;
   MHTracer_DTPStensorflowPSpythonPSutilPSutilDTcc mht_40(mht_40_v, 862, "", "./tensorflow/python/util/util.cc", "IsCompositeTensorHelper");

  static auto* const check_cache = new CachedTypeCheck([](PyObject* to_check) {
    return IsInstanceOfRegisteredType(to_check, "CompositeTensor");
  });
  return check_cache->CachedLookup(o);
}

// Returns 1 if `o` is an instance of TypeSpec, but is not TensorSpec or
// VariableSpec.
// Returns 0 otherwise.
// Returns -1 if an error occurred.
bool IsTypeSpecHelper(PyObject* o) {
   std::vector<std::string> mht_41_v;
   MHTracer_DTPStensorflowPSpythonPSutilPSutilDTcc mht_41(mht_41_v, 876, "", "./tensorflow/python/util/util.cc", "IsTypeSpecHelper");

  static auto* const check_cache = new CachedTypeCheck([](PyObject* to_check) {
    int is_type_spec = IsInstanceOfRegisteredType(to_check, "TypeSpec");
    int is_dense_spec = (IsInstanceOfRegisteredType(to_check, "TensorSpec") ||
                         IsInstanceOfRegisteredType(to_check, "VariableSpec"));
    if ((is_type_spec == -1) || (is_dense_spec == -1)) return -1;
    return static_cast<int>(is_type_spec && !is_dense_spec);
  });
  return check_cache->CachedLookup(o);
}

// Returns 1 if `o` is a (non-string) sequence or CompositeTensor or
// (non-TensorSpec and non-VariableSpec) TypeSpec.
// Returns 0 otherwise.
// Returns -1 if an error occurred.
int IsNestedOrCompositeHelper(PyObject* o) {
   std::vector<std::string> mht_42_v;
   MHTracer_DTPStensorflowPSpythonPSutilPSutilDTcc mht_42(mht_42_v, 894, "", "./tensorflow/python/util/util.cc", "IsNestedOrCompositeHelper");

  int is_nested = IsNestedHelper(o);
  int is_composite = IsCompositeTensorHelper(o);
  int is_type_spec = IsTypeSpecHelper(o);
  if ((is_nested == -1) || (is_composite == -1) || (is_type_spec == -1)) {
    return -1;
  }
  return is_nested || is_composite || is_type_spec;
}

int IsNestedForDataHelper(PyObject* o) {
   std::vector<std::string> mht_43_v;
   MHTracer_DTPStensorflowPSpythonPSutilPSutilDTcc mht_43(mht_43_v, 907, "", "./tensorflow/python/util/util.cc", "IsNestedForDataHelper");

  return IsNestedHelper(o) == 1 && !PyList_Check(o) &&
         !IsSparseTensorValueType(o);
}

ValueIteratorPtr GetValueIterator(PyObject* nested) {
   std::vector<std::string> mht_44_v;
   MHTracer_DTPStensorflowPSpythonPSutilPSutilDTcc mht_44(mht_44_v, 915, "", "./tensorflow/python/util/util.cc", "GetValueIterator");

  if (PyDict_Check(nested)) {
    return absl::make_unique<DictValueIterator>(nested);
  } else if (IsMappingHelper(nested)) {
    return absl::make_unique<MappingValueIterator>(nested);
  } else if (IsAttrsHelper(nested)) {
    return absl::make_unique<AttrsValueIterator>(nested);
  } else {
    return absl::make_unique<SequenceValueIterator>(nested);
  }
}

// Similar to above, just specialized for the functions in the data package.
ValueIteratorPtr GetValueIteratorForData(PyObject* nested) {
   std::vector<std::string> mht_45_v;
   MHTracer_DTPStensorflowPSpythonPSutilPSutilDTcc mht_45(mht_45_v, 931, "", "./tensorflow/python/util/util.cc", "GetValueIteratorForData");

  if (PyDict_Check(nested)) {
    return absl::make_unique<DictValueIterator>(nested);
  } else if (IsMappingHelper(nested)) {
    return absl::make_unique<MappingValueIterator>(nested);
  } else if (IsAttrsHelper(nested)) {
    return absl::make_unique<AttrsValueIterator>(nested);
  } else if (IsSparseTensorValueType(nested)) {
    return absl::make_unique<SingleValueIterator>(nested);
  } else {
    return absl::make_unique<SequenceValueIterator>(nested);
  }
}

// Similar to GetValueIterator above, but expands CompositeTensor and TypeSpec.
ValueIteratorPtr GetValueIteratorForComposite(PyObject* nested) {
   std::vector<std::string> mht_46_v;
   MHTracer_DTPStensorflowPSpythonPSutilPSutilDTcc mht_46(mht_46_v, 949, "", "./tensorflow/python/util/util.cc", "GetValueIteratorForComposite");

  if (IsCompositeTensor(nested)) {
    Safe_PyObjectPtr spec(PyObject_GetAttrString(nested, "_type_spec"));
    if (PyErr_Occurred() || !spec) {
      return absl::make_unique<ErrorValueIterator>();
    }

    static char to_components[] = "_to_components";
    static char argspec[] = "(O)";
    Safe_PyObjectPtr components(
        PyObject_CallMethod(spec.get(), to_components, argspec, nested));
    if (PyErr_Occurred() || components == nullptr) {
      return absl::make_unique<ErrorValueIterator>();
    }
    return absl::make_unique<SingleValueIterator>(components.get());
  }

  if (IsTypeSpec(nested)) {
    Safe_PyObjectPtr specs(PyObject_GetAttrString(nested, "_component_specs"));
    if (PyErr_Occurred() || specs == nullptr) {
      return absl::make_unique<ErrorValueIterator>();
    }
    return absl::make_unique<SingleValueIterator>(specs.get());
  }

  return GetValueIterator(nested);
}

bool FlattenHelper(
    PyObject* nested, PyObject* list,
    const std::function<int(PyObject*)>& is_nested_helper,
    const std::function<ValueIteratorPtr(PyObject*)>& value_iterator_getter) {
   std::vector<std::string> mht_47_v;
   MHTracer_DTPStensorflowPSpythonPSutilPSutilDTcc mht_47(mht_47_v, 983, "", "./tensorflow/python/util/util.cc", "FlattenHelper");

  // if nested is not a sequence, append itself and exit
  int is_nested = is_nested_helper(nested);
  if (is_nested == -1) return false;
  if (!is_nested) {
    return PyList_Append(list, nested) != -1;
  }

  ValueIteratorPtr iter = value_iterator_getter(nested);
  if (!iter->valid()) return false;

  for (Safe_PyObjectPtr item = iter->next(); item; item = iter->next()) {
    if (Py_EnterRecursiveCall(" in flatten")) {
      return false;
    }
    const bool success = FlattenHelper(item.get(), list, is_nested_helper,
                                       value_iterator_getter);
    Py_LeaveRecursiveCall();
    if (!success) {
      return false;
    }
  }
  return true;
}

// Sets error using keys of 'dict1' and 'dict2'.
// 'dict1' and 'dict2' are assumed to be Python dictionaries.
void SetDifferentKeysError(PyObject* dict1, PyObject* dict2, string* error_msg,
                           bool* is_type_error) {
   std::vector<std::string> mht_48_v;
   MHTracer_DTPStensorflowPSpythonPSutilPSutilDTcc mht_48(mht_48_v, 1014, "", "./tensorflow/python/util/util.cc", "SetDifferentKeysError");

  Safe_PyObjectPtr k1(MappingKeys(dict1));
  if (PyErr_Occurred() || k1.get() == nullptr) {
    *error_msg =
        ("The two dictionaries don't have the same set of keys. Failed to "
         "fetch keys.");
    return;
  }
  Safe_PyObjectPtr k2(MappingKeys(dict2));
  if (PyErr_Occurred() || k2.get() == nullptr) {
    *error_msg =
        ("The two dictionaries don't have the same set of keys. Failed to "
         "fetch keys.");
    return;
  }
  *is_type_error = false;
  *error_msg = tensorflow::strings::StrCat(
      "The two dictionaries don't have the same set of keys. "
      "First structure has keys ",
      PyObjectToString(k1.get()), ", while second structure has keys ",
      PyObjectToString(k2.get()));
}

// Returns true iff there were no "internal" errors. In other words,
// errors that has nothing to do with structure checking.
// If an "internal" error occurred, the appropriate Python error will be
// set and the caller can propage it directly to the user.
//
// Both `error_msg` and `is_type_error` must be non-null. `error_msg` must
// be empty.
// Leaves `error_msg` empty if structures matched. Else, fills `error_msg`
// with appropriate error and sets `is_type_error` to true iff
// the error to be raised should be TypeError.
bool AssertSameStructureHelper(
    PyObject* o1, PyObject* o2, bool check_types, string* error_msg,
    bool* is_type_error, const std::function<int(PyObject*)>& is_nested_helper,
    const std::function<ValueIteratorPtr(PyObject*)>& value_iterator_getter,
    bool check_composite_tensor_type_spec) {
   std::vector<std::string> mht_49_v;
   MHTracer_DTPStensorflowPSpythonPSutilPSutilDTcc mht_49(mht_49_v, 1054, "", "./tensorflow/python/util/util.cc", "AssertSameStructureHelper");

  DCHECK(error_msg);
  DCHECK(is_type_error);
  const bool is_nested1 = is_nested_helper(o1);
  const bool is_nested2 = is_nested_helper(o2);
  if (PyErr_Occurred()) return false;
  if (is_nested1 != is_nested2) {
    string seq_str = is_nested1 ? PyObjectToString(o1) : PyObjectToString(o2);
    string non_seq_str =
        is_nested1 ? PyObjectToString(o2) : PyObjectToString(o1);
    *is_type_error = false;
    *error_msg = tensorflow::strings::StrCat(
        "Substructure \"", seq_str, "\" is a sequence, while substructure \"",
        non_seq_str, "\" is not");
    return true;
  }

  // Got to objects that are considered non-sequences. Note that in tf.data
  // use case lists and sparse_tensors are not considered sequences. So finished
  // checking, structures are the same.
  if (!is_nested1) return true;

  if (check_types) {
    // Treat wrapped tuples as tuples.
    tensorflow::Safe_PyObjectPtr o1_wrapped;
    if (IsObjectProxy(o1)) {
      o1_wrapped.reset(PyObject_GetAttrString(o1, "__wrapped__"));
      o1 = o1_wrapped.get();
    }
    tensorflow::Safe_PyObjectPtr o2_wrapped;
    if (IsObjectProxy(o2)) {
      o2_wrapped.reset(PyObject_GetAttrString(o2, "__wrapped__"));
      o2 = o2_wrapped.get();
    }

    const PyTypeObject* type1 = o1->ob_type;
    const PyTypeObject* type2 = o2->ob_type;

    // We treat two different namedtuples with identical name and fields
    // as having the same type.
    const PyObject* o1_tuple = IsNamedtuple(o1, false);
    if (o1_tuple == nullptr) return false;
    const PyObject* o2_tuple = IsNamedtuple(o2, false);
    if (o2_tuple == nullptr) {
      Py_DECREF(o1_tuple);
      return false;
    }
    bool both_tuples = o1_tuple == Py_True && o2_tuple == Py_True;
    Py_DECREF(o1_tuple);
    Py_DECREF(o2_tuple);

    if (both_tuples) {
      const PyObject* same_tuples = SameNamedtuples(o1, o2);
      if (same_tuples == nullptr) return false;
      bool not_same_tuples = same_tuples != Py_True;
      Py_DECREF(same_tuples);
      if (not_same_tuples) {
        *is_type_error = true;
        *error_msg = tensorflow::strings::StrCat(
            "The two namedtuples don't have the same sequence type. "
            "First structure ",
            PyObjectToString(o1), " has type ", type1->tp_name,
            ", while second structure ", PyObjectToString(o2), " has type ",
            type2->tp_name);
        return true;
      }
    } else if (type1 != type2
               /* If both sequences are list types, don't complain. This allows
                  one to be a list subclass (e.g. _ListWrapper used for
                  automatic dependency tracking.) */
               && !(PyList_Check(o1) && PyList_Check(o2))
               /* Two mapping types will also compare equal, making _DictWrapper
                  and dict compare equal. */
               && !(IsMappingHelper(o1) && IsMappingHelper(o2))
               /* For CompositeTensor & TypeSpec, we check below. */
               && !(check_composite_tensor_type_spec &&
                    (IsCompositeTensor(o1) || IsCompositeTensor(o2)) &&
                    (IsTypeSpec(o1) || IsTypeSpec(o2)))) {
      *is_type_error = true;
      *error_msg = tensorflow::strings::StrCat(
          "The two namedtuples don't have the same sequence type. "
          "First structure ",
          PyObjectToString(o1), " has type ", type1->tp_name,
          ", while second structure ", PyObjectToString(o2), " has type ",
          type2->tp_name);
      return true;
    }

    if (PyDict_Check(o1) && PyDict_Check(o2)) {
      if (PyDict_Size(o1) != PyDict_Size(o2)) {
        SetDifferentKeysError(o1, o2, error_msg, is_type_error);
        return true;
      }

      PyObject* key;
      Py_ssize_t pos = 0;
      while (PyDict_Next(o1, &pos, &key, nullptr)) {
        if (PyDict_GetItem(o2, key) == nullptr) {
          SetDifferentKeysError(o1, o2, error_msg, is_type_error);
          return true;
        }
      }
    } else if (IsMappingHelper(o1)) {
      // Fallback for custom mapping types. Instead of using PyDict methods
      // which stay in C, we call iter(o1).
      if (PyMapping_Size(o1) != PyMapping_Size(o2)) {
        SetDifferentKeysError(o1, o2, error_msg, is_type_error);
        return true;
      }

      Safe_PyObjectPtr iter(PyObject_GetIter(o1));
      PyObject* key;
      while ((key = PyIter_Next(iter.get())) != nullptr) {
        if (!PyMapping_HasKey(o2, key)) {
          SetDifferentKeysError(o1, o2, error_msg, is_type_error);
          Py_DECREF(key);
          return true;
        }
        Py_DECREF(key);
      }
    }
  }

  if (check_composite_tensor_type_spec &&
      (IsCompositeTensor(o1) || IsCompositeTensor(o2))) {
    Safe_PyObjectPtr owned_type_spec_1;
    PyObject* type_spec_1 = o1;
    if (IsCompositeTensor(o1)) {
      owned_type_spec_1.reset(PyObject_GetAttrString(o1, "_type_spec"));
      type_spec_1 = owned_type_spec_1.get();
    }

    Safe_PyObjectPtr owned_type_spec_2;
    PyObject* type_spec_2 = o2;
    if (IsCompositeTensor(o2)) {
      owned_type_spec_2.reset(PyObject_GetAttrString(o2, "_type_spec"));
      type_spec_2 = owned_type_spec_2.get();
    }

    // Two composite tensors are considered to have the same structure if
    // they share a type spec that is a supertype of both of them. We do *not*
    // use is_subtype_of, since that would prevent us from e.g. using a
    // cond statement where the two sides have different shapes.

    // TODO(b/206014848): We have to explicitly remove the names.
    Safe_PyObjectPtr owned_nameless_type_spec_1(
        PyObject_CallMethod(type_spec_1, "_without_tensor_names", nullptr));
    Safe_PyObjectPtr owned_nameless_type_spec_2(
        PyObject_CallMethod(type_spec_2, "_without_tensor_names", nullptr));
    // TODO(b/222123181): Reconsider most_specific_common_supertype usage.
    static char compatible_type[] = "most_specific_common_supertype";
    static char argspec[] = "([O])";
    Safe_PyObjectPtr struct_compatible(
        PyObject_CallMethod(owned_nameless_type_spec_1.get(), compatible_type,
                            argspec, owned_nameless_type_spec_2.get()));
    if (PyErr_Occurred()) {
      return false;
    }
    if (struct_compatible.get() == Py_None) {
      *is_type_error = false;
      *error_msg = tensorflow::strings::StrCat(
          "Incompatible CompositeTensor TypeSpecs: ",
          PyObjectToString(type_spec_1), " vs. ",
          PyObjectToString(type_spec_2));
      return true;
    }
  }

  ValueIteratorPtr iter1 = value_iterator_getter(o1);
  ValueIteratorPtr iter2 = value_iterator_getter(o2);

  if (!iter1->valid() || !iter2->valid()) return false;

  while (true) {
    Safe_PyObjectPtr v1 = iter1->next();
    Safe_PyObjectPtr v2 = iter2->next();
    if (v1 && v2) {
      if (Py_EnterRecursiveCall(" in assert_same_structure")) {
        return false;
      }
      bool no_internal_errors = AssertSameStructureHelper(
          v1.get(), v2.get(), check_types, error_msg, is_type_error,
          is_nested_helper, value_iterator_getter,
          check_composite_tensor_type_spec);
      Py_LeaveRecursiveCall();
      if (!no_internal_errors) return false;
      if (!error_msg->empty()) return true;
    } else if (!v1 && !v2) {
      // Done with all recursive calls. Structure matched.
      return true;
    } else {
      *is_type_error = false;
      *error_msg = tensorflow::strings::StrCat(
          "The two structures don't have the same number of elements. ",
          "First structure: ", PyObjectToString(o1),
          ". Second structure: ", PyObjectToString(o2));
      return true;
    }
  }
}

}  // namespace

bool IsNested(PyObject* o) {
   std::vector<std::string> mht_50_v;
   MHTracer_DTPStensorflowPSpythonPSutilPSutilDTcc mht_50(mht_50_v, 1260, "", "./tensorflow/python/util/util.cc", "IsNested");
 return IsNestedHelper(o) == 1; }
bool IsMapping(PyObject* o) {
   std::vector<std::string> mht_51_v;
   MHTracer_DTPStensorflowPSpythonPSutilPSutilDTcc mht_51(mht_51_v, 1264, "", "./tensorflow/python/util/util.cc", "IsMapping");
 return IsMappingHelper(o) == 1; }
bool IsMutableMapping(PyObject* o) {
   std::vector<std::string> mht_52_v;
   MHTracer_DTPStensorflowPSpythonPSutilPSutilDTcc mht_52(mht_52_v, 1268, "", "./tensorflow/python/util/util.cc", "IsMutableMapping");
 return IsMutableMappingHelper(o) == 1; }
bool IsMappingView(PyObject* o) {
   std::vector<std::string> mht_53_v;
   MHTracer_DTPStensorflowPSpythonPSutilPSutilDTcc mht_53(mht_53_v, 1272, "", "./tensorflow/python/util/util.cc", "IsMappingView");
 return IsMappingViewHelper(o) == 1; }
bool IsAttrs(PyObject* o) {
   std::vector<std::string> mht_54_v;
   MHTracer_DTPStensorflowPSpythonPSutilPSutilDTcc mht_54(mht_54_v, 1276, "", "./tensorflow/python/util/util.cc", "IsAttrs");
 return IsAttrsHelper(o) == 1; }
bool IsTensor(PyObject* o) {
   std::vector<std::string> mht_55_v;
   MHTracer_DTPStensorflowPSpythonPSutilPSutilDTcc mht_55(mht_55_v, 1280, "", "./tensorflow/python/util/util.cc", "IsTensor");
 return IsTensorHelper(o) == 1; }
bool IsTensorSpec(PyObject* o) {
   std::vector<std::string> mht_56_v;
   MHTracer_DTPStensorflowPSpythonPSutilPSutilDTcc mht_56(mht_56_v, 1284, "", "./tensorflow/python/util/util.cc", "IsTensorSpec");
 return IsTensorSpecHelper(o) == 1; }
bool IsEagerTensorSlow(PyObject* o) {
   std::vector<std::string> mht_57_v;
   MHTracer_DTPStensorflowPSpythonPSutilPSutilDTcc mht_57(mht_57_v, 1288, "", "./tensorflow/python/util/util.cc", "IsEagerTensorSlow");
 return IsEagerTensorHelper(o) == 1; }
bool IsResourceVariable(PyObject* o) {
   std::vector<std::string> mht_58_v;
   MHTracer_DTPStensorflowPSpythonPSutilPSutilDTcc mht_58(mht_58_v, 1292, "", "./tensorflow/python/util/util.cc", "IsResourceVariable");

  return IsResourceVariableHelper(o) == 1;
}
bool IsOwnedIterator(PyObject* o) {
   std::vector<std::string> mht_59_v;
   MHTracer_DTPStensorflowPSpythonPSutilPSutilDTcc mht_59(mht_59_v, 1298, "", "./tensorflow/python/util/util.cc", "IsOwnedIterator");
 return IsOwnedIteratorHelper(o) == 1; }
bool IsVariable(PyObject* o) {
   std::vector<std::string> mht_60_v;
   MHTracer_DTPStensorflowPSpythonPSutilPSutilDTcc mht_60(mht_60_v, 1302, "", "./tensorflow/python/util/util.cc", "IsVariable");
 return IsVariableHelper(o) == 1; }
bool IsIndexedSlices(PyObject* o) {
   std::vector<std::string> mht_61_v;
   MHTracer_DTPStensorflowPSpythonPSutilPSutilDTcc mht_61(mht_61_v, 1306, "", "./tensorflow/python/util/util.cc", "IsIndexedSlices");
 return IsIndexedSlicesHelper(o) == 1; }
bool IsDispatchable(PyObject* o) {
   std::vector<std::string> mht_62_v;
   MHTracer_DTPStensorflowPSpythonPSutilPSutilDTcc mht_62(mht_62_v, 1310, "", "./tensorflow/python/util/util.cc", "IsDispatchable");
 return IsDispatchableHelper(o) == 1; }

bool IsTuple(PyObject* o) {
   std::vector<std::string> mht_63_v;
   MHTracer_DTPStensorflowPSpythonPSutilPSutilDTcc mht_63(mht_63_v, 1315, "", "./tensorflow/python/util/util.cc", "IsTuple");

  tensorflow::Safe_PyObjectPtr wrapped;
  if (IsObjectProxy(o)) {
    wrapped.reset(PyObject_GetAttrString(o, "__wrapped__"));
    o = wrapped.get();
  }
  return PyTuple_Check(o);
}

// Work around a writable-strings warning with Python 2's PyMapping_Keys macro,
// and while we're at it give them consistent behavior by making sure the
// returned value is a list.
//
// As with PyMapping_Keys, returns a new reference.
//
// On failure, returns nullptr.
PyObject* MappingKeys(PyObject* o) {
   std::vector<std::string> mht_64_v;
   MHTracer_DTPStensorflowPSpythonPSutilPSutilDTcc mht_64(mht_64_v, 1334, "", "./tensorflow/python/util/util.cc", "MappingKeys");

#if PY_MAJOR_VERSION >= 3
  return PyMapping_Keys(o);
#else
  static char key_method_name[] = "keys";
  Safe_PyObjectPtr raw_result(PyObject_CallMethod(o, key_method_name, nullptr));
  if (PyErr_Occurred() || raw_result.get() == nullptr) {
    return nullptr;
  }
  return PySequence_Fast(
      raw_result.get(),
      "The '.keys()' method of a custom mapping returned a non-sequence.");
#endif
}

PyObject* Flatten(PyObject* nested, bool expand_composites) {
   std::vector<std::string> mht_65_v;
   MHTracer_DTPStensorflowPSpythonPSutilPSutilDTcc mht_65(mht_65_v, 1352, "", "./tensorflow/python/util/util.cc", "Flatten");

  PyObject* list = PyList_New(0);
  const std::function<int(PyObject*)>& is_nested_helper =
      expand_composites ? IsNestedOrCompositeHelper : IsNestedHelper;
  const std::function<ValueIteratorPtr(PyObject*)>& get_value_iterator =
      expand_composites ? GetValueIteratorForComposite : GetValueIterator;
  if (FlattenHelper(nested, list, is_nested_helper, get_value_iterator)) {
    return list;
  } else {
    Py_DECREF(list);
    return nullptr;
  }
}

bool IsNestedOrComposite(PyObject* o) {
   std::vector<std::string> mht_66_v;
   MHTracer_DTPStensorflowPSpythonPSutilPSutilDTcc mht_66(mht_66_v, 1369, "", "./tensorflow/python/util/util.cc", "IsNestedOrComposite");

  return IsNestedOrCompositeHelper(o) == 1;
}

bool IsCompositeTensor(PyObject* o) {
   std::vector<std::string> mht_67_v;
   MHTracer_DTPStensorflowPSpythonPSutilPSutilDTcc mht_67(mht_67_v, 1376, "", "./tensorflow/python/util/util.cc", "IsCompositeTensor");
 return IsCompositeTensorHelper(o) == 1; }

bool IsTypeSpec(PyObject* o) {
   std::vector<std::string> mht_68_v;
   MHTracer_DTPStensorflowPSpythonPSutilPSutilDTcc mht_68(mht_68_v, 1381, "", "./tensorflow/python/util/util.cc", "IsTypeSpec");
 return IsTypeSpecHelper(o) == 1; }

bool IsNestedForData(PyObject* o) {
   std::vector<std::string> mht_69_v;
   MHTracer_DTPStensorflowPSpythonPSutilPSutilDTcc mht_69(mht_69_v, 1386, "", "./tensorflow/python/util/util.cc", "IsNestedForData");
 return IsNestedForDataHelper(o) == 1; }

PyObject* FlattenForData(PyObject* nested) {
   std::vector<std::string> mht_70_v;
   MHTracer_DTPStensorflowPSpythonPSutilPSutilDTcc mht_70(mht_70_v, 1391, "", "./tensorflow/python/util/util.cc", "FlattenForData");

  PyObject* list = PyList_New(0);
  if (FlattenHelper(nested, list, IsNestedForDataHelper,
                    GetValueIteratorForData)) {
    return list;
  } else {
    Py_DECREF(list);
    return nullptr;
  }
}

PyObject* IsNamedtuple(PyObject* o, bool strict) {
   std::vector<std::string> mht_71_v;
   MHTracer_DTPStensorflowPSpythonPSutilPSutilDTcc mht_71(mht_71_v, 1405, "", "./tensorflow/python/util/util.cc", "IsNamedtuple");

  // Some low-level CPython calls do not work with wrapt.ObjectProxy, so they
  // require some unwrapping if we want to treat them like the objects they're
  // wrapping.
  tensorflow::Safe_PyObjectPtr o_wrapped;
  if (IsObjectProxy(o)) {
    o_wrapped.reset(PyObject_GetAttrString(o, "__wrapped__"));
    o = o_wrapped.get();
  }

  // Must be subclass of tuple
  if (!PyTuple_Check(o)) {
    Py_RETURN_FALSE;
  }

  // If strict, o.__class__.__base__ must be tuple
  if (strict) {
    PyObject* klass = PyObject_GetAttrString(o, "__class__");
    if (klass == nullptr) return nullptr;
    PyObject* base = PyObject_GetAttrString(klass, "__base__");
    Py_DECREF(klass);
    if (base == nullptr) return nullptr;

    const PyTypeObject* base_type = reinterpret_cast<PyTypeObject*>(base);
    // built-in object types are singletons
    bool tuple_base = base_type == &PyTuple_Type;
    Py_DECREF(base);
    if (!tuple_base) {
      Py_RETURN_FALSE;
    }
  }

  // o must have attribute '_fields' and every element in
  // '_fields' must be a string.
  int has_fields = PyObject_HasAttrString(o, "_fields");
  if (!has_fields) {
    Py_RETURN_FALSE;
  }

  Safe_PyObjectPtr fields = make_safe(PyObject_GetAttrString(o, "_fields"));
  int is_instance = IsInstanceOfRegisteredType(fields.get(), "Sequence");
  if (is_instance == 0) {
    Py_RETURN_FALSE;
  } else if (is_instance == -1) {
    return nullptr;
  }

  Safe_PyObjectPtr seq = make_safe(PySequence_Fast(fields.get(), ""));
  const Py_ssize_t s = PySequence_Fast_GET_SIZE(seq.get());
  for (Py_ssize_t i = 0; i < s; ++i) {
    // PySequence_Fast_GET_ITEM returns borrowed ref
    PyObject* elem = PySequence_Fast_GET_ITEM(seq.get(), i);
    if (!IsString(elem)) {
      Py_RETURN_FALSE;
    }
  }

  Py_RETURN_TRUE;
}

PyObject* SameNamedtuples(PyObject* o1, PyObject* o2) {
   std::vector<std::string> mht_72_v;
   MHTracer_DTPStensorflowPSpythonPSutilPSutilDTcc mht_72(mht_72_v, 1468, "", "./tensorflow/python/util/util.cc", "SameNamedtuples");

  Safe_PyObjectPtr f1 = make_safe(PyObject_GetAttrString(o1, "_fields"));
  Safe_PyObjectPtr f2 = make_safe(PyObject_GetAttrString(o2, "_fields"));
  if (f1 == nullptr || f2 == nullptr) {
    PyErr_SetString(
        PyExc_RuntimeError,
        "Expected namedtuple-like objects (that have _fields attr)");
    return nullptr;
  }

  if (PyObject_RichCompareBool(f1.get(), f2.get(), Py_NE)) {
    Py_RETURN_FALSE;
  }

  if (GetClassName(o1).compare(GetClassName(o2)) == 0) {
    Py_RETURN_TRUE;
  } else {
    Py_RETURN_FALSE;
  }
}

PyObject* AssertSameStructure(PyObject* o1, PyObject* o2, bool check_types,
                              bool expand_composites) {
   std::vector<std::string> mht_73_v;
   MHTracer_DTPStensorflowPSpythonPSutilPSutilDTcc mht_73(mht_73_v, 1493, "", "./tensorflow/python/util/util.cc", "AssertSameStructure");

  const std::function<int(PyObject*)>& is_nested_helper =
      expand_composites ? IsNestedOrCompositeHelper : IsNestedHelper;
  const std::function<ValueIteratorPtr(PyObject*)>& get_value_iterator =
      expand_composites ? GetValueIteratorForComposite : GetValueIterator;
  const bool check_composite_tensor_type_spec = expand_composites;
  string error_msg;
  bool is_type_error = false;
  AssertSameStructureHelper(o1, o2, check_types, &error_msg, &is_type_error,
                            is_nested_helper, get_value_iterator,
                            check_composite_tensor_type_spec);
  if (PyErr_Occurred()) {
    // Don't hide Python exceptions while checking (e.g. errors fetching keys
    // from custom mappings).
    return nullptr;
  }
  if (!error_msg.empty()) {
    PyErr_SetString(
        is_type_error ? PyExc_TypeError : PyExc_ValueError,
        tensorflow::strings::StrCat(
            "The two structures don't have the same nested structure.\n\n",
            "First structure: ", PyObjectToString(o1), "\n\nSecond structure: ",
            PyObjectToString(o2), "\n\nMore specifically: ", error_msg)
            .c_str());
    return nullptr;
  }
  Py_RETURN_NONE;
}

PyObject* AssertSameStructureForData(PyObject* o1, PyObject* o2,
                                     bool check_types) {
   std::vector<std::string> mht_74_v;
   MHTracer_DTPStensorflowPSpythonPSutilPSutilDTcc mht_74(mht_74_v, 1526, "", "./tensorflow/python/util/util.cc", "AssertSameStructureForData");

  string error_msg;
  bool is_type_error = false;
  AssertSameStructureHelper(o1, o2, check_types, &error_msg, &is_type_error,
                            IsNestedForDataHelper, GetValueIterator, false);
  if (PyErr_Occurred()) {
    // Don't hide Python exceptions while checking (e.g. errors fetching keys
    // from custom mappings).
    return nullptr;
  }
  if (!error_msg.empty()) {
    PyErr_SetString(
        is_type_error ? PyExc_TypeError : PyExc_ValueError,
        tensorflow::strings::StrCat(
            "The two structures don't have the same nested structure.\n\n",
            "First structure: ", PyObjectToString(o1), "\n\nSecond structure: ",
            PyObjectToString(o2), "\n\nMore specifically: ", error_msg)
            .c_str());
    return nullptr;
  }
  Py_RETURN_NONE;
}

}  // namespace swig
}  // namespace tensorflow
