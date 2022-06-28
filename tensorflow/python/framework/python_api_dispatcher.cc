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
class MHTracer_DTPStensorflowPSpythonPSframeworkPSpython_api_dispatcherDTcc {
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
   MHTracer_DTPStensorflowPSpythonPSframeworkPSpython_api_dispatcherDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSpythonPSframeworkPSpython_api_dispatcherDTcc() {
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

#include "tensorflow/python/framework/python_api_dispatcher.h"

#include <set>

#include "absl/strings/str_join.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/python/lib/core/py_util.h"
#include "tensorflow/python/lib/core/safe_pyobject_ptr.h"
#include "tensorflow/python/util/util.h"

namespace tensorflow {
namespace py_dispatch {

namespace {

std::vector<Safe_PyObjectPtr>& GetRegisteredDispatchableTypes() {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSpythonPSframeworkPSpython_api_dispatcherDTcc mht_0(mht_0_v, 201, "", "./tensorflow/python/framework/python_api_dispatcher.cc", "GetRegisteredDispatchableTypes");

  static std::vector<Safe_PyObjectPtr>* registered_dispatchable_types =
      new std::vector<Safe_PyObjectPtr>();
  if (registered_dispatchable_types->empty()) {
    static PyObject* composite_tensor =
        swig::GetRegisteredPyObject("CompositeTensor");
    Py_INCREF(composite_tensor);
    registered_dispatchable_types->push_back(
        Safe_PyObjectPtr(composite_tensor));
  }
  return *registered_dispatchable_types;
}

// Returns true if `py_class` is a registered dispatchable type.
bool IsRegisteredDispatchableType(PyObject* py_class) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSpythonPSframeworkPSpython_api_dispatcherDTcc mht_1(mht_1_v, 218, "", "./tensorflow/python/framework/python_api_dispatcher.cc", "IsRegisteredDispatchableType");

  DCheckPyGilState();
  for (const auto& registered_type : GetRegisteredDispatchableTypes()) {
    int result = PyObject_IsSubclass(py_class, registered_type.get());
    if (result > 0) return true;
    if (result < 0) PyErr_Clear();
  }
  return false;
}

// Raises an exception indicating that multiple dispatch targets matched.
Safe_PyObjectPtr RaiseDispatchConflictError(const std::string& api_name,
                                            PyObject* selected,
                                            PyObject* target) {
   std::vector<std::string> mht_2_v;
   mht_2_v.push_back("api_name: \"" + api_name + "\"");
   MHTracer_DTPStensorflowPSpythonPSframeworkPSpython_api_dispatcherDTcc mht_2(mht_2_v, 235, "", "./tensorflow/python/framework/python_api_dispatcher.cc", "RaiseDispatchConflictError");

  Safe_PyObjectPtr s1(PyObject_Str(selected));
  Safe_PyObjectPtr s2(PyObject_Str(target));
  PyErr_SetString(PyExc_ValueError,
                  absl::StrCat("Multiple dispatch targets that were "
                               "registered with tf.dispatch_for (",
                               s1 ? PyUnicode_AsUTF8(s1.get()) : "?", " and ",
                               s2 ? PyUnicode_AsUTF8(s2.get()) : "?",
                               ") match the arguments to ", api_name)
                      .c_str());
  return nullptr;
}

}  // namespace

bool RegisterDispatchableType(PyObject* py_class) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPSpythonPSframeworkPSpython_api_dispatcherDTcc mht_3(mht_3_v, 253, "", "./tensorflow/python/framework/python_api_dispatcher.cc", "RegisterDispatchableType");

  DCheckPyGilState();
  if (!PyType_Check(py_class)) {
    PyErr_SetString(
        PyExc_ValueError,
        absl::StrCat("Expected a type object; got object with type ",
                     py_class->ob_type->tp_name)
            .c_str());
    return false;
  }
  if (IsRegisteredDispatchableType(py_class)) {
    Safe_PyObjectPtr s(PyObject_Str(py_class));
    PyErr_SetString(PyExc_ValueError,
                    absl::StrCat("Type ", s ? PyUnicode_AsUTF8(s.get()) : "?",
                                 " (or one of its bases clases) has "
                                 "already been registered")
                        .c_str());
    return false;
  }
  Py_INCREF(py_class);
  GetRegisteredDispatchableTypes().push_back(Safe_PyObjectPtr(py_class));
  return true;
}

PythonAPIDispatcher::PythonAPIDispatcher(const std::string& api_name,
                                         absl::Span<const char*> arg_names,
                                         absl::Span<PyObject*> defaults)
    : api_name_(api_name),
      canonicalizer_(arg_names, defaults),
      canonicalized_args_storage_(canonicalizer_.GetArgSize()),
      canonicalized_args_span_(canonicalized_args_storage_) {
   std::vector<std::string> mht_4_v;
   mht_4_v.push_back("api_name: \"" + api_name + "\"");
   MHTracer_DTPStensorflowPSpythonPSframeworkPSpython_api_dispatcherDTcc mht_4(mht_4_v, 287, "", "./tensorflow/python/framework/python_api_dispatcher.cc", "PythonAPIDispatcher::PythonAPIDispatcher");
}

void PythonAPIDispatcher::Register(PySignatureChecker signature_checker,
                                   PyObject* dispatch_target) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPSpythonPSframeworkPSpython_api_dispatcherDTcc mht_5(mht_5_v, 293, "", "./tensorflow/python/framework/python_api_dispatcher.cc", "PythonAPIDispatcher::Register");

  DCheckPyGilState();
  Py_INCREF(dispatch_target);
  targets_.emplace_back(std::move(signature_checker),
                        Safe_PyObjectPtr(dispatch_target));
}

Safe_PyObjectPtr PythonAPIDispatcher::Dispatch(PyObject* args,
                                               PyObject* kwargs) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPSpythonPSframeworkPSpython_api_dispatcherDTcc mht_6(mht_6_v, 304, "", "./tensorflow/python/framework/python_api_dispatcher.cc", "PythonAPIDispatcher::Dispatch");

  DCheckPyGilState();
  if (kwargs == Py_None) {
    kwargs = nullptr;
  }
  // Canonicalize args (so we don't need to deal with kwargs).
  if (!canonicalizer_.Canonicalize(args, kwargs, canonicalized_args_span_)) {
    return nullptr;
  }

  PyObject* selected = nullptr;
  for (auto& target : targets_) {
    if (target.first.CheckCanonicalizedArgs(canonicalized_args_span_)) {
      if (selected && selected != target.second.get()) {
        return RaiseDispatchConflictError(api_name_, selected,
                                          target.second.get());
      }
      selected = target.second.get();
    }
  }
  if (selected) {
    return Safe_PyObjectPtr(PyObject_Call(selected, args, kwargs));
  } else {
    Py_INCREF(Py_NotImplemented);
    return Safe_PyObjectPtr(Py_NotImplemented);
  }
}

// TODO(b/194903203) Raise an error if `func` is not registered.
void PythonAPIDispatcher::Unregister(PyObject* func) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPSpythonPSframeworkPSpython_api_dispatcherDTcc mht_7(mht_7_v, 336, "", "./tensorflow/python/framework/python_api_dispatcher.cc", "PythonAPIDispatcher::Unregister");

  DCheckPyGilState();
  using DispatchTargetPair = std::pair<PySignatureChecker, Safe_PyObjectPtr>;
  targets_.erase(std::remove_if(targets_.begin(), targets_.end(),
                                [func](const DispatchTargetPair& t) {
                                  return t.second.get() == func;
                                }),
                 targets_.end());
}

std::string PythonAPIDispatcher::DebugString() const {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPSpythonPSframeworkPSpython_api_dispatcherDTcc mht_8(mht_8_v, 349, "", "./tensorflow/python/framework/python_api_dispatcher.cc", "PythonAPIDispatcher::DebugString");

  DCheckPyGilState();
  std::string out = absl::StrCat("<Dispatch(", api_name_, "): ");

  const char* sep = "";
  for (const auto& target : targets_) {
    Safe_PyObjectPtr target_str(PyObject_Str(target.second.get()));
    absl::StrAppend(&out, sep, target.first.DebugString(), " -> ",
                    target_str ? PyUnicode_AsUTF8(target_str.get()) : "?");
    sep = ", ";
  }
  return out;
}

PySignatureChecker::PySignatureChecker(
    std::vector<ParamChecker> parameter_checkers)
    : positional_parameter_checkers_(std::move(parameter_checkers)) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPSpythonPSframeworkPSpython_api_dispatcherDTcc mht_9(mht_9_v, 368, "", "./tensorflow/python/framework/python_api_dispatcher.cc", "PySignatureChecker::PySignatureChecker");

  // Check less expensive parameters first.
  std::sort(positional_parameter_checkers_.begin(),
            positional_parameter_checkers_.end(),
            [](ParamChecker a, ParamChecker b) {
              return a.second->cost() < b.second->cost();
            });
}

bool PySignatureChecker::CheckCanonicalizedArgs(
    absl::Span<PyObject*> canon_args) const {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPSpythonPSframeworkPSpython_api_dispatcherDTcc mht_10(mht_10_v, 381, "", "./tensorflow/python/framework/python_api_dispatcher.cc", "PySignatureChecker::CheckCanonicalizedArgs");

  bool matched_dispatchable_type = false;
  for (auto& c : positional_parameter_checkers_) {
    int index = c.first;
    auto& param_checker = c.second;
    if (index >= canon_args.size()) {
      return false;
    }
    switch (param_checker->Check(canon_args[index])) {
      case PyTypeChecker::MatchType::NO_MATCH:
        return false;
      case PyTypeChecker::MatchType::MATCH_DISPATCHABLE:
        matched_dispatchable_type = true;
        break;
      case PyTypeChecker::MatchType::MATCH:
        break;
    }
  }
  return matched_dispatchable_type;
}

std::string PySignatureChecker::DebugString() const {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPSpythonPSframeworkPSpython_api_dispatcherDTcc mht_11(mht_11_v, 405, "", "./tensorflow/python/framework/python_api_dispatcher.cc", "PySignatureChecker::DebugString");

  return absl::StrJoin(positional_parameter_checkers_, ", ",
                       [](std::string* out, ParamChecker p) {
                         absl::StrAppend(out, "args[", p.first,
                                         "]:", p.second->DebugString());
                       });
}

PyInstanceChecker::PyInstanceChecker(const std::vector<PyObject*>& py_classes) {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPSpythonPSframeworkPSpython_api_dispatcherDTcc mht_12(mht_12_v, 416, "", "./tensorflow/python/framework/python_api_dispatcher.cc", "PyInstanceChecker::PyInstanceChecker");

  DCheckPyGilState();
  py_classes_.reserve(py_classes.size());
  for (PyObject* py_class : py_classes) {
    py_classes_.emplace_back(py_class);
    Py_INCREF(py_class);
  }
}

PyInstanceChecker::~PyInstanceChecker() {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPSpythonPSframeworkPSpython_api_dispatcherDTcc mht_13(mht_13_v, 428, "", "./tensorflow/python/framework/python_api_dispatcher.cc", "PyInstanceChecker::~PyInstanceChecker");

  DCheckPyGilState();
  for (const auto& pair : py_class_cache_) {
    Py_DECREF(pair.first);
  }
}

PyTypeChecker::MatchType PyInstanceChecker::Check(PyObject* value) {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPSpythonPSframeworkPSpython_api_dispatcherDTcc mht_14(mht_14_v, 438, "", "./tensorflow/python/framework/python_api_dispatcher.cc", "PyInstanceChecker::Check");

  DCheckPyGilState();
  auto* type = Py_TYPE(value);
  auto it = py_class_cache_.find(type);
  if (it != py_class_cache_.end()) {
    return it->second;
  }

  MatchType result = MatchType::NO_MATCH;
  for (const auto& py_class : py_classes_) {
    int is_instance = PyObject_IsInstance(value, py_class.get());
    if (is_instance == 1) {
      if (IsRegisteredDispatchableType(py_class.get())) {
        result = MatchType::MATCH_DISPATCHABLE;
        break;
      } else {
        result = MatchType::MATCH;
      }
    } else if (is_instance < 0) {
      PyErr_Clear();
      return MatchType::NO_MATCH;
    }
  }

  if (py_class_cache_.size() < kMaxItemsInCache) {
    Py_INCREF(type);
    auto insert_result = py_class_cache_.insert({type, result});
    if (!insert_result.second) {
      Py_DECREF(type);  // Result was added by a different thread.
    }
  }
  return result;
}

int PyInstanceChecker::cost() const {
   std::vector<std::string> mht_15_v;
   MHTracer_DTPStensorflowPSpythonPSframeworkPSpython_api_dispatcherDTcc mht_15(mht_15_v, 475, "", "./tensorflow/python/framework/python_api_dispatcher.cc", "PyInstanceChecker::cost");
 return py_classes_.size(); }

std::string PyInstanceChecker::DebugString() const {
   std::vector<std::string> mht_16_v;
   MHTracer_DTPStensorflowPSpythonPSframeworkPSpython_api_dispatcherDTcc mht_16(mht_16_v, 480, "", "./tensorflow/python/framework/python_api_dispatcher.cc", "PyInstanceChecker::DebugString");

  DCheckPyGilState();
  std::vector<const char*> type_names;
  for (const auto& py_class : py_classes_) {
    type_names.push_back(
        reinterpret_cast<PyTypeObject*>(py_class.get())->tp_name);
  }
  return absl::StrJoin(
      py_classes_, ", ", [](std::string* out, const Safe_PyObjectPtr& v) {
        out->append(reinterpret_cast<PyTypeObject*>(v.get())->tp_name);
      });
}

PyTypeChecker::MatchType PyListChecker::Check(PyObject* value) {
   std::vector<std::string> mht_17_v;
   MHTracer_DTPStensorflowPSpythonPSframeworkPSpython_api_dispatcherDTcc mht_17(mht_17_v, 496, "", "./tensorflow/python/framework/python_api_dispatcher.cc", "PyListChecker::Check");

  DCheckPyGilState();
  if (!(PyList_Check(value) || PyTuple_Check(value))) {
    return MatchType::NO_MATCH;
  }

  Safe_PyObjectPtr seq(PySequence_Fast(value, ""));
  if (!seq) {
    PyErr_Clear();
    return MatchType::NO_MATCH;  // value is not a sequence.
  }

  MatchType result = MatchType::MATCH;
  for (int i = 0; i < PySequence_Fast_GET_SIZE(seq.get()); ++i) {
    switch (element_type_->Check(PySequence_Fast_GET_ITEM(seq.get(), i))) {
      case MatchType::NO_MATCH:
        return MatchType::NO_MATCH;
      case MatchType::MATCH_DISPATCHABLE:
        result = MatchType::MATCH_DISPATCHABLE;
        break;
      case MatchType::MATCH:
        break;
    }
  }
  return result;
}

int PyListChecker::cost() const {
   std::vector<std::string> mht_18_v;
   MHTracer_DTPStensorflowPSpythonPSframeworkPSpython_api_dispatcherDTcc mht_18(mht_18_v, 526, "", "./tensorflow/python/framework/python_api_dispatcher.cc", "PyListChecker::cost");
 return 10 * element_type_->cost(); }

std::string PyListChecker::DebugString() const {
   std::vector<std::string> mht_19_v;
   MHTracer_DTPStensorflowPSpythonPSframeworkPSpython_api_dispatcherDTcc mht_19(mht_19_v, 531, "", "./tensorflow/python/framework/python_api_dispatcher.cc", "PyListChecker::DebugString");

  return absl::StrCat("List[", element_type_->DebugString(), "]");
}

PyTypeChecker::MatchType PyUnionChecker::Check(PyObject* value) {
   std::vector<std::string> mht_20_v;
   MHTracer_DTPStensorflowPSpythonPSframeworkPSpython_api_dispatcherDTcc mht_20(mht_20_v, 538, "", "./tensorflow/python/framework/python_api_dispatcher.cc", "PyUnionChecker::Check");

  MatchType result = MatchType::NO_MATCH;
  for (auto& type_option : options_) {
    switch (type_option->Check(value)) {
      case MatchType::MATCH:
        result = MatchType::MATCH;
        break;
      case MatchType::MATCH_DISPATCHABLE:
        return MatchType::MATCH_DISPATCHABLE;
      case MatchType::NO_MATCH:
        break;
    }
  }
  return result;
}

int PyUnionChecker::cost() const {
   std::vector<std::string> mht_21_v;
   MHTracer_DTPStensorflowPSpythonPSframeworkPSpython_api_dispatcherDTcc mht_21(mht_21_v, 557, "", "./tensorflow/python/framework/python_api_dispatcher.cc", "PyUnionChecker::cost");

  int cost = 1;
  for (auto& type_option : options_) {
    cost += type_option->cost();
  }
  return cost;
}

std::string PyUnionChecker::DebugString() const {
   std::vector<std::string> mht_22_v;
   MHTracer_DTPStensorflowPSpythonPSframeworkPSpython_api_dispatcherDTcc mht_22(mht_22_v, 568, "", "./tensorflow/python/framework/python_api_dispatcher.cc", "PyUnionChecker::DebugString");

  return absl::StrCat("Union[",
                      absl::StrJoin(options_, ", ",
                                    [](std::string* out, PyTypeChecker_ptr v) {
                                      out->append(v->DebugString());
                                    }),
                      "]");
}

}  // namespace py_dispatch
}  // namespace tensorflow
