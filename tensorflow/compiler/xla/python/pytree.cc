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
class MHTracer_DTPStensorflowPScompilerPSxlaPSpythonPSpytreeDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSxlaPSpythonPSpytreeDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSxlaPSpythonPSpytreeDTcc() {
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

// Caution: this code uses exceptions. The exception use is local to the
// binding code and the idiomatic way to emit Python exceptions.

#include "tensorflow/compiler/xla/python/pytree.h"

#include <algorithm>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_map.h"
#include "absl/hash/hash.h"
#include "absl/memory/memory.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "pybind11/pybind11.h"
#include "pybind11/pytypes.h"
#include "pybind11/stl.h"
#include "pybind11_abseil/absl_casters.h"  // from @pybind11_abseil
#include "tensorflow/core/platform/logging.h"

namespace xla {

namespace py = pybind11;

/*static*/ PyTreeTypeRegistry* PyTreeTypeRegistry::Singleton() {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpythonPSpytreeDTcc mht_0(mht_0_v, 213, "", "./tensorflow/compiler/xla/python/pytree.cc", "PyTreeTypeRegistry::Singleton");

  static auto* registry = []() -> PyTreeTypeRegistry* {
    auto* registry = new PyTreeTypeRegistry;

    auto add_builtin_type = [&](PyTypeObject* type_obj, PyTreeKind kind) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpythonPSpytreeDTcc mht_1(mht_1_v, 220, "", "./tensorflow/compiler/xla/python/pytree.cc", "lambda");

      py::object type = py::reinterpret_borrow<py::object>(
          reinterpret_cast<PyObject*>(type_obj));
      auto registration = absl::make_unique<Registration>();
      registration->kind = kind;
      registration->type = type;
      CHECK(registry->registrations_.emplace(type, std::move(registration))
                .second);
    };
    add_builtin_type(Py_TYPE(Py_None), PyTreeKind::kNone);
    add_builtin_type(&PyTuple_Type, PyTreeKind::kTuple);
    add_builtin_type(&PyList_Type, PyTreeKind::kList);
    add_builtin_type(&PyDict_Type, PyTreeKind::kDict);
    return registry;
  }();
  return registry;
}

/*static*/ void PyTreeTypeRegistry::Register(py::object type,
                                             py::function to_iterable,
                                             py::function from_iterable) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpythonPSpytreeDTcc mht_2(mht_2_v, 243, "", "./tensorflow/compiler/xla/python/pytree.cc", "PyTreeTypeRegistry::Register");

  PyTreeTypeRegistry* registry = Singleton();
  auto registration = absl::make_unique<Registration>();
  registration->kind = PyTreeKind::kCustom;
  registration->type = type;
  registration->to_iterable = std::move(to_iterable);
  registration->from_iterable = std::move(from_iterable);
  auto it = registry->registrations_.emplace(type, std::move(registration));
  if (!it.second) {
    throw std::invalid_argument(
        absl::StrFormat("Duplicate custom PyTreeDef type registration for %s.",
                        py::repr(type)));
  }
}

/*static*/ const PyTreeTypeRegistry::Registration* PyTreeTypeRegistry::Lookup(
    py::handle type) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpythonPSpytreeDTcc mht_3(mht_3_v, 262, "", "./tensorflow/compiler/xla/python/pytree.cc", "PyTreeTypeRegistry::Lookup");

  PyTreeTypeRegistry* registry = Singleton();
  auto it = registry->registrations_.find(type);
  return it == registry->registrations_.end() ? nullptr : it->second.get();
}

bool PyTreeDef::operator==(const PyTreeDef& other) const {
  if (traversal_.size() != other.traversal_.size()) {
    return false;
  }
  for (size_t i = 0; i < traversal_.size(); ++i) {
    const Node& a = traversal_[i];
    const Node& b = other.traversal_[i];
    if (a.kind != b.kind || a.arity != b.arity ||
        (a.node_data.ptr() == nullptr) != (b.node_data.ptr() == nullptr) ||
        a.custom != b.custom) {
      return false;
    }
    if (a.node_data && a.node_data.not_equal(b.node_data)) {
      return false;
    }
    // We don't need to test equality of num_leaves and num_nodes since they
    // are derivable from the other node data.
  }
  return true;
}

/*static*/ PyTreeKind PyTreeDef::GetKind(
    const py::handle& obj, PyTreeTypeRegistry::Registration const** custom) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpythonPSpytreeDTcc mht_4(mht_4_v, 293, "", "./tensorflow/compiler/xla/python/pytree.cc", "PyTreeDef::GetKind");

  const PyTreeTypeRegistry::Registration* registration =
      PyTreeTypeRegistry::Lookup(obj.get_type());
  if (registration) {
    if (registration->kind == PyTreeKind::kCustom) {
      *custom = registration;
    } else {
      *custom = nullptr;
    }
    return registration->kind;
  } else if (py::isinstance<py::tuple>(obj) && py::hasattr(obj, "_fields")) {
    // We can only identify namedtuples heuristically, here by the presence of
    // a _fields attribute.
    return PyTreeKind::kNamedTuple;
  } else {
    return PyTreeKind::kLeaf;
  }
}

template <typename T>
void PyTreeDef::FlattenIntoImpl(
    py::handle handle, T& leaves,
    const absl::optional<py::function>& leaf_predicate) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpythonPSpytreeDTcc mht_5(mht_5_v, 318, "", "./tensorflow/compiler/xla/python/pytree.cc", "PyTreeDef::FlattenIntoImpl");

  Node node;
  int start_num_nodes = traversal_.size();
  int start_num_leaves = leaves.size();
  if (leaf_predicate && (*leaf_predicate)(handle).cast<bool>()) {
    leaves.push_back(py::reinterpret_borrow<py::object>(handle));
  } else {
    node.kind = GetKind(handle, &node.custom);
    auto recurse = [this, &leaf_predicate, &leaves](py::handle child) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpythonPSpytreeDTcc mht_6(mht_6_v, 329, "", "./tensorflow/compiler/xla/python/pytree.cc", "lambda");

      FlattenInto(child, leaves, leaf_predicate);
    };
    switch (node.kind) {
      case PyTreeKind::kNone:
        // Nothing to do.
        break;
      case PyTreeKind::kTuple: {
        node.arity = PyTuple_GET_SIZE(handle.ptr());
        for (int i = 0; i < node.arity; ++i) {
          recurse(PyTuple_GET_ITEM(handle.ptr(), i));
        }
        break;
      }
      case PyTreeKind::kList: {
        node.arity = PyList_GET_SIZE(handle.ptr());
        for (int i = 0; i < node.arity; ++i) {
          recurse(PyList_GET_ITEM(handle.ptr(), i));
        }
        break;
      }
      case PyTreeKind::kDict: {
        py::dict dict = py::reinterpret_borrow<py::dict>(handle);
        py::list keys =
            py::reinterpret_steal<py::list>(PyDict_Keys(dict.ptr()));
        if (PyList_Sort(keys.ptr())) {
          throw py::error_already_set();
        }
        for (py::handle key : keys) {
          recurse(dict[key]);
        }
        node.arity = dict.size();
        node.node_data = std::move(keys);
        break;
      }
      case PyTreeKind::kCustom: {
        py::tuple out = py::cast<py::tuple>(node.custom->to_iterable(handle));
        if (out.size() != 2) {
          throw std::runtime_error(
              "PyTree custom to_iterable function should return a pair");
        }
        node.node_data = out[1];
        node.arity = 0;
        for (py::handle entry : py::cast<py::iterable>(out[0])) {
          ++node.arity;
          recurse(entry);
        }
        break;
      }
      case PyTreeKind::kNamedTuple: {
        py::tuple tuple = py::reinterpret_borrow<py::tuple>(handle);
        node.arity = tuple.size();
        node.node_data = py::reinterpret_borrow<py::object>(tuple.get_type());
        for (py::handle entry : tuple) {
          recurse(entry);
        }
        break;
      }
      default:
        DCHECK(node.kind == PyTreeKind::kLeaf);
        leaves.push_back(py::reinterpret_borrow<py::object>(handle));
    }
  }
  node.num_nodes = traversal_.size() - start_num_nodes + 1;
  node.num_leaves = leaves.size() - start_num_leaves;
  traversal_.push_back(std::move(node));
}

void PyTreeDef::FlattenInto(py::handle handle,
                            absl::InlinedVector<py::object, 2>& leaves,
                            absl::optional<py::function> leaf_predicate) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpythonPSpytreeDTcc mht_7(mht_7_v, 402, "", "./tensorflow/compiler/xla/python/pytree.cc", "PyTreeDef::FlattenInto");

  FlattenIntoImpl(handle, leaves, leaf_predicate);
}

void PyTreeDef::FlattenInto(py::handle handle, std::vector<py::object>& leaves,
                            absl::optional<py::function> leaf_predicate) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpythonPSpytreeDTcc mht_8(mht_8_v, 410, "", "./tensorflow/compiler/xla/python/pytree.cc", "PyTreeDef::FlattenInto");

  FlattenIntoImpl(handle, leaves, leaf_predicate);
}

/*static*/ std::pair<std::vector<py::object>, std::unique_ptr<PyTreeDef>>
PyTreeDef::Flatten(py::handle x, absl::optional<py::function> leaf_predicate) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpythonPSpytreeDTcc mht_9(mht_9_v, 418, "", "./tensorflow/compiler/xla/python/pytree.cc", "PyTreeDef::Flatten");

  std::vector<py::object> leaves;
  auto tree = absl::make_unique<PyTreeDef>();
  tree->FlattenInto(x, leaves, leaf_predicate);
  return std::make_pair(std::move(leaves), std::move(tree));
}

/*static*/ bool PyTreeDef::AllLeaves(const py::iterable& x) {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpythonPSpytreeDTcc mht_10(mht_10_v, 428, "", "./tensorflow/compiler/xla/python/pytree.cc", "PyTreeDef::AllLeaves");

  const PyTreeTypeRegistry::Registration* custom;
  for (const py::handle& h : x) {
    if (GetKind(h, &custom) != PyTreeKind::kLeaf) return false;
  }
  return true;
}

template <typename T>
py::object PyTreeDef::UnflattenImpl(T leaves) const {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpythonPSpytreeDTcc mht_11(mht_11_v, 440, "", "./tensorflow/compiler/xla/python/pytree.cc", "PyTreeDef::UnflattenImpl");

  absl::InlinedVector<py::object, 4> agenda;
  auto it = leaves.begin();
  int leaf_count = 0;
  for (const Node& node : traversal_) {
    if (agenda.size() < node.arity) {
      throw std::logic_error("Too few elements for TreeDef node.");
    }
    switch (node.kind) {
      case PyTreeKind::kLeaf:
        if (it == leaves.end()) {
          throw std::invalid_argument(absl::StrFormat(
              "Too few leaves for PyTreeDef; expected %d, got %d", num_leaves(),
              leaf_count));
        }
        agenda.push_back(py::reinterpret_borrow<py::object>(*it));
        ++it;
        ++leaf_count;
        break;

      case PyTreeKind::kNone:
      case PyTreeKind::kTuple:
      case PyTreeKind::kNamedTuple:
      case PyTreeKind::kList:
      case PyTreeKind::kDict:
      case PyTreeKind::kCustom: {
        const int size = agenda.size();
        absl::Span<py::object> span;
        if (node.arity > 0) {
          span = absl::Span<py::object>(&agenda[size - node.arity], node.arity);
        }
        py::object o = MakeNode(node, span);
        agenda.resize(size - node.arity);
        agenda.push_back(o);
        break;
      }
    }
  }
  if (it != leaves.end()) {
    throw std::invalid_argument(absl::StrFormat(
        "Too many leaves for PyTreeDef; expected %d.", num_leaves()));
  }
  if (agenda.size() != 1) {
    throw std::logic_error("PyTreeDef traversal did not yield a singleton.");
  }
  return std::move(agenda.back());
}

py::object PyTreeDef::Unflatten(py::iterable leaves) const {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpythonPSpytreeDTcc mht_12(mht_12_v, 491, "", "./tensorflow/compiler/xla/python/pytree.cc", "PyTreeDef::Unflatten");

  return UnflattenImpl(leaves);
}

py::object PyTreeDef::Unflatten(absl::Span<const py::object> leaves) const {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpythonPSpytreeDTcc mht_13(mht_13_v, 498, "", "./tensorflow/compiler/xla/python/pytree.cc", "PyTreeDef::Unflatten");

  return UnflattenImpl(leaves);
}

/*static*/ py::object PyTreeDef::MakeNode(const PyTreeDef::Node& node,
                                          absl::Span<py::object> children) {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpythonPSpytreeDTcc mht_14(mht_14_v, 506, "", "./tensorflow/compiler/xla/python/pytree.cc", "PyTreeDef::MakeNode");

  if (children.size() != node.arity) {
    throw std::logic_error("Node arity mismatch.");
  }
  switch (node.kind) {
    case PyTreeKind::kLeaf:
      throw std::logic_error("MakeNode not implemented for leaves.");

    case PyTreeKind::kNone:
      return py::none();

    case PyTreeKind::kTuple:
    case PyTreeKind::kNamedTuple: {
      py::tuple tuple(node.arity);
      for (int i = 0; i < node.arity; ++i) {
        tuple[i] = std::move(children[i]);
      }
      if (node.kind == PyTreeKind::kNamedTuple) {
        return node.node_data(*tuple);
      } else {
        return std::move(tuple);
      }
    }

    case PyTreeKind::kList: {
      py::list list(node.arity);
      for (int i = 0; i < node.arity; ++i) {
        list[i] = std::move(children[i]);
      }
      return std::move(list);
    }

    case PyTreeKind::kDict: {
      py::dict dict;
      py::list keys = py::reinterpret_borrow<py::list>(node.node_data);
      for (int i = 0; i < node.arity; ++i) {
        dict[keys[i]] = std::move(children[i]);
      }
      return std::move(dict);
      break;
    }
    case PyTreeKind::kCustom: {
      py::tuple tuple(node.arity);
      for (int i = 0; i < node.arity; ++i) {
        tuple[i] = std::move(children[i]);
      }
      return node.custom->from_iterable(node.node_data, tuple);
    }
  }
  throw std::logic_error("Unreachable code.");
}

py::list PyTreeDef::FlattenUpTo(py::handle xs) const {
   std::vector<std::string> mht_15_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpythonPSpytreeDTcc mht_15(mht_15_v, 561, "", "./tensorflow/compiler/xla/python/pytree.cc", "PyTreeDef::FlattenUpTo");

  py::list leaves(num_leaves());
  std::vector<py::object> agenda;
  agenda.push_back(py::reinterpret_borrow<py::object>(xs));
  auto it = traversal_.rbegin();
  int leaf = num_leaves() - 1;
  while (!agenda.empty()) {
    if (it == traversal_.rend()) {
      throw std::invalid_argument(absl::StrFormat(
          "Tree structures did not match: %s vs %s", py::repr(xs), ToString()));
    }
    const Node& node = *it;
    py::object object = agenda.back();
    agenda.pop_back();
    ++it;

    switch (node.kind) {
      case PyTreeKind::kLeaf:
        if (leaf < 0) {
          throw std::logic_error("Leaf count mismatch.");
        }
        leaves[leaf] = py::reinterpret_borrow<py::object>(object);
        --leaf;
        break;

      case PyTreeKind::kNone:
        break;

      case PyTreeKind::kTuple: {
        if (!PyTuple_CheckExact(object.ptr())) {
          throw std::invalid_argument(
              absl::StrFormat("Expected tuple, got %s.", py::repr(object)));
        }
        py::tuple tuple = py::reinterpret_borrow<py::tuple>(object);
        if (tuple.size() != node.arity) {
          throw std::invalid_argument(
              absl::StrFormat("Tuple arity mismatch: %d != %d; tuple: %s.",
                              tuple.size(), node.arity, py::repr(object)));
        }
        for (py::handle entry : tuple) {
          agenda.push_back(py::reinterpret_borrow<py::object>(entry));
        }
        break;
      }

      case PyTreeKind::kList: {
        if (!PyList_CheckExact(object.ptr())) {
          throw std::invalid_argument(
              absl::StrFormat("Expected list, got %s.", py::repr(object)));
        }
        py::list list = py::reinterpret_borrow<py::list>(object);
        if (list.size() != node.arity) {
          throw std::invalid_argument(
              absl::StrFormat("List arity mismatch: %d != %d; list: %s.",
                              list.size(), node.arity, py::repr(object)));
        }
        for (py::handle entry : list) {
          agenda.push_back(py::reinterpret_borrow<py::object>(entry));
        }
        break;
      }

      case PyTreeKind::kDict: {
        if (!PyDict_CheckExact(object.ptr())) {
          throw std::invalid_argument(
              absl::StrFormat("Expected dict, got %s.", py::repr(object)));
        }
        py::dict dict = py::reinterpret_borrow<py::dict>(object);
        py::list keys =
            py::reinterpret_steal<py::list>(PyDict_Keys(dict.ptr()));
        if (PyList_Sort(keys.ptr())) {
          throw std::runtime_error("Dictionary key sort failed.");
        }
        if (keys.not_equal(node.node_data)) {
          throw std::invalid_argument(
              absl::StrFormat("Dict key mismatch; expected keys: %s; dict: %s.",
                              py::repr(node.node_data), py::repr(object)));
        }
        for (py::handle key : keys) {
          agenda.push_back(dict[key]);
        }
        break;
      }

      case PyTreeKind::kNamedTuple: {
        if (!py::isinstance<py::tuple>(object) ||
            !py::hasattr(object, "_fields")) {
          throw std::invalid_argument(absl::StrFormat(
              "Expected named tuple, got %s.", py::repr(object)));
        }
        py::tuple tuple = py::reinterpret_borrow<py::tuple>(object);
        if (tuple.size() != node.arity) {
          throw std::invalid_argument(absl::StrFormat(
              "Named tuple arity mismatch: %d != %d; tuple: %s.", tuple.size(),
              node.arity, py::repr(object)));
        }
        if (tuple.get_type().not_equal(node.node_data)) {
          throw std::invalid_argument(absl::StrFormat(
              "Named tuple type mismatch: expected type: %s, tuple: %s.",
              py::repr(node.node_data), py::repr(object)));
        }
        for (py::handle entry : tuple) {
          agenda.push_back(py::reinterpret_borrow<py::object>(entry));
        }
        break;
      }

      case PyTreeKind::kCustom: {
        auto* registration = PyTreeTypeRegistry::Lookup(object.get_type());
        if (registration != node.custom) {
          throw std::invalid_argument(absl::StrFormat(
              "Custom node type mismatch: expected type: %s, value: %s.",
              py::repr(node.custom->type), py::repr(object)));
        }
        py::tuple out = py::cast<py::tuple>(node.custom->to_iterable(object));
        if (out.size() != 2) {
          throw std::runtime_error(
              "PyTree custom to_iterable function should return a pair");
        }
        if (node.node_data.not_equal(out[1])) {
          throw std::invalid_argument(absl::StrFormat(
              "Mismatch custom node data: %s != %s; value: %s.",
              py::repr(node.node_data), py::repr(out[1]), py::repr(object)));
        }
        int arity = 0;
        for (py::handle entry : py::cast<py::iterable>(out[0])) {
          ++arity;
          agenda.push_back(py::reinterpret_borrow<py::object>(entry));
        }
        if (arity != node.arity) {
          throw std::invalid_argument(absl::StrFormat(
              "Custom type arity mismatch: %d != %d; value: %s.", arity,
              node.arity, py::repr(object)));
        }
        break;
      }
    }
  }
  if (it != traversal_.rend() || leaf != -1) {
    throw std::invalid_argument(absl::StrFormat(
        "Tree structures did not match: %s vs %s", py::repr(xs), ToString()));
  }
  return leaves;
}

py::object PyTreeDef::Walk(const py::function& f_node, py::handle f_leaf,
                           py::iterable leaves) const {
   std::vector<std::string> mht_16_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpythonPSpytreeDTcc mht_16(mht_16_v, 710, "", "./tensorflow/compiler/xla/python/pytree.cc", "PyTreeDef::Walk");

  std::vector<py::object> agenda;
  auto it = leaves.begin();
  for (const Node& node : traversal_) {
    switch (node.kind) {
      case PyTreeKind::kLeaf: {
        if (it == leaves.end()) {
          throw std::invalid_argument("Too few leaves for PyTreeDef");
        }

        py::object leaf = py::reinterpret_borrow<py::object>(*it);
        agenda.push_back(f_leaf.is_none() ? std::move(leaf)
                                          : f_leaf(std::move(leaf)));
        ++it;
        break;
      }

      case PyTreeKind::kNone:
      case PyTreeKind::kTuple:
      case PyTreeKind::kNamedTuple:
      case PyTreeKind::kList:
      case PyTreeKind::kDict:
      case PyTreeKind::kCustom: {
        if (agenda.size() < node.arity) {
          throw std::logic_error("Too few elements for custom type.");
        }
        py::tuple tuple(node.arity);
        for (int i = node.arity - 1; i >= 0; --i) {
          tuple[i] = agenda.back();
          agenda.pop_back();
        }
        agenda.push_back(f_node(tuple));
      }
    }
  }
  if (it != leaves.end()) {
    throw std::invalid_argument("Too many leaves for PyTreeDef");
  }
  if (agenda.size() != 1) {
    throw std::logic_error("PyTreeDef traversal did not yield a singleton.");
  }
  return std::move(agenda.back());
}

py::object PyTreeDef::FromIterableTreeHelper(
    py::handle xs,
    absl::InlinedVector<PyTreeDef::Node, 1>::const_reverse_iterator* it) const {
   std::vector<std::string> mht_17_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpythonPSpytreeDTcc mht_17(mht_17_v, 759, "", "./tensorflow/compiler/xla/python/pytree.cc", "PyTreeDef::FromIterableTreeHelper");

  if (*it == traversal_.rend()) {
    throw std::invalid_argument("Tree structures did not match.");
  }
  const Node& node = **it;
  ++*it;
  if (node.kind == PyTreeKind::kLeaf) {
    return py::reinterpret_borrow<py::object>(xs);
  }
  py::iterable iterable = py::reinterpret_borrow<py::iterable>(xs);
  std::vector<py::object> ys;
  ys.reserve(node.arity);
  for (py::handle x : iterable) {
    ys.push_back(py::reinterpret_borrow<py::object>(x));
  }
  if (ys.size() != node.arity) {
    throw std::invalid_argument("Arity mismatch between trees");
  }
  for (int j = node.arity - 1; j >= 0; --j) {
    ys[j] = FromIterableTreeHelper(ys[j], it);
  }

  return MakeNode(node, absl::MakeSpan(ys));
}

py::object PyTreeDef::FromIterableTree(py::handle xs) const {
   std::vector<std::string> mht_18_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpythonPSpytreeDTcc mht_18(mht_18_v, 787, "", "./tensorflow/compiler/xla/python/pytree.cc", "PyTreeDef::FromIterableTree");

  auto it = traversal_.rbegin();
  py::object out = FromIterableTreeHelper(xs, &it);
  if (it != traversal_.rend()) {
    throw std::invalid_argument("Tree structures did not match.");
  }
  return out;
}

std::unique_ptr<PyTreeDef> PyTreeDef::Compose(const PyTreeDef& inner) const {
   std::vector<std::string> mht_19_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpythonPSpytreeDTcc mht_19(mht_19_v, 799, "", "./tensorflow/compiler/xla/python/pytree.cc", "PyTreeDef::Compose");

  auto out = absl::make_unique<PyTreeDef>();
  for (const Node& n : traversal_) {
    if (n.kind == PyTreeKind::kLeaf) {
      absl::c_copy(inner.traversal_, std::back_inserter(out->traversal_));
    } else {
      out->traversal_.push_back(n);
    }
  }
  const auto& root = traversal_.back();
  const auto& inner_root = inner.traversal_.back();
  // TODO(tomhennigan): This should update all nodes in the traversal.
  auto& out_root = out->traversal_.back();
  out_root.num_nodes = (root.num_nodes - root.num_leaves) +
                       (inner_root.num_nodes * root.num_leaves);
  out_root.num_leaves *= inner_root.num_leaves;
  return out;
}

/*static*/ std::unique_ptr<PyTreeDef> PyTreeDef::Tuple(
    const std::vector<PyTreeDef>& defs) {
   std::vector<std::string> mht_20_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpythonPSpytreeDTcc mht_20(mht_20_v, 822, "", "./tensorflow/compiler/xla/python/pytree.cc", "PyTreeDef::Tuple");

  auto out = absl::make_unique<PyTreeDef>();
  int num_leaves = 0;
  for (const PyTreeDef& def : defs) {
    absl::c_copy(def.traversal_, std::back_inserter(out->traversal_));
    num_leaves += def.num_leaves();
  }
  Node node;
  node.kind = PyTreeKind::kTuple;
  node.arity = defs.size();
  node.num_leaves = num_leaves;
  node.num_nodes = out->traversal_.size() + 1;
  out->traversal_.push_back(node);
  return out;
}

std::vector<std::unique_ptr<PyTreeDef>> PyTreeDef::Children() const {
   std::vector<std::string> mht_21_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpythonPSpytreeDTcc mht_21(mht_21_v, 841, "", "./tensorflow/compiler/xla/python/pytree.cc", "PyTreeDef::Children");

  std::vector<std::unique_ptr<PyTreeDef>> children;
  if (traversal_.empty()) {
    return children;
  }
  Node const& root = traversal_.back();
  children.resize(root.arity);
  int pos = traversal_.size() - 1;
  for (int i = root.arity - 1; i >= 0; --i) {
    children[i] = absl::make_unique<PyTreeDef>();
    const Node& node = traversal_.at(pos - 1);
    if (pos < node.num_nodes) {
      throw std::logic_error("children() walked off start of array");
    }
    std::copy(traversal_.begin() + pos - node.num_nodes,
              traversal_.begin() + pos,
              std::back_inserter(children[i]->traversal_));
    pos -= node.num_nodes;
  }
  if (pos != 0) {
    throw std::logic_error("pos != 0 at end of PyTreeDef::Children");
  }
  return children;
}

std::string PyTreeDef::ToString() const {
   std::vector<std::string> mht_22_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpythonPSpytreeDTcc mht_22(mht_22_v, 869, "", "./tensorflow/compiler/xla/python/pytree.cc", "PyTreeDef::ToString");

  std::vector<std::string> agenda;
  for (const Node& node : traversal_) {
    if (agenda.size() < node.arity) {
      throw std::logic_error("Too few elements for container.");
    }

    std::string children =
        absl::StrJoin(agenda.end() - node.arity, agenda.end(), ", ");
    std::string representation;
    switch (node.kind) {
      case PyTreeKind::kLeaf:
        agenda.push_back("*");
        continue;
      case PyTreeKind::kNone:
        representation = "None";
        break;
      case PyTreeKind::kTuple:
        // Tuples with only one element must have a trailing comma.
        if (node.arity == 1) children += ",";
        representation = absl::StrCat("(", children, ")");
        break;
      case PyTreeKind::kList:
        representation = absl::StrCat("[", children, "]");
        break;
      case PyTreeKind::kDict: {
        if (py::len(node.node_data) != node.arity) {
          throw std::logic_error("Number of keys and entries does not match.");
        }
        representation = "{";
        std::string separator;
        auto child_iter = agenda.end() - node.arity;
        for (const py::handle& key : node.node_data) {
          absl::StrAppendFormat(&representation, "%s%s: %s", separator,
                                py::repr(key), *child_iter);
          child_iter++;
          separator = ", ";
        }
        representation += "}";
        break;
      }

      case PyTreeKind::kNamedTuple:
      case PyTreeKind::kCustom: {
        std::string kind;
        if (node.kind == PyTreeKind::kNamedTuple) {
          kind = "namedtuple";
        } else {
          kind = static_cast<std::string>(py::str(node.custom->type));
        }

        std::string data;
        if (node.node_data) {
          data = absl::StrFormat("[%s]", py::str(node.node_data));
        }
        representation =
            absl::StrFormat("CustomNode(%s%s, [%s])", kind, data, children);
        break;
      }
    }
    agenda.erase(agenda.end() - node.arity, agenda.end());
    agenda.push_back(std::move(representation));
  }
  if (agenda.size() != 1) {
    throw std::logic_error("PyTreeDef traversal did not yield a singleton.");
  }
  return absl::StrCat("PyTreeDef(", agenda.back(), ")");
}

void BuildPytreeSubmodule(py::module& m) {
   std::vector<std::string> mht_23_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpythonPSpytreeDTcc mht_23(mht_23_v, 941, "", "./tensorflow/compiler/xla/python/pytree.cc", "BuildPytreeSubmodule");

  py::module pytree = m.def_submodule("pytree", "Python tree library");
  pytree.attr("version") = py::int_(1);
  pytree.def("flatten", &PyTreeDef::Flatten, py::arg("tree"),
             py::arg("leaf_predicate") = absl::nullopt);
  pytree.def("tuple", &PyTreeDef::Tuple);
  pytree.def("all_leaves", &PyTreeDef::AllLeaves);

  py::class_<PyTreeDef>(m, "PyTreeDef")
      .def("unflatten",
           static_cast<pybind11::object (PyTreeDef::*)(
               pybind11::iterable leaves) const>(&PyTreeDef::Unflatten))
      .def("flatten_up_to", &PyTreeDef::FlattenUpTo)
      .def("compose", &PyTreeDef::Compose)
      .def("walk", &PyTreeDef::Walk)
      .def("from_iterable_tree", &PyTreeDef::FromIterableTree)
      .def("children", &PyTreeDef::Children)
      .def_property_readonly("num_leaves", &PyTreeDef::num_leaves)
      .def_property_readonly("num_nodes", &PyTreeDef::num_nodes)
      .def("__repr__", &PyTreeDef::ToString)
      .def("__eq__",
           [](const PyTreeDef& a, const PyTreeDef& b) { return a == b; })
      .def("__ne__",
           [](const PyTreeDef& a, const PyTreeDef& b) { return a != b; })
      .def("__hash__", [](const PyTreeDef& t) { return absl::HashOf(t); });

  pytree.def("register_node", [](py::object type, py::function to_iterable,
                                 py::function from_iterable) {
    return PyTreeTypeRegistry::Register(type, to_iterable, from_iterable);
  });
}

}  // namespace xla
