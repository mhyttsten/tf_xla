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

#ifndef TENSORFLOW_COMPILER_XLA_PYTHON_PYTREE_H_
#define TENSORFLOW_COMPILER_XLA_PYTHON_PYTREE_H_
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
class MHTracer_DTPStensorflowPScompilerPSxlaPSpythonPSpytreeDTh {
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
   MHTracer_DTPStensorflowPScompilerPSxlaPSpythonPSpytreeDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSxlaPSpythonPSpytreeDTh() {
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


// See https://jax.readthedocs.io/en/latest/pytrees.html for the documentation
// about pytree.

// Caution: this code uses exceptions. The exception use is local to the
// binding code and the idiomatic way to emit Python exceptions.

#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/container/inlined_vector.h"
#include "absl/hash/hash.h"
#include "absl/memory/memory.h"
#include "pybind11/pybind11.h"
#include "pybind11/pytypes.h"
#include "pybind11/stl.h"

namespace xla {

enum class PyTreeKind {
  kLeaf,        // An opaque leaf node
  kNone,        // None.
  kTuple,       // A tuple
  kNamedTuple,  // A collections.namedtuple
  kList,        // A list
  kDict,        // A dict
  kCustom,      // A custom type.
};

// Registry of custom node types.
class PyTreeTypeRegistry {
 public:
  struct Registration {
    PyTreeKind kind;

    // The following values are populated for custom types.
    // The Python type object, used to identify the type.
    pybind11::object type;
    // A function with signature: object -> (iterable, aux_data)
    pybind11::function to_iterable;
    // A function with signature: (aux_data, iterable) -> object
    pybind11::function from_iterable;
  };

  // Registers a new custom type. Objects of `type` will be treated as container
  // node types in PyTrees.
  static void Register(pybind11::object type, pybind11::function to_iterable,
                       pybind11::function from_iterable);

  // Finds the custom type registration for `type`. Returns nullptr if none
  // exists.
  static const Registration* Lookup(pybind11::handle type);

 private:
  static PyTreeTypeRegistry* Singleton();

  struct TypeHash {
    using is_transparent = void;
    size_t operator()(const pybind11::object& t) const {
      return absl::HashOf(t.ptr());
    }
    size_t operator()(const pybind11::handle& t) const {
      return absl::HashOf(t.ptr());
    }
  };
  struct TypeEq {
    using is_transparent = void;
    bool operator()(const pybind11::object& a,
                    const pybind11::object& b) const {
      return a.ptr() == b.ptr();
    }
    bool operator()(const pybind11::object& a,
                    const pybind11::handle& b) const {
      return a.ptr() == b.ptr();
    }
  };
  absl::flat_hash_map<pybind11::object, std::unique_ptr<Registration>, TypeHash,
                      TypeEq>
      registrations_;
};

// A PyTreeDef describes the tree structure of a PyTree. A PyTree is a tree of
// Python values, where the interior nodes are tuples, lists, dictionaries, or
// user-defined containers, and the leaves are other objects.
class PyTreeDef {
 public:
  PyTreeDef() = default;

  // Flattens a Pytree into a list of leaves and a PyTreeDef.
  // Returns references to the flattened objects, which might be temporary
  // objects in the case of custom pytype handlers.
  static std::pair<std::vector<pybind11::object>, std::unique_ptr<PyTreeDef>>
  Flatten(pybind11::handle x,
          absl::optional<pybind11::function> leaf_predicate = absl::nullopt);

  // Recursive helper used to implement Flatten().
  void FlattenInto(
      pybind11::handle handle, std::vector<pybind11::object>& leaves,
      absl::optional<pybind11::function> leaf_predicate = absl::nullopt);
  void FlattenInto(
      pybind11::handle handle, absl::InlinedVector<pybind11::object, 2>& leaves,
      absl::optional<pybind11::function> leaf_predicate = absl::nullopt);

  // Tests whether the given list is a flat list of leaves.
  static bool AllLeaves(const pybind11::iterable& x);

  // Flattens a Pytree up to this PyTreeDef. 'this' must be a tree prefix of
  // the tree-structure of 'x'. For example, if we flatten a value
  // [(1, (2, 3)), {"foo": 4}] with a treedef [(*, *), *], the result is the
  // list of leaves [1, (2, 3), {"foo": 4}].
  pybind11::list FlattenUpTo(pybind11::handle x) const;

  // Returns an unflattened PyTree given an iterable of leaves and a PyTreeDef.
  pybind11::object Unflatten(pybind11::iterable leaves) const;
  pybind11::object Unflatten(absl::Span<const pybind11::object> leaves) const;

  // Composes two PyTreeDefs, replacing the leaves of this tree with copies of
  // `inner`.
  std::unique_ptr<PyTreeDef> Compose(const PyTreeDef& inner) const;

  // Makes a Tuple PyTreeDef out of a vector of PyTreeDefs.
  static std::unique_ptr<PyTreeDef> Tuple(const std::vector<PyTreeDef>& defs);

  std::vector<std::unique_ptr<PyTreeDef>> Children() const;

  // Maps a function over a PyTree structure, applying f_leaf to each leaf, and
  // f_node to each container node.
  // TODO(phawkins): use flattening everywhere instead and delete this method.
  pybind11::object Walk(const pybind11::function& f_node,
                        pybind11::handle f_leaf,
                        pybind11::iterable leaves) const;

  // Given a tree of iterables with the same node/leaf structure as this PyTree,
  // build the corresponding PyTree.
  // TODO(phawkins): use flattening everywhere instead and delete this method.
  pybind11::object FromIterableTree(pybind11::handle xs) const;

  int num_leaves() const {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpythonPSpytreeDTh mht_0(mht_0_v, 328, "", "./tensorflow/compiler/xla/python/pytree.h", "num_leaves");

    if (traversal_.empty()) {
      return 0;
    }
    return traversal_.back().num_leaves;
  }

  int num_nodes() const {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpythonPSpytreeDTh mht_1(mht_1_v, 338, "", "./tensorflow/compiler/xla/python/pytree.h", "num_nodes");
 return traversal_.size(); }

  size_t Hash() const;

  bool operator==(const PyTreeDef& other) const;
  bool operator!=(const PyTreeDef& other) const { return !(*this == other); }

  std::string ToString() const;

 private:
  struct Node {
    PyTreeKind kind = PyTreeKind::kLeaf;

    // Arity for non-kLeaf types.
    int arity = 0;

    // Kind-specific auxiliary data. For a kNamedTuple, contains the tuple type
    // object. For a kDict, contains a sorted list of keys. For a kCustom type,
    // contains the auxiliary data returned by the `to_iterable` function.
    pybind11::object node_data;

    // Custom type registration. Must be null for non-custom types.
    const PyTreeTypeRegistry::Registration* custom = nullptr;

    // Number of leaf nodes in the subtree rooted at this node.
    int num_leaves = 0;

    // Number of leaf and interior nodes in the subtree rooted at this node.
    int num_nodes = 0;
  };
  template <typename H>
  friend H AbslHashValue(H h, const Node& n);

  template <typename H>
  friend H AbslHashValue(H h, const PyTreeDef& t);

  // Helper that manufactures an instance of a node given its children.
  static pybind11::object MakeNode(const Node& node,
                                   absl::Span<pybind11::object> children);

  // Recursive helper used to implement FromIterableTree()
  pybind11::object FromIterableTreeHelper(
      pybind11::handle xs,
      absl::InlinedVector<PyTreeDef::Node, 1>::const_reverse_iterator* it)
      const;

  // Computes the node kind of a given Python object.
  static PyTreeKind GetKind(const pybind11::handle& obj,
                            PyTreeTypeRegistry::Registration const** custom);

  template <typename T>
  void FlattenIntoImpl(
      pybind11::handle handle, T& leaves,
      const absl::optional<pybind11::function>& leaf_predicate);

  template <typename T>
  pybind11::object UnflattenImpl(T leaves) const;

  // Nodes, in a post-order traversal. We use an ordered traversal to minimize
  // allocations, and post-order corresponds to the order we need to rebuild the
  // tree structure.
  absl::InlinedVector<Node, 1> traversal_;
};

template <typename H>
H AbslHashValue(H h, const PyTreeDef::Node& n) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpythonPSpytreeDTh mht_2(mht_2_v, 406, "", "./tensorflow/compiler/xla/python/pytree.h", "AbslHashValue");

  h = H::combine(std::move(h), n.kind, n.arity, n.custom);
  return h;
}

template <typename H>
H AbslHashValue(H h, const PyTreeDef& t) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpythonPSpytreeDTh mht_3(mht_3_v, 415, "", "./tensorflow/compiler/xla/python/pytree.h", "AbslHashValue");

  h = H::combine(std::move(h), t.traversal_);
  return h;
}

void BuildPytreeSubmodule(pybind11::module& m);

}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_PYTHON_PYTREE_H_
