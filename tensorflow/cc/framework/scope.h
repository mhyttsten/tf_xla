/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_CC_FRAMEWORK_SCOPE_H_
#define TENSORFLOW_CC_FRAMEWORK_SCOPE_H_
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
class MHTracer_DTPStensorflowPSccPSframeworkPSscopeDTh {
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
   MHTracer_DTPStensorflowPSccPSframeworkPSscopeDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSccPSframeworkPSscopeDTh() {
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


#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "absl/strings/str_cat.h"
#include "tensorflow/cc/framework/ops.h"
#include "tensorflow/core/common_runtime/graph_constructor.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/gtl/array_slice.h"

namespace tensorflow {

class Graph;
class GraphDef;
class NodeBuilder;
struct CompositeOpScopes;

/// @addtogroup core
/// @{

/// A `Scope` object represents a set of related TensorFlow ops that have the
/// same properties such as a common name prefix.
///
/// A Scope object is a container for TensorFlow Op properties. Op constructors
/// get a Scope object as a mandatory first argument and the constructed op
/// acquires the properties in the object.
///
/// A simple example:
///
///     using namespace ops;
///     Scope root = Scope::NewRootScope();
///     auto c1 = Const(root, { {1, 1} });
///     auto m = MatMul(root, c1, { {41}, {1} });
///     GraphDef gdef;
///     Status s = root.ToGraphDef(&gdef);
///     if (!s.ok()) { ... }
///
/// Scope hierarchy:
///
/// The Scope class provides various With<> functions that create a new scope.
/// The new scope typically has one property changed while other properties are
/// inherited from the parent scope.
/// NewSubScope(name) method appends `name` to the prefix of names for ops
/// created within the scope, and WithOpName() changes the suffix which
/// otherwise defaults to the type of the op.
///
/// Name examples:
///
///     Scope root = Scope::NewRootScope();
///     Scope linear = root.NewSubScope("linear");
///     // W will be named "linear/W"
///     auto W = Variable(linear.WithOpName("W"),
///                       {2, 2}, DT_FLOAT);
///     // b will be named "linear/b_3"
///     int idx = 3;
///     auto b = Variable(linear.WithOpName("b_", idx),
///                       {2}, DT_FLOAT);
///     auto x = Const(linear, {...});  // name: "linear/Const"
///     auto m = MatMul(linear, x, W);  // name: "linear/MatMul"
///     auto r = BiasAdd(linear, m, b); // name: "linear/BiasAdd"
///
/// Scope lifetime:
///
/// A new scope is created by calling Scope::NewRootScope. This creates some
/// resources that are shared by all the child scopes that inherit from this
/// scope, directly or transitively. For instance, a new scope creates a new
/// Graph object to which operations are added when the new scope or its
/// children are used by an Op constructor. The new scope also has a Status
/// object which will be used to indicate errors by Op-constructor functions
/// called on any child scope. The Op-constructor functions have to check the
/// scope's status by calling the ok() method before proceeding to construct the
/// op.
///
/// Thread safety:
///
/// A `Scope` object is NOT thread-safe. Threads cannot concurrently call
/// op-constructor functions on the same `Scope` object.
class Scope {
 public:
  Scope(const Scope& other);
  ~Scope();
  Scope& operator=(const Scope& other);

  // The following functions are for users making graphs. They return brand new
  // scopes, or scopes derived from an existing scope object.

  /// Return a new scope.
  /// This creates a new graph and all operations constructed in this graph
  /// should use the returned object as the "root" scope.
  static Scope NewRootScope();

  /// Return a new scope. Ops created with this scope will have
  /// `name/child_scope_name` as the prefix. The actual name will be unique
  /// in the current scope. All other properties are inherited from the current
  /// scope. If `child_scope_name` is empty, the `/` is elided.
  Scope NewSubScope(const string& child_scope_name) const;

  /// Return a new scope. All ops created within the returned scope will have
  /// names of the form `name/StrCat(fragments...)[_suffix]`
  template <typename... Ty>
  Scope WithOpName(Ty... fragments) const {
    return WithOpNameImpl(absl::StrCat(fragments...));
  }

  /// Return a new scope. All ops created within the returned scope will have as
  /// control dependencies the union of operations in the control_deps vector
  /// and the control dependencies of the current scope.
  Scope WithControlDependencies(
      const gtl::ArraySlice<Operation>& control_deps) const;
  /// Same as above, but convenient to add control dependency on the operation
  /// producing the control_dep output.
  Scope WithControlDependencies(const Output& control_dep) const;

  /// Return a new scope. All ops created within the returned scope will have no
  /// control dependencies on other operations.
  Scope WithNoControlDependencies() const;

  /// Return a new scope. All ops created within the returned scope will have
  /// the device field set to 'device'.
  Scope WithDevice(const string& device) const;

  /// Returns a new scope.  All ops created within the returned scope will have
  /// their assigned device set to `assigned_device`.
  Scope WithAssignedDevice(const string& assigned_device) const;

  /// Returns a new scope.  All ops created within the returned scope will have
  /// their _XlaCluster attribute set to `xla_cluster`.
  Scope WithXlaCluster(const string& xla_cluster) const;

  /// Return a new scope. All ops created within the returned scope will be
  /// co-located on the device where op is placed.
  /// NOTE: This function is intended to be use internal libraries only for
  /// controlling placement of ops on to devices. Public use is not encouraged
  /// because the implementation of device placement is subject to change.
  Scope ColocateWith(const Operation& op) const;
  /// Convenience function for above.
  Scope ColocateWith(const Output& out) const {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSccPSframeworkPSscopeDTh mht_0(mht_0_v, 326, "", "./tensorflow/cc/framework/scope.h", "ColocateWith");
 return ColocateWith(out.op()); }
  /// Clear all colocation constraints.
  Scope ClearColocation() const;

  /// Return a new scope. The op-constructor functions taking the returned scope
  /// as the scope argument will exit as soon as an error is detected, instead
  /// of setting the status on the scope.
  Scope ExitOnError() const;

  /// Return a new scope. All ops created with the new scope will have
  /// kernel_label as the value for their '_kernel' attribute;
  Scope WithKernelLabel(const string& kernel_label) const;

  // The following functions are for scope object consumers.

  /// Return a unique name, using default_name if an op name has not been
  /// specified.
  string GetUniqueNameForOp(const string& default_name) const;

  /// Update the status on this scope.
  /// Note: The status object is shared between all children of this scope.
  /// If the resulting status is not Status::OK() and exit_on_error_ is set on
  /// this scope, this function exits by calling LOG(FATAL).
  void UpdateStatus(const Status& s) const;

  // START_SKIP_DOXYGEN

  /// Update the builder with properties accumulated in this scope. Does not set
  /// status().
  // TODO(skyewm): NodeBuilder is not part of public API
  void UpdateBuilder(NodeBuilder* builder) const;
  // END_SKIP_DOXYGEN

  CompositeOpScopes GetCompositeOpScopes(const string& composite_op_name) const;

  bool ok() const;

  // TODO(skyewm): Graph is not part of public API
  Graph* graph() const;

  // TODO(skyewm): Graph is not part of public API
  std::shared_ptr<Graph> graph_as_shared_ptr() const;

  Status status() const;

  /// If status() is Status::OK(), convert the Graph object stored in this scope
  /// to a GraphDef proto and return Status::OK(). Otherwise, return the error
  /// status as is without performing GraphDef conversion.
  Status ToGraphDef(GraphDef* gdef) const;

  // START_SKIP_DOXYGEN

  /// If status() is Status::OK(), construct a Graph object using `opts` as the
  /// GraphConstructorOptions, and return Status::OK if graph construction was
  /// successful. Otherwise, return the error status.
  // TODO(josh11b, keveman): Make this faster; right now it converts
  // Graph->GraphDef->Graph.  This cleans up the graph (e.g. adds
  // edges from the source and to the sink node, resolves back edges
  // by name), and makes sure the resulting graph is valid.
  Status ToGraph(
      Graph* g, GraphConstructorOptions opts = GraphConstructorOptions{}) const;

  // Calls AddNode() using this scope's ShapeRefiner. This exists in the public
  // API to prevent custom op wrappers from needing access to shape_refiner.h or
  // scope_internal.h.
  // TODO(skyewm): remove this from public API
  Status DoShapeInference(Node* node) const;

  // Creates a new root scope that causes all DoShapeInference() calls to return
  // Status::OK() (on the returned scope and any subscopes). Used for testing.
  // TODO(skyewm): fix tests that still require this and eventually remove, or
  // at least remove from public API
  static Scope DisabledShapeInferenceScope();
  // END_SKIP_DOXYGEN

  const std::vector<Operation>& control_deps() const;

  // START_SKIP_DOXYGEN
  class Impl;
  Impl* impl() {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSccPSframeworkPSscopeDTh mht_1(mht_1_v, 408, "", "./tensorflow/cc/framework/scope.h", "impl");
 return impl_.get(); }
  const Impl* impl() const {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSccPSframeworkPSscopeDTh mht_2(mht_2_v, 412, "", "./tensorflow/cc/framework/scope.h", "impl");
 return impl_.get(); }
  // END_SKIP_DOXYGEN

 private:
  Scope WithOpNameImpl(const string& op_name) const;

  friend class InternalScope;
  std::unique_ptr<Impl> impl_;
  explicit Scope(Impl*);
};

/// A helper struct to hold the scopes that would be used by a function
/// constructing a composite op.
struct CompositeOpScopes {
  /// Scope to be used for creating the local ops (primitive or other composite
  /// ops).
  Scope child;
  /// Scope to be used for creating the last op.
  Scope last;
};

// Creates a node of the given operation, with the given inputs, and assigns the
// result to output. This does not support the ability to add additional
// attributes.
Status CreateOutputWithScope(string op_name,
                             absl::Span<const ::tensorflow::Input> inputs,
                             const Scope& scope, Output* output);
/// @}

}  // namespace tensorflow

#endif  // TENSORFLOW_CC_FRAMEWORK_SCOPE_H_
