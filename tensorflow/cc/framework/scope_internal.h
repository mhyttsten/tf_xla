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

#ifndef TENSORFLOW_CC_FRAMEWORK_SCOPE_INTERNAL_H_
#define TENSORFLOW_CC_FRAMEWORK_SCOPE_INTERNAL_H_
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
class MHTracer_DTPStensorflowPSccPSframeworkPSscope_internalDTh {
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
   MHTracer_DTPStensorflowPSccPSframeworkPSscope_internalDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSccPSframeworkPSscope_internalDTh() {
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


#include "tensorflow/cc/framework/scope.h"

namespace tensorflow {

class ShapeRefiner;

// NewInternalScope returns a new scope which doesn't take ownership of
// graph, status, name_map, and refiner.
// This is intended to enable the C API (which are used by other language
// bindings) to create a Scope and access C++ functionality (i.e. gradients).
//
// Shape inference is disabled if `refiner` is nullptr.
Scope NewInternalScope(Graph* graph, Status* status, ShapeRefiner* refiner);

class Scope::Impl {
 public:
  // A NameMap is used to keep track of suffixes for names used in a scope. A
  // name that has not been used so far in a scope will get no suffix. Later
  // uses of the same name will get suffixes _1, _2, _3, etc. Multiple scopes
  // can share the same NameMap. For instance, a new scope created using
  // WithControlDependencies() would share the same NameMap with the parent.
  typedef std::unordered_map<string, int> NameMap;

  Impl(const std::shared_ptr<Graph>& graph,
       const std::shared_ptr<Status>& status,
       const std::shared_ptr<NameMap>& name_map,
       const std::shared_ptr<ShapeRefiner>& refiner);

  const string& name() const {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSccPSframeworkPSscope_internalDTh mht_0(mht_0_v, 216, "", "./tensorflow/cc/framework/scope_internal.h", "name");
 return name_; }
  const std::vector<Operation>& control_deps() const {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSccPSframeworkPSscope_internalDTh mht_1(mht_1_v, 220, "", "./tensorflow/cc/framework/scope_internal.h", "control_deps");
 return control_deps_; }

 private:
  friend class Scope;

  // Tag types to choose the constructor to dispatch.
  struct Tags {
    enum class ScopeName;
    enum class OpName;
    enum class ControlDeps;
    enum class Device;
    enum class SingleUseScope;
    enum class ExitOnError;
    enum class KernelLabel;
    enum class Colocate;
    enum class AssignedDevice;
    enum class XlaCluster;
  };

  Impl(Graph* graph, Status* status, NameMap* name_map, ShapeRefiner* refiner,
       bool disable_shape_inference);
  Impl(const Scope& other, Tags::ScopeName, const string& name,
       bool copy_names);
  Impl(const Scope& other, Tags::OpName, const string& name,
       const string& op_name);
  Impl(const Scope& other, Tags::ControlDeps,
       std::vector<Operation> control_deps, bool clear_control_deps);
  Impl(const Scope& other, Tags::Device, const string& device);
  Impl(const Scope& other, Tags::SingleUseScope, const string& op_name);
  Impl(const Scope& other, Tags::ExitOnError);
  Impl(const Scope& other, Tags::KernelLabel, const string& kernel_label);
  Impl(const Scope& other, Tags::Colocate, const Operation& colocate_with_op,
       bool clear_colocations);
  Impl(const Scope& other, Tags::AssignedDevice, const string& assigned_device);
  Impl(const Scope& other, Tags::XlaCluster, const string& xla_cluster);

  std::unordered_set<string> GetColocationConstraints(
      const Operation& colocate_with_op) const;

  // Helper functions to get a unique names.
  string GetUniqueName(const string& prefix, bool check_single_use) const;
  string GetNameForOp(const string& default_name) const;

  bool single_use_scope() const {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSccPSframeworkPSscope_internalDTh mht_2(mht_2_v, 266, "", "./tensorflow/cc/framework/scope_internal.h", "single_use_scope");
 return scope_used_ != nullptr; }

  // The graph, status, and name maps are shared by all child scopes
  // created from a single 'root' scope. A root scope is created by calling the
  // Scope::NewRootScope function, which creates a new graph, a new status and
  // the name maps.
  std::shared_ptr<Graph> graph_ = nullptr;
  std::shared_ptr<Status> status_ = nullptr;
  std::shared_ptr<NameMap> name_map_ = nullptr;
  std::shared_ptr<ShapeRefiner> refiner_ = nullptr;

  // If scope_used_ is not nullptr, op_name_ should be empty and
  // GetUniqueNameForOp can only be called once on this scope. More calls to
  // GetUniqueNameForOp will cause an error status to be set on this scope.
  std::shared_ptr<bool> scope_used_ = nullptr;

  const std::vector<Operation> control_deps_;

  // The fully-qualified name of this scope (i.e. includes any parent scope
  // names).
  const string name_ = "";
  const string op_name_ = "";
  const bool exit_on_error_ = false;
  const string kernel_label_ = "";
  const string device_ = "";
  const string assigned_device_ = "";
  const string xla_cluster_ = "";
  const std::unordered_set<string> colocation_constraints_;

  // If true, Scope::DoShapeInference() always returns Status:OK().
  // TODO(skyewm): remove this when possible
  const bool disable_shape_inference_;
};

}  // namespace tensorflow

#endif  // TENSORFLOW_CC_FRAMEWORK_SCOPE_INTERNAL_H_
