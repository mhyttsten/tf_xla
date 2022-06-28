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

#ifndef TENSORFLOW_CORE_COMMON_RUNTIME_PLACER_INSPECTION_REQUIRED_OPS_UTILS_H_
#define TENSORFLOW_CORE_COMMON_RUNTIME_PLACER_INSPECTION_REQUIRED_OPS_UTILS_H_
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
class MHTracer_DTPStensorflowPScorePScommon_runtimePSplacer_inspection_required_ops_utilsDTh {
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
   MHTracer_DTPStensorflowPScorePScommon_runtimePSplacer_inspection_required_ops_utilsDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePScommon_runtimePSplacer_inspection_required_ops_utilsDTh() {
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


// Operations calling functions are becoming ubiquitous in TF 2.0.
// Examples include PartitionedCallOp, functional If/While, and Dataset ops.
// Such operations might require deep inspection - looking at the body of the
// called function - to place them and surrounding ops correctly.

// This file contains some utilities for placer to correctly place such ops
// including:
// - PlacerInspectionRequiredOpChecker: A simple class with a single
// IsPlacerInspectionRequired method.
// - IsolatePlacerInspectionRequiredOps: This function adds Identity ops for
// each input/output of ops requiring placer inspection. It greatly simplifies
// the implementation of placing such ops.

#include <vector>

#include "absl/types/optional.h"
#include "tensorflow/core/framework/function.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/lib/core/status.h"

namespace tensorflow {

// PlacerInspectionRequiredOpChecker allows one to check if Placer needs to
// look deeply into the op to place ops consuming the outputs correctly.
//
// It is a class instead of a standalone method because checking whether
// a function returns a resource takes non-trivial time and we cache the
// results.
class PlacerInspectionRequiredOpChecker {
 public:
  // Constructs a PlacerInspectionRequiredOpChecker for nodes of `graph`.
  // The functions referenced by nodes in `graph` will be looked up in
  // `flib_def`
  PlacerInspectionRequiredOpChecker(const Graph* graph,
                                    const FunctionLibraryDefinition* flib_def);

  // If `node` is considered a deep op, sets `*is_deep` to true and returns
  // Status::OK(). If an error occurs, returns that error, and the value of
  // `*is_deep` is undefined.
  // Currently, an op is considered deep, if it is a calling a function
  // returning a resource. This definition is driven by Placer's need to
  // look inside the op.
  // REQUIRES: `node` is part of `graph` passed into constructor.
  Status IsPlacerInspectionRequired(const Node& node, bool* is_deep);

 private:
  const Graph& graph_;
  const FunctionLibraryDefinition& flib_def_;
  // Indexed by the node id.
  // If cache_[node_id] is empty, the deepness of the node with id `node_id` has
  // not been computed yet. Else, it contains the value already computed.
  std::vector<absl::optional<bool>> cache_;
};

// Extracts `fdef` and `func` from `flib_def` for the function identified
// in "f" attribute of `node`.
Status GetFunctionDefAndAttrs(const FunctionLibraryDefinition& flib_def,
                              const Node& node, const FunctionDef** fdef,
                              NameAttrList* func);

// The "call" stack of functions.
// Useful for better error messages as well as for detecting recursion.
// Stores references to graph nodes. These references must outlive this.
class FunctionStack {
 public:
  explicit FunctionStack(const string& function_name);

  // `node_in_current_function` must outlive this.
  FunctionStack Push(const Node* node_in_current_function,
                     const string& new_current_function) const;

  // Returns true iff this stack already includes `function_name`.
  bool HasFunction(const string& function_name) const;

  const string& current_function_name() const {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSplacer_inspection_required_ops_utilsDTh mht_0(mht_0_v, 262, "", "./tensorflow/core/common_runtime/placer_inspection_required_ops_utils.h", "current_function_name");
 return current_function_name_; }

  // Format's this suitable for error interpolation that retrieves
  // Python files and line numbers.
  string FormatForError() const;

 private:
  struct Frame {
    Frame(const string& function, const Node* node)
        : function_name(function), node(node) {
   std::vector<std::string> mht_1_v;
   mht_1_v.push_back("function: \"" + function + "\"");
   MHTracer_DTPStensorflowPScorePScommon_runtimePSplacer_inspection_required_ops_utilsDTh mht_1(mht_1_v, 275, "", "./tensorflow/core/common_runtime/placer_inspection_required_ops_utils.h", "Frame");
}

    string function_name;
    const Node* node;
  };

  // The function at the top of the stack. In other words, the function
  // that is currently being inspected for placement.
  string current_function_name_;

  // The stack of frames that got the placement to the current_function_name_.
  // frames_[0].function_name is the top function that Placer was constructed
  // with. frames_[0].function_name can be empty if placer was constructed with
  // a nameless graph, not a function.  frames_[0].node_name is a name of a node
  // in frames_[0].function_name that required deep inspection (e.g. a
  // PartitionedCallOp). The function that this node invoked is
  // frames_[1].function_name, if frames_.size() > 1.  Else, the function that
  // this node invoked is current_function_name_.
  std::vector<Frame> frames_;
};

// Adds Identities for each input and output of function-calling ops in `graph`
//
// For example, the following graph calling a function on inputs `a` and `b`
// and producing output `y` will be rewritten to include identities on all
// edges:
//
//      a             b
//      |             |
//      v             v
//    f (PartitionedCallOp)
//         |
//         v
//         y
//
// is transformed to
//
//      a             b
//      |             |
//  a_f (Identity)   b_f (Identity)
//      |             |
//      v             v
//    f (PartitionedCallOp)
//         |
//      f_y (Identity)
//         |
//         v
//         y
//
Status IsolatePlacerInspectionRequiredOps(
    const FunctionLibraryDefinition& flib_def, Graph* graph);

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_COMMON_RUNTIME_PLACER_INSPECTION_REQUIRED_OPS_UTILS_H_
