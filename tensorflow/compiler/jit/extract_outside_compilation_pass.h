/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_COMPILER_JIT_EXTRACT_OUTSIDE_COMPILATION_PASS_H_
#define TENSORFLOW_COMPILER_JIT_EXTRACT_OUTSIDE_COMPILATION_PASS_H_
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
class MHTracer_DTPStensorflowPScompilerPSjitPSextract_outside_compilation_passDTh {
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
   MHTracer_DTPStensorflowPScompilerPSjitPSextract_outside_compilation_passDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSjitPSextract_outside_compilation_passDTh() {
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


#include "absl/types/optional.h"
#include "tensorflow/compiler/jit/encapsulate_util.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/core/graph/graph.h"

namespace tensorflow {

// Rewrite function for outside compilation subgraphs. It will perform the
// following steps:
//
// 1. Add a XLA computation key placeholder node (it will be used as input for
//    XlaRecvAtHost and XlaSendFromHost);
// 2. Replace all _Arg nodes with one single XlaRecvAtHost node;
// 3. Replace all _Retval nodes with one single XlaSendFromHost node;
// 4. Mark all nodes except key placeholder with attr `xla_cluster_attr_name`
//    and `outside_compilation_attr_name`;
// 5. For nodes marked with attr kXlaConnectedToXlaComputationAttrName, add a
//    control edge from the node to XlaSendFromHost; for nodes marked with attr
//    kXlaConnectedFromXlaComputationAttrName, add a control edge from
//    XlaRecvAtHost node to the node;
// 6. Try pruning XlaRecvAtHost/XlaSendFromHost/key placeholder node.
// 7. Add necessary attributes to `node_def`, so we can replace it with a
//    XlaHostCompute node later. If all input shapes for XlaSendFromHost are
//    known, "shapes" attr will be set to the list of input shapes; otherwise
//    "shape_inference_graph" attr will be set to shape inference function name.
class RewriteOutsideCompilationSubgraphFn {
 public:
  RewriteOutsideCompilationSubgraphFn(
      const string& xla_cluster_attr_name,
      const string& outside_compilation_attr_name,
      const string& xla_cluster_name, const string& new_function_name)
      : xla_cluster_attr_name_(xla_cluster_attr_name),
        outside_compilation_attr_name_(outside_compilation_attr_name),
        xla_cluster_name_(xla_cluster_name),
        new_function_name_(new_function_name) {
   std::vector<std::string> mht_0_v;
   mht_0_v.push_back("xla_cluster_attr_name: \"" + xla_cluster_attr_name + "\"");
   mht_0_v.push_back("outside_compilation_attr_name: \"" + outside_compilation_attr_name + "\"");
   mht_0_v.push_back("xla_cluster_name: \"" + xla_cluster_name + "\"");
   mht_0_v.push_back("new_function_name: \"" + new_function_name + "\"");
   MHTracer_DTPStensorflowPScompilerPSjitPSextract_outside_compilation_passDTh mht_0(mht_0_v, 226, "", "./tensorflow/compiler/jit/extract_outside_compilation_pass.h", "RewriteOutsideCompilationSubgraphFn");
}

  Status operator()(const std::vector<OutputTensor>&,
                    std::unique_ptr<Graph>* graph,
                    std::vector<int>* input_permutation,
                    std::vector<int>* output_permutation, NodeDef* node_def);

 private:
  string xla_cluster_attr_name_;
  string outside_compilation_attr_name_;
  string xla_cluster_name_;
  string new_function_name_;
};

// For an XLA computation function, replace all outside compilations with
// XlaHostCompute nodes. Each outside compilation subgraph will be rewritten by
// `RewriteOutsideCompilationSubgraphFn`, and they will be merged into one
// single host side graph (`host_graph`).
//
// xla_cluster_attr_name and outside_compilation_attr_name: attr name for XLA
//   computation and outside compilation. Required for
//   `RewriteOutsideCompilationSubgraphFn`.
// xla_cluster_name: XLA cluster name for this XLA computation. We need it
//   because XLA cluster name might be different from `func_name`.
// func_name_attrs: they will be used to instantiate the XLA computation func.
// new_func_name: new function name for rewritten XLA computation func.
// host_compute_core: mapping from outside compilation cluster name to XLA
//   device assignment.
// fld: FunctionLibraryDefinition object.
// host_graph: Graph object to store host side graph for all outside
//   compilations within this XLA computation func. If there is no outside
//   compilation, it will be empty.
// shape_inference_graphs: a list of outside compilation shape inference
//   function names. These functions need to be rewritten later.
// has_outside_compilation: a bool indicating whether this function has any
//   outside compilation nodes.
Status ExtractOutsideCompilationForFunction(
    const string& xla_cluster_attr_name,
    const string& outside_compilation_attr_name, const string& xla_cluster_name,
    const NameAttrList& func_name_attrs, const string& new_func_name,
    const string& host_graph_func_name,
    const std::map<string, int>& host_compute_core, FunctionLibraryRuntime* flr,
    FunctionLibraryDefinition* fld, std::vector<string>* shape_inference_graphs,
    bool* has_outside_compilation);

// Rewrites XLA computation in `clusters` to replace outside compilation nodes
// with XlaHostCompute, and moves those outside compilations into `g`. If shapes
// of outside compilation outputs cannot be determined now, we will store shape
// inference graph into `fld`.
Status ExtractOutsideCompilation(
    const string& xla_cluster_attr_name,
    const string& outside_compilation_attr_name,
    const std::unordered_map<string, XlaClusterInfo>& clusters, Graph* g,
    FunctionLibraryRuntime* flr, FunctionLibraryDefinition* fld,
    bool* modified);

}  // namespace tensorflow

#endif  // TENSORFLOW_COMPILER_JIT_EXTRACT_OUTSIDE_COMPILATION_PASS_H_
