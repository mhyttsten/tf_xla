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
#ifndef TENSORFLOW_CORE_TFRT_UTILS_TFRT_GRAPH_EXECUTION_STATE_H_
#define TENSORFLOW_CORE_TFRT_UTILS_TFRT_GRAPH_EXECUTION_STATE_H_
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
class MHTracer_DTPStensorflowPScorePStfrtPSutilsPStfrt_graph_execution_stateDTh {
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
   MHTracer_DTPStensorflowPScorePStfrtPSutilsPStfrt_graph_execution_stateDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePStfrtPSutilsPStfrt_graph_execution_stateDTh() {
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


#include <functional>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/synchronization/mutex.h"
#include "absl/time/time.h"
#include "tensorflow/compiler/mlir/tensorflow/translate/mlir_roundtrip_flags.h"
#include "tensorflow/core/common_runtime/graph_execution_state.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/platform/statusor.h"
#include "tensorflow/core/protobuf/config.pb.h"
#include "tensorflow/core/tfrt/fallback/fallback_state.h"

namespace tensorflow {
namespace tfrt_stub {

// This is a TFRT variant of `tensorflow::GraphExecutionState`. It wraps
// `tensorflow::GraphExecutionState` and adds TFRT-specific adjustments.
//
// Responsible for generating an executable `Graph` from the original `GraphDef`
// that specifies the complete graph and from `GraphImportConfig` that specifies
// input/output nodes.
//
// Thread-safe.
class TfrtGraphExecutionState {
 public:
  struct OptimizationResult {
    std::unique_ptr<tensorflow::Graph> graph;
    absl::Duration functionalization_duration;
    absl::Duration grappler_duration;
  };

  // Creates a `GraphExecutionState` given `graph_def` and `fallback_state`.
  static StatusOr<std::unique_ptr<TfrtGraphExecutionState>> Create(
      tensorflow::GraphDef graph_def, const FallbackState& fallback_state,
      bool run_placer_grappler_on_functions = false);

  // Ctor. Do not use directly. Public only for `std::make_unique<>()`.
  TfrtGraphExecutionState(
      std::unique_ptr<tensorflow::GraphExecutionState> graph_execution_state,
      const FallbackState& fallback_state,
      bool run_placer_grappler_on_functions,
      absl::flat_hash_set<std::string> functions_to_optimize)
      : graph_execution_state_(std::move(graph_execution_state)),
        fallback_state_(fallback_state),
        run_placer_grappler_on_functions_(run_placer_grappler_on_functions),
        functions_to_optimize_(std::move(functions_to_optimize)) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePStfrtPSutilsPStfrt_graph_execution_stateDTh mht_0(mht_0_v, 236, "", "./tensorflow/core/tfrt/utils/tfrt_graph_execution_state.h", "TfrtGraphExecutionState");
}

  // Creates an optimized graph by pruning with `graph_import_config` and
  // best-effort Grappler run.
  StatusOr<OptimizationResult> CreateOptimizedGraph(
      const tensorflow::GraphImportConfig& graph_import_config);

  // Extends the current graph by `graph`.
  Status Extend(const GraphDef& graph);

  // Return the preprocessed full graph. Note that it does not contain the
  // function library in the original graph.
  const tensorflow::Graph& graph() const {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePStfrtPSutilsPStfrt_graph_execution_stateDTh mht_1(mht_1_v, 251, "", "./tensorflow/core/tfrt/utils/tfrt_graph_execution_state.h", "graph");

    absl::MutexLock lock(&graph_execution_state_mu_);
    DCHECK(graph_execution_state_->full_graph());
    return *graph_execution_state_->full_graph();
  }

  // The original graph.
  const GraphDef* original_graph_def() const {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePStfrtPSutilsPStfrt_graph_execution_stateDTh mht_2(mht_2_v, 261, "", "./tensorflow/core/tfrt/utils/tfrt_graph_execution_state.h", "original_graph_def");

    absl::MutexLock lock(&graph_execution_state_mu_);
    return graph_execution_state_->original_graph_def();
  }

 private:
  // Return the function library in the original graph.
  const FunctionLibraryDefinition& flib_def() const {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePStfrtPSutilsPStfrt_graph_execution_stateDTh mht_3(mht_3_v, 271, "", "./tensorflow/core/tfrt/utils/tfrt_graph_execution_state.h", "flib_def");

    absl::MutexLock lock(&graph_execution_state_mu_);
    return graph_execution_state_->flib_def();
  }

  StatusOr<std::unique_ptr<tensorflow::Graph>> OptimizeGraph(
      const tensorflow::Graph& graph,
      const tensorflow::BuildGraphOptions& build_graph_options);

  std::unique_ptr<tensorflow::GraphExecutionState> graph_execution_state_
      ABSL_GUARDED_BY(graph_execution_state_mu_);
  // We need this mutex even thought `GraphExecutionState` is thread-safe,
  // because `swap()` is not thread-safe.
  mutable absl::Mutex graph_execution_state_mu_;

  const FallbackState& fallback_state_;
  bool run_placer_grappler_on_functions_;
  // Only valid if `run_placer_grappler_on_functions_` is true.
  absl::flat_hash_set<std::string> functions_to_optimize_;
};

// Prunes the `graph_def` using the feed/fetch nodes specified in
// `callable_options`. It is a TFRT-specific version that it performs more
// pruning (e.g., prunes the input edges to the feed nodes) than
// `ComputeTransitiveFanin()` so that the graph can be functionalized properly
// later.
Status PruneGraphDef(GraphDef& graph_def,
                     const CallableOptions& callable_options);

// Eliminates ref variables in V1 control flow, which is required for
// functionalization. Current strategy is to insert an identity node between
// each ref node and its ref input and in-place update the ref node to its
// non-ref counterpart.
Status EliminateRefVariablesFromV1ControlFlow(GraphDef& graph_def);

// Removes the "_input_shapes" attribute of functions in the graph.
void RemoveInputShapesInFunctions(tensorflow::GraphDef& graph_def);

}  // namespace tfrt_stub
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_TFRT_UTILS_TFRT_GRAPH_EXECUTION_STATE_H_
