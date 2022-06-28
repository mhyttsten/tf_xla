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
#ifndef TENSORFLOW_CORE_TFRT_GRAPH_EXECUTOR_GRAPH_EXECUTOR_H_
#define TENSORFLOW_CORE_TFRT_GRAPH_EXECUTOR_GRAPH_EXECUTOR_H_
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
class MHTracer_DTPStensorflowPScorePStfrtPSgraph_executorPSgraph_executorDTh {
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
   MHTracer_DTPStensorflowPScorePStfrtPSgraph_executorPSgraph_executorDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePStfrtPSgraph_executorPSgraph_executorDTh() {
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

#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "tensorflow/core/common_runtime/graph_execution_state.h"
#include "tensorflow/core/protobuf/config.pb.h"
#include "tensorflow/core/tfrt/fallback/fallback_state.h"
#include "tensorflow/core/tfrt/graph_executor/graph_execution_options.h"
#include "tensorflow/core/tfrt/runtime/work_queue_interface.h"
#include "tensorflow/core/tfrt/tpu/tpu_resources.h"
#include "tensorflow/core/tfrt/utils/tfrt_graph_execution_state.h"
#include "tfrt/bef/bef_buffer.h"  // from @tf_runtime
#include "tfrt/bef_executor/bef_file.h"  // from @tf_runtime
#include "tfrt/core_runtime/core_runtime.h"  // from @tf_runtime
#include "tfrt/host_context/execution_context.h"  // from @tf_runtime
#include "tfrt/host_context/function.h"  // from @tf_runtime
#include "tfrt/host_context/request_deadline_tracker.h"  // from @tf_runtime
#include "tfrt/support/ref_count.h"  // from @tf_runtime

namespace tensorflow {
namespace tfrt_stub {

// Contains request related info.
struct RequestInfo {
  tfrt::RCReference<tfrt::RequestContext> tfrt_request_context;
  std::unique_ptr<WorkQueueInterface> request_queue;
  std::function<void(std::function<void()>)> runner;
};

// Creates a `RequestInfo` given relative data.
StatusOr<std::unique_ptr<RequestInfo>> SetUpRequestContext(
    const GraphExecutionRunOptions& run_options,
    const SessionMetadata& model_metadata, tfrt::HostContext* host,
    tensorflow::tfrt_stub::WorkQueueInterface* work_queue,
    tfrt::ResourceContext* resource_context,
    const FallbackState& fallback_state);

// Runs on a function given input/output and other info.
tensorflow::Status GraphExecutionRunOnFunction(
    const GraphExecutionOptions& options,
    const GraphExecutionRunOptions& run_options,
    absl::string_view signature_name, const tfrt::Function& func,
    absl::Span<const tensorflow::Tensor> inputs,
    absl::Span<const tensorflow::Tensor> captures,
    std::vector<tensorflow::Tensor>* outputs,
    tfrt::ResourceContext* resource_context, const Runtime& runtime,
    const FallbackState& fallback_state,
    tfrt::RequestDeadlineTracker& req_deadline_tracker);

// Creates a ResourceContext and populate it with per model resource from
// Runtime. If `tpu_target` is set to kTpurt, also call a special
// `AddTpuResources` function to populate TPU related resources for tpurt.
//
// TODO(b/178227859): Remove the need for the special handling for TPU here.
std::unique_ptr<tfrt::ResourceContext> CreateResourceContext(
    const Runtime& runtime, tfrt::tpu::TpuModelResource* tpu_model_resource,
    tensorflow::TfrtTpuInfraTarget tpu_target);

// Loads (if not yet) and runs a subgraph in a graph as per each request.
class GraphExecutor {
 public:
  using Options = GraphExecutionOptions;
  using RunOptions = GraphExecutionRunOptions;

  // Creates a `GraphExecutor` given the args.
  static StatusOr<std::unique_ptr<GraphExecutor>> Create(
      Options options, const FallbackState& fallback_state,
      tfrt::tpu::TpuModelResource* tpu_model_resource,
      tensorflow::GraphDef graph_def);

  // Ctor. Public for `Create()`. Do not use directly.
  GraphExecutor(Options options, const FallbackState& fallback_state,
                tfrt::tpu::TpuModelResource* tpu_model_resource,
                std::unique_ptr<tensorflow::tfrt_stub::TfrtGraphExecutionState>
                    graph_execution_state)
      : options_(std::move(options)),
        fallback_state_(fallback_state),
        tpu_model_resource_(tpu_model_resource),
        graph_execution_state_(std::move(graph_execution_state)),
        req_deadline_tracker_(
            options_.runtime->core_runtime()->GetHostContext()) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePStfrtPSgraph_executorPSgraph_executorDTh mht_0(mht_0_v, 269, "", "./tensorflow/core/tfrt/graph_executor/graph_executor.h", "GraphExecutor");
}

  // Runs on the graph according to given input/output.
  tensorflow::Status Run(
      const RunOptions& run_options,
      absl::Span<const std::pair<std::string, tensorflow::Tensor>> inputs,
      absl::Span<const std::string> output_tensor_names,
      absl::Span<const std::string> target_tensor_names,
      std::vector<tensorflow::Tensor>* outputs);

  // Extends the current graph by `graph`.
  tensorflow::Status Extend(const GraphDef& graph);

  tensorflow::tfrt_stub::TfrtGraphExecutionState& graph_execution_state()
      const {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePStfrtPSgraph_executorPSgraph_executorDTh mht_1(mht_1_v, 286, "", "./tensorflow/core/tfrt/graph_executor/graph_executor.h", "graph_execution_state");

    return *graph_execution_state_;
  }

 private:
  // The loading result of a `ClientGraph`.
  struct LoadedClientGraph {
    std::string name;
    tfrt::BefBuffer bef;
    tfrt::RCReference<tfrt::BEFFile> bef_file;
    std::unique_ptr<tfrt::ResourceContext> resource_context;
  };

  // A subgraph constructed by specifying input/output tensors.
  struct ClientGraph {
    // A unique name by joining all the input/output/target names.
    std::string name;
    // The feed nodes for the corresponding inputs, but they might not be in the
    // original order and if there are more than one original inputs mapped to
    // the same feed node, only one is picked here.
    tensorflow::GraphImportConfig::InputArrays input_nodes;
    // The fetch nodes for the outputs, which should be in the original order.
    std::vector<std::string> output_nodes;
    // The target nodes that should be run but not returned as outputs.
    std::vector<std::string> target_nodes;
  };

  // A set of methods to load a client graph.
  StatusOr<std::unique_ptr<GraphExecutor::LoadedClientGraph>> LoadClientGraph(
      const GraphExecutor::ClientGraph& client_graph);
  tensorflow::StatusOr<mlir::OwningOpRef<mlir::ModuleOp>>
  ImportClientGraphToMlirModule(const GraphExecutor::ClientGraph& client_graph,
                                mlir::MLIRContext* context) const;
  StatusOr<tfrt::BefBuffer> CompileMlirModuleToBef(mlir::ModuleOp module) const;
  tensorflow::Status InitBef(tfrt::BEFFile* bef_file,
                             tfrt::ResourceContext* resource_context);

  // Returns a `LoadedClientGraph` given input/output tensor info. If there is
  // no existing one yet, creates one first.
  StatusOr<std::reference_wrapper<const GraphExecutor::LoadedClientGraph>>
  GetOrCreateLoadedClientGraph(
      absl::Span<const std::string> input_tensor_names,
      absl::Span<const tensorflow::DataType> input_tensor_dtypes,
      absl::Span<const std::string> output_tensor_names,
      absl::Span<const std::string> target_tensor_names)
      TF_LOCKS_EXCLUDED(loaded_client_graphs_mu_);

  const tensorflow::tfrt_stub::Runtime& runtime() const {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePStfrtPSgraph_executorPSgraph_executorDTh mht_2(mht_2_v, 336, "", "./tensorflow/core/tfrt/graph_executor/graph_executor.h", "runtime");

    DCHECK(options_.runtime);
    return *options_.runtime;
  }

  Options options_;
  std::reference_wrapper<const FallbackState> fallback_state_;
  tfrt::tpu::TpuModelResource* tpu_model_resource_;  // NOT owned.

  std::unique_ptr<tensorflow::tfrt_stub::TfrtGraphExecutionState>
      graph_execution_state_;

  tfrt::RequestDeadlineTracker req_deadline_tracker_;

  tensorflow::mutex loaded_client_graphs_mu_;
  // Caches `LoadedClientGraph` by the joined name.
  // For pointer stability of values in `absl::flat_hash_map<>`, additional
  // `std::unique_ptr<>` is necessary. (See https://abseil.io/tips/136.)
  absl::flat_hash_map<std::string /*joined_name*/,
                      std::unique_ptr<LoadedClientGraph>>
      loaded_client_graphs_ TF_GUARDED_BY(loaded_client_graphs_mu_);
};

}  // namespace tfrt_stub
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_TFRT_GRAPH_EXECUTOR_GRAPH_EXECUTOR_H_
