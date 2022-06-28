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
#ifndef TENSORFLOW_CORE_TFRT_SAVED_MODEL_SAVED_MODEL_H_
#define TENSORFLOW_CORE_TFRT_SAVED_MODEL_SAVED_MODEL_H_
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
class MHTracer_DTPStensorflowPScorePStfrtPSsaved_modelPSsaved_modelDTh {
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
   MHTracer_DTPStensorflowPScorePStfrtPSsaved_modelPSsaved_modelDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePStfrtPSsaved_modelPSsaved_modelDTh() {
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
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "tensorflow/compiler/mlir/tensorflow/translate/import_model.h"
#include "tensorflow/compiler/mlir/tfrt/saved_model/saved_model.h"
#include "tensorflow/compiler/mlir/tfrt/translate/import_model.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/platform/thread_annotations.h"
#include "tensorflow/core/platform/threadpool_interface.h"
#include "tensorflow/core/protobuf/meta_graph.pb.h"
#include "tensorflow/core/public/session_options.h"
#include "tensorflow/core/runtime_fallback/runtime/kernel_utils.h"
#include "tensorflow/core/tfrt/fallback/fallback_state.h"
#include "tensorflow/core/tfrt/graph_executor/graph_execution_options.h"
#include "tensorflow/core/tfrt/graph_executor/graph_executor.h"
#include "tensorflow/core/tfrt/runtime/runtime.h"
#include "tensorflow/core/tfrt/utils/tfrt_graph_execution_state.h"
#include "tfrt/host_context/function.h"  // from @tf_runtime
#include "tfrt/host_context/request_deadline_tracker.h"  // from @tf_runtime
#include "tfrt/host_context/resource_context.h"  // from @tf_runtime

namespace tfrt {

class BEFFile;
class HostContext;

namespace tpu {
class TpuModelResource;
}  // namespace tpu

}  // namespace tfrt

namespace tensorflow {
namespace tfrt_stub {

// TODO(tfrt-dev): Replace tfrt::TensorSpec with tensorflow::TensorSpec once the
// latter is checked in.
struct TensorSpec {
  tensorflow::DataType dtype;
  tensorflow::PartialTensorShape shape;

  explicit TensorSpec(tensorflow::DataType dtype) : dtype(dtype) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePStfrtPSsaved_modelPSsaved_modelDTh mht_0(mht_0_v, 235, "", "./tensorflow/core/tfrt/saved_model/saved_model.h", "TensorSpec");
}
  TensorSpec(tensorflow::DataType dtype, tensorflow::PartialTensorShape shape)
      : dtype(dtype), shape(std::move(shape)) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePStfrtPSsaved_modelPSsaved_modelDTh mht_1(mht_1_v, 240, "", "./tensorflow/core/tfrt/saved_model/saved_model.h", "TensorSpec");
}
};

inline bool operator==(const TensorSpec& a, const TensorSpec& b) {
  return a.dtype == b.dtype && a.shape.IsIdenticalTo(b.shape);
}

namespace internal {

struct Signature {
  std::vector<tensorflow::Tensor> captures;

  // The following three fields should have the same size.
  std::vector<std::string> input_names;
  std::vector<TensorSpec> input_specs;
  std::vector<std::string> input_devices;

  // The following two fields should have the same size.
  std::vector<std::string> output_names;
  std::vector<TensorSpec> output_specs;
};

}  // namespace internal

class FunctionMetadata {
 public:
  explicit FunctionMetadata(const internal::Signature* signature)
      : signature_(signature) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePStfrtPSsaved_modelPSsaved_modelDTh mht_2(mht_2_v, 270, "", "./tensorflow/core/tfrt/saved_model/saved_model.h", "FunctionMetadata");

    assert(signature);
  }

  const std::vector<std::string>& GetInputNames() const {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePStfrtPSsaved_modelPSsaved_modelDTh mht_3(mht_3_v, 277, "", "./tensorflow/core/tfrt/saved_model/saved_model.h", "GetInputNames");

    return signature_->input_names;
  }

  const std::vector<TensorSpec>& GetInputSpecs() const {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePStfrtPSsaved_modelPSsaved_modelDTh mht_4(mht_4_v, 284, "", "./tensorflow/core/tfrt/saved_model/saved_model.h", "GetInputSpecs");

    return signature_->input_specs;
  }

  const std::vector<std::string>& GetOutputNames() const {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePStfrtPSsaved_modelPSsaved_modelDTh mht_5(mht_5_v, 291, "", "./tensorflow/core/tfrt/saved_model/saved_model.h", "GetOutputNames");

    return signature_->output_names;
  }

  const std::vector<TensorSpec>& GetOutputSpecs() const {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePStfrtPSsaved_modelPSsaved_modelDTh mht_6(mht_6_v, 298, "", "./tensorflow/core/tfrt/saved_model/saved_model.h", "GetOutputSpecs");

    return signature_->output_specs;
  }

 private:
  friend class SavedModelImpl;

  const internal::Signature* signature_ = nullptr;
};

// SavedModel represents the in-memory states (graphs and variables) loaded from
// a tensorflow saved model directory.
class SavedModel {
 public:
  struct Options {
    explicit Options(const Runtime* rt) : graph_execution_options(rt) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePStfrtPSsaved_modelPSsaved_modelDTh mht_7(mht_7_v, 316, "", "./tensorflow/core/tfrt/saved_model/saved_model.h", "Options");
}

    // If true, the loading of any signature (or signature combination) will be
    // deferred until the first corresponding invocationof running. Otherwise,
    // the individual signatures will be loaded along with the saved model.
    bool enable_lazy_loading = false;

    GraphExecutionOptions graph_execution_options;
  };

  // Per-request options.
  using RunOptions = GraphExecutionRunOptions;

  explicit SavedModel(const Runtime* runtime) : runtime_(runtime) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScorePStfrtPSsaved_modelPSsaved_modelDTh mht_8(mht_8_v, 332, "", "./tensorflow/core/tfrt/saved_model/saved_model.h", "SavedModel");

    DCHECK(runtime_);
  }
  virtual ~SavedModel();

  const Runtime& runtime() const {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScorePStfrtPSsaved_modelPSsaved_modelDTh mht_9(mht_9_v, 340, "", "./tensorflow/core/tfrt/saved_model/saved_model.h", "runtime");

    DCHECK(runtime_);
    return *runtime_;
  }
  tfrt::HostContext* GetHostContext() const;

  // Returns meta graph def. Note that the graph_def field in the MetaGraphDef
  // has already been removed.
  //
  // TODO(b/191931702): Change the method to return SignatureDefs instead.
  virtual const tensorflow::MetaGraphDef& GetMetaGraphDef() const = 0;

  // Returns all the function names.
  virtual std::vector<std::string> GetFunctionNames() const = 0;

  // Returns the `FunctionMetadata` for a function. If the function is not
  // found, returns nullopt instead.
  virtual absl::optional<FunctionMetadata> GetFunctionMetadata(
      absl::string_view func_name) const = 0;

  // Runs the signature specified by `name`. Both `inputs` and `outputs`
  // are all host tensors. The `outputs` must be non-null. If the returned
  // status is non-OK, the `outputs` are invalid.
  virtual tensorflow::Status Run(const RunOptions& run_options,
                                 absl::string_view name,
                                 absl::Span<const tensorflow::Tensor> inputs,
                                 std::vector<tensorflow::Tensor>* outputs) = 0;

  // Runs the signatures specified by `names`. Both `inputs` and `outputs` are
  // all host tensors. The `outputs` must be non-null. If the returned status is
  // non-OK, the `outputs` are invalid.
  //
  // NOTE: If the given signatures have overlapping input nodes, the input
  // tensors for these overlapping nodes must be the same. Having different
  // input tensors for overlapping nodes results UNDEFINED BEHAVIOR.
  //
  // NOTE: The input/output tensors can only be dense tensors (as opposed to
  // sparse tensors or composite tensors).
  virtual tensorflow::Status RunMultipleSignatures(
      const RunOptions& run_options, absl::Span<const std::string> names,
      absl::Span<const std::vector<tensorflow::Tensor>> multi_inputs,
      std::vector<std::vector<tensorflow::Tensor>>* multi_outputs) = 0;

  // Runs the graphs specified by the tensor names terminal tensors (eg. feed
  // tensors, fetch tesnors) in the graph.
  virtual tensorflow::Status RunByTensorNames(
      const RunOptions& run_options,
      absl::Span<const std::pair<std::string, tensorflow::Tensor>> inputs,
      absl::Span<const std::string> output_tensor_names,
      absl::Span<const std::string> target_node_names,
      std::vector<tensorflow::Tensor>* outputs) = 0;

 private:
  const Runtime* runtime_ = nullptr;
};

class SavedModelImpl final : public SavedModel {
 public:
  struct JoinedSignature;

  // Loads all SignatureDefs in a MetaGraphDef that matches the `tags` in the
  // tensorflow saved model from `saved_model_dir`. Refer to
  // http://g3doc/learning/serving/g3doc/saved_model/overview.md
  // for explanations on SavedModel.
  static std::unique_ptr<SavedModel> LoadSavedModel(
      Options options, absl::string_view saved_model_dir,
      const std::unordered_set<std::string>& tags, tensorflow::Status* status);

  // Loads all SignatureDefs from `meta_graph_def`. Refer to
  // http://g3doc/learning/serving/g3doc/saved_model/overview.md
  // for explanations on SavedModel.
  static std::unique_ptr<SavedModel> LoadSavedModel(
      Options options, absl::string_view saved_model_dir,
      tensorflow::MetaGraphDef meta_graph_def, tensorflow::Status* status);

  SavedModelImpl(
      Options options, tensorflow::MetaGraphDef meta_graph_def,
      tfrt::BefBuffer bef, tfrt::RCReference<tfrt::BEFFile> bef_file,
      absl::flat_hash_map<std::string, internal::Signature> signatures,
      std::unique_ptr<FallbackState> fallback_state,
      std::unique_ptr<tfrt::tpu::TpuModelResource> tpu_model_resource,
      std::unique_ptr<tfrt::ResourceContext> resource_context,
      std::unique_ptr<GraphExecutor> graph_executor);

  ~SavedModelImpl() override;

  SavedModelImpl(const SavedModelImpl&) = delete;
  SavedModelImpl& operator=(const SavedModelImpl&) = delete;

  const tensorflow::MetaGraphDef& GetMetaGraphDef() const override;

  std::vector<std::string> GetFunctionNames() const override;

  absl::optional<FunctionMetadata> GetFunctionMetadata(
      absl::string_view func_name) const override;

  tensorflow::Status Run(const RunOptions& run_options, absl::string_view name,
                         absl::Span<const tensorflow::Tensor> inputs,
                         std::vector<tensorflow::Tensor>* outputs) override;

  tensorflow::Status RunMultipleSignatures(
      const RunOptions& run_options, absl::Span<const std::string> names,
      absl::Span<const std::vector<tensorflow::Tensor>> multi_inputs,
      std::vector<std::vector<tensorflow::Tensor>>* multi_outputs) override;

  tensorflow::Status RunByTensorNames(
      const RunOptions& run_options,
      absl::Span<const std::pair<std::string, tensorflow::Tensor>> inputs,
      absl::Span<const std::string> output_tensor_names,
      absl::Span<const std::string> target_node_names,
      std::vector<tensorflow::Tensor>* outputs) override;

 private:
  // The result of loading signature(s).
  struct LoadingResult {
    std::string name;
    tfrt::BefBuffer bef;
    tfrt::RCReference<tfrt::BEFFile> bef_file;
    std::unique_ptr<tfrt::ResourceContext> resource_context;
  };

  // Imports a subgraph as an MLIR module with the specified `input_nodes`,
  // `output_nodes`.
  tensorflow::StatusOr<mlir::OwningOpRef<mlir::ModuleOp>> ImportSubgraph(
      mlir::MLIRContext* context,
      const tensorflow::GraphImportConfig::InputArrays& input_nodes,
      const std::vector<std::string>& output_nodes,
      const std::vector<std::string>& target_nodes);

  // Given the joined signature, loads the subgraph and returns loading result.
  tensorflow::StatusOr<
      std::reference_wrapper<const SavedModelImpl::LoadingResult>>
  LoadJoinedSignature(const JoinedSignature& joined_signature)
      TF_EXCLUSIVE_LOCKS_REQUIRED(loading_result_cache_mu_);

  // Returns the loading result given the signature names.
  tensorflow::StatusOr<
      std::reference_wrapper<const SavedModelImpl::LoadingResult>>
  GetOrCreateLoadingResult(absl::Span<const std::string> names)
      TF_LOCKS_EXCLUDED(loading_result_cache_mu_);

  // Runs `func` with the given inputs, and outputs the result.
  tensorflow::Status RunInternal(const RunOptions& run_options,
                                 absl::string_view signature_name,
                                 const tfrt::Function& func,
                                 absl::Span<const tensorflow::Tensor> inputs,
                                 absl::Span<const tensorflow::Tensor> captures,
                                 std::vector<tensorflow::Tensor>* outputs,
                                 tfrt::ResourceContext* resource_context);

  Options options_;
  // `meta_graph_def_` only contains metadata of the model. The graph_def field
  // is removed.
  //
  // TODO(b/191931702): We should only keep content that are actually used
  // (eg. SignatureDefs), instead of keeping the whole saved model, to avoid
  // unnecessary memory usage.
  tensorflow::MetaGraphDef meta_graph_def_;
  tfrt::BefBuffer bef_;
  tfrt::RCReference<tfrt::BEFFile> bef_file_;
  tfrt::RequestDeadlineTracker req_deadline_tracker_;
  absl::flat_hash_map<std::string, internal::Signature> signatures_;
  std::unique_ptr<FallbackState> fallback_state_;
  // TODO(b/178227859): Change the hardcoding of this specific TPU resource
  // (TpuModelResource) to a general and plugable interface.
  std::unique_ptr<tfrt::tpu::TpuModelResource> tpu_model_resource_;
  std::unique_ptr<tfrt::ResourceContext> resource_context_;
  tensorflow::mutex loading_result_cache_mu_;
  // For pointer stability of values in `absl::flat_hash_map<>`, additional
  // `std::unique_ptr<>` is necessary. (See https://abseil.io/tips/136.)
  absl::flat_hash_map<std::string /*joined_name*/,
                      std::unique_ptr<LoadingResult>>
      loading_result_cache_ TF_GUARDED_BY(loading_result_cache_mu_);
  std::unique_ptr<GraphExecutor> graph_executor_;
};

}  // namespace tfrt_stub
}  // namespace tensorflow

namespace tfrt {

using SavedModel = ::tensorflow::tfrt_stub::SavedModel;
using SavedModelImpl = ::tensorflow::tfrt_stub::SavedModelImpl;
using TensorSpec = ::tensorflow::tfrt_stub::TensorSpec;
using FunctionMetadata = ::tensorflow::tfrt_stub::FunctionMetadata;

namespace internal {
using Signature = ::tensorflow::tfrt_stub::internal::Signature;
}

}  // namespace tfrt

#endif  // TENSORFLOW_CORE_TFRT_SAVED_MODEL_SAVED_MODEL_H_
