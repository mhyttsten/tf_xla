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

#ifndef TENSORFLOW_LITE_DELEGATES_GPU_CL_INFERENCE_CONTEXT_H_
#define TENSORFLOW_LITE_DELEGATES_GPU_CL_INFERENCE_CONTEXT_H_
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
class MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSclPSinference_contextDTh {
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
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSclPSinference_contextDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSclPSinference_contextDTh() {
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


#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "tensorflow/lite/delegates/gpu/cl/buffer.h"
#include "tensorflow/lite/delegates/gpu/cl/cl_command_queue.h"
#include "tensorflow/lite/delegates/gpu/cl/cl_operation.h"
#include "tensorflow/lite/delegates/gpu/cl/environment.h"
#include "tensorflow/lite/delegates/gpu/cl/gpu_object.h"
#include "tensorflow/lite/delegates/gpu/cl/opencl_wrapper.h"
#include "tensorflow/lite/delegates/gpu/cl/recordable_queue_builder.h"
#include "tensorflow/lite/delegates/gpu/cl/serialization_generated.h"
#include "tensorflow/lite/delegates/gpu/cl/tensor.h"
#include "tensorflow/lite/delegates/gpu/common/gpu_model.h"
#include "tensorflow/lite/delegates/gpu/common/model.h"
#include "tensorflow/lite/delegates/gpu/common/model_hints.h"
#include "tensorflow/lite/delegates/gpu/common/precision.h"
#include "tensorflow/lite/delegates/gpu/common/status.h"
#include "tensorflow/lite/delegates/gpu/common/task/tensor_desc.h"
#include "tensorflow/lite/delegates/gpu/common/tensor.h"

namespace tflite {
namespace gpu {
namespace cl {

struct CLNode {
  ClOperation cl_operation;
  std::vector<ValueId> inputs;
  std::vector<ValueId> outputs;

  // Mostly for debug purposes.
  std::string name;

  CLNode() = default;

  CLNode(CLNode&& node) = default;
  CLNode& operator=(CLNode&& node) = default;
  CLNode(const CLNode&) = delete;
  CLNode& operator=(const CLNode&) = delete;
};

enum class TensorType { kVariable, kConst, kExternal, kRuntime };

class InferenceContext {
 public:
  absl::Status InitFromGraph(const CreateGpuModelInfo& create_info,
                             const GraphFloat32& graph, Environment* env,
                             std::vector<uint8_t>* serialized_model = nullptr);

  absl::Status InitFromGpuModel(
      const CreateGpuModelInfo& create_info, GpuModel* gpu_model,
      Environment* env, std::vector<uint8_t>* serialized_model = nullptr,
      Buffer* shared_buffer = nullptr);

  // Applies OpenCL-specific transformations to the graph before the
  // initialization. These transformations are either impossible or useless in
  // other backends.
  absl::Status InitFromGraphWithTransforms(
      const CreateGpuModelInfo& create_info, GraphFloat32* graph,
      Environment* env, std::vector<uint8_t>* serialized_model = nullptr);

  absl::Status AddToQueue(CLCommandQueue* queue);
  absl::Status Profile(ProfilingCommandQueue* queue, ProfilingInfo* result);
  // for profiling and memory statistics
  uint64_t GetSizeOfMemoryAllocatedForIntermediateTensors() const;
  uint64_t GetConstantTensorsSize() const;

  absl::Status SetInputTensor(ValueId id, const TensorFloat32& tensor,
                              CLCommandQueue* queue);

  // It will work only with input/output tensor ids. For all other ids we don't
  // have any guarantees.
  Tensor* GetTensor(ValueId id);

  absl::Status GetOutputTensor(ValueId id, CLCommandQueue* queue,
                               TensorFloat32* result);

  const std::vector<ValueId>& GetInputIds() const {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSclPSinference_contextDTh mht_0(mht_0_v, 269, "", "./tensorflow/lite/delegates/gpu/cl/inference_context.h", "GetInputIds");
 return input_ids_; }
  const std::vector<ValueId>& GetOutputIds() const {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSclPSinference_contextDTh mht_1(mht_1_v, 273, "", "./tensorflow/lite/delegates/gpu/cl/inference_context.h", "GetOutputIds");
 return output_ids_; }

  absl::Status RestoreDeserialized(
      const absl::Span<const uint8_t> serialized_model, Environment* env,
      CreateGpuModelInfo* create_info = nullptr);

  // Can be used only with ids from external_mutable_tensors in create_info
  // Must be called after initialization and before execution
  absl::Status SetTensor(const ValueId& tensor_id, Tensor* tensor_ptr);

 private:
  flatbuffers::Offset<data::InferenceContext> Encode(
      const CLDevice& device, const ProgramCache& program_cache,
      flatbuffers::Offset<tflite::gpu::data::GpuModel> gpu_model_fb,
      flatbuffers::FlatBufferBuilder* builder);

  void InitFromGpuModel(GpuModel* gpu_model);

  absl::Status AllocateMemory(const GpuModel& gpu_model,
                              const GpuInfo& gpu_info,
                              const CreateGpuModelInfo* create_info,
                              CLContext* context);

  absl::Status AllocateConstTensors(const GpuModel& gpu_model,
                                    CLContext* context);

  absl::Status AllocateVariableTensors(const GpuModel& gpu_model,
                                       CLContext* context);

  absl::Status AllocateBufferBasedTensors(const GpuModel& gpu_model,
                                          const GpuInfo& gpu_info,
                                          const CreateGpuModelInfo* create_info,
                                          CLContext* context);

  absl::Status AllocateStrongShapesTensors(
      const GpuModel& gpu_model, const GpuInfo& gpu_info,
      const CreateGpuModelInfo* create_info, CLContext* context);

  void BindMemoryToOperations();
  absl::Status Compile(const CreationContext& creation_context);
  absl::Status Tune(TuningType tuning_type, const GpuInfo& gpu_info,
                    ProfilingCommandQueue* profiling_queue);
  absl::Status UpdateParams();
  void PrepareExternal();

  void InitRecordableQueue(Environment* env);

  absl::Status ProfileTime(ProfilingCommandQueue* queue, ProfilingInfo* result);

  struct ExecutionHints {
    bool need_flush = false;

    bool flush_periodically = false;
    int flush_period = 1;

    // In order to reduce memory leak on Mali a pipeline needs to be
    // synchronized with CPU to prevent growing internal global OpenCL kernel
    // pool. One trick is to enqueue an event from a previous run. Most of the
    // time is should already be executed on GPU and should not stall the
    // pipeline.
    bool need_manual_release = false;
    CLEvent prev_enqueue_start_point;

    void Init(const GpuInfo& gpu_info);
  };
  ExecutionHints execution_hints_;

  // Directly mapped nodes from graph, but some of them "inactive" due
  //  to fusion (inactive = fused).
  // Memory is allocated only once, in ConvertOperations, and is not modified
  //  anywhere.
  std::vector<CLNode> nodes_;

  absl::flat_hash_map<ValueId, Tensor*> external_immutable_tensors_;
  absl::flat_hash_map<ValueId, Tensor*> external_mutable_tensors_;
  absl::flat_hash_map<ValueId, std::vector<int>> external_tensor_to_nodes_;

  std::map<ValueId, Tensor> const_tensors_;

  std::map<ValueId, ValueId> variable_ids_and_refs_;
  std::map<ValueId, Tensor> variable_tensors_;

  std::unique_ptr<Buffer> shared_buffers_parent_;
  Buffer* shared_buffers_parent_ptr_ = nullptr;
  std::vector<Buffer> shared_buffers_;
  std::vector<Tensor>
      shared_buffer_tensors_;  // use references to memory from shared_buffers_
  std::map<ValueId, int> graph_ids_to_shared_buffer_tensors_;

  std::map<ValueId, Tensor> strong_shape_tensors_;
  std::map<ValueId, ValueId> graph_ids_to_strong_shape_tensors_;

  std::vector<ValueId> input_ids_;
  std::vector<ValueId> output_ids_;

  std::unique_ptr<RecordableQueue> recordable_queue_ = nullptr;

  GpuInfo gpu_info_;
};

absl::Status GetInOutRefs(const absl::Span<const uint8_t> serialized_model,
                          std::vector<int64_t>* in_refs,
                          std::vector<int64_t>* out_refs);

absl::Status GetTotalBufferSizeForTensors(const GpuModel& gpu_model,
                                          const CreateGpuModelInfo& create_info,
                                          const GpuInfo& gpu_info,
                                          uint64_t* result);

}  // namespace cl
}  // namespace gpu
}  // namespace tflite

#endif  // TENSORFLOW_LITE_DELEGATES_GPU_CL_INFERENCE_CONTEXT_H_
