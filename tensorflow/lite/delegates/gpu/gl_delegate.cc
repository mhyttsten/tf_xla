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
class MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSgl_delegateDTcc {
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
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSgl_delegateDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSgl_delegateDTcc() {
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

#include "tensorflow/lite/delegates/gpu/gl_delegate.h"

#include <EGL/egl.h>
#include <GLES3/gl31.h>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <string>
#include <unordered_set>
#include <vector>

#include "absl/types/span.h"
#include "tensorflow/lite/builtin_ops.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/delegates/gpu/common/convert.h"
#include "tensorflow/lite/delegates/gpu/common/model.h"
#include "tensorflow/lite/delegates/gpu/common/model_builder.h"
#include "tensorflow/lite/delegates/gpu/common/model_transformer.h"
#include "tensorflow/lite/delegates/gpu/common/shape.h"
#include "tensorflow/lite/delegates/gpu/common/status.h"
#include "tensorflow/lite/delegates/gpu/common/tensor.h"
#include "tensorflow/lite/delegates/gpu/common/transformations/model_transformations.h"
#include "tensorflow/lite/delegates/gpu/gl/api.h"
#include "tensorflow/lite/delegates/gpu/gl/command_queue.h"
#include "tensorflow/lite/delegates/gpu/gl/compiler.h"
#include "tensorflow/lite/delegates/gpu/gl/converters/bhwc_to_phwc4.h"
#include "tensorflow/lite/delegates/gpu/gl/converters/phwc4_to_bhwc.h"
#include "tensorflow/lite/delegates/gpu/gl/egl_environment.h"
#include "tensorflow/lite/delegates/gpu/gl/gl_call.h"
#include "tensorflow/lite/delegates/gpu/gl/kernels/registry.h"
#include "tensorflow/lite/delegates/gpu/gl/request_gpu_info.h"
#include "tensorflow/lite/delegates/gpu/gl/workgroups/best_effort_calculator.h"
#include "tensorflow/lite/minimal_logging.h"

#ifndef TFLITE_GPU_BINARY_RELEASE
#include "flatbuffers/flatbuffers.h"  // from @flatbuffers
#include "tensorflow/lite/delegates/gpu/gl/metadata_generated.h"
#include "tensorflow/lite/schema/schema_generated.h"
#endif  // TFLITE_GPU_BINARY_RELEASE

namespace tflite {
namespace gpu {
namespace gl {
namespace {

// Forward declarations.
TfLiteStatus DelegatePrepare(TfLiteContext* context, TfLiteDelegate* delegate);
TfLiteStatus DelegateCopyFromBufferHandle(
    TfLiteContext* context, TfLiteDelegate* delegate,
    TfLiteBufferHandle buffer_handle,  // ValueId
    TfLiteTensor* tensor);
TfLiteStatus DelegateCopyToBufferHandle(
    TfLiteContext* context, TfLiteDelegate* delegate,
    TfLiteBufferHandle buffer_handle,  // ValueId
    TfLiteTensor* tensor);

inline bool IsPHWC4(const BHWC& shape) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSgl_delegateDTcc mht_0(mht_0_v, 242, "", "./tensorflow/lite/delegates/gpu/gl_delegate.cc", "IsPHWC4");

  return shape.c == 4 || (shape.h == 1 && shape.w == 1 && shape.c % 4 == 0);
}

class Delegate {
  struct ValueRef {
    BHWC shape;
    int tensor_index;
  };

 public:
  explicit Delegate(const TfLiteGpuDelegateOptions* options) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSgl_delegateDTcc mht_1(mht_1_v, 256, "", "./tensorflow/lite/delegates/gpu/gl_delegate.cc", "Delegate");

    if (options) {
      options_ = *options;
    } else {
      // Default options.
      options_.metadata = nullptr;
      options_.compile_options.precision_loss_allowed = 0;
      options_.compile_options.preferred_gl_object_type =
          TFLITE_GL_OBJECT_TYPE_FASTEST;
      options_.compile_options.dynamic_batch_enabled = 0;
    }
  }

  absl::Status CopyFromBufferHandle(TfLiteBufferHandle handle,
                                    TfLiteTensor* tensor) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSgl_delegateDTcc mht_2(mht_2_v, 273, "", "./tensorflow/lite/delegates/gpu/gl_delegate.cc", "CopyFromBufferHandle");

    ValueRef ref;
    RETURN_IF_ERROR(FindObject(handle, &ref));
    auto buffer = phwc4_objects_.FindBuffer(handle);
    return buffer->MappedRead<float>([&](absl::Span<const float> data) {
      tensor->data_is_stale = false;
      return ConvertFromPHWC4(
          data, ref.shape,
          absl::MakeSpan(tensor->data.f, tensor->bytes / sizeof(float)));
    });
  }

  absl::Status CopyToBufferHandle(TfLiteBufferHandle handle,
                                  TfLiteTensor* tensor) const {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSgl_delegateDTcc mht_3(mht_3_v, 289, "", "./tensorflow/lite/delegates/gpu/gl_delegate.cc", "CopyToBufferHandle");

    ValueRef ref;
    RETURN_IF_ERROR(FindObject(handle, &ref));
    auto buffer = phwc4_objects_.FindBuffer(handle);
    return buffer->MappedWrite<float>([&](absl::Span<float> data) {
      return ConvertToPHWC4(
          absl::MakeConstSpan(tensor->data.f, tensor->bytes / sizeof(float)),
          ref.shape, data);
    });
  }

  absl::Status BindBufferToTensor(GLuint ssbo, int tensor_index) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSgl_delegateDTcc mht_4(mht_4_v, 303, "", "./tensorflow/lite/delegates/gpu/gl_delegate.cc", "BindBufferToTensor");

    int64_t bytes_size;
    RETURN_IF_ERROR(GetSSBOSize(ssbo, &bytes_size));
    return bhwc_objects_.RegisterBuffer(
        tensor_index, GlBuffer(GL_SHADER_STORAGE_BUFFER, ssbo, bytes_size,
                               /* offset = */ 0,
                               /* has_ownership = */ false));
  }

  absl::Status Prepare(TfLiteContext* context,
                       const TfLiteDelegateParams* delegate_params) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSgl_delegateDTcc mht_5(mht_5_v, 316, "", "./tensorflow/lite/delegates/gpu/gl_delegate.cc", "Prepare");

    // Extract TFLite delegate execution plan from the context and convert it
    // into GraphFloat32.
    GraphFloat32 graph;
    RETURN_IF_ERROR(BuildModel(context, delegate_params, &graph));

    // Apply general transformations on the graph.
    ModelTransformer transformer(&graph);
    if (!ApplyModelTransformations(&transformer)) {
      return absl::InternalError("Graph transformations failed");
    }

    if (!env_) RETURN_IF_ERROR(EglEnvironment::NewEglEnvironment(&env_));

    // TODO(impjdi): Remove code duplication.
    auto values = graph.values();
    auto find_value = [&](int tensor_index) -> Value* {
      for (auto value : values) {
        if (value->tensor.ref == tensor_index) return value;
      }
      return nullptr;
    };
    tensors_.reserve(values.back()->id + 1);
    for (auto value : values) {
      if (tensors_.size() <= value->id) {
        tensors_.resize(value->id + 1);
      }
      tensors_[value->id] = {value->tensor.shape, 0};
    }

    std::unordered_set<int> tflite_graph_io;  // NOLINT

    // Prepare graph inputs.
    //
    // Note that graph.inputs() cannot be used directly, as the notion of
    // graph input has a different meaning in public API and GPU-internal API.
    {
      inputs_.clear();
      inputs_.reserve(delegate_params->input_tensors->size);
      for (int i = 0; i < delegate_params->input_tensors->size; ++i) {
        const int tensor_index = delegate_params->input_tensors->data[i];
        auto* tensor = context->tensors + tensor_index;
        if (tensor->allocation_type == TfLiteAllocationType::kTfLiteMmapRo) {
          continue;
        }
        tflite_graph_io.insert(tensor_index);
        const auto* input = find_value(tensor_index);
        if (!input || tensor->type != TfLiteType::kTfLiteFloat32) {
          return absl::NotFoundError("Input tensor is not found in the graph.");
        }

        inputs_.push_back(input->id);
        tensor->buffer_handle = input->id;
        tensor->delegate = &delegate_;
        tensors_[input->id].tensor_index = tensor_index;

        // Create phwc4 input buffer.
        // Check whether there is externally provided object is already in
        // PHWC4. If yes, we may skip conversion step.
        // We need to keep same buffer in bhwc_objects_ to indicate there is
        // externally provided buffer.
        auto external_buffer = bhwc_objects_.FindBuffer(tensor_index);
        GlBuffer buffer;
        if (IsPHWC4(input->tensor.shape) && external_buffer) {
          buffer = external_buffer->MakeRef();
        } else {
          RETURN_IF_ERROR(CreateReadWriteShaderStorageBuffer<float>(
              GetElementsSizeForPHWC4(input->tensor.shape), &buffer));
        }
        RETURN_IF_ERROR(
            phwc4_objects_.RegisterBuffer(input->id, std::move(buffer)));
      }
    }

    // Prepare graph outputs.
    //
    // Note that graph.outputs() cannot be used directly, as the notion of
    // graph output has a different meaning in public API and GPU-internal API.
    {
      outputs_.clear();
      outputs_.reserve(delegate_params->output_tensors->size);
      for (int i = 0; i < delegate_params->output_tensors->size; ++i) {
        const int tensor_index = delegate_params->output_tensors->data[i];
        auto* tensor = context->tensors + tensor_index;
        tflite_graph_io.insert(tensor_index);
        const auto* output = find_value(tensor_index);
        if (!output || tensor->type != TfLiteType::kTfLiteFloat32) {
          return absl::NotFoundError(
              "Output tensor is not found in the graph.");
        }

        outputs_.push_back(output->id);
        tensor->buffer_handle = output->id;
        tensor->delegate = &delegate_;
        tensors_[output->id].tensor_index = tensor_index;

        // Create phwc4 output buffer.
        // Check whether there is externally provided object is already in
        // PHWC4. If yes, we may skip conversion step.
        auto external_buffer = bhwc_objects_.FindBuffer(tensor_index);
        GlBuffer buffer;
        if (IsPHWC4(output->tensor.shape) && external_buffer) {
          buffer = external_buffer->MakeRef();
        } else {
          RETURN_IF_ERROR(CreateReadWriteShaderStorageBuffer<float>(
              GetElementsSizeForPHWC4(output->tensor.shape), &buffer));
        }
        RETURN_IF_ERROR(
            phwc4_objects_.RegisterBuffer(output->id, std::move(buffer)));
      }
    }

    // Create shaders to convert from/to phwc4.
    RETURN_IF_ERROR(ConverterBhwcToPhwc4::Create(&bhwc_to_phwc4_));
    RETURN_IF_ERROR(ConverterPhwc4ToBhwc::Create(&phwc4_to_bhwc_));

    // Compile model.
    CompilationOptions compile_options;
    compile_options.allow_precision_loss =
        static_cast<bool>(options_.compile_options.precision_loss_allowed);
    compile_options.preferred_obj_type = static_cast<ObjectType>(
        options_.compile_options.preferred_gl_object_type);
    compile_options.ref_obj_type = static_cast<ObjectType>(
        options_.compile_options.preferred_gl_object_type);
    compile_options.dynamic_batch =
        static_cast<bool>(options_.compile_options.dynamic_batch_enabled);
    compile_options.inline_parameters =
        static_cast<bool>(options_.compile_options.inline_parameters);
    auto shaders = NewNodeShaderRegistry();
    GpuInfo gpu_info;
    RETURN_IF_ERROR(RequestGpuInfo(&gpu_info));
    command_queue_ = NewCommandQueue(gpu_info);
    auto workgroups_calculator =
        BestEffortWorkgroupsCalculator(options_.metadata, gpu_info);
    std::unique_ptr<CompiledModel> compiled_model;
    RETURN_IF_ERROR(Compile(compile_options, graph, tflite_graph_io, *shaders,
                            *workgroups_calculator, &compiled_model));

    // Create inference context.
    const RuntimeOptions runtime_options;
    RETURN_IF_ERROR(compiled_model->NewRun(runtime_options, &phwc4_objects_,
                                           command_queue_.get(),
                                           &inference_context_));
    return absl::OkStatus();
  }

  absl::Status Invoke(TfLiteContext* context) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSgl_delegateDTcc mht_6(mht_6_v, 465, "", "./tensorflow/lite/delegates/gpu/gl_delegate.cc", "Invoke");

    const EGLContext egl_context_at_delegate_init = env_->context().context();
    const EGLContext egl_context_at_delegate_invoke = eglGetCurrentContext();
    if (egl_context_at_delegate_init != egl_context_at_delegate_invoke) {
      return absl::FailedPreconditionError(
          "Delegate should run on the same thread where it was initialized.");
    }

    // Push input data from a tensor to GPU.
    for (ValueId id : inputs_) {
      const ValueRef& ref = tensors_[id];
      auto external_object = bhwc_objects_.FindBuffer(ref.tensor_index);
      if (external_object) {
        // Use input from GPU.
        // Conversion is needed only when external object is not phwc4.
        if (!IsPHWC4(tensors_[id].shape)) {
          RETURN_IF_ERROR(bhwc_to_phwc4_.Convert(
              ref.shape, *external_object, command_queue_.get(),
              phwc4_objects_.FindBuffer(id)));
        }
      } else {
        // Copy from CPU to GPU
        TfLiteTensor& tensor = context->tensors[ref.tensor_index];
        RETURN_IF_ERROR(CopyToBufferHandle(id, &tensor));
      }
    }

    // Run inference.
    RETURN_IF_ERROR(inference_context_->Reset());
    RETURN_IF_ERROR(inference_context_->Execute());

    // Push output data from GPU to a tensor.
    bool finished_gpu_processing = false;
    for (ValueId id : outputs_) {
      const ValueRef& ref = tensors_[id];
      auto external_object = bhwc_objects_.FindBuffer(ref.tensor_index);
      if (external_object) {
        // Convert data from PHWC4 to BHWC and leave it in GPU object.
        // Conversion is needed only when external object is not phwc4.
        if (!IsPHWC4(tensors_[id].shape)) {
          RETURN_IF_ERROR(
              phwc4_to_bhwc_.Convert(ref.shape, *phwc4_objects_.FindBuffer(id),
                                     command_queue_.get(), external_object));
        }
      } else {
        // Wait until all GPU command are completed. This call leads to a lower
        // processing latency because a buffer reading below will not stall if
        // data is not yet ready.
        if (!finished_gpu_processing) {
          RETURN_IF_ERROR(command_queue_->WaitForCompletion());
          finished_gpu_processing = true;
        }
        // Copy from GPU to CPU.
        TfLiteTensor& tensor = context->tensors[ref.tensor_index];
        RETURN_IF_ERROR(CopyFromBufferHandle(id, &tensor));
      }
    }
    return absl::OkStatus();
  }

  TfLiteDelegate* tflite_delegate() {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSgl_delegateDTcc mht_7(mht_7_v, 528, "", "./tensorflow/lite/delegates/gpu/gl_delegate.cc", "tflite_delegate");
 return &delegate_; }

 private:
  absl::Status FindObject(ValueId id, ValueRef* ref) const {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSgl_delegateDTcc mht_8(mht_8_v, 534, "", "./tensorflow/lite/delegates/gpu/gl_delegate.cc", "FindObject");

    if (id >= tensors_.size()) {
      return absl::InvalidArgumentError("Invalid buffer id");
    }
    *ref = tensors_[id];
    return absl::OkStatus();
  }

  TfLiteDelegate delegate_ = {
      reinterpret_cast<void*>(this),  // .data_
      DelegatePrepare,                // .Prepare
      DelegateCopyFromBufferHandle,   // .CopyFromBufferHandle
      DelegateCopyToBufferHandle,     // .CopyToBufferHandle
      nullptr,                        // .FreeBufferHandle
      kTfLiteDelegateFlagsNone,       // .flags
  };

  TfLiteGpuDelegateOptions options_;

  std::unique_ptr<EglEnvironment> env_;
  std::vector<ValueRef> tensors_;  // indexed by ValueId
  std::vector<ValueId> inputs_;
  std::vector<ValueId> outputs_;
  ObjectManager phwc4_objects_;
  ObjectManager bhwc_objects_;  // key is tensor_index
  ConverterPhwc4ToBhwc phwc4_to_bhwc_;
  ConverterBhwcToPhwc4 bhwc_to_phwc4_;
  std::unique_ptr<CommandQueue> command_queue_;
  std::unique_ptr<InferenceContext> inference_context_;
};

inline Delegate* GetGpuDelegate(TfLiteNode* node) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSgl_delegateDTcc mht_9(mht_9_v, 568, "", "./tensorflow/lite/delegates/gpu/gl_delegate.cc", "GetGpuDelegate");

  return reinterpret_cast<Delegate*>(node->user_data);
}

inline Delegate* GetGpuDelegate(TfLiteDelegate* delegate) {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSgl_delegateDTcc mht_10(mht_10_v, 575, "", "./tensorflow/lite/delegates/gpu/gl_delegate.cc", "GetGpuDelegate");

  return reinterpret_cast<Delegate*>(delegate->data_);
}

TfLiteStatus DelegatePrepare(TfLiteContext* context, TfLiteDelegate* delegate) {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSgl_delegateDTcc mht_11(mht_11_v, 582, "", "./tensorflow/lite/delegates/gpu/gl_delegate.cc", "DelegatePrepare");

  const TfLiteRegistration kRegistration = {
      // .init
      [](TfLiteContext* context, const char* buffer, size_t) -> void* {
        const auto* params =
            reinterpret_cast<const TfLiteDelegateParams*>(buffer);
        auto* gpu_delegate = GetGpuDelegate(params->delegate);
        // Everything below should happen in prepare function call, but TFLite
        // for whatever reason forbids that.
        const auto status = gpu_delegate->Prepare(context, params);
        if (status.ok()) return gpu_delegate;
        TF_LITE_KERNEL_LOG(context, "TfLiteGpuDelegate Prepare: %s",
                           std::string(status.message()).c_str());
        return nullptr;
      },
      // .free
      [](TfLiteContext*, void* buffer) -> void {},
      // .prepare
      [](TfLiteContext* context, TfLiteNode* node) -> TfLiteStatus {
        return node->user_data ? kTfLiteOk : kTfLiteError;
      },
      // .invoke
      [](TfLiteContext* context, TfLiteNode* node) -> TfLiteStatus {
        const auto status = GetGpuDelegate(node)->Invoke(context);
        if (status.ok()) return kTfLiteOk;
        TF_LITE_KERNEL_LOG(context, "TfLiteGpuDelegate Invoke: %s",
                           std::string(status.message()).c_str());
        return kTfLiteError;
      },
      nullptr,              // .profiling_string
      0,                    // .builtin_code
      "TfLiteGpuDelegate",  // .custom_name
      1,                    // .version
  };
  TfLiteIntArray* ops_to_replace = GetOpsToReplace(context);
  const auto status = context->ReplaceNodeSubsetsWithDelegateKernels(
      context, kRegistration, ops_to_replace, delegate);
  TfLiteIntArrayFree(ops_to_replace);
  return status;
}

TfLiteStatus DelegateCopyFromBufferHandle(TfLiteContext* context,
                                          TfLiteDelegate* delegate,
                                          TfLiteBufferHandle buffer_handle,
                                          TfLiteTensor* tensor) {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSgl_delegateDTcc mht_12(mht_12_v, 629, "", "./tensorflow/lite/delegates/gpu/gl_delegate.cc", "DelegateCopyFromBufferHandle");

  auto* gpu_delegate = GetGpuDelegate(delegate);
  if (!gpu_delegate) return kTfLiteError;
  const auto status = gpu_delegate->CopyFromBufferHandle(buffer_handle, tensor);
  if (status.ok()) return kTfLiteOk;
  TF_LITE_KERNEL_LOG(context, "TfLiteGpuDelegate CopyFromBufferHandle: %s",
                     std::string(status.message()).c_str());
  return kTfLiteError;
}

TfLiteStatus DelegateCopyToBufferHandle(TfLiteContext* context,
                                        TfLiteDelegate* delegate,
                                        TfLiteBufferHandle buffer_handle,
                                        TfLiteTensor* tensor) {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSgl_delegateDTcc mht_13(mht_13_v, 645, "", "./tensorflow/lite/delegates/gpu/gl_delegate.cc", "DelegateCopyToBufferHandle");

  auto* gpu_delegate = GetGpuDelegate(delegate);
  if (!gpu_delegate) return kTfLiteError;
  const auto status = gpu_delegate->CopyToBufferHandle(buffer_handle, tensor);
  if (status.ok()) return kTfLiteOk;
  TF_LITE_KERNEL_LOG(context, "TfLiteGpuDelegate CopyToBufferHandle: %s",
                     std::string(status.message()).c_str());
  return kTfLiteError;
}

}  // namespace
}  // namespace gl
}  // namespace gpu
}  // namespace tflite

TfLiteGlCompileOptions TfLiteGlCompileOptionsDefault() {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSgl_delegateDTcc mht_14(mht_14_v, 663, "", "./tensorflow/lite/delegates/gpu/gl_delegate.cc", "TfLiteGlCompileOptionsDefault");

  TfLiteGlCompileOptions options;
  options.precision_loss_allowed = 0;
  options.preferred_gl_object_type = TFLITE_GL_OBJECT_TYPE_FASTEST;
  options.dynamic_batch_enabled = 0;
  options.inline_parameters = 0;
  return options;
}

TfLiteGpuDelegateOptions TfLiteGpuDelegateOptionsDefault() {
   std::vector<std::string> mht_15_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSgl_delegateDTcc mht_15(mht_15_v, 675, "", "./tensorflow/lite/delegates/gpu/gl_delegate.cc", "TfLiteGpuDelegateOptionsDefault");

  TfLiteGpuDelegateOptions options;
  options.metadata = nullptr;
  options.compile_options = TfLiteGlCompileOptionsDefault();
  return options;
}

TfLiteDelegate* TfLiteGpuDelegateCreate(
    const TfLiteGpuDelegateOptions* options) {
   std::vector<std::string> mht_16_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSgl_delegateDTcc mht_16(mht_16_v, 686, "", "./tensorflow/lite/delegates/gpu/gl_delegate.cc", "TfLiteGpuDelegateCreate");

  TFLITE_LOG_PROD_ONCE(tflite::TFLITE_LOG_INFO,
                       "Created TensorFlow Lite delegate for GPU.");
  auto* gpu_delegate = new tflite::gpu::gl::Delegate(options);
  return gpu_delegate ? gpu_delegate->tflite_delegate() : nullptr;
}

void TfLiteGpuDelegateDelete(TfLiteDelegate* delegate) {
   std::vector<std::string> mht_17_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSgl_delegateDTcc mht_17(mht_17_v, 696, "", "./tensorflow/lite/delegates/gpu/gl_delegate.cc", "TfLiteGpuDelegateDelete");

  delete tflite::gpu::gl::GetGpuDelegate(delegate);
}

TfLiteStatus TfLiteGpuDelegateBindBufferToTensor(TfLiteDelegate* delegate,
                                                 GLuint buffer,
                                                 int tensor_index) {
   std::vector<std::string> mht_18_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSgl_delegateDTcc mht_18(mht_18_v, 705, "", "./tensorflow/lite/delegates/gpu/gl_delegate.cc", "TfLiteGpuDelegateBindBufferToTensor");

  auto* gpu_delegate = tflite::gpu::gl::GetGpuDelegate(delegate);
  return gpu_delegate &&
                 gpu_delegate->BindBufferToTensor(buffer, tensor_index).ok()
             ? kTfLiteOk
             : kTfLiteError;
}

#ifndef TFLITE_GPU_BINARY_RELEASE
const uint8_t* TfLiteGpuDelegateGetModelMetadata(const void* tflite_model) {
  const auto* model = reinterpret_cast<const tflite::Model*>(tflite_model);
  if (!model || !model->metadata_buffer() || !model->buffers()) return nullptr;
  for (int32_t buffer_index : *model->metadata_buffer()) {
    if (buffer_index < 0 && buffer_index >= model->buffers()->size()) continue;
    const tflite::Buffer* buffer = model->buffers()->Get(buffer_index);
    if (!buffer) continue;
    const uint8_t* data = buffer->data()->data();
    if (!flatbuffers::BufferHasIdentifier(
            data, tflite::gpu::gl::data::FlowMetadataIdentifier())) {
      continue;
    }
    flatbuffers::Verifier verifier(data, buffer->data()->size());
    return tflite::gpu::gl::data::VerifyFlowMetadataBuffer(verifier) ? data
                                                                     : nullptr;
  }
  return nullptr;
}
#endif  // TFLITE_GPU_BINARY_RELEASE
