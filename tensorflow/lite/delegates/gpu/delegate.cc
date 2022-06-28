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
class MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSdelegateDTcc {
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
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSdelegateDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSdelegateDTcc() {
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

#include "tensorflow/lite/delegates/gpu/delegate.h"

#include <cstdint>
#include <memory>
#include <thread>  // NOLINT(build/c++11)
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/memory/memory.h"
#include "absl/types/span.h"
#include "tensorflow/lite/builtin_ops.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/delegates/gpu/api.h"
#include "tensorflow/lite/delegates/gpu/cl/api.h"
#include "tensorflow/lite/delegates/gpu/cl/opencl_wrapper.h"
#include "tensorflow/lite/delegates/gpu/cl/tensor_type_util.h"
#include "tensorflow/lite/delegates/gpu/cl/util.h"
#include "tensorflow/lite/delegates/gpu/common/model.h"
#include "tensorflow/lite/delegates/gpu/common/model_builder.h"
#include "tensorflow/lite/delegates/gpu/common/model_transformer.h"
#include "tensorflow/lite/delegates/gpu/common/quantization_util.h"
#include "tensorflow/lite/delegates/gpu/common/status.h"
#include "tensorflow/lite/delegates/serialization.h"
#include "tensorflow/lite/kernels/internal/optimized/optimized_ops.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/minimal_logging.h"

#ifndef CL_DELEGATE_NO_GL
#include "tensorflow/lite/delegates/gpu/gl/api2.h"
#endif

namespace tflite {
namespace gpu {
namespace {

using delegates::Serialization;
using delegates::SerializationParams;

constexpr char kSerializedDataPrefix[] = "gpuv2_data_";

InferencePriority ToPriority(int32_t priority) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSdelegateDTcc mht_0(mht_0_v, 226, "", "./tensorflow/lite/delegates/gpu/delegate.cc", "ToPriority");

  switch (priority) {
    case TFLITE_GPU_INFERENCE_PRIORITY_AUTO:
      return InferencePriority::AUTO;
    case TFLITE_GPU_INFERENCE_PRIORITY_MAX_PRECISION:
      return InferencePriority::MAX_PRECISION;
    case TFLITE_GPU_INFERENCE_PRIORITY_MIN_LATENCY:
      return InferencePriority::MIN_LATENCY;
    case TFLITE_GPU_INFERENCE_PRIORITY_MIN_MEMORY_USAGE:
      return InferencePriority::MIN_MEMORY_USAGE;
  }
  return InferencePriority::UNKNOWN;
}

InferenceUsage ToUsage(int32_t usage) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSdelegateDTcc mht_1(mht_1_v, 243, "", "./tensorflow/lite/delegates/gpu/delegate.cc", "ToUsage");

  switch (usage) {
    case TFLITE_GPU_INFERENCE_PREFERENCE_FAST_SINGLE_ANSWER:
      return InferenceUsage::FAST_SINGLE_ANSWER;
    case TFLITE_GPU_INFERENCE_PREFERENCE_SUSTAINED_SPEED:
      return InferenceUsage::SUSTAINED_SPEED;
  }
  return InferenceUsage::UNKNOWN;
}

// Forward declarations.
TfLiteStatus DelegatePrepare(TfLiteContext* context, TfLiteDelegate* delegate);

class Delegate {
 public:
  explicit Delegate(const TfLiteGpuDelegateOptionsV2* options)
      : num_delegate_kernels_(0) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSdelegateDTcc mht_2(mht_2_v, 262, "", "./tensorflow/lite/delegates/gpu/delegate.cc", "Delegate");

    delegate_.data_ = reinterpret_cast<void*>(this);
    delegate_.Prepare = DelegatePrepare;
    delegate_.CopyFromBufferHandle = nullptr;
    delegate_.CopyToBufferHandle = nullptr;
    delegate_.FreeBufferHandle = nullptr;
    delegate_.flags = kTfLiteDelegateFlagsNone;
    options_ = options ? *options : TfLiteGpuDelegateOptionsV2Default();
    if (options_.max_delegated_partitions <= 0) {
      options_.max_delegated_partitions = 1;
    }
    if (options_.experimental_flags &
            TFLITE_GPU_EXPERIMENTAL_FLAGS_ENABLE_SERIALIZATION &&
        options_.model_token && options_.serialization_dir) {
      SerializationParams params;
      params.model_token = options_.model_token;
      params.cache_dir = options_.serialization_dir;
      serialization_.reset(new Serialization(params));
    }
  }

  TfLiteDelegate* tflite_delegate() {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSdelegateDTcc mht_3(mht_3_v, 286, "", "./tensorflow/lite/delegates/gpu/delegate.cc", "tflite_delegate");
 return &delegate_; }
  Serialization* serialization() {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSdelegateDTcc mht_4(mht_4_v, 290, "", "./tensorflow/lite/delegates/gpu/delegate.cc", "serialization");
 return serialization_.get(); }
  const TfLiteGpuDelegateOptionsV2& options() const {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSdelegateDTcc mht_5(mht_5_v, 294, "", "./tensorflow/lite/delegates/gpu/delegate.cc", "options");
 return options_; }

  bool IsQuantOpsAllowed() const {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSdelegateDTcc mht_6(mht_6_v, 299, "", "./tensorflow/lite/delegates/gpu/delegate.cc", "IsQuantOpsAllowed");

    return options_.experimental_flags &
           TFLITE_GPU_EXPERIMENTAL_FLAGS_ENABLE_QUANT;
  }
  int MaxDelegatedPartitions() const {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSdelegateDTcc mht_7(mht_7_v, 306, "", "./tensorflow/lite/delegates/gpu/delegate.cc", "MaxDelegatedPartitions");

    return options_.max_delegated_partitions;
  }
  int num_delegate_kernels() const {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSdelegateDTcc mht_8(mht_8_v, 312, "", "./tensorflow/lite/delegates/gpu/delegate.cc", "num_delegate_kernels");
 return num_delegate_kernels_; }

 private:
  TfLiteDelegate delegate_;
  TfLiteGpuDelegateOptionsV2 options_;
  int num_delegate_kernels_ = 0;

  std::unique_ptr<Serialization> serialization_;

  friend class DelegateKernel;
};

// Represent the execution of a subset of nodes on GPU.
class DelegateKernel {
 public:
  explicit DelegateKernel(Delegate* delegate) : delegate_(delegate) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSdelegateDTcc mht_9(mht_9_v, 330, "", "./tensorflow/lite/delegates/gpu/delegate.cc", "DelegateKernel");

    ++delegate_->num_delegate_kernels_;
  }
  ~DelegateKernel() {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSdelegateDTcc mht_10(mht_10_v, 336, "", "./tensorflow/lite/delegates/gpu/delegate.cc", "~DelegateKernel");
 --delegate_->num_delegate_kernels_; }

  absl::Status Prepare(TfLiteContext* context,
                       const TfLiteDelegateParams* delegate_params) {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSdelegateDTcc mht_11(mht_11_v, 342, "", "./tensorflow/lite/delegates/gpu/delegate.cc", "Prepare");

    thread_id_prepare_ = std::this_thread::get_id();

    // Extract TFLite delegate execution plan from the context and convert it
    // into GraphFloat32.
    GraphFloat32 graph;
    std::vector<uint32_t> input_refs;
    std::vector<uint32_t> output_refs;
    RETURN_IF_ERROR(InitializeGraph(context, delegate_params, &graph,
                                    &input_refs, &output_refs));

    std::unique_ptr<InferenceBuilder> builder;
    bool graph_is_destroyed;
    const int experimental_flags = delegate_->options().experimental_flags;
    if (experimental_flags & TFLITE_GPU_EXPERIMENTAL_FLAGS_CL_ONLY) {
      RETURN_IF_ERROR(InitializeOpenClApi(&graph, &builder, &graph_is_destroyed,
                                          context, delegate_params,
                                          delegate_->serialization()));
    } else if (experimental_flags & TFLITE_GPU_EXPERIMENTAL_FLAGS_GL_ONLY) {
      RETURN_IF_ERROR(InitializeOpenGlApi(&graph, &builder));
    } else {
      // By default, we try CL first & fall back to GL if that fails.
      absl::Status status =
          InitializeOpenClApi(&graph, &builder, &graph_is_destroyed, context,
                              delegate_params, delegate_->serialization());
      if (!status.ok()) {
        TF_LITE_KERNEL_LOG(context, std::string(status.message()).c_str());
        TF_LITE_KERNEL_LOG(context, "Falling back to OpenGL");

        // Graph needs to be re-created because it is moved above.
        GraphFloat32 graph2;
        if (graph_is_destroyed) {
          RETURN_IF_ERROR(InitializeGraph(context, delegate_params, &graph2,
                                          &input_refs, &output_refs));
        }
        RETURN_IF_ERROR(InitializeOpenGlApi(
            graph_is_destroyed ? &graph2 : &graph, &builder));
      }
    }

    // At this point, TFLite hasn't allocated tensors yet, therefore, collect
    // indices and set all input and output tensors from TFLite later.
    input_indices_.reserve(input_refs.size());
    for (uint32_t tensor_index : input_refs) {
      const int64_t object_index = input_indices_.size();
      input_indices_.push_back(tensor_index);
      RETURN_IF_ERROR(
          builder->SetInputObjectDef(object_index, GetObjectDef(tensor_index)));
    }
    output_indices_.reserve(output_refs.size());
    for (uint32_t tensor_index : output_refs) {
      const int64_t object_index = output_indices_.size();
      output_indices_.push_back(tensor_index);
      RETURN_IF_ERROR(builder->SetOutputObjectDef(object_index,
                                                  GetObjectDef(tensor_index)));
    }

    return builder->Build(&runner_);
  }

  // This directs the runtime to allocate memory for input/output temporary
  // tensors that require dequantization/quantization.
  absl::Status GetRequiredTemporaries(TfLiteContext* context, TfLiteNode* node,
                                      TfLiteIntArray** temporaries_array_ptr) {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSdelegateDTcc mht_12(mht_12_v, 408, "", "./tensorflow/lite/delegates/gpu/delegate.cc", "GetRequiredTemporaries");

    if (quant_conversion_map_.empty()) return absl::OkStatus();

    std::vector<int> temporary_tensors;
    for (auto index : input_indices_) {
      if (quant_conversion_map_.find(index) != quant_conversion_map_.end()) {
        temporary_tensors.push_back(index);
      }
    }
    for (auto index : output_indices_) {
      if (quant_conversion_map_.find(index) != quant_conversion_map_.end()) {
        temporary_tensors.push_back(index);
      }
    }
    *temporaries_array_ptr = TfLiteIntArrayCreate(temporary_tensors.size());
    for (int i = 0; i < temporary_tensors.size(); ++i) {
      (*temporaries_array_ptr)->data[i] = temporary_tensors[i];
    }
    return absl::OkStatus();
  }

  absl::Status Invoke(TfLiteContext* context) {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSdelegateDTcc mht_13(mht_13_v, 432, "", "./tensorflow/lite/delegates/gpu/delegate.cc", "Invoke");

    if (thread_id_prepare_ != std::this_thread::get_id()) {
      TFLITE_LOG(tflite::TFLITE_LOG_WARNING,
                 "GpuDelegate invoke thread != prepare thread");
      if (enforce_same_thread_) {
        return absl::FailedPreconditionError(
            "GpuDelegate must run on the same thread where it was "
            "initialized.");
      }
    }

    const bool is_dequant_required = !quant_conversion_map_.empty();
    if (is_dequant_required) {
      RETURN_IF_ERROR(
          DequantizeInputs(context, input_indices_, quant_conversion_map_));
    }
    RETURN_IF_ERROR(SetInputsAndOutputs(context));
    RETURN_IF_ERROR(runner_->Run());
    if (is_dequant_required) {
      RETURN_IF_ERROR(
          QuantizeOutputs(context, output_indices_, quant_conversion_map_));
    }
    return absl::OkStatus();
  }

 private:
  absl::Status SetInputsAndOutputs(TfLiteContext* context) {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSdelegateDTcc mht_14(mht_14_v, 461, "", "./tensorflow/lite/delegates/gpu/delegate.cc", "SetInputsAndOutputs");

    for (int i = 0; i < input_indices_.size(); ++i) {
      RETURN_IF_ERROR(runner_->SetInputObject(
          i, GetTensorObject(input_indices_[i], context)));
    }
    for (int i = 0; i < output_indices_.size(); ++i) {
      RETURN_IF_ERROR(runner_->SetOutputObject(
          i, GetTensorObject(output_indices_[i], context)));
    }
    return absl::OkStatus();
  }

  ObjectDef GetObjectDef(int index) const {
   std::vector<std::string> mht_15_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSdelegateDTcc mht_15(mht_15_v, 476, "", "./tensorflow/lite/delegates/gpu/delegate.cc", "GetObjectDef");

    ObjectDef default_object_def;
    default_object_def.data_type = DataType::FLOAT32;
    default_object_def.data_layout = DataLayout::BHWC;
    default_object_def.object_type = ObjectType::CPU_MEMORY;
    default_object_def.user_provided = true;
    return default_object_def;
  }

  TensorObject GetTensorObject(int index, TfLiteContext* context) const {
   std::vector<std::string> mht_16_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSdelegateDTcc mht_16(mht_16_v, 488, "", "./tensorflow/lite/delegates/gpu/delegate.cc", "GetTensorObject");

    auto& tensor = context->tensors[index];
    return MakeCpuMemory(absl::MakeSpan(tensor.data.raw, tensor.bytes));
  }

 private:
  absl::Status InitializeGraph(TfLiteContext* context,
                               const TfLiteDelegateParams* delegate_params,
                               GraphFloat32* graph,
                               std::vector<uint32_t>* input_refs,
                               std::vector<uint32_t>* output_refs) {
   std::vector<std::string> mht_17_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSdelegateDTcc mht_17(mht_17_v, 501, "", "./tensorflow/lite/delegates/gpu/delegate.cc", "InitializeGraph");

    quant_conversion_map_.clear();
    if (delegate_->IsQuantOpsAllowed()) {
      RETURN_IF_ERROR(BuildFinalModel(context, delegate_params, graph,
                                      &quant_conversion_map_));
    } else {
      RETURN_IF_ERROR(BuildFinalModel(context, delegate_params, graph));
    }

    // TfLiteDelegateParams.input_tensors is an array of all input tensors
    // including static weights.  GraphFloat32.inputs() is an array of runtime
    // tensors that don't have a producer and the order may not be the same as
    // defined by TfLiteDelegateParams.input_tensors.  These two sets are not
    // the same, especially on a multi-partition delegation.  These are matched
    // by filtering TfLiteDelegateParams.input_tensors with
    // !tflite::IsConstantTensor() and then inserting them in the order
    // specified by TfLiteDelegateParams.input_tensors.  This logic is shared
    // with ModelBuilder::PrecreateIOTensors() which is eventually called with
    // BuildFinalModel() above.
    //
    // Similarly, TfLiteDelegateParams.output_tensors is an array of all output
    // tensors, and can contain static tensors with buggy conversion.
    // GraphFloat32.outputs() is an array of runtime tensors that don't have a
    // consumer (this is a bug in the assumption) and the order may not be the
    // same as defined by TfLiteDelegateParams.output_tensors.  Again, these two
    // sets are not the same, especially on a multi-partition delegation.  These
    // are matched by inserting the tensors by the order defined by
    // TfLiteDelegateParams.output_tensors.  Similarly, this logic is shared
    // with ModelBuilder::PrecreateIOTensors() which is eventually called with
    // BuildFinalModel() above.
    //
    // The aforementioned matching in BuildFinalModel() is ported here to match
    // input/output_refs.
    // TODO(b/211393366): Fix this at GraphFloat32.inputs/outputs() level.
    const std::vector<Value*> inputs = graph->inputs();
    input_refs->clear();
    input_refs->reserve(delegate_params->input_tensors->size);
    for (int i = 0, j = 0; i < delegate_params->input_tensors->size; ++i) {
      const TfLiteTensor* tensor =
          context->tensors + delegate_params->input_tensors->data[i];
      if (tflite::IsConstantTensor(tensor)) continue;
      input_refs->push_back(inputs[j]->tensor.ref);
      ++j;
    }
    const std::vector<Value*> outputs = graph->outputs();
    output_refs->clear();
    output_refs->reserve(delegate_params->output_tensors->size);
    for (int i = 0; i < delegate_params->output_tensors->size; ++i) {
      output_refs->push_back(outputs[i]->tensor.ref);
    }

    return absl::OkStatus();
  }

  absl::Status InitializeOpenClApi(GraphFloat32* graph,
                                   std::unique_ptr<InferenceBuilder>* builder,
                                   bool* graph_is_destroyed,
                                   TfLiteContext* context,
                                   const TfLiteDelegateParams* delegate_params,
                                   Serialization* serialization = nullptr) {
   std::vector<std::string> mht_18_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSdelegateDTcc mht_18(mht_18_v, 563, "", "./tensorflow/lite/delegates/gpu/delegate.cc", "InitializeOpenClApi");

    *graph_is_destroyed = false;
    cl::InferenceEnvironmentOptions env_options;
    cl::InferenceEnvironmentProperties properties;

    // OpenCL initialization is parameterized by these InferenceOptions.
    auto delegate_options = delegate_->options();
    cl::InferenceOptions options;
    // If is_precision_loss_allowed == -1, then just use priorities instead
    // of paying attention to is_precision_loss_allowed value.
    if (delegate_options.is_precision_loss_allowed == -1) {
      options.priority1 = ToPriority(delegate_options.inference_priority1);
      options.priority2 = ToPriority(delegate_options.inference_priority2);
      options.priority3 = ToPriority(delegate_options.inference_priority3);
    } else {
      // Users set is_precision_loss_allowed explicitly, thus use it explicitly.
      if (delegate_options.is_precision_loss_allowed == 0) {
        options.priority1 = InferencePriority::MAX_PRECISION;
      } else {
        options.priority1 = InferencePriority::MIN_LATENCY;
      }
    }
    options.usage = ToUsage(delegate_options.inference_preference);

    if (!serialization) {
      // This path is faster when there is no serialization involved.
      RETURN_IF_ERROR(cl::NewInferenceEnvironment(env_options, &cl_environment_,
                                                  &properties));
      *graph_is_destroyed = true;
      RETURN_IF_ERROR(cl_environment_->NewInferenceBuilder(
          options, std::move(*graph), builder));
    } else {
      // If serialization data is found, initialize CL from it & return early.
      if (MaybeInitializeSerializedOpenCL(context, delegate_params, builder,
                                          &options, &env_options, &properties,
                                          serialization)
              .ok()) {
        return absl::OkStatus();
      }

      RETURN_IF_ERROR(cl::NewInferenceEnvironment(env_options, &cl_environment_,
                                                  &properties));
      *graph_is_destroyed = true;
      std::vector<uint8_t> serialized_model;
      RETURN_IF_ERROR(cl_environment_->BuildSerializedModel(
          options, std::move(*graph), &serialized_model));
      RETURN_IF_ERROR(
          cl_environment_->NewInferenceBuilder(serialized_model, builder));

      RETURN_IF_ERROR(SaveSerializedOpenCL(context, delegate_params, &options,
                                           serialization, serialized_model));
    }

    TFLITE_LOG_PROD_ONCE(tflite::TFLITE_LOG_INFO,
                         "Initialized OpenCL-based API.");
    return absl::OkStatus();
  }

  // Returns Ok only if serialized data is successsfully found.
  absl::Status MaybeInitializeSerializedOpenCL(
      TfLiteContext* context, const TfLiteDelegateParams* delegate_params,
      std::unique_ptr<InferenceBuilder>* builder, cl::InferenceOptions* options,
      cl::InferenceEnvironmentOptions* env_options,
      cl::InferenceEnvironmentProperties* properties,
      Serialization* serialization) {
   std::vector<std::string> mht_19_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSdelegateDTcc mht_19(mht_19_v, 630, "", "./tensorflow/lite/delegates/gpu/delegate.cc", "MaybeInitializeSerializedOpenCL");

    if (!serialization) return absl::InvalidArgumentError("No serialization");
    // We use a fingerprint of the options to ensure compatibility.
    std::string options_fingerprint =
        delegates::StrFingerprint(options, sizeof(cl::InferenceOptions));
    auto data_key = serialization->GetEntryForKernel(
        std::string(kSerializedDataPrefix) + options_fingerprint, context,
        delegate_params);

    std::string model_data;
    auto model_data_status = data_key.GetData(context, &model_data);
    if (model_data_status == kTfLiteOk) {
      absl::Span<const uint8_t> model_span = absl::Span<const uint8_t>{
          reinterpret_cast<const uint8_t*>(model_data.data()),
          model_data.size()};
      RETURN_IF_ERROR(cl::NewInferenceEnvironment(
          *env_options, &cl_environment_, properties));
      RETURN_IF_ERROR(
          cl_environment_->NewInferenceBuilder(model_span, builder));
      TFLITE_LOG_PROD_ONCE(
          tflite::TFLITE_LOG_INFO,
          "Initialized OpenCL-based API from serialized data.");
      return absl::OkStatus();
    }

    return absl::NotFoundError("Serialization data not found");
  }

  // Returns Ok only if serialization happens successfully.
  absl::Status SaveSerializedOpenCL(
      TfLiteContext* context, const TfLiteDelegateParams* delegate_params,
      cl::InferenceOptions* options, Serialization* serialization,
      const std::vector<uint8_t>& serialized_model) {
   std::vector<std::string> mht_20_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSdelegateDTcc mht_20(mht_20_v, 665, "", "./tensorflow/lite/delegates/gpu/delegate.cc", "SaveSerializedOpenCL");

    if (!serialization) return absl::InvalidArgumentError("No serialization");
    // We use a fingerprint of the options to ensure compatibility.
    std::string options_fingerprint =
        delegates::StrFingerprint(options, sizeof(cl::InferenceOptions));

    // Save data.
    auto data_key = serialization->GetEntryForKernel(
        std::string(kSerializedDataPrefix) + options_fingerprint, context,
        delegate_params);
    auto save_status = data_key.SetData(
        context, reinterpret_cast<const char*>(serialized_model.data()),
        serialized_model.size());
    if (save_status != kTfLiteOk) {
      return absl::InvalidArgumentError("Failed to save serialized data");
    }
    return absl::OkStatus();
  }

  absl::Status InitializeOpenGlApi(GraphFloat32* graph,
                                   std::unique_ptr<InferenceBuilder>* builder) {
   std::vector<std::string> mht_21_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSdelegateDTcc mht_21(mht_21_v, 688, "", "./tensorflow/lite/delegates/gpu/delegate.cc", "InitializeOpenGlApi");

#ifndef CL_DELEGATE_NO_GL
    gl::InferenceEnvironmentOptions env_options;
    gl::InferenceEnvironmentProperties properties;
    RETURN_IF_ERROR(
        NewInferenceEnvironment(env_options, &gl_environment_, &properties));
    auto delegate_options = delegate_->options();
    gl::InferenceOptions options;
    options.usage = ToUsage(delegate_options.inference_preference);
    options.priority1 = ToPriority(delegate_options.inference_priority1);
    options.priority2 = ToPriority(delegate_options.inference_priority2);
    options.priority3 = ToPriority(delegate_options.inference_priority3);
    RETURN_IF_ERROR(gl_environment_->NewInferenceBuilder(std::move(*graph),
                                                         options, builder));
    enforce_same_thread_ = true;
    TFLITE_LOG_PROD_ONCE(tflite::TFLITE_LOG_INFO,
                         "Initialized OpenGL-based API.");
    return absl::OkStatus();
#else
    return absl::UnavailableError("OpenGL-based API disabled");
#endif
  }

  // The Delegate instance that's shared across all DelegateKernel instances.
  Delegate* const delegate_;  // doesn't own the memory.
  std::unique_ptr<cl::InferenceEnvironment> cl_environment_;
#ifndef CL_DELEGATE_NO_GL
  std::unique_ptr<gl::InferenceEnvironment> gl_environment_;
#endif
  std::unique_ptr<InferenceRunner> runner_;
  std::vector<int64_t> input_indices_;
  std::vector<int64_t> output_indices_;
  // Whenever quantized inference is enabled, this maps the tensor index of each
  // originally quantized (8-bit) tensor to its float version added in
  // model_builder - and vice versa.
  absl::flat_hash_map<int, int> quant_conversion_map_;
  std::thread::id thread_id_prepare_;  // thread id used for Prapare()
  bool enforce_same_thread_ = false;   // flag to enforce same thread for Invoke
};

inline DelegateKernel* GetDelegateKernel(TfLiteNode* node) {
   std::vector<std::string> mht_22_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSdelegateDTcc mht_22(mht_22_v, 731, "", "./tensorflow/lite/delegates/gpu/delegate.cc", "GetDelegateKernel");

  return reinterpret_cast<DelegateKernel*>(node->user_data);
}

inline Delegate* GetDelegate(TfLiteDelegate* delegate) {
   std::vector<std::string> mht_23_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSdelegateDTcc mht_23(mht_23_v, 738, "", "./tensorflow/lite/delegates/gpu/delegate.cc", "GetDelegate");

  return reinterpret_cast<Delegate*>(delegate->data_);
}

TfLiteStatus DelegatePrepare(TfLiteContext* context, TfLiteDelegate* delegate) {
   std::vector<std::string> mht_24_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSdelegateDTcc mht_24(mht_24_v, 745, "", "./tensorflow/lite/delegates/gpu/delegate.cc", "DelegatePrepare");

  const TfLiteRegistration kRegistration = {
      // .init
      [](TfLiteContext* context, const char* buffer, size_t) -> void* {
        const auto* params =
            reinterpret_cast<const TfLiteDelegateParams*>(buffer);
        auto* gpu_delegate = GetDelegate(params->delegate);
        // Everything below should happen in prepare function call, but TFLite
        // for whatever reason forbids that.
        auto gpu_delegate_kernel =
            absl::make_unique<DelegateKernel>(gpu_delegate);
        const auto status = gpu_delegate_kernel->Prepare(context, params);
        if (!status.ok()) {
          TF_LITE_KERNEL_LOG(context, "TfLiteGpuDelegate Init: %s",
                             std::string(status.message()).c_str());
          return nullptr;
        }
        return gpu_delegate_kernel.release();
      },
      // .free
      [](TfLiteContext*, void* buffer) -> void {
        delete reinterpret_cast<DelegateKernel*>(buffer);
      },
      // .prepare
      [](TfLiteContext* context, TfLiteNode* node) -> TfLiteStatus {
        if (!node->user_data) {
          TF_LITE_KERNEL_LOG(
              context,
              "TfLiteGpuDelegate Prepare: delegate is not initialized");
          return kTfLiteError;
        }
        auto* gpu_delegate_kernel = GetDelegateKernel(node);
        const auto status = gpu_delegate_kernel->GetRequiredTemporaries(
            context, node, &node->temporaries);
        if (!status.ok()) {
          TF_LITE_KERNEL_LOG(context, "TfLiteGpuDelegate Prepare: %s",
                             std::string(status.message()).c_str());
          return kTfLiteError;
        }
        // TODO(akulik): tflite tensors are not allocated here either. It would
        // be good to set inputs and outputs only once here instead of setting
        // them every time in .invoke.
        return kTfLiteOk;
      },
      // .invoke
      [](TfLiteContext* context, TfLiteNode* node) -> TfLiteStatus {
        const auto status = GetDelegateKernel(node)->Invoke(context);
        if (!status.ok()) {
          TF_LITE_KERNEL_LOG(context, "TfLiteGpuDelegate Invoke: %s",
                             std::string(status.message()).c_str());
          return kTfLiteError;
        }
        return kTfLiteOk;
      },
      nullptr,                // .profiling_string
      0,                      // .builtin_code
      "TfLiteGpuDelegateV2",  // .custom_name
      1,                      // .version
  };

  auto* gpu_delegate = GetDelegate(delegate);
  absl::flat_hash_set<TfLiteBuiltinOperator> excluded_ops;
  if (!cl::OpenCLSupported()) {
    excluded_ops.insert(kTfLiteBuiltinSplit);
    excluded_ops.insert(kTfLiteBuiltinSplitV);
  }
  TfLiteIntArray* ops_to_replace =
      GetOpsToReplace(context, gpu_delegate->IsQuantOpsAllowed(),
                      gpu_delegate->MaxDelegatedPartitions(), &excluded_ops);
  const auto status = context->ReplaceNodeSubsetsWithDelegateKernels(
      context, kRegistration, ops_to_replace, delegate);
  TFLITE_LOG_PROD(TFLITE_LOG_INFO, "Created %d GPU delegate kernels.",
                  gpu_delegate->num_delegate_kernels());
  TfLiteIntArrayFree(ops_to_replace);
  return status;
}

}  // namespace
}  // namespace gpu
}  // namespace tflite

TfLiteGpuDelegateOptionsV2 TfLiteGpuDelegateOptionsV2Default() {
   std::vector<std::string> mht_25_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSdelegateDTcc mht_25(mht_25_v, 829, "", "./tensorflow/lite/delegates/gpu/delegate.cc", "TfLiteGpuDelegateOptionsV2Default");

  TfLiteGpuDelegateOptionsV2 options;
  // set it to -1 to detect whether it was later adjusted.
  options.is_precision_loss_allowed = -1;
  options.inference_preference =
      TFLITE_GPU_INFERENCE_PREFERENCE_FAST_SINGLE_ANSWER;
  options.inference_priority1 = TFLITE_GPU_INFERENCE_PRIORITY_MAX_PRECISION;
  options.inference_priority2 = TFLITE_GPU_INFERENCE_PRIORITY_AUTO;
  options.inference_priority3 = TFLITE_GPU_INFERENCE_PRIORITY_AUTO;
  options.experimental_flags = TFLITE_GPU_EXPERIMENTAL_FLAGS_ENABLE_QUANT;
  options.max_delegated_partitions = 1;
  options.model_token = nullptr;
  options.serialization_dir = nullptr;
  return options;
}

TfLiteDelegate* TfLiteGpuDelegateV2Create(
    const TfLiteGpuDelegateOptionsV2* options) {
   std::vector<std::string> mht_26_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSdelegateDTcc mht_26(mht_26_v, 849, "", "./tensorflow/lite/delegates/gpu/delegate.cc", "TfLiteGpuDelegateV2Create");

  auto* gpu_delegate = new tflite::gpu::Delegate(options);
  TFLITE_LOG_PROD_ONCE(tflite::TFLITE_LOG_INFO,
                       "Created TensorFlow Lite delegate for GPU.");
  return gpu_delegate ? gpu_delegate->tflite_delegate() : nullptr;
}

void TfLiteGpuDelegateV2Delete(TfLiteDelegate* delegate) {
   std::vector<std::string> mht_27_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSdelegateDTcc mht_27(mht_27_v, 859, "", "./tensorflow/lite/delegates/gpu/delegate.cc", "TfLiteGpuDelegateV2Delete");

  delete tflite::gpu::GetDelegate(delegate);
}
