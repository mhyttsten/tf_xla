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
class MHTracer_DTPStensorflowPScompilerPStf2tensorrtPSutilsPStrt_engine_utilsDTcc {
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
   MHTracer_DTPStensorflowPScompilerPStf2tensorrtPSutilsPStrt_engine_utilsDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPStf2tensorrtPSutilsPStrt_engine_utilsDTcc() {
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

/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/tf2tensorrt/utils/trt_engine_utils.h"

#include <string>
#include <vector>

#include "tensorflow/compiler/tf2tensorrt/common/utils.h"
#include "tensorflow/compiler/tf2tensorrt/convert/utils.h"
#include "tensorflow/compiler/tf2tensorrt/utils/trt_allocator.h"
#include "tensorflow/compiler/tf2tensorrt/utils/trt_execution_context.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/errors.h"

#if GOOGLE_CUDA && GOOGLE_TENSORRT
#include "third_party/tensorrt/NvInfer.h"

namespace tensorflow {
namespace tensorrt {

using absl::StrCat;

ExecutionContext ExecutionContext::Create(nvinfer1::ICudaEngine* cuda_engine) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPStf2tensorrtPSutilsPStrt_engine_utilsDTcc mht_0(mht_0_v, 207, "", "./tensorflow/compiler/tf2tensorrt/utils/trt_engine_utils.cc", "ExecutionContext::Create");

  bool has_int32_output = false;
  for (int i = 0; i < cuda_engine->getNbBindings(); i++) {
    if (!cuda_engine->bindingIsInput(i) &&
        cuda_engine->getBindingDataType(i) == nvinfer1::DataType::kINT32) {
      has_int32_output = true;
      break;
    }
  }
  if (!IS_TRT_VERSION_GE(8, 0, 0, 0) && has_int32_output) {
    // TODO(nvbugs/3390469): Remove this workaround when the bug is fixed.
    nvinfer1::IExecutionContext* execution_context =
        cuda_engine->createExecutionContext();
    return ExecutionContext(execution_context, true);
  }

  nvinfer1::IExecutionContext* execution_context =
      cuda_engine->createExecutionContextWithoutDeviceMemory();
  return ExecutionContext(execution_context, false);
}

Status GetTrtBindingShape(const nvinfer1::ICudaEngine* cuda_engine,
                          const nvinfer1::IExecutionContext* execution_context,
                          int binding_index, bool use_implicit_batch,
                          int batch_size, TensorShape& shape) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPStf2tensorrtPSutilsPStrt_engine_utilsDTcc mht_1(mht_1_v, 234, "", "./tensorflow/compiler/tf2tensorrt/utils/trt_engine_utils.cc", "GetTrtBindingShape");

  nvinfer1::Dims dims =
      use_implicit_batch
          ? cuda_engine->getBindingDimensions(binding_index)
          : execution_context->getBindingDimensions(binding_index);
  if (!use_implicit_batch) {
    if (dims.nbDims == -1) {
      return errors::Internal(
          "Binding index out of range. This can happen if profile is not set, "
          "or the network is invalid for the current profile.");
    }
  }
  TF_RETURN_IF_ERROR(DimsAdapter(dims).TensorShape(
      &shape,
      use_implicit_batch ? absl::optional<int>(batch_size) : absl::nullopt));
  return Status::OK();
}

Status SetupBindings(nvinfer1::ICudaEngine* cuda_engine, const Tensor& tensor,
                     std::vector<void*>& buffers, int binding_index) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPStf2tensorrtPSutilsPStrt_engine_utilsDTcc mht_2(mht_2_v, 256, "", "./tensorflow/compiler/tf2tensorrt/utils/trt_engine_utils.cc", "SetupBindings");

  const auto dtype = cuda_engine->getBindingDataType(binding_index);
  VLOG(2) << "<<<<<<<<< SetupBindings with dtype = " << (int)dtype;
  switch (dtype) {
    case nvinfer1::DataType::kFLOAT:
      buffers[binding_index] = const_cast<float*>(tensor.flat<float>().data());
      break;
    case nvinfer1::DataType::kHALF:
      buffers[binding_index] =
          const_cast<Eigen::half*>(tensor.flat<Eigen::half>().data());
      break;
    case nvinfer1::DataType::kINT8:
      return errors::Internal("INT8 inputs are not supported yet!");
    case nvinfer1::DataType::kINT32:
      buffers[binding_index] = const_cast<int32*>(tensor.flat<int32>().data());
      break;
#if IS_TRT_VERSION_GE(8, 2, 0, 0)
    case nvinfer1::DataType::kBOOL:
      buffers[binding_index] = const_cast<bool*>(tensor.flat<bool>().data());
      break;
#endif
    default:
      return errors::Internal("Unknown TRT data type: ",
                              static_cast<int>(dtype));
  }
  return Status::OK();
}

// Sets up bindings.
Status SetTrtEngineInputs(nvinfer1::ICudaEngine* cuda_engine,
                          nvinfer1::IExecutionContext* execution_context,
                          const int trt_profile_idx,
                          std::vector<void*>& buffers, bool use_implicit_batch,
                          int num_batch,
                          const TrtShapeOptimizationProfile& profiles,
                          OpKernelContext* ctx, const DataVec* input_vec) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPStf2tensorrtPSutilsPStrt_engine_utilsDTcc mht_3(mht_3_v, 294, "", "./tensorflow/compiler/tf2tensorrt/utils/trt_engine_utils.cc", "SetTrtEngineInputs");

  int n_inputs = ctx ? ctx->num_inputs() : (input_vec ? input_vec->size() : 0);
  // Setup engine inputs.
  for (int i = 0; i < n_inputs; i++) {
    const string input_name =
        ctx ? StrCat(IONamePrefixes::kInputPHName, i) : input_vec->at(i).name;
    int binding_index;
    Status status = GetTrtBindingIndex(input_name.c_str(), trt_profile_idx,
                                       cuda_engine, &binding_index);
    if (IS_TRT_VERSION_GE(8, 0, 0, 0)) {
      TF_RETURN_IF_ERROR(status);
    } else if (!status.ok()) {
      // Before TRT 8, an input tensor can be pruned if it is not used by the
      // network (e.g. only its shape is used, but the shape is already defined
      // by the optimization profile by setting min=max). nvbugs/3153064
      VLOG(2) << "Skipping pruned input " << input_name;
      continue;
    }
    const Tensor& input_tensor = ctx ? ctx->input(i) : input_vec->at(i).tensor;
    const TensorShape& input_shape = input_tensor.shape();

    if (use_implicit_batch && ctx) {
      // Ensure all inputs have the same batch size
      if (num_batch != input_shape.dim_size(0)) {
        const string msg =
            StrCat("Input data has inconsistent batch size: ", num_batch,
                   " vs ", input_shape.dim_size(0));
        return errors::NotFound(msg);
      }
    }
    // Set known input dimensions. This is necessary because TRT network
    // could be made with dynamic dimensions.
    if (!use_implicit_batch) {
      TF_RETURN_IF_ERROR(profiles.SetInputShapeBinding(
          i, binding_index, cuda_engine, execution_context));

      if (cuda_engine->isExecutionBinding(binding_index)) {
        nvinfer1::Dims trt_dims;
        auto adap = DimsAdapter::Create(input_shape);
        TRT_ENSURE_OK(adap);
        VLOG(2) << "Setting binding dimensions for idx " << binding_index;
        bool ret = execution_context->setBindingDimensions(binding_index,
                                                           adap->AsTrtDims());
        if (!ret) {
          VLOG(2) << "Error setting engine input " << binding_index << " "
                  << DebugString(trt_dims);
          return errors::Internal(
              "Binding dimension does not fit selected profile.");
        }
      }
    }
    // Setup input bindings.
    TF_RETURN_IF_ERROR(
        SetupBindings(cuda_engine, input_tensor, buffers, binding_index));
  }

  // Ensure all network dynamic dimensions (if any) are set in execution
  // context.
  if (!execution_context->allInputDimensionsSpecified()) {
    return errors::Internal(
        "Failed to set dimensions for all dynamic input tensors");
  }
  if (!execution_context->allInputShapesSpecified()) {
    return errors::Internal(
        "Failed to set dimensions for all shape input tensors.");
  }
  return Status::OK();
}

Status SetTrtEngineOutputs(nvinfer1::ICudaEngine* cuda_engine,
                           nvinfer1::IExecutionContext* execution_context,
                           int trt_profile_idx, std::vector<void*>& buffers,
                           bool use_implicit_batch, int batch_size,
                           OpKernelContext* ctx, DataVec* outputs) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScompilerPStf2tensorrtPSutilsPStrt_engine_utilsDTcc mht_4(mht_4_v, 370, "", "./tensorflow/compiler/tf2tensorrt/utils/trt_engine_utils.cc", "SetTrtEngineOutputs");

  // Either one of ctx or outpus should be specified
  int n_outputs = ctx ? ctx->num_outputs() : (outputs ? outputs->size() : 0);
  for (int i = 0; i < n_outputs; i++) {
    const string output_name =
        ctx ? StrCat(IONamePrefixes::kOutputPHName, i) : outputs->at(i).name;
    int binding_index;
    TF_RETURN_IF_ERROR(GetTrtBindingIndex(output_name.c_str(), trt_profile_idx,
                                          cuda_engine, &binding_index));

    // Get TRT output shapes for allocating output memory.
    TensorShape output_shape;
    TF_RETURN_IF_ERROR(GetTrtBindingShape(cuda_engine, execution_context,
                                          binding_index, use_implicit_batch,
                                          batch_size, output_shape));

    // Allocate output tensor of TRTEngineOp.
    Tensor* output_tensor = nullptr;
    if (ctx) {
      TF_RETURN_IF_ERROR(ctx->allocate_output(i, output_shape, &output_tensor));
    } else {
      // This path is used for unit tests. The tensor is already allocated.
      // Its shape is not necessarily set correctly, we fix that.
      VLOG(2) << "Applying shape " << output_shape.DebugString()
              << " on output.";
      output_tensor = &(outputs->at(i).tensor);
      bool status = output_tensor->CopyFrom(*output_tensor, output_shape);
      if (!status) {
        return errors::Internal(
            "Buffer size (", output_tensor->NumElements(),
            ") do not match while reshaping output tensors to shape ",
            output_shape.DebugString());
      }
    }

    // Set up output bindings.
    TF_RETURN_IF_ERROR(
        SetupBindings(cuda_engine, *output_tensor, buffers, binding_index));
  }
  return Status::OK();
}

Status TrtEnqueue(nvinfer1::IExecutionContext* execution_context,
                  std::vector<void*>& buffers, cudaStream_t stream,
                  bool use_implicit_batch, int batch_size) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScompilerPStf2tensorrtPSutilsPStrt_engine_utilsDTcc mht_5(mht_5_v, 417, "", "./tensorflow/compiler/tf2tensorrt/utils/trt_engine_utils.cc", "TrtEnqueue");

  bool ret = false;
  if (use_implicit_batch) {
    ret = execution_context->enqueue(batch_size, &buffers[0], stream, nullptr);
    VLOG(1) << "Called IExecutionContext::enqueue";
  } else {
    ret = execution_context->enqueueV2(&buffers[0], stream, nullptr);
    VLOG(1) << "Called IExecutionContext::enqueueV2";
  }
  if (!ret) {
    return errors::Internal("Failed to enqueue batch for TRT engine");
  }
  // Synchronization will be done by TF.
  return Status::OK();
}

}  // namespace tensorrt
}  // namespace tensorflow

#endif  // GOOGLE_CUDA && GOOGLE_TENSORRT
