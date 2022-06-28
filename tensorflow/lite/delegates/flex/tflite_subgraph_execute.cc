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
class MHTracer_DTPStensorflowPSlitePSdelegatesPSflexPStflite_subgraph_executeDTcc {
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
   MHTracer_DTPStensorflowPSlitePSdelegatesPSflexPStflite_subgraph_executeDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePSdelegatesPSflexPStflite_subgraph_executeDTcc() {
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
#include <string>

#include "absl/strings/numbers.h"
#include "absl/strings/str_split.h"
#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/resource_handle.h"
#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/tstring.h"
#include "tensorflow/lite/c/c_api_types.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/core/subgraph.h"
#include "tensorflow/lite/delegates/flex/buffer_map_util.h"
#include "tensorflow/lite/delegates/flex/subgraph_resource.h"
#include "tensorflow/lite/delegates/flex/util.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/string_util.h"

namespace tensorflow {

namespace {
constexpr int kTfLiteSubgraphResource = 0;
}

REGISTER_OP("TfLiteSubgraphExecute")
    .Input("subgraph_key: string")
    .Input("args: Tin")
    .Output("output: Tout")
    .Attr("Tin: list(type) >= 0")
    .Attr("Tout: list(type) >= 0")
    .SetShapeFn(shape_inference::UnknownShape);

// The `TfLiteSubgraphExecute` executes a tflite subgraph with the designated
// inputs. This op will first look up the tflite subgraph from TF resource
// manager based on the resource name stored on the first input, and then it
// will call that specific subgraph with the remaining arguments. The first
// input of this op is always a scalar string, which denotes the name of the
// subgraph resource. The remaining inputs will be fed to the subgraph as
// inputs, so the caller needs to ensure that the remaining inputs match with
// the subgraph's expected inputs. This is currently WIP/experimental and
// subject to change.
class TfLiteSubgraphExecute : public OpKernel {
 public:
  explicit TfLiteSubgraphExecute(OpKernelConstruction* ctx)
      : OpKernel(ctx), tfl_tensors_need_allocation_(true) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSflexPStflite_subgraph_executeDTcc mht_0(mht_0_v, 235, "", "./tensorflow/lite/delegates/flex/tflite_subgraph_execute.cc", "TfLiteSubgraphExecute");
}

  void Compute(OpKernelContext* ctx) override {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSflexPStflite_subgraph_executeDTcc mht_1(mht_1_v, 240, "", "./tensorflow/lite/delegates/flex/tflite_subgraph_execute.cc", "Compute");

    // Fetch the TF Lite subgraph to execute.
    tflite::flex::TFLiteSubgraphResource* resource = nullptr;
    OP_REQUIRES_OK(
        ctx,
        ctx->resource_manager()->Lookup<tflite::flex::TFLiteSubgraphResource>(
            "flex", ctx->input(kTfLiteSubgraphResource).flat<tstring>()(0),
            &resource));
    tensorflow::core::ScopedUnref unref_resource(resource);

    // Try to acquire a mutex lock from this resource. This is because tflite
    // subgraph is not thread-safe and we need to guarantee exclusive access to
    // it.
    mutex_lock lock(resource->GetExclusiveLock());
    tflite::Subgraph& subgraph_selected = resource->GetSubgraphResource();

    OP_REQUIRES(ctx, ctx->num_inputs() == subgraph_selected.inputs().size() + 1,
                errors::InvalidArgument("TF Lite subgraph expects ",
                                        subgraph_selected.inputs().size(),
                                        " inputs, but received ",
                                        ctx->num_inputs() - 1, "."));

    // Resize input tensors if necessary.
    ResizeInputTensor(ctx, subgraph_selected);

    if (tfl_tensors_need_allocation_) {
      OP_REQUIRES(ctx, subgraph_selected.AllocateTensors() == kTfLiteOk,
                  errors::Internal("Failed to call allocate tensors"));
      tfl_tensors_need_allocation_ = false;
    }

    // Copy input tensors to subgraph.
    SetSubgraphInput(ctx, subgraph_selected, resource->GetFlexDelegate());

    OP_REQUIRES(ctx, subgraph_selected.Invoke() == kTfLiteOk,
                errors::Internal("Failed to invoke tflite subgraph"));

    // Copy tflite results.
    CopyTFLiteSubgraphResult(ctx, subgraph_selected);
  }

 private:
  void ResizeInputTensor(OpKernelContext* ctx,
                         tflite::Subgraph& subgraph_selected) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSflexPStflite_subgraph_executeDTcc mht_2(mht_2_v, 286, "", "./tensorflow/lite/delegates/flex/tflite_subgraph_execute.cc", "ResizeInputTensor");

    for (int i = 0; i < subgraph_selected.inputs().size(); ++i) {
      // Shift index by 1 since the first input is always the resource name.
      const Tensor& tf_tensor = ctx->input(i + 1);
      TfLiteTensor* subgraph_input =
          subgraph_selected.tensor(subgraph_selected.inputs()[i]);

      bool need_resize = false;
      for (int dim = 0; dim < tf_tensor.shape().dims(); dim++) {
        if (tf_tensor.shape().dim_size(dim) !=
            subgraph_input->dims->data[dim]) {
          need_resize = true;
          break;
        }
      }
      if (need_resize) {
        std::vector<int> new_shape;
        for (auto dim : tf_tensor.shape().dim_sizes()) {
          new_shape.push_back(dim);
        }
        tfl_tensors_need_allocation_ = true;
        OP_REQUIRES(ctx,
                    subgraph_selected.ResizeInputTensor(
                        subgraph_selected.inputs()[i], new_shape) == kTfLiteOk,
                    errors::Internal("Failed to resize tflite tensor"));
      }
    }
  }

  void SetSubgraphInput(OpKernelContext* ctx,
                        tflite::Subgraph& subgraph_selected,
                        TfLiteDelegate* flex_delegate) const {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSflexPStflite_subgraph_executeDTcc mht_3(mht_3_v, 320, "", "./tensorflow/lite/delegates/flex/tflite_subgraph_execute.cc", "SetSubgraphInput");

    auto InitializeVariantOrResource = [flex_delegate](
                                           const Tensor& tf_tensor,
                                           TfLiteTensor* subgraph_input) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSflexPStflite_subgraph_executeDTcc mht_4(mht_4_v, 326, "", "./tensorflow/lite/delegates/flex/tflite_subgraph_execute.cc", "lambda");

      // The code here initializes the TfLiteTensor which points the data field
      // to the original TF resource or variant tensor. This requires the TF
      // tensor's lifetime must extend beyond the execution of callee subgraph.
      // TODO(b/179094265): This is an experimental implementation, subject to
      // change. This can be re-implemented with life cycle management
      // mechanism like reference counting.
      const size_t required_bytes = sizeof(tensorflow::Tensor**);
      const tensorflow::Tensor** tf_tensor_ptr =
          reinterpret_cast<const tensorflow::Tensor**>(malloc(required_bytes));
      *tf_tensor_ptr = &tf_tensor;

      TfLiteTensorDataFree(subgraph_input);
      subgraph_input->data.raw = reinterpret_cast<char*>(tf_tensor_ptr);
      subgraph_input->bytes = required_bytes;
      subgraph_input->data_is_stale = true;
      subgraph_input->delegate = flex_delegate;
    };

    for (int i = 0; i < subgraph_selected.inputs().size(); ++i) {
      const Tensor& tf_tensor = ctx->input(i + 1);
      TfLiteTensor* subgraph_input =
          subgraph_selected.tensor(subgraph_selected.inputs()[i]);

      if (subgraph_input->type == kTfLiteString) {
        OP_REQUIRES(ctx, tf_tensor.dtype() == tensorflow::DT_STRING,
                    errors::InvalidArgument("Tensor doesn't have string type"));
        tflite::DynamicBuffer dynamic_buffer;
        auto tf_data = tf_tensor.flat<tensorflow::tstring>();
        for (int i = 0; i < tf_tensor.NumElements(); ++i) {
          dynamic_buffer.AddString(tf_data(i).data(), tf_data(i).size());
        }

        dynamic_buffer.WriteToTensor(subgraph_input, /*new_shape=*/nullptr);
      } else if (subgraph_input->type == kTfLiteResource) {
        // Here we will try to parse the input tensor handle to see if it
        // contains a valid TF lite resource ID. If not, then we know that the
        // input is a TF resource tensor.
        tensorflow::ResourceHandle handle =
            tf_tensor.flat<tensorflow::ResourceHandle>()(0);
        if (!tflite::flex::GetTfLiteResourceTensorFromResourceHandle(
                handle, subgraph_input)) {
          InitializeVariantOrResource(tf_tensor, subgraph_input);
        }
      } else if (subgraph_input->type == kTfLiteVariant) {
        InitializeVariantOrResource(tf_tensor, subgraph_input);
      } else {
        tensorflow::StringPiece tensor_data = tf_tensor.tensor_data();
        OP_REQUIRES(ctx, subgraph_input->bytes == tensor_data.size(),
                    errors::Internal("tensor size doesn't match"));
        // TODO(b/181352924): This could incur some overhead in memory copy.
        // Optimize this away in the future.
        memcpy(subgraph_input->data.raw, tensor_data.data(),
               tensor_data.size());
      }
    }
  }

  void CopyTFLiteSubgraphResult(OpKernelContext* ctx,
                                tflite::Subgraph& subgraph_selected) const {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSflexPStflite_subgraph_executeDTcc mht_5(mht_5_v, 388, "", "./tensorflow/lite/delegates/flex/tflite_subgraph_execute.cc", "CopyTFLiteSubgraphResult");

    for (int i = 0; i < subgraph_selected.outputs().size(); ++i) {
      OP_REQUIRES(ctx,
                  subgraph_selected.EnsureTensorDataIsReadable(
                      subgraph_selected.outputs()[i]) == kTfLiteOk,
                  errors::Internal("TF lite subgraph output is not readable"));
      // Create an output tensor.
      TfLiteTensor* subgraph_output =
          subgraph_selected.tensor(subgraph_selected.outputs()[i]);

      Tensor tensor;
      OP_REQUIRES_OK(
          ctx, tflite::flex::SetTfTensorFromTfLite(subgraph_output, &tensor));
      ctx->set_output(i, std::move(tensor));
    }
  }

  // Tells if the target subgraph needs to invoko AllocateTensors().
  bool tfl_tensors_need_allocation_;
};

REGISTER_KERNEL_BUILDER(Name("TfLiteSubgraphExecute").Device(DEVICE_CPU),
                        TfLiteSubgraphExecute);

}  // namespace tensorflow
