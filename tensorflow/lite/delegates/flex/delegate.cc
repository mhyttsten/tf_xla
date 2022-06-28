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
class MHTracer_DTPStensorflowPSlitePSdelegatesPSflexPSdelegateDTcc {
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
   MHTracer_DTPStensorflowPSlitePSdelegatesPSflexPSdelegateDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePSdelegatesPSflexPSdelegateDTcc() {
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
#include "tensorflow/lite/delegates/flex/delegate.h"

#include <memory>
#include <utility>
#include <vector>

#include "absl/strings/str_cat.h"
#include "tensorflow/core/framework/variant.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/context_util.h"
#include "tensorflow/lite/core/macros.h"
#include "tensorflow/lite/delegates/flex/buffer_map.h"
#include "tensorflow/lite/delegates/flex/kernel.h"
#include "tensorflow/lite/delegates/flex/util.h"
#include "tensorflow/lite/minimal_logging.h"
#include "tensorflow/lite/string_util.h"
#include "tensorflow/lite/util.h"

namespace tflite {

TfLiteDelegateUniquePtr FlexDelegate::Create(
    std::unique_ptr<FlexDelegate> base_delegate) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSflexPSdelegateDTcc mht_0(mht_0_v, 206, "", "./tensorflow/lite/delegates/flex/delegate.cc", "FlexDelegate::Create");

  TFLITE_LOG_PROD_ONCE(TFLITE_LOG_INFO,
                       "Created TensorFlow Lite delegate for select TF ops.");
  if (base_delegate == nullptr) {
    base_delegate.reset(new FlexDelegate());
  }
  auto flex_delegate = TfLiteDelegateFactory::Create(std::move(base_delegate));
  flex_delegate->CopyFromBufferHandle =
      [](TfLiteContext* context, TfLiteDelegate* delegate,
         TfLiteBufferHandle buffer_handle,
         TfLiteTensor* tensor) -> TfLiteStatus {
    return reinterpret_cast<FlexDelegate*>(delegate->data_)
        ->CopyFromBufferHandle(context, buffer_handle, tensor);
  };
  flex_delegate->flags |= kTfLiteDelegateFlagsAllowDynamicTensors;
  reinterpret_cast<FlexDelegate*>(flex_delegate->data_)->base_delegate_ =
      flex_delegate.get();
  return flex_delegate;
}

TfLiteStatus FlexDelegate::Initialize(TfLiteContext* context) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSflexPSdelegateDTcc mht_1(mht_1_v, 229, "", "./tensorflow/lite/delegates/flex/delegate.cc", "FlexDelegate::Initialize");

  // If the TensorFlow Lite thread count is explicitly configured, use it,
  // otherwise rely on the default TensorFlow threading behavior.
  tensorflow::SessionOptions session_options;
  // We don't run multiple ops at the same time, so prefer using
  // 1 thread for inter-op parallelism.
  // Negative value means all are done on the caller thread.
  session_options.config.set_inter_op_parallelism_threads(-1);
  if (context->recommended_num_threads > 0) {
    session_options.config.set_intra_op_parallelism_threads(
        context->recommended_num_threads);
  }

  auto status = delegate_data_.Prepare(
      session_options, reinterpret_cast<Subgraph*>(context->impl_),
      base_delegate_);
  if (!status.ok()) {
    context->ReportError(context, "Failed to initialize TensorFlow context: %s",
                         status.error_message().c_str());
    return kTfLiteError;
  }

  // Initializes the cancellation manager.
  if (!cancellation_manager_) {
    cancellation_manager_ =
        absl::make_unique<tensorflow::CancellationManager>();
    delegate_data_.SetCancellationManager(cancellation_manager_.get());
  }

  return kTfLiteOk;
}

const char* FlexDelegate::Name() const {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSflexPSdelegateDTcc mht_2(mht_2_v, 264, "", "./tensorflow/lite/delegates/flex/delegate.cc", "FlexDelegate::Name");

  static constexpr char kName[] = "TfLiteFlexDelegate";
  return kName;
}

bool FlexDelegate::IsNodeSupportedByDelegate(
    const TfLiteRegistration* registration, const TfLiteNode* node,
    TfLiteContext* context) const {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSflexPSdelegateDTcc mht_3(mht_3_v, 274, "", "./tensorflow/lite/delegates/flex/delegate.cc", "FlexDelegate::IsNodeSupportedByDelegate");

  return IsFlexOp(registration->custom_name);
}

std::unique_ptr<SimpleDelegateKernelInterface>
FlexDelegate::CreateDelegateKernelInterface() {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSflexPSdelegateDTcc mht_4(mht_4_v, 282, "", "./tensorflow/lite/delegates/flex/delegate.cc", "FlexDelegate::CreateDelegateKernelInterface");

  return std::unique_ptr<SimpleDelegateKernelInterface>(
      new tflite::flex::DelegateKernel());
}

TfLiteStatus FlexDelegate::CopyFromBufferHandle(
    TfLiteContext* context, TfLiteBufferHandle buffer_handle,
    TfLiteTensor* output) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSflexPSdelegateDTcc mht_5(mht_5_v, 292, "", "./tensorflow/lite/delegates/flex/delegate.cc", "FlexDelegate::CopyFromBufferHandle");

  flex::BufferMap* buffer_map = delegate_data_.GetBufferMap(context);

  if (!buffer_map->HasTensor(buffer_handle)) {
    context->ReportError(context, "Invalid tensor index %d.", buffer_handle);
    return kTfLiteError;
  }

  tensorflow::Tensor t = buffer_map->GetTensor(buffer_handle);

  if (output->type == kTfLiteString) {
    if (t.dtype() != tensorflow::DT_STRING) {
      context->ReportError(context,
                           "Inconsistent type for TF string tensor index %d.",
                           buffer_handle);
      return kTfLiteError;
    }
    DynamicBuffer dynamic_buffer;

    auto tf_data = t.flat<tensorflow::tstring>();
    for (int i = 0; i < t.NumElements(); ++i) {
      dynamic_buffer.AddString(tf_data(i).data(), tf_data(i).size());
    }

    dynamic_buffer.WriteToTensor(output, /*new_shape=*/nullptr);
    return kTfLiteOk;
  }

  // TODO(b/179094265): This is an experimental implementation, subject to
  // change. This can be re-implemented with life cycle management mechanism
  // like reference counting.
  // When copying resource and variant tensors from Flex delegate to TensorFlow
  // Lite tensors, the CopyFromBufferHandle method of the Flex delegate is
  // invoked and it will store the `data` field of the given TensorFlow Lite
  // tensor and pass the TensorFlow Lite tensor pointer. Copying the `data`
  // field will act as passing pointers between TensorFlow Lite tensors.
  //
  // The life cycle of the pointer will be managed by the reference counting in
  // the TensorFlow world and the pointer will be freed when all the buffer
  // maps, who own it, are gone.
  if (flex::IsResourceOrVariant(output)) {
    const size_t required_bytes = sizeof(tensorflow::Tensor**);
    const tensorflow::Tensor** tf_tensor_ptr =
        reinterpret_cast<const tensorflow::Tensor**>(malloc(required_bytes));
    *tf_tensor_ptr = buffer_map->GetTensorPtr(buffer_handle);

    TfLiteTensorDataFree(output);
    output->data.raw = reinterpret_cast<char*>(tf_tensor_ptr);
    output->bytes = required_bytes;
    output->data_is_stale = true;
    return kTfLiteOk;
  }

  tensorflow::StringPiece t_data = t.tensor_data();

  if (output->bytes != t_data.size()) {
    context->ReportError(context,
                         absl::StrCat("The given ", output->bytes,
                                      " bytes are not enough to store "
                                      "TensorFlow's aligned buffer of size ",
                                      t_data.size(), " bytes.")
                             .c_str());
    return kTfLiteError;
  }

  memcpy(output->data.raw, t_data.data(), t_data.size());
  return kTfLiteOk;
}

void FlexDelegate::Cancel() {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSflexPSdelegateDTcc mht_6(mht_6_v, 364, "", "./tensorflow/lite/delegates/flex/delegate.cc", "FlexDelegate::Cancel");
 cancellation_manager_->StartCancel(); }

bool FlexDelegate::HasCancelled(void* data) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSflexPSdelegateDTcc mht_7(mht_7_v, 369, "", "./tensorflow/lite/delegates/flex/delegate.cc", "FlexDelegate::HasCancelled");

  if (data == nullptr) {
    return false;
  }

  auto* flex_delegate = static_cast<FlexDelegate*>(data);
  return flex_delegate->cancellation_manager_->IsCancelled();
}

}  // namespace tflite
