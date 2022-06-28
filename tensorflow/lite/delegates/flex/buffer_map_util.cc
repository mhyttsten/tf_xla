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
class MHTracer_DTPStensorflowPSlitePSdelegatesPSflexPSbuffer_map_utilDTcc {
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
   MHTracer_DTPStensorflowPSlitePSdelegatesPSflexPSbuffer_map_utilDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePSdelegatesPSflexPSbuffer_map_utilDTcc() {
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
#include "tensorflow/lite/delegates/flex/buffer_map_util.h"

#include "tensorflow/core/framework/log_memory.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/typed_allocator.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/lite/delegates/flex/util.h"
#include "tensorflow/lite/experimental/resource/resource_variable.h"
#include "tensorflow/lite/string_util.h"

namespace tflite {
namespace flex {

void BaseTfLiteTensorBuffer::FillAllocationDescription(
    tensorflow::AllocationDescription* proto) const {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSflexPSbuffer_map_utilDTcc mht_0(mht_0_v, 199, "", "./tensorflow/lite/delegates/flex/buffer_map_util.cc", "BaseTfLiteTensorBuffer::FillAllocationDescription");

  int64_t rb = size();
  proto->set_requested_bytes(rb);
  proto->set_allocator_name(tensorflow::cpu_allocator()->Name());
}

void BaseTfLiteTensorBuffer::LogAllocation() {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSflexPSbuffer_map_utilDTcc mht_1(mht_1_v, 208, "", "./tensorflow/lite/delegates/flex/buffer_map_util.cc", "BaseTfLiteTensorBuffer::LogAllocation");

  if (tensorflow::LogMemory::IsEnabled() && data() != nullptr) {
    tensorflow::LogMemory::RecordRawAllocation(
        "TfLiteTensorBuffer_New",
        tensorflow::LogMemory::EXTERNAL_TENSOR_ALLOCATION_STEP_ID, size(),
        data(), tensorflow::cpu_allocator());
  }
}
void BaseTfLiteTensorBuffer::LogDeallocation() {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSflexPSbuffer_map_utilDTcc mht_2(mht_2_v, 219, "", "./tensorflow/lite/delegates/flex/buffer_map_util.cc", "BaseTfLiteTensorBuffer::LogDeallocation");

  if (tensorflow::LogMemory::IsEnabled() && data() != nullptr) {
    tensorflow::LogMemory::RecordRawDeallocation(
        "TfLiteTensorBuffer_Delete",
        tensorflow::LogMemory::EXTERNAL_TENSOR_ALLOCATION_STEP_ID, data(),
        tensorflow::cpu_allocator(), false);
  }
}

TfLiteTensorBuffer::TfLiteTensorBuffer(const TfLiteTensor* tensor)
    : BaseTfLiteTensorBuffer(tensorflow::cpu_allocator()->AllocateRaw(
          EIGEN_MAX_ALIGN_BYTES, tensor->bytes)) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSflexPSbuffer_map_utilDTcc mht_3(mht_3_v, 233, "", "./tensorflow/lite/delegates/flex/buffer_map_util.cc", "TfLiteTensorBuffer::TfLiteTensorBuffer");

  // TODO(ahentz): if we can guarantee that TF Lite allocated tensors with
  // the same alignment as TensorFlow (EIGEN_MAX_ALIGN_BYTES), then we can
  // potentially eliminate the copy below.
  len_ = tensor->bytes;

  LogAllocation();

  if (data()) {
    std::memcpy(data(), tensor->data.raw, tensor->bytes);
  }
}

TfLiteTensorBuffer::~TfLiteTensorBuffer() {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSflexPSbuffer_map_utilDTcc mht_4(mht_4_v, 249, "", "./tensorflow/lite/delegates/flex/buffer_map_util.cc", "TfLiteTensorBuffer::~TfLiteTensorBuffer");

  LogDeallocation();
  tensorflow::cpu_allocator()->DeallocateRaw(data());
}

StringTfLiteTensorBuffer::StringTfLiteTensorBuffer(const TfLiteTensor* tensor)
    : StringTfLiteTensorBuffer(
          tensor, tensor->data.raw != nullptr ? GetStringCount(tensor) : 0) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSflexPSbuffer_map_utilDTcc mht_5(mht_5_v, 259, "", "./tensorflow/lite/delegates/flex/buffer_map_util.cc", "StringTfLiteTensorBuffer::StringTfLiteTensorBuffer");
}

StringTfLiteTensorBuffer::~StringTfLiteTensorBuffer() {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSflexPSbuffer_map_utilDTcc mht_6(mht_6_v, 264, "", "./tensorflow/lite/delegates/flex/buffer_map_util.cc", "StringTfLiteTensorBuffer::~StringTfLiteTensorBuffer");

  LogDeallocation();
  tensorflow::TypedAllocator::Deallocate<tensorflow::tstring>(
      tensorflow::cpu_allocator(), static_cast<tensorflow::tstring*>(data()),
      num_strings_);
}

StringTfLiteTensorBuffer::StringTfLiteTensorBuffer(const TfLiteTensor* tensor,
                                                   int num_strings)
    : BaseTfLiteTensorBuffer(
          num_strings != 0
              ? tensorflow::TypedAllocator::Allocate<tensorflow::tstring>(
                    tensorflow::cpu_allocator(), num_strings,
                    tensorflow::AllocationAttributes())
              : nullptr),
      num_strings_(num_strings) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSflexPSbuffer_map_utilDTcc mht_7(mht_7_v, 282, "", "./tensorflow/lite/delegates/flex/buffer_map_util.cc", "StringTfLiteTensorBuffer::StringTfLiteTensorBuffer");

  LogAllocation();

  if (data()) {
    tensorflow::tstring* p = static_cast<tensorflow::tstring*>(data());
    for (size_t i = 0; i < num_strings_; ++p, ++i) {
      auto ref = GetString(tensor, i);
      p->assign(ref.str, ref.len);
    }
  }
}

tensorflow::Status SetTfTensorFromTfLite(const TfLiteTensor* tensor,
                                         tensorflow::Tensor* tf_tensor) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSflexPSbuffer_map_utilDTcc mht_8(mht_8_v, 298, "", "./tensorflow/lite/delegates/flex/buffer_map_util.cc", "SetTfTensorFromTfLite");

  if (resource::IsBuiltinResource(tensor)) {
    // If this is native TF Lite resource variable, then we create a TF resource
    // tensor where the tensor handle encodes the identifier of the TF Lite
    // resource.
    // This approach assumes that there is only a single model being invoked
    // via the Interpreter instance, so that the resource IDs won't have any
    // collisions. If we plan to support concurrent execution in the future, we
    // should make sure the resource ID being encoded is unique between
    // different executions.
    tensorflow::Tensor t(tensorflow::DT_RESOURCE, tensorflow::TensorShape({}));
    tensorflow::ResourceHandle handle;
    handle.set_name(TfLiteResourceIdentifier(tensor));
    t.flat<tensorflow::ResourceHandle>()(0) = handle;
    *tf_tensor = t;
    return tensorflow::Status::OK();
  } else if (IsResourceOrVariant(tensor)) {
    // TODO(b/179094265): This is an experimental implementation, subject to
    // change. This can be re-implemented with life cycle management mechanism
    // like reference counting.
    // In a different subgraph, it can load the TensorFlow tensor pointer of the
    // given TensorFlow Lite tensor, which is stored in the `data` field. The
    // memory management cycle of the shared TensorFlow's tensor will be managed
    // by the buffer maps since the loaded tensors always will be kept in the
    // buffer map.
    //
    // The life cycle of the pointer will be managed by the reference counting
    // in the TensorFlow world and the pointer will be freed when all the buffer
    // maps, who own it, are gone.
    const tensorflow::Tensor** tf_tensor_ptr =
        reinterpret_cast<const tensorflow::Tensor**>(tensor->data.raw);
    *tf_tensor = **tf_tensor_ptr;
    return tensorflow::Status::OK();
  }

  tensorflow::TensorShape shape;
  int num_dims = tensor->dims->size;
  for (int i = 0; i < num_dims; ++i) {
    shape.AddDim(tensor->dims->data[i]);
  }
  // TODO(b/152916533): We assume this is a new tensor and allocate a new buffer
  // for it. This is not always the best approach. For example, this might
  // be a reallocation after resizing tensors. In that case it would be
  // preferable to somehow reuse the buffer.
  BaseTfLiteTensorBuffer* buf;
  if (tensor->type == kTfLiteString) {
    buf = new StringTfLiteTensorBuffer(tensor);
  } else {
    buf = new TfLiteTensorBuffer(tensor);
  }
  tensorflow::Tensor t = tensorflow::TensorCApi::MakeTensor(
      GetTensorFlowDataType(tensor->type), shape, buf);
  buf->Unref();

  *tf_tensor = std::move(t);
  return tensorflow::Status::OK();
}

}  // namespace flex
}  // namespace tflite
