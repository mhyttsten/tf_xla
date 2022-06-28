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
class MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSclPSbufferDTcc {
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
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSclPSbufferDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSclPSbufferDTcc() {
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

#include "tensorflow/lite/delegates/gpu/cl/buffer.h"

#include <string>

#include "absl/status/status.h"
#include "tensorflow/lite/delegates/gpu/common/data_type.h"
#include "tensorflow/lite/delegates/gpu/common/status.h"

namespace tflite {
namespace gpu {
namespace cl {
namespace {

absl::Status CreateBuffer(size_t size_in_bytes, bool gpu_read_only,
                          const void* data, CLContext* context,
                          Buffer* result) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSclPSbufferDTcc mht_0(mht_0_v, 200, "", "./tensorflow/lite/delegates/gpu/cl/buffer.cc", "CreateBuffer");

  cl_mem buffer;
  RETURN_IF_ERROR(CreateCLBuffer(context->context(), size_in_bytes,
                                 gpu_read_only, const_cast<void*>(data),
                                 &buffer));
  *result = Buffer(buffer, size_in_bytes);

  return absl::OkStatus();
}

absl::Status CreateSubBuffer(const Buffer& parent, size_t origin_in_bytes,
                             size_t size_in_bytes, bool gpu_read_only,
                             CLContext* context, Buffer* result) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSclPSbufferDTcc mht_1(mht_1_v, 215, "", "./tensorflow/lite/delegates/gpu/cl/buffer.cc", "CreateSubBuffer");

  cl_mem buffer;
  if (parent.IsSubBuffer()) {
    return absl::InvalidArgumentError(
        "Cannot create a sub-buffer from a sub-buffer!");
  }
  RETURN_IF_ERROR(CreateCLSubBuffer(context->context(), parent.GetMemoryPtr(),
                                    origin_in_bytes, size_in_bytes,
                                    gpu_read_only, &buffer));
  *result = Buffer(buffer, size_in_bytes, /*is_sub_buffer=*/true);

  return absl::OkStatus();
}
}  // namespace

Buffer::Buffer(cl_mem buffer, size_t size_in_bytes, bool is_sub_buffer)
    : buffer_(buffer), size_(size_in_bytes), is_sub_buffer_(is_sub_buffer) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSclPSbufferDTcc mht_2(mht_2_v, 234, "", "./tensorflow/lite/delegates/gpu/cl/buffer.cc", "Buffer::Buffer");
}

Buffer::Buffer(Buffer&& buffer)
    : buffer_(buffer.buffer_),
      size_(buffer.size_),
      is_sub_buffer_(buffer.is_sub_buffer_) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSclPSbufferDTcc mht_3(mht_3_v, 242, "", "./tensorflow/lite/delegates/gpu/cl/buffer.cc", "Buffer::Buffer");

  buffer.buffer_ = nullptr;
  buffer.size_ = 0;
  buffer.is_sub_buffer_ = false;
}

Buffer& Buffer::operator=(Buffer&& buffer) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSclPSbufferDTcc mht_4(mht_4_v, 251, "", "./tensorflow/lite/delegates/gpu/cl/buffer.cc", "=");

  if (this != &buffer) {
    Release();
    std::swap(size_, buffer.size_);
    std::swap(buffer_, buffer.buffer_);
    std::swap(is_sub_buffer_, buffer.is_sub_buffer_);
  }
  return *this;
}

void Buffer::Release() {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSclPSbufferDTcc mht_5(mht_5_v, 264, "", "./tensorflow/lite/delegates/gpu/cl/buffer.cc", "Buffer::Release");

  if (buffer_) {
    clReleaseMemObject(buffer_);
    buffer_ = nullptr;
    size_ = 0;
    is_sub_buffer_ = false;
  }
}

absl::Status Buffer::GetGPUResources(const GPUObjectDescriptor* obj_ptr,
                                     GPUResourcesWithValue* resources) const {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSclPSbufferDTcc mht_6(mht_6_v, 277, "", "./tensorflow/lite/delegates/gpu/cl/buffer.cc", "Buffer::GetGPUResources");

  const auto* buffer_desc = dynamic_cast<const BufferDescriptor*>(obj_ptr);
  if (!buffer_desc) {
    return absl::InvalidArgumentError("Expected BufferDescriptor on input.");
  }

  resources->buffers.push_back({"buffer", buffer_});
  return absl::OkStatus();
}

absl::Status Buffer::CreateFromBufferDescriptor(const BufferDescriptor& desc,
                                                CLContext* context) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSclPSbufferDTcc mht_7(mht_7_v, 291, "", "./tensorflow/lite/delegates/gpu/cl/buffer.cc", "Buffer::CreateFromBufferDescriptor");

  bool read_only = desc.memory_type == MemoryType::CONSTANT;
  uint8_t* data_ptr = desc.data.empty()
                          ? nullptr
                          : const_cast<unsigned char*>(desc.data.data());
  size_ = desc.size;
  return CreateCLBuffer(context->context(), desc.size, read_only, data_ptr,
                        &buffer_);
}

absl::Status CreateReadOnlyBuffer(size_t size_in_bytes, CLContext* context,
                                  Buffer* result) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSclPSbufferDTcc mht_8(mht_8_v, 305, "", "./tensorflow/lite/delegates/gpu/cl/buffer.cc", "CreateReadOnlyBuffer");

  return CreateBuffer(size_in_bytes, true, nullptr, context, result);
}

absl::Status CreateReadOnlyBuffer(size_t size_in_bytes, const void* data,
                                  CLContext* context, Buffer* result) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSclPSbufferDTcc mht_9(mht_9_v, 313, "", "./tensorflow/lite/delegates/gpu/cl/buffer.cc", "CreateReadOnlyBuffer");

  return CreateBuffer(size_in_bytes, true, data, context, result);
}

absl::Status CreateReadWriteBuffer(size_t size_in_bytes, CLContext* context,
                                   Buffer* result) {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSclPSbufferDTcc mht_10(mht_10_v, 321, "", "./tensorflow/lite/delegates/gpu/cl/buffer.cc", "CreateReadWriteBuffer");

  return CreateBuffer(size_in_bytes, false, nullptr, context, result);
}

absl::Status CreateReadWriteSubBuffer(const Buffer& parent,
                                      size_t origin_in_bytes,
                                      size_t size_in_bytes, CLContext* context,
                                      Buffer* result) {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSclPSbufferDTcc mht_11(mht_11_v, 331, "", "./tensorflow/lite/delegates/gpu/cl/buffer.cc", "CreateReadWriteSubBuffer");

  return CreateSubBuffer(parent, origin_in_bytes, size_in_bytes,
                         /*gpu_read_only=*/false, context, result);
}

}  // namespace cl
}  // namespace gpu
}  // namespace tflite
