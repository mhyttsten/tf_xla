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
class MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSmetalPSbufferDTcc {
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
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSmetalPSbufferDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSmetalPSbufferDTcc() {
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

#include "tensorflow/lite/delegates/gpu/metal/buffer.h"

#include <utility>

namespace tflite {
namespace gpu {
namespace metal {

Buffer::Buffer(id<MTLBuffer> buffer, size_t size_in_bytes)
    : buffer_(buffer), size_(size_in_bytes) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSmetalPSbufferDTcc mht_0(mht_0_v, 194, "", "./tensorflow/lite/delegates/gpu/metal/buffer.cc", "Buffer::Buffer");
}

Buffer::Buffer(Buffer&& buffer) : buffer_(buffer.buffer_), size_(buffer.size_) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSmetalPSbufferDTcc mht_1(mht_1_v, 199, "", "./tensorflow/lite/delegates/gpu/metal/buffer.cc", "Buffer::Buffer");

  buffer.buffer_ = nullptr;
  buffer.size_ = 0;
}

Buffer& Buffer::operator=(Buffer&& buffer) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSmetalPSbufferDTcc mht_2(mht_2_v, 207, "", "./tensorflow/lite/delegates/gpu/metal/buffer.cc", "=");

  if (this != &buffer) {
    Release();
    std::swap(size_, buffer.size_);
    std::swap(buffer_, buffer.buffer_);
  }
  return *this;
}

Buffer::~Buffer() {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSmetalPSbufferDTcc mht_3(mht_3_v, 219, "", "./tensorflow/lite/delegates/gpu/metal/buffer.cc", "Buffer::~Buffer");
 Release(); }

void Buffer::Release() {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSmetalPSbufferDTcc mht_4(mht_4_v, 224, "", "./tensorflow/lite/delegates/gpu/metal/buffer.cc", "Buffer::Release");

  if (buffer_) {
    buffer_ = nullptr;
    size_ = 0;
  }
}

absl::Status Buffer::GetGPUResources(const GPUObjectDescriptor* obj_ptr,
                                     GPUResourcesWithValue* resources) const {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSmetalPSbufferDTcc mht_5(mht_5_v, 235, "", "./tensorflow/lite/delegates/gpu/metal/buffer.cc", "Buffer::GetGPUResources");

  const auto* buffer_desc = dynamic_cast<const BufferDescriptor*>(obj_ptr);
  if (!buffer_desc) {
    return absl::InvalidArgumentError("Expected BufferDescriptor on input.");
  }

  resources->buffers.push_back({"buffer", {buffer_, 0}});
  return absl::OkStatus();
}

absl::Status Buffer::CreateFromBufferDescriptor(const BufferDescriptor& desc,
                                                id<MTLDevice> device) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSmetalPSbufferDTcc mht_6(mht_6_v, 249, "", "./tensorflow/lite/delegates/gpu/metal/buffer.cc", "Buffer::CreateFromBufferDescriptor");

  size_ = desc.size;
  if (desc.data.empty()) {
    buffer_ =
        [device newBufferWithLength:size_ options:MTLResourceStorageModeShared];
  } else {
    buffer_ = [device newBufferWithBytes:desc.data.data()
                                  length:size_
                                 options:MTLResourceStorageModeShared];
  }
  return absl::OkStatus();
}

absl::Status CreateBuffer(size_t size_in_bytes, const void* data,
                          id<MTLDevice> device, Buffer* result) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSmetalPSbufferDTcc mht_7(mht_7_v, 266, "", "./tensorflow/lite/delegates/gpu/metal/buffer.cc", "CreateBuffer");

  id<MTLBuffer> buffer;
  if (data) {
    buffer = [device newBufferWithBytes:data
                                 length:size_in_bytes
                                options:MTLResourceStorageModeShared];
  } else {
    buffer = [device newBufferWithLength:size_in_bytes
                                 options:MTLResourceStorageModeShared];
  }

  *result = Buffer(buffer, size_in_bytes);

  return absl::OkStatus();
}

}  // namespace metal
}  // namespace gpu
}  // namespace tflite
