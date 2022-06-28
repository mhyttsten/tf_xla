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
class MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSglPSobject_managerDTcc {
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
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSglPSobject_managerDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSglPSobject_managerDTcc() {
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

#include "tensorflow/lite/delegates/gpu/gl/object_manager.h"

#include "absl/memory/memory.h"
#include "absl/types/span.h"
#include "tensorflow/lite/delegates/gpu/common/convert.h"
#include "tensorflow/lite/delegates/gpu/common/status.h"

namespace tflite {
namespace gpu {
namespace gl {

absl::Status CreatePHWC4BufferFromTensor(const TensorFloat32& tensor,
                                         GlBuffer* gl_buffer) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSglPSobject_managerDTcc mht_0(mht_0_v, 197, "", "./tensorflow/lite/delegates/gpu/gl/object_manager.cc", "CreatePHWC4BufferFromTensor");

  std::vector<float> transposed(GetElementsSizeForPHWC4(tensor.shape));
  RETURN_IF_ERROR(
      ConvertToPHWC4(tensor.data, tensor.shape, absl::MakeSpan(transposed)));
  return CreateReadOnlyShaderStorageBuffer<float>(transposed, gl_buffer);
}

absl::Status CreatePHWC4BufferFromTensorRef(const TensorRef<BHWC>& tensor_ref,
                                            GlBuffer* gl_buffer) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSglPSobject_managerDTcc mht_1(mht_1_v, 208, "", "./tensorflow/lite/delegates/gpu/gl/object_manager.cc", "CreatePHWC4BufferFromTensorRef");

  return CreateReadWriteShaderStorageBuffer<float>(
      GetElementsSizeForPHWC4(tensor_ref.shape), gl_buffer);
}

absl::Status CopyFromPHWC4Buffer(const GlBuffer& buffer,
                                 TensorFloat32* tensor) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSglPSobject_managerDTcc mht_2(mht_2_v, 217, "", "./tensorflow/lite/delegates/gpu/gl/object_manager.cc", "CopyFromPHWC4Buffer");

  return buffer.MappedRead<float>([tensor](absl::Span<const float> data) {
    tensor->data.resize(tensor->shape.DimensionsProduct());
    return ConvertFromPHWC4(absl::MakeConstSpan(data), tensor->shape,
                            absl::MakeSpan(tensor->data));
  });
}

absl::Status ObjectManager::RegisterBuffer(uint32_t id, GlBuffer buffer) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSglPSobject_managerDTcc mht_3(mht_3_v, 228, "", "./tensorflow/lite/delegates/gpu/gl/object_manager.cc", "ObjectManager::RegisterBuffer");

  if (id >= buffers_.size()) {
    buffers_.resize(id + 1);
  }
  buffers_[id] = absl::make_unique<GlBuffer>(std::move(buffer));
  return absl::OkStatus();
}

void ObjectManager::RemoveBuffer(uint32_t id) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSglPSobject_managerDTcc mht_4(mht_4_v, 239, "", "./tensorflow/lite/delegates/gpu/gl/object_manager.cc", "ObjectManager::RemoveBuffer");

  if (id < buffers_.size()) {
    buffers_[id].reset(nullptr);
  }
}

GlBuffer* ObjectManager::FindBuffer(uint32_t id) const {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSglPSobject_managerDTcc mht_5(mht_5_v, 248, "", "./tensorflow/lite/delegates/gpu/gl/object_manager.cc", "ObjectManager::FindBuffer");

  return id >= buffers_.size() ? nullptr : buffers_[id].get();
}

absl::Status ObjectManager::RegisterTexture(uint32_t id, GlTexture texture) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSglPSobject_managerDTcc mht_6(mht_6_v, 255, "", "./tensorflow/lite/delegates/gpu/gl/object_manager.cc", "ObjectManager::RegisterTexture");

  if (id >= textures_.size()) {
    textures_.resize(id + 1);
  }
  textures_[id] = absl::make_unique<GlTexture>(std::move(texture));
  return absl::OkStatus();
}

void ObjectManager::RemoveTexture(uint32_t id) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSglPSobject_managerDTcc mht_7(mht_7_v, 266, "", "./tensorflow/lite/delegates/gpu/gl/object_manager.cc", "ObjectManager::RemoveTexture");

  if (id < textures_.size()) {
    textures_[id].reset(nullptr);
  }
}

GlTexture* ObjectManager::FindTexture(uint32_t id) const {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSglPSobject_managerDTcc mht_8(mht_8_v, 275, "", "./tensorflow/lite/delegates/gpu/gl/object_manager.cc", "ObjectManager::FindTexture");

  return id >= textures_.size() ? nullptr : textures_[id].get();
}

ObjectsStats ObjectManager::stats() const {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSglPSobject_managerDTcc mht_9(mht_9_v, 282, "", "./tensorflow/lite/delegates/gpu/gl/object_manager.cc", "ObjectManager::stats");

  ObjectsStats stats;
  for (auto& texture : textures_) {
    if (!texture || !texture->has_ownership()) continue;
    stats.textures.count++;
    stats.textures.total_bytes += texture->bytes_size();
  }
  for (auto& buffer : buffers_) {
    if (!buffer || !buffer->has_ownership()) continue;
    stats.buffers.count++;
    stats.buffers.total_bytes += buffer->bytes_size();
  }
  return stats;
}

}  // namespace gl
}  // namespace gpu
}  // namespace tflite
