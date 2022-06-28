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

#ifndef TENSORFLOW_LITE_DELEGATES_GPU_COMMON_TASK_GPU_OBJECT_DESC_H_
#define TENSORFLOW_LITE_DELEGATES_GPU_COMMON_TASK_GPU_OBJECT_DESC_H_
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
class MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStaskPSgpu_object_descDTh {
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
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStaskPSgpu_object_descDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStaskPSgpu_object_descDTh() {
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


#include <map>
#include <memory>
#include <string>
#include <vector>

#include "tensorflow/lite/delegates/gpu/common/access_type.h"
#include "tensorflow/lite/delegates/gpu/common/data_type.h"
#include "tensorflow/lite/delegates/gpu/common/gpu_info.h"
#include "tensorflow/lite/delegates/gpu/common/status.h"
#include "tensorflow/lite/delegates/gpu/common/task/serialization_base_generated.h"

namespace tflite {
namespace gpu {

struct GPUImage2DDescriptor {
  DataType data_type;
  bool normalized = false;   // used with INT data types, if normalized, we read
                             // in kernel float data.
  DataType normalized_type;  // can be FLOAT32 or FLOAT16, using with normalized
                             // = true
  AccessType access_type;
};

struct GPUImage3DDescriptor {
  DataType data_type;
  AccessType access_type;
};

struct GPUImage2DArrayDescriptor {
  DataType data_type;
  AccessType access_type;
};

struct GPUImageBufferDescriptor {
  DataType data_type;
  AccessType access_type;
};

struct GPUCustomMemoryDescriptor {
  std::string type_name;
};

enum class MemoryType { GLOBAL, CONSTANT, LOCAL };

struct GPUBufferDescriptor {
  DataType data_type;
  AccessType access_type;
  int element_size;
  MemoryType memory_type = MemoryType::GLOBAL;
  std::vector<std::string> attributes;
};

struct GPUResources {
  std::vector<std::string> ints;
  std::vector<std::string> floats;
  std::vector<std::pair<std::string, GPUBufferDescriptor>> buffers;
  std::vector<std::pair<std::string, GPUImage2DDescriptor>> images2d;
  std::vector<std::pair<std::string, GPUImage2DArrayDescriptor>> image2d_arrays;
  std::vector<std::pair<std::string, GPUImage3DDescriptor>> images3d;
  std::vector<std::pair<std::string, GPUImageBufferDescriptor>> image_buffers;
  std::vector<std::pair<std::string, GPUCustomMemoryDescriptor>>
      custom_memories;

  std::vector<std::string> GetNames() const {
    std::vector<std::string> names = ints;
    names.insert(names.end(), floats.begin(), floats.end());
    for (const auto& obj : buffers) {
      names.push_back(obj.first);
    }
    for (const auto& obj : images2d) {
      names.push_back(obj.first);
    }
    for (const auto& obj : image2d_arrays) {
      names.push_back(obj.first);
    }
    for (const auto& obj : images3d) {
      names.push_back(obj.first);
    }
    for (const auto& obj : image_buffers) {
      names.push_back(obj.first);
    }
    for (const auto& obj : custom_memories) {
      names.push_back(obj.first);
    }
    return names;
  }

  int GetReadImagesCount() const {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStaskPSgpu_object_descDTh mht_0(mht_0_v, 275, "", "./tensorflow/lite/delegates/gpu/common/task/gpu_object_desc.h", "GetReadImagesCount");

    int counter = 0;
    for (const auto& t : images2d) {
      if (t.second.access_type == tflite::gpu::AccessType::READ) {
        counter++;
      }
    }
    for (const auto& t : image2d_arrays) {
      if (t.second.access_type == tflite::gpu::AccessType::READ) {
        counter++;
      }
    }
    for (const auto& t : images3d) {
      if (t.second.access_type == tflite::gpu::AccessType::READ) {
        counter++;
      }
    }
    for (const auto& t : image_buffers) {
      if (t.second.access_type == tflite::gpu::AccessType::READ) {
        counter++;
      }
    }
    return counter;
  }

  int GetWriteImagesCount() const {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStaskPSgpu_object_descDTh mht_1(mht_1_v, 303, "", "./tensorflow/lite/delegates/gpu/common/task/gpu_object_desc.h", "GetWriteImagesCount");

    int counter = 0;
    for (const auto& t : images2d) {
      if (t.second.access_type == tflite::gpu::AccessType::WRITE) {
        counter++;
      }
    }
    for (const auto& t : image2d_arrays) {
      if (t.second.access_type == tflite::gpu::AccessType::WRITE) {
        counter++;
      }
    }
    for (const auto& t : images3d) {
      if (t.second.access_type == tflite::gpu::AccessType::WRITE) {
        counter++;
      }
    }
    for (const auto& t : image_buffers) {
      if (t.second.access_type == tflite::gpu::AccessType::WRITE) {
        counter++;
      }
    }
    return counter;
  }
};

class GPUObjectDescriptor {
 public:
  GPUObjectDescriptor() = default;
  GPUObjectDescriptor(const GPUObjectDescriptor&) = default;
  GPUObjectDescriptor& operator=(const GPUObjectDescriptor&) = default;
  GPUObjectDescriptor(GPUObjectDescriptor&& obj_desc) = default;
  GPUObjectDescriptor& operator=(GPUObjectDescriptor&& obj_desc) = default;
  virtual ~GPUObjectDescriptor() = default;

  void SetStateVar(const std::string& key, const std::string& value) const {
   std::vector<std::string> mht_2_v;
   mht_2_v.push_back("key: \"" + key + "\"");
   mht_2_v.push_back("value: \"" + value + "\"");
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStaskPSgpu_object_descDTh mht_2(mht_2_v, 343, "", "./tensorflow/lite/delegates/gpu/common/task/gpu_object_desc.h", "SetStateVar");

    state_vars_[key] = value;
  }

  virtual absl::Status PerformConstExpr(const tflite::gpu::GpuInfo& gpu_info,
                                        const std::string& const_expr,
                                        std::string* result) const {
   std::vector<std::string> mht_3_v;
   mht_3_v.push_back("const_expr: \"" + const_expr + "\"");
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStaskPSgpu_object_descDTh mht_3(mht_3_v, 353, "", "./tensorflow/lite/delegates/gpu/common/task/gpu_object_desc.h", "PerformConstExpr");

    return absl::UnimplementedError(
        "No implementation of perform const expression");
  }

  virtual absl::Status PerformSelector(
      const GpuInfo& gpu_info, const std::string& selector,
      const std::vector<std::string>& args,
      const std::vector<std::string>& template_args,
      std::string* result) const {
   std::vector<std::string> mht_4_v;
   mht_4_v.push_back("selector: \"" + selector + "\"");
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStaskPSgpu_object_descDTh mht_4(mht_4_v, 366, "", "./tensorflow/lite/delegates/gpu/common/task/gpu_object_desc.h", "PerformSelector");

    return absl::UnimplementedError("No implementation of perform selector");
  }
  virtual GPUResources GetGPUResources(const GpuInfo& gpu_info) const {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStaskPSgpu_object_descDTh mht_5(mht_5_v, 372, "", "./tensorflow/lite/delegates/gpu/common/task/gpu_object_desc.h", "GetGPUResources");

    return GPUResources();
  }

  virtual void Release() {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStaskPSgpu_object_descDTh mht_6(mht_6_v, 379, "", "./tensorflow/lite/delegates/gpu/common/task/gpu_object_desc.h", "Release");
}

  // For internal use, will work correct only for const objects and before
  // Release() call.
  virtual uint64_t GetSizeInBytes() const {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStaskPSgpu_object_descDTh mht_7(mht_7_v, 386, "", "./tensorflow/lite/delegates/gpu/common/task/gpu_object_desc.h", "GetSizeInBytes");
 return 0; }

  void SetAccess(AccessType access_type) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStaskPSgpu_object_descDTh mht_8(mht_8_v, 391, "", "./tensorflow/lite/delegates/gpu/common/task/gpu_object_desc.h", "SetAccess");
 access_type_ = access_type; }
  AccessType GetAccess() const {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStaskPSgpu_object_descDTh mht_9(mht_9_v, 395, "", "./tensorflow/lite/delegates/gpu/common/task/gpu_object_desc.h", "GetAccess");
 return access_type_; }

 protected:
  friend flatbuffers::Offset<tflite::gpu::data::GPUObjectDescriptor> Encode(
      const GPUObjectDescriptor& desc, flatbuffers::FlatBufferBuilder* builder);
  friend void Decode(const tflite::gpu::data::GPUObjectDescriptor* fb_obj,
                     GPUObjectDescriptor* obj);
  mutable std::map<std::string, std::string> state_vars_;
  AccessType access_type_;
};

using GPUObjectDescriptorPtr = std::unique_ptr<GPUObjectDescriptor>;

}  // namespace gpu
}  // namespace tflite

#endif  // TENSORFLOW_LITE_DELEGATES_GPU_COMMON_TASK_GPU_OBJECT_DESC_H_
