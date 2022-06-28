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

#ifndef TENSORFLOW_LITE_DELEGATES_GPU_GL_OBJECT_H_
#define TENSORFLOW_LITE_DELEGATES_GPU_GL_OBJECT_H_
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
class MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSglPSobjectDTh {
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
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSglPSobjectDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSglPSobjectDTh() {
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


#include <array>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

#include "absl/types/variant.h"
#include "tensorflow/lite/delegates/gpu/common/access_type.h"
#include "tensorflow/lite/delegates/gpu/common/data_type.h"
#include "tensorflow/lite/delegates/gpu/common/shape.h"
#include "tensorflow/lite/delegates/gpu/common/types.h"
#include "tensorflow/lite/delegates/gpu/common/util.h"

namespace tflite {
namespace gpu {
namespace gl {

using ObjectData = std::vector<uint8_t>;

// Generic identifier to be used to lookup an object.
using ObjectRef = uint32_t;

constexpr ObjectRef kInvalidObjectRef = ~0;

enum class ObjectType : int {
  UNKNOWN = 0,
  TEXTURE = 1,
  BUFFER = 2,
};

using ObjectSize = absl::variant<size_t, uint2, uint3>;

// An object represents a reference to or pre-defined constant OpenGL Buffer or
// Texture. NodeShader is supposed to set all fields but leave binding = 0
// that will be set later by a compiler.
struct Object {
  AccessType access;

  DataType data_type;

  ObjectType object_type;

  // OpenGL-specific binding information
  uint32_t binding;

  // Indicates size of 1D, 2D or 3D object in elements, where single element
  // consists of 4 values.
  ObjectSize size;

  absl::variant<ObjectData, ObjectRef> object;
};

// @return true if object is a reference.
inline bool IsRef(const Object& object) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSglPSobjectDTh mht_0(mht_0_v, 241, "", "./tensorflow/lite/delegates/gpu/gl/object.h", "IsRef");

  return !absl::holds_alternative<ObjectData>(object.object);
}

inline ObjectRef GetRef(const Object& object) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSglPSobjectDTh mht_1(mht_1_v, 248, "", "./tensorflow/lite/delegates/gpu/gl/object.h", "GetRef");

  auto ref = absl::get_if<ObjectRef>(&object.object);
  return ref ? *ref : kInvalidObjectRef;
}

inline const ObjectData* GetData(const Object& object) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSglPSobjectDTh mht_2(mht_2_v, 256, "", "./tensorflow/lite/delegates/gpu/gl/object.h", "GetData");

  return absl::get_if<ObjectData>(&object.object);
}

inline size_t ByteSizeOf(const Object& object);

// @return object that references an object created externally.
inline Object MakeObjectRef(ObjectRef unique_id, const ObjectSize& size,
                            AccessType access_type) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSglPSobjectDTh mht_3(mht_3_v, 267, "", "./tensorflow/lite/delegates/gpu/gl/object.h", "MakeObjectRef");

  return Object{access_type, DataType::FLOAT32, ObjectType::UNKNOWN, 0,
                size,        unique_id};
}

namespace internal_object {

template <typename T>
std::vector<uint8_t> ToBytesVector(const std::vector<T>& data,
                                   size_t alignment) {
  std::vector<uint8_t> t(AlignByN(data.size() * sizeof(T), alignment));
  std::memcpy(t.data(), data.data(), data.size() * sizeof(T));
  return t;
}

struct ObjectSizer {
  size_t operator()(const uint3& size) const {
    return size.x * size.y * size.z;
  }

  size_t operator()(const uint2& size) const { return size.x * size.y; }

  size_t operator()(uint32_t size) const { return size; }
};

}  // namespace internal_object

inline size_t NumElements(const ObjectSize& size) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSglPSobjectDTh mht_4(mht_4_v, 297, "", "./tensorflow/lite/delegates/gpu/gl/object.h", "NumElements");

  return absl::visit(internal_object::ObjectSizer{}, size);
}

inline size_t ByteSizeOf(const Object& object) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSglPSobjectDTh mht_5(mht_5_v, 304, "", "./tensorflow/lite/delegates/gpu/gl/object.h", "ByteSizeOf");

  return SizeOf(object.data_type) * /* vec4 */ 4 * NumElements(object.size);
}

inline Object MakeReadonlyObject(const ObjectSize& size,
                                 const std::vector<float>& data) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSglPSobjectDTh mht_6(mht_6_v, 312, "", "./tensorflow/lite/delegates/gpu/gl/object.h", "MakeReadonlyObject");

  return Object{AccessType::READ,
                DataType::FLOAT32,
                ObjectType::UNKNOWN,
                0,
                size,
                internal_object::ToBytesVector(data, 16)};
}

inline Object MakeReadonlyTexture(const ObjectSize& size,
                                  const std::vector<float>& data) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSglPSobjectDTh mht_7(mht_7_v, 325, "", "./tensorflow/lite/delegates/gpu/gl/object.h", "MakeReadonlyTexture");

  return Object{AccessType::READ,
                DataType::FLOAT32,
                ObjectType::TEXTURE,
                0,
                size,
                internal_object::ToBytesVector(data, 16)};
}

inline Object MakeReadonlyBuffer(const ObjectSize& size,
                                 const std::vector<float>& data) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSglPSobjectDTh mht_8(mht_8_v, 338, "", "./tensorflow/lite/delegates/gpu/gl/object.h", "MakeReadonlyBuffer");

  return Object{AccessType::READ,
                DataType::FLOAT32,
                ObjectType::BUFFER,
                0,
                size,
                internal_object::ToBytesVector(data, 16)};
}

inline Object MakeReadonlyObject(const std::vector<float>& data) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSglPSobjectDTh mht_9(mht_9_v, 350, "", "./tensorflow/lite/delegates/gpu/gl/object.h", "MakeReadonlyObject");

  return MakeReadonlyObject(
      DivideRoundUp(static_cast<uint32_t>(data.size()), 4U), data);
}

inline Object MakeReadonlyTexture(const std::vector<float>& data) {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSglPSobjectDTh mht_10(mht_10_v, 358, "", "./tensorflow/lite/delegates/gpu/gl/object.h", "MakeReadonlyTexture");

  return MakeReadonlyTexture(
      DivideRoundUp(static_cast<uint32_t>(data.size()), 4U), data);
}

inline Object MakeReadonlyBuffer(const std::vector<float>& data) {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSglPSobjectDTh mht_11(mht_11_v, 366, "", "./tensorflow/lite/delegates/gpu/gl/object.h", "MakeReadonlyBuffer");

  return MakeReadonlyBuffer(
      DivideRoundUp(static_cast<uint32_t>(data.size()), 4U), data);
}

// TODO(akulik): find better place for functions below.

inline uint3 GetPHWC4Size(const BHWC& shape) {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSglPSobjectDTh mht_12(mht_12_v, 376, "", "./tensorflow/lite/delegates/gpu/gl/object.h", "GetPHWC4Size");

  uint3 size;
  size.x = shape.w;
  size.y = shape.h;
  size.z = shape.b * DivideRoundUp(shape.c, 4);
  return size;
}

inline Object MakePHWC4Ref(uint32_t global_id, const BHWC& shape) {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSglPSobjectDTh mht_13(mht_13_v, 387, "", "./tensorflow/lite/delegates/gpu/gl/object.h", "MakePHWC4Ref");

  return MakeObjectRef(global_id, GetPHWC4Size(shape), AccessType::READ_WRITE);
}

}  // namespace gl
}  // namespace gpu
}  // namespace tflite

#endif  // TENSORFLOW_LITE_DELEGATES_GPU_GL_OBJECT_H_
