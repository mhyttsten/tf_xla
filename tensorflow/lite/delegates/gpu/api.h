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

#ifndef TENSORFLOW_LITE_DELEGATES_GPU_API_H_
#define TENSORFLOW_LITE_DELEGATES_GPU_API_H_
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
class MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSapiDTh {
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
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSapiDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSapiDTh() {
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


// Usage example:
//
//   // Builder is created from a model using GPU-specific parameters.
//   std::unique_ptr<InferenceBuilder> builder = ...;
//
//   // input data is coming from a texture
//   // output data goes to CPU
//   builder->SetInputObjectDef(0, {DataType::FLOAT16, DataLayout::PHWC4,
//                                  ObjectType::OPENGL_TEXTURE, true});
//   builder->SetOutputObjectDef(0, {DataType::FLOAT32, DataLayout::BHWC,
//                                  ObjectType::CPU_MEMORY, false});
//   std::unique_ptr<InferenceRunner> runner;
//   RETURN_IF_ERROR(builder->Build(&runner));  // may take significant time.
//   RETURN_IF_ERROR(
//       runner->SetInputObject(0, OpenGlTexture{texture_ud, texture_format}));
//   RETURN_IF_ERROR(runner->Run());

#include <cstdint>
#include <memory>
#include <vector>

#include "absl/types/span.h"
#include "absl/types/variant.h"
#include <CL/cl.h>
#include "tensorflow/lite/delegates/gpu/common/data_type.h"
#include "tensorflow/lite/delegates/gpu/common/status.h"
#include "tensorflow/lite/delegates/gpu/common/util.h"
#include "vulkan/vulkan.h"  // from @vulkan_headers

#define GL_NO_PROTOTYPES
#define EGL_NO_PROTOTYPES
#include "tensorflow/lite/delegates/gpu/gl/portable_gl31.h"
#undef GL_NO_PROTOTYPES
#undef EGL_NO_PROTOTYPES

namespace tflite {
namespace gpu {

// Common abbreviations:
//   B  - batch
//   H  - height
//   W  - width
//   C  - channels
//   D  - depth := DivideRoundUp(C, 4)
//   C4 - is the constant = 4.
enum class DataLayout {
  UNKNOWN,
  BHWC,
  DHWC4,
  HWDC4,
  HDWC4,
};

enum class ObjectType {
  UNKNOWN,
  OPENGL_SSBO,
  OPENGL_TEXTURE,
  CPU_MEMORY,
  OPENCL_TEXTURE,
  OPENCL_BUFFER,
  VULKAN_BUFFER,
  VULKAN_TEXTURE
};

struct OpenGlBuffer {
  OpenGlBuffer() = default;
  explicit OpenGlBuffer(GLuint new_id) : id(new_id) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSapiDTh mht_0(mht_0_v, 254, "", "./tensorflow/lite/delegates/gpu/api.h", "OpenGlBuffer");
}

  GLuint id = GL_INVALID_INDEX;
};

struct OpenGlTexture {
  OpenGlTexture() = default;
  OpenGlTexture(GLuint new_id, GLenum new_format)
      : id(new_id), format(new_format) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSapiDTh mht_1(mht_1_v, 265, "", "./tensorflow/lite/delegates/gpu/api.h", "OpenGlTexture");
}

  GLuint id = GL_INVALID_INDEX;
  GLenum format = GL_INVALID_ENUM;
};

struct OpenClBuffer {
  OpenClBuffer() = default;
  explicit OpenClBuffer(cl_mem new_memobj) : memobj(new_memobj) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSapiDTh mht_2(mht_2_v, 276, "", "./tensorflow/lite/delegates/gpu/api.h", "OpenClBuffer");
}

  cl_mem memobj = nullptr;
};

struct OpenClTexture {
  OpenClTexture() = default;
  explicit OpenClTexture(cl_mem new_memobj) : memobj(new_memobj) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSapiDTh mht_3(mht_3_v, 286, "", "./tensorflow/lite/delegates/gpu/api.h", "OpenClTexture");
}

  cl_mem memobj = nullptr;
  // TODO(akulik): should it specify texture format?
};

struct VulkanBuffer {
  VulkanBuffer() = default;
  explicit VulkanBuffer(VkBuffer buffer_, VkDeviceSize size_,
                        VkDeviceMemory memory_, VkDeviceSize offset_)
      : buffer(buffer_), size(size_), memory(memory_), offset(offset_) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSapiDTh mht_4(mht_4_v, 299, "", "./tensorflow/lite/delegates/gpu/api.h", "VulkanBuffer");
}

  VkBuffer buffer;
  VkDeviceSize size;
  VkDeviceMemory memory;
  VkDeviceSize offset;
};

struct VulkanTexture {
  VulkanTexture() = default;
  explicit VulkanTexture(VkDeviceMemory new_memory) : memory(new_memory) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSapiDTh mht_5(mht_5_v, 312, "", "./tensorflow/lite/delegates/gpu/api.h", "VulkanTexture");
}

  VkImage image;
  VkImageView image_view;
  VkFormat format;
  VkExtent3D extent;
  VkDeviceMemory memory;
  VkDeviceSize offset;
};

struct VulkanMemory {
  VulkanMemory() = default;
  explicit VulkanMemory(VkDeviceMemory new_memory) : memory(new_memory) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSapiDTh mht_6(mht_6_v, 327, "", "./tensorflow/lite/delegates/gpu/api.h", "VulkanMemory");
}

  VkDeviceMemory memory;
  VkDeviceSize size;
  VkDeviceSize offset;
};

struct CpuMemory {
  CpuMemory() = default;
  CpuMemory(void* new_data, size_t new_size_bytes)
      : data(new_data), size_bytes(new_size_bytes) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSapiDTh mht_7(mht_7_v, 340, "", "./tensorflow/lite/delegates/gpu/api.h", "CpuMemory");
}

  void* data = nullptr;
  size_t size_bytes = 0;
};

template <typename T>
inline CpuMemory MakeCpuMemory(absl::Span<T> t) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSapiDTh mht_8(mht_8_v, 350, "", "./tensorflow/lite/delegates/gpu/api.h", "MakeCpuMemory");

  CpuMemory m;
  m.data = t.data();
  m.size_bytes = t.size() * sizeof(T);
  return m;
}

template <typename T>
inline CpuMemory MakeReadableCpuMemory(absl::Span<const T> t) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSapiDTh mht_9(mht_9_v, 361, "", "./tensorflow/lite/delegates/gpu/api.h", "MakeReadableCpuMemory");

  CpuMemory m;
  m.data = const_cast<T*>(t.data());
  m.size_bytes = t.size() * sizeof(T);
  return m;
}

// Defines object representation.
struct ObjectDef {
  DataType data_type = DataType::UNKNOWN;
  DataLayout data_layout = DataLayout::UNKNOWN;
  ObjectType object_type = ObjectType::UNKNOWN;

  // If true, then object is managed externally and needs to be provided to
  // InferenceRunner by a user before running inference.
  //
  // User-provided objects will not be re-used internally for any purpose to
  // lower overall memory usage.
  bool user_provided = false;

  bool operator==(const ObjectDef& other) const {
    return data_type == other.data_type && data_layout == other.data_layout &&
           object_type == other.object_type &&
           user_provided == other.user_provided;
  }
};

bool IsValid(const ObjectDef& def);

struct Dimensions {
  Dimensions() : b(1), h(1), w(1), c(1) {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSapiDTh mht_10(mht_10_v, 394, "", "./tensorflow/lite/delegates/gpu/api.h", "Dimensions");
}

  Dimensions(int32_t batch, int32_t height, int32_t width, int32_t channels)
      : b(batch), h(height), w(width), c(channels) {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSapiDTh mht_11(mht_11_v, 400, "", "./tensorflow/lite/delegates/gpu/api.h", "Dimensions");
}

  int32_t d() const {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSapiDTh mht_12(mht_12_v, 405, "", "./tensorflow/lite/delegates/gpu/api.h", "d");
 return DivideRoundUp(c, 4); }

  int32_t product() const {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSapiDTh mht_13(mht_13_v, 410, "", "./tensorflow/lite/delegates/gpu/api.h", "product");
 return b * h * w * c; }

  bool operator==(const Dimensions& other) const {
    return b == other.b && h == other.h && w == other.w && c == other.c;
  }

  int32_t b;
  int32_t h;
  int32_t w;
  int32_t c;
};

// Connects tensor shape with corresponding object definition.
struct TensorObjectDef {
  // Dimensions semantic is defined by corresponding DataLayout.
  Dimensions dimensions;
  ObjectDef object_def;

  bool operator==(const TensorObjectDef& other) const {
    return dimensions == other.dimensions && object_def == other.object_def;
  }
};

// @return true if tensor object def is defined.
bool IsValid(const TensorObjectDef& def);

// @return the number of elements in a tensor object.
uint32_t NumElements(const TensorObjectDef& def);

using TensorObject =
    absl::variant<absl::monostate, OpenGlBuffer, OpenGlTexture, CpuMemory,
                  OpenClBuffer, OpenClTexture, VulkanBuffer, VulkanTexture>;

// @return true if object is set and corresponding values are defined.
bool IsValid(const TensorObjectDef& def, const TensorObject& object);

ObjectType GetType(const TensorObject& object);

// @return true if corresponding object is set for the given type
bool IsObjectPresent(ObjectType type, const TensorObject& obj);

// @return true if corresponding object has already been initialized and
// assigned with a specific ObjectType.
bool IsObjectInitialized(const TensorObject& obj);

class InferenceRunner;

// Allows to inspect and change input and output definitions before a graph is
// prepared for the inference.
class InferenceBuilder {
 public:
  virtual ~InferenceBuilder() {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSapiDTh mht_14(mht_14_v, 464, "", "./tensorflow/lite/delegates/gpu/api.h", "~InferenceBuilder");
}

  // Returns inference graph inputs and outputs definitions.
  virtual std::vector<TensorObjectDef> inputs() const = 0;
  virtual std::vector<TensorObjectDef> outputs() const = 0;

  // Sets new shape for the input if underlying implementation and graph
  // structure allows dynamic tensors.
  virtual absl::Status SetInputShape(int index,
                                     const Dimensions& dimensions) = 0;

  // Updates object definitions for the given index. Implementation may allow
  // to use different layouts and/or data type conversions between objects
  // defined in a graph and given objects, for example:
  //   input '0' is DataType::FLOAT32, DataLayout::BHWC.
  //   A user, however, has an input in DataType::FLOAT16, DataLayout::PHWC4.
  //   An implementation may allow this transformation to happen automatically
  //   under the hood.
  virtual absl::Status SetInputObjectDef(int index, ObjectDef def) = 0;
  virtual absl::Status SetOutputObjectDef(int index, ObjectDef def) = 0;
  virtual absl::Status SetAllInputObjectDefsTo(ObjectDef def) {
   std::vector<std::string> mht_15_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSapiDTh mht_15(mht_15_v, 487, "", "./tensorflow/lite/delegates/gpu/api.h", "SetAllInputObjectDefsTo");

    auto input_defs = inputs();
    for (int i = 0; i < input_defs.size(); ++i) {
      RETURN_IF_ERROR(SetInputObjectDef(i, def));
    }
    return absl::OkStatus();
  }
  virtual absl::Status SetAllOutputObjectDefsTo(ObjectDef def) {
   std::vector<std::string> mht_16_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSapiDTh mht_16(mht_16_v, 497, "", "./tensorflow/lite/delegates/gpu/api.h", "SetAllOutputObjectDefsTo");

    auto output_defs = outputs();
    for (int i = 0; i < output_defs.size(); ++i) {
      RETURN_IF_ERROR(SetOutputObjectDef(i, def));
    }
    return absl::OkStatus();
  }

  // Creates new instance of the inference runner. InferenceBuilder stays valid
  // and could be used to create another inference runner if needed.
  //
  // This method may take significant time to prepare new inference runner. For
  // example, it may require to compile OpenGL shaders.
  virtual absl::Status Build(std::unique_ptr<InferenceRunner>* runner) = 0;
};

// Runs prepared inference. Every object marked as external needs to be set
// prior calling Run method.
class InferenceRunner {
 public:
  virtual ~InferenceRunner() {
   std::vector<std::string> mht_17_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSapiDTh mht_17(mht_17_v, 520, "", "./tensorflow/lite/delegates/gpu/api.h", "~InferenceRunner");
}

  // Returns inference graph inputs and outputs definitions.
  virtual std::vector<TensorObjectDef> inputs() const = 0;
  virtual std::vector<TensorObjectDef> outputs() const = 0;

  // Getters provide access to underlying objects for the given index.
  // Setters allow to set or change external object for the given index. Note,
  // object need to match object definition set before in InferenceBuilder.

  virtual absl::Status GetInputObject(int index, TensorObject* object) = 0;
  virtual absl::Status GetOutputObject(int index, TensorObject* object) = 0;
  virtual absl::Status SetInputObject(int index, TensorObject object) = 0;
  virtual absl::Status SetOutputObject(int index, TensorObject object) = 0;

  virtual absl::Status Run() = 0;
};

// Encapsulated compilation/runtime tradeoffs.
enum class InferenceUsage {
  UNKNOWN,

  // InferenceRunner will be used only once. Therefore, it is important to
  // minimize bootstrap time as well.
  FAST_SINGLE_ANSWER,

  // Prefer maximizing the throughput. Same inference runner will be used
  // repeatedly on different inputs.
  SUSTAINED_SPEED,
};

// Defines aspects to control while instantiating a runner.
enum class InferencePriority {
  UNKNOWN,

  AUTO,

  MIN_LATENCY,

  MAX_PRECISION,

  MIN_MEMORY_USAGE,
};

struct InferenceOptions {
  InferenceUsage usage = InferenceUsage::SUSTAINED_SPEED;

  // Ordered priorities provide better understanding of desired semantics,
  // where priority(n) is more important than priority(n+1).
  // AUTO priority is needed when a single priority is the most important
  // factor. For example, priority1 = InferencePriority::MIN_LATENCY and leaving
  // everything else to AUTO would result in configuration that achieves maximum
  // performance.
  //
  // AUTO priority can only be used when higher priorities are fully specified.
  // For example:
  //   VALID:   priority1 = MIN_LATENCY, priority2 = AUTO, priority3 = AUTO
  //   VALID:   priority1 = MIN_LATENCY, priority2 = MAX_PRECISION,
  //            priority3 = AUTO
  //   INVALID: priority1 = AUTO, priority2 = MIN_LATENCY, priority3 = AUTO
  //   INVALID: priority1 = MIN_LATENCY, priority2 = AUTO,
  //            priority3 = MAX_PRECISION
  // Invalid priorities will result in error.
  InferencePriority priority1 = InferencePriority::MAX_PRECISION;

  InferencePriority priority2 = InferencePriority::AUTO;

  InferencePriority priority3 = InferencePriority::AUTO;
};

// Returns a position number for the priority. If priority is missing,
// then it would return 'max num priorities + 1'.
int GetPosition(const InferenceOptions& options, InferencePriority p);

// Return true if options are valid.
bool IsValid(const InferenceOptions& options);

// Resolves AUTO priorities and specifies them explicitly.
// Note, no-one should assume that these mappings will not change.
// Technically this function is declared here for code re-use purposes and
// by no means it should be treated as canonical way to resolve AUTO.
void ResolveAutoPriority(InferenceOptions* options);

enum class PriorityImportance {
  UNKNOWN,
  HIGHER,
  LOWER,
};

// If both p1 and p2 are not present in options, return UNKNOWN
// If p1 is present, but p2 is not, return HIGHER
// If p2 is present, but p1 is not, return LOWER
// If both are present, and p1 is more important, return HIGHER, otherwise,
// LOWER.
PriorityImportance GetRelativeImportance(const InferenceOptions& options,
                                         InferencePriority p1,
                                         InferencePriority p2);

}  // namespace gpu
}  // namespace tflite

#endif  // TENSORFLOW_LITE_DELEGATES_GPU_API_H_
