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
class MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSglPSapi2DTcc {
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
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSglPSapi2DTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSglPSapi2DTcc() {
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

#include "tensorflow/lite/delegates/gpu/gl/api2.h"

#include <algorithm>
#include <cstring>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/memory/memory.h"
#include "absl/types/span.h"
#include "tensorflow/lite/delegates/gpu/common/data_type.h"
#include "tensorflow/lite/delegates/gpu/common/model.h"
#include "tensorflow/lite/delegates/gpu/common/tensor.h"
#include "tensorflow/lite/delegates/gpu/gl/compiler.h"
#include "tensorflow/lite/delegates/gpu/gl/egl_environment.h"
#include "tensorflow/lite/delegates/gpu/gl/gl_call.h"
#include "tensorflow/lite/delegates/gpu/gl/kernels/converter.h"
#include "tensorflow/lite/delegates/gpu/gl/kernels/registry.h"
#include "tensorflow/lite/delegates/gpu/gl/object.h"
#include "tensorflow/lite/delegates/gpu/gl/portable_gl31.h"
#include "tensorflow/lite/delegates/gpu/gl/request_gpu_info.h"
#include "tensorflow/lite/delegates/gpu/gl/runtime.h"
#include "tensorflow/lite/delegates/gpu/gl/variable.h"
#include "tensorflow/lite/delegates/gpu/gl/workgroups/default_calculator.h"

namespace tflite {
namespace gpu {
namespace gl {
namespace {

std::string GetShaderHeader(uint3 localsize) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSglPSapi2DTcc mht_0(mht_0_v, 216, "", "./tensorflow/lite/delegates/gpu/gl/api2.cc", "GetShaderHeader");

  return absl::StrCat("#version 310 es\nlayout(local_size_x = ", localsize.x,
                      ", local_size_y = ", localsize.y,
                      ", local_size_z = ", localsize.z, ") in;\n");
}

// Wraps given SSBO into GlBuffer object that does not have ownership.
absl::Status WrapSSBO(OpenGlBuffer ssbo, GlBuffer* buffer) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSglPSapi2DTcc mht_1(mht_1_v, 226, "", "./tensorflow/lite/delegates/gpu/gl/api2.cc", "WrapSSBO");

  int64_t size_bytes;
  RETURN_IF_ERROR(GetSSBOSize(ssbo.id, &size_bytes));
  *buffer = GlBuffer(GL_SHADER_STORAGE_BUFFER, ssbo.id, size_bytes, 0, false);
  return absl::OkStatus();
}

absl::Status MaybeAllocateGlBuffer(const TensorObjectDef& def, GlBuffer* ssbo) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSglPSapi2DTcc mht_2(mht_2_v, 236, "", "./tensorflow/lite/delegates/gpu/gl/api2.cc", "MaybeAllocateGlBuffer");

  if (def.object_def.object_type != gpu::ObjectType::OPENGL_SSBO) {
    return absl::InvalidArgumentError("Tensor object is not GL SSBO");
  }
  const uint32_t num_elements = NumElements(def);
  switch (def.object_def.data_type) {
    case DataType::FLOAT32:
      return CreateReadWriteShaderStorageBuffer<float>(num_elements, ssbo);
    case DataType::FLOAT16:
      return CreateReadWriteShaderStorageBuffer<uint16_t>(num_elements, ssbo);
    default:
      return absl::InternalError(
          "Unable to create new GL SSBO. Unsupported data type.");
  }
  return absl::OkStatus();
}

// Does one-step conversion between internal and external objects.
// It may also allocate external objects if requested.
class DefaultTensorTie : public TensorTie {
 public:
  DefaultTensorTie(const TensorTieDef& def, TensorObject internal_obj,
                   ObjectManager* objects)
      : TensorTie(def), objects_(objects), internal_obj_(internal_obj) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSglPSapi2DTcc mht_3(mht_3_v, 262, "", "./tensorflow/lite/delegates/gpu/gl/api2.cc", "DefaultTensorTie");
}

  static bool IsSupported(
      const TensorTieDef& def,
      const TensorObjectConverterBuilder& converter_builder) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSglPSapi2DTcc mht_4(mht_4_v, 269, "", "./tensorflow/lite/delegates/gpu/gl/api2.cc", "IsSupported");

    return converter_builder.IsSupported(def.internal_def, def.external_def) &&
           converter_builder.IsSupported(def.external_def, def.internal_def);
  }

  static absl::Status New(const TensorTieDef& def,
                          TensorObjectConverterBuilder* converter_builder,
                          ObjectManager* objects,
                          std::unique_ptr<TensorTie>* tie) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSglPSapi2DTcc mht_5(mht_5_v, 280, "", "./tensorflow/lite/delegates/gpu/gl/api2.cc", "New");

    auto tie_impl =
        absl::make_unique<DefaultTensorTie>(def, TensorObject{}, objects);
    RETURN_IF_ERROR(tie_impl->Init(converter_builder));
    *tie = std::move(tie_impl);
    return absl::OkStatus();
  }

  static absl::Status New(const TensorTieDef& def,
                          TensorObjectConverterBuilder* converter_builder,
                          TensorObject internal_object,
                          std::unique_ptr<TensorTie>* tie) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSglPSapi2DTcc mht_6(mht_6_v, 294, "", "./tensorflow/lite/delegates/gpu/gl/api2.cc", "New");

    if (!IsValid(def.internal_def, internal_object)) {
      return absl::InternalError("Internal object does not match definition.");
    }

    auto tie_impl =
        absl::make_unique<DefaultTensorTie>(def, internal_object, nullptr);
    RETURN_IF_ERROR(tie_impl->Init(converter_builder));
    *tie = std::move(tie_impl);
    return absl::OkStatus();
  }

  absl::Status CopyToExternalObject() final {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSglPSapi2DTcc mht_7(mht_7_v, 309, "", "./tensorflow/lite/delegates/gpu/gl/api2.cc", "CopyToExternalObject");

    if (!converter_to_) {
      return absl::OkStatus();
    }
    return converter_to_->Convert(internal_obj_, GetExternalObject());
  }

  absl::Status CopyFromExternalObject() final {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSglPSapi2DTcc mht_8(mht_8_v, 319, "", "./tensorflow/lite/delegates/gpu/gl/api2.cc", "CopyFromExternalObject");

    if (!converter_from_) {
      return absl::OkStatus();
    }
    return converter_from_->Convert(GetExternalObject(), internal_obj_);
  }

  absl::Status SetExternalObject(TensorObject obj) final {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSglPSapi2DTcc mht_9(mht_9_v, 329, "", "./tensorflow/lite/delegates/gpu/gl/api2.cc", "SetExternalObject");

    if (!def().external_def.object_def.user_provided) {
      return absl::InvalidArgumentError("External object is read-only");
    }
    if (!IsValid(def().external_def, obj)) {
      return absl::InvalidArgumentError("Given object is not valid");
    }
    external_obj_ = obj;

    // Internal object is not initialized when external object is going to be
    // used as is, with not conversion. In this case we don't need to have a
    // separate internal object, we are just registering the appropriate
    // external object in the object manager for the future binding in the
    // inference runner.
    if (!IsObjectInitialized(internal_obj_)) {
      if (def().external_def.object_def.object_type ==
          gpu::ObjectType::OPENGL_SSBO) {
        auto ssbo = absl::get_if<OpenGlBuffer>(&obj);
        GlBuffer buffer;
        RETURN_IF_ERROR(WrapSSBO(*ssbo, &buffer));
        RETURN_IF_ERROR(objects_->RegisterBuffer(def().id, std::move(buffer)));
      } else {
        return absl::InternalError("Unexpected object type.");
      }
    }
    return absl::OkStatus();
  }

  TensorObject GetExternalObject() final {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSglPSapi2DTcc mht_10(mht_10_v, 360, "", "./tensorflow/lite/delegates/gpu/gl/api2.cc", "GetExternalObject");
 return external_obj_; }

 private:
  bool IsSameDef() const {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSglPSapi2DTcc mht_11(mht_11_v, 366, "", "./tensorflow/lite/delegates/gpu/gl/api2.cc", "IsSameDef");

    const auto& external_def = def().external_def.object_def;
    const auto& internal_def = def().internal_def.object_def;
    return (external_def.object_type == internal_def.object_type &&
            external_def.data_type == internal_def.data_type &&
            external_def.data_layout == internal_def.data_layout) ||
           // Check for equivalent layouts that have the same size.
           (external_def.object_type == internal_def.object_type &&
            external_def.data_type == internal_def.data_type &&
            external_def.data_layout == DataLayout::BHWC &&
            internal_def.data_layout == DataLayout::DHWC4 &&
            def().external_def.dimensions.c == 4);
  }

  absl::Status Init(TensorObjectConverterBuilder* converter_builder) {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSglPSapi2DTcc mht_12(mht_12_v, 383, "", "./tensorflow/lite/delegates/gpu/gl/api2.cc", "Init");

    // First check is an object is user provided.
    const auto& external_def = def().external_def.object_def;

    const bool is_same_def = IsSameDef();

    if (!is_same_def) {
      RETURN_IF_ERROR(converter_builder->MakeConverter(
          def().internal_def, def().external_def, &converter_to_));
      RETURN_IF_ERROR(converter_builder->MakeConverter(
          def().external_def, def().internal_def, &converter_from_));
    }

    if (external_def.user_provided) {
      if (is_same_def) {
        // Entering this scope indicates that external object is used with no
        // conversion to internal one. We still need to register the stub buffer
        // in the object manager, even that the real external object is not
        // available yet. Later, when the SetExternalObject() is called, the
        // proper external object will rewrite this record. The stub value will
        // allow us to correctly prepare the runtime for the late binding of
        // this object.
        GlBuffer invalid_buffer;
        RETURN_IF_ERROR(
            objects_->RegisterBuffer(def().id, std::move(invalid_buffer)));
        return absl::OkStatus();
      }
      // Object is provided by a user, but runtime expects different object
      // type. Therefore, we have to allocate internal object and convert.
      return MaybeAllocateInternalObject();
    } else {
      RETURN_IF_ERROR(MaybeAllocateInternalObject());

      if (is_same_def) {
        // Object is NOT provided by a user, but it matches definition expected
        // by runtime. Conversion is not needed.
        external_obj_ = internal_obj_;
        return absl::OkStatus();
      }

      // Object is NOT provided by a user.
      return MaybeAllocateExternalObject();
    }
    return absl::OkStatus();
  }

  absl::Status MaybeAllocateInternalObject() {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSglPSapi2DTcc mht_13(mht_13_v, 432, "", "./tensorflow/lite/delegates/gpu/gl/api2.cc", "MaybeAllocateInternalObject");

    const TensorObjectDef& d = def().internal_def;
    if (d.object_def.user_provided) {
      return absl::OkStatus();
    }
    switch (d.object_def.object_type) {
      case gpu::ObjectType::OPENGL_SSBO: {
        GlBuffer ssbo;
        RETURN_IF_ERROR(MaybeAllocateGlBuffer(d, &ssbo));
        internal_obj_ = OpenGlBuffer{ssbo.id()};
        RETURN_IF_ERROR(objects_->RegisterBuffer(def().id, std::move(ssbo)));
        break;
      }
      // TODO(akulik): support textures as internal object when compiler permits
      default:
        return absl::InternalError("Unexpected object type");
    }
    return absl::OkStatus();
  }

  absl::Status MaybeAllocateExternalObject() {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSglPSapi2DTcc mht_14(mht_14_v, 455, "", "./tensorflow/lite/delegates/gpu/gl/api2.cc", "MaybeAllocateExternalObject");

    const TensorObjectDef& d = def().external_def;
    switch (d.object_def.object_type) {
      case gpu::ObjectType::CPU_MEMORY: {
        size_t bytes_size = NumElements(d) * SizeOf(d.object_def.data_type);
        cpu_memory_.resize(bytes_size);
        external_obj_ = CpuMemory{cpu_memory_.data(), cpu_memory_.size()};
        break;
      }
      case gpu::ObjectType::OPENGL_SSBO: {
        RETURN_IF_ERROR(MaybeAllocateGlBuffer(d, &external_ssbo_));
        external_obj_ = OpenGlBuffer{external_ssbo_.id()};
        GlBuffer bbb;
        RETURN_IF_ERROR(WrapSSBO(OpenGlBuffer{external_ssbo_.id()}, &bbb));
        break;
      }
      default:
        return absl::InternalError("Unexpected object type");
    }
    return absl::OkStatus();
  }

  ObjectManager* objects_;

  // hold references to objects.
  TensorObject internal_obj_;
  TensorObject external_obj_;

  // Hold actual objects.
  GlBuffer external_ssbo_;
  std::vector<uint8_t> cpu_memory_;

  std::unique_ptr<TensorObjectConverter> converter_to_;
  std::unique_ptr<TensorObjectConverter> converter_from_;
};

// Copies data to intermediate OpenGL buffer and then does two step conversion.
// It drives the following cases were one-step conversion is not supported:
//   - CPU BHWC -> GL buffer BHWC -> GL texture DHWC4.
class TwoStepTensorTie : public TensorTie {
 public:
  explicit TwoStepTensorTie(const TensorTieDef& def) : TensorTie(def) {
   std::vector<std::string> mht_15_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSglPSapi2DTcc mht_15(mht_15_v, 499, "", "./tensorflow/lite/delegates/gpu/gl/api2.cc", "TwoStepTensorTie");
}

  static bool IsSupported(
      const TensorTieDef& def,
      const TensorObjectConverterBuilder& converter_builder) {
   std::vector<std::string> mht_16_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSglPSapi2DTcc mht_16(mht_16_v, 506, "", "./tensorflow/lite/delegates/gpu/gl/api2.cc", "IsSupported");

    auto defs = MakeOuterInnerDefs(def);
    return DefaultTensorTie::IsSupported(defs.first, converter_builder) &&
           DefaultTensorTie::IsSupported(defs.second, converter_builder);
  }

  static absl::Status New(const TensorTieDef& def,
                          TensorObjectConverterBuilder* converter_builder,
                          ObjectManager* objects,
                          std::unique_ptr<TensorTie>* tie) {
   std::vector<std::string> mht_17_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSglPSapi2DTcc mht_17(mht_17_v, 518, "", "./tensorflow/lite/delegates/gpu/gl/api2.cc", "New");

    auto tie_impl = absl::make_unique<TwoStepTensorTie>(def);
    RETURN_IF_ERROR(tie_impl->Init(converter_builder, objects));
    *tie = std::move(tie_impl);
    return absl::OkStatus();
  }

  absl::Status CopyToExternalObject() final {
   std::vector<std::string> mht_18_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSglPSapi2DTcc mht_18(mht_18_v, 528, "", "./tensorflow/lite/delegates/gpu/gl/api2.cc", "CopyToExternalObject");

    RETURN_IF_ERROR(inner_tie_->CopyToExternalObject());
    return outer_tie_->CopyToExternalObject();
  }

  absl::Status CopyFromExternalObject() final {
   std::vector<std::string> mht_19_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSglPSapi2DTcc mht_19(mht_19_v, 536, "", "./tensorflow/lite/delegates/gpu/gl/api2.cc", "CopyFromExternalObject");

    RETURN_IF_ERROR(outer_tie_->CopyFromExternalObject());
    return inner_tie_->CopyFromExternalObject();
  }

  absl::Status SetExternalObject(TensorObject obj) final {
   std::vector<std::string> mht_20_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSglPSapi2DTcc mht_20(mht_20_v, 544, "", "./tensorflow/lite/delegates/gpu/gl/api2.cc", "SetExternalObject");

    return outer_tie_->SetExternalObject(obj);
  }

  TensorObject GetExternalObject() final {
   std::vector<std::string> mht_21_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSglPSapi2DTcc mht_21(mht_21_v, 551, "", "./tensorflow/lite/delegates/gpu/gl/api2.cc", "GetExternalObject");

    return outer_tie_->GetExternalObject();
  }

 private:
  static std::pair<TensorTieDef, TensorTieDef> MakeOuterInnerDefs(
      const TensorTieDef& def) {
    TensorTieDef outer_def;
    outer_def.external_def = def.external_def;
    outer_def.internal_def = def.external_def;
    outer_def.internal_def.object_def.object_type =
        gpu::ObjectType::OPENGL_SSBO;
    // Will not allocate new SSBO
    outer_def.internal_def.object_def.user_provided = true;

    TensorTieDef inner_def;
    inner_def.id = def.id;
    inner_def.external_def = outer_def.internal_def;
    // Should not allocate external object.
    inner_def.external_def.object_def.user_provided = false;
    // Reflects what is actually supported by compiler.
    inner_def.internal_def.dimensions = inner_def.external_def.dimensions;
    inner_def.internal_def.object_def.data_type = DataType::FLOAT32;
    inner_def.internal_def.object_def.data_layout = DataLayout::DHWC4;
    inner_def.internal_def.object_def.object_type =
        gpu::ObjectType::OPENGL_SSBO;
    // It may allocate another internal object and should register it to
    // ObjectManager.
    inner_def.internal_def.object_def.user_provided = false;
    return std::make_pair(outer_def, inner_def);
  }

  absl::Status Init(TensorObjectConverterBuilder* converter_builder,
                    ObjectManager* objects) {
   std::vector<std::string> mht_22_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSglPSapi2DTcc mht_22(mht_22_v, 587, "", "./tensorflow/lite/delegates/gpu/gl/api2.cc", "Init");

    auto defs = MakeOuterInnerDefs(def());
    RETURN_IF_ERROR(DefaultTensorTie::New(defs.second, converter_builder,
                                          objects, &inner_tie_));
    return DefaultTensorTie::New(defs.first, converter_builder,
                                 inner_tie_->GetExternalObject(), &outer_tie_);
  }

  std::unique_ptr<TensorTie> inner_tie_;
  std::unique_ptr<TensorTie> outer_tie_;
};

// Responsible for creating new tensor tie objects.
class TensorTieFactory {
 public:
  explicit TensorTieFactory(const InferenceEnvironmentOptions& env_options)
      : converter_builder_(NewConverterBuilder(env_options.queue)) {
   std::vector<std::string> mht_23_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSglPSapi2DTcc mht_23(mht_23_v, 606, "", "./tensorflow/lite/delegates/gpu/gl/api2.cc", "TensorTieFactory");
}

  bool IsSupported(const TensorTieDef& def) const {
   std::vector<std::string> mht_24_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSglPSapi2DTcc mht_24(mht_24_v, 611, "", "./tensorflow/lite/delegates/gpu/gl/api2.cc", "IsSupported");

    return IsValid(def.external_def.object_def) &&
           (DefaultTensorTie::IsSupported(def, *converter_builder_) ||
            TwoStepTensorTie::IsSupported(def, *converter_builder_));
  }

  absl::Status NewTensorTie(const TensorTieDef& def, ObjectManager* objects,
                            std::unique_ptr<TensorTie>* tie) {
   std::vector<std::string> mht_25_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSglPSapi2DTcc mht_25(mht_25_v, 621, "", "./tensorflow/lite/delegates/gpu/gl/api2.cc", "NewTensorTie");

    auto converter = converter_builder_.get();
    if (DefaultTensorTie::IsSupported(def, *converter)) {
      return DefaultTensorTie::New(def, converter, objects, tie);
    }
    if (TwoStepTensorTie::IsSupported(def, *converter)) {
      return TwoStepTensorTie::New(def, converter, objects, tie);
    }
    return absl::UnimplementedError("Unsupported tensor tie definition.");
  }

 private:
  std::unique_ptr<TensorObjectConverterBuilder> converter_builder_;
};

class InferenceRunnerImpl : public InferenceRunner {
 public:
  InferenceRunnerImpl(std::unique_ptr<Runtime> runtime,
                      std::unique_ptr<ObjectManager> objects)
      : runtime_(std::move(runtime)), external_objects_(std::move(objects)) {
   std::vector<std::string> mht_26_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSglPSapi2DTcc mht_26(mht_26_v, 643, "", "./tensorflow/lite/delegates/gpu/gl/api2.cc", "InferenceRunnerImpl");
}

  absl::Status Initialize(const std::vector<TensorTieDef>& input_defs,
                          const std::vector<TensorTieDef>& output_defs,
                          TensorTieFactory* tie_factory) {
   std::vector<std::string> mht_27_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSglPSapi2DTcc mht_27(mht_27_v, 650, "", "./tensorflow/lite/delegates/gpu/gl/api2.cc", "Initialize");

    RETURN_IF_ERROR(LinkTensors(input_defs, tie_factory, &input_tensor_ties_));
    RETURN_IF_ERROR(
        LinkTensors(output_defs, tie_factory, &output_tensor_ties_));
    for (const auto& output_def : output_defs) {
      output_to_cpu_ |= output_def.external_def.object_def.object_type ==
                        gpu::ObjectType::CPU_MEMORY;
    }
    return absl::OkStatus();
  }

  std::vector<TensorObjectDef> inputs() const override {
    return GetExternalDefinitions(input_tensor_ties_);
  }

  std::vector<TensorObjectDef> outputs() const override {
    return GetExternalDefinitions(output_tensor_ties_);
  }

  absl::Status GetInputObject(int index, TensorObject* object) override {
   std::vector<std::string> mht_28_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSglPSapi2DTcc mht_28(mht_28_v, 672, "", "./tensorflow/lite/delegates/gpu/gl/api2.cc", "GetInputObject");

    if (index < 0 || index >= input_tensor_ties_.size()) {
      return absl::OutOfRangeError("Index is out of range");
    }
    *object = input_tensor_ties_[index]->GetExternalObject();
    return absl::OkStatus();
  }

  absl::Status GetOutputObject(int index, TensorObject* object) override {
   std::vector<std::string> mht_29_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSglPSapi2DTcc mht_29(mht_29_v, 683, "", "./tensorflow/lite/delegates/gpu/gl/api2.cc", "GetOutputObject");

    if (index < 0 || index >= output_tensor_ties_.size()) {
      return absl::OutOfRangeError("Index is out of range");
    }
    *object = output_tensor_ties_[index]->GetExternalObject();
    return absl::OkStatus();
  }

  absl::Status SetInputObject(int index, TensorObject object) override {
   std::vector<std::string> mht_30_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSglPSapi2DTcc mht_30(mht_30_v, 694, "", "./tensorflow/lite/delegates/gpu/gl/api2.cc", "SetInputObject");

    if (index < 0 || index >= input_tensor_ties_.size()) {
      return absl::OutOfRangeError("Index is out of range");
    }
    return input_tensor_ties_[index]->SetExternalObject(object);
  }

  absl::Status SetOutputObject(int index, TensorObject object) override {
   std::vector<std::string> mht_31_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSglPSapi2DTcc mht_31(mht_31_v, 704, "", "./tensorflow/lite/delegates/gpu/gl/api2.cc", "SetOutputObject");

    if (index < 0 || index >= output_tensor_ties_.size()) {
      return absl::OutOfRangeError("Index is out of range");
    }
    return output_tensor_ties_[index]->SetExternalObject(object);
  }

  absl::Status Run() override {
   std::vector<std::string> mht_32_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSglPSapi2DTcc mht_32(mht_32_v, 714, "", "./tensorflow/lite/delegates/gpu/gl/api2.cc", "Run");

    for (auto& obj : input_tensor_ties_) {
      RETURN_IF_ERROR(obj->CopyFromExternalObject());
    }
    RETURN_IF_ERROR(runtime_->Execute());
    for (auto& obj : output_tensor_ties_) {
      RETURN_IF_ERROR(obj->CopyToExternalObject());
    }
    RETURN_IF_ERROR(runtime_->command_queue()->Flush());
    if (output_to_cpu_) {
      RETURN_IF_ERROR(runtime_->command_queue()->WaitForCompletion());
    }
    return absl::OkStatus();
  }

 private:
  absl::Status LinkTensors(const std::vector<TensorTieDef>& defs,
                           TensorTieFactory* tie_factory,
                           std::vector<std::unique_ptr<TensorTie>>* objects) {
   std::vector<std::string> mht_33_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSglPSapi2DTcc mht_33(mht_33_v, 735, "", "./tensorflow/lite/delegates/gpu/gl/api2.cc", "LinkTensors");

    objects->reserve(defs.size());
    for (auto& def : defs) {
      std::unique_ptr<TensorTie> object;
      RETURN_IF_ERROR(
          tie_factory->NewTensorTie(def, external_objects_.get(), &object));
      objects->push_back(std::move(object));
    }
    return absl::OkStatus();
  }

  static std::vector<TensorObjectDef> GetExternalDefinitions(
      const std::vector<std::unique_ptr<TensorTie>>& objects) {
    std::vector<TensorObjectDef> defs;
    defs.reserve(objects.size());
    for (auto& obj : objects) {
      defs.push_back(obj->def().external_def);
    }
    return defs;
  }

  std::unique_ptr<Runtime> runtime_;
  std::unique_ptr<ObjectManager> external_objects_;
  std::vector<std::unique_ptr<TensorTie>> input_tensor_ties_;
  std::vector<std::unique_ptr<TensorTie>> output_tensor_ties_;
  bool output_to_cpu_ = false;
};

class InferenceBuilderImpl : public InferenceBuilder {
 public:
  InferenceBuilderImpl(const InferenceEnvironmentOptions& env_options,
                       const InferenceOptions& options, GraphFloat32 graph,
                       const GpuInfo* gpu_info)
      : env_options_(env_options),
        options_(options),
        graph_(std::move(graph)),
        gpu_info_(gpu_info),
        tie_factory_(env_options_) {
   std::vector<std::string> mht_34_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSglPSapi2DTcc mht_34(mht_34_v, 775, "", "./tensorflow/lite/delegates/gpu/gl/api2.cc", "InferenceBuilderImpl");
}

  absl::Status Initialize() {
   std::vector<std::string> mht_35_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSglPSapi2DTcc mht_35(mht_35_v, 780, "", "./tensorflow/lite/delegates/gpu/gl/api2.cc", "Initialize");

    inputs_ = LinkTensors(graph_.inputs());
    outputs_ = LinkTensors(graph_.outputs());
    return absl::OkStatus();
  }

  std::vector<TensorObjectDef> inputs() const final {
    return GetExternalDefinitions(inputs_);
  }

  std::vector<TensorObjectDef> outputs() const final {
    return GetExternalDefinitions(outputs_);
  }

  absl::Status SetInputShape(int index, const Dimensions& dimensions) final {
   std::vector<std::string> mht_36_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSglPSapi2DTcc mht_36(mht_36_v, 797, "", "./tensorflow/lite/delegates/gpu/gl/api2.cc", "SetInputShape");

    if (index < 0 || index >= inputs_.size()) {
      return absl::OutOfRangeError("Index is out of range");
    }
    return absl::UnimplementedError("Changing input shapes is not supported");
  }

  absl::Status SetInputObjectDef(int index, ObjectDef new_def) final {
   std::vector<std::string> mht_37_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSglPSapi2DTcc mht_37(mht_37_v, 807, "", "./tensorflow/lite/delegates/gpu/gl/api2.cc", "SetInputObjectDef");

    if (index < 0 || index >= inputs_.size()) {
      return absl::OutOfRangeError("Index is out of range");
    }
    auto def = inputs_[index];
    def.external_def.object_def = new_def;
    if (!tie_factory_.IsSupported(def)) {
      return absl::InvalidArgumentError(
          "New object definition is not supported.");
    }
    inputs_[index] = def;
    return absl::OkStatus();
  }

  absl::Status SetOutputObjectDef(int index, ObjectDef new_def) final {
   std::vector<std::string> mht_38_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSglPSapi2DTcc mht_38(mht_38_v, 824, "", "./tensorflow/lite/delegates/gpu/gl/api2.cc", "SetOutputObjectDef");

    if (index < 0 || index >= outputs_.size()) {
      return absl::OutOfRangeError("Index is out of range");
    }
    auto def = outputs_[index];
    def.external_def.object_def = new_def;
    if (!tie_factory_.IsSupported(def)) {
      return absl::InvalidArgumentError(
          "New object definition is not supported.");
    }
    outputs_[index] = def;
    return absl::OkStatus();
  }

  absl::Status Build(std::unique_ptr<InferenceRunner>* runner) final {
   std::vector<std::string> mht_39_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSglPSapi2DTcc mht_39(mht_39_v, 841, "", "./tensorflow/lite/delegates/gpu/gl/api2.cc", "Build");

    auto kernels = NewNodeShaderRegistry();
    CompilationOptions compiler_options;
    compiler_options.allow_precision_loss =
        GetPosition(options_, InferencePriority::MAX_PRECISION) > 1;
    compiler_options.inline_parameters =
        options_.usage == InferenceUsage::SUSTAINED_SPEED &&
        GetPosition(options_, InferencePriority::MIN_LATENCY) == 1;
    if (GetRelativeImportance(options_, InferencePriority::MIN_MEMORY_USAGE,
                              InferencePriority::MIN_LATENCY) ==
        PriorityImportance::HIGHER) {
      // Buffers have far better memory utilization.
      compiler_options.preferred_obj_type = ObjectType::BUFFER;
      compiler_options.ref_obj_type = ObjectType::BUFFER;
    }

    auto compiler = NewCompiler(kernels.get(), gpu_info_, compiler_options);
    auto workgroup_calculator = NewDefaultWorkgroupsCalculator(*gpu_info_);
    auto external_objects = absl::make_unique<ObjectManager>();
    std::vector<GlShader> shaders;
    absl::flat_hash_map<std::string, size_t> shader_to_index;
    RuntimeOptions runtime_options;
    auto runtime =
        absl::make_unique<Runtime>(runtime_options, *gpu_info_,
                                   env_options_.queue, external_objects.get());
    Runtime* runtime_ptr = runtime.get();
    auto runner_impl = absl::make_unique<InferenceRunnerImpl>(
        std::move(runtime), std::move(external_objects));
    RETURN_IF_ERROR(runner_impl->Initialize(inputs_, outputs_, &tie_factory_));
    RETURN_IF_ERROR(
        compiler->Compile(graph_, {}, [&](ShaderCode code) -> absl::Status {
          auto workgroup = workgroup_calculator->Calculate(code);
          size_t shader_index;
          std::string shader_src =
              GetShaderHeader(workgroup) + code.source_code;
          // Check if a shader was already compiled.
          auto it = shader_to_index.find(shader_src);
          if (it == shader_to_index.end()) {
            GlShader shader;
            RETURN_IF_ERROR(GlShader::CompileShader(GL_COMPUTE_SHADER,
                                                    shader_src, &shader));
            shaders.push_back(std::move(shader));
            shader_to_index.insert({shader_src, shader_to_index.size()});
            shader_index = shader_to_index.size() - 1;
          } else {
            shader_index = it->second;
          }
          auto num_workgroups = DivideRoundUp(code.workload, workgroup);
          return runtime_ptr->AddProgram(shaders[shader_index], code.parameters,
                                         code.objects, num_workgroups);
        }));
    RETURN_IF_ERROR(runtime_ptr->PrepareForExecution());
    *runner = std::move(runner_impl);
    return absl::OkStatus();
  }

 private:
  // Links internal tensors with external user-facing objects.
  std::vector<TensorTieDef> LinkTensors(const std::vector<Value*>& values) {
    std::vector<TensorTieDef> links;
    links.reserve(values.size());
    for (const auto& value : values) {
      TensorObjectDef external_def;
      // So far the compiler always forces inputs and outputs to be in the fixed
      // format.
      const auto& shape = value->tensor.shape;
      external_def.dimensions = Dimensions(shape.b, shape.h, shape.w, shape.c);
      external_def.object_def.data_type = DataType::FLOAT32;
      external_def.object_def.data_layout = DataLayout::DHWC4;
      external_def.object_def.object_type = gpu::ObjectType::OPENGL_SSBO;

      // Internal object is not expected to be provided by user because: if
      // external and internal objects have same defs, the external object is
      // propagated and just used as an internal one; otherwise, if they have
      // different defs, internal object will be created, because it is not
      // provided by user.
      TensorObjectDef internal_def = external_def;
      external_def.object_def.user_provided = true;
      internal_def.object_def.user_provided = false;
      AccessType access =
          graph_.IsGraphInput(value->id) ? AccessType::READ : AccessType::WRITE;
      links.push_back({value->id, access, internal_def, external_def});
    }
    return links;
  }

  static std::vector<TensorObjectDef> GetExternalDefinitions(
      const std::vector<TensorTieDef>& links) {
    std::vector<TensorObjectDef> defs;
    defs.reserve(links.size());
    for (auto& desc : links) {
      defs.push_back(desc.external_def);
    }
    return defs;
  }

  const InferenceEnvironmentOptions env_options_;
  const InferenceOptions options_;
  GraphFloat32 graph_;
  const GpuInfo* gpu_info_;
  std::vector<TensorTieDef> inputs_;
  std::vector<TensorTieDef> outputs_;
  TensorTieFactory tie_factory_;
};

class InferenceEnvironmentImpl : public InferenceEnvironment {
 public:
  explicit InferenceEnvironmentImpl(const InferenceEnvironmentOptions& options)
      : env_options_(options) {
   std::vector<std::string> mht_40_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSglPSapi2DTcc mht_40(mht_40_v, 952, "", "./tensorflow/lite/delegates/gpu/gl/api2.cc", "InferenceEnvironmentImpl");
}

  absl::Status Init() {
   std::vector<std::string> mht_41_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSglPSapi2DTcc mht_41(mht_41_v, 957, "", "./tensorflow/lite/delegates/gpu/gl/api2.cc", "Init");

    RETURN_IF_ERROR(EglEnvironment::NewEglEnvironment(&egl_env_));

    RETURN_IF_ERROR(RequestGpuInfo(&gpu_info_));
    properties_.is_opengl_available = gpu_info_.IsApiOpenGl31OrAbove();
    if (!properties_.is_opengl_available) {
      return absl::InternalError(
          "OpenGL ES 3.1 or above is required to use OpenGL inference.");
    }
    if (!env_options_.queue) {
      queue_ = NewCommandQueue(gpu_info_);
      env_options_.queue = queue_.get();
    }
    return absl::OkStatus();
  }

  absl::Status NewInferenceBuilder(
      GraphFloat32&& model, const InferenceOptions& options,
      std::unique_ptr<InferenceBuilder>* builder) final {
   std::vector<std::string> mht_42_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSglPSapi2DTcc mht_42(mht_42_v, 978, "", "./tensorflow/lite/delegates/gpu/gl/api2.cc", "NewInferenceBuilder");

    if (!IsValid(options)) {
      return absl::InvalidArgumentError("InferenceOptions are invalid.");
    }
    InferenceOptions resolved_options = options;
    ResolveAutoPriority(&resolved_options);
    RETURN_IF_ERROR(CheckBatchSizeForAllValues(model));
    auto builder_impl = absl::make_unique<InferenceBuilderImpl>(
        env_options_, resolved_options, std::move(model), &gpu_info_);
    RETURN_IF_ERROR(builder_impl->Initialize());
    *builder = std::move(builder_impl);
    return absl::OkStatus();
  }

  const InferenceEnvironmentProperties& properties() const {
   std::vector<std::string> mht_43_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSglPSapi2DTcc mht_43(mht_43_v, 995, "", "./tensorflow/lite/delegates/gpu/gl/api2.cc", "properties");

    return properties_;
  }

 private:
  std::unique_ptr<EglEnvironment> egl_env_;
  std::unique_ptr<CommandQueue> queue_;
  InferenceEnvironmentOptions env_options_;
  GpuInfo gpu_info_;
  InferenceEnvironmentProperties properties_;
};

}  // namespace

absl::Status NewInferenceEnvironment(
    const InferenceEnvironmentOptions& options,
    std::unique_ptr<InferenceEnvironment>* environment,
    InferenceEnvironmentProperties* properties) {
   std::vector<std::string> mht_44_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSglPSapi2DTcc mht_44(mht_44_v, 1015, "", "./tensorflow/lite/delegates/gpu/gl/api2.cc", "NewInferenceEnvironment");

  auto env_impl = absl::make_unique<InferenceEnvironmentImpl>(options);
  absl::Status status = env_impl->Init();
  if (properties) {
    *properties = env_impl->properties();
  }
  RETURN_IF_ERROR(status);
  *environment = std::move(env_impl);
  return absl::OkStatus();
}

}  // namespace gl
}  // namespace gpu
}  // namespace tflite
