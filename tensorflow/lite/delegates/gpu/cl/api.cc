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
class MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSclPSapiDTcc {
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
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSclPSapiDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSclPSapiDTcc() {
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

#include "tensorflow/lite/delegates/gpu/cl/api.h"

#ifndef CL_DELEGATE_NO_GL
#define CL_DELEGATE_ALLOW_GL
#endif

#include <algorithm>
#include <cstring>

#include "absl/memory/memory.h"
#include "absl/types/span.h"
#include "tensorflow/lite/delegates/gpu/cl/cl_command_queue.h"
#include "tensorflow/lite/delegates/gpu/cl/cl_errors.h"
#include "tensorflow/lite/delegates/gpu/cl/cl_event.h"
#include "tensorflow/lite/delegates/gpu/cl/environment.h"
#include "tensorflow/lite/delegates/gpu/cl/inference_context.h"
#include "tensorflow/lite/delegates/gpu/cl/kernels/converter.h"
#include "tensorflow/lite/delegates/gpu/cl/opencl_wrapper.h"
#include "tensorflow/lite/delegates/gpu/cl/tensor.h"
#include "tensorflow/lite/delegates/gpu/cl/tensor_type_util.h"
#include "tensorflow/lite/delegates/gpu/common/data_type.h"
#include "tensorflow/lite/delegates/gpu/common/precision.h"
#include "tensorflow/lite/delegates/gpu/common/shape.h"
#include "tensorflow/lite/delegates/gpu/common/task/tensor_desc.h"
#include "tensorflow/lite/delegates/gpu/common/tensor.h"

#ifdef CL_DELEGATE_ALLOW_GL
#include <EGL/eglext.h>

#include "tensorflow/lite/delegates/gpu/cl/egl_sync.h"
#include "tensorflow/lite/delegates/gpu/cl/gl_interop.h"
#endif

namespace tflite {
namespace gpu {
namespace cl {
namespace {

// Both internal and external defs are identical, therefore nothing to connect
// here.
class NoopTensorTie : public TensorTie {
 public:
  NoopTensorTie(const TensorTieDef& def, TensorObject obj)
      : TensorTie(def), obj_(obj) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSclPSapiDTcc mht_0(mht_0_v, 228, "", "./tensorflow/lite/delegates/gpu/cl/api.cc", "NoopTensorTie");
}

  static bool IsSupported(const TensorTieDef& def) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSclPSapiDTcc mht_1(mht_1_v, 233, "", "./tensorflow/lite/delegates/gpu/cl/api.cc", "IsSupported");

    return def.external_def == def.internal_def;
  }

  absl::Status SetExternalObject(TensorObject obj) final {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSclPSapiDTcc mht_2(mht_2_v, 240, "", "./tensorflow/lite/delegates/gpu/cl/api.cc", "SetExternalObject");

    if (!def().external_def.object_def.user_provided) {
      return absl::InvalidArgumentError("Tensor object is readonly.");
    }
    if (!IsValid(def().external_def, obj)) {
      return absl::InvalidArgumentError("Given object is not valid");
    }
    obj_ = obj;
    return absl::OkStatus();
  }

  TensorObject GetExternalObject() final {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSclPSapiDTcc mht_3(mht_3_v, 254, "", "./tensorflow/lite/delegates/gpu/cl/api.cc", "GetExternalObject");
 return obj_; }

  absl::Status CopyToExternalObject() final {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSclPSapiDTcc mht_4(mht_4_v, 259, "", "./tensorflow/lite/delegates/gpu/cl/api.cc", "CopyToExternalObject");
 return absl::OkStatus(); }

  absl::Status CopyFromExternalObject() final {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSclPSapiDTcc mht_5(mht_5_v, 264, "", "./tensorflow/lite/delegates/gpu/cl/api.cc", "CopyFromExternalObject");
 return absl::OkStatus(); }

 private:
  TensorObject obj_;
};

// Does one-step conversion between internal and external objects.
// It may also allocate external objects if requested.
class DefaultTensorTie : public TensorTie {
 public:
  DefaultTensorTie(const TensorTieDef& def, TensorObject internal_obj)
      : TensorTie(def), internal_obj_(internal_obj) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSclPSapiDTcc mht_6(mht_6_v, 278, "", "./tensorflow/lite/delegates/gpu/cl/api.cc", "DefaultTensorTie");
}

  static bool IsSupported(
      const TensorTieDef& def,
      const TensorObjectConverterBuilder& converter_builder) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSclPSapiDTcc mht_7(mht_7_v, 285, "", "./tensorflow/lite/delegates/gpu/cl/api.cc", "IsSupported");

    auto object_type = def.external_def.object_def.object_type;
#ifdef CL_DELEGATE_ALLOW_GL
    if (def.external_def.object_def.user_provided &&
        GlClBufferCopier::IsSupported(def.external_def.object_def,
                                      def.internal_def.object_def)) {
      return true;
    }
#endif
    return (object_type == ObjectType::OPENCL_BUFFER ||
            object_type == ObjectType::OPENCL_TEXTURE ||
            object_type == ObjectType::CPU_MEMORY) &&
           converter_builder.IsSupported(def.internal_def, def.external_def) &&
           converter_builder.IsSupported(def.external_def, def.internal_def);
  }

  static absl::Status New(const TensorTieDef& def, TensorObject internal_object,
                          TensorObjectConverterBuilder* converter_builder,
                          Environment* env, std::unique_ptr<TensorTie>* tie) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSclPSapiDTcc mht_8(mht_8_v, 306, "", "./tensorflow/lite/delegates/gpu/cl/api.cc", "New");

    auto tie_impl = absl::make_unique<DefaultTensorTie>(def, internal_object);
    RETURN_IF_ERROR(tie_impl->Init(converter_builder, env));
    *tie = std::move(tie_impl);
    return absl::OkStatus();
  }

  absl::Status CopyToExternalObject() final {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSclPSapiDTcc mht_9(mht_9_v, 316, "", "./tensorflow/lite/delegates/gpu/cl/api.cc", "CopyToExternalObject");

    if (!converter_to_) {
      return absl::UnavailableError("Conversion is not available");
    }
    return converter_to_->Convert(internal_obj_, GetExternalObject());
  }

  absl::Status CopyFromExternalObject() final {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSclPSapiDTcc mht_10(mht_10_v, 326, "", "./tensorflow/lite/delegates/gpu/cl/api.cc", "CopyFromExternalObject");

    if (!converter_from_) {
      return absl::UnavailableError("Conversion is not available");
    }
    return converter_from_->Convert(GetExternalObject(), internal_obj_);
  }

  absl::Status SetExternalObject(TensorObject obj) final {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSclPSapiDTcc mht_11(mht_11_v, 336, "", "./tensorflow/lite/delegates/gpu/cl/api.cc", "SetExternalObject");

    if (!def().external_def.object_def.user_provided) {
      return absl::InvalidArgumentError("External object is read-only");
    }
    if (!IsValid(def().external_def, obj)) {
      return absl::InvalidArgumentError("Given object is not valid");
    }
    external_obj_ = obj;
    return absl::OkStatus();
  }

  TensorObject GetExternalObject() final {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSclPSapiDTcc mht_12(mht_12_v, 350, "", "./tensorflow/lite/delegates/gpu/cl/api.cc", "GetExternalObject");
 return external_obj_; }

 private:
  absl::Status Init(TensorObjectConverterBuilder* converter_builder,
                    Environment* env) {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSclPSapiDTcc mht_13(mht_13_v, 357, "", "./tensorflow/lite/delegates/gpu/cl/api.cc", "Init");

#ifdef CL_DELEGATE_ALLOW_GL
    if (def().external_def.object_def.user_provided &&
        GlClBufferCopier::IsSupported(def().external_def.object_def,
                                      def().internal_def.object_def)) {
      converter_from_ = absl::make_unique<GlClBufferCopier>(
          def().internal_def, def().external_def, env);
    } else {
      RETURN_IF_ERROR(converter_builder->MakeConverter(
          def().external_def, def().internal_def, &converter_from_));
    }
    if (def().external_def.object_def.user_provided &&
        GlClBufferCopier::IsSupported(def().internal_def.object_def,
                                      def().external_def.object_def)) {
      converter_to_ = absl::make_unique<GlClBufferCopier>(
          def().internal_def, def().external_def, env);
    } else {
      RETURN_IF_ERROR(converter_builder->MakeConverter(
          def().internal_def, def().external_def, &converter_to_));
    }
#else
    RETURN_IF_ERROR(converter_builder->MakeConverter(
        def().external_def, def().internal_def, &converter_from_));
    RETURN_IF_ERROR(converter_builder->MakeConverter(
        def().internal_def, def().external_def, &converter_to_));
#endif
    return MaybeAllocateExternalObject(env);
  }

  absl::Status MaybeAllocateExternalObject(Environment* env) {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSclPSapiDTcc mht_14(mht_14_v, 389, "", "./tensorflow/lite/delegates/gpu/cl/api.cc", "MaybeAllocateExternalObject");

    const TensorObjectDef& d = def().external_def;
    if (d.object_def.user_provided) {
      return absl::OkStatus();
    }
    switch (d.object_def.object_type) {
      case ObjectType::CPU_MEMORY: {
        size_t bytes_size = NumElements(d) * SizeOf(d.object_def.data_type);
        cpu_memory_.resize(bytes_size);
        external_obj_ = CpuMemory{cpu_memory_.data(), cpu_memory_.size()};
        break;
      }
      case ObjectType::OPENCL_TEXTURE:
      case ObjectType::OPENCL_BUFFER: {
        auto& dims = d.dimensions;
        const BHWC shape(dims.b, dims.h, dims.w, dims.c);
        const TensorDescriptor desc{
            d.object_def.data_type,
            ToTensorStorageType(d.object_def.object_type,
                                d.object_def.data_layout),
            Layout::BHWC};
        RETURN_IF_ERROR(
            AllocateTensorMemory(env->context(), shape, desc, &cl_memory_));
        if (d.object_def.object_type == ObjectType::OPENCL_TEXTURE) {
          external_obj_ = OpenClTexture{cl_memory_.memory()};
        } else {
          external_obj_ = OpenClBuffer{cl_memory_.memory()};
        }
        break;
      }
      default:
        return absl::InternalError("Unexpected object type");
    }
    return absl::OkStatus();
  }

  const TensorObject internal_obj_;
  TensorObject external_obj_;
  CLMemory cl_memory_;
  std::vector<uint8_t> cpu_memory_;
  std::unique_ptr<TensorObjectConverter> converter_to_;
  std::unique_ptr<TensorObjectConverter> converter_from_;
};

// Copies data to intermediate OpenCL buffer and then does two step conversion.
// It drives the following cases were one-step conversion is not supported:
//   - CPU BHWC -> CL buffer BHWC -> CL texture DHWC4.
class TwoStepTensorTie : public TensorTie {
 public:
  explicit TwoStepTensorTie(const TensorTieDef& def) : TensorTie(def) {
   std::vector<std::string> mht_15_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSclPSapiDTcc mht_15(mht_15_v, 441, "", "./tensorflow/lite/delegates/gpu/cl/api.cc", "TwoStepTensorTie");
}

  static bool IsSupported(
      const TensorTieDef& def,
      const TensorObjectConverterBuilder& converter_builder) {
   std::vector<std::string> mht_16_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSclPSapiDTcc mht_16(mht_16_v, 448, "", "./tensorflow/lite/delegates/gpu/cl/api.cc", "IsSupported");

    auto defs = MakeOuterInnerDefs(def);
    return DefaultTensorTie::IsSupported(defs.first, converter_builder) &&
           DefaultTensorTie::IsSupported(defs.second, converter_builder);
  }

  static absl::Status New(const TensorTieDef& def, TensorObject internal_object,
                          TensorObjectConverterBuilder* converter_builder,
                          Environment* env, std::unique_ptr<TensorTie>* tie) {
   std::vector<std::string> mht_17_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSclPSapiDTcc mht_17(mht_17_v, 459, "", "./tensorflow/lite/delegates/gpu/cl/api.cc", "New");

    auto tie_impl = absl::make_unique<TwoStepTensorTie>(def);
    RETURN_IF_ERROR(tie_impl->Init(internal_object, converter_builder, env));
    *tie = std::move(tie_impl);
    return absl::OkStatus();
  }

  absl::Status CopyToExternalObject() final {
   std::vector<std::string> mht_18_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSclPSapiDTcc mht_18(mht_18_v, 469, "", "./tensorflow/lite/delegates/gpu/cl/api.cc", "CopyToExternalObject");

    RETURN_IF_ERROR(inner_tie_->CopyToExternalObject());
    return outer_tie_->CopyToExternalObject();
  }

  absl::Status CopyFromExternalObject() final {
   std::vector<std::string> mht_19_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSclPSapiDTcc mht_19(mht_19_v, 477, "", "./tensorflow/lite/delegates/gpu/cl/api.cc", "CopyFromExternalObject");

    RETURN_IF_ERROR(outer_tie_->CopyFromExternalObject());
    return inner_tie_->CopyFromExternalObject();
  }

  absl::Status SetExternalObject(TensorObject obj) final {
   std::vector<std::string> mht_20_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSclPSapiDTcc mht_20(mht_20_v, 485, "", "./tensorflow/lite/delegates/gpu/cl/api.cc", "SetExternalObject");

    return outer_tie_->SetExternalObject(obj);
  }

  TensorObject GetExternalObject() final {
   std::vector<std::string> mht_21_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSclPSapiDTcc mht_21(mht_21_v, 492, "", "./tensorflow/lite/delegates/gpu/cl/api.cc", "GetExternalObject");

    return outer_tie_->GetExternalObject();
  }

 private:
  static std::pair<TensorTieDef, TensorTieDef> MakeOuterInnerDefs(
      const TensorTieDef& def) {
    TensorTieDef outer_def;
    outer_def.external_def = def.external_def;
    outer_def.internal_def = def.external_def;
    outer_def.internal_def.object_def.object_type = ObjectType::OPENCL_BUFFER;
    outer_def.internal_def.object_def.user_provided = true;

    TensorTieDef inner_def;
    inner_def.external_def = outer_def.internal_def;
    inner_def.external_def.object_def.user_provided = false;
    inner_def.internal_def = def.internal_def;
    return std::make_pair(outer_def, inner_def);
  }

  absl::Status Init(TensorObject internal_object,
                    TensorObjectConverterBuilder* converter_builder,
                    Environment* env) {
   std::vector<std::string> mht_22_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSclPSapiDTcc mht_22(mht_22_v, 517, "", "./tensorflow/lite/delegates/gpu/cl/api.cc", "Init");

    auto defs = MakeOuterInnerDefs(def());
    RETURN_IF_ERROR(DefaultTensorTie::New(defs.second, internal_object,
                                          converter_builder, env, &inner_tie_));
    return DefaultTensorTie::New(defs.first, inner_tie_->GetExternalObject(),
                                 converter_builder, env, &outer_tie_);
  }

  std::unique_ptr<TensorTie> inner_tie_;
  std::unique_ptr<TensorTie> outer_tie_;
};

#ifdef CL_DELEGATE_ALLOW_GL
// Captures GL object into CL context before performing a conversion.
class GlBufferHolder : public TensorTie {
 public:
  GlBufferHolder(const TensorTieDef& def, GlInteropFabric* gl_interop_fabric,
                 Environment* env)
      : TensorTie(def),
        gl_interop_fabric_(gl_interop_fabric),
        environment_(env) {
   std::vector<std::string> mht_23_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSclPSapiDTcc mht_23(mht_23_v, 540, "", "./tensorflow/lite/delegates/gpu/cl/api.cc", "GlBufferHolder");
}

  static bool IsSupported(
      const TensorTieDef& def,
      const TensorObjectConverterBuilder& converter_builder) {
   std::vector<std::string> mht_24_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSclPSapiDTcc mht_24(mht_24_v, 547, "", "./tensorflow/lite/delegates/gpu/cl/api.cc", "IsSupported");

    if (!def.external_def.object_def.user_provided ||
        def.external_def.object_def.object_type != ObjectType::OPENGL_SSBO) {
      return false;
    }
    return DefaultTensorTie::IsSupported(MakeClDef(def), converter_builder);
  }

  static absl::Status New(const TensorTieDef& def, TensorObject internal_object,
                          TensorObjectConverterBuilder* converter_builder,
                          GlInteropFabric* gl_interop_fabric, Environment* env,
                          std::unique_ptr<TensorTie>* tie) {
   std::vector<std::string> mht_25_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSclPSapiDTcc mht_25(mht_25_v, 561, "", "./tensorflow/lite/delegates/gpu/cl/api.cc", "New");

    auto tie_impl =
        absl::make_unique<GlBufferHolder>(def, gl_interop_fabric, env);
    RETURN_IF_ERROR(DefaultTensorTie::New(MakeClDef(def), internal_object,
                                          converter_builder, env,
                                          &tie_impl->tie_));
    *tie = std::move(tie_impl);
    return absl::OkStatus();
  }

  absl::Status SetExternalObject(TensorObject obj) final {
   std::vector<std::string> mht_26_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSclPSapiDTcc mht_26(mht_26_v, 574, "", "./tensorflow/lite/delegates/gpu/cl/api.cc", "SetExternalObject");

    auto ssbo = absl::get_if<OpenGlBuffer>(&obj);
    if (!ssbo) {
      return absl::InvalidArgumentError("Missing OpenGL SSBO");
    }
    auto old_ssbo = absl::get_if<OpenGlBuffer>(&external_obj_);
    if (old_ssbo && ssbo->id == old_ssbo->id) {
      return absl::OkStatus();
    }
    if (cl_object_.memory()) {
      gl_interop_fabric_->UnregisterMemory(cl_object_.memory());
    }
    RETURN_IF_ERROR(CreateClMemoryFromGlBuffer(
        ssbo->id, def().access_type, &environment_->context(), &cl_object_));
    external_obj_ = obj;
    RETURN_IF_ERROR(tie_->SetExternalObject(OpenClBuffer{cl_object_.memory()}));
    gl_interop_fabric_->RegisterMemory(cl_object_.memory());
    return absl::OkStatus();
  }

  TensorObject GetExternalObject() final {
   std::vector<std::string> mht_27_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSclPSapiDTcc mht_27(mht_27_v, 597, "", "./tensorflow/lite/delegates/gpu/cl/api.cc", "GetExternalObject");
 return external_obj_; }

  absl::Status CopyFromExternalObject() final {
   std::vector<std::string> mht_28_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSclPSapiDTcc mht_28(mht_28_v, 602, "", "./tensorflow/lite/delegates/gpu/cl/api.cc", "CopyFromExternalObject");

    return tie_->CopyFromExternalObject();
  }

  absl::Status CopyToExternalObject() final {
   std::vector<std::string> mht_29_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSclPSapiDTcc mht_29(mht_29_v, 609, "", "./tensorflow/lite/delegates/gpu/cl/api.cc", "CopyToExternalObject");

    return tie_->CopyToExternalObject();
  }

 private:
  static TensorTieDef MakeClDef(const TensorTieDef& def) {
   std::vector<std::string> mht_30_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSclPSapiDTcc mht_30(mht_30_v, 617, "", "./tensorflow/lite/delegates/gpu/cl/api.cc", "MakeClDef");

    auto cl_def = def;
    cl_def.external_def.object_def.object_type = ObjectType::OPENCL_BUFFER;
    cl_def.external_def.object_def.user_provided = true;
    return cl_def;
  }

  CLMemory cl_object_;
  GlInteropFabric* gl_interop_fabric_;
  Environment* environment_;
  std::unique_ptr<TensorTie> tie_;
  TensorObject external_obj_;
};
#endif

TensorObject TensorToObj(const Tensor& tensor) {
  if (tensor.GetStorageType() == TensorStorageType::BUFFER) {
    return OpenClBuffer{tensor.GetMemoryPtr()};
  }
  if (tensor.GetStorageType() == TensorStorageType::IMAGE_BUFFER) {
    return OpenClBuffer{tensor.GetMemoryPtrForWriting()};
  }
  return OpenClTexture{tensor.GetMemoryPtr()};
}

// Responsible for creating new tensor objects.
class TensorTieFactory {
 public:
  TensorTieFactory(Environment* env, InferenceContext* context
#ifdef CL_DELEGATE_ALLOW_GL
                   ,
                   GlInteropFabric* gl_interop_fabric
#endif
                   )
      : env_(*env),
        context_(*context),
#ifdef CL_DELEGATE_ALLOW_GL
        gl_interop_fabric_(gl_interop_fabric),
#endif
        converter_builder_(NewConverterBuilder(env)) {
   std::vector<std::string> mht_31_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSclPSapiDTcc mht_31(mht_31_v, 659, "", "./tensorflow/lite/delegates/gpu/cl/api.cc", "TensorTieFactory");

  }

  bool IsSupported(const TensorTieDef& def) const {
   std::vector<std::string> mht_32_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSclPSapiDTcc mht_32(mht_32_v, 665, "", "./tensorflow/lite/delegates/gpu/cl/api.cc", "IsSupported");

    return IsValid(def.external_def.object_def) &&
           (NoopTensorTie::IsSupported(def) ||
            DefaultTensorTie::IsSupported(def, *converter_builder_) ||
#ifdef CL_DELEGATE_ALLOW_GL
            (gl_interop_fabric_ &&
             GlBufferHolder::IsSupported(def, *converter_builder_)) ||
#endif
            TwoStepTensorTie::IsSupported(def, *converter_builder_));
  }

  absl::Status NewTensorTie(const TensorTieDef& def,
                            std::unique_ptr<TensorTie>* tie) {
   std::vector<std::string> mht_33_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSclPSapiDTcc mht_33(mht_33_v, 680, "", "./tensorflow/lite/delegates/gpu/cl/api.cc", "NewTensorTie");

    TensorObject internal_object = TensorToObj(*context_.GetTensor(def.id));
    auto converter = converter_builder_.get();
    if (NoopTensorTie::IsSupported(def)) {
      *tie = absl::make_unique<NoopTensorTie>(def, internal_object);
      return absl::OkStatus();
    }
    if (DefaultTensorTie::IsSupported(def, *converter)) {
      return DefaultTensorTie::New(def, internal_object, converter, &env_, tie);
    }
#ifdef CL_DELEGATE_ALLOW_GL
    if (gl_interop_fabric_ && GlBufferHolder::IsSupported(def, *converter)) {
      return GlBufferHolder::New(def, internal_object, converter,
                                 gl_interop_fabric_, &env_, tie);
    }
#endif
    if (TwoStepTensorTie::IsSupported(def, *converter)) {
      return TwoStepTensorTie::New(def, internal_object, converter, &env_, tie);
    }
    return absl::UnimplementedError("Unsupported tensor tie definition.");
  }

 private:
  Environment& env_;
  InferenceContext& context_;
#ifdef CL_DELEGATE_ALLOW_GL
  GlInteropFabric* gl_interop_fabric_;
#endif
  std::unique_ptr<TensorObjectConverterBuilder> converter_builder_;
};

class InferenceRunnerImpl : public CLInferenceRunner {
 public:
  InferenceRunnerImpl(Environment* environment,
                      std::unique_ptr<InferenceContext> context
#ifdef CL_DELEGATE_ALLOW_GL
                      ,
                      std::unique_ptr<GlInteropFabric> gl_interop_fabric
#endif
                      )
      : queue_(environment->queue()),
        context_(std::move(context))
#ifdef CL_DELEGATE_ALLOW_GL
        ,
        gl_interop_fabric_(std::move(gl_interop_fabric))
#endif
  {
   std::vector<std::string> mht_34_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSclPSapiDTcc mht_34(mht_34_v, 729, "", "./tensorflow/lite/delegates/gpu/cl/api.cc", "InferenceRunnerImpl");

  }

  absl::Status Initialize(const std::vector<TensorTieDef>& inputs,
                          const std::vector<TensorTieDef>& outputs,
                          TensorTieFactory* factory) {
   std::vector<std::string> mht_35_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSclPSapiDTcc mht_35(mht_35_v, 737, "", "./tensorflow/lite/delegates/gpu/cl/api.cc", "Initialize");

    RETURN_IF_ERROR(LinkTensors(inputs, factory, &inputs_));
    return LinkTensors(outputs, factory, &outputs_);
  }

  std::vector<TensorObjectDef> inputs() const override {
    return GetExternalDefinitions(inputs_);
  }

  std::vector<TensorObjectDef> outputs() const override {
    return GetExternalDefinitions(outputs_);
  }

  absl::Status GetInputObject(int index, TensorObject* object) override {
   std::vector<std::string> mht_36_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSclPSapiDTcc mht_36(mht_36_v, 753, "", "./tensorflow/lite/delegates/gpu/cl/api.cc", "GetInputObject");

    if (index < 0 || index >= inputs_.size()) {
      return absl::OutOfRangeError("Index is out of range");
    }
    *object = inputs_[index]->GetExternalObject();
    return absl::OkStatus();
  }

  absl::Status GetOutputObject(int index, TensorObject* object) override {
   std::vector<std::string> mht_37_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSclPSapiDTcc mht_37(mht_37_v, 764, "", "./tensorflow/lite/delegates/gpu/cl/api.cc", "GetOutputObject");

    if (index < 0 || index >= outputs_.size()) {
      return absl::OutOfRangeError("Index is out of range");
    }
    *object = outputs_[index]->GetExternalObject();
    return absl::OkStatus();
  }

  absl::Status SetInputObject(int index, TensorObject object) override {
   std::vector<std::string> mht_38_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSclPSapiDTcc mht_38(mht_38_v, 775, "", "./tensorflow/lite/delegates/gpu/cl/api.cc", "SetInputObject");

    if (index < 0 || index >= inputs_.size()) {
      return absl::OutOfRangeError("Input index is out of range");
    }
    return inputs_[index]->SetExternalObject(object);
  }

  absl::Status SetOutputObject(int index, TensorObject object) override {
   std::vector<std::string> mht_39_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSclPSapiDTcc mht_39(mht_39_v, 785, "", "./tensorflow/lite/delegates/gpu/cl/api.cc", "SetOutputObject");

    if (index < 0 || index >= outputs_.size()) {
      return absl::OutOfRangeError("Output index is out of range");
    }
    return outputs_[index]->SetExternalObject(object);
  }

  absl::Status CopyFromExternalInput(int index) override {
   std::vector<std::string> mht_40_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSclPSapiDTcc mht_40(mht_40_v, 795, "", "./tensorflow/lite/delegates/gpu/cl/api.cc", "CopyFromExternalInput");

    if (index > inputs_.size()) {
      return absl::NotFoundError(
          absl::StrCat("Input id ", index, " is an invalid input index."));
    }
    return inputs_[index]->CopyFromExternalObject();
  }

  absl::Status CopyToExternalOutput(int index) override {
   std::vector<std::string> mht_41_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSclPSapiDTcc mht_41(mht_41_v, 806, "", "./tensorflow/lite/delegates/gpu/cl/api.cc", "CopyToExternalOutput");

    if (index > outputs_.size()) {
      return absl::NotFoundError(
          absl::StrCat("Output id ", index, " is an invalid output index"));
    }
    return outputs_[index]->CopyToExternalObject();
  }

  absl::Status Run() override {
   std::vector<std::string> mht_42_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSclPSapiDTcc mht_42(mht_42_v, 817, "", "./tensorflow/lite/delegates/gpu/cl/api.cc", "Run");

#ifdef CL_DELEGATE_ALLOW_GL
    if (gl_interop_fabric_) {
      RETURN_IF_ERROR(gl_interop_fabric_->Start());
    }
#endif
    for (const auto& input : inputs_) {
      RETURN_IF_ERROR(input->CopyFromExternalObject());
    }

    RETURN_IF_ERROR(RunWithoutExternalBufferCopy());

    bool has_async_copies = false;
    for (const auto& output : outputs_) {
      RETURN_IF_ERROR(output->CopyToExternalObject());
      if (output->def().external_def.object_def.object_type ==
          ObjectType::CPU_MEMORY) {
        has_async_copies = true;
      }
    }
#ifdef CL_DELEGATE_ALLOW_GL
    if (gl_interop_fabric_) {
      RETURN_IF_ERROR(gl_interop_fabric_->Finish());
    }
#endif
    if (has_async_copies) {
      RETURN_IF_ERROR(queue_->WaitForCompletion());
    }
    return absl::OkStatus();
  }

  absl::Status RunWithoutExternalBufferCopy() override {
   std::vector<std::string> mht_43_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSclPSapiDTcc mht_43(mht_43_v, 851, "", "./tensorflow/lite/delegates/gpu/cl/api.cc", "RunWithoutExternalBufferCopy");

    RETURN_IF_ERROR(context_->AddToQueue(queue_));
    clFlush(queue_->queue());

    return absl::OkStatus();
  }

 private:
  static absl::Status LinkTensors(
      const std::vector<TensorTieDef>& defs, TensorTieFactory* factory,
      std::vector<std::unique_ptr<TensorTie>>* objects) {
   std::vector<std::string> mht_44_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSclPSapiDTcc mht_44(mht_44_v, 864, "", "./tensorflow/lite/delegates/gpu/cl/api.cc", "LinkTensors");

    objects->reserve(defs.size());
    for (auto& def : defs) {
      std::unique_ptr<TensorTie> object;
      RETURN_IF_ERROR(factory->NewTensorTie(def, &object));
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

  CLCommandQueue* queue_;
  std::unique_ptr<InferenceContext> context_;
#ifdef CL_DELEGATE_ALLOW_GL
  std::unique_ptr<GlInteropFabric> gl_interop_fabric_;
#endif
  std::vector<std::unique_ptr<TensorTie>> inputs_;
  std::vector<std::unique_ptr<TensorTie>> outputs_;
};

TensorObjectDef TensorToDef(const Tensor& tensor) {
   std::vector<std::string> mht_45_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSclPSapiDTcc mht_45(mht_45_v, 896, "", "./tensorflow/lite/delegates/gpu/cl/api.cc", "TensorToDef");

  TensorObjectDef def;
  def.dimensions.b = tensor.Batch();
  def.dimensions.h = tensor.Height();
  def.dimensions.w = tensor.Width();
  def.dimensions.c = tensor.Channels();
  def.object_def.data_layout = ToDataLayout(tensor.GetStorageType());
  def.object_def.data_type = tensor.GetDataType();
  def.object_def.object_type = ToObjectType(tensor.GetStorageType());
  def.object_def.user_provided = false;
  return def;
}

CalculationsPrecision GetPrecision(const Environment& env,
                                   const InferenceOptions& options) {
   std::vector<std::string> mht_46_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSclPSapiDTcc mht_46(mht_46_v, 913, "", "./tensorflow/lite/delegates/gpu/cl/api.cc", "GetPrecision");

  CalculationsPrecision precision;
  switch (GetPosition(options, InferencePriority::MAX_PRECISION)) {
    case 1:
      precision = CalculationsPrecision::F32;
      break;
    case 2:
      precision = CalculationsPrecision::F32_F16;
      break;
    case 3:
      precision = CalculationsPrecision::F16;
      break;
    default:
      precision = CalculationsPrecision::F16;
      break;
  }
  // Increase precision if lower precision is not supported.
  if (!env.IsSupported(precision)) {
    precision = CalculationsPrecision::F32_F16;
    if (!env.IsSupported(precision)) {
      precision = CalculationsPrecision::F32;
    }
  }
  return precision;
}

TensorStorageType GetStorageTypeFromOptions(const Environment& env,
                                            const InferenceOptions& options) {
   std::vector<std::string> mht_47_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSclPSapiDTcc mht_47(mht_47_v, 943, "", "./tensorflow/lite/delegates/gpu/cl/api.cc", "GetStorageTypeFromOptions");

  // Fallback to BUFFER that should be supported by default.
  std::vector<TensorStorageType> preferred_storage_types;
  if (GetRelativeImportance(options, InferencePriority::MIN_LATENCY,
                            InferencePriority::MIN_MEMORY_USAGE) ==
      PriorityImportance::HIGHER) {
    preferred_storage_types = {GetFastestStorageType(env.device().GetInfo()),
                               TensorStorageType::BUFFER};
  } else {
    preferred_storage_types = {
        GetStorageTypeWithMinimalMemoryConsumption(env.device().GetInfo()),
        TensorStorageType::BUFFER};
  }

  for (TensorStorageType storage_type : preferred_storage_types) {
    if (env.IsSupported(storage_type)) {
      return storage_type;
    }
  }
  return TensorStorageType::UNKNOWN;
}

CreateGpuModelInfo GetCreateInfo(const Environment& environment,
                                 const InferenceOptions& options) {
   std::vector<std::string> mht_48_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSclPSapiDTcc mht_48(mht_48_v, 969, "", "./tensorflow/lite/delegates/gpu/cl/api.cc", "GetCreateInfo");

  CreateGpuModelInfo create_info;
  create_info.precision = GetPrecision(environment, options);
  create_info.storage_type = GetStorageTypeFromOptions(environment, options);
  if (options.usage == InferenceUsage::FAST_SINGLE_ANSWER) {
    create_info.hints.Add(ModelHints::kReduceKernelsCount);
    create_info.hints.Add(ModelHints::kFastTuning);
  } else if (options.usage == InferenceUsage::SUSTAINED_SPEED) {
    create_info.hints.Add(ModelHints::kAllowSpecialKernels);
  }
  if (GetRelativeImportance(options, InferencePriority::MIN_MEMORY_USAGE,
                            InferencePriority::MIN_LATENCY) ==
      PriorityImportance::HIGHER) {
    create_info.hints.Add(ModelHints::kNoWinogradOptimizations);
    create_info.hints.Add(ModelHints::kReuseConvWeights);
  }
  return create_info;
}

class InferenceBuilderImpl : public InferenceBuilder {
 public:
  explicit InferenceBuilderImpl(Environment* environment)
      : environment_(environment) {
   std::vector<std::string> mht_49_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSclPSapiDTcc mht_49(mht_49_v, 994, "", "./tensorflow/lite/delegates/gpu/cl/api.cc", "InferenceBuilderImpl");
}

  absl::Status Initialize(const InferenceOptions& options,
                          const InferenceEnvironmentOptions& env_options,
                          const GraphFloat32& graph) {
   std::vector<std::string> mht_50_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSclPSapiDTcc mht_50(mht_50_v, 1001, "", "./tensorflow/lite/delegates/gpu/cl/api.cc", "Initialize");

    context_ = absl::make_unique<InferenceContext>();
    CreateGpuModelInfo create_info = GetCreateInfo(*environment_, options);
    RETURN_IF_ERROR(context_->InitFromGraph(create_info, graph, environment_));

#ifdef CL_DELEGATE_ALLOW_GL
    if (env_options.IsGlAware() &&
        IsGlSharingSupported(environment_->device())) {
      gl_interop_fabric_ = absl::make_unique<GlInteropFabric>(
          env_options.egl_display, environment_);
    }
    tie_factory_ = absl::make_unique<TensorTieFactory>(
        environment_, context_.get(), gl_interop_fabric_.get());
#else
    tie_factory_ =
        absl::make_unique<TensorTieFactory>(environment_, context_.get());
#endif

    inputs_ = LinkTensors(context_->GetInputIds(), AccessType::READ);
    outputs_ = LinkTensors(context_->GetOutputIds(), AccessType::WRITE);
    return absl::OkStatus();
  }

  absl::Status Initialize(const InferenceEnvironmentOptions& env_options,
                          const absl::Span<const uint8_t> serialized_model) {
   std::vector<std::string> mht_51_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSclPSapiDTcc mht_51(mht_51_v, 1028, "", "./tensorflow/lite/delegates/gpu/cl/api.cc", "Initialize");

    context_ = absl::make_unique<InferenceContext>();
    RETURN_IF_ERROR(
        context_->RestoreDeserialized(serialized_model, environment_));

#ifdef CL_DELEGATE_ALLOW_GL
    if (env_options.IsGlAware() &&
        IsGlSharingSupported(environment_->device())) {
      gl_interop_fabric_ = absl::make_unique<GlInteropFabric>(
          env_options.egl_display, environment_);
    }
    tie_factory_ = absl::make_unique<TensorTieFactory>(
        environment_, context_.get(), gl_interop_fabric_.get());
#else
    tie_factory_ =
        absl::make_unique<TensorTieFactory>(environment_, context_.get());
#endif

    inputs_ = LinkTensors(context_->GetInputIds(), AccessType::READ);
    outputs_ = LinkTensors(context_->GetOutputIds(), AccessType::WRITE);
    return absl::OkStatus();
  }

  std::vector<TensorObjectDef> inputs() const override {
    return GetExternalDefinitions(inputs_);
  }

  std::vector<TensorObjectDef> outputs() const override {
    return GetExternalDefinitions(outputs_);
  }

  absl::Status SetInputShape(int index, const Dimensions& dimensions) override {
   std::vector<std::string> mht_52_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSclPSapiDTcc mht_52(mht_52_v, 1062, "", "./tensorflow/lite/delegates/gpu/cl/api.cc", "SetInputShape");

    if (index < 0 || index >= inputs_.size()) {
      return absl::OutOfRangeError("Index is out of range");
    }
    return absl::UnimplementedError("Changing input shapes is not supported");
  }

  absl::Status SetInputObjectDef(int index, ObjectDef new_def) override {
   std::vector<std::string> mht_53_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSclPSapiDTcc mht_53(mht_53_v, 1072, "", "./tensorflow/lite/delegates/gpu/cl/api.cc", "SetInputObjectDef");

    if (index < 0 || index >= inputs_.size()) {
      return absl::OutOfRangeError("Input index is out of range");
    }
    auto def = inputs_[index];
    def.external_def.object_def = new_def;
    if (!tie_factory_->IsSupported(def)) {
      return absl::InvalidArgumentError(
          "New input object definition is not supported.");
    }
    inputs_[index] = def;
    return absl::OkStatus();
  }

  absl::Status SetOutputObjectDef(int index, ObjectDef new_def) override {
   std::vector<std::string> mht_54_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSclPSapiDTcc mht_54(mht_54_v, 1089, "", "./tensorflow/lite/delegates/gpu/cl/api.cc", "SetOutputObjectDef");

    if (index < 0 || index >= outputs_.size()) {
      return absl::OutOfRangeError("Output index is out of range");
    }
    auto def = outputs_[index];
    def.external_def.object_def = new_def;
    if (!tie_factory_->IsSupported(def)) {
      return absl::InvalidArgumentError(
          "New output object definition is not supported.");
    }
    outputs_[index] = def;
    return absl::OkStatus();
  }

  absl::Status Build(std::unique_ptr<InferenceRunner>* runner) override {
   std::vector<std::string> mht_55_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSclPSapiDTcc mht_55(mht_55_v, 1106, "", "./tensorflow/lite/delegates/gpu/cl/api.cc", "Build");

#ifdef CL_DELEGATE_ALLOW_GL
    if (gl_interop_fabric_ && !HasGlObjects()) {
      // destroy interop layer when there are no GL objects to avoid
      // extra synchronization cost.
      gl_interop_fabric_.reset(nullptr);
    }
    auto runner_impl = absl::make_unique<InferenceRunnerImpl>(
        environment_, std::move(context_), std::move(gl_interop_fabric_));
#else
    auto runner_impl = absl::make_unique<InferenceRunnerImpl>(
        environment_, std::move(context_));
#endif
    RETURN_IF_ERROR(
        runner_impl->Initialize(inputs_, outputs_, tie_factory_.get()));
    *runner = std::move(runner_impl);
    return absl::OkStatus();
  }

 private:
  // Links internal tensors with external user-facing objects.
  std::vector<TensorTieDef> LinkTensors(const std::vector<ValueId>& ids,
                                        AccessType access) {
    std::vector<TensorTieDef> links;
    links.reserve(ids.size());
    for (const auto& id : ids) {
      TensorObjectDef def = TensorToDef(*context_->GetTensor(id));
      links.push_back({id, access, def, def});
    }
    return links;
  }

  bool HasGlObjects() const {
   std::vector<std::string> mht_56_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSclPSapiDTcc mht_56(mht_56_v, 1141, "", "./tensorflow/lite/delegates/gpu/cl/api.cc", "HasGlObjects");

#ifdef CL_DELEGATE_ALLOW_GL
    auto is_gl = [](ObjectType t) {
   std::vector<std::string> mht_57_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSclPSapiDTcc mht_57(mht_57_v, 1146, "", "./tensorflow/lite/delegates/gpu/cl/api.cc", "lambda");

      return t == ObjectType::OPENGL_SSBO || t == ObjectType::OPENGL_TEXTURE;
    };
    for (const TensorTieDef& def : inputs_) {
      if (is_gl(def.external_def.object_def.object_type)) {
        return true;
      }
    }
    for (const TensorTieDef& def : outputs_) {
      if (is_gl(def.external_def.object_def.object_type)) {
        return true;
      }
    }
#endif
    return false;
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

  std::unique_ptr<InferenceContext> context_;
#ifdef CL_DELEGATE_ALLOW_GL
  std::unique_ptr<GlInteropFabric> gl_interop_fabric_;
#endif
  Environment* environment_;

  std::vector<TensorTieDef> inputs_;
  std::vector<TensorTieDef> outputs_;
  std::unique_ptr<TensorTieFactory> tie_factory_;
};

class InferenceEnvironmentImpl : public InferenceEnvironment {
 public:
  explicit InferenceEnvironmentImpl(const InferenceEnvironmentOptions& options)
      : options_(options) {
   std::vector<std::string> mht_58_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSclPSapiDTcc mht_58(mht_58_v, 1190, "", "./tensorflow/lite/delegates/gpu/cl/api.cc", "InferenceEnvironmentImpl");
}

  absl::Status Init() {
   std::vector<std::string> mht_59_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSclPSapiDTcc mht_59(mht_59_v, 1195, "", "./tensorflow/lite/delegates/gpu/cl/api.cc", "Init");

    RETURN_IF_ERROR(LoadOpenCL());
    properties_.is_opencl_available = true;

    CLDevice device;
    if (options_.device) {
      cl_platform_id platform;
      RETURN_IF_ERROR(GetDeviceInfo<cl_platform_id>(
          options_.device, CL_DEVICE_PLATFORM, &platform));
      device = CLDevice(options_.device, platform);
    } else {
      RETURN_IF_ERROR(CreateDefaultGPUDevice(&device));
    }

#ifdef CL_DELEGATE_ALLOW_GL
    properties_.is_gl_sharing_supported = IsGlSharingSupported(device);
    properties_.is_gl_to_cl_fast_sync_supported =
        IsClEventFromEglSyncSupported(device);
    properties_.is_cl_to_gl_fast_sync_supported =
        IsEglSyncFromClEventSupported();
#endif

    CLContext context;
    if (options_.context) {
#ifdef CL_DELEGATE_ALLOW_GL
      if (options_.IsGlAware()) {
        return absl::InvalidArgumentError(
            "OpenCL context and EGL parameters are set in the same time.");
      }
#endif
      context = CLContext(options_.context, /* has_ownership = */ false);
    } else {
#ifdef CL_DELEGATE_ALLOW_GL
      if (options_.IsGlAware() && properties_.is_gl_sharing_supported) {
        RETURN_IF_ERROR(CreateCLGLContext(
            device,
            reinterpret_cast<cl_context_properties>(options_.egl_context),
            reinterpret_cast<cl_context_properties>(options_.egl_display),
            &context));
      } else {
        RETURN_IF_ERROR(CreateCLContext(device, &context));
      }
#else
      RETURN_IF_ERROR(CreateCLContext(device, &context));
#endif
    }

    CLCommandQueue queue;
    if (options_.command_queue) {
      queue =
          CLCommandQueue(options_.command_queue, /* has_ownership = */ false);
    } else {
      RETURN_IF_ERROR(CreateCLCommandQueue(device, context, &queue));
    }
    // Profiling queue is used for workgroup size tuning.
    ProfilingCommandQueue profiling_queue;
    RETURN_IF_ERROR(
        CreateProfilingCommandQueue(device, context, &profiling_queue));
    environment_ = Environment(std::move(device), std::move(context),
                               std::move(queue), std::move(profiling_queue));
    return environment_.Init();
  }

  absl::Status BuildSerializedModel(
      const InferenceOptions& options, GraphFloat32 model,
      std::vector<uint8_t>* serialized_model) final {
   std::vector<std::string> mht_60_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSclPSapiDTcc mht_60(mht_60_v, 1263, "", "./tensorflow/lite/delegates/gpu/cl/api.cc", "BuildSerializedModel");

    if (!IsValid(options)) {
      return absl::InvalidArgumentError("InferenceOptions are invalid.");
    }
    InferenceOptions resolved_options = options;
    ResolveAutoPriority(&resolved_options);
    if (environment_.program_cache() &&
        !options_.serialized_binary_cache.empty()) {
      // Ignore returned error. Cache is discarded.
      environment_.program_cache()
          ->AddSerializedCache(environment_.context(), environment_.device(),
                               options_.serialized_binary_cache)
          .IgnoreError();
    }

    RETURN_IF_ERROR(RunGraphTransformsForGpuModel(&model));
    InferenceContext context;
    CreateGpuModelInfo create_info = GetCreateInfo(environment_, options);
    RETURN_IF_ERROR(context.InitFromGraph(create_info, model, &environment_,
                                          serialized_model));
    return absl::OkStatus();
  }

  absl::Status NewInferenceBuilder(
      const InferenceOptions& options, GraphFloat32 model,
      std::unique_ptr<InferenceBuilder>* builder) final {
   std::vector<std::string> mht_61_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSclPSapiDTcc mht_61(mht_61_v, 1291, "", "./tensorflow/lite/delegates/gpu/cl/api.cc", "NewInferenceBuilder");

    if (!IsValid(options)) {
      return absl::InvalidArgumentError("InferenceOptions are invalid.");
    }
    InferenceOptions resolved_options = options;
    ResolveAutoPriority(&resolved_options);
    if (environment_.program_cache() &&
        !options_.serialized_binary_cache.empty()) {
      // Ignore returned error. Cache is discarded.
      environment_.program_cache()
          ->AddSerializedCache(environment_.context(), environment_.device(),
                               options_.serialized_binary_cache)
          .IgnoreError();
    }

    RETURN_IF_ERROR(RunGraphTransformsForGpuModel(&model));
    auto builder_impl = absl::make_unique<InferenceBuilderImpl>(&environment_);
    RETURN_IF_ERROR(
        builder_impl->Initialize(resolved_options, options_, model));
    *builder = std::move(builder_impl);
    return absl::OkStatus();
  }

  absl::Status NewInferenceBuilder(
      const absl::Span<const uint8_t> serialized_model,
      std::unique_ptr<InferenceBuilder>* builder) final {
   std::vector<std::string> mht_62_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSclPSapiDTcc mht_62(mht_62_v, 1319, "", "./tensorflow/lite/delegates/gpu/cl/api.cc", "NewInferenceBuilder");

    if (environment_.program_cache() &&
        !options_.serialized_binary_cache.empty()) {
      // Ignore returned error. Cache is discarded.
      environment_.program_cache()
          ->AddSerializedCache(environment_.context(), environment_.device(),
                               options_.serialized_binary_cache)
          .IgnoreError();
    }

    auto builder_impl = absl::make_unique<InferenceBuilderImpl>(&environment_);
    RETURN_IF_ERROR(builder_impl->Initialize(options_, serialized_model));
    *builder = std::move(builder_impl);
    return absl::OkStatus();
  }

  std::vector<uint8_t> GetSerializedBinaryCache() const final {
    std::vector<uint8_t> data;
    // Is there was a problem, data would be empty.
    environment_.program_cache()
        ->GetSerializedCache(environment_.device(), &data)
        .IgnoreError();
    return data;
  }

  const InferenceEnvironmentProperties& properties() const {
   std::vector<std::string> mht_63_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSclPSapiDTcc mht_63(mht_63_v, 1347, "", "./tensorflow/lite/delegates/gpu/cl/api.cc", "properties");

    return properties_;
  }

 private:
  const InferenceEnvironmentOptions options_;
  Environment environment_;
  InferenceEnvironmentProperties properties_;
};

}  // namespace

absl::Status NewInferenceEnvironment(
    const InferenceEnvironmentOptions& options,
    std::unique_ptr<InferenceEnvironment>* environment,
    InferenceEnvironmentProperties* properties) {
   std::vector<std::string> mht_64_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSclPSapiDTcc mht_64(mht_64_v, 1365, "", "./tensorflow/lite/delegates/gpu/cl/api.cc", "NewInferenceEnvironment");

  auto env_impl = absl::make_unique<InferenceEnvironmentImpl>(options);
  absl::Status status = env_impl->Init();
  if (properties) {
    *properties = env_impl->properties();
  }
  RETURN_IF_ERROR(status);
  *environment = std::move(env_impl);
  return absl::OkStatus();
}

}  // namespace cl
}  // namespace gpu
}  // namespace tflite
