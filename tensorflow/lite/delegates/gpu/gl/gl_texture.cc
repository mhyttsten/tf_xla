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
class MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSglPSgl_textureDTcc {
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
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSglPSgl_textureDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSglPSgl_textureDTcc() {
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

#include "tensorflow/lite/delegates/gpu/gl/gl_texture.h"

#include "tensorflow/lite/delegates/gpu/common/data_type.h"
#include "tensorflow/lite/delegates/gpu/common/status.h"
#include "tensorflow/lite/delegates/gpu/common/types.h"
#include "tensorflow/lite/delegates/gpu/gl/gl_call.h"
#include "tensorflow/lite/delegates/gpu/gl/gl_errors.h"
#include "tensorflow/lite/delegates/gpu/gl/gl_texture_helper.h"

namespace tflite {
namespace gpu {
namespace gl {

GlTexture::GlTexture(GlTexture&& texture)
    : GlTexture(texture.target_, texture.id_, texture.format_,
                texture.bytes_size_, texture.layer_, texture.owned_) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSglPSgl_textureDTcc mht_0(mht_0_v, 200, "", "./tensorflow/lite/delegates/gpu/gl/gl_texture.cc", "GlTexture::GlTexture");

  texture.owned_ = false;
}

GlTexture& GlTexture::operator=(GlTexture&& texture) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSglPSgl_textureDTcc mht_1(mht_1_v, 207, "", "./tensorflow/lite/delegates/gpu/gl/gl_texture.cc", "=");

  if (this != &texture) {
    Invalidate();

    target_ = texture.target_;
    format_ = texture.format_;
    bytes_size_ = texture.bytes_size_;
    layer_ = texture.layer_;
    owned_ = texture.owned_;
    id_ = texture.id_;
    texture.owned_ = false;
  }
  return *this;
}

GlTexture::~GlTexture() {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSglPSgl_textureDTcc mht_2(mht_2_v, 225, "", "./tensorflow/lite/delegates/gpu/gl/gl_texture.cc", "GlTexture::~GlTexture");

  Invalidate();
}

void GlTexture::Invalidate() {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSglPSgl_textureDTcc mht_3(mht_3_v, 232, "", "./tensorflow/lite/delegates/gpu/gl/gl_texture.cc", "GlTexture::Invalidate");

  if (owned_ && id_ != GL_INVALID_INDEX) {
    TFLITE_GPU_CALL_GL(glDeleteTextures, 1, &id_).IgnoreError();
    id_ = GL_INVALID_INDEX;
  }
}

absl::Status GlTexture::BindImage(uint32_t index, GLenum access) const {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSglPSgl_textureDTcc mht_4(mht_4_v, 242, "", "./tensorflow/lite/delegates/gpu/gl/gl_texture.cc", "GlTexture::BindImage");

  return TFLITE_GPU_CALL_GL(glBindImageTexture, index, id_, /* level = */ 0,
                            /* layered = */ GL_TRUE, layer_, access, format_);
}

absl::Status GlTexture::BindAsReadonlyImage(uint32_t index) const {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSglPSgl_textureDTcc mht_5(mht_5_v, 250, "", "./tensorflow/lite/delegates/gpu/gl/gl_texture.cc", "GlTexture::BindAsReadonlyImage");

  return BindImage(index, GL_READ_ONLY);
}

absl::Status GlTexture::BindAsWriteonlyImage(uint32_t index) const {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSglPSgl_textureDTcc mht_6(mht_6_v, 257, "", "./tensorflow/lite/delegates/gpu/gl/gl_texture.cc", "GlTexture::BindAsWriteonlyImage");

  return BindImage(index, GL_WRITE_ONLY);
}

absl::Status GlTexture::BindAsReadWriteImage(uint32_t index) const {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSglPSgl_textureDTcc mht_7(mht_7_v, 264, "", "./tensorflow/lite/delegates/gpu/gl/gl_texture.cc", "GlTexture::BindAsReadWriteImage");

  return BindImage(index, GL_READ_WRITE);
}

absl::Status GlTexture::BindAsSampler2D(uint32_t index) const {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSglPSgl_textureDTcc mht_8(mht_8_v, 271, "", "./tensorflow/lite/delegates/gpu/gl/gl_texture.cc", "GlTexture::BindAsSampler2D");

  RETURN_IF_ERROR(TFLITE_GPU_CALL_GL(glActiveTexture, GL_TEXTURE0 + index));
  return TFLITE_GPU_CALL_GL(glBindTexture, GL_TEXTURE_2D, id_);
}

namespace {

absl::Status SetTextureWrapAndFilter(GLenum target, GLenum texture_format) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSglPSgl_textureDTcc mht_9(mht_9_v, 281, "", "./tensorflow/lite/delegates/gpu/gl/gl_texture.cc", "SetTextureWrapAndFilter");

  if (texture_format == GL_RGBA32F) {
    RETURN_IF_ERROR(TFLITE_GPU_CALL_GL(glTexParameteri, target,
                                       GL_TEXTURE_WRAP_S, GL_REPEAT));
    RETURN_IF_ERROR(TFLITE_GPU_CALL_GL(glTexParameteri, target,
                                       GL_TEXTURE_WRAP_T, GL_REPEAT));
    if (target == GL_TEXTURE_2D_ARRAY || target == GL_TEXTURE_3D) {
      RETURN_IF_ERROR(TFLITE_GPU_CALL_GL(glTexParameteri, target,
                                         GL_TEXTURE_WRAP_R, GL_REPEAT));
    }
    // Texture filtering is not available for GL_RGBA32F, hence explicitly
    // specifying GL_NEAREST param for texture (Otherwise, we can end up
    // sampling some incorrect values from texture.)
    RETURN_IF_ERROR(TFLITE_GPU_CALL_GL(glTexParameteri, target,
                                       GL_TEXTURE_MAG_FILTER, GL_NEAREST));
    RETURN_IF_ERROR(TFLITE_GPU_CALL_GL(glTexParameteri, target,
                                       GL_TEXTURE_MIN_FILTER, GL_NEAREST));
  } else if (texture_format == GL_RGBA16F) {
    RETURN_IF_ERROR(TFLITE_GPU_CALL_GL(glTexParameteri, target,
                                       GL_TEXTURE_WRAP_S, GL_REPEAT));
    RETURN_IF_ERROR(TFLITE_GPU_CALL_GL(glTexParameteri, target,
                                       GL_TEXTURE_WRAP_T, GL_REPEAT));
    if (target == GL_TEXTURE_2D_ARRAY || target == GL_TEXTURE_3D) {
      RETURN_IF_ERROR(TFLITE_GPU_CALL_GL(glTexParameteri, target,
                                         GL_TEXTURE_WRAP_R, GL_REPEAT));
    }
    // Texture filtering is available for GL_RGBA16F, specifying that
    // explicitly improves quality for some operations like texture upscaling
    RETURN_IF_ERROR(TFLITE_GPU_CALL_GL(glTexParameteri, target,
                                       GL_TEXTURE_MAG_FILTER, GL_LINEAR));
    RETURN_IF_ERROR(TFLITE_GPU_CALL_GL(glTexParameteri, target,
                                       GL_TEXTURE_MIN_FILTER, GL_LINEAR));
  }
  return absl::OkStatus();
}

absl::Status CreateReadOnlyRgba2dImageTexture(DataType data_type,
                                              const uint2& size,
                                              const void* data,
                                              size_t byte_size,
                                              GlTexture* gl_texture) {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSglPSgl_textureDTcc mht_10(mht_10_v, 324, "", "./tensorflow/lite/delegates/gpu/gl/gl_texture.cc", "CreateReadOnlyRgba2dImageTexture");

  if (byte_size != /* RGBA=*/4 * SizeOf(data_type) * size.x * size.y) {
    return absl::InvalidArgumentError(
        "Creating image texture failed. Source data size is not matching "
        "expected dimensions.");
  }
  const GLenum kTarget = GL_TEXTURE_2D;
  GLenum internal_format = ToTextureInternalFormat(data_type);
  GLenum format = ToTextureFormat(data_type);
  GLenum type = ToTextureDataType(data_type);
  gl_texture_internal::TextureId id;
  gl_texture_internal::TextureBinder binder(kTarget, id.id());
  RETURN_IF_ERROR(SetTextureWrapAndFilter(kTarget, internal_format));
  RETURN_IF_ERROR(TFLITE_GPU_CALL_GL(glTexStorage2D, kTarget,
                                     /* num_levels = */ 1, internal_format,
                                     size.x, size.y));
  RETURN_IF_ERROR(TFLITE_GPU_CALL_GL(glTexSubImage2D, kTarget, /* level = */ 0,
                                     0, 0, size.x, size.y, format, type, data));
  *gl_texture = GlTexture(kTarget, id.Release(), internal_format, byte_size, 0,
                          /*owned=*/true);
  return absl::OkStatus();
}

absl::Status CreateReadOnlyRgba3dImageTexture(DataType data_type,
                                              const uint3& size,
                                              const void* data,
                                              size_t byte_size,
                                              GlTexture* gl_texture) {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSglPSgl_textureDTcc mht_11(mht_11_v, 354, "", "./tensorflow/lite/delegates/gpu/gl/gl_texture.cc", "CreateReadOnlyRgba3dImageTexture");

  if (byte_size != /* RGBA=*/4 * SizeOf(data_type) * size.x * size.y * size.z) {
    return absl::InvalidArgumentError(
        "Creating image texture failed. Source data is larger than dimensions "
        "product.");
  }
  const GLenum kTarget = GL_TEXTURE_2D_ARRAY;
  GLenum internal_format = ToTextureInternalFormat(data_type);
  GLenum format = ToTextureFormat(data_type);
  GLenum type = ToTextureDataType(data_type);
  gl_texture_internal::TextureId id;
  gl_texture_internal::TextureBinder binder(kTarget, id.id());
  RETURN_IF_ERROR(SetTextureWrapAndFilter(kTarget, internal_format));
  RETURN_IF_ERROR(TFLITE_GPU_CALL_GL(glTexStorage3D, kTarget,
                                     /* num_levels = */ 1, internal_format,
                                     size.x, size.y, size.z));
  RETURN_IF_ERROR(TFLITE_GPU_CALL_GL(glTexSubImage3D, kTarget, /* level = */ 0,
                                     0, 0, 0, size.x, size.y, size.z, format,
                                     type, data));
  *gl_texture = GlTexture(kTarget, id.Release(), internal_format, byte_size, 0,
                          /*owned=*/true);
  return absl::OkStatus();
}

}  // namespace

absl::Status CreateReadOnlyImageTexture(const uint2& size,
                                        absl::Span<const float> data,
                                        GlTexture* gl_texture) {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSglPSgl_textureDTcc mht_12(mht_12_v, 385, "", "./tensorflow/lite/delegates/gpu/gl/gl_texture.cc", "CreateReadOnlyImageTexture");

  return CreateReadOnlyRgba2dImageTexture(DataType::FLOAT32, size, data.data(),
                                          data.size() * sizeof(float),
                                          gl_texture);
}

absl::Status CreateReadOnlyImageTexture(const uint3& size,
                                        absl::Span<const float> data,
                                        GlTexture* gl_texture) {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSglPSgl_textureDTcc mht_13(mht_13_v, 396, "", "./tensorflow/lite/delegates/gpu/gl/gl_texture.cc", "CreateReadOnlyImageTexture");

  return CreateReadOnlyRgba3dImageTexture(DataType::FLOAT32, size, data.data(),
                                          data.size() * sizeof(float),
                                          gl_texture);
}

absl::Status CreateReadOnlyImageTextureU8(const uint2& size,
                                          absl::Span<const uint8_t> data,
                                          GlTexture* gl_texture) {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSglPSgl_textureDTcc mht_14(mht_14_v, 407, "", "./tensorflow/lite/delegates/gpu/gl/gl_texture.cc", "CreateReadOnlyImageTextureU8");

  return CreateReadOnlyRgba2dImageTexture(DataType::UINT8, size, data.data(),
                                          data.size() * sizeof(uint8_t),
                                          gl_texture);
}

absl::Status CreateReadOnlyImageTextureF16(const uint2& size,
                                           absl::Span<const uint16_t> data,
                                           GlTexture* gl_texture) {
   std::vector<std::string> mht_15_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSglPSgl_textureDTcc mht_15(mht_15_v, 418, "", "./tensorflow/lite/delegates/gpu/gl/gl_texture.cc", "CreateReadOnlyImageTextureF16");

  return CreateReadOnlyRgba2dImageTexture(DataType::FLOAT16, size, data.data(),
                                          data.size() * sizeof(uint16_t),
                                          gl_texture);
}

absl::Status CreateReadOnlyImageTextureF16(const uint3& size,
                                           absl::Span<const uint16_t> data,
                                           GlTexture* gl_texture) {
   std::vector<std::string> mht_16_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSglPSgl_textureDTcc mht_16(mht_16_v, 429, "", "./tensorflow/lite/delegates/gpu/gl/gl_texture.cc", "CreateReadOnlyImageTextureF16");

  return CreateReadOnlyRgba3dImageTexture(DataType::FLOAT16, size, data.data(),
                                          data.size() * sizeof(uint16_t),
                                          gl_texture);
}

absl::Status CreateReadWriteRgbaImageTexture(DataType data_type,
                                             const uint2& size,
                                             GlTexture* gl_texture) {
   std::vector<std::string> mht_17_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSglPSgl_textureDTcc mht_17(mht_17_v, 440, "", "./tensorflow/lite/delegates/gpu/gl/gl_texture.cc", "CreateReadWriteRgbaImageTexture");

  const GLenum kTarget = GL_TEXTURE_2D;
  const GLenum internal_format = ToTextureInternalFormat(data_type);
  gl_texture_internal::TextureId id;
  gl_texture_internal::TextureBinder binder(kTarget, id.id());
  RETURN_IF_ERROR(SetTextureWrapAndFilter(kTarget, internal_format));
  RETURN_IF_ERROR(TFLITE_GPU_CALL_GL(glTexStorage2D, kTarget,
                                     /* num_levels = */ 1, internal_format,
                                     size.x, size.y));
  size_t byte_size = /* RGBA = */ 4 * SizeOf(data_type) * size.x * size.y;
  *gl_texture = GlTexture(kTarget, id.Release(), internal_format, byte_size,
                          /* layer = */ 0,
                          /* owned = */ true);
  return absl::OkStatus();
}

absl::Status CreateReadWriteRgbaImageTexture(DataType data_type,
                                             const uint3& size,
                                             GlTexture* gl_texture) {
   std::vector<std::string> mht_18_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSglPSgl_textureDTcc mht_18(mht_18_v, 461, "", "./tensorflow/lite/delegates/gpu/gl/gl_texture.cc", "CreateReadWriteRgbaImageTexture");

  const GLenum kTarget = GL_TEXTURE_2D_ARRAY;
  GLenum internal_format = ToTextureInternalFormat(data_type);
  gl_texture_internal::TextureId id;
  gl_texture_internal::TextureBinder binder(kTarget, id.id());
  RETURN_IF_ERROR(SetTextureWrapAndFilter(kTarget, internal_format));
  RETURN_IF_ERROR(TFLITE_GPU_CALL_GL(glTexStorage3D, kTarget,
                                     /* num_levels = */ 1, internal_format,
                                     size.x, size.y, size.z));
  size_t byte_size =
      /* RGBA = */ 4 * SizeOf(data_type) * size.x * size.y * size.z;
  *gl_texture = GlTexture(kTarget, id.Release(), internal_format, byte_size,
                          /* layer = */ 0,
                          /* owned = */ true);
  return absl::OkStatus();
}

}  // namespace gl
}  // namespace gpu
}  // namespace tflite
