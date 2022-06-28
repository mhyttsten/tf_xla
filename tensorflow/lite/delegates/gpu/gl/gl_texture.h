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

#ifndef TENSORFLOW_LITE_DELEGATES_GPU_GL_GL_TEXTURE_H_
#define TENSORFLOW_LITE_DELEGATES_GPU_GL_GL_TEXTURE_H_
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
class MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSglPSgl_textureDTh {
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
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSglPSgl_textureDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSglPSgl_textureDTh() {
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


#include "absl/types/span.h"
#include "tensorflow/lite/delegates/gpu/common/data_type.h"
#include "tensorflow/lite/delegates/gpu/common/status.h"
#include "tensorflow/lite/delegates/gpu/common/tensor.h"
#include "tensorflow/lite/delegates/gpu/common/types.h"
#include "tensorflow/lite/delegates/gpu/gl/gl_call.h"
#include "tensorflow/lite/delegates/gpu/gl/portable_gl31.h"

namespace tflite {
namespace gpu {
namespace gl {

// Texture is an RAII wrapper for OpenGL texture object.
// See https://www.khronos.org/opengl/wiki/Texture for more information.
//
// Texture is moveable but not copyable.
class GlTexture {
 public:
  // Creates invalid texture.
  GlTexture()
      : GlTexture(GL_INVALID_ENUM, GL_INVALID_INDEX, GL_INVALID_ENUM, 0, 0,
                  false) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSglPSgl_textureDTh mht_0(mht_0_v, 209, "", "./tensorflow/lite/delegates/gpu/gl/gl_texture.h", "GlTexture");
}

  GlTexture(GLenum target, GLuint id, GLenum format, size_t bytes_size,
            GLint layer, bool owned)
      : id_(id),
        target_(target),
        format_(format),
        bytes_size_(bytes_size),
        layer_(layer),
        owned_(owned) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSglPSgl_textureDTh mht_1(mht_1_v, 221, "", "./tensorflow/lite/delegates/gpu/gl/gl_texture.h", "GlTexture");
}

  // Move-only
  GlTexture(GlTexture&& texture);
  GlTexture& operator=(GlTexture&& texture);
  GlTexture(const GlTexture&) = delete;
  GlTexture& operator=(const GlTexture&) = delete;

  ~GlTexture();

  // Binds a texture as an image to the given index.
  absl::Status BindAsReadonlyImage(uint32_t index) const;

  // Bind texture as an image for write access at given index.
  absl::Status BindAsWriteonlyImage(uint32_t index) const;

  // Bind texture as an image for read-write access at given index.
  absl::Status BindAsReadWriteImage(uint32_t index) const;

  // Binds a texture as a sampler to the given index.
  absl::Status BindAsSampler2D(uint32_t index) const;

  GLenum target() const {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSglPSgl_textureDTh mht_2(mht_2_v, 246, "", "./tensorflow/lite/delegates/gpu/gl/gl_texture.h", "target");
 return target_; }

  GLuint id() const {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSglPSgl_textureDTh mht_3(mht_3_v, 251, "", "./tensorflow/lite/delegates/gpu/gl/gl_texture.h", "id");
 return id_; }

  GLenum format() const {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSglPSgl_textureDTh mht_4(mht_4_v, 256, "", "./tensorflow/lite/delegates/gpu/gl/gl_texture.h", "format");
 return format_; }

  GLint layer() const {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSglPSgl_textureDTh mht_5(mht_5_v, 261, "", "./tensorflow/lite/delegates/gpu/gl/gl_texture.h", "layer");
 return layer_; }

  bool is_valid() const {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSglPSgl_textureDTh mht_6(mht_6_v, 266, "", "./tensorflow/lite/delegates/gpu/gl/gl_texture.h", "is_valid");
 return id_ != GL_INVALID_INDEX; }

  size_t bytes_size() const {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSglPSgl_textureDTh mht_7(mht_7_v, 271, "", "./tensorflow/lite/delegates/gpu/gl/gl_texture.h", "bytes_size");
 return bytes_size_; }

  // @return true if this object actually owns corresponding GL buffer
  //         and manages it's lifetime.
  bool has_ownership() const {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSglPSgl_textureDTh mht_8(mht_8_v, 278, "", "./tensorflow/lite/delegates/gpu/gl/gl_texture.h", "has_ownership");
 return owned_; }

 private:
  void Invalidate();

  absl::Status BindImage(uint32_t index, GLenum access) const;

  GLuint id_;
  GLenum target_;
  GLenum format_;
  size_t bytes_size_;
  GLint layer_;
  bool owned_;
};

// Creates new 2D image texture that will be filled with float32 data once which
// will be used for reading.
//
// @param size defines 2D image texture size where each pixel is RGBA.
absl::Status CreateReadOnlyImageTexture(const uint2& size,
                                        absl::Span<const float> data,
                                        GlTexture* gl_texture);

// Creates new 2D image texture that will be filled with float16 data once which
// will be used for reading.
//
// @param size defines 2D image texture size where each pixel is RGBA.
absl::Status CreateReadOnlyImageTextureF16(const uint2& size,
                                           absl::Span<const uint16_t> data,
                                           GlTexture* gl_texture);

// Creates new 2D image texture that will be filled with uint8 data once which
// will be used for reading.
//
// @param size defines 2D image texture size where each pixel is RGBA.
absl::Status CreateReadOnlyImageTextureU8(const uint2& size,
                                          absl::Span<const uint8_t> data,
                                          GlTexture* gl_texture);

// Creates new 3D RGBA image texture that will be filled with float32 data once
// which will be used for reading.
//
// @param size defines 3D image texture size where each pixel is RGBA.
absl::Status CreateReadOnlyImageTexture(const uint3& size,
                                        absl::Span<const float> data,
                                        GlTexture* gl_texture);

// Creates new 3D RGBA image texture that will be filled with float16 data once
// which will be used for reading.
//
// @param size defines 3D image texture size where each pixel is RGBA.
absl::Status CreateReadOnlyImageTextureF16(const uint3& size,
                                           absl::Span<const uint16_t> data,
                                           GlTexture* gl_texture);

// Creates new RGBA 2D image texture
//
// @param size defines 2D image texture size where each pixel is RGBA.
absl::Status CreateReadWriteRgbaImageTexture(DataType data_type,
                                             const uint2& size,
                                             GlTexture* gl_texture);

// Creates new RGBA 3D image texture
//
// @param size defines 3D image texture size where each pixel is RGBA.
absl::Status CreateReadWriteRgbaImageTexture(DataType data_type,
                                             const uint3& size,
                                             GlTexture* gl_texture);

namespace gl_texture_internal {

// RAII for creating and/or owning texture id.
class TextureId {
 public:
  TextureId() : id_(GL_INVALID_INDEX) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSglPSgl_textureDTh mht_9(mht_9_v, 355, "", "./tensorflow/lite/delegates/gpu/gl/gl_texture.h", "TextureId");

    TFLITE_GPU_CALL_GL(glGenTextures, 1 /* number of textures*/, &id_)
        .IgnoreError();
  }

  explicit TextureId(GLuint id) : id_(id) {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSglPSgl_textureDTh mht_10(mht_10_v, 363, "", "./tensorflow/lite/delegates/gpu/gl/gl_texture.h", "TextureId");
}

  ~TextureId() {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSglPSgl_textureDTh mht_11(mht_11_v, 368, "", "./tensorflow/lite/delegates/gpu/gl/gl_texture.h", "~TextureId");

    if (id_ != GL_INVALID_INDEX) {
      TFLITE_GPU_CALL_GL(glDeleteTextures, 1, &id_).IgnoreError();
    }
  }

  GLuint id() const {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSglPSgl_textureDTh mht_12(mht_12_v, 377, "", "./tensorflow/lite/delegates/gpu/gl/gl_texture.h", "id");
 return id_; }

  GLuint Release() {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSglPSgl_textureDTh mht_13(mht_13_v, 382, "", "./tensorflow/lite/delegates/gpu/gl/gl_texture.h", "Release");

    GLuint id = GL_INVALID_INDEX;
    std::swap(id, id_);
    return id;
  }

 private:
  GLuint id_;
};

// RAII for binding and unbinding a texture.
class TextureBinder {
 public:
  TextureBinder(GLenum target, GLuint id) : target_(target) {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSglPSgl_textureDTh mht_14(mht_14_v, 398, "", "./tensorflow/lite/delegates/gpu/gl/gl_texture.h", "TextureBinder");

    TFLITE_GPU_CALL_GL(glBindTexture, target_, id).IgnoreError();
  }

  ~TextureBinder() {
   std::vector<std::string> mht_15_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSglPSgl_textureDTh mht_15(mht_15_v, 405, "", "./tensorflow/lite/delegates/gpu/gl/gl_texture.h", "~TextureBinder");

    TFLITE_GPU_CALL_GL(glBindTexture, target_, 0).IgnoreError();
  }

 private:
  const GLenum target_;
};

}  // namespace gl_texture_internal
}  // namespace gl
}  // namespace gpu
}  // namespace tflite

#endif  // TENSORFLOW_LITE_DELEGATES_GPU_GL_GL_TEXTURE_H_
