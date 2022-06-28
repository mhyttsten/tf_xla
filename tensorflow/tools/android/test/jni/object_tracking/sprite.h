/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_EXAMPLES_ANDROID_JNI_OBJECT_TRACKING_SPRITE_H_
#define TENSORFLOW_EXAMPLES_ANDROID_JNI_OBJECT_TRACKING_SPRITE_H_
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
class MHTracer_DTPStensorflowPStoolsPSandroidPStestPSjniPSobject_trackingPSspriteDTh {
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
   MHTracer_DTPStensorflowPStoolsPSandroidPStestPSjniPSobject_trackingPSspriteDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPStoolsPSandroidPStestPSjniPSobject_trackingPSspriteDTh() {
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


#ifdef __RENDER_OPENGL__

#include <GLES/gl.h>
#include <GLES/glext.h>

#include "tensorflow/tools/android/test/jni/object_tracking/image-inl.h"
#include "tensorflow/tools/android/test/jni/object_tracking/image.h"

namespace tf_tracking {

// This class encapsulates the logic necessary to load an render image data
// at the same aspect ratio as the original source.
class Sprite {
 public:
  // Only create Sprites when you have an OpenGl context.
  explicit Sprite(const Image<uint8_t>& image) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPStoolsPSandroidPStestPSjniPSobject_trackingPSspriteDTh mht_0(mht_0_v, 203, "", "./tensorflow/tools/android/test/jni/object_tracking/sprite.h", "Sprite");
 LoadTexture(image, NULL); }

  Sprite(const Image<uint8_t>& image, const BoundingBox* const area) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPStoolsPSandroidPStestPSjniPSobject_trackingPSspriteDTh mht_1(mht_1_v, 208, "", "./tensorflow/tools/android/test/jni/object_tracking/sprite.h", "Sprite");

    LoadTexture(image, area);
  }

  // Also, try to only delete a Sprite when holding an OpenGl context.
  ~Sprite() {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPStoolsPSandroidPStestPSjniPSobject_trackingPSspriteDTh mht_2(mht_2_v, 216, "", "./tensorflow/tools/android/test/jni/object_tracking/sprite.h", "~Sprite");

    glDeleteTextures(1, &texture_);
  }

  inline int GetWidth() const {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPStoolsPSandroidPStestPSjniPSobject_trackingPSspriteDTh mht_3(mht_3_v, 223, "", "./tensorflow/tools/android/test/jni/object_tracking/sprite.h", "GetWidth");

    return actual_width_;
  }

  inline int GetHeight() const {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPStoolsPSandroidPStestPSjniPSobject_trackingPSspriteDTh mht_4(mht_4_v, 230, "", "./tensorflow/tools/android/test/jni/object_tracking/sprite.h", "GetHeight");

    return actual_height_;
  }

  // Draw the sprite at 0,0 - original width/height in the current reference
  // frame. Any transformations desired must be applied before calling this
  // function.
  void Draw() const {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPStoolsPSandroidPStestPSjniPSobject_trackingPSspriteDTh mht_5(mht_5_v, 240, "", "./tensorflow/tools/android/test/jni/object_tracking/sprite.h", "Draw");

    const float float_width = static_cast<float>(actual_width_);
    const float float_height = static_cast<float>(actual_height_);

    // Where it gets rendered to.
    const float vertices[] = { 0.0f, 0.0f, 0.0f,
                               0.0f, float_height, 0.0f,
                               float_width, 0.0f, 0.0f,
                               float_width, float_height, 0.0f,
                               };

    // The coordinates the texture gets drawn from.
    const float max_x = float_width / texture_width_;
    const float max_y = float_height / texture_height_;
    const float textureVertices[] = {
        0, 0,
        0, max_y,
        max_x, 0,
        max_x, max_y,
    };

    glEnable(GL_TEXTURE_2D);
    glBindTexture(GL_TEXTURE_2D, texture_);

    glEnableClientState(GL_VERTEX_ARRAY);
    glEnableClientState(GL_TEXTURE_COORD_ARRAY);

    glVertexPointer(3, GL_FLOAT, 0, vertices);
    glTexCoordPointer(2, GL_FLOAT, 0, textureVertices);

    glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);

    glDisableClientState(GL_VERTEX_ARRAY);
    glDisableClientState(GL_TEXTURE_COORD_ARRAY);
  }

 private:
  inline int GetNextPowerOfTwo(const int number) const {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPStoolsPSandroidPStestPSjniPSobject_trackingPSspriteDTh mht_6(mht_6_v, 280, "", "./tensorflow/tools/android/test/jni/object_tracking/sprite.h", "GetNextPowerOfTwo");

    int power_of_two = 1;
    while (power_of_two < number) {
      power_of_two *= 2;
    }
    return power_of_two;
  }

  // TODO(andrewharp): Allow sprites to have their textures reloaded.
  void LoadTexture(const Image<uint8_t>& texture_source,
                   const BoundingBox* const area) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPStoolsPSandroidPStestPSjniPSobject_trackingPSspriteDTh mht_7(mht_7_v, 293, "", "./tensorflow/tools/android/test/jni/object_tracking/sprite.h", "LoadTexture");

    glEnable(GL_TEXTURE_2D);

    glGenTextures(1, &texture_);

    glBindTexture(GL_TEXTURE_2D, texture_);

    int left = 0;
    int top = 0;

    if (area != NULL) {
      // If a sub-region was provided to pull the texture from, use that.
      left = area->left_;
      top = area->top_;
      actual_width_ = area->GetWidth();
      actual_height_ = area->GetHeight();
    } else {
      actual_width_ = texture_source.GetWidth();
      actual_height_ = texture_source.GetHeight();
    }

    // The textures must be a power of two, so find the sizes that are large
    // enough to contain the image data.
    texture_width_ = GetNextPowerOfTwo(actual_width_);
    texture_height_ = GetNextPowerOfTwo(actual_height_);

    bool allocated_data = false;
    uint8_t* texture_data;

    // Except in the lucky case where we're not using a sub-region of the
    // original image AND the source data has dimensions that are power of two,
    // care must be taken to copy data at the appropriate source and destination
    // strides so that the final block can be copied directly into texture
    // memory.
    // TODO(andrewharp): Figure out if data can be pulled directly from the
    // source image with some alignment modifications.
    if (left != 0 || top != 0 ||
        actual_width_ != texture_source.GetWidth() ||
        actual_height_ != texture_source.GetHeight()) {
      texture_data = new uint8_t[actual_width_ * actual_height_];

      for (int y = 0; y < actual_height_; ++y) {
        memcpy(texture_data + actual_width_ * y, texture_source[top + y] + left,
               actual_width_ * sizeof(uint8_t));
      }
      allocated_data = true;
    } else {
      // Cast away const-ness because for some reason glTexSubImage2D wants
      // a non-const data pointer.
      texture_data = const_cast<uint8_t*>(texture_source.data());
    }

    glTexImage2D(GL_TEXTURE_2D,
                 0,
                 GL_LUMINANCE,
                 texture_width_,
                 texture_height_,
                 0,
                 GL_LUMINANCE,
                 GL_UNSIGNED_BYTE,
                 NULL);

    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
    glTexSubImage2D(GL_TEXTURE_2D,
                    0,
                    0,
                    0,
                    actual_width_,
                    actual_height_,
                    GL_LUMINANCE,
                    GL_UNSIGNED_BYTE,
                    texture_data);

    if (allocated_data) {
      delete(texture_data);
    }

    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
  }

  // The id for the texture on the GPU.
  GLuint texture_;

  // The width and height to be used for display purposes, referring to the
  // dimensions of the original texture.
  int actual_width_;
  int actual_height_;

  // The allocated dimensions of the texture data, which must be powers of 2.
  int texture_width_;
  int texture_height_;

  TF_DISALLOW_COPY_AND_ASSIGN(Sprite);
};

}  // namespace tf_tracking

#endif  // __RENDER_OPENGL__

#endif  // TENSORFLOW_EXAMPLES_ANDROID_JNI_OBJECT_TRACKING_SPRITE_H_
