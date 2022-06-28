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

#ifndef TENSORFLOW_EXAMPLES_ANDROID_JNI_OBJECT_TRACKING_IMAGE_DATA_H_
#define TENSORFLOW_EXAMPLES_ANDROID_JNI_OBJECT_TRACKING_IMAGE_DATA_H_
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
class MHTracer_DTPStensorflowPStoolsPSandroidPStestPSjniPSobject_trackingPSimage_dataDTh {
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
   MHTracer_DTPStensorflowPStoolsPSandroidPStestPSjniPSobject_trackingPSimage_dataDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPStoolsPSandroidPStestPSjniPSobject_trackingPSimage_dataDTh() {
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


#include <stdint.h>

#include <memory>

#include "tensorflow/tools/android/test/jni/object_tracking/config.h"
#include "tensorflow/tools/android/test/jni/object_tracking/image-inl.h"
#include "tensorflow/tools/android/test/jni/object_tracking/image.h"
#include "tensorflow/tools/android/test/jni/object_tracking/image_utils.h"
#include "tensorflow/tools/android/test/jni/object_tracking/integral_image.h"
#include "tensorflow/tools/android/test/jni/object_tracking/time_log.h"
#include "tensorflow/tools/android/test/jni/object_tracking/utils.h"

namespace tf_tracking {

// Class that encapsulates all bulky processed data for a frame.
class ImageData {
 public:
  explicit ImageData(const int width, const int height)
      : uv_frame_width_(width << 1),
        uv_frame_height_(height << 1),
        timestamp_(0),
        image_(width, height) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPStoolsPSandroidPStestPSjniPSobject_trackingPSimage_dataDTh mht_0(mht_0_v, 209, "", "./tensorflow/tools/android/test/jni/object_tracking/image_data.h", "ImageData");

    InitPyramid(width, height);
    ResetComputationCache();
  }

 private:
  void ResetComputationCache() {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPStoolsPSandroidPStestPSjniPSobject_trackingPSimage_dataDTh mht_1(mht_1_v, 218, "", "./tensorflow/tools/android/test/jni/object_tracking/image_data.h", "ResetComputationCache");

    uv_data_computed_ = false;
    integral_image_computed_ = false;
    for (int i = 0; i < kNumPyramidLevels; ++i) {
      spatial_x_computed_[i] = false;
      spatial_y_computed_[i] = false;
      pyramid_sqrt2_computed_[i * 2] = false;
      pyramid_sqrt2_computed_[i * 2 + 1] = false;
    }
  }

  void InitPyramid(const int width, const int height) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPStoolsPSandroidPStestPSjniPSobject_trackingPSimage_dataDTh mht_2(mht_2_v, 232, "", "./tensorflow/tools/android/test/jni/object_tracking/image_data.h", "InitPyramid");

    int level_width = width;
    int level_height = height;

    for (int i = 0; i < kNumPyramidLevels; ++i) {
      pyramid_sqrt2_[i * 2] = NULL;
      pyramid_sqrt2_[i * 2 + 1] = NULL;
      spatial_x_[i] = NULL;
      spatial_y_[i] = NULL;

      level_width /= 2;
      level_height /= 2;
    }

    // Alias the first pyramid level to image_.
    pyramid_sqrt2_[0] = &image_;
  }

 public:
  ~ImageData() {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPStoolsPSandroidPStestPSjniPSobject_trackingPSimage_dataDTh mht_3(mht_3_v, 254, "", "./tensorflow/tools/android/test/jni/object_tracking/image_data.h", "~ImageData");

    // The first pyramid level is actually an alias to image_,
    // so make sure it doesn't get deleted here.
    pyramid_sqrt2_[0] = NULL;

    for (int i = 0; i < kNumPyramidLevels; ++i) {
      SAFE_DELETE(pyramid_sqrt2_[i * 2]);
      SAFE_DELETE(pyramid_sqrt2_[i * 2 + 1]);
      SAFE_DELETE(spatial_x_[i]);
      SAFE_DELETE(spatial_y_[i]);
    }
  }

  void SetData(const uint8_t* const new_frame, const int stride,
               const int64_t timestamp, const int downsample_factor) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPStoolsPSandroidPStestPSjniPSobject_trackingPSimage_dataDTh mht_4(mht_4_v, 271, "", "./tensorflow/tools/android/test/jni/object_tracking/image_data.h", "SetData");

    SetData(new_frame, NULL, stride, timestamp, downsample_factor);
  }

  void SetData(const uint8_t* const new_frame, const uint8_t* const uv_frame,
               const int stride, const int64_t timestamp,
               const int downsample_factor) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPStoolsPSandroidPStestPSjniPSobject_trackingPSimage_dataDTh mht_5(mht_5_v, 280, "", "./tensorflow/tools/android/test/jni/object_tracking/image_data.h", "SetData");

    ResetComputationCache();

    timestamp_ = timestamp;

    TimeLog("SetData!");

    pyramid_sqrt2_[0]->FromArray(new_frame, stride, downsample_factor);
    pyramid_sqrt2_computed_[0] = true;
    TimeLog("Downsampled image");

    if (uv_frame != NULL) {
      if (u_data_.get() == NULL) {
        u_data_.reset(new Image<uint8_t>(uv_frame_width_, uv_frame_height_));
        v_data_.reset(new Image<uint8_t>(uv_frame_width_, uv_frame_height_));
      }

      GetUV(uv_frame, u_data_.get(), v_data_.get());
      uv_data_computed_ = true;
      TimeLog("Copied UV data");
    } else {
      LOGV("No uv data!");
    }

#ifdef LOG_TIME
    // If profiling is enabled, precompute here to make it easier to distinguish
    // total costs.
    Precompute();
#endif
  }

  inline const uint64_t GetTimestamp() const {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPStoolsPSandroidPStestPSjniPSobject_trackingPSimage_dataDTh mht_6(mht_6_v, 314, "", "./tensorflow/tools/android/test/jni/object_tracking/image_data.h", "GetTimestamp");
 return timestamp_; }

  inline const Image<uint8_t>* GetImage() const {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPStoolsPSandroidPStestPSjniPSobject_trackingPSimage_dataDTh mht_7(mht_7_v, 319, "", "./tensorflow/tools/android/test/jni/object_tracking/image_data.h", "GetImage");

    SCHECK(pyramid_sqrt2_computed_[0], "image not set!");
    return pyramid_sqrt2_[0];
  }

  const Image<uint8_t>* GetPyramidSqrt2Level(const int level) const {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPStoolsPSandroidPStestPSjniPSobject_trackingPSimage_dataDTh mht_8(mht_8_v, 327, "", "./tensorflow/tools/android/test/jni/object_tracking/image_data.h", "GetPyramidSqrt2Level");

    if (!pyramid_sqrt2_computed_[level]) {
      SCHECK(level != 0, "Level equals 0!");
      if (level == 1) {
        const Image<uint8_t>& upper_level = *GetPyramidSqrt2Level(0);
        if (pyramid_sqrt2_[level] == NULL) {
          const int new_width =
              (static_cast<int>(upper_level.GetWidth() / sqrtf(2)) + 1) / 2 * 2;
          const int new_height =
              (static_cast<int>(upper_level.GetHeight() / sqrtf(2)) + 1) / 2 *
              2;

          pyramid_sqrt2_[level] = new Image<uint8_t>(new_width, new_height);
        }
        pyramid_sqrt2_[level]->DownsampleInterpolateLinear(upper_level);
      } else {
        const Image<uint8_t>& upper_level = *GetPyramidSqrt2Level(level - 2);
        if (pyramid_sqrt2_[level] == NULL) {
          pyramid_sqrt2_[level] = new Image<uint8_t>(
              upper_level.GetWidth() / 2, upper_level.GetHeight() / 2);
        }
        pyramid_sqrt2_[level]->DownsampleAveraged(
            upper_level.data(), upper_level.stride(), 2);
      }
      pyramid_sqrt2_computed_[level] = true;
    }
    return pyramid_sqrt2_[level];
  }

  inline const Image<int32_t>* GetSpatialX(const int level) const {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPStoolsPSandroidPStestPSjniPSobject_trackingPSimage_dataDTh mht_9(mht_9_v, 359, "", "./tensorflow/tools/android/test/jni/object_tracking/image_data.h", "GetSpatialX");

    if (!spatial_x_computed_[level]) {
      const Image<uint8_t>& src = *GetPyramidSqrt2Level(level * 2);
      if (spatial_x_[level] == NULL) {
        spatial_x_[level] = new Image<int32_t>(src.GetWidth(), src.GetHeight());
      }
      spatial_x_[level]->DerivativeX(src);
      spatial_x_computed_[level] = true;
    }
    return spatial_x_[level];
  }

  inline const Image<int32_t>* GetSpatialY(const int level) const {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPStoolsPSandroidPStestPSjniPSobject_trackingPSimage_dataDTh mht_10(mht_10_v, 374, "", "./tensorflow/tools/android/test/jni/object_tracking/image_data.h", "GetSpatialY");

    if (!spatial_y_computed_[level]) {
      const Image<uint8_t>& src = *GetPyramidSqrt2Level(level * 2);
      if (spatial_y_[level] == NULL) {
        spatial_y_[level] = new Image<int32_t>(src.GetWidth(), src.GetHeight());
      }
      spatial_y_[level]->DerivativeY(src);
      spatial_y_computed_[level] = true;
    }
    return spatial_y_[level];
  }

  // The integral image is currently only used for object detection, so lazily
  // initialize it on request.
  inline const IntegralImage* GetIntegralImage() const {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPStoolsPSandroidPStestPSjniPSobject_trackingPSimage_dataDTh mht_11(mht_11_v, 391, "", "./tensorflow/tools/android/test/jni/object_tracking/image_data.h", "GetIntegralImage");

    if (integral_image_.get() == NULL) {
      integral_image_.reset(new IntegralImage(image_));
    } else if (!integral_image_computed_) {
      integral_image_->Recompute(image_);
    }
    integral_image_computed_ = true;
    return integral_image_.get();
  }

  inline const Image<uint8_t>* GetU() const {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPStoolsPSandroidPStestPSjniPSobject_trackingPSimage_dataDTh mht_12(mht_12_v, 404, "", "./tensorflow/tools/android/test/jni/object_tracking/image_data.h", "GetU");

    SCHECK(uv_data_computed_, "UV data not provided!");
    return u_data_.get();
  }

  inline const Image<uint8_t>* GetV() const {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPStoolsPSandroidPStestPSjniPSobject_trackingPSimage_dataDTh mht_13(mht_13_v, 412, "", "./tensorflow/tools/android/test/jni/object_tracking/image_data.h", "GetV");

    SCHECK(uv_data_computed_, "UV data not provided!");
    return v_data_.get();
  }

 private:
  void Precompute() {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPStoolsPSandroidPStestPSjniPSobject_trackingPSimage_dataDTh mht_14(mht_14_v, 421, "", "./tensorflow/tools/android/test/jni/object_tracking/image_data.h", "Precompute");

    // Create the smoothed pyramids.
    for (int i = 0; i < kNumPyramidLevels * 2; i += 2) {
      (void) GetPyramidSqrt2Level(i);
    }
    TimeLog("Created smoothed pyramids");

    // Create the smoothed pyramids.
    for (int i = 1; i < kNumPyramidLevels * 2; i += 2) {
      (void) GetPyramidSqrt2Level(i);
    }
    TimeLog("Created smoothed sqrt pyramids");

    // Create the spatial derivatives for frame 1.
    for (int i = 0; i < kNumPyramidLevels; ++i) {
      (void) GetSpatialX(i);
      (void) GetSpatialY(i);
    }
    TimeLog("Created spatial derivatives");

    (void) GetIntegralImage();
    TimeLog("Got integral image!");
  }

  const int uv_frame_width_;
  const int uv_frame_height_;

  int64_t timestamp_;

  Image<uint8_t> image_;

  bool uv_data_computed_;
  std::unique_ptr<Image<uint8_t> > u_data_;
  std::unique_ptr<Image<uint8_t> > v_data_;

  mutable bool spatial_x_computed_[kNumPyramidLevels];
  mutable Image<int32_t>* spatial_x_[kNumPyramidLevels];

  mutable bool spatial_y_computed_[kNumPyramidLevels];
  mutable Image<int32_t>* spatial_y_[kNumPyramidLevels];

  // Mutable so the lazy initialization can work when this class is const.
  // Whether or not the integral image has been computed for the current image.
  mutable bool integral_image_computed_;
  mutable std::unique_ptr<IntegralImage> integral_image_;

  mutable bool pyramid_sqrt2_computed_[kNumPyramidLevels * 2];
  mutable Image<uint8_t>* pyramid_sqrt2_[kNumPyramidLevels * 2];

  TF_DISALLOW_COPY_AND_ASSIGN(ImageData);
};

}  // namespace tf_tracking

#endif  // TENSORFLOW_EXAMPLES_ANDROID_JNI_OBJECT_TRACKING_IMAGE_DATA_H_
