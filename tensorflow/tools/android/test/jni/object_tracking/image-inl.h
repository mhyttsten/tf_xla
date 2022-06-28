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

#ifndef TENSORFLOW_EXAMPLES_ANDROID_JNI_OBJECT_TRACKING_IMAGE_INL_H_
#define TENSORFLOW_EXAMPLES_ANDROID_JNI_OBJECT_TRACKING_IMAGE_INL_H_
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
class MHTracer_DTPStensorflowPStoolsPSandroidPStestPSjniPSobject_trackingPSimageSLinlDTh {
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
   MHTracer_DTPStensorflowPStoolsPSandroidPStestPSjniPSobject_trackingPSimageSLinlDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPStoolsPSandroidPStestPSjniPSobject_trackingPSimageSLinlDTh() {
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

#include "tensorflow/tools/android/test/jni/object_tracking/geom.h"
#include "tensorflow/tools/android/test/jni/object_tracking/image.h"
#include "tensorflow/tools/android/test/jni/object_tracking/utils.h"

namespace tf_tracking {

template <typename T>
Image<T>::Image(const int width, const int height)
    : width_less_one_(width - 1),
      height_less_one_(height - 1),
      data_size_(width * height),
      own_data_(true),
      width_(width),
      height_(height),
      stride_(width) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPStoolsPSandroidPStestPSjniPSobject_trackingPSimageSLinlDTh mht_0(mht_0_v, 204, "", "./tensorflow/tools/android/test/jni/object_tracking/image-inl.h", "Image<T>::Image");

  Allocate();
}

template <typename T>
Image<T>::Image(const Size& size)
    : width_less_one_(size.width - 1),
      height_less_one_(size.height - 1),
      data_size_(size.width * size.height),
      own_data_(true),
      width_(size.width),
      height_(size.height),
      stride_(size.width) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPStoolsPSandroidPStestPSjniPSobject_trackingPSimageSLinlDTh mht_1(mht_1_v, 219, "", "./tensorflow/tools/android/test/jni/object_tracking/image-inl.h", "Image<T>::Image");

  Allocate();
}

// Constructor that creates an image from preallocated data.
// Note: The image takes ownership of the data lifecycle, unless own_data is
// set to false.
template <typename T>
Image<T>::Image(const int width, const int height, T* const image_data,
      const bool own_data) :
    width_less_one_(width - 1),
    height_less_one_(height - 1),
    data_size_(width * height),
    own_data_(own_data),
    width_(width),
    height_(height),
    stride_(width) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPStoolsPSandroidPStestPSjniPSobject_trackingPSimageSLinlDTh mht_2(mht_2_v, 238, "", "./tensorflow/tools/android/test/jni/object_tracking/image-inl.h", "Image<T>::Image");

  image_data_ = image_data;
  SCHECK(image_data_ != NULL, "Can't create image with NULL data!");
}

template <typename T>
Image<T>::~Image() {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPStoolsPSandroidPStestPSjniPSobject_trackingPSimageSLinlDTh mht_3(mht_3_v, 247, "", "./tensorflow/tools/android/test/jni/object_tracking/image-inl.h", "Image<T>::~Image");

  if (own_data_) {
    delete[] image_data_;
  }
  image_data_ = NULL;
}

template<typename T>
template<class DstType>
bool Image<T>::ExtractPatchAtSubpixelFixed1616(const int fp_x,
                                               const int fp_y,
                                               const int patchwidth,
                                               const int patchheight,
                                               DstType* to_data) const {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPStoolsPSandroidPStestPSjniPSobject_trackingPSimageSLinlDTh mht_4(mht_4_v, 263, "", "./tensorflow/tools/android/test/jni/object_tracking/image-inl.h", "Image<T>::ExtractPatchAtSubpixelFixed1616");

  // Calculate weights.
  const int trunc_x = fp_x >> 16;
  const int trunc_y = fp_y >> 16;

  if (trunc_x < 0 || trunc_y < 0 ||
      (trunc_x + patchwidth) >= width_less_one_ ||
      (trunc_y + patchheight) >= height_less_one_) {
    return false;
  }

  // Now walk over destination patch and fill from interpolated source image.
  for (int y = 0; y < patchheight; ++y, to_data += patchwidth) {
    for (int x = 0; x < patchwidth; ++x) {
      to_data[x] =
          static_cast<DstType>(GetPixelInterpFixed1616(fp_x + (x << 16),
                                                       fp_y + (y << 16)));
    }
  }

  return true;
}

template <typename T>
Image<T>* Image<T>::Crop(
    const int left, const int top, const int right, const int bottom) const {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPStoolsPSandroidPStestPSjniPSobject_trackingPSimageSLinlDTh mht_5(mht_5_v, 291, "", "./tensorflow/tools/android/test/jni/object_tracking/image-inl.h", "Image<T>::Crop");

  SCHECK(left >= 0 && left < width_, "out of bounds at %d!", left);
  SCHECK(right >= 0 && right < width_, "out of bounds at %d!", right);
  SCHECK(top >= 0 && top < height_, "out of bounds at %d!", top);
  SCHECK(bottom >= 0 && bottom < height_, "out of bounds at %d!", bottom);

  SCHECK(left <= right, "mismatch!");
  SCHECK(top <= bottom, "mismatch!");

  const int new_width = right - left + 1;
  const int new_height = bottom - top + 1;

  Image<T>* const cropped_image = new Image(new_width, new_height);

  for (int y = 0; y < new_height; ++y) {
    memcpy((*cropped_image)[y], ((*this)[y + top] + left),
           new_width * sizeof(T));
  }

  return cropped_image;
}

template <typename T>
inline float Image<T>::GetPixelInterp(const float x, const float y) const {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPStoolsPSandroidPStestPSjniPSobject_trackingPSimageSLinlDTh mht_6(mht_6_v, 317, "", "./tensorflow/tools/android/test/jni/object_tracking/image-inl.h", "Image<T>::GetPixelInterp");

  // Do int conversion one time.
  const int floored_x = static_cast<int>(x);
  const int floored_y = static_cast<int>(y);

  // Note: it might be the case that the *_[min|max] values are clipped, and
  // these (the a b c d vals) aren't (for speed purposes), but that doesn't
  // matter. We'll just be blending the pixel with itself in that case anyway.
  const float b = x - floored_x;
  const float a = 1.0f - b;

  const float d = y - floored_y;
  const float c = 1.0f - d;

  SCHECK(ValidInterpPixel(x, y),
        "x or y out of bounds! %.2f [0 - %d), %.2f [0 - %d)",
        x, width_less_one_, y, height_less_one_);

  const T* const pix_ptr = (*this)[floored_y] + floored_x;

  // Get the pixel values surrounding this point.
  const T& p1 = pix_ptr[0];
  const T& p2 = pix_ptr[1];
  const T& p3 = pix_ptr[width_];
  const T& p4 = pix_ptr[width_ + 1];

  // Simple bilinear interpolation between four reference pixels.
  // If x is the value requested:
  //     a  b
  //   -------
  // c |p1 p2|
  //   |  x  |
  // d |p3 p4|
  //   -------
  return  c * ((a * p1) + (b * p2)) +
          d * ((a * p3) + (b * p4));
}


template <typename T>
inline T Image<T>::GetPixelInterpFixed1616(
    const int fp_x_whole, const int fp_y_whole) const {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPStoolsPSandroidPStestPSjniPSobject_trackingPSimageSLinlDTh mht_7(mht_7_v, 361, "", "./tensorflow/tools/android/test/jni/object_tracking/image-inl.h", "Image<T>::GetPixelInterpFixed1616");

  static const int kFixedPointOne = 0x00010000;
  static const int kFixedPointHalf = 0x00008000;
  static const int kFixedPointTruncateMask = 0xFFFF0000;

  int trunc_x = fp_x_whole & kFixedPointTruncateMask;
  int trunc_y = fp_y_whole & kFixedPointTruncateMask;
  const int fp_x = fp_x_whole - trunc_x;
  const int fp_y = fp_y_whole - trunc_y;

  // Scale the truncated values back to regular ints.
  trunc_x >>= 16;
  trunc_y >>= 16;

  const int one_minus_fp_x = kFixedPointOne - fp_x;
  const int one_minus_fp_y = kFixedPointOne - fp_y;

  const T* trunc_start = (*this)[trunc_y] + trunc_x;

  const T a = trunc_start[0];
  const T b = trunc_start[1];
  const T c = trunc_start[stride_];
  const T d = trunc_start[stride_ + 1];

  return (
      (one_minus_fp_y * static_cast<int64_t>(one_minus_fp_x * a + fp_x * b) +
       fp_y * static_cast<int64_t>(one_minus_fp_x * c + fp_x * d) +
       kFixedPointHalf) >>
      32);
}

template <typename T>
inline bool Image<T>::ValidPixel(const int x, const int y) const {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPStoolsPSandroidPStestPSjniPSobject_trackingPSimageSLinlDTh mht_8(mht_8_v, 396, "", "./tensorflow/tools/android/test/jni/object_tracking/image-inl.h", "Image<T>::ValidPixel");

  return InRange(x, ZERO, width_less_one_) &&
         InRange(y, ZERO, height_less_one_);
}

template <typename T>
inline BoundingBox Image<T>::GetContainingBox() const {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPStoolsPSandroidPStestPSjniPSobject_trackingPSimageSLinlDTh mht_9(mht_9_v, 405, "", "./tensorflow/tools/android/test/jni/object_tracking/image-inl.h", "Image<T>::GetContainingBox");

  return BoundingBox(
      0, 0, width_less_one_ - EPSILON, height_less_one_ - EPSILON);
}

template <typename T>
inline bool Image<T>::Contains(const BoundingBox& bounding_box) const {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPStoolsPSandroidPStestPSjniPSobject_trackingPSimageSLinlDTh mht_10(mht_10_v, 414, "", "./tensorflow/tools/android/test/jni/object_tracking/image-inl.h", "Image<T>::Contains");

  // TODO(andrewharp): Come up with a more elegant way of ensuring that bounds
  // are ok.
  return GetContainingBox().Contains(bounding_box);
}

template <typename T>
inline bool Image<T>::ValidInterpPixel(const float x, const float y) const {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPStoolsPSandroidPStestPSjniPSobject_trackingPSimageSLinlDTh mht_11(mht_11_v, 424, "", "./tensorflow/tools/android/test/jni/object_tracking/image-inl.h", "Image<T>::ValidInterpPixel");

  // Exclusive of max because we can be more efficient if we don't handle
  // interpolating on or past the last pixel.
  return (x >= ZERO) && (x < width_less_one_) &&
         (y >= ZERO) && (y < height_less_one_);
}

template <typename T>
void Image<T>::DownsampleAveraged(const T* const original, const int stride,
                                  const int factor) {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPStoolsPSandroidPStestPSjniPSobject_trackingPSimageSLinlDTh mht_12(mht_12_v, 436, "", "./tensorflow/tools/android/test/jni/object_tracking/image-inl.h", "Image<T>::DownsampleAveraged");

#ifdef __ARM_NEON
  if (factor == 4 || factor == 2) {
    DownsampleAveragedNeon(original, stride, factor);
    return;
  }
#endif

  // TODO(andrewharp): delete or enable this for non-uint8_t downsamples.
  const int pixels_per_block = factor * factor;

  // For every pixel in resulting image.
  for (int y = 0; y < height_; ++y) {
    const int orig_y = y * factor;
    const int y_bound = orig_y + factor;

    // Sum up the original pixels.
    for (int x = 0; x < width_; ++x) {
      const int orig_x = x * factor;
      const int x_bound = orig_x + factor;

      // Making this int32_t because type U or T might overflow.
      int32_t pixel_sum = 0;

      // Grab all the pixels that make up this pixel.
      for (int curr_y = orig_y; curr_y < y_bound; ++curr_y) {
        const T* p = original + curr_y * stride + orig_x;

        for (int curr_x = orig_x; curr_x < x_bound; ++curr_x) {
          pixel_sum += *p++;
        }
      }

      (*this)[y][x] = pixel_sum / pixels_per_block;
    }
  }
}

template <typename T>
void Image<T>::DownsampleInterpolateNearest(const Image<T>& original) {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPStoolsPSandroidPStestPSjniPSobject_trackingPSimageSLinlDTh mht_13(mht_13_v, 478, "", "./tensorflow/tools/android/test/jni/object_tracking/image-inl.h", "Image<T>::DownsampleInterpolateNearest");

  // Calculating the scaling factors based on target image size.
  const float factor_x = static_cast<float>(original.GetWidth()) /
      static_cast<float>(width_);
  const float factor_y = static_cast<float>(original.GetHeight()) /
      static_cast<float>(height_);

  // Calculating initial offset in x-axis.
  const float offset_x = 0.5f * (original.GetWidth() - width_) / width_;

  // Calculating initial offset in y-axis.
  const float offset_y = 0.5f * (original.GetHeight() - height_) / height_;

  float orig_y = offset_y;

  // For every pixel in resulting image.
  for (int y = 0; y < height_; ++y) {
    float orig_x = offset_x;

    // Finding nearest pixel on y-axis.
    const int nearest_y = static_cast<int>(orig_y + 0.5f);
    const T* row_data = original[nearest_y];

    T* pixel_ptr = (*this)[y];

    for (int x = 0; x < width_; ++x) {
      // Finding nearest pixel on x-axis.
      const int nearest_x = static_cast<int>(orig_x + 0.5f);

      *pixel_ptr++ = row_data[nearest_x];

      orig_x += factor_x;
    }

    orig_y += factor_y;
  }
}

template <typename T>
void Image<T>::DownsampleInterpolateLinear(const Image<T>& original) {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPStoolsPSandroidPStestPSjniPSobject_trackingPSimageSLinlDTh mht_14(mht_14_v, 520, "", "./tensorflow/tools/android/test/jni/object_tracking/image-inl.h", "Image<T>::DownsampleInterpolateLinear");

  // TODO(andrewharp): Turn this into a general compare sizes/bulk
  // copy method.
  if (original.GetWidth() == GetWidth() &&
      original.GetHeight() == GetHeight() &&
      original.stride() == stride()) {
    memcpy(image_data_, original.data(), data_size_ * sizeof(T));
    return;
  }

  // Calculating the scaling factors based on target image size.
  const float factor_x = static_cast<float>(original.GetWidth()) /
      static_cast<float>(width_);
  const float factor_y = static_cast<float>(original.GetHeight()) /
      static_cast<float>(height_);

  // Calculating initial offset in x-axis.
  const float offset_x = 0;
  const int offset_x_fp = RealToFixed1616(offset_x);

  // Calculating initial offset in y-axis.
  const float offset_y = 0;
  const int offset_y_fp = RealToFixed1616(offset_y);

  // Get the fixed point scaling factor value.
  // Shift by 8 so we can fit everything into a 4 byte int later for speed
  // reasons. This means the precision is limited to 1 / 256th of a pixel,
  // but this should be good enough.
  const int factor_x_fp = RealToFixed1616(factor_x) >> 8;
  const int factor_y_fp = RealToFixed1616(factor_y) >> 8;

  int src_y_fp = offset_y_fp >> 8;

  static const int kFixedPointOne8 = 0x00000100;
  static const int kFixedPointHalf8 = 0x00000080;
  static const int kFixedPointTruncateMask8 = 0xFFFFFF00;

  // For every pixel in resulting image.
  for (int y = 0; y < height_; ++y) {
    int src_x_fp = offset_x_fp >> 8;

    int trunc_y = src_y_fp & kFixedPointTruncateMask8;
    const int fp_y = src_y_fp - trunc_y;

    // Scale the truncated values back to regular ints.
    trunc_y >>= 8;

    const int one_minus_fp_y = kFixedPointOne8 - fp_y;

    T* pixel_ptr = (*this)[y];

    // Make sure not to read from an invalid row.
    const int trunc_y_b = MIN(original.height_less_one_, trunc_y + 1);
    const T* other_top_ptr = original[trunc_y];
    const T* other_bot_ptr = original[trunc_y_b];

    int last_trunc_x = -1;
    int trunc_x = -1;

    T a = 0;
    T b = 0;
    T c = 0;
    T d = 0;

    for (int x = 0; x < width_; ++x) {
      trunc_x = src_x_fp & kFixedPointTruncateMask8;

      const int fp_x = (src_x_fp - trunc_x) >> 8;

      // Scale the truncated values back to regular ints.
      trunc_x >>= 8;

      // It's possible we're reading from the same pixels
      if (trunc_x != last_trunc_x) {
        // Make sure not to read from an invalid column.
        const int trunc_x_b = MIN(original.width_less_one_, trunc_x + 1);
        a = other_top_ptr[trunc_x];
        b = other_top_ptr[trunc_x_b];
        c = other_bot_ptr[trunc_x];
        d = other_bot_ptr[trunc_x_b];
        last_trunc_x = trunc_x;
      }

      const int one_minus_fp_x = kFixedPointOne8 - fp_x;

      const int32_t value =
          ((one_minus_fp_y * one_minus_fp_x * a + fp_x * b) +
           (fp_y * one_minus_fp_x * c + fp_x * d) + kFixedPointHalf8) >>
          16;

      *pixel_ptr++ = value;

      src_x_fp += factor_x_fp;
    }
    src_y_fp += factor_y_fp;
  }
}

template <typename T>
void Image<T>::DownsampleSmoothed3x3(const Image<T>& original) {
   std::vector<std::string> mht_15_v;
   MHTracer_DTPStensorflowPStoolsPSandroidPStestPSjniPSobject_trackingPSimageSLinlDTh mht_15(mht_15_v, 622, "", "./tensorflow/tools/android/test/jni/object_tracking/image-inl.h", "Image<T>::DownsampleSmoothed3x3");

  for (int y = 0; y < height_; ++y) {
    const int orig_y = Clip(2 * y, ZERO, original.height_less_one_);
    const int min_y = Clip(orig_y - 1, ZERO, original.height_less_one_);
    const int max_y = Clip(orig_y + 1, ZERO, original.height_less_one_);

    for (int x = 0; x < width_; ++x) {
      const int orig_x = Clip(2 * x, ZERO, original.width_less_one_);
      const int min_x = Clip(orig_x - 1, ZERO, original.width_less_one_);
      const int max_x = Clip(orig_x + 1, ZERO, original.width_less_one_);

      // Center.
      int32_t pixel_sum = original[orig_y][orig_x] * 4;

      // Sides.
      pixel_sum += (original[orig_y][max_x] +
                    original[orig_y][min_x] +
                    original[max_y][orig_x] +
                    original[min_y][orig_x]) * 2;

      // Diagonals.
      pixel_sum += (original[min_y][max_x] +
                    original[min_y][min_x] +
                    original[max_y][max_x] +
                    original[max_y][min_x]);

      (*this)[y][x] = pixel_sum >> 4;  // 16
    }
  }
}

template <typename T>
void Image<T>::DownsampleSmoothed5x5(const Image<T>& original) {
   std::vector<std::string> mht_16_v;
   MHTracer_DTPStensorflowPStoolsPSandroidPStestPSjniPSobject_trackingPSimageSLinlDTh mht_16(mht_16_v, 657, "", "./tensorflow/tools/android/test/jni/object_tracking/image-inl.h", "Image<T>::DownsampleSmoothed5x5");

  const int max_x = original.width_less_one_;
  const int max_y = original.height_less_one_;

  // The JY Bouget paper on Lucas-Kanade recommends a
  // [1/16 1/4 3/8 1/4 1/16]^2 filter.
  // This works out to a [1 4 6 4 1]^2 / 256 array, precomputed below.
  static const int window_radius = 2;
  static const int window_size = window_radius*2 + 1;
  static const int window_weights[] = {1,  4,  6,  4, 1,   // 16 +
                                       4, 16, 24, 16, 4,   // 64 +
                                       6, 24, 36, 24, 6,   // 96 +
                                       4, 16, 24, 16, 4,   // 64 +
                                       1,  4,  6,  4, 1};  // 16 = 256

  // We'll multiply and sum with the whole numbers first, then divide by
  // the total weight to normalize at the last moment.
  for (int y = 0; y < height_; ++y) {
    for (int x = 0; x < width_; ++x) {
      int32_t pixel_sum = 0;

      const int* w = window_weights;
      const int start_x = Clip((x << 1) - window_radius, ZERO, max_x);

      // Clip the boundaries to the size of the image.
      for (int window_y = 0; window_y < window_size; ++window_y) {
        const int start_y =
            Clip((y << 1) - window_radius + window_y, ZERO, max_y);

        const T* p = original[start_y] + start_x;

        for (int window_x = 0; window_x < window_size; ++window_x) {
          pixel_sum +=  *p++ * *w++;
        }
      }

      // Conversion to type T will happen here after shifting right 8 bits to
      // divide by 256.
      (*this)[y][x] = pixel_sum >> 8;
    }
  }
}

template <typename T>
template <typename U>
inline T Image<T>::ScharrPixelX(const Image<U>& original,
                      const int center_x, const int center_y) const {
   std::vector<std::string> mht_17_v;
   MHTracer_DTPStensorflowPStoolsPSandroidPStestPSjniPSobject_trackingPSimageSLinlDTh mht_17(mht_17_v, 706, "", "./tensorflow/tools/android/test/jni/object_tracking/image-inl.h", "Image<T>::ScharrPixelX");

  const int min_x = Clip(center_x - 1, ZERO, original.width_less_one_);
  const int max_x = Clip(center_x + 1, ZERO, original.width_less_one_);
  const int min_y = Clip(center_y - 1, ZERO, original.height_less_one_);
  const int max_y = Clip(center_y + 1, ZERO, original.height_less_one_);

  // Convolution loop unrolled for performance...
  return (3 * (original[min_y][max_x]
               + original[max_y][max_x]
               - original[min_y][min_x]
               - original[max_y][min_x])
          + 10 * (original[center_y][max_x]
                  - original[center_y][min_x])) / 32;
}

template <typename T>
template <typename U>
inline T Image<T>::ScharrPixelY(const Image<U>& original,
                      const int center_x, const int center_y) const {
   std::vector<std::string> mht_18_v;
   MHTracer_DTPStensorflowPStoolsPSandroidPStestPSjniPSobject_trackingPSimageSLinlDTh mht_18(mht_18_v, 727, "", "./tensorflow/tools/android/test/jni/object_tracking/image-inl.h", "Image<T>::ScharrPixelY");

  const int min_x = Clip(center_x - 1, 0, original.width_less_one_);
  const int max_x = Clip(center_x + 1, 0, original.width_less_one_);
  const int min_y = Clip(center_y - 1, 0, original.height_less_one_);
  const int max_y = Clip(center_y + 1, 0, original.height_less_one_);

  // Convolution loop unrolled for performance...
  return (3 * (original[max_y][min_x]
               + original[max_y][max_x]
               - original[min_y][min_x]
               - original[min_y][max_x])
          + 10 * (original[max_y][center_x]
                  - original[min_y][center_x])) / 32;
}

template <typename T>
template <typename U>
inline void Image<T>::ScharrX(const Image<U>& original) {
   std::vector<std::string> mht_19_v;
   MHTracer_DTPStensorflowPStoolsPSandroidPStestPSjniPSobject_trackingPSimageSLinlDTh mht_19(mht_19_v, 747, "", "./tensorflow/tools/android/test/jni/object_tracking/image-inl.h", "Image<T>::ScharrX");

  for (int y = 0; y < height_; ++y) {
    for (int x = 0; x < width_; ++x) {
      SetPixel(x, y, ScharrPixelX(original, x, y));
    }
  }
}

template <typename T>
template <typename U>
inline void Image<T>::ScharrY(const Image<U>& original) {
   std::vector<std::string> mht_20_v;
   MHTracer_DTPStensorflowPStoolsPSandroidPStestPSjniPSobject_trackingPSimageSLinlDTh mht_20(mht_20_v, 760, "", "./tensorflow/tools/android/test/jni/object_tracking/image-inl.h", "Image<T>::ScharrY");

  for (int y = 0; y < height_; ++y) {
    for (int x = 0; x < width_; ++x) {
      SetPixel(x, y, ScharrPixelY(original, x, y));
    }
  }
}

template <typename T>
template <typename U>
void Image<T>::DerivativeX(const Image<U>& original) {
   std::vector<std::string> mht_21_v;
   MHTracer_DTPStensorflowPStoolsPSandroidPStestPSjniPSobject_trackingPSimageSLinlDTh mht_21(mht_21_v, 773, "", "./tensorflow/tools/android/test/jni/object_tracking/image-inl.h", "Image<T>::DerivativeX");

  for (int y = 0; y < height_; ++y) {
    const U* const source_row = original[y];
    T* const dest_row = (*this)[y];

    // Compute first pixel. Approximated with forward difference.
    dest_row[0] = source_row[1] - source_row[0];

    // All the pixels in between. Central difference method.
    const U* source_prev_pixel = source_row;
    T* dest_pixel = dest_row + 1;
    const U* source_next_pixel = source_row + 2;
    for (int x = 1; x < width_less_one_; ++x) {
      *dest_pixel++ = HalfDiff(*source_prev_pixel++, *source_next_pixel++);
    }

    // Last pixel. Approximated with backward difference.
    dest_row[width_less_one_] =
        source_row[width_less_one_] - source_row[width_less_one_ - 1];
  }
}

template <typename T>
template <typename U>
void Image<T>::DerivativeY(const Image<U>& original) {
   std::vector<std::string> mht_22_v;
   MHTracer_DTPStensorflowPStoolsPSandroidPStestPSjniPSobject_trackingPSimageSLinlDTh mht_22(mht_22_v, 800, "", "./tensorflow/tools/android/test/jni/object_tracking/image-inl.h", "Image<T>::DerivativeY");

  const int src_stride = original.stride();

  // Compute 1st row. Approximated with forward difference.
  {
    const U* const src_row = original[0];
    T* dest_row = (*this)[0];
    for (int x = 0; x < width_; ++x) {
      dest_row[x] = src_row[x + src_stride] - src_row[x];
    }
  }

  // Compute all rows in between using central difference.
  for (int y = 1; y < height_less_one_; ++y) {
    T* dest_row = (*this)[y];

    const U* source_prev_pixel = original[y - 1];
    const U* source_next_pixel = original[y + 1];
    for (int x = 0; x < width_; ++x) {
      *dest_row++ = HalfDiff(*source_prev_pixel++, *source_next_pixel++);
    }
  }

  // Compute last row. Approximated with backward difference.
  {
    const U* const src_row = original[height_less_one_];
    T* dest_row = (*this)[height_less_one_];
    for (int x = 0; x < width_; ++x) {
      dest_row[x] = src_row[x] - src_row[x - src_stride];
    }
  }
}

template <typename T>
template <typename U>
inline T Image<T>::ConvolvePixel3x3(const Image<U>& original,
                                    const int* const filter,
                                    const int center_x, const int center_y,
                                    const int total) const {
   std::vector<std::string> mht_23_v;
   MHTracer_DTPStensorflowPStoolsPSandroidPStestPSjniPSobject_trackingPSimageSLinlDTh mht_23(mht_23_v, 841, "", "./tensorflow/tools/android/test/jni/object_tracking/image-inl.h", "Image<T>::ConvolvePixel3x3");

  int32_t sum = 0;
  for (int filter_y = 0; filter_y < 3; ++filter_y) {
    const int y = Clip(center_y - 1 + filter_y, 0, original.GetHeight());
    for (int filter_x = 0; filter_x < 3; ++filter_x) {
      const int x = Clip(center_x - 1 + filter_x, 0, original.GetWidth());
      sum += original[y][x] * filter[filter_y * 3 + filter_x];
    }
  }
  return sum / total;
}

template <typename T>
template <typename U>
inline void Image<T>::Convolve3x3(const Image<U>& original,
                                  const int32_t* const filter) {
   std::vector<std::string> mht_24_v;
   MHTracer_DTPStensorflowPStoolsPSandroidPStestPSjniPSobject_trackingPSimageSLinlDTh mht_24(mht_24_v, 859, "", "./tensorflow/tools/android/test/jni/object_tracking/image-inl.h", "Image<T>::Convolve3x3");

  int32_t sum = 0;
  for (int i = 0; i < 9; ++i) {
    sum += abs(filter[i]);
  }
  for (int y = 0; y < height_; ++y) {
    for (int x = 0; x < width_; ++x) {
      SetPixel(x, y, ConvolvePixel3x3(original, filter, x, y, sum));
    }
  }
}

template <typename T>
inline void Image<T>::FromArray(const T* const pixels, const int stride,
                      const int factor) {
   std::vector<std::string> mht_25_v;
   MHTracer_DTPStensorflowPStoolsPSandroidPStestPSjniPSobject_trackingPSimageSLinlDTh mht_25(mht_25_v, 876, "", "./tensorflow/tools/android/test/jni/object_tracking/image-inl.h", "Image<T>::FromArray");

  if (factor == 1 && stride == width_) {
    // If not subsampling, memcpy per line should be faster.
    memcpy(this->image_data_, pixels, data_size_ * sizeof(T));
    return;
  }

  DownsampleAveraged(pixels, stride, factor);
}

}  // namespace tf_tracking

#endif  // TENSORFLOW_EXAMPLES_ANDROID_JNI_OBJECT_TRACKING_IMAGE_INL_H_
