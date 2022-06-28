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

#ifndef TENSORFLOW_EXAMPLES_ANDROID_JNI_OBJECT_TRACKING_IMAGE_H_
#define TENSORFLOW_EXAMPLES_ANDROID_JNI_OBJECT_TRACKING_IMAGE_H_
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
class MHTracer_DTPStensorflowPStoolsPSandroidPStestPSjniPSobject_trackingPSimageDTh {
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
   MHTracer_DTPStensorflowPStoolsPSandroidPStestPSjniPSobject_trackingPSimageDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPStoolsPSandroidPStestPSjniPSobject_trackingPSimageDTh() {
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
#include "tensorflow/tools/android/test/jni/object_tracking/utils.h"

// TODO(andrewharp): Make this a cast to uint32_t if/when we go unsigned for
// operations.
#define ZERO 0

#ifdef SANITY_CHECKS
  #define CHECK_PIXEL(IMAGE, X, Y) {\
    SCHECK((IMAGE)->ValidPixel((X), (Y)), \
          "CHECK_PIXEL(%d,%d) in %dx%d image.", \
          static_cast<int>(X), static_cast<int>(Y), \
          (IMAGE)->GetWidth(), (IMAGE)->GetHeight());\
  }

  #define CHECK_PIXEL_INTERP(IMAGE, X, Y) {\
    SCHECK((IMAGE)->validInterpPixel((X), (Y)), \
          "CHECK_PIXEL_INTERP(%.2f, %.2f) in %dx%d image.", \
          static_cast<float>(X), static_cast<float>(Y), \
          (IMAGE)->GetWidth(), (IMAGE)->GetHeight());\
  }
#else
  #define CHECK_PIXEL(image, x, y) {}
  #define CHECK_PIXEL_INTERP(IMAGE, X, Y) {}
#endif

namespace tf_tracking {

#ifdef SANITY_CHECKS
// Class which exists solely to provide bounds checking for array-style image
// data access.
template <typename T>
class RowData {
 public:
  RowData(T* const row_data, const int max_col)
      : row_data_(row_data), max_col_(max_col) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPStoolsPSandroidPStestPSjniPSobject_trackingPSimageDTh mht_0(mht_0_v, 225, "", "./tensorflow/tools/android/test/jni/object_tracking/image.h", "RowData");
}

  inline T& operator[](const int col) const {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPStoolsPSandroidPStestPSjniPSobject_trackingPSimageDTh mht_1(mht_1_v, 230, "", "./tensorflow/tools/android/test/jni/object_tracking/image.h", "lambda");

    SCHECK(InRange(col, 0, max_col_),
          "Column out of range: %d (%d max)", col, max_col_);
    return row_data_[col];
  }

  inline operator T*() const {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPStoolsPSandroidPStestPSjniPSobject_trackingPSimageDTh mht_2(mht_2_v, 239, "", "./tensorflow/tools/android/test/jni/object_tracking/image.h", "*");

    return row_data_;
  }

 private:
  T* const row_data_;
  const int max_col_;
};
#endif

// Naive templated sorting function.
template <typename T>
int Comp(const void* a, const void* b) {
  const T val1 = *reinterpret_cast<const T*>(a);
  const T val2 = *reinterpret_cast<const T*>(b);

  if (val1 == val2) {
    return 0;
  } else if (val1 < val2) {
    return -1;
  } else {
    return 1;
  }
}

// TODO(andrewharp): Make explicit which operations support negative numbers or
// struct/class types in image data (possibly create fast multi-dim array class
// for data where pixel arithmetic does not make sense).

// Image class optimized for working on numeric arrays as grayscale image data.
// Supports other data types as a 2D array class, so long as no pixel math
// operations are called (convolution, downsampling, etc).
template <typename T>
class Image {
 public:
  Image(const int width, const int height);
  explicit Image(const Size& size);

  // Constructor that creates an image from preallocated data.
  // Note: The image takes ownership of the data lifecycle, unless own_data is
  // set to false.
  Image(const int width, const int height, T* const image_data,
        const bool own_data = true);

  ~Image();

  // Extract a pixel patch from this image, starting at a subpixel location.
  // Uses 16:16 fixed point format for representing real values and doing the
  // bilinear interpolation.
  //
  // Arguments fp_x and fp_y tell the subpixel position in fixed point format,
  // patchwidth/patchheight give the size of the patch in pixels and
  // to_data must be a valid pointer to a *contiguous* destination data array.
  template<class DstType>
  bool ExtractPatchAtSubpixelFixed1616(const int fp_x,
                                       const int fp_y,
                                       const int patchwidth,
                                       const int patchheight,
                                       DstType* to_data) const;

  Image<T>* Crop(
      const int left, const int top, const int right, const int bottom) const;

  inline int GetWidth() const {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPStoolsPSandroidPStestPSjniPSobject_trackingPSimageDTh mht_3(mht_3_v, 305, "", "./tensorflow/tools/android/test/jni/object_tracking/image.h", "GetWidth");
 return width_; }
  inline int GetHeight() const {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPStoolsPSandroidPStestPSjniPSobject_trackingPSimageDTh mht_4(mht_4_v, 309, "", "./tensorflow/tools/android/test/jni/object_tracking/image.h", "GetHeight");
 return height_; }

  // Bilinearly sample a value between pixels.  Values must be within the image.
  inline float GetPixelInterp(const float x, const float y) const;

  // Bilinearly sample a pixels at a subpixel position using fixed point
  // arithmetic.
  // Avoids float<->int conversions.
  // Values must be within the image.
  // Arguments fp_x and fp_y tell the subpixel position in
  // 16:16 fixed point format.
  //
  // Important: This function only makes sense for integer-valued images, such
  // as Image<uint8_t> or Image<int> etc.
  inline T GetPixelInterpFixed1616(const int fp_x_whole,
                                   const int fp_y_whole) const;

  // Returns true iff the pixel is in the image's boundaries.
  inline bool ValidPixel(const int x, const int y) const;

  inline BoundingBox GetContainingBox() const;

  inline bool Contains(const BoundingBox& bounding_box) const;

  inline T GetMedianValue() {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPStoolsPSandroidPStestPSjniPSobject_trackingPSimageDTh mht_5(mht_5_v, 336, "", "./tensorflow/tools/android/test/jni/object_tracking/image.h", "GetMedianValue");

    qsort(image_data_, data_size_, sizeof(image_data_[0]), Comp<T>);
    return image_data_[data_size_ >> 1];
  }

  // Returns true iff the pixel is in the image's boundaries for interpolation
  // purposes.
  // TODO(andrewharp): check in interpolation follow-up change.
  inline bool ValidInterpPixel(const float x, const float y) const;

  // Safe lookup with boundary enforcement.
  inline T GetPixelClipped(const int x, const int y) const {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPStoolsPSandroidPStestPSjniPSobject_trackingPSimageDTh mht_6(mht_6_v, 350, "", "./tensorflow/tools/android/test/jni/object_tracking/image.h", "GetPixelClipped");

    return (*this)[Clip(y, ZERO, height_less_one_)]
                  [Clip(x, ZERO, width_less_one_)];
  }

#ifdef SANITY_CHECKS
  inline RowData<T> operator[](const int row) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPStoolsPSandroidPStestPSjniPSobject_trackingPSimageDTh mht_7(mht_7_v, 359, "", "./tensorflow/tools/android/test/jni/object_tracking/image.h", "lambda");

    SCHECK(InRange(row, 0, height_less_one_),
          "Row out of range: %d (%d max)", row, height_less_one_);
    return RowData<T>(image_data_ + row * stride_, width_less_one_);
  }

  inline const RowData<T> operator[](const int row) const {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPStoolsPSandroidPStestPSjniPSobject_trackingPSimageDTh mht_8(mht_8_v, 368, "", "./tensorflow/tools/android/test/jni/object_tracking/image.h", "lambda");

    SCHECK(InRange(row, 0, height_less_one_),
          "Row out of range: %d (%d max)", row, height_less_one_);
    return RowData<T>(image_data_ + row * stride_, width_less_one_);
  }
#else
  inline T* operator[](const int row) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPStoolsPSandroidPStestPSjniPSobject_trackingPSimageDTh mht_9(mht_9_v, 377, "", "./tensorflow/tools/android/test/jni/object_tracking/image.h", "lambda");

    return image_data_ + row * stride_;
  }

  inline const T* operator[](const int row) const {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPStoolsPSandroidPStestPSjniPSobject_trackingPSimageDTh mht_10(mht_10_v, 384, "", "./tensorflow/tools/android/test/jni/object_tracking/image.h", "lambda");

    return image_data_ + row * stride_;
  }
#endif

  const T* data() const { return image_data_; }

  inline int stride() const {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPStoolsPSandroidPStestPSjniPSobject_trackingPSimageDTh mht_11(mht_11_v, 394, "", "./tensorflow/tools/android/test/jni/object_tracking/image.h", "stride");
 return stride_; }

  // Clears image to a single value.
  inline void Clear(const T& val) {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPStoolsPSandroidPStestPSjniPSobject_trackingPSimageDTh mht_12(mht_12_v, 400, "", "./tensorflow/tools/android/test/jni/object_tracking/image.h", "Clear");

    memset(image_data_, val, sizeof(*image_data_) * data_size_);
  }

#ifdef __ARM_NEON
  void Downsample2x32ColumnsNeon(const uint8_t* const original,
                                 const int stride, const int orig_x);

  void Downsample4x32ColumnsNeon(const uint8_t* const original,
                                 const int stride, const int orig_x);

  void DownsampleAveragedNeon(const uint8_t* const original, const int stride,
                              const int factor);
#endif

  // Naive downsampler that reduces image size by factor by averaging pixels in
  // blocks of size factor x factor.
  void DownsampleAveraged(const T* const original, const int stride,
                          const int factor);

  // Naive downsampler that reduces image size by factor by averaging pixels in
  // blocks of size factor x factor.
  inline void DownsampleAveraged(const Image<T>& original, const int factor) {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPStoolsPSandroidPStestPSjniPSobject_trackingPSimageDTh mht_13(mht_13_v, 425, "", "./tensorflow/tools/android/test/jni/object_tracking/image.h", "DownsampleAveraged");

    DownsampleAveraged(original.data(), original.GetWidth(), factor);
  }

  // Native downsampler that reduces image size using nearest interpolation
  void DownsampleInterpolateNearest(const Image<T>& original);

  // Native downsampler that reduces image size using fixed-point bilinear
  // interpolation
  void DownsampleInterpolateLinear(const Image<T>& original);

  // Relatively efficient downsampling of an image by a factor of two with a
  // low-pass 3x3 smoothing operation thrown in.
  void DownsampleSmoothed3x3(const Image<T>& original);

  // Relatively efficient downsampling of an image by a factor of two with a
  // low-pass 5x5 smoothing operation thrown in.
  void DownsampleSmoothed5x5(const Image<T>& original);

  // Optimized Scharr filter on a single pixel in the X direction.
  // Scharr filters are like central-difference operators, but have more
  // rotational symmetry in their response because they also consider the
  // diagonal neighbors.
  template <typename U>
  inline T ScharrPixelX(const Image<U>& original,
                        const int center_x, const int center_y) const;

  // Optimized Scharr filter on a single pixel in the X direction.
  // Scharr filters are like central-difference operators, but have more
  // rotational symmetry in their response because they also consider the
  // diagonal neighbors.
  template <typename U>
  inline T ScharrPixelY(const Image<U>& original,
                        const int center_x, const int center_y) const;

  // Convolve the image with a Scharr filter in the X direction.
  // Much faster than an equivalent generic convolution.
  template <typename U>
  inline void ScharrX(const Image<U>& original);

  // Convolve the image with a Scharr filter in the Y direction.
  // Much faster than an equivalent generic convolution.
  template <typename U>
  inline void ScharrY(const Image<U>& original);

  static inline T HalfDiff(int32_t first, int32_t second) {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPStoolsPSandroidPStestPSjniPSobject_trackingPSimageDTh mht_14(mht_14_v, 473, "", "./tensorflow/tools/android/test/jni/object_tracking/image.h", "HalfDiff");

    return (second - first) / 2;
  }

  template <typename U>
  void DerivativeX(const Image<U>& original);

  template <typename U>
  void DerivativeY(const Image<U>& original);

  // Generic function for convolving pixel with 3x3 filter.
  // Filter pixels should be in row major order.
  template <typename U>
  inline T ConvolvePixel3x3(const Image<U>& original,
                            const int* const filter,
                            const int center_x, const int center_y,
                            const int total) const;

  // Generic function for convolving an image with a 3x3 filter.
  // TODO(andrewharp): Generalize this for any size filter.
  template <typename U>
  inline void Convolve3x3(const Image<U>& original,
                          const int32_t* const filter);

  // Load this image's data from a data array. The data at pixels is assumed to
  // have dimensions equivalent to this image's dimensions * factor.
  inline void FromArray(const T* const pixels, const int stride,
                        const int factor = 1);

  // Copy the image back out to an appropriately sized data array.
  inline void ToArray(T* const pixels) const {
   std::vector<std::string> mht_15_v;
   MHTracer_DTPStensorflowPStoolsPSandroidPStestPSjniPSobject_trackingPSimageDTh mht_15(mht_15_v, 506, "", "./tensorflow/tools/android/test/jni/object_tracking/image.h", "ToArray");

    // If not subsampling, memcpy should be faster.
    memcpy(pixels, this->image_data_, data_size_ * sizeof(T));
  }

  // Precompute these for efficiency's sake as they're used by a lot of
  // clipping code and loop code.
  // TODO(andrewharp): make these only accessible by other Images.
  const int width_less_one_;
  const int height_less_one_;

  // The raw size of the allocated data.
  const int data_size_;

 private:
  inline void Allocate() {
   std::vector<std::string> mht_16_v;
   MHTracer_DTPStensorflowPStoolsPSandroidPStestPSjniPSobject_trackingPSimageDTh mht_16(mht_16_v, 524, "", "./tensorflow/tools/android/test/jni/object_tracking/image.h", "Allocate");

    image_data_ = new T[data_size_];
    if (image_data_ == NULL) {
      LOGE("Couldn't allocate image data!");
    }
  }

  T* image_data_;

  bool own_data_;

  const int width_;
  const int height_;

  // The image stride (offset to next row).
  // TODO(andrewharp): Make sure that stride is honored in all code.
  const int stride_;

  TF_DISALLOW_COPY_AND_ASSIGN(Image);
};

template <typename t>
inline std::ostream& operator<<(std::ostream& stream, const Image<t>& image) {
   std::vector<std::string> mht_17_v;
   MHTracer_DTPStensorflowPStoolsPSandroidPStestPSjniPSobject_trackingPSimageDTh mht_17(mht_17_v, 549, "", "./tensorflow/tools/android/test/jni/object_tracking/image.h", "operator<<");

  for (int y = 0; y < image.GetHeight(); ++y) {
    for (int x = 0; x < image.GetWidth(); ++x) {
      stream << image[y][x] << " ";
    }
    stream << std::endl;
  }
  return stream;
}

}  // namespace tf_tracking

#endif  // TENSORFLOW_EXAMPLES_ANDROID_JNI_OBJECT_TRACKING_IMAGE_H_
