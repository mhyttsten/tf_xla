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

#ifndef TENSORFLOW_EXAMPLES_ANDROID_JNI_OBJECT_TRACKING_IMAGE_UTILS_H_
#define TENSORFLOW_EXAMPLES_ANDROID_JNI_OBJECT_TRACKING_IMAGE_UTILS_H_
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
class MHTracer_DTPStensorflowPStoolsPSandroidPStestPSjniPSobject_trackingPSimage_utilsDTh {
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
   MHTracer_DTPStensorflowPStoolsPSandroidPStestPSjniPSobject_trackingPSimage_utilsDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPStoolsPSandroidPStestPSjniPSobject_trackingPSimage_utilsDTh() {
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
#include "tensorflow/tools/android/test/jni/object_tracking/image-inl.h"
#include "tensorflow/tools/android/test/jni/object_tracking/image.h"
#include "tensorflow/tools/android/test/jni/object_tracking/utils.h"

namespace tf_tracking {

inline void GetUV(const uint8_t* const input, Image<uint8_t>* const u,
                  Image<uint8_t>* const v) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPStoolsPSandroidPStestPSjniPSobject_trackingPSimage_utilsDTh mht_0(mht_0_v, 198, "", "./tensorflow/tools/android/test/jni/object_tracking/image_utils.h", "GetUV");

  const uint8_t* pUV = input;

  for (int row = 0; row < u->GetHeight(); ++row) {
    uint8_t* u_curr = (*u)[row];
    uint8_t* v_curr = (*v)[row];
    for (int col = 0; col < u->GetWidth(); ++col) {
#ifdef __APPLE__
      *u_curr++ = *pUV++;
      *v_curr++ = *pUV++;
#else
      *v_curr++ = *pUV++;
      *u_curr++ = *pUV++;
#endif
    }
  }
}

// Marks every point within a circle of a given radius on the given boolean
// image true.
template <typename U>
inline static void MarkImage(const int x, const int y, const int radius,
                             Image<U>* const img) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPStoolsPSandroidPStestPSjniPSobject_trackingPSimage_utilsDTh mht_1(mht_1_v, 223, "", "./tensorflow/tools/android/test/jni/object_tracking/image_utils.h", "MarkImage");

  SCHECK(img->ValidPixel(x, y), "Marking invalid pixel in image! %d, %d", x, y);

  // Precomputed for efficiency.
  const int squared_radius = Square(radius);

  // Mark every row in the circle.
  for (int d_y = 0; d_y <= radius; ++d_y) {
    const int squared_y_dist = Square(d_y);

    const int min_y = MAX(y - d_y, 0);
    const int max_y = MIN(y + d_y, img->height_less_one_);

    // The max d_x of the circle must be strictly greater or equal to
    // radius - d_y for any positive d_y. Thus, starting from radius - d_y will
    // reduce the number of iterations required as compared to starting from
    // either 0 and counting up or radius and counting down.
    for (int d_x = radius - d_y; d_x <= radius; ++d_x) {
      // The first time this criteria is met, we know the width of the circle at
      // this row (without using sqrt).
      if (squared_y_dist + Square(d_x) >= squared_radius) {
        const int min_x = MAX(x - d_x, 0);
        const int max_x = MIN(x + d_x, img->width_less_one_);

        // Mark both above and below the center row.
        bool* const top_row_start = (*img)[min_y] + min_x;
        bool* const bottom_row_start = (*img)[max_y] + min_x;

        const int x_width = max_x - min_x + 1;
        memset(top_row_start, true, sizeof(*top_row_start) * x_width);
        memset(bottom_row_start, true, sizeof(*bottom_row_start) * x_width);

        // This row is marked, time to move on to the next row.
        break;
      }
    }
  }
}

#ifdef __ARM_NEON
void CalculateGNeon(
    const float* const vals_x, const float* const vals_y,
    const int num_vals, float* const G);
#endif

// Puts the image gradient matrix about a pixel into the 2x2 float array G.
// vals_x should be an array of the window x gradient values, whose indices
// can be in any order but are parallel to the vals_y entries.
// See http://robots.stanford.edu/cs223b04/algo_tracking.pdf for more details.
inline void CalculateG(const float* const vals_x, const float* const vals_y,
                       const int num_vals, float* const G) {
#ifdef __ARM_NEON
  CalculateGNeon(vals_x, vals_y, num_vals, G);
  return;
#endif

  // Non-accelerated version.
  for (int i = 0; i < num_vals; ++i) {
    G[0] += Square(vals_x[i]);
    G[1] += vals_x[i] * vals_y[i];
    G[3] += Square(vals_y[i]);
  }

  // The matrix is symmetric, so this is a given.
  G[2] = G[1];
}

inline void CalculateGInt16(const int16_t* const vals_x,
                            const int16_t* const vals_y, const int num_vals,
                            int* const G) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPStoolsPSandroidPStestPSjniPSobject_trackingPSimage_utilsDTh mht_2(mht_2_v, 295, "", "./tensorflow/tools/android/test/jni/object_tracking/image_utils.h", "CalculateGInt16");

  // Non-accelerated version.
  for (int i = 0; i < num_vals; ++i) {
    G[0] += Square(vals_x[i]);
    G[1] += vals_x[i] * vals_y[i];
    G[3] += Square(vals_y[i]);
  }

  // The matrix is symmetric, so this is a given.
  G[2] = G[1];
}


// Puts the image gradient matrix about a pixel into the 2x2 float array G.
// Looks up interpolated pixels, then calls above method for implementation.
inline void CalculateG(const int window_radius, const float center_x,
                       const float center_y, const Image<int32_t>& I_x,
                       const Image<int32_t>& I_y, float* const G) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPStoolsPSandroidPStestPSjniPSobject_trackingPSimage_utilsDTh mht_3(mht_3_v, 315, "", "./tensorflow/tools/android/test/jni/object_tracking/image_utils.h", "CalculateG");

  SCHECK(I_x.ValidPixel(center_x, center_y), "Problem in calculateG!");

  // Hardcoded to allow for a max window radius of 5 (9 pixels x 9 pixels).
  static const int kMaxWindowRadius = 5;
  SCHECK(window_radius <= kMaxWindowRadius,
        "Window %d > %d!", window_radius, kMaxWindowRadius);

  // Diameter of window is 2 * radius + 1 for center pixel.
  static const int kWindowBufferSize =
      (kMaxWindowRadius * 2 + 1) * (kMaxWindowRadius * 2 + 1);

  // Preallocate buffers statically for efficiency.
  static int16_t vals_x[kWindowBufferSize];
  static int16_t vals_y[kWindowBufferSize];

  const int src_left_fixed = RealToFixed1616(center_x - window_radius);
  const int src_top_fixed = RealToFixed1616(center_y - window_radius);

  int16_t* vals_x_ptr = vals_x;
  int16_t* vals_y_ptr = vals_y;

  const int window_size = 2 * window_radius + 1;
  for (int y = 0; y < window_size; ++y) {
    const int fp_y = src_top_fixed + (y << 16);

    for (int x = 0; x < window_size; ++x) {
      const int fp_x = src_left_fixed + (x << 16);

      *vals_x_ptr++ = I_x.GetPixelInterpFixed1616(fp_x, fp_y);
      *vals_y_ptr++ = I_y.GetPixelInterpFixed1616(fp_x, fp_y);
    }
  }

  int32_t g_temp[] = {0, 0, 0, 0};
  CalculateGInt16(vals_x, vals_y, window_size * window_size, g_temp);

  for (int i = 0; i < 4; ++i) {
    G[i] = g_temp[i];
  }
}

inline float ImageCrossCorrelation(const Image<float>& image1,
                                   const Image<float>& image2,
                                   const int x_offset, const int y_offset) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPStoolsPSandroidPStestPSjniPSobject_trackingPSimage_utilsDTh mht_4(mht_4_v, 362, "", "./tensorflow/tools/android/test/jni/object_tracking/image_utils.h", "ImageCrossCorrelation");

  SCHECK(image1.GetWidth() == image2.GetWidth() &&
         image1.GetHeight() == image2.GetHeight(),
        "Dimension mismatch! %dx%d vs %dx%d",
        image1.GetWidth(), image1.GetHeight(),
        image2.GetWidth(), image2.GetHeight());

  const int num_pixels = image1.GetWidth() * image1.GetHeight();
  const float* data1 = image1.data();
  const float* data2 = image2.data();
  return ComputeCrossCorrelation(data1, data2, num_pixels);
}

// Copies an arbitrary region of an image to another (floating point)
// image, scaling as it goes using bilinear interpolation.
inline void CopyArea(const Image<uint8_t>& image,
                     const BoundingBox& area_to_copy,
                     Image<float>* const patch_image) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPStoolsPSandroidPStestPSjniPSobject_trackingPSimage_utilsDTh mht_5(mht_5_v, 382, "", "./tensorflow/tools/android/test/jni/object_tracking/image_utils.h", "CopyArea");

  VLOG(2) << "Copying from: " << area_to_copy << std::endl;

  const int patch_width = patch_image->GetWidth();
  const int patch_height = patch_image->GetHeight();

  const float x_dist_between_samples = patch_width > 0 ?
      area_to_copy.GetWidth() / (patch_width - 1) : 0;

  const float y_dist_between_samples = patch_height > 0 ?
      area_to_copy.GetHeight() / (patch_height - 1) : 0;

  for (int y_index = 0; y_index < patch_height; ++y_index) {
    const float sample_y =
        y_index * y_dist_between_samples + area_to_copy.top_;

    for (int x_index = 0; x_index < patch_width; ++x_index) {
      const float sample_x =
          x_index * x_dist_between_samples + area_to_copy.left_;

      if (image.ValidInterpPixel(sample_x, sample_y)) {
        // TODO(andrewharp): Do area averaging when downsampling.
        (*patch_image)[y_index][x_index] =
            image.GetPixelInterp(sample_x, sample_y);
      } else {
        (*patch_image)[y_index][x_index] = -1.0f;
      }
    }
  }
}


// Takes a floating point image and normalizes it in-place.
//
// First, negative values will be set to the mean of the non-negative pixels
// in the image.
//
// Then, the resulting will be normalized such that it has mean value of 0.0 and
// a standard deviation of 1.0.
inline void NormalizeImage(Image<float>* const image) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPStoolsPSandroidPStestPSjniPSobject_trackingPSimage_utilsDTh mht_6(mht_6_v, 424, "", "./tensorflow/tools/android/test/jni/object_tracking/image_utils.h", "NormalizeImage");

  const float* const data_ptr = image->data();

  // Copy only the non-negative values to some temp memory.
  float running_sum = 0.0f;
  int num_data_gte_zero = 0;
  {
    float* const curr_data = (*image)[0];
    for (int i = 0; i < image->data_size_; ++i) {
      if (curr_data[i] >= 0.0f) {
        running_sum += curr_data[i];
        ++num_data_gte_zero;
      } else {
        curr_data[i] = -1.0f;
      }
    }
  }

  // If none of the pixels are valid, just set the entire thing to 0.0f.
  if (num_data_gte_zero == 0) {
    image->Clear(0.0f);
    return;
  }

  const float corrected_mean = running_sum / num_data_gte_zero;

  float* curr_data = (*image)[0];
  for (int i = 0; i < image->data_size_; ++i) {
    const float curr_val = *curr_data;
    *curr_data++ = curr_val < 0 ? 0 : curr_val - corrected_mean;
  }

  const float std_dev = ComputeStdDev(data_ptr, image->data_size_, 0.0f);

  if (std_dev > 0.0f) {
    curr_data = (*image)[0];
    for (int i = 0; i < image->data_size_; ++i) {
      *curr_data++ /= std_dev;
    }

#ifdef SANITY_CHECKS
    LOGV("corrected_mean: %1.2f  std_dev: %1.2f", corrected_mean, std_dev);
    const float correlation =
        ComputeCrossCorrelation(image->data(),
                                image->data(),
                                image->data_size_);

    if (std::abs(correlation - 1.0f) > EPSILON) {
      LOG(ERROR) << "Bad image!" << std::endl;
      LOG(ERROR) << *image << std::endl;
    }

    SCHECK(std::abs(correlation - 1.0f) < EPSILON,
           "Correlation wasn't 1.0f:  %.10f", correlation);
#endif
  }
}

}  // namespace tf_tracking

#endif  // TENSORFLOW_EXAMPLES_ANDROID_JNI_OBJECT_TRACKING_IMAGE_UTILS_H_
