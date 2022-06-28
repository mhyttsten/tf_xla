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
class MHTracer_DTPStensorflowPStoolsPSandroidPStestPSjniPSobject_trackingPSimage_neonDTcc {
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
   MHTracer_DTPStensorflowPStoolsPSandroidPStestPSjniPSobject_trackingPSimage_neonDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPStoolsPSandroidPStestPSjniPSobject_trackingPSimage_neonDTcc() {
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

// NEON implementations of Image methods for compatible devices.  Control
// should never enter this compilation unit on incompatible devices.

#ifdef __ARM_NEON

#include <arm_neon.h>
#include <stdint.h>

#include "tensorflow/tools/android/test/jni/object_tracking/image-inl.h"
#include "tensorflow/tools/android/test/jni/object_tracking/image.h"
#include "tensorflow/tools/android/test/jni/object_tracking/image_utils.h"
#include "tensorflow/tools/android/test/jni/object_tracking/utils.h"

namespace tf_tracking {

// This function does the bulk of the work.
template <>
void Image<uint8_t>::Downsample2x32ColumnsNeon(const uint8_t* const original,
                                               const int stride,
                                               const int orig_x) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPStoolsPSandroidPStestPSjniPSobject_trackingPSimage_neonDTcc mht_0(mht_0_v, 204, "", "./tensorflow/tools/android/test/jni/object_tracking/image_neon.cc", "Image<uint8_t>::Downsample2x32ColumnsNeon");

  // Divide input x offset by 2 to find output offset.
  const int new_x = orig_x >> 1;

  // Initial offset into top row.
  const uint8_t* offset = original + orig_x;

  // This points to the leftmost pixel of our 8 horizontally arranged
  // pixels in the destination data.
  uint8_t* ptr_dst = (*this)[0] + new_x;

  // Sum along vertical columns.
  // Process 32x2 input pixels and 16x1 output pixels per iteration.
  for (int new_y = 0; new_y < height_; ++new_y) {
    uint16x8_t accum1 = vdupq_n_u16(0);
    uint16x8_t accum2 = vdupq_n_u16(0);

    // Go top to bottom across the four rows of input pixels that make up
    // this output row.
    for (int row_num = 0; row_num < 2; ++row_num) {
      // First 16 bytes.
      {
        // Load 16 bytes of data from current offset.
        const uint8x16_t curr_data1 = vld1q_u8(offset);

        // Pairwise add and accumulate into accum vectors (16 bit to account
        // for values above 255).
        accum1 = vpadalq_u8(accum1, curr_data1);
      }

      // Second 16 bytes.
      {
        // Load 16 bytes of data from current offset.
        const uint8x16_t curr_data2 = vld1q_u8(offset + 16);

        // Pairwise add and accumulate into accum vectors (16 bit to account
        // for values above 255).
        accum2 = vpadalq_u8(accum2, curr_data2);
      }

      // Move offset down one row.
      offset += stride;
    }

    // Divide by 4 (number of input pixels per output
    // pixel) and narrow data from 16 bits per pixel to 8 bpp.
    const uint8x8_t tmp_pix1 = vqshrn_n_u16(accum1, 2);
    const uint8x8_t tmp_pix2 = vqshrn_n_u16(accum2, 2);

    // Concatenate 8x1 pixel strips into 16x1 pixel strip.
    const uint8x16_t allpixels = vcombine_u8(tmp_pix1, tmp_pix2);

    // Copy all pixels from composite 16x1 vector into output strip.
    vst1q_u8(ptr_dst, allpixels);

    ptr_dst += stride_;
  }
}

// This function does the bulk of the work.
template <>
void Image<uint8_t>::Downsample4x32ColumnsNeon(const uint8_t* const original,
                                               const int stride,
                                               const int orig_x) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPStoolsPSandroidPStestPSjniPSobject_trackingPSimage_neonDTcc mht_1(mht_1_v, 270, "", "./tensorflow/tools/android/test/jni/object_tracking/image_neon.cc", "Image<uint8_t>::Downsample4x32ColumnsNeon");

  // Divide input x offset by 4 to find output offset.
  const int new_x = orig_x >> 2;

  // Initial offset into top row.
  const uint8_t* offset = original + orig_x;

  // This points to the leftmost pixel of our 8 horizontally arranged
  // pixels in the destination data.
  uint8_t* ptr_dst = (*this)[0] + new_x;

  // Sum along vertical columns.
  // Process 32x4 input pixels and 8x1 output pixels per iteration.
  for (int new_y = 0; new_y < height_; ++new_y) {
    uint16x8_t accum1 = vdupq_n_u16(0);
    uint16x8_t accum2 = vdupq_n_u16(0);

    // Go top to bottom across the four rows of input pixels that make up
    // this output row.
    for (int row_num = 0; row_num < 4; ++row_num) {
      // First 16 bytes.
      {
        // Load 16 bytes of data from current offset.
        const uint8x16_t curr_data1 = vld1q_u8(offset);

        // Pairwise add and accumulate into accum vectors (16 bit to account
        // for values above 255).
        accum1 = vpadalq_u8(accum1, curr_data1);
      }

      // Second 16 bytes.
      {
        // Load 16 bytes of data from current offset.
        const uint8x16_t curr_data2 = vld1q_u8(offset + 16);

        // Pairwise add and accumulate into accum vectors (16 bit to account
        // for values above 255).
        accum2 = vpadalq_u8(accum2, curr_data2);
      }

      // Move offset down one row.
      offset += stride;
    }

    // Add and widen, then divide by 16 (number of input pixels per output
    // pixel) and narrow data from 32 bits per pixel to 16 bpp.
    const uint16x4_t tmp_pix1 = vqshrn_n_u32(vpaddlq_u16(accum1), 4);
    const uint16x4_t tmp_pix2 = vqshrn_n_u32(vpaddlq_u16(accum2), 4);

    // Combine 4x1 pixel strips into 8x1 pixel strip and narrow from
    // 16 bits to 8 bits per pixel.
    const uint8x8_t allpixels = vmovn_u16(vcombine_u16(tmp_pix1, tmp_pix2));

    // Copy all pixels from composite 8x1 vector into output strip.
    vst1_u8(ptr_dst, allpixels);

    ptr_dst += stride_;
  }
}


// Hardware accelerated downsampling method for supported devices.
// Requires that image size be a multiple of 16 pixels in each dimension,
// and that downsampling be by a factor of 2 or 4.
template <>
void Image<uint8_t>::DownsampleAveragedNeon(const uint8_t* const original,
                                            const int stride,
                                            const int factor) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPStoolsPSandroidPStestPSjniPSobject_trackingPSimage_neonDTcc mht_2(mht_2_v, 340, "", "./tensorflow/tools/android/test/jni/object_tracking/image_neon.cc", "Image<uint8_t>::DownsampleAveragedNeon");

  // TODO(andrewharp): stride is a bad approximation for the src image's width.
  // Better to pass that in directly.
  SCHECK(width_ * factor <= stride, "Uh oh!");
  const int last_starting_index = width_ * factor - 32;

  // We process 32 input pixels lengthwise at a time.
  // The output per pass of this loop is an 8 wide by downsampled height tall
  // pixel strip.
  int orig_x = 0;
  for (; orig_x <= last_starting_index; orig_x += 32) {
    if (factor == 2) {
      Downsample2x32ColumnsNeon(original, stride, orig_x);
    } else {
      Downsample4x32ColumnsNeon(original, stride, orig_x);
    }
  }

  // If a last pass is required, push it to the left enough so that it never
  // goes out of bounds. This will result in some extra computation on devices
  // whose frame widths are multiples of 16 and not 32.
  if (orig_x < last_starting_index + 32) {
    if (factor == 2) {
      Downsample2x32ColumnsNeon(original, stride, last_starting_index);
    } else {
      Downsample4x32ColumnsNeon(original, stride, last_starting_index);
    }
  }
}


// Puts the image gradient matrix about a pixel into the 2x2 float array G.
// vals_x should be an array of the window x gradient values, whose indices
// can be in any order but are parallel to the vals_y entries.
// See http://robots.stanford.edu/cs223b04/algo_tracking.pdf for more details.
void CalculateGNeon(const float* const vals_x, const float* const vals_y,
                    const int num_vals, float* const G) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPStoolsPSandroidPStestPSjniPSobject_trackingPSimage_neonDTcc mht_3(mht_3_v, 379, "", "./tensorflow/tools/android/test/jni/object_tracking/image_neon.cc", "CalculateGNeon");

  const float32_t* const arm_vals_x = (const float32_t*) vals_x;
  const float32_t* const arm_vals_y = (const float32_t*) vals_y;

  // Running sums.
  float32x4_t xx = vdupq_n_f32(0.0f);
  float32x4_t xy = vdupq_n_f32(0.0f);
  float32x4_t yy = vdupq_n_f32(0.0f);

  // Maximum index we can load 4 consecutive values from.
  // e.g. if there are 81 values, our last full pass can be from index 77:
  // 81-4=>77 (77, 78, 79, 80)
  const int max_i = num_vals - 4;

  // Defined here because we want to keep track of how many values were
  // processed by NEON, so that we can finish off the remainder the normal
  // way.
  int i = 0;

  // Process values 4 at a time, accumulating the sums of
  // the pixel-wise x*x, x*y, and y*y values.
  for (; i <= max_i; i += 4) {
    // Load xs
    float32x4_t x = vld1q_f32(arm_vals_x + i);

    // Multiply x*x and accumulate.
    xx = vmlaq_f32(xx, x, x);

    // Load ys
    float32x4_t y = vld1q_f32(arm_vals_y + i);

    // Multiply x*y and accumulate.
    xy = vmlaq_f32(xy, x, y);

    // Multiply y*y and accumulate.
    yy = vmlaq_f32(yy, y, y);
  }

  static float32_t xx_vals[4];
  static float32_t xy_vals[4];
  static float32_t yy_vals[4];

  vst1q_f32(xx_vals, xx);
  vst1q_f32(xy_vals, xy);
  vst1q_f32(yy_vals, yy);

  // Accumulated values are store in sets of 4, we have to manually add
  // the last bits together.
  for (int j = 0; j < 4; ++j) {
    G[0] += xx_vals[j];
    G[1] += xy_vals[j];
    G[3] += yy_vals[j];
  }

  // Finishes off last few values (< 4) from above.
  for (; i < num_vals; ++i) {
    G[0] += Square(vals_x[i]);
    G[1] += vals_x[i] * vals_y[i];
    G[3] += Square(vals_y[i]);
  }

  // The matrix is symmetric, so this is a given.
  G[2] = G[1];
}

}  // namespace tf_tracking

#endif
