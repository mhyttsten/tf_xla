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

#ifndef TENSORFLOW_EXAMPLES_ANDROID_JNI_OBJECT_TRACKING_INTEGRAL_IMAGE_H_
#define TENSORFLOW_EXAMPLES_ANDROID_JNI_OBJECT_TRACKING_INTEGRAL_IMAGE_H_
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
class MHTracer_DTPStensorflowPStoolsPSandroidPStestPSjniPSobject_trackingPSintegral_imageDTh {
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
   MHTracer_DTPStensorflowPStoolsPSandroidPStestPSjniPSobject_trackingPSintegral_imageDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPStoolsPSandroidPStestPSjniPSobject_trackingPSintegral_imageDTh() {
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


#include "tensorflow/tools/android/test/jni/object_tracking/geom.h"
#include "tensorflow/tools/android/test/jni/object_tracking/image-inl.h"
#include "tensorflow/tools/android/test/jni/object_tracking/image.h"
#include "tensorflow/tools/android/test/jni/object_tracking/utils.h"

namespace tf_tracking {

typedef uint8_t Code;

class IntegralImage : public Image<uint32_t> {
 public:
  explicit IntegralImage(const Image<uint8_t>& image_base)
      : Image<uint32_t>(image_base.GetWidth(), image_base.GetHeight()) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPStoolsPSandroidPStestPSjniPSobject_trackingPSintegral_imageDTh mht_0(mht_0_v, 200, "", "./tensorflow/tools/android/test/jni/object_tracking/integral_image.h", "IntegralImage");

    Recompute(image_base);
  }

  IntegralImage(const int width, const int height)
      : Image<uint32_t>(width, height) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPStoolsPSandroidPStestPSjniPSobject_trackingPSintegral_imageDTh mht_1(mht_1_v, 208, "", "./tensorflow/tools/android/test/jni/object_tracking/integral_image.h", "IntegralImage");
}

  void Recompute(const Image<uint8_t>& image_base) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPStoolsPSandroidPStestPSjniPSobject_trackingPSintegral_imageDTh mht_2(mht_2_v, 213, "", "./tensorflow/tools/android/test/jni/object_tracking/integral_image.h", "Recompute");

    SCHECK(image_base.GetWidth() == GetWidth() &&
          image_base.GetHeight() == GetHeight(), "Dimensions don't match!");

    // Sum along first row.
    {
      int x_sum = 0;
      for (int x = 0; x < image_base.GetWidth(); ++x) {
        x_sum += image_base[0][x];
        (*this)[0][x] = x_sum;
      }
    }

    // Sum everything else.
    for (int y = 1; y < image_base.GetHeight(); ++y) {
      uint32_t* curr_sum = (*this)[y];

      // Previously summed pointers.
      const uint32_t* up_one = (*this)[y - 1];

      // Current value pointer.
      const uint8_t* curr_delta = image_base[y];

      uint32_t row_till_now = 0;

      for (int x = 0; x < GetWidth(); ++x) {
        // Add the one above and the one to the left.
        row_till_now += *curr_delta;
        *curr_sum = *up_one + row_till_now;

        // Scoot everything along.
        ++curr_sum;
        ++up_one;
        ++curr_delta;
      }
    }

    SCHECK(VerifyData(image_base), "Images did not match!");
  }

  bool VerifyData(const Image<uint8_t>& image_base) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPStoolsPSandroidPStestPSjniPSobject_trackingPSintegral_imageDTh mht_3(mht_3_v, 256, "", "./tensorflow/tools/android/test/jni/object_tracking/integral_image.h", "VerifyData");

    for (int y = 0; y < GetHeight(); ++y) {
      for (int x = 0; x < GetWidth(); ++x) {
        uint32_t curr_val = (*this)[y][x];

        if (x > 0) {
          curr_val -= (*this)[y][x - 1];
        }

        if (y > 0) {
          curr_val -= (*this)[y - 1][x];
        }

        if (x > 0 && y > 0) {
          curr_val += (*this)[y - 1][x - 1];
        }

        if (curr_val != image_base[y][x]) {
          LOGE("Mismatch! %d vs %d", curr_val, image_base[y][x]);
          return false;
        }

        if (GetRegionSum(x, y, x, y) != curr_val) {
          LOGE("Mismatch!");
        }
      }
    }

    return true;
  }

  // Returns the sum of all pixels in the specified region.
  inline uint32_t GetRegionSum(const int x1, const int y1, const int x2,
                               const int y2) const {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPStoolsPSandroidPStestPSjniPSobject_trackingPSintegral_imageDTh mht_4(mht_4_v, 292, "", "./tensorflow/tools/android/test/jni/object_tracking/integral_image.h", "GetRegionSum");

    SCHECK(x1 >= 0 && y1 >= 0 &&
          x2 >= x1 && y2 >= y1 && x2 < GetWidth() && y2 < GetHeight(),
          "indices out of bounds! %d-%d / %d, %d-%d / %d, ",
          x1, x2, GetWidth(), y1, y2, GetHeight());

    const uint32_t everything = (*this)[y2][x2];

    uint32_t sum = everything;
    if (x1 > 0 && y1 > 0) {
      // Most common case.
      const uint32_t left = (*this)[y2][x1 - 1];
      const uint32_t top = (*this)[y1 - 1][x2];
      const uint32_t top_left = (*this)[y1 - 1][x1 - 1];

      sum = everything - left - top + top_left;
      SCHECK(sum >= 0, "Both: %d - %d - %d + %d => %d! indices: %d %d %d %d",
            everything, left, top, top_left, sum, x1, y1, x2, y2);
    } else if (x1 > 0) {
      // Flush against top of image.
      // Subtract out the region to the left only.
      const uint32_t top = (*this)[y2][x1 - 1];
      sum = everything - top;
      SCHECK(sum >= 0, "Top: %d - %d => %d!", everything, top, sum);
    } else if (y1 > 0) {
      // Flush against left side of image.
      // Subtract out the region above only.
      const uint32_t left = (*this)[y1 - 1][x2];
      sum = everything - left;
      SCHECK(sum >= 0, "Left: %d - %d => %d!", everything, left, sum);
    }

    SCHECK(sum >= 0, "Negative sum!");

    return sum;
  }

  // Returns the 2bit code associated with this region, which represents
  // the overall gradient.
  inline Code GetCode(const BoundingBox& bounding_box) const {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPStoolsPSandroidPStestPSjniPSobject_trackingPSintegral_imageDTh mht_5(mht_5_v, 334, "", "./tensorflow/tools/android/test/jni/object_tracking/integral_image.h", "GetCode");

    return GetCode(bounding_box.left_, bounding_box.top_,
                   bounding_box.right_, bounding_box.bottom_);
  }

  inline Code GetCode(const int x1, const int y1,
                      const int x2, const int y2) const {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPStoolsPSandroidPStestPSjniPSobject_trackingPSintegral_imageDTh mht_6(mht_6_v, 343, "", "./tensorflow/tools/android/test/jni/object_tracking/integral_image.h", "GetCode");

    SCHECK(x1 < x2 && y1 < y2, "Bounds out of order!! TL:%d,%d BR:%d,%d",
           x1, y1, x2, y2);

    // Gradient computed vertically.
    const int box_height = (y2 - y1) / 2;
    const int top_sum = GetRegionSum(x1, y1, x2, y1 + box_height);
    const int bottom_sum = GetRegionSum(x1, y2 - box_height, x2, y2);
    const bool vertical_code = top_sum > bottom_sum;

    // Gradient computed horizontally.
    const int box_width = (x2 - x1) / 2;
    const int left_sum = GetRegionSum(x1, y1, x1 + box_width, y2);
    const int right_sum = GetRegionSum(x2 - box_width, y1, x2, y2);
    const bool horizontal_code = left_sum > right_sum;

    const Code final_code = (vertical_code << 1) | horizontal_code;

    SCHECK(InRange(final_code, static_cast<Code>(0), static_cast<Code>(3)),
          "Invalid code! %d", final_code);

    // Returns a value 0-3.
    return final_code;
  }

 private:
  TF_DISALLOW_COPY_AND_ASSIGN(IntegralImage);
};

}  // namespace tf_tracking

#endif  // TENSORFLOW_EXAMPLES_ANDROID_JNI_OBJECT_TRACKING_INTEGRAL_IMAGE_H_
