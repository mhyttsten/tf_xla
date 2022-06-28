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

#ifndef TENSORFLOW_EXAMPLES_ANDROID_JNI_OBJECT_TRACKING_GEOM_H_
#define TENSORFLOW_EXAMPLES_ANDROID_JNI_OBJECT_TRACKING_GEOM_H_
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
class MHTracer_DTPStensorflowPStoolsPSandroidPStestPSjniPSobject_trackingPSgeomDTh {
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
   MHTracer_DTPStensorflowPStoolsPSandroidPStestPSjniPSobject_trackingPSgeomDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPStoolsPSandroidPStestPSjniPSobject_trackingPSgeomDTh() {
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


#include "tensorflow/tools/android/test/jni/object_tracking/logging.h"
#include "tensorflow/tools/android/test/jni/object_tracking/utils.h"

namespace tf_tracking {

struct Size {
  Size(const int width, const int height) : width(width), height(height) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPStoolsPSandroidPStestPSjniPSobject_trackingPSgeomDTh mht_0(mht_0_v, 194, "", "./tensorflow/tools/android/test/jni/object_tracking/geom.h", "Size");
}

  int width;
  int height;
};


class Point2f {
 public:
  Point2f() : x(0.0f), y(0.0f) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPStoolsPSandroidPStestPSjniPSobject_trackingPSgeomDTh mht_1(mht_1_v, 206, "", "./tensorflow/tools/android/test/jni/object_tracking/geom.h", "Point2f");
}
  Point2f(const float x, const float y) : x(x), y(y) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPStoolsPSandroidPStestPSjniPSobject_trackingPSgeomDTh mht_2(mht_2_v, 210, "", "./tensorflow/tools/android/test/jni/object_tracking/geom.h", "Point2f");
}

  inline Point2f operator- (const Point2f& that) const {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPStoolsPSandroidPStestPSjniPSobject_trackingPSgeomDTh mht_3(mht_3_v, 215, "", "./tensorflow/tools/android/test/jni/object_tracking/geom.h", "-");

    return Point2f(this->x - that.x, this->y - that.y);
  }

  inline Point2f operator+ (const Point2f& that) const {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPStoolsPSandroidPStestPSjniPSobject_trackingPSgeomDTh mht_4(mht_4_v, 222, "", "./tensorflow/tools/android/test/jni/object_tracking/geom.h", "+");

    return Point2f(this->x + that.x, this->y + that.y);
  }

  inline Point2f& operator+= (const Point2f& that) {
    this->x += that.x;
    this->y += that.y;
    return *this;
  }

  inline Point2f& operator-= (const Point2f& that) {
    this->x -= that.x;
    this->y -= that.y;
    return *this;
  }

  inline Point2f operator- (const Point2f& that) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPStoolsPSandroidPStestPSjniPSobject_trackingPSgeomDTh mht_5(mht_5_v, 241, "", "./tensorflow/tools/android/test/jni/object_tracking/geom.h", "-");

    return Point2f(this->x - that.x, this->y - that.y);
  }

  inline float LengthSquared() {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPStoolsPSandroidPStestPSjniPSobject_trackingPSgeomDTh mht_6(mht_6_v, 248, "", "./tensorflow/tools/android/test/jni/object_tracking/geom.h", "LengthSquared");

    return Square(this->x) + Square(this->y);
  }

  inline float Length() {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPStoolsPSandroidPStestPSjniPSobject_trackingPSgeomDTh mht_7(mht_7_v, 255, "", "./tensorflow/tools/android/test/jni/object_tracking/geom.h", "Length");

    return sqrtf(LengthSquared());
  }

  inline float DistanceSquared(const Point2f& that) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPStoolsPSandroidPStestPSjniPSobject_trackingPSgeomDTh mht_8(mht_8_v, 262, "", "./tensorflow/tools/android/test/jni/object_tracking/geom.h", "DistanceSquared");

    return Square(this->x - that.x) + Square(this->y - that.y);
  }

  inline float Distance(const Point2f& that) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPStoolsPSandroidPStestPSjniPSobject_trackingPSgeomDTh mht_9(mht_9_v, 269, "", "./tensorflow/tools/android/test/jni/object_tracking/geom.h", "Distance");

    return sqrtf(DistanceSquared(that));
  }

  float x;
  float y;
};

inline std::ostream& operator<<(std::ostream& stream, const Point2f& point) {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPStoolsPSandroidPStestPSjniPSobject_trackingPSgeomDTh mht_10(mht_10_v, 280, "", "./tensorflow/tools/android/test/jni/object_tracking/geom.h", "operator<<");

  stream << point.x << "," << point.y;
  return stream;
}

class BoundingBox {
 public:
  BoundingBox()
      : left_(0),
        top_(0),
        right_(0),
        bottom_(0) {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPStoolsPSandroidPStestPSjniPSobject_trackingPSgeomDTh mht_11(mht_11_v, 294, "", "./tensorflow/tools/android/test/jni/object_tracking/geom.h", "BoundingBox");
}

  BoundingBox(const BoundingBox& bounding_box)
      : left_(bounding_box.left_),
        top_(bounding_box.top_),
        right_(bounding_box.right_),
        bottom_(bounding_box.bottom_) {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPStoolsPSandroidPStestPSjniPSobject_trackingPSgeomDTh mht_12(mht_12_v, 303, "", "./tensorflow/tools/android/test/jni/object_tracking/geom.h", "BoundingBox");

    SCHECK(left_ < right_, "Bounds out of whack! %.2f vs %.2f!", left_, right_);
    SCHECK(top_ < bottom_, "Bounds out of whack! %.2f vs %.2f!", top_, bottom_);
  }

  BoundingBox(const float left,
              const float top,
              const float right,
              const float bottom)
      : left_(left),
        top_(top),
        right_(right),
        bottom_(bottom) {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPStoolsPSandroidPStestPSjniPSobject_trackingPSgeomDTh mht_13(mht_13_v, 318, "", "./tensorflow/tools/android/test/jni/object_tracking/geom.h", "BoundingBox");

    SCHECK(left_ < right_, "Bounds out of whack! %.2f vs %.2f!", left_, right_);
    SCHECK(top_ < bottom_, "Bounds out of whack! %.2f vs %.2f!", top_, bottom_);
  }

  BoundingBox(const Point2f& point1, const Point2f& point2)
      : left_(MIN(point1.x, point2.x)),
        top_(MIN(point1.y, point2.y)),
        right_(MAX(point1.x, point2.x)),
        bottom_(MAX(point1.y, point2.y)) {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPStoolsPSandroidPStestPSjniPSobject_trackingPSgeomDTh mht_14(mht_14_v, 330, "", "./tensorflow/tools/android/test/jni/object_tracking/geom.h", "BoundingBox");
}

  inline void CopyToArray(float* const bounds_array) const {
   std::vector<std::string> mht_15_v;
   MHTracer_DTPStensorflowPStoolsPSandroidPStestPSjniPSobject_trackingPSgeomDTh mht_15(mht_15_v, 335, "", "./tensorflow/tools/android/test/jni/object_tracking/geom.h", "CopyToArray");

    bounds_array[0] = left_;
    bounds_array[1] = top_;
    bounds_array[2] = right_;
    bounds_array[3] = bottom_;
  }

  inline float GetWidth() const {
   std::vector<std::string> mht_16_v;
   MHTracer_DTPStensorflowPStoolsPSandroidPStestPSjniPSobject_trackingPSgeomDTh mht_16(mht_16_v, 345, "", "./tensorflow/tools/android/test/jni/object_tracking/geom.h", "GetWidth");

    return right_ - left_;
  }

  inline float GetHeight() const {
   std::vector<std::string> mht_17_v;
   MHTracer_DTPStensorflowPStoolsPSandroidPStestPSjniPSobject_trackingPSgeomDTh mht_17(mht_17_v, 352, "", "./tensorflow/tools/android/test/jni/object_tracking/geom.h", "GetHeight");

    return bottom_ - top_;
  }

  inline float GetArea() const {
   std::vector<std::string> mht_18_v;
   MHTracer_DTPStensorflowPStoolsPSandroidPStestPSjniPSobject_trackingPSgeomDTh mht_18(mht_18_v, 359, "", "./tensorflow/tools/android/test/jni/object_tracking/geom.h", "GetArea");

    const float width = GetWidth();
    const float height = GetHeight();
    if (width <= 0 || height <= 0) {
      return 0.0f;
    }

    return width * height;
  }

  inline Point2f GetCenter() const {
   std::vector<std::string> mht_19_v;
   MHTracer_DTPStensorflowPStoolsPSandroidPStestPSjniPSobject_trackingPSgeomDTh mht_19(mht_19_v, 372, "", "./tensorflow/tools/android/test/jni/object_tracking/geom.h", "GetCenter");

    return Point2f((left_ + right_) / 2.0f,
                   (top_ + bottom_) / 2.0f);
  }

  inline bool ValidBox() const {
   std::vector<std::string> mht_20_v;
   MHTracer_DTPStensorflowPStoolsPSandroidPStestPSjniPSobject_trackingPSgeomDTh mht_20(mht_20_v, 380, "", "./tensorflow/tools/android/test/jni/object_tracking/geom.h", "ValidBox");

    return GetArea() > 0.0f;
  }

  // Returns a bounding box created from the overlapping area of these two.
  inline BoundingBox Intersect(const BoundingBox& that) const {
   std::vector<std::string> mht_21_v;
   MHTracer_DTPStensorflowPStoolsPSandroidPStestPSjniPSobject_trackingPSgeomDTh mht_21(mht_21_v, 388, "", "./tensorflow/tools/android/test/jni/object_tracking/geom.h", "Intersect");

    const float new_left = MAX(this->left_, that.left_);
    const float new_right = MIN(this->right_, that.right_);

    if (new_left >= new_right) {
      return BoundingBox();
    }

    const float new_top = MAX(this->top_, that.top_);
    const float new_bottom = MIN(this->bottom_, that.bottom_);

    if (new_top >= new_bottom) {
      return BoundingBox();
    }

    return BoundingBox(new_left, new_top,  new_right, new_bottom);
  }

  // Returns a bounding box that can contain both boxes.
  inline BoundingBox Union(const BoundingBox& that) const {
   std::vector<std::string> mht_22_v;
   MHTracer_DTPStensorflowPStoolsPSandroidPStestPSjniPSobject_trackingPSgeomDTh mht_22(mht_22_v, 410, "", "./tensorflow/tools/android/test/jni/object_tracking/geom.h", "Union");

    return BoundingBox(MIN(this->left_, that.left_),
                       MIN(this->top_, that.top_),
                       MAX(this->right_, that.right_),
                       MAX(this->bottom_, that.bottom_));
  }

  inline float PascalScore(const BoundingBox& that) const {
   std::vector<std::string> mht_23_v;
   MHTracer_DTPStensorflowPStoolsPSandroidPStestPSjniPSobject_trackingPSgeomDTh mht_23(mht_23_v, 420, "", "./tensorflow/tools/android/test/jni/object_tracking/geom.h", "PascalScore");

    SCHECK(GetArea() > 0.0f, "Empty bounding box!");
    SCHECK(that.GetArea() > 0.0f, "Empty bounding box!");

    const float intersect_area = this->Intersect(that).GetArea();

    if (intersect_area <= 0) {
      return 0;
    }

    const float score =
        intersect_area / (GetArea() + that.GetArea() - intersect_area);
    SCHECK(InRange(score, 0.0f, 1.0f), "Invalid score! %.2f", score);
    return score;
  }

  inline bool Intersects(const BoundingBox& that) const {
   std::vector<std::string> mht_24_v;
   MHTracer_DTPStensorflowPStoolsPSandroidPStestPSjniPSobject_trackingPSgeomDTh mht_24(mht_24_v, 439, "", "./tensorflow/tools/android/test/jni/object_tracking/geom.h", "Intersects");

    return InRange(that.left_, left_, right_)
        || InRange(that.right_, left_, right_)
        || InRange(that.top_, top_, bottom_)
        || InRange(that.bottom_, top_, bottom_);
  }

  // Returns whether another bounding box is completely inside of this bounding
  // box. Sharing edges is ok.
  inline bool Contains(const BoundingBox& that) const {
   std::vector<std::string> mht_25_v;
   MHTracer_DTPStensorflowPStoolsPSandroidPStestPSjniPSobject_trackingPSgeomDTh mht_25(mht_25_v, 451, "", "./tensorflow/tools/android/test/jni/object_tracking/geom.h", "Contains");

    return that.left_ >= left_ &&
        that.right_ <= right_ &&
        that.top_ >= top_ &&
        that.bottom_ <= bottom_;
  }

  inline bool Contains(const Point2f& point) const {
   std::vector<std::string> mht_26_v;
   MHTracer_DTPStensorflowPStoolsPSandroidPStestPSjniPSobject_trackingPSgeomDTh mht_26(mht_26_v, 461, "", "./tensorflow/tools/android/test/jni/object_tracking/geom.h", "Contains");

    return InRange(point.x, left_, right_) && InRange(point.y, top_, bottom_);
  }

  inline void Shift(const Point2f shift_amount) {
   std::vector<std::string> mht_27_v;
   MHTracer_DTPStensorflowPStoolsPSandroidPStestPSjniPSobject_trackingPSgeomDTh mht_27(mht_27_v, 468, "", "./tensorflow/tools/android/test/jni/object_tracking/geom.h", "Shift");

    left_ += shift_amount.x;
    top_ += shift_amount.y;
    right_ += shift_amount.x;
    bottom_ += shift_amount.y;
  }

  inline void ScaleOrigin(const float scale_x, const float scale_y) {
   std::vector<std::string> mht_28_v;
   MHTracer_DTPStensorflowPStoolsPSandroidPStestPSjniPSobject_trackingPSgeomDTh mht_28(mht_28_v, 478, "", "./tensorflow/tools/android/test/jni/object_tracking/geom.h", "ScaleOrigin");

    left_ *= scale_x;
    right_ *= scale_x;
    top_ *= scale_y;
    bottom_ *= scale_y;
  }

  inline void Scale(const float scale_x, const float scale_y) {
   std::vector<std::string> mht_29_v;
   MHTracer_DTPStensorflowPStoolsPSandroidPStestPSjniPSobject_trackingPSgeomDTh mht_29(mht_29_v, 488, "", "./tensorflow/tools/android/test/jni/object_tracking/geom.h", "Scale");

    const Point2f center = GetCenter();
    const float half_width = GetWidth() / 2.0f;
    const float half_height = GetHeight() / 2.0f;

    left_ = center.x - half_width * scale_x;
    right_ = center.x + half_width * scale_x;

    top_ = center.y - half_height * scale_y;
    bottom_ = center.y + half_height * scale_y;
  }

  float left_;
  float top_;
  float right_;
  float bottom_;
};
inline std::ostream& operator<<(std::ostream& stream, const BoundingBox& box) {
   std::vector<std::string> mht_30_v;
   MHTracer_DTPStensorflowPStoolsPSandroidPStestPSjniPSobject_trackingPSgeomDTh mht_30(mht_30_v, 508, "", "./tensorflow/tools/android/test/jni/object_tracking/geom.h", "operator<<");

  stream << "[" << box.left_ << " - " << box.right_
         << ", " << box.top_ << " - " << box.bottom_
         << ",  w:" << box.GetWidth() << " h:" << box.GetHeight() << "]";
  return stream;
}


class BoundingSquare {
 public:
  BoundingSquare(const float x, const float y, const float size)
      : x_(x), y_(y), size_(size) {
   std::vector<std::string> mht_31_v;
   MHTracer_DTPStensorflowPStoolsPSandroidPStestPSjniPSobject_trackingPSgeomDTh mht_31(mht_31_v, 522, "", "./tensorflow/tools/android/test/jni/object_tracking/geom.h", "BoundingSquare");
}

  explicit BoundingSquare(const BoundingBox& box)
      : x_(box.left_), y_(box.top_), size_(box.GetWidth()) {
   std::vector<std::string> mht_32_v;
   MHTracer_DTPStensorflowPStoolsPSandroidPStestPSjniPSobject_trackingPSgeomDTh mht_32(mht_32_v, 528, "", "./tensorflow/tools/android/test/jni/object_tracking/geom.h", "BoundingSquare");

#ifdef SANITY_CHECKS
    if (std::abs(box.GetWidth() - box.GetHeight()) > 0.1f) {
      LOG(WARNING) << "This is not a square: " << box << std::endl;
    }
#endif
  }

  inline BoundingBox ToBoundingBox() const {
   std::vector<std::string> mht_33_v;
   MHTracer_DTPStensorflowPStoolsPSandroidPStestPSjniPSobject_trackingPSgeomDTh mht_33(mht_33_v, 539, "", "./tensorflow/tools/android/test/jni/object_tracking/geom.h", "ToBoundingBox");

    return BoundingBox(x_, y_, x_ + size_, y_ + size_);
  }

  inline bool ValidBox() {
   std::vector<std::string> mht_34_v;
   MHTracer_DTPStensorflowPStoolsPSandroidPStestPSjniPSobject_trackingPSgeomDTh mht_34(mht_34_v, 546, "", "./tensorflow/tools/android/test/jni/object_tracking/geom.h", "ValidBox");

    return size_ > 0.0f;
  }

  inline void Shift(const Point2f shift_amount) {
   std::vector<std::string> mht_35_v;
   MHTracer_DTPStensorflowPStoolsPSandroidPStestPSjniPSobject_trackingPSgeomDTh mht_35(mht_35_v, 553, "", "./tensorflow/tools/android/test/jni/object_tracking/geom.h", "Shift");

    x_ += shift_amount.x;
    y_ += shift_amount.y;
  }

  inline void Scale(const float scale) {
   std::vector<std::string> mht_36_v;
   MHTracer_DTPStensorflowPStoolsPSandroidPStestPSjniPSobject_trackingPSgeomDTh mht_36(mht_36_v, 561, "", "./tensorflow/tools/android/test/jni/object_tracking/geom.h", "Scale");

    const float new_size = size_ * scale;
    const float position_diff = (new_size - size_) / 2.0f;
    x_ -= position_diff;
    y_ -= position_diff;
    size_ = new_size;
  }

  float x_;
  float y_;
  float size_;
};
inline std::ostream& operator<<(std::ostream& stream,
                                const BoundingSquare& square) {
   std::vector<std::string> mht_37_v;
   MHTracer_DTPStensorflowPStoolsPSandroidPStestPSjniPSobject_trackingPSgeomDTh mht_37(mht_37_v, 577, "", "./tensorflow/tools/android/test/jni/object_tracking/geom.h", "operator<<");

  stream << "[" << square.x_ << "," << square.y_ << " " << square.size_ << "]";
  return stream;
}


inline BoundingSquare GetCenteredSquare(const BoundingBox& original_box,
                                        const float size) {
   std::vector<std::string> mht_38_v;
   MHTracer_DTPStensorflowPStoolsPSandroidPStestPSjniPSobject_trackingPSgeomDTh mht_38(mht_38_v, 587, "", "./tensorflow/tools/android/test/jni/object_tracking/geom.h", "GetCenteredSquare");

  const float width_diff = (original_box.GetWidth() - size) / 2.0f;
  const float height_diff = (original_box.GetHeight() - size) / 2.0f;
  return BoundingSquare(original_box.left_ + width_diff,
                        original_box.top_ + height_diff,
                        size);
}

inline BoundingSquare GetCenteredSquare(const BoundingBox& original_box) {
   std::vector<std::string> mht_39_v;
   MHTracer_DTPStensorflowPStoolsPSandroidPStestPSjniPSobject_trackingPSgeomDTh mht_39(mht_39_v, 598, "", "./tensorflow/tools/android/test/jni/object_tracking/geom.h", "GetCenteredSquare");

  return GetCenteredSquare(
      original_box, MIN(original_box.GetWidth(), original_box.GetHeight()));
}

}  // namespace tf_tracking

#endif  // TENSORFLOW_EXAMPLES_ANDROID_JNI_OBJECT_TRACKING_GEOM_H_
