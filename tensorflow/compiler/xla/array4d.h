/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_COMPILER_XLA_ARRAY4D_H_
#define TENSORFLOW_COMPILER_XLA_ARRAY4D_H_
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
class MHTracer_DTPStensorflowPScompilerPSxlaPSarray4dDTh {
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
   MHTracer_DTPStensorflowPScompilerPSxlaPSarray4dDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSxlaPSarray4dDTh() {
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


#include <algorithm>
#include <functional>
#include <initializer_list>
#include <iterator>
#include <memory>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include "absl/strings/str_cat.h"
#include "absl/types/span.h"
#include "tensorflow/compiler/xla/array.h"
#include "tensorflow/compiler/xla/array2d.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/core/platform/logging.h"

namespace xla {

// Simple 4D array structure, similar in form to Array2D, for use primarily in
// testing and describing to XLA APIs values in the 4D array structures used
// in convolutions.
//
// The data layout is, in order from major to minor:
//
//    First dimension: plane, batch, n1
//   Second dimension: depth, feature, z, n2
//    Third dimension: height, y, n3
//   Fourth dimension: width, x, n4
//
// These dimensions are referred to by various names, so that is why
// more than one name is given above. See operator() for the exact
// calculation of 1d indices from 4d indices.
template <typename T>
class Array4D : public Array<T> {
 public:
  Array4D() : Array<T>(std::vector<int64_t>{0, 0, 0, 0}) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSarray4dDTh mht_0(mht_0_v, 224, "", "./tensorflow/compiler/xla/array4d.h", "Array4D");
}

  // Creates a 4D array, uninitialized values.
  Array4D(int64_t planes, int64_t depth, int64_t height, int64_t width)
      : Array<T>(std::vector<int64_t>{planes, depth, height, width}) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSarray4dDTh mht_1(mht_1_v, 231, "", "./tensorflow/compiler/xla/array4d.h", "Array4D");
}

  // Creates a 4D array, initialized to value.
  Array4D(int64_t planes, int64_t depth, int64_t height, int64_t width, T value)
      : Array<T>(std::vector<int64_t>{planes, depth, height, width}, value) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSarray4dDTh mht_2(mht_2_v, 238, "", "./tensorflow/compiler/xla/array4d.h", "Array4D");
}

  // Creates a 4D array, filled with values.
  //
  // We need to set a default type for Container so that code like
  // Array4D(1, 1, 1, 1, {1}) will work. The template cannot infer the
  // initializer_list type in that case without this default.
  template <typename Container = std::initializer_list<T>>
  Array4D(int64_t planes, int64_t depth, int64_t height, int64_t width,
          const Container& values)
      : Array4D(planes, depth, height, width) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSarray4dDTh mht_3(mht_3_v, 251, "", "./tensorflow/compiler/xla/array4d.h", "Array4D");

    this->SetValues(values);
  }

  // Construct an Array4D with the given nested initializer list.
  Array4D(std::initializer_list<std::initializer_list<
              std::initializer_list<std::initializer_list<T>>>>
              values)
      : Array<T>(values) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSarray4dDTh mht_4(mht_4_v, 262, "", "./tensorflow/compiler/xla/array4d.h", "Array4D");
}

  // Creates an array of a floating-point type (half, bfloat16, float,
  // or double) from the given nested initializer list of float values.
  template <typename T2, typename = typename std::enable_if<
                             (std::is_same<T, Eigen::half>::value ||
                              std::is_same<T, bfloat16>::value ||
                              std::is_same<T, float>::value ||
                              std::is_same<T, double>::value) &&
                             std::is_same<T2, float>::value>::type>
  Array4D(std::initializer_list<std::initializer_list<
              std::initializer_list<std::initializer_list<T2>>>>
              values)
      : Array<T>(values) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSarray4dDTh mht_5(mht_5_v, 278, "", "./tensorflow/compiler/xla/array4d.h", "Array4D");
}

  // Numerically-named aliases for the various dimensions. This matches the
  // dimension names used in array3d.
  int64_t n4() const {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSarray4dDTh mht_6(mht_6_v, 285, "", "./tensorflow/compiler/xla/array4d.h", "n4");
 return this->dim(3); }
  int64_t n3() const {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSarray4dDTh mht_7(mht_7_v, 289, "", "./tensorflow/compiler/xla/array4d.h", "n3");
 return this->dim(2); }
  int64_t n2() const {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSarray4dDTh mht_8(mht_8_v, 293, "", "./tensorflow/compiler/xla/array4d.h", "n2");
 return this->dim(1); }
  int64_t n1() const {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSarray4dDTh mht_9(mht_9_v, 297, "", "./tensorflow/compiler/xla/array4d.h", "n1");
 return this->dim(0); }

  int64_t width() const {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSarray4dDTh mht_10(mht_10_v, 302, "", "./tensorflow/compiler/xla/array4d.h", "width");
 return this->dim(3); }
  int64_t height() const {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSarray4dDTh mht_11(mht_11_v, 306, "", "./tensorflow/compiler/xla/array4d.h", "height");
 return this->dim(2); }
  int64_t depth() const {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSarray4dDTh mht_12(mht_12_v, 310, "", "./tensorflow/compiler/xla/array4d.h", "depth");
 return this->dim(1); }
  int64_t planes() const {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSarray4dDTh mht_13(mht_13_v, 314, "", "./tensorflow/compiler/xla/array4d.h", "planes");
 return this->dim(0); }

  // Fills all of the {p,z} with the array provided, which specifies {y,x}.
  void FillWithYX(const Array2D<T>& value) {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSarray4dDTh mht_14(mht_14_v, 320, "", "./tensorflow/compiler/xla/array4d.h", "FillWithYX");

    CHECK_EQ(value.height(), height());
    CHECK_EQ(value.width(), width());
    for (int64_t plane = 0; plane < planes(); ++plane) {
      for (int64_t depth = 0; depth < this->depth(); ++depth) {
        for (int64_t height = 0; height < this->height(); ++height) {
          for (int64_t width = 0; width < this->width(); ++width) {
            (*this)(plane, depth, height, width) = value(height, width);
          }
        }
      }
    }
  }

  // Fills all of the {p,x} with the array provided, which specifies {z,y}.
  void FillWithZY(const Array2D<T>& value) {
   std::vector<std::string> mht_15_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSarray4dDTh mht_15(mht_15_v, 338, "", "./tensorflow/compiler/xla/array4d.h", "FillWithZY");

    CHECK_EQ(value.height(), depth());
    CHECK_EQ(value.width(), height());
    for (int64_t plane = 0; plane < planes(); ++plane) {
      for (int64_t depth = 0; depth < this->depth(); ++depth) {
        for (int64_t height = 0; height < this->height(); ++height) {
          for (int64_t width = 0; width < this->width(); ++width) {
            (*this)(plane, depth, height, width) = value(depth, height);
          }
        }
      }
    }
  }

  // Fills all of the {x,y} with the array provided, which specifies {p,z}.
  void FillWithPZ(const Array2D<T>& value) {
   std::vector<std::string> mht_16_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSarray4dDTh mht_16(mht_16_v, 356, "", "./tensorflow/compiler/xla/array4d.h", "FillWithPZ");

    CHECK_EQ(value.height(), planes());
    CHECK_EQ(value.width(), depth());
    for (int64_t height = 0; height < this->height(); ++height) {
      for (int64_t width = 0; width < this->width(); ++width) {
        for (int64_t plane = 0; plane < planes(); ++plane) {
          for (int64_t depth = 0; depth < this->depth(); ++depth) {
            (*this)(plane, depth, height, width) = value(plane, depth);
          }
        }
      }
    }
  }

  // Fills each of the minor-dim matrices with a number designating which minor
  // dim matrix is enclosed by the shape.
  void FillWithMinorDimNum() {
   std::vector<std::string> mht_17_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSarray4dDTh mht_17(mht_17_v, 375, "", "./tensorflow/compiler/xla/array4d.h", "FillWithMinorDimNum");

    LOG(INFO) << "width: " << this->width();
    LOG(INFO) << "height: " << this->height();
    LOG(INFO) << "depth: " << this->depth();
    LOG(INFO) << "planes: " << this->planes();
    for (int64_t height = 0; height < this->height(); ++height) {
      for (int64_t width = 0; width < this->width(); ++width) {
        for (int64_t plane = 0; plane < planes(); ++plane) {
          for (int64_t depth = 0; depth < this->depth(); ++depth) {
            float this_val = plane * this->depth() + depth;
            (*this)(plane, depth, height, width) = this_val;
          }
        }
      }
    }
  }
};

}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_ARRAY4D_H_
