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

#ifndef TENSORFLOW_CORE_KERNELS_WINOGRAD_TRANSFORM_H_
#define TENSORFLOW_CORE_KERNELS_WINOGRAD_TRANSFORM_H_
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
class MHTracer_DTPStensorflowPScorePSkernelsPSwinograd_transformDTh {
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
   MHTracer_DTPStensorflowPScorePSkernelsPSwinograd_transformDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSkernelsPSwinograd_transformDTh() {
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


#include "tensorflow/core/kernels/deep_conv2d.h"

namespace tensorflow {

// Winograd DeepConv2DTransform implementation for 3x3 filters.
// Details:
// *) Arithmetic complexity of computations: Shmuel Winograd
// *) Fast Algorithms for Convolutional Neural Networks: Lavin, Gray

template <typename T>
class WinogradTransform : public DeepConv2DTransform<T> {
 public:
  typedef typename DeepConv2DTransform<T>::Shape Shape;

  WinogradTransform()
      : filter_shape_(3, 3), input_shape_(4, 4), output_shape_(2, 2) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSwinograd_transformDTh mht_0(mht_0_v, 203, "", "./tensorflow/core/kernels/winograd_transform.h", "WinogradTransform");
}

  virtual void GetFilterTransformMatrix(const int64_t rows, const int64_t cols,
                                        T* transform_matrix) const;

  virtual void GetInputTransformMatrix(const int64_t rows, const int64_t cols,
                                       T* transform_matrix) const;

  virtual void GetOutputTransformMatrix(const int64_t rows, const int64_t cols,
                                        T* transform_matrix) const;

  virtual const Shape& filter_shape() const {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSwinograd_transformDTh mht_1(mht_1_v, 217, "", "./tensorflow/core/kernels/winograd_transform.h", "filter_shape");
 return filter_shape_; }
  virtual const Shape& input_shape() const {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSwinograd_transformDTh mht_2(mht_2_v, 221, "", "./tensorflow/core/kernels/winograd_transform.h", "input_shape");
 return input_shape_; }
  virtual const Shape& output_shape() const {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSwinograd_transformDTh mht_3(mht_3_v, 225, "", "./tensorflow/core/kernels/winograd_transform.h", "output_shape");
 return output_shape_; }

 private:
  const Shape filter_shape_;
  const Shape input_shape_;
  const Shape output_shape_;
};

// The filter transform matrix is the kronecker product 'M * M' of the
// following matrix 'M':
//
//   [ 1    0   0   ]
//   [ 1/2  1/2 1/2 ]
//   [ 1/2 -1/2 1/2 ]
//   [ 0    0   1   ]
//
// The data layout of 'transform_matrix':
//   [input_tile_spatial_size, filter_spatial_size]
//
template <typename T>
void WinogradTransform<T>::GetFilterTransformMatrix(const int64_t rows,
                                                    const int64_t cols,
                                                    T* transform_matrix) const {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSwinograd_transformDTh mht_4(mht_4_v, 250, "", "./tensorflow/core/kernels/winograd_transform.h", "WinogradTransform<T>::GetFilterTransformMatrix");

  CHECK_GT(rows, 0);
  CHECK_GT(cols, 0);
  memset(transform_matrix, 0, sizeof(T) * rows * cols);

  // Sub matrix [0,0]
  transform_matrix[0 * cols + 0] = T(1.0);

  transform_matrix[1 * cols + 0] = T(0.5);
  transform_matrix[1 * cols + 1] = T(0.5);
  transform_matrix[1 * cols + 2] = T(0.5);

  transform_matrix[2 * cols + 0] = T(0.5);
  transform_matrix[2 * cols + 1] = T(-0.5);
  transform_matrix[2 * cols + 2] = T(0.5);

  transform_matrix[3 * cols + 2] = T(1.0);

  // Sub matrix [1,0]
  transform_matrix[4 * cols + 0] = T(0.5);

  transform_matrix[5 * cols + 0] = T(0.25);
  transform_matrix[5 * cols + 1] = T(0.25);
  transform_matrix[5 * cols + 2] = T(0.25);

  transform_matrix[6 * cols + 0] = T(0.25);
  transform_matrix[6 * cols + 1] = T(-0.25);
  transform_matrix[6 * cols + 2] = T(0.25);

  transform_matrix[7 * cols + 2] = T(0.5);

  // Sub matrix [1,1]
  transform_matrix[4 * cols + 3] = T(0.5);

  transform_matrix[5 * cols + 3] = T(0.25);
  transform_matrix[5 * cols + 4] = T(0.25);
  transform_matrix[5 * cols + 5] = T(0.25);

  transform_matrix[6 * cols + 3] = T(0.25);
  transform_matrix[6 * cols + 4] = T(-0.25);
  transform_matrix[6 * cols + 5] = T(0.25);

  transform_matrix[7 * cols + 5] = T(0.5);

  // Sub matrix [1,2]
  transform_matrix[4 * cols + 6] = T(0.5);

  transform_matrix[5 * cols + 6] = T(0.25);
  transform_matrix[5 * cols + 7] = T(0.25);
  transform_matrix[5 * cols + 8] = T(0.25);

  transform_matrix[6 * cols + 6] = T(0.25);
  transform_matrix[6 * cols + 7] = T(-0.25);
  transform_matrix[6 * cols + 8] = T(0.25);

  transform_matrix[7 * cols + 8] = T(0.5);

  // Sub matrix [2,0]
  transform_matrix[8 * cols + 0] = T(0.5);

  transform_matrix[9 * cols + 0] = T(0.25);
  transform_matrix[9 * cols + 1] = T(0.25);
  transform_matrix[9 * cols + 2] = T(0.25);

  transform_matrix[10 * cols + 0] = T(0.25);
  transform_matrix[10 * cols + 1] = T(-0.25);
  transform_matrix[10 * cols + 2] = T(0.25);

  transform_matrix[11 * cols + 2] = T(0.5);

  // Sub matrix [2,1]
  transform_matrix[8 * cols + 3] = T(-0.5);

  transform_matrix[9 * cols + 3] = T(-0.25);
  transform_matrix[9 * cols + 4] = T(-0.25);
  transform_matrix[9 * cols + 5] = T(-0.25);

  transform_matrix[10 * cols + 3] = T(-0.25);
  transform_matrix[10 * cols + 4] = T(0.25);
  transform_matrix[10 * cols + 5] = T(-0.25);

  transform_matrix[11 * cols + 5] = T(-0.5);

  // Sub matrix [2,2]
  transform_matrix[8 * cols + 6] = T(0.5);

  transform_matrix[9 * cols + 6] = T(0.25);
  transform_matrix[9 * cols + 7] = T(0.25);
  transform_matrix[9 * cols + 8] = T(0.25);

  transform_matrix[10 * cols + 6] = T(0.25);
  transform_matrix[10 * cols + 7] = T(-0.25);
  transform_matrix[10 * cols + 8] = T(0.25);

  transform_matrix[11 * cols + 8] = T(0.5);

  // Sub matrix [3,2]
  transform_matrix[12 * cols + 6] = T(1.0);

  transform_matrix[13 * cols + 6] = T(0.5);
  transform_matrix[13 * cols + 7] = T(0.5);
  transform_matrix[13 * cols + 8] = T(0.5);

  transform_matrix[14 * cols + 6] = T(0.5);
  transform_matrix[14 * cols + 7] = T(-0.5);
  transform_matrix[14 * cols + 8] = T(0.5);

  transform_matrix[15 * cols + 8] = T(1.0);
}

// The input transform matrix is the kronecker product 'M * M' of the
// following matrix 'M':
//
//   [1   0  -1   0]
//   [0   1   1   0]
//   [0  -1   1   0]
//   [0   1   0  -1]
//
// Data layout of 'transform_matrix':
//   [tile_spatial_size, tile_spatial_size]
//
template <typename T>
void WinogradTransform<T>::GetInputTransformMatrix(const int64_t rows,
                                                   const int64_t cols,
                                                   T* transform_matrix) const {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSwinograd_transformDTh mht_5(mht_5_v, 377, "", "./tensorflow/core/kernels/winograd_transform.h", "WinogradTransform<T>::GetInputTransformMatrix");

  CHECK_GT(rows, 0);
  CHECK_GT(cols, 0);
  memset(transform_matrix, 0, sizeof(T) * rows * cols);

  // Sub matrix [0,0]
  transform_matrix[0 * cols + 0] = T(1.0);
  transform_matrix[0 * cols + 2] = T(-1.0);

  transform_matrix[1 * cols + 1] = T(1.0);
  transform_matrix[1 * cols + 2] = T(1.0);

  transform_matrix[2 * cols + 1] = T(-1.0);
  transform_matrix[2 * cols + 2] = T(1.0);

  transform_matrix[3 * cols + 1] = T(1.0);
  transform_matrix[3 * cols + 3] = T(-1.0);

  // Sub matrix [0,2]
  transform_matrix[0 * cols + 8] = T(-1.0);
  transform_matrix[0 * cols + 10] = T(1.0);

  transform_matrix[1 * cols + 9] = T(-1.0);
  transform_matrix[1 * cols + 10] = T(-1.0);

  transform_matrix[2 * cols + 9] = T(1.0);
  transform_matrix[2 * cols + 10] = T(-1.0);

  transform_matrix[3 * cols + 9] = T(-1.0);
  transform_matrix[3 * cols + 11] = T(1.0);

  // Sub matrix [1,1]
  transform_matrix[4 * cols + 4] = T(1.0);
  transform_matrix[4 * cols + 6] = T(-1.0);

  transform_matrix[5 * cols + 5] = T(1.0);
  transform_matrix[5 * cols + 6] = T(1.0);

  transform_matrix[6 * cols + 5] = T(-1.0);
  transform_matrix[6 * cols + 6] = T(1.0);

  transform_matrix[7 * cols + 5] = T(1.0);
  transform_matrix[7 * cols + 7] = T(-1.0);

  // Sub matrix [1,2]
  transform_matrix[4 * cols + 8] = T(1.0);
  transform_matrix[4 * cols + 10] = T(-1.0);

  transform_matrix[5 * cols + 9] = T(1.0);
  transform_matrix[5 * cols + 10] = T(1.0);

  transform_matrix[6 * cols + 9] = T(-1.0);
  transform_matrix[6 * cols + 10] = T(1.0);

  transform_matrix[7 * cols + 9] = T(1.0);
  transform_matrix[7 * cols + 11] = T(-1.0);

  // Sub matrix [2,1]
  transform_matrix[8 * cols + 4] = T(-1.0);
  transform_matrix[8 * cols + 6] = T(1.0);

  transform_matrix[9 * cols + 5] = T(-1.0);
  transform_matrix[9 * cols + 6] = T(-1.0);

  transform_matrix[10 * cols + 5] = T(1.0);
  transform_matrix[10 * cols + 6] = T(-1.0);

  transform_matrix[11 * cols + 5] = T(-1.0);
  transform_matrix[11 * cols + 7] = T(1.0);

  // Sub matrix [2,2]
  transform_matrix[8 * cols + 8] = T(1.0);
  transform_matrix[8 * cols + 10] = T(-1.0);

  transform_matrix[9 * cols + 9] = T(1.0);
  transform_matrix[9 * cols + 10] = T(1.0);

  transform_matrix[10 * cols + 9] = T(-1.0);
  transform_matrix[10 * cols + 10] = T(1.0);

  transform_matrix[11 * cols + 9] = T(1.0);
  transform_matrix[11 * cols + 11] = T(-1.0);

  // Sub matrix [3,1]
  transform_matrix[12 * cols + 4] = T(1.0);
  transform_matrix[12 * cols + 6] = T(-1.0);

  transform_matrix[13 * cols + 5] = T(1.0);
  transform_matrix[13 * cols + 6] = T(1.0);

  transform_matrix[14 * cols + 5] = T(-1.0);
  transform_matrix[14 * cols + 6] = T(1.0);

  transform_matrix[15 * cols + 5] = T(1.0);
  transform_matrix[15 * cols + 7] = T(-1.0);

  // Sub matrix [3,3]
  transform_matrix[12 * cols + 12] = T(-1.0);
  transform_matrix[12 * cols + 14] = T(1.0);

  transform_matrix[13 * cols + 13] = T(-1.0);
  transform_matrix[13 * cols + 14] = T(-1.0);

  transform_matrix[14 * cols + 13] = T(1.0);
  transform_matrix[14 * cols + 14] = T(-1.0);

  transform_matrix[15 * cols + 13] = T(-1.0);
  transform_matrix[15 * cols + 15] = T(1.0);
};

// The output transform matrix is the kronecker product 'M * M' of the
// following matrix 'M':
//
//   [1  1  1  0]
//   [0  1 -1 -1]
//
// Data layout of 'transform_matrix':
//   [out_tile_spatial_size, tile_spatial_size]
//
template <typename T>
void WinogradTransform<T>::GetOutputTransformMatrix(const int64_t rows,
                                                    const int64_t cols,
                                                    T* transform_matrix) const {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSwinograd_transformDTh mht_6(mht_6_v, 502, "", "./tensorflow/core/kernels/winograd_transform.h", "WinogradTransform<T>::GetOutputTransformMatrix");

  CHECK_GT(rows, 0);
  CHECK_GT(cols, 0);
  memset(transform_matrix, 0, sizeof(T) * rows * cols);

  // Sub matrix [0,0]
  transform_matrix[0 * cols + 0] = T(1.0);
  transform_matrix[0 * cols + 1] = T(1.0);
  transform_matrix[0 * cols + 2] = T(1.0);

  transform_matrix[1 * cols + 1] = T(1.0);
  transform_matrix[1 * cols + 2] = T(-1.0);
  transform_matrix[1 * cols + 3] = T(-1.0);

  // Sub matrix [0,1]
  transform_matrix[0 * cols + 4] = T(1.0);
  transform_matrix[0 * cols + 5] = T(1.0);
  transform_matrix[0 * cols + 6] = T(1.0);

  transform_matrix[1 * cols + 5] = T(1.0);
  transform_matrix[1 * cols + 6] = T(-1.0);
  transform_matrix[1 * cols + 7] = T(-1.0);

  // Sub matrix [0,2]
  transform_matrix[0 * cols + 8] = T(1.0);
  transform_matrix[0 * cols + 9] = T(1.0);
  transform_matrix[0 * cols + 10] = T(1.0);

  transform_matrix[1 * cols + 9] = T(1.0);
  transform_matrix[1 * cols + 10] = T(-1.0);
  transform_matrix[1 * cols + 11] = T(-1.0);

  // Sub matrix [1,1]
  transform_matrix[2 * cols + 4] = T(1.0);
  transform_matrix[2 * cols + 5] = T(1.0);
  transform_matrix[2 * cols + 6] = T(1.0);

  transform_matrix[3 * cols + 5] = T(1.0);
  transform_matrix[3 * cols + 6] = T(-1.0);
  transform_matrix[3 * cols + 7] = T(-1.0);

  // Sub matrix [1,2]
  transform_matrix[2 * cols + 8] = T(-1.0);
  transform_matrix[2 * cols + 9] = T(-1.0);
  transform_matrix[2 * cols + 10] = T(-1.0);

  transform_matrix[3 * cols + 9] = T(-1.0);
  transform_matrix[3 * cols + 10] = T(1.0);
  transform_matrix[3 * cols + 11] = T(1.0);

  // Sub matrix [1,3]
  transform_matrix[2 * cols + 12] = T(-1.0);
  transform_matrix[2 * cols + 13] = T(-1.0);
  transform_matrix[2 * cols + 14] = T(-1.0);

  transform_matrix[3 * cols + 13] = T(-1.0);
  transform_matrix[3 * cols + 14] = T(1.0);
  transform_matrix[3 * cols + 15] = T(1.0);
};

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_KERNELS_WINOGRAD_TRANSFORM_H_
