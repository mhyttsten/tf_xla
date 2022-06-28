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

#ifndef TENSORFLOW_CORE_KERNELS_DEEP_CONV2D_H_
#define TENSORFLOW_CORE_KERNELS_DEEP_CONV2D_H_
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
class MHTracer_DTPStensorflowPScorePSkernelsPSdeep_conv2dDTh {
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
   MHTracer_DTPStensorflowPScorePSkernelsPSdeep_conv2dDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSkernelsPSdeep_conv2dDTh() {
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


#include "tensorflow/core/framework/types.h"

namespace tensorflow {

class OpKernelContext;

// DeepConv2D is a Conv2D implementation specialized for deep (i.e. large
// in_depth * out_depth product) convolutions (see deep_conv2d.cc for details).

// DeepConv2DTransform is an interface for implementing transforms for
// DeepConv2D. Implementations must specify transform matrices and
// input/output/filter shapes. DeepConv2d computes:
//
//   y = C[Ad * Bg]
//
//   C: output transform matrix
//   A: input data transform matrix
//   B: filter transform matrix
//   d: vectorized 2D data tile
//   g: vectorized 2D filter tile
//   y: vectorized 2D output tile

template <typename T>
class DeepConv2DTransform {
 public:
  virtual ~DeepConv2DTransform() {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdeep_conv2dDTh mht_0(mht_0_v, 213, "", "./tensorflow/core/kernels/deep_conv2d.h", "~DeepConv2DTransform");
}

  virtual void GetFilterTransformMatrix(const int64_t rows, const int64_t cols,
                                        T* transform_matrix) const = 0;

  virtual void GetInputTransformMatrix(const int64_t rows, const int64_t cols,
                                       T* transform_matrix) const = 0;

  virtual void GetOutputTransformMatrix(const int64_t rows, const int64_t cols,
                                        T* transform_matrix) const = 0;

  struct Shape {
    Shape(int64_t r, int64_t c) : rows(r), cols(c) {}
    int64_t rows;
    int64_t cols;
  };

  virtual const Shape& filter_shape() const = 0;
  virtual const Shape& input_shape() const = 0;
  virtual const Shape& output_shape() const = 0;
};

// Conv2D arguments used by DeepConv2D implementation.
struct Conv2DArgs {
  // Input layer dimensions
  int batch;
  int in_rows;
  int in_cols;
  int in_depth;
  int filter_rows;
  int filter_cols;
  int pad_rows;
  int pad_cols;

  // Output layer dimensions
  int out_rows;
  int out_cols;
  int out_depth;

  Conv2DArgs()
      : batch(0),
        in_rows(0),
        in_cols(0),
        in_depth(0),
        filter_rows(0),
        filter_cols(0),
        pad_rows(0),
        pad_cols(0),
        out_rows(0),
        out_cols(0),
        out_depth(0) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdeep_conv2dDTh mht_1(mht_1_v, 266, "", "./tensorflow/core/kernels/deep_conv2d.h", "Conv2DArgs");
}
};

// Returns true if convolution operation specified by function arguments
// can use DeepConv2D implementation, and false otherwise.
// May return false based on parameters, cost, or whether feature is disabled.
bool CanUseDeepConv2D(int stride_rows, int stride_cols, int filter_rows,
                      int filter_cols, int in_depth, int out_depth,
                      int out_rows, int out_cols);

namespace functor {

// Calls DeepConv2D implementation (see deep_conv2d.cc for details).
template <typename Device, typename T>
struct DeepConv2D {
  void operator()(OpKernelContext* ctx, const Conv2DArgs& args, const T* input,
                  const T* filter, T* output);
};

}  // namespace functor

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_KERNELS_DEEP_CONV2D_H_
