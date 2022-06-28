/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_CORE_FRAMEWORK_OPS_UTIL_H_
#define TENSORFLOW_CORE_FRAMEWORK_OPS_UTIL_H_
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
class MHTracer_DTPStensorflowPScorePSframeworkPSops_utilDTh {
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
   MHTracer_DTPStensorflowPScorePSframeworkPSops_utilDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSframeworkPSops_utilDTh() {
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


// This file contains utilities for various operations.

#include <array>

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/util/padding.h"

namespace tensorflow {

// Calculates broadcast starting index and size.  For SAME padding, addition
// padding could be applied to right, left, top and bottom.  Depending on the
// current index, input size, kernel size, stride, padding size, the starting
// index and size for broadcast for that dimension are different from the
// current index and kernel size.
// This is mainly used by gradient algorithms for pooling operations.
Status GetBroadcastSize(const int index, const int in_size, const int ksize,
                        const int stride, const int pad_size, int* bindex,
                        int* bsize);

// Converts Brain's Padding to Eigen's PaddingType.
Eigen::PaddingType BrainPadding2EigenPadding(Padding padding);

// Given a shape 's' of a tensor of type T. Returns true iff the
// number of bytes occupied by each dim 0 (i.e., &tensor(i + 1, ...) -
// &tensor(i, ...)) is multiple of EIGEN_MAX_ALIGN_BYTES.
template <typename T>
bool IsInnerDimsSizeAligned(const TensorShape& s) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSops_utilDTh mht_0(mht_0_v, 217, "", "./tensorflow/core/framework/ops_util.h", "IsInnerDimsSizeAligned");

  if (s.dims() == 0) return false;
  const int64_t dim0_size = s.dim_size(0);
  if (dim0_size == 0) return false;
#if EIGEN_MAX_ALIGN_BYTES == 0
  return true;
#else
  const int64_t bytes_per_dim0 = (s.num_elements() / dim0_size) * sizeof(T);
  return bytes_per_dim0 % EIGEN_MAX_ALIGN_BYTES == 0;
#endif
}

// Given a shape 's' of a tensor of type T and the `start` and `end` index of a
// dim 0 slice, returns true iff slice is aligned with respect to original
// tensor. Here aligned implies the address is a multiple of
// EIGEN_MAX_ALIGN_BYTES.
template <typename T>
bool IsDim0SliceAligned(const TensorShape& s, int64_t start,
                        int64_t end_or_size) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSops_utilDTh mht_1(mht_1_v, 238, "", "./tensorflow/core/framework/ops_util.h", "IsDim0SliceAligned");

  if (s.dims() == 1) {
#if EIGEN_MAX_ALIGN_BYTES == 0
    return true;
#else
    bool start_aligned = (start * sizeof(T)) % EIGEN_MAX_ALIGN_BYTES == 0;
    // End is aligned if either the explicit end index is passed and is a
    // a multiple of EIGEN_MAX_ALIGN_BYTES, or the start index is aligned and
    // the size is aligned. So for convenience we can either pass start and
    // index, or start and size.
    bool end_aligned = (end_or_size * sizeof(T)) % EIGEN_MAX_ALIGN_BYTES == 0;
    return start_aligned && end_aligned;
#endif
  } else {
    return IsInnerDimsSizeAligned<T>(s);
  }
}

// Returns <suffix> sanitized to have only [a-zA-Z0-9-_].
std::string SanitizeThreadSuffix(std::string suffix);

// Helper to compute 'strides' given a tensor 'shape'. I.e.,
// strides[i] = prod(shape.dim_size[(i+1):])
template <typename T>
gtl::InlinedVector<T, 8> ComputeStride(const TensorShape& shape) {
  const int ndims = shape.dims();
  gtl::InlinedVector<T, 8> strides(ndims);
  T stride = 1;
  for (int i = ndims - 1; i >= 0; --i) {
    strides[i] = stride;
    stride *= static_cast<T>(shape.dim_size(i));
  }
  return strides;
}

// Helper to compute 'strides' given an Eigen TensorDimensions
template <typename T, typename EigenDimensions>
gtl::InlinedVector<T, 8> ComputeEigenStrides(const EigenDimensions& shape) {
  const int ndims = shape.rank();
  gtl::InlinedVector<T, 8> strides(ndims);
  T stride = 1;
  for (int i = ndims - 1; i >= 0; --i) {
    strides[i] = stride;
    stride *= static_cast<T>(shape[i]);
  }
  return strides;
}

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_FRAMEWORK_OPS_UTIL_H_
