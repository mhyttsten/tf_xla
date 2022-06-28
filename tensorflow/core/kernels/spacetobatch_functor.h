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

#ifndef TENSORFLOW_CORE_KERNELS_SPACETOBATCH_FUNCTOR_H_
#define TENSORFLOW_CORE_KERNELS_SPACETOBATCH_FUNCTOR_H_
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
class MHTracer_DTPStensorflowPScorePSkernelsPSspacetobatch_functorDTh {
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
   MHTracer_DTPStensorflowPScorePSkernelsPSspacetobatch_functorDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSkernelsPSspacetobatch_functorDTh() {
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


#include <type_traits>

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/bounds_check.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {

// Maximum number of non-collapsible blocked dimensions supported by the
// {SpaceToBatch,BatchToSpace}ND operation.  To change the limit, modify this
// constant and the TF_SPACETOBATCH_FOR_EACH_NUM_BLOCK_DIMS macro definition
// below.
constexpr int kMaxSpaceToBatchBlockDims = 4;

// Expands to:
//   MACRO(1, ## __VA_ARGS__)
//   ...
//   MACRO(kMaxSpaceToBatchBlockDims, ## __VA_ARGS__)
//
// Note: The space between the number and the comma is necessary for proper GCC
// comma handling: https://gcc.gnu.org/onlinedocs/cpp/Variadic-Macros.html
#define TF_SPACETOBATCH_FOR_EACH_NUM_BLOCK_DIMS(MACRO, ...) \
  MACRO(1 /**/, ##__VA_ARGS__)                              \
  MACRO(2 /**/, ##__VA_ARGS__)                              \
  MACRO(3 /**/, ##__VA_ARGS__)                              \
  MACRO(4 /**/, ##__VA_ARGS__)                              \
  /**/

namespace internal {
namespace spacetobatch {

template <typename InputType, typename OutputType>
void SubtleMustCopyFlatHelper(const Tensor& t, OutputType* output) {
  const int64_t num_elements = t.shape().num_elements();
  output->resize(num_elements);
  auto eigen_vec = t.flat<InputType>();
  for (int64_t i = 0; i < num_elements; ++i) {
    (*output)[i] = SubtleMustCopy(eigen_vec(i));
  }
}

// Copies flat contents of `t` to std::vector-like `*output`, which is resized
// as needed.  `OutputType` may be either `std::vector<int64_t>` or
// `gtl::InlinedVector<int64_t>`.
//
// Precondition: t.dtype() must be either DT_INT32 or DT_INT64.
template <typename OutputType>
void SubtleMustCopyFlat(const Tensor& t, OutputType* output) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSspacetobatch_functorDTh mht_0(mht_0_v, 237, "", "./tensorflow/core/kernels/spacetobatch_functor.h", "SubtleMustCopyFlat");

  if (t.dtype() == DT_INT32) {
    SubtleMustCopyFlatHelper<int32, OutputType>(t, output);
  } else {
    SubtleMustCopyFlatHelper<int64_t, OutputType>(t, output);
  }
}

}  // namespace spacetobatch
}  // namespace internal

namespace functor {

// Functor used by {SpaceToBatch,BatchToSpace}{ND,}Op to do the conversion.
//
// If B2S is false, then this performs the space-to-batch conversion.  If B2S is
// true, then this performs the inverse batch-to-space conversion.
template <typename Device, typename T, int NUM_BLOCK_DIMS, bool B2S = false>
struct SpaceToBatchFunctor {
  using InputT = typename std::conditional<B2S, T, const T>::type;
  using OutputT = typename std::conditional<B2S, const T, T>::type;
  // Implements the space to batch conversion.
  //
  // space_tensor: input tensor of space-to-batch operation.  If B2S = false,
  //     then this is the input to the conversion.  If B2S = true, then this
  //     is the output of the conversion.
  // block_size: array of shape [NUM_BLOCK_DIMS] specifying the block sizes for
  //     dimensions 1 through NUM_BLOCK_DIMS.
  // paddings: row-major array of shape [NUM_BLOCK_DIMS, 2] specifying the
  //     start and end padding for dimensions 1 through NUM_BLOCK_DIMS.
  // batch_tensor: output tensor of the space-to-batch operation.  If
  //     B2S = false, then this is the output of the conversion.  If B2S = true,
  //     then this is the input to the conversion.
  //
  // The caller must ensure that the dimensions of the tensors are correct.
  Status operator()(
      const Device& d,
      typename TTypes<InputT, NUM_BLOCK_DIMS + 2>::Tensor space_tensor,
      const int64_t block_shape[NUM_BLOCK_DIMS],
      const int64_t paddings[NUM_BLOCK_DIMS * 2],
      typename TTypes<OutputT, NUM_BLOCK_DIMS + 2>::Tensor batch_tensor);
};

}  // namespace functor
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_KERNELS_SPACETOBATCH_FUNCTOR_H_
