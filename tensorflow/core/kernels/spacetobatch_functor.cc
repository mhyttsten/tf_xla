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
class MHTracer_DTPStensorflowPScorePSkernelsPSspacetobatch_functorDTcc {
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
   MHTracer_DTPStensorflowPScorePSkernelsPSspacetobatch_functorDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSkernelsPSspacetobatch_functorDTcc() {
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

// Specialization of SpaceToBatchFunctor for a CPUDevice.

#define EIGEN_USE_THREADS

#include "tensorflow/core/kernels/spacetobatch_functor.h"

#include "tensorflow/core/framework/register_types.h"

namespace tensorflow {

typedef Eigen::ThreadPoolDevice CPUDevice;

namespace functor {

namespace {

// Implementation of nested loops for SpaceToBatchOpFunctor.
//
// To simplify template implementation given lack of constexpr if, both the
// input and output pointers are non-const.
template <int N, bool B2S>
struct SpaceToBatchHelper {
  template <typename T>
  static void run(T* space_tensor_ptr, const int64_t* space_tensor_shape,
                  const int64_t* space_tensor_strides,
                  const int64_t* block_shape, const int64_t* pad_start,
                  const int64_t* block_offsets,
                  const int64_t* batch_tensor_shape,
                  const int64_t* batch_tensor_strides, T* batch_tensor_ptr) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSspacetobatch_functorDTcc mht_0(mht_0_v, 213, "", "./tensorflow/core/kernels/spacetobatch_functor.cc", "run");

    for (int64_t batch_tensor_pos = 0; batch_tensor_pos < batch_tensor_shape[0];
         ++batch_tensor_pos) {
      const int64_t space_tensor_pos =
          batch_tensor_pos * block_shape[0] + block_offsets[0] - pad_start[0];
      if (space_tensor_pos >= 0 && space_tensor_pos < space_tensor_shape[0]) {
        SpaceToBatchHelper<N - 1, B2S>::run(
            space_tensor_ptr + space_tensor_pos * space_tensor_strides[0],
            space_tensor_shape + 1, space_tensor_strides + 1, block_shape + 1,
            pad_start + 1, block_offsets + 1, batch_tensor_shape + 1,
            batch_tensor_strides + 1, batch_tensor_ptr);
      } else {
        if (B2S == false) {
          // Copy in padding.
          for (int64_t i = 0; i < batch_tensor_strides[0]; ++i) {
            batch_tensor_ptr[i] = static_cast<T>(0);
          }
        }
      }
      batch_tensor_ptr += batch_tensor_strides[0];
    }
  }
};

template <bool B2S>
struct SpaceToBatchHelper<0, B2S> {
  template <typename T>
  static void run(T* space_tensor_ptr, const int64_t* space_tensor_shape,
                  const int64_t* space_tensor_strides,
                  const int64_t* block_shape, const int64_t* pad_start,
                  const int64_t* block_offsets,
                  const int64_t* batch_tensor_shape,
                  const int64_t* batch_tensor_strides, T* batch_tensor_ptr) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSspacetobatch_functorDTcc mht_1(mht_1_v, 248, "", "./tensorflow/core/kernels/spacetobatch_functor.cc", "run");

    for (int64_t i = 0; i < batch_tensor_strides[-1]; ++i) {
      if (B2S == false) {
        batch_tensor_ptr[i] = space_tensor_ptr[i];
      } else {
        space_tensor_ptr[i] = batch_tensor_ptr[i];
      }
    }
  }
};

}  // namespace

template <typename T, int NUM_BLOCK_DIMS, bool B2S>
struct SpaceToBatchFunctor<CPUDevice, T, NUM_BLOCK_DIMS, B2S> {
  using SpaceT = typename std::conditional<B2S, T, const T>::type;
  using BatchT = typename std::conditional<B2S, const T, T>::type;
  Status operator()(
      const CPUDevice& d,
      typename TTypes<SpaceT, NUM_BLOCK_DIMS + 2>::Tensor space_tensor,
      const int64_t block_shape_tensor[NUM_BLOCK_DIMS],
      const int64_t paddings_tensor[NUM_BLOCK_DIMS * 2],
      typename TTypes<BatchT, NUM_BLOCK_DIMS + 2>::Tensor batch_tensor) {
    const int64_t batch_tensor_batch = batch_tensor.dimension(0);

    const int64_t space_tensor_batch = space_tensor.dimension(0);

    // Copy into local array so that the compiler is free to place in a
    // register.
    int64_t pad_start[NUM_BLOCK_DIMS];
    int64_t block_shape[NUM_BLOCK_DIMS];
    int64_t space_tensor_shape[NUM_BLOCK_DIMS],
        batch_tensor_shape[NUM_BLOCK_DIMS];
    for (int block_dim = 0; block_dim < NUM_BLOCK_DIMS; ++block_dim) {
      pad_start[block_dim] = paddings_tensor[block_dim * 2];
      block_shape[block_dim] = block_shape_tensor[block_dim];
      space_tensor_shape[block_dim] = space_tensor.dimension(block_dim + 1);
      batch_tensor_shape[block_dim] = batch_tensor.dimension(block_dim + 1);
    }

    int64_t space_tensor_strides[NUM_BLOCK_DIMS + 2],
        batch_tensor_strides[NUM_BLOCK_DIMS + 2];
    space_tensor_strides[NUM_BLOCK_DIMS + 1] =
        batch_tensor_strides[NUM_BLOCK_DIMS + 1] = 1;
    for (int dim = NUM_BLOCK_DIMS; dim >= 0; --dim) {
      space_tensor_strides[dim] =
          space_tensor_strides[dim + 1] * space_tensor.dimension(dim + 1);
      batch_tensor_strides[dim] =
          batch_tensor_strides[dim + 1] * batch_tensor.dimension(dim + 1);
    }

    // Use non-const pointers for both input and output to simplify template
    // implementation given lack of constexpr if.
    T* space_tensor_ptr = const_cast<T*>(space_tensor.data());
    T* batch_tensor_ptr = const_cast<T*>(batch_tensor.data());

    for (int64_t batch_tensor_b = 0; batch_tensor_b < batch_tensor_batch;
         ++batch_tensor_b) {
      const int64_t space_tensor_b = batch_tensor_b % space_tensor_batch;
      int64_t block_index = batch_tensor_b / space_tensor_batch;
      int64_t block_offsets[NUM_BLOCK_DIMS];
      for (int block_dim = NUM_BLOCK_DIMS - 1; block_dim >= 0; --block_dim) {
        // Skip unnecessary remainder operation for block_dim == 0.
        block_offsets[block_dim] =
            block_dim > 0 ? block_index % block_shape[block_dim] : block_index;
        block_index /= block_shape[block_dim];
      }

      // The compiler should inline the nested loops generated by this template.
      SpaceToBatchHelper<NUM_BLOCK_DIMS, B2S>::run(
          space_tensor_ptr + space_tensor_b * space_tensor_strides[0],
          space_tensor_shape, &space_tensor_strides[1], block_shape, pad_start,
          block_offsets, batch_tensor_shape, &batch_tensor_strides[1],
          batch_tensor_ptr + batch_tensor_b * batch_tensor_strides[0]);
    }
    return Status::OK();
  }
};

// Instantiate.
#define INSTANTIATE(NUM_BLOCK_DIMS, T)                                      \
  template struct SpaceToBatchFunctor<CPUDevice, T, NUM_BLOCK_DIMS, false>; \
  template struct SpaceToBatchFunctor<CPUDevice, T, NUM_BLOCK_DIMS, true>;  \
  /**/

#define INSTANTIATE_FOR_T(T) \
  TF_SPACETOBATCH_FOR_EACH_NUM_BLOCK_DIMS(INSTANTIATE, T)

TF_CALL_REAL_NUMBER_TYPES(INSTANTIATE_FOR_T)

#undef INSTANTIATE_FOR_T
#undef INSTANTIATE

}  // namespace functor
}  // end namespace tensorflow
