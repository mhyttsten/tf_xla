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

#ifndef TENSORFLOW_CORE_KERNELS_GATHER_FUNCTOR_H_
#define TENSORFLOW_CORE_KERNELS_GATHER_FUNCTOR_H_
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
class MHTracer_DTPStensorflowPScorePSkernelsPSgather_functorDTh {
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
   MHTracer_DTPStensorflowPScorePSkernelsPSgather_functorDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSkernelsPSgather_functorDTh() {
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


#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

#include "tensorflow/core/framework/bounds_check.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/framework/type_traits.h"
#include "tensorflow/core/framework/variant.h"
#include "tensorflow/core/platform/prefetch.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/util/work_sharder.h"

namespace tensorflow {
typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;

namespace functor {

// Helper method to copy using memcpy.
template <typename T, typename Index, typename SliceIndex,
          SliceIndex static_slice_elems>
SliceIndex HandleCopies(OpKernelContext* ctx,
                        typename TTypes<T, 3>::ConstTensor params,
                        typename TTypes<Index>::ConstFlat indices,
                        SliceIndex slice_elems,
                        typename TTypes<T, 3>::Tensor out) {
  const SliceIndex indices_size = static_cast<SliceIndex>(indices.dimension(0));
  const SliceIndex batch_size = static_cast<SliceIndex>(params.dimension(0));
  const Index limit = static_cast<Index>(params.dimension(1));
  T* out_base = out.data();
  const T* params_base = params.data();
  if (static_slice_elems >= 0) {
    // Give compiler static knowledge of the number of elements/bytes
    slice_elems = static_slice_elems;
  }
  // Compute slice_bytes here so that static knowledge is available
  const size_t slice_bytes = slice_elems * sizeof(T);
  auto* worker_threads = ctx->device()->tensorflow_cpu_worker_threads();
  mutex mu;
  // Store the value of invalidate index for printing error information, it's a
  // shared variable.
  SliceIndex result = -1;
  auto work = [&](int64_t start, int64_t end) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSgather_functorDTh mht_0(mht_0_v, 229, "", "./tensorflow/core/kernels/gather_functor.h", "lambda");

    SliceIndex batch_idx = static_cast<SliceIndex>(start / indices_size);
    SliceIndex indices_idx = static_cast<SliceIndex>(start % indices_size);
    SliceIndex batch_idx_end = static_cast<SliceIndex>(end / indices_size);
    SliceIndex indices_idx_end = static_cast<SliceIndex>(end % indices_size);

    while ((batch_idx < batch_idx_end) ||
           (batch_idx == batch_idx_end && indices_idx < indices_idx_end)) {
      SliceIndex i_next = indices_idx + 1;
      SliceIndex b_next = batch_idx + 1;
      const Index index = internal::SubtleMustCopy(indices(indices_idx));
      if (!FastBoundsCheck(index, limit)) {
        mutex_lock l(mu);
        result = indices_idx;
        return;
      }
      if ((batch_idx == batch_idx_end && i_next < indices_idx_end) ||
          (i_next < indices_size)) {
        port::prefetch<port::PREFETCH_HINT_T0>(
            &params(batch_idx, indices(i_next), 0));
        port::prefetch<port::PREFETCH_HINT_T0>(&out(batch_idx, i_next, 0));
        b_next = batch_idx;
      } else if (b_next <= batch_idx_end) {
        port::prefetch<port::PREFETCH_HINT_T0>(&params(b_next, indices(0), 0));
        port::prefetch<port::PREFETCH_HINT_T0>(&out(b_next, 0, 0));
        i_next = 0;
      }
      // Copy using memcpy if possible, otherwise an Eigen loop
      // TODO(cwhipkey): avoid linking to framework to get Allocator (to improve
      // ahead-of-time compilation binary size).
      if (is_simple_type<T>::value) {
        // Avoid auto-promotion to Index from SliceIndex by casting.
        memcpy(
            out_base + (batch_idx * indices_size + indices_idx) * slice_elems,
            params_base + (batch_idx * static_cast<SliceIndex>(limit) +
                           static_cast<SliceIndex>(index)) *
                              slice_elems,
            slice_bytes);
      } else {
        // For non-"simple" types (e.g. strings).
        out.template chip<0>(batch_idx).template chip<0>(indices_idx) =
            params.template chip<0>(batch_idx).template chip<0>(index);
      }
      indices_idx = i_next;
      batch_idx = b_next;
    }
  };

  Shard(worker_threads->num_threads, worker_threads->workers,
        batch_size * indices_size, slice_elems * sizeof(T), work);
  return result;
}

template <typename T, typename Index>
struct GatherFunctorCPU {
  int64_t operator()(OpKernelContext* ctx,
                     typename TTypes<T, 3>::ConstTensor params,
                     typename TTypes<Index>::ConstFlat indices,
                     typename TTypes<T, 3>::Tensor out) {
    const int64_t indices_size = indices.size();
    const int64_t slice_size = out.dimension(2);
    int64_t bad_i;

    const int64_t batch_size = params.dimension(0);

    bool use_large = (slice_size > std::numeric_limits<int32>::max() ||
                      params.size() > std::numeric_limits<int32>::max() ||
                      indices_size > std::numeric_limits<int32>::max() ||
                      batch_size * indices_size * slice_size >
                          std::numeric_limits<int32>::max());
#define CALL(elems)                                                        \
  do {                                                                     \
    if (use_large) {                                                       \
      bad_i = HandleCopies<T, Index, int64_t, elems>(ctx, params, indices, \
                                                     slice_size, out);     \
    } else {                                                               \
      const int32 small_slice = static_cast<int32>(slice_size);            \
      bad_i = HandleCopies<T, Index, int32, elems>(ctx, params, indices,   \
                                                   small_slice, out);      \
    }                                                                      \
  } while (0)

    if (slice_size == 10)
      CALL(10);
    else if (slice_size == 20)
      CALL(20);
    else
      CALL(-1);
#undef CALL

    return bad_i;
  }
};

template <typename Device, typename T, typename Index>
struct GatherFunctor {
  int64_t operator()(OpKernelContext* ctx,
                     typename TTypes<T, 3>::ConstTensor params,
                     typename TTypes<Index>::ConstFlat indices,
                     typename TTypes<T, 3>::Tensor out);
};

template <typename T, typename Index>
struct GatherFunctor<CPUDevice, T, Index> {
  int64_t operator()(OpKernelContext* ctx,
                     typename TTypes<T, 3>::ConstTensor params,
                     typename TTypes<Index>::ConstFlat indices,
                     typename TTypes<T, 3>::Tensor out) {
    return GatherFunctorCPU<T, Index>()(ctx, params, indices, out);
  }
};

template <typename Index>
struct GatherFunctor<GPUDevice, Variant, Index> {
  int64_t operator()(OpKernelContext* ctx,
                     typename TTypes<Variant, 3>::ConstTensor params,
                     typename TTypes<Index>::ConstFlat indices,
                     typename TTypes<Variant, 3>::Tensor out) {
    return GatherFunctorCPU<Variant, Index>()(ctx, params, indices, out);
  }
};

}  // namespace functor
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_KERNELS_GATHER_FUNCTOR_H_
