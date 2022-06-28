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

#ifndef TENSORFLOW_CORE_KERNELS_GATHER_ND_OP_CPU_IMPL_H_
#define TENSORFLOW_CORE_KERNELS_GATHER_ND_OP_CPU_IMPL_H_
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
class MHTracer_DTPStensorflowPScorePSkernelsPSgather_nd_op_cpu_implDTh {
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
   MHTracer_DTPStensorflowPScorePSkernelsPSgather_nd_op_cpu_implDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSkernelsPSgather_nd_op_cpu_implDTh() {
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


// Specialization of GatherNdSlice to CPU

#define EIGEN_USE_THREADS

#include <atomic>

#include "tensorflow/core/framework/bounds_check.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/kernels/gather_nd_op.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/mem.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/util/util.h"

namespace tensorflow {

typedef Eigen::ThreadPoolDevice CPUDevice;

namespace generator {

template <typename T, typename Index, int IXDIM>
class GatherNdSliceGenerator {
 public:
  EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE GatherNdSliceGenerator(
      const Index slice_size, typename TTypes<Index>::ConstMatrix Tindices,
      typename TTypes<T, IXDIM + 1>::ConstTensor Tparams,
      typename TTypes<T>::Matrix Tout, std::atomic<Index>* error_loc)
      : slice_size_(slice_size),
        Tindices_(Tindices),
        Tparams_(Tparams),
        Tout_(Tout),
        error_loc_(error_loc) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSgather_nd_op_cpu_implDTh mht_0(mht_0_v, 221, "", "./tensorflow/core/kernels/gather_nd_op_cpu_impl.h", "GatherNdSliceGenerator");
}

  EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE bool GenerateIndices(
      const Index loc, Eigen::array<Eigen::DenseIndex, IXDIM + 1>* ix) const {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSgather_nd_op_cpu_implDTh mht_1(mht_1_v, 227, "", "./tensorflow/core/kernels/gather_nd_op_cpu_impl.h", "GenerateIndices");

    (*ix)[IXDIM] = 0;
    bool out_of_bounds = false;
    for (int i = 0; i < IXDIM; ++i) {
      const Index ix_i = internal::SubtleMustCopy(Tindices_(loc, i));
      (*ix)[i] = ix_i;
      out_of_bounds |= !FastBoundsCheck(ix_i, Tparams_.dimension(i));
    }
    return out_of_bounds;
  }

  EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE int32
  operator()(const Eigen::array<Eigen::DenseIndex, 1>& loc_array) const {
    const Index loc = loc_array[0];
    Eigen::array<Eigen::DenseIndex, IXDIM + 1> ix;
    Eigen::array<Eigen::DenseIndex, 2> ix_out;
    ix_out[0] = loc;
    ix_out[1] = 0;
    const bool out_of_bounds = GenerateIndices(loc, &ix);
    if (TF_PREDICT_FALSE(out_of_bounds)) {
      error_loc_->store(loc);
      std::fill_n(&Tout_(ix_out), slice_size_, T());
    } else {
      std::copy_n(&Tparams_(ix), slice_size_, &Tout_(ix_out));
    }

    return static_cast<int32>(0);  // Return something...
  }

 private:
  const Index slice_size_;
  const typename TTypes<Index>::ConstMatrix Tindices_;
  const typename TTypes<T, IXDIM + 1>::ConstTensor Tparams_;
  mutable typename TTypes<T>::Matrix Tout_;
  std::atomic<Index>* error_loc_;
};

}  // namespace generator

namespace functor {

template <typename T, typename Index, int IXDIM>
struct GatherNdSlice<CPUDevice, T, Index, IXDIM> {
  Index operator()(const CPUDevice& d, const Index slice_size,
                   typename TTypes<int32>::Scalar Tscratch,
                   typename TTypes<T, IXDIM + 1>::ConstTensor Tparams,
                   typename TTypes<Index>::ConstMatrix Tindices,
                   typename TTypes<T>::Matrix Tout) {
    std::atomic<Index> error_loc(-1);
    const Eigen::Index batch_size = Tindices.dimension(0);
    generator::GatherNdSliceGenerator<T, Index, IXDIM> gather_nd_generator(
        slice_size, Tindices, Tparams, Tout, &error_loc);

    auto compute_shard = [&](Eigen::Index begin, Eigen::Index end) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSgather_nd_op_cpu_implDTh mht_2(mht_2_v, 283, "", "./tensorflow/core/kernels/gather_nd_op_cpu_impl.h", "lambda");

      for (Eigen::Index i = begin; i < end; ++i) {
        const Eigen::array<Eigen::Index, 1> loc{i};
        gather_nd_generator(loc);
      }
    };
    Eigen::Index bytes_moved = sizeof(T) * (slice_size + IXDIM);
    auto cost = Eigen::TensorOpCost(bytes_moved /* bytes loaded */,
                                    bytes_moved /* bytes stored */,
                                    slice_size + IXDIM /* compute cycles */);
    d.parallelFor(batch_size, cost, compute_shard);

    // error_loc() returns -1 if there's no out-of-bounds index,
    // otherwise it returns the location of an OOB index in Tindices.
    return error_loc.load();
  }
};

#define REGISTER_GATHER_ND_FULL(T, Index)                                     \
  template Index GatherNdSlice<CPUDevice, T, Index, CPU_PROVIDED_IXDIM>::     \
  operator()(const CPUDevice& d, const Index slice_size,                      \
             typename TTypes<int32>::Scalar Tscratch,                         \
             typename TTypes<T, CPU_PROVIDED_IXDIM + 1>::ConstTensor Tparams, \
             typename TTypes<Index>::ConstMatrix Tindices,                    \
             typename TTypes<T>::Matrix Tout);

#define REGISTER_GATHER_ND_CPU(type)    \
  REGISTER_GATHER_ND_FULL(type, int32); \
  REGISTER_GATHER_ND_FULL(type, int64)

TF_CALL_ALL_TYPES(REGISTER_GATHER_ND_CPU);
TF_CALL_QUANTIZED_TYPES(REGISTER_GATHER_ND_CPU);

}  // namespace functor

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_KERNELS_GATHER_ND_OP_CPU_IMPL_H_
