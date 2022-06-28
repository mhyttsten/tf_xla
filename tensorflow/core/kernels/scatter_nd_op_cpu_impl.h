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

#ifndef TENSORFLOW_CORE_KERNELS_SCATTER_ND_OP_CPU_IMPL_H_
#define TENSORFLOW_CORE_KERNELS_SCATTER_ND_OP_CPU_IMPL_H_
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
class MHTracer_DTPStensorflowPScorePSkernelsPSscatter_nd_op_cpu_implDTh {
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
   MHTracer_DTPStensorflowPScorePSkernelsPSscatter_nd_op_cpu_implDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSkernelsPSscatter_nd_op_cpu_implDTh() {
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


// Functor definitions for ScatterND ops, must be compilable by nvcc.

#define EIGEN_USE_THREADS

#include <atomic>

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

#include "tensorflow/core/framework/bounds_check.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/kernels/fill_functor.h"
#include "tensorflow/core/kernels/scatter_nd_op.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/util/util.h"

namespace tensorflow {

typedef Eigen::ThreadPoolDevice CPUDevice;

class OpKernelContext;

// Specialization of UpdateExecutor to CPU
namespace update_executor {

template <typename T, typename Input, typename Update, typename Output,
          scatter_nd_op::UpdateOp OP>
class UpdateExecutor {
 public:
  EIGEN_STRONG_INLINE static void Execute(const T& device, Input value,
                                          Update update, Output output);
};

template <typename T, typename Input, typename Update, typename Output>
class UpdateExecutor<T, Input, Update, Output,
                     scatter_nd_op::UpdateOp::ASSIGN> {
 public:
  EIGEN_STRONG_INLINE static void Execute(const T& device, Input /* input */,
                                          Update update, Output output) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSscatter_nd_op_cpu_implDTh mht_0(mht_0_v, 229, "", "./tensorflow/core/kernels/scatter_nd_op_cpu_impl.h", "Execute");

    output.device(device) = update;
  }
};

template <typename T, typename Input, typename Update, typename Output>
class UpdateExecutor<T, Input, Update, Output, scatter_nd_op::UpdateOp::ADD> {
 public:
  EIGEN_STRONG_INLINE static void Execute(const T& device, Input /* input */,
                                          Update update, Output output) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSscatter_nd_op_cpu_implDTh mht_1(mht_1_v, 241, "", "./tensorflow/core/kernels/scatter_nd_op_cpu_impl.h", "Execute");

    output.device(device) += update;
  }
};

template <typename T, typename Input, typename Update, typename Output>
class UpdateExecutor<T, Input, Update, Output, scatter_nd_op::UpdateOp::SUB> {
 public:
  EIGEN_STRONG_INLINE static void Execute(const T& device, Input /* input */,
                                          Update update, Output output) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSscatter_nd_op_cpu_implDTh mht_2(mht_2_v, 253, "", "./tensorflow/core/kernels/scatter_nd_op_cpu_impl.h", "Execute");

    output.device(device) -= update;
  }
};

template <typename T, typename Input, typename Update, typename Output>
class UpdateExecutor<T, Input, Update, Output, scatter_nd_op::UpdateOp::MIN> {
 public:
  EIGEN_STRONG_INLINE static void Execute(const T& device, Input /* input */,
                                          Update update, Output output) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSscatter_nd_op_cpu_implDTh mht_3(mht_3_v, 265, "", "./tensorflow/core/kernels/scatter_nd_op_cpu_impl.h", "Execute");

    output.device(device) = output.cwiseMin(update);
  }
};

template <typename T, typename Input, typename Update, typename Output>
class UpdateExecutor<T, Input, Update, Output, scatter_nd_op::UpdateOp::MAX> {
 public:
  EIGEN_STRONG_INLINE static void Execute(const T& device, Input /* input */,
                                          Update update, Output output) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSscatter_nd_op_cpu_implDTh mht_4(mht_4_v, 277, "", "./tensorflow/core/kernels/scatter_nd_op_cpu_impl.h", "Execute");

    output.device(device) = output.cwiseMax(update);
  }
};

}  // namespace update_executor

namespace functor {

// Implementation of update functor for CPU.
template <typename T, typename Index, scatter_nd_op::UpdateOp OP, int IXDIM>
struct ScatterNdFunctor<CPUDevice, T, Index, OP, IXDIM> {
  Index operator()(
      const CPUDevice& d, const Index slice_size,
      const Eigen::array<Eigen::DenseIndex, IXDIM> output_shape_prefix,
      typename TTypes<T, 2>::Tensor Tparams,
      typename TTypes<Index, 2>::ConstTensor Tindices,
      typename TTypes<T, 2>::ConstTensor Tupdates,
      typename TTypes<T, 2>::Tensor Toutput) {
    // error_loc is -1 if there's no out-of-bounds index,
    // otherwise it is the location of an OOB index in Tindices.
    Index error_loc = -1;

    const Eigen::DenseIndex batch_size = Tindices.dimension(0);

    Index batch_strides[IXDIM];
    for (int dim = IXDIM - 1; dim >= 0; --dim) {
      if (dim == IXDIM - 1) {
        batch_strides[dim] = 1;
      } else {
        batch_strides[dim] =
            batch_strides[dim + 1] * output_shape_prefix[dim + 1];
      }
    }

    for (Eigen::DenseIndex loc = 0; loc < batch_size; ++loc) {
      Index i = 0;
      bool out_of_bounds = false;
      for (int dim = 0; dim < IXDIM; ++dim) {
        const Index ix_d = internal::SubtleMustCopy(Tindices(loc, dim));
        out_of_bounds |= !FastBoundsCheck(ix_d, output_shape_prefix[dim]);
        i += ix_d * batch_strides[dim];
      }
      if (TF_PREDICT_FALSE(out_of_bounds)) {
        error_loc = loc;
        break;
      } else {
        auto input_chip = Toutput.template chip<0>(i);
        auto output_chip = input_chip;
        auto update_chip = Tupdates.template chip<0>(loc);
        update_executor::UpdateExecutor<
            CPUDevice, decltype(input_chip), decltype(update_chip),
            decltype(output_chip), OP>::Execute(d, input_chip, update_chip,
                                                output_chip);
      }
    }

    return error_loc;
  }
};

#define REGISTER_SCATTER_ND_FULL(T, Index, op)                               \
  template Index                                                             \
  ScatterNdFunctor<CPUDevice, T, Index, op, CPU_PROVIDED_IXDIM>::operator()( \
      const CPUDevice& d, const Index slice_size,                            \
      const Eigen::array<Eigen::DenseIndex, CPU_PROVIDED_IXDIM>              \
          output_shape_prefix,                                               \
      typename TTypes<T, 2>::Tensor Tparams,                                 \
      typename TTypes<Index, 2>::ConstTensor Tindices,                       \
      typename TTypes<T, 2>::ConstTensor Tupdates,                           \
      typename TTypes<T, 2>::Tensor Toutput)

#define REGISTER_SCATTER_ND_INDEX(type, op)  \
  REGISTER_SCATTER_ND_FULL(type, int32, op); \
  REGISTER_SCATTER_ND_FULL(type, int64, op)

#define REGISTER_SCATTER_ND_UPDATE(type) \
  REGISTER_SCATTER_ND_INDEX(type, scatter_nd_op::UpdateOp::ASSIGN);

#define REGISTER_SCATTER_ND_MATH(type)                           \
  REGISTER_SCATTER_ND_INDEX(type, scatter_nd_op::UpdateOp::ADD); \
  REGISTER_SCATTER_ND_INDEX(type, scatter_nd_op::UpdateOp::SUB);

#define REGISTER_SCATTER_ND_MIN_MAX(type)                        \
  REGISTER_SCATTER_ND_INDEX(type, scatter_nd_op::UpdateOp::MAX); \
  REGISTER_SCATTER_ND_INDEX(type, scatter_nd_op::UpdateOp::MIN);

TF_CALL_ALL_TYPES(REGISTER_SCATTER_ND_UPDATE);
REGISTER_SCATTER_ND_INDEX(tstring, scatter_nd_op::UpdateOp::ADD);
TF_CALL_NUMBER_TYPES(REGISTER_SCATTER_ND_MATH);
TF_CALL_REAL_NUMBER_TYPES(REGISTER_SCATTER_ND_MIN_MAX);
TF_CALL_bool(REGISTER_SCATTER_ND_MATH);

#undef REGISTER_SCATTER_ND_MATH
#undef REGISTER_SCATTER_ND_MIN_MAX
#undef REGISTER_SCATTER_ND_UPDATE
#undef REGISTER_SCATTER_ND_INDEX
#undef REGISTER_SCATTER_ND_FULL
}  // namespace functor

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_KERNELS_SCATTER_ND_OP_CPU_IMPL_H_
