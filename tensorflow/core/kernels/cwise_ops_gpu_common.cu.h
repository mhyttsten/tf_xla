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

#if !GOOGLE_CUDA && !TENSORFLOW_USE_ROCM
#error This file must only be included when building with Cuda or ROCm support
#endif

#ifndef TENSORFLOW_CORE_KERNELS_CWISE_OPS_GPU_COMMON_CU_H_
#define TENSORFLOW_CORE_KERNELS_CWISE_OPS_GPU_COMMON_CU_H_
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
class MHTracer_DTPStensorflowPScorePSkernelsPScwise_ops_gpu_commonDTcuDTh {
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
   MHTracer_DTPStensorflowPScorePSkernelsPScwise_ops_gpu_commonDTcuDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSkernelsPScwise_ops_gpu_commonDTcuDTh() {
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


#define _USE_MATH_DEFINES
#include <cmath>
#include <complex>

#define EIGEN_USE_GPU
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/kernels/cwise_ops.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/types.h"
namespace tensorflow {
namespace functor {

typedef Eigen::GpuDevice GPUDevice;
typedef std::complex<float> complex64;
typedef std::complex<double> complex128;

// Partial specialization of UnaryFunctor<Device=GPUDevice, Functor>.
template <typename Functor>
struct UnaryFunctor<GPUDevice, Functor> {
  void operator()(const GPUDevice& d, typename Functor::tout_type out,
                  typename Functor::tin_type in) {
    MaybeWith32BitIndexing<GPUDevice>(
        [&](auto out32, auto in32) {
          out32.device(d) = in32.unaryExpr(typename Functor::func());
        },
        out, in);
  }
};

// Partial specialization of BinaryFunctor<Device=GPUDevice, Functor>.
template <typename Functor, int NDIMS, bool has_errors>
struct BinaryFunctor<GPUDevice, Functor, NDIMS, has_errors> {
  void operator()(const GPUDevice& d, typename Functor::tout_type out,
                  typename Functor::tin_type in0,
                  typename Functor::tin_type in1, bool* error) {
    MaybeWith32BitIndexing<GPUDevice>(
        [&](auto out32, auto in0_32, auto in1_32) {
          out32.device(d) = in0_32.binaryExpr(in1_32, typename Functor::func());
        },
        out, in0, in1);
  }

  void Left(const GPUDevice& d, typename Functor::tout_type out,
            typename Functor::tscalar_type scalar,
            typename Functor::tin_type in, bool* error) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSkernelsPScwise_ops_gpu_commonDTcuDTh mht_0(mht_0_v, 236, "", "./tensorflow/core/kernels/cwise_ops_gpu_common.cu.h", "Left");

    typedef typename Functor::out_type Tout;
    typedef typename Functor::in_type Tin;
    typedef typename Functor::func Binary;
    typedef typename Eigen::internal::scalar_left<Tout, Tin, Binary> Unary;
    MaybeWith32BitIndexing<GPUDevice>(
        [&](auto out32, auto in32) {
          out32.device(d) = in32.unaryExpr(Unary(scalar.data()));
        },
        out, in);
  }

  void Right(const GPUDevice& d, typename Functor::tout_type out,
             typename Functor::tin_type in,
             typename Functor::tscalar_type scalar, bool* error) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSkernelsPScwise_ops_gpu_commonDTcuDTh mht_1(mht_1_v, 253, "", "./tensorflow/core/kernels/cwise_ops_gpu_common.cu.h", "Right");

    typedef typename Functor::out_type Tout;
    typedef typename Functor::in_type Tin;
    typedef typename Functor::func Binary;
    typedef typename Eigen::internal::scalar_right<Tout, Tin, Binary> Unary;
    MaybeWith32BitIndexing<GPUDevice>(
        [&](auto out32, auto in32) {
          out32.device(d) = in32.unaryExpr(Unary(scalar.data()));
        },
        out, in);
  }

  void BCast(const GPUDevice& d,
             typename TTypes<typename Functor::out_type, NDIMS>::Tensor out,
             typename TTypes<typename Functor::in_type, NDIMS>::ConstTensor in0,
             typename Eigen::array<Eigen::DenseIndex, NDIMS> bcast0,
             typename TTypes<typename Functor::in_type, NDIMS>::ConstTensor in1,
             typename Eigen::array<Eigen::DenseIndex, NDIMS> bcast1,
             bool* error) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSkernelsPScwise_ops_gpu_commonDTcuDTh mht_2(mht_2_v, 274, "", "./tensorflow/core/kernels/cwise_ops_gpu_common.cu.h", "BCast");

    typedef typename Functor::in_type T;
    typename Functor::func func;
    if ((NDIMS == 2) && Functor::use_bcast_optimization &&
        use_bcast_optimization<T>::value) {
      const bool bcast0_all_one = AllOne<NDIMS>(bcast0);
      const bool bcast1_all_one = AllOne<NDIMS>(bcast1);
      if (bcast0_all_one && !bcast1_all_one) {
        MaybeWith32BitIndexing<GPUDevice>(
            [&](auto out32, auto in0_32, auto in1_32) {
              out32.device(d) =
                  in0_32.binaryExpr(in1_32.broadcast(bcast1), func);
            },
            out, in0, in1);
        return;
      }
      if (!bcast0_all_one && bcast1_all_one) {
        MaybeWith32BitIndexing<GPUDevice>(
            [&](auto out32, auto in0_32, auto in1_32) {
              out32.device(d) =
                  in0_32.broadcast(bcast0).binaryExpr(in1_32, func);
            },
            out, in0, in1);
        return;
      }
    }
    MaybeWith32BitIndexing<GPUDevice>(
        [&](auto out32, auto in0_32, auto in1_32) {
          out32.device(d) = in0_32.broadcast(bcast0).binaryExpr(
              in1_32.broadcast(bcast1), func);
        },
        out, in0, in1);
  }
};

// Partial specialization of ApproximateEqual<Device=GPUDevice, T>.
template <typename T>
struct ApproximateEqual<GPUDevice, T> {
  void operator()(const GPUDevice& d, typename TTypes<T>::ConstFlat x,
                  typename TTypes<T>::ConstFlat y, T tolerance,
                  typename TTypes<bool>::Flat z) {
    auto diff = x - y;
    z.device(d) = diff.abs() <= tolerance;
  }
};

// Macros to explicitly instantiate kernels on GPU for multiple types
// (T0, T1, etc.) for UnaryFunctor (e.g., functor::sqrt).
#define DEFINE_UNARY1(F, T) template struct UnaryFunctor<GPUDevice, F<T> >
#define DEFINE_UNARY2(F, T0, T1) \
  DEFINE_UNARY1(F, T0);          \
  DEFINE_UNARY1(F, T1)
#define DEFINE_UNARY3(F, T0, T1, T2) \
  DEFINE_UNARY2(F, T0, T1);          \
  DEFINE_UNARY1(F, T2)
#define DEFINE_UNARY4(F, T0, T1, T2, T3) \
  DEFINE_UNARY2(F, T0, T1);              \
  DEFINE_UNARY2(F, T2, T3)
#define DEFINE_UNARY5(F, T0, T1, T2, T3, T4) \
  DEFINE_UNARY2(F, T0, T1);                  \
  DEFINE_UNARY3(F, T2, T3, T4)
#define DEFINE_UNARY6(F, T0, T1, T2, T3, T4, T5) \
  DEFINE_UNARY2(F, T0, T1);                      \
  DEFINE_UNARY4(F, T2, T3, T4, T5)
#define DEFINE_UNARY7(F, T0, T1, T2, T3, T4, T5, T6) \
  DEFINE_UNARY2(F, T0, T1);                          \
  DEFINE_UNARY5(F, T2, T3, T4, T5, T6)
#define DEFINE_UNARY8(F, T0, T1, T2, T3, T4, T5, T6, T7) \
  DEFINE_UNARY4(F, T0, T1, T2, T3);                      \
  DEFINE_UNARY4(F, T4, T5, T6, T7)

// Macros to explicitly instantiate kernels on GPU for multiple types
// (T0, T1, etc.) for BinaryFunctor.
#define DEFINE_BINARY1(F, T)                         \
  template struct BinaryFunctor<GPUDevice, F<T>, 1>; \
  template struct BinaryFunctor<GPUDevice, F<T>, 2>; \
  template struct BinaryFunctor<GPUDevice, F<T>, 3>; \
  template struct BinaryFunctor<GPUDevice, F<T>, 4>; \
  template struct BinaryFunctor<GPUDevice, F<T>, 5>
#define DEFINE_BINARY2(F, T0, T1) \
  DEFINE_BINARY1(F, T0);          \
  DEFINE_BINARY1(F, T1)
#define DEFINE_BINARY3(F, T0, T1, T2) \
  DEFINE_BINARY2(F, T0, T1);          \
  DEFINE_BINARY1(F, T2)
#define DEFINE_BINARY4(F, T0, T1, T2, T3) \
  DEFINE_BINARY2(F, T0, T1);              \
  DEFINE_BINARY2(F, T2, T3)
#define DEFINE_BINARY5(F, T0, T1, T2, T3, T4) \
  DEFINE_BINARY2(F, T0, T1);                  \
  DEFINE_BINARY3(F, T2, T3, T4)
#define DEFINE_BINARY6(F, T0, T1, T2, T3, T4, T5) \
  DEFINE_BINARY3(F, T0, T1, T2);                  \
  DEFINE_BINARY3(F, T3, T4, T5)
#define DEFINE_BINARY7(F, T0, T1, T2, T3, T4, T5, T6) \
  DEFINE_BINARY3(F, T0, T1, T2);                      \
  DEFINE_BINARY4(F, T3, T4, T5, T6)
#define DEFINE_BINARY8(F, T0, T1, T2, T3, T4, T5, T6, T7) \
  DEFINE_BINARY4(F, T0, T1, T2, T3);                      \
  DEFINE_BINARY4(F, T4, T5, T6, T7)
#define DEFINE_BINARY9(F, T0, T1, T2, T3, T4, T5, T6, T7, T8) \
  DEFINE_BINARY4(F, T0, T1, T2, T3);                          \
  DEFINE_BINARY5(F, T4, T5, T6, T7, T8)
#define DEFINE_BINARY10(F, T0, T1, T2, T3, T4, T5, T6, T7, T8, T9) \
  DEFINE_BINARY5(F, T0, T1, T2, T3, T4);                           \
  DEFINE_BINARY5(F, T5, T6, T7, T8, T9)
#define DEFINE_BINARY11(F, T0, T1, T2, T3, T4, T5, T6, T7, T8, T9, T10) \
  DEFINE_BINARY5(F, T0, T1, T2, T3, T4);                                \
  DEFINE_BINARY6(F, T5, T6, T7, T8, T9, T10)

#define DEFINE_APPROXIMATE_EQUAL1(T) \
  template struct ApproximateEqual<GPUDevice, T>;
#define DEFINE_APPROXIMATE_EQUAL2(T0, T1) \
  DEFINE_APPROXIMATE_EQUAL1(T0);          \
  DEFINE_APPROXIMATE_EQUAL1(T1);

}  // end namespace functor
}  // end namespace tensorflow

#endif  // TENSORFLOW_CORE_KERNELS_CWISE_OPS_GPU_COMMON_CU_H_
