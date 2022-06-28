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

#ifndef TENSORFLOW_CORE_KERNELS_CAST_OP_H_
#define TENSORFLOW_CORE_KERNELS_CAST_OP_H_
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
class MHTracer_DTPStensorflowPScorePSkernelsPScast_opDTh {
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
   MHTracer_DTPStensorflowPScorePSkernelsPScast_opDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSkernelsPScast_opDTh() {
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
#include "tensorflow/core/framework/bfloat16.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/platform/byte_order.h"
#include "tensorflow/core/platform/types.h"

// Note that the GPU cast functor templates need to be instantiated unlike the
// CPU ones, and hence their specializations are different than that for CPUs.
#ifdef SPECIALIZE_FOR_GPUS
#define SPECIALIZE_CAST(DEVICE, OUT_TYPE, IN_OUT)                   \
  template <typename Device>                                        \
  struct CastFunctor<Device, OUT_TYPE, IN_OUT> {                    \
    void operator()(const Device& d,                                \
                    typename TTypes<OUT_TYPE>::Flat out_tensor,     \
                    typename TTypes<IN_OUT>::ConstFlat in_tensor,   \
                    bool truncate = false) {                        \
      if (truncate) {                                               \
        out_tensor.device(d) =                                      \
            in_tensor.unaryExpr(LSBZeroSetter<IN_OUT, OUT_TYPE>())  \
                .template cast<OUT_TYPE>();                         \
      } else {                                                      \
        out_tensor.device(d) = in_tensor.template cast<OUT_TYPE>(); \
      }                                                             \
    }                                                               \
  };                                                                \
  template struct CastFunctor<DEVICE, OUT_TYPE, IN_OUT>;
#else
#define SPECIALIZE_CAST(DEVICE, OUT_TYPE, IN_OUT)                   \
  template <>                                                       \
  struct CastFunctor<DEVICE, OUT_TYPE, IN_OUT> {                    \
    void operator()(const DEVICE& d,                                \
                    typename TTypes<OUT_TYPE>::Flat out_tensor,     \
                    typename TTypes<IN_OUT>::ConstFlat in_tensor,   \
                    bool truncate = false) {                        \
      if (truncate) {                                               \
        out_tensor.device(d) =                                      \
            in_tensor.unaryExpr(LSBZeroSetter<IN_OUT, OUT_TYPE>())  \
                .template cast<OUT_TYPE>();                         \
      } else {                                                      \
        out_tensor.device(d) = in_tensor.template cast<OUT_TYPE>(); \
      }                                                             \
    }                                                               \
  };
#endif

#define CAST_FUNCTORS(devname)                                        \
  SPECIALIZE_CAST(devname, float, double)                             \
  SPECIALIZE_CAST(devname, float, std::complex<double>)               \
  SPECIALIZE_CAST(devname, std::complex<float>, std::complex<double>) \
  SPECIALIZE_CAST(devname, std::complex<float>, double)               \
  SPECIALIZE_CAST(devname, Eigen::half, double)                       \
  SPECIALIZE_CAST(devname, Eigen::half, float)                        \
  SPECIALIZE_CAST(devname, Eigen::half, std::complex<double>)         \
  SPECIALIZE_CAST(devname, Eigen::half, std::complex<float>)          \
  SPECIALIZE_CAST(devname, bfloat16, float)                           \
  template <typename OUT_TYPE, typename IN_OUT>                       \
  struct CastFunctor<devname, OUT_TYPE, IN_OUT> {                     \
    void operator()(const devname& d,                                 \
                    typename TTypes<OUT_TYPE>::Flat out_tensor,       \
                    typename TTypes<IN_OUT>::ConstFlat in_tensor,     \
                    bool truncate = false) {                          \
      out_tensor.device(d) = in_tensor.template cast<OUT_TYPE>();     \
    }                                                                 \
  };

#if defined(MLIR_GENERATED_GPU_KERNELS_ENABLED)
// If MLIR kernels are enabled, we don't need the specialized cast from float to
// double or from Eigen::half to double. We still need the specialized cast from
// Eigen::half to float, because it is used in depthwise_conv_grad_op.cc. We
// still need the specialized cast from float to double because it is used in
// resize_bilinear_op.cc.
#define CAST_FUNCTORS_SUBSET(devname)                                 \
  SPECIALIZE_CAST(devname, float, double)                             \
  SPECIALIZE_CAST(devname, float, std::complex<double>)               \
  SPECIALIZE_CAST(devname, std::complex<float>, std::complex<double>) \
  SPECIALIZE_CAST(devname, std::complex<float>, double)               \
  SPECIALIZE_CAST(devname, Eigen::half, float)                        \
  SPECIALIZE_CAST(devname, Eigen::half, std::complex<double>)         \
  SPECIALIZE_CAST(devname, Eigen::half, std::complex<float>)          \
  SPECIALIZE_CAST(devname, bfloat16, float)                           \
  template <typename OUT_TYPE, typename IN_OUT>                       \
  struct CastFunctor<devname, OUT_TYPE, IN_OUT> {                     \
    void operator()(const devname& d,                                 \
                    typename TTypes<OUT_TYPE>::Flat out_tensor,       \
                    typename TTypes<IN_OUT>::ConstFlat in_tensor,     \
                    bool truncate = false) {                          \
      out_tensor.device(d) = in_tensor.template cast<OUT_TYPE>();     \
    }                                                                 \
  };
#endif

namespace tensorflow {

typedef std::function<void(OpKernelContext*, const Tensor&, Tensor*,
                           bool trunc)>
    CastFunctorType;

// Common base class of Cast kernels
class CastOpBase : public OpKernel {
 public:
  explicit CastOpBase(OpKernelConstruction* ctx);

  void Compute(OpKernelContext* ctx) override;

 protected:
  DataType src_dtype_;
  DataType dst_dtype_;
  DataType external_src_dtype_;
  DataType external_dst_dtype_;
  bool use_truncation_;
  CastFunctorType work_ = nullptr;
  Status Unimplemented();

  TF_DISALLOW_COPY_AND_ASSIGN(CastOpBase);
};

// CPU implementation of Cast
class CpuCastOp : public CastOpBase {
 public:
  explicit CpuCastOp(OpKernelConstruction* ctx);

 private:
  Status Prepare();
};

namespace functor {

template <typename I>
constexpr int MantissaWidth() {
  return std::numeric_limits<I>::digits;
}

template <>
constexpr int MantissaWidth<Eigen::half>() {
  // Remember, there's 1 hidden bit
  return 10 + 1;
}

template <>
constexpr int MantissaWidth<bfloat16>() {
  // Remember, there's 1 hidden bit
  return 7 + 1;
}

template <typename Device, typename Tout, typename Tin>
void Cast(const Device& d, typename TTypes<Tout>::Flat o,
          typename TTypes<Tin>::ConstFlat i) {
  o.device(d) = i.template cast<Tout>();
}

template <typename Device, typename Tout, typename Tin>
struct CastFunctor {
  void operator()(const Device& d, typename TTypes<Tout>::Flat o,
                  typename TTypes<Tin>::ConstFlat i, bool truncate = false);
};

// Only enable LSBZeroSetterHelper for 64 and 32 bit input data types.
// Specialize for others if needed in future.
template <typename I>
typename std::enable_if<sizeof(I) == 8, void>::type EIGEN_DEVICE_FUNC
    EIGEN_STRONG_INLINE static LSBZeroSetterHelper(I& t, int n) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSkernelsPScast_opDTh mht_0(mht_0_v, 350, "", "./tensorflow/core/kernels/cast_op.h", "LSBZeroSetterHelper");

  // Only zero the bits for non-NaNs.
  // For NaNs, let the non-truncation version handle it.
  if (!std::isnan(t)) {
    uint64_t* p = reinterpret_cast<uint64_t*>(&t);
    *p &= (0xFFFFFFFFFFFFFFFF << n);
  }
}

template <typename I>
typename std::enable_if<sizeof(I) == 4, void>::type EIGEN_DEVICE_FUNC
    EIGEN_STRONG_INLINE static LSBZeroSetterHelper(I& t, int n) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSkernelsPScast_opDTh mht_1(mht_1_v, 364, "", "./tensorflow/core/kernels/cast_op.h", "LSBZeroSetterHelper");

  // Only zero the bits for non-NaNs.
  // For NaNs, let the non-truncation version handle it.
  if (!std::isnan(t)) {
    uint32_t* p = reinterpret_cast<uint32_t*>(&t);
    *p &= (0xFFFFFFFF << n);
  }
}

// Set n least significant bits to 0
template <typename I, typename O>
struct LSBZeroSetter {
  EIGEN_EMPTY_STRUCT_CTOR(LSBZeroSetter)

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE const I operator()(const I& a) const {
    constexpr int bits = MantissaWidth<I>() - MantissaWidth<O>();
    static_assert(
        bits > 0,
        "The output type must have fewer mantissa bits than the input type\n");
    I t = a;
    LSBZeroSetterHelper(t, bits);
    return t;
  }
};

template <typename I, typename O>
struct LSBZeroSetter<std::complex<I>, std::complex<O>> {
  EIGEN_EMPTY_STRUCT_CTOR(LSBZeroSetter)

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE const std::complex<I> operator()(
      const std::complex<I>& a) const {
    constexpr int bits = MantissaWidth<I>() - MantissaWidth<O>();
    static_assert(
        bits > 0,
        "The output type must have fewer mantissa bits than the input type\n");
    I re = std::real(a);
    I img = std::imag(a);
    LSBZeroSetterHelper(re, bits);
    LSBZeroSetterHelper(img, bits);
    std::complex<I> toReturn(re, img);
    return toReturn;
  }
};

template <typename I, typename O>
struct LSBZeroSetter<std::complex<I>, O> {
  EIGEN_EMPTY_STRUCT_CTOR(LSBZeroSetter)
  // Sets the 16 LSBits of the float to 0
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE const std::complex<I> operator()(
      const std::complex<I>& a) const {
    constexpr int bits = MantissaWidth<I>() - MantissaWidth<O>();
    static_assert(
        bits > 0,
        "The output type must have fewer mantissa bits than the input type\n");
    I re = std::real(a);
    I img = std::imag(a);
    LSBZeroSetterHelper(re, bits);
    LSBZeroSetterHelper(img, bits);
    std::complex<I> toReturn(re, img);
    return toReturn;
  }
};

}  // end namespace functor
}  // end namespace tensorflow

namespace Eigen {
namespace internal {

// Eigen can't convert to/from complex numbers, because it is limited to cases
// that can be static_casted. But numpy is able to cast to/from complex, which
// we want to replicate. So we add specializations for complex here.
template <typename From, typename To>
struct scalar_cast_op<std::complex<From>, To> {
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE To
  operator()(const std::complex<From>& a) const {
    // Replicate numpy behavior of returning just the real part
    return static_cast<To>(a.real());
  }
};

template <typename From, typename To>
struct scalar_cast_op<From, std::complex<To>> {
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE std::complex<To> operator()(
      const From& a) const {
    // Replicate numpy behavior of setting the imaginary part to 0
    return std::complex<To>(static_cast<To>(a), To(0));
  }
};

template <typename From, typename To>
struct scalar_cast_op<std::complex<From>, std::complex<To>> {
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE std::complex<To> operator()(
      const std::complex<From>& a) const {
    return std::complex<To>(static_cast<To>(a.real()),
                            static_cast<To>(a.imag()));
  }
};

template <typename From, typename To>
struct functor_traits_complex_impl {
  enum { Cost = NumTraits<To>::AddCost, PacketAccess = false };
};

template <typename From, typename To>
struct functor_traits<scalar_cast_op<std::complex<From>, To>>
    : functor_traits_complex_impl<std::complex<From>, To> {};
template <typename From, typename To>
struct functor_traits<scalar_cast_op<From, std::complex<To>>>
    : functor_traits_complex_impl<From, std::complex<To>> {};
// Needed to avoid ambiguous partial specialization
template <typename From, typename To>
struct functor_traits<scalar_cast_op<std::complex<From>, std::complex<To>>>
    : functor_traits_complex_impl<std::complex<From>, std::complex<To>> {};

}  // namespace internal
}  // namespace Eigen

#endif  // TENSORFLOW_CORE_KERNELS_CAST_OP_H_
