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

#ifndef TENSORFLOW_CORE_KERNELS_CWISE_OPS_H_
#define TENSORFLOW_CORE_KERNELS_CWISE_OPS_H_
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
class MHTracer_DTPStensorflowPScorePSkernelsPScwise_opsDTh {
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
   MHTracer_DTPStensorflowPScorePSkernelsPScwise_opsDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSkernelsPScwise_opsDTh() {
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
#include <functional>
#include <type_traits>

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/bounds_check.h"
#include "tensorflow/core/framework/numeric_types.h"
#include "tensorflow/core/framework/tensor_types.h"

namespace Eigen {
namespace internal {

#if GOOGLE_CUDA
template <>
struct scalar_arg_op<std::complex<float>> {
  EIGEN_EMPTY_STRUCT_CTOR(scalar_arg_op)
  typedef typename Eigen::NumTraits<std::complex<float>>::Real result_type;
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE float operator()(
      const std::complex<float>& a) const {
    return ::atan2f(a.imag(), a.real());
  }
};

template <>
struct scalar_arg_op<std::complex<double>> {
  EIGEN_EMPTY_STRUCT_CTOR(scalar_arg_op)
  typedef typename Eigen::NumTraits<std::complex<double>>::Real result_type;
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE double operator()(
      const std::complex<double>& a) const {
    return ::atan2(a.imag(), a.real());
  }
};
#endif

#if EIGEN_HAS_CXX11_MATH == 0
template <typename T>
struct scalar_asinh_op {
  EIGEN_EMPTY_STRUCT_CTOR(scalar_asinh_op)
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE T operator()(const T& a) const {
    return static_cast<T>(std::asinh(a));
  }
};
template <typename T>
struct functor_traits<scalar_asinh_op<T>> {
  enum { Cost = 5 * NumTraits<T>::MulCost, PacketAccess = false };
};

template <typename T>
struct scalar_acosh_op {
  EIGEN_EMPTY_STRUCT_CTOR(scalar_acosh_op)
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE T operator()(const T& a) const {
    return static_cast<T>(std::acosh(a));
  }
};
template <typename T>
struct functor_traits<scalar_acosh_op<T>> {
  enum { Cost = 5 * NumTraits<T>::MulCost, PacketAccess = false };
};

template <typename T>
struct scalar_atanh_op {
  EIGEN_EMPTY_STRUCT_CTOR(scalar_atanh_op)
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE T operator()(const T& a) const {
    return static_cast<T>(std::atanh(a));
  }
};
template <typename T>
struct functor_traits<scalar_atanh_op<T>> {
  enum { Cost = 5 * NumTraits<T>::MulCost, PacketAccess = false };
};
#endif

template <typename Scalar, typename Exponent>
struct safe_scalar_binary_pow_op {
  static_assert(std::is_integral<Scalar>::value, "Integer type expected");
  static_assert(std::is_integral<Exponent>::value &&
                    std::is_signed<Exponent>::value,
                "Signed integer type expected");

  bool* const error;

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE safe_scalar_binary_pow_op(bool* error)
      : error(error) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSkernelsPScwise_opsDTh mht_0(mht_0_v, 271, "", "./tensorflow/core/kernels/cwise_ops.h", "safe_scalar_binary_pow_op");
}

  EIGEN_DEVICE_FUNC inline Scalar operator()(const Scalar& a,
                                             const Exponent& b) const {
    const Exponent safe_b = tensorflow::internal::SubtleMustCopy(b);
    if (TF_PREDICT_TRUE(safe_b >= 0)) {
      return numext::pow(a, safe_b);
    } else {
      *error = true;
      return 0;
    }
  }
};

template <typename Scalar, typename Exponent>
struct functor_traits<safe_scalar_binary_pow_op<Scalar, Exponent>> {
  enum { Cost = 5 * NumTraits<Scalar>::MulCost, PacketAccess = false };
};

template <typename T, typename DivOrMod>
struct safe_div_or_mod_op {
  static_assert(std::is_integral<T>::value, "Integer type expected");

  bool* const error;

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE safe_div_or_mod_op(bool* error)
      : error(error) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSkernelsPScwise_opsDTh mht_1(mht_1_v, 300, "", "./tensorflow/core/kernels/cwise_ops.h", "safe_div_or_mod_op");
}

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE T operator()(const T& a,
                                                     const T& b) const {
    const T safe_b = tensorflow::internal::SubtleMustCopy(b);
    if (TF_PREDICT_TRUE(safe_b != 0)) {
      // Avoid FPE for INT_MIN/-1.
      const T safe_a = tensorflow::internal::SubtleMustCopy(a);
      if (TF_PREDICT_FALSE(std::is_signed<T>::value &&
                           safe_a == std::numeric_limits<T>::min() &&
                           safe_b == T(-1))) {
        // Prefer to overflow 'a' instead of crashing.
        return DivOrMod()(-safe_a, 1);
      }
      return DivOrMod()(safe_a, safe_b);
    } else {
      *error = true;
      return 0;
    }
  }
};

template <typename T, typename DivOrMod>
struct functor_traits<safe_div_or_mod_op<T, DivOrMod>> {
  enum {
    Cost = functor_traits<DivOrMod>::Cost + NumTraits<T>::AddCost,
    PacketAccess = false,
  };
};

template <typename T, typename Binary>
struct no_nan_op {
  EIGEN_EMPTY_STRUCT_CTOR(no_nan_op)
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE T operator()(const T& a,
                                                     const T& b) const {
    if (b != T(0)) {
      return Binary()(a, b);
    } else {
      return T(0);
    }
  }
  template <typename Packet>
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Packet packetOp(const Packet& a,
                                                        const Packet& b) const {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSkernelsPScwise_opsDTh mht_2(mht_2_v, 346, "", "./tensorflow/core/kernels/cwise_ops.h", "packetOp");

    const Packet mask = pcmp_eq(b, pzero(b));
    const Packet quotient = Binary().packetOp(a, b);
    return pandnot(quotient, mask);
  }
};

template <typename T, bool IsComplex = Eigen::NumTraits<T>::IsComplex>
struct div_no_nan_op;

template <typename T>
struct div_no_nan_op<T, /*IsComplex=*/false>
    : public no_nan_op<T, scalar_quotient_op<T>> {
  EIGEN_EMPTY_STRUCT_CTOR(div_no_nan_op)
};

template <typename T>
struct functor_traits<div_no_nan_op<T, /*IsComplex=*/false>> {
  enum {
    Cost = functor_traits<scalar_quotient_op<T>>::Cost + NumTraits<T>::AddCost,
    PacketAccess = true,
  };
};

// Whether or not complex division produces a NaN depends on the underlying
// implementation. Some compilers (e.g. gcc) use a simple method that divides
// by |b|^2, which may underflow to 0 for b != 0.
template <typename T>
struct div_no_nan_op<T, /*IsComplex=*/true> {
  EIGEN_EMPTY_STRUCT_CTOR(div_no_nan_op)
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE T operator()(const T& a,
                                                     const T& b) const {
    if (b == T(0)) {
      return T(0);
    } else {
      // If the numerator is zero, then the result must be zero even if |b|^2
      // underflows to zero.
      const T numerator =
          scalar_product_op<T>()(a, scalar_conjugate_op<T>()(b));
      if (numerator == T(0)) {
        return T(0);
      }
    }
    return scalar_quotient_op<T>()(a, b);
  }
  template <typename Packet>
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Packet packetOp(const Packet& a,
                                                        const Packet& b) const {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSkernelsPScwise_opsDTh mht_3(mht_3_v, 396, "", "./tensorflow/core/kernels/cwise_ops.h", "packetOp");

    const Packet numerator = pmul(a, pconj(b));
    const Packet mask = por(pcmp_eq(b, pzero(a)), pcmp_eq(numerator, pzero(a)));
    const Packet quotient = pdiv(a, b);
    return pandnot(quotient, mask);
  }
};

template <typename T>
struct functor_traits<div_no_nan_op<T, /*IsComplex=*/true>> {
  enum {
    Cost = functor_traits<scalar_quotient_op<T>>::Cost + NumTraits<T>::MulCost,
    PacketAccess = packet_traits<T>::HasMul && packet_traits<T>::HasDiv &&
                   packet_traits<T>::HasConj,
  };
};

template <typename T>
struct mul_no_nan_op : public no_nan_op<T, scalar_product_op<T>> {
  EIGEN_EMPTY_STRUCT_CTOR(mul_no_nan_op)
};

template <typename T>
struct functor_traits<mul_no_nan_op<T>> {
  enum {
    Cost = functor_traits<scalar_product_op<T>>::Cost + NumTraits<T>::AddCost,
    PacketAccess = true,
  };
};

// scalar_left and scalar_right are template helpers to partially
// apply a binary function.
//
// Suppose Binary is a binary functor f(x, y), scalar_left<> is a
// unary functor g_x(y) = f(x, y), where x is provided via the
// constructor. Similarly, scalar_right<> is a unary functor g_y(x) =
// f(x, y).

template <typename Tout, typename Tin, typename Binary,
          bool is_scalar_in_host_memory = false>
struct scalar_left : private Binary {
  using result_type = Tout;
  using TinPacket = typename Eigen::internal::packet_traits<Tin>::type;

  const Tin* left;
  TinPacket left_packet;  // initialized iff is_scalar_in_host_memory == true

  EIGEN_DEVICE_FUNC inline scalar_left(const scalar_left& other) = default;

  template <typename... Args>
  EIGEN_DEVICE_FUNC inline explicit scalar_left(const Tin* c, Args... args)
      : Binary(args...), left(c) {
    if (is_scalar_in_host_memory) {
      left_packet = Eigen::internal::pset1<TinPacket>(*left);
    }
  }

  EIGEN_DEVICE_FUNC inline Tout operator()(const Tin& right) const {
    return Binary::operator()(*left, right);
  }

  template <typename Packet,
            typename std::enable_if<!is_scalar_in_host_memory ||
                                        !std::is_same<TinPacket, Packet>::value,
                                    int>::type = 0>
  EIGEN_DEVICE_FUNC inline Packet packetOp(const Packet& right_packet) const {
    const Packet left_packet = Eigen::internal::pset1<Packet>(*left);
    return Binary::packetOp(left_packet, right_packet);
  }

  template <typename Packet,
            typename std::enable_if<is_scalar_in_host_memory &&
                                        std::is_same<TinPacket, Packet>::value,
                                    int>::type = 0>
  EIGEN_DEVICE_FUNC inline Packet packetOp(const Packet& right_packet) const {
    return Binary::packetOp(left_packet, right_packet);
  }
};

template <typename Tout, typename Tin, typename Binary,
          bool is_scalar_in_host_memory>
struct functor_traits<
    scalar_left<Tout, Tin, Binary, is_scalar_in_host_memory>> {
  enum {
    Cost = functor_traits<Binary>::Cost,
    PacketAccess = functor_traits<Binary>::PacketAccess,
  };
};

template <typename Tout, typename Tin, typename Binary,
          bool is_scalar_in_host_memory = false>
struct scalar_right : private Binary {
  using result_type = Tout;
  using TinPacket = typename Eigen::internal::packet_traits<Tin>::type;

  const Tin* right;
  TinPacket right_packet;  // initialized iff is_scalar_in_host_memory == true

  EIGEN_DEVICE_FUNC inline scalar_right(const scalar_right& other) = default;

  template <typename... Args>
  EIGEN_DEVICE_FUNC inline explicit scalar_right(const Tin* c, Args... args)
      : Binary(args...), right(c) {
    if (is_scalar_in_host_memory) {
      right_packet = Eigen::internal::pset1<TinPacket>(*right);
    }
  }

  EIGEN_DEVICE_FUNC inline Tout operator()(const Tin& left) const {
    return Binary::operator()(left, *right);
  }

  template <typename Packet,
            typename std::enable_if<!is_scalar_in_host_memory ||
                                        !std::is_same<TinPacket, Packet>::value,
                                    int>::type = 0>
  EIGEN_DEVICE_FUNC inline Packet packetOp(const Packet& left_packet) const {
    const Packet right_packet = Eigen::internal::pset1<Packet>(*right);
    return Binary::packetOp(left_packet, right_packet);
  }

  template <typename Packet,
            typename std::enable_if<is_scalar_in_host_memory &&
                                        std::is_same<TinPacket, Packet>::value,
                                    int>::type = 0>
  EIGEN_DEVICE_FUNC inline Packet packetOp(const Packet& left_packet) const {
    return Binary::packetOp(left_packet, right_packet);
  }
};

template <typename Tout, typename Tin, typename Binary,
          bool is_scalar_in_host_memory>
struct functor_traits<
    scalar_right<Tout, Tin, Binary, is_scalar_in_host_memory>> {
  enum {
    Cost = functor_traits<Binary>::Cost,
    PacketAccess = functor_traits<Binary>::PacketAccess,
  };
};

// similar to std::equal_to, but with the DEVICE_FUNC qualifier
template <class T>
struct equal_to : std::binary_function<T, T, bool> {
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE bool operator()(const T& x,
                                                        const T& y) const {
    return x == y;
  }
};

// similar to std::not_equal_to, but with the DEVICE_FUNC qualifier
template <class T>
struct not_equal_to : std::binary_function<T, T, bool> {
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE bool operator()(const T& x,
                                                        const T& y) const {
    return x != y;
  }
};

// similar to std::greater, but with the DEVICE_FUNC qualifier
template <class T>
struct greater : std::binary_function<T, T, bool> {
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE bool operator()(const T& x,
                                                        const T& y) const {
    return x > y;
  }
};

// similar to std::less, but with the DEVICE_FUNC qualifier
template <class T>
struct less : std::binary_function<T, T, bool> {
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE bool operator()(const T& x,
                                                        const T& y) const {
    return x < y;
  }
};

// similar to std::greater_equal, but with the DEVICE_FUNC qualifier
template <class T>
struct greater_equal : std::binary_function<T, T, bool> {
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE bool operator()(const T& x,
                                                        const T& y) const {
    return x >= y;
  }
};

// similar to std::less_equal, but with the DEVICE_FUNC qualifier
template <class T>
struct less_equal : std::binary_function<T, T, bool> {
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE bool operator()(const T& x,
                                                        const T& y) const {
    return x <= y;
  }
};

// Functor that enables squared difference functor.
template <typename Scalar>
struct scalar_squared_difference_op {
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Scalar
  operator()(const Scalar& a, const Scalar& b) const {
    const Scalar v = scalar_difference_op<Scalar>()(a, b);
    return scalar_product_op<Scalar>()(v, scalar_conjugate_op<Scalar>()(v));
  }
  template <typename Packet>
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Packet packetOp(const Packet& a,
                                                        const Packet& b) const {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSkernelsPScwise_opsDTh mht_4(mht_4_v, 603, "", "./tensorflow/core/kernels/cwise_ops.h", "packetOp");

    const Packet v = scalar_difference_op<Scalar>().packetOp(a, b);
    return scalar_product_op<Scalar>().packetOp(
        v, scalar_conjugate_op<Scalar>().packetOp(v));
  }
};

template <typename Scalar>
struct functor_traits<scalar_squared_difference_op<Scalar>> {
  enum {
    Cost = functor_traits<scalar_difference_op<Scalar>>::Cost +
           functor_traits<scalar_conjugate_op<Scalar>>::Cost +
           functor_traits<scalar_product_op<Scalar>>::Cost,
    PacketAccess = functor_traits<scalar_difference_op<Scalar>>::PacketAccess &&
                   functor_traits<scalar_conjugate_op<Scalar>>::PacketAccess &&
                   functor_traits<scalar_product_op<Scalar>>::PacketAccess
  };
};

// TODO(b/32239616): This kernel should be moved into Eigen and vectorized.
template <typename T, typename Enable = void>
struct google_floor_div {
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE T operator()(const T& x,
                                                     const T& y) const {
    const T z = x / y;
    // Subtract one if there is a remainder and if the inputs have opposite
    // signs. This approach avoids unnecessary overflows.
    return z * y != x && (x < T(0) != y < T(0)) ? z - T(1) : z;
  }
  template <typename Packet>
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Packet packetOp(const Packet& x,
                                                        const Packet& y) const {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSkernelsPScwise_opsDTh mht_5(mht_5_v, 637, "", "./tensorflow/core/kernels/cwise_ops.h", "packetOp");

    Packet zeros = pzero(x);
    Packet x_mask = pcmp_lt(x, zeros);
    Packet y_mask = pcmp_lt(y, zeros);
    Packet x_div_y = pdiv(x, y);
    Packet x_div_y_times_y = pmul(x_div_y, y);
    return pselect(por(peq(x_div_y_times_y, x), peq(x_mask, y_mask)), x_div_y,
                   psub(x_div_y, pones(x)));
  }
};

template <typename T>
struct google_floor_div<
    T, typename std::enable_if<std::is_unsigned<T>::value>::type> {
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE T operator()(const T& x,
                                                     const T& y) const {
    return x / y;
  }
  template <typename Packet>
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Packet packetOp(const Packet& x,
                                                        const Packet& y) const {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePSkernelsPScwise_opsDTh mht_6(mht_6_v, 660, "", "./tensorflow/core/kernels/cwise_ops.h", "packetOp");

    return pdiv(x, y);
  }
};

template <typename Scalar>
struct functor_traits<google_floor_div<Scalar>> {
  enum {
    Cost = 2 * Eigen::internal::scalar_div_cost<
                   Scalar, packet_traits<Scalar>::HasDiv>::value +
           NumTraits<Scalar>::AddCost,
    PacketAccess = packet_traits<Scalar>::HasDiv
  };
};

template <typename T, typename Enable = void>
struct google_floor_div_real {
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE T operator()(const T& x,
                                                     const T& y) const {
    return Eigen::numext::floor(x / y);
  }
  template <typename Packet>
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Packet packetOp(const Packet& x,
                                                        const Packet& y) const {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePSkernelsPScwise_opsDTh mht_7(mht_7_v, 686, "", "./tensorflow/core/kernels/cwise_ops.h", "packetOp");

    return pfloor(pdiv(x, y));
  }
};

template <typename Scalar>
struct functor_traits<google_floor_div_real<Scalar>> {
  enum {
    Cost = 2 * Eigen::internal::scalar_div_cost<
                   Scalar, packet_traits<Scalar>::HasDiv>::value +
           2 * NumTraits<Scalar>::AddCost,
    PacketAccess =
        packet_traits<Scalar>::HasDiv && packet_traits<Scalar>::HasFloor
  };
};

// TODO(rmlarsen): Add vectorized mod & fmod in Eigen and use it here.
template <typename T>
struct google_floor_fmod {
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE T operator()(const T& x,
                                                     const T& y) const {
    // EIGEN_STATIC_ASSERT(NUMERIC_TYPE_MUST_BE_REAL);
    T trunc_mod = scalar_fmod_op<T>()(x, y);
    return trunc_mod != T(0) && (y < T(0) != trunc_mod < T(0)) ? trunc_mod + y
                                                               : trunc_mod;
  }
};

template <typename Scalar>
struct functor_traits<google_floor_fmod<Scalar>> {
  enum {
    Cost = functor_traits<Eigen::internal::scalar_fmod_op<Scalar>>::Cost +
           NumTraits<Scalar>::AddCost,
    PacketAccess = false
  };
};

// TODO(rmlarsen): Add vectorized mod & fmod in Eigen and use it here.
template <typename T>
struct google_floor_mod {
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE T operator()(const T& x,
                                                     const T& y) const {
    // EIGEN_STATIC_ASSERT(!NUMERIC_TYPE_MUST_BE_REAL);
    T trunc_mod = Eigen::internal::scalar_mod2_op<T>()(x, y);
    return trunc_mod != T(0) && (y < T(0) != trunc_mod < T(0)) ? trunc_mod + y
                                                               : trunc_mod;
  }
};

template <typename Scalar>
struct functor_traits<google_floor_mod<Scalar>> {
  enum {
    Cost = functor_traits<Eigen::internal::scalar_mod2_op<Scalar>>::Cost +
           NumTraits<Scalar>::AddCost,
    PacketAccess = false
  };
};

#if EIGEN_COMP_GNUC && __cplusplus > 199711L
#define DISABLE_FLOAT_EQUALITY_WARNING \
  _Pragma("GCC diagnostic push")       \
      _Pragma("GCC diagnostic ignored \"-Wfloat-equal\"")
#define ENABLE_FLOAT_EQUALITY_WARNING _Pragma("GCC diagnostic pop")
#else
#define DISABLE_FLOAT_EQUALITY_WARNING
#define ENABLE_FLOAT_EQUALITY_WARNING
#endif

template <typename Scalar, bool IsInteger = Eigen::NumTraits<Scalar>::IsInteger,
          bool HasRint = packet_traits<Scalar>::HasRint>
struct scalar_round_half_to_even_op {
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Scalar
  operator()(const Scalar& x) const {
    EIGEN_STATIC_ASSERT((!NumTraits<Scalar>::IsComplex),
                        NUMERIC_TYPE_MUST_BE_REAL)

    const Scalar round_val = Eigen::numext::floor(x + Scalar(0.5));
    const Scalar fraction = round_val - x;
    if (TF_PREDICT_FALSE(fraction == Scalar(.5))) {
      return Scalar(2) * Eigen::numext::floor(Scalar(.5) * x + Scalar(0.5));
    } else {
      return round_val;
    }
  }

  template <typename Packet>
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Packet packetOp(const Packet& x) const {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScorePSkernelsPScwise_opsDTh mht_8(mht_8_v, 775, "", "./tensorflow/core/kernels/cwise_ops.h", "packetOp");

    Packet half = pset1<Packet>(Scalar(0.5));
    Packet round_val = pfloor(padd(x, half));
    Packet fraction = psub(round_val, x);
    Packet half_mask = pcmp_eq(fraction, half);
    bool any_halves = predux_any(half_mask);
    if (TF_PREDICT_FALSE(any_halves)) {
      Packet two = pset1<Packet>(Scalar(2));
      Packet nearest_even = pmul(two, pfloor(pmadd(half, x, half)));
      return pselect(half_mask, nearest_even, round_val);
    } else {
      return round_val;
    }
  }
};

template <typename Scalar>
struct scalar_round_half_to_even_op<Scalar, true, false> {
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Scalar
  operator()(const Scalar& x) const {
    return x;
  }
  template <typename Packet>
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Packet packetOp(const Packet& x) const {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScorePSkernelsPScwise_opsDTh mht_9(mht_9_v, 801, "", "./tensorflow/core/kernels/cwise_ops.h", "packetOp");

    return x;
  }
};

template <typename Scalar>
struct scalar_round_half_to_even_op<Scalar, false, true> {
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Scalar
  operator()(const Scalar& x) const {
    return Eigen::numext::rint(x);
  }
  template <typename Packet>
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Packet packetOp(const Packet& x) const {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScorePSkernelsPScwise_opsDTh mht_10(mht_10_v, 816, "", "./tensorflow/core/kernels/cwise_ops.h", "packetOp");

    return print(x);
  }
};

template <typename Scalar>
struct functor_traits<scalar_round_half_to_even_op<Scalar>> {
  enum {
    Cost = Eigen::NumTraits<Scalar>::IsInteger ? 0
                                               : 4 * NumTraits<Scalar>::AddCost,
    PacketAccess = packet_traits<Scalar>::HasFloor &&
                   packet_traits<Scalar>::HasAdd &&
                   packet_traits<Scalar>::HasMul,
  };
};

template <typename Scalar, bool IsInteger = Eigen::NumTraits<Scalar>::IsInteger>
struct scalar_round_up_op {
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Scalar
  operator()(const Scalar& x) const {
    EIGEN_STATIC_ASSERT((!NumTraits<Scalar>::IsComplex),
                        NUMERIC_TYPE_MUST_BE_REAL)
    return Eigen::numext::floor(x + Scalar(0.5));
  }

  template <typename Packet>
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Packet packetOp(const Packet& x) const {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScorePSkernelsPScwise_opsDTh mht_11(mht_11_v, 845, "", "./tensorflow/core/kernels/cwise_ops.h", "packetOp");

    return pfloor(padd(x, pset1<Packet>(0.5)));
  }
};

template <typename Scalar>
struct scalar_round_up_op<Scalar, true> {
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Scalar
  operator()(const Scalar& x) const {
    return x;
  }

  template <typename Packet>
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Packet packetOp(const Packet& x) const {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScorePSkernelsPScwise_opsDTh mht_12(mht_12_v, 861, "", "./tensorflow/core/kernels/cwise_ops.h", "packetOp");

    return x;
  }
};

template <typename Scalar, bool IsInteger>
struct functor_traits<scalar_round_up_op<Scalar, IsInteger>> {
  enum {
    Cost = IsInteger ? 0 : 4 * NumTraits<Scalar>::AddCost,
    PacketAccess = IsInteger || packet_traits<Scalar>::HasFloor
  };
};

#undef ENABLE_FLOAT_EQUALITY_WARNING
#undef DISABLE_FLOAT_EQUALITY_WARNING

template <typename Scalar>
struct bitwise_xor_op {
  EIGEN_EMPTY_STRUCT_CTOR(bitwise_xor_op)
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Scalar
  operator()(const Scalar& x, const Scalar& y) const {
    return x ^ y;
  }
  template <typename Packet>
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Packet packetOp(const Packet& a,
                                                        const Packet& b) const {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPScorePSkernelsPScwise_opsDTh mht_13(mht_13_v, 889, "", "./tensorflow/core/kernels/cwise_ops.h", "packetOp");

    return Eigen::internal::pxor(a, b);
  }
};

template <typename Scalar>
struct functor_traits<bitwise_xor_op<Scalar>> {
  enum { Cost = Eigen::NumTraits<Scalar>::AddCost, PacketAccess = true };
};

template <typename Scalar>
struct xlogy_op {
  EIGEN_EMPTY_STRUCT_CTOR(xlogy_op)
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Scalar
  operator()(const Scalar& x, const Scalar& y) const {
    if (x == Scalar(0.)) {
      return Scalar(0.);
    }
    return x * numext::log(y);
  }
  template <typename Packet>
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Packet packetOp(const Packet& x,
                                                        const Packet& y) const {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPScorePSkernelsPScwise_opsDTh mht_14(mht_14_v, 914, "", "./tensorflow/core/kernels/cwise_ops.h", "packetOp");

    Packet zeros = pzero(x);
    Packet mask = pcmp_eq(x, zeros);
    scalar_log_op<Scalar> log_op;
    Packet log_y = log_op.packetOp(y);
    Packet x_log_y = pmul(x, log_y);
    return pselect(mask, x, x_log_y);
  }
};

template <typename Scalar>
struct functor_traits<xlogy_op<Scalar>> {
  enum {
    Cost = functor_traits<scalar_log_op<Scalar>>::Cost +
           Eigen::NumTraits<Scalar>::MulCost,
    PacketAccess = functor_traits<scalar_log_op<Scalar>>::PacketAccess
  };
};

template <typename Scalar>
struct xlog1py_op {
  EIGEN_EMPTY_STRUCT_CTOR(xlog1py_op)
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Scalar
  operator()(const Scalar& x, const Scalar& y) const {
    if (x == Scalar(0.)) {
      return Scalar(0.);
    }
    return x * numext::log1p(y);
  }
  template <typename Packet>
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Packet packetOp(const Packet& x,
                                                        const Packet& y) const {
   std::vector<std::string> mht_15_v;
   MHTracer_DTPStensorflowPScorePSkernelsPScwise_opsDTh mht_15(mht_15_v, 948, "", "./tensorflow/core/kernels/cwise_ops.h", "packetOp");

    Packet zeros = pzero(x);
    Packet mask = pcmp_eq(x, zeros);
    scalar_log1p_op<Scalar> log1p_op;
    Packet log1p_y = log1p_op.packetOp(y);
    Packet x_log1p_y = pmul(x, log1p_y);
    return pselect(mask, x, x_log1p_y);
  }
};

template <typename Scalar>
struct functor_traits<xlog1py_op<Scalar>> {
  enum {
    Cost = functor_traits<scalar_log1p_op<Scalar>>::Cost +
           Eigen::NumTraits<Scalar>::MulCost,
#if TENSORFLOW_USE_ROCM
    PacketAccess = false,
#else
    PacketAccess = functor_traits<scalar_log1p_op<Scalar>>::PacketAccess
#endif
  };
};

template <typename Scalar>
struct xdivy_op {
  EIGEN_EMPTY_STRUCT_CTOR(xdivy_op)
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Scalar
  operator()(const Scalar& x, const Scalar& y) const {
    if (x == Scalar(0.)) {
      return Scalar(0.);
    }
    return x / y;
  }
  template <typename Packet>
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Packet packetOp(const Packet& x,
                                                        const Packet& y) const {
   std::vector<std::string> mht_16_v;
   MHTracer_DTPStensorflowPScorePSkernelsPScwise_opsDTh mht_16(mht_16_v, 986, "", "./tensorflow/core/kernels/cwise_ops.h", "packetOp");

    Packet zeros = pzero(x);
    Packet mask = pcmp_eq(x, zeros);
    Packet x_div_y = pdiv(x, y);
    return pselect(mask, x, x_div_y);
  }
};

template <typename Scalar>
struct functor_traits<xdivy_op<Scalar>> {
  enum {
    Cost =
        Eigen::NumTraits<Scalar>::AddCost +
        Eigen::internal::scalar_div_cost<Scalar,
                                         packet_traits<Scalar>::HasDiv>::value,
    PacketAccess = packet_traits<Scalar>::HasDiv
  };
};

template <typename T>
struct scalar_erfinv_op {
  EIGEN_EMPTY_STRUCT_CTOR(scalar_erfinv_op)
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE T operator()(const T& x) const {
    constexpr T half = T(0.5);
    T y = numext::ndtri(half * x + half);
    constexpr T half_sqrt = T(M_SQRT1_2);
    return y * half_sqrt;
  }
  template <typename Packet>
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Packet packetOp(const Packet& x) const {
   std::vector<std::string> mht_17_v;
   MHTracer_DTPStensorflowPScorePSkernelsPScwise_opsDTh mht_17(mht_17_v, 1018, "", "./tensorflow/core/kernels/cwise_ops.h", "packetOp");

    Packet half = pset1<Packet>(T(0.5));
    Packet y = pndtri<Packet>(pmadd(half, x, half));
    Packet half_sqrt = pset1<Packet>(T(M_SQRT1_2));
    return pmul(y, half_sqrt);
  }
};

template <typename T>
struct functor_traits<scalar_erfinv_op<T>> {
  enum {
    Cost = functor_traits<scalar_ndtri_op<T>>::Cost + NumTraits<T>::AddCost,
    PacketAccess = packet_traits<T>::HasNdtri,
  };
};

}  // end namespace internal
}  // end namespace Eigen

namespace tensorflow {
namespace functor {

////////////////////////////////////////////////////////////////////////////////
// Helpers
////////////////////////////////////////////////////////////////////////////////

// Base template for functors whose input scalar type is T and
// output scalar type is R.
template <typename T, typename F, typename R = T>
struct base {
  // func defines operator() and its vectorized version packetOp().
  typedef F func;

  // If true, the functor's corresponding binary op will instantiate
  // specialized kernels to perform an optimized broadcast
  // operation. Each functor for which this is enabled increases the
  // code size, so by default this is disabled for binary functors and
  // is enabled on a per-op basis as needed.
  static constexpr bool use_bcast_optimization = false;

  // operator() has the signature:
  //  out_type operator()(in_type in0, in_type in1 ...)
  typedef R out_type;
  typedef T in_type;

  // TensorFlow provides tensor-ized version of "func". Roughly
  // speaking, the tensorflow operation has the signature:
  //   tout_type op(tin_type in0)
  //   tout_type op(tin_type in0, tin_type in1)
  //   tout_type op(tin_type in0, in_type scalar)
  typedef typename TTypes<out_type>::Flat tout_type;
  typedef typename TTypes<in_type>::ConstFlat tin_type;
  typedef typename TTypes<in_type>::ConstScalar tscalar_type;

  // Whether the functor can error out.  Currently applies only to integer
  // div and mod.
  static constexpr bool has_errors = false;
};

// For now, we only apply certain speed optimization for
// float/double's broadcast binary op.
template <typename T>
struct use_bcast_optimization {
  static constexpr bool value = false;
};

template <>
struct use_bcast_optimization<float> {
  static constexpr bool value = true;
};

template <>
struct use_bcast_optimization<double> {
  static constexpr bool value = true;
};

////////////////////////////////////////////////////////////////////////////////
// Unary functors
////////////////////////////////////////////////////////////////////////////////

// abs(x) = |x|
// neg(x) = - x
// inverse(x) = 1 / x
// square(x) = x^2
// sqrt(x) = x^(1/2)
// rsqrt(x) = x^(-1/2)
// exp(x) = e^x
// expm1(x) = e^x - 1
// log(x) = natural logarithm of x
// log1p(x) = natural logarithm of 1 + x
// tanh = (exp(x) - exp(-x)) / (exp(x) + exp(-x))
// sigmoid = 1 / (1 + exp(-x))  // a.k.a, logistic
//
// NOTE: We may eventually implement common functions used in NN
// here. E.g., rectifier, softplus, derivatives of tanh, sigmod, etc.
// For reference, see speech/lstm/eigen_functors.h.

template <typename T>
struct abs : base<T, Eigen::internal::scalar_abs_op<T>,
                  typename Eigen::internal::scalar_abs_op<T>::result_type> {};

template <typename T>
struct neg : base<T, Eigen::internal::scalar_opposite_op<T>> {};

template <typename T>
struct inverse : base<T, Eigen::internal::scalar_inverse_op<T>> {};

template <typename T>
struct square : base<T, Eigen::internal::scalar_square_op<T>> {};

template <typename T>
struct sqrt : base<T, Eigen::internal::scalar_sqrt_op<T>> {};

template <typename T>
struct rsqrt : base<T, Eigen::internal::scalar_rsqrt_op<T>> {};

template <typename T>
struct exp : base<T, Eigen::internal::scalar_exp_op<T>> {};

template <typename T>
struct expm1 : base<T, Eigen::internal::scalar_expm1_op<T>> {};

template <typename T>
struct log : base<T, Eigen::internal::scalar_log_op<T>> {};

template <typename T>
struct log1p : base<T, Eigen::internal::scalar_log1p_op<T>> {};

template <typename T>
struct sign : base<T, Eigen::internal::scalar_sign_op<T>> {};

template <typename T>
struct sinh : base<T, Eigen::internal::scalar_sinh_op<T>> {};

template <typename T>
struct cosh : base<T, Eigen::internal::scalar_cosh_op<T>> {};

template <typename T>
struct tanh : base<T, Eigen::internal::scalar_tanh_op<T>> {};

template <typename T>
struct asinh : base<T, Eigen::internal::scalar_asinh_op<T>> {};

template <typename T>
struct acosh : base<T, Eigen::internal::scalar_acosh_op<T>> {};

template <typename T>
struct atanh : base<T, Eigen::internal::scalar_atanh_op<T>> {};

template <typename T>
struct lgamma : base<T, Eigen::internal::scalar_lgamma_op<T>> {};

template <typename T>
struct digamma : base<T, Eigen::internal::scalar_digamma_op<T>> {};

template <typename T>
struct erf : base<T, Eigen::internal::scalar_erf_op<T>> {};

template <typename T>
struct erfc : base<T, Eigen::internal::scalar_erfc_op<T>> {};

template <typename T>
struct ndtri : base<T, Eigen::internal::scalar_ndtri_op<T>> {};

template <typename T>
struct erfinv : base<T, Eigen::internal::scalar_erfinv_op<T>> {};

template <typename T>
struct sigmoid : base<T, Eigen::internal::scalar_logistic_op<T>> {};

template <typename T>
struct sin : base<T, Eigen::internal::scalar_sin_op<T>> {};

template <typename T>
struct cos : base<T, Eigen::internal::scalar_cos_op<T>> {};

template <typename T>
struct tan : base<T, Eigen::internal::scalar_tan_op<T>> {};

template <typename T>
struct asin : base<T, Eigen::internal::scalar_asin_op<T>> {};

template <typename T>
struct acos : base<T, Eigen::internal::scalar_acos_op<T>> {};

template <typename T>
struct atan : base<T, Eigen::internal::scalar_atan_op<T>> {};

struct logical_not : base<bool, Eigen::internal::scalar_boolean_not_op<bool>> {
};

// Flip all bits. Named invert to be consistent with numpy.
template <typename T>
struct invert_op {
  EIGEN_EMPTY_STRUCT_CTOR(invert_op)
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE T operator()(const T& a) const {
    return ~a;
  }
};

template <typename T>
struct invert : base<T, invert_op<T>> {};

// NOTE: std::isinf, std::isnan, std::isfinite are plain function.
// Therefore we need to wrap them in functors to be used with Eigen's
// type system.
template <typename T>
struct isinf : base<T, Eigen::internal::scalar_isinf_op<T>, bool> {};

template <typename T>
struct isnan : base<T, Eigen::internal::scalar_isnan_op<T>, bool> {};

template <typename T>
struct isfinite : base<T, Eigen::internal::scalar_isfinite_op<T>, bool> {};

template <typename T>
struct floor : base<T, Eigen::internal::scalar_floor_op<T>> {};

template <typename T>
struct round : base<T, Eigen::internal::scalar_round_half_to_even_op<T>> {};

template <typename T>
struct ceil : base<T, Eigen::internal::scalar_ceil_op<T>> {};

// Note: rint rounds half values to even, just like round_half_to_even_op.
template <typename T>
struct rint : base<T, Eigen::internal::scalar_rint_op<T>> {};

////////////////////////////////////////////////////////////////////////////////
// Binary functors
////////////////////////////////////////////////////////////////////////////////

// Binary functors:
//
// add(x, y) = x + y
// sub(x, y) = x - y
// mul(x, y) = x * y
// div(x, y) = x / y
// mod(x, y) = x % y         (int32 and int64 only)
// fmod(x, y) = fmod(x, y)   (float and double only)
// pow(x, y) = x ^ y
// maximum(x, y) = x > y ? x : y
// minimum(x, y) = x < y ? x : y
// squared_difference(x, y) = conj(x - y) * (x - y)

template <typename T>
struct add : base<T, Eigen::internal::scalar_sum_op<T>> {
  static constexpr bool use_bcast_optimization = true;
};

template <typename T>
struct sub : base<T, Eigen::internal::scalar_difference_op<T>> {
  static constexpr bool use_bcast_optimization = true;
};

template <typename T>
struct mul : base<T, Eigen::internal::scalar_product_op<T>> {
  static constexpr bool use_bcast_optimization = true;
};

template <typename T>
struct mul_no_nan : base<T, Eigen::internal::mul_no_nan_op<T>> {};

template <typename T>
struct div : base<T, Eigen::internal::scalar_quotient_op<T>> {};

template <typename T>
struct safe_div : base<T, Eigen::internal::safe_div_or_mod_op<
                              T, Eigen::internal::scalar_quotient_op<T>>> {
  static constexpr bool has_errors = true;
};

template <typename T>
struct div_no_nan : base<T, Eigen::internal::div_no_nan_op<T>> {};

template <typename T>
struct fmod : base<T, Eigen::internal::scalar_fmod_op<T>> {};

template <typename T>
struct mod : base<T, Eigen::internal::scalar_mod2_op<T>> {};

template <typename T>
struct safe_mod : base<T, Eigen::internal::safe_div_or_mod_op<
                              T, Eigen::internal::scalar_mod2_op<T>>> {
  static constexpr bool has_errors = true;
};

template <typename T>
struct floor_fmod : base<T, Eigen::internal::google_floor_fmod<T>> {};

template <typename T>
struct safe_floor_mod : base<T, Eigen::internal::safe_div_or_mod_op<
                                    T, Eigen::internal::google_floor_mod<T>>> {
  static constexpr bool has_errors = true;
};

template <typename T>
struct floor_div : base<T, Eigen::internal::google_floor_div<T>> {};

template <typename T>
struct safe_floor_div : base<T, Eigen::internal::safe_div_or_mod_op<
                                    T, Eigen::internal::google_floor_div<T>>> {
  static constexpr bool has_errors = true;
};

template <typename T>
struct floor_div_real : base<T, Eigen::internal::google_floor_div_real<T>> {};

template <typename T>
struct pow : base<T, Eigen::internal::scalar_pow_op<T, T>> {};

template <typename T>
struct safe_pow : base<T, Eigen::internal::safe_scalar_binary_pow_op<T, T>> {
  static constexpr bool has_errors = true;
};

// Version of safe_pow for integers which returns 0 if RHS is negative and LHS
// is not 1 or -1. For use on GPUs, where we cannot raise an error.
template <typename T>
struct safe_pow_ignore_error_op {
  static_assert(std::is_integral<T>::value, "Integer type expected");
  EIGEN_EMPTY_STRUCT_CTOR(safe_pow_ignore_error_op)
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE T operator()(const T& x,
                                                     const T& y) const {
    if (TF_PREDICT_FALSE(y < 0)) {
      if (x == T(-1)) {
        T trunc_mod = Eigen::internal::scalar_mod2_op<T>()(y, T(2));
        return trunc_mod == T(-1) ? T(-1) : T(1);
      }
      return x == T(1) ? T(1) : T(0);
    }
    return Eigen::internal::scalar_pow_op<T, T>{}(x, y);
  }
};

template <typename T>
struct safe_pow_ignore_error : base<T, safe_pow_ignore_error_op<T>> {};

template <typename T>
struct maximum
    : base<T, Eigen::internal::scalar_max_op<T, T, Eigen::PropagateNaN>> {};

template <typename T>
struct minimum
    : base<T, Eigen::internal::scalar_min_op<T, T, Eigen::PropagateNaN>> {};

template <typename T>
struct igamma : base<T, Eigen::internal::scalar_igamma_op<T>> {};

template <typename T>
struct random_gamma_grad
    : base<T, Eigen::internal::scalar_gamma_sample_der_alpha_op<T>> {};

template <typename T>
struct igammac : base<T, Eigen::internal::scalar_igammac_op<T>> {};

template <typename T>
struct zeta : base<T, Eigen::internal::scalar_zeta_op<T>> {};

template <typename T>
struct polygamma : base<T, Eigen::internal::scalar_polygamma_op<T>> {};

template <typename Scalar>
struct scalar_atan2_op {
  EIGEN_EMPTY_STRUCT_CTOR(scalar_atan2_op)
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Scalar
  operator()(const Scalar& y, const Scalar& x) const {
#if TENSORFLOW_USE_ROCM
    return static_cast<Scalar>(::atan2(y, x));
#else
    return static_cast<Scalar>(std::atan2(y, x));
#endif
  }
};

template <typename T>
struct atan2 : base<T, scalar_atan2_op<T>> {};

template <typename T>
struct squared_difference
    : base<T, Eigen::internal::scalar_squared_difference_op<T>> {};

template <typename T>
struct xdivy : base<T, Eigen::internal::xdivy_op<T>> {};

template <typename T>
struct xlogy : base<T, Eigen::internal::xlogy_op<T>> {};

template <typename T>
struct xlog1py : base<T, Eigen::internal::xlog1py_op<T>> {};

template <typename T>
struct less : base<T, Eigen::internal::less<T>, bool> {};

template <typename T>
struct less_equal : base<T, Eigen::internal::less_equal<T>, bool> {};

template <typename T>
struct greater : base<T, Eigen::internal::greater<T>, bool> {};

template <typename T>
struct greater_equal : base<T, Eigen::internal::greater_equal<T>, bool> {};

template <typename T>
struct equal_to : base<T, Eigen::internal::equal_to<T>, bool> {};

template <typename T>
struct not_equal_to : base<T, Eigen::internal::not_equal_to<T>, bool> {};

struct logical_and : base<bool, Eigen::internal::scalar_boolean_and_op> {};

struct logical_or : base<bool, Eigen::internal::scalar_boolean_or_op> {};

template <typename T>
struct bitwise_and_op {
  EIGEN_EMPTY_STRUCT_CTOR(bitwise_and_op)
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE T operator()(const T& x,
                                                     const T& y) const {
    return x & y;
  }
};

template <typename T>
struct bitwise_or_op {
  EIGEN_EMPTY_STRUCT_CTOR(bitwise_or_op)
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE T operator()(const T& x,
                                                     const T& y) const {
    return x | y;
  }
};

template <typename T>
struct bitwise_and : base<T, bitwise_and_op<T>> {};

template <typename T>
struct bitwise_or : base<T, bitwise_or_op<T>> {};

template <typename T>
struct bitwise_xor : base<T, Eigen::internal::bitwise_xor_op<T>> {};

template <typename T>
struct left_shift_op {
  EIGEN_EMPTY_STRUCT_CTOR(left_shift_op)
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE T operator()(const T& x,
                                                     const T& y) const {
    // Avoids UB: don't shift by larger than the bitwidth of T, and
    // performs left shifts as unsigned shifts.
    T y_clamped = y;
    if (y_clamped < 0) {
      y_clamped = 0;
    } else if (y_clamped > sizeof(T) * CHAR_BIT - 1) {
      y_clamped = sizeof(T) * CHAR_BIT - 1;
    }
    using U = typename std::make_unsigned<T>::type;
    return static_cast<T>(static_cast<U>(x) << static_cast<U>(y_clamped));
  }
};

template <typename T>
struct right_shift_op {
  EIGEN_EMPTY_STRUCT_CTOR(right_shift_op)
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE T operator()(const T& x,
                                                     const T& y) const {
    // Avoids UB: don't shift by larger than the bitwidth of T.
    T y_clamped = y;
    if (y_clamped < 0) {
      y_clamped = 0;
    } else if (y_clamped > sizeof(T) * CHAR_BIT - 1) {
      y_clamped = sizeof(T) * CHAR_BIT - 1;
    }
    // Technically right shifts of signed integers are not necessarily
    // arithmetic shifts according to the C++ standard. However in practice most
    // implementations are arithmetic shifts. If this proves to be a problem in
    // practice, we may need to use an alternative implementation.
    return x >> y_clamped;
  }
};

template <typename T>
struct left_shift : base<T, left_shift_op<T>> {};

template <typename T>
struct right_shift : base<T, right_shift_op<T>> {};

template <typename T>
struct make_complex_func {
  typedef std::complex<T> result_type;
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE result_type operator()(T real,
                                                               T imag) const {
    return std::complex<T>(real, imag);
  }
};

template <typename T>
struct make_complex : base<T, make_complex_func<T>, std::complex<T>> {};

template <typename T>
struct get_real
    : base<T, Eigen::internal::scalar_real_op<T>, typename T::value_type> {};

template <typename T>
struct get_imag
    : base<T, Eigen::internal::scalar_imag_op<T>, typename T::value_type> {};

template <typename T>
struct get_angle
    : base<T, Eigen::internal::scalar_arg_op<T>, typename T::value_type> {};

template <typename T>
struct conj : base<T, Eigen::internal::scalar_conjugate_op<T>> {};

////////////////////////////////////////////////////////////////////////////////
// Functors takes 1 or 2 tensors, computes the base functor on
// coefficient of the input tensors and puts the results in the output
// tensor.
////////////////////////////////////////////////////////////////////////////////
template <typename Device, typename Functor>
struct UnaryFunctor {
  // Computes on device "d": out[i] = Functor(in[i])
  void operator()(const Device& d, typename Functor::tout_type out,
                  typename Functor::tin_type in);
};

template <typename Device, typename Functor, typename Targ>
struct UnaryFunctorWithArg {
  // Computes on device "d": out[i] = Functor(in[i])
  void operator()(const Device& d, typename Functor::tout_type out,
                  typename Functor::tin_type in, Targ val);
};

template <typename Device, typename Functor, int NDIMS,
          bool has_errors = Functor::has_errors>
struct BinaryFunctor {
  // Computes on device "d": out[i] = Functor(in0[i], in1[i])
  void operator()(const Device& d, typename Functor::tout_type out,
                  typename Functor::tin_type in0,
                  typename Functor::tin_type in1, bool* error);

  // Computes on device "d": out[i] = Functor(scalar[0], in[i])
  void Left(const Device& d, typename Functor::tout_type out,
            typename Functor::tscalar_type scalar,
            typename Functor::tin_type in, bool* error);

  // Computes on device "d": out[i] = Functor(in[i], scalar[0])
  void Right(const Device& d, typename Functor::tout_type out,
             typename Functor::tin_type in,
             typename Functor::tscalar_type scalar, bool* error);

  // Computes on device "d":
  //   out = Functor(in0.broadcast(bcast0), in1.broadcast(bcast1))
  //
  // TODO(zhifengc): makes BCast a template member function on NDIMS
  // instead making BinaryFunctor templates on NDIMS.
  void BCast(const Device& d,
             typename TTypes<typename Functor::out_type, NDIMS>::Tensor out,
             typename TTypes<typename Functor::in_type, NDIMS>::ConstTensor in0,
             typename Eigen::array<Eigen::DenseIndex, NDIMS> bcast0,
             typename TTypes<typename Functor::in_type, NDIMS>::ConstTensor in1,
             typename Eigen::array<Eigen::DenseIndex, NDIMS> bcast1,
             bool* error);
};

template <typename Device, typename T>
struct ApproximateEqual {
  void operator()(const Device& d, typename TTypes<T>::ConstFlat x,
                  typename TTypes<T>::ConstFlat y, T tolerance,
                  typename TTypes<bool>::Flat z);
};

template <int NDIMS>
bool AllOne(const typename Eigen::array<Eigen::DenseIndex, NDIMS>& a) {
   std::vector<std::string> mht_18_v;
   MHTracer_DTPStensorflowPScorePSkernelsPScwise_opsDTh mht_18(mht_18_v, 1591, "", "./tensorflow/core/kernels/cwise_ops.h", "AllOne");

  for (size_t i = 0; i < a.size(); ++i) {
    if (a[i] != 1) return false;
  }
  return true;
}

template <typename Device, typename T>
struct SelectFunctor {
  void operator()(const Device& d, typename TTypes<T>::Flat out,
                  typename TTypes<bool>::ConstFlat cond_flat,
                  typename TTypes<T>::ConstFlat then_flat,
                  typename TTypes<T>::ConstFlat else_flat);
};

template <typename Device, typename T>
struct SelectScalarFunctor {
  void operator()(const Device& d, typename TTypes<T>::Flat out,
                  typename TTypes<bool>::ConstScalar cond,
                  typename TTypes<T>::ConstFlat then_flat,
                  typename TTypes<T>::ConstFlat else_flat);
};

template <typename Device, typename T>
struct BatchSelectFunctor {
  void operator()(const Device& d,
                  typename TTypes<T>::Matrix output_flat_outer_dims,
                  TTypes<bool>::ConstVec cond_vec,
                  typename TTypes<T>::ConstMatrix then_flat_outer_dims,
                  typename TTypes<T>::ConstMatrix else_flat_outer_dims);
};

template <typename Device, typename T, int NDIMS>
struct BCastSelectFunctor {
  void operator()(const Device& d,
                  typename TTypes<T, NDIMS>::Tensor output_tensor,
                  typename TTypes<bool, NDIMS>::ConstTensor cond_tensor,
                  typename TTypes<T, NDIMS>::ConstTensor then_tensor,
                  typename TTypes<T, NDIMS>::ConstTensor else_tensor,
                  typename Eigen::array<Eigen::DenseIndex, NDIMS> cond_bcast,
                  typename Eigen::array<Eigen::DenseIndex, NDIMS> then_bcast,
                  typename Eigen::array<Eigen::DenseIndex, NDIMS> else_bcast);
};

}  // end namespace functor
}  // end namespace tensorflow

#endif  // TENSORFLOW_CORE_KERNELS_CWISE_OPS_H_
