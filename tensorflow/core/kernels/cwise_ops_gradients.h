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

#ifndef TENSORFLOW_CORE_KERNELS_CWISE_OPS_GRADIENTS_H_
#define TENSORFLOW_CORE_KERNELS_CWISE_OPS_GRADIENTS_H_
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
class MHTracer_DTPStensorflowPScorePSkernelsPScwise_ops_gradientsDTh {
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
   MHTracer_DTPStensorflowPScorePSkernelsPScwise_ops_gradientsDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSkernelsPScwise_ops_gradientsDTh() {
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


#define EIGEN_USE_THREADS
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/kernels/cwise_ops.h"

namespace Eigen {
namespace internal {

// Gradient for the tanh function
template <typename T>
struct scalar_tanh_gradient_op {
  EIGEN_EMPTY_STRUCT_CTOR(scalar_tanh_gradient_op)
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE const T
  operator()(const T& output, const T& output_gradient) const {
    return output_gradient * (T(1) - output * output);
  }
  template <typename Packet>
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE const Packet
  packetOp(const Packet& output, const Packet& output_gradient) const {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSkernelsPScwise_ops_gradientsDTh mht_0(mht_0_v, 205, "", "./tensorflow/core/kernels/cwise_ops_gradients.h", "packetOp");

    return pmul(output_gradient,
                psub(pset1<Packet>(T(1)), pmul(output, output)));
  }
};
template <typename T>
struct functor_traits<scalar_tanh_gradient_op<T>> {
  enum {
    Cost = NumTraits<T>::AddCost + 2 * NumTraits<T>::MulCost,
    PacketAccess = packet_traits<T>::HasSub && packet_traits<T>::HasMul,
  };
};

// Gradient for the sigmoid function
template <typename T>
struct scalar_sigmoid_gradient_op {
  EIGEN_EMPTY_STRUCT_CTOR(scalar_sigmoid_gradient_op)
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE const T
  operator()(const T& output, const T& output_gradient) const {
    return output_gradient * output * (T(1) - output);
  }
  template <typename Packet>
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE const Packet
  packetOp(const Packet& output, const Packet& output_gradient) const {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSkernelsPScwise_ops_gradientsDTh mht_1(mht_1_v, 231, "", "./tensorflow/core/kernels/cwise_ops_gradients.h", "packetOp");

    return pmul(output_gradient,
                pmul(output, psub(pset1<Packet>(T(1)), output)));
  }
};
template <typename T>
struct functor_traits<scalar_sigmoid_gradient_op<T>> {
  enum {
    Cost = NumTraits<T>::AddCost + 2 * NumTraits<T>::MulCost,
    PacketAccess = packet_traits<T>::HasSub && packet_traits<T>::HasMul,
  };
};

// Gradient for the inverse function
template <typename T>
struct scalar_inverse_gradient_op {
  EIGEN_EMPTY_STRUCT_CTOR(scalar_inverse_gradient_op)
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE const T
  operator()(const T& output, const T& output_gradient) const {
    if (output_gradient == T(0)) {
      return T(0);
    } else {
      const T out_conj = numext::conj(output);
      return -out_conj * out_conj * output_gradient;
    }
  }
  template <typename Packet>
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE const Packet
  packetOp(const Packet& output, const Packet& output_gradient) const {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSkernelsPScwise_ops_gradientsDTh mht_2(mht_2_v, 262, "", "./tensorflow/core/kernels/cwise_ops_gradients.h", "packetOp");

    const Packet out_conj = pconj(output);
    return mul_no_nan_op<T>().packetOp(pnegate(pmul(out_conj, out_conj)),
                                       output_gradient);
  }
};
template <typename T>
struct functor_traits<scalar_inverse_gradient_op<T>> {
  enum {
    Cost = NumTraits<T>::AddCost + 2 * NumTraits<T>::MulCost,
    PacketAccess = packet_traits<T>::HasMul,
  };
};

// Gradient for the sqrt function
template <typename T>
struct scalar_sqrt_gradient_op {
  EIGEN_EMPTY_STRUCT_CTOR(scalar_sqrt_gradient_op)
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE const T
  operator()(const T& output, const T& output_gradient) const {
    if (output_gradient == T(0)) {
      return T(0);
    } else {
      const T out_conj = numext::conj(output);
      return (static_cast<T>(0.5) * output_gradient) / out_conj;
    }
  }
  template <typename Packet>
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE const Packet
  packetOp(const Packet& output, const Packet& output_gradient) const {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSkernelsPScwise_ops_gradientsDTh mht_3(mht_3_v, 294, "", "./tensorflow/core/kernels/cwise_ops_gradients.h", "packetOp");

    const Packet const_half = pset1<Packet>(static_cast<T>(0.5));
    const Packet out_conj = pconj(output);
    return mul_no_nan_op<T>().packetOp(pdiv(const_half, out_conj),
                                       output_gradient);
  }
};
template <typename T>
struct functor_traits<scalar_sqrt_gradient_op<T>> {
  enum {
    PacketAccess = packet_traits<T>::HasMul & packet_traits<T>::HasDiv,
    Cost = NumTraits<T>::MulCost + scalar_div_cost<T, PacketAccess>::value,
  };
};

// Gradient for the rsqrt function
template <typename T>
struct scalar_rsqrt_gradient_op {
  EIGEN_EMPTY_STRUCT_CTOR(scalar_rsqrt_gradient_op)
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE const T
  operator()(const T& output, const T& output_gradient) const {
    if (output_gradient == T(0)) {
      return T(0);
    } else {
      const T out_conj = numext::conj(output);
      return static_cast<T>(-0.5) * (output_gradient * out_conj) *
             (out_conj * out_conj);
    }
  }
  template <typename Packet>
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE const Packet
  packetOp(const Packet& output, const Packet& output_gradient) const {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSkernelsPScwise_ops_gradientsDTh mht_4(mht_4_v, 328, "", "./tensorflow/core/kernels/cwise_ops_gradients.h", "packetOp");

    const Packet const_half = pset1<Packet>(static_cast<T>(-0.5));
    const Packet out_conj = pconj(output);
    auto safe_pmul = [](const Packet& a, const Packet& b) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSkernelsPScwise_ops_gradientsDTh mht_5(mht_5_v, 334, "", "./tensorflow/core/kernels/cwise_ops_gradients.h", "lambda");

      return mul_no_nan_op<T>().packetOp(a, b);
    };
    return safe_pmul(pmul(const_half, pmul(out_conj, out_conj)),
                     safe_pmul(out_conj, output_gradient));
  }
};
template <typename T>
struct functor_traits<scalar_rsqrt_gradient_op<T>> {
  enum {
    Cost = 4 * NumTraits<T>::MulCost,
    PacketAccess = packet_traits<T>::HasMul,
  };
};

}  // end namespace internal
}  // end namespace Eigen

namespace tensorflow {

namespace functor {

template <typename Device, typename Functor>
struct SimpleBinaryFunctor {
  void operator()(const Device& d, typename Functor::tout_type out,
                  typename Functor::tin_type in0,
                  typename Functor::tin_type in1);
};

// Partial specialization of BinaryFunctor for CPU devices
typedef Eigen::ThreadPoolDevice CPUDevice;

template <typename Functor>
struct SimpleBinaryFunctor<CPUDevice, Functor> {
  void operator()(const CPUDevice& d, typename Functor::tout_type out,
                  typename Functor::tin_type in0,
                  typename Functor::tin_type in1) {
    out.device(d) = in0.binaryExpr(in1, typename Functor::func());
  }
};


template <typename T>
struct tanh_grad : base<T, Eigen::internal::scalar_tanh_gradient_op<T>> {};

template <typename T>
struct sigmoid_grad : base<T, Eigen::internal::scalar_sigmoid_gradient_op<T>> {
};

template <typename T>
struct inverse_grad : base<T, Eigen::internal::scalar_inverse_gradient_op<T>> {
};

template <typename T>
struct sqrt_grad : base<T, Eigen::internal::scalar_sqrt_gradient_op<T>> {};

template <typename T>
struct rsqrt_grad : base<T, Eigen::internal::scalar_rsqrt_gradient_op<T>> {};

template <typename T>
struct igamma_grad_a : base<T, Eigen::internal::scalar_igamma_der_a_op<T>> {};

}  // end namespace functor

}  // end namespace tensorflow
#endif  // TENSORFLOW_CORE_KERNELS_CWISE_OPS_GRADIENTS_H_
