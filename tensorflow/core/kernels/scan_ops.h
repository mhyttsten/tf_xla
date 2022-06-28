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

#ifndef TENSORFLOW_CORE_KERNELS_SCAN_OPS_H_
#define TENSORFLOW_CORE_KERNELS_SCAN_OPS_H_
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
class MHTracer_DTPStensorflowPScorePSkernelsPSscan_opsDTh {
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
   MHTracer_DTPStensorflowPScorePSkernelsPSscan_opsDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSkernelsPSscan_opsDTh() {
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
#include "tensorflow/core/framework/tensor_types.h"

namespace tensorflow {
namespace functor {

typedef Eigen::Index Index;

// TODO(b/154339590): Needs to be vectorized.
template <typename Device, typename Reducer, typename T>
struct Scan {
  void operator()(const Device& d, typename TTypes<T, 3>::ConstTensor in,
                  typename TTypes<T, 3>::Tensor out, const Reducer& reducer,
                  const bool reverse, const bool exclusive) {
    // Perform the reverse ops directly with Eigen, which avoids copying the
    // tensor twice compared to using individual ops.
    Eigen::array<bool, 3> dims;
    dims[0] = false;
    dims[1] = reverse;
    dims[2] = false;
    MaybeWith32BitIndexing<Device>(
        [&](auto in32, auto out32) {
          out32.device(d) =
              in32.reverse(dims).scan(1, reducer, exclusive).reverse(dims);
        },
        in, out);
  }
};

template <typename T>
struct LogSumExp {
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE T operator()(const T& a,
                                                     const T& b) const {
    auto mi = Eigen::internal::scalar_min_op<T>()(a, b);
    auto ma = Eigen::internal::scalar_max_op<T>()(a, b);

    auto sub = Eigen::internal::scalar_difference_op<T>();
    auto add = Eigen::internal::scalar_sum_op<T>();
    auto exp = Eigen::internal::scalar_exp_op<T>();
    auto log1p = Eigen::internal::scalar_log1p_op<T>();
    auto cmp_lt =
        Eigen::internal::scalar_cmp_op<T, T, Eigen::internal::cmp_LT>();

    auto logsumexp = add(log1p(exp(sub(mi, ma))), ma);
    return cmp_lt(ma, Eigen::NumTraits<T>::lowest()) ? ma : logsumexp;
  }
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE T packetOp(const T& a,
                                                   const T& b) const {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSscan_opsDTh mht_0(mht_0_v, 235, "", "./tensorflow/core/kernels/scan_ops.h", "packetOp");

    auto mi = Eigen::internal::pmin(a, b);
    auto ma = Eigen::internal::pmax(a, b);
    using Eigen::internal::padd;
    using Eigen::internal::pcmp_lt;
    using Eigen::internal::pexp;
    using Eigen::internal::plog1p;
    using Eigen::internal::pset1;
    using Eigen::internal::psub;

    auto logsumexp = padd(plog1p(pexp(psub(mi, ma))), ma);
    return pselect(pcmp_lt(ma, pset1(Eigen::NumTraits<T>::lowest())), ma,
                   logsumexp);
  }
};

template <typename T>
struct LogSumExpReducer {
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE void reduce(const T t, T* accum) const {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSscan_opsDTh mht_1(mht_1_v, 256, "", "./tensorflow/core/kernels/scan_ops.h", "reduce");

    LogSumExp<T> logsumexp;
    *accum = logsumexp(*accum, t);
  }

  template <typename Packet>
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE void reducePacket(const Packet& p,
                                                          Packet* accum) const {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSscan_opsDTh mht_2(mht_2_v, 266, "", "./tensorflow/core/kernels/scan_ops.h", "reducePacket");

    LogSumExp<T> logsumexp;
    *accum = logsumexp.packetOp(*accum, p);
  }

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE T initialize() const {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSscan_opsDTh mht_3(mht_3_v, 274, "", "./tensorflow/core/kernels/scan_ops.h", "initialize");

    return -Eigen::NumTraits<T>::infinity();
  }

  template <typename Packet>
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Packet initializePacket() const {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSscan_opsDTh mht_4(mht_4_v, 282, "", "./tensorflow/core/kernels/scan_ops.h", "initializePacket");

    return Eigen::internal::pset1(initialize());
  }

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE T finalize(const T accum) const {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSscan_opsDTh mht_5(mht_5_v, 289, "", "./tensorflow/core/kernels/scan_ops.h", "finalize");

    return accum;
  }

  template <typename Packet>
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Packet
  finalizePacket(const Packet& vaccum) const {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSscan_opsDTh mht_6(mht_6_v, 298, "", "./tensorflow/core/kernels/scan_ops.h", "finalizePacket");

    return vaccum;
  }

  template <typename Packet>
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE T
  finalizeBoth(const T saccum, const Packet& vaccum) const {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSscan_opsDTh mht_7(mht_7_v, 307, "", "./tensorflow/core/kernels/scan_ops.h", "finalizeBoth");

    auto max_reducer = Eigen::internal::MaxReducer<T, Eigen::PropagateNaN>();
    auto sum_reducer = Eigen::internal::SumReducer<T>();
    auto exp = Eigen::internal::scalar_exp_op<T>();
    auto cmp_lt =
        Eigen::internal::scalar_cmp_op<T, T, Eigen::internal::cmp_LT>();
    auto log = Eigen::internal::scalar_log_op<T>();
    auto add = Eigen::internal::scalar_sum_op<T>();

    using Eigen::internal::pexp;
    using Eigen::internal::psub;

    // `ma = max(x1, ..., xn)`
    // If the max of all of the `xi` is `-infinity` then the result is
    // -infinity. If the max is larger than `-infinity` then it's safe to use
    // for normalization even if the other elements are `-infinity`.
    //
    // `logsumexp(x1, ..., xn) = ma + log (exp(x1 - ma) + ... + exp(xn - ma))`
    auto ma = max_reducer.finalizeBoth(saccum, vaccum);
    auto logsumexp = add(log(sum_reducer.finalizeBoth(
                             exp(saccum - ma), pexp(psub(vaccum, pset1(ma))))),
                         ma);
    return cmp_lt(ma, Eigen::NumTraits<T>::lowest()) ? initialize() : logsumexp;
  }
};

}  // namespace functor
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_KERNELS_SCAN_OPS_H_
