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

#ifndef TENSORFLOW_CORE_KERNELS_REDUCTION_OPS_H_
#define TENSORFLOW_CORE_KERNELS_REDUCTION_OPS_H_
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
class MHTracer_DTPStensorflowPScorePSkernelsPSreduction_opsDTh {
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
   MHTracer_DTPStensorflowPScorePSkernelsPSreduction_opsDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSkernelsPSreduction_opsDTh() {
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


// Functor definitions for Reduction ops, must be compilable by nvcc.

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor_types.h"

namespace tensorflow {
namespace functor {

template <typename Reducer>
struct ReducerTraits {
  enum { IsScalarIdentity = true };
};

// Dummy class used for template specialization for mean reduction, which is
// accomplished by SumReducer and on-the-fly division by the reduction factor.
template <typename Scalar>
struct MeanReducer {
  Scalar initialize() const {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSreduction_opsDTh mht_0(mht_0_v, 206, "", "./tensorflow/core/kernels/reduction_ops.h", "initialize");
 return Scalar(0); }
};

// Dummy class used for template specialization for l2-norm reduction.
template <typename Scalar>
struct EuclideanNormReducer {
  Scalar initialize() const {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSreduction_opsDTh mht_1(mht_1_v, 215, "", "./tensorflow/core/kernels/reduction_ops.h", "initialize");
 return Scalar(0); }
};

template <typename Scalar>
struct ReducerTraits<EuclideanNormReducer<Scalar>> {
  enum { IsScalarIdentity = false };
};

template <typename Device, typename OUT_T, typename IN_T,
          typename ReductionAxes, typename Reducer>
struct ReduceEigenImpl {
  void operator()(const Device& d, OUT_T out, IN_T in,
                  const ReductionAxes& reduction_axes, const Reducer& reducer) {
    out.device(d) = in.reduce(reduction_axes, reducer);
  }
};

// Specialization for BF16 Reducer to fix accuracy.
// TODO: All BF16 reducers should have specializations to fix accuracy.
#define CASTING_SPECIALIZATION(Reducer, ScalarType, IntermediateType)        \
  template <typename Device, typename OUT_T, typename IN_T,                  \
            typename ReductionAxes>                                          \
  struct ReduceEigenImpl<Device, OUT_T, IN_T, ReductionAxes,                 \
                         Reducer<ScalarType>> {                              \
    void operator()(const Device& d, OUT_T out, IN_T in,                     \
                    const ReductionAxes& reduction_axes,                     \
                    const Reducer<ScalarType>& reducer) {                    \
      static_assert(std::is_same<ScalarType, typename OUT_T::Scalar>::value, \
                    "");                                                     \
      Reducer<IntermediateType> intermediate_reducer;                        \
      auto in_as_intermediate = in.template cast<IntermediateType>();        \
      out.device(d) =                                                        \
          in_as_intermediate.reduce(reduction_axes, intermediate_reducer)    \
              .template cast<ScalarType>();                                  \
    }                                                                        \
  };

CASTING_SPECIALIZATION(Eigen::internal::SumReducer, bfloat16, float);
#undef CASTING_SPECIALIZATION

template <typename Device, typename OUT_T, typename IN_T,
          typename ReductionAxes, typename Scalar>
struct ReduceEigenImpl<Device, OUT_T, IN_T, ReductionAxes,
                       functor::MeanReducer<Scalar>> {
  void operator()(const Device& d, OUT_T out, IN_T in,
                  const ReductionAxes& reduction_axes,
                  const functor::MeanReducer<Scalar>& reducer) {
    static_assert(std::is_same<Scalar, typename OUT_T::Scalar>::value, "");
    Eigen::internal::SumReducer<Scalar> sum_reducer;
    out.device(d) = in.reduce(reduction_axes, sum_reducer) /
                    static_cast<Scalar>(in.size() / out.size());
  }
};

// Specialization for which we do the reduction in IntermediateType to
// avoid integer overflow.
#define CASTING_SPECIALIZATION(ScalarType, IntermediateType)                  \
  template <typename Device, typename OUT_T, typename IN_T,                   \
            typename ReductionAxes>                                           \
  struct ReduceEigenImpl<Device, OUT_T, IN_T, ReductionAxes,                  \
                         functor::MeanReducer<ScalarType>> {                  \
    void operator()(const Device& d, OUT_T out, IN_T in,                      \
                    const ReductionAxes& reduction_axes,                      \
                    const functor::MeanReducer<ScalarType>& reducer) {        \
      static_assert(std::is_same<ScalarType, typename OUT_T::Scalar>::value,  \
                    "");                                                      \
      Eigen::internal::SumReducer<IntermediateType> sum_reducer;              \
      out.device(d) = (in.template cast<IntermediateType>().reduce(           \
                           reduction_axes, sum_reducer) /                     \
                       static_cast<IntermediateType>(in.size() / out.size())) \
                          .template cast<ScalarType>();                       \
    }                                                                         \
  }

CASTING_SPECIALIZATION(uint8, uint64);
CASTING_SPECIALIZATION(uint16, uint64);
CASTING_SPECIALIZATION(uint32, uint64);
CASTING_SPECIALIZATION(int8, int64_t);
CASTING_SPECIALIZATION(int16, int64_t);
CASTING_SPECIALIZATION(int32, int64_t);
#undef CASTING_SPECIALIZATION

// TODO(rmlarsen): Refactor this such that taking the sqrt can be optional
// controlled by an attribute.
template <typename Device, typename OUT_T, typename IN_T,
          typename ReductionAxes, typename Scalar>
struct ReduceEigenImpl<Device, OUT_T, IN_T, ReductionAxes,
                       functor::EuclideanNormReducer<Scalar>> {
  void operator()(const Device& d, OUT_T out, IN_T in,
                  const ReductionAxes& reduction_axes,
                  const functor::EuclideanNormReducer<Scalar>& reducer) {
    static_assert(std::is_same<Scalar, typename OUT_T::Scalar>::value, "");
    Eigen::internal::SumReducer<Scalar> sum_reducer;
    out.device(d) =
        (in * in.conjugate()).reduce(reduction_axes, sum_reducer).sqrt();
  }
};

template <typename Device, typename OUT_T, typename IN_T,
          typename ReductionAxes>
struct ReduceEigenImpl<Device, OUT_T, IN_T, ReductionAxes,
                       functor::EuclideanNormReducer<bfloat16>> {
  void operator()(const Device& d, OUT_T out, IN_T in,
                  const ReductionAxes& reduction_axes,
                  const functor::EuclideanNormReducer<bfloat16>& reducer) {
    static_assert(std::is_same<bfloat16, typename OUT_T::Scalar>::value, "");
    Eigen::internal::SumReducer<float> sum_reducer;
    auto in_as_float = in.template cast<float>();
    out.device(d) = (in_as_float * in_as_float.conjugate())
                        .reduce(reduction_axes, sum_reducer)
                        .sqrt()
                        .template cast<bfloat16>();
  }
};

// For most reducers, the identity is Reducer::initialize()
template <typename Reducer>
struct Identity {
  static auto identity(const Reducer& reducer)
      -> decltype(reducer.initialize()) {
    return reducer.initialize();
  }
};

// MeanReducer is a special case, since it doesn't technically have an identity.
// Thus, ideally we'd return nan.  However, mean is instantiated for integer
// types as well, so we do the nan override only for floating point types.
#define FIX_MEAN_IDENTITY(T)                            \
  template <>                                           \
  struct Identity<functor::MeanReducer<T>> {            \
    static T identity(const functor::MeanReducer<T>&) { \
      return Eigen::NumTraits<T>::quiet_NaN();          \
    }                                                   \
  };
FIX_MEAN_IDENTITY(Eigen::half)
FIX_MEAN_IDENTITY(float)
FIX_MEAN_IDENTITY(double)
#undef FIX_MEAN_IDENTITY

template <typename Device, typename OUT_T, typename Reducer>
void FillIdentityEigenImpl(const Device& d, OUT_T out, const Reducer& reducer) {
  MaybeWith32BitIndexing<Device>(
      [&](auto out32) {
        out32.device(d) = out32.constant(Identity<Reducer>::identity(reducer));
      },
      out);
}

template <typename Device, typename Reducer>
struct ReduceFunctor {
  template <typename OUT_T, typename IN_T, typename ReductionAxes>
  static void Reduce(OpKernelContext* ctx, OUT_T out, IN_T in,
                     const ReductionAxes& reduction_axes,
                     const Reducer& reducer);

  template <typename OUT_T>
  static void FillIdentity(const Device& d, OUT_T out, const Reducer& reducer);
};

}  // namespace functor
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_KERNELS_REDUCTION_OPS_H_
