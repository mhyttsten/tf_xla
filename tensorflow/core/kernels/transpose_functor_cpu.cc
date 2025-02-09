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
class MHTracer_DTPStensorflowPScorePSkernelsPStranspose_functor_cpuDTcc {
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
   MHTracer_DTPStensorflowPScorePSkernelsPStranspose_functor_cpuDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSkernelsPStranspose_functor_cpuDTcc() {
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

#define EIGEN_USE_THREADS

#include <complex>

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/attr_value.pb.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/kernels/ops_util.h"
#include "tensorflow/core/kernels/transpose_functor.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/gtl/array_slice.h"
#include "tensorflow/core/lib/gtl/inlined_vector.h"

typedef Eigen::ThreadPoolDevice CPUDevice;

namespace tensorflow {
namespace {

template <typename T, bool conjugate>
void TransposeSimple(const CPUDevice& device, const Tensor& in,
                     const gtl::ArraySlice<int32> perm, Tensor* out) {
  const int ndims = in.dims();
  gtl::InlinedVector<int64_t, 8> in_strides =
      ComputeStride<int64_t>(in.shape());
  gtl::InlinedVector<int64_t, 8> out_strides =
      ComputeStride<int64_t>(out->shape());
  const T* p = reinterpret_cast<const T*>(in.tensor_data().data());
  T* q = reinterpret_cast<T*>(const_cast<char*>((out->tensor_data().data())));
  auto transpose_fn = [=, &in_strides, &out_strides, &perm](int64_t begin,
                                                            int64_t end) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSkernelsPStranspose_functor_cpuDTcc mht_0(mht_0_v, 214, "", "./tensorflow/core/kernels/transpose_functor_cpu.cc", "lambda");

    for (int64_t o_idx = begin; o_idx < end; ++o_idx) {
      int64_t i_idx = 0;
      int64_t t = o_idx;
      for (int i = 0; i < ndims; ++i) {
        const int64_t ratio = t / out_strides[i];
        t -= ratio * out_strides[i];
        i_idx += ratio * in_strides[perm[i]];
      }
      if (conjugate) {
        q[o_idx] = Eigen::numext::conj(p[i_idx]);
      } else {
        q[o_idx] = p[i_idx];
      }
    }
  };
  double cycles_per_element =
      (conjugate ? 1 : 0) +
      ndims * (Eigen::TensorOpCost::DivCost<int64_t>() +
               2 * Eigen::TensorOpCost::MulCost<int64_t>() +
               2 * Eigen::TensorOpCost::AddCost<int64_t>());
  Eigen::TensorOpCost cost(/*bytes_loaded=*/sizeof(T),
                           /*bytes_stored=*/sizeof(T), cycles_per_element);
  device.parallelFor(in.NumElements(), cost, std::move(transpose_fn));
}

}  // namespace

template <typename T, bool conjugate>
struct Transpose<CPUDevice, T, conjugate> {
  static void run(const CPUDevice& d, const Tensor& in,
                  const gtl::ArraySlice<int32> perm, Tensor* out) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSkernelsPStranspose_functor_cpuDTcc mht_1(mht_1_v, 248, "", "./tensorflow/core/kernels/transpose_functor_cpu.cc", "run");

    switch (in.dims()) {
      case 2:
        internal::TransposeUsingEigen<CPUDevice, T, 2>(d, in, perm, conjugate,
                                                       out);
        break;
      case 3:
        internal::TransposeUsingEigen<CPUDevice, T, 3>(d, in, perm, conjugate,
                                                       out);
        break;
      case 4:
        internal::TransposeUsingEigen<CPUDevice, T, 4>(d, in, perm, conjugate,
                                                       out);
        break;
      case 5:
        internal::TransposeUsingEigen<CPUDevice, T, 5>(d, in, perm, conjugate,
                                                       out);
        break;
      case 6:
        internal::TransposeUsingEigen<CPUDevice, T, 6>(d, in, perm, conjugate,
                                                       out);
        break;
      case 7:
        internal::TransposeUsingEigen<CPUDevice, T, 7>(d, in, perm, conjugate,
                                                       out);
        break;
      case 8:
        internal::TransposeUsingEigen<CPUDevice, T, 8>(d, in, perm, conjugate,
                                                       out);
        break;
      default:
        TransposeSimple<T, conjugate>(d, in, perm, out);
        break;
    }
  }
};

#define INSTANTIATE(DEVICE)                                                 \
  template <>                                                               \
  Status DoTranspose(const DEVICE& device, const Tensor& in,                \
                     const gtl::ArraySlice<int32> perm, Tensor* out) {      \
    return internal::DoTransposeImpl(device, in, perm, /*conjugate=*/false, \
                                     out);                                  \
  }                                                                         \
  template <>                                                               \
  Status DoConjugateTranspose(const DEVICE& device, const Tensor& in,       \
                              const gtl::ArraySlice<int32> perm,            \
                              Tensor* out) {                                \
    return internal::DoTransposeImpl(device, in, perm, /*conjugate=*/true,  \
                                     out);                                  \
  }                                                                         \
  template <>                                                               \
  Status DoMatrixTranspose(const DEVICE& device, const Tensor& in,          \
                           Tensor* out) {                                   \
    return internal::DoMatrixTransposeImpl(device, in, /*conjugate=*/false, \
                                           out);                            \
  }                                                                         \
  template <>                                                               \
  Status DoConjugateMatrixTranspose(const DEVICE& device, const Tensor& in, \
                                    Tensor* out) {                          \
    return internal::DoMatrixTransposeImpl(device, in, /*conjugate=*/true,  \
                                           out);                            \
  }

INSTANTIATE(CPUDevice)


}  // namespace tensorflow
