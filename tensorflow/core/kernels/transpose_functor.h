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

#ifndef TENSORFLOW_CORE_KERNELS_TRANSPOSE_FUNCTOR_H_
#define TENSORFLOW_CORE_KERNELS_TRANSPOSE_FUNCTOR_H_
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
class MHTracer_DTPStensorflowPScorePSkernelsPStranspose_functorDTh {
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
   MHTracer_DTPStensorflowPScorePSkernelsPStranspose_functorDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSkernelsPStranspose_functorDTh() {
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


#include <numeric>
#include <string>
#include <vector>

#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/platform/logging.h"

namespace tensorflow {
// Transpose tensor 'in' into tensor 'out' according to dimension
// permutation 'perm'.
//
// REQUIRES: in.dtype() == out->dtype()
// REQUIRES: in.dims() == out->dims()
// REQUIRES: in.dims() == perm.size()
// REQUIRES: in.dim_size(perm[i]) == out->dim_size(i)
template <typename Device>
Status DoTranspose(const Device& device, const Tensor& in,
                   const gtl::ArraySlice<int32> perm, Tensor* out);

// Conjugate and transpose tensor 'in' into tensor 'out' according to dimension
// permutation 'perm'.
//
// REQUIRES: in.dtype() == out->dtype()
// REQUIRES: in.dims() == out->dims()
// REQUIRES: in.dims() == perm.size()
// REQUIRES: in.dim_size(perm[i]) == out->dim_size(i)
template <typename Device>
Status DoConjugateTranspose(const Device& device, const Tensor& in,
                            const gtl::ArraySlice<int32> perm, Tensor* out);

// Convenience versions of DoTranspose that only swap the last (inner) two
// dimensions.
template <typename Device>
Status DoMatrixTranspose(const Device& device, const Tensor& in, Tensor* out);

// Convenience versions of DoConjugateTranspose that only swap the last (inner)
// two dimensions.
template <typename Device>
Status DoConjugateMatrixTranspose(const Device& device, const Tensor& in,
                                  Tensor* out);

// Primary device specific functor to be specialized for each device and type.
template <typename Device, typename T, bool conjugate = false>
struct Transpose {
  static void run(const Device& d, const Tensor& in,
                  const gtl::ArraySlice<int32> perm, Tensor* out);
};

// Implementation details.
namespace internal {

typedef gtl::InlinedVector<int64_t, 8> TransposeDimsVec;
typedef gtl::InlinedVector<int32, 8> TransposePermsVec;

// Helper function that takes a tensor shape, a permutation, combines the
// neighboring shapes if their indices in the permutation are consecutive.
// The function outputs the combined shape and new permutation.
// Example: Tensor shape {2, 3, 4, 5, 120} and permutation {0, 4, 1, 2, 3} will
// produce new shape {2, 60, 120} and new permutation {0, 2, 1}.
inline void ReduceTransposeDimensions(const TensorShape& shape,
                                      gtl::ArraySlice<int32> perm,
                                      TransposePermsVec* new_perm,
                                      TransposeDimsVec* new_dims) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSkernelsPStranspose_functorDTh mht_0(mht_0_v, 251, "", "./tensorflow/core/kernels/transpose_functor.h", "ReduceTransposeDimensions");

  CHECK_EQ(shape.dims(), perm.size());
  if (shape.dims() == 1) {
    // If input dimension is already 1, no need to reduce dimension.
    new_perm->resize(1);
    (*new_perm)[0] = perm[0];
    (*new_dims)[0] = shape.dim_size(0);
    return;
  }
  TransposePermsVec new_dim_position(shape.dims(), -1);
  TransposeDimsVec combined_dims(shape.dims(), 0);
  int cur_head = perm[0];
  new_dim_position[cur_head] = 0;
  combined_dims[0] = shape.dim_size(cur_head);
  int dim_idx = 0;
  for (int perm_idx = 1; perm_idx < shape.dims(); ++perm_idx) {
    // If two indices in permutation are consecutive numbers, combine their
    // dimensions.
    if (cur_head + 1 == perm[perm_idx]) {
      cur_head = perm[perm_idx];
      combined_dims[dim_idx] *= shape.dim_size(cur_head);
    } else {
      // Else start a new dimension.
      cur_head = perm[perm_idx];
      dim_idx++;
      new_dim_position[cur_head] = dim_idx;
      combined_dims[dim_idx] = shape.dim_size(cur_head);
    }
  }
  // Compact the new permutations and dimension sizes.
  new_perm->resize(dim_idx + 1);
  new_dims->resize(dim_idx + 1);
  dim_idx = 0;
  for (int i = 0; i < new_dim_position.size(); ++i) {
    if (new_dim_position[i] >= 0) {
      int new_perm_idx = new_dim_position[i];
      (*new_perm)[dim_idx] = new_perm_idx;
      (*new_dims)[dim_idx] = combined_dims[new_perm_idx];
      dim_idx++;
    }
  }
}

// If all non-singleton dimensions remain in ascending order, the shuffled
// singletons can be transposed by a reshape, saving a memory allocation & copy.
// |permutation| must be a permutation of {0, .., input_shape.dims() - 1}.
// That is, for all i, 0 <= perm[i] < input_shape.dims().
// In practice, this is checked in TransposeOp::Compute prior to calling this
// function, and the function sits here to facilitate unit testing.
inline bool NonSingletonDimensionsAlign(const TensorShape& input_shape,
                                        const std::vector<int32>& permutation) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSkernelsPStranspose_functorDTh mht_1(mht_1_v, 304, "", "./tensorflow/core/kernels/transpose_functor.h", "NonSingletonDimensionsAlign");

  int last_nonsingleton_perm_dim = -1;
  for (int perm_dim : permutation) {
    if (input_shape.dim_size(perm_dim) == 1) {
      continue;
    }
    if (perm_dim < last_nonsingleton_perm_dim) {
      return false;
    }
    last_nonsingleton_perm_dim = perm_dim;
  }
  return true;
}

// Uses Eigen to transpose.
template <typename Device, typename T, int NDIMS>
void TransposeUsingEigen(const Device& d, const Tensor& in,
                         const gtl::ArraySlice<int32> perm, bool conjugate,
                         Tensor* out) {
  Eigen::array<int, NDIMS> p;
  for (int i = 0; i < NDIMS; ++i) p[i] = perm[i];
  auto x = typename TTypes<T, NDIMS>::ConstTensor(
      reinterpret_cast<const T*>(in.tensor_data().data()),
      in.shape().AsEigenDSizes<NDIMS>());
  auto y = typename TTypes<T, NDIMS>::Tensor(
      reinterpret_cast<T*>(const_cast<char*>(out->tensor_data().data())),
      out->shape().AsEigenDSizes<NDIMS>());
  if (conjugate) {
    y.device(d) = x.conjugate().shuffle(p);
  } else {
    y.device(d) = x.shuffle(p);
  }
}

template <typename Device>
Status DoTransposeImpl(const Device& d, const Tensor& in,
                       const gtl::ArraySlice<int32> perm, bool conjugate,
                       Tensor* out) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSkernelsPStranspose_functorDTh mht_2(mht_2_v, 344, "", "./tensorflow/core/kernels/transpose_functor.h", "DoTransposeImpl");

  CHECK_EQ(in.dims(), out->dims());
  CHECK_EQ(in.dims(), perm.size());
  CHECK_EQ(in.dtype(), out->dtype());
  switch (in.dtype()) {
    case DT_BOOL:
    case DT_INT8:
    case DT_QINT8:
    case DT_QUINT8:
    case DT_UINT8:
      Transpose<Device, uint8>::run(d, in, perm, out);
      break;

    case DT_BFLOAT16:
    case DT_HALF:
    case DT_INT16:
    case DT_QINT16:
    case DT_QUINT16:
    case DT_UINT16:
      Transpose<Device, uint16>::run(d, in, perm, out);
      break;

    case DT_FLOAT:
    case DT_INT32:
    case DT_QINT32:
    case DT_UINT32:
      Transpose<Device, uint32>::run(d, in, perm, out);
      break;

    case DT_DOUBLE:
    case DT_INT64:
    case DT_UINT64:
      Transpose<Device, uint64>::run(d, in, perm, out);
      break;

    case DT_COMPLEX64:
      if (conjugate) {
#if defined(__ANDROID__) and !defined(__clang__)
        // Workaround for GCC compiler bug in Android toolchain.
        return errors::Unimplemented(
            "Conjugate transpose of complex64 not supported for GCC on "
            "Android.");
#else
        Transpose<Device, complex64, /*conjugate=*/true>::run(d, in, perm, out);
#endif
      } else {
        Transpose<Device, uint64>::run(d, in, perm, out);
      }
      break;

    case DT_COMPLEX128:
      if (conjugate) {
        Transpose<Device, complex128, /*conjugate=*/true>::run(d, in, perm,
                                                               out);
      } else {
        Transpose<Device, complex128, /*conjugate=*/false>::run(d, in, perm,
                                                                out);
      }
      break;

    case DT_STRING:
      Transpose<Device, tstring>::run(d, in, perm, out);
      break;

    default:
      return errors::Unimplemented("Unsupported dtype on CPU: ", in.dtype());
  }
  return Status::OK();
}

template <typename Device>
inline Status DoMatrixTransposeImpl(const Device& device, const Tensor& in,
                                    bool conjugate, Tensor* out) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSkernelsPStranspose_functorDTh mht_3(mht_3_v, 419, "", "./tensorflow/core/kernels/transpose_functor.h", "DoMatrixTransposeImpl");

  const int ndims = in.dims();
  if (ndims == 0) return Status::OK();
  TransposePermsVec perm(ndims);
  std::iota(perm.begin(), perm.end(), 0);
  std::swap(perm[ndims - 2], perm[ndims - 1]);
  return DoTransposeImpl(device, in, perm, conjugate, out);
}

}  // namespace internal
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_KERNELS_TRANSPOSE_FUNCTOR_H_
