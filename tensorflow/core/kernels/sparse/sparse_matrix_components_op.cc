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
class MHTracer_DTPStensorflowPScorePSkernelsPSsparsePSsparse_matrix_components_opDTcc {
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
   MHTracer_DTPStensorflowPScorePSkernelsPSsparsePSsparse_matrix_components_opDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSkernelsPSsparsePSsparse_matrix_components_opDTcc() {
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

/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
#define EIGEN_USE_GPU
#endif

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/framework/variant_op_registry.h"
#include "tensorflow/core/kernels/dense_update_functor.h"
#include "tensorflow/core/kernels/slice_op.h"
#include "tensorflow/core/kernels/sparse/kernels.h"
#include "tensorflow/core/kernels/sparse/sparse_matrix.h"

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
#include "tensorflow/core/util/cuda_sparse.h"
#include "tensorflow/core/util/gpu_solvers.h"
#endif

namespace tensorflow {

typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;

template <typename Device, typename T>
class CSRSparseMatrixComponentsOp : public OpKernel {
 public:
  explicit CSRSparseMatrixComponentsOp(OpKernelConstruction* c) : OpKernel(c) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSsparsePSsparse_matrix_components_opDTcc mht_0(mht_0_v, 214, "", "./tensorflow/core/kernels/sparse/sparse_matrix_components_op.cc", "CSRSparseMatrixComponentsOp");
}

  void Compute(OpKernelContext* c) final {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSsparsePSsparse_matrix_components_opDTcc mht_1(mht_1_v, 219, "", "./tensorflow/core/kernels/sparse/sparse_matrix_components_op.cc", "Compute");

    const CSRSparseMatrix* csr_sparse_matrix;
    OP_REQUIRES_OK(c, ExtractVariantFromInput(c, 0, &csr_sparse_matrix));

    const Tensor& index_t = c->input(1);
    OP_REQUIRES(c, DataTypeToEnum<T>::value == csr_sparse_matrix->dtype(),
                errors::InvalidArgument(
                    "dtype of input is not equal to 'type': ",
                    DataTypeString(csr_sparse_matrix->dtype()), " vs. ",
                    DataTypeString(DataTypeToEnum<T>::value)));
    OP_REQUIRES(c, index_t.dims() == 0,
                errors::InvalidArgument("index should be a scalar, but saw: ",
                                        index_t.DebugString()));
    int32_t index = index_t.scalar<int32>()();
    OP_REQUIRES(c, index >= 0 && index < csr_sparse_matrix->batch_size(),
                errors::InvalidArgument("index (", index, ") not in [0, ",
                                        csr_sparse_matrix->batch_size(), ")"));

    if (csr_sparse_matrix->dims() == 2) {
      c->set_output(0, csr_sparse_matrix->row_pointers());
      c->set_output(1, csr_sparse_matrix->col_indices());
      c->set_output(2, csr_sparse_matrix->values());
    } else {
      auto batch_ptrs = csr_sparse_matrix->batch_pointers().vec<int32>();
      auto dense_shape = csr_sparse_matrix->dense_shape().vec<int64_t>();
      int64_t rows = dense_shape(1);
      int nnz = batch_ptrs(index + 1) - batch_ptrs(index);
      Tensor* row_ptrs_t;
      Tensor* col_inds_t;
      Tensor* values_t;
      OP_REQUIRES_OK(
          c, c->allocate_output(0, TensorShape({rows + 1}), &row_ptrs_t));
      OP_REQUIRES_OK(c, c->allocate_output(1, TensorShape({nnz}), &col_inds_t));
      OP_REQUIRES_OK(c, c->allocate_output(2, TensorShape({nnz}), &values_t));
      auto row_ptrs = row_ptrs_t->vec<int32>();
      auto col_inds = col_inds_t->vec<int32>();
      auto values = values_t->vec<T>();

      functor::Slice<Device, int32, 1> slice_int;
      functor::Slice<Device, T, 1> slice_t;
      typedef Eigen::DSizes<Eigen::DenseIndex, 1> EVec;
      const Device& d = c->eigen_device<Device>();
      slice_int(d,
                /*output*/ row_ptrs,
                /*input*/ csr_sparse_matrix->row_pointers().vec<int32>(),
                /*slice_indices*/
                EVec{static_cast<Eigen::DenseIndex>(index * (rows + 1))},
                /*slice_sizes*/ EVec{static_cast<Eigen::DenseIndex>(rows + 1)});
      slice_int(d,
                /*output*/ col_inds,
                /*input*/ csr_sparse_matrix->col_indices().vec<int32>(),
                /*slice_indices*/ EVec{batch_ptrs(index)},
                /*slice_sizes*/ EVec{nnz});
      slice_t(d,
              /*output*/ values, /*input*/ csr_sparse_matrix->values().vec<T>(),
              /*slice_indices*/ EVec{batch_ptrs(index)},
              /*slice_sizes*/ EVec{nnz});
    }
  }
};

#define REGISTER(DEV, T)                                    \
  REGISTER_KERNEL_BUILDER(Name("CSRSparseMatrixComponents") \
                              .Device(DEVICE_##DEV)         \
                              .TypeConstraint<T>("type")    \
                              .HostMemory("index"),         \
                          CSRSparseMatrixComponentsOp<DEV##Device, T>);

REGISTER(CPU, float)
REGISTER(CPU, double)
REGISTER(CPU, complex64)
REGISTER(CPU, complex128)

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM

REGISTER(GPU, float)
REGISTER(GPU, double)
REGISTER(GPU, complex64)
REGISTER(GPU, complex128)

#undef REGISTER

namespace functor {
// TODO(ebrevdo): This should move to a slice_functor.cc
#define DECLARE_GPU_SPEC(T)                                     \
  template <>                                                   \
  void Slice<GPUDevice, T, 1>::operator()(                      \
      const GPUDevice& d, typename TTypes<T, 1>::Tensor output, \
      typename TTypes<T, 1>::ConstTensor input,                 \
      const Eigen::DSizes<Eigen::DenseIndex, 1>& indices,       \
      const Eigen::DSizes<Eigen::DenseIndex, 1>& sizes);        \
  extern template struct Slice<GPUDevice, T, 1>;

DECLARE_GPU_SPEC(int32);
DECLARE_GPU_SPEC(float);
DECLARE_GPU_SPEC(double);
DECLARE_GPU_SPEC(complex64);
DECLARE_GPU_SPEC(complex128);

#undef DECLARE_GPU_SPEC
}  // namespace functor

#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM

}  // namespace tensorflow
