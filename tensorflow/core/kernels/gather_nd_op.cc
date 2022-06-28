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
class MHTracer_DTPStensorflowPScorePSkernelsPSgather_nd_opDTcc {
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
   MHTracer_DTPStensorflowPScorePSkernelsPSgather_nd_opDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSkernelsPSgather_nd_opDTcc() {
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

// See docs in ../ops/array_ops.cc.
#define EIGEN_USE_THREADS

#include "tensorflow/core/kernels/gather_nd_op.h"
#include "tensorflow/core/framework/bounds_check.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/mem.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {

typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;

template <typename Device, typename T, typename Index>
class GatherNdOp : public OpKernel {
 public:
  explicit GatherNdOp(OpKernelConstruction* c) : OpKernel(c) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSgather_nd_opDTcc mht_0(mht_0_v, 204, "", "./tensorflow/core/kernels/gather_nd_op.cc", "GatherNdOp");

    const DataType dt = DataTypeToEnum<T>::v();
    const DataType index_t = DataTypeToEnum<Index>::v();
    OP_REQUIRES_OK(c, c->MatchSignature({dt, index_t}, {dt}));
  }

  void Compute(OpKernelContext* c) override {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSgather_nd_opDTcc mht_1(mht_1_v, 213, "", "./tensorflow/core/kernels/gather_nd_op.cc", "Compute");

    const Tensor& params = c->input(0);
    const Tensor& indices = c->input(1);

    Tensor out;
    OP_REQUIRES_OK(
        c, functor::DoGatherNd<Device, T, Index>(c, params, indices, &out));
    c->set_output(0, out);
  }
};

#define REGISTER_GATHER_ND_FULL(dev, type, index_type)                 \
  REGISTER_KERNEL_BUILDER(Name("GatherNd")                             \
                              .Device(DEVICE_##dev)                    \
                              .TypeConstraint<type>("Tparams")         \
                              .TypeConstraint<index_type>("Tindices"), \
                          GatherNdOp<dev##Device, type, index_type>)

#define REGISTER_GATHER_ND_ALL_INDICES(dev, type) \
  REGISTER_GATHER_ND_FULL(dev, type, int32);      \
  REGISTER_GATHER_ND_FULL(dev, type, int64_t)

#define REGISTER_GATHER_ND_CPU(type) REGISTER_GATHER_ND_ALL_INDICES(CPU, type)

// TODO(ebrevdo): This is a pure data-movement kernel. It shouldn't be
// instantiated for all different types. Instead, all the types should
// be coalesced. So we should only have int8, int16, int32, int64 support.
// And float is redirected to int32, double is redirected to int64,
// and complex<float> is redirected to int32 with twice the number of
// entries, similarly for complex<double>.
//
// Same for the GPU kernel.
TF_CALL_ALL_TYPES(REGISTER_GATHER_ND_CPU);
TF_CALL_QUANTIZED_TYPES(REGISTER_GATHER_ND_CPU);

#undef REGISTER_GATHER_ND_CPU

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
// Forward declarations of the functor specializations for GPU.
namespace functor {
#define DECLARE_GPU_SPECS_INDEX_NDIM(T, Index, NDIM)          \
  template <>                                                 \
  Index GatherNdSlice<GPUDevice, T, Index, NDIM>::operator()( \
      const GPUDevice& d, const Index slice_size,             \
      typename TTypes<int32>::Scalar Tscratch,                \
      typename TTypes<T, NDIM + 1>::ConstTensor Tparams,      \
      typename TTypes<Index>::ConstMatrix Tindices,           \
      typename TTypes<T>::Matrix Tout);                       \
  extern template struct GatherNdSlice<GPUDevice, T, Index, NDIM>;

#define DECLARE_GPU_SPECS_INDEX(T, Index)    \
  DECLARE_GPU_SPECS_INDEX_NDIM(T, Index, 0); \
  DECLARE_GPU_SPECS_INDEX_NDIM(T, Index, 1); \
  DECLARE_GPU_SPECS_INDEX_NDIM(T, Index, 2); \
  DECLARE_GPU_SPECS_INDEX_NDIM(T, Index, 3); \
  DECLARE_GPU_SPECS_INDEX_NDIM(T, Index, 4); \
  DECLARE_GPU_SPECS_INDEX_NDIM(T, Index, 5); \
  DECLARE_GPU_SPECS_INDEX_NDIM(T, Index, 6); \
  DECLARE_GPU_SPECS_INDEX_NDIM(T, Index, 7);

#define DECLARE_GPU_SPECS(T)         \
  DECLARE_GPU_SPECS_INDEX(T, int32); \
  DECLARE_GPU_SPECS_INDEX(T, int64_t)

TF_CALL_int32(DECLARE_GPU_SPECS);
TF_CALL_int64(DECLARE_GPU_SPECS);
TF_CALL_GPU_NUMBER_TYPES(DECLARE_GPU_SPECS);
TF_CALL_COMPLEX_TYPES(DECLARE_GPU_SPECS);

#undef DECLARE_GPU_SPECS
#undef DECLARE_GPU_SPECS_INDEX
}  // namespace functor

// Registration of the GPU implementations.
#define REGISTER_GATHER_ND_GPU(type) REGISTER_GATHER_ND_ALL_INDICES(GPU, type)

TF_CALL_int32(REGISTER_GATHER_ND_GPU);
TF_CALL_int64(REGISTER_GATHER_ND_GPU);
TF_CALL_GPU_NUMBER_TYPES(REGISTER_GATHER_ND_GPU);
TF_CALL_COMPLEX_TYPES(REGISTER_GATHER_ND_GPU);

#undef REGISTER_GATHER_ND_GPU

#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM

#undef REGISTER_GATHER_ND_ALL_INDICES
#undef REGISTER_GATHER_ND_FULL

}  // namespace tensorflow
