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
class MHTracer_DTPStensorflowPScorePSkernelsPSdense_update_opsDTcc {
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
   MHTracer_DTPStensorflowPScorePSkernelsPSdense_update_opsDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSkernelsPSdense_update_opsDTcc() {
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

#define EIGEN_USE_THREADS

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
#define EIGEN_USE_GPU
#endif

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/kernels/assign_op.h"
#include "tensorflow/core/kernels/dense_update_functor.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {

template <typename Device, typename T>
class AssignOpT : public AssignOp {
 public:
  using AssignOp::AssignOp;

  void Copy(OpKernelContext* context, Tensor* lhs, const Tensor& rhs) override {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdense_update_opsDTcc mht_0(mht_0_v, 207, "", "./tensorflow/core/kernels/dense_update_ops.cc", "Copy");

    functor::DenseUpdate<Device, T, ASSIGN> copy;
    copy(context->eigen_device<Device>(), lhs->flat<T>(), rhs.flat<T>());
  }
};

// TODO(jeff): Get rid of use_exclusive_lock_ option
template <typename Device, typename T, DenseUpdateType OP>
class DenseUpdateOp : public OpKernel {
 public:
  explicit DenseUpdateOp(OpKernelConstruction* context) : OpKernel(context) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdense_update_opsDTcc mht_1(mht_1_v, 220, "", "./tensorflow/core/kernels/dense_update_ops.cc", "DenseUpdateOp");

    OP_REQUIRES_OK(context,
                   context->GetAttr("use_locking", &use_exclusive_lock_));
    const DataType dt = DataTypeToEnum<T>::v();
    OP_REQUIRES_OK(context, context->MatchSignature({MakeRefType(dt), dt},
                                                    {MakeRefType(dt)}));
  }

  void Compute(OpKernelContext* context) override {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdense_update_opsDTcc mht_2(mht_2_v, 231, "", "./tensorflow/core/kernels/dense_update_ops.cc", "Compute");

    // We always return the input ref.
    context->forward_ref_input_to_ref_output(0, 0);

    if (use_exclusive_lock_) {
      mutex_lock l(*context->input_ref_mutex(0));
      DoUpdate(context);
    } else {
      DoUpdate(context);
    }
  }

 private:
  void DoUpdate(OpKernelContext* context) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdense_update_opsDTcc mht_3(mht_3_v, 247, "", "./tensorflow/core/kernels/dense_update_ops.cc", "DoUpdate");

    Tensor Tparams = context->mutable_input(0, use_exclusive_lock_);
    const Tensor& Tupdate = context->input(1);
    OP_REQUIRES(context, Tparams.IsInitialized(),
                errors::FailedPrecondition("Attempting to use uninitialized "
                                           "parameters: ",
                                           requested_input(0)));
    OP_REQUIRES(
        context, Tparams.IsSameSize(Tupdate),
        errors::InvalidArgument("Parameters and update must be the same size"));

    functor::DenseUpdate<Device, T, OP> update_functor;
    update_functor(context->template eigen_device<Device>(), Tparams.flat<T>(),
                   Tupdate.flat<T>());
  }

  bool use_exclusive_lock_;
};

typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;

#define REGISTER_KERNELS(type)                                     \
  REGISTER_KERNEL_BUILDER(                                         \
      Name("Assign").Device(DEVICE_CPU).TypeConstraint<type>("T"), \
      AssignOpT<CPUDevice, type>);

TF_CALL_ALL_TYPES(REGISTER_KERNELS);
// uint32 not included in ALL_TYPES
TF_CALL_QUANTIZED_TYPES(REGISTER_KERNELS);
// quint16 not included in QUANTIZIED_TYPES
TF_CALL_quint16(REGISTER_KERNELS);
#undef REGISTER_KERNELS

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
// Only register 'Assign' on GPU for the subset of types also supported by
// 'Variable' (see variable_ops.cc.)
#define REGISTER_GPU_KERNELS(type)                                 \
  REGISTER_KERNEL_BUILDER(                                         \
      Name("Assign").Device(DEVICE_GPU).TypeConstraint<type>("T"), \
      AssignOpT<GPUDevice, type>);

TF_CALL_GPU_ALL_TYPES(REGISTER_GPU_KERNELS);
TF_CALL_int64(REGISTER_GPU_KERNELS);
TF_CALL_uint32(REGISTER_GPU_KERNELS);
TF_CALL_uint8(REGISTER_GPU_KERNELS);
#undef REGISTER_GPU_KERNELS
#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM


#define REGISTER_KERNELS(type)                                        \
  REGISTER_KERNEL_BUILDER(                                            \
      Name("AssignAdd").Device(DEVICE_CPU).TypeConstraint<type>("T"), \
      DenseUpdateOp<CPUDevice, type, DenseUpdateType::ADD>);          \
  REGISTER_KERNEL_BUILDER(                                            \
      Name("AssignSub").Device(DEVICE_CPU).TypeConstraint<type>("T"), \
      DenseUpdateOp<CPUDevice, type, DenseUpdateType::SUB>);

TF_CALL_NUMBER_TYPES(REGISTER_KERNELS);
#undef REGISTER_KERNELS

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
#define REGISTER_GPU_KERNELS(type)                                    \
  REGISTER_KERNEL_BUILDER(                                            \
      Name("AssignAdd").Device(DEVICE_GPU).TypeConstraint<type>("T"), \
      DenseUpdateOp<GPUDevice, type, DenseUpdateType::ADD>);          \
  REGISTER_KERNEL_BUILDER(                                            \
      Name("AssignSub").Device(DEVICE_GPU).TypeConstraint<type>("T"), \
      DenseUpdateOp<GPUDevice, type, DenseUpdateType::SUB>);
TF_CALL_GPU_NUMBER_TYPES(REGISTER_GPU_KERNELS);
TF_CALL_int64(REGISTER_GPU_KERNELS);
TF_CALL_uint8(REGISTER_GPU_KERNELS);
#undef REGISTER_GPU_KERNELS
#endif  // end GOOGLE_CUDA || TENSORFLOW_USE_ROCM

}  // namespace tensorflow
