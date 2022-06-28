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
class MHTracer_DTPStensorflowPScorePSkernelsPStensor_arrayDTcc {
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
   MHTracer_DTPStensorflowPScorePSkernelsPStensor_arrayDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSkernelsPStensor_arrayDTcc() {
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
#include "tensorflow/core/kernels/tensor_array.h"

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor_util.h"
#include "tensorflow/core/kernels/aggregate_ops_cpu.h"

namespace tensorflow {

typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;

namespace tensor_array {

#define TENSOR_ARRAY_WRITE_OR_ADD(Device, T)                                \
  template <>                                                               \
  Status AddToTensor<Device, T>(OpKernelContext * ctx, Tensor * sum,        \
                                const Tensor* current, const Tensor* add) { \
    functor::Add2Functor<Device, T> add_functor;                            \
    add_functor(ctx->template eigen_device<Device>(), sum->flat<T>(),       \
                current->flat<T>(), add->flat<T>());                        \
    return Status::OK();                                                    \
  }

#define TENSOR_ARRAY_WRITE_OR_ADD_CPU(T) TENSOR_ARRAY_WRITE_OR_ADD(CPUDevice, T)
TF_CALL_NUMBER_TYPES(TENSOR_ARRAY_WRITE_OR_ADD_CPU)
#undef TENSOR_ARRAY_WRITE_OR_ADD_CPU

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM

#define TENSOR_ARRAY_WRITE_OR_ADD_GPU(T) TENSOR_ARRAY_WRITE_OR_ADD(GPUDevice, T)
TF_CALL_GPU_NUMBER_TYPES(TENSOR_ARRAY_WRITE_OR_ADD_GPU);
TF_CALL_COMPLEX_TYPES(TENSOR_ARRAY_WRITE_OR_ADD_GPU);
#undef TENSOR_ARRAY_WRITE_OR_ADD_GPU

#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM

#undef TENSOR_ARRAY_WRITE_OR_ADD

#define TENSOR_ARRAY_SET_ZERO(Device, T)                                      \
  template <>                                                                 \
  Status TensorSetZero<Device, T>(OpKernelContext * ctx, Tensor * value) {    \
    functor::SetZeroFunctor<Device, T> set_zero_functor;                      \
    set_zero_functor(ctx->template eigen_device<Device>(), value->flat<T>()); \
    return Status::OK();                                                      \
  }

#define TENSOR_ARRAY_SET_ZERO_CPU(T) TENSOR_ARRAY_SET_ZERO(CPUDevice, T)
TF_CALL_NUMBER_TYPES(TENSOR_ARRAY_SET_ZERO_CPU);
TF_CALL_bool(TENSOR_ARRAY_SET_ZERO_CPU);
#undef TENSOR_ARRAY_SET_ZERO_CPU

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM

#define TENSOR_ARRAY_SET_ZERO_GPU(T) TENSOR_ARRAY_SET_ZERO(GPUDevice, T)
TF_CALL_GPU_NUMBER_TYPES(TENSOR_ARRAY_SET_ZERO_GPU);
TF_CALL_COMPLEX_TYPES(TENSOR_ARRAY_SET_ZERO_GPU);
#undef TENSOR_ARRAY_SET_ZERO_GPU

#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM

#undef TENSOR_ARRAY_SET_ZERO

}  // namespace tensor_array

std::atomic<int64_t> TensorArray::tensor_array_counter{0};

Status TensorArray::CopyShapesFrom(TensorArray* rhs,
                                   const TensorShape* shape_to_prepend) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSkernelsPStensor_arrayDTcc mht_0(mht_0_v, 254, "", "./tensorflow/core/kernels/tensor_array.cc", "TensorArray::CopyShapesFrom");

  mutex_lock l(mu_);
  mutex_lock l_rhs(rhs->mu_);
  TF_RETURN_IF_ERROR(LockedReturnIfClosed());
  TF_RETURN_IF_ERROR(rhs->LockedReturnIfClosed());
  if (tensors_.size() != rhs->tensors_.size()) {
    return errors::InvalidArgument(
        "TensorArray sizes do not match during CopyShapesFrom: ",
        handle_.vec<tstring>()(1), " has size ", tensors_.size(), " but rhs ",
        rhs->handle_.vec<tstring>()(1), " has size ", rhs->tensors_.size());
  }
  for (std::size_t i = 0; i < tensors_.size(); ++i) {
    // Skip "soft copy" of indices which have not been written.
    if (!rhs->tensors_[i].written) continue;

    // Copy the shape over.
    if (shape_to_prepend) {
      tensors_[i].shape = *shape_to_prepend;
      tensors_[i].shape.AppendShape(rhs->tensors_[i].shape);
    } else {
      tensors_[i].shape = rhs->tensors_[i].shape;
    }
    // Mark as written.  Reads will know that if written is true and
    // read is false, and cleared is false, to return zeros of the
    // appropriate shape.  Future aggregating writes will only use the shape
    // for validation.
    tensors_[i].written = true;
  }

  return Status::OK();
}

}  // namespace tensorflow
