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
class MHTracer_DTPStensorflowPScorePSkernelsPSdense_update_functorDTcc {
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
   MHTracer_DTPStensorflowPScorePSkernelsPSdense_update_functorDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSkernelsPSdense_update_functorDTcc() {
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

#include "tensorflow/core/kernels/dense_update_functor.h"

#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/variant_op_registry.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {

typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;

namespace functor {

template <>
struct DenseUpdate<CPUDevice, string, ASSIGN> {
  void operator()(const CPUDevice& d, typename TTypes<tstring>::Flat params,
                  typename TTypes<tstring>::ConstFlat update) {
    if (params.dimension(0) == 1) {
      params.data()->resize(update.data()->size());
      auto work = [&params, &update](int64_t start, int64_t end) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdense_update_functorDTcc mht_0(mht_0_v, 208, "", "./tensorflow/core/kernels/dense_update_functor.cc", "lambda");

        memmove(const_cast<char*>(params.data()->data()) + start,
                update.data()->data() + start, end - start);
      };
      d.parallelFor(update.data()->size(),
                    Eigen::TensorOpCost(.1,  // chosen to force large chunks
                                        .1, 0),
                    work);
    } else {
      auto work = [&params, &update](int64_t start, int64_t end) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdense_update_functorDTcc mht_1(mht_1_v, 220, "", "./tensorflow/core/kernels/dense_update_functor.cc", "lambda");

        for (int i = start; i < end; ++i) {
          params.data()[i].resize(update.data()[i].size());
          memmove(const_cast<char*>(params.data()[i].data()),
                  update.data()[i].data(), update.data()[i].size());
        }
      };
      int64_t estimated_string_size;
      if (update.size() > 0) {
        // first element of the tensor seems as good a guess as any of the sizes
        // of the strings contained within...
        estimated_string_size =
            std::max(update.data()[0].size(), sizeof(tstring));
      } else {
        estimated_string_size = sizeof(tstring);
      }
      d.parallelFor(
          params.dimension(0),
          Eigen::TensorOpCost(estimated_string_size, estimated_string_size, 0),
          work);
    }
  }
};

}  // namespace functor

#define CPU_DENSE_COPY(T)                                               \
  case DataTypeToEnum<T>::value: {                                      \
    functor::DenseUpdate<CPUDevice, T, ASSIGN> copy_functor_;           \
    copy_functor_(context->eigen_device<CPUDevice>(), tensor.flat<T>(), \
                  from.flat<T>());                                      \
    break;                                                              \
  }

#define INSTANTIATE_GET_VARIANT_COPY_FN(DEVICE, TYPE_CALLER, TYPE_DENSE_COPY) \
  template <>                                                                 \
  Status VariantCopyFn<DEVICE>(OpKernelContext * context, const Tensor& from, \
                               Tensor* to) {                                  \
    Tensor tensor;                                                            \
    AllocatorAttributes attr;                                                 \
    attr.set_gpu_compatible(true);                                            \
    attr.set_nic_compatible(true);                                            \
    TF_RETURN_IF_ERROR(                                                       \
        context->allocate_temp(from.dtype(), from.shape(), &tensor, attr));   \
    switch (from.dtype()) {                                                   \
      TYPE_CALLER(TYPE_DENSE_COPY);                                           \
      default:                                                                \
        return errors::InvalidArgument(                                       \
            "VariantCopyFn: Could not perform a deep copy of variant "        \
            "element of type: ",                                              \
            DataTypeString(from.dtype()),                                     \
            " using device: ", context->device()->name());                    \
    }                                                                         \
    *to = tensor;                                                             \
    return Status::OK();                                                      \
  }

INSTANTIATE_GET_VARIANT_COPY_FN(CPUDevice, TF_CALL_ALL_TYPES, CPU_DENSE_COPY);

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
#define GPU_DENSE_COPY(T)                                               \
  case DataTypeToEnum<T>::value: {                                      \
    functor::DenseUpdate<GPUDevice, T, ASSIGN> copy_functor_;           \
    copy_functor_(context->eigen_device<GPUDevice>(), tensor.flat<T>(), \
                  from.flat<T>());                                      \
    break;                                                              \
  }
#define TF_CALL_GPU_AND_ADDITIONAL_TYPES(T) \
  TF_CALL_GPU_ALL_TYPES(T);                 \
  TF_CALL_int32(T);                         \
  TF_CALL_int64(T);
INSTANTIATE_GET_VARIANT_COPY_FN(GPUDevice, TF_CALL_GPU_AND_ADDITIONAL_TYPES,
                                GPU_DENSE_COPY);
#undef TF_CALL_GPU_AND_ADDITIONAL_TYPES
#undef GPU_DENSE_COPY
#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM

#undef CPU_DENSE_COPY
#undef INSTANTIATE_GET_VARIANT_COPY_FN

}  // namespace tensorflow
