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
class MHTracer_DTPStensorflowPScorePSkernelsPSdata_format_opsDTcc {
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
   MHTracer_DTPStensorflowPScorePSkernelsPSdata_format_opsDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSkernelsPSdata_format_opsDTcc() {
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

// See docs in ../ops/nn_ops.cc.

#define EIGEN_USE_THREADS

#include "tensorflow/core/kernels/data_format_ops.h"

#include <map>

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/platform/errors.h"

namespace tensorflow {

typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;

// Ensure that `src` and `dst` define a valid permutation.
// Ops defined in this file assume that user specifies a permutation via two
// string attributes. This check validates that these attributes properly define
// it to prevent security vulnerabilities.
static bool IsValidPermutation(const std::string& src, const std::string& dst) {
   std::vector<std::string> mht_0_v;
   mht_0_v.push_back("src: \"" + src + "\"");
   mht_0_v.push_back("dst: \"" + dst + "\"");
   MHTracer_DTPStensorflowPScorePSkernelsPSdata_format_opsDTcc mht_0(mht_0_v, 210, "", "./tensorflow/core/kernels/data_format_ops.cc", "IsValidPermutation");

  if (src.size() != dst.size()) {
    return false;
  }

  std::map<char, bool> characters;

  // Every character in `src` must be present only once
  for (const auto c : src) {
    if (characters[c]) {
      return false;
    }
    characters[c] = true;
  }

  // Every character in `dst` must show up in `src` exactly once
  for (const auto c : dst) {
    if (!characters[c]) {
      return false;
    }
    characters[c] = false;
  }

  // At this point, characters[] has been switched to true and false exactly
  // once for all character in `src` (and `dst`) so we have a valid permutation
  return true;
}

template <typename Device, typename T>
class DataFormatDimMapOp : public OpKernel {
 public:
  explicit DataFormatDimMapOp(OpKernelConstruction* context)
      : OpKernel(context) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdata_format_opsDTcc mht_1(mht_1_v, 245, "", "./tensorflow/core/kernels/data_format_ops.cc", "DataFormatDimMapOp");

    string src_format;
    OP_REQUIRES_OK(context, context->GetAttr("src_format", &src_format));
    string dst_format;
    OP_REQUIRES_OK(context, context->GetAttr("dst_format", &dst_format));
    OP_REQUIRES(context, src_format.size() == 4 || src_format.size() == 5,
                errors::InvalidArgument(
                    "Source format must be of length 4 or 5, received "
                    "src_format = ",
                    src_format));
    OP_REQUIRES(context, dst_format.size() == 4 || dst_format.size() == 5,
                errors::InvalidArgument("Destination format must be of length "
                                        "4 or 5, received dst_format = ",
                                        dst_format));
    OP_REQUIRES(
        context, IsValidPermutation(src_format, dst_format),
        errors::InvalidArgument(
            "Destination and source format must determine a permutation, got ",
            src_format, " and ", dst_format));
    dst_idx_ = Tensor(DT_INT32, {static_cast<int64_t>(src_format.size())});
    for (int i = 0; i < src_format.size(); ++i) {
      for (int j = 0; j < dst_format.size(); ++j) {
        if (dst_format[j] == src_format[i]) {
          dst_idx_.vec<int>()(i) = j;
          break;
        }
      }
    }
  }

  void Compute(OpKernelContext* context) override {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdata_format_opsDTcc mht_2(mht_2_v, 278, "", "./tensorflow/core/kernels/data_format_ops.cc", "Compute");

    const Tensor& input = context->input(0);
    Tensor* output;
    OP_REQUIRES_OK(context,
                   context->allocate_output(0, input.shape(), &output));
    functor::DataFormatDimMap<Device, T>()(context->eigen_device<Device>(),
                                           input.flat<T>(), output->flat<T>(),
                                           dst_idx_.vec<int>());
  }

  Tensor dst_idx_;
};

template <typename Device, typename T>
class DataFormatVecPermuteOp : public OpKernel {
 public:
  explicit DataFormatVecPermuteOp(OpKernelConstruction* context)
      : OpKernel(context) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdata_format_opsDTcc mht_3(mht_3_v, 298, "", "./tensorflow/core/kernels/data_format_ops.cc", "DataFormatVecPermuteOp");

    string src_format;
    OP_REQUIRES_OK(context, context->GetAttr("src_format", &src_format));
    OP_REQUIRES(context, src_format.size() == 4 || src_format.size() == 5,
                errors::InvalidArgument(
                    "Source format must be of length 4 or 5, received "
                    "src_format = ",
                    src_format));
    string dst_format;
    OP_REQUIRES_OK(context, context->GetAttr("dst_format", &dst_format));
    OP_REQUIRES(context, dst_format.size() == 4 || dst_format.size() == 5,
                errors::InvalidArgument("Destination format must be of length "
                                        "4 or 5, received dst_format = ",
                                        dst_format));
    OP_REQUIRES(
        context, IsValidPermutation(src_format, dst_format),
        errors::InvalidArgument(
            "Destination and source format must determine a permutation, got ",
            src_format, " and ", dst_format));
    src_format_ = src_format;
    dst_format_ = dst_format;
  }

  void Compute(OpKernelContext* context) override {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdata_format_opsDTcc mht_4(mht_4_v, 324, "", "./tensorflow/core/kernels/data_format_ops.cc", "Compute");

    const Tensor& input = context->input(0);
    OP_REQUIRES(context, input.dims() == 1 || input.dims() == 2,
                errors::InvalidArgument(
                    "input must be a vector or 2D tensor, but got shape ",
                    input.shape().DebugString()));

    const int full_dim_count = src_format_.size();
    const int spatial_dim_count = full_dim_count - 2;

    if (input.dims() == 1) {
      OP_REQUIRES(context,
                  input.NumElements() == spatial_dim_count ||
                      input.NumElements() == full_dim_count,
                  errors::InvalidArgument("1D input must be of size ",
                                          spatial_dim_count, " or ",
                                          full_dim_count, ", but got shape ",
                                          input.shape().DebugString()));
    } else if (input.dims() == 2) {
      OP_REQUIRES(context,
                  input.dim_size(0) == spatial_dim_count ||
                      input.dim_size(0) == full_dim_count,
                  errors::InvalidArgument("First dimension of 2D input must be "
                                          "of size ",
                                          spatial_dim_count, " or ",
                                          full_dim_count, ", but got shape ",
                                          input.shape().DebugString()));
      OP_REQUIRES(
          context, input.dim_size(1) == 2,
          errors::InvalidArgument(
              "Second dimension of 2D input must be of size 2, but got shape ",
              input.shape().DebugString()));
    }

    Tensor* output = nullptr;
    OP_REQUIRES_OK(context,
                   context->allocate_output(0, input.shape(), &output));
    // Support 1D and 2D cases.
    Eigen::DSizes<Eigen::DenseIndex, 10> dst_idx;
    string src_format_str = src_format_;
    string dst_format_str = dst_format_;
    if (input.dim_size(0) == spatial_dim_count) {
      // If the input is a vector of size spatial_dim_count, treat the elements
      // as spatial dimensions.
      auto keep_only_spatial_dimensions =
          [spatial_dim_count](string* format_str) -> void {
        auto new_end =
            std::remove_if(format_str->begin(), format_str->end(),
                           [spatial_dim_count](const char dim) {
                             return dim != 'H' && dim != 'W' &&
                                    (spatial_dim_count == 2 || dim != 'D');
                           });
        format_str->erase(new_end, format_str->end());
      };
      keep_only_spatial_dimensions(&src_format_str);
      keep_only_spatial_dimensions(&dst_format_str);
      if (spatial_dim_count == 3) {
        OP_REQUIRES(
            context, src_format_str.size() == 3 && dst_format_str.size() == 3,
            errors::InvalidArgument(
                "Format specifier must contain D, H and W for 2D case"));
      } else {
        DCHECK(spatial_dim_count == 2);
        OP_REQUIRES(context,
                    src_format_str.size() == 2 && dst_format_str.size() == 2,
                    errors::InvalidArgument(
                        "Format specifier must contain H and W for 2D case"));
      }
    }
    ComputeDstIndex(src_format_str, dst_format_str, input.dims(), &dst_idx);

    functor::DataFormatVecPermute<Device, T>()(context->eigen_device<Device>(),
                                               input.flat<T>(),
                                               output->flat<T>(), dst_idx);
  }

 private:
  // Finds out the destination index. Support 1D and 2D cases.
  // Example: HWNC --> NHWC
  // 1D: dst = [1, 2, 0, 3],
  // 2D: dst = [2, 3, 4, 5, 0, 1, 6, 7]
  static void ComputeDstIndex(const string& src_format_str,
                              const string& dst_format_str, int num_dim,
                              Eigen::DSizes<Eigen::DenseIndex, 10>* dst) {
   std::vector<std::string> mht_5_v;
   mht_5_v.push_back("src_format_str: \"" + src_format_str + "\"");
   mht_5_v.push_back("dst_format_str: \"" + dst_format_str + "\"");
   MHTracer_DTPStensorflowPScorePSkernelsPSdata_format_opsDTcc mht_5(mht_5_v, 412, "", "./tensorflow/core/kernels/data_format_ops.cc", "ComputeDstIndex");

    for (int i = 0; i < src_format_str.size(); ++i) {
      for (int j = 0; j < dst_format_str.size(); ++j) {
        if (dst_format_str[j] != src_format_str[i]) continue;
        // Found the dst index. Set output based on the number of dims.
        for (int k = 0; k < num_dim; ++k) {
          (*dst)[i * num_dim + k] = j * num_dim + k;
        }
      }
    }
  }

  string src_format_;
  string dst_format_;
};

#define REGISTER_KERNEL(T)                                                \
  REGISTER_KERNEL_BUILDER(                                                \
      Name("DataFormatDimMap").Device(DEVICE_CPU).TypeConstraint<T>("T"), \
      DataFormatDimMapOp<CPUDevice, T>);
TF_CALL_int32(REGISTER_KERNEL);
TF_CALL_int64(REGISTER_KERNEL);
#undef REGISTER_KERNEL

#define REGISTER_KERNEL(T)                                                    \
  REGISTER_KERNEL_BUILDER(                                                    \
      Name("DataFormatVecPermute").Device(DEVICE_CPU).TypeConstraint<T>("T"), \
      DataFormatVecPermuteOp<CPUDevice, T>);
TF_CALL_int32(REGISTER_KERNEL);
TF_CALL_int64(REGISTER_KERNEL);
#undef REGISTER_KERNEL

#define REGISTER_KERNEL(T)                             \
  REGISTER_KERNEL_BUILDER(Name("DataFormatDimMap")     \
                              .Device(DEVICE_CPU)      \
                              .Label("host")           \
                              .TypeConstraint<T>("T"), \
                          DataFormatDimMapOp<CPUDevice, T>);
TF_CALL_int32(REGISTER_KERNEL);
TF_CALL_int64(REGISTER_KERNEL);
#undef REGISTER_KERNEL

#define REGISTER_KERNEL(T)                             \
  REGISTER_KERNEL_BUILDER(Name("DataFormatVecPermute") \
                              .Device(DEVICE_CPU)      \
                              .Label("host")           \
                              .TypeConstraint<T>("T"), \
                          DataFormatVecPermuteOp<CPUDevice, T>);
TF_CALL_int32(REGISTER_KERNEL);
TF_CALL_int64(REGISTER_KERNEL);
#undef REGISTER_KERNEL

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
// Forward declarations of the functor specializations for GPU.
namespace functor {
#define DECLARE_GPU_SPEC(T)                                    \
  template <>                                                  \
  void DataFormatDimMap<GPUDevice, T>::operator()(             \
      const GPUDevice& d, typename TTypes<T>::ConstFlat x,     \
      typename TTypes<T>::Flat y, const TTypes<int>::Vec dst); \
  extern template struct DataFormatDimMap<GPUDevice, T>;
#define DECLARE_GPU_SPECS(T) DECLARE_GPU_SPEC(T);
TF_CALL_int32(DECLARE_GPU_SPECS);
TF_CALL_int64(DECLARE_GPU_SPECS);
#undef DECLARE_GPU_SPEC

#define DECLARE_GPU_SPEC(T)                                 \
  template <>                                               \
  void DataFormatVecPermute<GPUDevice, T>::operator()(      \
      const GPUDevice& d, typename TTypes<T>::ConstFlat x,  \
      typename TTypes<T>::Vec y,                            \
      const Eigen::DSizes<Eigen::DenseIndex, 10>& dst_idx); \
  extern template struct DataFormatVecPermute<GPUDevice, T>;
#define DECLARE_GPU_SPECS(T) DECLARE_GPU_SPEC(T);
TF_CALL_int32(DECLARE_GPU_SPECS);
TF_CALL_int64(DECLARE_GPU_SPECS);
#undef DECLARE_GPU_SPEC
}  // namespace functor

// Registration of the GPU implementations.
#define REGISTER_GPU_KERNEL(T)                                            \
  REGISTER_KERNEL_BUILDER(                                                \
      Name("DataFormatDimMap").Device(DEVICE_GPU).TypeConstraint<T>("T"), \
      DataFormatDimMapOp<GPUDevice, T>);                                  \
  REGISTER_KERNEL_BUILDER(Name("DataFormatDimMap")                        \
                              .Device(DEVICE_GPU)                         \
                              .HostMemory("x")                            \
                              .HostMemory("y")                            \
                              .Label("host")                              \
                              .TypeConstraint<T>("T"),                    \
                          DataFormatDimMapOp<CPUDevice, T>);
TF_CALL_int32(REGISTER_GPU_KERNEL);
TF_CALL_int64(REGISTER_GPU_KERNEL);
#undef REGISTER_GPU_KERNEL

#define REGISTER_GPU_KERNEL(T)                                                \
  REGISTER_KERNEL_BUILDER(                                                    \
      Name("DataFormatVecPermute").Device(DEVICE_GPU).TypeConstraint<T>("T"), \
      DataFormatVecPermuteOp<GPUDevice, T>);                                  \
  REGISTER_KERNEL_BUILDER(Name("DataFormatVecPermute")                        \
                              .Device(DEVICE_GPU)                             \
                              .HostMemory("x")                                \
                              .HostMemory("y")                                \
                              .Label("host")                                  \
                              .TypeConstraint<T>("T"),                        \
                          DataFormatVecPermuteOp<CPUDevice, T>);
TF_CALL_int32(REGISTER_GPU_KERNEL);
TF_CALL_int64(REGISTER_GPU_KERNEL);
#undef REGISTER_GPU_KERNEL
#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM

}  // namespace tensorflow
