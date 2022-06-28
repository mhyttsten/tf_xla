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
class MHTracer_DTPStensorflowPScorePSkernelsPSimagePSextract_image_patches_opDTcc {
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
   MHTracer_DTPStensorflowPScorePSkernelsPSimagePSextract_image_patches_opDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSkernelsPSimagePSextract_image_patches_opDTcc() {
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

// See docs in ../ops/image_ops.cc.

#define USE_EIGEN_TENSOR
#define EIGEN_USE_THREADS

#include "tensorflow/core/kernels/image/extract_image_patches_op.h"

#include <vector>

#include "tensorflow/core/framework/bounds_check.h"
#include "tensorflow/core/framework/kernel_shape_util.h"
#include "tensorflow/core/framework/numeric_op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/kernels/ops_util.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/util/tensor_format.h"

namespace tensorflow {

typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;

static inline void ParseAttributeVec4(OpKernelConstruction* context,
                                      const string& attr_name,
                                      std::vector<int32>* attr) {
   std::vector<std::string> mht_0_v;
   mht_0_v.push_back("attr_name: \"" + attr_name + "\"");
   MHTracer_DTPStensorflowPScorePSkernelsPSimagePSextract_image_patches_opDTcc mht_0(mht_0_v, 214, "", "./tensorflow/core/kernels/image/extract_image_patches_op.cc", "ParseAttributeVec4");

  OP_REQUIRES_OK(context, context->GetAttr(attr_name, attr));
  OP_REQUIRES(
      context, (*attr)[0] == 1 && (*attr)[3] == 1,
      errors::Unimplemented("Only support ", attr_name, " across space."));
  OP_REQUIRES(context, (*attr)[1] >= 1 && (*attr)[2] >= 1,
              errors::OutOfRange(attr_name, " is out of range."));
}

template <typename Device, typename T>
class ExtractImagePatchesOp : public UnaryOp<T> {
 public:
  explicit ExtractImagePatchesOp(OpKernelConstruction* context)
      : UnaryOp<T>(context) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSimagePSextract_image_patches_opDTcc mht_1(mht_1_v, 230, "", "./tensorflow/core/kernels/image/extract_image_patches_op.cc", "ExtractImagePatchesOp");

    ParseAttributeVec4(context, "ksizes", &ksizes_);
    ParseAttributeVec4(context, "strides", &strides_);
    ParseAttributeVec4(context, "rates", &rates_);
    OP_REQUIRES_OK(context, context->GetAttr("padding", &padding_));
  }

  void Compute(OpKernelContext* context) override {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSimagePSextract_image_patches_opDTcc mht_2(mht_2_v, 240, "", "./tensorflow/core/kernels/image/extract_image_patches_op.cc", "Compute");

    // Input tensor is of the following dimensions:
    // [ batch, in_rows, in_cols, channels ]
    const Tensor& input = context->input(0);
    OP_REQUIRES(context, input.dims() == 4,
                errors::InvalidArgument("input must be 4-dimensional",
                                        input.shape().DebugString()));

    const int batch = input.dim_size(0);
    const int in_rows = input.dim_size(1);
    const int in_cols = input.dim_size(2);
    const int depth = input.dim_size(3);

    const int ksize_rows = ksizes_[1];
    const int ksize_cols = ksizes_[2];

    const int stride_rows = strides_[1];
    const int stride_cols = strides_[2];

    const int rate_rows = rates_[1];
    const int rate_cols = rates_[2];

    const int ksize_rows_eff = ksize_rows + (ksize_rows - 1) * (rate_rows - 1);
    const int ksize_cols_eff = ksize_cols + (ksize_cols - 1) * (rate_cols - 1);

    int64_t out_rows = 0, out_cols = 0;
    int64_t pad_rows = 0, pad_cols = 0;
    OP_REQUIRES_OK(context,
                   GetWindowedOutputSize(in_rows, ksize_rows_eff, stride_rows,
                                         padding_, &out_rows, &pad_rows));
    OP_REQUIRES_OK(context,
                   GetWindowedOutputSize(in_cols, ksize_cols_eff, stride_cols,
                                         padding_, &out_cols, &pad_cols));

    const std::vector<int64_t> out_sizes = {batch, out_rows, out_cols,
                                            ksize_rows * ksize_cols * depth};
    TensorShape out_shape(out_sizes);

    Tensor* output = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(0, out_shape, &output));

    // If there is nothing to compute, return.
    if (out_shape.num_elements() == 0) {
      return;
    }

    functor::ExtractImagePatchesForward<Device, T>()(
        context->eigen_device<Device>(), input.tensor<T, 4>(), ksize_rows,
        ksize_cols, stride_rows, stride_cols, rate_rows, rate_cols,
        BrainPadding2EigenPadding(padding_), output->tensor<T, 4>());
  }

 private:
  std::vector<int32> ksizes_;
  std::vector<int32> strides_;
  std::vector<int32> rates_;

  Padding padding_;

  TF_DISALLOW_COPY_AND_ASSIGN(ExtractImagePatchesOp);
};

// Registration of the CPU implementations.
#define REGISTER(T)                                                          \
  REGISTER_KERNEL_BUILDER(                                                   \
      Name("ExtractImagePatches").Device(DEVICE_CPU).TypeConstraint<T>("T"), \
      ExtractImagePatchesOp<CPUDevice, T>);

TF_CALL_NUMBER_TYPES(REGISTER);

#undef REGISTER

#if (defined(GOOGLE_CUDA) && GOOGLE_CUDA) || \
    (defined(TENSORFLOW_USE_ROCM) && TENSORFLOW_USE_ROCM)

// Forward declarations of the functor specializations for GPU.
namespace functor {

#define DECLARE_GPU_SPEC(T)                                             \
  template <>                                                           \
  void ExtractImagePatchesForward<GPUDevice, T>::operator()(            \
      const GPUDevice& d, typename TTypes<T, 4>::ConstTensor input,     \
      int patch_rows, int patch_cols, int stride_rows, int stride_cols, \
      int rate_rows, int rate_cols, const Eigen::PaddingType& padding,  \
      typename TTypes<T, 4>::Tensor output);                            \
  extern template struct ExtractImagePatchesForward<GPUDevice, T>;

TF_CALL_GPU_ALL_TYPES(DECLARE_GPU_SPEC);

#undef DECLARE_GPU_SPEC

}  // namespace functor

// Registration of the GPU implementations.
#define REGISTER(T)                                                          \
  REGISTER_KERNEL_BUILDER(                                                   \
      Name("ExtractImagePatches").Device(DEVICE_GPU).TypeConstraint<T>("T"), \
      ExtractImagePatchesOp<GPUDevice, T>);

TF_CALL_GPU_ALL_TYPES(REGISTER);

#undef REGISTER

#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM

}  // namespace tensorflow
