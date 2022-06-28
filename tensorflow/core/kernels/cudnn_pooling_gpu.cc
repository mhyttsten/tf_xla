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
class MHTracer_DTPStensorflowPScorePSkernelsPScudnn_pooling_gpuDTcc {
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
   MHTracer_DTPStensorflowPScorePSkernelsPScudnn_pooling_gpuDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSkernelsPScudnn_pooling_gpuDTcc() {
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

#define USE_EIGEN_TENSOR
#define EIGEN_USE_THREADS

#include <array>

#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/kernels/conv_2d.h"
#include "tensorflow/core/kernels/conv_3d.h"
#include "tensorflow/core/kernels/conv_ops_gpu.h"
#include "tensorflow/core/kernels/cudnn_pooling_gpu.h"

typedef Eigen::GpuDevice GPUDevice;

namespace tensorflow {

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM

template <typename T>
void DnnPooling3dOp<T>::Compute(OpKernelContext* context,
                                se::dnn::PoolingMode pooling_mode,
                                const std::array<int64_t, 3>& window,
                                const std::array<int64_t, 3>& stride,
                                const std::array<int64_t, 3>& padding,
                                TensorFormat data_format,
                                const Tensor& tensor_in, Tensor* output) {
  const auto in_shape = tensor_in.shape();
  const auto out_shape = output->shape();

  const int64_t in_batch = GetTensorDim(tensor_in, data_format, 'N');
  const int64_t in_features = GetTensorDim(tensor_in, data_format, 'C');

  Tensor transformed_input;
  if (data_format == FORMAT_NHWC) {
    OP_REQUIRES_OK(context, context->allocate_temp(
                                DataTypeToEnum<T>::value,
                                ShapeFromFormat(FORMAT_NCHW, tensor_in.shape(),
                                                data_format),
                                &transformed_input));
    functor::NHWCToNCHW<GPUDevice, T, 5>()(context->eigen_device<GPUDevice>(),
                                           tensor_in.tensor<T, 5>(),
                                           transformed_input.tensor<T, 5>());
  } else {
    transformed_input = tensor_in;
  }
  Tensor transformed_output;
  if (data_format == FORMAT_NHWC) {
    OP_REQUIRES_OK(context,
                   context->allocate_temp(
                       DataTypeToEnum<T>::value,
                       ShapeFromFormat(FORMAT_NCHW, out_shape, data_format),
                       &transformed_output));
  } else {
    transformed_output = *output;
  }

  se::dnn::PoolingDescriptor pooling_desc(3);
  pooling_desc.set_pooling_mode(pooling_mode);
  se::dnn::BatchDescriptor input_desc(3);
  input_desc.set_count(in_batch)
      .set_feature_map_count(in_features)
      .set_layout(se::dnn::DataLayout::kBatchDepthYX);
  se::dnn::BatchDescriptor output_desc(3);
  output_desc.set_count(in_batch)
      .set_feature_map_count(in_features)
      .set_layout(se::dnn::DataLayout::kBatchDepthYX);
  for (size_t i = 0; i < window.size(); ++i) {
    const auto dim_i = static_cast<se::dnn::DimIndex>(i);
    pooling_desc.set_window(dim_i, window[i]);
    pooling_desc.set_stride(dim_i, stride[i]);
    pooling_desc.set_padding(dim_i, padding[i]);
    input_desc.set_spatial_dim(dim_i,
                               GetTensorDim(tensor_in, data_format, '2' - i));
    output_desc.set_spatial_dim(dim_i,
                                GetTensorDim(out_shape, data_format, '2' - i));
  }

  auto input_data = AsDeviceMemory(transformed_input.template flat<T>().data(),
                                   transformed_input.template flat<T>().size());
  auto output_data =
      AsDeviceMemory(transformed_output.template flat<T>().data(),
                     transformed_output.template flat<T>().size());

  auto* stream = context->op_device_context()->stream();
  OP_REQUIRES(context, stream, errors::Internal("No GPU stream available."));

#if TENSORFLOW_USE_ROCM
  static int64 PoolingScratchSize = GetDnnWorkspaceLimit(
      // default value is in bytes despite the name of the environment variable
      "TF_CUDNN_WORKSPACE_LIMIT_IN_MB", 1LL << 32  // 4GB
  );

  DnnScratchAllocator scratch_allocator(PoolingScratchSize, context);
  bool status =
      stream
          ->ThenPoolForward(pooling_desc, input_desc, input_data, output_desc,
                            &output_data, &scratch_allocator)
          .ok();
#else
  bool status = stream
                    ->ThenPoolForward(pooling_desc, input_desc, input_data,
                                      output_desc, &output_data)
                    .ok();
#endif

  OP_REQUIRES(context, status,
              errors::Internal("dnn PoolForward launch failed"));

  if (data_format == FORMAT_NHWC) {
    auto toConstTensor = [](const Tensor& x) -> const Tensor { return x; };
    functor::NCHWToNHWC<GPUDevice, T, 5>()(
        context->eigen_device<GPUDevice>(),
        toConstTensor(transformed_output).template tensor<T, 5>(),
        output->tensor<T, 5>());
  }
}

template <typename T>
void DnnPooling3dGradOp<T>::Compute(
    OpKernelContext* context, se::dnn::PoolingMode pooling_mode,
    const std::array<int64_t, 3>& window, const std::array<int64_t, 3>& stride,
    const std::array<int64_t, 3>& padding,
    const std::array<int64_t, 3>& output_size, TensorFormat data_format,
    const Tensor& out_backprop, const TensorShape& tensor_in_shape,
    const Tensor* tensor_in, const Tensor* tensor_out, Tensor* input_backprop) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSkernelsPScudnn_pooling_gpuDTcc mht_0(mht_0_v, 308, "", "./tensorflow/core/kernels/cudnn_pooling_gpu.cc", "DnnPooling3dGradOp<T>::Compute");

  CHECK((pooling_mode != se::dnn::PoolingMode::kMaximum) ||
        (tensor_in && tensor_out))
      << "For MaxPoolGrad, both tensor_in and tensor_out needs to be "
         "specified";

  const int64_t in_batch = GetTensorDim(tensor_in_shape, data_format, 'N');
  const int64_t in_features = GetTensorDim(tensor_in_shape, data_format, 'C');

  Tensor transformed_input;
  TensorShape transformed_input_shape;
  if (data_format == FORMAT_NHWC || tensor_in == nullptr) {
    transformed_input_shape =
        ShapeFromFormat(FORMAT_NCHW, tensor_in_shape, data_format);
    OP_REQUIRES_OK(context, context->allocate_temp(DataTypeToEnum<T>::value,
                                                   transformed_input_shape,
                                                   &transformed_input));
  } else {
    transformed_input = *tensor_in;
  }
  Tensor transformed_output;
  TensorShape transformed_output_shape;
  if (data_format == FORMAT_NHWC || tensor_out == nullptr) {
    transformed_output_shape =
        ShapeFromFormat(FORMAT_NCHW, out_backprop.shape(), data_format);
    OP_REQUIRES_OK(context, context->allocate_temp(DataTypeToEnum<T>::value,
                                                   transformed_output_shape,
                                                   &transformed_output));
  } else {
    transformed_output = *tensor_out;
  }
  Tensor transformed_input_backprop;
  if (data_format == FORMAT_NHWC) {
    OP_REQUIRES_OK(context,
                   context->allocate_temp(DataTypeToEnum<T>::value,
                                          transformed_input_shape,
                                          &transformed_input_backprop));
  } else {
    transformed_input_backprop = *input_backprop;
  }
  Tensor transformed_output_backprop;
  if (data_format == FORMAT_NHWC) {
    OP_REQUIRES_OK(context,
                   context->allocate_temp(DataTypeToEnum<T>::value,
                                          transformed_output_shape,
                                          &transformed_output_backprop));
  } else {
    transformed_output_backprop = out_backprop;
  }
  if (data_format == FORMAT_NHWC) {
    if (tensor_in != nullptr) {
      functor::NHWCToNCHW<GPUDevice, T, 5>()(context->eigen_device<GPUDevice>(),
                                             tensor_in->tensor<T, 5>(),
                                             transformed_input.tensor<T, 5>());
    }
    if (tensor_out != nullptr) {
      functor::NHWCToNCHW<GPUDevice, T, 5>()(context->eigen_device<GPUDevice>(),
                                             tensor_out->tensor<T, 5>(),
                                             transformed_output.tensor<T, 5>());
    }
    functor::NHWCToNCHW<GPUDevice, T, 5>()(
        context->eigen_device<GPUDevice>(), out_backprop.tensor<T, 5>(),
        transformed_output_backprop.tensor<T, 5>());
  }

  se::dnn::PoolingDescriptor pooling_desc(3);
  pooling_desc.set_pooling_mode(pooling_mode);

  se::dnn::BatchDescriptor orig_output_desc(3);
  orig_output_desc.set_count(in_batch)
      .set_feature_map_count(in_features)
      .set_layout(se::dnn::DataLayout::kBatchDepthYX);

  se::dnn::BatchDescriptor orig_input_desc(3);
  orig_input_desc.set_count(in_batch)
      .set_feature_map_count(in_features)
      .set_layout(se::dnn::DataLayout::kBatchDepthYX);

  for (size_t i = 0; i < window.size(); ++i) {
    const auto dim_i = static_cast<se::dnn::DimIndex>(i);
    pooling_desc.set_window(dim_i, window[i]);
    pooling_desc.set_stride(dim_i, stride[i]);
    pooling_desc.set_padding(dim_i, padding[i]);
    orig_input_desc.set_spatial_dim(
        dim_i, GetTensorDim(tensor_in_shape, data_format, '2' - i));
    orig_output_desc.set_spatial_dim(dim_i, output_size[i]);
  }

  auto orig_output_data =
      AsDeviceMemory(transformed_output.template flat<T>().data(),
                     transformed_output.template flat<T>().size());
  auto orig_input_data =
      AsDeviceMemory(transformed_input.template flat<T>().data(),
                     transformed_input.template flat<T>().size());
  auto output_backprop_data =
      AsDeviceMemory(transformed_output_backprop.template flat<T>().data(),
                     transformed_output_backprop.template flat<T>().size());
  auto input_backprop_data =
      AsDeviceMemory(transformed_input_backprop.template flat<T>().data(),
                     transformed_input_backprop.template flat<T>().size());

  auto* stream = context->op_device_context()->stream();
  OP_REQUIRES(context, stream, errors::Internal("No GPU stream available."));

#if TENSORFLOW_USE_ROCM
  static int64 PoolingScratchSize = GetDnnWorkspaceLimit(
      // default value is in bytes despite the name of the environment variable
      "TF_CUDNN_WORKSPACE_LIMIT_IN_MB", 1LL << 32  // 4GB
  );

  DnnScratchAllocator scratch_allocator(PoolingScratchSize, context);
  bool status = stream
                    ->ThenPoolBackward(pooling_desc, orig_input_desc,
                                       orig_input_data, orig_output_desc,
                                       orig_output_data, output_backprop_data,
                                       &input_backprop_data, &scratch_allocator)
                    .ok();
#else
  bool status =
      stream
          ->ThenPoolBackward(pooling_desc, orig_input_desc, orig_input_data,
                             orig_output_desc, orig_output_data,
                             output_backprop_data, &input_backprop_data)
          .ok();
#endif

  OP_REQUIRES(context, status,
              errors::Internal("dnn PoolBackward launch failed"));

  if (data_format == FORMAT_NHWC) {
    auto toConstTensor = [](const Tensor& x) -> const Tensor { return x; };
    functor::NCHWToNHWC<GPUDevice, T, 5>()(
        context->eigen_device<GPUDevice>(),
        toConstTensor(transformed_input_backprop).template tensor<T, 5>(),
        input_backprop->tensor<T, 5>());
  }
}

#define DEFINE_DNN_OPS(T)           \
  template class DnnPooling3dOp<T>; \
  template class DnnPooling3dGradOp<T>;
TF_CALL_float(DEFINE_DNN_OPS) TF_CALL_half(DEFINE_DNN_OPS)
#undef DEFINE_DNN_OPS

#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM

}  // namespace tensorflow
