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
class MHTracer_DTPStensorflowPScorePSkernelsPSctc_loss_opDTcc {
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
   MHTracer_DTPStensorflowPScorePSkernelsPSctc_loss_opDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSkernelsPSctc_loss_opDTcc() {
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

// See docs in ../ops/ctc_ops.cc.

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
#define EIGEN_USE_GPU
#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM

#include "tensorflow/core/framework/bounds_check.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/util/ctc/ctc_loss_calculator.h"
#include "tensorflow/core/util/sparse/sparse_tensor.h"

#if GOOGLE_CUDA
#include "third_party/gpus/cudnn/cudnn.h"
#endif  // GOOGLE_CUDA

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
#include "tensorflow/core/kernels/conv_ops_gpu.h"
#include "tensorflow/core/util/stream_executor_util.h"
#include "tensorflow/core/util/tensor_format.h"
#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM

namespace tensorflow {

typedef Eigen::ThreadPoolDevice CPUDevice;
#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
using GPUDevice = Eigen::GpuDevice;

namespace {
using se::Stream;
using se::StreamExecutor;
using se::dnn::RnnStateTensorDescriptor;
using se::dnn::ToDataType;

template <typename T>
void DoHistogram(OpKernelContext* ctx, const Tensor* labels_indices,
                 int num_indices, int batch_size,
                 std::vector<int>* labels_lengths) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSctc_loss_opDTcc mht_0(mht_0_v, 226, "", "./tensorflow/core/kernels/ctc_loss_op.cc", "DoHistogram");

  const T* h_in = labels_indices->flat<T>().data();
  for (int i = 0; i < num_indices; i++) {
    const T& key = h_in[i * 2];
    (*labels_lengths)[key]++;
  }
}

}  // end namespace
#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM

template <typename T>
class CTCLossOp : public OpKernel {
  typedef Eigen::Map<
      const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> >
      InputMap;
  typedef Eigen::Map<
      Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> >
      OutputMap;

 public:
  explicit CTCLossOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSctc_loss_opDTcc mht_1(mht_1_v, 250, "", "./tensorflow/core/kernels/ctc_loss_op.cc", "CTCLossOp");

    OP_REQUIRES_OK(ctx, ctx->GetAttr("preprocess_collapse_repeated",
                                     &preprocess_collapse_repeated_));
    OP_REQUIRES_OK(ctx,
                   ctx->GetAttr("ctc_merge_repeated", &ctc_merge_repeated_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("ignore_longer_outputs_than_inputs",
                                     &ignore_longer_outputs_than_inputs_));
  }

  void Compute(OpKernelContext* ctx) override {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSctc_loss_opDTcc mht_2(mht_2_v, 262, "", "./tensorflow/core/kernels/ctc_loss_op.cc", "Compute");

    const Tensor* inputs;
    const Tensor* labels_indices;
    const Tensor* labels_values;
    const Tensor* seq_len;
    OP_REQUIRES_OK(ctx, ctx->input("inputs", &inputs));
    OP_REQUIRES_OK(ctx, ctx->input("labels_indices", &labels_indices));
    OP_REQUIRES_OK(ctx, ctx->input("labels_values", &labels_values));
    OP_REQUIRES_OK(ctx, ctx->input("sequence_length", &seq_len));

    OP_REQUIRES(ctx, inputs->shape().dims() == 3,
                errors::InvalidArgument("inputs is not a 3-Tensor"));
    OP_REQUIRES(ctx, TensorShapeUtils::IsVector(seq_len->shape()),
                errors::InvalidArgument("sequence_length is not a vector"));
    OP_REQUIRES(ctx, TensorShapeUtils::IsMatrix(labels_indices->shape()),
                errors::InvalidArgument("labels_indices is not a matrix"));
    OP_REQUIRES(ctx, labels_indices->dim_size(1) > 1,
                errors::InvalidArgument(
                    "labels_indices second dimension must be >= 1. Received ",
                    labels_indices->dim_size(1)));
    OP_REQUIRES(ctx, TensorShapeUtils::IsVector(labels_values->shape()),
                errors::InvalidArgument("labels_values is not a vector"));

    const TensorShape& inputs_shape = inputs->shape();
    const int64_t max_time = inputs_shape.dim_size(0);
    OP_REQUIRES(ctx, max_time != 0,
                errors::InvalidArgument(
                    "Max time or first dimension of input cannot be 0."));
    const int64_t batch_size = inputs_shape.dim_size(1);
    const int64_t num_classes_raw = inputs_shape.dim_size(2);
    OP_REQUIRES(
        ctx, FastBoundsCheck(num_classes_raw, std::numeric_limits<int>::max()),
        errors::InvalidArgument("num_classes cannot exceed max int"));
    const int num_classes = static_cast<const int>(num_classes_raw);

    OP_REQUIRES(
        ctx, batch_size == seq_len->dim_size(0),
        errors::InvalidArgument("len(sequence_length) != batch_size.  ",
                                "len(sequence_length):  ", seq_len->dim_size(0),
                                " batch_size: ", batch_size));
    auto seq_len_t = seq_len->vec<int32>();

    OP_REQUIRES(ctx, labels_indices->dim_size(0) == labels_values->dim_size(0),
                errors::InvalidArgument(
                    "labels_indices and labels_values must contain the "
                    "same number of rows, but saw shapes: ",
                    labels_indices->shape().DebugString(), " vs. ",
                    labels_values->shape().DebugString()));

    OP_REQUIRES(ctx, batch_size != 0,
                errors::InvalidArgument("batch_size must not be 0"));

    // Figure out the maximum label length to use as sparse tensor dimension.
    auto labels_indices_t = labels_indices->matrix<int64_t>();
    int64_t max_label_len = 0;
    for (int i = 0; i < labels_indices->dim_size(0); i++) {
      max_label_len = std::max(max_label_len, labels_indices_t(i, 1) + 1);
    }

    TensorShape labels_shape({batch_size, max_label_len});
    std::vector<int64_t> order{0, 1};
    sparse::SparseTensor labels_sp;
    OP_REQUIRES_OK(
        ctx, sparse::SparseTensor::Create(*labels_indices, *labels_values,
                                          labels_shape, order, &labels_sp));

    Status labels_sp_valid = labels_sp.IndicesValid();
    OP_REQUIRES(ctx, labels_sp_valid.ok(),
                errors::InvalidArgument("label SparseTensor is not valid: ",
                                        labels_sp_valid.error_message()));

    typename ctc::CTCLossCalculator<T>::LabelSequences labels_t(batch_size);
    for (const auto& g : labels_sp.group({0})) {  // iterate by batch
      const int64_t batch_indices = g.group()[0];
      OP_REQUIRES(ctx, FastBoundsCheck(batch_indices, batch_size),
                  errors::InvalidArgument("labels batch index must be between ",
                                          0, " and ", batch_size,
                                          " but saw: ", batch_indices));

      auto values = g.values<int32>();
      std::vector<int>* b_values = &labels_t[batch_indices];
      b_values->resize(values.size());
      for (int i = 0; i < values.size(); ++i) (*b_values)[i] = values(i);
    }

    OP_REQUIRES(ctx, static_cast<size_t>(batch_size) == labels_t.size(),
                errors::InvalidArgument("len(labels) != batch_size.  ",
                                        "len(labels):  ", labels_t.size(),
                                        " batch_size: ", batch_size));

    for (int64_t b = 0; b < batch_size; ++b) {
      OP_REQUIRES(
          ctx, seq_len_t(b) <= max_time,
          errors::InvalidArgument("sequence_length(", b, ") <= ", max_time));
    }

    Tensor* loss = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output("loss", seq_len->shape(), &loss));
    auto loss_t = loss->vec<T>();

    Tensor* gradient;
    OP_REQUIRES_OK(ctx,
                   ctx->allocate_output("gradient", inputs_shape, &gradient));
    auto gradient_t = gradient->tensor<T, 3>();
    auto inputs_t = inputs->tensor<T, 3>();
    std::vector<OutputMap> gradient_list_t;
    std::vector<InputMap> input_list_t;

    for (std::size_t t = 0; t < max_time; ++t) {
      input_list_t.emplace_back(inputs_t.data() + t * batch_size * num_classes,
                                batch_size, num_classes);
      gradient_list_t.emplace_back(
          gradient_t.data() + t * batch_size * num_classes, batch_size,
          num_classes);
    }

    gradient_t.setZero();

    // Assumption: the blank index is num_classes - 1
    ctc::CTCLossCalculator<T> ctc_loss_calculator(num_classes - 1, 0);
    DeviceBase::CpuWorkerThreads workers =
        *ctx->device()->tensorflow_cpu_worker_threads();
    OP_REQUIRES_OK(ctx, ctc_loss_calculator.CalculateLoss(
                            seq_len_t, labels_t, input_list_t,
                            preprocess_collapse_repeated_, ctc_merge_repeated_,
                            ignore_longer_outputs_than_inputs_, &loss_t,
                            &gradient_list_t, &workers));
  }

 private:
  bool preprocess_collapse_repeated_;
  bool ctc_merge_repeated_;
  bool ignore_longer_outputs_than_inputs_;

  TF_DISALLOW_COPY_AND_ASSIGN(CTCLossOp<T>);
};

#define REGISTER_CPU(T)                                          \
  REGISTER_KERNEL_BUILDER(                                       \
      Name("CTCLoss").Device(DEVICE_CPU).TypeConstraint<T>("T"), \
      CTCLossOp<T>);

REGISTER_CPU(float);
REGISTER_CPU(double);

#undef REGISTER_CPU

#if ((GOOGLE_CUDA && CUDNN_VERSION >= 7603) || TENSORFLOW_USE_ROCM)
class CTCLossOpGPU : public OpKernel {
 public:
  explicit CTCLossOpGPU(OpKernelConstruction* ctx) : OpKernel(ctx) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSctc_loss_opDTcc mht_3(mht_3_v, 415, "", "./tensorflow/core/kernels/ctc_loss_op.cc", "CTCLossOpGPU");

    bool preprocess_collapse_repeated;
    bool ctc_merge_repeated;
    bool ignore_longer_outputs_than_inputs;
    OP_REQUIRES_OK(ctx, ctx->GetAttr("preprocess_collapse_repeated",
                                     &preprocess_collapse_repeated));
    OP_REQUIRES_OK(ctx,
                   ctx->GetAttr("ctc_merge_repeated", &ctc_merge_repeated));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("ignore_longer_outputs_than_inputs",
                                     &ignore_longer_outputs_than_inputs));

    OP_REQUIRES(ctx, !preprocess_collapse_repeated,
                errors::InvalidArgument("GPU CTCLossOp requires "
                                        "preprocess_collapse_repeated to be "
                                        "false"));
    OP_REQUIRES(ctx, ctc_merge_repeated,
                errors::InvalidArgument("GPU CTCLossOp requires "
                                        "ctc_merge_repeated to be "
                                        "true"));
    OP_REQUIRES(ctx, !ignore_longer_outputs_than_inputs,
                errors::InvalidArgument("GPU CTCLossOp requires "
                                        "ignore_longer_outputs_than_inputs to"
                                        "be false"));
  }

  void Compute(OpKernelContext* ctx) override {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSctc_loss_opDTcc mht_4(mht_4_v, 443, "", "./tensorflow/core/kernels/ctc_loss_op.cc", "Compute");

    const Tensor* inputs;
    const Tensor* labels_indices;
    const Tensor* labels_values;
    const Tensor* seq_len;
    OP_REQUIRES_OK(ctx, ctx->input("inputs", &inputs));
    OP_REQUIRES_OK(ctx, ctx->input("labels_indices", &labels_indices));
    OP_REQUIRES_OK(ctx, ctx->input("labels_values", &labels_values));
    OP_REQUIRES_OK(ctx, ctx->input("sequence_length", &seq_len));

    OP_REQUIRES(ctx, inputs->shape().dims() == 3,
                errors::InvalidArgument("inputs is not a 3-Tensor"));
    OP_REQUIRES(ctx, TensorShapeUtils::IsVector(seq_len->shape()),
                errors::InvalidArgument("sequence_length is not a vector"));
    OP_REQUIRES(ctx, TensorShapeUtils::IsMatrix(labels_indices->shape()),
                errors::InvalidArgument("labels_indices is not a matrix"));
    OP_REQUIRES(ctx, TensorShapeUtils::IsVector(labels_values->shape()),
                errors::InvalidArgument("labels_values is not a vector"));

    const TensorShape& inputs_shape = inputs->shape();
    const int64_t max_time_raw = inputs_shape.dim_size(0);
    const int64_t batch_size_raw = inputs_shape.dim_size(1);
    const int64_t num_classes_raw = inputs_shape.dim_size(2);
    OP_REQUIRES(ctx,
                FastBoundsCheck(max_time_raw, std::numeric_limits<int>::max()),
                errors::InvalidArgument("max_time_ cannot exceed max int"));
    OP_REQUIRES(
        ctx, FastBoundsCheck(batch_size_raw, std::numeric_limits<int>::max()),
        errors::InvalidArgument("batch_size cannot exceed max int"));
    OP_REQUIRES(
        ctx, FastBoundsCheck(num_classes_raw, std::numeric_limits<int>::max()),
        errors::InvalidArgument("num_classes cannot exceed max int"));
    const int max_time = static_cast<const int>(max_time_raw);
    const int batch_size = static_cast<const int>(batch_size_raw);
    const int num_classes = static_cast<const int>(num_classes_raw);

    OP_REQUIRES(
        ctx, batch_size == seq_len->dim_size(0),
        errors::InvalidArgument("len(sequence_length) != batch_size.  ",
                                "len(sequence_length):  ", seq_len->dim_size(0),
                                " batch_size: ", batch_size));

    OP_REQUIRES(ctx, labels_indices->dim_size(0) == labels_values->dim_size(0),
                errors::InvalidArgument(
                    "labels_indices and labels_values must contain the "
                    "same number of rows, but saw shapes: ",
                    labels_indices->shape().DebugString(), " vs. ",
                    labels_values->shape().DebugString()));
    auto num_indices = labels_indices->dim_size(0);

    OP_REQUIRES(ctx, batch_size != 0,
                errors::InvalidArgument("batch_size must not be 0"));

    Tensor* loss = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output("loss", seq_len->shape(), &loss));

    Tensor* gradient = nullptr;
    OP_REQUIRES_OK(ctx,
                   ctx->allocate_output("gradient", inputs_shape, &gradient));

    // Convert the labels_indices to labels_lengths.
    std::vector<int> labels_lengths(batch_size, 0);
    DoHistogram<int64_t>(ctx, labels_indices, num_indices, batch_size,
                         &labels_lengths);

    StreamExecutor* executor = ctx->op_device_context()->stream()->parent();
    se::dnn::DataType data_type = ToDataType<float>::value;

    auto probs_desc_s = executor->createRnnStateTensorDescriptor(
        max_time, batch_size, num_classes, data_type);
    OP_REQUIRES_OK(ctx, probs_desc_s.status());
    std::unique_ptr<RnnStateTensorDescriptor> probs_desc =
        probs_desc_s.ConsumeValueOrDie();

    auto grads_desc_s = executor->createRnnStateTensorDescriptor(
        max_time, batch_size, num_classes, data_type);
    OP_REQUIRES_OK(ctx, grads_desc_s.status());
    std::unique_ptr<RnnStateTensorDescriptor> grads_desc =
        grads_desc_s.ConsumeValueOrDie();

    absl::Span<const int32> labels_data(labels_values->flat<int32>().data(),
                                        num_indices);
    absl::Span<const int32> labels_lengths_data(labels_lengths.data(),
                                                batch_size);
    absl::Span<const int32> input_lengths_data(seq_len->flat<int32>().data(),
                                               batch_size);

    auto probs_data = StreamExecutorUtil::AsDeviceMemory<float>(*inputs);
    auto costs_data = StreamExecutorUtil::AsDeviceMemory<float>(*loss);
    auto grads_data = StreamExecutorUtil::AsDeviceMemory<float>(*gradient);

    // Set the memory limitation to 4GB for workspace memory.
    DnnScratchAllocator workspace_allocator(1LL << 32, ctx);

    Stream* stream = ctx->op_device_context()->stream();
    bool cudnn_launch_status =
        stream
            ->ThenCtcLoss(*probs_desc, probs_data, labels_data,
                          labels_lengths_data, input_lengths_data, &costs_data,
                          *grads_desc, &grads_data, &workspace_allocator)
            .ok();

    if (!cudnn_launch_status) {
      ctx->SetStatus(errors::Internal("cuDNN CTCLoss launch failure"));
    }
  }

 private:
  TF_DISALLOW_COPY_AND_ASSIGN(CTCLossOpGPU);
};

REGISTER_KERNEL_BUILDER(Name("CTCLossV2")
                            .Device(DEVICE_GPU)
                            .HostMemory("labels_indices")
                            .HostMemory("labels_values")
                            .HostMemory("sequence_length"),
                        CTCLossOpGPU);
#endif  // ((GOOGLE_CUDA && CUDNN_VERSION >= 7603)  || TENSORFLOW_USE_ROCM)
}  // end namespace tensorflow
