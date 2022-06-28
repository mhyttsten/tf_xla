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
class MHTracer_DTPStensorflowPScorePSkernelsPSmklPSmkl_avgpooling_opDTcc {
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
   MHTracer_DTPStensorflowPScorePSkernelsPSmklPSmkl_avgpooling_opDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSkernelsPSmklPSmkl_avgpooling_opDTcc() {
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

/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#ifdef INTEL_MKL
#define EIGEN_USE_THREADS

#include "dnnl.hpp"
#include "tensorflow/core/common_runtime/device.h"
#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/kernels/mkl/mkl_pooling_ops_common.h"

using dnnl::algorithm;
using dnnl::engine;
using dnnl::error;
using dnnl::memory;
using dnnl::pooling_backward;
using dnnl::pooling_forward;
using dnnl::prop_kind;

namespace tensorflow {

typedef Eigen::ThreadPoolDevice CPUDevice;

template <typename Device, typename T, bool native_format = false>
class MklAvgPoolingOp : public MklPoolingForwardOpBase<T> {
 public:
  explicit MklAvgPoolingOp(OpKernelConstruction* context)
      : MklPoolingForwardOpBase<T>(context) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSmklPSmkl_avgpooling_opDTcc mht_0(mht_0_v, 210, "", "./tensorflow/core/kernels/mkl/mkl_avgpooling_op.cc", "MklAvgPoolingOp");

    // Workspace is an oneDNN construct that is only used in Max Pooling.
    // So set workspace_enabled_ to false.
    this->workspace_enabled_ = false;
    this->native_format_ = native_format;
  }

  void Compute(OpKernelContext* context) override {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSmklPSmkl_avgpooling_opDTcc mht_1(mht_1_v, 220, "", "./tensorflow/core/kernels/mkl/mkl_avgpooling_op.cc", "Compute");

    try {
      const Tensor& input_tensor =
          MklGetInput(context, this->kInputTensorIndexInput);
      MklDnnShape dnn_shape_input;
      GetMklShape(context, this->kInputTensorIndexInput, &dnn_shape_input,
                  this->native_format_);
      this->SanityCheckInput(context, input_tensor, dnn_shape_input);
      if (!context->status().ok()) return;

      MklDnnData<T> dnn_data_input(&cpu_engine_);

      // Initialize variables for the pooling op.
      MklPoolParameters pool_params;
      // Check whether pooling is 2D or 3D.
      bool is_pool2d = (this->ksize_.size() == 4);
      // Get the input tensor and initialize the pooling parameters.
      TensorShape input_tensor_shape = input_tensor.shape();
      this->InitMklPoolParameters(context, &pool_params, dnn_shape_input,
                                  input_tensor_shape);
      OP_REQUIRES_OK(context, context->status());

      Tensor* output_tensor = nullptr;
      memory::dims output_dims_mkl_order;
      this->GetOutputDims(pool_params, &output_dims_mkl_order);

      // If input is an empty tensor, allocate an empty output tensor.
      if (input_tensor.NumElements() == 0) {
        const int kOutputIndex = 0;
        this->AllocateEmptyOutputTensor(context, kOutputIndex, &pool_params,
                                        output_dims_mkl_order, &output_tensor);
        return;
      }

      memory::dims filter_dims, strides, padding_left, padding_right;
      // Get src/filter/stride/padding information.
      this->PoolParamsToDims(&pool_params, &filter_dims, &strides,
                             &padding_left, &padding_right, is_pool2d);

      // Get the input memory descriptor.
      memory::dims src_dims =
          dnn_shape_input.IsMklTensor()
              ? dnn_shape_input.GetSizesAsMklDnnDims()
              : is_pool2d ? TFShapeToMklDnnDimsInNCHW(input_tensor.shape(),
                                                      this->data_format_tf_)
                          : TFShapeToMklDnnDimsInNCDHW(input_tensor.shape(),
                                                       this->data_format_tf_);
      memory::desc input_md = dnn_shape_input.IsMklTensor()
                                  ? dnn_shape_input.GetMklLayout()
                                  : memory::desc(src_dims, MklDnnType<T>(),
                                                 this->data_format_mkldnn_);

      // Get an average pooling primitive from the op pool.
      MklPoolingFwdPrimitive<T>* pooling_fwd = nullptr;
      prop_kind pooling_prop_kind;
      bool int8_forward_inference =
          std::is_same<T, qint8>::value || std::is_same<T, quint8>::value;
      if (int8_forward_inference)
        pooling_prop_kind = prop_kind::forward_inference;
      else
        pooling_prop_kind = prop_kind::forward_training;

      MklPoolingParams fwdParams(
          src_dims, output_dims_mkl_order, filter_dims, strides, padding_left,
          padding_right, dnnl::algorithm::pooling_avg_exclude_padding,
          pooling_prop_kind,
          static_cast<memory::format_tag>(this->data_format_mkldnn_), input_md,
          this->native_format_);
      pooling_fwd = MklPoolingFwdPrimitiveFactory<T>::Get(fwdParams);

      // Allocate output tensor.
      this->AllocateOutputTensor(context, *(pooling_fwd->GetPoolingFwdPd()),
                                 output_dims_mkl_order,
                                 this->tensor_format_mkldnn_, &output_tensor);
      DCHECK(output_tensor);
      OP_REQUIRES_OK(context, context->status());

      const T* src_data = input_tensor.flat<T>().data();

      T* dst_data = output_tensor->flat<T>().data();
      std::shared_ptr<stream> fwd_cpu_stream;
      MklDnnThreadPool eigen_tp(context);
      fwd_cpu_stream.reset(CreateStream(&eigen_tp, pooling_fwd->GetEngine()));
      // Execute pooling op.
      pooling_fwd->Execute(src_data, dst_data, nullptr, fwd_cpu_stream);

      // Pass min, max from input to output.
      if (int8_forward_inference) {
        const Tensor& min_input_t = MklGetInput(context, 1);
        const Tensor& max_input_t = MklGetInput(context, 2);
        const float min_input = min_input_t.flat<float>()(0);
        const float max_input = max_input_t.flat<float>()(0);

        Tensor* output_min = nullptr;
        Tensor* output_max = nullptr;
        MklDnnShape output_min_mkl_shape, output_max_mkl_shape;
        output_min_mkl_shape.SetMklTensor(false);
        output_max_mkl_shape.SetMklTensor(false);
        AllocateOutputSetMklShape(context, 1, &output_min, {},
                                  output_min_mkl_shape, this->native_format_);
        AllocateOutputSetMklShape(context, 2, &output_max, {},
                                  output_max_mkl_shape, this->native_format_);
        output_min->flat<float>()(0) = min_input;
        output_max->flat<float>()(0) = max_input;
      }
    } catch (dnnl::error& e) {
      string error_msg = "Status: " + std::to_string(e.status) +
                         ", message: " + string(e.message) + ", in file " +
                         string(__FILE__) + ":" + std::to_string(__LINE__);
      OP_REQUIRES_OK(
          context,
          errors::Aborted("Operation received an exception:", error_msg));
    }
  }  // Compute

 private:
  engine cpu_engine_ = engine(engine::kind::cpu, 0);
};  // MklAvgPoolingOp

template <class Device, class T, bool native_format = false>
class MklAvgPoolingGradOp : public MklPoolingBackwardOpBase<T> {
 public:
  explicit MklAvgPoolingGradOp(OpKernelConstruction* context)
      : MklPoolingBackwardOpBase<T>(context) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSmklPSmkl_avgpooling_opDTcc mht_2(mht_2_v, 346, "", "./tensorflow/core/kernels/mkl/mkl_avgpooling_op.cc", "MklAvgPoolingGradOp");

    this->native_format_ = native_format;
  }

  void Compute(OpKernelContext* context) override {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSmklPSmkl_avgpooling_opDTcc mht_3(mht_3_v, 353, "", "./tensorflow/core/kernels/mkl/mkl_avgpooling_op.cc", "Compute");

    try {
      const Tensor& orig_input_tensor =
          MklGetInput(context, kInputTensorIndexInputShape);
      const Tensor& grad_tensor =
          MklGetInput(context, kInputTensorIndexInputGradient);

      MklDnnShape orig_input_mkl_shape, grad_mkl_shape;
      GetMklShape(context, kInputTensorIndexInputShape, &orig_input_mkl_shape,
                  this->native_format_);
      GetMklShape(context, kInputTensorIndexInputGradient, &grad_mkl_shape,
                  this->native_format_);
      if (!context->status().ok()) return;

      // Used to allocate output_diff_src/diff_src.
      MklDnnData<T> grad_dnn_data(&cpu_engine_);
      MklPoolParameters pool_params;
      auto shape_vec = orig_input_tensor.vec<int32>();
      TensorShape orig_input_shape;
      for (int i = 0; i < orig_input_tensor.NumElements(); i++) {
        orig_input_shape.AddDim(shape_vec(i));
      }

      bool is_pool2d = (this->ksize_.size() == 4);
      this->InitMklPoolParameters(context, &pool_params, orig_input_mkl_shape,
                                  orig_input_shape);

      memory::dims filter_dims, strides, padding_left, padding_right;
      this->PoolParamsToDims(&pool_params, &filter_dims, &strides,
                             &padding_left, &padding_right, is_pool2d);

      memory::dims orig_input_dims_mkl_order =
          orig_input_mkl_shape.IsMklTensor()
              ? orig_input_mkl_shape.GetSizesAsMklDnnDims()
              : is_pool2d ? TFShapeToMklDnnDimsInNCHW(orig_input_shape,
                                                      this->data_format_tf_)
                          : TFShapeToMklDnnDimsInNCDHW(orig_input_shape,
                                                       this->data_format_tf_);

      memory::dims diff_dst_dims =
          grad_mkl_shape.IsMklTensor()
              ? grad_mkl_shape.GetSizesAsMklDnnDims()
              : is_pool2d ? TFShapeToMklDnnDimsInNCHW(grad_tensor.shape(),
                                                      this->data_format_tf_)
                          : TFShapeToMklDnnDimsInNCDHW(grad_tensor.shape(),
                                                       this->data_format_tf_);
      memory::dims output_dims_mkl_order;
      this->GetOutputDims(pool_params, &output_dims_mkl_order);

      // get src memory::desc
      memory::desc src_md =
          orig_input_mkl_shape.IsMklTensor()
              ? orig_input_mkl_shape.GetMklLayout()
              : memory::desc(orig_input_dims_mkl_order, MklDnnType<T>(),
                             this->data_format_mkldnn_);

      // Get diff_dst memory::desc.
      memory::desc diff_dst_md =
          grad_mkl_shape.IsMklTensor()
              ? grad_mkl_shape.GetMklLayout()
              : memory::desc(diff_dst_dims, MklDnnType<T>(),
                             this->data_format_mkldnn_);

      // Pass prop_kind::forward_training to create a forward primitive
      // that is used in the backward pass.
      MklPoolingParams bwdParams(
          orig_input_dims_mkl_order, output_dims_mkl_order, filter_dims,
          strides, padding_left, padding_right,
          dnnl::algorithm::pooling_avg_exclude_padding,
          prop_kind::forward_training,
          static_cast<memory::format_tag>(this->data_format_mkldnn_), src_md,
          this->native_format_);
      MklPoolingBwdPrimitive<T>* pooling_bwd =
          MklPoolingBwdPrimitiveFactory<T>::Get(bwdParams);

      std::shared_ptr<stream> bwd_cpu_stream;
      MklDnnThreadPool eigen_tp(context);
      bwd_cpu_stream.reset(CreateStream(&eigen_tp, pooling_bwd->GetEngine()));
      Tensor* output_tensor = nullptr;
      this->AllocateOutputTensor(context, *(pooling_bwd->GetPoolingBwdPd()),
                                 orig_input_dims_mkl_order,
                                 this->tensor_format_mkldnn_, &output_tensor);

      // TODO(intel-tf): Refactor (lines 249-262) common code for
      // max & avg pooling into superclass or common utils function.
      // Check whether we need to reorder diff_dst.
      std::shared_ptr<PoolingBwdPd> pooling_bwd_pd =
          pooling_bwd->GetPoolingBwdPd();
      T* diff_dst_data = nullptr;
      if (!this->native_format_ &&
          (diff_dst_md != pooling_bwd_pd->diff_dst_desc())) {
        grad_dnn_data.SetUsrMem(diff_dst_md, &grad_tensor);
        grad_dnn_data.CheckReorderToOpMem(pooling_bwd_pd->diff_dst_desc(),
                                          cpu_engine_);
        diff_dst_data =
            static_cast<T*>(grad_dnn_data.GetOpMem().get_data_handle());
      } else {
        diff_dst_data =
            static_cast<T*>(const_cast<T*>(grad_tensor.flat<T>().data()));
      }

      T* diff_src_data = output_tensor->flat<T>().data();

      // Execute pooling op.
      pooling_bwd->Execute(diff_dst_data, diff_src_data, nullptr,
                           bwd_cpu_stream);
    } catch (dnnl::error& e) {
      string error_msg = "Status: " + std::to_string(e.status) +
                         ", message: " + string(e.message) + ", in file " +
                         string(__FILE__) + ":" + std::to_string(__LINE__);
      OP_REQUIRES_OK(context, errors::Aborted("Compute received an exception:",
                                              error_msg));
    }
  }

 private:
  // 0. Input("orig_input_shape: int32")
  // 1. Input("grad: T")
  const int kInputTensorIndexInputShape = 0;
  const int kInputTensorIndexInputGradient = 1;
  engine cpu_engine_ = engine(engine::kind::cpu, 0);
};  // MklAvgPoolingGradOp

#define REGISTER_MKL_AVGPOOL3D_KERNELS(T)                                     \
  REGISTER_KERNEL_BUILDER(                                                    \
      Name("_MklAvgPool3D")                                                   \
          .Device(DEVICE_CPU)                                                 \
          .TypeConstraint<T>("T")                                             \
          .Label(mkl_op_registry::kMklLayoutDependentOpLabel),                \
      MklAvgPoolingOp<CPUDevice, T>);                                         \
  REGISTER_KERNEL_BUILDER(                                                    \
      Name("_MklAvgPool3DGrad")                                               \
          .Device(DEVICE_CPU)                                                 \
          .TypeConstraint<T>("T")                                             \
          .Label(mkl_op_registry::kMklLayoutDependentOpLabel),                \
      MklAvgPoolingGradOp<CPUDevice, T>);                                     \
  REGISTER_KERNEL_BUILDER(Name("_MklNativeAvgPool3D")                         \
                              .Device(DEVICE_CPU)                             \
                              .TypeConstraint<T>("T")                         \
                              .Label(mkl_op_registry::kMklNameChangeOpLabel), \
                          MklAvgPoolingOp<CPUDevice, T, true>);               \
  REGISTER_KERNEL_BUILDER(Name("_MklNativeAvgPool3DGrad")                     \
                              .Device(DEVICE_CPU)                             \
                              .TypeConstraint<T>("T")                         \
                              .Label(mkl_op_registry::kMklNameChangeOpLabel), \
                          MklAvgPoolingGradOp<CPUDevice, T, true>);

TF_CALL_float(REGISTER_MKL_AVGPOOL3D_KERNELS);
TF_CALL_bfloat16(REGISTER_MKL_AVGPOOL3D_KERNELS);

#define REGISTER_MKL_AVGPOOL_KERNELS(T)                                       \
  REGISTER_KERNEL_BUILDER(                                                    \
      Name("_MklAvgPool")                                                     \
          .Device(DEVICE_CPU)                                                 \
          .TypeConstraint<T>("T")                                             \
          .Label(mkl_op_registry::kMklLayoutDependentOpLabel),                \
      MklAvgPoolingOp<CPUDevice, T>);                                         \
  REGISTER_KERNEL_BUILDER(                                                    \
      Name("_MklAvgPoolGrad")                                                 \
          .Device(DEVICE_CPU)                                                 \
          .TypeConstraint<T>("T")                                             \
          .Label(mkl_op_registry::kMklLayoutDependentOpLabel),                \
      MklAvgPoolingGradOp<CPUDevice, T>);                                     \
  REGISTER_KERNEL_BUILDER(Name("_MklNativeAvgPool")                           \
                              .Device(DEVICE_CPU)                             \
                              .TypeConstraint<T>("T")                         \
                              .Label(mkl_op_registry::kMklNameChangeOpLabel), \
                          MklAvgPoolingOp<CPUDevice, T, true>);               \
  REGISTER_KERNEL_BUILDER(Name("_MklNativeAvgPoolGrad")                       \
                              .Device(DEVICE_CPU)                             \
                              .TypeConstraint<T>("T")                         \
                              .Label(mkl_op_registry::kMklNameChangeOpLabel), \
                          MklAvgPoolingGradOp<CPUDevice, T, true>);

TF_CALL_float(REGISTER_MKL_AVGPOOL_KERNELS);
TF_CALL_bfloat16(REGISTER_MKL_AVGPOOL_KERNELS);

REGISTER_KERNEL_BUILDER(Name("_MklQuantizedAvgPool")
                            .Device(DEVICE_CPU)
                            .TypeConstraint<quint8>("T")
                            .Label(mkl_op_registry::kMklQuantizedOpLabel),
                        MklAvgPoolingOp<CPUDevice, quint8, true>);

REGISTER_KERNEL_BUILDER(Name("_MklQuantizedAvgPool")
                            .Device(DEVICE_CPU)
                            .TypeConstraint<qint8>("T")
                            .Label(mkl_op_registry::kMklQuantizedOpLabel),
                        MklAvgPoolingOp<CPUDevice, qint8, true>);

}  // namespace tensorflow

#endif  // INTEL_MKL
