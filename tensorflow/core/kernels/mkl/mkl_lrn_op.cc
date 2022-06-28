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
class MHTracer_DTPStensorflowPScorePSkernelsPSmklPSmkl_lrn_opDTcc {
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
   MHTracer_DTPStensorflowPScorePSkernelsPSmklPSmkl_lrn_opDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSkernelsPSmklPSmkl_lrn_opDTcc() {
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

// LRN = Local Response Normalization
// See docs in ../ops/nn_ops.cc. This opkernel uses MKL library, create MKL
// layout and primitives, use MKL dnn primitives to compute local
// response normalization

#ifdef INTEL_MKL

#define EIGEN_USE_THREADS

#include <unordered_map>
#include <vector>

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "dnnl.hpp"
#include "tensorflow/core/framework/bounds_check.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/kernels/ops_util.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/util/mkl_util.h"
#include "tensorflow/core/util/tensor_format.h"

#if !defined(IS_MOBILE_PLATFORM)
#include "tensorflow/core/util/work_sharder.h"
#endif

using dnnl::lrn_backward;
using dnnl::lrn_forward;
using dnnl::prop_kind;
using dnnl::stream;

namespace tensorflow {

namespace {
// Create a depth-by-depth band matrix with 1s along a swath of size (2 *
// depth_radius + 1) around the diagonal.
template <typename T>
void GetBandMatrix(int depth, int depth_radius,
                   Eigen::Tensor<T, 2, Eigen::RowMajor>* result) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSmklPSmkl_lrn_opDTcc mht_0(mht_0_v, 224, "", "./tensorflow/core/kernels/mkl/mkl_lrn_op.cc", "GetBandMatrix");

  result->setZero();
  for (int row = 0; row < depth; ++row) {
    const int begin = std::max<int>(0, row - depth_radius);
    const int end = std::min<int>(depth, row + depth_radius + 1);
    Eigen::DSizes<Eigen::DenseIndex, 2> start(row, begin);
    Eigen::DSizes<Eigen::DenseIndex, 2> sizes(1, end - begin);
    result->slice(start, sizes).setConstant(T(1));
  }
}

}  // namespace

template <typename T>
class MklLRNOp : public OpKernel {
 public:
  ~MklLRNOp() {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSmklPSmkl_lrn_opDTcc mht_1(mht_1_v, 243, "", "./tensorflow/core/kernels/mkl/mkl_lrn_op.cc", "~MklLRNOp");
}

  explicit MklLRNOp(OpKernelConstruction* context)
      : OpKernel(context), cpu_engine_(engine::kind::cpu, 0) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSmklPSmkl_lrn_opDTcc mht_2(mht_2_v, 249, "", "./tensorflow/core/kernels/mkl/mkl_lrn_op.cc", "MklLRNOp");

    int64 depth_radius64;
    OP_REQUIRES_OK(context, context->GetAttr("depth_radius", &depth_radius64));
    OP_REQUIRES(
        context,
        FastBoundsCheck(depth_radius64, std::numeric_limits<int>::max()),
        errors::InvalidArgument("depth_radius = ", depth_radius64,
                                " larger than int max"));
    depth_radius_ = static_cast<size_t>(depth_radius64);

    OP_REQUIRES_OK(context, context->GetAttr("bias", &bias_));
    OP_REQUIRES_OK(context, context->GetAttr("alpha", &alpha_));
    OP_REQUIRES_OK(context, context->GetAttr("beta", &beta_));
    workspace_enabled_ = false;
    OP_REQUIRES_OK(context,
                   context->GetAttr("workspace_enabled", &workspace_enabled_));
  }

  void Compute(OpKernelContext* context) override {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSmklPSmkl_lrn_opDTcc mht_3(mht_3_v, 270, "", "./tensorflow/core/kernels/mkl/mkl_lrn_op.cc", "Compute");

    try {
      SanityCheckInputs(context);
      if (!context->status().ok()) return;

      const Tensor& src_tensor = MklGetInput(context, kIdxInput);
      MklDnnShape src_dnn_shape;
      GetMklShape(context, kIdxInput, &src_dnn_shape);

      // MKL-DNN has a notion of kernel_size and not depth_radius.
      int kernel_size = 2 * depth_radius_ + 1;
      float new_alpha = alpha_ * kernel_size;

      // if the input tensor is not an MKL Tensor, or if the last
      // dimension is not channel, then just use Eigen.
      // MKL only support normalization over the channel dimension.
      if (!src_dnn_shape.IsMklTensor()) {
        MklDefaultToEigen(context, src_tensor);
        return;
      } else if (!src_dnn_shape.IsMklChannelDim(src_dnn_shape.GetDimension() -
                                                1)) {
        Tensor converted_tensor;
        OP_REQUIRES_OK(context,
                       ConvertMklToTF<T>(context, src_tensor, src_dnn_shape,
                                         &converted_tensor));
        MklDefaultToEigen(context, converted_tensor);
        return;
      }
      // At this point, we can assume that the src is an MklTensor
      // and we can enable the workspace
      workspace_enabled_ = true;

      MklDnnData<T> src_dnn_data(&cpu_engine_);
      MklDnnData<T> dst_dnn_data(&cpu_engine_);
      MklDnnData<uint8> workspace_dnn_data(&cpu_engine_);

      TensorShape tf_output_shape = src_tensor.shape();

      memory::desc src_md = src_dnn_shape.GetCurLayout();
      memory::dims input_dims = src_dnn_shape.GetSizesAsMklDnnDims();

      // Create memory for user input.
      // Since Tensorflow always performs normalization over last dimension,
      // and MKL-DNN performs normalization over Channel, we tell MKL-DNN
      // that input is in NHWC layout with Channel being the last dimension.
      src_dnn_data.SetUsrMem(src_md, &src_tensor);
      src_dnn_data.SetOpMemDesc(input_dims, memory::format_tag::nhwc);
      src_dnn_data.SetUsrMemDataHandle(&src_tensor, fwd_stream_);

      // dst_dnn_data has the same shape as input.
      dst_dnn_data.SetUsrMem(src_md);
      dst_dnn_data.SetOpMemDesc(input_dims, memory::format_tag::nhwc);

      // Create LRN primitive descriptor.
      // Tensorflow's normalization semantics is across channels.
      // MKL-DNN also supports normalization within channel.
      auto lrn_desc = lrn_forward::desc(
          prop_kind::forward, dnnl::algorithm::lrn_across_channels,
          src_dnn_data.GetUsrMemDesc(), kernel_size, new_alpha, beta_, bias_);
      auto lrn_prim_desc = lrn_forward::primitive_desc(lrn_desc, cpu_engine_);

      // Allocate output_dnn_data tensor.
      Tensor* output_tensor = nullptr;
      auto input_format = src_dnn_shape.GetTfDataFormat();
      AllocateOutputTensor(context, lrn_prim_desc, input_dims, input_format,
                           &output_tensor);
      OP_REQUIRES_OK(context, context->status());
      DCHECK(output_tensor != nullptr);
      dst_dnn_data.SetUsrMemDataHandle(output_tensor, fwd_stream_);

      // Handle workspace required for MKL-DNN.
      AllocateWorkspaceTensor(context, lrn_prim_desc, &workspace_dnn_data);
      OP_REQUIRES_OK(context, context->status());

      // Check for input reorder
      src_dnn_data.CheckReorderToOpMem(lrn_prim_desc.src_desc(), cpu_engine_);

      std::vector<primitive> net;
      MklDnnThreadPool eigen_tp(context);
      fwd_stream_.reset(CreateStream(&eigen_tp, cpu_engine_));
      net.push_back(lrn_forward(lrn_prim_desc));
      std::vector<std::unordered_map<int, memory>> net_args;
      net_args.push_back({{DNNL_ARG_SRC, src_dnn_data.GetOpMem()},
                          {DNNL_ARG_WORKSPACE, workspace_dnn_data.GetOpMem()},
                          {DNNL_ARG_DST, dst_dnn_data.GetOpMem()}});
      net.push_back(lrn_forward(lrn_prim_desc));
      net.at(0).execute(*fwd_stream_, net_args.at(0));
    } catch (dnnl::error& e) {
      string error_msg = "Status: " + std::to_string(e.status) +
                         ", message: " + string(e.message) + ", in file " +
                         string(__FILE__) + ":" + std::to_string(__LINE__);
      OP_REQUIRES_OK(
          context,
          errors::Aborted("Operation received an exception:", error_msg));
    }
  }

 private:
  void AllocateOutputTensor(
      OpKernelContext* context,
      const lrn_forward::primitive_desc& lrn_fwd_prim_desc,
      const memory::dims output_dims_mkl_order,
      const MklTensorFormat& output_tf_format, Tensor** output_tensor) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSmklPSmkl_lrn_opDTcc mht_4(mht_4_v, 375, "", "./tensorflow/core/kernels/mkl/mkl_lrn_op.cc", "AllocateOutputTensor");

    DCHECK(output_tensor != nullptr);
    memory::desc dst_pd = lrn_fwd_prim_desc.dst_desc();

    MklDnnShape output_mkl_shape;
    // We only handle the case when the inputs and output are in Mkl format
    // Any other case is handled by Eigen
    output_mkl_shape.SetMklTensor(true);
    output_mkl_shape.SetMklLayout(&dst_pd);
    output_mkl_shape.SetElemType(MklDnnType<T>());
    output_mkl_shape.SetTfLayout(output_dims_mkl_order.size(),
                                 output_dims_mkl_order, output_tf_format);
    TensorShape output_tf_shape;
    // only allocate enough space for the elements we need.
    size_t num_bytes = dst_pd.get_size();
    CHECK_EQ(num_bytes % sizeof(T), 0);
    output_tf_shape.AddDim(num_bytes / sizeof(T));
    AllocateOutputSetMklShape(context, kIdxOutput, output_tensor,
                              output_tf_shape, output_mkl_shape);
  }

  // Fallback implementation - Taken from lrn_op.cc
  // TODO(intel-tf) Check if we can use EigenLRNOp directly instead of making a
  // copy.
  void MklDefaultToEigen(OpKernelContext* context, const Tensor& input) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSmklPSmkl_lrn_opDTcc mht_5(mht_5_v, 402, "", "./tensorflow/core/kernels/mkl/mkl_lrn_op.cc", "MklDefaultToEigen");

    const int batch = static_cast<int>(input.dim_size(0));
    const int rows = static_cast<int>(input.dim_size(1));
    const int cols = static_cast<int>(input.dim_size(2));
    const int depth = static_cast<int>(input.dim_size(3));
    const int nodes = cols * rows;

    auto in_shaped = input.shaped<T, 2>({nodes * batch, depth});
    // Multiplying the input with the band matrix has the effect of reducing
    // the correct patch along the depth.
    Eigen::Tensor<T, 2, Eigen::RowMajor> multiplier(depth, depth);
    GetBandMatrix<T>(depth, depth_radius_, &multiplier);

    Tensor* output_dnn_data = nullptr;
    MklDnnShape mkl_output_mkl_shape;
    mkl_output_mkl_shape.SetMklTensor(false);
    mkl_output_mkl_shape.SetDimensions(4);
    AllocateOutputSetMklShape(context, kIdxOutput, &output_dnn_data,
                              input.shape(), mkl_output_mkl_shape);
    DCHECK(output_dnn_data != nullptr);

    Tensor* workspace_tensor = nullptr;
    MklDnnShape workspace_mkl_shape;
    workspace_mkl_shape.SetMklTensor(false);
    TensorShape workspace_tf_shape;
    workspace_tf_shape.AddDim(0);
    AllocateOutputSetMklShape(context, kIdxWorkspace, &workspace_tensor,
                              workspace_tf_shape, workspace_mkl_shape);
    DCHECK(workspace_tensor);

    auto out_shaped = output_dnn_data->shaped<T, 2>({nodes * batch, depth});
    Eigen::array<DimPair, 1> dims = {{DimPair(1, 0)}};
    auto tmp = in_shaped.square().contract(multiplier, dims) * alpha_ + bias_;
    if (beta_ == T(1)) {
      out_shaped.device(context->eigen_cpu_device()) =
          in_shaped * tmp.inverse();
    } else if (beta_ == T(0.5)) {
      out_shaped.device(context->eigen_cpu_device()) = in_shaped * tmp.rsqrt();
    } else {
      out_shaped.device(context->eigen_cpu_device()) =
          in_shaped * (tmp.log() * -beta_).exp();
    }
  }

  void AllocateWorkspaceTensor(
      OpKernelContext* context,
      const lrn_forward::primitive_desc& lrn_fwd_prim_desc,
      MklDnnData<uint8>* dnn_data_wksp) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSmklPSmkl_lrn_opDTcc mht_6(mht_6_v, 452, "", "./tensorflow/core/kernels/mkl/mkl_lrn_op.cc", "AllocateWorkspaceTensor");

    DCHECK(dnn_data_wksp != nullptr);
    Tensor* workspace_tensor = nullptr;
    memory::desc workspace_pd = lrn_fwd_prim_desc.workspace_desc();
    size_t workspace_bytes = workspace_pd.get_size();
    MklDnnShape workspace_mkl_shape;
    // the workspace tensor is a uint8 tensor that has
    // exactly the number of bytes necessary
    workspace_mkl_shape.SetMklTensor(false);
    TensorShape workspace_tf_shape;
    workspace_tf_shape.AddDim(workspace_bytes);
    AllocateOutputSetMklShape(context, kIdxWorkspace, &workspace_tensor,
                              workspace_tf_shape, workspace_mkl_shape);
    DCHECK(workspace_tensor != nullptr);
    dnn_data_wksp->SetUsrMem(workspace_pd, workspace_tensor);
  }

  void SanityCheckInputs(OpKernelContext* context) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSmklPSmkl_lrn_opDTcc mht_7(mht_7_v, 472, "", "./tensorflow/core/kernels/mkl/mkl_lrn_op.cc", "SanityCheckInputs");

    const Tensor& src_tensor = MklGetInput(context, kIdxInput);
    MklDnnShape src_dnn_shape;
    GetMklShape(context, kIdxInput, &src_dnn_shape);
    if (src_dnn_shape.IsMklTensor()) {
      OP_REQUIRES(context, src_dnn_shape.GetDimension() == 4,
                  errors::InvalidArgument("input must be 4-dimensional"));
      OP_REQUIRES(context,
                  FastBoundsCheck(src_tensor.NumElements(),
                                  std::numeric_limits<int>::max()),
                  errors::InvalidArgument("argument to LRN too large"));
    } else {
      OP_REQUIRES(context, src_tensor.dims() == 4,
                  errors::InvalidArgument("input must be 4-dimensional"));
      OP_REQUIRES(context,
                  FastBoundsCheck(src_tensor.NumElements(),
                                  std::numeric_limits<int>::max()),
                  errors::InvalidArgument("argument to LRN too large"));
    }
  }
  const int kIdxInput = 0, kIdxOutput = 0, kIdxWorkspace = 1;

  typedef typename Eigen::Tensor<T, 1, Eigen::RowMajor>::DimensionPair DimPair;
  bool workspace_enabled_;
  int depth_radius_;
  float bias_;
  float alpha_;
  float beta_;
  engine cpu_engine_;
  std::shared_ptr<stream> fwd_stream_;
};

template <typename T>
class MklLRNGradOp : public OpKernel {
 public:
  explicit MklLRNGradOp(OpKernelConstruction* context)
      : OpKernel(context), cpu_engine_(engine::kind::cpu, 0) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSmklPSmkl_lrn_opDTcc mht_8(mht_8_v, 511, "", "./tensorflow/core/kernels/mkl/mkl_lrn_op.cc", "MklLRNGradOp");

    int64 depth_radius64;
    OP_REQUIRES_OK(context, context->GetAttr("depth_radius", &depth_radius64));
    OP_REQUIRES(
        context,
        FastBoundsCheck(depth_radius64, std::numeric_limits<int>::max()),
        errors::InvalidArgument("depth_radius = ", depth_radius64,
                                " larger than int max"));
    depth_radius_ = static_cast<int>(depth_radius64);
    OP_REQUIRES_OK(context, context->GetAttr("bias", &bias_));
    OP_REQUIRES_OK(context, context->GetAttr("alpha", &alpha_));
    OP_REQUIRES_OK(context, context->GetAttr("beta", &beta_));
    workspace_enabled_ = false;
    OP_REQUIRES_OK(context,
                   context->GetAttr("workspace_enabled", &workspace_enabled_));
    bwd_stream_.reset(new stream(cpu_engine_));
  }

  void Compute(OpKernelContext* context) override {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSmklPSmkl_lrn_opDTcc mht_9(mht_9_v, 532, "", "./tensorflow/core/kernels/mkl/mkl_lrn_op.cc", "Compute");

    try {
      SanityCheckInputs(context);
      if (!context->status().ok()) return;

      MklDnnData<T> input_grad_dnn_data(&cpu_engine_);
      MklDnnData<T> orig_input_dnn_data(&cpu_engine_);
      MklDnnData<T> orig_output_dnn_data(&cpu_engine_);
      MklDnnData<T> output_dnn_data(&cpu_engine_);

      MklDnnShape input_grad_dnn_shape, orig_input_dnn_shape,
          orig_output_dnn_shape;
      GetMklShape(context, kIdxGradient, &input_grad_dnn_shape);
      GetMklShape(context, kIdxOrigInput, &orig_input_dnn_shape);
      GetMklShape(context, kIdxOrigOutput, &orig_output_dnn_shape);

      // We only use oneDNN if all of the necessary inputs are present
      // in oneDNN format, and Channel is the last dimension
      bool can_use_mkldnn = workspace_enabled_ &&
                            input_grad_dnn_shape.IsMklTensor() &&
                            orig_input_dnn_shape.IsMklTensor() &&
                            orig_output_dnn_shape.IsMklTensor() &&
                            input_grad_dnn_shape.IsMklChannelDim(
                                input_grad_dnn_shape.GetDimension() - 1) &&
                            orig_input_dnn_shape.IsMklChannelDim(
                                orig_input_dnn_shape.GetDimension() - 1) &&
                            orig_output_dnn_shape.IsMklChannelDim(
                                orig_output_dnn_shape.GetDimension() - 1);

      if (!can_use_mkldnn) {
        // Fallback to eigen
        MklDefaultToEigen(context);
        return;
      }
      // At this point, we have the all clear to use MklDnn constructs
      // Naming: diff_dst is input_gradient_tensor; src is orig_input_tensor.
      const Tensor& input_grad_tensor = MklGetInput(context, kIdxGradient);
      const Tensor& orig_input_tensor = MklGetInput(context, kIdxOrigInput);

      // Get input sizes in MKL-DNN required NCHW format.
      // LRN does not have data_format attribute. But by default it has
      // NHWC format.
      memory::desc original_output_md = orig_output_dnn_shape.GetCurLayout();
      memory::desc target_diff_dst_md = ConfigureInputGradient(
          input_grad_tensor, input_grad_dnn_shape, &input_grad_dnn_data);

      memory::desc orig_input_md = orig_input_dnn_shape.GetCurLayout();
      memory::dims orig_input_dims =
          orig_input_dnn_shape.GetSizesAsMklDnnDims();
      orig_input_dnn_data.SetUsrMem(orig_input_md, &orig_input_tensor);
      orig_input_dnn_data.SetOpMemDesc(orig_input_dims,
                                       memory::format_tag::nhwc);
      orig_input_dnn_data.SetUsrMemDataHandle(&orig_input_tensor, bwd_stream_);

      // output_dnn_data has the same shape as original input
      output_dnn_data.SetUsrMem(orig_input_md);
      output_dnn_data.SetOpMemDesc(orig_input_dims, memory::format_tag::nhwc);

      // MKL-DNN has a notion of kernel_size and not depth_radius.
      int kernel_size = 2 * depth_radius_ + 1;
      float new_alpha = alpha_ * kernel_size;

      // Create LRN backward primitive descriptor. It requires LRN forward
      // primitive descriptor also.
      auto lrn_fwd_desc = lrn_forward::desc(
          prop_kind::forward, dnnl::algorithm::lrn_across_channels,
          orig_input_md, kernel_size, new_alpha, beta_, bias_);
      auto lrn_fwd_prim_desc =
          lrn_forward::primitive_desc(lrn_fwd_desc, cpu_engine_);
      auto lrn_bwd_desc = lrn_backward::desc(
          dnnl::algorithm::lrn_across_channels, original_output_md,
          target_diff_dst_md, kernel_size, new_alpha, beta_, bias_);
      auto lrn_bwd_prim_desc = lrn_backward::primitive_desc(
          lrn_bwd_desc, cpu_engine_, lrn_fwd_prim_desc);

      Tensor* output_tensor = nullptr;
      auto orig_input_format = orig_input_dnn_shape.GetTfDataFormat();
      AllocateOutputTensor(context, lrn_bwd_prim_desc, orig_input_dims,
                           orig_input_format, &output_tensor);
      OP_REQUIRES_OK(context, context->status());
      DCHECK(output_tensor != nullptr);
      output_dnn_data.SetUsrMemDataHandle(output_tensor, bwd_stream_);

      // Create LRN primitive and add it to the net
      // At this point, workspace is enabled, so we don't need
      // to check. Pass input workspace to LRN backward primitive.
      const Tensor& workspace_tensor = MklGetInput(context, kIdxWorkspace);
      MklDnnData<uint8> workspace_dnn_data(&cpu_engine_);
      ConfigureWorkspace(workspace_tensor, lrn_fwd_prim_desc.workspace_desc(),
                         &workspace_dnn_data);

      // Check for input reordering on the diff dst input
      input_grad_dnn_data.CheckReorderToOpMem(lrn_bwd_prim_desc.diff_dst_desc(),
                                              cpu_engine_);

      // Check for input reordering on the original input
      orig_input_dnn_data.CheckReorderToOpMem(lrn_fwd_prim_desc.src_desc(),
                                              cpu_engine_);

      std::vector<primitive> net;
      std::vector<std::unordered_map<int, memory>> net_args;
      net.push_back(lrn_backward(lrn_bwd_prim_desc));
      net_args.push_back({{DNNL_ARG_SRC, orig_input_dnn_data.GetOpMem()},
                          {DNNL_ARG_DIFF_DST, input_grad_dnn_data.GetOpMem()},
                          {DNNL_ARG_DST, output_dnn_data.GetOpMem()}});
      net.push_back(lrn_backward(lrn_bwd_prim_desc));
      net.at(0).execute(*bwd_stream_, net_args.at(0));
    } catch (dnnl::error& e) {
      string error_msg = "Status: " + std::to_string(e.status) +
                         ", message: " + string(e.message) + ", in file " +
                         string(__FILE__) + ":" + std::to_string(__LINE__);
      OP_REQUIRES_OK(
          context,
          errors::Aborted("Operation received an exception:", error_msg));
    }
  }

  void AllocateOutputTensor(
      OpKernelContext* context,
      const lrn_backward::primitive_desc& lrn_bkwd_prim_desc,
      const memory::dims output_dims_mkl_order,
      const MklTensorFormat& output_tf_format, Tensor** output_tensor) {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSmklPSmkl_lrn_opDTcc mht_10(mht_10_v, 656, "", "./tensorflow/core/kernels/mkl/mkl_lrn_op.cc", "AllocateOutputTensor");

    DCHECK(output_tensor != nullptr);
    memory::desc dst_pd = lrn_bkwd_prim_desc.diff_src_desc();
    MklDnnShape output_mkl_shape;

    // We assume that all outputs at this point are MKL Tensors
    output_mkl_shape.SetMklTensor(true);
    output_mkl_shape.SetMklLayout(&dst_pd);
    output_mkl_shape.SetElemType(MklDnnType<T>());
    output_mkl_shape.SetTfLayout(output_dims_mkl_order.size(),
                                 output_dims_mkl_order, output_tf_format);

    TensorShape output_tf_shape;
    size_t num_bytes = dst_pd.get_size();
    CHECK_EQ(num_bytes % sizeof(T), 0);
    output_tf_shape.AddDim(num_bytes / sizeof(T));
    AllocateOutputSetMklShape(context, kIdxOutput, output_tensor,
                              output_tf_shape, output_mkl_shape);
  }

  memory::desc ConfigureInputGradient(const Tensor& input_grad_tensor,
                                      const MklDnnShape& input_grad_dnn_shape,
                                      MklDnnData<T>* input_grad_dnn_data) {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSmklPSmkl_lrn_opDTcc mht_11(mht_11_v, 681, "", "./tensorflow/core/kernels/mkl/mkl_lrn_op.cc", "ConfigureInputGradient");

    DCHECK(input_grad_dnn_data != nullptr);
    // This shouldn't be necessary at this point, but just in case
    DCHECK(input_grad_dnn_shape.IsMklTensor() == true);

    memory::desc input_grad_md = input_grad_dnn_shape.GetCurLayout();
    memory::dims orig_input_dims = input_grad_dnn_shape.GetSizesAsMklDnnDims();
    input_grad_dnn_data->SetUsrMem(input_grad_md, &input_grad_tensor);
    input_grad_dnn_data->SetOpMemDesc(orig_input_dims,
                                      memory::format_tag::nhwc);
    return input_grad_md;
  }

  void ConfigureWorkspace(const Tensor& workspace_tensor,
                          memory::desc workspace_pd,
                          MklDnnData<uint8>* workspace_dnn_data) {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSmklPSmkl_lrn_opDTcc mht_12(mht_12_v, 699, "", "./tensorflow/core/kernels/mkl/mkl_lrn_op.cc", "ConfigureWorkspace");

    DCHECK(workspace_dnn_data);

    workspace_dnn_data->SetUsrMem(workspace_pd, &workspace_tensor);
  }

  // Fallback implementation - Taken from lrn_op.cc
  // TODO(intel-tf) Check if we can use EigenLRNOp directly
  // instead of making a copy.
  void MklDefaultToEigen(OpKernelContext* context) {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSmklPSmkl_lrn_opDTcc mht_13(mht_13_v, 711, "", "./tensorflow/core/kernels/mkl/mkl_lrn_op.cc", "MklDefaultToEigen");

    Tensor input_gradient_tensor;
    Tensor orig_input_tensor;
    Tensor orig_output_tensor;

    MklDnnShape input_grad_dnn_shape, orig_input_dnn_shape,
        orig_output_dnn_shape;
    GetMklShape(context, kIdxGradient, &input_grad_dnn_shape);
    GetMklShape(context, kIdxOrigInput, &orig_input_dnn_shape);
    GetMklShape(context, kIdxOrigOutput, &orig_output_dnn_shape);

    if (input_grad_dnn_shape.IsMklTensor()) {
      OP_REQUIRES_OK(
          context,
          ConvertMklToTF<T>(context, MklGetInput(context, kIdxGradient),
                            input_grad_dnn_shape, &input_gradient_tensor));
    } else {
      input_gradient_tensor = MklGetInput(context, kIdxGradient);
    }

    if (orig_input_dnn_shape.IsMklTensor()) {
      OP_REQUIRES_OK(context, ConvertMklToTF<T>(
                                  context, MklGetInput(context, kIdxOrigInput),
                                  orig_input_dnn_shape, &orig_input_tensor));
    } else {
      orig_input_tensor = MklGetInput(context, kIdxOrigInput);
    }

    if (orig_output_dnn_shape.IsMklTensor()) {
      OP_REQUIRES_OK(context, ConvertMklToTF<T>(
                                  context, MklGetInput(context, kIdxOrigOutput),
                                  orig_output_dnn_shape, &orig_output_tensor));
    } else {
      orig_output_tensor = MklGetInput(context, kIdxOrigOutput);
    }

    const int64 batch = static_cast<int64_t>(input_gradient_tensor.dim_size(0));
    const int64 rows = static_cast<int64_t>(input_gradient_tensor.dim_size(1));
    const int64 cols = static_cast<int64_t>(input_gradient_tensor.dim_size(2));
    const int64 depth = static_cast<int64_t>(input_gradient_tensor.dim_size(3));
    const auto nodes = cols * rows;

    auto grads_shaped =
        input_gradient_tensor.shaped<T, 2>({nodes * batch, depth});

    auto in_shaped = orig_input_tensor.shaped<T, 2>({nodes * batch, depth});
    auto activations = orig_output_tensor.shaped<T, 2>({nodes * batch, depth});

    Tensor* output_dnn_data;
    MklDnnShape mkl_output_mkl_shape;
    mkl_output_mkl_shape.SetMklTensor(false);
    mkl_output_mkl_shape.SetDimensions(4);
    AllocateOutputSetMklShape(context, kIdxOutput, &output_dnn_data,
                              input_gradient_tensor.shape(),
                              mkl_output_mkl_shape);

    auto out_shaped = output_dnn_data->shaped<T, 2>({nodes * batch, depth});
    out_shaped.setZero();
    auto shard = [this, activations, in_shaped, grads_shaped, out_shaped,
                  depth](int64 begin, int64 end) {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSmklPSmkl_lrn_opDTcc mht_14(mht_14_v, 773, "", "./tensorflow/core/kernels/mkl/mkl_lrn_op.cc", "lambda");

      for (int64 i = begin; i < end; ++i) {
        for (int64 j = 0; j < depth; ++j) {
          int64 depth_begin = std::max<int64_t>(0, j - depth_radius_);
          int64 depth_end = std::min<int64_t>(depth, j + depth_radius_ + 1);

          T norm(0);
          for (int64 k = depth_begin; k < depth_end; ++k) {
            norm += in_shaped(i, k) * in_shaped(i, k);
          }
          norm = alpha_ * norm + bias_;
          DCHECK_GT(norm, T(1e-6));
          for (int64 k = depth_begin; k < depth_end; ++k) {
            T dyi = T(-2) * alpha_ * beta_ * in_shaped(i, k) *
                    activations(i, j) / norm;
            if (k == j) {
              dyi += Eigen::numext::pow(norm, -beta_);
            }
            dyi *= grads_shaped(i, j);
            const_cast<typename TTypes<T, 2>::Tensor&>(out_shaped)(i, k) += dyi;
          }
        }
      }
    };
    auto worker_threads = *(context->device()->tensorflow_cpu_worker_threads());
    Shard(worker_threads.num_threads, worker_threads.workers, nodes * batch,
          depth * depth, shard);
  }

  void SanityCheckInputs(OpKernelContext* context) {
   std::vector<std::string> mht_15_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSmklPSmkl_lrn_opDTcc mht_15(mht_15_v, 805, "", "./tensorflow/core/kernels/mkl/mkl_lrn_op.cc", "SanityCheckInputs");

    const Tensor& input_gradient_tensor = MklGetInput(context, kIdxGradient);
    const Tensor& orig_input_tensor = MklGetInput(context, kIdxOrigInput);
    const Tensor& orig_output_tensor = MklGetInput(context, kIdxOrigOutput);
    const Tensor& workspace_tensor = MklGetInput(context, kIdxWorkspace);
    MklDnnShape in_grads_dnn_shape, in_image_dnn_shape, out_image_dnn_shape,
        workspace_dnn_shape;
    GetMklShape(context, kIdxGradient, &in_grads_dnn_shape);
    GetMklShape(context, kIdxOrigInput, &in_image_dnn_shape);
    GetMklShape(context, kIdxOrigOutput, &out_image_dnn_shape);
    GetMklShape(context, kIdxWorkspace, &workspace_dnn_shape);
    if (in_grads_dnn_shape.IsMklTensor()) {
      OP_REQUIRES(context, in_grads_dnn_shape.GetDimension() == 4,
                  errors::InvalidArgument("Input gradient must be "
                                          "4-dimensional"));
    } else {
      OP_REQUIRES(
          context, input_gradient_tensor.dims() == 4,
          errors::InvalidArgument("input gradient must be 4-dimensional"));
    }

    if (in_image_dnn_shape.IsMklTensor()) {
      OP_REQUIRES(context, in_image_dnn_shape.GetDimension() == 4,
                  errors::InvalidArgument("input images must be "
                                          "4-dimensional"));
    } else {
      OP_REQUIRES(context, orig_input_tensor.dims() == 4,
                  errors::InvalidArgument("input images must be "
                                          "4-dimensional"));
    }

    if (out_image_dnn_shape.IsMklTensor()) {
      OP_REQUIRES(context, out_image_dnn_shape.GetDimension() == 4,
                  errors::InvalidArgument("Output image must be "
                                          "4-dimensional"));
    } else {
      OP_REQUIRES(
          context, orig_output_tensor.dims() == 4,
          errors::InvalidArgument("Output image must be 4-dimensional"));
    }

    if (workspace_enabled_) {
      if (workspace_dnn_shape.IsMklTensor()) {
        OP_REQUIRES(
            context, workspace_dnn_shape.IsMklTensor() == false,
            errors::InvalidArgument("Workspace should not be MKL Tensor."));
      } else {
        OP_REQUIRES(context, workspace_tensor.dims() == 1,
                    errors::InvalidArgument("Workspace must be 1-dimensional"));
      }
    }
  }

  // Input("input_grads: T")
  // Input("input_image: T")
  // Input("output_image: T")
  // Input("workspace: uint8")
  const int kIdxGradient = 0, kIdxOrigInput = 1, kIdxOrigOutput = 2,
            kIdxWorkspace = 3, kIdxOutput = 0;

  typedef typename Eigen::Tensor<T, 1, Eigen::RowMajor>::DimensionPair DimPair;
  bool workspace_enabled_;
  int depth_radius_;
  float bias_;
  float alpha_;
  float beta_;
  engine cpu_engine_;
  std::shared_ptr<stream> bwd_stream_;
};

#define REGISTER_MKL_LRN_CPU(T)                                \
  REGISTER_KERNEL_BUILDER(                                     \
      Name("_MklLRN")                                          \
          .Device(DEVICE_CPU)                                  \
          .TypeConstraint<T>("T")                              \
          .Label(mkl_op_registry::kMklLayoutDependentOpLabel), \
      MklLRNOp<T>);                                            \
  REGISTER_KERNEL_BUILDER(                                     \
      Name("_MklLRNGrad")                                      \
          .Device(DEVICE_CPU)                                  \
          .TypeConstraint<T>("T")                              \
          .Label(mkl_op_registry::kMklLayoutDependentOpLabel), \
      MklLRNGradOp<T>);

TF_CALL_float(REGISTER_MKL_LRN_CPU);

}  // namespace tensorflow

#endif  // INTEL_MKL
