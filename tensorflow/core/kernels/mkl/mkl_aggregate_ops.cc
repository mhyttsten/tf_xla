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
class MHTracer_DTPStensorflowPScorePSkernelsPSmklPSmkl_aggregate_opsDTcc {
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
   MHTracer_DTPStensorflowPScorePSkernelsPSmklPSmkl_aggregate_opsDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSkernelsPSmklPSmkl_aggregate_opsDTcc() {
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

// See docs in ../ops/math_ops.cc.

#ifdef INTEL_MKL
#define EIGEN_USE_THREADS

#include <numeric>

#include "dnnl.hpp"
#include "tensorflow/core/framework/numeric_op.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/lib/gtl/inlined_vector.h"
#include "tensorflow/core/util/mkl_util.h"

using dnnl::stream;
using dnnl::sum;

namespace tensorflow {
typedef Eigen::ThreadPoolDevice CPUDevice;

template <typename Device, typename T>
class MklAddNOp : public OpKernel {
 public:
  ~MklAddNOp() {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSmklPSmkl_aggregate_opsDTcc mht_0(mht_0_v, 207, "", "./tensorflow/core/kernels/mkl/mkl_aggregate_ops.cc", "~MklAddNOp");
}
  explicit MklAddNOp(OpKernelConstruction* context) : OpKernel(context) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSmklPSmkl_aggregate_opsDTcc mht_1(mht_1_v, 211, "", "./tensorflow/core/kernels/mkl/mkl_aggregate_ops.cc", "MklAddNOp");
}

  TensorShape GetTensorShape(OpKernelContext* ctx, size_t src_index) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSmklPSmkl_aggregate_opsDTcc mht_2(mht_2_v, 216, "", "./tensorflow/core/kernels/mkl/mkl_aggregate_ops.cc", "GetTensorShape");

    const Tensor& src_tensor = MklGetInput(ctx, src_index);
    MklDnnShape src_mkl_shape;
    GetMklShape(ctx, src_index, &src_mkl_shape);
    return src_mkl_shape.IsMklTensor() ? src_mkl_shape.GetTfShape()
                                       : src_tensor.shape();
  }

  bool CheckInputShape(OpKernelContext* ctx) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSmklPSmkl_aggregate_opsDTcc mht_3(mht_3_v, 227, "", "./tensorflow/core/kernels/mkl/mkl_aggregate_ops.cc", "CheckInputShape");

    const int num_inputs = ctx->num_inputs() / 2;
    const TensorShape src0_shape = GetTensorShape(ctx, 0);

    for (size_t i = 1; i < num_inputs; ++i) {
      if (!src0_shape.IsSameSize(GetTensorShape(ctx, i))) {
        ctx->SetStatus(errors::InvalidArgument(
            "Inputs to operation ", this->name(), " of type ",
            this->type_string(),
            " must have the same size and shape.  Input 0: ",
            src0_shape.DebugString(), " != input : ", i,
            GetTensorShape(ctx, i).DebugString()));

        return false;
      }
    }

    return true;
  }

  // Return first tensor index which is in MKL layout, or -1 with no MKL input.
  int FindMKLInputIndex(OpKernelContext* ctx) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSmklPSmkl_aggregate_opsDTcc mht_4(mht_4_v, 251, "", "./tensorflow/core/kernels/mkl/mkl_aggregate_ops.cc", "FindMKLInputIndex");

    int mkl_index = -1;
    const int num_inputs = ctx->num_inputs() / 2;

    MklDnnShape src_mkl_shape;
    for (size_t i = 0; i < num_inputs; ++i) {
      GetMklShape(ctx, i, &src_mkl_shape);
      if (src_mkl_shape.IsMklTensor()) {
        mkl_index = i;
        break;
      }
    }

    return mkl_index;
  }

  void ComputeScalar(OpKernelContext* ctx) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSmklPSmkl_aggregate_opsDTcc mht_5(mht_5_v, 270, "", "./tensorflow/core/kernels/mkl/mkl_aggregate_ops.cc", "ComputeScalar");

    const int num_inputs = ctx->num_inputs() / 2;
    const size_t kOutputIdx = 0;
    TensorShape output_tf_shape;
    MklDnnShape output_mkl_shape;
    Tensor* dst_tensor = nullptr;

    T sum = static_cast<T>(0);
    for (int src_idx = 0; src_idx < num_inputs; ++src_idx) {
      const Tensor& src_tensor = MklGetInput(ctx, src_idx);
      T* src_i = const_cast<T*>(src_tensor.flat<T>().data());
      sum += src_i[0];
    }

    output_mkl_shape.SetMklTensor(false);
    output_tf_shape = MklGetInput(ctx, kOutputIdx).shape();
    AllocateOutputSetMklShape(ctx, kOutputIdx, &dst_tensor, output_tf_shape,
                              output_mkl_shape);

    T* out_o = dst_tensor->flat<T>().data();
    out_o[0] = sum;
  }

  void Compute(OpKernelContext* ctx) override {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSmklPSmkl_aggregate_opsDTcc mht_6(mht_6_v, 296, "", "./tensorflow/core/kernels/mkl/mkl_aggregate_ops.cc", "Compute");

    // Each input tensor in MKL layout has additional meta-tensor carrying
    // layout information. So the number of actual tensors is half the total
    // number of inputs.
    const int num_inputs = ctx->num_inputs() / 2;

    MklDnnShape mkl_shape;
    const size_t kSrc0Idx = 0;
    const size_t kOutputIdx = 0;

    if (num_inputs == 1) {
      GetMklShape(ctx, kSrc0Idx, &mkl_shape);
      bool input_in_mkl_format = mkl_shape.IsMklTensor();

      if (input_in_mkl_format) {
        ForwardMklTensorInToOut(ctx, kSrc0Idx, kOutputIdx);
      } else {
        ForwardTfTensorInToOut(ctx, kSrc0Idx, kOutputIdx);
      }
      return;
    }

    // Check if the input shape is same
    if (!CheckInputShape(ctx)) return;

    try {
      TensorShape output_tf_shape;
      MklDnnShape output_mkl_shape;
      const Tensor& src_tensor = MklGetInput(ctx, kSrc0Idx);

      Tensor* dst_tensor = nullptr;

      // Nothing to compute, return.
      if (src_tensor.shape().num_elements() == 0) {
        output_mkl_shape.SetMklTensor(false);
        output_tf_shape = src_tensor.shape();
        AllocateOutputSetMklShape(ctx, kOutputIdx, &dst_tensor, output_tf_shape,
                                  output_mkl_shape);
        return;
      }

      if (src_tensor.dims() == 0) {
        ComputeScalar(ctx);
        return;
      }

      auto cpu_engine = engine(engine::kind::cpu, 0);
      std::vector<float> coeff(num_inputs, 1.0);
      std::vector<memory::desc> srcs_pd;
      std::vector<memory> inputs;

      MklDnnData<T> dst(&cpu_engine);
      MklDnnData<T> src(&cpu_engine);
      bool has_mkl_input = false;
      int mkl_input_index = FindMKLInputIndex(ctx);
      MklTensorFormat mkl_data_format;
      TensorFormat tf_data_format;
      memory::format_tag dnn_fmt = memory::format_tag::any;
      if (mkl_input_index >= 0) {
        has_mkl_input = true;
        GetMklShape(ctx, mkl_input_index, &mkl_shape);
        // MKL input has the data format information.
        mkl_data_format = mkl_shape.GetTfDataFormat();
        tf_data_format = MklDnnDataFormatToTFDataFormat(mkl_data_format);
        dnn_fmt = MklTensorFormatToMklDnnDataFormat(mkl_data_format);
      }

      std::shared_ptr<stream> fwd_cpu_stream;
      MklDnnThreadPool eigen_tp(ctx);
      fwd_cpu_stream.reset(CreateStream(&eigen_tp, cpu_engine));

      // Create memory descriptor for MKL-DNN.
      // If all input in Tensorflow format, create block memory descriptor,
      // else convert TF format to MKL memory descriptor
      for (int src_idx = 0; src_idx < num_inputs; ++src_idx) {
        MklDnnShape src_mkl_shape;
        GetMklShape(ctx, src_idx, &src_mkl_shape);
        memory::desc md({}, memory::data_type::undef,
                        memory::format_tag::undef);
        const Tensor& src_tensor = MklGetInput(ctx, src_idx);

        if (src_mkl_shape.IsMklTensor()) {
          md = src_mkl_shape.GetMklLayout();
        } else {
          if (has_mkl_input) {
            memory::dims src_dims;
            if (src_tensor.dims() == 4) {
              src_dims =
                  TFShapeToMklDnnDimsInNCHW(src_tensor.shape(), tf_data_format);
            } else {
              DCHECK(src_tensor.dims() == 5);
              src_dims = TFShapeToMklDnnDimsInNCDHW(src_tensor.shape(),
                                                    tf_data_format);
            }
            md = memory::desc(src_dims, MklDnnType<T>(), dnn_fmt);
          } else {
            // Create block memory descriptor for TensorFlow format input.
            auto dims = TFShapeToMklDnnDims(src_tensor.shape());
            auto strides = CalculateTFStrides(dims);
            md = MklDnnData<T>::CreateBlockedMemDesc(dims, strides);
          }
        }
        srcs_pd.push_back(memory::desc(md));
        src.SetUsrMem(md, &src_tensor);
        src.SetUsrMemDataHandle(&src_tensor, fwd_cpu_stream);
        inputs.push_back(src.GetOpMem());
      }

      auto sum_pd = sum::primitive_desc(coeff, srcs_pd, cpu_engine);
      output_mkl_shape.SetMklTensor(has_mkl_input);
      auto output_pd = sum_pd.dst_desc();
      dst.SetUsrMem(output_pd);

      if (has_mkl_input) {
        output_mkl_shape.SetMklLayout(&output_pd);
        output_mkl_shape.SetElemType(MklDnnType<T>());
        output_mkl_shape.SetTfLayout(mkl_shape.GetDimension(),
                                     mkl_shape.GetSizesAsMklDnnDims(),
                                     mkl_shape.GetTfDataFormat());
        output_tf_shape.AddDim((output_pd.get_size() / sizeof(T)));
      } else {
        // All inputs have TF shapes, get the shape from first one.
        output_tf_shape = MklGetInput(ctx, kSrc0Idx).shape();
      }
      AllocateOutputSetMklShape(ctx, kOutputIdx, &dst_tensor, output_tf_shape,
                                output_mkl_shape);
      dst.SetUsrMemDataHandle(dst_tensor, fwd_cpu_stream);

      // Create Sum op, and submit net for execution.
      std::vector<primitive> net;
      dnnl::sum sum_op(sum_pd);
      std::unordered_map<int, memory> net_args = {
          {DNNL_ARG_DST, dst.GetOpMem()}};
      for (int i = 0; i < num_inputs; ++i) {
        net_args.insert({DNNL_ARG_MULTIPLE_SRC + i, inputs[i]});
      }
      sum_op.execute(*fwd_cpu_stream, net_args);
    } catch (dnnl::error& e) {
      string error_msg = "Status: " + std::to_string(e.status) +
                         ", message: " + string(e.message) + ", in file " +
                         string(__FILE__) + ":" + std::to_string(__LINE__);
      OP_REQUIRES_OK(
          ctx, errors::Aborted("Operation received an exception:", error_msg));
    }
  }
};

#define REGISTER_MKL_CPU(T)                                    \
  REGISTER_KERNEL_BUILDER(                                     \
      Name("_MklAddN")                                         \
          .Device(DEVICE_CPU)                                  \
          .TypeConstraint<T>("T")                              \
          .Label(mkl_op_registry::kMklLayoutDependentOpLabel), \
      MklAddNOp<CPUDevice, T>);

TF_CALL_float(REGISTER_MKL_CPU);
TF_CALL_bfloat16(REGISTER_MKL_CPU);
#undef REGISTER_MKL_CPU
}  // namespace tensorflow
#endif  // INTEL_MKL
