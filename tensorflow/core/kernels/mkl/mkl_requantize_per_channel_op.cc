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
class MHTracer_DTPStensorflowPScorePSkernelsPSmklPSmkl_requantize_per_channel_opDTcc {
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
   MHTracer_DTPStensorflowPScorePSkernelsPSmklPSmkl_requantize_per_channel_opDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSkernelsPSmklPSmkl_requantize_per_channel_opDTcc() {
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

/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

// See docs in ../ops/array_ops.cc.

#ifdef INTEL_MKL
#define EIGEN_USE_THREADS

#include <math.h>

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "dnnl.hpp"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/type_traits.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/kernels/meta_support.h"
#include "tensorflow/core/kernels/no_op.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/util/mkl_util.h"

namespace tensorflow {

typedef Eigen::ThreadPoolDevice CPUDevice;

template <typename Device, typename Toutput>
class MklRequantizePerChannelOp : public OpKernel {
 public:
  explicit MklRequantizePerChannelOp(OpKernelConstruction* ctx)
      : OpKernel(ctx) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSmklPSmkl_requantize_per_channel_opDTcc mht_0(mht_0_v, 211, "", "./tensorflow/core/kernels/mkl/mkl_requantize_per_channel_op.cc", "MklRequantizePerChannelOp");

    OP_REQUIRES_OK(ctx, ctx->GetAttr("out_type", &out_type_));
    OP_REQUIRES(ctx, out_type_ == DT_QINT8 || out_type_ == DT_QUINT8,
                errors::InvalidArgument(
                    "out_type must be qint8 or quint8, but got: ", out_type_));
  }
  virtual ~MklRequantizePerChannelOp() {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSmklPSmkl_requantize_per_channel_opDTcc mht_1(mht_1_v, 220, "", "./tensorflow/core/kernels/mkl/mkl_requantize_per_channel_op.cc", "~MklRequantizePerChannelOp");
}
  void Compute(OpKernelContext* ctx) override {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSmklPSmkl_requantize_per_channel_opDTcc mht_2(mht_2_v, 224, "", "./tensorflow/core/kernels/mkl/mkl_requantize_per_channel_op.cc", "Compute");

    try {
      const Tensor& input = ctx->input(kInputTensorIndex);
      OP_REQUIRES(
          ctx, input.dims() == 4,
          errors::InvalidArgument("Current RequantizePerChannel operator"
                                  "supports 4D tensors only."));

      const Tensor& input_min_vec = ctx->input(kInputMinVecIndex);
      size_t depth = input_min_vec.NumElements();
      float* input_min_vec_data = (float*)const_cast<void*>(
          static_cast<const void*>(input_min_vec.flat<float>().data()));

      const Tensor& input_max_vec = ctx->input(kInputMaxVecIndex);
      OP_REQUIRES(
          ctx, input_max_vec.NumElements() == depth,
          errors::InvalidArgument("input_max has incorrect size, expected ",
                                  depth, " was ", input_max_vec.NumElements()));
      float* input_max_vec_data = (float*)const_cast<void*>(
          static_cast<const void*>(input_max_vec.flat<float>().data()));

      const Tensor& input_requested_min = ctx->input(this->kRequestMinIndex);
      OP_REQUIRES(
          ctx, input_requested_min.NumElements() == 1,
          errors::InvalidArgument("requested_output_min must be a scalar"));
      const float input_requested_min_float =
          input_requested_min.flat<float>()(0);

      const Tensor& input_requested_max = ctx->input(this->kRequestMaxIndex);
      OP_REQUIRES(
          ctx, input_requested_min.NumElements() == 1,
          errors::InvalidArgument("requested_output_max must be a scalar"));
      const float input_requested_max_float =
          input_requested_max.flat<float>()(0);

      if (out_type_ == DT_QINT8) {
        OP_REQUIRES(ctx, input_requested_min_float < 0.0f,
                    errors::InvalidArgument(
                        "If out_type is QINT8, requested_output_max must be "
                        "non negative, got ",
                        input_requested_min_float));
      }

      const float factor = (out_type_ == DT_QINT8) ? 127.0f : 255.0f;
      const float requested_min_max =
          std::max(std::abs(input_requested_min_float),
                   std::abs(input_requested_max_float));
      Tensor* output = nullptr;
      OP_REQUIRES_OK(ctx, ctx->allocate_output(kOutputTensorIndex,
                                               input.shape(), &output));

      std::vector<float> scales(depth);
      for (int i = 0; i < depth; ++i) {
        float min_max_from_vec = std::max(std::abs(input_min_vec_data[i]),
                                          std::abs(input_max_vec_data[i]));
        scales[i] = factor * (min_max_from_vec / requested_min_max /
                              static_cast<float>(1L << 31));
      }

      dnnl::primitive_attr reorder_attr;
      reorder_attr.set_output_scales(2, scales);

      memory::dims dims_mkl_order =
          TFShapeToMklDnnDimsInNCHW(input.shape(), FORMAT_NHWC);
      memory::desc input_md = memory::desc(dims_mkl_order, MklDnnType<qint32>(),
                                           memory::format_tag::nhwc);
      memory::desc output_md =
          (out_type_ == DT_QINT8)
              ? memory::desc(dims_mkl_order, MklDnnType<qint8>(),
                             memory::format_tag::nhwc)
              : memory::desc(dims_mkl_order, MklDnnType<quint8>(),
                             memory::format_tag::nhwc);

      void* input_buf =
          static_cast<void*>(const_cast<qint32*>(input.flat<qint32>().data()));
      void* output_buf;
      if (out_type_ == DT_QINT8) {
        output_buf = static_cast<void*>(
            const_cast<qint8*>(output->flat<qint8>().data()));
      } else {
        output_buf = static_cast<void*>(
            const_cast<quint8*>(output->flat<quint8>().data()));
      }

      std::unique_ptr<memory> input_mem_prim(
          new memory(input_md, cpu_engine_, input_buf));
      std::unique_ptr<memory> output_mem_prim(
          new memory(output_md, cpu_engine_, output_buf));

      dnnl::reorder::primitive_desc reorder_pd =
          ReorderPd(cpu_engine_, input_mem_prim->get_desc(), cpu_engine_,
                    output_mem_prim->get_desc(), reorder_attr);
      std::shared_ptr<stream> reorder_stream;
      MklDnnThreadPool eigen_tp(ctx);
      reorder_stream.reset(CreateStream(&eigen_tp, cpu_engine_));
      std::unordered_map<int, dnnl::memory> reorder_args = {
          {DNNL_ARG_FROM, *input_mem_prim}, {DNNL_ARG_TO, *output_mem_prim}};
      std::unique_ptr<dnnl::primitive> reorder_prim(
          new dnnl::reorder(reorder_pd));
      reorder_prim->execute(*reorder_stream, reorder_args);

      Tensor* output_min = nullptr;
      Tensor* output_max = nullptr;
      OP_REQUIRES_OK(ctx,
                     ctx->allocate_output(kOutputMinIndex, {}, &output_min));
      OP_REQUIRES_OK(ctx,
                     ctx->allocate_output(kOutputMaxIndex, {}, &output_max));

      output_min->flat<float>()(0) = input_requested_min_float;
      output_max->flat<float>()(0) = input_requested_max_float;
    } catch (dnnl::error& e) {
      string error_msg = "Status: " + std::to_string(e.status) +
                         ", message: " + std::string(e.message) + ", in file " +
                         std::string(__FILE__) + ":" + std::to_string(__LINE__);
      OP_REQUIRES_OK(
          ctx, errors::Aborted("Operation received an exception:", error_msg));
    }
  }

 private:
  const int kInputTensorIndex = 0;
  const int kInputMinVecIndex = 1;
  const int kInputMaxVecIndex = 2;
  const int kRequestMinIndex = 3;
  const int kRequestMaxIndex = 4;
  const int kOutputTensorIndex = 0;
  const int kOutputMinIndex = 1;
  const int kOutputMaxIndex = 2;
  DataType out_type_;
  engine cpu_engine_ = engine(engine::kind::cpu, 0);
};

// Registration for out_type: qint8
REGISTER_KERNEL_BUILDER(Name("RequantizePerChannel")
                            .Device(DEVICE_CPU)
                            .TypeConstraint<qint32>("T")
                            .TypeConstraint<qint8>("out_type"),
                        MklRequantizePerChannelOp<CPUDevice, qint8>);
// Registration for out_type: quint8
REGISTER_KERNEL_BUILDER(Name("RequantizePerChannel")
                            .Device(DEVICE_CPU)
                            .TypeConstraint<qint32>("T")
                            .TypeConstraint<quint8>("out_type"),
                        MklRequantizePerChannelOp<CPUDevice, quint8>);

}  // namespace tensorflow
#endif  // INTEL_MKL
