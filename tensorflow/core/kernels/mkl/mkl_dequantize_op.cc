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
class MHTracer_DTPStensorflowPScorePSkernelsPSmklPSmkl_dequantize_opDTcc {
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
   MHTracer_DTPStensorflowPScorePSkernelsPSmklPSmkl_dequantize_opDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSkernelsPSmklPSmkl_dequantize_opDTcc() {
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

#ifdef INTEL_MKL

#define EIGEN_USE_THREADS

#include "dnnl.hpp"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/type_traits.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/graph/mkl_graph_util.h"
#include "tensorflow/core/kernels/meta_support.h"
#include "tensorflow/core/kernels/quantization_utils.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/util/mkl_util.h"

using dnnl::primitive_attr;
using dnnl::stream;

namespace tensorflow {

typedef Eigen::ThreadPoolDevice CPUDevice;

template <typename Device, typename T, bool native_format = false>
class MklDequantizeOp : public OpKernel {
 public:
  explicit MklDequantizeOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSmklPSmkl_dequantize_opDTcc mht_0(mht_0_v, 210, "", "./tensorflow/core/kernels/mkl/mkl_dequantize_op.cc", "MklDequantizeOp");

    string mode_string;
    OP_REQUIRES_OK(ctx, ctx->GetAttr("mode", &mode_string));
    OP_REQUIRES(ctx, mode_string == "SCALED",
                errors::InvalidArgument(
                    "MklDequantizeOp only supports 'SCALED' mode, but got '" +
                    mode_string + "'"));
  }

  void Compute(OpKernelContext* ctx) override {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSmklPSmkl_dequantize_opDTcc mht_1(mht_1_v, 222, "", "./tensorflow/core/kernels/mkl/mkl_dequantize_op.cc", "Compute");

    try {
      // Using CPU device
      auto cpu_engine = engine(engine::kind::cpu, 0);

      // Get the inputs
      const Tensor& src_tensor = MklGetInput(ctx, kSrcIndex);
      const float min_range =
          MklGetInput(ctx, kMinIndex).template flat<float>()(0);
      const float max_range =
          MklGetInput(ctx, kMaxIndex).template flat<float>()(0);

      // Get MklShape
      MklDnnShape src_mkl_shape;
      GetMklShape(ctx, kSrcIndex, &src_mkl_shape, native_format);

      // src_dims is the dimension of src_tensor
      // output_dims are same as src_dims
      auto src_dims = src_mkl_shape.IsMklTensor()
                          ? src_mkl_shape.GetSizesAsMklDnnDims()
                          : TFShapeToMklDnnDims(src_tensor.shape());
      auto output_dims = src_dims;

      // Create reorder memory for src and dst
      MklDnnData<T> src(&cpu_engine);
      MklDnnData<float> dst(&cpu_engine);

      std::shared_ptr<stream> reorder_stream;
      MklDnnThreadPool eigen_tp(ctx);
      reorder_stream.reset(CreateStream(&eigen_tp, cpu_engine));

      // If input is in MKL layout, then simply grab input layout; otherwise,
      // construct input TF layout. For TF layout, although input shape
      // (src_dims) required is in MKL-DNN order, the layout is Tensorflow's
      // layout
      auto src_md =
          src_mkl_shape.IsMklTensor()
              ? src_mkl_shape.GetMklLayout()
              : memory::desc(src_dims, MklDnnType<T>(),
                             src_dims.size() == 4 ? memory::format_tag::nhwc
                                                  : memory::format_tag::nc);

      src.SetUsrMem(src_md, &src_tensor);
      src.SetUsrMemDataHandle(&src_tensor, reorder_stream);

      Tensor* output_tensor = nullptr;
      MklDnnShape output_mkl_shape;
      TensorShape output_tf_shape;
      memory::desc dst_md = memory::desc();
      if (src_mkl_shape.IsMklTensor()) {
        dst_md = memory::desc(src_mkl_shape.GetMklLayout().data);
        // There is no API in MKL-DNN v1.x to construct memory descriptor with
        // same .data field but different type.
        dst_md.data.data_type = memory::convert_to_c(MklDnnType<float>());
      } else {
        dst_md = memory::desc(src_dims, MklDnnType<float>(),
                              src_dims.size() == 4 ? memory::format_tag::nhwc
                                                   : memory::format_tag::nc);
      }

      // If input is MKL shape, output is also MKL shape.
      // If input is TF shape, output is also TF shape.
      if (src_mkl_shape.IsMklTensor()) {
        output_mkl_shape.SetMklTensor(true);
        output_mkl_shape.SetMklLayout(&dst_md);
        output_mkl_shape.SetElemType(MklDnnType<float>());
        output_mkl_shape.SetTfLayout(src_mkl_shape.GetDimension(),
                                     src_mkl_shape.GetSizesAsMklDnnDims(),
                                     src_mkl_shape.GetTfDataFormat());
        output_tf_shape.AddDim(dst_md.get_size() / sizeof(float));
      } else {
        output_mkl_shape.SetMklTensor(false);
        output_tf_shape = MklDnnDimsToTFShape(output_dims);
      }

      // Allocate MKL or TF output shape based on the above
      AllocateOutputSetMklShape(ctx, 0, &output_tensor, output_tf_shape,
                                output_mkl_shape, native_format);
      dst.SetUsrMem(dst_md, output_tensor);
      dst.SetUsrMemDataHandle(output_tensor, reorder_stream);

      // The quantization logic here for mode SCALED is similar to the logic
      // in QuantizeAndDequantizeV2 and QuantizeAndDequantizeV3.
      static constexpr int num_bits = sizeof(T) * 8;
      const float max_abs = std::max(std::abs(min_range), std::abs(max_range));
      bool is_signed = std::is_signed<T>::value;
      // If it is signed, we try to keep 0.0 being 0 and drop one bucket. For
      // example, if it is 8 bits, we have the range [-127, 127]. So for input
      // range of [-x, x], the scale should be (2*x)/254.
      //
      // If it is unsigned and num_bits == 8, the range with 8 bits is [0, 255].
      // If the input range is [0, x], then the scale is x/255 instead of 254 as
      // in the case above.
      const int target_bits = is_signed ? (num_bits - 1) : num_bits;
      const float target_range =
          static_cast<float>((uint64_t{1} << target_bits) - 1);
      const float scale_factor = max_abs / target_range;
      std::vector<float> scales;
      scales.push_back(scale_factor);
      primitive_attr attr;
      attr.set_output_scales(0, scales);
      std::vector<primitive> net;

      // Create reorder primitive and then execute.
      auto reorder_pd =
          ReorderPd(cpu_engine, src.GetUsrMem()->get_desc(), cpu_engine,
                    dst.GetUsrMem()->get_desc(), attr);
      net.push_back(reorder(reorder_pd));
      std::vector<std::unordered_map<int, memory>> reorder_net_args;
      reorder_net_args.push_back(
          {{DNNL_ARG_FROM, *src.GetUsrMem()}, {DNNL_ARG_TO, *dst.GetUsrMem()}});
      execute_primitives(net, reorder_stream, reorder_net_args);
    } catch (dnnl::error& e) {
      string error_msg = "Status: " + std::to_string(e.status) +
                         ", message: " + string(e.message) + ", in file " +
                         string(__FILE__) + ":" + std::to_string(__LINE__);
      OP_REQUIRES_OK(
          ctx, errors::Aborted("Operation received an exception:", error_msg));
    }
  }

 private:
  const size_t kSrcIndex = 0;
  const size_t kMinIndex = 1;
  const size_t kMaxIndex = 2;
};

REGISTER_KERNEL_BUILDER(Name("_MklDequantize")
                            .Device(DEVICE_CPU)
                            .TypeConstraint<quint8>("T")
                            .Label(mkl_op_registry::kMklQuantizedOpLabel),
                        MklDequantizeOp<CPUDevice, quint8, true>);
REGISTER_KERNEL_BUILDER(Name("_MklDequantize")
                            .Device(DEVICE_CPU)
                            .TypeConstraint<qint8>("T")
                            .Label(mkl_op_registry::kMklQuantizedOpLabel),
                        MklDequantizeOp<CPUDevice, qint8, true>);

}  // namespace tensorflow

#endif  // INTEL_MKL
