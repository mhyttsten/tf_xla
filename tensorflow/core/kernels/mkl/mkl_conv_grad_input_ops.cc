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
class MHTracer_DTPStensorflowPScorePSkernelsPSmklPSmkl_conv_grad_input_opsDTcc {
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
   MHTracer_DTPStensorflowPScorePSkernelsPSmklPSmkl_conv_grad_input_opsDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSkernelsPSmklPSmkl_conv_grad_input_opsDTcc() {
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

// See docs in ../ops/nn_ops.cc. This opkernel uses MKL library, create MKL
// layout and primitives, use MKL dnn primitives to compute convolution backward
// input

#ifdef INTEL_MKL

#define USE_EIGEN_TENSOR
#define EIGEN_USE_THREADS

#include <algorithm>
#include <vector>

#include "tensorflow/core/framework/tensor_util.h"
#include "tensorflow/core/kernels/conv_grad_shape_utils.h"
#include "tensorflow/core/kernels/mkl/mkl_conv_ops.h"
#include "tensorflow/core/util/use_cudnn.h"
#include "tensorflow/core/util/work_sharder.h"
#ifdef DNNL_AARCH64_USE_ACL
#include "tensorflow/core/platform/mutex.h"
#endif

using dnnl::convolution_backward_data;
using dnnl::prop_kind;
using dnnl::stream;

namespace tensorflow {

using ConvBwdDataDesc = dnnl::convolution_backward_data::desc;
using ConvBwdDataPd = dnnl::convolution_backward_data::primitive_desc;

// Utility classes for enabling primitive reuse for conv bwd input.
struct MklConvBwdInputParams {
  memory::dims diff_src_dims;
  memory::dims filter_dims;
  memory::dims diff_dst_dims;
  memory::dims strides;
  MklTensorFormat tf_fmt;
  bool native_format;
  memory::dims dilations;
  memory::dims padding_left;
  memory::dims padding_right;

  MklConvBwdInputParams(memory::dims diff_src_dims, memory::dims filter_dims,
                        memory::dims diff_dst_dims, memory::dims strides,
                        MklTensorFormat tf_fmt, bool native_format,
                        memory::dims dilations, memory::dims padding_left,
                        memory::dims padding_right)
      : diff_src_dims(diff_src_dims),
        filter_dims(filter_dims),
        diff_dst_dims(diff_dst_dims),
        strides(strides),
        tf_fmt(tf_fmt),
        native_format(native_format),
        dilations(dilations),
        padding_left(padding_left),
        padding_right(padding_right) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSmklPSmkl_conv_grad_input_opsDTcc mht_0(mht_0_v, 240, "", "./tensorflow/core/kernels/mkl/mkl_conv_grad_input_ops.cc", "MklConvBwdInputParams");
}
};

template <typename T>
class MklConvBwdInputPrimitive : public MklPrimitive {
 public:
  explicit MklConvBwdInputPrimitive(
      const MklConvBwdInputParams& convBwdInputDims)
      : MklPrimitive(engine(engine::kind::cpu, 0)) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSmklPSmkl_conv_grad_input_opsDTcc mht_1(mht_1_v, 251, "", "./tensorflow/core/kernels/mkl/mkl_conv_grad_input_ops.cc", "MklConvBwdInputPrimitive");

    // Create conv bwd input primitive
    if (context_.conv_bwd_input == nullptr) {
      Setup(convBwdInputDims);
    }
  }

  ~MklConvBwdInputPrimitive() {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSmklPSmkl_conv_grad_input_opsDTcc mht_2(mht_2_v, 261, "", "./tensorflow/core/kernels/mkl/mkl_conv_grad_input_ops.cc", "~MklConvBwdInputPrimitive");
}

  // Convolution backward input (data) execution.
  //   diff_src_data: output data buffer for diff_src
  //   filter_data:   input data buffer for filter (weights)
  //   diff_dst_data: input data buffer for dst
  // Bias does not matter here
  void Execute(const T* diff_src_data, const T* filter_data,
               const T* diff_dst_data,
               std::shared_ptr<stream> bwd_input_stream) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSmklPSmkl_conv_grad_input_opsDTcc mht_3(mht_3_v, 273, "", "./tensorflow/core/kernels/mkl/mkl_conv_grad_input_ops.cc", "Execute");

#ifdef DNNL_AARCH64_USE_ACL
    mutex_lock lock(primitive_execution_mu_);
#endif
#ifndef ENABLE_ONEDNN_OPENMP
    // TODO(intel-tf): Create a common function and avoid the duplicate code
    context_.diff_src_mem->set_data_handle(
        static_cast<T*>(const_cast<T*>(diff_src_data)), *bwd_input_stream);
    context_.filter_mem->set_data_handle(
        static_cast<T*>(const_cast<T*>(filter_data)), *bwd_input_stream);
    context_.diff_dst_mem->set_data_handle(
        static_cast<T*>(const_cast<T*>(diff_dst_data)), *bwd_input_stream);
#else
    context_.diff_src_mem->set_data_handle(
        static_cast<T*>(const_cast<T*>(diff_src_data)));
    context_.filter_mem->set_data_handle(
        static_cast<T*>(const_cast<T*>(filter_data)));
    context_.diff_dst_mem->set_data_handle(
        static_cast<T*>(const_cast<T*>(diff_dst_data)));
#endif  // !ENABLE_ONEDNN_OPENMP
    execute_primitives(context_.bwd_input_primitives, bwd_input_stream,
                       context_.bwd_input_primitives_args);

    // Set data handle back to DummyData.
    context_.diff_src_mem->set_data_handle(DummyData);
    context_.filter_mem->set_data_handle(DummyData);
    context_.diff_dst_mem->set_data_handle(DummyData);
    return;
  }

  std::shared_ptr<ConvBwdDataPd> GetPrimitiveDesc() const {
    return context_.bwd_input_pd;
  }

 private:
  // Primitive reuse context for conv bwd input.
  struct ConvBwdInputContext {
    // MKL-DNN memory.
    std::shared_ptr<dnnl::memory> diff_src_mem;
    std::shared_ptr<dnnl::memory> filter_mem;
    std::shared_ptr<dnnl::memory> diff_dst_mem;

    // Conv backward input primitive descriptor and descriptor.
    std::shared_ptr<ConvBwdDataPd> bwd_input_pd;
    std::shared_ptr<ConvBwdDataDesc> bwd_input_desc;

    // Primitive descriptor and descriptor for conv fwd
    std::shared_ptr<ConvFwdPd> fwd_pd;
    std::shared_ptr<ConvFwdDesc> fwd_desc;

    // Conv bwd input primitive.
    std::shared_ptr<dnnl::primitive> conv_bwd_input;

    // Memory descriptors: forward & backward share the same descriptors.
    std::shared_ptr<memory::desc> diff_src_md;
    std::shared_ptr<memory::desc> filter_md;
    std::shared_ptr<memory::desc> diff_dst_md;

    // MKL-DNN pipeline for executing primitives.
    std::vector<dnnl::primitive> bwd_input_primitives;
    std::vector<std::unordered_map<int, memory>> bwd_input_primitives_args;

    ConvBwdInputContext()
        : diff_src_mem(nullptr),
          filter_mem(nullptr),
          diff_dst_mem(nullptr),
          bwd_input_pd(nullptr),
          bwd_input_desc(nullptr),
          fwd_pd(nullptr),
          fwd_desc(nullptr),
          conv_bwd_input(nullptr),
          diff_src_md(nullptr),
          filter_md(nullptr),
          diff_dst_md(nullptr) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSmklPSmkl_conv_grad_input_opsDTcc mht_4(mht_4_v, 349, "", "./tensorflow/core/kernels/mkl/mkl_conv_grad_input_ops.cc", "ConvBwdInputContext");
}
  };

  void Setup(const MklConvBwdInputParams& convBwdInputDims) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSmklPSmkl_conv_grad_input_opsDTcc mht_5(mht_5_v, 355, "", "./tensorflow/core/kernels/mkl/mkl_conv_grad_input_ops.cc", "Setup");

    memory::format_tag user_data_fmt;
    if (convBwdInputDims.native_format) {
      user_data_fmt =
          MklTensorFormatToMklDnnDataFormat(convBwdInputDims.tf_fmt);
    } else {
      // Create memory descriptors for conv bwd input without any specified
      // format so that MKL-DNN can pick an appropriate one depending on the
      // input parameters.
      user_data_fmt = memory::format_tag::any;
    }
    context_.diff_dst_md.reset(new memory::desc(
        {convBwdInputDims.diff_dst_dims}, MklDnnType<T>(), user_data_fmt));
    context_.diff_src_md.reset(new memory::desc(
        {convBwdInputDims.diff_src_dims}, MklDnnType<T>(), user_data_fmt));
    context_.filter_md.reset(new memory::desc({convBwdInputDims.filter_dims},
                                              MklDnnType<T>(),
                                              memory::format_tag::any));

    // Create descriptors for both conv fwd and conv bwd input.
    context_.bwd_input_desc.reset(new ConvBwdDataDesc(
        dnnl::algorithm::convolution_direct, *context_.diff_src_md,
        *context_.filter_md, *context_.diff_dst_md, convBwdInputDims.strides,
        convBwdInputDims.dilations, convBwdInputDims.padding_left,
        convBwdInputDims.padding_right));

    context_.fwd_desc.reset(new ConvFwdDesc(
        prop_kind::forward, dnnl::algorithm::convolution_direct,
        *context_.diff_src_md, *context_.filter_md, *context_.diff_dst_md,
        convBwdInputDims.strides, convBwdInputDims.dilations,
        convBwdInputDims.padding_left, convBwdInputDims.padding_right));

    // Create primitive descriptors for conv fwd and conv bwd input.
    context_.fwd_pd.reset(new ConvFwdPd(*context_.fwd_desc, cpu_engine_));
    context_.bwd_input_pd.reset(new ConvBwdDataPd(
        *context_.bwd_input_desc, cpu_engine_, *context_.fwd_pd));

    // Create memory using dummy data.
    context_.diff_src_mem.reset(new memory(
        context_.bwd_input_pd.get()->diff_src_desc(), cpu_engine_, DummyData));
    context_.filter_mem.reset(new memory(
        context_.bwd_input_pd.get()->weights_desc(), cpu_engine_, DummyData));
    context_.diff_dst_mem.reset(new memory(
        context_.bwd_input_pd.get()->diff_dst_desc(), cpu_engine_, DummyData));

    // Create conv bwd input primitive and add it to the net
    context_.conv_bwd_input.reset(
        new convolution_backward_data(*context_.bwd_input_pd));
    context_.bwd_input_primitives_args.push_back(
        {{DNNL_ARG_DIFF_DST, *context_.diff_dst_mem},
         {DNNL_ARG_WEIGHTS, *context_.filter_mem},
         {DNNL_ARG_DIFF_SRC, *context_.diff_src_mem}});

    context_.bwd_input_primitives.push_back(*context_.conv_bwd_input);
  }

  struct ConvBwdInputContext context_;
#ifdef DNNL_AARCH64_USE_ACL
  mutex primitive_execution_mu_;
#endif
};

template <typename T>
class MklConvBwdInputPrimitiveFactory : public MklPrimitiveFactory<T> {
 private:
  MklConvBwdInputPrimitiveFactory() {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSmklPSmkl_conv_grad_input_opsDTcc mht_6(mht_6_v, 423, "", "./tensorflow/core/kernels/mkl/mkl_conv_grad_input_ops.cc", "MklConvBwdInputPrimitiveFactory");
}
  ~MklConvBwdInputPrimitiveFactory() {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSmklPSmkl_conv_grad_input_opsDTcc mht_7(mht_7_v, 427, "", "./tensorflow/core/kernels/mkl/mkl_conv_grad_input_ops.cc", "~MklConvBwdInputPrimitiveFactory");
}

 public:
  static MklConvBwdInputPrimitive<T>* Get(
      const MklConvBwdInputParams& convBwdInputDims, bool do_not_cache) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSmklPSmkl_conv_grad_input_opsDTcc mht_8(mht_8_v, 434, "", "./tensorflow/core/kernels/mkl/mkl_conv_grad_input_ops.cc", "Get");

    MklConvBwdInputPrimitive<T>* conv_bwd_input = nullptr;

    if (do_not_cache) {  // Always allocate primitive.
      conv_bwd_input = new MklConvBwdInputPrimitive<T>(convBwdInputDims);
    } else {
      // look into the pool for reusable primitive.
      conv_bwd_input = dynamic_cast<MklConvBwdInputPrimitive<T>*>(
          MklConvBwdInputPrimitiveFactory<T>::GetInstance().GetConvBwdInput(
              convBwdInputDims));
      if (conv_bwd_input == nullptr) {
        conv_bwd_input = new MklConvBwdInputPrimitive<T>(convBwdInputDims);
        MklConvBwdInputPrimitiveFactory<T>::GetInstance().SetConvBwdInput(
            convBwdInputDims, conv_bwd_input);
      }
    }

    return conv_bwd_input;
  }

 private:
  static MklConvBwdInputPrimitiveFactory& GetInstance() {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSmklPSmkl_conv_grad_input_opsDTcc mht_9(mht_9_v, 458, "", "./tensorflow/core/kernels/mkl/mkl_conv_grad_input_ops.cc", "GetInstance");

    static MklConvBwdInputPrimitiveFactory instance_;
    return instance_;
  }

  static string CreateKey(const MklConvBwdInputParams& convBwdInputDims) {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSmklPSmkl_conv_grad_input_opsDTcc mht_10(mht_10_v, 466, "", "./tensorflow/core/kernels/mkl/mkl_conv_grad_input_ops.cc", "CreateKey");

    string prefix = "conv_bwd_input";
    FactoryKeyCreator key_creator;
    key_creator.AddAsKey(prefix);
    key_creator.AddAsKey(convBwdInputDims.diff_src_dims);
    key_creator.AddAsKey(convBwdInputDims.filter_dims);
    key_creator.AddAsKey(convBwdInputDims.diff_dst_dims);
    key_creator.AddAsKey(convBwdInputDims.strides);
    key_creator.AddAsKey(convBwdInputDims.dilations);
    key_creator.AddAsKey(convBwdInputDims.padding_left);
    key_creator.AddAsKey(convBwdInputDims.padding_right);
    if (convBwdInputDims.native_format) {
      key_creator.AddAsKey(convBwdInputDims.tf_fmt);
    }
    return key_creator.GetKey();
  }

  MklPrimitive* GetConvBwdInput(const MklConvBwdInputParams& convBwdInputDims) {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSmklPSmkl_conv_grad_input_opsDTcc mht_11(mht_11_v, 486, "", "./tensorflow/core/kernels/mkl/mkl_conv_grad_input_ops.cc", "GetConvBwdInput");

    string key = CreateKey(convBwdInputDims);
    return this->GetOp(key);
  }

  void SetConvBwdInput(const MklConvBwdInputParams& convBwdInputDims,
                       MklPrimitive* op) {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSmklPSmkl_conv_grad_input_opsDTcc mht_12(mht_12_v, 495, "", "./tensorflow/core/kernels/mkl/mkl_conv_grad_input_ops.cc", "SetConvBwdInput");

    string key = CreateKey(convBwdInputDims);
    this->SetOp(key, op);
  }
};

template <typename Device, class T, bool is_depthwise, bool native_format>
class MklConvCustomBackpropInputOp
    : public MklConvBackpropCommonOp<Device, T, is_depthwise> {
 public:
  explicit MklConvCustomBackpropInputOp(OpKernelConstruction* context)
      : MklConvBackpropCommonOp<Device, T, is_depthwise>(context) {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSmklPSmkl_conv_grad_input_opsDTcc mht_13(mht_13_v, 509, "", "./tensorflow/core/kernels/mkl/mkl_conv_grad_input_ops.cc", "MklConvCustomBackpropInputOp");
}

  ~MklConvCustomBackpropInputOp() {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSmklPSmkl_conv_grad_input_opsDTcc mht_14(mht_14_v, 514, "", "./tensorflow/core/kernels/mkl/mkl_conv_grad_input_ops.cc", "~MklConvCustomBackpropInputOp");
}

  void Compute(OpKernelContext* context) {
   std::vector<std::string> mht_15_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSmklPSmkl_conv_grad_input_opsDTcc mht_15(mht_15_v, 519, "", "./tensorflow/core/kernels/mkl/mkl_conv_grad_input_ops.cc", "Compute");

    try {
      // Input tensors.
      const Tensor& src_tensor = MklGetInput(context, kInputIdx);
      const Tensor& filter_tensor = MklGetInput(context, kFilterIdx);
      const Tensor& diff_dst_tensor = MklGetInput(context, kOutbpropIdx);

      MklDnnShape src_mkl_shape, filter_mkl_shape, diff_dst_mkl_shape;
      GetMklShape(context, kInputIdx, &src_mkl_shape, native_format);
      GetMklShape(context, kFilterIdx, &filter_mkl_shape, native_format);
      GetMklShape(context, kOutbpropIdx, &diff_dst_mkl_shape, native_format);
      // Allow operator-specific sanity checking of shapes.
      ValidateMklShapes(src_mkl_shape, filter_mkl_shape, diff_dst_mkl_shape);

      // Allow operator-specific generation of shapes.
      // E.g., ConvBackpropFilter gets filter as filter_sizes. It is a
      // tensor containing shape of filter. So filter.shape() is not
      // a correct way to get filter shape. These operator-specific calls
      // allow this class to handle this case.
      TensorShape src_tf_shape;
      if (src_tensor.dim_size(0) == 2) {
        OP_REQUIRES_OK(context, Conv2DBackpropComputeInputShape(
                                    src_tensor, filter_tensor.shape(),
                                    diff_dst_tensor.shape(), this->data_format_,
                                    &src_tf_shape));
      } else {
        src_tf_shape = MakeInputTfShape(context, src_tensor);
      }

      TensorShape filter_tf_shape = MakeFilterTfShape(context, filter_tensor);
      TensorShape diff_dst_tf_shape =
          GetTfShape(context, kOutbpropIdx, native_format);

      // Corner cases: output with 0 elements and 0 batch size.
      Tensor* diff_src_tensor = nullptr;
      if (src_tf_shape.num_elements() == 0 ||
          filter_tf_shape.num_elements() == 0 ||
          diff_dst_tf_shape.num_elements() == 0) {
        MklDnnShape diff_src_mkl_shape;
        diff_src_mkl_shape.SetMklTensor(false);
        TensorShape diff_src_tf_shape =
            GetOutputTfShape(src_tf_shape, filter_tf_shape, diff_dst_tf_shape);
        const int kOutputIdx = 0;
        AllocateOutputSetMklShape(context, kOutputIdx, &diff_src_tensor,
                                  diff_src_tf_shape, diff_src_mkl_shape,
                                  native_format);
        DCHECK(diff_src_tensor != nullptr);

        // If output tensor has more than 0 elements, we need to 0 them out.
        auto diff_src_data = diff_src_tensor->flat<T>().data();
        for (size_t i = 0; i < diff_src_tf_shape.num_elements(); ++i) {
          diff_src_data[i] = static_cast<T>(0);
        }
        return;
      }

      // By default, all dims are in MKL order except those that are suffixed
      // with `tf_order`.
      memory::dims diff_dst_dims, fwd_src_dims, fwd_filter_dims;
      memory::dims padding_left, padding_right, dilations, strides;
      memory::dims fwd_output_dims, fwd_output_dims_tf_order;

      // Get conv fwd parameters.
      bool is_grouped_convolution = false;
      MklDnnConvUtil conv_util(context, this->strides_, this->padding_,
                               this->data_format_, this->dilations_);
      conv_util.GetConvFwdSizesInMklOrder(
          src_tf_shape, filter_tf_shape, &fwd_src_dims, &fwd_filter_dims,
          &strides, &dilations, &fwd_output_dims_tf_order, &fwd_output_dims,
          &padding_left, &padding_right, &is_grouped_convolution, false,
          is_depthwise);
      if (!context->status().ok()) return;

      bool is_conv2d = (this->strides_.size() == 4);

      // Create conv fwd descriptor since conv bwd input API needs it.
      // For that, we first need to create input, filter and output memory
      // descriptors.
      auto tf_fmt = is_conv2d
                        ? TFDataFormatToMklDnnDataFormat(this->data_format_)
                        : TFDataFormatToMklDnn3DDataFormat(this->data_format_);

      auto mkl_fmt_tag = MklTensorFormatToMklDnnDataFormat(tf_fmt);
      OP_REQUIRES(context, mkl_fmt_tag != memory::format_tag::undef,
                  errors::InvalidArgument("Invalid data format"));

      // If filter is in MKL layout, then simply grab filter layout;
      // otherwise, construct filter in TF layout.
      // For TF layout, filter is in HWIO format.
      auto fwd_filter_md =
          filter_mkl_shape.IsMklTensor()
              ? filter_mkl_shape.GetMklLayout()
              : memory::desc(fwd_filter_dims, MklDnnType<T>(),
                             (is_depthwise || is_grouped_convolution)
                                 ? memory::format_tag::hwigo
                                 : (is_conv2d ? memory::format_tag::hwio
                                              : memory::format_tag::dhwio));

      conv_util.GetInputSizeInMklOrder(diff_dst_tf_shape, &diff_dst_dims);
      if (!context->status().ok()) return;

      auto diff_dst_md =
          diff_dst_mkl_shape.IsMklTensor()
              ? diff_dst_mkl_shape.GetMklLayout()
              : memory::desc(diff_dst_dims, MklDnnType<T>(), mkl_fmt_tag);

      // The default dilation factor for each dimension is 1 in TF and
      // 0 in MKL-DNN.
      for (int i = 0; i < dilations.size(); ++i) --dilations[i];
      MklConvBwdInputParams convBwdInputDims(
          fwd_src_dims, fwd_filter_dims, diff_dst_dims, strides, tf_fmt,
          native_format, dilations, padding_left, padding_right);

      // We don't cache those primitives if the environment variable
      // TF_MKL_OPTIMIZE_PRIMITIVE_MEMUSE is true and if primitive descriptor
      // includes potentially large buffers. MKL-DNN allocates buffers
      // in the following cases
      //   1. Legacy CPU without AVX512/AVX2, or
      //   2. 1x1 convolution with stride != 1
      bool do_not_cache = MklPrimitiveFactory<T>::IsPrimitiveMemOptEnabled() &&
                          (MklPrimitiveFactory<T>::IsLegacyPlatform() ||
                           IsConv1x1StrideNot1(fwd_filter_dims, strides));

      MklConvBwdInputPrimitive<T>* conv_bwd_input =
          MklConvBwdInputPrimitiveFactory<T>::Get(convBwdInputDims,
                                                  do_not_cache);

      auto bwd_input_pd = conv_bwd_input->GetPrimitiveDesc();
      auto diff_src_pd = bwd_input_pd.get()->diff_src_desc();
      auto bwd_diff_src_dims = GetOutputDims(fwd_src_dims, fwd_filter_dims);
      auto bwd_diff_src_format = GetOutputFormat(tf_fmt);

      // Allocate output tensor.
      MklDnnShape diff_src_mkl_shape;
      diff_src_mkl_shape.SetMklTensor(true);
      diff_src_mkl_shape.SetMklLayout(&diff_src_pd);
      diff_src_mkl_shape.SetElemType(MklDnnType<T>());
      diff_src_mkl_shape.SetTfLayout(bwd_diff_src_dims.size(),
                                     bwd_diff_src_dims, bwd_diff_src_format);
      TensorShape diff_src_tf_shape;
      diff_src_tf_shape.AddDim(diff_src_pd.get_size() / sizeof(T));
      if (native_format) {
        diff_src_tf_shape = diff_src_mkl_shape.GetTfShape();
      }
      AllocateOutputSetMklShape(context, 0, &diff_src_tensor, diff_src_tf_shape,
                                diff_src_mkl_shape, native_format);
      T* diff_src_data =
          static_cast<T*>(const_cast<T*>(diff_src_tensor->flat<T>().data()));

      // Check if filter and diff_dst need to be reordered.
      T* filter_data = nullptr;
      MklDnnData<T> filter(&cpu_engine_);
      if (fwd_filter_md != bwd_input_pd->weights_desc()) {
        filter.SetUsrMem(fwd_filter_md, &filter_tensor);
        filter.CheckReorderToOpMem(bwd_input_pd.get()->weights_desc(),
                                   cpu_engine_, context);
        filter_data = static_cast<T*>(filter.GetOpMem().get_data_handle());
      } else {
        filter_data =
            static_cast<T*>(const_cast<T*>(filter_tensor.flat<T>().data()));
      }

      T* diff_dst_data = nullptr;
      MklDnnData<T> diff_dst(&cpu_engine_);
      if (diff_dst_md != bwd_input_pd->diff_dst_desc()) {
        diff_dst.SetUsrMem(diff_dst_md, &diff_dst_tensor);
        diff_dst.CheckReorderToOpMem(bwd_input_pd.get()->diff_dst_desc(),
                                     cpu_engine_, context);
        diff_dst_data = static_cast<T*>(diff_dst.GetOpMem().get_data_handle());
      } else {
        diff_dst_data =
            static_cast<T*>(const_cast<T*>(diff_dst_tensor.flat<T>().data()));
      }

      std::shared_ptr<stream> bwd_cpu_stream;
      MklDnnThreadPool eigen_tp(context);
      bwd_cpu_stream.reset(
          CreateStream(&eigen_tp, conv_bwd_input->GetEngine()));
      // Execute conv bwd input primitive.
      conv_bwd_input->Execute(diff_src_data, filter_data, diff_dst_data,
                              bwd_cpu_stream);

      // Delete primitive since it is not cached.
      if (do_not_cache) {
        delete conv_bwd_input;
      }
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
  const int kInputIdx = 0, kFilterIdx = 1, kOutbpropIdx = 2;
  const int kDilationH = 0, kDilationW = 1;

  engine cpu_engine_ = engine(engine::kind::cpu, 0);

  // Assert that input shapes are valid.
  void ValidateMklShapes(const MklDnnShape& input_mkl_shape,
                         const MklDnnShape& filter_mkl_shape,
                         const MklDnnShape& obp_mkl_shape) {
   std::vector<std::string> mht_16_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSmklPSmkl_conv_grad_input_opsDTcc mht_16(mht_16_v, 727, "", "./tensorflow/core/kernels/mkl/mkl_conv_grad_input_ops.cc", "ValidateMklShapes");

    // Tensor that feeds to 'Input' slot of BackpropInput is always just a shape
    // of the Tensor and never an actual tensor. So it will never be in MKL
    // layout.
    CHECK(!input_mkl_shape.IsMklTensor())
        << "ConvBackpropInput: input should not be in MKL Layout";
  }

  // Get TensorFlow shape of input tensor.
  TensorShape MakeInputTfShape(OpKernelContext* context,
                               const Tensor& input_tensor) {
   std::vector<std::string> mht_17_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSmklPSmkl_conv_grad_input_opsDTcc mht_17(mht_17_v, 740, "", "./tensorflow/core/kernels/mkl/mkl_conv_grad_input_ops.cc", "MakeInputTfShape");

    TensorShape input_tf_shape;
    CHECK_EQ(TensorShapeUtils::IsVector(input_tensor.shape()), true);
    // Conv[2D|3D]BackpropInputV2 supports both DT_INT32 and DT_INT64
    // output_shape tensor::MakeShape is able to handle both DT_INT32 and
    // DT_INT64 for input_tensor.
    TF_CHECK_OK(tensor::MakeShape(input_tensor, &input_tf_shape));
    return input_tf_shape;
  }

  // Get TensorFlow shape of filter tensor.
  TensorShape MakeFilterTfShape(OpKernelContext* context,
                                const Tensor& filter_tensor) {
   std::vector<std::string> mht_18_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSmklPSmkl_conv_grad_input_opsDTcc mht_18(mht_18_v, 755, "", "./tensorflow/core/kernels/mkl/mkl_conv_grad_input_ops.cc", "MakeFilterTfShape");

    return GetTfShape(context, kFilterIdx, native_format);
  }

  // Get the Tensorflow shape of Output (diff_src),
  // which is same as shape of Conv 'input'.
  TensorShape GetOutputTfShape(const TensorShape& input_shape,
                               const TensorShape& filter_shape,
                               const TensorShape& outbprop_shape) {
   std::vector<std::string> mht_19_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSmklPSmkl_conv_grad_input_opsDTcc mht_19(mht_19_v, 766, "", "./tensorflow/core/kernels/mkl/mkl_conv_grad_input_ops.cc", "GetOutputTfShape");

    return input_shape;
  }

  const memory::dims& GetOutputDims(const memory::dims& fwd_input_dims,
                                    const memory::dims& fwd_filter_dims) {
   std::vector<std::string> mht_20_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSmklPSmkl_conv_grad_input_opsDTcc mht_20(mht_20_v, 774, "", "./tensorflow/core/kernels/mkl/mkl_conv_grad_input_ops.cc", "GetOutputDims");

    return fwd_input_dims;
  }

  // Output layout is Tensorflow's layout in data format order.
  MklTensorFormat GetOutputFormat(const MklTensorFormat data_format) {
   std::vector<std::string> mht_21_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSmklPSmkl_conv_grad_input_opsDTcc mht_21(mht_21_v, 782, "", "./tensorflow/core/kernels/mkl/mkl_conv_grad_input_ops.cc", "GetOutputFormat");

    return data_format;
  }

  // TODO(intel-tf): Move this function to mkl_util.h since it is common to
  // both the forward and backward implementations
  void AllocateOutputTensor(OpKernelContext* context,
                            const ConvBwdDataPd& conv_pd,
                            const memory::dims& output_dims_mkl_order,
                            MklTensorFormat output_tf_format,
                            Tensor** output_tensor) {
   std::vector<std::string> mht_22_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSmklPSmkl_conv_grad_input_opsDTcc mht_22(mht_22_v, 795, "", "./tensorflow/core/kernels/mkl/mkl_conv_grad_input_ops.cc", "AllocateOutputTensor");

    DCHECK(output_tensor != nullptr);

    // Output primitive descriptor for backward data is diff_src.
    auto dst_pd = conv_pd.diff_src_desc();

    // Allocate shape of MKL tensor.
    MklDnnShape output_mkl_shape;
    output_mkl_shape.SetMklTensor(true);
    output_mkl_shape.SetMklLayout(&dst_pd);
    output_mkl_shape.SetElemType(MklDnnType<T>());
    output_mkl_shape.SetTfLayout(output_dims_mkl_order.size(),
                                 output_dims_mkl_order, output_tf_format);

    // Allocate shape of TF tensor.
    TensorShape output_tf_shape;
    output_tf_shape.AddDim(dst_pd.get_size() / sizeof(T));

    AllocateOutputSetMklShape(context, 0, output_tensor, output_tf_shape,
                              output_mkl_shape);
  }
};

#define REGISTER_MKL_CPU_KERNELS(T)                              \
  REGISTER_KERNEL_BUILDER(                                       \
      Name("_MklConv2DBackpropInput")                            \
          .Device(DEVICE_CPU)                                    \
          .TypeConstraint<T>("T")                                \
          .Label(mkl_op_registry::kMklLayoutDependentOpLabel),   \
      MklConvCustomBackpropInputOp<CPUDevice, T, false, false>); \
  REGISTER_KERNEL_BUILDER(                                       \
      Name("_MklConv3DBackpropInputV2")                          \
          .Device(DEVICE_CPU)                                    \
          .TypeConstraint<T>("T")                                \
          .Label(mkl_op_registry::kMklLayoutDependentOpLabel),   \
      MklConvCustomBackpropInputOp<CPUDevice, T, false, false>); \
  REGISTER_KERNEL_BUILDER(                                       \
      Name("_MklDepthwiseConv2dNativeBackpropInput")             \
          .Device(DEVICE_CPU)                                    \
          .TypeConstraint<T>("T")                                \
          .Label(mkl_op_registry::kMklLayoutDependentOpLabel),   \
      MklConvCustomBackpropInputOp<CPUDevice, T, true, false>);  \
  REGISTER_KERNEL_BUILDER(                                       \
      Name("_MklNativeConv2DBackpropInput")                      \
          .Device(DEVICE_CPU)                                    \
          .TypeConstraint<T>("T")                                \
          .Label(mkl_op_registry::kMklNameChangeOpLabel),        \
      MklConvCustomBackpropInputOp<CPUDevice, T, false, true>);  \
  REGISTER_KERNEL_BUILDER(                                       \
      Name("_MklNativeConv3DBackpropInputV2")                    \
          .Device(DEVICE_CPU)                                    \
          .TypeConstraint<T>("T")                                \
          .Label(mkl_op_registry::kMklNameChangeOpLabel),        \
      MklConvCustomBackpropInputOp<CPUDevice, T, false, true>);  \
  REGISTER_KERNEL_BUILDER(                                       \
      Name("_MklNativeDepthwiseConv2dNativeBackpropInput")       \
          .Device(DEVICE_CPU)                                    \
          .TypeConstraint<T>("T")                                \
          .Label(mkl_op_registry::kMklNameChangeOpLabel),        \
      MklConvCustomBackpropInputOp<CPUDevice, T, true, true>);

TF_CALL_float(REGISTER_MKL_CPU_KERNELS);
TF_CALL_bfloat16(REGISTER_MKL_CPU_KERNELS);

#undef REGISTER_MKL_CPU_KERNELS

}  // namespace tensorflow
#endif  // INTEL_MKL
