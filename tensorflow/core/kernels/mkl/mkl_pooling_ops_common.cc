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
class MHTracer_DTPStensorflowPScorePSkernelsPSmklPSmkl_pooling_ops_commonDTcc {
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
   MHTracer_DTPStensorflowPScorePSkernelsPSmklPSmkl_pooling_ops_commonDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSkernelsPSmklPSmkl_pooling_ops_commonDTcc() {
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

#include "tensorflow/core/kernels/mkl/mkl_pooling_ops_common.h"

#include <limits>
#include <vector>

#include "tensorflow/core/common_runtime/device.h"
#include "tensorflow/core/framework/bounds_check.h"
#include "tensorflow/core/framework/kernel_shape_util.h"

namespace tensorflow {
using dnnl::prop_kind;

template <typename T>
void MklPoolingFwdPrimitive<T>::Setup(const MklPoolingParams& fwdParams) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSmklPSmkl_pooling_ops_commonDTcc mht_0(mht_0_v, 200, "", "./tensorflow/core/kernels/mkl/mkl_pooling_ops_common.cc", "MklPoolingFwdPrimitive<T>::Setup");

  DCHECK(fwdParams.alg_kind == dnnl::algorithm::pooling_max ||
         fwdParams.alg_kind == dnnl::algorithm::pooling_avg ||
         fwdParams.alg_kind == dnnl::algorithm::pooling_avg_include_padding ||
         fwdParams.alg_kind == dnnl::algorithm::pooling_avg_exclude_padding)
      << "Pooling algorithm kind is not supported";

  context_.alg_kind = fwdParams.alg_kind;
  context_.prop_kind = fwdParams.prop_kind;

  // Create memory descriptor
  // TODO(intel-tf): Pooling doesn't expose to get the src_primitive_desc,
  //                 so src format is currently hard-coded.
  //                 A utility function is used to do this,
  //                 which may be broken with future CPU architectures
  context_.src_md.reset(new memory::desc(fwdParams.src_md.data));
  context_.dst_md.reset(new memory::desc({fwdParams.dst_dims}, MklDnnType<T>(),
                                         fwdParams.native_format
                                             ? fwdParams.src_format
                                             : memory::format_tag::any));

  // Create a pooling descriptor.
  context_.fwd_desc.reset(new pooling_forward::desc(
      fwdParams.prop_kind, fwdParams.alg_kind, *context_.src_md,
      *context_.dst_md, fwdParams.strides, fwdParams.filter_dims,
      fwdParams.padding_left, fwdParams.padding_right));
  context_.fwd_pd.reset(
      new pooling_forward::primitive_desc(*context_.fwd_desc, cpu_engine_));
  context_.dst_fmt = static_cast<memory::format_tag>(memory::format_tag::any);

  // Create oneDNN internal memory object with dummy data.
  context_.src_mem.reset(
      new memory(context_.fwd_pd.get()->src_desc(), cpu_engine_, DummyData));
  context_.dst_mem.reset(
      new memory(context_.fwd_pd.get()->dst_desc(), cpu_engine_, DummyData));

  // For max pooling, need to return workspace (ws) for backward computing.
  if (fwdParams.alg_kind == dnnl::algorithm::pooling_max &&
      fwdParams.prop_kind == prop_kind::forward_training) {
    context_.ws_mem.reset(new memory(context_.fwd_pd.get()->workspace_desc(),
                                     cpu_engine_, DummyData));
    context_.net_args.push_back({{DNNL_ARG_SRC, *context_.src_mem},
                                 {DNNL_ARG_DST, *context_.dst_mem},
                                 {DNNL_ARG_WORKSPACE, *context_.ws_mem}});
    context_.fwd.reset(new pooling_forward(*context_.fwd_pd));
  } else {
    context_.net_args.push_back(
        {{DNNL_ARG_SRC, *context_.src_mem}, {DNNL_ARG_DST, *context_.dst_mem}});
    context_.fwd.reset(new pooling_forward(*context_.fwd_pd));
  }

  context_.fwd_primitives.push_back(*context_.fwd);
}

template <typename T>
void MklPoolingFwdPrimitive<T>::Execute(const T* src_data, T* dst_data,
                                        void* ws_data,
                                        std::shared_ptr<stream> fwd_stream) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSmklPSmkl_pooling_ops_commonDTcc mht_1(mht_1_v, 260, "", "./tensorflow/core/kernels/mkl/mkl_pooling_ops_common.cc", "MklPoolingFwdPrimitive<T>::Execute");

#ifdef DNNL_AARCH64_USE_ACL
  mutex_lock lock(primitive_execution_mu_);
#endif
#ifndef ENABLE_ONEDNN_OPENMP
  context_.src_mem->set_data_handle(
      static_cast<void*>(const_cast<T*>(src_data)), *fwd_stream);
  context_.dst_mem->set_data_handle(static_cast<void*>(dst_data), *fwd_stream);
  if (context_.alg_kind == dnnl::algorithm::pooling_max &&
      context_.prop_kind ==
          prop_kind::forward_training) {  // Max pooling must have workspace.
    DCHECK(ws_data != nullptr);
    context_.ws_mem->set_data_handle(ws_data, *fwd_stream);
  }
#else
  context_.src_mem->set_data_handle(
      static_cast<void*>(const_cast<T*>(src_data)));
  context_.dst_mem->set_data_handle(static_cast<void*>(dst_data));
  if (context_.alg_kind == dnnl::algorithm::pooling_max &&
      context_.prop_kind ==
          prop_kind::forward_training) {  // Max pooling must have workspace.
    DCHECK(ws_data != nullptr);
    context_.ws_mem->set_data_handle(ws_data);
  }
#endif  // !ENABLE_ONEDNN_OPENMP
  execute_primitives(context_.fwd_primitives, fwd_stream, context_.net_args);

  // Set back data handle.
  context_.src_mem->set_data_handle(DummyData);
  context_.dst_mem->set_data_handle(DummyData);
  if (context_.alg_kind == dnnl::algorithm::pooling_max &&
      context_.prop_kind ==
          prop_kind::forward_training) {  // Max pooling must have workspace.
    DCHECK(ws_data != nullptr);
    context_.ws_mem->set_data_handle(DummyData);
  }
}

template class MklPoolingFwdPrimitive<float>;
template class MklPoolingFwdPrimitive<quint8>;
template class MklPoolingFwdPrimitive<qint8>;
template class MklPoolingFwdPrimitive<bfloat16>;

template <typename T>
void MklPoolingBwdPrimitive<T>::Setup(const MklPoolingParams& bwdParams) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSmklPSmkl_pooling_ops_commonDTcc mht_2(mht_2_v, 307, "", "./tensorflow/core/kernels/mkl/mkl_pooling_ops_common.cc", "MklPoolingBwdPrimitive<T>::Setup");

  DCHECK(bwdParams.alg_kind == dnnl::algorithm::pooling_max ||
         bwdParams.alg_kind == dnnl::algorithm::pooling_avg ||
         bwdParams.alg_kind == dnnl::algorithm::pooling_avg_include_padding ||
         bwdParams.alg_kind == dnnl::algorithm::pooling_avg_exclude_padding)
      << "Pooling algorithm kind is not supported";
  context_.alg_kind = bwdParams.alg_kind;

  // Create memory descriptor.
  context_.src_md.reset(new memory::desc({bwdParams.src_dims}, MklDnnType<T>(),
                                         memory::format_tag::any));
  context_.src_md.reset(new memory::desc(bwdParams.src_md.data));
  context_.dst_md.reset(new memory::desc({bwdParams.dst_dims}, MklDnnType<T>(),
                                         bwdParams.native_format
                                             ? bwdParams.src_format
                                             : memory::format_tag::any));

  // Create a backward primitive. The implementation for backward must comply to
  // the workspace format it gets from forward pass, so we directly use src_md
  // and dst_md here.
  context_.bwd_desc.reset(new pooling_backward::desc(
      bwdParams.alg_kind, *context_.src_md, *context_.dst_md, bwdParams.strides,
      bwdParams.filter_dims, bwdParams.padding_left, bwdParams.padding_right));
  // Create a forward primitive,
  // which will be used as a hint for creating backward primitive.
  context_.fwd_desc.reset(new pooling_forward::desc(
      bwdParams.prop_kind, bwdParams.alg_kind, *context_.src_md,
      *context_.dst_md, bwdParams.strides, bwdParams.filter_dims,
      bwdParams.padding_left, bwdParams.padding_right));
  context_.fwd_pd.reset(
      new pooling_forward::primitive_desc(*context_.fwd_desc, cpu_engine_));
  context_.bwd_pd.reset(new pooling_backward::primitive_desc(
      *context_.bwd_desc, cpu_engine_, *context_.fwd_pd));

  // Create oneDNN internal memory object with dummy data.
  context_.diff_src_mem.reset(new memory(context_.bwd_pd.get()->diff_src_desc(),
                                         cpu_engine_, DummyData));
  context_.diff_dst_mem.reset(new memory(context_.bwd_pd.get()->diff_dst_desc(),
                                         cpu_engine_, DummyData));

  // For max pooling, need to return workspace for backward computing.
  if (bwdParams.alg_kind == dnnl::algorithm::pooling_max) {
    context_.ws_mem.reset(
        new memory(context_.fwd_pd.get()->workspace_desc(), cpu_engine_));
    context_.net_args.push_back({{DNNL_ARG_DIFF_DST, *context_.diff_dst_mem},
                                 {DNNL_ARG_WORKSPACE, *context_.ws_mem},
                                 {DNNL_ARG_DIFF_SRC, *context_.diff_src_mem}});
    context_.bwd.reset(new pooling_backward(*context_.bwd_pd));
  } else {
    context_.net_args.push_back({{DNNL_ARG_DIFF_DST, *context_.diff_dst_mem},
                                 {DNNL_ARG_DIFF_SRC, *context_.diff_src_mem}});
    context_.bwd.reset(new pooling_backward(*context_.bwd_pd));
  }
  context_.bwd_primitives.push_back(*context_.bwd);
}

template <typename T>
void MklPoolingBwdPrimitive<T>::Execute(const T* diff_dst_data,
                                        T* diff_src_data, const void* ws_data,
                                        std::shared_ptr<stream> bwd_stream) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSmklPSmkl_pooling_ops_commonDTcc mht_3(mht_3_v, 369, "", "./tensorflow/core/kernels/mkl/mkl_pooling_ops_common.cc", "MklPoolingBwdPrimitive<T>::Execute");

#ifdef DNNL_AARCH64_USE_ACL
  mutex_lock lock(primitive_execution_mu_);
#endif
#ifndef ENABLE_ONEDNN_OPENMP
  context_.diff_dst_mem->set_data_handle(
      static_cast<void*>(const_cast<T*>(diff_dst_data)), *bwd_stream);
  context_.diff_src_mem->set_data_handle(static_cast<void*>(diff_src_data),
                                         *bwd_stream);
  if (context_.alg_kind == dnnl::algorithm::pooling_max) {
    DCHECK(ws_data != nullptr);
    context_.ws_mem->set_data_handle(const_cast<void*>(ws_data), *bwd_stream);
  }
#else
  context_.diff_dst_mem->set_data_handle(
      static_cast<void*>(const_cast<T*>(diff_dst_data)));
  context_.diff_src_mem->set_data_handle(static_cast<void*>(diff_src_data));
  if (context_.alg_kind == dnnl::algorithm::pooling_max) {
    DCHECK(ws_data != nullptr);
    context_.ws_mem->set_data_handle(const_cast<void*>(ws_data));
  }
#endif  // !ENABLE_ONEDNN_OPENMP

  execute_primitives(context_.bwd_primitives, bwd_stream, context_.net_args);

  // Set back data handle.
  context_.diff_dst_mem->set_data_handle(DummyData);
  context_.diff_src_mem->set_data_handle(DummyData);
  if (context_.alg_kind == dnnl::algorithm::pooling_max) {
    DCHECK(ws_data != nullptr);
    context_.ws_mem->set_data_handle(DummyData);
  }
}

template class MklPoolingBwdPrimitive<float>;
template class MklPoolingBwdPrimitive<bfloat16>;

// Initialization for TensorFlow format
void MklPoolParameters::Init(OpKernelContext* context,
                             const std::vector<int32>& ksize,
                             const std::vector<int32>& stride, Padding padding,
                             TensorFormat data_format,
                             const TensorShape& tensor_in_shape) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSmklPSmkl_pooling_ops_commonDTcc mht_4(mht_4_v, 414, "", "./tensorflow/core/kernels/mkl/mkl_pooling_ops_common.cc", "MklPoolParameters::Init");

  // For max pooling, tensor_in should have 4 or 5 dimensions.
  OP_REQUIRES(context,
              tensor_in_shape.dims() == 4 || tensor_in_shape.dims() == 5,
              errors::InvalidArgument("tensor_in must be 4 or 5-dimensional"));

  depth = GetTensorDim(tensor_in_shape, data_format, 'C');
  if (tensor_in_shape.dims() == 4) {
    // Pool2D
    tensor_in_cols = GetTensorDim(tensor_in_shape, data_format, 'W');
    tensor_in_rows = GetTensorDim(tensor_in_shape, data_format, 'H');
  } else {
    // Pool3D
    tensor_in_planes = GetTensorDim(tensor_in_shape, data_format, '0');
    tensor_in_rows = GetTensorDim(tensor_in_shape, data_format, '1');
    tensor_in_cols = GetTensorDim(tensor_in_shape, data_format, '2');
  }
  tensor_in_batch = GetTensorDim(tensor_in_shape, data_format, 'N');

  Init(context, ksize, stride, padding, data_format);
}

// Initialization for oneDNN format.
void MklPoolParameters::Init(OpKernelContext* context,
                             const std::vector<int32>& ksize,
                             const std::vector<int32>& stride, Padding padding,
                             TensorFormat data_format,
                             const MklDnnShape* mklInputShape) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSmklPSmkl_pooling_ops_commonDTcc mht_5(mht_5_v, 444, "", "./tensorflow/core/kernels/mkl/mkl_pooling_ops_common.cc", "MklPoolParameters::Init");

  // Get the input sizes.
  if (ksize.size() == 4) {
    // Pool2D
    depth = mklInputShape->GetDimension('C');
    tensor_in_cols = mklInputShape->GetDimension('W');
    tensor_in_rows = mklInputShape->GetDimension('H');
    tensor_in_batch = mklInputShape->GetDimension('N');
  } else {
    // Pool3D
    depth = mklInputShape->GetDimension3D('C');
    tensor_in_cols = mklInputShape->GetDimension3D('W');
    tensor_in_rows = mklInputShape->GetDimension3D('H');
    tensor_in_planes = mklInputShape->GetDimension3D('D');
    tensor_in_batch = mklInputShape->GetDimension3D('N');
  }

  Init(context, ksize, stride, padding, data_format);
}

// Common Initialization for TensorFlow and MKL formats.
void MklPoolParameters::Init(OpKernelContext* context,
                             const std::vector<int32>& ksize,
                             const std::vector<int32>& stride, Padding padding,
                             TensorFormat data_format) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSmklPSmkl_pooling_ops_commonDTcc mht_6(mht_6_v, 471, "", "./tensorflow/core/kernels/mkl/mkl_pooling_ops_common.cc", "MklPoolParameters::Init");

  // Get the data format.
  this->data_format = data_format;

  bool is_pool2d = (ksize.size() == 4);
  if (is_pool2d) {
    // Pool2D
    // Get the output sizes.
    window_rows = GetTensorDim(ksize, data_format, 'H');
    window_cols = GetTensorDim(ksize, data_format, 'W');
    depth_window = GetTensorDim(ksize, data_format, 'C');

    // Get the strides.
    row_stride = GetTensorDim(stride, data_format, 'H');
    col_stride = GetTensorDim(stride, data_format, 'W');
    depth_stride = GetTensorDim(stride, data_format, 'C');

    // We only support 2D pooling across width/height and depthwise
    // pooling, not a combination.
    OP_REQUIRES(context,
                (depth_window == 1 || (window_rows == 1 && window_cols == 1)),
                errors::Unimplemented(
                    "MaxPooling supports exactly one of pooling across depth "
                    "or pooling across width/height."));
  } else {
    // Pool3D
    // Get the output sizes.
    window_planes = GetTensorDim(ksize, data_format, '0');
    window_rows = GetTensorDim(ksize, data_format, '1');
    window_cols = GetTensorDim(ksize, data_format, '2');
    depth_window = GetTensorDim(ksize, data_format, 'C');

    // Get the strides.
    planes_stride = GetTensorDim(stride, data_format, '0');
    row_stride = GetTensorDim(stride, data_format, '1');
    col_stride = GetTensorDim(stride, data_format, '2');
    depth_stride = GetTensorDim(stride, data_format, 'C');

    // We only support 3D pooling across depth/width/height and depthwise
    // pooling, not a combination.
    OP_REQUIRES(context,
                (depth_window == 1 ||
                 (window_rows == 1 && window_cols == 1 && window_planes == 1)),
                errors::Unimplemented(
                    "AvgPooling3D supports exactly one of pooling across depth "
                    "or pooling across depth/width/height."));
  }

  if (depth_window == 1) {  // We are pooling in the D (Pool3D only), H and W.
    if (!is_pool2d) {
      OP_REQUIRES_OK(
          context, GetWindowedOutputSizeVerbose(tensor_in_planes, window_planes,
                                                planes_stride, padding,
                                                &out_planes, &pad_P1, &pad_P2));
    }

    OP_REQUIRES_OK(context, GetWindowedOutputSizeVerbose(
                                tensor_in_rows, window_rows, row_stride,
                                padding, &out_height, &pad_top, &pad_bottom));

    OP_REQUIRES_OK(context, GetWindowedOutputSizeVerbose(
                                tensor_in_cols, window_cols, col_stride,
                                padding, &out_width, &pad_left, &pad_right));

    // TF can work with int64, but oneDNN only supports int32.
    // Fail if the depth, height or width are greater than MAX_INT.
    // We check depth only for 3D pooling case.
    if (!is_pool2d) {
      OP_REQUIRES(context,
                  FastBoundsCheck(out_planes, std::numeric_limits<int>::max()),
                  errors::InvalidArgument("output depth/planes is too large"));
    }

    OP_REQUIRES(context,
                FastBoundsCheck(out_height, std::numeric_limits<int>::max()),
                errors::InvalidArgument("output height is too large"));

    OP_REQUIRES(context,
                FastBoundsCheck(out_width, std::numeric_limits<int>::max()),
                errors::InvalidArgument("output width is too large"));

    out_depth = depth;  // Output will have the same depth as the input.
  } else {              // We are pooling in the depth dimension.
    // Our current version of depthwise max pooling does not support
    // any padding, and expects the depth_window to equal the depth
    // stride (no overlapping).
    OP_REQUIRES(context, depth % depth_window == 0,
                errors::Unimplemented("Depthwise max pooling requires the"
                                      " depth window to evenly divide the"
                                      " input depth"));
    OP_REQUIRES(context, depth_stride == depth_window,
                errors::Unimplemented("Depthwise max pooling requires the"
                                      " depth window to equal the depth"
                                      " stride"));

    // The current version of depthwise max is only implemented on CPU.
    OP_REQUIRES(context,
                (DeviceType(static_cast<Device*>(context->device())
                                ->attributes()
                                .device_type()) == DeviceType(DEVICE_CPU)),
                errors::Unimplemented("Depthwise max pooling is currently "
                                      "only implemented for CPU devices."));

    out_depth = depth / depth_window;
  }
}

}  // namespace tensorflow

#endif  // INTEL_MKL
