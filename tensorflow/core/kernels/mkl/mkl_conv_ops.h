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

#ifndef TENSORFLOW_CORE_KERNELS_MKL_MKL_CONV_OPS_H_
#define TENSORFLOW_CORE_KERNELS_MKL_MKL_CONV_OPS_H_
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
class MHTracer_DTPStensorflowPScorePSkernelsPSmklPSmkl_conv_opsDTh {
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
   MHTracer_DTPStensorflowPScorePSkernelsPSmklPSmkl_conv_opsDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSkernelsPSmklPSmkl_conv_opsDTh() {
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


#ifdef INTEL_MKL
#include <limits>
#include <memory>
#include <vector>

#include "dnnl.hpp"
#include "tensorflow/core/framework/bounds_check.h"
#include "tensorflow/core/framework/kernel_shape_util.h"
#include "tensorflow/core/framework/numeric_op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/tensor_slice.h"
#include "tensorflow/core/kernels/conv_grad_ops.h"
#include "tensorflow/core/kernels/ops_util.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/gtl/array_slice.h"
#include "tensorflow/core/lib/strings/numbers.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/util/mkl_util.h"
#include "tensorflow/core/util/onednn_env_vars.h"
#include "tensorflow/core/util/padding.h"
#include "tensorflow/core/util/tensor_format.h"

using dnnl::convolution_forward;
using dnnl::prop_kind;
using dnnl::stream;

namespace tensorflow {

using ConvFwdDesc = dnnl::convolution_forward::desc;
using ConvFwdPd = dnnl::convolution_forward::primitive_desc;

class MklDnnConvUtil {
 protected:
  OpKernelContext* context_;  // We don't own this.
  std::vector<int32> strides_;
  std::vector<int32> dilations_;
  Padding padding_;
  TensorFormat data_format_;

 public:
  MklDnnConvUtil(OpKernelContext* context, const std::vector<int32>& strides,
                 Padding pad, TensorFormat fm,
                 const std::vector<int32>& dilations, bool is_depthwise = false)
      : context_(context),
        strides_(strides),
        dilations_(dilations),
        padding_(pad),
        data_format_(fm) {}

  virtual ~MklDnnConvUtil() { context_ = nullptr; }

  // Calculate Convolution strides
  virtual inline void GetStridesInMklOrder(memory::dims* strides) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSmklPSmkl_conv_opsDTh mht_0(mht_0_v, 244, "", "./tensorflow/core/kernels/mkl/mkl_conv_ops.h", "GetStridesInMklOrder");

    // For now we take the stride from the second and third dimensions only
    // (we do not support striding on the batch or depth dimension).
    DCHECK(strides);
    if (strides_.size() == 4) {
      int stride_rows = GetTensorDim(strides_, data_format_, 'H');
      int stride_cols = GetTensorDim(strides_, data_format_, 'W');
      *strides = {stride_rows, stride_cols};
    } else if (strides_.size() == 5) {
      int stride_planes = GetTensorDim(strides_, data_format_, '0');
      int stride_rows = GetTensorDim(strides_, data_format_, '1');
      int stride_cols = GetTensorDim(strides_, data_format_, '2');
      *strides = {stride_planes, stride_rows, stride_cols};
    }
  }

  // Calculate Convolution dilations
  virtual inline void GetDilationsInMklOrder(memory::dims* dilations) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSmklPSmkl_conv_opsDTh mht_1(mht_1_v, 264, "", "./tensorflow/core/kernels/mkl/mkl_conv_ops.h", "GetDilationsInMklOrder");

    // For now we take the dilation from the second and third dimensions only
    // (we do not support dilation on the batch or depth dimension).
    DCHECK(dilations);
    if (dilations_.size() == 4) {
      int dilations_rows = GetTensorDim(dilations_, data_format_, 'H');
      int dilations_cols = GetTensorDim(dilations_, data_format_, 'W');
      *dilations = {dilations_rows, dilations_cols};
    } else if (dilations_.size() == 5) {
      int dilations_planes = GetTensorDim(dilations_, data_format_, '0');
      int dilations_rows = GetTensorDim(dilations_, data_format_, '1');
      int dilations_cols = GetTensorDim(dilations_, data_format_, '2');
      *dilations = {dilations_planes, dilations_rows, dilations_cols};
    }
  }

  // Calculate Convolution input size in oneDNN order. oneDNN
  // requires input in NCHW/NCDHW format. Function does not return anything.
  // But errors arising from sanity checks are returned in context's
  // status.
  virtual inline void GetInputSizeInMklOrder(const TensorShape& input_shape,
                                             memory::dims* input_dims) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSmklPSmkl_conv_opsDTh mht_2(mht_2_v, 288, "", "./tensorflow/core/kernels/mkl/mkl_conv_ops.h", "GetInputSizeInMklOrder");

#define CHECK_BOUNDS(val, err_msg)                                     \
  do {                                                                 \
    OP_REQUIRES(context_,                                              \
                FastBoundsCheck(val, std::numeric_limits<int>::max()), \
                errors::InvalidArgument(err_msg));                     \
  } while (0)

    DCHECK(input_dims);

    // Input channel
    int64 input_depth_raw = GetTensorDim(input_shape, data_format_, 'C');
    int input_depth = static_cast<int>(input_depth_raw);

    // Input batch
    int64 input_batch_raw = GetTensorDim(input_shape, data_format_, 'N');
    CHECK_BOUNDS(input_batch_raw, "Input batch too large");
    int input_batch = static_cast<int>(input_batch_raw);

    if (strides_.size() == 4) {  // NCHW format for Conv2D
      // Input rows/height
      int64 input_rows_raw = GetTensorDim(input_shape, data_format_, 'H');
      CHECK_BOUNDS(input_rows_raw, "Input rows too large");
      int input_rows = static_cast<int>(input_rows_raw);

      // Input columns/width
      int64 input_cols_raw = GetTensorDim(input_shape, data_format_, 'W');
      CHECK_BOUNDS(input_cols_raw, "Input cols too large");
      int input_cols = static_cast<int>(input_cols_raw);

      // oneDNN always requires input in NCHW format Conv2D.
      std::vector<memory::dim> input_sizes(4, -1);
      input_sizes[MklDnnDims::Dim_N] = input_batch;
      input_sizes[MklDnnDims::Dim_C] = input_depth;
      input_sizes[MklDnnDims::Dim_H] = input_rows;
      input_sizes[MklDnnDims::Dim_W] = input_cols;
      *input_dims = input_sizes;
    } else if (strides_.size() == 5) {  // NCDHW format for Conv3D
      // Input planes/third-dimension
      int64 input_planes_raw = GetTensorDim(input_shape, data_format_, '0');
      CHECK_BOUNDS(input_planes_raw, "Input depth too large");
      int input_planes = static_cast<int>(input_planes_raw);

      // Input rows/height
      int64 input_rows_raw = GetTensorDim(input_shape, data_format_, '1');
      CHECK_BOUNDS(input_rows_raw, "Input rows too large");
      int input_rows = static_cast<int>(input_rows_raw);

      // Input columns/width
      int64 input_cols_raw = GetTensorDim(input_shape, data_format_, '2');
      CHECK_BOUNDS(input_cols_raw, "Input cols too large");
      int input_cols = static_cast<int>(input_cols_raw);

      // oneDNN always requires input in NCDHW format for Conv3D.
      std::vector<memory::dim> input_sizes(5, -1);
      input_sizes[MklDnnDims3D::Dim3d_N] = input_batch;
      input_sizes[MklDnnDims3D::Dim3d_C] = input_depth;
      input_sizes[MklDnnDims3D::Dim3d_D] = input_planes;
      input_sizes[MklDnnDims3D::Dim3d_H] = input_rows;
      input_sizes[MklDnnDims3D::Dim3d_W] = input_cols;
      *input_dims = input_sizes;
    }
#undef CHECK_BOUNDS
  }

  // Calculate Convolution filter size in oneDNN order.
  // oneDNN requires filter in OIHW (Conv2D) or OIDHW (Conv3D) format.
  // Function does not return anything.
  // But errors arising from sanity checks are returned in context's
  // status. This function differs from GetConvFilterSizeInMklOrder in
  // parameter for input - it accepts src_shape since Convolution Backward
  // Input gets shape of input tensor rather than actual tensor (Convolution
  // forward gets actual tensor as input).
  //
  // TODO(intel-tf): Add similar function for input and filter in MklShape.
  virtual inline void GetFilterSizeInMklOrder(const TensorShape& input_shape,
                                              const TensorShape& filter_shape,
                                              memory::dims* filter_dims,
                                              bool* is_grouped_convolution,
                                              bool is_depthwise) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSmklPSmkl_conv_opsDTh mht_3(mht_3_v, 370, "", "./tensorflow/core/kernels/mkl/mkl_conv_ops.h", "GetFilterSizeInMklOrder");

    DCHECK(filter_dims);

    OP_REQUIRES(context_, filter_shape.dims() == strides_.size(),
                errors::InvalidArgument((strides_.size() == 4)
                                            ? "filter must be 4-dimensional: "
                                            : "filter must be 5-dimensional: ",
                                        filter_shape.DebugString()));

    for (int i = 0; i < ((strides_.size() == 4) ? 3 : 5); i++) {
      OP_REQUIRES(context_,
                  FastBoundsCheck(filter_shape.dim_size(i),
                                  std::numeric_limits<int>::max()),
                  errors::InvalidArgument("filter too large"));
    }

    int input_depth = GetTensorDim(input_shape, data_format_, 'C');

    if (strides_.size() == 4) {  // Conv2D
      // TF filter is always in (rows, cols, in_depth, out_depth) order.
      int filter_rows =
          static_cast<int>(filter_shape.dim_size(TF_2DFILTER_DIM_H));
      int filter_cols =
          static_cast<int>(filter_shape.dim_size(TF_2DFILTER_DIM_W));
      int filter_in_depth =
          static_cast<int>(filter_shape.dim_size(TF_2DFILTER_DIM_I));
      int filter_out_depth =
          static_cast<int>(filter_shape.dim_size(TF_2DFILTER_DIM_O));
      OP_REQUIRES(context_, input_depth % filter_in_depth == 0,
                  errors::InvalidArgument(
                      "input depth must be evenly divisible by filter depth: ",
                      input_depth, " vs ", filter_in_depth));
      *is_grouped_convolution = filter_in_depth != input_depth;
      int group_count = input_depth / filter_in_depth;
      // oneDNN always needs filter in OIHW format for regular convolutions
      // and GOIHW for grouped/depthwise convolutions,
      // OIHW = (out_depth, in_depth, rows, cols)
      // GOIHW = (group, out_depth, in_depth, rows, cols)
      // Specifically for depthwise G=filter_indepth, O=filter_outdepth, I=1
      if (is_depthwise) {
        std::vector<memory::dim> filter_sizes(5, -1);
        filter_sizes[MKL_GROUP_FILTER_DIM_G] = filter_in_depth;
        filter_sizes[MKL_GROUP_FILTER_DIM_O] = filter_out_depth;
        filter_sizes[MKL_GROUP_FILTER_DIM_I] = 1;
        filter_sizes[MKL_GROUP_FILTER_DIM_H] = filter_rows;
        filter_sizes[MKL_GROUP_FILTER_DIM_W] = filter_cols;
        *filter_dims = filter_sizes;
      } else if (*is_grouped_convolution) {
        // TODO(intel-tf): Directly set filter_dims. Same for other places.
        std::vector<memory::dim> filter_sizes(5, -1);
        filter_sizes[MKL_GROUP_FILTER_DIM_G] = group_count;
        filter_sizes[MKL_GROUP_FILTER_DIM_O] = filter_out_depth / group_count;
        filter_sizes[MKL_GROUP_FILTER_DIM_I] = filter_in_depth;
        filter_sizes[MKL_GROUP_FILTER_DIM_H] = filter_rows;
        filter_sizes[MKL_GROUP_FILTER_DIM_W] = filter_cols;
        *filter_dims = filter_sizes;
      } else {
        std::vector<memory::dim> filter_sizes(4, -1);
        filter_sizes[MklDnnDims::Dim_O] = filter_out_depth;
        filter_sizes[MklDnnDims::Dim_I] = filter_in_depth;
        filter_sizes[MklDnnDims::Dim_H] = filter_rows;
        filter_sizes[MklDnnDims::Dim_W] = filter_cols;
        *filter_dims = filter_sizes;
      }
    } else {  // Conv3D
      OP_REQUIRES(context_, input_depth == filter_shape.dim_size(3),
                  errors::InvalidArgument(
                      "input and filter must have the same depth: ",
                      input_depth, " vs ", filter_shape.dim_size(3)));

      // TF filter is always in (planes, rows, cols, in_depth, out_depth) order.
      int filter_planes =
          static_cast<int>(filter_shape.dim_size(TF_3DFILTER_DIM_P));
      int filter_rows =
          static_cast<int>(filter_shape.dim_size(TF_3DFILTER_DIM_H));
      int filter_cols =
          static_cast<int>(filter_shape.dim_size(TF_3DFILTER_DIM_W));
      int filter_in_depth =
          static_cast<int>(filter_shape.dim_size(TF_3DFILTER_DIM_I));
      int filter_out_depth =
          static_cast<int>(filter_shape.dim_size(TF_3DFILTER_DIM_O));

      // oneDNN always needs filter in OIDHW format.
      // OIDHW = (out_depth, in_depth, planes, rows, cols)
      std::vector<memory::dim> filter_sizes(5, -1);
      filter_sizes[MklDnnDims3D::Dim3d_O] = filter_out_depth;
      filter_sizes[MklDnnDims3D::Dim3d_I] = filter_in_depth;
      filter_sizes[MklDnnDims3D::Dim3d_D] = filter_planes;
      filter_sizes[MklDnnDims3D::Dim3d_H] = filter_rows;
      filter_sizes[MklDnnDims3D::Dim3d_W] = filter_cols;
      *filter_dims = filter_sizes;
    }
  }

  // Calculate Convolution filter size in oneDNN order.
  // oneDNN requires filter in OIHW (Conv2D) or OIDHW(Conv3D format.
  // Function does not return anything. But errors arising from sanity
  // checks are returned in context's status.
  virtual inline void GetFilterSizeInMklOrder(size_t src_index,
                                              size_t filter_index,
                                              memory::dims* filter_dims,
                                              bool* is_grouped_convolution,
                                              bool is_depthwise) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSmklPSmkl_conv_opsDTh mht_4(mht_4_v, 475, "", "./tensorflow/core/kernels/mkl/mkl_conv_ops.h", "GetFilterSizeInMklOrder");

    DCHECK(filter_dims);
    GetFilterSizeInMklOrder(GetTfShape(context_, src_index),
                            GetTfShape(context_, filter_index), filter_dims,
                            is_grouped_convolution, is_depthwise);
  }

  // Calculate Bias size for 2D or 3D Convolution. Function does not
  // return anything, but may set an error in context status.
  virtual inline void GetBiasSizeInMklOrder(size_t bias_index,
                                            memory::dims* bias_dims) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSmklPSmkl_conv_opsDTh mht_5(mht_5_v, 488, "", "./tensorflow/core/kernels/mkl/mkl_conv_ops.h", "GetBiasSizeInMklOrder");

    const Tensor& bias = MklGetInput(context_, bias_index);
    if (bias.dims() > 1) {
      if (strides_.size() == 4) {
        OP_REQUIRES(
            context_, bias.dims() <= 4,
            errors::InvalidArgument("For NHWC format, bias should have  "
                                    "4 or less dimensions",
                                    bias.shape().DebugString()));
      } else if (strides_.size() == 5) {
        OP_REQUIRES(
            context_, bias.dims() <= 5,
            errors::InvalidArgument("For NDHWC format, bias should have  "
                                    "5 or less dimensions",
                                    bias.shape().DebugString()));
      }
      // Make sure all the dims except channel(last) is 1
      for (int i = 0; i < bias.dims() - 1; i++) {
        OP_REQUIRES(
            context_, bias.dim_size(i) == 1,
            errors::InvalidArgument("For bias_dims > 1, all except the last "
                                    "dimension (channel) must be 1: ",
                                    bias.shape().DebugString()));
      }
      *bias_dims = {static_cast<int>(bias.dim_size(bias.dims() - 1))};
    } else {
      *bias_dims = {static_cast<int>(bias.dim_size(0))};
    }
  }

  // Function to calculate output and padding size for 2D/3D convolution.
  //
  // Calculate output shape of Convolution in oneDNN and TensorFlow order.
  // oneDNN uses NCHW(Conv2D) or NCDHW(Conv3D) for output order.
  // But TensorFlow output will be in NHWC||NCHW(Conv2D) or
  // NDHWC||NCDHW(Conv3D) format depending on data format.
  // Function also calculates left, right, top and bottom pads.
  // Function does not return any status which is set with context status.
  //
  // TODO(intel-tf): Add similar function for input and filter in MklShape.
  virtual inline void GetOutputAndPadSizeInMklOrder(
      const TensorShape& input_shape, const TensorShape& filter_shape,
      const memory::dims& strides, const memory::dims& dilations,
      memory::dims* output_dims_tf_order, memory::dims* output_dims_mkl_order,
      memory::dims* pad_l, memory::dims* pad_r, bool is_grouped_convolution,
      bool pad_enabled = false, bool is_depthwise = false) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSmklPSmkl_conv_opsDTh mht_6(mht_6_v, 536, "", "./tensorflow/core/kernels/mkl/mkl_conv_ops.h", "GetOutputAndPadSizeInMklOrder");

    DCHECK(output_dims_tf_order);
    DCHECK(output_dims_mkl_order);
    DCHECK(pad_l);
    DCHECK(pad_r);

    bool is_conv2d = (strides_.size() == 4);
    int input_planes, input_rows, input_cols;
    if (is_conv2d) {
      input_rows = GetTensorDim(input_shape, data_format_, 'H');
      input_cols = GetTensorDim(input_shape, data_format_, 'W');
    } else {
      input_planes = GetTensorDim(input_shape, data_format_, '0');
      input_rows = GetTensorDim(input_shape, data_format_, '1');
      input_cols = GetTensorDim(input_shape, data_format_, '2');
    }

    // Filter dimension
    // Conv2D:
    //    First dimension: rows/height.
    //    Second dimension: cols/width.
    // Conv3D:
    //    First dimension: planes/depth.
    //    Second dimension: rows/height.
    //    Third dimension: cols/width.

    int filter_planes, filter_rows, filter_cols;
    if (is_conv2d) {
      filter_rows = filter_shape.dim_size(TF_2DFILTER_DIM_H);
      filter_cols = filter_shape.dim_size(TF_2DFILTER_DIM_W);
    } else {
      filter_planes = filter_shape.dim_size(TF_3DFILTER_DIM_P);
      filter_rows = filter_shape.dim_size(TF_3DFILTER_DIM_H);
      filter_cols = filter_shape.dim_size(TF_3DFILTER_DIM_W);
    }

    int stride_planes, stride_rows, stride_cols;
    int dilation_planes, dilation_rows, dilation_cols;
    if (is_conv2d) {
      // Conv2D stride is a vector of 2 elements: {s_r, s_c}
      stride_rows = strides[0];
      stride_cols = strides[1];
      dilation_rows = dilations[0];
      dilation_cols = dilations[1];
    } else {
      // Conv3D stride is a vector of 3 elements: {s_d, s_r, s_c}
      stride_planes = strides[0];
      stride_rows = strides[1];
      stride_cols = strides[2];
      dilation_planes = dilations[0];
      dilation_rows = dilations[1];
      dilation_cols = dilations[2];
    }

    // Output batch is same as input batch.
    int out_batch = GetTensorDim(input_shape, data_format_, 'N');
    int out_depth;

    // TODO(intel-tf) add support for 3-D Depthwise

    // Output depth is same as last dimension for filters for regular
    // convolutions and group convolutions. For depthwise it is in_depth *
    // channel_multiplier. The channel_multiplier is the last dimension of
    // TF filter for depthwise convolutions.
    if (is_depthwise) {
      out_depth = (filter_shape.dim_size(TF_2DFILTER_DIM_I) *
                   filter_shape.dim_size(TF_2DFILTER_DIM_O));
    } else if (is_grouped_convolution) {
      out_depth = filter_shape.dim_size(TF_2DFILTER_DIM_O);
    } else {
      out_depth = filter_shape.dim_size(
          is_conv2d ? static_cast<int>(TF_2DFILTER_DIM_O)
                    : static_cast<int>(TF_3DFILTER_DIM_O));
    }

    int64 out_rows = 0, out_cols = 0, out_planes = 0;
    int64 pad_top = 0, pad_bottom = 0, pad_left = 0, pad_right = 0;
    int64 pad_front, pad_back;

    if (is_conv2d) {
      Padding padding_type;
      if (pad_enabled) {
        padding_type = Padding::EXPLICIT;
        pad_top = static_cast<int64_t>((*pad_l)[0]);
        pad_left = static_cast<int64_t>((*pad_l)[1]);
        pad_bottom = static_cast<int64_t>((*pad_r)[0]);
        pad_right = static_cast<int64_t>((*pad_r)[1]);
      } else {
        padding_type = padding_;
      }
      OP_REQUIRES_OK(context_,
                     GetWindowedOutputSizeVerboseV2(
                         input_rows, filter_rows, dilation_rows, stride_rows,
                         padding_type, &out_rows, &pad_top, &pad_bottom));
      OP_REQUIRES_OK(context_,
                     GetWindowedOutputSizeVerboseV2(
                         input_cols, filter_cols, dilation_cols, stride_cols,
                         padding_type, &out_cols, &pad_left, &pad_right));
    } else {
      Padding padding_type;
      if (pad_enabled) {
        padding_type = Padding::EXPLICIT;
        pad_front = static_cast<int64>((*pad_l)[0]);
        pad_top = static_cast<int64>((*pad_l)[1]);
        pad_left = static_cast<int64>((*pad_l)[2]);
        pad_back = static_cast<int64>((*pad_r)[0]);
        pad_bottom = static_cast<int64>((*pad_r)[1]);
        pad_right = static_cast<int64>((*pad_r)[2]);
      } else {
        padding_type = padding_;
      }
      OP_REQUIRES_OK(context_, GetWindowedOutputSizeVerboseV2(
                                   input_planes, filter_planes, dilation_planes,
                                   stride_planes, padding_type, &out_planes,
                                   &pad_front, &pad_back));
      OP_REQUIRES_OK(context_,
                     GetWindowedOutputSizeVerboseV2(
                         input_rows, filter_rows, dilation_rows, stride_rows,
                         padding_type, &out_rows, &pad_top, &pad_bottom));
      OP_REQUIRES_OK(context_,
                     GetWindowedOutputSizeVerboseV2(
                         input_cols, filter_cols, dilation_cols, stride_cols,
                         padding_type, &out_cols, &pad_left, &pad_right));
    }

    if (is_conv2d) {
      // If pad_enabled, i.e., pad and conv op are fused, then
      // all pads are already passed from pad op through
      // *pad_l and *pad_r and they don't need to be set here.
      if (!pad_enabled) {
        *pad_l = {static_cast<int>(pad_top), static_cast<int>(pad_left)};
        *pad_r = {static_cast<int>(pad_bottom), static_cast<int>(pad_right)};
      }
    } else {
      // If pad_enabled, i.e., pad and conv op are fused, then
      // all pads are already passed from pad op through
      // *pad_l and *pad_r and they don't need to be set here.
      if (!pad_enabled) {
        *pad_l = {static_cast<int>(pad_front), static_cast<int>(pad_top),
                  static_cast<int>(pad_left)};
        *pad_r = {static_cast<int>(pad_back), static_cast<int>(pad_bottom),
                  static_cast<int>(pad_right)};
      }
    }
    // Tensorflow output is in data_format order.
    //     Conv2D: NHWC or NCHW
    //     Conv3D: NDHWC or NCDHW
    // oneDNN uses asymmetric padding.
    TensorShape out_shape =
        is_conv2d
            ? ShapeFromFormat(data_format_, out_batch, out_rows, out_cols,
                              out_depth)
            : ShapeFromFormat(data_format_, out_batch,
                              {{out_planes, out_rows, out_cols}}, out_depth);
    *output_dims_tf_order = TFShapeToMklDnnDims(out_shape);
    if (is_grouped_convolution) {
      int out_depth = GetTensorDim(out_shape, data_format_, 'C');
      int input_depth = GetTensorDim(input_shape, data_format_, 'C');
      int filter_in_depth =
          static_cast<int>(filter_shape.dim_size(TF_2DFILTER_DIM_I));
      int num_groups = input_depth / filter_in_depth;
      OP_REQUIRES(
          context_, out_depth % num_groups == 0 && out_depth >= num_groups,
          errors::InvalidArgument(
              "output depth must be evenly divisible by number of groups: ",
              out_depth, " vs ", num_groups));
    }
    if (is_conv2d) {
      // For Conv2D, oneDNN always needs output in NCHW format.
      std::vector<memory::dim> output_sizes(4, -1);
      output_sizes[MklDnnDims::Dim_N] = out_batch;
      output_sizes[MklDnnDims::Dim_C] = out_depth;
      output_sizes[MklDnnDims::Dim_H] = static_cast<int>(out_rows);
      output_sizes[MklDnnDims::Dim_W] = static_cast<int>(out_cols);
      *output_dims_mkl_order = output_sizes;
    } else {
      std::vector<memory::dim> output_sizes(5, -1);
      output_sizes[MklDnnDims3D::Dim3d_N] = out_batch;
      output_sizes[MklDnnDims3D::Dim3d_C] = out_depth;
      output_sizes[MklDnnDims3D::Dim3d_D] = static_cast<int>(out_planes);
      output_sizes[MklDnnDims3D::Dim3d_H] = static_cast<int>(out_rows);
      output_sizes[MklDnnDims3D::Dim3d_W] = static_cast<int>(out_cols);
      *output_dims_mkl_order = output_sizes;
    }
  }

  // Calculate output and pad size of forward Convolution operator.
  // See comment on GetConvOutputAndPadSizeInMklOrder for parameters.
  //
  // Function does not return anything, but sets error in context status.
  inline void GetOutputAndPadSizeInMklOrder(
      size_t src_index, size_t filter_index, const memory::dims& strides,
      const memory::dims& dilations, memory::dims* output_dims_tf_order,
      memory::dims* output_dims_mkl_order, memory::dims* pad_l,
      memory::dims* pad_r, bool is_grouped_convolution, bool is_depthwise) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSmklPSmkl_conv_opsDTh mht_7(mht_7_v, 733, "", "./tensorflow/core/kernels/mkl/mkl_conv_ops.h", "GetOutputAndPadSizeInMklOrder");

    DCHECK(output_dims_tf_order);
    DCHECK(output_dims_mkl_order);
    DCHECK(pad_l);
    DCHECK(pad_r);

    auto input_tf_shape = GetTfShape(context_, src_index);
    auto filter_tf_shape = GetTfShape(context_, filter_index);

    if (strides_.size() == 4) {
      // Conv2D
      OP_REQUIRES(context_, input_tf_shape.dims() == 4,
                  errors::InvalidArgument("input must be 4-dimensional",
                                          input_tf_shape.DebugString()));
    } else {
      // Conv3D
      OP_REQUIRES(context_, input_tf_shape.dims() == 5,
                  errors::InvalidArgument("input must be 5-dimensional",
                                          input_tf_shape.DebugString()));
    }

    GetOutputAndPadSizeInMklOrder(input_tf_shape, filter_tf_shape, strides,
                                  dilations, output_dims_tf_order,
                                  output_dims_mkl_order, pad_l, pad_r,
                                  is_grouped_convolution, is_depthwise);
  }

  // Wrapper function to calculate input, filter, and output sizes of
  // Conv2D/Conv3D in MKL order:
  //     Conv2D: NCHW for input and output; OIHW for filter.
  //     Conv3D: NCDHW for input and output; OIDHW for filter.
  // Function also calculates output shape in Tensorflow order.
  // Additionally, it also calculates strides and paddings.
  //
  // Function does not return anything, but sets error in context status.
  inline void GetConvFwdSizesInMklOrder(
      const TensorShape& input_shape, const TensorShape& filter_shape,
      memory::dims* input_dims, memory::dims* filter_dims,
      memory::dims* strides, memory::dims* dilations,
      memory::dims* output_dims_tf_order, memory::dims* output_dims_mkl_order,
      memory::dims* pad_l, memory::dims* pad_r, bool* is_grouped_convolution,
      bool pad_enabled = false, bool is_depthwise = false) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSmklPSmkl_conv_opsDTh mht_8(mht_8_v, 777, "", "./tensorflow/core/kernels/mkl/mkl_conv_ops.h", "GetConvFwdSizesInMklOrder");

    DCHECK(input_dims);
    DCHECK(filter_dims);
    DCHECK(strides);
    DCHECK(dilations);
    DCHECK(output_dims_tf_order);
    DCHECK(output_dims_mkl_order);
    DCHECK(pad_l);
    DCHECK(pad_r);

    GetInputSizeInMklOrder(input_shape, input_dims);
    if (!context_->status().ok()) return;
    GetFilterSizeInMklOrder(input_shape, filter_shape, filter_dims,
                            is_grouped_convolution, is_depthwise);
    if (!context_->status().ok()) return;
    GetStridesInMklOrder(strides);
    GetDilationsInMklOrder(dilations);
    GetOutputAndPadSizeInMklOrder(
        input_shape, filter_shape, *strides, *dilations, output_dims_tf_order,
        output_dims_mkl_order, pad_l, pad_r, *is_grouped_convolution,
        pad_enabled, is_depthwise);
    if (!context_->status().ok()) return;
  }
};

/////////////////////////////////////////////////////////////////////
///  Common class that implements ConvBackpropFilter and Input
/////////////////////////////////////////////////////////////////////

template <typename Device, class T, bool is_depthwise>
class MklConvBackpropCommonOp : public OpKernel {
 public:
  ~MklConvBackpropCommonOp() {}
  explicit MklConvBackpropCommonOp(OpKernelConstruction* context)
      : OpKernel(context) {
    string data_format_str;
    OP_REQUIRES_OK(context, context->GetAttr("data_format", &data_format_str));
    OP_REQUIRES(context, FormatFromString(data_format_str, &data_format_),
                errors::InvalidArgument("Invalid data format"));
    OP_REQUIRES_OK(context, context->GetAttr("strides", &strides_));
    int stride_n = GetTensorDim(strides_, data_format_, 'N');
    int stride_c = GetTensorDim(strides_, data_format_, 'C');
    OP_REQUIRES(
        context, (stride_n == 1 && stride_c == 1),
        errors::InvalidArgument("Current implementation does not yet support "
                                "strides in the batch and depth dimensions."));

    // Depthwise Convolution doesn't have dilation parameter
    if (!is_depthwise) {
      OP_REQUIRES_OK(context, context->GetAttr("dilations", &dilations_));
      if (strides_.size() == 4) {
        // Check Conv2D dilations
        OP_REQUIRES(
            context, dilations_.size() == 4,
            errors::InvalidArgument("Sliding window dilations field must "
                                    "specify 4 dimensions"));
        int dilation_n = GetTensorDim(dilations_, data_format_, 'N');
        int dilation_c = GetTensorDim(dilations_, data_format_, 'C');
        int dilation_h = GetTensorDim(dilations_, data_format_, 'H');
        int dilation_w = GetTensorDim(dilations_, data_format_, 'W');
        OP_REQUIRES(context, (dilation_n == 1 && dilation_c == 1),
                    errors::InvalidArgument(
                        "Current implementation does not yet support "
                        "dilations in the batch and depth dimensions."));
        OP_REQUIRES(
            context, dilation_h > 0 && dilation_w > 0,
            errors::InvalidArgument("Dilated rates should be larger than 0."));
      }
    } else {
      // Set dilations as 1 for depthwise conv
      // for future support to align with Tensorflow
      dilations_ = {1, 1, 1, 1};
    }

    OP_REQUIRES_OK(context, context->GetAttr("padding", &padding_));
  }

 protected:
  // data members accessible to derived classes.
  std::vector<int32> dilations_;
  std::vector<int32> strides_;
  Padding padding_;
  TensorFormat data_format_;  // NCHW or NHWC
};

/////////////////////////////////////////////////////////////////////
///  Dummy Mkl op that is just used for operators that are intermediate
///  output of node fusion in the graph
/////////////////////////////////////////////////////////////////////

template <typename Device, typename T>
class MklDummyOp : public OpKernel {
 public:
  ~MklDummyOp() {}

  explicit MklDummyOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSmklPSmkl_conv_opsDTh mht_9(mht_9_v, 877, "", "./tensorflow/core/kernels/mkl/mkl_conv_ops.h", "Compute");

    TF_CHECK_OK(
        errors::Unimplemented("This is a dummy op."
                              "It should not have been invoked."));
  }
};

}  // namespace tensorflow

#endif  // INTEL_MKL
#endif  // TENSORFLOW_CORE_KERNELS_MKL_MKL_CONV_OPS_H_
