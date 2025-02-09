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
class MHTracer_DTPStensorflowPScorePSkernelsPSmklPSmkl_reshape_opDTcc {
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
   MHTracer_DTPStensorflowPScorePSkernelsPSmklPSmkl_reshape_opDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSkernelsPSmklPSmkl_reshape_opDTcc() {
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

#ifdef INTEL_MKL

#include <memory>

#include "dnnl.hpp"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/util/mkl_util.h"

using dnnl::stream;

namespace tensorflow {

using CPUDevice = Eigen::ThreadPoolDevice;

template <typename Device, typename T>
class MklReshapeOp : public OpKernel {
 public:
  explicit MklReshapeOp(OpKernelConstruction* context) : OpKernel(context) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSmklPSmkl_reshape_opDTcc mht_0(mht_0_v, 207, "", "./tensorflow/core/kernels/mkl/mkl_reshape_op.cc", "MklReshapeOp");
}

 private:
  // When the input tensor is in MKL layout and we are reshaping the tensor to a
  // different shape than its actual shape, then we use oneDNN reorder primitive
  // to put tensor back in Tensorflow layout. But we can skip this reordering
  // some times. This function checks for all such cases.
  bool SkipReorder(const MklDnnShape& mkl_shape_input,
                   const TensorShape& reshape_to) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSmklPSmkl_reshape_opDTcc mht_1(mht_1_v, 218, "", "./tensorflow/core/kernels/mkl/mkl_reshape_op.cc", "SkipReorder");

    CHECK_EQ(mkl_shape_input.IsMklTensor(), true);

    // If Tensorflow's data format and the underlying format maintained by
    // oneDNN are equivalent (both are NHWC or both are NCHW), then we can
    // safely return true.
    // TODO(intel-tf): In the future, do not force skip reorder for all blocked
    // format. Use blocking_desc_is_equal() for checking all the stride arrays
    // in mkl-dnn/blob/master/src/common/type_helpers.hpp
    return (mkl_shape_input.GetTfDataFormat() ==
            MklTensorFormat::FORMAT_BLOCKED);
  }

 public:
  void Compute(OpKernelContext* context) override {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSmklPSmkl_reshape_opDTcc mht_2(mht_2_v, 235, "", "./tensorflow/core/kernels/mkl/mkl_reshape_op.cc", "Compute");

    const Tensor& input_tensor = MklGetInput(context, 0);
    const Tensor& sizes = MklGetInput(context, 1);

    MklDnnShape mkl_shape_input;
    GetMklShape(context, kInputSlotIdx, &mkl_shape_input);
    bool input_in_mkl_format = mkl_shape_input.IsMklTensor();
    TensorShape input_shape = input_in_mkl_format ? mkl_shape_input.GetTfShape()
                                                  : input_tensor.shape();
    const int64 nelems = input_in_mkl_format ? input_shape.num_elements()
                                             : input_tensor.NumElements();

    // Preliminary validation of sizes.
    OP_REQUIRES(context, TensorShapeUtils::IsVector(sizes.shape()),
                errors::InvalidArgument("sizes input must be 1-D, not shape ",
                                        sizes.shape().DebugString()));

    // Compute the output shape.  Determine product of specified
    // dimensions, and find the index of the unspecified one.
    TensorShape shape;
    int64 product = 1;
    int unknown_index = -1;
    bool sizes_has_zero_dim = false;
    switch (sizes.dtype()) {
      case DT_INT32:
        OP_REQUIRES_OK(context,
                       ValidateSizes<int32>(sizes, &product, &unknown_index,
                                            &shape, &sizes_has_zero_dim));
        break;
      case DT_INT64:
        OP_REQUIRES_OK(context,
                       ValidateSizes<int64_t>(sizes, &product, &unknown_index,
                                              &shape, &sizes_has_zero_dim));
        break;
      default:
        context->CtxFailure(errors::InvalidArgument(
            "desired shape must be a DT_INT32 or DT_INT64 vector, not a ",
            DataTypeString(sizes.dtype())));
        return;
    }
    if (unknown_index != -1) {
      int64 input_num_elements = 1;
      bool input_has_zero_dim = false;
      for (int dim = 0; dim < input_shape.dims(); ++dim) {
        // For zero dimension, we don't count it into `input_num_elements`
        // unless `sizes` has no zero dimension, so we are still able to
        // infer shapes for other dimensions.
        if (input_shape.dim_size(dim) > 0 || !sizes_has_zero_dim) {
          input_num_elements *= input_shape.dim_size(dim);
        } else {
          input_has_zero_dim = true;
        }
      }

      const int64 missing = input_num_elements / product;
      if (!input_has_zero_dim) {
        OP_REQUIRES(
            context, product * missing == input_num_elements,
            errors::InvalidArgument(
                "Input to reshape is a tensor with ", input_num_elements,
                " values, but the requested shape requires a multiple of ",
                product));
      }
      shape.set_dim(unknown_index, missing);
    }
    OP_REQUIRES(
        context, shape.num_elements() == nelems,
        errors::InvalidArgument("Input to reshape is a tensor with ", nelems,
                                " values, but the requested shape has ",
                                shape.num_elements()));

    if (input_in_mkl_format && !SkipReorder(mkl_shape_input, shape)) {
      TensorShape& shape_to = shape;
      TensorShape shape_from = mkl_shape_input.GetTfShape();
      if (shape_from == shape_to) {
        CopyMklTensorInToOut(context, kInputSlotIdx, kOutputSlotIdx);
        return;
      } else {
        try {
          auto cpu_engine = engine(engine::kind::cpu, 0);
          MklDnnData<T> dnn_data_input(&cpu_engine);
          // Reshape is just a logical view change operation for a tensor.
          // It does not change underlying layout. But oneDNN may maintain
          // tensor data in different layout than that specified by Tensorflow.
          // If oneDNN maintains input tensor in different layout than that
          // specified by Tensorflow, we will need to reorder tensor and then
          // put it in the shape expected by Tensorflow.

          // If dimensions that are being expanded or collapsed are not
          // maintained contiguously by oneDNN, then we use reorder.

          // Get Mkl layout of input tensor.
          auto input_mkl_md = mkl_shape_input.GetMklLayout();
          // Set input Mkl layout as the user layout.
          dnn_data_input.SetUsrMem(input_mkl_md, &input_tensor);
          // Get expected Tensorflow layout of input tensor.
          auto output_tf_md = mkl_shape_input.GetTfLayout();

          Tensor* output_tensor = nullptr;
          MklDnnShape mkl_shape_output;
          mkl_shape_output.SetMklTensor(false);
          // We allocate output tensor in the shape expected by Reshape.
          AllocateOutputSetMklShape(context, kOutputSlotIdx, &output_tensor,
                                    shape_to, mkl_shape_output);

          // Insert reorder between Mkl layout and TensorFlow layout if
          // needed. If reorder is not needed but reshape is needed (since
          // shape_from != shape_to), then we just copy input tensor to
          // output tensor with target shape (we cannot forward Mkl layout
          // in such case because shape has changed.)
          if (dnn_data_input.CheckReorderToOpMem(output_tf_md, output_tensor,
                                                 context)) {
          } else {
            OP_REQUIRES(context,
                        output_tensor->CopyFrom(input_tensor, shape_to),
                        errors::InvalidArgument("invalid input tensor shape"));
          }
          return;
        } catch (dnnl::error& e) {
          string error_msg = "Status: " + std::to_string(e.status) +
                             ", message: " + string(e.message) + ", in file " +
                             string(__FILE__) + ":" + std::to_string(__LINE__);
          OP_REQUIRES_OK(
              context,
              errors::Aborted("Operation received an exception:", error_msg));
        }
      }
    } else {
      // If input tensor is not in Mkl format, then just copy Tensorflow tensor
      // to output with specified shape.
      CopyTfTensorInToOutWithShape(context, kInputSlotIdx, kOutputSlotIdx,
                                   shape);
    }
  }

 private:
  const int kInputSlotIdx = 0;
  const int kOutputSlotIdx = 0;

  template <typename Tshape>
  Status ValidateSizes(const Tensor& sizes, int64* product, int* unknown_index,
                       TensorShape* shape, bool* has_zero_dim) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSmklPSmkl_reshape_opDTcc mht_3(mht_3_v, 379, "", "./tensorflow/core/kernels/mkl/mkl_reshape_op.cc", "ValidateSizes");

    *product = 1;
    *unknown_index = -1;
    *has_zero_dim = false;
    const int64 num_dims = sizes.NumElements();
    auto Svec = sizes.flat<Tshape>();
    for (int d = 0; d < num_dims; ++d) {
      const Tshape size = Svec(d);
      if (size == -1) {
        if (*unknown_index != -1) {
          return errors::InvalidArgument(
              "Only one input size may be -1, not both ", *unknown_index,
              " and ", d);
        }
        *unknown_index = d;
        shape->AddDim(1);
      } else if (size < 0) {
        return errors::InvalidArgument("Size ", d,
                                       " must be non-negative, not ", size);
      } else if (size == 0) {
        // We don't include zero-sized dimension in product, so that we can
        // still calculate number of elements for non-zero-sized dimensions and
        // therefore infer their shapes.
        shape->AddDim(size);
        *has_zero_dim = true;
      } else {
        shape->AddDim(size);
        (*product) *= size;
      }
    }
    return Status::OK();
  }
};

#define REGISTER_MKL_CPU(T)                                    \
  REGISTER_KERNEL_BUILDER(                                     \
      Name("_MklReshape")                                      \
          .Device(DEVICE_CPU)                                  \
          .HostMemory("shape")                                 \
          .TypeConstraint<T>("T")                              \
          .TypeConstraint("Tshape", {DT_INT32, DT_INT64})      \
          .Label(mkl_op_registry::kMklLayoutDependentOpLabel), \
      MklReshapeOp<CPUDevice, T>);

TF_CALL_float(REGISTER_MKL_CPU);
TF_CALL_bfloat16(REGISTER_MKL_CPU);

#undef REGISTER_MKL_CPU

}  // namespace tensorflow

#endif  // INTEL_MKL
