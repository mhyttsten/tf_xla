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
class MHTracer_DTPStensorflowPScorePSkernelsPSmklPSmkl_input_conversion_opDTcc {
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
   MHTracer_DTPStensorflowPScorePSkernelsPSmklPSmkl_input_conversion_opDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSkernelsPSmklPSmkl_input_conversion_opDTcc() {
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

#include <algorithm>
#include <vector>

#include "tensorflow/core/kernels/mkl/mkl_tfconv_op.h"

namespace tensorflow {

///////////////////////////////////////////////////////////
//               Op kernel
// Checks and ensures that the 2 inputs are compatible for mkl binary ops.
// Here's the basic logic:
//
// if both inputs are in TF format:
//   pass the inputs through to the output
// else if both inputs are in mkl format:
//   if both have the same shape:
//     pass the inputs through to the output
//   else:
//     convert both to TF
// else if one is TF and one is MKL:
//   if broadcast is needed:
//     convert the MKL format input to TF format
//   else:
//     convert the TF format input to MKL format
///////////////////////////////////////////////////////////

template <typename Device, typename T>
class MklInputConversionOp : public OpKernel {
 public:
  explicit MklInputConversionOp(OpKernelConstruction* context)
      : OpKernel(context) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSmklPSmkl_input_conversion_opDTcc mht_0(mht_0_v, 217, "", "./tensorflow/core/kernels/mkl/mkl_input_conversion_op.cc", "MklInputConversionOp");

    OP_REQUIRES_OK(context, context->GetAttr("data_format", &data_format_str));
    OP_REQUIRES_OK(context, context->GetAttr("T", &op_data_type));
    has_avx512f_ = port::TestCPUFeature(port::CPUFeature::AVX512F);
  }

 private:
  void Compute(OpKernelContext* context) override {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSmklPSmkl_input_conversion_opDTcc mht_1(mht_1_v, 227, "", "./tensorflow/core/kernels/mkl/mkl_input_conversion_op.cc", "Compute");

    const int kInputIndex_0 = 0, kInputIndex_1 = 1;
    const Tensor& input_tensor_0 = MklGetInput(context, kInputIndex_0);
    MklDnnShape input_shape_0;
    GetMklShape(context, kInputIndex_0, &input_shape_0);

    const Tensor& input_tensor_1 = MklGetInput(context, kInputIndex_1);
    MklDnnShape input_shape_1;
    GetMklShape(context, kInputIndex_1, &input_shape_1);

    VLOG(1) << "MklInputConversionOp: Input shapes are: "
            << context->input(kInputIndex_0).shape().DebugString() << " and "
            << context->input(kInputIndex_1).shape().DebugString();

    // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    // if both inputs are in TF format, just copy input tensors to output.
    if (!input_shape_0.IsMklTensor() && !input_shape_1.IsMklTensor()) {
      VLOG(1) << "MklInputConversionOp: No conversion needed, "
              << "copying TF inputs to output";

      ForwardTfTensorInToOut(context, kInputIndex_0, kInputIndex_0);
      ForwardTfTensorInToOut(context, kInputIndex_1, kInputIndex_1);
      return;
    }

    // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    // If both inputs are in MKL format
    if (input_shape_0.IsMklTensor() && input_shape_1.IsMklTensor()) {
      // It is safer to compare the original TensorFlow shapes than to compare
      // Mkl shapes since element wise ops are forwarded to Eigen
      // implementation.
      TensorShape tf_shape0 = input_shape_0.GetTfShape();
      TensorShape tf_shape1 = input_shape_1.GetTfShape();
      TensorShape tensor_shape0 = input_tensor_0.shape();
      TensorShape tensor_shape1 = input_tensor_1.shape();
      if (tf_shape0 == tf_shape1 && tensor_shape0 == tensor_shape1) {
        auto input0_md = input_shape_0.GetMklLayout();
        auto input1_md = input_shape_1.GetMklLayout();

        // If both have the same shape and same format, pass them through
        if (input_shape_0.GetTfDataFormat() ==
            input_shape_1.GetTfDataFormat()) {
          VLOG(1) << "MklInputConversionOp: No conversion needed, "
                  << "copying MKL inputs with identical shapes to output";

          ForwardMklTensorInToOut(context, kInputIndex_0, kInputIndex_0);
          ForwardMklTensorInToOut(context, kInputIndex_1, kInputIndex_1);
          return;
        } else {
          VLOG(1) << "MklInputConversionOp: Shape is same, but format is "
                     "different, "
                  << "need to convert to same format";
          // TODO(intel-tf): For now, input0 is converted and input1 is
          // unchanged. We should choose the optimal oneDNN format to convert
          // to.
          Tensor* tensor_out;
          MklDnnShape mkl_output_mkl_shape;
          mkl_output_mkl_shape.SetMklTensor(true);
          mkl_output_mkl_shape.SetElemType(MklDnnType<T>());
          mkl_output_mkl_shape.SetTfLayout(input_shape_0.GetDimension(),
                                           input_shape_0.GetSizesAsMklDnnDims(),
                                           input_shape_0.GetTfDataFormat());

          // Get MKL layout from input1 as destination layout
          mkl_output_mkl_shape.SetMklLayout(&input1_md);

          // Create output Mkl tensor for index 0
          AllocateOutputSetMklShape(context, kInputIndex_0, &tensor_out,
                                    input_tensor_0.shape(),
                                    mkl_output_mkl_shape);

          // Create MklDnnData object for input0 tensor
          auto cpu_engine = engine(engine::kind::cpu, 0);
          MklDnnData<T> input(&cpu_engine);
          input.SetUsrMem(input0_md, &input_tensor_0);
          // Create reorder from input0's layout to input1's layout
          std::vector<primitive> net;
          std::vector<MemoryArgsMap> net_args;
          // TODO(intel-tf): Refactor CheckReorderToOpMem() to create and
          // execute reorder
          OP_REQUIRES(
              context,
              input.CheckReorderToOpMem(input1_md, tensor_out, net, net_args,
                                        cpu_engine),
              errors::Internal(
                  "MklInputConversionOp: Failed to create reorder for input0"));
          ExecutePrimitive(net, &net_args, cpu_engine, context);
          // Input1 will be passed through
          ForwardMklTensorInToOut(context, kInputIndex_1, kInputIndex_1);
          return;
        }
      }

      // Sanity check
      bool mkl_shapes_are_same = ((input_shape_0 == input_shape_1) &&
                                  (tensor_shape0 == tensor_shape1));
      if (mkl_shapes_are_same) {
        CHECK(false) << "MklInputConversionOp: Unexpected: TF shapes are "
                        "different but MKL shapes are same";
      }

      // Both have different shapes, so broadcast will be necessary.
      // Convert to TF and pass both tensors through (we can't do broadcast
      // with MKL tensors)
      VLOG(1) << "MklInputConversionOp: Broadcast needed, "
              << "converted MKL inputs to TF format";
      // TODO(intel-tf): Cleanup op_data_type and has_avx512f_ after these two
      //     parameters are removed from ConvertMklToTf
      MklToTfOp<Device, T>::ConvertMklToTf(this, context, data_format_str,
                                           op_data_type, has_avx512f_,
                                           kInputIndex_0);
      MklToTfOp<Device, T>::ConvertMklToTf(this, context, data_format_str,
                                           op_data_type, has_avx512f_,
                                           kInputIndex_1);
      SetDummyMklDnnShapeOutput(context, kInputIndex_0);
      SetDummyMklDnnShapeOutput(context, kInputIndex_1);
      return;
    }

    // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    // One input is MKL and one is TF. If no broadcast is needed, convert
    // the TF tensor to MKL, otherwise convert the MKL tensor to TF format
    VLOG(1) << "MklInputConversionOp: Inputs in different formats (MKL/TF)";

    const Tensor* mkl_tensor;
    const MklDnnShape* mkl_shape;
    const Tensor* tf_tensor;
    uint mkl_tensor_index;
    uint tf_tensor_index;
    if (input_shape_0.IsMklTensor() && !input_shape_1.IsMklTensor()) {
      mkl_tensor = &input_tensor_0;
      mkl_shape = &input_shape_0;
      mkl_tensor_index = 0;
      tf_tensor = &input_tensor_1;
      tf_tensor_index = 1;
    } else if (!input_shape_0.IsMklTensor() && input_shape_1.IsMklTensor()) {
      mkl_tensor = &input_tensor_1;
      mkl_shape = &input_shape_1;
      mkl_tensor_index = 1;
      tf_tensor = &input_tensor_0;
      tf_tensor_index = 0;
    } else {
      CHECK(false) << "MklInputConversionOp: Unexpected combination of input "
                      "shapes for MKL "
                   << "element-wise op";
    }

    // Broadcast is needed if the shapes are not the same
    if (mkl_shape->GetTfShape().num_elements() ==
        tf_tensor->shape().num_elements()) {
      // Both shapes are same, convert the TF input to MKL
      VLOG(1) << "MklInputConversionOp: No broadcast needed.";
      VLOG(1) << "MklInputConversionOp: Converting input " << tf_tensor_index
              << " to MKL format";

      // Create MklDnnShape for output Mkl tensor.
      Tensor* tensor_out;
      MklDnnShape mkl_output_mkl_shape;
      mkl_output_mkl_shape.SetMklTensor(true);
      mkl_output_mkl_shape.SetElemType(MklDnnType<T>());
      mkl_output_mkl_shape.SetTfLayout(mkl_shape->GetDimension(),
                                       mkl_shape->GetSizesAsMklDnnDims(),
                                       mkl_shape->GetTfDataFormat());
      // ** Temporarily borrow the layout from the MKL input **
      auto output_mkl_md = mkl_shape->GetMklLayout();
      mkl_output_mkl_shape.SetMklLayout(&output_mkl_md);

      // Create output Mkl tensor
      AllocateOutputSetMklShape(context, tf_tensor_index, &tensor_out,
                                mkl_tensor->shape(), mkl_output_mkl_shape);

      // Create MklDnnData object for input tensor. Input tensor is in
      // Tensorflow layout.
      auto cpu_engine = engine(engine::kind::cpu, 0);
      MklDnnData<T> tf_input(&cpu_engine);
      auto input_tf_md = mkl_output_mkl_shape.GetTfLayout();
      tf_input.SetUsrMem(input_tf_md, tf_tensor);
      // Create reorder between TF layout and MKL layout if necessary
      std::vector<primitive> net;
      std::vector<MemoryArgsMap> net_args;
      bool reordered = tf_input.CheckReorderToOpMem(output_mkl_md, tensor_out,
                                                    net, net_args, cpu_engine);
      if (!reordered) {
        // This is the case that the TF tensor has the same shape and format of
        // mkl tensor. However, tf_tensor can not be simply forwarded to the
        // output tensor since mkl data tensor is always one dimensional tensor.
        // Tensor::CopyFrom shares the buffer of the other tensor while set its
        // shape to the other tensor.
        OP_REQUIRES(context,
                    tensor_out->CopyFrom(*tf_tensor, tensor_out->shape()),
                    errors::Internal("MklInputConversionOp: Failed to forward "
                                     "input tensor to output"));
      } else {
        ExecutePrimitive(net, &net_args, cpu_engine, context);
      }

      // -- The tensor in MKL format passes through --
      ForwardMklTensorInToOut(context, mkl_tensor_index, mkl_tensor_index);
    } else {
      // Broadcast is needed, so convert the MKL input to TF
      VLOG(1) << "MklInputConversionOp: Broadcast needed.";
      VLOG(1) << "MklInputConversionOp: Converting input " << mkl_tensor_index
              << " to TF format";
      MklToTfOp<Device, T>::ConvertMklToTf(this, context, data_format_str,
                                           op_data_type, has_avx512f_,
                                           mkl_tensor_index);
      SetDummyMklDnnShapeOutput(context, mkl_tensor_index);

      // The tensor in TF format passes through
      ForwardTfTensorInToOut(context, tf_tensor_index, tf_tensor_index);
    }

    VLOG(1) << "MklInputConversionOp: Shapes (output): "
            << context->mutable_output(kInputIndex_0)->shape().DebugString()
            << " and "
            << context->mutable_output(kInputIndex_1)->shape().DebugString();

    VLOG(1) << "MklInputConversion completed successfully.";
  }

 private:
  /// Data format of the operation
  string data_format_str;

  /// Data type of the operation
  DataType op_data_type;

  /// CPUIDInfo
  bool has_avx512f_ = false;
};

///////////////////////////////////////////////////////////
//               Register kernel
///////////////////////////////////////////////////////////

#define REGISTER_CPU(T)                                        \
  REGISTER_KERNEL_BUILDER(                                     \
      Name("_MklInputConversion")                              \
          .Device(DEVICE_CPU)                                  \
          .TypeConstraint<T>("T")                              \
          .Label(mkl_op_registry::kMklLayoutDependentOpLabel), \
      MklInputConversionOp<CPUDevice, T>);

TF_CALL_float(REGISTER_CPU);
TF_CALL_bfloat16(REGISTER_CPU);

#undef REGISTER_CPU

}  // namespace tensorflow
#endif  // INTEL_MKL
