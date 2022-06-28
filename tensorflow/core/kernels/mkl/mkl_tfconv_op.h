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

#ifndef TENSORFLOW_CORE_KERNELS_MKL_MKL_TFCONV_OP_H_
#define TENSORFLOW_CORE_KERNELS_MKL_MKL_TFCONV_OP_H_
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
class MHTracer_DTPStensorflowPScorePSkernelsPSmklPSmkl_tfconv_opDTh {
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
   MHTracer_DTPStensorflowPScorePSkernelsPSmklPSmkl_tfconv_opDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSkernelsPSmklPSmkl_tfconv_opDTh() {
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

#include <algorithm>
#include <string>
#include <vector>

#include "tensorflow/core/framework/numeric_op.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/kernels/ops_util.h"
#include "tensorflow/core/platform/byte_order.h"
#include "tensorflow/core/platform/cpu_info.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/util/mkl_util.h"
#include "tensorflow/core/util/tensor_format.h"

using dnnl::stream;

namespace tensorflow {

typedef Eigen::ThreadPoolDevice CPUDevice;

///////////////////////////////////////////////////////////
//               Op kernel
///////////////////////////////////////////////////////////

template <typename Device, typename T>
class MklToTfOp : public OpKernel {
 public:
  explicit MklToTfOp(OpKernelConstruction* context) : OpKernel(context) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSmklPSmkl_tfconv_opDTh mht_0(mht_0_v, 220, "", "./tensorflow/core/kernels/mkl/mkl_tfconv_op.h", "MklToTfOp");

    OP_REQUIRES_OK(context, context->GetAttr("data_format", &data_format_str));
    OP_REQUIRES_OK(context, context->GetAttr("T", &op_data_type));
    has_avx512f_ = port::TestCPUFeature(port::CPUFeature::AVX512F);
  }

  void Compute(OpKernelContext* context) override {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSmklPSmkl_tfconv_opDTh mht_1(mht_1_v, 229, "", "./tensorflow/core/kernels/mkl/mkl_tfconv_op.h", "Compute");

    ConvertMklToTf(this, context, data_format_str, op_data_type, has_avx512f_,
                   0);
    VLOG(1) << "MKLToTFConversion complete successfully.";
  }

  // TODO(intel-tf): Move the below ConvertMklToTf() to mkl_util.h
  static void ConvertMklToTf(OpKernel* op_kernel, OpKernelContext* context,
                             string data_format_str, DataType op_data_type,
                             bool has_avx512f, uint input_number) {
   std::vector<std::string> mht_2_v;
   mht_2_v.push_back("data_format_str: \"" + data_format_str + "\"");
   MHTracer_DTPStensorflowPScorePSkernelsPSmklPSmkl_tfconv_opDTh mht_2(mht_2_v, 242, "", "./tensorflow/core/kernels/mkl/mkl_tfconv_op.h", "ConvertMklToTf");

    try {
      // Check that input tensor is in MKL format.
      const Tensor& input_tensor = MklGetInput(context, input_number);
      MklDnnShape input_shape;
      GetMklShape(context, input_number, &input_shape);

      // if input is already in Tf format, then copy input tensor to output.
      if (!input_shape.IsMklTensor()) {
        context->set_output(input_number, input_tensor);
        VLOG(1) << "MKLToTFConversion: No conversion needed, "
                << "copying input to output";
        return;
      }

      // Check that input data type is same as operator data type and that it
      // is same as output data type.
      DataType input_data_type = op_kernel->input_type(input_number);
      DataType output_data_type = op_kernel->output_type(input_number);
      CHECK_EQ(op_data_type, input_data_type);
      CHECK_EQ(op_data_type, output_data_type);

      auto cpu_engine = engine(engine::kind::cpu, 0);
      MklDnnData<T> input(&cpu_engine);

      // Get MKL layout of input tensor.
      auto input_mkl_md = input_shape.GetMklLayout();
      // Get TensorFlow layout of input tensor. Expected output of conversion
      // has same layout as Tensorflow layout of input tensor.
      auto output_tf_md = input_shape.GetTfLayout();
      // Set input MKL layout as the user layout.
      input.SetUsrMem(input_mkl_md, &input_tensor);

      // Allocate output tensor.
      TensorShape output_shape = input_shape.GetTfShape();
      Tensor* output_tensor = nullptr;
      OP_REQUIRES_OK(context, context->allocate_output(
                                  input_number, output_shape, &output_tensor));
      DCHECK(output_tensor);

      // Check if input needs to be reordered
      if (input.IsReorderNeeded(output_tf_md)) {
        // Insert reorder between MKL layout and TensorFlow layout
        OP_REQUIRES(
            context,
            input.CheckReorderToOpMem(output_tf_md, output_tensor, context),
            errors::Internal("MklToTfOp: Failed to create input reorder"));
      } else {
        // If not, just forward input tensor to output tensor.
        OP_REQUIRES(context,
                    output_tensor->CopyFrom(input_tensor, output_shape),
                    errors::Internal(
                        "MklToTfOp: Failed to forward input tensor to output"));
      }
    } catch (dnnl::error& e) {
      OP_REQUIRES_OK(
          context,
          errors::Aborted("Operation received an exception: Status: ", e.status,
                          ", message: ", StringPiece(e.message), ", in file ",
                          __FILE__, ":", __LINE__));
    }
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
      Name("_MklToTf")                                         \
          .Device(DEVICE_CPU)                                  \
          .TypeConstraint<T>("T")                              \
          .Label(mkl_op_registry::kMklLayoutDependentOpLabel), \
      MklToTfOp<CPUDevice, T>);

TF_CALL_NUMBER_TYPES(REGISTER_CPU);
TF_CALL_QUANTIZED_TYPES(REGISTER_CPU);

#undef REGISTER_CPU

}  // namespace tensorflow
#endif  // INTEL_MKL
#endif  // TENSORFLOW_CORE_KERNELS_MKL_MKL_TFCONV_OP_H_
