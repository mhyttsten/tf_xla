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
class MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSin_topk_opDTcc {
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
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSin_topk_opDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSin_topk_opDTcc() {
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

/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/tf2xla/type_util.h"
#include "tensorflow/compiler/tf2xla/xla_op_kernel.h"
#include "tensorflow/compiler/tf2xla/xla_op_registry.h"
#include "tensorflow/compiler/xla/client/lib/arithmetic.h"
#include "tensorflow/compiler/xla/client/lib/constants.h"
#include "tensorflow/compiler/xla/client/lib/sorting.h"
#include "tensorflow/compiler/xla/client/xla_builder.h"
#include "tensorflow/compiler/xla/literal.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/framework/kernel_def_builder.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/platform/macros.h"

namespace tensorflow {
namespace {

class InTopKOp : public XlaOpKernel {
 public:
  explicit InTopKOp(OpKernelConstruction* context) : XlaOpKernel(context) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSin_topk_opDTcc mht_0(mht_0_v, 204, "", "./tensorflow/compiler/tf2xla/kernels/in_topk_op.cc", "InTopKOp");

    OP_REQUIRES_OK(context, context->GetAttr("T", &targets_dtype_));
    OP_REQUIRES_OK(context,
                   DataTypeToPrimitiveType(targets_dtype_, &targets_type_));
  }

  void Compile(XlaOpKernelContext* context) override {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSin_topk_opDTcc mht_1(mht_1_v, 213, "", "./tensorflow/compiler/tf2xla/kernels/in_topk_op.cc", "Compile");

    int64_t k;
    OP_REQUIRES_OK(context, context->ConstantInputAsIntScalar(2, &k));
    OP_REQUIRES(context, k >= 0,
                errors::InvalidArgument("Need k >= 0, got ", k));
    const TensorShape predictions_shape = context->InputShape(0);
    OP_REQUIRES(
        context, predictions_shape.dims() == 2,
        errors::InvalidArgument("predictions must be == 2-D, got shape ",
                                predictions_shape.DebugString()));
    const TensorShape targets_shape = context->InputShape(1);
    OP_REQUIRES(context, targets_shape.dims() == 1,
                errors::InvalidArgument("targets must be == 1-D, got shape ",
                                        targets_shape.DebugString()));

    int64_t batch_size = predictions_shape.dim_size(0);
    OP_REQUIRES(context, batch_size == targets_shape.dim_size(0),
                errors::InvalidArgument(
                    "targets must have same elements as predictions rows. Had ",
                    targets_shape.dim_size(0), ", needed ", batch_size));

    // Given `predictions` with shape batch_size*num_classes and `target` with
    // shape num_classes, we generate `targets_values_r1` with shape num_classes
    // which the elements are the corresponding values of `targets` in
    // `predictions` for each example. This step can be done using xla::Gather
    // as well.
    xla::XlaOp predictions_r2 = context->Input(0);
    xla::XlaOp targets_r1 = context->Input(1);

    xla::XlaBuilder* xla_builder = context->builder();
    xla::XlaOp iota_r1 =
        xla::Iota(xla_builder, targets_type_, predictions_shape.dim_size(1));
    xla::XlaOp iota_r2 = xla::Broadcast(iota_r1, {batch_size});

    xla::XlaOp eq_r2 = xla::Eq(targets_r1, iota_r2, {0});
    xla::XlaOp zero_r0_f32 = xla::Zero(xla_builder, xla::F32);
    xla::XlaOp zero_r2_f32 = xla::ZerosLike(predictions_r2);
    xla::XlaOp select_r2 = xla::Select(eq_r2, predictions_r2, zero_r2_f32);
    xla::XlaOp targets_values_r1 = xla::Reduce(
        select_r2, zero_r0_f32,
        xla::CreateScalarAddComputation(xla::F32, xla_builder), {1});

    // Calculate in each row of `predictions`, how many values are larger than
    // the value of target class. Then return the result whether the count < k,
    // which indicates the target is in topk.
    xla::XlaOp gt_r2 = xla::Gt(predictions_r2, targets_values_r1, {0});
    xla::XlaOp zero_r0 = xla::Zero(xla_builder, xla::S32);
    xla::XlaOp zero_r2 = xla::Broadcast(zero_r0, predictions_shape.dim_sizes());
    xla::XlaOp one_r0 = xla::One(xla_builder, xla::S32);
    xla::XlaOp one_r2 = xla::Broadcast(one_r0, predictions_shape.dim_sizes());
    xla::XlaOp one_hot_r2 = xla::Select(gt_r2, one_r2, zero_r2);
    xla::XlaOp num_gt_r1 = xla::Reduce(
        one_hot_r2, zero_r0,
        xla::CreateScalarAddComputation(xla::S32, xla_builder), {1});

    xla::XlaOp result =
        xla::And(xla::Lt(num_gt_r1, xla::ConstantR0<int32>(xla_builder, k)),
                 xla::IsFinite(targets_values_r1));

    context->SetOutput(0, result);
  }

 protected:
  DataType targets_dtype_;
  xla::PrimitiveType targets_type_;

  TF_DISALLOW_COPY_AND_ASSIGN(InTopKOp);
};

REGISTER_XLA_OP(Name("InTopKV2")
                    .CompileTimeConstantInput("k")
                    .TypeConstraint("T", {DT_INT32, DT_INT64}),
                InTopKOp);

}  // namespace
}  // namespace tensorflow
