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
class MHTracer_DTPStensorflowPSlitePSkernelsPSmaximum_minimumDTcc {
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
   MHTracer_DTPStensorflowPSlitePSkernelsPSmaximum_minimumDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePSkernelsPSmaximum_minimumDTcc() {
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
#include "tensorflow/lite/kernels/internal/reference/maximum_minimum.h"

#include <stdint.h>

#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/internal/compatibility.h"
#include "tensorflow/lite/kernels/internal/optimized/optimized_ops.h"
#include "tensorflow/lite/kernels/internal/reference/process_broadcast_shapes.h"
#include "tensorflow/lite/kernels/internal/reference/reference_ops.h"
#include "tensorflow/lite/kernels/internal/tensor.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/internal/types.h"
#include "tensorflow/lite/kernels/kernel_util.h"

namespace tflite {
namespace ops {
namespace builtin {
namespace maximum_minimum {

// This file has a reference implementation of TFMaximum/TFMinimum.
enum KernelType {
  kReference,
  kGenericOptimized,
};

constexpr int kInputTensor1 = 0;
constexpr int kInputTensor2 = 1;
constexpr int kOutputTensor = 0;

struct OpContext {
  OpContext(TfLiteContext* context, TfLiteNode* node) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSmaximum_minimumDTcc mht_0(mht_0_v, 214, "", "./tensorflow/lite/kernels/maximum_minimum.cc", "OpContext");

    input1 = GetInput(context, node, kInputTensor1);
    input2 = GetInput(context, node, kInputTensor2);
    output = GetOutput(context, node, kOutputTensor);
  }
  const TfLiteTensor* input1;
  const TfLiteTensor* input2;
  TfLiteTensor* output;
};

TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSmaximum_minimumDTcc mht_1(mht_1_v, 227, "", "./tensorflow/lite/kernels/maximum_minimum.cc", "Prepare");

  TF_LITE_ENSURE_EQ(context, NumInputs(node), 2);
  TF_LITE_ENSURE_EQ(context, NumOutputs(node), 1);

  OpContext op_context(context, node);
  TF_LITE_ENSURE_TYPES_EQ(context, op_context.input1->type,
                          op_context.input2->type);
  op_context.output->type = op_context.input1->type;

  bool requires_broadcast =
      !HaveSameShapes(op_context.input1, op_context.input2);

  TfLiteIntArray* output_size = nullptr;
  if (requires_broadcast) {
    TF_LITE_ENSURE_OK(
        context, CalculateShapeForBroadcast(context, op_context.input1,
                                            op_context.input2, &output_size));
  } else {
    output_size = TfLiteIntArrayCopy(op_context.input1->dims);
  }

  return context->ResizeTensor(context, op_context.output, output_size);
}

struct MaximumOp {
  template <typename data_type>
  static data_type op(data_type el1, data_type el2) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSmaximum_minimumDTcc mht_2(mht_2_v, 256, "", "./tensorflow/lite/kernels/maximum_minimum.cc", "op");

    return el1 > el2 ? el1 : el2;
  }
};

struct MinimumOp {
  template <typename data_type>
  static data_type op(data_type el1, data_type el2) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSmaximum_minimumDTcc mht_3(mht_3_v, 266, "", "./tensorflow/lite/kernels/maximum_minimum.cc", "op");

    return el1 < el2 ? el1 : el2;
  }
};

template <KernelType kernel_type, typename data_type, typename op_type>
void TFLiteOperation(TfLiteContext* context, TfLiteNode* node,
                     const OpContext& op_context) {
  reference_ops::MaximumMinimumBroadcastSlow(
      GetTensorShape(op_context.input1),
      GetTensorData<data_type>(op_context.input1),
      GetTensorShape(op_context.input2),
      GetTensorData<data_type>(op_context.input2),
      GetTensorShape(op_context.output),
      GetTensorData<data_type>(op_context.output),
      op_type::template op<data_type>);
}

// Maximum generic opt int8.
template <>
void TFLiteOperation<maximum_minimum::kGenericOptimized, int8, MaximumOp>(
    TfLiteContext* context, TfLiteNode* node, const OpContext& op_context) {
  tflite::ArithmeticParams op_params;
  const bool need_broadcast = optimized_ops::ProcessBroadcastShapes(
      GetTensorShape(op_context.input1), GetTensorShape(op_context.input2),
      &op_params);
  if (need_broadcast) {
    optimized_ops::BroadcastMaximumDispatch(
        op_params, GetTensorShape(op_context.input1),
        GetTensorData<int8>(op_context.input1),
        GetTensorShape(op_context.input2),
        GetTensorData<int8>(op_context.input2),
        GetTensorShape(op_context.output),
        GetTensorData<int8>(op_context.output), MaximumOp::template op<int8>);
    return;
  }
  reference_ops::MaximumMinimumBroadcastSlow(
      GetTensorShape(op_context.input1), GetTensorData<int8>(op_context.input1),
      GetTensorShape(op_context.input2), GetTensorData<int8>(op_context.input2),
      GetTensorShape(op_context.output), GetTensorData<int8>(op_context.output),
      MaximumOp::template op<int8>);
}

// Minimum generic opt int8.
template <>
void TFLiteOperation<maximum_minimum::kGenericOptimized, int8, MinimumOp>(
    TfLiteContext* context, TfLiteNode* node, const OpContext& op_context) {
  tflite::ArithmeticParams op_params;
  const bool need_broadcast = optimized_ops::ProcessBroadcastShapes(
      GetTensorShape(op_context.input1), GetTensorShape(op_context.input2),
      &op_params);
  if (need_broadcast) {
    optimized_ops::BroadcastMinimumDispatch(
        op_params, GetTensorShape(op_context.input1),
        GetTensorData<int8>(op_context.input1),
        GetTensorShape(op_context.input2),
        GetTensorData<int8>(op_context.input2),
        GetTensorShape(op_context.output),
        GetTensorData<int8>(op_context.output), MinimumOp::template op<int8>);
    return;
  }
  reference_ops::MaximumMinimumBroadcastSlow(
      GetTensorShape(op_context.input1), GetTensorData<int8>(op_context.input1),
      GetTensorShape(op_context.input2), GetTensorData<int8>(op_context.input2),
      GetTensorShape(op_context.output), GetTensorData<int8>(op_context.output),
      MinimumOp::template op<int8>);
}

template <KernelType kernel_type, typename OpType>
TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
  OpContext op_context(context, node);

  // If inputs have no element, shortcircuit.
  if (NumElements(op_context.input1) == 0 ||
      NumElements(op_context.input2) == 0) {
    return kTfLiteOk;
  }

  switch (op_context.output->type) {
    case kTfLiteFloat32:
      TFLiteOperation<kernel_type, float, OpType>(context, node, op_context);
      break;
    case kTfLiteUInt8:
      TFLiteOperation<kernel_type, uint8_t, OpType>(context, node, op_context);
      break;
    case kTfLiteInt8:
      TFLiteOperation<kernel_type, int8_t, OpType>(context, node, op_context);
      break;
    case kTfLiteInt32:
      TFLiteOperation<kernel_type, int32_t, OpType>(context, node, op_context);
      break;
    case kTfLiteInt64:
      TFLiteOperation<kernel_type, int64_t, OpType>(context, node, op_context);
      break;
    case kTfLiteInt16:
      TFLiteOperation<kernel_type, int16_t, OpType>(context, node, op_context);
      break;
    default:
      context->ReportError(context,
                           "Type %d is currently not supported by Maximum.",
                           op_context.output->type);
      return kTfLiteError;
  }
  return kTfLiteOk;
}

}  // namespace maximum_minimum

TfLiteRegistration* Register_MAXIMUM_REF() {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSmaximum_minimumDTcc mht_4(mht_4_v, 377, "", "./tensorflow/lite/kernels/maximum_minimum.cc", "Register_MAXIMUM_REF");

  static TfLiteRegistration r = {
      nullptr, nullptr, maximum_minimum::Prepare,
      maximum_minimum::Eval<maximum_minimum::kReference,
                            maximum_minimum::MaximumOp>};
  return &r;
}

TfLiteRegistration* Register_MAXIMUM_GENERIC_OPT() {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSmaximum_minimumDTcc mht_5(mht_5_v, 388, "", "./tensorflow/lite/kernels/maximum_minimum.cc", "Register_MAXIMUM_GENERIC_OPT");

  static TfLiteRegistration r = {
      nullptr, nullptr, maximum_minimum::Prepare,
      maximum_minimum::Eval<maximum_minimum::kGenericOptimized,
                            maximum_minimum::MaximumOp>};
  return &r;
}

TfLiteRegistration* Register_MINIMUM_REF() {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSmaximum_minimumDTcc mht_6(mht_6_v, 399, "", "./tensorflow/lite/kernels/maximum_minimum.cc", "Register_MINIMUM_REF");

  static TfLiteRegistration r = {
      nullptr, nullptr, maximum_minimum::Prepare,
      maximum_minimum::Eval<maximum_minimum::kReference,
                            maximum_minimum::MinimumOp>};
  return &r;
}

TfLiteRegistration* Register_MINIMUM_GENERIC_OPT() {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSmaximum_minimumDTcc mht_7(mht_7_v, 410, "", "./tensorflow/lite/kernels/maximum_minimum.cc", "Register_MINIMUM_GENERIC_OPT");

  static TfLiteRegistration r = {
      nullptr, nullptr, maximum_minimum::Prepare,
      maximum_minimum::Eval<maximum_minimum::kGenericOptimized,
                            maximum_minimum::MinimumOp>};
  return &r;
}

TfLiteRegistration* Register_MAXIMUM() {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSmaximum_minimumDTcc mht_8(mht_8_v, 421, "", "./tensorflow/lite/kernels/maximum_minimum.cc", "Register_MAXIMUM");

  return Register_MAXIMUM_GENERIC_OPT();
}
TfLiteRegistration* Register_MINIMUM() {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSmaximum_minimumDTcc mht_9(mht_9_v, 427, "", "./tensorflow/lite/kernels/maximum_minimum.cc", "Register_MINIMUM");

  return Register_MINIMUM_GENERIC_OPT();
}

}  // namespace builtin
}  // namespace ops
}  // namespace tflite
