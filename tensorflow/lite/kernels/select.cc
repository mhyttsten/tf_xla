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
class MHTracer_DTPStensorflowPSlitePSkernelsPSselectDTcc {
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
   MHTracer_DTPStensorflowPSlitePSkernelsPSselectDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePSkernelsPSselectDTcc() {
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
#include <stddef.h>
#include <stdint.h>

#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/internal/reference/reference_ops.h"
#include "tensorflow/lite/kernels/internal/tensor.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/kernel_util.h"

namespace tflite {
namespace ops {
namespace builtin {
namespace select {

constexpr int kInputTensorCondition = 0;
constexpr int kInputTensorX = 1;
constexpr int kInputTensorY = 2;
constexpr int kOutputTensor = 0;

enum KernelType {
  kVersionOne,
  kVersionTwo,
};

struct OpData {
  bool requires_broadcast;
  // True if input condition is scalar or input condition has rank one and
  // matches the first dimension of other inputs.
  bool has_low_rank_input_condition;
};

void* SelectInit(TfLiteContext* context, const char* buffer, size_t length) {
   std::vector<std::string> mht_0_v;
   mht_0_v.push_back("buffer: \"" + (buffer == nullptr ? std::string("nullptr") : std::string((char*)buffer)) + "\"");
   MHTracer_DTPStensorflowPSlitePSkernelsPSselectDTcc mht_0(mht_0_v, 216, "", "./tensorflow/lite/kernels/select.cc", "SelectInit");

  auto* data = new OpData;
  data->requires_broadcast = false;
  data->has_low_rank_input_condition = false;
  return data;
}

void SelectFree(TfLiteContext* context, void* buffer) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSselectDTcc mht_1(mht_1_v, 226, "", "./tensorflow/lite/kernels/select.cc", "SelectFree");

  delete reinterpret_cast<OpData*>(buffer);
}

template <KernelType kernel_type>
TfLiteStatus SelectPrepare(TfLiteContext* context, TfLiteNode* node) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSselectDTcc mht_2(mht_2_v, 234, "", "./tensorflow/lite/kernels/select.cc", "SelectPrepare");

  OpData* data = reinterpret_cast<OpData*>(node->user_data);

  TF_LITE_ENSURE_EQ(context, NumInputs(node), 3);
  TF_LITE_ENSURE_EQ(context, NumOutputs(node), 1);

  const TfLiteTensor* input_condition;
  TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, kInputTensorCondition,
                                          &input_condition));
  const TfLiteTensor* input_x;
  TF_LITE_ENSURE_OK(context,
                    GetInputSafe(context, node, kInputTensorX, &input_x));
  const TfLiteTensor* input_y;
  TF_LITE_ENSURE_OK(context,
                    GetInputSafe(context, node, kInputTensorY, &input_y));
  TfLiteTensor* output;
  TF_LITE_ENSURE_OK(context,
                    GetOutputSafe(context, node, kOutputTensor, &output));

  // Input must be bool.
  TF_LITE_ENSURE_TYPES_EQ(context, input_condition->type, kTfLiteBool);
  TF_LITE_ENSURE_TYPES_EQ(context, input_x->type, input_y->type);
  output->type = input_x->type;

  // Respect the original output shape when there are mixed shapes to represent
  // a scalar data.
  if (GetTensorShape(input_condition).FlatSize() == 1 &&
      GetTensorShape(input_x).FlatSize() == 1 &&
      GetTensorShape(input_y).FlatSize() == 1 &&
      GetTensorShape(output).FlatSize() == 1) {
    return kTfLiteOk;
  }

  bool same_shape = HaveSameShapes(input_condition, input_x) &&
                    HaveSameShapes(input_x, input_y);
  TfLiteIntArray* output_size;
  if (!same_shape) {
    switch (kernel_type) {
      case kVersionOne: {
        bool is_input_condition_scalar = NumDimensions(input_condition) == 0;
        bool has_rank_one_input_condition =
            NumDimensions(input_condition) == 1 &&
            SizeOfDimension(input_condition, 0) == SizeOfDimension(input_x, 0);
        data->has_low_rank_input_condition =
            is_input_condition_scalar || has_rank_one_input_condition;
        TF_LITE_ENSURE(context, data->has_low_rank_input_condition);

        output_size = TfLiteIntArrayCopy(input_x->dims);

        // Input tensors must have the same type and size
        TF_LITE_ENSURE(context, HaveSameShapes(input_x, input_y));
        break;
      }
      case kVersionTwo: {
        TF_LITE_ENSURE_OK(context, CalculateShapeForBroadcast(
                                       context, input_condition, input_x,
                                       input_y, &output_size));
        data->requires_broadcast = true;
        break;
      }
      default:
        return kTfLiteError;
    }
  } else {
    output_size = TfLiteIntArrayCopy(input_x->dims);
  }

  return context->ResizeTensor(context, output, output_size);
}

TfLiteStatus SelectEval(TfLiteContext* context, TfLiteNode* node) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSselectDTcc mht_3(mht_3_v, 307, "", "./tensorflow/lite/kernels/select.cc", "SelectEval");

  OpData* data = reinterpret_cast<OpData*>(node->user_data);
  const TfLiteTensor* input_condition;
  TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, kInputTensorCondition,
                                          &input_condition));
  const TfLiteTensor* input_x;
  TF_LITE_ENSURE_OK(context,
                    GetInputSafe(context, node, kInputTensorX, &input_x));
  const TfLiteTensor* input_y;
  TF_LITE_ENSURE_OK(context,
                    GetInputSafe(context, node, kInputTensorY, &input_y));
  TfLiteTensor* output;
  TF_LITE_ENSURE_OK(context,
                    GetOutputSafe(context, node, kOutputTensor, &output));

#define TF_LITE_SELECT(type, op)                                           \
  reference_ops::op(GetTensorShape(input_condition),                       \
                    GetTensorData<bool>(input_condition),                  \
                    GetTensorShape(input_x), GetTensorData<type>(input_x), \
                    GetTensorShape(input_y), GetTensorData<type>(input_y), \
                    GetTensorShape(output), GetTensorData<type>(output));

#define TF_LITE_SWITCH(type, op)                                               \
  switch (type) {                                                              \
    break;                                                                     \
    case kTfLiteBool:                                                          \
      TF_LITE_SELECT(bool, op);                                                \
      break;                                                                   \
    case kTfLiteFloat32:                                                       \
      TF_LITE_SELECT(float, op);                                               \
      break;                                                                   \
    case kTfLiteUInt8:                                                         \
      TF_LITE_SELECT(uint8_t, op);                                             \
      break;                                                                   \
    case kTfLiteInt8:                                                          \
      TF_LITE_SELECT(int8_t, op);                                              \
      break;                                                                   \
    case kTfLiteInt16:                                                         \
      TF_LITE_SELECT(int16_t, op);                                             \
      break;                                                                   \
    case kTfLiteInt32:                                                         \
      TF_LITE_SELECT(int32_t, op);                                             \
      break;                                                                   \
    case kTfLiteInt64:                                                         \
      TF_LITE_SELECT(int64_t, op);                                             \
      break;                                                                   \
    default:                                                                   \
      context->ReportError(context,                                            \
                           "Does not support type other than bool|float|int, " \
                           "got %d",                                           \
                           type);                                              \
      return kTfLiteError;                                                     \
  }

  if (data->has_low_rank_input_condition) {
    TF_LITE_SWITCH(input_x->type, RankOneSelect);
  } else if (data->requires_broadcast) {
    TF_LITE_SWITCH(input_x->type, BroadcastSelect4DSlow);
  } else {
    TF_LITE_SWITCH(input_x->type, Select);
  }

#undef TF_LITE_SELECT
#undef TF_LITE_SWITCH
  return kTfLiteOk;
}

}  // namespace select

// Select op selects values of 'x' if the corresponding value of 'condition' is
// true or the value of 'y' if false. There are valid condition input sizes:
//
// 1. Either the same shape (in which case the select is elementwise), or
// 2. condition must be Rank 1 and match over the first dimension, or
// 3. condition is scalar
TfLiteRegistration* Register_SELECT() {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSselectDTcc mht_4(mht_4_v, 385, "", "./tensorflow/lite/kernels/select.cc", "Register_SELECT");

  static TfLiteRegistration r = {select::SelectInit, select::SelectFree,
                                 select::SelectPrepare<select::kVersionOne>,
                                 select::SelectEval};
  return &r;
}

// SelectV2 op selects values of 'x' if the corresponding value of 'condition'
// is true or the value of 'y' if false. There are valid condition input sizes:
//
// 1. Either the same shape (in which case the select is elementwise), or
// 2. Broadcastable shapes between 'condition', 'x' and 'y'.
TfLiteRegistration* Register_SELECT_V2() {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSselectDTcc mht_5(mht_5_v, 400, "", "./tensorflow/lite/kernels/select.cc", "Register_SELECT_V2");

  static TfLiteRegistration r = {select::SelectInit, select::SelectFree,
                                 select::SelectPrepare<select::kVersionTwo>,
                                 select::SelectEval};
  return &r;
}

}  // namespace builtin
}  // namespace ops
}  // namespace tflite
