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
class MHTracer_DTPStensorflowPSlitePSkernelsPSsegment_sumDTcc {
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
   MHTracer_DTPStensorflowPSlitePSkernelsPSsegment_sumDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePSkernelsPSsegment_sumDTcc() {
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

/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#include <stdint.h>

#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/internal/reference/reference_ops.h"
#include "tensorflow/lite/kernels/internal/tensor.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/kernel_util.h"

namespace tflite {
namespace ops {
namespace builtin {
namespace segment_sum {

static const int kInputDataTensor = 0;
static const int kInputSegmentIdsTensor = 1;
static const int kOutputTensor = 0;

TfLiteStatus ResizeOutputTensor(TfLiteContext* context,
                                const TfLiteTensor* data,
                                const TfLiteTensor* segment_ids,
                                TfLiteTensor* output) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSsegment_sumDTcc mht_0(mht_0_v, 205, "", "./tensorflow/lite/kernels/segment_sum.cc", "ResizeOutputTensor");

  // Segment ids should be of same cardinality as first input dimension and they
  // should be increasing by at most 1, from 0 (e.g., [0, 0, 1, 2, 3] is valid)
  const int segment_id_size = segment_ids->dims->data[0];
  TF_LITE_ENSURE_EQ(context, segment_id_size, data->dims->data[0]);
  int previous_segment_id = -1;
  for (int i = 0; i < segment_id_size; i++) {
    const int current_segment_id = GetTensorData<int32_t>(segment_ids)[i];
    if (i == 0) {
      TF_LITE_ENSURE_EQ(context, current_segment_id, 0);
    } else {
      int delta = current_segment_id - previous_segment_id;
      TF_LITE_ENSURE(context, delta == 0 || delta == 1);
    }
    previous_segment_id = current_segment_id;
  }

  const int max_index = previous_segment_id;

  const int data_rank = NumDimensions(data);
  TfLiteIntArray* output_shape = TfLiteIntArrayCreate(NumDimensions(data));
  output_shape->data[0] = max_index + 1;
  for (int i = 1; i < data_rank; ++i) {
    output_shape->data[i] = data->dims->data[i];
  }
  return context->ResizeTensor(context, output, output_shape);
}

TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSsegment_sumDTcc mht_1(mht_1_v, 236, "", "./tensorflow/lite/kernels/segment_sum.cc", "Prepare");

  TF_LITE_ENSURE_EQ(context, NumInputs(node), 2);
  TF_LITE_ENSURE_EQ(context, NumOutputs(node), 1);
  const TfLiteTensor* data;
  TF_LITE_ENSURE_OK(context,
                    GetInputSafe(context, node, kInputDataTensor, &data));
  const TfLiteTensor* segment_ids;
  TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, kInputSegmentIdsTensor,
                                          &segment_ids));
  TfLiteTensor* output;
  TF_LITE_ENSURE_OK(context,
                    GetOutputSafe(context, node, kOutputTensor, &output));
  TF_LITE_ENSURE(context,
                 data->type == kTfLiteInt32 || data->type == kTfLiteFloat32);
  TF_LITE_ENSURE_EQ(context, segment_ids->type, kTfLiteInt32);

  if (!IsConstantTensor(data) || !IsConstantTensor(segment_ids)) {
    SetTensorToDynamic(output);
    return kTfLiteOk;
  }

  return ResizeOutputTensor(context, data, segment_ids, output);
}

TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSsegment_sumDTcc mht_2(mht_2_v, 263, "", "./tensorflow/lite/kernels/segment_sum.cc", "Eval");

  const TfLiteTensor* data;
  TF_LITE_ENSURE_OK(context,
                    GetInputSafe(context, node, kInputDataTensor, &data));
  const TfLiteTensor* segment_ids;
  TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, kInputSegmentIdsTensor,
                                          &segment_ids));
  TfLiteTensor* output;
  TF_LITE_ENSURE_OK(context,
                    GetOutputSafe(context, node, kOutputTensor, &output));

  if (IsDynamicTensor(output)) {
    TF_LITE_ENSURE_OK(context,
                      ResizeOutputTensor(context, data, segment_ids, output));
  }

#define TF_LITE_SEGMENT_SUM(dtype)                                      \
  reference_ops::SegmentSum<dtype>(                                     \
      GetTensorShape(data), GetTensorData<dtype>(data),                 \
      GetTensorShape(segment_ids), GetTensorData<int32_t>(segment_ids), \
      GetTensorShape(output), GetTensorData<dtype>(output));
  switch (data->type) {
    case kTfLiteInt32:
      TF_LITE_SEGMENT_SUM(int32_t);
      break;
    case kTfLiteFloat32:
      TF_LITE_SEGMENT_SUM(float);
      break;
    default:
      context->ReportError(context,
                           "Currently SegmentSum doesn't support type: %s",
                           TfLiteTypeGetName(data->type));
      return kTfLiteError;
  }
#undef TF_LITE_SEGMENT_SUM
  return kTfLiteOk;
}

}  // namespace segment_sum

TfLiteRegistration* Register_SEGMENT_SUM() {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSsegment_sumDTcc mht_3(mht_3_v, 306, "", "./tensorflow/lite/kernels/segment_sum.cc", "Register_SEGMENT_SUM");

  static TfLiteRegistration r = {nullptr, nullptr, segment_sum::Prepare,
                                 segment_sum::Eval};
  return &r;
}

}  // namespace builtin
}  // namespace ops
}  // namespace tflite
