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
class MHTracer_DTPStensorflowPSlitePSkernelsPSembedding_lookupDTcc {
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
   MHTracer_DTPStensorflowPSlitePSkernelsPSembedding_lookupDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePSkernelsPSembedding_lookupDTcc() {
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

// Ops that looks up items from matrix.
//
// Input:
//     Tensor[0]: Row number to lookup, dim.size == 1, int32
//     Tensor[1]: 2-dimensional matrix of multi-dimensional items
//                dim.size >= 2, any data type.
//                first dimension is row, second dimension is column.
//
// Output:
//   Output.dim[0] == Tensor[0].dim[0], num of lookups
//   Output.dim[1] == Tensor[1].dim[1],  num of items per row
//   Each item in output is a raw bytes copy of the corresponding item in input,
//   or a dequantized value in the case of a uint8 input.
//   When indices are out of bound, the ops will not succeed.
//

#include <stdint.h>

#include <cstring>

#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/kernel_util.h"

namespace tflite {
namespace ops {
namespace builtin {
namespace embedding_lookup {

TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSembedding_lookupDTcc mht_0(mht_0_v, 214, "", "./tensorflow/lite/kernels/embedding_lookup.cc", "Prepare");

  TF_LITE_ENSURE_EQ(context, NumInputs(node), 2);
  TF_LITE_ENSURE_EQ(context, NumOutputs(node), 1);

  const TfLiteTensor* lookup;
  TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, 0, &lookup));
  TF_LITE_ENSURE_EQ(context, NumDimensions(lookup), 1);
  TF_LITE_ENSURE_EQ(context, lookup->type, kTfLiteInt32);

  const TfLiteTensor* value;
  TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, 1, &value));
  TF_LITE_ENSURE(context, NumDimensions(value) >= 2);

  TfLiteTensor* output;
  TF_LITE_ENSURE_OK(context, GetOutputSafe(context, node, 0, &output));
  TfLiteIntArray* outputSize = TfLiteIntArrayCreate(NumDimensions(value));

  outputSize->data[0] = SizeOfDimension(lookup, 0);
  outputSize->data[1] = SizeOfDimension(value, 1);
  for (int i = 2; i < NumDimensions(value); i++) {
    outputSize->data[i] = SizeOfDimension(value, i);
  }
  return context->ResizeTensor(context, output, outputSize);
}

TfLiteStatus EvalSimple(TfLiteContext* context, TfLiteNode* node,
                        const TfLiteTensor* lookup, const TfLiteTensor* value,
                        TfLiteTensor* output) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSembedding_lookupDTcc mht_1(mht_1_v, 244, "", "./tensorflow/lite/kernels/embedding_lookup.cc", "EvalSimple");

  const int row_size = SizeOfDimension(value, 0);
  if (row_size == 0) {
    // Propagate empty tensor if input is empty
    return kTfLiteOk;
  }
  const int row_bytes = value->bytes / row_size;

  char* output_raw = GetTensorData<char>(output);
  const char* value_raw = GetTensorData<char>(value);
  const int32_t* lookup_data = GetTensorData<int32_t>(lookup);
  for (int i = 0; i < SizeOfDimension(lookup, 0); i++) {
    int idx = lookup_data[i];
    if (idx >= row_size || idx < 0) {
      context->ReportError(context,
                           "Embedding Lookup: index out of bounds. "
                           "Got %d, and bounds are [0, %d]",
                           idx, row_size - 1);
      return kTfLiteError;
    } else {
      std::memcpy(output_raw + i * row_bytes, value_raw + idx * row_bytes,
                  row_bytes);
    }
  }

  return kTfLiteOk;
}

TfLiteStatus EvalHybrid(TfLiteContext* context, TfLiteNode* node,
                        const TfLiteTensor* lookup, const TfLiteTensor* value,
                        TfLiteTensor* output) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSembedding_lookupDTcc mht_2(mht_2_v, 277, "", "./tensorflow/lite/kernels/embedding_lookup.cc", "EvalHybrid");

  const int row_size = SizeOfDimension(value, 0);
  const double scaling_factor = value->params.scale;

  // col_size after we flatten tensor into 2D.
  int col_size = 1;
  for (int i = 1; i < NumDimensions(value); i++) {
    col_size *= SizeOfDimension(value, i);
  }

  float* output_ptr = GetTensorData<float>(output);
  const int8_t* value_ptr = GetTensorData<int8_t>(value);
  const int32_t* lookup_data = GetTensorData<int32_t>(lookup);

  for (int i = 0; i < SizeOfDimension(lookup, 0); i++) {
    int idx = lookup_data[i];
    if (idx >= row_size || idx < 0) {
      context->ReportError(context,
                           "Embedding Lookup: index out of bounds. "
                           "Got %d, and bounds are [0, %d]",
                           idx, row_size - 1);
      return kTfLiteError;
    } else {
      // Dequantize embedding values.
      // TODO(alanchiao): refactor scalar multiply into separate function
      // for ease of adding a neon equivalent if ever necessary.
      for (int j = 0; j < col_size; j++) {
        output_ptr[j + i * col_size] =
            value_ptr[j + idx * col_size] * scaling_factor;
      }
    }
  }

  return kTfLiteOk;
}

TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSembedding_lookupDTcc mht_3(mht_3_v, 316, "", "./tensorflow/lite/kernels/embedding_lookup.cc", "Eval");

  const TfLiteTensor* lookup;
  TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, 0, &lookup));
  const TfLiteTensor* value;
  TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, 1, &value));
  TfLiteTensor* output;
  TF_LITE_ENSURE_OK(context, GetOutputSafe(context, node, 0, &output));
  switch (value->type) {
    case kTfLiteFloat32:
      return EvalSimple(context, node, lookup, value, output);
    case kTfLiteUInt8:
    case kTfLiteInt8:
      if (output->type == kTfLiteFloat32) {
        return EvalHybrid(context, node, lookup, value, output);
      } else {
        return EvalSimple(context, node, lookup, value, output);
      }
    default:
      context->ReportError(context, "Type not currently supported.");
      return kTfLiteError;
  }
}

}  // namespace embedding_lookup

TfLiteRegistration* Register_EMBEDDING_LOOKUP() {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSembedding_lookupDTcc mht_4(mht_4_v, 344, "", "./tensorflow/lite/kernels/embedding_lookup.cc", "Register_EMBEDDING_LOOKUP");

  static TfLiteRegistration r = {nullptr, nullptr, embedding_lookup::Prepare,
                                 embedding_lookup::Eval};
  return &r;
}

}  // namespace builtin
}  // namespace ops
}  // namespace tflite
