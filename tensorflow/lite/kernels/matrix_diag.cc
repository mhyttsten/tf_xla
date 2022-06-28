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
class MHTracer_DTPStensorflowPSlitePSkernelsPSmatrix_diagDTcc {
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
   MHTracer_DTPStensorflowPSlitePSkernelsPSmatrix_diagDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePSkernelsPSmatrix_diagDTcc() {
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

/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/lite/kernels/internal/optimized/optimized_ops.h"
#include "tensorflow/lite/kernels/internal/reference/reference_ops.h"
#include "tensorflow/lite/kernels/internal/tensor.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/kernel_util.h"

namespace tflite {
namespace ops {
namespace builtin {
namespace matrix_diag {

constexpr int kInputTensor = 0;
constexpr int kOutputTensor = 0;

TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSmatrix_diagDTcc mht_0(mht_0_v, 201, "", "./tensorflow/lite/kernels/matrix_diag.cc", "Prepare");

  TF_LITE_ENSURE_EQ(context, NumInputs(node), 1);
  TF_LITE_ENSURE_EQ(context, NumOutputs(node), 1);
  const TfLiteTensor* input;
  TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, kInputTensor, &input));
  TfLiteIntArray* input_dims = input->dims;
  int input_dims_size = input_dims->size;
  TF_LITE_ENSURE(context, input_dims_size >= 1);

  TfLiteTensor* output;
  TF_LITE_ENSURE_OK(context,
                    GetOutputSafe(context, node, kOutputTensor, &output));
  // Resize the output tensor.
  TfLiteIntArray* output_shape = TfLiteIntArrayCreate(input_dims_size + 1);
  for (int i = 0; i < input_dims_size; i++) {
    output_shape->data[i] = input_dims->data[i];
  }
  // Last dimension in the output is the same as the last dimension in the
  // input.
  output_shape->data[input_dims_size] = input_dims->data[input_dims_size - 1];
  output->type = input->type;
  TF_LITE_ENSURE_OK(context,
                    context->ResizeTensor(context, output, output_shape));

  return kTfLiteOk;
}

// Fill the tensor to make a diagonal matrix in each batch, i.e., when
// row index and column index are the same, fill with the next input value.
// All other entries get zero.
// TODO(b/128636574) Move to reference_ops.
template <typename T>
void FillDiagImpl(const T* in, T* out, const int batch_size, const int row_size,
                  const int col_size) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSmatrix_diagDTcc mht_1(mht_1_v, 237, "", "./tensorflow/lite/kernels/matrix_diag.cc", "FillDiagImpl");

  int idx = 0;
  for (int b = 0; b < batch_size; b++) {
    for (int i = 0; i < row_size; i++) {
      for (int j = 0; j < col_size; ++j) {
        // input values go on the diagonal, 0 elsewhere
        if (i == j) {
          out[i * col_size + j] = in[idx];
          idx++;
        } else {
          out[i * col_size + j] = 0;
        }
      }
    }
    out += row_size * col_size;
  }
}

template <typename T>
void FillDiag(const TfLiteTensor* input, TfLiteTensor* output,
              const int batch_size, const int row_size, const int col_size) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSmatrix_diagDTcc mht_2(mht_2_v, 260, "", "./tensorflow/lite/kernels/matrix_diag.cc", "FillDiag");

  FillDiagImpl<T>(GetTensorData<T>(input), GetTensorData<T>(output), batch_size,
                  row_size, col_size);
}

// Fill a tensor with given input on the diagonal, zero elsewhere
void FillDiagHelper(const TfLiteTensor* input, TfLiteTensor* output) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSmatrix_diagDTcc mht_3(mht_3_v, 269, "", "./tensorflow/lite/kernels/matrix_diag.cc", "FillDiagHelper");

  const int num_output_dims = output->dims->size;
  int batch_size = 1;
  for (int i = 0; i < num_output_dims - 2; ++i) {
    batch_size *= output->dims->data[i];
  }

  const int row_size = output->dims->data[num_output_dims - 2];
  const int col_size = output->dims->data[num_output_dims - 1];
  switch (output->type) {
    case kTfLiteInt64: {
      return FillDiag<int64_t>(input, output, batch_size, row_size, col_size);
    }
    case kTfLiteInt32: {
      return FillDiag<int32_t>(input, output, batch_size, row_size, col_size);
    }
    case kTfLiteInt16: {
      return FillDiag<int16_t>(input, output, batch_size, row_size, col_size);
    }
    case kTfLiteInt8: {
      return FillDiag<int8_t>(input, output, batch_size, row_size, col_size);
    }
    case kTfLiteUInt8: {
      return FillDiag<uint8_t>(input, output, batch_size, row_size, col_size);
    }
    default:
      return FillDiag<float>(input, output, batch_size, row_size, col_size);
  }
}

TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSmatrix_diagDTcc mht_4(mht_4_v, 302, "", "./tensorflow/lite/kernels/matrix_diag.cc", "Eval");

  TfLiteTensor* output;
  TF_LITE_ENSURE_OK(context,
                    GetOutputSafe(context, node, kOutputTensor, &output));
  const TfLiteTensor* input;
  TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, kInputTensor, &input));
  FillDiagHelper(input, output);
  return kTfLiteOk;
}

}  // namespace matrix_diag

TfLiteRegistration* Register_MATRIX_DIAG() {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSmatrix_diagDTcc mht_5(mht_5_v, 317, "", "./tensorflow/lite/kernels/matrix_diag.cc", "Register_MATRIX_DIAG");

  static TfLiteRegistration r = {nullptr, nullptr, matrix_diag::Prepare,
                                 matrix_diag::Eval};
  return &r;
}

}  // namespace builtin
}  // namespace ops
}  // namespace tflite
