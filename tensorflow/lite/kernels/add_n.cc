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
class MHTracer_DTPStensorflowPSlitePSkernelsPSadd_nDTcc {
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
   MHTracer_DTPStensorflowPSlitePSkernelsPSadd_nDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePSkernelsPSadd_nDTcc() {
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
#include "tensorflow/lite/kernels/cpu_backend_threadpool.h"
#include "tensorflow/lite/kernels/internal/optimized/optimized_ops.h"
#include "tensorflow/lite/kernels/internal/tensor.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/kernel_util.h"

namespace tflite {
namespace ops {
namespace builtin {
namespace add_n {

constexpr int kInputTensor1 = 0;
constexpr int kOutputTensor = 0;

struct OpData {
  // The index of the temporary tensor where temporary accumulations are kept.
  int scratch_tensor_index;
};

void* Init(TfLiteContext* context, const char* buffer, size_t length) {
   std::vector<std::string> mht_0_v;
   mht_0_v.push_back("buffer: \"" + (buffer == nullptr ? std::string("nullptr") : std::string((char*)buffer)) + "\"");
   MHTracer_DTPStensorflowPSlitePSkernelsPSadd_nDTcc mht_0(mht_0_v, 207, "", "./tensorflow/lite/kernels/add_n.cc", "Init");

  auto* op_data = new OpData();
  context->AddTensors(context, 1, &op_data->scratch_tensor_index);
  return op_data;
}

void Free(TfLiteContext* context, void* buffer) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSadd_nDTcc mht_1(mht_1_v, 216, "", "./tensorflow/lite/kernels/add_n.cc", "Free");

  delete reinterpret_cast<OpData*>(buffer);
}

TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSadd_nDTcc mht_2(mht_2_v, 223, "", "./tensorflow/lite/kernels/add_n.cc", "Prepare");

  int num_inputs = NumInputs(node);
  TF_LITE_ENSURE(context, num_inputs >= 2);
  TF_LITE_ENSURE_EQ(context, NumOutputs(node), 1);

  const TfLiteTensor* input1;
  TF_LITE_ENSURE_OK(context,
                    GetInputSafe(context, node, kInputTensor1, &input1));
  TfLiteTensor* output;
  TF_LITE_ENSURE_OK(context,
                    GetOutputSafe(context, node, kOutputTensor, &output));
  output->type = input1->type;

  OpData* op_data = reinterpret_cast<OpData*>(node->user_data);
  TfLiteIntArrayFree(node->temporaries);
  node->temporaries = TfLiteIntArrayCreate(1);
  node->temporaries->data[0] = op_data->scratch_tensor_index;
  TfLiteTensor* scratch_tensor;
  TF_LITE_ENSURE_OK(
      context, GetTemporarySafe(context, node, /*index=*/0, &scratch_tensor));
  scratch_tensor->type = input1->type;
  scratch_tensor->allocation_type = kTfLiteArenaRw;

  CpuBackendContext* cpu_backend_context =
      CpuBackendContext::GetFromContext(context);
  // Choose the proper number of thread so that:
  // (1) Each thread gets at least two tensors (1 if we only have 1 input
  // tensor).
  // (2) Total thread_count should be bounded by the maximimum allowed threads.
  // (3) Tensors are distributed evenly across different threads.
  const int thread_count =
      std::min(std::max(1, static_cast<int>(num_inputs) / 2),
               cpu_backend_context->max_num_threads());

  TfLiteIntArray* scratch_shape = TfLiteIntArrayCreate(1);
  scratch_shape->data[0] = thread_count * NumElements(input1);
  TF_LITE_ENSURE_OK(
      context, context->ResizeTensor(context, scratch_tensor, scratch_shape));

  // Check that all input tensors have the same shape and type.
  for (int i = kInputTensor1 + 1; i < num_inputs; ++i) {
    const TfLiteTensor* input;
    TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, i, &input));
    TF_LITE_ENSURE(context, HaveSameShapes(input1, input));
    TF_LITE_ENSURE_TYPES_EQ(context, input1->type, input->type);
  }

  // Use the first input node's dimension to be the dimension of the output
  // node.
  TfLiteIntArray* input1_dims = input1->dims;
  TfLiteIntArray* output_dims = TfLiteIntArrayCopy(input1_dims);
  return context->ResizeTensor(context, output, output_dims);
}

template <typename T>
TfLiteStatus EvalAddN(TfLiteContext* context, TfLiteNode* node) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSadd_nDTcc mht_3(mht_3_v, 281, "", "./tensorflow/lite/kernels/add_n.cc", "EvalAddN");

  // TODO(haoliang): Initialize all_inputs only once during init.
  VectorOfTensors<T> all_inputs(*context, *node->inputs);
  // Safe to use unchecked since caller checks that tensor is valid
  TfLiteTensor* output = GetOutput(context, node, kOutputTensor);
  int num_inputs = NumInputs(node);
  // Safe to use unchecked since caller checks that tensor is valid
  const TfLiteTensor* input1 = GetInput(context, node, kInputTensor1);

  // Fetch backend context and number of threads.
  CpuBackendContext* cpu_backend_context =
      CpuBackendContext::GetFromContext(context);
  TfLiteTensor* scratch_tensor;
  TF_LITE_ENSURE_OK(context,
                    GetTemporarySafe(context, node, 0, &scratch_tensor));
  optimized_ops::AddN<T>(GetTensorShape(input1), num_inputs, all_inputs.data(),
                         GetTensorData<T>(output),
                         GetTensorData<T>(scratch_tensor), cpu_backend_context);
  return kTfLiteOk;
}

TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSadd_nDTcc mht_4(mht_4_v, 305, "", "./tensorflow/lite/kernels/add_n.cc", "Eval");

  const TfLiteTensor* input1;
  TF_LITE_ENSURE_OK(context,
                    GetInputSafe(context, node, kInputTensor1, &input1));
  TfLiteTensor* output;
  TF_LITE_ENSURE_OK(context,
                    GetOutputSafe(context, node, kOutputTensor, &output));
  if (output->type == kTfLiteFloat32) {
    TF_LITE_ENSURE_OK(context, EvalAddN<float>(context, node));
  } else if (output->type == kTfLiteInt32) {
    TF_LITE_ENSURE_OK(context, EvalAddN<int32_t>(context, node));
  } else {
    context->ReportError(context,
                         "AddN only supports FLOAT32|INT32 now, got %s.",
                         TfLiteTypeGetName(output->type));
    return kTfLiteError;
  }
  return kTfLiteOk;
}

}  // namespace add_n

TfLiteRegistration* Register_ADD_N() {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSadd_nDTcc mht_5(mht_5_v, 330, "", "./tensorflow/lite/kernels/add_n.cc", "Register_ADD_N");

  static TfLiteRegistration r = {add_n::Init, add_n::Free, add_n::Prepare,
                                 add_n::Eval};
  return &r;
}

}  // namespace builtin
}  // namespace ops
}  // namespace tflite
