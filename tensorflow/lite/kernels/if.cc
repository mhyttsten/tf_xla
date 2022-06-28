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
class MHTracer_DTPStensorflowPSlitePSkernelsPSifDTcc {
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
   MHTracer_DTPStensorflowPSlitePSkernelsPSifDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePSkernelsPSifDTcc() {
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

#include <cstring>
#include <memory>
#include <vector>

#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/core/subgraph.h"
#include "tensorflow/lite/kernels/internal/compatibility.h"
#include "tensorflow/lite/kernels/kernel_util.h"

namespace tflite {
namespace ops {
namespace builtin {
namespace if_kernel {

struct OpData {
  int then_subgraph_index;
  int else_subgraph_index;
};

void* Init(TfLiteContext* context, const char* buffer, size_t length) {
   std::vector<std::string> mht_0_v;
   mht_0_v.push_back("buffer: \"" + (buffer == nullptr ? std::string("nullptr") : std::string((char*)buffer)) + "\"");
   MHTracer_DTPStensorflowPSlitePSkernelsPSifDTcc mht_0(mht_0_v, 208, "", "./tensorflow/lite/kernels/if.cc", "Init");

  auto* op_data = new OpData;
  const auto* params = reinterpret_cast<const TfLiteIfParams*>(buffer);
  op_data->then_subgraph_index = params->then_subgraph_index;
  op_data->else_subgraph_index = params->else_subgraph_index;
  return op_data;
}

void Free(TfLiteContext* context, void* buffer) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSifDTcc mht_1(mht_1_v, 219, "", "./tensorflow/lite/kernels/if.cc", "Free");

  delete reinterpret_cast<OpData*>(buffer);
}

TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSifDTcc mht_2(mht_2_v, 226, "", "./tensorflow/lite/kernels/if.cc", "Prepare");

  const OpData* op_data = reinterpret_cast<OpData*>(node->user_data);

  TF_LITE_ENSURE(context, node->inputs->size > 0);

  // The first input is the condition.
  const TfLiteTensor* cond;
  TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, 0, &cond));
  // Currently only bool is supported.
  // TODO(ycling): Support other types since TensorFlow also support
  // non-bool types as condition.
  TF_LITE_ENSURE_EQ(context, cond->type, kTfLiteBool);
  TF_LITE_ENSURE_EQ(context, NumElements(cond), 1);

  // The first input of the node is the condition. The rest of inputs are
  // passed to the branch subgraphs. Therefore, the number of subgraph inputs
  // will be the number of node inputs - 1.
  int num_inputs = node->inputs->size - 1;
  int num_outputs = node->outputs->size;

  Subgraph* this_subgraph = reinterpret_cast<Subgraph*>(context->impl_);
  auto* subgraphs = this_subgraph->GetSubgraphs();
  TF_LITE_ENSURE(context, op_data->then_subgraph_index < subgraphs->size());
  TF_LITE_ENSURE(context, op_data->else_subgraph_index < subgraphs->size());

  Subgraph* then_subgraph = (*subgraphs)[op_data->then_subgraph_index].get();
  Subgraph* else_subgraph = (*subgraphs)[op_data->else_subgraph_index].get();

  for (auto* subgraph : {then_subgraph, else_subgraph}) {
    TF_LITE_ENSURE_EQ(context, num_inputs, subgraph->inputs().size());
    TF_LITE_ENSURE_EQ(context, num_outputs, subgraph->outputs().size());
  }

  bool has_dynamic_output_tensors = false;
  for (auto* subgraph : {then_subgraph, else_subgraph}) {
    for (int i = 0; i < num_inputs; ++i) {
      // The first input of the node is the condition. The indices of the inputs
      // passed to the subgraphs are offset by 1.
      const TfLiteTensor* input;
      TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, i + 1, &input));
      std::vector<int> dims(input->dims->data,
                            input->dims->data + input->dims->size);
      subgraph->ResizeInputTensor(i, dims);
      TfLiteTensor* subgraph_input = subgraph->tensor(subgraph->inputs()[i]);
      TF_LITE_ENSURE_TYPES_EQ(context, input->type, subgraph_input->type);
    }
    // Note: The `Prepare` function is responsible to run `AllocateTensors` on
    // both subgraphs. It's intentionally not to break out of the loop when
    // finding a dynamic output tensor.
    TF_LITE_ENSURE_OK(context, subgraph->AllocateTensors());
    has_dynamic_output_tensors |= subgraph->HasDynamicTensors();
  }

  if (!has_dynamic_output_tensors) {
    for (int i = 0; i < num_outputs; ++i) {
      TfLiteTensor* then_output =
          then_subgraph->tensor(then_subgraph->outputs()[i]);
      TfLiteTensor* else_output =
          else_subgraph->tensor(else_subgraph->outputs()[i]);
      // If the 2 subgraphs have static but different output shapes, the output
      // tensors of the IF op have dynamic sizes.
      if (!TfLiteIntArrayEqual(then_output->dims, else_output->dims)) {
        has_dynamic_output_tensors = true;
        break;
      }
    }
  }

  for (int i = 0; i < num_outputs; ++i) {
    TfLiteTensor* output;
    TF_LITE_ENSURE_OK(context, GetOutputSafe(context, node, i, &output));
    if (has_dynamic_output_tensors) {
      SetTensorToDynamic(output);
    } else {
      // When there's no dynamic output tensors, the 2 subgraph has exactly
      // the same static sized outputs.
      TfLiteTensor* then_output =
          then_subgraph->tensor(then_subgraph->outputs()[i]);
      TfLiteIntArray* output_size = TfLiteIntArrayCopy(then_output->dims);
      TF_LITE_ENSURE_OK(context,
                        context->ResizeTensor(context, output, output_size));
    }
  }

  return kTfLiteOk;
}

TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSifDTcc mht_3(mht_3_v, 316, "", "./tensorflow/lite/kernels/if.cc", "Eval");

  const OpData* op_data = reinterpret_cast<OpData*>(node->user_data);

  const TfLiteTensor* cond;
  TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, 0, &cond));
  bool cond_value = cond->data.b[0];

  Subgraph* this_subgraph = reinterpret_cast<Subgraph*>(context->impl_);
  auto* subgraphs = this_subgraph->GetSubgraphs();

  // Currently we copy the input / output between the subgraphs. This isn't
  // optimized yet.
  // TODO(b/120234921): Optimize and avoid copying tensors between subgraphs.
  int active_branch_subgraph_index =
      cond_value ? op_data->then_subgraph_index : op_data->else_subgraph_index;
  Subgraph& active_branch_subgraph =
      *(*subgraphs)[active_branch_subgraph_index];
  for (int i = 0; i < active_branch_subgraph.inputs().size(); ++i) {
    const TfLiteTensor* input;
    TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, i + 1, &input));
    TfLiteTensor* subgraph_input =
        active_branch_subgraph.tensor(active_branch_subgraph.inputs()[i]);

    if (IsDynamicTensor(subgraph_input)) {
      TfLiteTensorRealloc(input->bytes, subgraph_input);
    }

    TF_LITE_ENSURE_EQ(context, input->bytes, subgraph_input->bytes);
    TfLiteTensorCopy(input, subgraph_input);
  }

  // Note: It's guaranteed that the subgraphs' `AllocateTensors` are called
  // in `Prepare`, so we don't need to do it here again.
  TF_LITE_ENSURE_OK(context, active_branch_subgraph.Invoke());

  for (int tensor_index : active_branch_subgraph.outputs()) {
    active_branch_subgraph.EnsureTensorDataIsReadable(tensor_index);
  }

  bool has_dynamic_output_tensors = false;
  for (int i = 0; i < node->outputs->size; ++i) {
    TfLiteTensor* output;
    TF_LITE_ENSURE_OK(context, GetOutputSafe(context, node, i, &output));
    if (IsDynamicTensor(output)) {
      has_dynamic_output_tensors = true;
      break;
    }
  }

  if (has_dynamic_output_tensors) {
    for (int i = 0; i < node->outputs->size; ++i) {
      TfLiteTensor* output;
      TF_LITE_ENSURE_OK(context, GetOutputSafe(context, node, i, &output));
      TfLiteTensor* subgraph_output =
          active_branch_subgraph.tensor(active_branch_subgraph.outputs()[i]);
      TfLiteIntArray* output_size = TfLiteIntArrayCopy(subgraph_output->dims);
      TF_LITE_ENSURE_OK(context,
                        context->ResizeTensor(context, output, output_size));
    }
  }

  for (int i = 0; i < active_branch_subgraph.outputs().size(); ++i) {
    const TfLiteTensor* subgraph_output =
        active_branch_subgraph.tensor(active_branch_subgraph.outputs()[i]);
    TfLiteTensor* output;
    TF_LITE_ENSURE_OK(context, GetOutputSafe(context, node, i, &output));

    if (IsDynamicTensor(output)) {
      TfLiteTensorRealloc(subgraph_output->bytes, output);
    }

    TF_LITE_ENSURE_EQ(context, output->bytes, subgraph_output->bytes);
    TfLiteTensorCopy(subgraph_output, output);
  }
  return kTfLiteOk;
}

}  // namespace if_kernel

TfLiteRegistration* Register_IF() {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSifDTcc mht_4(mht_4_v, 398, "", "./tensorflow/lite/kernels/if.cc", "Register_IF");

  static TfLiteRegistration r = {if_kernel::Init, if_kernel::Free,
                                 if_kernel::Prepare, if_kernel::Eval};
  return &r;
}

}  // namespace builtin
}  // namespace ops
}  // namespace tflite
