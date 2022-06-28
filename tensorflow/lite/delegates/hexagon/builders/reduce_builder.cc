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
class MHTracer_DTPStensorflowPSlitePSdelegatesPShexagonPSbuildersPSreduce_builderDTcc {
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
   MHTracer_DTPStensorflowPSlitePSdelegatesPShexagonPSbuildersPSreduce_builderDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePSdelegatesPShexagonPSbuildersPSreduce_builderDTcc() {
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
#include "tensorflow/lite/delegates/hexagon/builders/reduce_builder.h"

#include <stdint.h>

#include <limits>

#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/delegates/hexagon/hexagon_nn/hexagon_nn.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/util.h"

namespace tflite {
namespace delegates {
namespace hexagon {
TfLiteStatus ReduceOpBuilder::PopulateSubGraph(const TfLiteIntArray* inputs,
                                               const TfLiteIntArray* outputs,
                                               TfLiteContext* context) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPShexagonPSbuildersPSreduce_builderDTcc mht_0(mht_0_v, 200, "", "./tensorflow/lite/delegates/hexagon/builders/reduce_builder.cc", "ReduceOpBuilder::PopulateSubGraph");

  // Input data tensor.
  int tensor_id = inputs->data[0];
  const auto& input_tensor = context->tensors[tensor_id];
  AddInput(graph_builder_->GetHexagonTensorId(tensor_id));
  TF_LITE_ENSURE_STATUS(ComputeAndAddMinAndMax(context, input_tensor));

  // Axes tensor should be constant.
  tensor_id = inputs->data[1];
  const auto& axes_tensor = context->tensors[tensor_id];
  if (axes_tensor.allocation_type == kTfLiteMmapRo) {
    // If the axes input is a constant, bake it into the Hexagon graph as a
    // Const node.
    auto* const_axes_node =
        graph_builder_->AddConstNodeWithData(tensor_id, axes_tensor);
    AddInput(TensorID(const_axes_node->GetID(), 0));
  } else {
    context->ReportError(context, "Reduction op doesn't have constant axis");
    return kTfLiteError;
  }

  auto& output_tensor = context->tensors[outputs->data[0]];
  int output_batch_size, output_height_size, output_width_size,
      output_depth_size;
  GetDims(&output_batch_size, &output_height_size, &output_width_size,
          &output_depth_size, output_tensor.dims);

  float output_min = -1, output_max = -1;
  ComputeMinAndMaxQuantValues(output_tensor, &output_min, &output_max);
  auto* output_min_const = graph_builder_->AddConstNodeWithData(
      kScalarShape, reinterpret_cast<char*>(&output_min), sizeof(output_min));
  auto* output_max_const = graph_builder_->AddConstNodeWithData(
      kScalarShape, reinterpret_cast<char*>(&output_max), sizeof(output_max));
  // Min/max values for output tensor.
  AddInput(TensorID(output_min_const->GetID(), 0));
  AddInput(TensorID(output_max_const->GetID(), 0));

  // Add outputs
  size_t output_element_size = 0;
  TF_LITE_ENSURE_STATUS(
      GetSizeOfType(context, output_tensor.type, &output_element_size));
  auto mean_output = AddOutput(output_element_size, 4,
                               {output_batch_size, output_height_size,
                                output_width_size, output_depth_size});
  auto mean_out_min = AddOutput(output_element_size, 4, kScalarShape);
  auto mean_out_max = AddOutput(output_element_size, 4, kScalarShape);
  // Mean op doesn't honor the passed min/max for output, so we need
  // to add requantize.
  auto* requantize_op = graph_builder_->AddNode(GetTFLiteNodeID());
  requantize_op->SetOpType(OP_Requantize_8to8);
  requantize_op->AddInput(mean_output);
  requantize_op->AddInput(mean_out_min);
  requantize_op->AddInput(mean_out_max);
  requantize_op->AddInput(TensorID(output_min_const->GetID(), 0));
  requantize_op->AddInput(TensorID(output_max_const->GetID(), 0));
  node_output_ =
      requantize_op->AddOutput(sizeof(uint8_t), 4,
                               {output_batch_size, output_height_size,
                                output_width_size, output_depth_size});
  requantize_op->AddOutput(sizeof(float), 4, kScalarShape);
  requantize_op->AddOutput(sizeof(float), 4, kScalarShape);

  return kTfLiteOk;
}

TfLiteStatus ReduceOpBuilder::RegisterOutputs(const TfLiteIntArray* outputs,
                                              TfLiteContext* context) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPShexagonPSbuildersPSreduce_builderDTcc mht_1(mht_1_v, 269, "", "./tensorflow/lite/delegates/hexagon/builders/reduce_builder.cc", "ReduceOpBuilder::RegisterOutputs");

  // Should be only 1 output.
  graph_builder_->AddTensorWithID(outputs->data[0], node_output_.first,
                                  node_output_.second);

  return kTfLiteOk;
}

ReduceOpBuilder::~ReduceOpBuilder() {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPShexagonPSbuildersPSreduce_builderDTcc mht_2(mht_2_v, 280, "", "./tensorflow/lite/delegates/hexagon/builders/reduce_builder.cc", "ReduceOpBuilder::~ReduceOpBuilder");
}

OpBuilder* CreateReduceBuilder(GraphBuilder* graph_builder, int op_type) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPShexagonPSbuildersPSreduce_builderDTcc mht_3(mht_3_v, 285, "", "./tensorflow/lite/delegates/hexagon/builders/reduce_builder.cc", "CreateReduceBuilder");

  return new ReduceOpBuilder(graph_builder, op_type);
}

}  // namespace hexagon
}  // namespace delegates
}  // namespace tflite
