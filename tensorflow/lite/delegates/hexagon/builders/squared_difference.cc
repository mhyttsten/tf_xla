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
class MHTracer_DTPStensorflowPSlitePSdelegatesPShexagonPSbuildersPSsquared_differenceDTcc {
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
   MHTracer_DTPStensorflowPSlitePSdelegatesPShexagonPSbuildersPSsquared_differenceDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePSdelegatesPShexagonPSbuildersPSsquared_differenceDTcc() {
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
#include "tensorflow/lite/delegates/hexagon/builders/op_builder.h"

namespace tflite {
namespace delegates {
namespace hexagon {
// Builder for SquaredDifference op by computing Mul(Sub(A,B), Sub(A,B))
class SquaredDifferenceOpBuilder : public OpBuilder {
 public:
  explicit SquaredDifferenceOpBuilder(GraphBuilder* graph_builder, int op_type)
      : OpBuilder(graph_builder, op_type) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPShexagonPSbuildersPSsquared_differenceDTcc mht_0(mht_0_v, 193, "", "./tensorflow/lite/delegates/hexagon/builders/squared_difference.cc", "SquaredDifferenceOpBuilder");
}
  TfLiteStatus PopulateSubGraph(const TfLiteIntArray* inputs,
                                const TfLiteIntArray* outputs,
                                TfLiteContext* context) override;

  TfLiteStatus RegisterOutputs(const TfLiteIntArray* outputs,
                               TfLiteContext* context) override;

 private:
  TensorID node_output_;
};

TfLiteStatus SquaredDifferenceOpBuilder::PopulateSubGraph(
    const TfLiteIntArray* inputs, const TfLiteIntArray* outputs,
    TfLiteContext* context) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPShexagonPSbuildersPSsquared_differenceDTcc mht_1(mht_1_v, 210, "", "./tensorflow/lite/delegates/hexagon/builders/squared_difference.cc", "SquaredDifferenceOpBuilder::PopulateSubGraph");

  // We model Squared Diff as Mul(Sub(a,b), Sub(a,b))

  // Add first Sub op.
  const int tensor_a_index = inputs->data[0];
  const int tensor_b_index = inputs->data[1];
  const auto& tensor_a = context->tensors[tensor_a_index];
  const auto& tensor_b = context->tensors[tensor_b_index];
  AddInput(graph_builder_->GetHexagonTensorId(tensor_a_index));
  AddInput(graph_builder_->GetHexagonTensorId(tensor_b_index));
  // Inputs min/max
  TF_LITE_ENSURE_STATUS(ComputeAndAddMinAndMax(context, tensor_a));
  TF_LITE_ENSURE_STATUS(ComputeAndAddMinAndMax(context, tensor_b));
  // Output details.
  float output_min = -1, output_max = -1;
  TF_LITE_ENSURE_STATUS(ComputeMinAndMaxQuantValues(
      context->tensors[outputs->data[0]], &output_min, &output_max));
  auto* output_min_const = graph_builder_->AddConstNodeWithData(
      kScalarShape, reinterpret_cast<char*>(&output_min), sizeof(output_min));
  auto* output_max_const = graph_builder_->AddConstNodeWithData(
      kScalarShape, reinterpret_cast<char*>(&output_max), sizeof(output_max));
  int output_batch_size, output_height_size, output_width_size,
      output_depth_size;
  GetDims(&output_batch_size, &output_height_size, &output_width_size,
          &output_depth_size, context->tensors[outputs->data[0]].dims);

  auto sub_out = AddOutput(sizeof(uint8_t), 4,
                           {output_batch_size, output_height_size,
                            output_width_size, output_depth_size});
  auto sub_min = AddOutput(sizeof(float), 4, kScalarShape);
  auto sub_max = AddOutput(sizeof(float), 4, kScalarShape);

  // Add Mul
  auto* mul_op = graph_builder_->AddNode(GetTFLiteNodeID());
  mul_op->SetOpType(OP_QuantizedMul_8x8to8);
  mul_op->AddInput(sub_out);
  mul_op->AddInput(sub_out);
  mul_op->AddInput(sub_min);
  mul_op->AddInput(sub_max);
  mul_op->AddInput(sub_min);
  mul_op->AddInput(sub_max);
  mul_op->AddInput(TensorID(output_min_const->GetID(), 0));
  mul_op->AddInput(TensorID(output_max_const->GetID(), 0));
  node_output_ = mul_op->AddOutput(sizeof(uint8_t), 4,
                                   {output_batch_size, output_height_size,
                                    output_width_size, output_depth_size});
  mul_op->AddOutput(sizeof(float), 4, kScalarShape);
  mul_op->AddOutput(sizeof(float), 4, kScalarShape);

  return kTfLiteOk;
}

TfLiteStatus SquaredDifferenceOpBuilder::RegisterOutputs(
    const TfLiteIntArray* outputs, TfLiteContext* context) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPShexagonPSbuildersPSsquared_differenceDTcc mht_2(mht_2_v, 266, "", "./tensorflow/lite/delegates/hexagon/builders/squared_difference.cc", "SquaredDifferenceOpBuilder::RegisterOutputs");

  graph_builder_->AddTensorWithID(outputs->data[0], node_output_.first,
                                  node_output_.second);
  return kTfLiteOk;
}

OpBuilder* CreateSquaredDifferenceOpBuilder(GraphBuilder* graph_builder,
                                            int op_type) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPShexagonPSbuildersPSsquared_differenceDTcc mht_3(mht_3_v, 276, "", "./tensorflow/lite/delegates/hexagon/builders/squared_difference.cc", "CreateSquaredDifferenceOpBuilder");

  return new SquaredDifferenceOpBuilder(graph_builder, op_type);
}

}  // namespace hexagon
}  // namespace delegates
}  // namespace tflite
