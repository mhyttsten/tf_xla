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
#ifndef TENSORFLOW_LITE_DELEGATES_HEXAGON_BUILDERS_CONV_2D_BUILDER_H_
#define TENSORFLOW_LITE_DELEGATES_HEXAGON_BUILDERS_CONV_2D_BUILDER_H_
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
class MHTracer_DTPStensorflowPSlitePSdelegatesPShexagonPSbuildersPSconv_2d_builderDTh {
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
   MHTracer_DTPStensorflowPSlitePSdelegatesPShexagonPSbuildersPSconv_2d_builderDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePSdelegatesPShexagonPSbuildersPSconv_2d_builderDTh() {
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


#include <vector>

#include "tensorflow/lite/delegates/hexagon/builders/op_builder.h"

namespace tflite {
namespace delegates {
namespace hexagon {

// Stores quantization data for Conv/TransposeConv nodes.
// This information is used to handle the per-channel quantized weights & biases
// correctly in the Hexagon delegate.
struct PerChannelQuantData {
  // This is initialized while processing quantized weights, and acts as an
  // input to Hexagon Conv nodes.
  OpBuilder* channel_scales_node = nullptr;
  // Scale information is obtained from TfLiteAffineQuantization in the weights
  // tensor.
  float* scales_data = nullptr;
  int num_scale_values = 1;
};

class Conv2dOpBuilder : public OpBuilder {
 public:
  explicit Conv2dOpBuilder(GraphBuilder* graph_builder, int op_type)
      : OpBuilder(graph_builder, op_type) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPShexagonPSbuildersPSconv_2d_builderDTh mht_0(mht_0_v, 211, "", "./tensorflow/lite/delegates/hexagon/builders/conv_2d_builder.h", "Conv2dOpBuilder");
}
  TfLiteStatus PopulateSubGraph(const TfLiteIntArray* inputs,
                                const TfLiteIntArray* outputs,
                                TfLiteContext* context) override;

  TfLiteStatus RegisterOutputs(const TfLiteIntArray* outputs,
                               TfLiteContext* context) override;

  ~Conv2dOpBuilder() override;

 private:
  TfLiteStatus InitializeWeightsNodes(const TfLiteIntArray* inputs,
                                      const TfLiteIntArray* outputs,
                                      TfLiteContext* context,
                                      const int input_depth);

  TfLiteStatus InitializeBiasNodes(const TfLiteIntArray* inputs,
                                   const TfLiteIntArray* outputs,
                                   TfLiteContext* context);

  TensorID node_output_;
  std::vector<float> transposed_weights_;
  std::vector<int> stride_shape_;
  std::vector<int> weight_shape_;
  OpBuilder* weights_min_node_ = nullptr;
  OpBuilder* weights_max_node_ = nullptr;
  OpBuilder* bias_min_node_ = nullptr;
  OpBuilder* bias_max_node_ = nullptr;

  // Modified only if node has per-channel quantized weights/biases.
  PerChannelQuantData per_channel_quant_;

  // Only used for dilated Depthwise Conv.
  std::vector<int> dilation_factors_h_w_;
  std::vector<int> space_to_batch_paddings_;
  std::vector<int> batch_to_space_crops_;
};

// ProcessPerChannelQuantizedWeights & ProcessPerChannelQuantizedBias can be
// used to pre-process per-channel quantized weights & biases for Hexagon.
// NOTE: ProcessPerChannelQuantizedWeights should be run before
// ProcessPerChannelQuantizedBias. This is becase we set PerChannelQuantData
// based on the weights tensor, which is utilized while preprocessing bias.

TfLiteStatus ProcessPerChannelQuantizedWeights(
    const TfLiteTensor& weights_tensor, TfLiteContext* context,
    float* weights_min, float* weights_max, GraphBuilder* graph_builder,
    PerChannelQuantData* per_channel_quant);

TfLiteStatus ProcessPerChannelQuantizedBias(
    const TfLiteTensor& data_tensor, const TfLiteTensor& bias_tensor,
    const int bias_tensor_idx, TfLiteContext* context, float* bias_min,
    float* bias_max, GraphBuilder* graph_builder,
    PerChannelQuantData* per_channel_quant,
    OpBuilder** bias_const_node = nullptr);

}  // namespace hexagon
}  // namespace delegates
}  // namespace tflite

#endif  // TENSORFLOW_LITE_DELEGATES_HEXAGON_BUILDERS_CONV_2D_BUILDER_H_
