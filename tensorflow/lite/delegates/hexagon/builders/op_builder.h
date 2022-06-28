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
#ifndef TENSORFLOW_LITE_DELEGATES_HEXAGON_BUILDERS_OP_BUILDER_H_
#define TENSORFLOW_LITE_DELEGATES_HEXAGON_BUILDERS_OP_BUILDER_H_
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
class MHTracer_DTPStensorflowPSlitePSdelegatesPShexagonPSbuildersPSop_builderDTh {
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
   MHTracer_DTPStensorflowPSlitePSdelegatesPShexagonPSbuildersPSop_builderDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePSdelegatesPShexagonPSbuildersPSop_builderDTh() {
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


#include <limits>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "hexagon/hexagon_nn_ops.h"
#include "tensorflow/lite/builtin_ops.h"
#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/delegates/hexagon/hexagon_implementation.h"
#include "tensorflow/lite/delegates/hexagon/hexagon_nn/hexagon_nn.h"

namespace tflite {
namespace delegates {
namespace hexagon {

// Wrapper that holds all data representing a single node in the Hexagon graph.
struct OpNode {
  std::vector<hexagon_nn_input> inputs;
  std::vector<hexagon_nn_output> outputs;
  // Value from the Enum of Ops in hexagon_nn_ops
  int op_type;
  hexagon_nn_padding_type padding_type = NN_PAD_NA;
  // Id of node in the Hexagon graph.
  int node_id = -1;
  // Index/ID of node in the tflite graph.
  // This ID can be duplicate if one TFLite node creates multiple Hexagon op
  // nodes.
  int tflite_node_index = -1;
};

class GraphBuilder;

// Represents a single Op in the TFLite graph.
// For each op in TFLite there should be an OpBuidler, this builder is
// responsible for constructing equivalent node(s) in the hexagon graph. A
// single builder can create one or more ops in the hexagon graph. When adding
// new op* users should inherit from this class and implement
// - PopulateSubgraph: which given inputs/outputs should construct the
// equivalent hexagon nodes.
// - RegisterOutputs: Which should have logic that maps final outputs from a
// given node to the equivalent in Hexagon graph.
class OpBuilder {
 public:
  // Const representing the shape of a scalar value.
  static constexpr int kScalarShape[] = {1, 1, 1, 1};

  OpBuilder(GraphBuilder* graph_builder, int hexagon_op_type)
      : graph_builder_(graph_builder) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPShexagonPSbuildersPSop_builderDTh mht_0(mht_0_v, 237, "", "./tensorflow/lite/delegates/hexagon/builders/op_builder.h", "OpBuilder");

    op_node_.op_type = hexagon_op_type;
  }
  // A tensor is identified in the graph using a pair of IDs
  // (Node ID, output Tensor ID)
  // Node producing this tensor, and the index of the tensor in this
  // node output list.
  using TensorID = std::pair<int, int>;

  virtual ~OpBuilder() {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPShexagonPSbuildersPSop_builderDTh mht_1(mht_1_v, 249, "", "./tensorflow/lite/delegates/hexagon/builders/op_builder.h", "~OpBuilder");
}

  // Sets the op type in the hexagon graph.
  void SetOpType(int op_type) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPShexagonPSbuildersPSop_builderDTh mht_2(mht_2_v, 255, "", "./tensorflow/lite/delegates/hexagon/builders/op_builder.h", "SetOpType");
 op_node_.op_type = op_type; }

  // Sets the node id in the hexagon graph.
  void SetNodeId(int node_id) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPShexagonPSbuildersPSop_builderDTh mht_3(mht_3_v, 261, "", "./tensorflow/lite/delegates/hexagon/builders/op_builder.h", "SetNodeId");
 op_node_.node_id = node_id; }

  // Sets the TfLite node index in the TfLite graph.
  void SetTFLiteNodeId(int node_index) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPShexagonPSbuildersPSop_builderDTh mht_4(mht_4_v, 267, "", "./tensorflow/lite/delegates/hexagon/builders/op_builder.h", "SetTFLiteNodeId");

    op_node_.tflite_node_index = node_index;
  }

  // Marks this node as Const node.
  void SetConstNode() {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPShexagonPSbuildersPSop_builderDTh mht_5(mht_5_v, 275, "", "./tensorflow/lite/delegates/hexagon/builders/op_builder.h", "SetConstNode");
 op_node_.op_type = OP_Const; }

  // Sets the padding type of the current node.
  void SetPaddingType(hexagon_nn_padding_type padding_type) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPShexagonPSbuildersPSop_builderDTh mht_6(mht_6_v, 281, "", "./tensorflow/lite/delegates/hexagon/builders/op_builder.h", "SetPaddingType");

    op_node_.padding_type = padding_type;
  }

  // Sets the builtin_data of TFLite node that this Builder is responsible for.
  void SetBuiltinData(void* builtin_data) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPShexagonPSbuildersPSop_builderDTh mht_7(mht_7_v, 289, "", "./tensorflow/lite/delegates/hexagon/builders/op_builder.h", "SetBuiltinData");
 builtin_data_ = builtin_data; }

  // Returns true if the current op is a const Op.
  bool IsConstNode() const {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPShexagonPSbuildersPSop_builderDTh mht_8(mht_8_v, 295, "", "./tensorflow/lite/delegates/hexagon/builders/op_builder.h", "IsConstNode");
 return op_node_.op_type == OP_Const; }

  // Subclasses should override it and have logic which handles initializing
  // hexagon node(s) for the current op, given 'inputs' 'outputs'
  virtual TfLiteStatus PopulateSubGraph(const TfLiteIntArray* inputs,
                                        const TfLiteIntArray* outputs,
                                        TfLiteContext* context) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPShexagonPSbuildersPSop_builderDTh mht_9(mht_9_v, 304, "", "./tensorflow/lite/delegates/hexagon/builders/op_builder.h", "PopulateSubGraph");

    return kTfLiteOk;
  }

  // Subclasses should override it and register the final output(s) from the
  // node to the equivalent in hexagon graph.
  virtual TfLiteStatus RegisterOutputs(const TfLiteIntArray* outputs,
                                       TfLiteContext* context) {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPShexagonPSbuildersPSop_builderDTh mht_10(mht_10_v, 314, "", "./tensorflow/lite/delegates/hexagon/builders/op_builder.h", "RegisterOutputs");

    return kTfLiteOk;
  }

  // Constructs OpNode which represents a node in the Hexagon graph.
  const OpNode* Build();

  // Returns the Node index in TFLite graph.
  int GetTFLiteNodeID() const {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPShexagonPSbuildersPSop_builderDTh mht_11(mht_11_v, 325, "", "./tensorflow/lite/delegates/hexagon/builders/op_builder.h", "GetTFLiteNodeID");
 return op_node_.tflite_node_index; }

  // Returns the Op type of the current Op (in Hexagon graph)
  int GetOpType() const {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPShexagonPSbuildersPSop_builderDTh mht_12(mht_12_v, 331, "", "./tensorflow/lite/delegates/hexagon/builders/op_builder.h", "GetOpType");
 return op_node_.op_type; }

  // Returns the node id in the hexagon graph.
  int GetID() const {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPShexagonPSbuildersPSop_builderDTh mht_13(mht_13_v, 337, "", "./tensorflow/lite/delegates/hexagon/builders/op_builder.h", "GetID");
 return op_node_.node_id; }

  // Adds tensor identified by 'tensor_id' as input to the current Op.
  void AddInput(const TensorID& tensor_id) {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPShexagonPSbuildersPSop_builderDTh mht_14(mht_14_v, 343, "", "./tensorflow/lite/delegates/hexagon/builders/op_builder.h", "AddInput");
 input_ids_.push_back(tensor_id); }

  // Adds Output to the current node, the output has shape defined in 'dims'.
  // The size of each element is defined using 'element_size'.
  // Returns the TensorID identifying this output in the graph.
  TensorID AddOutput(const TfLiteIntArray* dims, int element_size);

  // Adds Output to the current node, each element in the output has
  // size 'elementsize' and rank 'rank' and for each dimension in the output
  // the maximum size is max_sizes[i].
  // Returns the TensorID identifying this output in the graph.
  TensorID AddOutput(int elementsize, int rank,
                     const std::vector<int>& max_sizes);

  // Same as above but accepts pointer instead of std::vector.
  TensorID AddOutput(int elementsize, int rank, const int* max_sizes_vect);

  // Sets the node that corresponds to this builder in TFLite graph.
  void SetTfLiteNode(const TfLiteNode* node) {
   std::vector<std::string> mht_15_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPShexagonPSbuildersPSop_builderDTh mht_15(mht_15_v, 364, "", "./tensorflow/lite/delegates/hexagon/builders/op_builder.h", "SetTfLiteNode");
 tflite_node_ = node; }

  // Static
  // Computes the min/max values of 'tensor' and sets the values in
  // the out params 'min' and 'max'.
  // Returns kTfLiteOk on success.
  static TfLiteStatus ComputeMinAndMaxQuantValues(const TfLiteTensor& tensor,
                                                  float* min, float* max) {
   std::vector<std::string> mht_16_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPShexagonPSbuildersPSop_builderDTh mht_16(mht_16_v, 374, "", "./tensorflow/lite/delegates/hexagon/builders/op_builder.h", "ComputeMinAndMaxQuantValues");

    if (tensor.type == kTfLiteUInt8) {
      return ComputeMinAndMaxQuantValues(tensor, min, max,
                                         std::numeric_limits<uint8_t>::min(),
                                         std::numeric_limits<uint8_t>::max());
    } else if (tensor.type == kTfLiteInt8) {
      return ComputeMinAndMaxQuantValues(tensor, min, max,
                                         std::numeric_limits<int8_t>::min(),
                                         std::numeric_limits<int8_t>::max());
    } else if (tensor.type == kTfLiteInt32) {
      return ComputeMinAndMaxQuantValues(tensor, min, max,
                                         std::numeric_limits<int>::min(),
                                         std::numeric_limits<int>::max());
    }
    return kTfLiteError;
  }

 protected:
  // Helper method to fetch dimensions.
  // TODO(karimnosseir): Move to a shared place.
  void GetDims(int* batch_size, int* height_size, int* width_size,
               int* depth_size, const TfLiteIntArray* dims) {
   std::vector<std::string> mht_17_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPShexagonPSbuildersPSop_builderDTh mht_17(mht_17_v, 398, "", "./tensorflow/lite/delegates/hexagon/builders/op_builder.h", "GetDims");

    int* dim[] = {batch_size, height_size, width_size, depth_size};
    for (int i = 0; i < 4; ++i) *(dim[i]) = 1;
    for (int i = 4 - dims->size; i < 4; ++i) {
      *dim[i] = dims->data[i - (4 - dims->size)];
    }
  }

  // Computes the min and max for 'tensor' and adds them as input
  // to the node.
  TfLiteStatus ComputeAndAddMinAndMax(TfLiteContext* context,
                                      const TfLiteTensor& tensor);

  // Computes the float min and max for 'tensor', given 'min_value' and
  // 'max_value' data range. The float min and max will be set in 'min' and
  // 'max' params
  template <typename T>
  static TfLiteStatus ComputeMinAndMaxQuantValues(const TfLiteTensor& tensor,
                                                  float* min, float* max,
                                                  T min_value, T max_value) {
   std::vector<std::string> mht_18_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPShexagonPSbuildersPSop_builderDTh mht_18(mht_18_v, 420, "", "./tensorflow/lite/delegates/hexagon/builders/op_builder.h", "ComputeMinAndMaxQuantValues");

    *min = 0;
    *max = 0;
    const TfLiteQuantization& quant = tensor.quantization;
    if (quant.type != TfLiteQuantizationType::kTfLiteAffineQuantization) {
      printf("Tensor not quantized: %s\n", tensor.name);
      return kTfLiteError;
    }
    const TfLiteAffineQuantization* params =
        static_cast<const TfLiteAffineQuantization*>(quant.params);
    float scale = params->scale->data[0];
    float zero_point = static_cast<float>(params->zero_point->data[0]);
    *min = scale * (static_cast<float>(min_value) - zero_point);
    *max = scale * (static_cast<float>(max_value) - zero_point);

    return kTfLiteOk;
  }

  OpNode op_node_;
  // inputs to the current op. Each pair identifies a single output from
  // another node (node_id, output_id).
  std::vector<TensorID> input_ids_;
  // Pointer to the graph builder.
  GraphBuilder* graph_builder_ = nullptr;
  // Data needed by this node.
  void* builtin_data_ = nullptr;
  // TODO(karimnosseir): Currently we only use it for getting output
  // size. Can we avoid passing it ?
  const TfLiteNode* tflite_node_ = nullptr;
};

class GraphBuilder {
 public:
  GraphBuilder(const HexagonNN* hexagon_nn, TfLiteContext* context,
               int graph_id)
      : hexagon_nn_(hexagon_nn), context_(context), graph_id_(graph_id) {
   std::vector<std::string> mht_19_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPShexagonPSbuildersPSop_builderDTh mht_19(mht_19_v, 458, "", "./tensorflow/lite/delegates/hexagon/builders/op_builder.h", "GraphBuilder");
}

  // Returns per OP builder. 'op_type' is the TfLite builtinOperator.
  OpBuilder* AddNodeFromTfLiteOp(int op_type, TfLiteNode* node,
                                 int tflite_node_index);

  // Add node to the graph. The caller responsible for setting correct
  // data in the Op.
  // 'tflite_node_index' is the node index in TFLite that creates this op.
  OpBuilder* AddNode(int tflite_node_index = -1);

  // Add const node that provides the data held by 'tensor'.
  // If `int8_to_uint8` is true, then the data will be casted to uint8 from
  // int8.
  OpBuilder* AddConstNodeWithData(int tensor_id, const TfLiteTensor& tensor,
                                  bool int8_to_uint8 = false);

  // Same as above but takes shape of the tensor that will holds the data.
  OpBuilder* AddConstNodeWithData(const int shape[], char* data, int data_size);

  OpBuilder* CreateOpBuilderFromTfLiteOp(int op_type, TfLiteNode* node);

  // Construct Input node with 'input_tensors' as output.
  TfLiteStatus AddInputTensors(const TfLiteIntArray* input_tensors,
                               TfLiteContext* context);

  // Construct Output node with 'output_tensors' as input.
  TfLiteStatus AddOutputTensors(const TfLiteIntArray* output_tensors,
                                TfLiteContext* context);

  // Adds BatchSeqConfig node to the graph. This is configuration
  // for a dynamic batch size for the graph.
  // A graph can have only one node of this type.
  void AddBatchSeqConfig(int max_size_for_batch,
                         TfLiteIntArray* input_batch_dimensions,
                         TfLiteIntArray* output_batch_dimensions);

  // Returns tensor id inside Hexagon graph.
  OpBuilder::TensorID GetHexagonTensorId(int tflite_tensor_index) {
   std::vector<std::string> mht_20_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPShexagonPSbuildersPSop_builderDTh mht_20(mht_20_v, 499, "", "./tensorflow/lite/delegates/hexagon/builders/op_builder.h", "GetHexagonTensorId");

    if (!HasTensor(tflite_tensor_index)) {
      // Return invalid ID.
      return OpBuilder::TensorID(-1, -1);
    }
    return tensors_[tflite_tensor_index];
  }

  // Return true if this tensor was added before to the graph.
  bool HasTensor(int tflite_tensor_index) {
   std::vector<std::string> mht_21_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPShexagonPSbuildersPSop_builderDTh mht_21(mht_21_v, 511, "", "./tensorflow/lite/delegates/hexagon/builders/op_builder.h", "HasTensor");

    if (tensors_.size() <= tflite_tensor_index) {
      return false;
    }
    // the first field is node ID and id = 0 is reserved
    // so anything > 0 is correctly initialized.
    return tensors_[tflite_tensor_index].first != 0;
  }

  void AddDebugNode() {
   std::vector<std::string> mht_22_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPShexagonPSbuildersPSop_builderDTh mht_22(mht_22_v, 523, "", "./tensorflow/lite/delegates/hexagon/builders/op_builder.h", "AddDebugNode");
}

  void Build() {
   std::vector<std::string> mht_23_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPShexagonPSbuildersPSop_builderDTh mht_23(mht_23_v, 528, "", "./tensorflow/lite/delegates/hexagon/builders/op_builder.h", "Build");

    for (int i = 0; i < builders_.size(); ++i) {
      if (builders_[i]->IsConstNode()) {
        continue;
      }
      const OpNode* op_node = builders_[i]->Build();
      int error = hexagon_nn_->hexagon_nn_append_node(
          graph_id_, op_node->node_id, op_node->op_type, op_node->padding_type,
          op_node->inputs.data(), op_node->inputs.size(),
          op_node->outputs.data(), op_node->outputs.size());
      if (error != 0) {
        printf("Error adding node: id:%d, op_type:%d\n", op_node->node_id,
               op_node->op_type);
      }
    }
  }

  void print() {
   std::vector<std::string> mht_24_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPShexagonPSbuildersPSop_builderDTh mht_24(mht_24_v, 548, "", "./tensorflow/lite/delegates/hexagon/builders/op_builder.h", "print");

    printf("------------------------------\n");
    std::vector<unsigned char> buf(10000);
    hexagon_nn_->hexagon_nn_snpprint(graph_id_, buf.data(), buf.size());
    printf("%s", buf.data());
    printf("------------------------------\n");
    fflush(stdout);
  }

  // Add new tensor mapping to the tensor list.
  bool AddTensorWithID(int tflite_tensor_id, int hexagon_node_id,
                       int hexagon_node_output_id, bool overwrite = false) {
   std::vector<std::string> mht_25_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPShexagonPSbuildersPSop_builderDTh mht_25(mht_25_v, 562, "", "./tensorflow/lite/delegates/hexagon/builders/op_builder.h", "AddTensorWithID");

    if (!overwrite && HasTensor(tflite_tensor_id)) {
      TF_LITE_KERNEL_LOG(
          context_,
          "Trying to add duplicate tensor without overwrite, tflite_tensor_id "
          "%d, hexagon_node_id %d, hexagon_node_output_id %d",
          tflite_tensor_id, hexagon_node_id, hexagon_node_output_id);
      return false;
    }
    if (tensors_.size() <= tflite_tensor_id) {
      tensors_.resize(tflite_tensor_id + 1);
    }
    if (hexagon_node_id == -1 || hexagon_node_output_id == -1)
      TF_LITE_KERNEL_LOG(context_,
                         "Trying to add invalid id, tflite_tensor_id "
                         "%d, hexagon_node_id %d, hexagon_node_output_id %d",
                         tflite_tensor_id, hexagon_node_id,
                         hexagon_node_output_id);
    tensors_[tflite_tensor_id] =
        OpBuilder::TensorID(hexagon_node_id, hexagon_node_output_id);
    return true;
  }

  int GetOpTypeId(int node_id) {
   std::vector<std::string> mht_26_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPShexagonPSbuildersPSop_builderDTh mht_26(mht_26_v, 588, "", "./tensorflow/lite/delegates/hexagon/builders/op_builder.h", "GetOpTypeId");

    if (node_id > builders_.size()) {
      return -1;
    }
    return builders_[node_id - 1]->GetOpType();
  }

  int GetTFLiteNodeID(int node_id) const {
   std::vector<std::string> mht_27_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPShexagonPSbuildersPSop_builderDTh mht_27(mht_27_v, 598, "", "./tensorflow/lite/delegates/hexagon/builders/op_builder.h", "GetTFLiteNodeID");

    if (node_id > builders_.size()) {
      return -1;
    }
    return builders_[node_id - 1]->GetTFLiteNodeID();
  }

  // Returns true if the graph supports dynamic batch. False otherwise.
  bool GraphHasDynamicBatch() const {
   std::vector<std::string> mht_28_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPShexagonPSbuildersPSop_builderDTh mht_28(mht_28_v, 609, "", "./tensorflow/lite/delegates/hexagon/builders/op_builder.h", "GraphHasDynamicBatch");
 return max_size_for_batch_ != -1; }

  // Returns the maximum value for batch dimension the graph supports.
  // -1 if the graph doesn't support dynamic batch.
  int GetMaxBatchSize() const {
   std::vector<std::string> mht_29_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPShexagonPSbuildersPSop_builderDTh mht_29(mht_29_v, 616, "", "./tensorflow/lite/delegates/hexagon/builders/op_builder.h", "GetMaxBatchSize");
 return max_size_for_batch_; }

 private:
  // Lookup in cache if data with key 'cache_key' is present.
  // Return OpBuilder* for the data if found, nullptr otherwise.
  OpBuilder* LookupConstData(uint64_t cache_key);

  // Inserts 'value' in cache, with key equals 'cache_key'.
  // If data in cache with same key then it will be overwritten.
  void AddToCache(uint64_t cache_key, OpBuilder* value);

  // Helper method to fetch dimensions.
  // TODO(karimnosseir): Move this method to shared place.
  void GetDims(int* batch_size, int* height_size, int* width_size,
               int* depth_size, const TfLiteIntArray* dims) {
   std::vector<std::string> mht_30_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPShexagonPSbuildersPSop_builderDTh mht_30(mht_30_v, 633, "", "./tensorflow/lite/delegates/hexagon/builders/op_builder.h", "GetDims");

    int* dim[] = {batch_size, height_size, width_size, depth_size};
    for (int i = 0; i < 4; ++i) *(dim[i]) = 1;
    for (int i = 4 - dims->size; i < 4; ++i) {
      *dim[i] = dims->data[i - (4 - dims->size)];
    }
  }

  // Adds a Cast op to convert a tensor from int8 to uint8 (or vice versa).
  // The builder which has the casting operator is filled in 'cast_op_builder'
  // if not nullptr.
  TfLiteStatus AddCastOp(TfLiteContext* context, int op_type, int tensor_id,
                         OpBuilder** cast_op_builder);

  const HexagonNN* hexagon_nn_ = nullptr;
  TfLiteContext* context_ = nullptr;
  int graph_id_ = -1;
  std::vector<std::unique_ptr<OpBuilder>> builders_;
  // Index in the vector is the tflite_tensor_index, the value
  // is the ID in the hexgon graph.
  std::vector<OpBuilder::TensorID> tensors_;

  // If the graph being built supports dynamic batch, this represents
  // the maximum value for batch.
  int max_size_for_batch_ = -1;

  // Cache for const data in the graph.
  // Key is hash of the data, value is pointer to the OpBuilder* for the added
  // data.
  std::map<uint64_t, OpBuilder*> cache_;
};

}  // namespace hexagon
}  // namespace delegates
}  // namespace tflite

#endif  // TENSORFLOW_LITE_DELEGATES_HEXAGON_BUILDERS_OP_BUILDER_H_
