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
#ifndef TENSORFLOW_LITE_EXPERIMENTAL_DELEGATES_COREML_BUILDERS_OP_BUILDER_H_
#define TENSORFLOW_LITE_EXPERIMENTAL_DELEGATES_COREML_BUILDERS_OP_BUILDER_H_
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
class MHTracer_DTPStensorflowPSlitePSdelegatesPScoremlPSbuildersPSop_builderDTh {
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
   MHTracer_DTPStensorflowPSlitePSdelegatesPScoremlPSbuildersPSop_builderDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePSdelegatesPScoremlPSbuildersPSop_builderDTh() {
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


#include <string>

#include "mlmodel/format/Model.pb.h"
#include "mlmodel/format/NeuralNetwork.pb.h"
#include "tensorflow/lite/c/common.h"

namespace tflite {
namespace delegates {
namespace coreml {
class OpBuilder;

// A class represents an ID in the coreML graph.
// A node is represented by a pair (node_id, and output_index)
// API is experimental and subject to change.
class TensorID {
 public:
  TensorID() {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPScoremlPSbuildersPSop_builderDTh mht_0(mht_0_v, 203, "", "./tensorflow/lite/delegates/coreml/builders/op_builder.h", "TensorID");
}
  TensorID(int node, int output_id) : node_(node), output_id_(output_id) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPScoremlPSbuildersPSop_builderDTh mht_1(mht_1_v, 207, "", "./tensorflow/lite/delegates/coreml/builders/op_builder.h", "TensorID");
}

  std::string ToString() const;

  int NodeID() const;

  int OutputID() const;

 private:
  int node_ = -1;
  int output_id_ = -1;
};

// Builder for the whole graph.
// All op builders should be added using AddBuilder
// and then BuildModel should be called to return the CoreML generated.
//
// API is experimental and subject to change.
class GraphBuilder {
 public:
  explicit GraphBuilder(int coreml_version) : coreml_version_(coreml_version) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPScoremlPSbuildersPSop_builderDTh mht_2(mht_2_v, 230, "", "./tensorflow/lite/delegates/coreml/builders/op_builder.h", "GraphBuilder");
}

  // Returns pointer to the created builder. Ownership still belongs
  // to the GraphBuilder.
  OpBuilder* AddBuilder(int builtin_code, const TfLiteNode* node);

  // Returns pointer to the created builder with op builder function provided.
  OpBuilder* AddBuilder(const std::function<OpBuilder*(GraphBuilder*)>& builder,
                        const TfLiteNode* node);

  // Builds Model instance and returns it.
  CoreML::Specification::Model* BuildModel();

  // Returns string representing tensor 'tensor_id' in coreML.
  // tensor_id should have been added before calling this method.
  std::string GetTensorName(int tensor_id);

  // Returns Core ML Tensor ID for TFL 'tensor_id'.
  // tensor_id should have been added before calling this method.
  const TensorID GetTensorID(int tensor_id);

  void AddTensorWithID(int tf_tensor_id, const TensorID& tensor_id);

  // Return true if this tensor was added before to the graph.
  bool HasTensor(int tflite_tensor_index);
  // Return if this tensor is used in the graph (not as data).
  // This information is used to mark constant tensors that are used as input.
  bool IsTensorUsed(int tflite_tensor_index);

  const int coreml_version_;

 private:
  std::vector<std::unique_ptr<OpBuilder>> builders_;
  // Index in the vector is the tflite_tensor_index, the value
  // is the ID in the coreml graph.
  std::vector<TensorID> tensors_;
  std::vector<bool> used_tensor_;
};

// Interface for all op layers
// API is experimental and subject to change.
class OpBuilder {
 public:
  explicit OpBuilder(GraphBuilder* graph_builder)
      : graph_builder_(graph_builder) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPScoremlPSbuildersPSop_builderDTh mht_3(mht_3_v, 277, "", "./tensorflow/lite/delegates/coreml/builders/op_builder.h", "OpBuilder");
}
  virtual ~OpBuilder() {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPScoremlPSbuildersPSop_builderDTh mht_4(mht_4_v, 281, "", "./tensorflow/lite/delegates/coreml/builders/op_builder.h", "~OpBuilder");
}

  // Returns the Layer this builder responsible for.
  // Ownership is transferred to caller.
  virtual CoreML::Specification::NeuralNetworkLayer* Build();

  // Associates TfLite input tensors to Core ML layer's inputs and properties.
  // Verification for input constraints should happen here.
  virtual TfLiteStatus RegisterInputs(const TfLiteIntArray* inputs,
                                      TfLiteContext* context) = 0;

  // Associates TFLite output tensor with the node's output. If the OpBuilder
  // has subgraphs, The final output of that subgraph should be associated with
  // the output tensor.
  virtual TfLiteStatus RegisterOutputs(const TfLiteIntArray* outputs,
                                       TfLiteContext* context) = 0;

  // Adds additional required OpBuilders, and populate builder_output_ with
  // Actual output that corresponds to output tensor of TFL Node.
  // Clients need to override this in cases where the nodes can be used for
  // composing other ops. For example, Relu6 in TfLite can be converted to
  // Relu -> Threshold -> Neg.
  // TODO(b/147211734): have this called automatically when necessary.
  virtual TfLiteStatus PopulateSubgraph(TfLiteContext* context);

  virtual const std::string& DebugName() = 0;

  void SetBuiltinData(void* builtin_data);

  void SetNodeID(int id);

  void SetTfLiteNode(const TfLiteNode* node);

  int GetID() const;

  // Adds input with tensor name.
  void AddInput(const std::string& input_name);

  // Adds input with CoreML tensor ID.
  void AddInput(const TensorID& input_id);

  // Adds input with TF Lite tensor ID.
  // TODO(taeheej): cleanup AddInput use cases and used tensor tracking.
  void AddInput(int tf_input_id);

  // Simply adds new output to the underlying layer.
  TensorID AddOutput();

  // Should set builder_output_ (if unset) and return it as the output of
  // this node. To be used by clients that needs the output of the node.
  virtual TensorID GetOutput(TfLiteContext* context);

 protected:
  // Sets layer's name.
  void SetDebugName(const char* layer_name, int id);

  GraphBuilder* graph_builder_ = nullptr;
  // Data needed by this node.
  void* builtin_data_ = nullptr;
  int node_id_ = -1;
  int num_outputs_ = 0;
  const TfLiteNode* tflite_node_ = nullptr;
  TensorID builder_output_;
  std::string debug_name_;
  std::unique_ptr<CoreML::Specification::NeuralNetworkLayer> layer_;
};

}  // namespace coreml
}  // namespace delegates
}  // namespace tflite

#endif  // TENSORFLOW_LITE_EXPERIMENTAL_DELEGATES_COREML_BUILDERS_OP_BUILDER_H_
