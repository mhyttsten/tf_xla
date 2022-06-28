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

#ifndef TENSORFLOW_LITE_DELEGATES_GPU_COMMON_MODEL_H_
#define TENSORFLOW_LITE_DELEGATES_GPU_COMMON_MODEL_H_
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
class MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPSmodelDTh {
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
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPSmodelDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPSmodelDTh() {
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


#include <algorithm>
#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "absl/memory/memory.h"
#include "absl/types/any.h"
#include "absl/types/optional.h"
#include "tensorflow/lite/delegates/gpu/common/shape.h"
#include "tensorflow/lite/delegates/gpu/common/status.h"
#include "tensorflow/lite/delegates/gpu/common/tensor.h"

namespace tflite {
namespace gpu {

// There is yet another representation of CNN graph. The primary purpose of this
// representation is to simplify graph manipulation.

using ValueId = uint32_t;

using NodeId = uint32_t;

// Used to emulate quantized behavior.
struct QuantizationParams {
  float min = 0;
  float max = 0;
  float scale = 0;
};

// Connects tensor's producer and operation that depends on this tensor.
struct Value {
  const ValueId id;
  TensorRef<BHWC> tensor;
  absl::optional<QuantizationParams> quant_params;
};

struct Operation {
  std::string type;
  absl::any attributes;
};

struct Node {
  const NodeId id;
  Operation operation;
};

// A DAG that consists of nodes and values. Each value may have a single
// producer node and multiple consumer nodes. Therefore, each node may have
// multiple input and output values.
//
// Value that does not have a producer is a graph's input. Value that does not
// have a consumer is a graph's output.
//
// It keeps values and nodes referenced by their index in a vector. Therefore,
// nodes and values are never deleted, but rather erased, where corresponding
// index remains.
//
// It is possible to re-use removed indices, but it is not implemented yet.
class GraphFloat32 {
 public:
  // @return a collection of nodes in this graph.
  std::vector<Node*> nodes() const;

  // @return a collection of values in this graph.
  std::vector<Value*> values() const;

  // @return graph inputs, that are values without producers.
  std::vector<Value*> inputs() const;

  // @return graph outputs, that are values without consumers.
  std::vector<Value*> outputs() const;

  // @return values updated in place with a previously defined tensor reference.
  std::vector<Value*> variable_inputs() const;

  // @return inputs into the given node. Returns empty vector for deleted node.
  std::vector<Value*> FindInputs(NodeId id) const;

  // @return outputs from the given node. Returns empty vector for deleted node.
  std::vector<Value*> FindOutputs(NodeId id) const;

  bool IsGraphInput(ValueId id) const;

  bool IsGraphOutput(ValueId id) const;

  // @return producer of the given value. Returns nullptr for deleted value.
  Node* FindProducer(ValueId id) const;

  // @return consumers of the given value. Returns empty vector for deleted
  // value.
  std::vector<Node*> FindConsumers(ValueId id) const;

  // @return a node or nullptr if node with the given id is not present.
  Node* GetNode(NodeId id) const;

  // @return a value or nullptr if value with the given id is not present.
  Value* GetValue(ValueId id) const;

  //////////////////////////////////////////////////////////////////////////////
  // Graph manipulation functions are below
  //////////////////////////////////////////////////////////////////////////////

  // @return new node created in this graph
  // NOTE: nodes should be created in the topological order, e.g. node A that
  // depends on a value from node B should be created after node B.
  Node* NewNode();

  // Insert Node after another in the execution plan.
  absl::Status InsertNodeAfter(NodeId id, Node** new_node);

  // @return new value created in this graph
  Value* NewValue();

  // Sets a producer for the given value. There could be a single producer
  // for a value. If a value had another producer, it will reassign producer
  // appropriately. If a value didn't have a producer, it will be removed
  // from a graph's input.
  absl::Status SetProducer(NodeId producer, ValueId value);

  // Removes a producer for the given value. Value becomes producer-less and
  // therefore becomes graph's input.
  absl::Status RemoveProducer(ValueId value);

  // Sets a consumer for the given value. There could be multiple consumers
  // for a value.
  absl::Status AddConsumer(NodeId consumer, ValueId value);

  // Replace input value for given node.
  absl::Status ReplaceInput(NodeId node, ValueId old_value, ValueId new_value);

  // Removes a consumer for the given value. If value does not have any
  // consumers it becomes graph's output.
  absl::Status RemoveConsumer(NodeId consumer, ValueId value);

  // Removes node from this graph. For all input values this node will be
  // removed from consumers and for all output values a producer will be
  // removed.
  absl::Status DeleteNode(NodeId id);

  // Removes value from this graph. It will be removed from inputs for all
  // dependent nodes. A node that was a producer of this value will loose its
  // output.
  absl::Status DeleteValue(ValueId id);

  absl::Status MakeExactCopy(GraphFloat32* model) const;

 private:
  struct NodeDef {
    std::vector<Value*> inputs;
    std::vector<Value*> outputs;
    std::unique_ptr<Node> node;
  };

  struct ValueDef {
    Node* producer = nullptr;
    std::vector<Node*> consumers;
    std::unique_ptr<Value> value;
  };

  bool IsInput(NodeId node, ValueId value);

  template <typename T>
  static void Erase(std::vector<T>* values, T value) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPSmodelDTh mht_0(mht_0_v, 352, "", "./tensorflow/lite/delegates/gpu/common/model.h", "Erase");

    values->erase(std::find(values->begin(), values->end(), value));
  }

  // @return non-nullptr NodeDef that has valid Node or an error
  absl::Status LookupNode(NodeId id, NodeDef** node_def);

  // @return non-nullptr ValueDef that has valid Value or an error
  absl::Status LookupValue(ValueId id, ValueDef** value_def);

  template <typename Pred>
  std::vector<Value*> FilterValues(const Pred& predicate) const {
    std::vector<Value*> values;
    values.reserve(values_.size());
    for (auto& v : values_) {
      if (v.value != nullptr && predicate(v)) {
        values.push_back(v.value.get());
      }
    }
    return values;
  }

  template <typename Pred>
  std::vector<Node*> FilterNodes(const Pred& predicate) const {
    std::vector<Node*> nodes;
    nodes.reserve(nodes_.size());
    for (const auto id : execution_plan_) {
      auto& n = nodes_.at(id);
      if (n.node != nullptr && predicate(n)) {
        nodes.push_back(n.node.get());
      }
    }
    return nodes;
  }

  // There are two approaches possible: wrap entire NodeDef and ValueDef into
  // unique_ptr and store it in values_ and nodes_ or store it by value.
  // We store it by value here to make introspection calls cheaper.
  std::vector<ValueDef> values_;

  std::map<NodeId, NodeDef> nodes_;
  // Node Ids in order of execution.
  std::vector<NodeId> execution_plan_;
};

// Removes to_remove node that precedes to_keep node only if to_remove has
// outputs that are consumed only by to_keep. In such case to_keep inherits all
// to_remove inputs.
absl::Status RemovePrecedingNode(GraphFloat32* graph, const Node* to_remove,
                                 const Node* to_keep);

// Removes to_remove node that follows to_keep node only if to_remove has inputs
// that are produced by to_keep. to_keep inherits all to_remove inputs.
absl::Status RemoveFollowingNode(GraphFloat32* graph, const Node* to_remove,
                                 const Node* to_keep);

// Removes simple_node and its output value from the graph. Node is considered
// simple if it has only one input and one output value. Input value is kept.
absl::Status RemoveSimpleNodeKeepInput(GraphFloat32* graph,
                                       const Node* simple_node);

// Removes simple_node and its input value from the graph. Node is considered
// simple if it has only one input and one output value. Output value is kept.
// simple_node should be an exclusive consumer of its input value.
absl::Status RemoveSimpleNodeKeepOutput(GraphFloat32* graph,
                                        const Node* simple_node);

absl::Status AddOutput(GraphFloat32* graph, const Node* from_node,
                       Value** output);

// Makes a direct connection between from_node and to_node. All input parameters
// except output are expected to be initialized before passing to the function.
// If from_node already has an output value, which is not yet consumed by
// to_node, it may be passed as output parameter.
absl::Status ConnectTwoNodes(GraphFloat32* graph, const Node* from_node,
                             const Node* to_node, Value** output);

// @return OkStatus if all tensors have the same batch value, otherwise an
// invalid argument error is returned.
absl::Status CheckBatchSizeForAllValues(const GraphFloat32& model);

}  // namespace gpu
}  // namespace tflite

#endif  // TENSORFLOW_LITE_DELEGATES_GPU_COMMON_MODEL_H_
