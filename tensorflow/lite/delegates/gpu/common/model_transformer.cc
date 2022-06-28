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
class MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPSmodel_transformerDTcc {
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
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPSmodel_transformerDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPSmodel_transformerDTcc() {
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

#include "tensorflow/lite/delegates/gpu/common/model_transformer.h"

#include <deque>
#include <string>
#include <vector>

#include "absl/container/flat_hash_set.h"
#include "absl/strings/str_join.h"
#include "tensorflow/lite/delegates/gpu/common/model.h"

namespace tflite {
namespace gpu {

bool ModelTransformer::Apply(const std::string& name,
                             SequenceTransformation* transformation) {
   std::vector<std::string> mht_0_v;
   mht_0_v.push_back("name: \"" + name + "\"");
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPSmodel_transformerDTcc mht_0(mht_0_v, 200, "", "./tensorflow/lite/delegates/gpu/common/model_transformer.cc", "ModelTransformer::Apply");

  // Seed transformations with starting node. Each node may start a chain of
  // transformations.
  for (auto input : graph_->inputs()) {
    for (auto node : graph_->FindConsumers(input->id)) {
      AddNodeToProcess(node);
    }
  }
  while (!to_process_.empty()) {
    auto node = graph_->GetNode(to_process_.front());
    if (node) {
      if (!ApplyStartingWithNode(name, transformation, node)) {
        return false;
      }
    }
    to_process_.pop_front();
  }
  processed_.clear();
  return true;
}

bool ModelTransformer::Apply(const std::string& name,
                             NodeTransformation* transformation) {
   std::vector<std::string> mht_1_v;
   mht_1_v.push_back("name: \"" + name + "\"");
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPSmodel_transformerDTcc mht_1(mht_1_v, 226, "", "./tensorflow/lite/delegates/gpu/common/model_transformer.cc", "ModelTransformer::Apply");

  // Apply a transformation only to nodes that are present in the graph before
  // transformation.
  std::vector<NodeId> nodes;
  for (auto node : graph_->nodes()) {
    nodes.push_back(node->id);
  }
  for (auto node_id : nodes) {
    auto node = graph_->GetNode(node_id);
    if (!node) {
      continue;
    }
    auto result = transformation->ApplyToNode(node, graph_);
    last_transformation_message_ = result.message;
    if (result.status == TransformStatus::INVALID) {
      return false;
    }
  }
  return true;
}

const std::string& ModelTransformer::last_transformation_message() const {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPSmodel_transformerDTcc mht_2(mht_2_v, 250, "", "./tensorflow/lite/delegates/gpu/common/model_transformer.cc", "ModelTransformer::last_transformation_message");

  return last_transformation_message_;
}

bool ModelTransformer::ApplyStartingWithNode(
    const std::string& name, SequenceTransformation* transformation,
    Node* begin) {
   std::vector<std::string> mht_3_v;
   mht_3_v.push_back("name: \"" + name + "\"");
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPSmodel_transformerDTcc mht_3(mht_3_v, 260, "", "./tensorflow/lite/delegates/gpu/common/model_transformer.cc", "ModelTransformer::ApplyStartingWithNode");

  int expected_sequence_length = transformation->ExpectedSequenceLength();

  std::deque<NodeId> sequence;
  std::vector<Node*> nodes;
  nodes.reserve(transformation->ExpectedSequenceLength());
  sequence.push_back(begin->id);

  // Go over nodes with sequence sliding window of size
  // expected_sequence_length until a node with multiple dependents is found.
  while (true) {
    // Apply transformation if possible.
    if (sequence.size() == expected_sequence_length) {
      nodes.clear();
      for (NodeId id : sequence) {
        // Nodes present in sequence should be present in a graph. If they are
        // not, then this transformation changes a graph but didn't say it.
        Node* node = graph_->GetNode(id);
        if (node == nullptr) {
          return false;
        }
        nodes.push_back(node);
      }

      NodeId first_in_sequence = sequence.front();
      auto preceding_node =
          graph_->FindProducer(graph_->FindInputs(first_in_sequence)[0]->id);
      auto result = transformation->ApplyToNodesSequence(nodes, graph_);
      last_transformation_message_ = result.message;
      if (result.status == TransformStatus::INVALID) {
        // graph is broken now.
        return false;
      }
      if (result.status == TransformStatus::APPLIED) {
        // Also remove first node of a sequence from a set of processed node.
        // Out of all nodes in a sequence only first one may have been added
        // to "processed" set because other nodes do not have more than one
        // dependent. However, if a sequence is changed, then processing needs
        // to be restarted again.
        processed_.erase(first_in_sequence);
        // Transformation was successful. Restart sequence from the node that
        // precedes current sequence.
        if (preceding_node) {
          processed_.erase(preceding_node->id);
          AddNodeToProcess(preceding_node);
        } else {
          // This is the first node in the graph. Re-seed transformation.
          for (auto input : graph_->inputs()) {
            for (auto node : graph_->FindConsumers(input->id)) {
              AddNodeToProcess(node);
            }
          }
        }
        return true;
      }
    }

    // Try to extend current sequence.
    Node* next_node_in_sequence = nullptr;
    bool has_multiple_children = false;

    // Check that all outputs from last node are consumed by a single node.
    for (auto output_value : graph_->FindOutputs(sequence.back())) {
      for (auto dependent : graph_->FindConsumers(output_value->id)) {
        if (has_multiple_children) {
          AddNodeToProcess(dependent);
        } else if (next_node_in_sequence == nullptr) {
          next_node_in_sequence = dependent;
        } else if (next_node_in_sequence != dependent) {
          // There are more than two nodes depend on the output from end node,
          // therefore here a sequence stops and new will start. Push all such
          // nodes.
          has_multiple_children = true;
          AddNodeToProcess(dependent);
          AddNodeToProcess(next_node_in_sequence);
        }
      }
    }

    // Now check that next node has inputs only produced by the last node.
    if (!has_multiple_children && next_node_in_sequence) {
      for (auto input : graph_->FindInputs(next_node_in_sequence->id)) {
        auto producer = graph_->FindProducer(input->id);
        if (producer == nullptr || producer->id != sequence.back()) {
          has_multiple_children = true;
          AddNodeToProcess(next_node_in_sequence);
          break;
        }
      }
    }

    if (has_multiple_children || next_node_in_sequence == nullptr) {
      // reached end of this transformation sequence.
      return true;
    }

    sequence.push_back(next_node_in_sequence->id);
    // Decrease sequence until it matches expected length.
    if (sequence.size() > expected_sequence_length) {
      sequence.pop_front();
    }
  }
  return true;
}

}  // namespace gpu
}  // namespace tflite
