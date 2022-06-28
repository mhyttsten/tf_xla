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

#ifndef TENSORFLOW_LITE_DELEGATES_GPU_COMMON_MODEL_TRANSFORMER_H_
#define TENSORFLOW_LITE_DELEGATES_GPU_COMMON_MODEL_TRANSFORMER_H_
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
class MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPSmodel_transformerDTh {
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
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPSmodel_transformerDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPSmodel_transformerDTh() {
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


#include <deque>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_set.h"
#include "tensorflow/lite/delegates/gpu/common/model.h"

namespace tflite {
namespace gpu {

struct TransformationContext {
  GraphFloat32* graph;
};

enum class TransformStatus {
  // Transformation was not applied due to trivial conditions mismatch.
  //
  // This is different from DECLINED code below that provides in-depth
  // explanation why a transformation that could have been applied but was not
  // due to some issues.
  SKIPPED,

  // Transformation was declined, therefore, a model was not modified.
  DECLINED,

  // Transformation was applied successfully
  APPLIED,

  // Transformation may partially be applied, but left a model in an invalid
  // state. This error should be considered unrecoverable.
  INVALID,
};

struct TransformResult {
  TransformStatus status;
  std::string message;
  bool operator==(const TransformResult& result) const {
    return this->status == result.status && this->message == result.message;
  }
};

// Class responsible for applying a transformation to a single node.
class NodeTransformation {
 public:
  virtual ~NodeTransformation() = default;

  virtual TransformResult ApplyToNode(Node* node, GraphFloat32* graph) = 0;
};

// Class responsible for applying a transformation to a sequence of nodes.
// Nodes are guaranteed to depend on each other without extra dependents being
// spilled.
class SequenceTransformation {
 public:
  virtual ~SequenceTransformation() = default;

  // @return number of nodes in a sequence to apply this transformation.
  virtual int ExpectedSequenceLength() const = 0;

  // Applies transformations to a sequence of nodes. Transformation
  // implementation is free manipulate with sequence nodes including adding
  // and/or deleting nodes. if there were updates to nodes in the end and/or
  // beginning of the sequence, then referential consistency should be
  // maintained by updating relevant references in nodes that precede this
  // sequence or depend on a last node of the sequence.
  virtual TransformResult ApplyToNodesSequence(
      const std::vector<Node*>& sequence, GraphFloat32* graph) = 0;
};

// Performs model transformations.
class ModelTransformer {
 public:
  explicit ModelTransformer(GraphFloat32* graph) : graph_(graph) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPSmodel_transformerDTh mht_0(mht_0_v, 261, "", "./tensorflow/lite/delegates/gpu/common/model_transformer.h", "ModelTransformer");
}

  // @return false if a graph is in the broken states can not be used any more
  bool Apply(const std::string& name, SequenceTransformation* transformation);

  // @return false if a graph is in the broken states can not be used any more
  bool Apply(const std::string& name, NodeTransformation* transformation);

  // @return last recorded error for graph transformations.
  const std::string& last_transformation_message() const;

 private:
  bool ApplyStartingWithNode(const std::string& name,
                             SequenceTransformation* transformation,
                             Node* begin);

  void AddNodeToProcess(Node* node) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPSmodel_transformerDTh mht_1(mht_1_v, 280, "", "./tensorflow/lite/delegates/gpu/common/model_transformer.h", "AddNodeToProcess");

    if (node && processed_.insert(node->id).second) {
      to_process_.push_back(node->id);
    }
  }

  GraphFloat32* graph_;

  // TODO(b/163423950): Clean up messaging mechanism.
  std::string last_transformation_message_;
  std::deque<NodeId> to_process_;
  absl::flat_hash_set<NodeId> processed_;
};

}  // namespace gpu
}  // namespace tflite

#endif  // TENSORFLOW_LITE_DELEGATES_GPU_COMMON_MODEL_TRANSFORMER_H_
