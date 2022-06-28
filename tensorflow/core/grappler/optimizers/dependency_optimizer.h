/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_CORE_GRAPPLER_OPTIMIZERS_DEPENDENCY_OPTIMIZER_H_
#define TENSORFLOW_CORE_GRAPPLER_OPTIMIZERS_DEPENDENCY_OPTIMIZER_H_
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
class MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSdependency_optimizerDTh {
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
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSdependency_optimizerDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSdependency_optimizerDTh() {
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


#include <unordered_set>
#include "tensorflow/core/grappler/optimizers/graph_optimizer.h"
#include "tensorflow/core/grappler/utils.h"
#include "tensorflow/core/protobuf/rewriter_config.pb.h"

namespace tensorflow {
namespace grappler {

// Optimize TF computations by removing control dependencies or re-arranging
// them to shorten the critical path for a model step or enable other
// optimizations, such as removing nodes that are effectively noops.
class DependencyOptimizer : public GraphOptimizer {
 public:
  DependencyOptimizer() {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSdependency_optimizerDTh mht_0(mht_0_v, 201, "", "./tensorflow/core/grappler/optimizers/dependency_optimizer.h", "DependencyOptimizer");
}
  explicit DependencyOptimizer(RewriterConfig::Toggle opt_level) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSdependency_optimizerDTh mht_1(mht_1_v, 205, "", "./tensorflow/core/grappler/optimizers/dependency_optimizer.h", "DependencyOptimizer");
}
  ~DependencyOptimizer() override {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSdependency_optimizerDTh mht_2(mht_2_v, 209, "", "./tensorflow/core/grappler/optimizers/dependency_optimizer.h", "~DependencyOptimizer");
}

  string name() const override {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSdependency_optimizerDTh mht_3(mht_3_v, 214, "", "./tensorflow/core/grappler/optimizers/dependency_optimizer.h", "name");
 return "dependency_optimizer"; };

  bool UsesFunctionLibrary() const override {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSdependency_optimizerDTh mht_4(mht_4_v, 219, "", "./tensorflow/core/grappler/optimizers/dependency_optimizer.h", "UsesFunctionLibrary");
 return false; }

  Status Optimize(Cluster* cluster, const GrapplerItem& item,
                  GraphDef* optimized_graph) override;

 private:
  // Returns true if bypassing node does not increase the number of edges or
  // number of edges crossing a device boundary.
  bool BypassingNodeIsBeneficial(
      const NodeDef& node, const std::vector<NodeDef*>& input_nodes,
      const std::vector<NodeDef*>& output_nodes) const;
  int NumEdgesIfBypassed(const NodeDef& node,
                         const std::vector<NodeDef*>& output_nodes) const;
  // Returns true if node is not an Identity node or if it is an Identity
  // that is safe to remove.
  bool SafeToRemoveIdentity(const NodeDef& node) const;
  // Returns true if it is safe to convert node to NoOp.
  bool SafeToConvertToNoOp(const NodeDef& node) const;
  // Removes all duplicate control dependencies.
  void CleanControlInputs();
  // Builds a map from the &optimized_graph_->node(i) to i.
  void BuildNodeToIdx();
  // Tries to optimize the node with the given index, possibly additional
  // optimizations by inserting nodes in nodes_to_simplify, and pruning nodes by
  // inserting them in nodes_to_delete.
  void OptimizeNode(int node_idx, SetVector<int>* nodes_to_simplify,
                    std::set<int>* nodes_to_delete);
  // Eliminates redundant control dependencies by computing the transitive
  // reduction of the graph.
  Status TransitiveReduction();
  // Main driver of dependency optimizations.
  Status OptimizeDependencies();
  // Replaces multiple cross-device control edges from the same device with a
  // single control edge.  If `host_granularity` is true then group control
  // edges from all devices on the same host.
  void GroupCrossDeviceControlEdges(bool host_granularity);

  bool fetch_nodes_known_;
  std::unordered_set<string> nodes_to_preserve_;
  std::unique_ptr<NodeMap> node_map_;
  std::unordered_map<const NodeDef*, int> node_to_idx_;
  GraphDef* optimized_graph_;  // Not owned.
};

}  // end namespace grappler
}  // end namespace tensorflow

#endif  // TENSORFLOW_CORE_GRAPPLER_OPTIMIZERS_DEPENDENCY_OPTIMIZER_H_
