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

#ifndef TENSORFLOW_CORE_GRAPPLER_OPTIMIZERS_GRAPH_OPTIMIZER_STAGE_H_
#define TENSORFLOW_CORE_GRAPPLER_OPTIMIZERS_GRAPH_OPTIMIZER_STAGE_H_
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
class MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSgraph_optimizer_stageDTh {
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
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSgraph_optimizer_stageDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSgraph_optimizer_stageDTh() {
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


#include <unordered_map>
#include <unordered_set>

#include "absl/strings/str_cat.h"
#include "tensorflow/core/grappler/costs/graph_properties.h"
#include "tensorflow/core/grappler/grappler_item.h"
#include "tensorflow/core/grappler/utils.h"
#include "tensorflow/core/lib/gtl/flatset.h"
#include "tensorflow/core/protobuf/rewriter_config.pb.h"

namespace tensorflow {
namespace grappler {

struct NodeScopeAndName {
  string scope;
  string name;
};

// Parse scope and name: "a/b/c/Add_1" -> {"a/b/c", "Add_1"}
const NodeScopeAndName ParseNodeScopeAndName(const string& node_name);

// Context owned by GraphOptimizer, and passed to every stage at construction
// time. Each optimizer stage is responsible for updating it according to the
// changes it made to the graph.
//
// If an optimizer needs access to some helper class that is not present in this
// context, consider creating an extension context, specific to that
// optimizer (see example of ArithmeticOptimizerContext). GraphOptimizerContext
// should only have members that are useful to almost all optimizers.
struct GraphOptimizerContext {
  GraphOptimizerContext(const std::unordered_set<string>* nodes_to_preserve,
                        GraphDef* optimized_graph,
                        GraphProperties* graph_properties, NodeMap* node_map,
                        gtl::FlatSet<string>* feed_nodes,
                        RewriterConfig::Toggle opt_level)
      : nodes_to_preserve(nodes_to_preserve),
        optimized_graph(optimized_graph),
        graph_properties(graph_properties),
        node_map(node_map),
        feed_nodes(feed_nodes),
        opt_level(opt_level) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSgraph_optimizer_stageDTh mht_0(mht_0_v, 228, "", "./tensorflow/core/grappler/optimizers/graph_optimizer_stage.h", "GraphOptimizerContext");
}

  const std::unordered_set<string>* nodes_to_preserve;
  GraphDef* optimized_graph;
  GraphProperties* graph_properties;
  NodeMap* node_map;
  gtl::FlatSet<string>* feed_nodes;
  RewriterConfig::Toggle opt_level;
};

Status GetInputNode(const GraphOptimizerContext& ctx, const string& input,
                    NodeDef** node);
Status GetTensorProperties(const GraphOptimizerContext& ctx,
                           const string& tensor,
                           const OpInfo::TensorProperties** properties);

NodeDef* AddCopyNode(const GraphOptimizerContext& ctx, const string& name,
                     const NodeDef* node_to_copy);
NodeDef* AddEmptyNode(const GraphOptimizerContext& ctx, const string& name);

// WARNING:
// Optimizer stage must try to re-use original nodes of a graph and
// make all updates in place. This helps to make robust node placement
// decisions. Create new nodes only if there is a reason for that.

// Make a name for a new node obtained by optimizing a single node of the
// original graph. The optimized node is placed under the original node scope.
//
// Node name uniqueness is guaranteed by unique name of an original node in
// a same scope.
//
// Empty sub_scope or prefix ignored. At least one of them must be non-empty.
//
// Example: a/b/c/Add -> a/b/c/${sub_scope}/${prefix}_Add.
const string MakeOptimizedNodeName(const NodeScopeAndName& node,
                                   const string& sub_scope,
                                   const string& prefix);
// Make a name for a new node obtained by optimizing multiple nodes of the
// original graph, starting from "root". The optimized node is placed under
// the original scope of a "root" node.
//
// Example: [a/b/c/Add, x/y/z/Mul] -> a/b/c/${sub_scope}/${prefix}_Add_Mul
const string MakeOptimizedNodeName(const NodeScopeAndName& root,
                                   const std::vector<string> node_names,
                                   const string& sub_scope,
                                   const string& prefix);

// Base class for multi-stage GraphOptimizers (ArithmeticOptimizer, etc...).
//
// If a graph optimizer consists of large number of small independent
// rewrites, each of them should be implemented as a separate stage.
//
// * Result:
// Each graph optimizer choose what result is reported by each stage
// (e.g. each stage can fill in the name of optimized nodes, or have more
// complex result).
template <typename Result>
class GraphOptimizerStage {
 public:
  explicit GraphOptimizerStage(const string& optimizer_name,
                               const string& stage_name,
                               const GraphOptimizerContext& ctx)
      : optimizer_name_(optimizer_name), stage_name_(stage_name), ctx_(ctx) {
   std::vector<std::string> mht_1_v;
   mht_1_v.push_back("optimizer_name: \"" + optimizer_name + "\"");
   mht_1_v.push_back("stage_name: \"" + stage_name + "\"");
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSgraph_optimizer_stageDTh mht_1(mht_1_v, 295, "", "./tensorflow/core/grappler/optimizers/graph_optimizer_stage.h", "GraphOptimizerStage");
}
  virtual ~GraphOptimizerStage() = default;

  const string& stage_name() const {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSgraph_optimizer_stageDTh mht_2(mht_2_v, 301, "", "./tensorflow/core/grappler/optimizers/graph_optimizer_stage.h", "stage_name");
 return stage_name_; }
  const string& optimizer_name() const {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSgraph_optimizer_stageDTh mht_3(mht_3_v, 305, "", "./tensorflow/core/grappler/optimizers/graph_optimizer_stage.h", "optimizer_name");
 return optimizer_name_; }

  // Check if we should try to simplify node. Returning true doesn't
  // guarantee that node will be simplified.
  //
  // Should implement just a basic sanity check, without any expensive graph
  // traversals.
  virtual bool IsSupported(const NodeDef* node) const = 0;

  // Try to simplify the given node.
  //
  // Return error status only if some precondition is failed, or got an
  // incorrect graph. In every other case return Status:OK(), even if didn't
  // simplify anything.
  //
  // Report result using output argument. Each GraphOptimizer can choose it's
  // own Result type.
  // TODO(ezhulenev): if it will appear that Result output parameter is not
  // sufficiently useful (used with a reason by most optimizers), get rid of it,
  // and remove template parameter.
  virtual Status TrySimplify(NodeDef* node, Result* result) = 0;

  // Return InvalidArgumentError if node is not supported by the optimizer
  // stage.
  // TODO(ezhulenev): make this check part of non-virtual public API
  // (TrySimplify), and make virtual implementation protected.
  Status EnsureNodeIsSupported(const NodeDef* node) const {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSgraph_optimizer_stageDTh mht_4(mht_4_v, 334, "", "./tensorflow/core/grappler/optimizers/graph_optimizer_stage.h", "EnsureNodeIsSupported");

    return IsSupported(node)
               ? Status::OK()
               : errors::InvalidArgument(
                     "Node ", node->name(), " is not supported by optimizer ",
                     optimizer_name_, " and stage ", stage_name_);
  }

  // Get a name for a new node, created by this stage, based on one or multiple
  // nodes of an original graph.
  const string OptimizedNodeName(const NodeScopeAndName& node) const {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSgraph_optimizer_stageDTh mht_5(mht_5_v, 347, "", "./tensorflow/core/grappler/optimizers/graph_optimizer_stage.h", "OptimizedNodeName");

    return MakeOptimizedNodeName(node, optimizer_name_, stage_name_);
  }
  const string OptimizedNodeName(const NodeScopeAndName& root,
                                 const std::vector<string>& nodes) const {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSgraph_optimizer_stageDTh mht_6(mht_6_v, 354, "", "./tensorflow/core/grappler/optimizers/graph_optimizer_stage.h", "OptimizedNodeName");

    return MakeOptimizedNodeName(root, nodes, optimizer_name_, stage_name_);
  }
  const string OptimizedNodeName(const NodeScopeAndName& node,
                                 const string& rewrite_rule) const {
   std::vector<std::string> mht_7_v;
   mht_7_v.push_back("rewrite_rule: \"" + rewrite_rule + "\"");
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSgraph_optimizer_stageDTh mht_7(mht_7_v, 362, "", "./tensorflow/core/grappler/optimizers/graph_optimizer_stage.h", "OptimizedNodeName");

    const string prefix = strings::StrCat(stage_name_, "_", rewrite_rule);
    return MakeOptimizedNodeName(node, optimizer_name_, prefix);
  }

  const string UniqueOptimizedNodeName(const NodeScopeAndName& node) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSgraph_optimizer_stageDTh mht_8(mht_8_v, 370, "", "./tensorflow/core/grappler/optimizers/graph_optimizer_stage.h", "UniqueOptimizedNodeName");

    const string node_name = OptimizedNodeName(node);
    return UniqueNodeName(node_name);
  }
  const string UniqueOptimizedNodeName(const NodeScopeAndName& node,
                                       const string& rewrite_rule) {
   std::vector<std::string> mht_9_v;
   mht_9_v.push_back("rewrite_rule: \"" + rewrite_rule + "\"");
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSgraph_optimizer_stageDTh mht_9(mht_9_v, 379, "", "./tensorflow/core/grappler/optimizers/graph_optimizer_stage.h", "UniqueOptimizedNodeName");

    const string node_name = OptimizedNodeName(node, rewrite_rule);
    return UniqueNodeName(node_name);
  }

  // Get a node by input name from a node map. Return an error if node was not
  // found.
  Status GetInputNode(const string& input, NodeDef** node) const {
   std::vector<std::string> mht_10_v;
   mht_10_v.push_back("input: \"" + input + "\"");
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSgraph_optimizer_stageDTh mht_10(mht_10_v, 390, "", "./tensorflow/core/grappler/optimizers/graph_optimizer_stage.h", "GetInputNode");

    return ::tensorflow::grappler::GetInputNode(ctx_, input, node);
  }
  // Lookup tensor properties by name. Tensor name might have non-zero port
  // number. Return an error if tensor node doesn't exists in a graph, or it
  // doesn't have properties defined for requested port.
  Status GetTensorProperties(
      const string& tensor, const OpInfo::TensorProperties** properties) const {
   std::vector<std::string> mht_11_v;
   mht_11_v.push_back("tensor: \"" + tensor + "\"");
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSgraph_optimizer_stageDTh mht_11(mht_11_v, 401, "", "./tensorflow/core/grappler/optimizers/graph_optimizer_stage.h", "GetTensorProperties");

    return ::tensorflow::grappler::GetTensorProperties(ctx_, tensor,
                                                       properties);
  }

  NodeDef* AddCopyNode(const string& name, const NodeDef* node_to_copy) {
   std::vector<std::string> mht_12_v;
   mht_12_v.push_back("name: \"" + name + "\"");
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSgraph_optimizer_stageDTh mht_12(mht_12_v, 410, "", "./tensorflow/core/grappler/optimizers/graph_optimizer_stage.h", "AddCopyNode");

    return ::tensorflow::grappler::AddCopyNode(ctx_, name, node_to_copy);
  }
  NodeDef* AddEmptyNode(const string& name) {
   std::vector<std::string> mht_13_v;
   mht_13_v.push_back("name: \"" + name + "\"");
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSgraph_optimizer_stageDTh mht_13(mht_13_v, 417, "", "./tensorflow/core/grappler/optimizers/graph_optimizer_stage.h", "AddEmptyNode");

    return ::tensorflow::grappler::AddEmptyNode(ctx_, name);
  }

 protected:
  const GraphOptimizerContext& ctx() const {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSgraph_optimizer_stageDTh mht_14(mht_14_v, 425, "", "./tensorflow/core/grappler/optimizers/graph_optimizer_stage.h", "ctx");
 return ctx_; }

 private:
  const string UniqueNodeName(absl::string_view name) {
   std::vector<std::string> mht_15_v;
   mht_15_v.push_back("name: \"" + std::string(name.data(), name.size()) + "\"");
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSgraph_optimizer_stageDTh mht_15(mht_15_v, 432, "", "./tensorflow/core/grappler/optimizers/graph_optimizer_stage.h", "UniqueNodeName");

    string node_name = string(name);
    while (ctx_.node_map->NodeExists(node_name)) {
      node_name = absl::StrCat(name, "_unique",
                               optimized_node_name_counter_.fetch_add(1));
    }

    return node_name;
  }

  const string optimizer_name_;
  const string stage_name_;
  const GraphOptimizerContext ctx_;
  std::atomic<int64_t> optimized_node_name_counter_ = {0};
};

template <typename Result>
class GraphOptimizerStagePipeline {
 public:
  // Break predicate specifies if a pipeline should stop early, and not pass
  // a node to the next registered optimizer stage, typically that should be the
  // case when a stage successfully optimized a node, and it wants to yield
  // control to the optimizer.
  explicit GraphOptimizerStagePipeline(
      const std::function<bool(const Result&)> break_predicate)
      : break_predicate_(break_predicate) {
   std::vector<std::string> mht_16_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSgraph_optimizer_stageDTh mht_16(mht_16_v, 460, "", "./tensorflow/core/grappler/optimizers/graph_optimizer_stage.h", "GraphOptimizerStagePipeline");
}

  // Add a stage to the pipeline. It should be called with the arguments for the
  // stage constructor:
  //
  //   pipeline.AddStage<FooStage>(constructor_arg1, constructor_arg2);
  //
  // Returns a reference to the added stage.
  template <typename T, typename... Args>
  T& AddStage(Args&&... args) {
    auto stage = new T(std::forward<Args>(args)...);
    stages_.push_back(std::unique_ptr<T>(stage));
    return *stage;
  }

  // Pass a node through all registered optimizer stages, until break predicate
  // is true.
  //
  // Return true, if pipeline exited after a break predicate was evaluated as
  // 'true', which typically means that a node was optimized by one of the
  // registered stages.
  //
  // Return false, if node was not optimized by any of registered stages.
  bool PassThroughAllStages(NodeDef* node, Result* result) {
   std::vector<std::string> mht_17_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSgraph_optimizer_stageDTh mht_17(mht_17_v, 486, "", "./tensorflow/core/grappler/optimizers/graph_optimizer_stage.h", "PassThroughAllStages");

    for (auto& stage : stages_) {
      if (stage->IsSupported(node)) {
        const Status stage_status = stage->TrySimplify(node, result);
        // Each stage must be "error safe" (just like exception safe). In
        // case of any error it must leave optimized graph unmodified.
        if (!stage_status.ok()) {
          VLOG(2) << "Failed to run optimizer " << stage->optimizer_name()
                  << ", stage " << stage->stage_name() << " node "
                  << node->name()
                  << ". Error: " << stage_status.error_message();
        }
        if (break_predicate_(*result)) return true;
      }
    }
    return false;
  }

  // Pass a node through all registered optimizer stages, until break predicate
  // is true or a stage fails.
  //
  // Returns any stage failure status, or else Status::OK().
  Status PassThroughAllStagesWithStatus(NodeDef* node, Result* result) {
   std::vector<std::string> mht_18_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSgraph_optimizer_stageDTh mht_18(mht_18_v, 511, "", "./tensorflow/core/grappler/optimizers/graph_optimizer_stage.h", "PassThroughAllStagesWithStatus");

    for (auto& stage : stages_) {
      if (!stage->IsSupported(node)) {
        continue;
      }
      const Status stage_status = stage->TrySimplify(node, result);
      if (!stage_status.ok()) {
        return stage_status;
      } else if (break_predicate_(*result)) {
        break;
      }
    }
    return Status::OK();
  }

  std::size_t NumStages() {
   std::vector<std::string> mht_19_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSgraph_optimizer_stageDTh mht_19(mht_19_v, 529, "", "./tensorflow/core/grappler/optimizers/graph_optimizer_stage.h", "NumStages");
 return stages_.size(); }

  std::vector<string> StageNames() {
    std::vector<string> names;
    for (const auto& stage : stages_) {
      names.push_back(stage->stage_name());
    }
    return names;
  }

 private:
  std::vector<std::unique_ptr<GraphOptimizerStage<Result>>> stages_;
  std::function<bool(const Result&)> break_predicate_;

  TF_DISALLOW_COPY_AND_ASSIGN(GraphOptimizerStagePipeline);
};

}  // end namespace grappler
}  // end namespace tensorflow

#endif  // TENSORFLOW_CORE_GRAPPLER_OPTIMIZERS_GRAPH_OPTIMIZER_STAGE_H_
