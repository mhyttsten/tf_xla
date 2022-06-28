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

#ifndef TENSORFLOW_CORE_GRAPPLER_OPTIMIZERS_META_OPTIMIZER_H_
#define TENSORFLOW_CORE_GRAPPLER_OPTIMIZERS_META_OPTIMIZER_H_
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
class MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSmeta_optimizerDTh {
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
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSmeta_optimizerDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSmeta_optimizerDTh() {
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


#include "tensorflow/core/common_runtime/device_set.h"
#include "tensorflow/core/framework/device_base.h"
#include "tensorflow/core/framework/function.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/grappler/grappler_item.h"
#include "tensorflow/core/grappler/optimizers/graph_optimizer.h"
#include "tensorflow/core/grappler/verifiers/graph_verifier.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/protobuf/config.pb.h"
#include "tensorflow/core/protobuf/rewriter_config.pb.h"
#include "tensorflow/core/protobuf/verifier_config.pb.h"

namespace tensorflow {
namespace grappler {

// Run the other grappler optimizers based on the specified rewriter config.
class MetaOptimizer : public GraphOptimizer {
 public:
  MetaOptimizer(DeviceBase* cpu_device, const ConfigProto& cfg);
  ~MetaOptimizer() override = default;

  string name() const override {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSmeta_optimizerDTh mht_0(mht_0_v, 209, "", "./tensorflow/core/grappler/optimizers/meta_optimizer.h", "name");
 return "meta_optimizer"; };

  bool UsesFunctionLibrary() const override {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSmeta_optimizerDTh mht_1(mht_1_v, 214, "", "./tensorflow/core/grappler/optimizers/meta_optimizer.h", "UsesFunctionLibrary");
 return true; }

  Status Optimize(Cluster* cluster, const GrapplerItem& item,
                  GraphDef* optimized_graph) override {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSmeta_optimizerDTh mht_2(mht_2_v, 220, "", "./tensorflow/core/grappler/optimizers/meta_optimizer.h", "Optimize");

    GrapplerItem copy(item);
    return OptimizeConsumeItem(cluster, std::move(copy), optimized_graph);
  }

  Status OptimizeConsumeItem(Cluster* cluster, GrapplerItem&& item,
                             GraphDef* optimized_graph);

  string GetResultString() const;

  void PrintResult();

 private:
  std::unique_ptr<GraphOptimizer> MakeNewOptimizer(
      const string& optimizer, const std::set<string>& device_types) const;

  // When grappler should lower control flow to V1 switch/merge style nodes.
  bool LowerControlFlow() const;

  // Initialize active optimizers from RewriterConfig toggles.
  Status InitializeOptimizers(
      const std::set<string>& device_types,
      std::vector<std::unique_ptr<GraphOptimizer>>* optimizers) const;
  // Initialize active optimizers from RewriterConfig optimizer names.
  Status InitializeOptimizersByName(
      const std::set<string>& device_types,
      std::vector<std::unique_ptr<GraphOptimizer>>* optimizers) const;
  // Initialize active optimizers from RewriterConfig.custom_optimizers.
  Status InitializeCustomGraphOptimizers(
      const std::set<string>& device_types,
      const std::set<string>& pre_initialized_optimizers,
      std::vector<std::unique_ptr<GraphOptimizer>>* optimizers) const;
  Status InitializePluginGraphOptimizers(
      const std::set<string>& device_types,
      std::vector<std::unique_ptr<GraphOptimizer>>* optimizers) const;
  // Returns the config for a custom graph optimizer. Null if none was found.
  const RewriterConfig::CustomGraphOptimizer* GetCustomGraphOptimizerConfig(
      const string& name) const;

  // Initialize active verifiers from the RewriterConfig toggles.
  void InitializeVerifiers(
      std::vector<std::unique_ptr<GraphVerifier>>* inter_optimizer_verifiers,
      std::vector<std::unique_ptr<GraphVerifier>>* post_optimization_verifiers)
      const;

  void PrintUserAndPluginConfigs(const std::set<string>& device_types) const;

  // Run optimization pass over a single GrapplerItem. Meta optimizer might run
  // multiple such passes: 1) for the main graph 2) for the function library
  Status OptimizeGraph(Cluster* cluster, GrapplerItem&& item,
                       GraphDef* optimized_graph);

  DeviceBase* const cpu_device_;  // may be NULL
  ConfigProto config_proto_;
  RewriterConfig& cfg_;
  bool xla_auto_clustering_on_;

  struct OptimizerResult {
    string optimizer_name;
    string message;
    Status status;
  };

  struct GraphOptimizationResult {
    explicit GraphOptimizationResult(const string& id) : id(id) {
   std::vector<std::string> mht_3_v;
   mht_3_v.push_back("id: \"" + id + "\"");
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSmeta_optimizerDTh mht_3(mht_3_v, 288, "", "./tensorflow/core/grappler/optimizers/meta_optimizer.h", "GraphOptimizationResult");
}
    string id;
    std::vector<OptimizerResult> results;
  };

  Status RunOptimizer(GraphOptimizer* optimizer, Cluster* cluster,
                      GrapplerItem* optimized_item, GraphDef* optimized_graph,
                      GraphOptimizationResult* optimization_result);

  std::vector<GraphOptimizationResult> optimization_results_;
};

bool MetaOptimizerEnabled(const ConfigProto& cfg);

// Run the meta optimizer.
//
// If <cpu_device> is non-null, it is the device to be used for executing ops
// during constant folding; if NULL, a new device is created for doing constant
// folding. For performance, it is recommended to pass in an existing cpu_device
// when possible.
Status RunMetaOptimizer(GrapplerItem&& item, const ConfigProto& cfg,
                        DeviceBase* cpu_device, Cluster* cluster,
                        GraphDef* optimized_graph);

// Wrapper around RunMetaOptimizer convenient for optimizing
// function graphs.
//
// Runs grappler optimizations on `g` based on `config_proto`.
// `ret_node_names`: a vector of node names whose outputs are returned,
//    aka fetches. when `g` represent a function, these are _Retval nodes.
// `lib`: function library to use with `g`.
// `device_set`: the set of devices that graph can refer to.
// `cpu_device`: the CPU device.
// `config_proto`: Grapper configuration.
// `grappler_item_id': Grappler item id (e.g. optimized function name).
// `optimization_options`: Grappler optimization constraints that are known only
//    at runtime.
//
// **g is a graph constructed based on the runtime library 'lib'.
// OptimizeGraph mutates **g extensively and replaces '*g' with a
// complete copy. Therefore, the caller should not keep any references
// to nodes *g.
Status OptimizeGraph(
    std::vector<string> ret_node_names, std::vector<string> keep_node_names,
    FunctionLibraryDefinition* lib, const DeviceSet& device_set,
    Device* cpu_device, const ConfigProto& config_proto,
    const string& grappler_item_id,
    const GrapplerItem::OptimizationOptions& optimization_options,
    std::unique_ptr<tensorflow::Graph>* g);

}  // namespace grappler
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_GRAPPLER_OPTIMIZERS_META_OPTIMIZER_H_
