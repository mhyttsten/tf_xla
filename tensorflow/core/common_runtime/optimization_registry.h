/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

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

// Classes to maintain a static registry of whole-graph optimization
// passes to be applied by the Session when it initializes a graph.
#ifndef TENSORFLOW_CORE_COMMON_RUNTIME_OPTIMIZATION_REGISTRY_H_
#define TENSORFLOW_CORE_COMMON_RUNTIME_OPTIMIZATION_REGISTRY_H_
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
class MHTracer_DTPStensorflowPScorePScommon_runtimePSoptimization_registryDTh {
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
   MHTracer_DTPStensorflowPScorePScommon_runtimePSoptimization_registryDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePScommon_runtimePSoptimization_registryDTh() {
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


#include <functional>
#include <map>
#include <vector>

#include "tensorflow/core/common_runtime/composite_device.h"
#include "tensorflow/core/common_runtime/device_set.h"
#include "tensorflow/core/framework/function.h"
#include "tensorflow/core/graph/costmodel.h"
#include "tensorflow/core/graph/graph.h"

namespace tensorflow {
struct SessionOptions;

// All the parameters used by an optimization pass are packaged in
// this struct. They should be enough for the optimization pass to use
// as a key into a state dictionary if it wants to keep state across
// calls.
struct GraphOptimizationPassOptions {
  // Filled in by DirectSession for PRE_PLACEMENT optimizations. Can be empty.
  string session_handle;
  const SessionOptions* session_options = nullptr;
  const CostModel* cost_model = nullptr;

  FunctionLibraryDefinition* flib_def = nullptr;  // Not owned.
  // The DeviceSet contains all the devices known to the system and is
  // filled in for optimizations run by the session master, i.e.,
  // PRE_PLACEMENT, POST_PLACEMENT, and POST_REWRITE_FOR_EXEC. It is
  // nullptr for POST_PARTITIONING optimizations which are run at the
  // workers.
  const DeviceSet* device_set = nullptr;  // Not owned.

  // Maps from a CompositeDevice name to a list of underlying physical
  // devices.
  const std::vector<CompositeDevice*>* composite_devices =
      nullptr;  // Not owned.

  // The graph to optimize, for optimization passes that run before
  // partitioning. Null for post-partitioning passes.
  // An optimization pass may replace *graph with a new graph object.
  std::unique_ptr<Graph>* graph = nullptr;

  // Graphs for each partition, if running post-partitioning. Optimization
  // passes may alter the graphs, but must not add or remove partitions.
  // Null for pre-partitioning passes.
  std::unordered_map<string, std::unique_ptr<Graph>>* partition_graphs =
      nullptr;

  // Indicator of whether or not the graph was derived from a function.
  bool is_function_graph = false;
  // Set when is_function_graph is true. The default device where the function
  // runs. If nullptr, it runs on the local host.
  const Device* default_function_device = nullptr;
  // Set when is_function_graph is true. The function where the graph was
  // derived. `graph` doesn't contain all the information in the function_def,
  // e.g. function attributes.
  const FunctionDef* function_def = nullptr;

  // TODO(b/176491312): Remove this if shape inference on import flag is
  // removed. If True, allows mlir roundtrip to run shape inference on import.
  bool shape_inference_on_tfe_dialect_import = true;
};

// Optimization passes are implemented by inheriting from
// GraphOptimizationPass.
class GraphOptimizationPass {
 public:
  virtual ~GraphOptimizationPass() {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSoptimization_registryDTh mht_0(mht_0_v, 256, "", "./tensorflow/core/common_runtime/optimization_registry.h", "~GraphOptimizationPass");
}
  virtual Status Run(const GraphOptimizationPassOptions& options) = 0;
  void set_name(const string& name) {
   std::vector<std::string> mht_1_v;
   mht_1_v.push_back("name: \"" + name + "\"");
   MHTracer_DTPStensorflowPScorePScommon_runtimePSoptimization_registryDTh mht_1(mht_1_v, 262, "", "./tensorflow/core/common_runtime/optimization_registry.h", "set_name");
 name_ = name; }
  string name() const {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSoptimization_registryDTh mht_2(mht_2_v, 266, "", "./tensorflow/core/common_runtime/optimization_registry.h", "name");
 return name_; }

 private:
  // The name of the optimization pass, which is the same as the inherited
  // class name.
  string name_;
};

// The key is a 'phase' number. Phases are executed in increasing
// order. Within each phase the order of passes is undefined.
typedef std::map<int, std::vector<std::unique_ptr<GraphOptimizationPass>>>
    GraphOptimizationPasses;

// A global OptimizationPassRegistry is used to hold all passes.
class OptimizationPassRegistry {
 public:
  // Groups of passes are run at different points in initialization.
  enum Grouping {
    PRE_PLACEMENT,          // after cost model assignment, before placement.
    POST_PLACEMENT,         // after placement.
    POST_REWRITE_FOR_EXEC,  // after re-write using feed/fetch endpoints.
    POST_PARTITIONING,      // after partitioning
  };

  // Add an optimization pass to the registry.
  void Register(Grouping grouping, int phase,
                std::unique_ptr<GraphOptimizationPass> pass);

  const std::map<Grouping, GraphOptimizationPasses>& groups() {
    return groups_;
  }

  // Run all passes in grouping, ordered by phase, with the same
  // options.
  Status RunGrouping(Grouping grouping,
                     const GraphOptimizationPassOptions& options);

  // Returns the global registry of optimization passes.
  static OptimizationPassRegistry* Global();

  // Prints registered optimization passes for debugging.
  void LogGrouping(Grouping grouping, int vlog_level);
  void LogAllGroupings(int vlog_level);

 private:
  std::map<Grouping, GraphOptimizationPasses> groups_;
};

namespace optimization_registration {

class OptimizationPassRegistration {
 public:
  OptimizationPassRegistration(OptimizationPassRegistry::Grouping grouping,
                               int phase,
                               std::unique_ptr<GraphOptimizationPass> pass,
                               string optimization_pass_name) {
   std::vector<std::string> mht_3_v;
   mht_3_v.push_back("optimization_pass_name: \"" + optimization_pass_name + "\"");
   MHTracer_DTPStensorflowPScorePScommon_runtimePSoptimization_registryDTh mht_3(mht_3_v, 325, "", "./tensorflow/core/common_runtime/optimization_registry.h", "OptimizationPassRegistration");

    pass->set_name(optimization_pass_name);
    OptimizationPassRegistry::Global()->Register(grouping, phase,
                                                 std::move(pass));
  }
};

}  // namespace optimization_registration

#define REGISTER_OPTIMIZATION(grouping, phase, optimization) \
  REGISTER_OPTIMIZATION_UNIQ_HELPER(__COUNTER__, grouping, phase, optimization)

#define REGISTER_OPTIMIZATION_UNIQ_HELPER(ctr, grouping, phase, optimization) \
  REGISTER_OPTIMIZATION_UNIQ(ctr, grouping, phase, optimization)

#define REGISTER_OPTIMIZATION_UNIQ(ctr, grouping, phase, optimization)         \
  static ::tensorflow::optimization_registration::OptimizationPassRegistration \
      register_optimization_##ctr(                                             \
          grouping, phase,                                                     \
          ::std::unique_ptr<::tensorflow::GraphOptimizationPass>(              \
              new optimization()),                                             \
          #optimization)

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_COMMON_RUNTIME_OPTIMIZATION_REGISTRY_H_
