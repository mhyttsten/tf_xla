/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_COMPILER_MLIR_MLIR_GRAPH_OPTIMIZATION_PASS_H_
#define TENSORFLOW_COMPILER_MLIR_MLIR_GRAPH_OPTIMIZATION_PASS_H_
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
class MHTracer_DTPStensorflowPScompilerPSmlirPSmlir_graph_optimization_passDTh {
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
   MHTracer_DTPStensorflowPScompilerPSmlirPSmlir_graph_optimization_passDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSmlirPSmlir_graph_optimization_passDTh() {
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

#include "tensorflow/compiler/mlir/mlir_bridge_rollout_policy.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "tensorflow/core/common_runtime/function_optimization_registry.h"
#include "tensorflow/core/common_runtime/optimization_registry.h"

namespace tensorflow {

// -------------------------------------------------------------------------- //
// MLIR passes running on Tensorflow function graphs (Tensorflow V2).
// -------------------------------------------------------------------------- //

// Disabled - skip execution of the pass.
// Enabled - execute the pass, propagate errors to the caller if any.
// FallbackEnabled - execute the pass and commit all the changes to the MLIR
//   module in case of success. Do not commit any changes in case of failures,
//   let the rest of the pipeline run.
enum class MlirOptimizationPassState { Disabled, Enabled, FallbackEnabled };

// An API for registering MLIR ModulePass with the Tensorflow runtime. These
// passes are running only for function graphs built by Tensorflow V2 and
// instantiated by the process_function_library_runtime (see
// FunctionOptimizationPass for details).
class MlirOptimizationPass {
 public:
  virtual ~MlirOptimizationPass() = default;
  virtual llvm::StringRef name() const = 0;

  // Returns an enum value:
  //   Enabled if the pass is enabled for the given graph with specified config.
  //   Disabled if the pass is disabled.
  //   FallbackEnabled if the pass needs to be executed in fallback mode.
  //
  // When the pass is FallbackEnabled, the pass is executed and the changes it
  // makes to the MLIR module will be committed only if the pass was successful,
  // otherwise no changes are committed and the rest of the pipeline is run.
  //
  // `device_set` can be nullptr if the devices information is not
  // available or no device specific filtering is required.
  // `function_library` contains function definitions for function calls in
  // `graph` not included in the `graph` FunctionLibraryDefinition.
  virtual MlirOptimizationPassState GetPassState(
      const DeviceSet* device_set, const ConfigProto& config_proto,
      const Graph& graph,
      const FunctionLibraryDefinition& function_library) const = 0;

  virtual Status Run(const ConfigProto& config_proto, mlir::ModuleOp module,
                     const Graph& graph,
                     const FunctionLibraryDefinition& function_library) = 0;
};

class MlirOptimizationPassRegistry {
 public:
  struct PassRegistration {
    int priority;
    std::unique_ptr<MlirOptimizationPass> pass;
  };

  struct PriorityComparator {
    bool operator()(const PassRegistration& x,
                    const PassRegistration& y) const {
      return x.priority < y.priority;
    }
  };

  using Passes = std::set<PassRegistration, PriorityComparator>;

  // Returns the global registry of MLIR optimization passes.
  static MlirOptimizationPassRegistry& Global();

  // Register optimization `pass` with the given `priority`.
  void Add(int priority, std::unique_ptr<MlirOptimizationPass> pass) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSmlir_graph_optimization_passDTh mht_0(mht_0_v, 261, "", "./tensorflow/compiler/mlir/mlir_graph_optimization_pass.h", "Add");

    auto inserted = passes_.insert({priority, std::move(pass)});
    CHECK(inserted.second)
        << "Pass priority must be unique. "
        << "Previously registered pass with the same priority: "
        << inserted.first->pass->name().str();
  }

  // Free the memory allocated for all passes.
  void ClearPasses() {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSmlir_graph_optimization_passDTh mht_1(mht_1_v, 273, "", "./tensorflow/compiler/mlir/mlir_graph_optimization_pass.h", "ClearPasses");
 passes_.clear(); }

  const Passes& passes() const {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSmlir_graph_optimization_passDTh mht_2(mht_2_v, 278, "", "./tensorflow/compiler/mlir/mlir_graph_optimization_pass.h", "passes");
 return passes_; }

 private:
  Passes passes_;
};

// Function optimization pass that runs all MLIR passes registered in
// MlirOptimizationPassRegistry.
class MlirFunctionOptimizationPass : public FunctionOptimizationPass {
 public:
  explicit MlirFunctionOptimizationPass(
      const MlirOptimizationPassRegistry* registry =
          &MlirOptimizationPassRegistry::Global())
      : registry_(registry) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSmlir_graph_optimization_passDTh mht_3(mht_3_v, 294, "", "./tensorflow/compiler/mlir/mlir_graph_optimization_pass.h", "MlirFunctionOptimizationPass");
}

  // Executes all of the underlying registered MlirOptimizationPasses.
  Status Run(const DeviceSet& device_set, const ConfigProto& config_proto,
             std::unique_ptr<Graph>* graph, FunctionLibraryDefinition* flib_def,
             std::vector<std::string>* control_ret_node_names,
             bool* control_rets_updated) override;

 private:
  const MlirOptimizationPassRegistry* registry_;
};

// -------------------------------------------------------------------------- //
// MLIR passes running on Tensorflow V1 graphs.
// -------------------------------------------------------------------------- //

// An API for registering MLIR ModulePass with the Tensorflow runtime. These
// passes are running only for V1 graphs (legacy graphs) executed via Session
// runtime. Graph importer updates legacy graph behavior to V2 constructs (e.g.
// it raises control flow from Switch/Merge nodes to functional control flow
// with If/While operations).
class MlirV1CompatOptimizationPass {
 public:
  virtual ~MlirV1CompatOptimizationPass() = default;
  virtual llvm::StringRef name() const = 0;

  // Returns a MlirOptimizationPassState based on the given graph and
  // config. See comments on `MlirOptimizationPassState` enum for more info
  // on exact values.
  virtual MlirOptimizationPassState GetPassState(
      const DeviceSet* device_set, const ConfigProto& config_proto,
      const Graph& graph,
      const FunctionLibraryDefinition& function_library) const = 0;

  virtual Status Run(const GraphOptimizationPassOptions& options,
                     mlir::ModuleOp module) = 0;
};

class MlirV1CompatOptimizationPassRegistry {
 public:
  // Returns the global registry of MLIR optimization passes.
  static MlirV1CompatOptimizationPassRegistry& Global();

  void Add(std::unique_ptr<MlirV1CompatOptimizationPass> pass) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSmlir_graph_optimization_passDTh mht_4(mht_4_v, 340, "", "./tensorflow/compiler/mlir/mlir_graph_optimization_pass.h", "Add");

    CHECK(pass_ == nullptr) << "Only a single pass can be registered";
    pass_ = std::move(pass);
  }

  MlirV1CompatOptimizationPass* pass() const {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSmlir_graph_optimization_passDTh mht_5(mht_5_v, 348, "", "./tensorflow/compiler/mlir/mlir_graph_optimization_pass.h", "pass");

    return pass_ ? pass_.get() : nullptr;
  }

 private:
  std::unique_ptr<MlirV1CompatOptimizationPass> pass_{};
};

class MlirV1CompatGraphOptimizationPass : public GraphOptimizationPass {
 public:
  explicit MlirV1CompatGraphOptimizationPass(
      const MlirV1CompatOptimizationPassRegistry* registry =
          &MlirV1CompatOptimizationPassRegistry::Global())
      : registry_(registry) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSmlir_graph_optimization_passDTh mht_6(mht_6_v, 364, "", "./tensorflow/compiler/mlir/mlir_graph_optimization_pass.h", "MlirV1CompatGraphOptimizationPass");
}

  Status Run(const GraphOptimizationPassOptions& options) override;

 private:
  const MlirV1CompatOptimizationPassRegistry* registry_;
};

// -------------------------------------------------------------------------- //
// Helper classes for static registration of MLIR (V1 Compat) passes in the
// corresponding registry.
// -------------------------------------------------------------------------- //

namespace mlir_pass_registration {

class MlirOptimizationPassRegistration {
 public:
  explicit MlirOptimizationPassRegistration(
      int priority, std::unique_ptr<MlirOptimizationPass> pass) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSmlir_graph_optimization_passDTh mht_7(mht_7_v, 385, "", "./tensorflow/compiler/mlir/mlir_graph_optimization_pass.h", "MlirOptimizationPassRegistration");

    MlirOptimizationPassRegistry::Global().Add(priority, std::move(pass));
  }
};

class MlirV1CompatOptimizationPassRegistration {
 public:
  explicit MlirV1CompatOptimizationPassRegistration(
      std::unique_ptr<MlirV1CompatOptimizationPass> pass) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSmlir_graph_optimization_passDTh mht_8(mht_8_v, 396, "", "./tensorflow/compiler/mlir/mlir_graph_optimization_pass.h", "MlirV1CompatOptimizationPassRegistration");

    MlirV1CompatOptimizationPassRegistry::Global().Add(std::move(pass));
  }
};

}  // namespace mlir_pass_registration

}  // namespace tensorflow

#endif  // TENSORFLOW_COMPILER_MLIR_MLIR_GRAPH_OPTIMIZATION_PASS_H_
