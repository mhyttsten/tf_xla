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

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_HLO_PASS_PIPELINE_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_HLO_PASS_PIPELINE_H_
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
class MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_pass_pipelineDTh {
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
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_pass_pipelineDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_pass_pipelineDTh() {
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
#include <memory>
#include <string>
#include <vector>

#include "absl/memory/memory.h"
#include "absl/strings/str_cat.h"
#include "tensorflow/compiler/xla/service/compilation_stats.h"
#include "tensorflow/compiler/xla/service/hlo_module.h"
#include "tensorflow/compiler/xla/service/hlo_pass_interface.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/compiler/xla/types.h"

namespace xla {

class PhaseOrderPipeline;

// Pipeline of HLO passes.
class HloPassPipeline : public HloPassInterface {
 public:
  explicit HloPassPipeline(const std::string& name,
                           CompilationStats* compilation_stats = nullptr)
      : name_(name), compilation_stats_(compilation_stats) {
   std::vector<std::string> mht_0_v;
   mht_0_v.push_back("name: \"" + name + "\"");
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_pass_pipelineDTh mht_0(mht_0_v, 211, "", "./tensorflow/compiler/xla/service/hlo_pass_pipeline.h", "HloPassPipeline");

    if (compilation_stats == nullptr) {
      empty_compilation_stats_ = CompilationStats::MakeNoopStats();
      compilation_stats_ = empty_compilation_stats_.get();
    }
  }
  absl::string_view name() const override {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_pass_pipelineDTh mht_1(mht_1_v, 220, "", "./tensorflow/compiler/xla/service/hlo_pass_pipeline.h", "name");
 return name_; }

  // Add a pass to the pipeline. It should be called with the arguments for the
  // pass constructor:
  //
  //   pipeline.AddPass<FooPass>(constructor_arg1, constructor_arg2);
  //
  // Returns a reference to the added pass.
  template <typename T, typename... Args>
  T& AddPass(Args&&... args) {
    CHECK(!run_called_) << "AddPass cannot be called after Run";
    auto pass = new T(std::forward<Args>(args)...);
    passes_.push_back(std::unique_ptr<T>(pass));
    return *pass;
  }

  // Add an invariant-checking pass to the pipeline. It will be run before and
  // after each HLO pass. The invariant checking pass must not mutate the graph
  // (it is required to always return "false" from its Run() method).
  template <typename T, typename... Args>
  T& AddInvariantChecker(Args&&... args) {
    CHECK(!run_called_) << "AddInvariantChecker cannot be called after Run";
    auto pass = new T(std::forward<Args>(args)...);
    invariant_checkers_.push_back(std::unique_ptr<T>(pass));
    return *pass;
  }

  // Add an invariant-checking pass to the pipeline on debug builds only.
  template <typename T, typename... Args>
  void AddInvariantCheckerDebug(Args&&... args) {
#ifndef NDEBUG
    AddInvariantChecker<T>(std::forward<Args>(args)...);
#endif  // NDEBUG
  }

  StatusOr<bool> Run(HloModule* module) override;
  StatusOr<bool> RunOnModuleGroup(HloModuleGroup* module_group) override;

  bool IsPassPipeline() override {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_pass_pipelineDTh mht_2(mht_2_v, 261, "", "./tensorflow/compiler/xla/service/hlo_pass_pipeline.h", "IsPassPipeline");
 return true; }

  // Return size of passes_.
  int PassesSize() {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_pass_pipelineDTh mht_3(mht_3_v, 267, "", "./tensorflow/compiler/xla/service/hlo_pass_pipeline.h", "PassesSize");
 return passes_.size(); }
  // Return reference to pass specified by index.
  HloPassInterface& GetPass(int index) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_pass_pipelineDTh mht_4(mht_4_v, 272, "", "./tensorflow/compiler/xla/service/hlo_pass_pipeline.h", "GetPass");
 return *passes_[index]; }

 private:
  // Returns the set of passes which are enabled. DebugOptions can selectively
  // disable passes via --xla_disable_hlo_passes flag.
  std::vector<HloPassInterface*> GetEnabledPasses(
      const DebugOptions& debug_options);

  // Maybe dumps the given module or module group depending on flag values
  // contained in DebugOptions of module config. If it is dumped, saves the
  // filenames of the dumps into module metadata.
  void MaybeDumpHloAndSaveFilenames(HloModuleGroup& module_group,
                                    absl::string_view after_pass_name,
                                    absl::string_view before_pass_name);
  void MaybeDumpHloAndSaveFilenames(HloModule& module,
                                    absl::string_view after_pass_name,
                                    absl::string_view before_pass_name);

  // Runs the invariant checker on the given HLO. HloT can be either HloModule
  // or HloModuleGroup.
  template <typename HloT>
  Status RunInvariantCheckers(HloT* hlo, absl::string_view after_pass_name);

  // Helper which runs the given pass on the given HLO. HloT can be either
  // HloModule or HloModuleGroup.
  template <typename HloT>
  StatusOr<bool> RunPassesInternal(HloT* hlo,
                                   const DebugOptions& debug_options);

  // Helpers which run the given passes on the given HLO construct. These
  // helpers enable templating of the core of the pipeline logic by providing
  // HloModule and HloModuleGroup specific methods with the same name.
  static StatusOr<bool> RunHelper(HloPassInterface* pass, HloModule* module) {
    TF_ASSIGN_OR_RETURN(bool changed, pass->Run(module));
    module->Cleanup();
    return changed;
  }
  static StatusOr<bool> RunHelper(HloPassInterface* pass,
                                  HloModuleGroup* module_group) {
    TF_ASSIGN_OR_RETURN(bool changed, pass->RunOnModuleGroup(module_group));
    module_group->Cleanup();
    return changed;
  }

  const std::string name_;
  std::vector<std::unique_ptr<HloPassInterface>> passes_;
  std::vector<std::unique_ptr<HloPassInterface>> invariant_checkers_;
  bool run_called_ = false;

  CompilationStats* compilation_stats_;
  // Default stats instance for when one is not passed in the constructor.
  // Use via compilation_stats_, not directly.
  std::unique_ptr<CompilationStats> empty_compilation_stats_;

  // Allow PhaseOrderPipeline to modify private passes_ member in order to
  // perform PhaseOrdering.
  friend class ::xla::PhaseOrderPipeline;
};

}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_HLO_PASS_PIPELINE_H_
