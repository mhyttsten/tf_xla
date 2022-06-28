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

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_HLO_PASS_INTERFACE_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_HLO_PASS_INTERFACE_H_
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
class MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_pass_interfaceDTh {
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
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_pass_interfaceDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_pass_interfaceDTh() {
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


#include "absl/container/flat_hash_set.h"
#include "tensorflow/compiler/xla/service/hlo_module.h"
#include "tensorflow/compiler/xla/service/hlo_module_group.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/compiler/xla/types.h"

namespace xla {

// Base class for HLO passes. These are used with the HloPassPipeline to
// organize a sequence of passes. An HLO pass should not extend this class
// directly; it should extend HloModulePass or HloModuleGroupPass.
class HloPassInterface {
 public:
  // Struct that holds states of pass runs across multiple iterations.
  struct RunState {
    // The current iteration number.
    int iteration = 0;
    // Set of all changed computations from all pass runs using this state.
    absl::flat_hash_set<HloComputation*> changed;
    // Set of changed computation from previous iteration.
    absl::flat_hash_set<HloComputation*> changed_last_iteration;
    // Set of changed computation from current iteration.
    absl::flat_hash_set<HloComputation*> changed_this_iteration;

    RunState() = default;
    explicit RunState(HloModule* module)
        : changed_last_iteration(module->computations().begin(),
                                 module->computations().end()) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_pass_interfaceDTh mht_0(mht_0_v, 216, "", "./tensorflow/compiler/xla/service/hlo_pass_interface.h", "RunState");
}

    // Transition to the next iteration.
    //
    // Depending on the pass implmentation, one iteration includes all the work
    // done between two IncrementIteration calls, there can be arbitrary number
    // of passes that ran arbitrary times with this state.
    void IncrementIteration() {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_pass_interfaceDTh mht_1(mht_1_v, 226, "", "./tensorflow/compiler/xla/service/hlo_pass_interface.h", "IncrementIteration");

      using std::swap;
      changed.insert(changed_this_iteration.begin(),
                     changed_this_iteration.end());
      swap(changed_last_iteration, changed_this_iteration);
      changed_this_iteration.clear();
      ++iteration;
    }
  };
  virtual ~HloPassInterface() = default;
  virtual absl::string_view name() const = 0;

  // Run the pass on the given HLO module.  Returns whether it modified the
  // module.
  virtual StatusOr<bool> Run(HloModule* module) = 0;

  // Run the pass on computation on changed computations from last iteration in
  // given HLO module, with caller provided RunState which holds the state
  // information across multiple iterations.
  //
  // NOTE: This is a temporary default implementation that conservatively treats
  // all computations as changed. Eventually all passes should override this
  // method instead of Run() and Run() will call into this method instead.
  virtual Status RunOnChangedComputations(HloModule* module,
                                          RunState* run_state) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_pass_interfaceDTh mht_2(mht_2_v, 253, "", "./tensorflow/compiler/xla/service/hlo_pass_interface.h", "RunOnChangedComputations");

    TF_ASSIGN_OR_RETURN(bool changed, Run(module));
    if (changed) {
      auto computations = module->computations();
      run_state->changed_this_iteration.insert(computations.begin(),
                                               computations.end());
    }
    return Status::OK();
  }

  // Run the pass on the given HLO module group. Returns whether it modified the
  // module group. Ideally, the module group variant would be named "Run" as
  // well, but C++ does not handle overloaded virtual methods well.
  virtual StatusOr<bool> RunOnModuleGroup(HloModuleGroup* module_group) = 0;

  virtual bool IsPassPipeline() {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_pass_interfaceDTh mht_3(mht_3_v, 271, "", "./tensorflow/compiler/xla/service/hlo_pass_interface.h", "IsPassPipeline");
 return false; }
};

// Base class for passes which are module-scoped.
class HloModulePass : public HloPassInterface {
 public:
  // Runs the pass on a module group by iterating through each module in the
  // group.
  StatusOr<bool> RunOnModuleGroup(HloModuleGroup* module_group) override {
    bool changed = false;
    for (HloModule* module : module_group->modules()) {
      TF_ASSIGN_OR_RETURN(bool module_changed, Run(module));
      changed |= module_changed;
    }
    return changed;
  };

  // Update the layout of a Shape to one that is supported by a given backend.
  // One can call this function after modifying the Shape in case that modifying
  // the Shape requires changes to the layout for the given Backend.
  //
  // TODO(b/129084868): Make this Backend dependent instead of requiring
  // deriving from the pass and overriding this function.
  virtual void UpdateLayout(Shape* shape) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_pass_interfaceDTh mht_4(mht_4_v, 297, "", "./tensorflow/compiler/xla/service/hlo_pass_interface.h", "UpdateLayout");
}
};

// Base class for passes which are module-group scoped. These passes cannot run
// on an HLO module.
class HloModuleGroupPass : public HloPassInterface {
 public:
  StatusOr<bool> Run(HloModule* module) override {
    return InternalError("Module group pass cannot be run on a module");
  }
};

}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_HLO_PASS_INTERFACE_H_
