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
class MHTracer_DTPStensorflowPScompilerPSmlirPSlitePStransformsPStrim_functions_tfDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePStransformsPStrim_functions_tfDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSmlirPSlitePStransformsPStrim_functions_tfDTcc() {
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

#include <queue>
#include <string>

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/Support/CommandLine.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/Location.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/IR/SymbolTable.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/lite/transforms/passes.h"

// The cmd line flag to specify the allowlist of functions. Rest are trimmed
// after this pass is run.
// NOLINTNEXTLINE
static llvm::cl::list<std::string> trim_funcs_allowlist(
    "tfl-trim-funcs-allowlist", llvm::cl::value_desc("list"),
    llvm::cl::desc("comma separated list of allowlisted functions. The first "
                   "function specified will be used as main."),
    llvm::cl::CommaSeparated);

namespace mlir {
namespace TFL {
namespace {

// The pass to trim functions before we legalize to TFL
// dialect using the specified allowlist.
class TrimFunctionsPass
    : public mlir::PassWrapper<TrimFunctionsPass, OperationPass<ModuleOp>> {
 public:
  explicit TrimFunctionsPass() : trim_funcs_allowlist_(trim_funcs_allowlist) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePStransformsPStrim_functions_tfDTcc mht_0(mht_0_v, 218, "", "./tensorflow/compiler/mlir/lite/transforms/trim_functions_tf.cc", "TrimFunctionsPass");
}
  explicit TrimFunctionsPass(llvm::ArrayRef<std::string> trim_funcs_allowlist)
      : trim_funcs_allowlist_(trim_funcs_allowlist) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePStransformsPStrim_functions_tfDTcc mht_1(mht_1_v, 223, "", "./tensorflow/compiler/mlir/lite/transforms/trim_functions_tf.cc", "TrimFunctionsPass");
}

  StringRef getArgument() const final {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePStransformsPStrim_functions_tfDTcc mht_2(mht_2_v, 228, "", "./tensorflow/compiler/mlir/lite/transforms/trim_functions_tf.cc", "getArgument");

    // This is the argument used to refer to the pass in
    // the textual format (on the commandline for example).
    return "tfl-trim-funcs-tf";
  }
  StringRef getDescription() const final {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePStransformsPStrim_functions_tfDTcc mht_3(mht_3_v, 236, "", "./tensorflow/compiler/mlir/lite/transforms/trim_functions_tf.cc", "getDescription");

    // This is a brief description of the pass.
    return "Trim functions to restrict them to a specified allowlist prior to "
           "legalization to TensorFlow lite dialect";
  }

 private:
  void runOnOperation() override;
  bool TrimModule();
  void Verify();

  llvm::ArrayRef<std::string> trim_funcs_allowlist_;
};

void TrimFunctionsPass::runOnOperation() {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePStransformsPStrim_functions_tfDTcc mht_4(mht_4_v, 253, "", "./tensorflow/compiler/mlir/lite/transforms/trim_functions_tf.cc", "TrimFunctionsPass::runOnOperation");

  // trim the functions in the module using the trim_funcs_allowlist_
  // by removing functions not in the allowlist.
  if (TrimModule()) {
    // verify the updated module is still valid, if not signal the
    // pass as failed.
    Verify();
  }
}

bool TrimFunctionsPass::TrimModule() {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePStransformsPStrim_functions_tfDTcc mht_5(mht_5_v, 266, "", "./tensorflow/compiler/mlir/lite/transforms/trim_functions_tf.cc", "TrimFunctionsPass::TrimModule");

  // if no trim_funcs_allowlist_ is specified, this pass is a no-op.
  if (trim_funcs_allowlist_.empty()) return false;

  llvm::SmallVector<FuncOp, 4> funcs_to_trim;
  for (auto func : getOperation().getOps<FuncOp>()) {
    if (llvm::is_contained(trim_funcs_allowlist_, func.getName())) {
      // If no main is specified in the allowlist, use the 1st func
      // in trim_funcs_allowlist as the main.
      // TODO(ashwinm): Currently tflite flatbuffer export assumes there is
      // always a main. This is strictly not required for TFlite. We need to
      // remove that restriction once we have support to attribute the main
      // tensorflow function in MLIR TF import using an entry_point attr.
      if (!llvm::is_contained(trim_funcs_allowlist_, "main") &&
          func.getName() == trim_funcs_allowlist_[0]) {
        func.setName(StringAttr::get(func.getContext(), "main"));
      }
    } else {
      funcs_to_trim.push_back(func);
    }
  }

  // remove all unexported functions from the module.
  for (auto func : funcs_to_trim) {
    func.erase();
  }
  return true;
}

// validate that all reachable functions from the remaining functions are
// also in the allowlist.
void TrimFunctionsPass::Verify() {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePStransformsPStrim_functions_tfDTcc mht_6(mht_6_v, 300, "", "./tensorflow/compiler/mlir/lite/transforms/trim_functions_tf.cc", "TrimFunctionsPass::Verify");

  // TODO(ashwinm): Instead, we should make sure that references to all
  // SymbolRefAttrs of all ops are present.
  SymbolTable symbol_table = SymbolTable(getOperation());
  llvm::SetVector<FuncOp> reachable_funcs;
  for (auto func : getOperation().getOps<FuncOp>()) {
    auto walk_result = func.walk([&](func::CallOp op) -> WalkResult {
      if (!symbol_table.lookup<FuncOp>(op.getCallee()))
        return getOperation().emitError()
               << func.getName() << " is not in the funcs allowlist";
      return WalkResult::advance();
    });
    if (walk_result.wasInterrupted()) return signalPassFailure();
  }
}

}  // namespace

// Creates an instance of the TensorFlow Lite dialect TrimFunctions
/// pass.
std::unique_ptr<OperationPass<ModuleOp>> CreateTrimFunctionsPass(
    llvm::ArrayRef<std::string> trim_funcs_allowlist) {
  return std::make_unique<TrimFunctionsPass>(trim_funcs_allowlist);
}

static PassRegistration<TrimFunctionsPass> pass;

}  // namespace TFL
}  // namespace mlir
