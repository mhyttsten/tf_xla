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
class MHTracer_DTPStensorflowPScompilerPSmlirPSlitePStransformsPSanalyze_variablesDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePStransformsPSanalyze_variablesDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSmlirPSlitePStransformsPSanalyze_variablesDTcc() {
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

/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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
#include "mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/lite/ir/tfl_ops.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"

namespace mlir {
namespace TFL {
namespace {
// Attribute name to be added on the module to identify whether
// variables should be legalized to TFLite or not.
const char kLegalizeTflVariables[] = "tfl._legalize_tfl_variables";

// Returns true if 'op' is TF op that accepts resource type, but is
// supported by TFLite.
bool IsSupportedTFLiteResourceOp(Operation* op) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePStransformsPSanalyze_variablesDTcc mht_0(mht_0_v, 199, "", "./tensorflow/compiler/mlir/lite/transforms/analyze_variables.cc", "IsSupportedTFLiteResourceOp");

  return llvm::isa<TF::ReadVariableOp, TF::AssignVariableOp, TF::VarHandleOp,
                   TF::LookupTableFindV2Op, TF::LookupTableImportV2Op,
                   TF::LookupTableSizeV2Op>(op);
}

// Returns true if 'op' is TF/TFLite control flow op that can accept resource
// type. Usually these ops are just pass through, they call another subgraph and
// pass the operands to.
bool IsSupportedTFLiteControlFlow(Operation* op) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePStransformsPSanalyze_variablesDTcc mht_1(mht_1_v, 211, "", "./tensorflow/compiler/mlir/lite/transforms/analyze_variables.cc", "IsSupportedTFLiteControlFlow");

  return llvm::isa<TFL::WhileOp, TFL::IfOp, TFL::CallOnceOp>(op);
}

// Returns true if the 'op' is one of the supported TF control flow ops or
// dataset ops. Those ops just forward the operands to other subgraphs.
bool IsSupportedTFDataForwardingOp(Operation* op) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePStransformsPSanalyze_variablesDTcc mht_2(mht_2_v, 220, "", "./tensorflow/compiler/mlir/lite/transforms/analyze_variables.cc", "IsSupportedTFDataForwardingOp");

  return llvm::isa<TF::MapDatasetOp, TF::ReduceDatasetOp,
                   TF::TakeWhileDatasetOp, TF::IfOp, TF::WhileOp>(op);
}

}  // namespace

// Pass which analyzes the variables in the graph and add an attribute whether
// variables should be legalized to TFLite native ones.
// This pass needs to run post TF->TFL legalization and before variable
// legalization.
class AnalyzeVariablesPass
    : public PassWrapper<AnalyzeVariablesPass, OperationPass<ModuleOp>> {
 public:
  AnalyzeVariablesPass() = default;
  AnalyzeVariablesPass(const AnalyzeVariablesPass&) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePStransformsPSanalyze_variablesDTcc mht_3(mht_3_v, 238, "", "./tensorflow/compiler/mlir/lite/transforms/analyze_variables.cc", "AnalyzeVariablesPass");
}

  StringRef getArgument() const final {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePStransformsPSanalyze_variablesDTcc mht_4(mht_4_v, 243, "", "./tensorflow/compiler/mlir/lite/transforms/analyze_variables.cc", "getArgument");

    // This is the argument used to refer to the pass in
    // the textual format (on the commandline for example).
    return "tfl-analyze-variables-pass";
  }
  StringRef getDescription() const final {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePStransformsPSanalyze_variablesDTcc mht_5(mht_5_v, 251, "", "./tensorflow/compiler/mlir/lite/transforms/analyze_variables.cc", "getDescription");

    // This is a brief description of the pass.
    return "Analyze variables in the graph.";
  }

  void runOnOperation() override {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePStransformsPSanalyze_variablesDTcc mht_6(mht_6_v, 259, "", "./tensorflow/compiler/mlir/lite/transforms/analyze_variables.cc", "runOnOperation");

    auto* context = &getContext();
    auto module = getOperation();
    bool legalize_to_tfl = true;

    module.walk([&](Operation* op) {
      // Skip ops that are supported natively by TFLite.
      if (IsSupportedTFLiteResourceOp(op)) return WalkResult::advance();
      if (IsSupportedTFLiteControlFlow(op)) return WalkResult::advance();

      // Check for ops that are legalized to TFLite.
      if (op->getDialect()->getNamespace() == "tfl") {
        return WalkResult::advance();
      }
      // Check for ops that are not legalized to TFLite.
      if (IsSupportedTFDataForwardingOp(op)) {
        return WalkResult::advance();
      }

      // If any of the operands is a resource type, then we break
      // and mark the module as not valid for TFLite legalization.
      // Note: this might disable native variables in more than needed cases.
      // TODO(b/189370197): Enhance variable analysis.
      for (auto operand : op->getOperands()) {
        if (getElementTypeOrSelf(operand.getType()).isa<TF::ResourceType>()) {
          legalize_to_tfl = false;
          return WalkResult::interrupt();
        }
      }
      return WalkResult::advance();
    });
    module->setAttr(kLegalizeTflVariables,
                    BoolAttr::get(context, legalize_to_tfl));
  }
};

std::unique_ptr<OperationPass<ModuleOp>> CreateAnalyzeVariablesPass() {
  return std::make_unique<AnalyzeVariablesPass>();
}

static PassRegistration<AnalyzeVariablesPass> pass([] {
  return CreateAnalyzeVariablesPass();
});

}  // namespace TFL
}  // namespace mlir
