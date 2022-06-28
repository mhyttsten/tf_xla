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
class MHTracer_DTPStensorflowPScompilerPSmlirPStfrtPSanalysisPScompatibility_analysisDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSmlirPStfrtPSanalysisPScompatibility_analysisDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSmlirPStfrtPSanalysisPScompatibility_analysisDTcc() {
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

#include "tensorflow/compiler/mlir/tfrt/analysis/compatibility_analysis.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/Tools/mlir-translate/Translation.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/dialect_registration.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_executor.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_types.h"
#include "tensorflow/core/platform/protobuf.h"

namespace tensorflow {
namespace {

class CompatibilityAnalysis {
 public:
  void AnalyzeOperation(mlir::Operation* op);

  const mlir::tfrt::CompatibilityAnalysisProto& GetResult() const {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStfrtPSanalysisPScompatibility_analysisDTcc mht_0(mht_0_v, 205, "", "./tensorflow/compiler/mlir/tfrt/analysis/compatibility_analysis.cc", "GetResult");

    return analysis_;
  }

 private:
  // Return true if some attributes in the op are not supported.
  bool AnalyzeOpAttributes(mlir::Operation* op);
  // Return true if this op has unsupported operation (eg. mutate) on resource
  // variables.
  bool AnalyzeVariable(mlir::Operation* op);

  void UpdateReport(
      const mlir::tfrt::CompatibilityAnalysisReportProto& new_report,
      mlir::tfrt::CompatibilityAnalysisReportProto* old_report);

  mlir::tfrt::CompatibilityAnalysisProto analysis_;
};

void CompatibilityAnalysis::AnalyzeOperation(mlir::Operation* op) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStfrtPSanalysisPScompatibility_analysisDTcc mht_1(mht_1_v, 226, "", "./tensorflow/compiler/mlir/tfrt/analysis/compatibility_analysis.cc", "CompatibilityAnalysis::AnalyzeOperation");

  // Skip the standard ops that are allowed in tf dialect.
  if (llvm::isa<mlir::func::ReturnOp, mlir::func::FuncOp, mlir::ModuleOp>(op))
    return;

  auto op_name = op->getName();

  std::string name = op_name.getStringRef().str();

  mlir::tfrt::CompatibilityAnalysisReportProto op_report;

  if (op_name.getDialectNamespace() ==
      mlir::TF::TensorFlowDialect::getDialectNamespace()) {
    // Analyze op attributes.
    if (AnalyzeOpAttributes(op)) op_report.set_incompatible_attribute(true);

    // Analyze variable operations.
    if (AnalyzeVariable(op)) op_report.set_incompatible_variable(true);

    // Reference variable is not supported.
    if (op_name.getStringRef() == "tf.VariableV2")
      op_report.set_ref_variable(true);
  } else if (op_name.getDialectNamespace() == "tf_executor") {
    if (llvm::isa<mlir::tf_executor::SwitchOp, mlir::tf_executor::SwitchNOp,
                  mlir::tf_executor::MergeOp, mlir::tf_executor::EnterOp,
                  mlir::tf_executor::NextIterationSourceOp,
                  mlir::tf_executor::NextIterationSinkOp>(op)) {
      op_report.set_control_flow_v1(true);
    } else {
      // Skip the rest of the tf_executor ops as they can be handled.
      //
      // TODO(chky): consider adding allowlist here.
      return;
    }
  } else {
    // Mark unknown dialect in the report.
    op_report.set_unknown_dialect(true);
  }

  auto& op_info = (*analysis_.mutable_ops())[name];
  op_info.set_count(op_info.count() + 1);

  UpdateReport(op_report, op_info.mutable_report());
  UpdateReport(op_report, analysis_.mutable_summary());
}

bool CompatibilityAnalysis::AnalyzeOpAttributes(mlir::Operation* op) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStfrtPSanalysisPScompatibility_analysisDTcc mht_2(mht_2_v, 275, "", "./tensorflow/compiler/mlir/tfrt/analysis/compatibility_analysis.cc", "CompatibilityAnalysis::AnalyzeOpAttributes");

  // tf.Const gets special handling so it is always compatible.
  if (llvm::isa<mlir::TF::ConstOp>(op)) return false;

  // TODO(chky): Derived attributes should be also analyzed here.
  for (auto attr : op->getAttrs()) {
    if (attr.getName().strref() == "_output_shapes") continue;
    if (attr.getName().strref() == "_class") continue;

    // Symbol attributes (eg. function names) is currently not supported.
    //
    // TODO(chky): CoreRT should ideally support function call operatoins.
    // Remove this condition once that is implemented.
    if (attr.getValue().isa<mlir::FlatSymbolRefAttr>()) return true;

    // Currently only tensors of simple dtypes (i1, i32, i64, f32, f64) are
    // supported.
    if (auto elements_attr = attr.getValue().dyn_cast<mlir::ElementsAttr>()) {
      if (!elements_attr.isa<mlir::DenseElementsAttr>()) return true;
      auto element_type = elements_attr.getType().getElementType();
      if (element_type.isa<mlir::TF::TensorFlowType>()) return true;
    }

    // Currently only arrays of simple element types (i1, i32, i64, f32, f64)
    // are supported.
    if (auto array_attr = attr.getValue().dyn_cast<mlir::ArrayAttr>()) {
      if (!array_attr.empty()) {
        if (array_attr[0].isa<mlir::ElementsAttr>()) return true;

        if (array_attr[0].isa<mlir::StringAttr>()) return true;

        if (array_attr[0].isa<mlir::FlatSymbolRefAttr>()) return true;
      }
    }
  }
  return false;
}

bool CompatibilityAnalysis::AnalyzeVariable(mlir::Operation* op) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStfrtPSanalysisPScompatibility_analysisDTcc mht_3(mht_3_v, 316, "", "./tensorflow/compiler/mlir/tfrt/analysis/compatibility_analysis.cc", "CompatibilityAnalysis::AnalyzeVariable");

  // Currently only supported variable op is ReadVariableOp.
  if (llvm::isa<mlir::TF::ReadVariableOp>(op)) return false;

  for (auto value : op->getOperands()) {
    auto type = value.getType();
    if (auto tensor_type = type.dyn_cast<mlir::TensorType>()) {
      auto element_type = tensor_type.getElementType();
      if (element_type.isa<mlir::TF::ResourceType>()) return true;
    }
  }

  return false;
}

void CompatibilityAnalysis::UpdateReport(
    const mlir::tfrt::CompatibilityAnalysisReportProto& new_report,
    mlir::tfrt::CompatibilityAnalysisReportProto* old_report) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStfrtPSanalysisPScompatibility_analysisDTcc mht_4(mht_4_v, 336, "", "./tensorflow/compiler/mlir/tfrt/analysis/compatibility_analysis.cc", "CompatibilityAnalysis::UpdateReport");

  if (new_report.unknown_dialect()) old_report->set_unknown_dialect(true);

  if (new_report.ref_variable()) old_report->set_ref_variable(true);

  if (new_report.incompatible_variable())
    old_report->set_incompatible_variable(true);

  if (new_report.incompatible_attribute())
    old_report->set_incompatible_attribute(true);

  if (new_report.control_flow_v1()) old_report->set_control_flow_v1(true);
}

}  // namespace

mlir::tfrt::CompatibilityAnalysisProto AnalyzeTFCompatibility(
    mlir::ModuleOp op) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStfrtPSanalysisPScompatibility_analysisDTcc mht_5(mht_5_v, 356, "", "./tensorflow/compiler/mlir/tfrt/analysis/compatibility_analysis.cc", "AnalyzeTFCompatibility");

  CompatibilityAnalysis analysis;
  op.walk([&analysis](mlir::Operation* op) { analysis.AnalyzeOperation(op); });
  return analysis.GetResult();
}

static mlir::TranslateFromMLIRRegistration registration(
    "analyze-tf-for-tfrt",
    [](mlir::ModuleOp op, llvm::raw_ostream& output) {
      auto analysis_proto = AnalyzeTFCompatibility(op);
      std::string text_proto;
      if (tensorflow::protobuf::TextFormat::PrintToString(analysis_proto,
                                                          &text_proto)) {
        output << text_proto;
        return mlir::success();
      }

      return mlir::failure();
    },
    [](mlir::DialectRegistry& registry) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStfrtPSanalysisPScompatibility_analysisDTcc mht_6(mht_6_v, 378, "", "./tensorflow/compiler/mlir/tfrt/analysis/compatibility_analysis.cc", "lambda");

      mlir::RegisterAllTensorFlowDialects(registry);
    });

}  // namespace tensorflow
