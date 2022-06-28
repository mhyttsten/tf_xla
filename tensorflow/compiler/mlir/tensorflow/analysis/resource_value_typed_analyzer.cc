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
class MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPSanalysisPSresource_value_typed_analyzerDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPSanalysisPSresource_value_typed_analyzerDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPSanalysisPSresource_value_typed_analyzerDTcc() {
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

#include "tensorflow/compiler/mlir/tensorflow/analysis/resource_value_typed_analyzer.h"

#include "llvm/Support/Casting.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/Transforms/RegionUtils.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_saved_model.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_types.h"

namespace mlir {
namespace TF {
namespace {
bool IsResourceType(Type type) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPSanalysisPSresource_value_typed_analyzerDTcc mht_0(mht_0_v, 198, "", "./tensorflow/compiler/mlir/tensorflow/analysis/resource_value_typed_analyzer.cc", "IsResourceType");

  if (auto tensor_type = type.dyn_cast<TensorType>()) {
    return tensor_type.getElementType().isa<TF::ResourceType>();
  }
  return false;
}

bool IsResource(Value value) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPSanalysisPSresource_value_typed_analyzerDTcc mht_1(mht_1_v, 208, "", "./tensorflow/compiler/mlir/tensorflow/analysis/resource_value_typed_analyzer.cc", "IsResource");
 return IsResourceType(value.getType()); }

// Helper that returns the FuncOp that is the SessionInit function which
// will be called to initialize all resources.
// Returns nullptr if no function is found.
FuncOp GetSessionInitializerFunc(ModuleOp module) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPSanalysisPSresource_value_typed_analyzerDTcc mht_2(mht_2_v, 216, "", "./tensorflow/compiler/mlir/tensorflow/analysis/resource_value_typed_analyzer.cc", "GetSessionInitializerFunc");

  auto session_init_op = tf_saved_model::GetSessionInitializerOp(module);
  if (session_init_op && !session_init_op.initializers().empty()) {
    SymbolTable symbol_table(module);
    FuncOp init_func_op = symbol_table.lookup<mlir::func::FuncOp>(
        session_init_op.initializers()[0].cast<FlatSymbolRefAttr>().getValue());
    return init_func_op;
  }
  return nullptr;
}

// Returns ID for identifying a resource.
std::tuple<llvm::StringRef, llvm::StringRef, llvm::StringRef> GetResourceKey(
    Operation* op) {
  llvm::StringRef device;
  if (auto attr = op->getAttrOfType<mlir::StringAttr>("device")) {
    device = attr.getValue();
  }

  llvm::StringRef container;
  if (auto attr = op->getAttrOfType<mlir::StringAttr>("container")) {
    container = attr.getValue();
  }

  llvm::StringRef shared_name;
  if (auto attr = op->getAttrOfType<mlir::StringAttr>("shared_name")) {
    shared_name = attr.getValue();
  }

  return std::tuple<llvm::StringRef, llvm::StringRef, llvm::StringRef>{
      device, container, shared_name};
}
}  // namespace
ResourceAnalyzer::ResourceAnalyzer(ModuleOp module, bool skip_session_init) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPSanalysisPSresource_value_typed_analyzerDTcc mht_3(mht_3_v, 252, "", "./tensorflow/compiler/mlir/tensorflow/analysis/resource_value_typed_analyzer.cc", "ResourceAnalyzer::ResourceAnalyzer");

  auto session_init_func = GetSessionInitializerFunc(module);
  for (auto func : module.getOps<FuncOp>()) {
    if (skip_session_init && func == session_init_func) continue;
    (void)AnalyzeRegion(func.getRegion());
  }
}

void ResourceAnalyzer::SetPotentiallyWritten(Value resource) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPSanalysisPSresource_value_typed_analyzerDTcc mht_4(mht_4_v, 263, "", "./tensorflow/compiler/mlir/tensorflow/analysis/resource_value_typed_analyzer.cc", "ResourceAnalyzer::SetPotentiallyWritten");

  assert(IsResource(resource));
  resource_infos_[resource].potentially_written = true;
  auto* operation = resource.getDefiningOp();
  if (operation && llvm::isa<TF::VarHandleOp>(operation)) {
    mutable_variables_.insert(GetResourceKey(operation));
  }
}

bool ResourceAnalyzer::IsPotentiallyWritten(Value resource) const {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPSanalysisPSresource_value_typed_analyzerDTcc mht_5(mht_5_v, 275, "", "./tensorflow/compiler/mlir/tensorflow/analysis/resource_value_typed_analyzer.cc", "ResourceAnalyzer::IsPotentiallyWritten");

  assert(IsResource(resource));
  auto* operation = resource.getDefiningOp();
  if (operation && llvm::isa<TF::VarHandleOp>(operation))
    return mutable_variables_.contains(GetResourceKey(operation));
  auto it = resource_infos_.find(resource);
  if (it == resource_infos_.end()) {
    return false;
  }
  return it->second.potentially_written;
}

// Analyze the specified region for resource mutating operations, namely
// TF::AssignVariableOp, if so, set the resource associated as "potentially
// written". Do this recursively across the chain of regions via call or
// control flow ops.
// TODO(ashwinm): Move to iterative traversal.
LogicalResult ResourceAnalyzer::AnalyzeRegion(Region& region) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPSanalysisPSresource_value_typed_analyzerDTcc mht_6(mht_6_v, 295, "", "./tensorflow/compiler/mlir/tensorflow/analysis/resource_value_typed_analyzer.cc", "ResourceAnalyzer::AnalyzeRegion");

  // Avoid infinite recursion.
  if (!discovered_.insert(&region).second) {
    return success();
  }

  region.walk([&](Operation* op) {
    if (isa<TF::ReadVariableOp, func::ReturnOp, YieldOp>(op)) {
      return;
    }
    if (auto assign_variable = dyn_cast<TF::AssignVariableOp>(op)) {
      SetPotentiallyWritten(assign_variable.resource());
      return;
    }
    if (auto call = dyn_cast<CallOpInterface>(op)) {
      if (auto func = dyn_cast<FuncOp>(call.resolveCallable())) {
        PropagatePotentiallyWrittenUpFromCallee(func.getRegion(),
                                                call.getArgOperands());
      }
      return;
    }
    if (auto if_op = dyn_cast<TF::IfOp>(op)) {
      for (auto callee : {if_op.then_function(), if_op.else_function()}) {
        PropagatePotentiallyWrittenUpFromCallee(callee.getRegion(),
                                                if_op.input());
      }
      return;
    }
    if (auto if_op = dyn_cast<TF::IfRegionOp>(op)) {
      PropagatePotentiallyWrittenUpFromCallee(if_op.then_branch(),
                                              if_op.getODSOperands(1));
      PropagatePotentiallyWrittenUpFromCallee(if_op.else_branch(),
                                              if_op.getODSOperands(1));
      return;
    }
    if (auto while_op = dyn_cast<TF::WhileOp>(op)) {
      for (auto callee : {while_op.cond_function(), while_op.body_function()}) {
        PropagatePotentiallyWrittenUpFromCallee(callee.getRegion(),
                                                while_op.input());
      }
      return;
    }
    if (auto while_op = dyn_cast<TF::WhileRegionOp>(op)) {
      PropagatePotentiallyWrittenUpFromCallee(while_op.cond(),
                                              while_op.input());
      PropagatePotentiallyWrittenUpFromCallee(while_op.body(),
                                              while_op.input());
      return;
    }
    // For all other ops, we assume it mutates all resources it uses, so
    // this errs on the side of being conservative. We should improve
    // this by using either a property or a trait that clearly
    // identifies ops with resource mutating behavior.
    PropagatePotentiallyWrittenWithinUnhandledOp(op);
  });
  return success();
}

void ResourceAnalyzer::PropagatePotentiallyWrittenWithinUnhandledOp(
    Operation* op) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPSanalysisPSresource_value_typed_analyzerDTcc mht_7(mht_7_v, 357, "", "./tensorflow/compiler/mlir/tensorflow/analysis/resource_value_typed_analyzer.cc", "ResourceAnalyzer::PropagatePotentiallyWrittenWithinUnhandledOp");

  for (auto operand : op->getOperands()) {
    if (IsResource(operand)) {
      SetPotentiallyWritten(operand);
    }
  }
  visitUsedValuesDefinedAbove(op->getRegions(), [&](OpOperand* operand) {
    if (IsResource(operand->get())) {
      SetPotentiallyWritten(operand->get());
    }
  });
}

void ResourceAnalyzer::PropagatePotentiallyWrittenUpFromCallee(
    Region& region, Operation::operand_range propagate_to) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPSanalysisPSresource_value_typed_analyzerDTcc mht_8(mht_8_v, 374, "", "./tensorflow/compiler/mlir/tensorflow/analysis/resource_value_typed_analyzer.cc", "ResourceAnalyzer::PropagatePotentiallyWrittenUpFromCallee");

  (void)AnalyzeRegion(region);
  for (auto t : llvm::zip(region.getArguments(), propagate_to)) {
    if (!IsResource(std::get<0>(t))) {
      continue;
    }
    if (IsPotentiallyWritten(std::get<0>(t))) {
      SetPotentiallyWritten(std::get<1>(t));
    }
  }
}
}  // namespace TF
}  // namespace mlir
