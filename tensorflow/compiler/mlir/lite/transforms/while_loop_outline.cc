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
class MHTracer_DTPStensorflowPScompilerPSmlirPSlitePStransformsPSwhile_loop_outlineDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePStransformsPSwhile_loop_outlineDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSmlirPSlitePStransformsPSwhile_loop_outlineDTcc() {
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

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/Support/CommandLine.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/IR/Location.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/IR/Matchers.h"  // from @llvm-project
#include "mlir/IR/SymbolTable.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Transforms/RegionUtils.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/lite/ir/tfl_ops.h"
#include "tensorflow/compiler/mlir/lite/transforms/passes.h"
#include "tensorflow/compiler/mlir/op_or_arg_name_mapper.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"

namespace mlir {
namespace TFL {
namespace {

// This pass outlines the cond/body region of the TFL WhileOp into functions and
// replaces the regions with calls to these outlined functions.
class WhileOutlinePass
    : public mlir::PassWrapper<WhileOutlinePass, OperationPass<ModuleOp>> {
  void getDependentDialects(DialectRegistry& registry) const override {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePStransformsPSwhile_loop_outlineDTcc mht_0(mht_0_v, 212, "", "./tensorflow/compiler/mlir/lite/transforms/while_loop_outline.cc", "getDependentDialects");

    registry.insert<TF::TensorFlowDialect>();
  }

 public:
  explicit WhileOutlinePass() {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePStransformsPSwhile_loop_outlineDTcc mht_1(mht_1_v, 220, "", "./tensorflow/compiler/mlir/lite/transforms/while_loop_outline.cc", "WhileOutlinePass");
}

  StringRef getArgument() const final {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePStransformsPSwhile_loop_outlineDTcc mht_2(mht_2_v, 225, "", "./tensorflow/compiler/mlir/lite/transforms/while_loop_outline.cc", "getArgument");

    // This is the argument used to refer to the pass in
    // the textual format (on the commandline for example).
    return "tfl-while-loop-outline";
  }
  StringRef getDescription() const final {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePStransformsPSwhile_loop_outlineDTcc mht_3(mht_3_v, 233, "", "./tensorflow/compiler/mlir/lite/transforms/while_loop_outline.cc", "getDescription");

    // This is a brief description of the pass.
    return "Hoist while op regions into functions";
  }

 private:
  void runOnOperation() override;

  // Outlines the regions of the WhileOp's cond and body and insert function
  // calls instead,
  void OutlineWhile(WhileOp while_op);

  // Get unique name by using the loc to name mapping.
  std::string GetName(Operation* op, StringRef suffix);

  tensorflow::OpOrArgLocNameMapper mapper_;
};

std::string WhileOutlinePass::GetName(Operation* op, StringRef suffix) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePStransformsPSwhile_loop_outlineDTcc mht_4(mht_4_v, 254, "", "./tensorflow/compiler/mlir/lite/transforms/while_loop_outline.cc", "WhileOutlinePass::GetName");

  return (mapper_.GetUniqueName(op) + suffix).str();
}

// Returns whether the WhileOp is already outlined (e.g., only consists of calls
// to functions).
bool IsAlreadyOutlined(WhileOp while_op) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePStransformsPSwhile_loop_outlineDTcc mht_5(mht_5_v, 263, "", "./tensorflow/compiler/mlir/lite/transforms/while_loop_outline.cc", "IsAlreadyOutlined");

  auto just_call = [](Region& region) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePStransformsPSwhile_loop_outlineDTcc mht_6(mht_6_v, 267, "", "./tensorflow/compiler/mlir/lite/transforms/while_loop_outline.cc", "lambda");

    auto it = region.front().begin();
    if (!isa<func::CallOp>(*it)) return false;
    ++it;
    if (!isa<YieldOp>(*it)) return false;
    return true;
  };
  return just_call(while_op.body()) && just_call(while_op.cond());
}

bool IsCompatibleTypeWithTFLCastOp(Type type) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePStransformsPSwhile_loop_outlineDTcc mht_7(mht_7_v, 280, "", "./tensorflow/compiler/mlir/lite/transforms/while_loop_outline.cc", "IsCompatibleTypeWithTFLCastOp");

  auto elemType = getElementTypeOrSelf(type);
  // F32 and BF16 types are allowed.
  if (elemType.isBF16() || elemType.isF32()) return true;

  // I1, I8 I16, I32, I64 types are allowed.
  if (elemType.isInteger(1) || elemType.isInteger(8) ||
      elemType.isInteger(16) || elemType.isInteger(32) ||
      elemType.isInteger(64))
    return true;

  // Complex<F<32>> is allowed.
  if (elemType.isa<ComplexType>() &&
      elemType.cast<ComplexType>().getElementType().isF32())
    return true;

  // QUINT8 and UI8 are allowed.
  if (elemType.isa<TF::Quint8Type>() ||
      (elemType.isInteger(8) && elemType.cast<IntegerType>().isUnsigned()))
    return true;

  return false;
}

FuncOp CreateOutlineFunc(StringRef name, Region& region,
                         bool passthru_extra_args, int num_loop_carried,
                         const llvm::SetVector<Value>& extern_values,
                         const SmallVectorImpl<Type>& types, Location loc) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePStransformsPSwhile_loop_outlineDTcc mht_8(mht_8_v, 310, "", "./tensorflow/compiler/mlir/lite/transforms/while_loop_outline.cc", "CreateOutlineFunc");

  MLIRContext* context = loc.getContext();
  OpBuilder builder(context);
  FunctionType type;
  if (passthru_extra_args) {
    type = FunctionType::get(context, types, types);
  } else {
    SmallVector<Type, 4> result_types;
    auto operands = region.front().getTerminator()->getOperandTypes();
    result_types.append(operands.begin(), operands.end());
    type = FunctionType::get(context, types, result_types);
  }

  auto outlined_func = builder.create<FuncOp>(loc, name, type);
  outlined_func.getBody().takeBody(region);
  Region& func_region = outlined_func.getBody();

  // Replace all external uses with block args and update uses.
  llvm::SmallVector<Value, 4> new_args;
  new_args.reserve(extern_values.size());
  Block& block = func_region.front();
  for (Value value : extern_values) {
    auto arg = block.addArgument(value.getType(), loc);
    replaceAllUsesInRegionWith(value, arg, func_region);
    new_args.push_back(arg);
  }

  // Replace yield op with return.
  Operation* yield_op = outlined_func.getBody().front().getTerminator();
  OpBuilder b(yield_op);
  llvm::SmallVector<Value, 4> args;
  auto loop_carried_yield_operands =
      yield_op->getOperands().take_front(num_loop_carried);
  args.reserve(loop_carried_yield_operands.size() + new_args.size());
  if (passthru_extra_args) {
    // Add operands of yield to the return, inserting casts if needed.
    for (auto it : llvm::zip_first(loop_carried_yield_operands, types)) {
      auto value = std::get<0>(it);
      auto type = std::get<1>(it);
      if (value.getType() == type) {
        args.push_back(value);
      } else {
        if (IsCompatibleTypeWithTFLCastOp(value.getType()) &&
            IsCompatibleTypeWithTFLCastOp(type)) {
          auto cast = b.create<CastOp>(yield_op->getLoc(), type, value);
          args.push_back(cast);
        } else {
          auto cast = b.create<TF::CastOp>(yield_op->getLoc(), type, value);
          args.push_back(cast);
        }
      }
    }
    args.append(new_args.begin(), new_args.end());
  } else {
    args.append(yield_op->operand_begin(), yield_op->operand_end());
  }
  b.create<func::ReturnOp>(yield_op->getLoc(), args);
  yield_op->erase();
  SymbolTable(region.getParentOfType<ModuleOp>()).insert(outlined_func);
  outlined_func.setPrivate();
  return outlined_func;
}

// Replace region with call to outline function.
void ReplaceRegionWithCall(StringRef name, Region& region,
                           bool passthru_extra_args, int num_loop_carried,
                           const llvm::SetVector<Value>& extern_values,
                           const SmallVectorImpl<Type>& types, Location loc) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePStransformsPSwhile_loop_outlineDTcc mht_9(mht_9_v, 380, "", "./tensorflow/compiler/mlir/lite/transforms/while_loop_outline.cc", "ReplaceRegionWithCall");

  auto func = CreateOutlineFunc(name, region, passthru_extra_args,
                                num_loop_carried, extern_values, types, loc);
  OpBuilder b(region);
  // The body of the region is empty/has been outlined into the function.
  auto block = b.createBlock(&region);
  SmallVector<Value, 4> new_operands;
  new_operands.reserve(types.size());
  for (Type t : llvm::makeArrayRef(types).drop_back(extern_values.size()))
    new_operands.push_back(block->addArgument(t, loc));
  for (Value v : extern_values) new_operands.push_back(v);
  auto call = b.create<func::CallOp>(loc, func, new_operands);
  b.create<YieldOp>(loc, call.getResults());
}

void WhileOutlinePass::OutlineWhile(WhileOp while_op) {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePStransformsPSwhile_loop_outlineDTcc mht_10(mht_10_v, 398, "", "./tensorflow/compiler/mlir/lite/transforms/while_loop_outline.cc", "WhileOutlinePass::OutlineWhile");

  OpBuilder builder(&getContext());
  // Collect external values used.
  llvm::SetVector<Value> extern_values;

  // The basic block arguments correspond to values that are loop carried, while
  // all those post are loop independent. Initialize extern_values with while_op
  // not loop carried operands.
  auto num_loop_carried = while_op.cond().getNumArguments();
  auto not_carried_operands =
      while_op.getOperands().drop_front(num_loop_carried);
  extern_values.insert(not_carried_operands.begin(),
                       not_carried_operands.end());
  auto old_extern_values_size = extern_values.size();

  llvm::SmallVector<Region*, 2> regions{&while_op.cond(), &while_op.body()};
  for (auto it : llvm::enumerate(regions)) {
    llvm::SetVector<Value> region_extern_values;
    getUsedValuesDefinedAbove(*it.value(), region_extern_values);

    // Sink down constants into the functions.
    for (auto extern_value : region_extern_values) {
      if (!matchPattern(extern_value, m_Constant())) {
        extern_values.insert(extern_value);
        continue;
      }
      // Add constant at start of region.
      auto const_builder =
          OpBuilder(&it.value()->front(), it.value()->front().begin());
      auto const_value = const_builder.clone(*extern_value.getDefiningOp());
      replaceAllUsesInRegionWith(extern_value, const_value->getResult(0),
                                 *it.value());
    }
  }

  bool has_extra_extern_values = old_extern_values_size != extern_values.size();
  // If an extern value is already an operand post the loop carried operands,
  // then it need not be passed in again.
  // Compute all the extra operands that have to be added to the while.
  llvm::SetVector<Value> extra_operands;
  if (has_extra_extern_values) {
    auto new_extern =
        extern_values.getArrayRef().drop_front(old_extern_values_size);
    extra_operands.insert(new_extern.begin(), new_extern.end());
  }

  // Skip if already just calls.
  if (extra_operands.empty() && IsAlreadyOutlined(while_op)) return;

  // Collect new types.
  SmallVector<Type, 4> types;
  types.reserve(extra_operands.size() + while_op.getNumOperands());
  for (Type type : while_op.cond().getArgumentTypes()) types.push_back(type);
  for (Value operand : extern_values) types.push_back(operand.getType());

  // Create outline function from region. Optional pass extra arguments through
  // to yield.
  ReplaceRegionWithCall(GetName(while_op.getOperation(), "_cond"),
                        while_op.cond(), false, num_loop_carried, extern_values,
                        types, while_op.getLoc());
  ReplaceRegionWithCall(GetName(while_op.getOperation(), "_body"),
                        while_op.body(), true, num_loop_carried, extern_values,
                        types, while_op.getLoc());

  // If there are extern values used then the result type of the while has to
  // change, so replace with new while op.
  if (extra_operands.empty()) return;

  const int operands_size = while_op.getNumOperands() + extra_operands.size();
  SmallVector<Value, 4> operands;
  operands.reserve(operands_size);
  operands.append(while_op.getOperands().begin(), while_op.getOperands().end());
  operands.append(extra_operands.begin(), extra_operands.end());
  SmallVector<Type, 4> new_types;
  new_types.reserve(operands_size);
  new_types.append(while_op.getResultTypes().begin(),
                   while_op.getResultTypes().end());
  for (auto extra_operand : extra_operands)
    new_types.push_back(extra_operand.getType());

  auto new_while_op = OpBuilder(while_op).create<WhileOp>(
      while_op.getLoc(), new_types, operands, while_op->getAttrs());
  new_while_op.cond().takeBody(while_op.cond());
  new_while_op.body().takeBody(while_op.body());
  while_op.replaceAllUsesWith(
      new_while_op.getResults().take_front(while_op.getNumResults()));
  while_op.erase();
}

void WhileOutlinePass::runOnOperation() {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePStransformsPSwhile_loop_outlineDTcc mht_11(mht_11_v, 490, "", "./tensorflow/compiler/mlir/lite/transforms/while_loop_outline.cc", "WhileOutlinePass::runOnOperation");

  getOperation().walk(
      [&](mlir::TFL::WhileOp while_op) { OutlineWhile(while_op); });
}
}  // namespace

// Creates an instance of the TensorFlow Lite dialect WhileOp outline pass.
std::unique_ptr<OperationPass<ModuleOp>> CreateWhileOutlinePass() {
  return std::make_unique<WhileOutlinePass>();
}

static PassRegistration<WhileOutlinePass> pass;

}  // namespace TFL
}  // namespace mlir
