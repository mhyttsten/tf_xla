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
class MHTracer_DTPStensorflowPScompilerPSmlirPStoolsPSkernel_genPStransformsPSembed_memref_printsDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSmlirPStoolsPSkernel_genPStransformsPSembed_memref_printsDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSmlirPStoolsPSkernel_genPStransformsPSembed_memref_printsDTcc() {
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

#include <memory>
#include <string>

#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/Dialect/Linalg/IR/Linalg.h"  // from @llvm-project
#include "mlir/Dialect/MemRef/IR/MemRef.h"  // from @llvm-project
#include "mlir/Dialect/SCF/SCF.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/hlo/include/mlir-hlo/Dialect/gml_st/IR/gml_st_ops.h"
#include "tensorflow/compiler/mlir/tools/kernel_gen/transforms/passes.h"
#include "tensorflow/compiler/mlir/tools/kernel_gen/transforms/utils.h"

namespace mlir {
namespace kernel_gen {
namespace transforms {
namespace {

constexpr StringRef kPrintStringFuncName = "print_c_string";

#define GEN_PASS_CLASSES
#include "tensorflow/compiler/mlir/tools/kernel_gen/transforms/kernel_gen_passes.h.inc"

Operation* EmitMemRefPrint(Location loc, Type element_type, Value arg,
                           OpBuilder* b) {
  StringRef func_name;
  if (element_type.isF32()) {
    func_name = "print_memref_f32";
  }
  if (element_type.isF64()) {
    func_name = "print_memref_f64";
  }
  if (element_type.isInteger(32)) {
    func_name = "print_memref_i32";
  }
  if (element_type.isInteger(64) || element_type.isIndex()) {
    func_name = "print_memref_i64";
  }
  assert(!func_name.empty() &&
         "Did not find a print function for the element type");

  auto caller_func =
      b->getInsertionBlock()->getParent()->getParentOfType<FuncOp>();
  auto func_name_attr = b->getStringAttr(func_name);

  auto callee_func =
      SymbolTable::lookupNearestSymbolFrom<FuncOp>(caller_func, func_name_attr);
  if (!callee_func) {
    OpBuilder::InsertionGuard insertGuard(*b);

    auto module = caller_func->getParentOfType<ModuleOp>();
    b->setInsertionPointToStart(module.getBody());
    auto func_type = FunctionType::get(b->getContext(), arg.getType(),
                                       /*results=*/llvm::None);
    callee_func = b->create<FuncOp>(module.getLoc(), func_name, func_type);
    callee_func.setPrivate();
  }
  return b->create<func::CallOp>(loc, callee_func, arg);
}

bool IsElementTypePrintalble(Type element_type) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStoolsPSkernel_genPStransformsPSembed_memref_printsDTcc mht_0(mht_0_v, 243, "", "./tensorflow/compiler/mlir/tools/kernel_gen/transforms/embed_memref_prints.cc", "IsElementTypePrintalble");

  return element_type.isF32() || element_type.isF64() ||
         element_type.isInteger(32) || element_type.isInteger(64) ||
         element_type.isIndex();
}

void EmitMemRefPrint(Location loc, Value memref, OpBuilder* b) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStoolsPSkernel_genPStransformsPSembed_memref_printsDTcc mht_1(mht_1_v, 252, "", "./tensorflow/compiler/mlir/tools/kernel_gen/transforms/embed_memref_prints.cc", "EmitMemRefPrint");

  auto memref_type = memref.getType();
  if (auto unranked_type = memref_type.dyn_cast<UnrankedMemRefType>()) {
    Type element_type = unranked_type.getElementType();
    if (!IsElementTypePrintalble(element_type)) return;

    EmitMemRefPrint(loc, element_type, memref, b);
  }
  if (auto ranked_type = memref_type.dyn_cast<MemRefType>()) {
    Type element_type = ranked_type.getElementType();
    if (!IsElementTypePrintalble(element_type)) return;

    if (element_type.isIndex()) {
      element_type = b->getI64Type();
      ranked_type = MemRefType::get(ranked_type.getShape(), element_type,
                                    ranked_type.getLayout(),
                                    ranked_type.getMemorySpace());
      memref = b->create<arith::IndexCastOp>(loc, ranked_type, memref);
    }

    auto unranked_type = UnrankedMemRefType::get(
        element_type, ranked_type.getMemorySpaceAsInt());
    Value unranked_memref =
        b->create<memref::CastOp>(loc, unranked_type, memref);
    EmitMemRefPrint(loc, element_type, unranked_memref, b);
  }
}

SmallVector<Value> ExtractValuesToPrint(Operation* op) {
  if (isa<memref::ReinterpretCastOp>(op) || isa<memref::ReshapeOp>(op) ||
      isa<memref::ExpandShapeOp>(op) || isa<memref::CollapseShapeOp>(op)) {
    return {op->getResult(0)};
  }
  if (auto linalg = dyn_cast<linalg::LinalgOp>(op)) {
    return linalg.getOutputBufferOperands();
  }
  if (auto loop = dyn_cast<gml_st::LoopOp>(op)) {
    return loop.outputs();
  }
  if (auto loop = dyn_cast<scf::ForOp>(op)) {
    return loop.getIterOperands();
  }
  if (auto copy = dyn_cast<memref::CopyOp>(op)) {
    return {copy.target()};
  }
  return {};
}

void EmitOperationPrint(Operation* op, OpBuilder* b) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStoolsPSkernel_genPStransformsPSembed_memref_printsDTcc mht_2(mht_2_v, 303, "", "./tensorflow/compiler/mlir/tools/kernel_gen/transforms/embed_memref_prints.cc", "EmitOperationPrint");

  std::string debug_str = "\n\nPrint memref content after the following op\n";
  llvm::raw_string_ostream output_stream(debug_str);

  mlir::OpPrintingFlags flags;
  op->print(output_stream, flags);
  output_stream << "\n\n";

  Location loc = op->getLoc();
  Value message_constant = CreateOrFindGlobalStringConstant(
      loc, GetGlobalName("debug_op", debug_str), debug_str, b);

  // Insert function call.
  MLIRContext* ctx = op->getContext();
  auto func_type = LLVM::LLVMFunctionType::get(
      LLVM::LLVMVoidType::get(op->getContext()),
      {LLVM::LLVMPointerType::get(IntegerType::get(ctx, 8))});
  FlatSymbolRefAttr tf_func_ref =
      GetOrInsertLLVMFunction(kPrintStringFuncName, func_type, op, b);
  b->create<LLVM::CallOp>(loc, llvm::None, tf_func_ref,
                          llvm::makeArrayRef({message_constant}));
}

// The pass inserts printing on every mutation of memrefs.
struct EmbedMemRefPrintsPass
    : public EmbedMemRefPrintsPassBase<EmbedMemRefPrintsPass> {
  void runOnOperation() override {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStoolsPSkernel_genPStransformsPSembed_memref_printsDTcc mht_3(mht_3_v, 332, "", "./tensorflow/compiler/mlir/tools/kernel_gen/transforms/embed_memref_prints.cc", "runOnOperation");

    ModuleOp module = getOperation();
    module.walk([&](FuncOp func) {
      if (func.isDeclaration()) return;
      Block* body = &func.getBody().front();

      // Print arguments.
      OpBuilder b(&getContext());
      b.setInsertionPointToStart(body);
      Location loc = func.getLoc();
      auto args = func.getArguments();
      if (!args.empty()) {
        EmitOperationPrint(func, &b);
      }
      for (auto arg : args) {
        EmitMemRefPrint(loc, arg, &b);
      }
      // Print buffers after every change.
      for (auto& op : func.getBody().front().getOperations()) {
        b.setInsertionPointAfter(&op);
        auto memrefs = ExtractValuesToPrint(&op);
        if (!memrefs.empty()) {
          EmitOperationPrint(&op, &b);
        }
        for (auto memref : memrefs) {
          EmitMemRefPrint(op.getLoc(), memref, &b);
        }
      }
    });
  }
};

}  // namespace

std::unique_ptr<OperationPass<ModuleOp>> CreateEmbedMemRefPrintsPass() {
  return std::make_unique<EmbedMemRefPrintsPass>();
}

}  // namespace transforms
}  // namespace kernel_gen
}  // namespace mlir
