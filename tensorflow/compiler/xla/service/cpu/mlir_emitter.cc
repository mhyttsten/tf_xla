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
class MHTracer_DTPStensorflowPScompilerPSxlaPSservicePScpuPSmlir_emitterDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePScpuPSmlir_emitterDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSxlaPSservicePScpuPSmlir_emitterDTcc() {
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

#include "tensorflow/compiler/xla/service/cpu/mlir_emitter.h"

#include "llvm/Linker/Linker.h"
#include "llvm/Transforms/IPO/Internalize.h"
#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"  // from @llvm-project
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"  // from @llvm-project
#include "mlir/Conversion/VectorToLLVM/ConvertVectorToLLVM.h"  // from @llvm-project
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/Dialect/Linalg/Passes.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Pass/PassManager.h"  // from @llvm-project
#include "mlir/Target/LLVMIR/Export.h"  // from @llvm-project
#include "mlir/Transforms/Passes.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/xla/hlo_utils.h"

namespace xla {
namespace cpu {
namespace {

// Lower an MLIR module to an LLVM module.
std::unique_ptr<llvm::Module> MakeLLVMModule(
    mlir::OwningOpRef<mlir::ModuleOp> module, llvm::LLVMContext *context) {
  // When set, the LLVM backend will be allowed to reassociate floating-point
  // reductions, which enables much more efficient "horizontal" SIMD
  // implementations.
  // TODO(kramerb): link this to the right option, command line flag, etc.
  constexpr bool kReassociateFPReductions = true;

  mlir::PassManager manager(module->getContext(),
                            mlir::OpPassManager::Nesting::Implicit);
  manager.addPass(mlir::createConvertLinalgToLoopsPass());
  manager.addPass(mlir::createLowerAffinePass());
  manager.addPass(mlir::createConvertSCFToCFPass());
  manager.addPass(mlir::createConvertVectorToLLVMPass(
      mlir::LowerVectorToLLVMOptions().enableReassociateFPReductions(
          kReassociateFPReductions)));
  CHECK(succeeded(manager.run(*module)));
  return mlir::translateModuleToLLVMIR(*module, *context);
}

// Get arguments to pass a memref to an mlir function.
void BuildViewForBuffer(llvm::SmallVectorImpl<llvm::Value *> *args,
                        llvm::IRBuilder<> *b, const Shape &opShape,
                        llvm::Value *op_val) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePScpuPSmlir_emitterDTcc mht_0(mht_0_v, 229, "", "./tensorflow/compiler/xla/service/cpu/mlir_emitter.cc", "BuildViewForBuffer");

  llvm::Type *ty = op_val->getType();
  while (auto aty =
             llvm::dyn_cast<llvm::ArrayType>(ty->getPointerElementType())) {
    ty = aty->getElementType()->getPointerTo();
  }
  op_val = b->CreateBitCast(op_val, ty);

  args->push_back(op_val);          // Allocated pointer.
  args->push_back(op_val);          // Aligned pointer.
  args->push_back(b->getInt64(0));  // Offset.

  // Sizes.
  for (int64_t dim : opShape.dimensions()) {
    args->push_back(b->getInt64(dim));
  }

  int64_t accumulated_stride = 1;
  llvm::SmallVector<int64_t, 4> strides(opShape.rank(), 1);
  for (int64_t dim : LayoutUtil::MinorToMajor(opShape)) {
    strides[dim] = accumulated_stride;
    accumulated_stride *= opShape.dimensions(dim);
  }

  // Strides.
  for (int64_t stride : strides) {
    args->push_back(b->getInt64(stride));
  }
}
}  // namespace

Status EmitMlirFuncAndCall(
    mlir::MLIRContext *context, llvm::IRBuilder<> *b, const Shape &result_shape,
    llvm::ArrayRef<Shape> operand_shapes, llvm::Value *result_ptr,
    llvm::ArrayRef<llvm::Value *> operand_ptrs, llvm::StringRef func_name,
    llvm::function_ref<void(mlir::OpBuilder *, mlir::func::FuncOp)> emitter) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePScpuPSmlir_emitterDTcc mht_1(mht_1_v, 267, "", "./tensorflow/compiler/xla/service/cpu/mlir_emitter.cc", "EmitMlirFuncAndCall");

  llvm::Module *llvm_module = b->GetInsertBlock()->getParent()->getParent();
  mlir::Builder mlir_builder(context);

  // Get memref types for the inputs and output.
  TF_ASSIGN_OR_RETURN(mlir::Type ret_memref, ConvertTensorShapeToMemRefType(
                                                 result_shape, mlir_builder));
  std::vector<mlir::Type> operand_types = {ret_memref};
  for (int i = 0; i != operand_shapes.size(); ++i) {
    TF_ASSIGN_OR_RETURN(
        mlir::Type op_memref,
        ConvertTensorShapeToMemRefType(operand_shapes[i], mlir_builder));
    operand_types.push_back(op_memref);
  }

  // Create the function an call the emission callback.
  mlir::Location loc = mlir::UnknownLoc::get(context);
  auto function = mlir::func::FuncOp::create(
      loc, func_name, mlir::FunctionType::get(context, operand_types, {}));
  function.addEntryBlock();
  mlir::OwningOpRef<mlir::ModuleOp> mlir_module = mlir::ModuleOp::create(loc);
  mlir_module->push_back(function);
  mlir::OpBuilder op_builder(&function.getBody());
  emitter(&op_builder, function);

  // Now link it all into the main LLVM module.
  auto mlir_llvm_module =
      MakeLLVMModule(std::move(mlir_module), &b->getContext());
  mlir_llvm_module->setDataLayout(llvm_module->getDataLayout());
  llvm::Linker::linkModules(
      *llvm_module, std::move(mlir_llvm_module), llvm::Linker::None,
      [](llvm::Module &M, const llvm::StringSet<> &GVS) {
        llvm::internalizeModule(M, [&GVS](const llvm::GlobalValue &GV) {
          return !GV.hasName() || (GVS.count(GV.getName()) == 0);
        });
      });

  // And leave behind a call to the function generated by MLIR.
  llvm::Function *func = llvm_module->getFunction(func_name);
  llvm::SmallVector<llvm::Value *, 4> op_vals;
  BuildViewForBuffer(&op_vals, b, result_shape, result_ptr);
  for (int i = 0; i != operand_shapes.size(); ++i) {
    BuildViewForBuffer(&op_vals, b, operand_shapes[i], operand_ptrs[i]);
  }
  b->CreateCall(func, op_vals);

  return Status::OK();
}

}  // namespace cpu
}  // namespace xla
