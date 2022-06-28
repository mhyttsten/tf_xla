/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_LLVM_IR_IR_BUILDER_MIXIN_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_LLVM_IR_IR_BUILDER_MIXIN_H_
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
class MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSllvm_irPSir_builder_mixinDTh {
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
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSllvm_irPSir_builder_mixinDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSllvm_irPSir_builder_mixinDTh() {
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


#include "llvm/IR/IRBuilder.h"

namespace xla {

// Mixin class that injects more ergonomic versions of llvm::IRBuilder methods
// into a class.  Intended to be used as a CRTP base class, like:
//
//  class MyIrEmitter : public IrBuilderMixin<MyIrEmitter> {
//    llvm::IRBuilder<>* builder() { return builder_; }
//
//    void EmitFoo(HloInstruction* foo) {
//      Add(Mul(...), FPToUI(...));
//    }
//  };

template <typename Derived>
class IrBuilderMixin {
 protected:
  template <class... Args>
  llvm::Value* Add(Args&&... args) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSllvm_irPSir_builder_mixinDTh mht_0(mht_0_v, 207, "", "./tensorflow/compiler/xla/service/llvm_ir/ir_builder_mixin.h", "Add");

    return mixin_builder()->CreateAdd(std::forward<Args>(args)...);
  }

  template <class... Args>
  llvm::LoadInst* AlignedLoad(llvm::Type* type, Args&&... args) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSllvm_irPSir_builder_mixinDTh mht_1(mht_1_v, 215, "", "./tensorflow/compiler/xla/service/llvm_ir/ir_builder_mixin.h", "AlignedLoad");

    return mixin_builder()->CreateAlignedLoad(type,
                                              std::forward<Args>(args)...);
  }

  template <class... Args>
  llvm::LoadInst* AlignedLoad(llvm::Value* value, Args&&... args) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSllvm_irPSir_builder_mixinDTh mht_2(mht_2_v, 224, "", "./tensorflow/compiler/xla/service/llvm_ir/ir_builder_mixin.h", "AlignedLoad");

    // LLVM has deprecated CreateAlignedLoad without a type argument. Provide it
    // for convenience.
    return mixin_builder()->CreateAlignedLoad(
        value->getType()->getPointerElementType(), value,
        std::forward<Args>(args)...);
  }

  template <class... Args>
  llvm::StoreInst* AlignedStore(Args&&... args) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSllvm_irPSir_builder_mixinDTh mht_3(mht_3_v, 236, "", "./tensorflow/compiler/xla/service/llvm_ir/ir_builder_mixin.h", "AlignedStore");

    return mixin_builder()->CreateAlignedStore(std::forward<Args>(args)...);
  }

  template <class... Args>
  llvm::AllocaInst* Alloca(Args&&... args) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSllvm_irPSir_builder_mixinDTh mht_4(mht_4_v, 244, "", "./tensorflow/compiler/xla/service/llvm_ir/ir_builder_mixin.h", "Alloca");

    return mixin_builder()->CreateAlloca(std::forward<Args>(args)...);
  }

  template <class... Args>
  llvm::Value* And(Args&&... args) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSllvm_irPSir_builder_mixinDTh mht_5(mht_5_v, 252, "", "./tensorflow/compiler/xla/service/llvm_ir/ir_builder_mixin.h", "And");

    return mixin_builder()->CreateAnd(std::forward<Args>(args)...);
  }

  template <class... Args>
  llvm::Value* AtomicCmpXchg(Args&&... args) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSllvm_irPSir_builder_mixinDTh mht_6(mht_6_v, 260, "", "./tensorflow/compiler/xla/service/llvm_ir/ir_builder_mixin.h", "AtomicCmpXchg");

    return mixin_builder()->CreateAtomicCmpXchg(std::forward<Args>(args)...);
  }

  template <class... Args>
  llvm::Value* AtomicRMW(Args&&... args) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSllvm_irPSir_builder_mixinDTh mht_7(mht_7_v, 268, "", "./tensorflow/compiler/xla/service/llvm_ir/ir_builder_mixin.h", "AtomicRMW");

    return mixin_builder()->CreateAtomicRMW(std::forward<Args>(args)...);
  }

  template <class... Args>
  llvm::Value* BitCast(Args&&... args) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSllvm_irPSir_builder_mixinDTh mht_8(mht_8_v, 276, "", "./tensorflow/compiler/xla/service/llvm_ir/ir_builder_mixin.h", "BitCast");

    return mixin_builder()->CreateBitCast(std::forward<Args>(args)...);
  }

  template <class... Args>
  llvm::Value* Br(Args&&... args) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSllvm_irPSir_builder_mixinDTh mht_9(mht_9_v, 284, "", "./tensorflow/compiler/xla/service/llvm_ir/ir_builder_mixin.h", "Br");

    return mixin_builder()->CreateBr(std::forward<Args>(args)...);
  }

  llvm::CallInst* Call(llvm::FunctionCallee func_callee,
                       llvm::ArrayRef<llvm::Value*> args = llvm::None,
                       const llvm::Twine& name = "",
                       llvm::MDNode* fp_math_tag = nullptr) {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSllvm_irPSir_builder_mixinDTh mht_10(mht_10_v, 294, "", "./tensorflow/compiler/xla/service/llvm_ir/ir_builder_mixin.h", "Call");

    return mixin_builder()->CreateCall(func_callee, args, name, fp_math_tag);
  }

  llvm::CallInst* Call(llvm::FunctionType* func_type, llvm::Value* callee,
                       llvm::ArrayRef<llvm::Value*> args = llvm::None,
                       const llvm::Twine& name = "",
                       llvm::MDNode* fp_math_tag = nullptr) {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSllvm_irPSir_builder_mixinDTh mht_11(mht_11_v, 304, "", "./tensorflow/compiler/xla/service/llvm_ir/ir_builder_mixin.h", "Call");

    return mixin_builder()->CreateCall(func_type, callee, args, name,
                                       fp_math_tag);
  }

  template <class... Args>
  llvm::BranchInst* CondBr(Args&&... args) {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSllvm_irPSir_builder_mixinDTh mht_12(mht_12_v, 313, "", "./tensorflow/compiler/xla/service/llvm_ir/ir_builder_mixin.h", "CondBr");

    return mixin_builder()->CreateCondBr(std::forward<Args>(args)...);
  }

  template <class... Args>
  llvm::Value* ConstInBoundsGEP1_32(Args&&... args) {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSllvm_irPSir_builder_mixinDTh mht_13(mht_13_v, 321, "", "./tensorflow/compiler/xla/service/llvm_ir/ir_builder_mixin.h", "ConstInBoundsGEP1_32");

    return mixin_builder()->CreateConstInBoundsGEP1_32(
        std::forward<Args>(args)...);
  }

  template <class... Args>
  llvm::Value* FAdd(Args&&... args) {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSllvm_irPSir_builder_mixinDTh mht_14(mht_14_v, 330, "", "./tensorflow/compiler/xla/service/llvm_ir/ir_builder_mixin.h", "FAdd");

    return mixin_builder()->CreateFAdd(std::forward<Args>(args)...);
  }

  template <class... Args>
  llvm::Value* FMul(Args&&... args) {
   std::vector<std::string> mht_15_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSllvm_irPSir_builder_mixinDTh mht_15(mht_15_v, 338, "", "./tensorflow/compiler/xla/service/llvm_ir/ir_builder_mixin.h", "FMul");

    return mixin_builder()->CreateFMul(std::forward<Args>(args)...);
  }

  llvm::Value* GEP(llvm::Value* ptr, llvm::ArrayRef<llvm::Value*> idx_list,
                   const llvm::Twine& name = "") {
   std::vector<std::string> mht_16_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSllvm_irPSir_builder_mixinDTh mht_16(mht_16_v, 346, "", "./tensorflow/compiler/xla/service/llvm_ir/ir_builder_mixin.h", "GEP");

    return mixin_builder()->CreateGEP(ptr->getType()->getPointerElementType(),
                                      ptr, idx_list, name);
  }

  template <class... Args>
  llvm::Value* ICmpEQ(Args&&... args) {
   std::vector<std::string> mht_17_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSllvm_irPSir_builder_mixinDTh mht_17(mht_17_v, 355, "", "./tensorflow/compiler/xla/service/llvm_ir/ir_builder_mixin.h", "ICmpEQ");

    return mixin_builder()->CreateICmpEQ(std::forward<Args>(args)...);
  }

  template <class... Args>
  llvm::Value* ICmpNE(Args&&... args) {
   std::vector<std::string> mht_18_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSllvm_irPSir_builder_mixinDTh mht_18(mht_18_v, 363, "", "./tensorflow/compiler/xla/service/llvm_ir/ir_builder_mixin.h", "ICmpNE");

    return mixin_builder()->CreateICmpNE(std::forward<Args>(args)...);
  }

  template <class... Args>
  llvm::Value* ICmpULE(Args&&... args) {
   std::vector<std::string> mht_19_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSllvm_irPSir_builder_mixinDTh mht_19(mht_19_v, 371, "", "./tensorflow/compiler/xla/service/llvm_ir/ir_builder_mixin.h", "ICmpULE");

    return mixin_builder()->CreateICmpULE(std::forward<Args>(args)...);
  }

  template <class... Args>
  llvm::Value* ICmpULT(Args&&... args) {
   std::vector<std::string> mht_20_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSllvm_irPSir_builder_mixinDTh mht_20(mht_20_v, 379, "", "./tensorflow/compiler/xla/service/llvm_ir/ir_builder_mixin.h", "ICmpULT");

    return mixin_builder()->CreateICmpULT(std::forward<Args>(args)...);
  }

  llvm::Value* InBoundsGEP(llvm::Value* ptr,
                           llvm::ArrayRef<llvm::Value*> idx_list,
                           const llvm::Twine& name = "") {
   std::vector<std::string> mht_21_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSllvm_irPSir_builder_mixinDTh mht_21(mht_21_v, 388, "", "./tensorflow/compiler/xla/service/llvm_ir/ir_builder_mixin.h", "InBoundsGEP");

    return mixin_builder()->CreateInBoundsGEP(
        ptr->getType()->getPointerElementType(), ptr, idx_list, name);
  }

  llvm::Value* ExtractValue(llvm::Value* agg, llvm::ArrayRef<unsigned> idxs,
                            const llvm::Twine& name = "") {
   std::vector<std::string> mht_22_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSllvm_irPSir_builder_mixinDTh mht_22(mht_22_v, 397, "", "./tensorflow/compiler/xla/service/llvm_ir/ir_builder_mixin.h", "ExtractValue");

    return mixin_builder()->CreateExtractValue(agg, idxs, name);
  }

  llvm::Value* InsertValue(llvm::Value* agg, llvm::Value* val,
                           llvm::ArrayRef<unsigned> idxs,
                           const llvm::Twine& name = "") {
   std::vector<std::string> mht_23_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSllvm_irPSir_builder_mixinDTh mht_23(mht_23_v, 406, "", "./tensorflow/compiler/xla/service/llvm_ir/ir_builder_mixin.h", "InsertValue");

    return mixin_builder()->CreateInsertValue(agg, val, idxs, name);
  }

  template <class... Args>
  llvm::Value* IntToPtr(Args&&... args) {
   std::vector<std::string> mht_24_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSllvm_irPSir_builder_mixinDTh mht_24(mht_24_v, 414, "", "./tensorflow/compiler/xla/service/llvm_ir/ir_builder_mixin.h", "IntToPtr");

    return mixin_builder()->CreateIntToPtr(std::forward<Args>(args)...);
  }

  template <class... Args>
  llvm::LoadInst* Load(llvm::Type* type, Args&&... args) {
   std::vector<std::string> mht_25_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSllvm_irPSir_builder_mixinDTh mht_25(mht_25_v, 422, "", "./tensorflow/compiler/xla/service/llvm_ir/ir_builder_mixin.h", "Load");

    return mixin_builder()->CreateLoad(type, std::forward<Args>(args)...);
  }

  template <class... Args>
  llvm::LoadInst* Load(llvm::Value* value, Args&&... args) {
   std::vector<std::string> mht_26_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSllvm_irPSir_builder_mixinDTh mht_26(mht_26_v, 430, "", "./tensorflow/compiler/xla/service/llvm_ir/ir_builder_mixin.h", "Load");

    // LLVM has deprecated CreateLoad without a type argument. Provide it for
    // convenience.
    return mixin_builder()->CreateLoad(
        value->getType()->getPointerElementType(), value,
        std::forward<Args>(args)...);
  }

  template <class... Args>
  llvm::CallInst* MemCpy(Args&&... args) {
   std::vector<std::string> mht_27_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSllvm_irPSir_builder_mixinDTh mht_27(mht_27_v, 442, "", "./tensorflow/compiler/xla/service/llvm_ir/ir_builder_mixin.h", "MemCpy");

    return mixin_builder()->CreateMemCpy(std::forward<Args>(args)...);
  }

  template <class... Args>
  llvm::Value* Mul(Args&&... args) {
   std::vector<std::string> mht_28_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSllvm_irPSir_builder_mixinDTh mht_28(mht_28_v, 450, "", "./tensorflow/compiler/xla/service/llvm_ir/ir_builder_mixin.h", "Mul");

    return mixin_builder()->CreateMul(std::forward<Args>(args)...);
  }

  template <class... Args>
  llvm::Value* NSWAdd(Args&&... args) {
   std::vector<std::string> mht_29_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSllvm_irPSir_builder_mixinDTh mht_29(mht_29_v, 458, "", "./tensorflow/compiler/xla/service/llvm_ir/ir_builder_mixin.h", "NSWAdd");

    return mixin_builder()->CreateNSWAdd(std::forward<Args>(args)...);
  }

  template <class... Args>
  llvm::Value* NSWMul(Args&&... args) {
   std::vector<std::string> mht_30_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSllvm_irPSir_builder_mixinDTh mht_30(mht_30_v, 466, "", "./tensorflow/compiler/xla/service/llvm_ir/ir_builder_mixin.h", "NSWMul");

    return mixin_builder()->CreateNSWMul(std::forward<Args>(args)...);
  }

  template <class... Args>
  llvm::Value* NSWSub(Args&&... args) {
   std::vector<std::string> mht_31_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSllvm_irPSir_builder_mixinDTh mht_31(mht_31_v, 474, "", "./tensorflow/compiler/xla/service/llvm_ir/ir_builder_mixin.h", "NSWSub");

    return mixin_builder()->CreateNSWSub(std::forward<Args>(args)...);
  }

  template <class... Args>
  llvm::Value* Or(Args&&... args) {
   std::vector<std::string> mht_32_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSllvm_irPSir_builder_mixinDTh mht_32(mht_32_v, 482, "", "./tensorflow/compiler/xla/service/llvm_ir/ir_builder_mixin.h", "Or");

    return mixin_builder()->CreateOr(std::forward<Args>(args)...);
  }

  template <class... Args>
  llvm::Value* PointerCast(Args&&... args) {
   std::vector<std::string> mht_33_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSllvm_irPSir_builder_mixinDTh mht_33(mht_33_v, 490, "", "./tensorflow/compiler/xla/service/llvm_ir/ir_builder_mixin.h", "PointerCast");

    return mixin_builder()->CreatePointerCast(std::forward<Args>(args)...);
  }

  template <class... Args>
  llvm::Value* PtrToInt(Args&&... args) {
   std::vector<std::string> mht_34_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSllvm_irPSir_builder_mixinDTh mht_34(mht_34_v, 498, "", "./tensorflow/compiler/xla/service/llvm_ir/ir_builder_mixin.h", "PtrToInt");

    return mixin_builder()->CreatePtrToInt(std::forward<Args>(args)...);
  }

  template <class... Args>
  llvm::Value* SDiv(Args&&... args) {
   std::vector<std::string> mht_35_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSllvm_irPSir_builder_mixinDTh mht_35(mht_35_v, 506, "", "./tensorflow/compiler/xla/service/llvm_ir/ir_builder_mixin.h", "SDiv");

    return mixin_builder()->CreateSDiv(std::forward<Args>(args)...);
  }

  template <class... Args>
  llvm::Value* Select(Args&&... args) {
   std::vector<std::string> mht_36_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSllvm_irPSir_builder_mixinDTh mht_36(mht_36_v, 514, "", "./tensorflow/compiler/xla/service/llvm_ir/ir_builder_mixin.h", "Select");

    return mixin_builder()->CreateSelect(std::forward<Args>(args)...);
  }

  template <class... Args>
  llvm::Value* SRem(Args&&... args) {
   std::vector<std::string> mht_37_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSllvm_irPSir_builder_mixinDTh mht_37(mht_37_v, 522, "", "./tensorflow/compiler/xla/service/llvm_ir/ir_builder_mixin.h", "SRem");

    return mixin_builder()->CreateSRem(std::forward<Args>(args)...);
  }

  template <class... Args>
  llvm::StoreInst* Store(Args&&... args) {
   std::vector<std::string> mht_38_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSllvm_irPSir_builder_mixinDTh mht_38(mht_38_v, 530, "", "./tensorflow/compiler/xla/service/llvm_ir/ir_builder_mixin.h", "Store");

    return mixin_builder()->CreateStore(std::forward<Args>(args)...);
  }

  template <class... Args>
  llvm::Value* UDiv(Args&&... args) {
   std::vector<std::string> mht_39_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSllvm_irPSir_builder_mixinDTh mht_39(mht_39_v, 538, "", "./tensorflow/compiler/xla/service/llvm_ir/ir_builder_mixin.h", "UDiv");

    return mixin_builder()->CreateUDiv(std::forward<Args>(args)...);
  }

  template <class... Args>
  llvm::Value* URem(Args&&... args) {
   std::vector<std::string> mht_40_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSllvm_irPSir_builder_mixinDTh mht_40(mht_40_v, 546, "", "./tensorflow/compiler/xla/service/llvm_ir/ir_builder_mixin.h", "URem");

    return mixin_builder()->CreateURem(std::forward<Args>(args)...);
  }

  template <class... Args>
  llvm::Value* VectorSplat(Args&&... args) {
   std::vector<std::string> mht_41_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSllvm_irPSir_builder_mixinDTh mht_41(mht_41_v, 554, "", "./tensorflow/compiler/xla/service/llvm_ir/ir_builder_mixin.h", "VectorSplat");

    return mixin_builder()->CreateVectorSplat(std::forward<Args>(args)...);
  }

  template <class... Args>
  llvm::Value* ZExtOrTrunc(Args&&... args) {
   std::vector<std::string> mht_42_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSllvm_irPSir_builder_mixinDTh mht_42(mht_42_v, 562, "", "./tensorflow/compiler/xla/service/llvm_ir/ir_builder_mixin.h", "ZExtOrTrunc");

    return mixin_builder()->CreateZExtOrTrunc(std::forward<Args>(args)...);
  }

  template <class... Args>
  llvm::Value* AShr(Args&&... args) {
   std::vector<std::string> mht_43_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSllvm_irPSir_builder_mixinDTh mht_43(mht_43_v, 570, "", "./tensorflow/compiler/xla/service/llvm_ir/ir_builder_mixin.h", "AShr");

    return mixin_builder()->CreateAShr(std::forward<Args>(args)...);
  }

  template <class... Args>
  llvm::Value* FCmpOEQ(Args&&... args) {
   std::vector<std::string> mht_44_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSllvm_irPSir_builder_mixinDTh mht_44(mht_44_v, 578, "", "./tensorflow/compiler/xla/service/llvm_ir/ir_builder_mixin.h", "FCmpOEQ");

    return mixin_builder()->CreateFCmpOEQ(std::forward<Args>(args)...);
  }

  template <class... Args>
  llvm::Value* FCmpOGT(Args&&... args) {
   std::vector<std::string> mht_45_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSllvm_irPSir_builder_mixinDTh mht_45(mht_45_v, 586, "", "./tensorflow/compiler/xla/service/llvm_ir/ir_builder_mixin.h", "FCmpOGT");

    return mixin_builder()->CreateFCmpOGT(std::forward<Args>(args)...);
  }

  template <class... Args>
  llvm::Value* FCmpOGE(Args&&... args) {
   std::vector<std::string> mht_46_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSllvm_irPSir_builder_mixinDTh mht_46(mht_46_v, 594, "", "./tensorflow/compiler/xla/service/llvm_ir/ir_builder_mixin.h", "FCmpOGE");

    return mixin_builder()->CreateFCmpOGE(std::forward<Args>(args)...);
  }

  template <class... Args>
  llvm::Value* FCmpOLT(Args&&... args) {
   std::vector<std::string> mht_47_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSllvm_irPSir_builder_mixinDTh mht_47(mht_47_v, 602, "", "./tensorflow/compiler/xla/service/llvm_ir/ir_builder_mixin.h", "FCmpOLT");

    return mixin_builder()->CreateFCmpOLT(std::forward<Args>(args)...);
  }

  template <class... Args>
  llvm::Value* FCmpULT(Args&&... args) {
   std::vector<std::string> mht_48_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSllvm_irPSir_builder_mixinDTh mht_48(mht_48_v, 610, "", "./tensorflow/compiler/xla/service/llvm_ir/ir_builder_mixin.h", "FCmpULT");

    return mixin_builder()->CreateFCmpULT(std::forward<Args>(args)...);
  }

  template <class... Args>
  llvm::Value* FCmpULE(Args&&... args) {
   std::vector<std::string> mht_49_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSllvm_irPSir_builder_mixinDTh mht_49(mht_49_v, 618, "", "./tensorflow/compiler/xla/service/llvm_ir/ir_builder_mixin.h", "FCmpULE");

    return mixin_builder()->CreateFCmpULE(std::forward<Args>(args)...);
  }

  template <class... Args>
  llvm::Value* FCmpOLE(Args&&... args) {
   std::vector<std::string> mht_50_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSllvm_irPSir_builder_mixinDTh mht_50(mht_50_v, 626, "", "./tensorflow/compiler/xla/service/llvm_ir/ir_builder_mixin.h", "FCmpOLE");

    return mixin_builder()->CreateFCmpOLE(std::forward<Args>(args)...);
  }

  template <class... Args>
  llvm::Value* FCmpONE(Args&&... args) {
   std::vector<std::string> mht_51_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSllvm_irPSir_builder_mixinDTh mht_51(mht_51_v, 634, "", "./tensorflow/compiler/xla/service/llvm_ir/ir_builder_mixin.h", "FCmpONE");

    return mixin_builder()->CreateFCmpONE(std::forward<Args>(args)...);
  }

  template <class... Args>
  llvm::Value* FCmpUNE(Args&&... args) {
   std::vector<std::string> mht_52_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSllvm_irPSir_builder_mixinDTh mht_52(mht_52_v, 642, "", "./tensorflow/compiler/xla/service/llvm_ir/ir_builder_mixin.h", "FCmpUNE");

    return mixin_builder()->CreateFCmpUNE(std::forward<Args>(args)...);
  }

  template <class... Args>
  llvm::Value* FCmpUNO(Args&&... args) {
   std::vector<std::string> mht_53_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSllvm_irPSir_builder_mixinDTh mht_53(mht_53_v, 650, "", "./tensorflow/compiler/xla/service/llvm_ir/ir_builder_mixin.h", "FCmpUNO");

    return mixin_builder()->CreateFCmpUNO(std::forward<Args>(args)...);
  }

  template <class... Args>
  llvm::Value* FDiv(Args&&... args) {
   std::vector<std::string> mht_54_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSllvm_irPSir_builder_mixinDTh mht_54(mht_54_v, 658, "", "./tensorflow/compiler/xla/service/llvm_ir/ir_builder_mixin.h", "FDiv");

    return mixin_builder()->CreateFDiv(std::forward<Args>(args)...);
  }

  template <class... Args>
  llvm::Value* FNeg(Args&&... args) {
   std::vector<std::string> mht_55_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSllvm_irPSir_builder_mixinDTh mht_55(mht_55_v, 666, "", "./tensorflow/compiler/xla/service/llvm_ir/ir_builder_mixin.h", "FNeg");

    return mixin_builder()->CreateFNeg(std::forward<Args>(args)...);
  }

  template <class... Args>
  llvm::Value* FPCast(Args&&... args) {
   std::vector<std::string> mht_56_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSllvm_irPSir_builder_mixinDTh mht_56(mht_56_v, 674, "", "./tensorflow/compiler/xla/service/llvm_ir/ir_builder_mixin.h", "FPCast");

    return mixin_builder()->CreateFPCast(std::forward<Args>(args)...);
  }

  template <class... Args>
  llvm::Value* FPToSI(Args&&... args) {
   std::vector<std::string> mht_57_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSllvm_irPSir_builder_mixinDTh mht_57(mht_57_v, 682, "", "./tensorflow/compiler/xla/service/llvm_ir/ir_builder_mixin.h", "FPToSI");

    return mixin_builder()->CreateFPToSI(std::forward<Args>(args)...);
  }

  template <class... Args>
  llvm::Value* FPToUI(Args&&... args) {
   std::vector<std::string> mht_58_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSllvm_irPSir_builder_mixinDTh mht_58(mht_58_v, 690, "", "./tensorflow/compiler/xla/service/llvm_ir/ir_builder_mixin.h", "FPToUI");

    return mixin_builder()->CreateFPToUI(std::forward<Args>(args)...);
  }

  template <class... Args>
  llvm::Value* FPTrunc(Args&&... args) {
   std::vector<std::string> mht_59_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSllvm_irPSir_builder_mixinDTh mht_59(mht_59_v, 698, "", "./tensorflow/compiler/xla/service/llvm_ir/ir_builder_mixin.h", "FPTrunc");

    return mixin_builder()->CreateFPTrunc(std::forward<Args>(args)...);
  }

  template <class... Args>
  llvm::Value* FRem(Args&&... args) {
   std::vector<std::string> mht_60_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSllvm_irPSir_builder_mixinDTh mht_60(mht_60_v, 706, "", "./tensorflow/compiler/xla/service/llvm_ir/ir_builder_mixin.h", "FRem");

    return mixin_builder()->CreateFRem(std::forward<Args>(args)...);
  }

  template <class... Args>
  llvm::Value* FSub(Args&&... args) {
   std::vector<std::string> mht_61_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSllvm_irPSir_builder_mixinDTh mht_61(mht_61_v, 714, "", "./tensorflow/compiler/xla/service/llvm_ir/ir_builder_mixin.h", "FSub");

    return mixin_builder()->CreateFSub(std::forward<Args>(args)...);
  }

  template <class... Args>
  llvm::Value* ICmpSGE(Args&&... args) {
   std::vector<std::string> mht_62_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSllvm_irPSir_builder_mixinDTh mht_62(mht_62_v, 722, "", "./tensorflow/compiler/xla/service/llvm_ir/ir_builder_mixin.h", "ICmpSGE");

    return mixin_builder()->CreateICmpSGE(std::forward<Args>(args)...);
  }

  template <class... Args>
  llvm::Value* ICmpSLT(Args&&... args) {
   std::vector<std::string> mht_63_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSllvm_irPSir_builder_mixinDTh mht_63(mht_63_v, 730, "", "./tensorflow/compiler/xla/service/llvm_ir/ir_builder_mixin.h", "ICmpSLT");

    return mixin_builder()->CreateICmpSLT(std::forward<Args>(args)...);
  }

  template <class... Args>
  llvm::Value* IntCast(Args&&... args) {
   std::vector<std::string> mht_64_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSllvm_irPSir_builder_mixinDTh mht_64(mht_64_v, 738, "", "./tensorflow/compiler/xla/service/llvm_ir/ir_builder_mixin.h", "IntCast");

    return mixin_builder()->CreateIntCast(std::forward<Args>(args)...);
  }

  template <class... Args>
  llvm::Value* LShr(Args&&... args) {
   std::vector<std::string> mht_65_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSllvm_irPSir_builder_mixinDTh mht_65(mht_65_v, 746, "", "./tensorflow/compiler/xla/service/llvm_ir/ir_builder_mixin.h", "LShr");

    return mixin_builder()->CreateLShr(std::forward<Args>(args)...);
  }

  template <class... Args>
  llvm::Value* MemSet(Args&&... args) {
   std::vector<std::string> mht_66_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSllvm_irPSir_builder_mixinDTh mht_66(mht_66_v, 754, "", "./tensorflow/compiler/xla/service/llvm_ir/ir_builder_mixin.h", "MemSet");

    return mixin_builder()->CreateMemSet(std::forward<Args>(args)...);
  }

  template <class... Args>
  llvm::Value* Neg(Args&&... args) {
   std::vector<std::string> mht_67_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSllvm_irPSir_builder_mixinDTh mht_67(mht_67_v, 762, "", "./tensorflow/compiler/xla/service/llvm_ir/ir_builder_mixin.h", "Neg");

    return mixin_builder()->CreateNeg(std::forward<Args>(args)...);
  }

  template <class... Args>
  llvm::Value* Not(Args&&... args) {
   std::vector<std::string> mht_68_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSllvm_irPSir_builder_mixinDTh mht_68(mht_68_v, 770, "", "./tensorflow/compiler/xla/service/llvm_ir/ir_builder_mixin.h", "Not");

    return mixin_builder()->CreateNot(std::forward<Args>(args)...);
  }

  template <class... Args>
  llvm::PHINode* PHI(Args&&... args) {
   std::vector<std::string> mht_69_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSllvm_irPSir_builder_mixinDTh mht_69(mht_69_v, 778, "", "./tensorflow/compiler/xla/service/llvm_ir/ir_builder_mixin.h", "PHI");

    return mixin_builder()->CreatePHI(std::forward<Args>(args)...);
  }

  template <class... Args>
  llvm::Value* RetVoid(Args&&... args) {
   std::vector<std::string> mht_70_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSllvm_irPSir_builder_mixinDTh mht_70(mht_70_v, 786, "", "./tensorflow/compiler/xla/service/llvm_ir/ir_builder_mixin.h", "RetVoid");

    return mixin_builder()->CreateRetVoid(std::forward<Args>(args)...);
  }

  template <class... Args>
  llvm::Value* SExtOrTrunc(Args&&... args) {
   std::vector<std::string> mht_71_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSllvm_irPSir_builder_mixinDTh mht_71(mht_71_v, 794, "", "./tensorflow/compiler/xla/service/llvm_ir/ir_builder_mixin.h", "SExtOrTrunc");

    return mixin_builder()->CreateSExtOrTrunc(std::forward<Args>(args)...);
  }

  template <class... Args>
  llvm::Value* Shl(Args&&... args) {
   std::vector<std::string> mht_72_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSllvm_irPSir_builder_mixinDTh mht_72(mht_72_v, 802, "", "./tensorflow/compiler/xla/service/llvm_ir/ir_builder_mixin.h", "Shl");

    return mixin_builder()->CreateShl(std::forward<Args>(args)...);
  }

  template <class... Args>
  llvm::Value* SIToFP(Args&&... args) {
   std::vector<std::string> mht_73_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSllvm_irPSir_builder_mixinDTh mht_73(mht_73_v, 810, "", "./tensorflow/compiler/xla/service/llvm_ir/ir_builder_mixin.h", "SIToFP");

    return mixin_builder()->CreateSIToFP(std::forward<Args>(args)...);
  }

  template <class... Args>
  llvm::Value* Sub(Args&&... args) {
   std::vector<std::string> mht_74_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSllvm_irPSir_builder_mixinDTh mht_74(mht_74_v, 818, "", "./tensorflow/compiler/xla/service/llvm_ir/ir_builder_mixin.h", "Sub");

    return mixin_builder()->CreateSub(std::forward<Args>(args)...);
  }

  template <class... Args>
  llvm::Value* Trunc(Args&&... args) {
   std::vector<std::string> mht_75_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSllvm_irPSir_builder_mixinDTh mht_75(mht_75_v, 826, "", "./tensorflow/compiler/xla/service/llvm_ir/ir_builder_mixin.h", "Trunc");

    return mixin_builder()->CreateTrunc(std::forward<Args>(args)...);
  }

  template <class... Args>
  llvm::Value* UIToFP(Args&&... args) {
   std::vector<std::string> mht_76_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSllvm_irPSir_builder_mixinDTh mht_76(mht_76_v, 834, "", "./tensorflow/compiler/xla/service/llvm_ir/ir_builder_mixin.h", "UIToFP");

    return mixin_builder()->CreateUIToFP(std::forward<Args>(args)...);
  }

  template <class... Args>
  llvm::Value* Unreachable(Args&&... args) {
   std::vector<std::string> mht_77_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSllvm_irPSir_builder_mixinDTh mht_77(mht_77_v, 842, "", "./tensorflow/compiler/xla/service/llvm_ir/ir_builder_mixin.h", "Unreachable");

    return mixin_builder()->CreateUnreachable(std::forward<Args>(args)...);
  }

  template <class... Args>
  llvm::Value* Xor(Args&&... args) {
   std::vector<std::string> mht_78_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSllvm_irPSir_builder_mixinDTh mht_78(mht_78_v, 850, "", "./tensorflow/compiler/xla/service/llvm_ir/ir_builder_mixin.h", "Xor");

    return mixin_builder()->CreateXor(std::forward<Args>(args)...);
  }

 private:
  llvm::IRBuilder<>* mixin_builder() {
   std::vector<std::string> mht_79_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSllvm_irPSir_builder_mixinDTh mht_79(mht_79_v, 858, "", "./tensorflow/compiler/xla/service/llvm_ir/ir_builder_mixin.h", "mixin_builder");

    return static_cast<Derived*>(this)->builder();
  }
};

}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_LLVM_IR_IR_BUILDER_MIXIN_H_
