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
class MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSllvm_irPSllvm_loopDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSllvm_irPSllvm_loopDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSllvm_irPSllvm_loopDTcc() {
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

#include "tensorflow/compiler/xla/service/llvm_ir/llvm_loop.h"

#include <numeric>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/strings/str_cat.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Instructions.h"
#include "tensorflow/compiler/xla/service/llvm_ir/llvm_util.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/core/platform/logging.h"

namespace xla {
namespace llvm_ir {

ForLoop::ForLoop(absl::string_view prefix, absl::string_view suffix,
                 llvm::Value* start_index, llvm::Value* end_index,
                 llvm::Value* step, UnrollMode unroll_mode,
                 bool prevent_vectorization)
    : prefix_(prefix),
      suffix_(suffix),
      start_index_(start_index),
      end_index_(end_index),
      step_(step),
      insert_before_bb_(nullptr),
      unroll_mode_(unroll_mode),
      prevent_vectorization_(prevent_vectorization) {
   std::vector<std::string> mht_0_v;
   mht_0_v.push_back("prefix: \"" + std::string(prefix.data(), prefix.size()) + "\"");
   mht_0_v.push_back("suffix: \"" + std::string(suffix.data(), suffix.size()) + "\"");
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSllvm_irPSllvm_loopDTcc mht_0(mht_0_v, 216, "", "./tensorflow/compiler/xla/service/llvm_ir/llvm_loop.cc", "ForLoop::ForLoop");
}

/* static */ std::unique_ptr<ForLoop> ForLoop::EmitForLoop(
    absl::string_view prefix, llvm::Value* start_index, llvm::Value* end_index,
    llvm::Value* step, llvm::IRBuilder<>* b, UnrollMode unroll_mode,
    bool prevent_vectorization) {
   std::vector<std::string> mht_1_v;
   mht_1_v.push_back("prefix: \"" + std::string(prefix.data(), prefix.size()) + "\"");
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSllvm_irPSllvm_loopDTcc mht_1(mht_1_v, 225, "", "./tensorflow/compiler/xla/service/llvm_ir/llvm_loop.cc", "ForLoop::EmitForLoop");

  std::unique_ptr<ForLoop> loop(new ForLoop(prefix, /*suffix=*/"", start_index,
                                            end_index, step, unroll_mode,
                                            prevent_vectorization));
  loop->Emit(b);
  return loop;
}

void ForLoop::Emit(llvm::IRBuilder<>* b) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSllvm_irPSllvm_loopDTcc mht_2(mht_2_v, 236, "", "./tensorflow/compiler/xla/service/llvm_ir/llvm_loop.cc", "ForLoop::Emit");

  // The preheader block is the block the builder is currently emitting
  // code into.
  preheader_bb_ = b->GetInsertBlock();

  llvm::BasicBlock::iterator insert_point = b->GetInsertPoint();
  if (insert_point == preheader_bb_->end()) {
    // We're emitting the loop at the end of a basic block. Verify there is no
    // terminator (eg, branch) in the basic block.
    CHECK_EQ(nullptr, preheader_bb_->getTerminator());

    exit_bb_ = CreateLoopBB("loop_exit", b);
  } else {
    // We're emitting the loop into the middle of a basic block. splitBasicBlock
    // requires that this basic block be well-formed (have a terminator).
    CHECK_NE(nullptr, preheader_bb_->getTerminator());

    // Split the preheader to create an exit basic block. The exit basic block
    // will contain all instructions at or after insert_point.
    exit_bb_ = preheader_bb_->splitBasicBlock(insert_point,
                                              GetQualifiedName("loop_exit"));

    // splitBasicBlock adds an unconditional branch between the split basic
    // blocks. Remove it. An unconditional branch will be added below from the
    // preheader to the header.
    preheader_bb_->getTerminator()->eraseFromParent();
  }
  insert_before_bb_ = exit_bb_;

  // Create remaining basic block which form the inside of the loop.
  header_bb_ = CreateLoopBB("loop_header", b);
  body_bb_ = CreateLoopBB("loop_body", b);

  // Function entry basic block.
  // Emit alloca for the induction variable. We do this at the entry to the
  // basic block to ensure the alloc only executes once per function (we could
  // be emitting a nested loop).
  llvm::Function* func = preheader_bb_->getParent();
  b->SetInsertPoint(&func->getEntryBlock(),
                    func->getEntryBlock().getFirstInsertionPt());
  llvm::Value* indvar_address = b->CreateAlloca(
      start_index_->getType(), nullptr, GetQualifiedName("invar_address"));

  // Preheader basic block.
  // Initialize induction variable starting index. Create branch to the header.
  b->SetInsertPoint(preheader_bb_);
  b->CreateStore(start_index_, indvar_address);
  // The preheader should not have a branch yet.
  CHECK_EQ(preheader_bb_->getTerminator(), nullptr);
  b->CreateBr(header_bb_);

  // Header basic block.
  // Emit the loop conditional branch. Load and compare indvar with ending
  // index and jump to loop exit if equal. Jump to body otherwise.
  b->SetInsertPoint(header_bb_);
  indvar_ = b->CreateLoad(start_index_->getType(), indvar_address,
                          GetQualifiedName("indvar"));
  llvm::Value* exit_cond = b->CreateICmpUGE(indvar_, end_index_);
  b->CreateCondBr(/*Cond=*/exit_cond,
                  /*True=*/exit_bb_, /*False=*/body_bb_);

  // Body basic block.
  // Increment indvar, store indvar, and jump to header.
  b->SetInsertPoint(body_bb_);
  llvm::Value* step = step_;
  llvm::Value* indvar = indvar_;

  llvm::Value* indvar_inc = b->CreateAdd(indvar, step, "invar.inc",
                                         /*HasNUW=*/true, /*HasNSW=*/true);
  b->CreateStore(indvar_inc, indvar_address);
  llvm::BranchInst* back_branch = b->CreateBr(header_bb_);

  std::vector<llvm::Metadata*> loop_metadata = GetLoopMetadata(b);
  if (!loop_metadata.empty()) {
    llvm::LLVMContext* ctx = &start_index_->getContext();
    auto temp_node = llvm::MDNode::getTemporary(*ctx, llvm::None);
    loop_metadata.insert(loop_metadata.begin(), temp_node.get());
    auto loop_id = llvm::MDNode::get(*ctx, loop_metadata);
    loop_id->replaceOperandWith(0, loop_id);
    back_branch->setMetadata(llvm::LLVMContext::MD_loop, loop_id);
  }

  // Re-point the IR builder to the loop exit block.
  b->SetInsertPoint(exit_bb_);
}

std::vector<llvm::Metadata*> ForLoop::GetLoopMetadata(llvm::IRBuilder<>* b) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSllvm_irPSllvm_loopDTcc mht_3(mht_3_v, 325, "", "./tensorflow/compiler/xla/service/llvm_ir/llvm_loop.cc", "ForLoop::GetLoopMetadata");

  const char* const kLlvmLoopUnrollDisableMDName = "llvm.loop.unroll.disable";
  const char* const kLlvmLoopUnrollFullMDName = "llvm.loop.unroll.full";
  const char* const kLlvmLoopVectorizeMDName = "llvm.loop.vectorize.enable";
  llvm::LLVMContext* ctx = &start_index_->getContext();

  std::vector<llvm::Metadata*> result;
  if (unroll_mode_ == xla::llvm_ir::UnrollMode::kNoUnroll) {
    result.push_back(llvm::MDNode::get(
        *ctx, {llvm::MDString::get(*ctx, kLlvmLoopUnrollDisableMDName)}));
  }

  if (prevent_vectorization_) {
    result.push_back(llvm::MDNode::get(
        *ctx, {llvm::MDString::get(*ctx, kLlvmLoopVectorizeMDName),
               llvm::ConstantAsMetadata::get(b->getFalse())}));
  }

  if (unroll_mode_ == xla::llvm_ir::UnrollMode::kFullyUnroll) {
    result.push_back(llvm::MDNode::get(
        *ctx, {llvm::MDString::get(*ctx, kLlvmLoopUnrollFullMDName)}));
  }
  return result;
}

std::string ForLoop::GetQualifiedName(absl::string_view name) {
   std::vector<std::string> mht_4_v;
   mht_4_v.push_back("name: \"" + std::string(name.data(), name.size()) + "\"");
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSllvm_irPSllvm_loopDTcc mht_4(mht_4_v, 354, "", "./tensorflow/compiler/xla/service/llvm_ir/llvm_loop.cc", "ForLoop::GetQualifiedName");

  return llvm_ir::IrName(prefix_, llvm_ir::IrName(name, suffix_));
}

llvm::BasicBlock* ForLoop::CreateLoopBB(absl::string_view name,
                                        llvm::IRBuilder<>* b) {
   std::vector<std::string> mht_5_v;
   mht_5_v.push_back("name: \"" + std::string(name.data(), name.size()) + "\"");
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSllvm_irPSllvm_loopDTcc mht_5(mht_5_v, 363, "", "./tensorflow/compiler/xla/service/llvm_ir/llvm_loop.cc", "ForLoop::CreateLoopBB");

  return CreateBasicBlock(insert_before_bb_, GetQualifiedName(name), b);
}

std::unique_ptr<ForLoop> ForLoopNest::AddLoop(absl::string_view suffix,
                                              llvm::Value* start_index,
                                              llvm::Value* end_index,
                                              UnrollMode unroll_mode,
                                              bool prevent_vectorization) {
   std::vector<std::string> mht_6_v;
   mht_6_v.push_back("suffix: \"" + std::string(suffix.data(), suffix.size()) + "\"");
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSllvm_irPSllvm_loopDTcc mht_6(mht_6_v, 375, "", "./tensorflow/compiler/xla/service/llvm_ir/llvm_loop.cc", "ForLoopNest::AddLoop");

  return AddLoop(suffix, start_index, end_index, GetConstantWithIndexType(1),
                 unroll_mode, prevent_vectorization);
}

std::unique_ptr<ForLoop> ForLoopNest::AddLoop(
    absl::string_view suffix, llvm::Value* start_index, llvm::Value* end_index,
    llvm::Value* stride, UnrollMode unroll_mode, bool prevent_vectorization) {
   std::vector<std::string> mht_7_v;
   mht_7_v.push_back("suffix: \"" + std::string(suffix.data(), suffix.size()) + "\"");
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSllvm_irPSllvm_loopDTcc mht_7(mht_7_v, 386, "", "./tensorflow/compiler/xla/service/llvm_ir/llvm_loop.cc", "ForLoopNest::AddLoop");

  if (inner_loop_body_bb_ != nullptr) {
    // Create this loop inside the previous one.
    b_->SetInsertPoint(&*inner_loop_body_bb_->getFirstInsertionPt());
  }
  std::unique_ptr<ForLoop> loop(new ForLoop(
      /*prefix=*/name_, suffix, start_index, end_index, stride, unroll_mode,
      prevent_vectorization));
  loop->Emit(b_);

  if (outer_loop_preheader_bb_ == nullptr) {
    outer_loop_preheader_bb_ = loop->GetPreheaderBasicBlock();
  }

  if (outer_loop_exit_bb_ == nullptr) {
    outer_loop_exit_bb_ = loop->GetExitBasicBlock();
  }

  inner_loop_body_bb_ = loop->GetBodyBasicBlock();

  return loop;
}

std::unique_ptr<ForLoop> ForLoopNest::AddLoop(int64_t start_index,
                                              int64_t end_index,
                                              absl::string_view suffix,
                                              UnrollMode unroll_mode,
                                              bool prevent_vectorization) {
   std::vector<std::string> mht_8_v;
   mht_8_v.push_back("suffix: \"" + std::string(suffix.data(), suffix.size()) + "\"");
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSllvm_irPSllvm_loopDTcc mht_8(mht_8_v, 417, "", "./tensorflow/compiler/xla/service/llvm_ir/llvm_loop.cc", "ForLoopNest::AddLoop");

  CHECK_LE(start_index, end_index);
  return AddLoop(suffix, GetConstantWithIndexType(start_index),
                 GetConstantWithIndexType(end_index), unroll_mode,
                 prevent_vectorization);
}

std::unique_ptr<ForLoop> ForLoopNest::AddLoop(int64_t start_index,
                                              int64_t end_index, int64_t stride,
                                              absl::string_view suffix,
                                              UnrollMode unroll_mode,
                                              bool prevent_vectorization) {
   std::vector<std::string> mht_9_v;
   mht_9_v.push_back("suffix: \"" + std::string(suffix.data(), suffix.size()) + "\"");
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSllvm_irPSllvm_loopDTcc mht_9(mht_9_v, 432, "", "./tensorflow/compiler/xla/service/llvm_ir/llvm_loop.cc", "ForLoopNest::AddLoop");

  CHECK_LE(start_index, end_index);
  return AddLoop(suffix, GetConstantWithIndexType(start_index),
                 GetConstantWithIndexType(end_index),
                 GetConstantWithIndexType(stride), unroll_mode,
                 prevent_vectorization);
}

IrArray::Index ForLoopNest::AddLoopsForShape(const Shape& shape,
                                             absl::string_view suffix) {
   std::vector<std::string> mht_10_v;
   mht_10_v.push_back("suffix: \"" + std::string(suffix.data(), suffix.size()) + "\"");
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSllvm_irPSllvm_loopDTcc mht_10(mht_10_v, 445, "", "./tensorflow/compiler/xla/service/llvm_ir/llvm_loop.cc", "ForLoopNest::AddLoopsForShape");

  std::vector<int64_t> dimensions(shape.rank());
  std::iota(dimensions.begin(), dimensions.end(), 0);
  return IrArray::Index(AddLoopsForShapeOnDimensions(shape, dimensions, suffix),
                        shape, index_type_);
}

std::vector<llvm::Value*> ForLoopNest::AddLoopsForShapeOnDimensions(
    const Shape& shape, absl::Span<const int64_t> dimensions,
    absl::string_view suffix) {
   std::vector<std::string> mht_11_v;
   mht_11_v.push_back("suffix: \"" + std::string(suffix.data(), suffix.size()) + "\"");
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSllvm_irPSllvm_loopDTcc mht_11(mht_11_v, 458, "", "./tensorflow/compiler/xla/service/llvm_ir/llvm_loop.cc", "ForLoopNest::AddLoopsForShapeOnDimensions");

  std::vector<llvm::Value*> multi_index(shape.dimensions_size());
  for (int64_t dimension : dimensions) {
    std::unique_ptr<llvm_ir::ForLoop> loop = AddLoop(
        /*start_index=*/0,
        /*end_index=*/shape.dimensions(dimension),
        /*suffix=*/
        llvm_ir::IrName(suffix, absl::StrCat(dimension)));
    multi_index[dimension] = loop->GetIndVarValue();
  }
  return multi_index;
}

std::vector<llvm::Value*> ForLoopNest::EmitOperandArrayLoopNest(
    const llvm_ir::IrArray& operand_array, int64_t dimension_to_skip,
    absl::string_view name_suffix) {
   std::vector<std::string> mht_12_v;
   mht_12_v.push_back("name_suffix: \"" + std::string(name_suffix.data(), name_suffix.size()) + "\"");
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSllvm_irPSllvm_loopDTcc mht_12(mht_12_v, 477, "", "./tensorflow/compiler/xla/service/llvm_ir/llvm_loop.cc", "ForLoopNest::EmitOperandArrayLoopNest");

  // Prepares the dimension list we will use to emit the loop nest. Outermost
  // loops are added first. Add loops in major-to-minor order, and skip the
  // 'dimension_to_skip' dimension.
  std::vector<int64_t> dimensions;
  const Shape& shape = operand_array.GetShape();
  // Initially get the dimensions in minor to major order, then reverse them.
  for (int64_t dimension : LayoutUtil::MinorToMajor(shape)) {
    if (dimension != dimension_to_skip) {
      dimensions.push_back(dimension);
    }
  }
  absl::c_reverse(dimensions);

  // Create loop nest with one for-loop for each dimension of the
  // output.
  std::vector<llvm::Value*> multi_index =
      AddLoopsForShapeOnDimensions(shape, dimensions, name_suffix);
  // Verify every dimension except the 'dimension_to_skip' dimension was set in
  // the index.
  for (size_t dimension = 0; dimension < multi_index.size(); ++dimension) {
    if (dimension == dimension_to_skip) {
      DCHECK_EQ(nullptr, multi_index[dimension]);
    } else {
      DCHECK_NE(nullptr, multi_index[dimension]);
    }
  }
  return multi_index;
}

}  // namespace llvm_ir
}  // namespace xla
