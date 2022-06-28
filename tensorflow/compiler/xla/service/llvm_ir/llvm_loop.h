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

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_LLVM_IR_LLVM_LOOP_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_LLVM_IR_LLVM_LOOP_H_
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
class MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSllvm_irPSllvm_loopDTh {
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
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSllvm_irPSllvm_loopDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSllvm_irPSllvm_loopDTh() {
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


#include <memory>
#include <string>

#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Value.h"
#include "tensorflow/compiler/xla/service/llvm_ir/ir_array.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"

namespace xla {
namespace llvm_ir {

enum class UnrollMode {
  kDefaultUnroll,
  kFullyUnroll,
  kNoUnroll,
};

// A class for constructing a for-loop in LLVM IR.
class ForLoop {
 public:
  ForLoop(const ForLoop&) = delete;
  ForLoop& operator=(const ForLoop&) = delete;

  // Emit a for-loop at the current insert point of the given IRBuilder.
  //
  // start_index and end_index are the loop bounds (end_index is not inclusive).
  // `step` is the increment of the loop index after each iteration.
  //
  // The current insert basic block of the builder is the preheader to the loop
  // (see below for definition of basic block names). All instructions (if any)
  // at or after the insert point in the insert basic block are moved to a newly
  // created exit basic block. Instructions before the insert point remain in
  // the insert BB:
  //
  //                   +--------------+         +----------------+
  //                   |  insert BB   |         |   insert BB    |
  //                   |     ...      |         | (preheader BB) |
  //                   | %foo = ...   |         |      ...       |
  //    insert point ->| %bar = ...   |  ===>   | %foo = ...     |
  //                   |     ...      |         +----------------+
  //                   +--------------+                 |
  //                                                    V
  //                                              [[ LOOP BBs ]]
  //                                                    |
  //                                                    V
  //                                             +--------------+
  //                                             |   exit BB    |
  //                                             | %bar = ...   |
  //                                             |     ...      |
  //                                             +--------------+
  //
  // `prefix` is used to disambiguate variable and basic block names emitted in
  // LLVM IR. If non-empty, it is prepended to the name of the induction
  // variable value and each basic block created for the loop.
  //
  // `unroll_mode` specifies the desired LLVM unrolling behavior for generated
  //  loop.
  static std::unique_ptr<ForLoop> EmitForLoop(
      absl::string_view prefix, llvm::Value* start_index,
      llvm::Value* end_index, llvm::Value* step, llvm::IRBuilder<>* b,
      UnrollMode unroll_mode = llvm_ir::UnrollMode::kDefaultUnroll,
      bool prevent_vectorization = false);

  // The names of the blocks follow LLVM's conventions. Control flow amongst the
  // blocks for the example C code looks like:
  //
  //   for (int i = 0; i < n; ++i) {
  //     do_stuff(i);
  //   }
  //
  //      +--------------+
  //      | preheader BB |
  //      |     i = 0    |
  //      +--------------+
  //              |
  //              V
  //      +-------------+
  //      |  header BB  |<-+
  //      | if i < n:   |  |
  //      |   goto body |  |
  //      | else:       |  |
  //      |   goto exit |  |
  //      +-------------+  |
  //            | |        |
  //   +--------+ |        |
  //   |          V        |
  //   |  +-------------+  |
  //   |  |   body BB   |  |
  //   |  | dostuff(i)  |--+
  //   |  | ++i         |
  //   |  +-------------+
  //   |
  //   |  +-------------+
  //   +->|   exit BB   |
  //      +-------------+
  //
  // Caller-emitted code to execute within the loop should be placed within the
  // "body" basic block.
  //
  // Return pointers to various blocks in the loop.
  llvm::BasicBlock* GetPreheaderBasicBlock() const {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSllvm_irPSllvm_loopDTh mht_0(mht_0_v, 293, "", "./tensorflow/compiler/xla/service/llvm_ir/llvm_loop.h", "GetPreheaderBasicBlock");
 return preheader_bb_; }
  llvm::BasicBlock* GetHeaderBasicBlock() const {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSllvm_irPSllvm_loopDTh mht_1(mht_1_v, 297, "", "./tensorflow/compiler/xla/service/llvm_ir/llvm_loop.h", "GetHeaderBasicBlock");
 return header_bb_; }
  llvm::BasicBlock* GetBodyBasicBlock() const {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSllvm_irPSllvm_loopDTh mht_2(mht_2_v, 301, "", "./tensorflow/compiler/xla/service/llvm_ir/llvm_loop.h", "GetBodyBasicBlock");
 return body_bb_; }
  llvm::BasicBlock* GetExitBasicBlock() const {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSllvm_irPSllvm_loopDTh mht_3(mht_3_v, 305, "", "./tensorflow/compiler/xla/service/llvm_ir/llvm_loop.h", "GetExitBasicBlock");
 return exit_bb_; }

  // Return the Value representing the induction variable in the body basic
  // block of the loop.
  llvm::Value* GetIndVarValue() const {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSllvm_irPSllvm_loopDTh mht_4(mht_4_v, 312, "", "./tensorflow/compiler/xla/service/llvm_ir/llvm_loop.h", "GetIndVarValue");
 return indvar_; }

 private:
  // Allow ForLoopNest to call this private constructor.
  friend class ForLoopNest;

  ForLoop(absl::string_view prefix, absl::string_view suffix,
          llvm::Value* start_index, llvm::Value* end_index, llvm::Value* step,
          UnrollMode unroll_mode, bool prevent_vectorization);

  // Emit the loop at the insert point of the builder.
  void Emit(llvm::IRBuilder<>* b);

  llvm::BasicBlock* CreateLoopBB(absl::string_view name, llvm::IRBuilder<>* b);

  // Creates a name for an LLVM construct, appending prefix_ and suffix_, if
  // they are set.
  std::string GetQualifiedName(absl::string_view name);

  // Return a list of metadata nodes that should be associated with the
  // llvm::Loop for this `ForLoop`.
  std::vector<llvm::Metadata*> GetLoopMetadata(llvm::IRBuilder<>* b);

  std::string prefix_;
  std::string suffix_;
  llvm::Value* start_index_;
  llvm::Value* end_index_;
  llvm::Value* step_;

  // To improve readability of the IR, we want the basic blocks to appear
  // consecutively in the following order: preheader, header, body, loop,
  // exit. The member insert_before_bb_ points to where the next basic block
  // should be created to ensure this ordering.
  llvm::BasicBlock* insert_before_bb_;

  llvm::BasicBlock* preheader_bb_;
  llvm::BasicBlock* header_bb_;
  llvm::BasicBlock* body_bb_;
  llvm::BasicBlock* exit_bb_;
  llvm::Value* indvar_;
  UnrollMode unroll_mode_;
  bool prevent_vectorization_;
};

// A simple class for constructing nested for-loops.
class ForLoopNest {
 public:
  ForLoopNest(absl::string_view name, llvm::IRBuilder<>* b,
              llvm::Type* index_ty = nullptr)
      : name_(name),
        outer_loop_preheader_bb_(nullptr),
        outer_loop_exit_bb_(nullptr),
        inner_loop_body_bb_(nullptr),
        b_(b) {
   std::vector<std::string> mht_5_v;
   mht_5_v.push_back("name: \"" + std::string(name.data(), name.size()) + "\"");
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSllvm_irPSllvm_loopDTh mht_5(mht_5_v, 369, "", "./tensorflow/compiler/xla/service/llvm_ir/llvm_loop.h", "ForLoopNest");

    SetIndexType(index_ty);
  }
  ForLoopNest(const ForLoopNest&) = delete;
  ForLoopNest& operator=(const ForLoopNest&) = delete;

  // Adds a loop to the nest. If no loop has been added yet then emit a loop at
  // the current insert point of the given builder. If one or more loops have
  // been added then emit loop inside the body of the last added loop.
  // unroll_mode is used to emit metadata that controls LLVM unrolling.
  std::unique_ptr<ForLoop> AddLoop(
      absl::string_view suffix, llvm::Value* start_index,
      llvm::Value* end_index, llvm::Value* stride,
      UnrollMode unroll_mode = xla::llvm_ir::UnrollMode::kDefaultUnroll,
      bool prevent_vectorization = false);

  // Like the above, except that it defaults to a stride of one.
  std::unique_ptr<ForLoop> AddLoop(
      absl::string_view suffix, llvm::Value* start_index,
      llvm::Value* end_index,
      UnrollMode unroll_mode = xla::llvm_ir::UnrollMode::kDefaultUnroll,
      bool prevent_vectorization = false);

  // A convenient wrapper of the other flavor of AddLoop. The given start and
  // end index are constant.
  std::unique_ptr<ForLoop> AddLoop(
      int64_t start_index, int64_t end_index, int64_t stride,
      absl::string_view suffix,
      UnrollMode unroll_mode = xla::llvm_ir::UnrollMode::kDefaultUnroll,
      bool prevent_vectorization = false);

  // Like the above, except that it defaults to a stride of one.
  std::unique_ptr<ForLoop> AddLoop(
      int64_t start_index, int64_t end_index, absl::string_view suffix,
      UnrollMode unroll_mode = xla::llvm_ir::UnrollMode::kDefaultUnroll,
      bool prevent_vectorization = false);

  // Add loops to iterate through the indices within the specified
  // shape. The returned index collects the induction variables of the
  // loops so that it will iterate through all coordinates within the
  // specified shape.
  //
  // E.g. if you pass in a 2x3 shape, you will get back an index with
  // two entries that are induction variables of the two loops that
  // will be added. That index will iterate through the 6 coordinates
  // within the shape. One possible order for that sequence would be:
  //
  //   (0,0), (0,1), (0,2), (1,0), (1,1), (1,2)
  IrArray::Index AddLoopsForShape(const Shape& shape, absl::string_view suffix);

  // Add a loop for each dimension in "dimensions". "suffix" is the
  // name suffix of the indvar and basic blocks in this new loop nest.
  //
  // The return value is an index with the induction variables. The
  // size equals the rank of shape and there is a null for each
  // dimension that is not in "dimensions".
  std::vector<llvm::Value*> AddLoopsForShapeOnDimensions(
      const Shape& shape, absl::Span<const int64_t> dimensions,
      absl::string_view suffix);

  // Emits a series of nested loops for iterating over an operand array. Loops
  // are constructed in major to minor dimension layout order. No loop is
  // emitted for the given 'dimension_to_skip'. The function returns an IrArray
  // index for the given operand_array containing the indvars of the loops. All
  // dimensions of the index are filled except for 'dimension_to_skip'.
  // name_suffix is the string to append to the names of LLVM constructs (eg,
  // basic blocks) constructed by this method.
  std::vector<llvm::Value*> EmitOperandArrayLoopNest(
      const llvm_ir::IrArray& operand_array, int64_t dimension_to_skip,
      absl::string_view name_suffix);

  // Convenience methods which return particular basic blocks of the outermost
  // or innermost loops. These methods return nullptr if no loops have been
  // added yet.
  llvm::BasicBlock* GetOuterLoopPreheaderBasicBlock() {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSllvm_irPSllvm_loopDTh mht_6(mht_6_v, 446, "", "./tensorflow/compiler/xla/service/llvm_ir/llvm_loop.h", "GetOuterLoopPreheaderBasicBlock");

    return outer_loop_preheader_bb_;
  }
  llvm::BasicBlock* GetOuterLoopExitBasicBlock() {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSllvm_irPSllvm_loopDTh mht_7(mht_7_v, 452, "", "./tensorflow/compiler/xla/service/llvm_ir/llvm_loop.h", "GetOuterLoopExitBasicBlock");
 return outer_loop_exit_bb_; }
  llvm::BasicBlock* GetInnerLoopBodyBasicBlock() {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSllvm_irPSllvm_loopDTh mht_8(mht_8_v, 456, "", "./tensorflow/compiler/xla/service/llvm_ir/llvm_loop.h", "GetInnerLoopBodyBasicBlock");
 return inner_loop_body_bb_; }

 private:
  void SetIndexType(llvm::Type* index_ty) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSllvm_irPSllvm_loopDTh mht_9(mht_9_v, 462, "", "./tensorflow/compiler/xla/service/llvm_ir/llvm_loop.h", "SetIndexType");

    index_type_ = index_ty == nullptr ? b_->getInt64Ty() : index_ty;
  }

  llvm::Constant* GetConstantWithIndexType(int64_t c) const {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSllvm_irPSllvm_loopDTh mht_10(mht_10_v, 469, "", "./tensorflow/compiler/xla/service/llvm_ir/llvm_loop.h", "GetConstantWithIndexType");

    return llvm::ConstantInt::get(index_type_, c);
  }

  // Human-friendly name of the loop nest.
  std::string name_;

  // The preheader and exit basic block of the outermost loop, or nullptr if no
  // loop has been added yet.
  llvm::BasicBlock* outer_loop_preheader_bb_;
  llvm::BasicBlock* outer_loop_exit_bb_;

  // The body basic block of the most-recently added loop, or nullptr if no loop
  // has been added yet.
  llvm::BasicBlock* inner_loop_body_bb_;

  llvm::IRBuilder<>* b_;

  llvm::Type* index_type_;
};

}  // namespace llvm_ir
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_LLVM_IR_LLVM_LOOP_H_
