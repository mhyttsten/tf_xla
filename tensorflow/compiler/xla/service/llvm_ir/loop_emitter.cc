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
class MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSllvm_irPSloop_emitterDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSllvm_irPSloop_emitterDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSllvm_irPSloop_emitterDTcc() {
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

#include "tensorflow/compiler/xla/service/llvm_ir/loop_emitter.h"

#include <memory>
#include <utility>

#include "absl/strings/str_format.h"
#include "tensorflow/compiler/xla/service/llvm_ir/llvm_loop.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/protobuf.h"

namespace xla {
namespace llvm_ir {

LoopEmitter::LoopEmitter(const BodyEmitter& body_emitter, const Shape& shape,
                         llvm::IRBuilder<>* b)
    : body_emitter_(body_emitter), shape_(shape), b_(b) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSllvm_irPSloop_emitterDTcc mht_0(mht_0_v, 204, "", "./tensorflow/compiler/xla/service/llvm_ir/loop_emitter.cc", "LoopEmitter::LoopEmitter");
}

LoopEmitter::LoopEmitter(const BodyEmitter& body_emitter, const Shape& shape,
                         std::vector<llvm::Value*> dynamic_dims,
                         llvm::IRBuilder<>* b)
    : LoopEmitter::LoopEmitter(body_emitter, shape, b) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSllvm_irPSloop_emitterDTcc mht_1(mht_1_v, 212, "", "./tensorflow/compiler/xla/service/llvm_ir/loop_emitter.cc", "LoopEmitter::LoopEmitter");

  CHECK_EQ(dynamic_dims.size(), shape_.dimensions_size());
  dynamic_dims_ = std::move(dynamic_dims);
}

LoopEmitter::LoopEmitter(const ElementGenerator& target_element_generator,
                         const IrArray& target_array, llvm::IRBuilder<>* b)
    : body_emitter_(MakeBodyEmitter(target_element_generator, {target_array}, b,
                                    /*is_tuple=*/false)),
      shape_(target_array.GetShape()),
      b_(b) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSllvm_irPSloop_emitterDTcc mht_2(mht_2_v, 225, "", "./tensorflow/compiler/xla/service/llvm_ir/loop_emitter.cc", "LoopEmitter::LoopEmitter");
}

LoopEmitter::LoopEmitter(const ElementGenerator& target_element_generator,
                         absl::Span<const IrArray> target_arrays,
                         llvm::IRBuilder<>* b)
    : body_emitter_(MakeBodyEmitter(target_element_generator, target_arrays, b,
                                    /*is_tuple=*/true)),
      shape_(target_arrays[0].GetShape()),
      b_(b) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSllvm_irPSloop_emitterDTcc mht_3(mht_3_v, 236, "", "./tensorflow/compiler/xla/service/llvm_ir/loop_emitter.cc", "LoopEmitter::LoopEmitter");

  // Sanity check: In multi-output fusion, all shapes produced must have the
  // same dimensions.
  for (const IrArray& array : target_arrays) {
    CHECK(ShapeUtil::SameDimensions(shape_, array.GetShape()))
        << ": '" << shape_.ShortDebugString() << "' does not match '"
        << array.GetShape().ShortDebugString() << "'";
  }
}

BodyEmitter MakeBodyEmitter(const ElementGenerator& target_element_generator,
                            absl::Span<IrArray const> target_arrays,
                            llvm::IRBuilder<>* b, bool is_tuple) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSllvm_irPSloop_emitterDTcc mht_4(mht_4_v, 251, "", "./tensorflow/compiler/xla/service/llvm_ir/loop_emitter.cc", "MakeBodyEmitter");

  std::vector<IrArray> target_arrays_vec(target_arrays.begin(),
                                         target_arrays.end());
  if (!is_tuple) {
    CHECK_EQ(target_arrays.size(), 1);
    return [=](const llvm_ir::IrArray::Index array_index) -> Status {
      // Convert target_element_generator to a BodyEmitter.
      TF_ASSIGN_OR_RETURN(llvm::Value * target_element,
                          target_element_generator(array_index));
      target_arrays_vec[0].EmitWriteArrayElement(array_index, target_element,
                                                 b);
      return Status::OK();
    };
  }

  return [=](const llvm_ir::IrArray::Index array_index) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSllvm_irPSloop_emitterDTcc mht_5(mht_5_v, 269, "", "./tensorflow/compiler/xla/service/llvm_ir/loop_emitter.cc", "lambda");

    TF_ASSIGN_OR_RETURN(llvm::Value * target_element,
                        target_element_generator(array_index));
    CHECK(target_element->getType()->isStructTy())
        << "This BodyEmitter is for multi-output, but target element "
           "generator does not produce values of struct type.";
    CHECK_EQ(target_element->getType()->getStructNumElements(),
             target_arrays_vec.size());

    for (int64_t i = 0; i < target_arrays_vec.size(); ++i) {
      target_arrays_vec[i].EmitWriteArrayElement(
          array_index, b->CreateExtractValue(target_element, i), b);
    }
    return Status::OK();
  };
}

IrArray::Index LoopEmitter::EmitStaticIndex(ForLoopNest* loop_nest,
                                            llvm::Type* index_type) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSllvm_irPSloop_emitterDTcc mht_6(mht_6_v, 290, "", "./tensorflow/compiler/xla/service/llvm_ir/loop_emitter.cc", "LoopEmitter::EmitStaticIndex");

  // Create loop nest with one for-loop for each dimension of the target shape.
  // Loops are added from outermost to innermost order with the ForLoopNest
  // class so emit loops in order from most-major dimension down to most-minor
  // dimension (of the target shape).
  std::vector<llvm::Value*> array_multi_index(shape_.dimensions_size());
  for (int i = 0; i < LayoutUtil::MinorToMajor(shape_).size(); ++i) {
    int64_t dimension = LayoutUtil::Major(shape_.layout(), i);
    std::unique_ptr<ForLoop> loop = loop_nest->AddLoop(
        /*start_index=*/0,
        /*end_index=*/shape_.dimensions(dimension),
        /*suffix=*/absl::StrFormat("dim.%d", dimension));
    array_multi_index[dimension] = loop->GetIndVarValue();
  }
  return IrArray::Index(array_multi_index, shape_, index_type);
}

IrArray::Index LoopEmitter::EmitDynamicIndex(ForLoopNest* loop_nest,
                                             llvm::Type* index_type) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSllvm_irPSloop_emitterDTcc mht_7(mht_7_v, 311, "", "./tensorflow/compiler/xla/service/llvm_ir/loop_emitter.cc", "LoopEmitter::EmitDynamicIndex");

  CHECK_EQ(shape_.is_dynamic(), true);
  // Create loop nest with one for-loop for each dynamic dimensions.
  // Loops are added from outermost to innermost order with the ForLoopNest
  // class so emit loops in order from most-major dimension down to most-minor
  // dimension (of the target shape).
  std::vector<llvm::Value*> array_multi_index(shape_.dimensions_size());
  for (int i = 0; i < LayoutUtil::MinorToMajor(shape_).size(); ++i) {
    int64_t dimension = LayoutUtil::Major(shape_.layout(), i);
    std::unique_ptr<ForLoop> loop = loop_nest->AddLoop(
        /*suffix=*/absl::StrFormat("dim.%d", dimension),
        /*start_index=*/llvm::ConstantInt::get(index_type, 0),
        /*end_index=*/dynamic_dims_[dimension]);
    array_multi_index[dimension] = loop->GetIndVarValue();
  }
  return IrArray::Index(array_multi_index, shape_, index_type);
}

std::vector<IrArray::Index> LoopEmitter::EmitIndexAndSetExitBasicBlock(
    absl::string_view loop_name, llvm::Type* index_type,
    llvm::Value* base_index) {
   std::vector<std::string> mht_8_v;
   mht_8_v.push_back("loop_name: \"" + std::string(loop_name.data(), loop_name.size()) + "\"");
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSllvm_irPSloop_emitterDTcc mht_8(mht_8_v, 335, "", "./tensorflow/compiler/xla/service/llvm_ir/loop_emitter.cc", "LoopEmitter::EmitIndexAndSetExitBasicBlock");

  CHECK_NE(index_type, nullptr);
  CHECK_EQ(base_index, nullptr)
      << "XLA CPU implementation of"
      << " LoopEmitter::EmitIndexAndSetExitBasicBlock doesn't support"
      << " base_index, but it was requested.";

  if (ShapeUtil::IsScalar(shape_)) {
    // No loop needed, so set exit_bb_ to nullptr.
    exit_bb_ = nullptr;
    return {IrArray::Index(index_type)};
  }

  ForLoopNest loop_nest(loop_name, b_);

  IrArray::Index array_index = dynamic_dims_.empty()
                                   ? EmitStaticIndex(&loop_nest, index_type)
                                   : EmitDynamicIndex(&loop_nest, index_type);

  // Set IR builder insertion point to the loop body basic block of the
  // innermost loop.
  llvm::BasicBlock* innermost_body_bb = loop_nest.GetInnerLoopBodyBasicBlock();
  b_->SetInsertPoint(innermost_body_bb,
                     innermost_body_bb->getFirstInsertionPt());

  // Set exit_bb_ to the exit block of the loop nest.
  exit_bb_ = loop_nest.GetOuterLoopExitBasicBlock();
  CHECK_NOTNULL(exit_bb_);

  return {array_index};
}

Status LoopEmitter::EmitLoop(absl::string_view loop_name,
                             llvm::Type* index_type) {
   std::vector<std::string> mht_9_v;
   mht_9_v.push_back("loop_name: \"" + std::string(loop_name.data(), loop_name.size()) + "\"");
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSllvm_irPSloop_emitterDTcc mht_9(mht_9_v, 372, "", "./tensorflow/compiler/xla/service/llvm_ir/loop_emitter.cc", "LoopEmitter::EmitLoop");

  if (index_type == nullptr) {
    index_type = b_->getInt64Ty();
  }

  for (const IrArray::Index& array_index :
       EmitIndexAndSetExitBasicBlock(loop_name, index_type,
                                     /*base_index*/ nullptr)) {
    TF_RETURN_IF_ERROR(body_emitter_(array_index));
  }

  // Set the insertion point of b_ to the loop exit, so that
  // code emitted for later instructions will be correctly placed.
  if (exit_bb_ != nullptr) {
    b_->SetInsertPoint(exit_bb_);
  }
  return Status::OK();
}

}  // namespace llvm_ir
}  // namespace xla
