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
class MHTracer_DTPStensorflowPScompilerPSxlaPSservicePScpuPSparallel_loop_emitterDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePScpuPSparallel_loop_emitterDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSxlaPSservicePScpuPSparallel_loop_emitterDTcc() {
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

#include "tensorflow/compiler/xla/service/cpu/parallel_loop_emitter.h"

#include "absl/strings/str_format.h"
#include "tensorflow/compiler/xla/service/llvm_ir/llvm_loop.h"
#include "tensorflow/compiler/xla/service/llvm_ir/llvm_util.h"

namespace xla {
namespace cpu {

ParallelLoopEmitter::ParallelLoopEmitter(
    const llvm_ir::ElementGenerator& target_element_generator,
    const llvm_ir::IrArray& target_array,
    const DynamicLoopBounds* dynamic_loop_bounds, llvm::IRBuilder<>* b)
    : LoopEmitter(target_element_generator, target_array, b),
      dynamic_loop_bounds_(dynamic_loop_bounds) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePScpuPSparallel_loop_emitterDTcc mht_0(mht_0_v, 199, "", "./tensorflow/compiler/xla/service/cpu/parallel_loop_emitter.cc", "ParallelLoopEmitter::ParallelLoopEmitter");
}

std::vector<llvm_ir::IrArray::Index>
ParallelLoopEmitter::EmitIndexAndSetExitBasicBlock(absl::string_view loop_name,
                                                   llvm::Type* index_type,
                                                   llvm::Value* base_index) {
   std::vector<std::string> mht_1_v;
   mht_1_v.push_back("loop_name: \"" + std::string(loop_name.data(), loop_name.size()) + "\"");
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePScpuPSparallel_loop_emitterDTcc mht_1(mht_1_v, 208, "", "./tensorflow/compiler/xla/service/cpu/parallel_loop_emitter.cc", "ParallelLoopEmitter::EmitIndexAndSetExitBasicBlock");

  CHECK_NE(index_type, nullptr);

  CHECK_EQ(base_index, nullptr)
      << "XLA CPU implementation of"
      << " ParallelLoopEmitter::EmitIndexAndSetExitBasicBlock doesn't support"
      << " base_index, but it was requested.";

  CHECK(!shape_.IsTuple());
  CHECK(!ShapeUtil::IsScalar(shape_));

  llvm_ir::ForLoopNest loop_nest(loop_name, b_);
  const int64_t num_dims = shape_.dimensions_size();
  std::vector<llvm::Value*> array_multi_index(num_dims);

  // Add loops from outer-most to inner-most dimensions.
  for (int i = LayoutUtil::MinorToMajor(shape_).size() - 1; i >= 0; --i) {
    const int64_t dimension = LayoutUtil::Minor(shape_.layout(), i);
    const int bounds_index = num_dims - 1 - i;
    if (bounds_index < dynamic_loop_bounds_->size()) {
      // Emit dynamic loop bounds for this dimension. Dynamic loop bounds
      // are read from ir function dynamic loop bounds argument.
      llvm::Value* start_index = (*dynamic_loop_bounds_)[bounds_index].first;
      llvm::Value* end_index = (*dynamic_loop_bounds_)[bounds_index].second;

      std::unique_ptr<llvm_ir::ForLoop> loop = loop_nest.AddLoop(
          /*suffix=*/absl::StrFormat("dim.%d", dimension), start_index,
          end_index);
      array_multi_index[dimension] = loop->GetIndVarValue();
    } else {
      // Emit static loop bounds for this dimension.
      std::unique_ptr<llvm_ir::ForLoop> loop = loop_nest.AddLoop(
          /*start_index=*/0,
          /*end_index=*/shape_.dimensions(dimension),
          /*suffix=*/absl::StrFormat("dim.%d", dimension));
      array_multi_index[dimension] = loop->GetIndVarValue();
    }
  }
  // Point IR builder at inner loop BB.
  llvm_ir::SetToFirstInsertPoint(loop_nest.GetInnerLoopBodyBasicBlock(), b_);

  // Set exit_bb_ to the exit block of the loop nest.
  exit_bb_ = loop_nest.GetOuterLoopExitBasicBlock();
  CHECK(exit_bb_ != nullptr);

  llvm_ir::IrArray::Index array_index(array_multi_index, shape_, index_type);
  return {array_index};
}

}  // namespace cpu
}  // namespace xla
