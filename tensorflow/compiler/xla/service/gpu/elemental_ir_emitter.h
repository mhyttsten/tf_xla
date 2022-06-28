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

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_GPU_ELEMENTAL_IR_EMITTER_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_GPU_ELEMENTAL_IR_EMITTER_H_
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
class MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPSelemental_ir_emitterDTh {
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
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPSelemental_ir_emitterDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPSelemental_ir_emitterDTh() {
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


#include <functional>
#include <string>
#include <utility>

#include "absl/types/span.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Value.h"
#include "tensorflow/compiler/xla/service/elemental_ir_emitter.h"
#include "tensorflow/compiler/xla/service/gpu/target_util.h"
#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_module_config.h"
#include "tensorflow/compiler/xla/service/llvm_ir/loop_emitter.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"

namespace xla {
namespace gpu {

class GpuElementalIrEmitter : public ElementalIrEmitter {
 public:
  // A NestedComputer computes an element of the output of the given computation
  // given a Span of its input elements.
  using NestedComputer = std::function<StatusOr<std::vector<llvm::Value*>>(
      const HloComputation&, absl::Span<llvm::Value* const>)>;

  GpuElementalIrEmitter(const HloModuleConfig& hlo_module_config,
                        llvm::Module* module, llvm::IRBuilder<>* b,
                        NestedComputer compute_nested);

 protected:
  llvm_ir::IrArray::Index GetSourceIndexOfBitcast(
      const llvm_ir::IrArray::Index& index, const HloInstruction* hlo) override;

  StatusOr<llvm::Value*> EmitFloatBinaryOp(const HloInstruction* op,
                                           llvm::Value* lhs_value,
                                           llvm::Value* rhs_value) override;

  StatusOr<llvm::Value*> EmitLog(PrimitiveType prim_type,
                                 llvm::Value* value) override;

  StatusOr<llvm::Value*> EmitLog1p(PrimitiveType prim_type,
                                   llvm::Value* value) override;

  StatusOr<llvm::Value*> EmitSin(PrimitiveType prim_type,
                                 llvm::Value* value) override;

  StatusOr<llvm::Value*> EmitCos(PrimitiveType prim_type,
                                 llvm::Value* value) override;

  StatusOr<llvm::Value*> EmitExp(PrimitiveType prim_type, llvm::Value* value,
                                 absl::string_view name) override;

  StatusOr<llvm::Value*> EmitExpm1(PrimitiveType prim_type,
                                   llvm::Value* value) override;

  StatusOr<llvm::Value*> EmitSqrt(PrimitiveType prim_type,
                                  llvm::Value* value) override;

  StatusOr<llvm::Value*> EmitRsqrt(PrimitiveType prim_type,
                                   llvm::Value* value) override;

  StatusOr<llvm::Value*> EmitPow(PrimitiveType prim_type, llvm::Value* lhs,
                                 llvm::Value* rhs,
                                 absl::string_view name) override;

  StatusOr<llvm::Value*> EmitAtan2(PrimitiveType prim_type, llvm::Value* lhs,
                                   llvm::Value* rhs,
                                   absl::string_view name) override;

  StatusOr<llvm::Value*> EmitTanh(PrimitiveType prim_type,
                                  llvm::Value* value) override;

  StatusOr<llvm::Value*> EmitComplexAbs(PrimitiveType prim_type,
                                        llvm::Value* value) override;

  StatusOr<std::vector<llvm::Value*>> EmitThreadLocalCall(
      const HloComputation& callee, absl::Span<llvm::Value* const> parameters,
      absl::string_view, bool /*is_reducer*/) override {
    return compute_nested_(callee, parameters);
  }

  llvm::Value* EmitThreadId() override;

  bool fast_min_max() override {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPSelemental_ir_emitterDTh mht_0(mht_0_v, 273, "", "./tensorflow/compiler/xla/service/gpu/elemental_ir_emitter.h", "fast_min_max");

    return hlo_module_config_.debug_options().xla_gpu_enable_fast_min_max();
  }

 private:
  // Emits IR for op, which must have opcode kPower.
  StatusOr<llvm::Value*> EmitPowerOp(const HloInstruction* op,
                                     llvm::Value* lhs_value,
                                     llvm::Value* rhs_value);

  // Emits IR to call an LLVM intrinsic of type [T] -> T.  Adjusts
  // callee_name according to T.  Returns the IR value that represents the
  // return value of the function.
  StatusOr<llvm::Value*> EmitLlvmIntrinsicMathCall(
      const std::string& callee_name, absl::Span<llvm::Value* const> operands,
      absl::Span<const PrimitiveType> input_types, PrimitiveType output_type);

  // Emits IR to call a device function of type [T] -> T.  Adjusts
  // callee_name according to T.  Returns the IR value that represents the
  // return value of the function.
  StatusOr<llvm::Value*> EmitDeviceMathCall(
      TargetDeviceFunctionID funcid, absl::Span<llvm::Value* const> operands,
      absl::Span<const PrimitiveType> input_types, PrimitiveType output_type,
      absl::string_view name = "");

  // Emits IR to call a function of type [T] -> T.  Does not munge callee_name.
  // Returns the IR value that represents the return value of the function.
  StatusOr<llvm::Value*> EmitMathCall(
      const std::string& callee_name, absl::Span<llvm::Value* const> operands,
      absl::Span<const PrimitiveType> input_types, PrimitiveType output_type,
      absl::string_view name = "");

  const HloModuleConfig& hlo_module_config_;

  NestedComputer compute_nested_;
};

}  // namespace gpu
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_GPU_ELEMENTAL_IR_EMITTER_H_
