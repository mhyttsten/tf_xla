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

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_GPU_IR_EMITTER_NESTED_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_GPU_IR_EMITTER_NESTED_H_
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
class MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPSir_emitter_nestedDTh {
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
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPSir_emitter_nestedDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPSir_emitter_nestedDTh() {
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


#include "llvm/IR/Function.h"
#include "tensorflow/compiler/xla/service/gpu/ir_emitter.h"

namespace xla {
namespace gpu {

// Emits LLVM IR for a "nested computation" into a non-kernel device function.
//
// This is used to emit code for HloComputations that don't require a separate
// kernel call.  For example, IrEmitterNested is used to emit code for a kReduce
// HLO's elementwise reduction computation.  Notably, IrEmitterNested is *not*
// used to emit code for fusion nodes -- fusion nodes use FusedIrEmitter, which
// is a different beast altogether.
//
// IrEmitterNested generates a non-kernel function with the following
// parameters:
//
//   - N pointers to the buffers of each of the N parameters to the computation,
//   - a pointer to the output buffer of the computation, and
//   - a pointer to the top-level temp buffer.
//
class IrEmitterNested : public IrEmitter {
 public:
  // Constructs IrEmitterNested and emits code for global constants, and also
  // populates related information to 'ir_emitter_context_' for large-constant
  // initializations. Large constants don't get initializers in the generated
  // code and so must be initialized by XLA. The value of these constants will
  // be stored in 'content'. Constants with initializers in the generated code
  // will have empty 'content'.
  //
  // The allocation index for these constants will always be -1 (i.e. doesn't
  // correspond to any allocation).
  static StatusOr<std::unique_ptr<IrEmitterNested>> Create(
      const HloModuleConfig& hlo_module_config,
      const HloComputation& nested_computation,
      IrEmitterContext* ir_emitter_context);

  IrEmitterNested(const IrEmitterNested&) = delete;
  IrEmitterNested& operator=(const IrEmitterNested&) = delete;

  // Overrides the default empty implementation. Binds the given instruction
  // "parameter" with the parameter of the IR function.
  Status HandleParameter(HloInstruction* parameter) override;

  llvm::Function* GetEmittedFunction() const {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPSir_emitter_nestedDTh mht_0(mht_0_v, 232, "", "./tensorflow/compiler/xla/service/gpu/ir_emitter_nested.h", "GetEmittedFunction");
 return emitted_function_; }

  // Generate the code for the computation passed in the constructor.
  Status CodegenNestedComputation();

 protected:
  Status EmitTargetElementLoop(
      const HloInstruction& hlo,
      const llvm_ir::ElementGenerator& body_emitter) override;

 private:
  // Constructs an LLVM IR emitter for a nested HLO computation. `function` is
  // the containing IR function this emitter produces IR to. See
  // IrEmitter::IrEmitter for the meanings of other arguments.
  IrEmitterNested(const HloModuleConfig& hlo_module_config,
                  const HloComputation& nested_computation,
                  IrEmitterContext* ir_emitter_context);

  // Emits constants to generated LLVM IR, and also populates related
  // information to 'ir_emitter_context_' for large-constant initializations.
  Status EmitConstants(const HloComputation& computation);

  const HloComputation& nested_computation_;
  llvm::Function* emitted_function_;
};

}  // namespace gpu
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_GPU_IR_EMITTER_NESTED_H_
