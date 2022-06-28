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

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_LLVM_IR_FUSED_IR_EMITTER_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_LLVM_IR_FUSED_IR_EMITTER_H_
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
class MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSllvm_irPSfused_ir_emitterDTh {
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
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSllvm_irPSfused_ir_emitterDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSllvm_irPSfused_ir_emitterDTh() {
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


#include <utility>

#include "absl/container/flat_hash_map.h"
#include "llvm/IR/Value.h"
#include "tensorflow/compiler/xla/service/elemental_ir_emitter.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/statusor.h"

namespace xla {

// FusedIrEmitter is used to generate code for fusion nodes.
//
// Unlike IrEmitter and its ilk, which directly create LLVM IR in an LLVM
// Module, FusedIrEmitter is better understood as "IR generator generator".
// FusedIrEmitter recursively creates a generator (a host function) which the
// compiler can invoke at a later time.  Invoking the generator emits LLVM IR
// that, when run, produces the value at a particular index of the output.
//
// After building this generator, the compiler creates a loop (or its moral
// equivalent, e.g. a GPU kernel) and calls the generator from within the loop.
// This generates code that produces each element of the output.
//
// This class handles both vanilla fusion and multi-output fusion.  In the MOF
// case, the fusion node ends with a kTuple instruction, and the generator
// created produces an LLVM struct with N elements, one for each element of the
// arrays in the tuple.  It follows that the arrays in the tuple must have the
// same length.
class FusedIrEmitter {
 public:
  using IndexedGenerator = llvm_ir::ElementGenerator;

  explicit FusedIrEmitter(ElementalIrEmitter& elemental_emitter)
      : elemental_emitter_(elemental_emitter) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSllvm_irPSfused_ir_emitterDTh mht_0(mht_0_v, 220, "", "./tensorflow/compiler/xla/service/llvm_ir/fused_ir_emitter.h", "FusedIrEmitter");
}

  void BindGenerator(const HloInstruction& instruction,
                     llvm_ir::ElementGenerator generator) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSllvm_irPSfused_ir_emitterDTh mht_1(mht_1_v, 226, "", "./tensorflow/compiler/xla/service/llvm_ir/fused_ir_emitter.h", "BindGenerator");

    indexed_generators_[&instruction] = std::move(generator);
  }

  // Returns the generator function for the given instruction.
  StatusOr<IndexedGenerator> GetGenerator(const HloInstruction& instruction);

  // Evaluates whether fusing 'producer' into 'consumer' might cause exponential
  // behavior in FusedIrEmitter. We currently can have exponential time/memory
  // requirements for emitting certain fusion kernels, in which case we don't
  // want to fuse.
  // TODO(b/119692968): Remove this once we have fixed our fusion emitter.
  static bool IsFusedIrEmitterInefficient(const HloInstruction& consumer,
                                          const HloInstruction& producer);

 private:
  StatusOr<IndexedGenerator> CreateGenerator(const HloInstruction& instruction);
  StatusOr<IndexedGenerator> DefaultAction(const HloInstruction& instruction);
  IndexedGenerator HandleConstant(const HloInstruction& constant);
  StatusOr<IndexedGenerator> HandleTuple(const HloInstruction& tuple);

  ElementalIrEmitter& elemental_emitter_;

  // Map from instructions to functions that generate code for the output
  // elements. If an instruction is a GetTupleElement instruction, the
  // instruction produces non-tuple result.
  absl::flat_hash_map<const HloInstruction*, IndexedGenerator>
      indexed_generators_;

  // Cache of generated values, lest we regenerate an element of a node with
  // multiple outgoing edges.
  // Use instruction and index values as the key.
  using ValueCacheKey =
      std::pair<const HloInstruction*, std::vector<llvm::Value*>>;
  absl::flat_hash_map<ValueCacheKey, llvm::Value*> value_cache_;
};

}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_LLVM_IR_FUSED_IR_EMITTER_H_
