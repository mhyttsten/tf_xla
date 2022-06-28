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

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_GPU_HLO_TO_IR_BINDINGS_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_GPU_HLO_TO_IR_BINDINGS_H_
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
class MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPShlo_to_ir_bindingsDTh {
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
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPShlo_to_ir_bindingsDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPShlo_to_ir_bindingsDTh() {
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


#include "absl/container/flat_hash_map.h"
#include "absl/types/span.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Value.h"
#include "tensorflow/compiler/xla/map_util.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/llvm_ir/ir_array.h"

namespace xla {
namespace gpu {

// This class encapsulates the bindings between HloInstructions and LLVM IR
// values that represent their addresses.
class HloToIrBindings {
 public:
  HloToIrBindings(llvm::IRBuilder<>* b, llvm::Module* llvm_module,
                  bool is_nested)
      : is_nested_(is_nested), b_(b), module_(llvm_module) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPShlo_to_ir_bindingsDTh mht_0(mht_0_v, 205, "", "./tensorflow/compiler/xla/service/gpu/hlo_to_ir_bindings.h", "HloToIrBindings");
}

  void EmitBasePointersForHlos(
      absl::Span<const HloInstruction* const> io_hlos,
      absl::Span<const HloInstruction* const> non_io_hlos);

  // Rebinds the given HLO to the LLVM IR value that represent its address.
  void BindHloToIrValue(const HloInstruction& hlo, llvm::Value* ir_value,
                        ShapeIndexView shape_index = {});

  // Unbinds all IR values that's defined in an LLVM function, e.g., function
  // arguments and stack variables. Global variables will be kept in bindings_.
  //
  // This method is called after emitting code for each top-level HLO. The local
  // IR values are out of scope at that point and should not be used.
  void UnbindAllLocalIrValues();

  // Returns whether `hlo` is bound to an LLVM IR value.
  bool BoundToIrValue(const HloInstruction& hlo) const {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPShlo_to_ir_bindingsDTh mht_1(mht_1_v, 226, "", "./tensorflow/compiler/xla/service/gpu/hlo_to_ir_bindings.h", "BoundToIrValue");

    return base_ptrs_.contains(&hlo);
  }

  llvm::Value* GetTempBufferBase() const {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPShlo_to_ir_bindingsDTh mht_2(mht_2_v, 233, "", "./tensorflow/compiler/xla/service/gpu/hlo_to_ir_bindings.h", "GetTempBufferBase");
 return temp_buffer_base_; }
  void SetTempBufferBase(llvm::Value* v) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPShlo_to_ir_bindingsDTh mht_3(mht_3_v, 237, "", "./tensorflow/compiler/xla/service/gpu/hlo_to_ir_bindings.h", "SetTempBufferBase");
 temp_buffer_base_ = v; }

  // A helper method that returns the base pointer of the IrArray containing the
  // output of "inst".at the given ShapeIndex.
  llvm::Value* GetBasePointer(const HloInstruction& hlo,
                              ShapeIndexView shape_index = {}) const {
    auto it = base_ptrs_.find(&hlo);
    CHECK(it != base_ptrs_.end()) << hlo.ToString();
    return it->second.element(shape_index);
  }

  // Returns the IrArray which contains the output of hlo.
  //
  // consumer is the HLO in which this IrArray is used -- we use this to (try
  // to) add metadata indicating that the array is invariant within consumer.
  //
  // To get the buffer into which hlo should write its own output, call
  // GetIrArray(hlo, hlo).
  llvm_ir::IrArray GetIrArray(const HloInstruction& hlo,
                              const HloInstruction& consumer,
                              const ShapeIndex& shape_index = {});

  std::string ToString() const;

 private:
  // Emits IR to resolve (possibly) recursive GetTupleElement instructions.
  llvm::Value* EmitGetTupleElement(const HloInstruction* gte,
                                   llvm::Value* base_ptr);

  // Returns an llvm typed ir representation of 'ir_value' based on 'hlo' shape.
  llvm::Value* GetTypedIrValue(const HloInstruction& hlo,
                               ShapeIndexView shape_index,
                               llvm::Value* ir_value);

  const bool is_nested_;

  llvm::IRBuilder<>* b_;
  llvm::Module* module_;

  // Stores the underlying llvm::IrArray for each HloInstruction.
  // For an instruction that generates multiple outputs, the root will be a
  // tuple shape. The IrArray for each element output is stored in the subnode
  // in the ShapeTree.
  absl::flat_hash_map<const HloInstruction*, ShapeTree<llvm::Value*>>
      base_ptrs_;

  // The address of the memory block that contains all temporary buffers.
  llvm::Value* temp_buffer_base_ = nullptr;
};

// Converts `ir_value` with type i8* to a typed LLVM Value* based on `shape`.
llvm::Value* CastToTypedValue(const Shape& shape, llvm::Value* ir_value,
                              llvm::IRBuilder<>* b);

}  // namespace gpu
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_GPU_HLO_TO_IR_BINDINGS_H_
