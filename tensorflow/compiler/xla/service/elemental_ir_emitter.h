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

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_ELEMENTAL_IR_EMITTER_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_ELEMENTAL_IR_EMITTER_H_
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
class MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSelemental_ir_emitterDTh {
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
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSelemental_ir_emitterDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSelemental_ir_emitterDTh() {
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


#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Value.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_instructions.h"
#include "tensorflow/compiler/xla/service/llvm_ir/ir_array.h"
#include "tensorflow/compiler/xla/service/llvm_ir/ir_builder_mixin.h"
#include "tensorflow/compiler/xla/service/llvm_ir/loop_emitter.h"
#include "tensorflow/compiler/xla/statusor.h"

namespace xla {

class ElementalIrEmitter : public IrBuilderMixin<ElementalIrEmitter> {
 public:
  using HloToElementGeneratorMap =
      absl::flat_hash_map<const HloInstruction*, llvm_ir::ElementGenerator>;

  ElementalIrEmitter(llvm::Module* module, llvm::IRBuilder<>* b)
      : b_(b), module_(module) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSelemental_ir_emitterDTh mht_0(mht_0_v, 211, "", "./tensorflow/compiler/xla/service/elemental_ir_emitter.h", "ElementalIrEmitter");
}

  virtual ~ElementalIrEmitter() = default;

  // Returns a function to generate an element of the output of `hlo`, given a
  // map of functions to generate elements of its operands.
  llvm_ir::ElementGenerator MakeElementGenerator(
      const HloInstruction* hlo,
      const HloToElementGeneratorMap& operand_to_generator);

  llvm::IRBuilder<>* b() {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSelemental_ir_emitterDTh mht_1(mht_1_v, 224, "", "./tensorflow/compiler/xla/service/elemental_ir_emitter.h", "b");
 return b_; }

  // builder() is for IrBuilderMixin.
  llvm::IRBuilder<>* builder() {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSelemental_ir_emitterDTh mht_2(mht_2_v, 230, "", "./tensorflow/compiler/xla/service/elemental_ir_emitter.h", "builder");
 return b_; }

  llvm::Module* module() {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSelemental_ir_emitterDTh mht_3(mht_3_v, 235, "", "./tensorflow/compiler/xla/service/elemental_ir_emitter.h", "module");
 return module_; }

 protected:
  virtual llvm_ir::IrArray::Index GetSourceIndexOfBitcast(
      const llvm_ir::IrArray::Index& index, const HloInstruction* hlo) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSelemental_ir_emitterDTh mht_4(mht_4_v, 242, "", "./tensorflow/compiler/xla/service/elemental_ir_emitter.h", "GetSourceIndexOfBitcast");

    return index.SourceIndexOfBitcast(hlo->shape(), hlo->operand(0)->shape(),
                                      b_);
  }

  virtual StatusOr<llvm::Value*> EmitFloatBinaryOp(const HloInstruction* op,
                                                   llvm::Value* lhs_value,
                                                   llvm::Value* rhs_value);

  virtual llvm::Value* EmitExtractReal(llvm::Value* value);
  virtual llvm::Value* EmitExtractImag(llvm::Value* value);

 private:
  virtual StatusOr<llvm::Value*> EmitUnaryOp(const HloInstruction* op,
                                             llvm::Value* operand_value);

  virtual StatusOr<llvm::Value*> EmitBinaryOp(const HloInstruction* op,
                                              llvm::Value* lhs_value,
                                              llvm::Value* rhs_value);

  virtual StatusOr<llvm::Value*> EmitIntegerUnaryOp(const HloInstruction* op,
                                                    llvm::Value* operand_value);

  virtual StatusOr<llvm::Value*> EmitFloatUnaryOp(const HloInstruction* op,
                                                  llvm::Value* operand_value);

  virtual StatusOr<llvm::Value*> EmitComplexUnaryOp(const HloInstruction* op,
                                                    llvm::Value* operand_value);

  llvm::Value* IsZero(llvm::Value* v);
  llvm::Value* IsIntMinDivisionOverflow(llvm::Value* lhs, llvm::Value* rhs);
  llvm::Value* GetZero(llvm::Type* type);
  llvm::Value* GetOne(llvm::Type* type);
  llvm::Value* GetIntSMin(llvm::Type* type);
  llvm::Value* GetMinusOne(llvm::Type* type);

  llvm::Value* EmitIntegerDivide(llvm::Value* lhs, llvm::Value* rhs,
                                 bool is_signed);
  llvm::Value* EmitIntegerRemainder(llvm::Value* lhs, llvm::Value* rhs,
                                    bool is_signed);
  llvm::Value* EmitIntegerPow(llvm::Value* lhs, llvm::Value* rhs,
                              bool is_signed);

  virtual StatusOr<llvm::Value*> EmitPredBinaryOp(const HloInstruction* op,
                                                  llvm::Value* lhs_value,
                                                  llvm::Value* rhs_value);

  virtual StatusOr<llvm::Value*> EmitIntegerBinaryOp(const HloInstruction* op,
                                                     llvm::Value* lhs_value,
                                                     llvm::Value* rhs_value,
                                                     bool is_signed);

  virtual StatusOr<llvm::Value*> EmitComplexBinaryOp(const HloInstruction* op,
                                                     llvm::Value* lhs_value,
                                                     llvm::Value* rhs_value);

  virtual llvm::Value* EmitFloatMax(llvm::Value* lhs_value,
                                    llvm::Value* rhs_value,
                                    absl::string_view name);

  virtual llvm::Value* EmitFloatMin(llvm::Value* lhs_value,
                                    llvm::Value* rhs_value,
                                    absl::string_view name);

  llvm::Value* EmitIntegralMax(llvm::Value* lhs_value, llvm::Value* rhs_value,
                               bool is_signed);

  llvm::Value* EmitIntegralMin(llvm::Value* lhs_value, llvm::Value* rhs_value,
                               bool is_signed);

  virtual StatusOr<llvm::Value*> EmitAtan2(PrimitiveType prim_type,
                                           llvm::Value* lhs, llvm::Value* rhs,
                                           absl::string_view name);

  virtual StatusOr<llvm::Value*> EmitLog(PrimitiveType prim_type,
                                         llvm::Value* value);

  virtual StatusOr<llvm::Value*> EmitSqrt(PrimitiveType prim_type,
                                          llvm::Value* value);

  virtual StatusOr<llvm::Value*> EmitCbrt(PrimitiveType prim_type,
                                          llvm::Value* value);

  virtual StatusOr<llvm::Value*> EmitRsqrt(PrimitiveType prim_type,
                                           llvm::Value* value);

  virtual StatusOr<llvm::Value*> EmitLog1p(PrimitiveType prim_type,
                                           llvm::Value* value);

  virtual StatusOr<llvm::Value*> EmitSin(PrimitiveType prim_type,
                                         llvm::Value* value);

  virtual StatusOr<llvm::Value*> EmitCos(PrimitiveType prim_type,
                                         llvm::Value* value);

  virtual StatusOr<llvm::Value*> EmitExp(PrimitiveType prim_type,
                                         llvm::Value* value,
                                         absl::string_view name);

  virtual StatusOr<llvm::Value*> EmitExpm1(PrimitiveType prim_type,
                                           llvm::Value* value);

  virtual StatusOr<llvm::Value*> EmitPow(PrimitiveType prim_type,
                                         llvm::Value* lhs, llvm::Value* rhs,
                                         absl::string_view name);

  virtual StatusOr<llvm::Value*> EmitTanh(PrimitiveType prim_type,
                                          llvm::Value* value);

  virtual StatusOr<llvm::Value*> EmitReducePrecision(const HloInstruction* hlo,
                                                     llvm::Value* x);

  virtual StatusOr<std::tuple<llvm::Value*, llvm::Value*, llvm::Value*>>
  EmitComplexAbsHelper(PrimitiveType prim_type, llvm::Value* operand_value,
                       bool return_sqrt);

  virtual StatusOr<llvm::Value*> EmitComplexAbs(PrimitiveType prim_type,
                                                llvm::Value* operand_value);

  virtual StatusOr<llvm::Value*> EmitSqrtComplexAbs(PrimitiveType prim_type,
                                                    llvm::Value* operand_value);
  virtual StatusOr<llvm::Value*> EmitRsqrtComplexAbs(
      PrimitiveType prim_type, llvm::Value* operand_value);

  virtual StatusOr<llvm::Value*> EmitComplexAdd(const HloInstruction* op,
                                                llvm::Value* lhs_value,
                                                llvm::Value* rhs_value);

  virtual StatusOr<llvm::Value*> EmitComplexSubtract(const HloInstruction* op,
                                                     llvm::Value* lhs_value,
                                                     llvm::Value* rhs_value);

  virtual StatusOr<llvm::Value*> EmitComplexMultiply(const HloInstruction* op,
                                                     llvm::Value* lhs_value,
                                                     llvm::Value* rhs_value);

  virtual StatusOr<llvm::Value*> EmitComplexDivide(const HloInstruction* op,
                                                   llvm::Value* lhs_value,
                                                   llvm::Value* rhs_value);

  virtual StatusOr<llvm::Value*> EmitComplexLog(const HloInstruction* op,
                                                llvm::Value* operand_value);

  virtual StatusOr<llvm::Value*> EmitComplexSqrt(const HloInstruction* op,
                                                 PrimitiveType prim_type,
                                                 llvm::Value* operand_value);

  virtual StatusOr<llvm::Value*> EmitComplexCbrt(const HloInstruction* op,
                                                 PrimitiveType prim_type,
                                                 llvm::Value* operand_value);

  virtual StatusOr<llvm::Value*> EmitComplexRsqrt(const HloInstruction* op,
                                                  PrimitiveType prim_type,
                                                  llvm::Value* operand_value);

  StatusOr<llvm::Value*> EmitAccumResult(
      absl::Span<llvm::Value* const> accumulator_addrs,
      llvm::ArrayRef<llvm::Type*> accumulator_types, bool is_variadic);

  // Composes a complex struct. imag may be nullptr for simple cast operations.
  llvm::Value* EmitComposeComplex(const HloInstruction* op, llvm::Value* real,
                                  llvm::Value* imag);

  // Emit `accumulator + lhs * rhs` for the given primitive type.
  llvm::Value* EmitMulAdd(llvm::Value* lhs, llvm::Value* rhs,
                          llvm::Value* accumulator,
                          xla::PrimitiveType primitive_type);

  // Identifier of the thread unique among all threads on the device
  virtual llvm::Value* EmitThreadId() {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSelemental_ir_emitterDTh mht_5(mht_5_v, 414, "", "./tensorflow/compiler/xla/service/elemental_ir_emitter.h", "EmitThreadId");
 return b_->getIntN(128, 0); }

  StatusOr<llvm::Value*> EmitElementalSelect(
      const HloInstruction* hlo,
      const HloToElementGeneratorMap& operand_to_generator,
      const llvm_ir::IrArray::Index& index);

  StatusOr<llvm::Value*> EmitElementalClamp(
      const HloInstruction* hlo,
      const HloToElementGeneratorMap& operand_to_generator,
      const llvm_ir::IrArray::Index& index);

  StatusOr<llvm::Value*> EmitElementalConcatenate(
      const HloInstruction* hlo,
      const HloToElementGeneratorMap& operand_to_generator,
      const llvm_ir::IrArray::Index& target_index);

  StatusOr<llvm::Value*> EmitElementalDynamicSlice(
      const HloInstruction* hlo,
      const HloToElementGeneratorMap& operand_to_generator,
      const llvm_ir::IrArray::Index& index);

  StatusOr<llvm::Value*> EmitElementalGather(
      const HloInstruction* hlo,
      const HloToElementGeneratorMap& operand_to_generator,
      const llvm_ir::IrArray::Index& index);

  StatusOr<llvm::Value*> EmitElementalDynamicUpdateSlice(
      const HloInstruction* hlo,
      const HloToElementGeneratorMap& operand_to_generator,
      const llvm_ir::IrArray::Index& index);

  StatusOr<llvm::Value*> EmitElementalPad(
      const HloInstruction* hlo,
      const HloToElementGeneratorMap& operand_to_generator,
      const llvm_ir::IrArray::Index& padded_index);

  StatusOr<llvm::Value*> EmitElementalDot(
      const HloInstruction* hlo,
      const HloToElementGeneratorMap& operand_to_generator,
      const llvm_ir::IrArray::Index& dot_result_index);

  virtual StatusOr<std::vector<llvm::Value*>> EmitThreadLocalCall(
      const HloComputation& callee, absl::Span<llvm::Value* const> parameters,
      absl::string_view name, bool is_reducer) = 0;

  StatusOr<llvm::Value*> EmitElementalMap(
      const HloMapInstruction* map_instr,
      absl::Span<llvm::Value* const> elemental_operands);

  StatusOr<llvm::Value*> EmitElementalReduceWindow(
      const HloReduceWindowInstruction* reduce_window,
      std::vector<llvm_ir::ElementGenerator> input_generators,
      std::vector<llvm_ir::ElementGenerator> initial_value_generators,
      const llvm_ir::IrArray::Index& index);

  StatusOr<llvm::Value*> EmitElementalReduce(
      const HloReduceInstruction* reduce,
      std::vector<llvm_ir::ElementGenerator> input_generators,
      std::vector<llvm_ir::ElementGenerator> initial_value_generators,
      const llvm_ir::IrArray::Index& index);

  virtual StatusOr<llvm::Value*> EmitConvolution(
      const HloInstruction* hlo,
      const HloToElementGeneratorMap& operand_to_generator,
      const llvm_ir::IrArray::Index& index);

  // Computes the complex power function, returns (a + i*b)^(c + i*d).
  StatusOr<llvm::Value*> EmitComplexPower(const HloInstruction* op,
                                          llvm::Value* a, llvm::Value* b,
                                          llvm::Value* c, llvm::Value* d);

  // Evaluates a polynomial using Horner's method.
  StatusOr<llvm::Value*> EvaluatePolynomial(
      llvm::Type* type, llvm::Value* x, absl::Span<const double> coefficients);

  virtual bool fast_min_max() = 0;

  llvm::IRBuilder<>* const b_;

  llvm::Module* module_;
};

}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_ELEMENTAL_IR_EMITTER_H_
