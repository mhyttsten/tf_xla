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

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_CPU_IR_FUNCTION_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_CPU_IR_FUNCTION_H_
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
class MHTracer_DTPStensorflowPScompilerPSxlaPSservicePScpuPSir_functionDTh {
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
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePScpuPSir_functionDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSxlaPSservicePScpuPSir_functionDTh() {
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


#include "absl/types/span.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Value.h"
#include "tensorflow/compiler/xla/service/cpu/ir_emission_utils.h"
#include "tensorflow/compiler/xla/service/hlo_module_config.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/compiler/xla/types.h"

namespace xla {
namespace cpu {

// IrFunction creates and encapsulates an llvm::Function, exposing methods to
// emitters for function and function argument access.
// The llvm::Function is created with the standard function signature
// used in the XLA CPU backend (see ir_function.cc for argument details).
// In addition IrFunction saves the callers IR insert point during construction,
// and restores it after destruction.
//
// Example usage:
//
//    // Create and initialize new IrFunction.
//    std::unique_ptr<IrFunction> compute_function(new IrFunction(...));
//    // Emit IR for function body using IrFunction helper methods.
//    ...
//    // Store reference to llvm::Function for future invocation.
//    ir_functions.push_back(compute_function.function());
//    // Delete IrFunction (finalizes IR function and restores caller insertion
//    // point).
//    compute_function.reset();
//

class IrFunction {
 public:
  IrFunction(const std::string& function_name,
             llvm::Function::LinkageTypes linkage,
             const HloModuleConfig& module_config, llvm::Module* llvm_module,
             llvm::IRBuilder<>* b, int64_t num_dynamic_loop_bounds);
  ~IrFunction();

  // Emit IR to read and return the set of IR values representing the dynamic
  // loop bounds argument of this function. These bounds delimit the subset
  // of the output that will be written by the computation's root instruction at
  // runtime. This is used for parallel computations, where a single computation
  // is partitioned into N calls to a function with parallel loop bounds, and
  // then called N times in parallel with loop bounds limiting each call to
  // producing 1/N of the output.
  //
  // Each element in returned vector is a pair of ir values representing the
  // loop bounds for a specific dimension, where the first element of the pair
  // is the dimension start index, and the second element of the pair is the
  // dimension limit.
  //
  // EX: [dimension_i_index_start_ir_value, // dimension_i_index_limit_ir_value]
  DynamicLoopBounds GetDynamicLoopBounds();

  // Returns the encapculated llvm::Function.
  llvm::Function* function() {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePScpuPSir_functionDTh mht_0(mht_0_v, 247, "", "./tensorflow/compiler/xla/service/cpu/ir_function.h", "function");
 return function_; }

  // Get the llvm::Value* that represents this functions "retval" argument.
  llvm::Argument* result_arg() {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePScpuPSir_functionDTh mht_1(mht_1_v, 253, "", "./tensorflow/compiler/xla/service/cpu/ir_function.h", "result_arg");
 return result_arg_; }

  // Get the xla::ExecutableRunOptions that represents this functions
  // "run_options" argument.
  llvm::Value* exec_run_options_arg() {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePScpuPSir_functionDTh mht_2(mht_2_v, 260, "", "./tensorflow/compiler/xla/service/cpu/ir_function.h", "exec_run_options_arg");
 return exec_run_options_arg_; }

  // Get the llvm::Value* that represents this functions parameters argument.
  llvm::Value* parameters_arg() {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePScpuPSir_functionDTh mht_3(mht_3_v, 266, "", "./tensorflow/compiler/xla/service/cpu/ir_function.h", "parameters_arg");
 return parameters_arg_; }

  // Get the llvm::Value* that represents this functions "buffer_table"
  // argument.
  llvm::Value* buffer_table_arg() {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePScpuPSir_functionDTh mht_4(mht_4_v, 273, "", "./tensorflow/compiler/xla/service/cpu/ir_function.h", "buffer_table_arg");
 return buffer_table_arg_; }

  // Get the llvm::Value* that represents this functions "prof_counters"
  // argument.
  llvm::Value* profile_counters_arg() {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePScpuPSir_functionDTh mht_5(mht_5_v, 280, "", "./tensorflow/compiler/xla/service/cpu/ir_function.h", "profile_counters_arg");
 return profile_counters_arg_; }

  // Get the llvm::BasicBlock* that contains this function's "ret" instruction.
  llvm::BasicBlock* return_block() {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePScpuPSir_functionDTh mht_6(mht_6_v, 286, "", "./tensorflow/compiler/xla/service/cpu/ir_function.h", "return_block");
 return return_block_; }

  // Get the llvm::Value* that represents this function's "status" argument.
  llvm::Value* status_arg() {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePScpuPSir_functionDTh mht_7(mht_7_v, 292, "", "./tensorflow/compiler/xla/service/cpu/ir_function.h", "status_arg");
 return status_arg_; }

 private:
  // Initialize an llvm::Function with standard signature based on arguments.
  void Initialize(const std::string& function_name,
                  llvm::Function::LinkageTypes linkage,
                  const HloModuleConfig& module_config);

  // Emit ir to read and return the ir value for the dynamic loop bound at
  // 'offset' from the "dynamic_loop_bounds" argument of this function.
  llvm::Value* GetDynamicLoopBound(int64_t offset);

  llvm::IRBuilder<>* b_;
  llvm::Module* llvm_module_;
  llvm::IRBuilder<>::InsertPointGuard caller_insert_point_guard_;

  int64_t num_dynamic_loop_bounds_ = 0;
  // Encapsulated llvm::Function.
  llvm::Function* function_;
  // Function argument IR values.
  llvm::Argument* result_arg_;
  llvm::Value* exec_run_options_arg_;
  llvm::Value* parameters_arg_;
  llvm::Value* buffer_table_arg_;
  llvm::Value* dynamic_loop_bounds_arg_ = nullptr;
  llvm::Value* profile_counters_arg_;
  llvm::Value* status_arg_;
  // Basic block containing return.
  llvm::BasicBlock* return_block_;
};

// Returns arguments in `arguments` encoded as a single buffer, suitable for a
// function call.
llvm::Value* EncodeArrayFunctionArguments(
    absl::Span<llvm::Value* const> arguments, absl::string_view name,
    llvm::IRBuilder<>* b);

// Returns an array of compute function call argument ir values.
std::vector<llvm::Value*> GetArrayFunctionCallArguments(
    absl::Span<llvm::Value* const> parameter_addresses, llvm::IRBuilder<>* b,
    absl::string_view name, llvm::Value* return_value_buffer,
    llvm::Value* exec_run_options_arg, llvm::Value* buffer_table_arg,
    llvm::Value* status_arg, llvm::Value* profile_counters_arg);

// Emits a call to a runtime fork/join function which dispatches parallel
// calls to 'parallel_function' (and joins threads before returning).
Status EmitCallToParallelForkJoin(
    const std::vector<llvm::Value*>& arguments, const Shape& shape,
    const std::vector<int64_t>& dimension_partition_counts,
    llvm::IRBuilder<>* b, llvm::Function* parallel_function,
    const std::string& name);

}  // namespace cpu
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_CPU_IR_FUNCTION_H_
