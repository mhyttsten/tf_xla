/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_COMPILER_MLIR_LITE_QUANTIZATION_QUANTIZATION_CONTEXT_H_
#define TENSORFLOW_COMPILER_MLIR_LITE_QUANTIZATION_QUANTIZATION_CONTEXT_H_
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
class MHTracer_DTPStensorflowPScompilerPSmlirPSlitePSquantizationPSquantization_contextDTh {
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
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePSquantizationPSquantization_contextDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSmlirPSlitePSquantizationPSquantization_contextDTh() {
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


#include "llvm/ADT/DenseMap.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/Dialect/Quant/QuantOps.h"  // from @llvm-project
#include "mlir/Dialect/Quant/QuantTypes.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/Operation.h"  // from @llvm-project
#include "mlir/IR/Value.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/lite/quantization/device_target.h"
#include "tensorflow/compiler/mlir/lite/quantization/quantization_utils.h"

namespace mlir {
namespace quant {

static bool EmptyParams(QuantParams p) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePSquantizationPSquantization_contextDTh mht_0(mht_0_v, 202, "", "./tensorflow/compiler/mlir/lite/quantization/quantization_context.h", "EmptyParams");
 return p == quant::QuantizedType(); }

// The state for each op result during the quantization parameters propagation.
struct QuantState {
  // Quantization parameters propagated to an op result.
  QuantParams params;
  // A flag indicates this state (the params) shouldn't be changed after it is
  // initialized. This flag will be set to true if the quantization parameters
  // are from the quantization-aware training.
  const bool immutable;

  bool IsEmpty() {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePSquantizationPSquantization_contextDTh mht_1(mht_1_v, 216, "", "./tensorflow/compiler/mlir/lite/quantization/quantization_context.h", "IsEmpty");
 return EmptyParams(params); }
};

// The state for rescaling the propagated quantization parameters. This can be
// on the input side to satisfy the constraint of previous operation, or on the
// output side to satisfy the constraint of the next operation.
struct RequantizeState {
  // Sometimes, we have to "requantize" the quantization result to satisfy all
  // the constraints. The "requantize" can happen either on the input or output
  // of the quantization result.
  enum RequantizePosition {
    NO_REQUANTIZE,
    ON_INPUT,
    ON_OUTPUT
  } pos = NO_REQUANTIZE;

  // Quantization parameters will be used to add the requantize ops.
  QuantParams params;
};

// This class manages all the intermediate quantization states.
class QuantizeContext {
 public:
  QuantizeContext(FuncOp func, const DeviceTarget &spec);

  // Returns all the quant region ops.
  std::vector<quant::QuantizeRegionOp> GetAllOps();

  // For each quant region op, propagates its quantization parameters according
  // to the kernel specification and also returns the adjacent quant region ops
  // which get the new quantization parameters propagated.
  LogicalResult Handle(quant::QuantizeRegionOp op,
                       llvm::SmallVectorImpl<Operation *> *new_items,
                       bool *changed);

  // Updates the port quantization specifications of all the quant region ops
  // with the propagation results.
  LogicalResult Finalize();

  // Dumps the states stores in the state manager.
  void DumpStates(QuantizeRegionOp current_op = {});

  // Update the quantization parameter for certain result of the op. By this
  // method, the quantization parameter is propagated to all the users of the
  // result as well.
  bool SetResultParams(Operation *op, int index, QuantParams params) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePSquantizationPSquantization_contextDTh mht_2(mht_2_v, 264, "", "./tensorflow/compiler/mlir/lite/quantization/quantization_context.h", "SetResultParams");

    return states_manager_.SetResultParams(op, index, params);
  }

  // Update the quantization parameter for certain operand of the op. By this
  // method, the quantization parameter is propagated to the defining op of
  // operand as well.
  bool SetOperandParams(Operation *op, int index, QuantParams params) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePSquantizationPSquantization_contextDTh mht_3(mht_3_v, 274, "", "./tensorflow/compiler/mlir/lite/quantization/quantization_context.h", "SetOperandParams");

    return states_manager_.SetOperandParams(op, index, params);
  }

  // Return the quantization parameter of certain result of the op.
  QuantParams GetResultParams(Operation *op, int index) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePSquantizationPSquantization_contextDTh mht_4(mht_4_v, 282, "", "./tensorflow/compiler/mlir/lite/quantization/quantization_context.h", "GetResultParams");

    return states_manager_.GetResultParams(op, index);
  }

  // Return the quantization parameter of certain operand of the op.
  QuantParams GetOperandParams(Operation *op, int index) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePSquantizationPSquantization_contextDTh mht_5(mht_5_v, 290, "", "./tensorflow/compiler/mlir/lite/quantization/quantization_context.h", "GetOperandParams");

    return states_manager_.GetOperandParams(op, index);
  }

  // Return the signature of the op.
  KernelSpecs::Signature GetSignature(QuantizeRegionOp op);

  // A heuristic to get quantization parameters satisfies the same scale
  // constraints:
  // - If there are immutable states,
  //   - use the single input, or,
  //   - use the single output, or,
  //   - use the first one in the collection,
  // - use the single input if it is ready, or,
  // - use the single output if it is ready, or,
  // - use the first ready one in the collection.
  QuantParams GetQuantParamsForSameScaleConstraint(Operation *op);

  // Propagate `params` to all the quantizable port of the `op`. The adjacent
  // ops, which have the parameters propagated to, are collected by `new_items`,
  // so they can be added to the working queue. `changed` is set to true if
  // there are any new elements being added to `new_items`.
  LogicalResult PropagateQuantParams(Operation *op, const QuantParams params,
                                     AdjacentOperations *new_items,
                                     bool *changed);

 private:
  class StatesManager {
   public:
    // Sets the quantization parameters of the constant result according to its
    // content.
    //
    // Always returns true.
    bool SetConstantResultParams(Operation *op);

    // Sets the quantization parameters of the result to a fixed value. If any
    // quantization parameters have been propagated, a `requantize` will happen
    // on the input of propagated quantization.
    //
    // Returns true, if the users of the result needs to be added to the
    // worklist.
    bool SetResultParams(Operation *op, int index, QuantParams params);

    // Sets the quantization parameters of the operand to a fixed value. If any
    // quantization parameters have been propagated, a `requantize` will happen
    // on the output of propagated quantization.
    //
    // Returns true, if the defining op of the operand needs to be added to the
    // worklist.
    bool SetOperandParams(Operation *op, int index, QuantParams params);

    // Returns the quantization parameters of the index-th result of the op.
    QuantParams GetResultParams(Operation *op, int index) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePSquantizationPSquantization_contextDTh mht_6(mht_6_v, 345, "", "./tensorflow/compiler/mlir/lite/quantization/quantization_context.h", "GetResultParams");

      return states_[result_states_[{op, index}]].params;
    }

    // Returns the quantization parameters of the index-th operand of the op.
    QuantParams GetOperandParams(Operation *op, int index) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePSquantizationPSquantization_contextDTh mht_7(mht_7_v, 353, "", "./tensorflow/compiler/mlir/lite/quantization/quantization_context.h", "GetOperandParams");

      return states_[operand_states_[{op, index}]].params;
    }

   private:
    friend class QuantizeContext;

    // Uses the type of `val` to set the initial state of the index-th result if
    // `as_result` is true or index-th operand if `as_result` is false. The
    // state is immutable if the type is a quantized type. Returns the index of
    // this new state in the state vector.
    int InitializeState(quant::QuantizeRegionOp op, int index, bool as_result);

    // Sets the state of the index-th operand of the op. If this operand is
    // cached, uses the cached result without creating new entry in the state
    // vector. Otherwise, allocate a new entry in the state vector.
    void InitializeOperandState(quant::QuantizeRegionOp op, int index,
                                llvm::DenseMap<Value, int> *cache);

    // Sets the state of the index-th result of the op. If this result is
    // cached, uses the cached result without creating new entry in the state
    // vector. Otherwise, allocate a new entry in the state vector.
    void InitializeResultState(quant::QuantizeRegionOp op, int index,
                               llvm::DenseMap<Value, int> *cache);

    // Returns the state of the index-th operand of the op.
    QuantState &GetOperandQuantState(Operation *op, int index) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePSquantizationPSquantization_contextDTh mht_8(mht_8_v, 382, "", "./tensorflow/compiler/mlir/lite/quantization/quantization_context.h", "GetOperandQuantState");

      return states_[operand_states_[{op, index}]];
    }

    // Returns the state of the index-th result of the op.
    QuantState &GetResultQuantState(Operation *op, int index) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePSquantizationPSquantization_contextDTh mht_9(mht_9_v, 390, "", "./tensorflow/compiler/mlir/lite/quantization/quantization_context.h", "GetResultQuantState");

      return states_[result_states_[{op, index}]];
    }

    // Returns the state of the index-th operand of the op.
    RequantizeState &GetOperandRequantizeState(Operation *op, int index) {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePSquantizationPSquantization_contextDTh mht_10(mht_10_v, 398, "", "./tensorflow/compiler/mlir/lite/quantization/quantization_context.h", "GetOperandRequantizeState");

      return rescale_states_[operand_states_[{op, index}]];
    }

    // Returns the state of the index-th result of the op.
    RequantizeState &GetResultRequantizeState(Operation *op, int index) {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePSquantizationPSquantization_contextDTh mht_11(mht_11_v, 406, "", "./tensorflow/compiler/mlir/lite/quantization/quantization_context.h", "GetResultRequantizeState");

      return rescale_states_[result_states_[{op, index}]];
    }

   private:
    // This is used to identify an operand or result of an op. The second
    // element of this pair is the index of the operand or result.
    using OpValue = std::pair<mlir::Operation *, int>;

    // The vector contains all the quantization parameters propagated from the
    // defining operations of the value, or from the quantization aware
    // training.
    std::vector<QuantState> states_;

    // The map contains all the quantization parameters which are required to
    // satisfy the same operands and results constraint. The keys of this map
    // are the values from `operand_states_` and `result_state_`.
    std::unordered_map<int, RequantizeState> rescale_states_;

    // Maps of indexes to the propagation state vector from the ops operands,
    // results and arguments.
    llvm::DenseMap<OpValue, int> operand_states_;
    llvm::DenseMap<OpValue, int> result_states_;
  };

  FuncOp func_;

  DeviceTarget target_spec_;

  StatesManager states_manager_;
};

}  // namespace quant
}  // namespace mlir

#endif  // TENSORFLOW_COMPILER_MLIR_LITE_QUANTIZATION_QUANTIZATION_CONTEXT_H_
