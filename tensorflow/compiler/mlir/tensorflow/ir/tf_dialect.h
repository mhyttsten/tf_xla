/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

// This file defines the standard MLIR TensorFlow dialect after control
// dependences are raise to the standard form.

#ifndef TENSORFLOW_COMPILER_MLIR_TENSORFLOW_IR_TF_DIALECT_H_
#define TENSORFLOW_COMPILER_MLIR_TENSORFLOW_IR_TF_DIALECT_H_
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
class MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPSirPStf_dialectDTh {
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
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPSirPStf_dialectDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPSirPStf_dialectDTh() {
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


#include "mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/IR/Dialect.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_types.h"

namespace mlir {
namespace TF {

class TensorFlowRegistryEffectInterfaceFallback;

class TensorFlowDialect final : public Dialect {
 public:
  explicit TensorFlowDialect(MLIRContext *context);
  ~TensorFlowDialect() override;

  static StringRef getDialectNamespace() {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPSirPStf_dialectDTh mht_0(mht_0_v, 205, "", "./tensorflow/compiler/mlir/tensorflow/ir/tf_dialect.h", "getDialectNamespace");
 return "tf"; }

  // Overrides to redirect to tf_type dialect.
  Attribute parseAttribute(DialectAsmParser &parser, Type type) const override;
  Type parseType(DialectAsmParser &parser) const override;

  // Gradient attribute ("tf.gradient") in the list of NamedAttributes in a
  // function references to its gradient function. This attribute in TensorFlow
  // Dialect is used to model TF GradientDef. GetGradientAttrName() returns the
  // string description of gradient attribute.
  static StringRef GetGradientAttrName() {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPSirPStf_dialectDTh mht_1(mht_1_v, 218, "", "./tensorflow/compiler/mlir/tensorflow/ir/tf_dialect.h", "GetGradientAttrName");
 return "tf.gradient"; }

  // This attribute marks if a function is stateful.
  // Returns the string description of stateful attribute.
  static StringRef GetStatefulAttrName() {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPSirPStf_dialectDTh mht_2(mht_2_v, 225, "", "./tensorflow/compiler/mlir/tensorflow/ir/tf_dialect.h", "GetStatefulAttrName");
 return "tf.signature.is_stateful"; }

  // Returns true if the op can be duplicated during transformations.
  static bool CanDuplicate(Operation *op);

  // Returns true if the op can have side effects.
  static bool CanHaveSideEffects(Operation *op);

  // Registered hook to materialize a constant operation from a given attribute
  // value with the desired resultant type.
  Operation *materializeConstant(OpBuilder &builder, Attribute value, Type type,
                                 Location loc) override;

  typedef std::function<void(TensorFlowDialect &dialect)> AdditionalOpFunction;

  // Register an op registration hook which is invoked during construction.
  //
  // A hook may use the public addOperations() method to add additional
  // operations to the dialect. Hooks will only apply to subsequent
  // instantations of the Dialect/MLIRContext.
  static void RegisterAdditionalOperationHook(TypeID uniqueId,
                                              AdditionalOpFunction fn);

  // Re-define publicly the protected addOperations() method from the Dialect
  // class, usually used in a Dialect constructor. This allows hook
  // functions to register operations on the TensorFlow dialect using the
  // same interface.
  template <typename... Args>
  void addOperations() {
    Dialect::addOperations<Args...>();
  }

  using ConstantFoldHook = LogicalResult (*)(Operation *, ArrayRef<Attribute>,
                                             SmallVectorImpl<OpFoldResult> &);
  static void RegisterConstantFoldHook(ConstantFoldHook fn) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPSirPStf_dialectDTh mht_3(mht_3_v, 262, "", "./tensorflow/compiler/mlir/tensorflow/ir/tf_dialect.h", "RegisterConstantFoldHook");

    constant_fold_hook_ = std::move(fn);
  }

  static LogicalResult constantFold(Operation *op, ArrayRef<Attribute> operands,
                                    SmallVectorImpl<OpFoldResult> &results) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPSirPStf_dialectDTh mht_4(mht_4_v, 270, "", "./tensorflow/compiler/mlir/tensorflow/ir/tf_dialect.h", "constantFold");

    if (constant_fold_hook_) return constant_fold_hook_(op, operands, results);
    return failure();
  }

  using DecodeConstantHook = LogicalResult (*)(OpaqueElementsAttr input,
                                               ElementsAttr &output);
  static void RegisterDecodeConstantHook(DecodeConstantHook fn) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPSirPStf_dialectDTh mht_5(mht_5_v, 280, "", "./tensorflow/compiler/mlir/tensorflow/ir/tf_dialect.h", "RegisterDecodeConstantHook");

    decode_constant_hook_ = std::move(fn);
  }
  static LogicalResult decode(OpaqueElementsAttr input, ElementsAttr &output) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPSirPStf_dialectDTh mht_6(mht_6_v, 286, "", "./tensorflow/compiler/mlir/tensorflow/ir/tf_dialect.h", "decode");

    if (decode_constant_hook_) return decode_constant_hook_(input, output);
    return failure();
  }

  // Provides a hook for op interface.
  void *getRegisteredInterfaceForOp(mlir::TypeID interface,
                                    mlir::OperationName opName) override;

 private:
  static ConstantFoldHook constant_fold_hook_;
  static DecodeConstantHook decode_constant_hook_;

  // Storage for a custom fallback interface.
  TensorFlowRegistryEffectInterfaceFallback *fallback_effect_op_interface_;
};

}  // namespace TF
}  // namespace mlir

#define TF_DIALECT_REGISTER_ADDITIONAL_OPERATIONS(hookFn)           \
  {                                                                 \
    static bool key;                                                \
    ::mlir::TF::TensorFlowDialect::RegisterAdditionalOperationHook( \
        ::mlir::TypeID::getFromOpaquePointer(&key), hookFn);        \
  }

#endif  // TENSORFLOW_COMPILER_MLIR_TENSORFLOW_IR_TF_DIALECT_H_
