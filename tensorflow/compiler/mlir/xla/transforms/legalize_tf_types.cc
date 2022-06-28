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
class MHTracer_DTPStensorflowPScompilerPSmlirPSxlaPStransformsPSlegalize_tf_typesDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSmlirPSxlaPStransformsPSlegalize_tf_typesDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSmlirPSxlaPStransformsPSlegalize_tf_typesDTcc() {
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

/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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

// The TF dialect uses some TF types that are illegal in the MHLO dialect and
// some generic types that are legal in MHLO. This pass legalizes TF types into
// types that are legal in MHLO. For example, TF::Qint8Type is converted to i8.
// Rewrites here should run before TF to MHLO op legalizations are run.
// TODO(b/180234029): The rewrite here should be part of the LegalizeTF pass
// rather than its own pass.

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/TypeSwitch.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/IR/PatternMatch.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "mlir/Transforms/DialectConversion.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_types.h"
#include "tensorflow/compiler/mlir/xla/transforms/xla_legalize_tf_passes_detail.h"

#define DEBUG_TYPE "xla-legalize-tf-types"

namespace mlir {
namespace mhlo {
namespace {

bool IsIllegalElementType(Type type) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSxlaPStransformsPSlegalize_tf_typesDTcc mht_0(mht_0_v, 211, "", "./tensorflow/compiler/mlir/xla/transforms/legalize_tf_types.cc", "IsIllegalElementType");

  return type
      .isa<mlir::TF::Qint8Type, mlir::TF::Qint16Type, mlir::TF::Qint32Type,
           mlir::TF::Quint8Type, mlir::TF::Quint16Type>();
}

Type ToLegalElementType(Type type) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSxlaPStransformsPSlegalize_tf_typesDTcc mht_1(mht_1_v, 220, "", "./tensorflow/compiler/mlir/xla/transforms/legalize_tf_types.cc", "ToLegalElementType");

  return TypeSwitch<Type, Type>(type)
      .Case<mlir::TF::Qint8Type>([&type](Type) {
        return mlir::IntegerType::get(type.getContext(), 8);
      })
      .Case<mlir::TF::Qint16Type>([&type](Type) {
        return mlir::IntegerType::get(type.getContext(), 16);
      })
      .Case<mlir::TF::Qint32Type>([&type](Type) {
        return mlir::IntegerType::get(type.getContext(), 32);
      })
      .Case<mlir::TF::Quint8Type>([&type](Type) {
        return mlir::IntegerType::get(
            type.getContext(), 8,
            mlir::IntegerType::SignednessSemantics::Unsigned);
      })
      .Case<mlir::TF::Quint16Type>([&type](Type) {
        return mlir::IntegerType::get(
            type.getContext(), 16,
            mlir::IntegerType::SignednessSemantics::Unsigned);
      })
      .Default([&type](Type) { return type; });
}

// TODO(b/180234863): What's below this line is generic so convert it to a
// utility.

bool IsIllegalType(Type type) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSxlaPStransformsPSlegalize_tf_typesDTcc mht_2(mht_2_v, 250, "", "./tensorflow/compiler/mlir/xla/transforms/legalize_tf_types.cc", "IsIllegalType");

  return IsIllegalElementType(getElementTypeOrSelf(type));
}

Type ToLegalType(Type type) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSxlaPStransformsPSlegalize_tf_typesDTcc mht_3(mht_3_v, 257, "", "./tensorflow/compiler/mlir/xla/transforms/legalize_tf_types.cc", "ToLegalType");

  if (IsIllegalElementType(type)) return ToLegalElementType(type);
  if (auto shaped = type.dyn_cast<ShapedType>()) {
    Type elem = shaped.getElementType();
    if (IsIllegalType(elem)) return shaped.clone(ToLegalType(elem));
  }
  return type;
}

class TfTypeConverter : public TypeConverter {
 public:
  TfTypeConverter() {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSxlaPStransformsPSlegalize_tf_typesDTcc mht_4(mht_4_v, 271, "", "./tensorflow/compiler/mlir/xla/transforms/legalize_tf_types.cc", "TfTypeConverter");

    addConversion([](Type type) -> Type {
      return IsIllegalType(type) ? ToLegalType(type) : type;
    });
  }
};

// An Op is illegal iff it contains an illegalType.
class TfTypeConversionTarget : public ConversionTarget {
 public:
  explicit TfTypeConversionTarget(MLIRContext &ctx, TfTypeConverter &converter)
      : ConversionTarget(ctx), converter_(converter) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSxlaPStransformsPSlegalize_tf_typesDTcc mht_5(mht_5_v, 285, "", "./tensorflow/compiler/mlir/xla/transforms/legalize_tf_types.cc", "TfTypeConversionTarget");

    markUnknownOpDynamicallyLegal([this](Operation *op) {
      // The FuncOp type can contain types that the op's operand and result
      // types do not contain.
      if (auto func = dyn_cast<FuncOp>(op)) {
        if (!converter_.isSignatureLegal(func.getFunctionType())) return false;
      }
      return converter_.isLegal(op);
    });
  }

 private:
  TfTypeConverter &converter_;
};

class TfTypePattern : public ConversionPattern {
 public:
  TfTypePattern(MLIRContext *ctx, TypeConverter &converter)
      : ConversionPattern(converter, MatchAnyOpTypeTag(), 1, ctx) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSxlaPStransformsPSlegalize_tf_typesDTcc mht_6(mht_6_v, 306, "", "./tensorflow/compiler/mlir/xla/transforms/legalize_tf_types.cc", "TfTypePattern");
}

  // The dialect conversion framework will call this matchAndRewrite on each
  // Operation in the IR tree. This call matchAndRewrite needs to update the
  // Operation's results and child regions.
  LogicalResult matchAndRewrite(
      Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSxlaPStransformsPSlegalize_tf_typesDTcc mht_7(mht_7_v, 316, "", "./tensorflow/compiler/mlir/xla/transforms/legalize_tf_types.cc", "matchAndRewrite");

    // Update the results.
    llvm::SmallVector<Type, 4> new_results;
    if (failed(getTypeConverter()->convertTypes(op->getResultTypes(),
                                                new_results)))
      return failure();

    // Update the regions. The dialect conversion framework wants new regions to
    // be created and updated, rather than updating the old op. Thus we use an
    // OperationState so we can add regions to the new up.
    OperationState state(op->getLoc(), op->getName().getStringRef(), operands,
                         new_results, op->getAttrs(), op->getSuccessors());
    for (Region &region : op->getRegions()) {
      Region &new_region = *state.addRegion();
      rewriter.inlineRegionBefore(region, new_region, new_region.begin());
      if (failed(rewriter.convertRegionTypes(&new_region, *getTypeConverter())))
        return failure();
    }
    rewriter.replaceOp(op, rewriter.create(state)->getResults());

    return success();
  }
};

struct LegalizeTfTypesPass
    : public LegalizeTfTypesPassBase<LegalizeTfTypesPass> {
  void runOnOperation() override;
};

void LegalizeTfTypesPass::runOnOperation() {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSxlaPStransformsPSlegalize_tf_typesDTcc mht_8(mht_8_v, 348, "", "./tensorflow/compiler/mlir/xla/transforms/legalize_tf_types.cc", "LegalizeTfTypesPass::runOnOperation");

  TfTypeConverter converter;
  RewritePatternSet patterns(&getContext());
  patterns.add<TfTypePattern>(&getContext(), converter);
  populateFunctionOpInterfaceTypeConversionPattern<FuncOp>(patterns, converter);
  TfTypeConversionTarget target(getContext(), converter);
  if (failed(applyFullConversion(getOperation(), target, std::move(patterns))))
    return signalPassFailure();
}

}  // namespace

std::unique_ptr<OperationPass<>> CreateLegalizeTfTypesPass() {
  return std::make_unique<LegalizeTfTypesPass>();
}

}  // namespace mhlo
}  // namespace mlir
