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
class MHTracer_DTPStensorflowPScompilerPSmlirPSlitePStransformsPSlegalize_jax_randomDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePStransformsPSlegalize_jax_randomDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSmlirPSlitePStransformsPSlegalize_jax_randomDTcc() {
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

// The full pipline of converting jax random include 2 steps.
// 1. Rename the jax random functions to tflite wrapped functions with the aid
//    of "jax.named_call". For example, in the dumped hlo, the
//    jax.random.uniform will have name "tfl_wrapped_jax_random_uniform".
// 2. Replace the body of "tfl_wrapped_jax_random_uniform" and
//    "tfl_wrapped_jax_random_normal" with tfl.CustomOp("RandomUniform") and
//     tfl.CustomOp("RandomStandardNormal"), respectively.

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/IR/Block.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/IR/ImplicitLocOpBuilder.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/IR/OperationSupport.h"  // from @llvm-project
#include "mlir/IR/PatternMatch.h"  // from @llvm-project
#include "mlir/IR/Region.h"  // from @llvm-project
#include "mlir/IR/TypeRange.h"  // from @llvm-project
#include "mlir/IR/Visitors.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "mlir/Transforms/DialectConversion.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/hlo/include/mlir-hlo/Dialect/mhlo/IR/hlo_ops.h"
#include "tensorflow/compiler/mlir/lite/ir/tfl_ops.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_dialect.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"

namespace mlir {
namespace TFL {
namespace {

struct LegalizeJaxRandomPass
    : public PassWrapper<LegalizeJaxRandomPass, OperationPass<FuncOp>> {
 public:
  StringRef getArgument() const final {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePStransformsPSlegalize_jax_randomDTcc mht_0(mht_0_v, 228, "", "./tensorflow/compiler/mlir/lite/transforms/legalize_jax_random.cc", "getArgument");
 return "tfl-legalize-random"; }
  StringRef getDescription() const final {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePStransformsPSlegalize_jax_randomDTcc mht_1(mht_1_v, 232, "", "./tensorflow/compiler/mlir/lite/transforms/legalize_jax_random.cc", "getDescription");

    return "Replace jax.random.uniform/normal with tfl.custom.";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePStransformsPSlegalize_jax_randomDTcc mht_2(mht_2_v, 239, "", "./tensorflow/compiler/mlir/lite/transforms/legalize_jax_random.cc", "getDependentDialects");

    registry.insert<TFL::TensorFlowLiteDialect, mhlo::MhloDialect>();
  }
  void runOnOperation() override;
};

inline OpaqueElementsAttr CustomOption(ImplicitLocOpBuilder *builder,
                                       const std::string &content) {
   std::vector<std::string> mht_3_v;
   mht_3_v.push_back("content: \"" + content + "\"");
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePStransformsPSlegalize_jax_randomDTcc mht_3(mht_3_v, 250, "", "./tensorflow/compiler/mlir/lite/transforms/legalize_jax_random.cc", "CustomOption");

  ShapedType type = RankedTensorType::get(
      {static_cast<int64_t>(content.size())}, builder->getIntegerType(8));
  return OpaqueElementsAttr::get(builder->getContext()->getLoadedDialect("tfl"),
                                 type,
                                 StringRef(content.data(), content.size()));
}

inline bool IsJaxRandomUniform(mlir::func::FuncOp func) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePStransformsPSlegalize_jax_randomDTcc mht_4(mht_4_v, 261, "", "./tensorflow/compiler/mlir/lite/transforms/legalize_jax_random.cc", "IsJaxRandomUniform");

  return func.getName().contains("tfl_wrapped_jax_random_uniform");
}

inline bool IsJaxRandomNormal(mlir::func::FuncOp func) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePStransformsPSlegalize_jax_randomDTcc mht_5(mht_5_v, 268, "", "./tensorflow/compiler/mlir/lite/transforms/legalize_jax_random.cc", "IsJaxRandomNormal");

  return func.getName().contains("tfl_wrapped_jax_random_normal");
}

void LegalizeJaxRandomPass::runOnOperation() {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePStransformsPSlegalize_jax_randomDTcc mht_6(mht_6_v, 275, "", "./tensorflow/compiler/mlir/lite/transforms/legalize_jax_random.cc", "LegalizeJaxRandomPass::runOnOperation");

  auto func = getOperation();
  if (!IsJaxRandomUniform(func) && !IsJaxRandomNormal(func)) return;
  auto result_tuple_ty =
      func.getFunctionType().getResult(0).dyn_cast_or_null<TupleType>();
  if (!result_tuple_ty) return;
  if (result_tuple_ty.size() != 1) return;
  auto result_ty = result_tuple_ty.getType(0).dyn_cast<ShapedType>();

  func.eraseBody();
  func.addEntryBlock();
  ImplicitLocOpBuilder builder(func.getLoc(), func.getBody());
  llvm::SmallVector<int32_t> result_shape_i32;
  auto result_shape = result_ty.getShape();
  for (auto element : result_shape) {
    result_shape_i32.push_back(static_cast<int32_t>(element));
  }
  auto result_shape_attr = builder.getI32TensorAttr(result_shape_i32);
  Value result_shape_tensor = builder.create<mhlo::ConstOp>(result_shape_attr);
  auto custom_code =
      IsJaxRandomUniform(func) ? "RandomUniform" : "RandomStandardNormal";

  llvm::SmallVector<Type> result_ty_vec({result_ty});
  llvm::SmallVector<Value> result_shape_tensor_vec({result_shape_tensor});
  auto attr = CustomOption(&builder, "");
  Value random_result =
      builder
          .create<TFL::CustomOp>(TypeRange(result_ty_vec),
                                 ValueRange(result_shape_tensor_vec),
                                 custom_code, attr)
          .getResult(0);
  Value tulple_result = builder.create<mhlo::TupleOp>(random_result);
  builder.create<mlir::func::ReturnOp>(tulple_result);
}

static PassRegistration<LegalizeJaxRandomPass> pass;
}  // namespace

std::unique_ptr<OperationPass<FuncOp>> CreateLegalizeJaxRandomPass() {
  return std::make_unique<LegalizeJaxRandomPass>();
}

}  // namespace TFL
}  // namespace mlir
