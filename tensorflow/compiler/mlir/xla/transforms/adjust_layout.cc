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
class MHTracer_DTPStensorflowPScompilerPSmlirPSxlaPStransformsPSadjust_layoutDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSmlirPSxlaPStransformsPSadjust_layoutDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSmlirPSxlaPStransformsPSadjust_layoutDTcc() {
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

#include <algorithm>
#include <cstdint>
#include <memory>
#include <numeric>
#include <vector>

#include "absl/types/span.h"
#include "llvm/ADT/StringRef.h"
#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/IR/Dialect.h"  // from @llvm-project
#include "mlir/IR/OpDefinition.h"  // from @llvm-project
#include "mlir/IR/Operation.h"  // from @llvm-project
#include "mlir/IR/Types.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/hlo/include/mlir-hlo/Dialect/mhlo/IR/hlo_ops.h"
#include "tensorflow/compiler/mlir/xla/type_to_shape.h"
#include "tensorflow/compiler/xla/layout.h"
#include "tensorflow/compiler/xla/shape.h"
#include "tensorflow/core/tpu/tpu_api.h"
#include "tensorflow/stream_executor/tpu/c_api_conversions.h"

namespace mlir {
namespace mhlo {
static FailureOr<std::vector<int64_t>> GetTPUInfeedLayoutFromAPI(
    RankedTensorType t) {
  // Call the TPU API to determine the right infeed layout. Note that
  // this can fail if we're not running on a TPU-enabled node.
  // TODO(kramm): Move this into a separate pass. See b/184944903
  xla::Shape old_shape = xla::TypeToShape(t);
  XLA_Shape old_shape_c = {};
  XLA_Shape new_shape_c = {};
  TfTpu_ExecutorApiFn *executor = tensorflow::tpu::ExecutorApiFn();
  if (!tensorflow::tpu::IsInitialized(executor)) {
    return failure();
  }
  ApiConverter::ToC(old_shape, &old_shape_c);
  executor->TpuTransferManager_GetInfeedLayoutFn(&old_shape_c, &new_shape_c);
  xla::Shape new_shape = ApiConverter::FromC(&new_shape_c);
  ApiConverter::Destroy(&old_shape_c);
  ApiConverter::Destroy(&new_shape_c);

  auto minor_to_major = new_shape.layout().minor_to_major();
  return std::vector<int64_t>(minor_to_major.begin(), minor_to_major.end());
}

FailureOr<Attribute> GetTPUInfeedLayout(const ArrayRef<Type> types,
                                        OpBuilder &rewriter) {
  auto i64_type = rewriter.getIntegerType(64);
  if (types.size() > 1) {
    llvm::SmallVector<mlir::Attribute> v;
    v.reserve(types.size());
    for (const mlir::Type &t : types) {
      if (t.isa<TokenType>()) continue;
      auto layout = GetTPUInfeedLayout({t}, rewriter);
      if (failed(layout)) return failure();
      v.push_back(layout.getValue());
    }
    ArrayRef<Attribute> shape(v);
    return rewriter.getArrayAttr(shape);
  } else if (types[0].isa<TupleType>()) {
    auto tuple_type = types[0].dyn_cast<TupleType>();
    const auto &types = tuple_type.getTypes();
    llvm::SmallVector<mlir::Attribute> v;
    v.reserve(types.size());
    for (const mlir::Type &t : types) {
      if (t.isa<TokenType>()) continue;
      auto layout = GetTPUInfeedLayout({t}, rewriter);
      if (failed(layout)) return failure();
      v.push_back(layout.getValue());
    }
    ArrayRef<Attribute> shape(v);
    return rewriter.getArrayAttr(shape);
  } else if (auto t = types[0].dyn_cast<RankedTensorType>()) {
    if (!t.hasStaticShape()) return failure();
    auto layout = GetTPUInfeedLayoutFromAPI(t);
    std::vector<int64_t> minor_to_major;
    if (succeeded(layout)) {
      minor_to_major = layout.getValue();
    } else {
      /* If we're not running on a TPU node, we might not be able to
       * actually call the part of the TPU API that gives us layout.
       * This happens e.g. for unit tests. Below we just create a reasonable
       * layout.  We sort by dimension size, which makes the layout agree with
       * the "correct" TPU layout in surprisingly many cases.
       * Note that the corresponding InfeedEnqueue op will be generated
       * through another path, and might still generate an (incompatible)
       * layout using the TPU API. Running legalize_tf.cc on non-TPU nodes
       * thus is a potential source of bugs.
       */
      minor_to_major.resize(t.getRank());
      std::iota(minor_to_major.begin(), minor_to_major.end(), 0);
      std::sort(minor_to_major.begin(), minor_to_major.end(),
                [=](int64_t a, int64_t b) {
                  int64_t da = t.getDimSize(a);
                  int64_t db = t.getDimSize(b);
                  return da > db || (da == db && a > b);
                });
    }
    std::vector<Attribute> elements;
    elements.reserve(minor_to_major.size());
    for (auto e : minor_to_major) {
      elements.push_back(rewriter.getIntegerAttr(i64_type, e));
    }
    return rewriter.getArrayAttr(elements);
  } else {
    // types.size() == 1 and types[0] == TokenType
    // For this case, we return an empty array attribute.
    return rewriter.getArrayAttr({});
  }
}

namespace {
class AdjustLayout : public PassWrapper<AdjustLayout, OperationPass<FuncOp>> {
  void getDependentDialects(DialectRegistry &registry) const override {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSxlaPStransformsPSadjust_layoutDTcc mht_0(mht_0_v, 301, "", "./tensorflow/compiler/mlir/xla/transforms/adjust_layout.cc", "getDependentDialects");

    registry.insert<mhlo::MhloDialect>();
  }

 public:
  StringRef getArgument() const final {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSxlaPStransformsPSadjust_layoutDTcc mht_1(mht_1_v, 309, "", "./tensorflow/compiler/mlir/xla/transforms/adjust_layout.cc", "getArgument");
 return "xla-adjust-layout"; }
  StringRef getDescription() const final {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSxlaPStransformsPSadjust_layoutDTcc mht_2(mht_2_v, 313, "", "./tensorflow/compiler/mlir/xla/transforms/adjust_layout.cc", "getDescription");

    return "Adjust layouts so infeed send & receive use the same format.";
  }

  static void runOnInfeedOp(::mlir::mhlo::InfeedOp op) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSxlaPStransformsPSadjust_layoutDTcc mht_3(mht_3_v, 320, "", "./tensorflow/compiler/mlir/xla/transforms/adjust_layout.cc", "runOnInfeedOp");

    OpBuilder builder(op.getContext());
    SmallVector<Type> result_types(op.getResultTypes().begin(),
                                   op.getResultTypes().end());
    if (!op->getAttr("layout")) {
      auto layout = GetTPUInfeedLayout(result_types, builder);
      if (failed(layout)) return;

      op->setAttr("layout", layout.getValue());
    }
  }

  void runOnOperation() override {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSxlaPStransformsPSadjust_layoutDTcc mht_4(mht_4_v, 335, "", "./tensorflow/compiler/mlir/xla/transforms/adjust_layout.cc", "runOnOperation");
 getOperation().walk(runOnInfeedOp); }
};
}  // anonymous namespace

// Header for this is in passes.h, which pulls into many deps. NOLINTNEXTLINE
std::unique_ptr<Pass> CreateAdjustLayoutPass() {
  return std::make_unique<AdjustLayout>();
}

void RegisterAdjustLayoutPass() {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSxlaPStransformsPSadjust_layoutDTcc mht_5(mht_5_v, 347, "", "./tensorflow/compiler/mlir/xla/transforms/adjust_layout.cc", "RegisterAdjustLayoutPass");
 static PassRegistration<AdjustLayout> pass; }

}  // namespace mhlo

}  // namespace mlir
