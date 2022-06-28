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
class MHTracer_DTPStensorflowPScompilerPSmlirPStfrtPSjitPStransformsPStf_jitrt_math_approximationDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSmlirPStfrtPSjitPStransformsPStf_jitrt_math_approximationDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSmlirPStfrtPSjitPStransformsPStf_jitrt_math_approximationDTcc() {
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

#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/ArrayRef.h"
#include "tensorflow/compiler/mlir/tfrt/jit/transforms/tf_jitrt_passes.h"

namespace tensorflow {
namespace {

#define GEN_PASS_CLASSES
#include "tensorflow/compiler/mlir/tfrt/jit/transforms/tf_jitrt_passes.h.inc"

using ::llvm::ArrayRef;
using ::llvm::SmallVector;

using ::mlir::ImplicitLocOpBuilder;
using ::mlir::LogicalResult;
using ::mlir::OpRewritePattern;
using ::mlir::PatternRewriter;
using ::mlir::RewritePatternSet;
using ::mlir::Type;
using ::mlir::Value;
using ::mlir::VectorType;

namespace arith = ::mlir::arith;
namespace math = ::mlir::math;
namespace vector = ::mlir::vector;

using TypePredicate = ::llvm::function_ref<bool(Type)>;

// Returns vector shape if the element type is matching the predicate (scalars
// that do match the predicate have shape equal to `{1}`).
static llvm::Optional<SmallVector<int64_t, 2>> vectorShape(Type type,
                                                           TypePredicate pred) {
  // If the type matches the predicate then its shape is `{1}`.
  if (pred(type)) return SmallVector<int64_t, 2>{1};

  // Otherwise check if the type is a vector type.
  auto vectorType = type.dyn_cast<VectorType>();
  if (vectorType && pred(vectorType.getElementType())) {
    return llvm::to_vector<2>(vectorType.getShape());
  }

  return llvm::None;
}

// Returns vector shape of the type. If the type is a scalar returns `1`.
static SmallVector<int64_t, 2> vectorShape(Type type) {
  auto vectorType = type.dyn_cast<VectorType>();
  return vectorType ? llvm::to_vector<2>(vectorType.getShape())
                    : SmallVector<int64_t, 2>{1};
}

// Returns vector element type. If the type is a scalar returns the argument.
static Type elementType(Type type) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStfrtPSjitPStransformsPStf_jitrt_math_approximationDTcc mht_0(mht_0_v, 241, "", "./tensorflow/compiler/mlir/tfrt/jit/transforms/tf_jitrt_math_approximation.cc", "elementType");

  auto vectorType = type.dyn_cast<VectorType>();
  return vectorType ? vectorType.getElementType() : type;
}

static bool isF32(Type type) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStfrtPSjitPStransformsPStf_jitrt_math_approximationDTcc mht_1(mht_1_v, 249, "", "./tensorflow/compiler/mlir/tfrt/jit/transforms/tf_jitrt_math_approximation.cc", "isF32");
 return type.isF32(); }

//----------------------------------------------------------------------------//
// Broadcast scalar types and values into vector types and values.
//----------------------------------------------------------------------------//

// Returns true if shape != {1}.
static bool isNonScalarShape(ArrayRef<int64_t> shape) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStfrtPSjitPStransformsPStf_jitrt_math_approximationDTcc mht_2(mht_2_v, 259, "", "./tensorflow/compiler/mlir/tfrt/jit/transforms/tf_jitrt_math_approximation.cc", "isNonScalarShape");

  return shape.size() > 1 || shape[0] > 1;
}

// Broadcasts scalar type into vector type (iff shape is non-scalar).
static Type broadcast(Type type, ArrayRef<int64_t> shape) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStfrtPSjitPStransformsPStf_jitrt_math_approximationDTcc mht_3(mht_3_v, 267, "", "./tensorflow/compiler/mlir/tfrt/jit/transforms/tf_jitrt_math_approximation.cc", "broadcast");

  assert(!type.isa<VectorType>() && "must be scalar type");
  return isNonScalarShape(shape) ? VectorType::get(shape, type) : type;
}

// Broadcasts scalar value into vector (iff shape is non-scalar).
static Value broadcast(ImplicitLocOpBuilder &builder, Value value,
                       ArrayRef<int64_t> shape) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStfrtPSjitPStransformsPStf_jitrt_math_approximationDTcc mht_4(mht_4_v, 277, "", "./tensorflow/compiler/mlir/tfrt/jit/transforms/tf_jitrt_math_approximation.cc", "broadcast");

  assert(!value.getType().isa<VectorType>() && "must be scalar value");
  auto type = broadcast(value.getType(), shape);
  return isNonScalarShape(shape)
             ? builder.create<vector::BroadcastOp>(type, value)
             : value;
}

//----------------------------------------------------------------------------//
// Helper functions to create constants.
//----------------------------------------------------------------------------//

static Value f32Cst(ImplicitLocOpBuilder &builder, float value) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStfrtPSjitPStransformsPStf_jitrt_math_approximationDTcc mht_5(mht_5_v, 292, "", "./tensorflow/compiler/mlir/tfrt/jit/transforms/tf_jitrt_math_approximation.cc", "f32Cst");

  return builder.create<arith::ConstantOp>(builder.getF32FloatAttr(value));
}

static Value i32Cst(ImplicitLocOpBuilder &builder, int32_t value) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStfrtPSjitPStransformsPStf_jitrt_math_approximationDTcc mht_6(mht_6_v, 299, "", "./tensorflow/compiler/mlir/tfrt/jit/transforms/tf_jitrt_math_approximation.cc", "i32Cst");

  return builder.create<arith::ConstantOp>(builder.getI32IntegerAttr(value));
}

//----------------------------------------------------------------------------//
// Helper functions to build math function approximations.
//----------------------------------------------------------------------------//

static Value min(ImplicitLocOpBuilder &builder, Value a, Value b) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStfrtPSjitPStransformsPStf_jitrt_math_approximationDTcc mht_7(mht_7_v, 310, "", "./tensorflow/compiler/mlir/tfrt/jit/transforms/tf_jitrt_math_approximation.cc", "min");

  return builder.create<mlir::arith::SelectOp>(
      builder.create<arith::CmpFOp>(arith::CmpFPredicate::OLT, a, b), a, b);
}

static Value max(ImplicitLocOpBuilder &builder, Value a, Value b) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStfrtPSjitPStransformsPStf_jitrt_math_approximationDTcc mht_8(mht_8_v, 318, "", "./tensorflow/compiler/mlir/tfrt/jit/transforms/tf_jitrt_math_approximation.cc", "max");

  return builder.create<mlir::arith::SelectOp>(
      builder.create<arith::CmpFOp>(arith::CmpFPredicate::OGT, a, b), a, b);
}

static Value clamp(ImplicitLocOpBuilder &builder, Value value, Value lowerBound,
                   Value upperBound) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStfrtPSjitPStransformsPStf_jitrt_math_approximationDTcc mht_9(mht_9_v, 327, "", "./tensorflow/compiler/mlir/tfrt/jit/transforms/tf_jitrt_math_approximation.cc", "clamp");

  return max(builder, min(builder, value, upperBound), lowerBound);
}

// Eigen's implementation of ldexp.
// ldexp(x, exp) = x * 2^exp
// Set e = min(max(exp, -278), 278)
//     b = floor(e/4)
// Then out = ((((x * 2^(b)) * 2^(b)) * 2^(b)) * 2^(e-3*b))
static Value ldexp(ImplicitLocOpBuilder &builder, Value x, Value exp) {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStfrtPSjitPStransformsPStf_jitrt_math_approximationDTcc mht_10(mht_10_v, 339, "", "./tensorflow/compiler/mlir/tfrt/jit/transforms/tf_jitrt_math_approximation.cc", "ldexp");

  assert(isF32(elementType(x.getType())) && "argument x must be f32 type");
  assert(isF32(elementType(exp.getType())) && "argument exp must be f32 type");

  auto shape = vectorShape(x.getType());
  auto exp_shape = vectorShape(exp.getType());
  assert(shape == exp_shape && "x and exp must be of equal shape");
  auto f32Vec = broadcast(builder.getF32Type(), shape);
  auto i32Vec = broadcast(builder.getI32Type(), shape);

  auto bcast = [&](Value value) -> Value {
    return broadcast(builder, value, shape);
  };
  auto mulf = [&](Value a, Value b) -> Value {
    return builder.create<arith::MulFOp>(a, b);
  };
  auto subi = [&](Value a, Value b) -> Value {
    return builder.create<arith::SubIOp>(a, b);
  };
  auto shli = [&](Value a, Value pos) -> Value {
    return builder.create<arith::ShLIOp>(a, pos);
  };

  Value cstMantBitsI = bcast(i32Cst(builder, 23));
  Value cstMaxExponent = bcast(f32Cst(builder, 278.0f));
  Value cstMinExponent = bcast(f32Cst(builder, -278.0f));
  Value cstBiasI = bcast(i32Cst(builder, 127));
  Value cst2I = bcast(i32Cst(builder, 2));

  Value e = clamp(builder, exp, cstMinExponent, cstMaxExponent);
  Value eI = builder.create<arith::FPToSIOp>(i32Vec, e);
  Value bI = builder.create<arith::ShRSIOp>(eI, cst2I);
  Value biasedBI = builder.create<arith::AddIOp>(bI, cstBiasI);
  Value c = builder.create<arith::BitcastOp>(
      f32Vec, shli(biasedBI, cstMantBitsI));               // 2^b
  Value out = mulf(mulf(mulf(x, c), c), c);                // x * 2^(3b)
  bI = subi(subi(subi(eI, bI), bI), bI);                   // e - 3b
  biasedBI = builder.create<arith::AddIOp>(bI, cstBiasI);  // 2^(e - 3b)
  c = builder.create<arith::BitcastOp>(f32Vec, shli(biasedBI, cstMantBitsI));
  out = mulf(out, c);
  return out;
}

struct EigenExpApproximation : public OpRewritePattern<math::ExpOp> {
 public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(math::ExpOp op,
                                PatternRewriter &rewriter) const final;
};

LogicalResult EigenExpApproximation::matchAndRewrite(
    math::ExpOp op, PatternRewriter &rewriter) const {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStfrtPSjitPStransformsPStf_jitrt_math_approximationDTcc mht_11(mht_11_v, 394, "", "./tensorflow/compiler/mlir/tfrt/jit/transforms/tf_jitrt_math_approximation.cc", "EigenExpApproximation::matchAndRewrite");

  auto shape = vectorShape(op.getOperand().getType(), isF32);
  if (!shape.hasValue())
    return rewriter.notifyMatchFailure(op, "unsupported operand type");
  ImplicitLocOpBuilder builder(op->getLoc(), rewriter);

  auto addf = [&](Value a, Value b) -> Value {
    return builder.create<arith::AddFOp>(a, b);
  };
  auto bcast = [&](Value value) -> Value {
    return broadcast(builder, value, *shape);
  };
  auto floor = [&](Value a) {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStfrtPSjitPStransformsPStf_jitrt_math_approximationDTcc mht_12(mht_12_v, 409, "", "./tensorflow/compiler/mlir/tfrt/jit/transforms/tf_jitrt_math_approximation.cc", "lambda");
 return builder.create<math::FloorOp>(a); };
  auto fma = [&](Value a, Value b, Value c) {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStfrtPSjitPStransformsPStf_jitrt_math_approximationDTcc mht_13(mht_13_v, 413, "", "./tensorflow/compiler/mlir/tfrt/jit/transforms/tf_jitrt_math_approximation.cc", "lambda");

    return builder.create<math::FmaOp>(a, b, c);
  };
  auto mulf = [&](Value a, Value b) -> Value {
    return builder.create<arith::MulFOp>(a, b);
  };

  Value cstOne = bcast(f32Cst(builder, 1.0f));
  Value cstHalf = bcast(f32Cst(builder, 0.5f));
  Value cstExpHi = bcast(f32Cst(builder, 88.723f));
  Value cstExpLo = bcast(f32Cst(builder, -88.723f));

  Value cstCephesLog2E = bcast(f32Cst(builder, 1.44269504088896341f));
  Value cstCephesExpP0 = bcast(f32Cst(builder, 1.9875691500E-4f));
  Value cstCephesExpP1 = bcast(f32Cst(builder, 1.3981999507E-3f));
  Value cstCephesExpP2 = bcast(f32Cst(builder, 8.3334519073E-3f));
  Value cstCephesExpP3 = bcast(f32Cst(builder, 4.1665795894E-2f));
  Value cstCephesExpP4 = bcast(f32Cst(builder, 1.6666665459E-1f));
  Value cstCephesExpP5 = bcast(f32Cst(builder, 5.0000001201E-1f));

  Value x = clamp(builder, op.getOperand(), cstExpLo, cstExpHi);
  Value m = floor(fma(x, cstCephesLog2E, cstHalf));

  Value cstCephesExpC1 = bcast(f32Cst(builder, -0.693359375f));
  Value cstCephesExpC2 = bcast(f32Cst(builder, 2.12194440e-4f));
  Value r = fma(m, cstCephesExpC1, x);
  r = fma(m, cstCephesExpC2, r);

  Value r2 = mulf(r, r);
  Value r3 = mulf(r2, r);

  Value y = fma(cstCephesExpP0, r, cstCephesExpP1);
  Value y1 = fma(cstCephesExpP3, r, cstCephesExpP4);
  Value y2 = addf(r, cstOne);
  y = fma(y, r, cstCephesExpP2);
  y1 = fma(y1, r, cstCephesExpP5);
  y = fma(y, r3, y1);
  y = fma(y, r2, y2);
  Value ret = max(builder, ldexp(builder, y, m), op.getOperand());
  rewriter.replaceOp(op, ret);
  return mlir::success();
}

static void populateMathApproximationPatterns(RewritePatternSet &patterns,
                                              ArrayRef<std::string> oplist) {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStfrtPSjitPStransformsPStf_jitrt_math_approximationDTcc mht_14(mht_14_v, 460, "", "./tensorflow/compiler/mlir/tfrt/jit/transforms/tf_jitrt_math_approximation.cc", "populateMathApproximationPatterns");

  for (const std::string &op : oplist) {
    if (op == "exp" || op == "all")
      patterns.add<EigenExpApproximation>(patterns.getContext());
  }
}

struct MathApproximationPass
    : public MathApproximationBase<MathApproximationPass> {
  explicit MathApproximationPass(ArrayRef<std::string> approx_oplist) {
   std::vector<std::string> mht_15_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStfrtPSjitPStransformsPStf_jitrt_math_approximationDTcc mht_15(mht_15_v, 472, "", "./tensorflow/compiler/mlir/tfrt/jit/transforms/tf_jitrt_math_approximation.cc", "MathApproximationPass");

    this->oplist = approx_oplist;
  }

  void runOnOperation() override;
};

void MathApproximationPass::runOnOperation() {
   std::vector<std::string> mht_16_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStfrtPSjitPStransformsPStf_jitrt_math_approximationDTcc mht_16(mht_16_v, 482, "", "./tensorflow/compiler/mlir/tfrt/jit/transforms/tf_jitrt_math_approximation.cc", "MathApproximationPass::runOnOperation");

  mlir::RewritePatternSet patterns(&getContext());
  populateMathApproximationPatterns(patterns, oplist);
  if (failed(mlir::applyPatternsAndFoldGreedily(getOperation(),
                                                std::move(patterns))))
    signalPassFailure();
}

}  // namespace

std::unique_ptr<mlir::OperationPass<mlir::func::FuncOp>>
CreateMathApproximationPass(ArrayRef<std::string> oplist) {
  return std::make_unique<MathApproximationPass>(oplist);
}

}  // namespace tensorflow
