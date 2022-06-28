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
class MHTracer_DTPStensorflowPScompilerPSmlirPShloPSlibPSutilsPShlo_utilsDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSmlirPShloPSlibPSutilsPShlo_utilsDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSmlirPShloPSlibPSutilsPShlo_utilsDTcc() {
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

#include "mlir-hlo/utils/hlo_utils.h"

#include <numeric>

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Attributes.h"

namespace mlir {
namespace hlo {

static constexpr size_t kPaddingSize = 64;

DenseIntElementsAttr getBroadcastDimensionsAttr(Builder* b, Value x, Value y,
                                                bool allow_empty) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPShloPSlibPSutilsPShlo_utilsDTcc mht_0(mht_0_v, 198, "", "./tensorflow/compiler/mlir/hlo/lib/utils/hlo_utils.cc", "getBroadcastDimensionsAttr");

  TensorType xType = x.getType().dyn_cast<RankedTensorType>();
  TensorType yType = y.getType().dyn_cast<RankedTensorType>();
  if (!xType || !yType) return {};
  if (allow_empty && xType == yType) return {};

  // If the shapes have the same rank, then there is nothing to do.
  auto xRank = xType.getRank(), yRank = yType.getRank();
  if (allow_empty && xRank == yRank) return {};

  // Otherwise if the ranks of the inputs don't match, TensorFlow automatically
  // reshapes the smaller by padding with dimensions of size 1 as a prefix. In
  // other words to pad a 5-vector to a 3-dimensional tensor it is reshaped to
  // have shape [1,1,5]. XLA's automatic broadcast code is able to broadcast
  // from lower to higher rank, but doesn't assume you want to pad as a prefix
  // of the dimensions, and instead needs to be told which dimensions of the
  // higher rank tensor to match to the lower rank tensor.
  auto maxRank = std::max(xRank, yRank);
  auto minRank = std::min(xRank, yRank);

  // Match the lower rank tensor along the larger-numbered dimensions of the
  // higher rank tensor.
  SmallVector<int64_t, 4> broadcastDimensions(minRank);
  std::iota(broadcastDimensions.begin(), broadcastDimensions.end(),
            maxRank - minRank);

  RankedTensorType type =
      RankedTensorType::get({minRank}, b->getIntegerType(64));
  return DenseIntElementsAttr::get(type, broadcastDimensions);
}

DenseElementsAttr GetScalarOfType(Type ty, int64_t raw_value) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPShloPSlibPSutilsPShlo_utilsDTcc mht_1(mht_1_v, 232, "", "./tensorflow/compiler/mlir/hlo/lib/utils/hlo_utils.cc", "GetScalarOfType");

  RankedTensorType scalar_ty = RankedTensorType::get({}, ty);

  if (auto float_ty = ty.dyn_cast<FloatType>()) {
    APFloat value(float_ty.getFloatSemantics(), raw_value);
    return DenseElementsAttr::get(scalar_ty, value);
  }
  if (auto int_ty = ty.dyn_cast<IntegerType>()) {
    APInt value(int_ty.getWidth(), static_cast<int64_t>(raw_value),
                /*isSigned=*/true);
    return DenseElementsAttr::get(scalar_ty, value);
  }
  if (auto complex_ty = ty.dyn_cast<ComplexType>()) {
    if (auto float_ty = complex_ty.getElementType().cast<FloatType>()) {
      APFloat real(float_ty.getFloatSemantics(), raw_value);
      APFloat imag = APFloat::getZero(float_ty.getFloatSemantics());
      return DenseElementsAttr::get(scalar_ty,
                                    std::complex<APFloat>(real, imag));
    }
  }
  llvm_unreachable("unsupported type");
}

DenseElementsAttr GetScalarNegZeroOfType(Type ty) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPShloPSlibPSutilsPShlo_utilsDTcc mht_2(mht_2_v, 258, "", "./tensorflow/compiler/mlir/hlo/lib/utils/hlo_utils.cc", "GetScalarNegZeroOfType");

  RankedTensorType scalar_ty = RankedTensorType::get({}, ty);

  if (auto float_ty = ty.dyn_cast<FloatType>()) {
    APFloat neg_zero =
        APFloat::getZero(float_ty.getFloatSemantics(), /*Negative=*/true);
    return DenseElementsAttr::get(scalar_ty, neg_zero);
  }
  if (auto int_ty = ty.dyn_cast<IntegerType>()) {
    return DenseElementsAttr::get(scalar_ty, APInt::getZero(int_ty.getWidth()));
  }
  if (auto complex_ty = ty.dyn_cast<ComplexType>()) {
    if (auto float_ty = complex_ty.getElementType().cast<FloatType>()) {
      APFloat neg_zero =
          APFloat::getZero(float_ty.getFloatSemantics(), /*Negative=*/true);
      return DenseElementsAttr::get(scalar_ty,
                                    std::complex<APFloat>(neg_zero, neg_zero));
    }
  }
  llvm_unreachable("unsupported type");
}

static APFloat GetScalarLimitOfFloatType(FloatType float_ty,
                                         ScalarLimit limit) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPShloPSlibPSutilsPShlo_utilsDTcc mht_3(mht_3_v, 284, "", "./tensorflow/compiler/mlir/hlo/lib/utils/hlo_utils.cc", "GetScalarLimitOfFloatType");

  auto& semantics = float_ty.getFloatSemantics();
  switch (limit) {
    case kLowest:
      return APFloat::getLargest(semantics, /*negative=*/true);
    case kInfinityLowest:
      return APFloat::getInf(semantics, /*negative=*/true);
    case kMax:
      return APFloat::getLargest(semantics, /*negative=*/false);
    case kInfinityMax:
      return APFloat::getInf(semantics, /*negative=*/false);
  }
  llvm_unreachable("invalid limit");
}

// Returns a scalar value for the given integer type.
//
// The argument 'scalar' describes which scalar value to return. `integer_value`
// is used to specify the integer value for kInteger. For any other scalar,
// integer_value is ignored.
static APInt GetScalarLimitOfIntegerType(IntegerType integer_ty,
                                         ScalarLimit limit) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPShloPSlibPSutilsPShlo_utilsDTcc mht_4(mht_4_v, 308, "", "./tensorflow/compiler/mlir/hlo/lib/utils/hlo_utils.cc", "GetScalarLimitOfIntegerType");

  unsigned width = integer_ty.getWidth();
  bool is_bool = (width == 1);
  switch (limit) {
    case kLowest:
    case kInfinityLowest:
      if (integer_ty.isUnsigned() || is_bool) {
        return APInt::getMinValue(width);
      } else {
        return APInt::getSignedMinValue(width);
      }

    case kMax:
    case kInfinityMax:
      if (integer_ty.isUnsigned() || is_bool) {
        return APInt::getMaxValue(width);
      } else {
        return APInt::getSignedMaxValue(width);
      }
  }
  llvm_unreachable("invalid limit");
}

DenseElementsAttr GetScalarLimitOfType(Type ty, ScalarLimit limit) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPShloPSlibPSutilsPShlo_utilsDTcc mht_5(mht_5_v, 334, "", "./tensorflow/compiler/mlir/hlo/lib/utils/hlo_utils.cc", "GetScalarLimitOfType");

  RankedTensorType scalar_ty = RankedTensorType::get({}, ty);
  if (auto float_ty = ty.dyn_cast<FloatType>()) {
    return DenseElementsAttr::get(scalar_ty,
                                  GetScalarLimitOfFloatType(float_ty, limit));
  }
  if (auto integer_ty = ty.dyn_cast<IntegerType>()) {
    return DenseElementsAttr::get(
        scalar_ty, GetScalarLimitOfIntegerType(integer_ty, limit));
  }
  llvm_unreachable("unsupported type");
}

std::string LmhloToMhloOpName(llvm::StringRef op_name,
                              mlir::MLIRContext* context) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPShloPSlibPSutilsPShlo_utilsDTcc mht_6(mht_6_v, 351, "", "./tensorflow/compiler/mlir/hlo/lib/utils/hlo_utils.cc", "LmhloToMhloOpName");

  assert(op_name.startswith("lmhlo.") && "Expected an LMHLO op");

  if (op_name == "lmhlo.dot") {
    return "mhlo.dot_general";
  }

  if (op_name == "lmhlo.dynamic_slice") {
    return "mhlo.dynamic-slice";
  }

  std::string mhlo_op_name(op_name.drop_front(1));
  if (context->isOperationRegistered(mhlo_op_name)) return mhlo_op_name;
  return "";
}

bool IsSequenceStartingWith0(Attribute attr) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPShloPSlibPSutilsPShlo_utilsDTcc mht_7(mht_7_v, 370, "", "./tensorflow/compiler/mlir/hlo/lib/utils/hlo_utils.cc", "IsSequenceStartingWith0");

  DenseIntElementsAttr denseAttr = attr.dyn_cast<DenseIntElementsAttr>();
  for (int64_t i = 0, e = denseAttr.getNumElements(); i < e; ++i)
    if (denseAttr.getValues<APInt>()[i].getSExtValue() != i) return false;
  return true;
}

int64_t getArgumentIndex(mlir::func::FuncOp op, Value value) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPShloPSlibPSutilsPShlo_utilsDTcc mht_8(mht_8_v, 380, "", "./tensorflow/compiler/mlir/hlo/lib/utils/hlo_utils.cc", "getArgumentIndex");

  BlockArgument arg = value.dyn_cast<BlockArgument>();
  if (!arg || arg.getOwner() != &op.front()) return -1;
  return arg.getArgNumber();
}

/// Computes the memory usage of the given allocations.
std::pair<size_t, size_t> computeMemory(const std::vector<Value>& allocs) {
  size_t totalSize = 0;
  size_t allocCounter = 0;
  for (const Value alloc : allocs) {
    auto shape = alloc.getType().cast<ShapedType>();
    size_t shapeBytes = llvm::divideCeil(shape.getSizeInBits(), 8);
    size_t alignFactor = llvm::divideCeil(shapeBytes, kPaddingSize);
    size_t size = alignFactor * kPaddingSize;
    totalSize += size;
    allocCounter++;
  }
  return std::make_pair(totalSize, allocCounter);
}

}  // namespace hlo
}  // namespace mlir
