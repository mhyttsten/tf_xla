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
class MHTracer_DTPStensorflowPScompilerPSmlirPSxlaPShlo_utilsDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSmlirPSxlaPShlo_utilsDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSmlirPSxlaPShlo_utilsDTcc() {
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

// This file defines helpers useful when creating or manipulating lhlo/hlo.

#include "tensorflow/compiler/mlir/xla/hlo_utils.h"

#include "mlir/IR/AffineMap.h"  // from @llvm-project
#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/IR/TypeUtilities.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/hlo/include/mlir-hlo/Dialect/lhlo/IR/lhlo_ops.h"
#include "tensorflow/compiler/xla/literal.h"
#include "tensorflow/core/platform/bfloat16.h"
#include "tensorflow/core/platform/logging.h"

namespace xla {
namespace {

using mlir::AffineMap;
using mlir::Builder;
using mlir::DenseElementsAttr;
using mlir::ShapedType;
using xla::LiteralBase;
using xla::StatusOr;

template <typename CppType>
::mlir::DenseElementsAttr CreateDenseAttrFromLiteral(
    const ShapedType& type, const LiteralBase& literal) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSxlaPShlo_utilsDTcc mht_0(mht_0_v, 210, "", "./tensorflow/compiler/mlir/xla/hlo_utils.cc", "CreateDenseAttrFromLiteral");

  auto data_span = literal.data<CppType>();
  return ::mlir::DenseElementsAttr::get(
      type, llvm::makeArrayRef(data_span.data(), data_span.size()));
}

StatusOr<AffineMap> GetPermutationIfAvailable(const Shape& shape,
                                              mlir::Builder builder) {
  // N.B. IsMonotonicWithDim0Major ignores tiling, and I can't change it because
  // some XLA code relies on it treating tiled layouts as equivalent to untiled
  // layouts, so the check to rule out tiling has to come /before/ the
  // early-return branch, or we'd miss tiled monotonic layouts.
  if (!shape.layout().tiles().empty()) {
    return tensorflow::errors::Internal("Tiled layouts are not yet supported");
  }
  if (!shape.has_layout() ||
      LayoutUtil::IsMonotonicWithDim0Major(shape.layout())) {
    return AffineMap();
  }
  if (!shape.is_static()) {
    return tensorflow::errors::Internal(
        "Permutations for dynamic shapes are not yet supported");
  }
  int64_t accumulated_stride = 1;
  llvm::SmallVector<int64_t, 4> strides(shape.rank(), 1);
  for (int64_t dim : LayoutUtil::MinorToMajor(shape)) {
    strides[dim] = accumulated_stride;
    accumulated_stride *= shape.dimensions(dim);
  }
  if (accumulated_stride == 0) {
    return AffineMap();
  }
  return makeStridedLinearLayoutMap(strides, /*offset=*/0,
                                    builder.getContext());
}

template <typename T>
void CopyDenseElementsBy(mlir::DenseElementsAttr data,
                         std::vector<uint8_t>* output) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSxlaPShlo_utilsDTcc mht_1(mht_1_v, 251, "", "./tensorflow/compiler/mlir/xla/hlo_utils.cc", "CopyDenseElementsBy");

  output->resize(data.getNumElements() * sizeof(T));
  int i = 0;
  for (T element : data.getValues<T>()) {
    std::memcpy(&(*output)[i], &element, sizeof(T));
    i += sizeof(T);
  }
}

}  // namespace

StatusOr<mlir::MemRefType> ConvertTensorShapeToMemRefType(
    const Shape& shape, mlir::Builder builder) {
  auto element_type_or =
      ConvertPrimitiveTypeToMLIRType(shape.element_type(), builder);
  if (!element_type_or.ok()) return element_type_or.status();

  using mlir::MemRefType;
  auto dimensions = shape.dimensions();
  llvm::SmallVector<int64_t, 4> array(dimensions.begin(), dimensions.end());
  auto permutation_or = GetPermutationIfAvailable(shape, builder);
  if (!permutation_or.ok()) return permutation_or.status();
  return MemRefType::get(array, element_type_or.ValueOrDie(),
                         permutation_or.ValueOrDie());
}

StatusOr<mlir::DenseElementsAttr> CreateDenseElementsAttrFromLiteral(
    const LiteralBase& literal, Builder builder) {
  TF_ASSIGN_OR_RETURN(auto type,
                      ConvertTensorShapeToType<mlir::RankedTensorType>(
                          literal.shape(), builder));

  // TODO(hinsu): Support remaining XLA primitive types.
  auto element_type = literal.shape().element_type();
  switch (element_type) {
    case PrimitiveType::PRED:
      return CreateDenseAttrFromLiteral<bool>(type, literal);
    case PrimitiveType::F16:
      return CreateDenseAttrFromLiteral<half>(type, literal);
    case PrimitiveType::BF16:
      return CreateDenseAttrFromLiteral<bfloat16>(type, literal);
    case PrimitiveType::F32:
      return CreateDenseAttrFromLiteral<float>(type, literal);
    case PrimitiveType::F64:
      return CreateDenseAttrFromLiteral<double>(type, literal);
    case PrimitiveType::S8:
      return CreateDenseAttrFromLiteral<int8_t>(type, literal);
    case PrimitiveType::S16:
      return CreateDenseAttrFromLiteral<int16_t>(type, literal);
    case PrimitiveType::S32:
      return CreateDenseAttrFromLiteral<int32_t>(type, literal);
    case PrimitiveType::S64:
      return CreateDenseAttrFromLiteral<int64_t>(type, literal);
    case PrimitiveType::U8:
      return CreateDenseAttrFromLiteral<uint8_t>(type, literal);
    case PrimitiveType::U16:
      return CreateDenseAttrFromLiteral<uint16_t>(type, literal);
    case PrimitiveType::U32:
      return CreateDenseAttrFromLiteral<uint32_t>(type, literal);
    case PrimitiveType::U64:
      return CreateDenseAttrFromLiteral<uint64_t>(type, literal);
    case PrimitiveType::C64:
      return CreateDenseAttrFromLiteral<complex64>(type, literal);
    case PrimitiveType::C128:
      return CreateDenseAttrFromLiteral<complex128>(type, literal);
    default:
      return tensorflow::errors::Internal(
          absl::StrCat("Unsupported type: ", PrimitiveType_Name(element_type)));
  }
}

Status CopyDenseElementsDataToXlaFormat(mlir::DenseElementsAttr data,
                                        std::vector<uint8_t>* output) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSxlaPShlo_utilsDTcc mht_2(mht_2_v, 326, "", "./tensorflow/compiler/mlir/xla/hlo_utils.cc", "CopyDenseElementsDataToXlaFormat");

  mlir::Type element_type = data.getType().getElementType();

  // TODO(hinsu): Support remaining XLA primitive types.
  if (element_type.isInteger(1)) {
    CopyDenseElementsBy<bool>(data, output);
    return Status::OK();
  }
  if (element_type.isInteger(8)) {
    CopyDenseElementsBy<uint8_t>(data, output);
    return Status::OK();
  }
  if (element_type.isInteger(16)) {
    CopyDenseElementsBy<uint16_t>(data, output);
    return Status::OK();
  }
  if (element_type.isInteger(32)) {
    CopyDenseElementsBy<uint32_t>(data, output);
    return Status::OK();
  }
  if (element_type.isInteger(64)) {
    CopyDenseElementsBy<uint64_t>(data, output);
    return Status::OK();
  }
  if (element_type.isBF16()) {
    CopyDenseElementsBy<bfloat16>(data, output);
    return Status::OK();
  }
  if (element_type.isF16()) {
    CopyDenseElementsBy<half>(data, output);
    return Status::OK();
  }
  if (element_type.isF32()) {
    CopyDenseElementsBy<float>(data, output);
    return Status::OK();
  }
  if (element_type.isF64()) {
    CopyDenseElementsBy<double>(data, output);
    return Status::OK();
  }
  if (auto complex_type = element_type.dyn_cast<mlir::ComplexType>()) {
    if (complex_type.getElementType().isF32()) {
      CopyDenseElementsBy<complex64>(data, output);
      return Status::OK();
    }
    if (complex_type.getElementType().isF64()) {
      CopyDenseElementsBy<complex128>(data, output);
      return Status::OK();
    }
  }
  return tensorflow::errors::Internal(
      "Unsupported type in CopyDenseElementsDataToXlaFormat");
}

StatusOr<int> GetElementTypeBytes(mlir::Type type) {
  if (type.isInteger(1)) {
    return 1;
  }
  if (auto complex_type = type.dyn_cast<mlir::ComplexType>()) {
    TF_ASSIGN_OR_RETURN(int bytes,
                        GetElementTypeBytes(complex_type.getElementType()));
    return bytes * 2;
  }
  int width = type.getIntOrFloatBitWidth();
  TF_RET_CHECK(width % 8 == 0);
  return width / 8;
}

mlir::DenseIntElementsAttr CreateDenseIntElementsAttrFromVector(
    const llvm::ArrayRef<int64_t> vector, mlir::Builder builder,
    llvm::ArrayRef<int64_t> shape) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSxlaPShlo_utilsDTcc mht_3(mht_3_v, 399, "", "./tensorflow/compiler/mlir/xla/hlo_utils.cc", "CreateDenseIntElementsAttrFromVector");

  return mlir::DenseIntElementsAttr::get(
      mlir::RankedTensorType::get(shape.empty() ? vector.size() : shape,
                                  builder.getIntegerType(64)),
      vector);
}

StatusOr<mlir::Type> ConvertPrimitiveTypeToMLIRType(PrimitiveType element_type,
                                                    mlir::Builder builder) {
  switch (element_type) {
    case PrimitiveType::PRED:
      return builder.getI1Type();
    case PrimitiveType::F16:
      return builder.getF16Type();
    case PrimitiveType::BF16:
      return builder.getBF16Type();
    case PrimitiveType::F32:
      return builder.getF32Type();
    case PrimitiveType::F64:
      return builder.getF64Type();
    case PrimitiveType::S8:
      return builder.getIntegerType(8);
    case PrimitiveType::S16:
      return builder.getIntegerType(16);
    case PrimitiveType::S32:
      return builder.getIntegerType(32);
    case PrimitiveType::S64:
      return builder.getIntegerType(64);
    case PrimitiveType::U8:
      return builder.getIntegerType(8, /*isSigned=*/false);
    case PrimitiveType::U16:
      return builder.getIntegerType(16, /*isSigned=*/false);
    case PrimitiveType::U32:
      return builder.getIntegerType(32, /*isSigned=*/false);
    case PrimitiveType::U64:
      return builder.getIntegerType(64, /*isSigned=*/false);
    case PrimitiveType::C64:
      return mlir::ComplexType::get(builder.getF32Type());
    case PrimitiveType::C128:
      return mlir::ComplexType::get(builder.getF64Type());
    // TODO(b/130356985): Support unsigned primitive types.
    default:
      return tensorflow::errors::Internal(
          absl::StrCat("Unsupported type: ", PrimitiveType_Name(element_type)));
  }
}

mlir::mhlo::GatherDimensionNumbersAttr CreateGatherDimensionNumbers(
    const GatherDimensionNumbers& input, mlir::Builder builder) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSxlaPShlo_utilsDTcc mht_4(mht_4_v, 450, "", "./tensorflow/compiler/mlir/xla/hlo_utils.cc", "CreateGatherDimensionNumbers");

  auto get_i64_array = [](absl::Span<const int64_t> container) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSxlaPShlo_utilsDTcc mht_5(mht_5_v, 454, "", "./tensorflow/compiler/mlir/xla/hlo_utils.cc", "lambda");

    return llvm::ArrayRef<int64_t>{container.data(), container.size()};
  };
  return mlir::mhlo::GatherDimensionNumbersAttr::get(
      builder.getContext(), get_i64_array(input.offset_dims()),
      get_i64_array(input.collapsed_slice_dims()),
      get_i64_array(input.start_index_map()), input.index_vector_dim());
}

StatusOr<::xla::HloOpcode> MhloToHloOpcode(mlir::Operation* op) {
  using mlir::isa;

  if (isa<mlir::mhlo::ConstOp, mlir::lmhlo::ConstOp>(op)) {
    return xla::HloOpcode::kConstant;
  } else if (isa<mlir::mhlo::IotaOp, mlir::lmhlo::IotaOp>(op)) {
    return xla::HloOpcode::kIota;
  } else if (isa<mlir::mhlo::ConvertOp, mlir::lmhlo::ConvertOp>(op)) {
    return xla::HloOpcode::kConvert;
  } else if (isa<mlir::mhlo::AddOp, mlir::lmhlo::AddOp>(op)) {
    return xla::HloOpcode::kAdd;
  } else if (isa<mlir::mhlo::Atan2Op, mlir::lmhlo::Atan2Op>(op)) {
    return xla::HloOpcode::kAtan2;
  } else if (isa<mlir::mhlo::DivOp, mlir::lmhlo::DivOp>(op)) {
    return xla::HloOpcode::kDivide;
  } else if (isa<mlir::mhlo::MaxOp, mlir::lmhlo::MaxOp>(op)) {
    return xla::HloOpcode::kMaximum;
  } else if (isa<mlir::mhlo::MinOp, mlir::lmhlo::MinOp>(op)) {
    return xla::HloOpcode::kMinimum;
  } else if (isa<mlir::mhlo::MulOp, mlir::lmhlo::MulOp>(op)) {
    return xla::HloOpcode::kMultiply;
  } else if (isa<mlir::mhlo::PowOp, mlir::lmhlo::PowOp>(op)) {
    return xla::HloOpcode::kPower;
  } else if (isa<mlir::mhlo::RemOp, mlir::lmhlo::RemOp>(op)) {
    return xla::HloOpcode::kRemainder;
  } else if (isa<mlir::mhlo::ShiftLeftOp, mlir::lmhlo::ShiftLeftOp>(op)) {
    return xla::HloOpcode::kShiftLeft;
  } else if (isa<mlir::mhlo::ShiftRightArithmeticOp,
                 mlir::lmhlo::ShiftRightArithmeticOp>(op)) {
    return xla::HloOpcode::kShiftRightArithmetic;
  } else if (isa<mlir::mhlo::ShiftRightLogicalOp,
                 mlir::lmhlo::ShiftRightLogicalOp>(op)) {
    return xla::HloOpcode::kShiftRightLogical;
  } else if (isa<mlir::mhlo::SubOp, mlir::lmhlo::SubOp>(op)) {
    return xla::HloOpcode::kSubtract;
  } else if (isa<mlir::mhlo::XorOp, mlir::lmhlo::XorOp>(op)) {
    return xla::HloOpcode::kXor;
  } else if (isa<mlir::mhlo::InfeedOp, mlir::lmhlo::InfeedOp>(op)) {
    return xla::HloOpcode::kInfeed;
  } else if (isa<mlir::mhlo::OutfeedOp, mlir::lmhlo::OutfeedOp>(op)) {
    return xla::HloOpcode::kOutfeed;
  } else if (isa<mlir::mhlo::SendOp>(op)) {
    return xla::HloOpcode::kSend;
  } else if (isa<mlir::mhlo::RecvOp>(op)) {
    return xla::HloOpcode::kRecv;
  } else if (isa<mlir::mhlo::ReplicaIdOp, mlir::lmhlo::ReplicaIdOp>(op)) {
    return xla::HloOpcode::kReplicaId;
  } else if (isa<mlir::mhlo::AfterAllOp>(op)) {
    return xla::HloOpcode::kAfterAll;
  } else if (isa<mlir::mhlo::AllReduceOp, mlir::lmhlo::AllReduceOp>(op)) {
    return xla::HloOpcode::kAllReduce;
  } else if (isa<mlir::mhlo::AllToAllOp>(op)) {
    return xla::HloOpcode::kAllToAll;
  } else if (isa<mlir::mhlo::TupleOp>(op)) {
    return xla::HloOpcode::kTuple;
  } else if (isa<mlir::mhlo::BatchNormGradOp, mlir::lmhlo::BatchNormGradOp>(
                 op)) {
    return xla::HloOpcode::kBatchNormGrad;
  } else if (isa<mlir::mhlo::BatchNormInferenceOp,
                 mlir::lmhlo::BatchNormInferenceOp>(op)) {
    return xla::HloOpcode::kBatchNormInference;
  } else if (isa<mlir::mhlo::BatchNormTrainingOp,
                 mlir::lmhlo::BatchNormTrainingOp>(op)) {
    return xla::HloOpcode::kBatchNormTraining;
  } else if (isa<mlir::mhlo::BitcastConvertOp, mlir::lmhlo::BitcastConvertOp>(
                 op)) {
    return xla::HloOpcode::kBitcastConvert;
  } else if (isa<mlir::mhlo::BroadcastOp, mlir::lmhlo::BroadcastOp>(op)) {
    return xla::HloOpcode::kBroadcast;
  } else if (isa<mlir::mhlo::CholeskyOp, mlir::lmhlo::CholeskyOp>(op)) {
    return xla::HloOpcode::kCholesky;
  } else if (isa<mlir::mhlo::ClampOp, mlir::lmhlo::ClampOp>(op)) {
    return xla::HloOpcode::kClamp;
  } else if (isa<mlir::mhlo::ConcatenateOp, mlir::lmhlo::ConcatenateOp>(op)) {
    return xla::HloOpcode::kConcatenate;
  } else if (isa<mlir::mhlo::ConvOp, mlir::lmhlo::ConvOp>(op)) {
    return xla::HloOpcode::kConvolution;
  } else if (isa<mlir::mhlo::SortOp, mlir::lmhlo::SortOp>(op)) {
    return xla::HloOpcode::kSort;
  } else if (isa<mlir::mhlo::RngBitGeneratorOp>(op)) {
    return xla::HloOpcode::kRngBitGenerator;
  } else if (isa<mlir::mhlo::XlaRngGetAndUpdateStateOp>(op)) {
    return xla::HloOpcode::kRngGetAndUpdateState;
  } else if (isa<mlir::mhlo::FusionOp, mlir::lmhlo::FusionOp>(op)) {
    return xla::HloOpcode::kFusion;
  } else if (isa<mlir::mhlo::BitcastOp>(op)) {
    return xla::HloOpcode::kBitcast;
  } else if (isa<mlir::mhlo::AbsOp, mlir::lmhlo::AbsOp>(op)) {
    return xla::HloOpcode::kAbs;
  } else if (isa<mlir::mhlo::CbrtOp, mlir::lmhlo::CbrtOp>(op)) {
    return xla::HloOpcode::kCbrt;
  } else if (isa<mlir::mhlo::CeilOp, mlir::lmhlo::CeilOp>(op)) {
    return xla::HloOpcode::kCeil;
  } else if (isa<mlir::mhlo::ClzOp, mlir::lmhlo::ClzOp>(op)) {
    return xla::HloOpcode::kClz;
  } else if (isa<mlir::mhlo::CosOp, mlir::lmhlo::CosOp>(op)) {
    return xla::HloOpcode::kCos;
  } else if (isa<mlir::mhlo::ExpOp, mlir::lmhlo::ExpOp>(op)) {
    return xla::HloOpcode::kExp;
  } else if (isa<mlir::mhlo::Expm1Op, mlir::lmhlo::Expm1Op>(op)) {
    return xla::HloOpcode::kExpm1;
  } else if (isa<mlir::mhlo::FloorOp, mlir::lmhlo::FloorOp>(op)) {
    return xla::HloOpcode::kFloor;
  } else if (isa<mlir::mhlo::ImagOp, mlir::lmhlo::ImagOp>(op)) {
    return xla::HloOpcode::kImag;
  } else if (isa<mlir::mhlo::IsFiniteOp, mlir::lmhlo::IsFiniteOp>(op)) {
    return xla::HloOpcode::kIsFinite;
  } else if (isa<mlir::mhlo::LogOp, mlir::lmhlo::LogOp>(op)) {
    return xla::HloOpcode::kLog;
  } else if (isa<mlir::mhlo::Log1pOp, mlir::lmhlo::Log1pOp>(op)) {
    return xla::HloOpcode::kLog1p;
  } else if (isa<mlir::mhlo::LogisticOp>(op)) {
    return xla::HloOpcode::kLogistic;
  } else if (isa<mlir::mhlo::NotOp, mlir::lmhlo::NotOp>(op)) {
    return xla::HloOpcode::kNot;
  } else if (isa<mlir::mhlo::NegOp, mlir::lmhlo::NegOp>(op)) {
    return xla::HloOpcode::kNegate;
  } else if (isa<mlir::mhlo::PopulationCountOp, mlir::lmhlo::PopulationCountOp>(
                 op)) {
    return xla::HloOpcode::kPopulationCount;
  } else if (isa<mlir::mhlo::RealOp, mlir::lmhlo::RealOp>(op)) {
    return xla::HloOpcode::kReal;
  } else if (isa<mlir::mhlo::RoundOp, mlir::lmhlo::RoundOp>(op)) {
    return xla::HloOpcode::kRoundNearestAfz;
  } else if (isa<mlir::mhlo::RsqrtOp, mlir::lmhlo::RsqrtOp>(op)) {
    return xla::HloOpcode::kRsqrt;
  } else if (isa<mlir::mhlo::SignOp, mlir::lmhlo::SignOp>(op)) {
    return xla::HloOpcode::kSign;
  } else if (isa<mlir::mhlo::SinOp, mlir::lmhlo::SinOp>(op)) {
    return xla::HloOpcode::kSin;
  } else if (isa<mlir::mhlo::SqrtOp, mlir::lmhlo::SqrtOp>(op)) {
    return xla::HloOpcode::kSqrt;
  } else if (isa<mlir::mhlo::TanhOp, mlir::lmhlo::TanhOp>(op)) {
    return xla::HloOpcode::kTanh;
  } else if (isa<mlir::mhlo::ComplexOp, mlir::lmhlo::ComplexOp>(op)) {
    return xla::HloOpcode::kComplex;
  } else if (isa<mlir::mhlo::AndOp, mlir::lmhlo::AndOp>(op)) {
    return xla::HloOpcode::kAnd;
  } else if (isa<mlir::mhlo::OrOp, mlir::lmhlo::OrOp>(op)) {
    return xla::HloOpcode::kOr;
  } else if (isa<mlir::mhlo::WhileOp, mlir::lmhlo::WhileOp>(op)) {
    return xla::HloOpcode::kWhile;
  } else if (isa<mlir::mhlo::ReduceOp, mlir::lmhlo::ReduceOp>(op)) {
    return xla::HloOpcode::kReduce;
  } else if (isa<mlir::mhlo::GetTupleElementOp>(op)) {
    return xla::HloOpcode::kGetTupleElement;
  } else if (isa<mlir::mhlo::CompareOp, mlir::lmhlo::CompareOp>(op)) {
    return xla::HloOpcode::kCompare;
  } else if (isa<mlir::mhlo::SliceOp, mlir::lmhlo::SliceOp>(op)) {
    return xla::HloOpcode::kSlice;
  } else if (isa<mlir::mhlo::DynamicSliceOp, mlir::lmhlo::DynamicSliceOp>(op)) {
    return xla::HloOpcode::kDynamicSlice;
  } else if (isa<mlir::mhlo::DynamicUpdateSliceOp,
                 mlir::lmhlo::DynamicUpdateSliceOp>(op)) {
    return xla::HloOpcode::kDynamicUpdateSlice;
  } else if (isa<mlir::mhlo::CollectivePermuteOp,
                 mlir::lmhlo::CollectivePermuteOp>(op)) {
    return xla::HloOpcode::kCollectivePermute;
  } else if (isa<mlir::mhlo::CopyOp, mlir::lmhlo::CopyOp>(op)) {
    return xla::HloOpcode::kCopy;
  } else if (isa<mlir::mhlo::CustomCallOp, mlir::lmhlo::CustomCallOp>(op)) {
    return xla::HloOpcode::kCustomCall;
  } else if (isa<mlir::mhlo::DotOp, mlir::lmhlo::DotOp>(op)) {
    return xla::HloOpcode::kDot;
  } else if (isa<mlir::mhlo::FftOp, mlir::lmhlo::FftOp>(op)) {
    return xla::HloOpcode::kFft;
  } else if (isa<mlir::mhlo::GatherOp, mlir::lmhlo::GatherOp>(op)) {
    return xla::HloOpcode::kGather;
  } else if (isa<mlir::mhlo::GetDimensionSizeOp>(op)) {
    return xla::HloOpcode::kGetDimensionSize;
  } else if (isa<mlir::mhlo::MapOp, mlir::lmhlo::MapOp>(op)) {
    return xla::HloOpcode::kMap;
  } else if (isa<mlir::mhlo::ReshapeOp, mlir::lmhlo::ReshapeOp>(op)) {
    return xla::HloOpcode::kReshape;
  } else if (isa<mlir::mhlo::DynamicReshapeOp>(op)) {
    return xla::HloOpcode::kDynamicReshape;
  } else if (isa<mlir::mhlo::ScatterOp, mlir::lmhlo::ScatterOp>(op)) {
    return xla::HloOpcode::kScatter;
  } else if (isa<mlir::mhlo::SelectOp, mlir::lmhlo::SelectOp>(op)) {
    return xla::HloOpcode::kSelect;
  } else if (isa<mlir::mhlo::SelectAndScatterOp,
                 mlir::lmhlo::SelectAndScatterOp>(op)) {
    return xla::HloOpcode::kSelectAndScatter;
  } else if (isa<mlir::mhlo::SetDimensionSizeOp>(op)) {
    return xla::HloOpcode::kSetDimensionSize;
  } else if (isa<mlir::mhlo::ReverseOp, mlir::lmhlo::ReverseOp>(op)) {
    return xla::HloOpcode::kReverse;
  } else if (isa<mlir::mhlo::PadOp, mlir::lmhlo::PadOp>(op)) {
    return xla::HloOpcode::kPad;
  } else if (isa<mlir::mhlo::TraceOp>(op)) {
    return xla::HloOpcode::kTrace;
  } else if (isa<mlir::mhlo::TransposeOp, mlir::lmhlo::TransposeOp>(op)) {
    return xla::HloOpcode::kTranspose;
  } else if (isa<mlir::mhlo::TriangularSolveOp, mlir::lmhlo::TriangularSolveOp>(
                 op)) {
    return xla::HloOpcode::kTriangularSolve;
  } else if (isa<mlir::mhlo::ReduceWindowOp, mlir::lmhlo::ReduceWindowOp>(op)) {
    return xla::HloOpcode::kReduceWindow;
  } else if (isa<mlir::mhlo::ReducePrecisionOp, mlir::lmhlo::ReducePrecisionOp>(
                 op)) {
    return xla::HloOpcode::kReducePrecision;
  } else if (isa<mlir::mhlo::DotGeneralOp>(op)) {
    return xla::HloOpcode::kDot;
  } else if (isa<mlir::mhlo::BroadcastInDimOp, mlir::lmhlo::BroadcastInDimOp>(
                 op)) {
    return xla::HloOpcode::kBroadcast;
  } else {
    std::string s;
    {
      llvm::raw_string_ostream os(s);
      op->print(os);
    }
    return tensorflow::errors::Unimplemented(
        "Unimplemented MHLO -> HloOpcode: ", s);
  }
}

}  // namespace xla
