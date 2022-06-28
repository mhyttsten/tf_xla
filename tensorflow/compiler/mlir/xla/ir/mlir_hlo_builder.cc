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
class MHTracer_DTPStensorflowPScompilerPSmlirPSxlaPSirPSmlir_hlo_builderDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSmlirPSxlaPSirPSmlir_hlo_builderDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSmlirPSxlaPSirPSmlir_hlo_builderDTcc() {
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
#include "tensorflow/compiler/mlir/xla/ir/mlir_hlo_builder.h"

#include <string>
#include <utility>

#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/hlo/include/mlir-hlo/Dialect/mhlo/IR/hlo_ops.h"
#include "tensorflow/compiler/mlir/xla/attribute_importer.h"
#include "tensorflow/compiler/mlir/xla/hlo_function_importer.h"
#include "tensorflow/compiler/mlir/xla/hlo_utils.h"
#include "tensorflow/compiler/mlir/xla/type_to_shape.h"
#include "tensorflow/compiler/xla/comparison_util.h"
#include "tensorflow/compiler/xla/service/hlo_module.h"
#include "tensorflow/compiler/xla/service/shape_inference.h"
#include "tensorflow/compiler/xla/util.h"

namespace xla {

static std::string GetMlirOpName(HloOpcode opcode) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSxlaPSirPSmlir_hlo_builderDTcc mht_0(mht_0_v, 205, "", "./tensorflow/compiler/mlir/xla/ir/mlir_hlo_builder.cc", "GetMlirOpName");

  std::string op_name = HloOpcodeString(opcode);
  absl::c_replace(op_name, '-', '_');
  return mlir::mhlo::MhloDialect::getDialectNamespace().str() + "." + op_name;
}

static std::string ToString(mlir::Type ty) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSxlaPSirPSmlir_hlo_builderDTcc mht_1(mht_1_v, 214, "", "./tensorflow/compiler/mlir/xla/ir/mlir_hlo_builder.cc", "ToString");

  std::string str;
  llvm::raw_string_ostream sstream(str);
  ty.print(sstream);
  sstream.flush();
  return str;
}

// Returns 1D 64-bit dense elements attribute with the given values.
static mlir::DenseIntElementsAttr GetI64ElementsAttr(
    absl::Span<const int64_t> values, mlir::Builder* builder) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSxlaPSirPSmlir_hlo_builderDTcc mht_2(mht_2_v, 227, "", "./tensorflow/compiler/mlir/xla/ir/mlir_hlo_builder.cc", "GetI64ElementsAttr");

  auto ty = mlir::RankedTensorType::get({static_cast<int64_t>(values.size())},
                                        builder->getIntegerType(64));
  return mlir::DenseIntElementsAttr::get(
      ty, llvm::makeArrayRef(values.data(), values.size()));
}

static mlir::DenseIntElementsAttr ConvertPadding(
    absl::Span<const std::pair<int64_t, int64_t>> padding,
    mlir::Builder* builder) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSxlaPSirPSmlir_hlo_builderDTcc mht_3(mht_3_v, 239, "", "./tensorflow/compiler/mlir/xla/ir/mlir_hlo_builder.cc", "ConvertPadding");

  llvm::SmallVector<int64_t, 8> elements;
  elements.reserve(padding.size() * 2);
  for (const auto& vals : padding) {
    elements.push_back(vals.first);
    elements.push_back(vals.second);
  }
  auto ty = mlir::RankedTensorType::get(
      {static_cast<int64_t>(padding.size()), 2}, builder->getIntegerType(64));
  return mlir::DenseIntElementsAttr::get(ty, elements);
}

MlirHloBuilder::~MlirHloBuilder() = default;

StatusOr<XlaOp> MlirHloBuilder::MakeXlaOp(mlir::Value val) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSxlaPSirPSmlir_hlo_builderDTcc mht_4(mht_4_v, 256, "", "./tensorflow/compiler/mlir/xla/ir/mlir_hlo_builder.cc", "MlirHloBuilder::MakeXlaOp");

  mlir::Type ty = val.getType();
  auto shape = std::make_unique<Shape>(TypeToShape(ty));
  if (shape->element_type() == PrimitiveType::PRIMITIVE_TYPE_INVALID) {
    return InvalidArgument("unsupported type: %s", ToString(ty).c_str());
  }

  int64_t handle = reinterpret_cast<int64_t>(val.getAsOpaquePointer());
  handle_to_shape_[handle] = std::move(shape);
  return XlaOp(handle, this);
}

XlaOp MlirHloBuilder::ConstantLiteral(const LiteralSlice& literal) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSxlaPSirPSmlir_hlo_builderDTcc mht_5(mht_5_v, 271, "", "./tensorflow/compiler/mlir/xla/ir/mlir_hlo_builder.cc", "MlirHloBuilder::ConstantLiteral");

  return ReportErrorOrReturn([&]() -> StatusOr<XlaOp> {
    TF_ASSIGN_OR_RETURN(mlir::DenseElementsAttr attr,
                        CreateDenseElementsAttrFromLiteral(literal, builder_));
    auto op = builder_.create<mlir::mhlo::ConstOp>(loc_, attr);
    return MakeXlaOp(op);
  });
}

StatusOr<XlaOp> MlirHloBuilder::ConvGeneralDilatedInternal(
    const Shape& shape, XlaOp lhs, XlaOp rhs, const Window& window,
    absl::Span<const int64_t> window_strides,
    absl::Span<const std::pair<int64_t, int64_t>> padding,
    absl::Span<const int64_t> lhs_dilation,
    absl::Span<const int64_t> rhs_dilation,
    const ConvolutionDimensionNumbers& dimension_numbers,
    int64_t feature_group_count, int64_t batch_group_count,
    const PrecisionConfig* precision_config) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSxlaPSirPSmlir_hlo_builderDTcc mht_6(mht_6_v, 291, "", "./tensorflow/compiler/mlir/xla/ir/mlir_hlo_builder.cc", "MlirHloBuilder::ConvGeneralDilatedInternal");

  TF_ASSIGN_OR_RETURN(mlir::Type ty, ConvertShapeToType<mlir::RankedTensorType>(
                                         shape, builder_));
  mlir::ArrayAttr config_attr;
  if (precision_config)
    config_attr = ConvertPrecisionConfig(precision_config, &builder_);
  auto op = builder_.create<mlir::mhlo::ConvOp>(
      loc_, ty, GetValue(lhs), GetValue(rhs),
      GetI64ElementsAttr(window_strides, &builder_),
      ConvertPadding(padding, &builder_),
      GetI64ElementsAttr(lhs_dilation, &builder_),
      GetI64ElementsAttr(rhs_dilation, &builder_),
      /*window_reversal=*/nullptr,
      ConvertConvDimensionNumbers(dimension_numbers, &builder_),
      builder_.getI64IntegerAttr(feature_group_count),
      builder_.getI64IntegerAttr(batch_group_count), config_attr);
  return MakeXlaOp(op);
}

StatusOr<XlaOp> MlirHloBuilder::FftInternal(
    const Shape& shape, XlaOp operand, FftType fft_type,
    absl::Span<const int64_t> fft_length) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSxlaPSirPSmlir_hlo_builderDTcc mht_7(mht_7_v, 315, "", "./tensorflow/compiler/mlir/xla/ir/mlir_hlo_builder.cc", "MlirHloBuilder::FftInternal");

  TF_ASSIGN_OR_RETURN(mlir::Type ty, ConvertShapeToType<mlir::RankedTensorType>(
                                         shape, builder_));
  auto fft_type_attr = mlir::mhlo::symbolizeFftType(FftType_Name(fft_type));
  auto op = builder_.create<mlir::mhlo::FftOp>(
      loc_, ty, GetValue(operand),
      mlir::mhlo::FftTypeAttr::get(builder_.getContext(),
                                   fft_type_attr.getValue()),
      GetI64ElementsAttr(fft_length, &builder_));
  return MakeXlaOp(op);
}

StatusOr<XlaOp> MlirHloBuilder::CustomCallInternal(
    const std::string& call_target_name, absl::Span<const XlaOp> operands,
    const Shape& shape, const std::string& opaque,
    absl::optional<absl::Span<const Shape>> operand_shapes_with_layout,
    bool has_side_effect,
    absl::Span<const std::pair<ShapeIndex, std::pair<int64_t, ShapeIndex>>>
        output_operand_aliasing,
    const Literal* literal, absl::optional<Window> window,
    absl::optional<ConvolutionDimensionNumbers> dnums,
    CustomCallSchedule schedule, CustomCallApiVersion api_version) {
   std::vector<std::string> mht_8_v;
   mht_8_v.push_back("call_target_name: \"" + call_target_name + "\"");
   mht_8_v.push_back("opaque: \"" + opaque + "\"");
   MHTracer_DTPStensorflowPScompilerPSmlirPSxlaPSirPSmlir_hlo_builderDTcc mht_8(mht_8_v, 341, "", "./tensorflow/compiler/mlir/xla/ir/mlir_hlo_builder.cc", "MlirHloBuilder::CustomCallInternal");

  TF_RET_CHECK(output_operand_aliasing.empty())
      << "MLIR CustomCallOp does not support output_operand_aliasing yet";
  TF_RET_CHECK(literal == nullptr)
      << "MLIR CustomCallOp does not support literal yet";
  TF_RET_CHECK(!window.has_value())
      << "MLIR CustomCallOp does not support ConvolutionDimensionNumbers yet";
  TF_RET_CHECK(!dnums.has_value())
      << "MLIR CustomCallOp does not support ConvolutionDimensionNumbers yet";
  TF_RET_CHECK(schedule == CustomCallSchedule::SCHEDULE_NONE)
      << "MLIR CustomCallOp does not support custom-call-schedule yet";

  llvm::SmallVector<mlir::NamedAttribute> attributes;
  if (operand_shapes_with_layout.has_value()) {
    TF_ASSIGN_OR_RETURN(mlir::ArrayAttr operand_layouts,
                        ExtractLayoutsFromShapes(
                            operand_shapes_with_layout.value(), &builder_));
    attributes.push_back(
        builder_.getNamedAttr("operand_layouts", operand_layouts));

    mlir::ArrayAttr result_layouts;
    if (shape.IsTuple()) {
      TF_ASSIGN_OR_RETURN(result_layouts,
                          ExtractLayoutsFromTuple(shape, &builder_));
    } else {
      TF_ASSIGN_OR_RETURN(result_layouts,
                          ExtractLayoutsFromShapes({shape}, &builder_));
    }
    attributes.push_back(
        builder_.getNamedAttr("result_layouts", result_layouts));
  }
  TF_ASSIGN_OR_RETURN(mlir::Type ty, ConvertShapeToType<mlir::RankedTensorType>(
                                         shape, builder_));
  TF_ASSIGN_OR_RETURN(auto mlir_api_version,
                      ConvertCustomCallApiVersion(api_version));
  attributes.push_back(builder_.getNamedAttr(
      "api_version", mlir::mhlo::CustomCallApiVersionAttr::get(
                         builder_.getContext(), mlir_api_version)));

  attributes.push_back(builder_.getNamedAttr(
      "call_target_name", builder_.getStringAttr(call_target_name)));
  attributes.push_back(builder_.getNamedAttr(
      "has_side_effect", builder_.getBoolAttr(has_side_effect)));
  attributes.push_back(
      builder_.getNamedAttr("backend_config", builder_.getStringAttr(opaque)));

  auto op = builder_.create<mlir::mhlo::CustomCallOp>(
      loc_, ty, GetValues(operands), attributes);
  return MakeXlaOp(op.getResult(0));
}

StatusOr<XlaOp> MlirHloBuilder::ReduceInternal(
    const Shape& shape, absl::Span<const XlaOp> all_operands,
    const XlaComputation& computation,
    absl::Span<const int64_t> dimensions_to_reduce) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSxlaPSirPSmlir_hlo_builderDTcc mht_9(mht_9_v, 398, "", "./tensorflow/compiler/mlir/xla/ir/mlir_hlo_builder.cc", "MlirHloBuilder::ReduceInternal");

  // Reduce takes two set of variadic operands inputs and init_values.
  // all_operands contains both of these so split operands into two parts.
  int64_t num_args = all_operands.size() / 2;
  auto op = builder_.create<mlir::mhlo::ReduceOp>(
      loc_, GetValues(all_operands.first(num_args)),
      GetValues(all_operands.subspan(num_args)),
      GetI64ElementsAttr(dimensions_to_reduce, &builder_));
  TF_RETURN_IF_ERROR(ImportComputation(computation.proto(), &op.body(),
                                       /*flatten_region_arg_tuple*/ true));
  if (op.getNumResults() == 1) return MakeXlaOp(op.getResult(0));
  auto tuple = builder_.create<mlir::mhlo::TupleOp>(loc_, op.getResults());
  return MakeXlaOp(tuple);
}

StatusOr<XlaOp> MlirHloBuilder::ReduceWindowInternal(
    const Shape& shape, XlaOp operand, XlaOp init_value,
    const XlaComputation& computation, Window window) {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSxlaPSirPSmlir_hlo_builderDTcc mht_10(mht_10_v, 418, "", "./tensorflow/compiler/mlir/xla/ir/mlir_hlo_builder.cc", "MlirHloBuilder::ReduceWindowInternal");

  TF_ASSIGN_OR_RETURN(mlir::Type ty, ConvertShapeToType<mlir::RankedTensorType>(
                                         shape, builder_));
  llvm::SmallVector<int64_t, 4> sizes, strides, base_dilations, win_dilations;
  llvm::SmallVector<int64_t, 8> padding;
  for (const auto& dim : window.dimensions()) {
    sizes.push_back(dim.size());
    strides.push_back(dim.stride());
    base_dilations.push_back(dim.base_dilation());
    win_dilations.push_back(dim.window_dilation());
    padding.push_back(dim.padding_low());
    padding.push_back(dim.padding_high());
  }
  auto padding_ty =
      mlir::RankedTensorType::get({static_cast<int64_t>(padding.size()) / 2, 2},
                                  builder_.getIntegerType(64));
  auto op = builder_.create<mlir::mhlo::ReduceWindowOp>(
      loc_, ty, GetValue(operand), GetValue(init_value),
      GetI64ElementsAttr(sizes, &builder_),
      GetI64ElementsAttr(strides, &builder_),
      GetI64ElementsAttr(base_dilations, &builder_),
      GetI64ElementsAttr(win_dilations, &builder_),
      mlir::DenseIntElementsAttr::get(padding_ty, padding));
  TF_RETURN_IF_ERROR(ImportComputation(computation.proto(), &op.body(),
                                       /*flatten_region_arg_tuple*/ true));
  return MakeXlaOp(op.getResult(0));
}

XlaOp MlirHloBuilder::Iota(const Shape& shape, int64_t iota_dimension) {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSxlaPSirPSmlir_hlo_builderDTcc mht_11(mht_11_v, 449, "", "./tensorflow/compiler/mlir/xla/ir/mlir_hlo_builder.cc", "MlirHloBuilder::Iota");

  return ReportErrorOrReturn([&]() -> StatusOr<XlaOp> {
    TF_ASSIGN_OR_RETURN(
        mlir::Type ty,
        ConvertShapeToType<mlir::RankedTensorType>(shape, builder_));
    auto op = builder_.create<mlir::mhlo::IotaOp>(
        loc_, ty,
        builder_.getIntegerAttr(builder_.getI64Type(), iota_dimension));
    return MakeXlaOp(op);
  });
}

StatusOr<XlaOp> MlirHloBuilder::BitcastConvertTypeInternal(const Shape& shape,
                                                           XlaOp operand) {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSxlaPSirPSmlir_hlo_builderDTcc mht_12(mht_12_v, 465, "", "./tensorflow/compiler/mlir/xla/ir/mlir_hlo_builder.cc", "MlirHloBuilder::BitcastConvertTypeInternal");

  TF_ASSIGN_OR_RETURN(mlir::Type ty, ConvertShapeToType<mlir::RankedTensorType>(
                                         shape, builder_));
  auto op = builder_.create<mlir::mhlo::BitcastConvertOp>(loc_, ty,
                                                          GetValue(operand));
  return MakeXlaOp(op);
}

StatusOr<XlaOp> MlirHloBuilder::TransposeInternal(
    const Shape& shape, XlaOp operand, absl::Span<const int64_t> permutation) {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSxlaPSirPSmlir_hlo_builderDTcc mht_13(mht_13_v, 477, "", "./tensorflow/compiler/mlir/xla/ir/mlir_hlo_builder.cc", "MlirHloBuilder::TransposeInternal");

  TF_ASSIGN_OR_RETURN(mlir::Type ty, ConvertShapeToType<mlir::RankedTensorType>(
                                         shape, builder_));
  auto op = builder_.create<mlir::mhlo::TransposeOp>(
      loc_, ty, GetValue(operand), GetI64ElementsAttr(permutation, &builder_));
  return MakeXlaOp(op);
}

StatusOr<XlaOp> MlirHloBuilder::RevInternal(
    const Shape& shape, XlaOp operand, absl::Span<const int64_t> dimensions) {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSxlaPSirPSmlir_hlo_builderDTcc mht_14(mht_14_v, 489, "", "./tensorflow/compiler/mlir/xla/ir/mlir_hlo_builder.cc", "MlirHloBuilder::RevInternal");

  TF_ASSIGN_OR_RETURN(mlir::Type ty, ConvertShapeToType<mlir::RankedTensorType>(
                                         shape, builder_));
  auto op = builder_.create<mlir::mhlo::ReverseOp>(
      loc_, ty, GetValue(operand), GetI64ElementsAttr(dimensions, &builder_));
  return MakeXlaOp(op);
}

StatusOr<XlaOp> MlirHloBuilder::SortInternal(const Shape& shape,
                                             absl::Span<const XlaOp> operands,
                                             const XlaComputation& comparator,
                                             int64_t dimension,
                                             bool is_stable) {
   std::vector<std::string> mht_15_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSxlaPSirPSmlir_hlo_builderDTcc mht_15(mht_15_v, 504, "", "./tensorflow/compiler/mlir/xla/ir/mlir_hlo_builder.cc", "MlirHloBuilder::SortInternal");

  TF_ASSIGN_OR_RETURN(mlir::Type ty, ConvertShapeToType<mlir::RankedTensorType>(
                                         shape, builder_));
  llvm::SmallVector<mlir::Type, 4> sort_types = {ty};
  if (auto tuple_ty = ty.dyn_cast<mlir::TupleType>()) {
    sort_types = llvm::to_vector<6>(tuple_ty.getTypes());
  }

  auto op = builder_.create<mlir::mhlo::SortOp>(
      loc_, sort_types, GetValues(operands),
      builder_.getI64IntegerAttr(dimension), builder_.getBoolAttr(is_stable));
  TF_RETURN_IF_ERROR(ImportComputation(comparator.proto(), &op.comparator()));

  if (ty.isa<mlir::TupleType>()) {
    auto tuple = builder_.create<mlir::mhlo::TupleOp>(loc_, op.getResults());
    return MakeXlaOp(tuple);
  }

  return MakeXlaOp(op.getResult(0));
}

StatusOr<XlaOp> MlirHloBuilder::WhileInternal(const Shape& shape,
                                              const XlaComputation& condition,
                                              const XlaComputation& body,
                                              XlaOp init) {
   std::vector<std::string> mht_16_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSxlaPSirPSmlir_hlo_builderDTcc mht_16(mht_16_v, 531, "", "./tensorflow/compiler/mlir/xla/ir/mlir_hlo_builder.cc", "MlirHloBuilder::WhileInternal");

  TF_ASSIGN_OR_RETURN(mlir::Type ty, ConvertShapeToType<mlir::RankedTensorType>(
                                         shape, builder_));

  llvm::SmallVector<mlir::Value> flattened_operands;
  llvm::SmallVector<mlir::Type> flattened_operand_types;

  HloFunctionImporter::FlattenTupleType(ty, flattened_operand_types);
  HloFunctionImporter::FlattenTupleValue(&builder_, loc_, GetValue(init),
                                         flattened_operands);

  auto op = builder_.create<mlir::mhlo::WhileOp>(loc_, flattened_operand_types,
                                                 flattened_operands);

  TF_RETURN_IF_ERROR(ImportComputation(condition.proto(), &op.cond(),
                                       /*flatten_region_arg_tuple*/ true));
  TF_RETURN_IF_ERROR(ImportComputation(body.proto(), &op.body(),
                                       /*flatten_region_arg_tuple*/ true));

  if (ty.isa<mlir::TupleType>()) {
    llvm::SmallVector<mlir::Value> flattened_results = op->getResults();
    llvm::MutableArrayRef<mlir::Value> flattened_results_ref(flattened_results);
    auto result = HloFunctionImporter::CreateTupleValue(
        &builder_, loc_, flattened_results_ref, ty);
    auto defining_tuple_op = result.getDefiningOp<mlir::mhlo::TupleOp>();
    return MakeXlaOp(defining_tuple_op);
  }

  return MakeXlaOp(op.getResult(0));
}

StatusOr<XlaOp> MlirHloBuilder::ReducePrecisionInternal(
    const Shape& shape, XlaOp operand, const int exponent_bits,
    const int mantissa_bits) {
   std::vector<std::string> mht_17_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSxlaPSirPSmlir_hlo_builderDTcc mht_17(mht_17_v, 567, "", "./tensorflow/compiler/mlir/xla/ir/mlir_hlo_builder.cc", "MlirHloBuilder::ReducePrecisionInternal");

  TF_ASSIGN_OR_RETURN(mlir::Type ty, ConvertShapeToType<mlir::RankedTensorType>(
                                         shape, builder_));
  auto op = builder_.create<mlir::mhlo::ReducePrecisionOp>(
      loc_, ty, GetValue(operand), builder_.getI32IntegerAttr(exponent_bits),
      builder_.getI32IntegerAttr(mantissa_bits));
  return MakeXlaOp(op);
}

StatusOr<XlaOp> MlirHloBuilder::GatherInternal(
    const Shape& shape, XlaOp input, XlaOp start_indices,
    const GatherDimensionNumbers& dimension_numbers,
    absl::Span<const int64_t> slice_sizes, bool indices_are_sorted) {
   std::vector<std::string> mht_18_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSxlaPSirPSmlir_hlo_builderDTcc mht_18(mht_18_v, 582, "", "./tensorflow/compiler/mlir/xla/ir/mlir_hlo_builder.cc", "MlirHloBuilder::GatherInternal");

  TF_ASSIGN_OR_RETURN(mlir::Type ty, ConvertShapeToType<mlir::RankedTensorType>(
                                         shape, builder_));
  auto op = builder_.create<mlir::mhlo::GatherOp>(
      loc_, ty, GetValue(input), GetValue(start_indices),
      ConvertGatherDimensionNumbers(dimension_numbers, &builder_),
      GetI64ElementsAttr(slice_sizes, &builder_));
  return MakeXlaOp(op);
}

StatusOr<XlaOp> MlirHloBuilder::ScatterInternal(
    const Shape& shape, XlaOp input, XlaOp scatter_indices, XlaOp updates,
    const XlaComputation& update_computation,
    const ScatterDimensionNumbers& dimension_numbers, bool indices_are_sorted,
    bool unique_indices) {
   std::vector<std::string> mht_19_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSxlaPSirPSmlir_hlo_builderDTcc mht_19(mht_19_v, 599, "", "./tensorflow/compiler/mlir/xla/ir/mlir_hlo_builder.cc", "MlirHloBuilder::ScatterInternal");

  TF_ASSIGN_OR_RETURN(mlir::Type ty, ConvertShapeToType<mlir::RankedTensorType>(
                                         shape, builder_));
  auto op = builder_.create<mlir::mhlo::ScatterOp>(
      loc_, ty, GetValue(input), GetValue(scatter_indices), GetValue(updates),
      ConvertScatterDimensionNumbers(dimension_numbers, &builder_),
      builder_.getBoolAttr(indices_are_sorted),
      builder_.getBoolAttr(unique_indices));

  TF_RETURN_IF_ERROR(
      ImportComputation(update_computation.proto(), &op.update_computation()));
  return MakeXlaOp(op);
}

StatusOr<XlaOp> MlirHloBuilder::SetDimensionSizeInternal(const Shape& shape,
                                                         XlaOp operand,
                                                         XlaOp val,
                                                         int64_t dimension) {
   std::vector<std::string> mht_20_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSxlaPSirPSmlir_hlo_builderDTcc mht_20(mht_20_v, 619, "", "./tensorflow/compiler/mlir/xla/ir/mlir_hlo_builder.cc", "MlirHloBuilder::SetDimensionSizeInternal");

  TF_ASSIGN_OR_RETURN(mlir::Type ty, ConvertShapeToType<mlir::RankedTensorType>(
                                         shape, builder_));
  auto op = builder_.create<mlir::mhlo::SetDimensionSizeOp>(
      loc_, ty, GetValue(operand), GetValue(val),
      builder_.getI64IntegerAttr(dimension));
  return MakeXlaOp(op);
}

StatusOr<XlaOp> MlirHloBuilder::RngOpInternal(
    RandomDistribution distribution, absl::Span<const XlaOp> parameters,
    const Shape& shape) {
   std::vector<std::string> mht_21_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSxlaPSirPSmlir_hlo_builderDTcc mht_21(mht_21_v, 633, "", "./tensorflow/compiler/mlir/xla/ir/mlir_hlo_builder.cc", "MlirHloBuilder::RngOpInternal");

  // TODO(hinsu): Introduce RngOp in the HLO dialect in MLIR and then RngUniform
  // and RngNormal can be mapped to the new op.
  std::string op_name;
  if (distribution == xla::RandomDistribution::RNG_UNIFORM) {
    op_name = "mhlo.rng_uniform";
  } else {
    TF_RET_CHECK(distribution == xla::RandomDistribution::RNG_NORMAL)
        << "Unexpected distribution: " << distribution;
    op_name = "mhlo.rng_normal";
  }

  if (shape.is_dynamic())
    return Unimplemented("RngOp with dynamic dims not supported");
  llvm::SmallVector<XlaOp, 3> operands;
  operands.append(parameters.begin(), parameters.end());
  operands.push_back(
      ConstantLiteral(LiteralUtil::CreateR1<int64_t>(shape.dimensions())));
  return CreateOp(op_name, shape, operands);
}

StatusOr<XlaOp> MlirHloBuilder::RngBitGeneratorInternal(
    const Shape& full_result_shape, RandomAlgorithm algorithm,
    XlaOp initial_state) {
   std::vector<std::string> mht_22_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSxlaPSirPSmlir_hlo_builderDTcc mht_22(mht_22_v, 659, "", "./tensorflow/compiler/mlir/xla/ir/mlir_hlo_builder.cc", "MlirHloBuilder::RngBitGeneratorInternal");

  TF_ASSIGN_OR_RETURN(mlir::Type ty, ConvertShapeToType<mlir::RankedTensorType>(
                                         full_result_shape, builder_));

  llvm::SmallVector<mlir::Type> flattened_ret_types;
  HloFunctionImporter::FlattenTupleType(ty, flattened_ret_types);

  auto op = builder_.create<mlir::mhlo::RngBitGeneratorOp>(
      loc_, flattened_ret_types, builder_.getI32IntegerAttr(algorithm),
      GetValue(initial_state));

  if (ty.isa<mlir::TupleType>()) {
    llvm::SmallVector<mlir::Value> flattened_results = op->getResults();
    llvm::MutableArrayRef<mlir::Value> flattened_results_ref(flattened_results);
    auto result = HloFunctionImporter::CreateTupleValue(
        &builder_, loc_, flattened_results_ref, ty);
    auto defining_tuple_op = result.getDefiningOp<mlir::mhlo::TupleOp>();
    return MakeXlaOp(defining_tuple_op);
  }

  return MakeXlaOp(op.getResult(0));
}

StatusOr<XlaOp> MlirHloBuilder::ReshapeInternal(const Shape& shape,
                                                XlaOp operand,
                                                int64_t inferred_dimension) {
   std::vector<std::string> mht_23_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSxlaPSirPSmlir_hlo_builderDTcc mht_23(mht_23_v, 687, "", "./tensorflow/compiler/mlir/xla/ir/mlir_hlo_builder.cc", "MlirHloBuilder::ReshapeInternal");

  TF_RETURN_IF_ERROR(first_error());

  if (inferred_dimension != -1)
    return Unimplemented("inferred_dimension not yet supported for Reshape op");
  TF_ASSIGN_OR_RETURN(mlir::Type ty, ConvertShapeToType<mlir::RankedTensorType>(
                                         shape, builder_));
  mlir::Value value = GetValue(operand);
  auto op = builder_.create<mlir::mhlo::ReshapeOp>(loc_, ty, value);
  return MakeXlaOp(op.getResult());
}

StatusOr<XlaOp> MlirHloBuilder::DotGeneralInternal(
    const Shape& shape, XlaOp lhs, XlaOp rhs,
    const DotDimensionNumbers& dimension_number,
    const PrecisionConfig* precision_config) {
   std::vector<std::string> mht_24_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSxlaPSirPSmlir_hlo_builderDTcc mht_24(mht_24_v, 705, "", "./tensorflow/compiler/mlir/xla/ir/mlir_hlo_builder.cc", "MlirHloBuilder::DotGeneralInternal");

  TF_ASSIGN_OR_RETURN(mlir::Type ty, ConvertShapeToType<mlir::RankedTensorType>(
                                         shape, builder_));
  auto op = builder_.create<mlir::mhlo::DotGeneralOp>(
      loc_, ty, GetValue(lhs), GetValue(rhs),
      ConvertDotDimensionNumbers(dimension_number, &builder_),
      ConvertPrecisionConfig(precision_config, &builder_));
  return MakeXlaOp(op.getResult());
}

StatusOr<XlaOp> MlirHloBuilder::InDimBroadcast(
    const Shape& shape, XlaOp operand,
    absl::Span<const int64_t> broadcast_dimensions) {
   std::vector<std::string> mht_25_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSxlaPSirPSmlir_hlo_builderDTcc mht_25(mht_25_v, 720, "", "./tensorflow/compiler/mlir/xla/ir/mlir_hlo_builder.cc", "MlirHloBuilder::InDimBroadcast");

  TF_RETURN_IF_ERROR(first_error());
  TF_ASSIGN_OR_RETURN(mlir::Type ty, ConvertShapeToType<mlir::RankedTensorType>(
                                         shape, builder_));
  mlir::Value value = GetValue(operand);
  auto op = builder_.create<mlir::mhlo::BroadcastInDimOp>(
      loc_, ty, value, GetI64ElementsAttr(broadcast_dimensions, &builder_));
  return MakeXlaOp(op.getResult());
}

StatusOr<XlaOp> MlirHloBuilder::AddInstruction(
    HloInstructionProto&& instr, HloOpcode opcode,
    absl::Span<const XlaOp> operands) {
   std::vector<std::string> mht_26_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSxlaPSirPSmlir_hlo_builderDTcc mht_26(mht_26_v, 735, "", "./tensorflow/compiler/mlir/xla/ir/mlir_hlo_builder.cc", "MlirHloBuilder::AddInstruction");

  return Unimplemented("MlirHloBuilder does not support op %s",
                       HloOpcodeString(opcode));
}

StatusOr<XlaOp> MlirHloBuilder::Compare(const Shape& shape, XlaOp lhs,
                                        XlaOp rhs,
                                        ComparisonDirection direction,
                                        Comparison::Type type) {
   std::vector<std::string> mht_27_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSxlaPSirPSmlir_hlo_builderDTcc mht_27(mht_27_v, 746, "", "./tensorflow/compiler/mlir/xla/ir/mlir_hlo_builder.cc", "MlirHloBuilder::Compare");

  TF_ASSIGN_OR_RETURN(mlir::Type ty, ConvertShapeToType<mlir::RankedTensorType>(
                                         shape, builder_));
  auto op = builder_.create<mlir::mhlo::CompareOp>(
      loc_, ty, GetValue(lhs), GetValue(rhs),
      mlir::mhlo::ComparisonDirectionAttr::get(
          builder_.getContext(), mlir::mhlo::symbolizeComparisonDirection(
                                     ComparisonDirectionToString(direction))
                                     .getValue()),
      mlir::mhlo::ComparisonTypeAttr::get(
          builder_.getContext(),
          mlir::mhlo::symbolizeComparisonType(ComparisonTypeToString(type))
              .getValue()));
  return MakeXlaOp(op.getResult());
}

XlaOp MlirHloBuilder::BinaryOpNoBroadcast(HloOpcode binop, const Shape& shape,
                                          XlaOp lhs, XlaOp rhs) {
   std::vector<std::string> mht_28_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSxlaPSirPSmlir_hlo_builderDTcc mht_28(mht_28_v, 766, "", "./tensorflow/compiler/mlir/xla/ir/mlir_hlo_builder.cc", "MlirHloBuilder::BinaryOpNoBroadcast");

  return ReportErrorOrReturn([&]() -> StatusOr<XlaOp> {
    return CreateOp(GetMlirOpName(binop), shape, {lhs, rhs});
  });
}

StatusOr<XlaOp> MlirHloBuilder::AddOpWithShape(
    HloOpcode opcode, const Shape& shape, absl::Span<const XlaOp> operands) {
   std::vector<std::string> mht_29_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSxlaPSirPSmlir_hlo_builderDTcc mht_29(mht_29_v, 776, "", "./tensorflow/compiler/mlir/xla/ir/mlir_hlo_builder.cc", "MlirHloBuilder::AddOpWithShape");

  return CreateOp(GetMlirOpName(opcode), shape,
                  llvm::makeArrayRef<XlaOp>(operands.data(), operands.size()));
}

XlaOp MlirHloBuilder::CreateToken() {
   std::vector<std::string> mht_30_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSxlaPSirPSmlir_hlo_builderDTcc mht_30(mht_30_v, 784, "", "./tensorflow/compiler/mlir/xla/ir/mlir_hlo_builder.cc", "MlirHloBuilder::CreateToken");

  return ReportErrorOrReturn([&]() -> StatusOr<XlaOp> {
    return MakeXlaOp(builder_.create<mlir::mhlo::CreateTokenOp>(
        loc_, mlir::mhlo::TokenType::get(builder_.getContext())));
  });
}

StatusOr<XlaOp> MlirHloBuilder::TriangularSolveInternal(
    const Shape& shape, XlaOp a, XlaOp b, TriangularSolveOptions options) {
   std::vector<std::string> mht_31_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSxlaPSirPSmlir_hlo_builderDTcc mht_31(mht_31_v, 795, "", "./tensorflow/compiler/mlir/xla/ir/mlir_hlo_builder.cc", "MlirHloBuilder::TriangularSolveInternal");

  TF_ASSIGN_OR_RETURN(
      mlir::Type result_ty,
      ConvertShapeToType<mlir::RankedTensorType>(shape, builder_));
  auto op = builder_.create<mlir::mhlo::TriangularSolveOp>(
      loc_, result_ty, GetValue(a), GetValue(b),
      builder_.getBoolAttr(options.left_side()),
      builder_.getBoolAttr(options.lower()),
      builder_.getBoolAttr(options.unit_diagonal()),
      mlir::mhlo::TransposeAttr::get(
          builder_.getContext(),
          ::mlir::mhlo::symbolizeTranspose(
              TriangularSolveOptions::Transpose_Name(options.transpose_a()))
              .getValue()));
  return MakeXlaOp(op);
}

StatusOr<XlaOp> MlirHloBuilder::CholeskyInternal(const Shape& shape, XlaOp a,
                                                 bool lower) {
   std::vector<std::string> mht_32_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSxlaPSirPSmlir_hlo_builderDTcc mht_32(mht_32_v, 816, "", "./tensorflow/compiler/mlir/xla/ir/mlir_hlo_builder.cc", "MlirHloBuilder::CholeskyInternal");

  TF_ASSIGN_OR_RETURN(
      mlir::Type result_ty,
      ConvertShapeToType<mlir::RankedTensorType>(shape, builder_));
  auto op = builder_.create<mlir::mhlo::CholeskyOp>(
      loc_, result_ty, GetValue(a), builder_.getBoolAttr(lower));
  return MakeXlaOp(op);
}

StatusOr<XlaOp> MlirHloBuilder::InfeedWithTokenInternal(
    const Shape& infeed_instruction_shape, XlaOp token,
    const std::string& config) {
   std::vector<std::string> mht_33_v;
   mht_33_v.push_back("config: \"" + config + "\"");
   MHTracer_DTPStensorflowPScompilerPSmlirPSxlaPSirPSmlir_hlo_builderDTcc mht_33(mht_33_v, 831, "", "./tensorflow/compiler/mlir/xla/ir/mlir_hlo_builder.cc", "MlirHloBuilder::InfeedWithTokenInternal");

  TF_ASSIGN_OR_RETURN(mlir::Type result_type,
                      ConvertShapeToType<mlir::RankedTensorType>(
                          infeed_instruction_shape, builder_));
  llvm::SmallVector<mlir::Type> flattened_ret_types;
  HloFunctionImporter::FlattenTupleType(result_type, flattened_ret_types);

  mlir::ArrayAttr layout;
  auto op = builder_.create<mlir::mhlo::InfeedOp>(loc_, flattened_ret_types,
                                                  GetValue(token),
                                                  /*infeed_config=*/config,
                                                  /*layout=*/layout);

  llvm::SmallVector<mlir::Value> flattened_results = op->getResults();
  llvm::MutableArrayRef<mlir::Value> flattened_results_ref(flattened_results);
  auto result = HloFunctionImporter::CreateTupleValue(
      &builder_, loc_, flattened_results_ref, result_type);
  auto defining_tuple_op = result.getDefiningOp<mlir::mhlo::TupleOp>();
  return MakeXlaOp(defining_tuple_op);
}

StatusOr<XlaOp> MlirHloBuilder::OutfeedWithTokenInternal(
    XlaOp operand, XlaOp token, const Shape& shape_with_layout,
    const std::string& outfeed_config) {
   std::vector<std::string> mht_34_v;
   mht_34_v.push_back("outfeed_config: \"" + outfeed_config + "\"");
   MHTracer_DTPStensorflowPScompilerPSmlirPSxlaPSirPSmlir_hlo_builderDTcc mht_34(mht_34_v, 858, "", "./tensorflow/compiler/mlir/xla/ir/mlir_hlo_builder.cc", "MlirHloBuilder::OutfeedWithTokenInternal");

  auto token_type = mlir::mhlo::TokenType::get(builder_.getContext());
  llvm::SmallVector<mlir::Value> flattened_operands;
  HloFunctionImporter::FlattenTupleValue(&builder_, loc_, GetValue(operand),
                                         flattened_operands);
  return MakeXlaOp(builder_.create<mlir::mhlo::OutfeedOp>(
      loc_, token_type, flattened_operands, GetValue(token), outfeed_config));
}

StatusOr<XlaOp> MlirHloBuilder::ConcatInDimInternal(
    const Shape& shape, absl::Span<const XlaOp> operands, int64_t dimension) {
   std::vector<std::string> mht_35_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSxlaPSirPSmlir_hlo_builderDTcc mht_35(mht_35_v, 871, "", "./tensorflow/compiler/mlir/xla/ir/mlir_hlo_builder.cc", "MlirHloBuilder::ConcatInDimInternal");

  TF_ASSIGN_OR_RETURN(
      mlir::Type result_type,
      ConvertShapeToType<mlir::RankedTensorType>(shape, builder_));
  auto mlir_operands = GetValues(operands);
  return MakeXlaOp(builder_.create<mlir::mhlo::ConcatenateOp>(
      loc_, result_type, mlir_operands, builder_.getI64IntegerAttr(dimension)));
}

StatusOr<XlaOp> MlirHloBuilder::GetTupleElementInternal(const Shape& shape,
                                                        XlaOp tuple_data,
                                                        int64_t index) {
   std::vector<std::string> mht_36_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSxlaPSirPSmlir_hlo_builderDTcc mht_36(mht_36_v, 885, "", "./tensorflow/compiler/mlir/xla/ir/mlir_hlo_builder.cc", "MlirHloBuilder::GetTupleElementInternal");

  TF_ASSIGN_OR_RETURN(
      mlir::Type result_type,
      ConvertShapeToType<mlir::RankedTensorType>(shape, builder_));
  return MakeXlaOp(builder_.create<mlir::mhlo::GetTupleElementOp>(
      loc_, result_type, GetValue(tuple_data),
      builder_.getI32IntegerAttr(index)));
}

StatusOr<XlaOp> MlirHloBuilder::SliceInternal(
    const Shape& shape, XlaOp operand, absl::Span<const int64_t> start_indices,
    absl::Span<const int64_t> limit_indices,
    absl::Span<const int64_t> strides) {
   std::vector<std::string> mht_37_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSxlaPSirPSmlir_hlo_builderDTcc mht_37(mht_37_v, 900, "", "./tensorflow/compiler/mlir/xla/ir/mlir_hlo_builder.cc", "MlirHloBuilder::SliceInternal");

  return MakeXlaOp(builder_.create<mlir::mhlo::SliceOp>(
      loc_, GetValue(operand), GetI64ElementsAttr(start_indices, &builder_),
      GetI64ElementsAttr(limit_indices, &builder_),
      GetI64ElementsAttr(strides, &builder_)));
}

StatusOr<XlaOp> MlirHloBuilder::DynamicSliceInternal(
    const Shape& shape, XlaOp operand, absl::Span<const XlaOp> start_indices,
    absl::Span<const int64_t> slice_sizes) {
   std::vector<std::string> mht_38_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSxlaPSirPSmlir_hlo_builderDTcc mht_38(mht_38_v, 912, "", "./tensorflow/compiler/mlir/xla/ir/mlir_hlo_builder.cc", "MlirHloBuilder::DynamicSliceInternal");

  TF_ASSIGN_OR_RETURN(
      mlir::Type result_ty,
      ConvertShapeToType<mlir::RankedTensorType>(shape, builder_));
  return MakeXlaOp(builder_.create<mlir::mhlo::DynamicSliceOp>(
      loc_, result_ty, GetValue(operand), GetValues(start_indices),
      GetI64ElementsAttr(slice_sizes, &builder_)));
}

StatusOr<XlaOp> MlirHloBuilder::DynamicUpdateSliceInternal(
    const Shape& shape, XlaOp operand, XlaOp update,
    absl::Span<const XlaOp> start_indices) {
   std::vector<std::string> mht_39_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSxlaPSirPSmlir_hlo_builderDTcc mht_39(mht_39_v, 926, "", "./tensorflow/compiler/mlir/xla/ir/mlir_hlo_builder.cc", "MlirHloBuilder::DynamicUpdateSliceInternal");

  TF_ASSIGN_OR_RETURN(
      mlir::Type result_ty,
      ConvertShapeToType<mlir::RankedTensorType>(shape, builder_));
  return MakeXlaOp(builder_.create<mlir::mhlo::DynamicUpdateSliceOp>(
      loc_, result_ty, GetValue(operand), GetValue(update),
      GetValues(start_indices)));
}

StatusOr<XlaOp> MlirHloBuilder::PadInternal(
    const Shape& shape, XlaOp operand, XlaOp padding_value,
    const PaddingConfig& padding_config) {
   std::vector<std::string> mht_40_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSxlaPSirPSmlir_hlo_builderDTcc mht_40(mht_40_v, 940, "", "./tensorflow/compiler/mlir/xla/ir/mlir_hlo_builder.cc", "MlirHloBuilder::PadInternal");

  TF_ASSIGN_OR_RETURN(
      mlir::Type result_type,
      ConvertShapeToType<mlir::RankedTensorType>(shape, builder_));
  llvm::SmallVector<int64_t> low, high, internal;
  for (auto& dimension : padding_config.dimensions()) {
    low.push_back(dimension.edge_padding_low());
    high.push_back(dimension.edge_padding_high());
    internal.push_back(dimension.interior_padding());
  }
  return MakeXlaOp(builder_.create<mlir::mhlo::PadOp>(
      loc_, result_type, GetValue(operand), GetValue(padding_value),
      GetI64ElementsAttr(low, &builder_), GetI64ElementsAttr(high, &builder_),
      GetI64ElementsAttr(internal, &builder_)));
}

StatusOr<XlaOp> MlirHloBuilder::TupleInternal(
    const Shape& shape, absl::Span<const XlaOp> elements) {
   std::vector<std::string> mht_41_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSxlaPSirPSmlir_hlo_builderDTcc mht_41(mht_41_v, 960, "", "./tensorflow/compiler/mlir/xla/ir/mlir_hlo_builder.cc", "MlirHloBuilder::TupleInternal");

  mlir::SmallVector<mlir::Value, 4> operands;
  for (auto& element : elements) {
    operands.push_back(GetValue(element));
  }
  return MakeXlaOp(builder_.create<mlir::mhlo::TupleOp>(loc_, operands));
}

StatusOr<XlaOp> MlirHloBuilder::CreateOp(
    const std::string& op_name, const Shape& shape,
    llvm::ArrayRef<XlaOp> operands,
    llvm::ArrayRef<mlir::NamedAttribute> attributes) {
   std::vector<std::string> mht_42_v;
   mht_42_v.push_back("op_name: \"" + op_name + "\"");
   MHTracer_DTPStensorflowPScompilerPSmlirPSxlaPSirPSmlir_hlo_builderDTcc mht_42(mht_42_v, 975, "", "./tensorflow/compiler/mlir/xla/ir/mlir_hlo_builder.cc", "MlirHloBuilder::CreateOp");

  llvm::SmallVector<mlir::Value, 4> operand_values;
  operand_values.reserve(operands.size());
  for (XlaOp xla_op : operands) {
    operand_values.push_back(GetValue(xla_op));
  }
  TF_ASSIGN_OR_RETURN(mlir::Type ty, ConvertShapeToType<mlir::RankedTensorType>(
                                         shape, builder_));
  mlir::OperationState state(loc_, op_name, operand_values, {ty}, attributes);
  mlir::Operation* op = builder_.create(state);
  return MakeXlaOp(op->getResult(0));
}

Status MlirHloBuilder::ImportComputation(const HloModuleProto& computation,
                                         mlir::Region* region,
                                         bool flatten_region_arg_tuple) {
   std::vector<std::string> mht_43_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSxlaPSirPSmlir_hlo_builderDTcc mht_43(mht_43_v, 993, "", "./tensorflow/compiler/mlir/xla/ir/mlir_hlo_builder.cc", "MlirHloBuilder::ImportComputation");

  TF_ASSIGN_OR_RETURN(auto module_config,
                      xla::HloModule::CreateModuleConfigFromProto(
                          computation, xla::DebugOptions()));
  TF_ASSIGN_OR_RETURN(auto hlo_module, xla::HloModule::CreateFromProto(
                                           computation, module_config));

  return HloFunctionImporter::ImportAsRegion(*hlo_module->entry_computation(),
                                             region, &builder_,
                                             flatten_region_arg_tuple);
}

StatusOr<const Shape*> MlirHloBuilder::GetShapePtr(XlaOp op) const {
   std::vector<std::string> mht_44_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSxlaPSirPSmlir_hlo_builderDTcc mht_44(mht_44_v, 1008, "", "./tensorflow/compiler/mlir/xla/ir/mlir_hlo_builder.cc", "MlirHloBuilder::GetShapePtr");

  TF_RETURN_IF_ERROR(first_error());
  TF_RETURN_IF_ERROR(CheckOpBuilder(op));
  auto it = handle_to_shape_.find(op.handle());
  if (it == handle_to_shape_.end()) {
    return InvalidArgument("No XlaOp with handle %d", op.handle());
  }
  return it->second.get();
}

}  // namespace xla
