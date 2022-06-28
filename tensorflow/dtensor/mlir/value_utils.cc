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
class MHTracer_DTPStensorflowPSdtensorPSmlirPSvalue_utilsDTcc {
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
   MHTracer_DTPStensorflowPSdtensorPSmlirPSvalue_utilsDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSdtensorPSmlirPSvalue_utilsDTcc() {
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

/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/dtensor/mlir/value_utils.h"

#include "mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/transforms/collection_ops_util.h"
#include "tensorflow/dtensor/mlir/ir/tf_dtensor.h"
#include "tensorflow/dtensor/mlir/op_utils.h"

namespace tensorflow {
namespace dtensor {
namespace {

// Given a mlir::Value will trace the value back through
// DTensorLayout and basic blocks of while loops.
// This is like a reverse version of TraceUseToNextTFOp.
mlir::Value GetForwardedInput(mlir::Value value) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSdtensorPSmlirPSvalue_utilsDTcc mht_0(mht_0_v, 199, "", "./tensorflow/dtensor/mlir/value_utils.cc", "GetForwardedInput");

  bool value_updated;
  do {
    value_updated = false;
    if (mlir::BlockArgument argument = value.dyn_cast<mlir::BlockArgument>()) {
      mlir::Region* region = argument.getParentRegion();
      if (region == nullptr) break;
      mlir::Operation* parent_op = region->getParentOp();
      // TODO(bfontain): handle if and other control flow blocks.
      if (mlir::TF::WhileRegionOp while_op =
              mlir::dyn_cast<mlir::TF::WhileRegionOp>(parent_op)) {
        value = while_op.getOperand(argument.getArgNumber());
        value_updated = true;
      }
    } else {
      mlir::Operation* op = value.getDefiningOp();
      // TODO(bfontain): Add cases for identity and control flow return values.
      if (mlir::TF::DTensorLayout layout_op =
              mlir::dyn_cast<mlir::TF::DTensorLayout>(op)) {
        value = layout_op.input();
        value_updated = true;
      }
    }
  } while (value_updated);

  return value;
}
}  // namespace

namespace ops_util = ::mlir::TF::collection_ops_util;

int ValueRank(mlir::Value operand_value) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSdtensorPSmlirPSvalue_utilsDTcc mht_1(mht_1_v, 233, "", "./tensorflow/dtensor/mlir/value_utils.cc", "ValueRank");

  mlir::Type type = GetSubtypeOrSelf(operand_value);
  const auto operand_type = type.cast<mlir::TensorType>();
  if (!operand_type.hasRank()) return -1;
  return operand_type.getRank();
}

mlir::RankedTensorType EffectivelyScalarR1Type(mlir::Type element_type) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSdtensorPSmlirPSvalue_utilsDTcc mht_2(mht_2_v, 243, "", "./tensorflow/dtensor/mlir/value_utils.cc", "EffectivelyScalarR1Type");

  return mlir::RankedTensorType::get({1}, element_type);
}

mlir::Value ReshapeSizeTypeToScalar(mlir::OpBuilder builder, mlir::Location loc,
                                    mlir::Value tensor) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPSdtensorPSmlirPSvalue_utilsDTcc mht_3(mht_3_v, 251, "", "./tensorflow/dtensor/mlir/value_utils.cc", "ReshapeSizeTypeToScalar");

  auto scalar_type =
      mlir::RankedTensorType::get({}, builder.getIntegerType(32));
  mlir::Value scalar_shape =
      ops_util::GetR1Const(scalar_type.getShape(), builder, loc);
  return builder.create<mlir::TF::ReshapeOp>(
      loc, mlir::ArrayRef<mlir::Type>{scalar_type},
      mlir::ArrayRef<mlir::Value>{tensor, scalar_shape});
}

mlir::Value IntConst(mlir::OpBuilder& builder, mlir::Location loc,
                     llvm::ArrayRef<int32> values) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPSdtensorPSmlirPSvalue_utilsDTcc mht_4(mht_4_v, 265, "", "./tensorflow/dtensor/mlir/value_utils.cc", "IntConst");

  auto const_type = mlir::RankedTensorType::get(
      {static_cast<int64>(values.size())}, builder.getIntegerType(32));
  mlir::Attribute const_attr =
      mlir::DenseIntElementsAttr::get(const_type, values);
  return builder.create<mlir::TF::ConstOp>(loc, const_attr).getResult();
}

mlir::Value Int64Const(mlir::OpBuilder& builder, mlir::Location loc,
                       llvm::ArrayRef<int64_t> values) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPSdtensorPSmlirPSvalue_utilsDTcc mht_5(mht_5_v, 277, "", "./tensorflow/dtensor/mlir/value_utils.cc", "Int64Const");

  auto const_type = mlir::RankedTensorType::get(
      {static_cast<int64>(values.size())}, builder.getIntegerType(64));
  mlir::Attribute const_attr =
      mlir::DenseIntElementsAttr::get(const_type, values);
  return builder.create<mlir::TF::ConstOp>(loc, const_attr).getResult();
}

mlir::Value FloatConst(mlir::OpBuilder& builder, mlir::Location loc,
                       llvm::ArrayRef<float> values) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPSdtensorPSmlirPSvalue_utilsDTcc mht_6(mht_6_v, 289, "", "./tensorflow/dtensor/mlir/value_utils.cc", "FloatConst");

  mlir::RankedTensorType const_type = mlir::RankedTensorType::get(
      {static_cast<int64>(values.size())}, builder.getF32Type());
  mlir::Attribute const_attr =
      mlir::DenseFPElementsAttr::get(const_type, values);
  return builder.create<mlir::TF::ConstOp>(loc, const_attr).getResult();
}

mlir::Value StringConst(mlir::OpBuilder& builder, mlir::Location loc,
                        llvm::ArrayRef<llvm::StringRef> values) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPSdtensorPSmlirPSvalue_utilsDTcc mht_7(mht_7_v, 301, "", "./tensorflow/dtensor/mlir/value_utils.cc", "StringConst");

  auto const_type =
      mlir::RankedTensorType::get({static_cast<int64>(values.size())},
                                  builder.getType<mlir::TF::StringType>());
  mlir::Attribute const_attr =
      mlir::DenseStringElementsAttr::get(const_type, values);
  return builder.create<mlir::TF::ConstOp>(loc, const_attr).getResult();
}

StatusOr<int64_t> ExtractConstIntFromValue(mlir::Value value) {
  value = GetForwardedInput(value);
  if (value.isa<mlir::BlockArgument>())
    return errors::Internal("unable get constant value from block argument");
  mlir::DenseIntElementsAttr attr;
  if (!matchPattern(value, m_Constant(&attr))) {
    return errors::Internal(absl::StrCat("required constant value for ",
                                         OpName(value.getDefiningOp())));
  }
  if (attr.size() != 1) {
    return errors::Internal(absl::StrCat("expected 1 element, got ",
                                         attr.size(), " for ",
                                         OpName(value.getDefiningOp())));
  }
  auto a = *attr.value_begin<llvm::APInt>();
  return a.getSExtValue();
}

Status ExtractConstVectorFromValue(mlir::Value value,
                                   llvm::SmallVector<int64_t, 4>* out_vector) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPSdtensorPSmlirPSvalue_utilsDTcc mht_8(mht_8_v, 332, "", "./tensorflow/dtensor/mlir/value_utils.cc", "ExtractConstVectorFromValue");

  value = GetForwardedInput(value);
  if (value.isa<mlir::BlockArgument>())
    return errors::Internal("unable get constant value from block argument");
  mlir::DenseIntElementsAttr attr;
  if (!matchPattern(value, m_Constant(&attr))) {
    return errors::Internal(
        absl::StrCat("failed to extract constant value from ",
                     value.getDefiningOp()->getName().getStringRef().str()));
  }
  for (const mlir::APInt& index : attr)
    out_vector->emplace_back(index.getSExtValue());
  return Status::OK();
}

mlir::Value CreateIntScalarConst(const int64_t value, mlir::OpBuilder builder,
                                 mlir::Location loc, bool use_int64) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPSdtensorPSmlirPSvalue_utilsDTcc mht_9(mht_9_v, 351, "", "./tensorflow/dtensor/mlir/value_utils.cc", "CreateIntScalarConst");

  if (use_int64) {
    return builder.create<mlir::TF::ConstOp>(
        loc, mlir::DenseIntElementsAttr::get(
                 mlir::RankedTensorType::get({}, builder.getI64Type()), value));
  } else {
    return builder.create<mlir::TF::ConstOp>(
        loc, mlir::DenseIntElementsAttr::get(
                 mlir::RankedTensorType::get({}, builder.getI32Type()),
                 static_cast<int32_t>(value)));
  }
}

absl::optional<mlir::Value> CreateZeroScalarConst(mlir::OpBuilder& builder,
                                                  mlir::Location loc,
                                                  mlir::Type type) {
  if (type.isF64()) {
    return builder.create<mlir::TF::ConstOp>(
        loc, mlir::DenseFPElementsAttr::get(
                 mlir::RankedTensorType::get({}, builder.getF64Type()),
                 static_cast<double>(0.)));
  } else if (type.isF32()) {
    return builder.create<mlir::TF::ConstOp>(
        loc, mlir::DenseFPElementsAttr::get(
                 mlir::RankedTensorType::get({}, builder.getF32Type()),
                 static_cast<float>(0.f)));
  } else if (type.isInteger(32)) {
    return builder.create<mlir::TF::ConstOp>(
        loc, mlir::DenseIntElementsAttr::get(
                 mlir::RankedTensorType::get({}, builder.getI32Type()),
                 static_cast<int32_t>(0)));
  } else if (type.isInteger(64)) {
    return builder.create<mlir::TF::ConstOp>(
        loc, mlir::DenseIntElementsAttr::get(
                 mlir::RankedTensorType::get({}, builder.getI64Type()),
                 static_cast<int64_t>(0)));
  } else {
    return absl::nullopt;
  }
}

mlir::Type GetSubtypeOrSelf(mlir::Value val) {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPSdtensorPSmlirPSvalue_utilsDTcc mht_10(mht_10_v, 395, "", "./tensorflow/dtensor/mlir/value_utils.cc", "GetSubtypeOrSelf");

  mlir::Type type = val.getType();
  if (auto type_with_subtype =
          mlir::getElementTypeOrSelf(val)
              .dyn_cast<mlir::TF::TensorFlowTypeWithSubtype>()) {
    if (type_with_subtype.GetSubtypes().size() == 1) {
      type = type_with_subtype.GetSubtypes().front();
    }
  }
  return type;
}

}  // namespace dtensor
}  // namespace tensorflow
