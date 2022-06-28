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
class MHTracer_DTPStensorflowPScompilerPSmlirPSlitePSflatbuffer_operatorDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePSflatbuffer_operatorDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSmlirPSlitePSflatbuffer_operatorDTcc() {
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

#include "tensorflow/compiler/mlir/lite/flatbuffer_operator.h"

#include <vector>

#include "absl/strings/str_cat.h"
#include "flatbuffers/flatbuffers.h"  // from @flatbuffers
#include "flatbuffers/flexbuffers.h"  // from @flatbuffers
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/ADT/Twine.h"
#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/lite/ir/tfl_ops.h"
#include "tensorflow/compiler/mlir/lite/utils/convert_type.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_types.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/lite/kernels/internal/kernel_utils.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/schema/schema_utils.h"

namespace {

using ::tensorflow::Status;
using ::tensorflow::errors::InvalidArgument;
using ::xla::StatusOr;

StatusOr<mlir::StringAttr> GetPaddingAttr(TfLitePadding pad_params,
                                          mlir::Builder builder,
                                          mlir::Location loc) {
  auto padding = tflite::Padding::Padding_VALID;
  if (pad_params == TfLitePadding::kTfLitePaddingSame) {
    padding = tflite::Padding_SAME;
  } else if (pad_params == TfLitePadding::kTfLitePaddingValid) {
    padding = tflite::Padding_VALID;
  } else {
    return InvalidArgument(
        absl::StrCat("Invalid padding type", std::to_string(pad_params)));
  }

  const char* option_name = tflite::EnumNamePadding(padding);
  return builder.getStringAttr(option_name);
}

}  // namespace

std::string mlir::GetMlirOpNameFromOpCode(
    const tflite::OperatorCodeT& op_code) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePSflatbuffer_operatorDTcc mht_0(mht_0_v, 234, "", "./tensorflow/compiler/mlir/lite/flatbuffer_operator.cc", "mlir::GetMlirOpNameFromOpCode");

  auto builtin_code = tflite::GetBuiltinCode(&op_code);
  if (builtin_code == tflite::BuiltinOperator_CUSTOM) {
    return std::string("tfl.custom");
  }
  if (builtin_code == tflite::BuiltinOperator_IF) {
    return std::string("tf.If");
  }
  if (builtin_code == tflite::BuiltinOperator_WHILE) {
    return std::string("tfl.while");
  }

  llvm::StringRef op_name(tflite::EnumNameBuiltinOperator(builtin_code));
  return llvm::Twine("tfl.", op_name.lower()).str();
}

// TODO(jpienaar): This is a placeholder. This should be done in more efficient
// way when part of the translation of module.
static tflite::ActivationFunctionType ConvertTFL_AFAttrForOptionWriter(
    llvm::StringRef str, flatbuffers::FlatBufferBuilder* builder) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePSflatbuffer_operatorDTcc mht_1(mht_1_v, 256, "", "./tensorflow/compiler/mlir/lite/flatbuffer_operator.cc", "ConvertTFL_AFAttrForOptionWriter");

  return llvm::StringSwitch<tflite::ActivationFunctionType>(str)
      .Case("NONE", tflite::ActivationFunctionType_NONE)
      .Case("RELU", tflite::ActivationFunctionType_RELU)
      .Case("RELU_N1_TO_1", tflite::ActivationFunctionType_RELU_N1_TO_1)
      .Case("RELU6", tflite::ActivationFunctionType_RELU6)
      .Case("TANH", tflite::ActivationFunctionType_TANH)
      .Case("SIGN_BIT", tflite::ActivationFunctionType_SIGN_BIT);
}

static tflite::TensorType ConvertDerivedTFLiteTypeAttrForOptionWriter(
    tflite::TensorType type, flatbuffers::FlatBufferBuilder* builder) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePSflatbuffer_operatorDTcc mht_2(mht_2_v, 270, "", "./tensorflow/compiler/mlir/lite/flatbuffer_operator.cc", "ConvertDerivedTFLiteTypeAttrForOptionWriter");

  if (type == tflite::TensorType_INT64) {
    return tflite::TensorType_INT64;
  } else if (type == tflite::TensorType_INT32) {
    return tflite::TensorType_INT32;
  }
  llvm_unreachable("invalid type in conversion.");
}

static tflite::Padding ConvertTFL_PaddingAttrForOptionWriter(
    llvm::StringRef str, flatbuffers::FlatBufferBuilder* builder) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePSflatbuffer_operatorDTcc mht_3(mht_3_v, 283, "", "./tensorflow/compiler/mlir/lite/flatbuffer_operator.cc", "ConvertTFL_PaddingAttrForOptionWriter");

  return llvm::StringSwitch<tflite::Padding>(str)
      .Case("SAME", tflite::Padding_SAME)
      .Case("VALID", tflite::Padding_VALID);
}

static tflite::MirrorPadMode ConvertTFL_MirrorPaddingAttrForOptionWriter(
    llvm::StringRef str, flatbuffers::FlatBufferBuilder* builder) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePSflatbuffer_operatorDTcc mht_4(mht_4_v, 293, "", "./tensorflow/compiler/mlir/lite/flatbuffer_operator.cc", "ConvertTFL_MirrorPaddingAttrForOptionWriter");

  return llvm::StringSwitch<tflite::MirrorPadMode>(str)
      .Case("REFLECT", tflite::MirrorPadMode_REFLECT)
      .Case("SYMMETRIC", tflite::MirrorPadMode_SYMMETRIC);
}

static tflite::TensorType ConvertDerivedTypeAttrForOptionWriter(
    mlir::Type type, flatbuffers::FlatBufferBuilder* builder) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePSflatbuffer_operatorDTcc mht_5(mht_5_v, 303, "", "./tensorflow/compiler/mlir/lite/flatbuffer_operator.cc", "ConvertDerivedTypeAttrForOptionWriter");

  return tflite::ConvertTypeToTensorType(type);
}

// I32Attr already returns an int as required by flatbuffer builders.
static int ConvertI32AttrForOptionWriter(
    int i, flatbuffers::FlatBufferBuilder* builder) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePSflatbuffer_operatorDTcc mht_6(mht_6_v, 312, "", "./tensorflow/compiler/mlir/lite/flatbuffer_operator.cc", "ConvertI32AttrForOptionWriter");

  return i;
}

// I64Attr already returns a int64_t as required by flatbuffer builders.
static int64_t ConvertI64AttrForOptionWriter(
    int64_t i, flatbuffers::FlatBufferBuilder* builder) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePSflatbuffer_operatorDTcc mht_7(mht_7_v, 321, "", "./tensorflow/compiler/mlir/lite/flatbuffer_operator.cc", "ConvertI64AttrForOptionWriter");

  return i;
}

static int ConvertPositiveI32AttrForOptionWriter(
    int i, flatbuffers::FlatBufferBuilder* builder) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePSflatbuffer_operatorDTcc mht_8(mht_8_v, 329, "", "./tensorflow/compiler/mlir/lite/flatbuffer_operator.cc", "ConvertPositiveI32AttrForOptionWriter");

  return ConvertI32AttrForOptionWriter(i, builder);
}

static flatbuffers::Offset<flatbuffers::Vector<int32_t>>
ConvertI64ArrayAttrForOptionWriter(mlir::ArrayAttr attrArray,
                                   flatbuffers::FlatBufferBuilder* builder) {
  std::vector<int32_t> intVec;
  intVec.reserve(attrArray.getValue().size());
  for (auto attr : attrArray.getValue()) {
    intVec.push_back(attr.cast<mlir::IntegerAttr>().getInt());
  }
  return builder->CreateVector(intVec);
}

static flatbuffers::Offset<flatbuffers::Vector<float>>
ConvertF32ArrayAttrForOptionWriter(mlir::ArrayAttr attrArray,
                                   flatbuffers::FlatBufferBuilder* builder) {
  std::vector<float> floatVec;
  floatVec.reserve(attrArray.getValue().size());
  for (auto attr : attrArray.getValue()) {
    floatVec.push_back(
        attr.cast<mlir::FloatAttr>().getValue().convertToFloat());
  }
  return builder->CreateVector(floatVec);
}

// F32Attr already returns a float as required by flatbuffer builders.
static float ConvertF32AttrForOptionWriter(
    llvm::APFloat f, flatbuffers::FlatBufferBuilder* builder) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePSflatbuffer_operatorDTcc mht_9(mht_9_v, 361, "", "./tensorflow/compiler/mlir/lite/flatbuffer_operator.cc", "ConvertF32AttrForOptionWriter");

  return f.convertToFloat();
}

// BoolAttr already returns a bool as required by flatbuffer builders.
static bool ConvertBoolAttrForOptionWriter(
    bool b, flatbuffers::FlatBufferBuilder* builder) {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePSflatbuffer_operatorDTcc mht_10(mht_10_v, 370, "", "./tensorflow/compiler/mlir/lite/flatbuffer_operator.cc", "ConvertBoolAttrForOptionWriter");

  return b;
}

// Overloading of ConvertBoolAttrForOptionWriter which takes Optional<bool> as
// an input. If value is not specified, false is set for the attribute.
static bool ConvertBoolAttrForOptionWriter(
    mlir::Optional<bool> b, flatbuffers::FlatBufferBuilder* builder) {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePSflatbuffer_operatorDTcc mht_11(mht_11_v, 380, "", "./tensorflow/compiler/mlir/lite/flatbuffer_operator.cc", "ConvertBoolAttrForOptionWriter");

  return b.hasValue() ? b.getValue() : false;
}

static flatbuffers::Offset<flatbuffers::String> ConvertStrAttrForOptionWriter(
    llvm::StringRef str, flatbuffers::FlatBufferBuilder* builder) {
  return builder->CreateString(str.str());
}

static tflite::TensorType ConvertTypeAttrForOptionWriter(
    mlir::Type type, flatbuffers::FlatBufferBuilder* builder) {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePSflatbuffer_operatorDTcc mht_12(mht_12_v, 393, "", "./tensorflow/compiler/mlir/lite/flatbuffer_operator.cc", "ConvertTypeAttrForOptionWriter");

  return tflite::ConvertTypeToTensorType(type);
}

static flatbuffers::Offset<flatbuffers::Vector<int32_t>>
ConvertDerivedShapeAttrForOptionWriter(
    llvm::ArrayRef<int64_t> r, flatbuffers::FlatBufferBuilder* builder) {
  std::vector<int> intVec(r.begin(), r.end());
  return builder->CreateVector(intVec);
}

static tflite::FullyConnectedOptionsWeightsFormat
ConvertTFL_FullyConnectedOptionsWeightFormatAttrForOptionWriter(
    llvm::StringRef str, flatbuffers::FlatBufferBuilder* builder) {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePSflatbuffer_operatorDTcc mht_13(mht_13_v, 409, "", "./tensorflow/compiler/mlir/lite/flatbuffer_operator.cc", "ConvertTFL_FullyConnectedOptionsWeightFormatAttrForOptionWriter");

  return llvm::StringSwitch<tflite::FullyConnectedOptionsWeightsFormat>(str)
      .Case("DEFAULT", tflite::FullyConnectedOptionsWeightsFormat_DEFAULT)
      .Case("SHUFFLED4x16INT8",
            tflite::FullyConnectedOptionsWeightsFormat_SHUFFLED4x16INT8);
}

static tflite::LSTMKernelType ConvertTFL_LSTMKernelTypeAttrForOptionWriter(
    llvm::StringRef str, flatbuffers::FlatBufferBuilder* builder) {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePSflatbuffer_operatorDTcc mht_14(mht_14_v, 420, "", "./tensorflow/compiler/mlir/lite/flatbuffer_operator.cc", "ConvertTFL_LSTMKernelTypeAttrForOptionWriter");

  return llvm::StringSwitch<tflite::LSTMKernelType>(str)
      .Case("FULL", tflite::LSTMKernelType_FULL)
      .Case("BASIC", tflite::LSTMKernelType_BASIC);
}

static mlir::Attribute BuildBoolAttr(bool value, mlir::Builder builder) {
   std::vector<std::string> mht_15_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePSflatbuffer_operatorDTcc mht_15(mht_15_v, 429, "", "./tensorflow/compiler/mlir/lite/flatbuffer_operator.cc", "BuildBoolAttr");

  return builder.getBoolAttr(value);
}

static mlir::Attribute BuildStrAttr(llvm::StringRef str,
                                    mlir::Builder builder) {
   std::vector<std::string> mht_16_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePSflatbuffer_operatorDTcc mht_16(mht_16_v, 437, "", "./tensorflow/compiler/mlir/lite/flatbuffer_operator.cc", "BuildStrAttr");

  return builder.getStringAttr(str);
}

static mlir::Attribute BuildF32Attr(float value, mlir::Builder builder) {
   std::vector<std::string> mht_17_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePSflatbuffer_operatorDTcc mht_17(mht_17_v, 444, "", "./tensorflow/compiler/mlir/lite/flatbuffer_operator.cc", "BuildF32Attr");

  return builder.getF32FloatAttr(value);
}

static mlir::Attribute BuildI32Attr(int32_t value, mlir::Builder builder) {
   std::vector<std::string> mht_18_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePSflatbuffer_operatorDTcc mht_18(mht_18_v, 451, "", "./tensorflow/compiler/mlir/lite/flatbuffer_operator.cc", "BuildI32Attr");

  return builder.getI32IntegerAttr(value);
}

static mlir::Attribute BuildI64Attr(int64_t value, mlir::Builder builder) {
   std::vector<std::string> mht_19_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePSflatbuffer_operatorDTcc mht_19(mht_19_v, 458, "", "./tensorflow/compiler/mlir/lite/flatbuffer_operator.cc", "BuildI64Attr");

  return builder.getI64IntegerAttr(value);
}

static mlir::Attribute BuildI64ArrayAttr(std::vector<int32_t> value,
                                         mlir::Builder builder) {
   std::vector<std::string> mht_20_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePSflatbuffer_operatorDTcc mht_20(mht_20_v, 466, "", "./tensorflow/compiler/mlir/lite/flatbuffer_operator.cc", "BuildI64ArrayAttr");

  std::vector<int64_t> typecast(value.begin(), value.end());
  return builder.getI64ArrayAttr(typecast);
}

static mlir::Attribute BuildF32ArrayAttr(std::vector<float> value,
                                         mlir::Builder builder) {
   std::vector<std::string> mht_21_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePSflatbuffer_operatorDTcc mht_21(mht_21_v, 475, "", "./tensorflow/compiler/mlir/lite/flatbuffer_operator.cc", "BuildF32ArrayAttr");

  std::vector<float> typecast(value.begin(), value.end());
  return builder.getF32ArrayAttr(typecast);
}

static mlir::Attribute BuildPositiveI32Attr(int32_t value,
                                            mlir::Builder builder) {
   std::vector<std::string> mht_22_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePSflatbuffer_operatorDTcc mht_22(mht_22_v, 484, "", "./tensorflow/compiler/mlir/lite/flatbuffer_operator.cc", "BuildPositiveI32Attr");

  return builder.getI32IntegerAttr(value);
}

static mlir::Attribute BuildTypeAttr(tflite::TensorType value,
                                     mlir::Builder builder) {
   std::vector<std::string> mht_23_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePSflatbuffer_operatorDTcc mht_23(mht_23_v, 492, "", "./tensorflow/compiler/mlir/lite/flatbuffer_operator.cc", "BuildTypeAttr");

  return mlir::TypeAttr::get(ConvertElementType(value, builder));
}

static mlir::Attribute BuildTFL_AFAttr(tflite::ActivationFunctionType value,
                                       mlir::Builder builder) {
   std::vector<std::string> mht_24_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePSflatbuffer_operatorDTcc mht_24(mht_24_v, 500, "", "./tensorflow/compiler/mlir/lite/flatbuffer_operator.cc", "BuildTFL_AFAttr");

  const char* option_name = tflite::EnumNameActivationFunctionType(value);
  return builder.getStringAttr(option_name);
}

static mlir::Attribute BuildTFL_FullyConnectedOptionsWeightFormatAttr(
    tflite::FullyConnectedOptionsWeightsFormat value, mlir::Builder builder) {
   std::vector<std::string> mht_25_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePSflatbuffer_operatorDTcc mht_25(mht_25_v, 509, "", "./tensorflow/compiler/mlir/lite/flatbuffer_operator.cc", "BuildTFL_FullyConnectedOptionsWeightFormatAttr");

  const char* option_name =
      tflite::EnumNameFullyConnectedOptionsWeightsFormat(value);
  return builder.getStringAttr(option_name);
}

static mlir::Attribute BuildTFL_LSTMKernelTypeAttr(tflite::LSTMKernelType value,
                                                   mlir::Builder builder) {
   std::vector<std::string> mht_26_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePSflatbuffer_operatorDTcc mht_26(mht_26_v, 519, "", "./tensorflow/compiler/mlir/lite/flatbuffer_operator.cc", "BuildTFL_LSTMKernelTypeAttr");

  const char* option_name = tflite::EnumNameLSTMKernelType(value);
  return builder.getStringAttr(option_name);
}

static mlir::Attribute BuildTFL_MirrorPaddingAttr(tflite::MirrorPadMode value,
                                                  mlir::Builder builder) {
   std::vector<std::string> mht_27_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePSflatbuffer_operatorDTcc mht_27(mht_27_v, 528, "", "./tensorflow/compiler/mlir/lite/flatbuffer_operator.cc", "BuildTFL_MirrorPaddingAttr");

  const char* option_name = tflite::EnumNameMirrorPadMode(value);
  return builder.getStringAttr(option_name);
}

static mlir::Attribute BuildTFL_PaddingAttr(tflite::Padding value,
                                            mlir::Builder builder) {
   std::vector<std::string> mht_28_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePSflatbuffer_operatorDTcc mht_28(mht_28_v, 537, "", "./tensorflow/compiler/mlir/lite/flatbuffer_operator.cc", "BuildTFL_PaddingAttr");

  const char* option_name = tflite::EnumNamePadding(value);
  return builder.getStringAttr(option_name);
}

Status mlir::CustomOptionsToAttributes(
    const std::string& custom_code, const std::vector<uint8_t>& custom_options,
    mlir::Builder builder, mlir::Location loc,
    llvm::SmallVectorImpl<mlir::NamedAttribute>* attributes) {
   std::vector<std::string> mht_29_v;
   mht_29_v.push_back("custom_code: \"" + custom_code + "\"");
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePSflatbuffer_operatorDTcc mht_29(mht_29_v, 549, "", "./tensorflow/compiler/mlir/lite/flatbuffer_operator.cc", "mlir::CustomOptionsToAttributes");

  attributes->emplace_back(
      builder.getNamedAttr("custom_code", builder.getStringAttr(custom_code)));
  std::string content;
  content.assign(reinterpret_cast<const char*>(custom_options.data()),
                 custom_options.size());
  ShapedType type = RankedTensorType::get(
      {static_cast<int64_t>(custom_options.size())}, builder.getIntegerType(8));
  attributes->emplace_back(builder.getNamedAttr(
      "custom_option",
      OpaqueElementsAttr::get(builder.getContext()->getLoadedDialect("tfl"),
                              type, content)));

  return Status::OK();
}

// Pull in FlatBuffer writers for TFLite generated using TableGen
#include "tensorflow/compiler/mlir/lite/operator_converters.inc"
