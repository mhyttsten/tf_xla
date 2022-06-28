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
class MHTracer_DTPStensorflowPScompilerPSmlirPSlitePSutilsPSconvert_typeDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePSutilsPSconvert_typeDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSmlirPSlitePSutilsPSconvert_typeDTcc() {
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

#include "tensorflow/compiler/mlir/lite/utils/convert_type.h"

#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/IR/Types.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/lite/ir/tfl_ops.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_types.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/lite/schema/schema_generated.h"

namespace tflite {

using xla::StatusOr;

namespace errors = tensorflow::errors;

tflite::TensorType ConvertTypeToTensorType(mlir::Type type) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePSutilsPSconvert_typeDTcc mht_0(mht_0_v, 203, "", "./tensorflow/compiler/mlir/lite/utils/convert_type.cc", "ConvertTypeToTensorType");

  if (type.isF16()) {
    return tflite::TensorType_FLOAT16;
  } else if (type.isF32()) {
    return tflite::TensorType_FLOAT32;
  } else if (type.isF64()) {
    return tflite::TensorType_FLOAT64;
  } else if (type.isa<mlir::TF::StringType>()) {
    return tflite::TensorType_STRING;
  } else if (auto complex_type = type.dyn_cast<mlir::ComplexType>()) {
    if (complex_type.getElementType().isF32()) {
      return tflite::TensorType_COMPLEX64;
    } else if (complex_type.getElementType().isF64()) {
      return tflite::TensorType_COMPLEX128;
    }
    llvm_unreachable("invalid complex Type in conversion");
  } else if (auto itype = type.dyn_cast<mlir::IntegerType>()) {
    switch (itype.getWidth()) {
      case 1:
        return tflite::TensorType_BOOL;
      case 8:
        if (itype.isUnsigned())
          return tflite::TensorType_UINT8;
        else
          return tflite::TensorType_INT8;
      case 16:
        return tflite::TensorType_INT16;
      case 32:
        return tflite::TensorType_INT32;
      case 64:
        if (itype.isUnsigned())
          return tflite::TensorType_UINT64;
        else
          return tflite::TensorType_INT64;
      default:
        llvm_unreachable("invalid integer Type in conversion");
    }
  }
  llvm_unreachable("invalid Type in conversion");
}

mlir::Type ConvertElementType(tflite::TensorType type, mlir::Builder builder) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePSutilsPSconvert_typeDTcc mht_1(mht_1_v, 247, "", "./tensorflow/compiler/mlir/lite/utils/convert_type.cc", "ConvertElementType");

  switch (type) {
    case tflite::TensorType_FLOAT16:
      return builder.getF16Type();
    case tflite::TensorType_FLOAT32:
      return builder.getF32Type();
    case tflite::TensorType_FLOAT64:
      return builder.getF64Type();
    case tflite::TensorType_INT32:
      return builder.getIntegerType(32);
    case tflite::TensorType_UINT16:
      return builder.getIntegerType(16, /*isSigned=*/false);
    case tflite::TensorType_UINT32:
      return builder.getIntegerType(32, /*isSigned=*/false);
    case tflite::TensorType_UINT8:
      return builder.getIntegerType(8, /*isSigned=*/false);
    case tflite::TensorType_INT64:
      return builder.getIntegerType(64);
    case tflite::TensorType_STRING:
      return mlir::TF::StringType::get(builder.getContext());
    case tflite::TensorType_BOOL:
      return builder.getI1Type();
    case tflite::TensorType_INT16:
      return builder.getIntegerType(16);
    case tflite::TensorType_COMPLEX64:
      return mlir::ComplexType::get(builder.getF32Type());
    case tflite::TensorType_COMPLEX128:
      return mlir::ComplexType::get(builder.getF64Type());
    case tflite::TensorType_INT8:
      return builder.getIntegerType(8);
    case tflite::TensorType_UINT64:
      return builder.getIntegerType(64, /*isSigned=*/false);
    case tflite::TensorType_RESOURCE:
      return mlir::TF::ResourceType::get(builder.getContext());
    case tflite::TensorType_VARIANT:
      return mlir::TF::VariantType::get(builder.getContext());
  }
}

tensorflow::DataType TflTypeToTfType(tflite::TensorType type) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePSutilsPSconvert_typeDTcc mht_2(mht_2_v, 289, "", "./tensorflow/compiler/mlir/lite/utils/convert_type.cc", "TflTypeToTfType");

  switch (type) {
    case tflite::TensorType_BOOL:
      return tensorflow::DT_BOOL;
    case tflite::TensorType_COMPLEX64:
      return tensorflow::DT_COMPLEX64;
    case tflite::TensorType_COMPLEX128:
      return tensorflow::DT_COMPLEX128;
    case tflite::TensorType_FLOAT16:
      return tensorflow::DT_HALF;
    case tflite::TensorType_FLOAT32:
      return tensorflow::DT_FLOAT;
    case tflite::TensorType_FLOAT64:
      return tensorflow::DT_DOUBLE;
    case tflite::TensorType_INT8:
      return tensorflow::DT_INT8;
    case tflite::TensorType_INT16:
      return tensorflow::DT_INT16;
    case tflite::TensorType_INT32:
      return tensorflow::DT_INT32;
    case tflite::TensorType_UINT32:
      return tensorflow::DT_UINT32;
    case tflite::TensorType_INT64:
      return tensorflow::DT_INT64;
    case tflite::TensorType_STRING:
      return tensorflow::DT_STRING;
    case tflite::TensorType_UINT8:
      return tensorflow::DT_UINT8;
    case tflite::TensorType_UINT16:
      return tensorflow::DT_UINT16;
    case tflite::TensorType_UINT64:
      return tensorflow::DT_UINT64;
    case tflite::TensorType_RESOURCE:
      return tensorflow::DT_RESOURCE;
    case tflite::TensorType_VARIANT:
      return tensorflow::DT_VARIANT;
  }
}

StatusOr<tflite::TensorType> TfTypeToTflType(tensorflow::DataType type) {
  switch (type) {
    case tensorflow::DT_BOOL:
      return tflite::TensorType_BOOL;
    case tensorflow::DT_COMPLEX64:
      return tflite::TensorType_COMPLEX64;
    case tensorflow::DT_COMPLEX128:
      return tflite::TensorType_COMPLEX128;
    case tensorflow::DT_HALF:
      return tflite::TensorType_FLOAT16;
    case tensorflow::DT_FLOAT:
      return tflite::TensorType_FLOAT32;
    case tensorflow::DT_DOUBLE:
      return tflite::TensorType_FLOAT64;
    case tensorflow::DT_INT8:
      return tflite::TensorType_INT8;
    case tensorflow::DT_INT16:
      return tflite::TensorType_INT16;
    case tensorflow::DT_INT32:
      return tflite::TensorType_INT32;
    case tensorflow::DT_UINT32:
      return tflite::TensorType_UINT32;
    case tensorflow::DT_INT64:
      return tflite::TensorType_INT64;
    case tensorflow::DT_UINT64:
      return tflite::TensorType_UINT64;
    case tensorflow::DT_STRING:
      return tflite::TensorType_STRING;
    case tensorflow::DT_UINT8:
      return tflite::TensorType_UINT8;
    case tensorflow::DT_RESOURCE:
      return tflite::TensorType_RESOURCE;
    case tensorflow::DT_VARIANT:
      return tflite::TensorType_VARIANT;
    default:
      return errors::InvalidArgument("unsupported tensor data type", type);
  }
}

mlir::Type GetShapeStrippedType(mlir::TypeAttr type_attr) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePSutilsPSconvert_typeDTcc mht_3(mht_3_v, 370, "", "./tensorflow/compiler/mlir/lite/utils/convert_type.cc", "GetShapeStrippedType");

  auto type = type_attr.getValue();
  auto shaped_type = type.dyn_cast<mlir::ShapedType>();
  if (shaped_type) {
    return shaped_type.getElementType();
  } else {
    return type;
  }
}

bool NotFromQuantOpOrSameQuantType(mlir::Value val, mlir::TypeAttr qtype_attr) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePSutilsPSconvert_typeDTcc mht_4(mht_4_v, 383, "", "./tensorflow/compiler/mlir/lite/utils/convert_type.cc", "NotFromQuantOpOrSameQuantType");

  auto val_defn_op = val.getDefiningOp();
  mlir::TFL::QuantizeOp q_op =
      llvm::dyn_cast_or_null<mlir::TFL::QuantizeOp>(val_defn_op);
  if (!q_op) return true;

  // Ignore shape details - we're really only trying to
  // check if quantization is the same.
  auto stripped_src_qtype = GetShapeStrippedType(q_op.qtypeAttr());
  auto stripped_qtype = GetShapeStrippedType(qtype_attr);
  return stripped_src_qtype == stripped_qtype;
}

}  // namespace tflite
