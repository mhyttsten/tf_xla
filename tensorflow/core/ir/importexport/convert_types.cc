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
class MHTracer_DTPStensorflowPScorePSirPSimportexportPSconvert_typesDTcc {
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
   MHTracer_DTPStensorflowPScorePSirPSimportexportPSconvert_typesDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSirPSimportexportPSconvert_typesDTcc() {
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

#include "tensorflow/core/ir/importexport/convert_types.h"

#include "absl/strings/str_cat.h"
#include "llvm/Support/Casting.h"
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/IR/Types.h"  // from @llvm-project
#include "mlir/Support/DebugStringHelper.h"  // from @llvm-project
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/ir/dialect.h"
#include "tensorflow/core/lib/core/errors.h"

namespace mlir {
namespace tfg {

using tensorflow::DataType;
using tensorflow::Status;
using tensorflow::TensorShape;
using tensorflow::TensorShapeProto;
using tensorflow::errors::InvalidArgument;
using tensorflow::errors::Unimplemented;

Status ConvertDataType(DataType dtype, Builder builder, Type* type) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSirPSimportexportPSconvert_typesDTcc mht_0(mht_0_v, 207, "", "./tensorflow/core/ir/importexport/convert_types.cc", "ConvertDataType");

  switch (dtype) {
    case tensorflow::DT_HALF:
      *type = builder.getF16Type();
      return Status::OK();
    case tensorflow::DT_FLOAT:
      *type = builder.getF32Type();
      return Status::OK();
    case tensorflow::DT_DOUBLE:
      *type = builder.getF64Type();
      return Status::OK();
    case tensorflow::DT_BOOL:
      *type = builder.getIntegerType(1);
      return Status::OK();
    case tensorflow::DT_INT8:
      *type = builder.getIntegerType(8);
      return Status::OK();
    case tensorflow::DT_INT16:
      *type = builder.getIntegerType(16);
      return Status::OK();
    case tensorflow::DT_INT32:
      *type = builder.getIntegerType(32);
      return Status::OK();
    case tensorflow::DT_INT64:
      *type = builder.getIntegerType(64);
      return Status::OK();
    case tensorflow::DT_UINT8:
      *type = builder.getIntegerType(8, /*isSigned=*/false);
      return Status::OK();
    case tensorflow::DT_UINT16:
      *type = builder.getIntegerType(16, /*isSigned=*/false);
      return Status::OK();
    case tensorflow::DT_UINT32:
      *type = builder.getIntegerType(32, /*isSigned=*/false);
      return Status::OK();
    case tensorflow::DT_UINT64:
      *type = builder.getIntegerType(64, /*isSigned=*/false);
      return Status::OK();
    case tensorflow::DT_BFLOAT16:
      *type = builder.getBF16Type();
      return Status::OK();
    case tensorflow::DT_COMPLEX64:
      *type = ComplexType::get(builder.getF32Type());
      return Status::OK();
    case tensorflow::DT_COMPLEX128:
      *type = ComplexType::get(builder.getF64Type());
      return Status::OK();
#define HANDLE_TF_TYPE(tftype, enumerant, name) \
  case tensorflow::DT_##enumerant:              \
    *type = builder.getType<tftype##Type>();    \
    return Status::OK();
#include "tensorflow/core/ir/types/types.def"

    default:
      return Unimplemented(absl::StrCat(
          "Converting DataType '", DataTypeString(dtype), "' to MLIR Type"));
  }
}

Status ConvertScalarTypeToDataType(Type type, DataType* dtype) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSirPSimportexportPSconvert_typesDTcc mht_1(mht_1_v, 269, "", "./tensorflow/core/ir/importexport/convert_types.cc", "ConvertScalarTypeToDataType");

  if (type.isF16()) {
    *dtype = tensorflow::DT_HALF;
    return Status::OK();
  } else if (type.isF32()) {
    *dtype = tensorflow::DT_FLOAT;
    return Status::OK();
  } else if (type.isF64()) {
    *dtype = tensorflow::DT_DOUBLE;
    return Status::OK();
  } else if (type.isBF16()) {
    *dtype = tensorflow::DT_BFLOAT16;
    return Status::OK();
  } else if (auto itype = type.dyn_cast<IntegerType>()) {
    switch (itype.getWidth()) {
      case 1:
        *dtype = tensorflow::DT_BOOL;
        return Status::OK();
      case 8:
        *dtype =
            itype.isUnsigned() ? tensorflow::DT_UINT8 : tensorflow::DT_INT8;
        return Status::OK();
      case 16:
        *dtype =
            itype.isUnsigned() ? tensorflow::DT_UINT16 : tensorflow::DT_INT16;
        return Status::OK();
      case 32:
        *dtype =
            itype.isUnsigned() ? tensorflow::DT_UINT32 : tensorflow::DT_INT32;
        return Status::OK();
      case 64:
        *dtype =
            itype.isUnsigned() ? tensorflow::DT_UINT64 : tensorflow::DT_INT64;
        return Status::OK();
      default:
        return Unimplemented(
            absl::StrCat("Converting ", debugString(type), " to DataType"));
    }
  } else if (auto complex_type = type.dyn_cast<ComplexType>()) {
    auto etype = complex_type.getElementType();
    if (etype.isF32()) {
      *dtype = tensorflow::DT_COMPLEX64;
      return Status::OK();
    } else if (etype.isF64()) {
      *dtype = tensorflow::DT_COMPLEX128;
      return Status::OK();
    }
    return Unimplemented(
        absl::StrCat("Converting ", debugString(type), " to DataType"));
  }

#define HANDLE_TF_TYPE(tftype, enumerant, name) \
  if (type.isa<tftype##Type>()) {               \
    *dtype = tensorflow::DT_##enumerant;        \
    return Status::OK();                        \
  }
// NOLINTNEXTLINE
#include "tensorflow/core/ir/types/types.def"

  return Unimplemented(
      absl::StrCat("Converting ", debugString(type), " to DataType"));
}

Status ConvertToDataType(Type type, DataType* dtype) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSirPSimportexportPSconvert_typesDTcc mht_2(mht_2_v, 335, "", "./tensorflow/core/ir/importexport/convert_types.cc", "ConvertToDataType");

  if (auto stype = type.dyn_cast<ShapedType>()) {
    TF_RETURN_IF_ERROR(
        ConvertScalarTypeToDataType(stype.getElementType(), dtype));
  } else {
    TF_RETURN_IF_ERROR(ConvertScalarTypeToDataType(type, dtype));
  }
  return Status::OK();
}

void ConvertToMlirShape(const TensorShape& input_shape,
                        llvm::SmallVectorImpl<int64_t>* shape) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSirPSimportexportPSconvert_typesDTcc mht_3(mht_3_v, 349, "", "./tensorflow/core/ir/importexport/convert_types.cc", "ConvertToMlirShape");

  shape->reserve(input_shape.dims());
  for (const auto& d : input_shape) {
    shape->push_back(d.size);
  }
}

Status ConvertToMlirShape(const TensorShapeProto& input_shape,
                          llvm::SmallVectorImpl<int64_t>* shape) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSirPSimportexportPSconvert_typesDTcc mht_4(mht_4_v, 360, "", "./tensorflow/core/ir/importexport/convert_types.cc", "ConvertToMlirShape");

  shape->reserve(input_shape.dim_size());
  auto& dims = input_shape.dim();
  for (auto& d : dims) {
    if (d.size() > std::numeric_limits<int64_t>::max()) {
      return InvalidArgument("Shape element overflows");
    }
    // This isn't really expected, but Grappler is using such shapes for its
    // symbolic shape analysis and it may spill into here.
    if (d.size() < ShapedType::kDynamicSize)
      shape->push_back(ShapedType::kDynamicSize);
    else
      shape->push_back(d.size());
  }
  return Status::OK();
}

tensorflow::StatusOr<Type> ConvertToMlirTensorType(
    const TensorShapeProto& shape, DataType dtype, Builder* builder) {
  Type element_type;
  TF_RETURN_IF_ERROR(ConvertDataType(dtype, *builder, &element_type));
  if (shape.unknown_rank()) {
    return UnrankedTensorType::get(element_type);
  }
  llvm::SmallVector<int64_t, 4> shape_dims;
  TF_RETURN_IF_ERROR(ConvertToMlirShape(shape, &shape_dims));
  return RankedTensorType::get(shape_dims, element_type);
}

}  // namespace tfg
}  // namespace mlir
