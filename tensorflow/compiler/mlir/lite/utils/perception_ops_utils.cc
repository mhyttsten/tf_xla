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
class MHTracer_DTPStensorflowPScompilerPSmlirPSlitePSutilsPSperception_ops_utilsDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePSutilsPSperception_ops_utilsDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSmlirPSlitePSutilsPSperception_ops_utilsDTcc() {
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
#include "tensorflow/compiler/mlir/lite/utils/perception_ops_utils.h"

#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/OpDefinition.h"  // from @llvm-project
#include "mlir/IR/Types.h"  // from @llvm-project
#include "mlir/IR/Value.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/lite/ir/tfl_ops.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "tensorflow/lite/c/builtin_op_data.h"

namespace mlir {
namespace TFL {

namespace {

constexpr char kTFImplements[] = "tf._implements";
constexpr char kMaxUnpooling[] = "MaxUnpooling2D";
constexpr char kImageWarping[] = "DenseImageWarp";

inline OpaqueElementsAttr CustomOption(OpBuilder* builder,
                                       const std::string& content) {
   std::vector<std::string> mht_0_v;
   mht_0_v.push_back("content: \"" + content + "\"");
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePSutilsPSperception_ops_utilsDTcc mht_0(mht_0_v, 207, "", "./tensorflow/compiler/mlir/lite/utils/perception_ops_utils.cc", "CustomOption");

  ShapedType type = RankedTensorType::get(
      {static_cast<int64_t>(content.size())}, builder->getIntegerType(8));
  return OpaqueElementsAttr::get(builder->getContext()->getLoadedDialect("tfl"),
                                 type,
                                 StringRef(content.data(), content.size()));
}

inline LogicalResult HasIntegerArrayWithSize(FuncOp* func,
                                             const DictionaryAttr& attrs,
                                             const std::string& attr_name,
                                             int N) {
   std::vector<std::string> mht_1_v;
   mht_1_v.push_back("attr_name: \"" + attr_name + "\"");
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePSutilsPSperception_ops_utilsDTcc mht_1(mht_1_v, 222, "", "./tensorflow/compiler/mlir/lite/utils/perception_ops_utils.cc", "HasIntegerArrayWithSize");

  ArrayAttr array_attr = attrs.get(attr_name).dyn_cast_or_null<ArrayAttr>();
  if (array_attr == nullptr || array_attr.size() != N) {
    return func->emitWarning()
           << "'" << attr_name << "' attribute for " << kMaxUnpooling
           << " must be set and has size of " << N;
  }
  for (Attribute integer_attr : array_attr.getValue()) {
    IntegerAttr value = integer_attr.dyn_cast<IntegerAttr>();
    if (!value) {
      return func->emitWarning()
             << "'" << attr_name << "' attribute for " << kMaxUnpooling
             << " does not contain integer values";
    }
  }
  return success();
}

inline LogicalResult GetIntegerArraySafe(
    FuncOp* func, const DictionaryAttr& attrs, const std::string& attr_name,
    llvm::SmallVectorImpl<int32_t>* results, int N) {
   std::vector<std::string> mht_2_v;
   mht_2_v.push_back("attr_name: \"" + attr_name + "\"");
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePSutilsPSperception_ops_utilsDTcc mht_2(mht_2_v, 246, "", "./tensorflow/compiler/mlir/lite/utils/perception_ops_utils.cc", "GetIntegerArraySafe");

  ArrayAttr array_attr = attrs.get(attr_name).dyn_cast_or_null<ArrayAttr>();
  if (array_attr == nullptr || array_attr.size() != N) {
    return func->emitError()
           << "'" << attr_name << "' attribute for " << kMaxUnpooling
           << " must be set and has size of " << N;
  }
  results->reserve(N);

  for (Attribute integer_attr : array_attr.getValue()) {
    IntegerAttr value = integer_attr.dyn_cast<IntegerAttr>();
    if (!value) {
      return func->emitError()
             << "'" << attr_name << "' attribute for " << kMaxUnpooling
             << " does not contain integer values";
    }
    results->push_back(value.getInt());
  }
  return success();
}

}  // namespace

LogicalResult ConvertMaxUnpoolingFunc::RewriteFunc() {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePSutilsPSperception_ops_utilsDTcc mht_3(mht_3_v, 272, "", "./tensorflow/compiler/mlir/lite/utils/perception_ops_utils.cc", "ConvertMaxUnpoolingFunc::RewriteFunc");

  func_.eraseBody();
  func_.addEntryBlock();
  func_->setAttr(kTFImplements,
                 StringAttr::get(func_.getContext(), kMaxUnpooling));

  OpBuilder builder(func_.getBody());
  std::string custom_option_buffer;
  if (failed(CreateCustomOptions(custom_option_buffer))) {
    return failure();
  }
  auto op = builder.create<CustomOp>(
      func_.getLoc(), func_.getFunctionType().getResults(),
      func_.getArguments(), kMaxUnpooling,
      CustomOption(&builder, custom_option_buffer));
  builder.create<func::ReturnOp>(func_.getLoc(), op.getResults());

  return success();
}

LogicalResult ConvertMaxUnpoolingFunc::VerifySignature() {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePSutilsPSperception_ops_utilsDTcc mht_4(mht_4_v, 295, "", "./tensorflow/compiler/mlir/lite/utils/perception_ops_utils.cc", "ConvertMaxUnpoolingFunc::VerifySignature");

  // Verify high-level function signature.
  if (func_.getNumArguments() != 2) {
    return func_.emitWarning()
           << "Invalid number of arguments to " << kMaxUnpooling << ": "
           << func_.getNumArguments();
  }
  if (func_.getFunctionType().getNumResults() != 1) {
    return func_.emitWarning()
           << "Invalid number of results from " << kMaxUnpooling << ": "
           << func_.getFunctionType().getNumResults();
  }

  auto attrs = attr_.getAttrs();

  if (failed(HasIntegerArrayWithSize(&func_, attrs, "pool_size", 2))) {
    return failure();
  }

  if (failed(HasIntegerArrayWithSize(&func_, attrs, "strides", 2))) {
    return failure();
  }

  // Retrieves padding.
  auto padding = attrs.get("padding").dyn_cast_or_null<StringAttr>();
  if (!padding) {
    return func_.emitWarning() << "'padding' attribute for " << kMaxUnpooling
                               << " is not set or not a string";
  }
  if (!padding.getValue().equals("VALID") &&
      !padding.getValue().equals("SAME")) {
    return func_.emitWarning()
           << "Padding for " << kMaxUnpooling << " must be 'SAME' or 'VALID'";
  }
  return success();
}

LogicalResult ConvertMaxUnpoolingFunc::CreateCustomOptions(
    std::string& custom_option_buffer) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePSutilsPSperception_ops_utilsDTcc mht_5(mht_5_v, 336, "", "./tensorflow/compiler/mlir/lite/utils/perception_ops_utils.cc", "ConvertMaxUnpoolingFunc::CreateCustomOptions");

  auto attrs = attr_.getAttrs();
  TfLitePoolParams pool_params;

  llvm::SmallVector<int32_t, 2> pool_size;
  if (failed(GetIntegerArraySafe(&func_, attrs, "pool_size", &pool_size, 2))) {
    return failure();
  }
  pool_params.filter_height = pool_size[0];
  pool_params.filter_width = pool_size[1];

  // Retrieve strides.
  llvm::SmallVector<int32_t, 2> strides;
  if (failed(GetIntegerArraySafe(&func_, attrs, "strides", &strides, 2))) {
    return failure();
  }
  pool_params.stride_height = strides[0];
  pool_params.stride_width = strides[1];

  // Retrieves padding.
  auto padding = attrs.get("padding").dyn_cast_or_null<StringAttr>();
  if (!padding) {
    return func_.emitError() << "'padding' attribute for " << kMaxUnpooling
                             << " is not set or not a string";
  }
  if (padding.getValue().equals("VALID")) {
    pool_params.padding = kTfLitePaddingValid;
  } else if (padding.getValue().equals("SAME")) {
    pool_params.padding = kTfLitePaddingSame;
  } else {
    return func_.emitError()
           << "Padding for " << kMaxUnpooling << " must be 'SAME' or 'VALID'";
  }

  pool_params.activation = kTfLiteActNone;
  pool_params.computed.padding = TfLitePaddingValues{0, 0, 0, 0};

  custom_option_buffer.assign(reinterpret_cast<char*>(&pool_params),
                              sizeof(TfLitePoolParams));
  return success();
}

LogicalResult ConvertDenseImageWarpFunc::RewriteFunc() {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePSutilsPSperception_ops_utilsDTcc mht_6(mht_6_v, 381, "", "./tensorflow/compiler/mlir/lite/utils/perception_ops_utils.cc", "ConvertDenseImageWarpFunc::RewriteFunc");

  func_.eraseBody();
  func_.addEntryBlock();
  func_->setAttr(kTFImplements,
                 StringAttr::get(func_.getContext(), kImageWarping));

  OpBuilder builder(func_.getBody());
  auto op = builder.create<CustomOp>(func_.getLoc(),
                                     func_.getFunctionType().getResults(),
                                     func_.getArguments(), kImageWarping,
                                     CustomOption(&builder, /*content=*/""));
  builder.create<func::ReturnOp>(func_.getLoc(), op.getResults());

  return success();
}

LogicalResult ConvertDenseImageWarpFunc::VerifySignature() {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePSutilsPSperception_ops_utilsDTcc mht_7(mht_7_v, 400, "", "./tensorflow/compiler/mlir/lite/utils/perception_ops_utils.cc", "ConvertDenseImageWarpFunc::VerifySignature");

  // Verify high-level function signature.
  if (func_.getNumArguments() != 2) {
    return func_.emitWarning()
           << "Invalid number of arguments to " << kImageWarping << ": "
           << func_.getNumArguments();
  }
  if (func_.getFunctionType().getNumResults() != 1) {
    return func_.emitWarning()
           << "Invalid number of results from " << kImageWarping << ": "
           << func_.getFunctionType().getNumResults();
  }

  // Check types and shapes.
  auto image_type =
      func_.getFunctionType().getInput(0).dyn_cast_or_null<RankedTensorType>();
  if (!image_type || !image_type.getElementType().isF32() ||
      image_type.getRank() != 4) {
    return func_.emitWarning() << "Image should be a 4D float tensor";
  }

  auto flow_type =
      func_.getFunctionType().getInput(1).dyn_cast_or_null<RankedTensorType>();
  if (!flow_type || !flow_type.getElementType().isF32() ||
      flow_type.getRank() != 4) {
    return func_.emitWarning() << "Flow should be a 4D float tensor";
  }

  auto output_type =
      func_.getFunctionType().getResult(0).dyn_cast_or_null<RankedTensorType>();
  if (!output_type || !output_type.getElementType().isF32() ||
      output_type.getRank() != 4) {
    return func_.emitWarning() << "Output should be a 4D float tensor";
  }

  return success();
}

}  // namespace TFL
}  // namespace mlir
