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
class MHTracer_DTPStensorflowPScompilerPSmlirPSlitePSutilsPSnms_utilsDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePSutilsPSnms_utilsDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSmlirPSlitePSutilsPSnms_utilsDTcc() {
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

#include "tensorflow/compiler/mlir/lite/utils/nms_utils.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/lite/ir/tfl_ops.h"

namespace mlir {
namespace TFL {

namespace {

// TODO(b/162842801): Consolidate all util definitions of kTFImplements.
constexpr char kTFImplements[] = "tf._implements";
constexpr char kCustomSSDPostprocessing[] = "TFLite_Detection_PostProcess";
constexpr char kTfNMSPadded[] = "non_max_suppression_padded_v2";

inline OpaqueElementsAttr CustomOption(OpBuilder* builder,
                                       const std::string& content) {
   std::vector<std::string> mht_0_v;
   mht_0_v.push_back("content: \"" + content + "\"");
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePSutilsPSnms_utilsDTcc mht_0(mht_0_v, 202, "", "./tensorflow/compiler/mlir/lite/utils/nms_utils.cc", "CustomOption");

  ShapedType type = RankedTensorType::get(
      {static_cast<int64_t>(content.size())}, builder->getIntegerType(8));
  return OpaqueElementsAttr::get(builder->getContext()->getLoadedDialect("tfl"),
                                 type,
                                 StringRef(content.data(), content.size()));
}

}  // namespace

void ConvertNMSPaddedFunc::RewriteFunc() {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePSutilsPSnms_utilsDTcc mht_1(mht_1_v, 215, "", "./tensorflow/compiler/mlir/lite/utils/nms_utils.cc", "ConvertNMSPaddedFunc::RewriteFunc");

  func_->setAttr(kTFImplements,
                 StringAttr::get(func_.getContext(), kTfNMSPadded));
  Value boxes = func_.getArgument(0);
  Value scores = func_.getArgument(1);
  Value max_output_size = func_.getArgument(2);
  Value iou_threshold = func_.getArgument(3);
  Value score_threshold = func_.getArgument(4);
  auto output_type0 = func_.getFunctionType().getResult(0);
  auto output_type1 = func_.getFunctionType().getResult(1);

  OpBuilder builder(func_.getBody());
  auto op = builder.create<mlir::TFL::NonMaxSuppressionV4Op>(
      func_.getLoc(), output_type0, output_type1, boxes, scores,
      max_output_size, iou_threshold, score_threshold);

  builder.create<mlir::func::ReturnOp>(func_.getLoc(), op.getResults());
}

LogicalResult ConvertNMSPaddedFunc::VerifySignature() {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePSutilsPSnms_utilsDTcc mht_2(mht_2_v, 237, "", "./tensorflow/compiler/mlir/lite/utils/nms_utils.cc", "ConvertNMSPaddedFunc::VerifySignature");

  // Verify high-level function signature.
  // Relevant argument characteristics are checked by the TFL op definition.
  if (func_.getNumArguments() < 5) {
    return func_.emitWarning()
           << "Invalid number of arguments to "
              "non_max_suppression_padded_v2 (need at least 5): "
           << func_.getNumArguments();
  }
  if (func_.getFunctionType().getNumResults() != 2) {
    return func_.emitWarning() << "Invalid number of results from "
                                  "non_max_suppression_padded_v2 (need 2): "
                               << func_.getFunctionType().getNumResults();
  }
  // The TFLite fused op does not support batching yet.
  // TODO(b/158709815): Add support for batches with padded NMS.
  auto boxes_type =
      func_.getFunctionType().getInput(0).dyn_cast<RankedTensorType>();
  if (boxes_type == nullptr || !boxes_type.hasRank() ||
      boxes_type.getRank() != 2) {
    return func_.emitWarning() << "TFLite does not support batched input for "
                                  "non_max_suppression_padded";
  }
  return success();
}

LogicalResult ConvertSSDPostProcessFunc::RewriteFunc() {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePSutilsPSnms_utilsDTcc mht_3(mht_3_v, 266, "", "./tensorflow/compiler/mlir/lite/utils/nms_utils.cc", "ConvertSSDPostProcessFunc::RewriteFunc");

  func_.eraseBody();
  func_.addEntryBlock();
  func_->setAttr(kTFImplements,
                 StringAttr::get(func_.getContext(), kCustomSSDPostprocessing));

  OpBuilder builder(func_.getBody());
  std::string custom_option_buffer;
  if (failed(CreateNMSCustomOptions(func_, attr_.getAttrs(),
                                    custom_option_buffer))) {
    return failure();
  }
  auto op = builder.create<CustomOp>(
      func_.getLoc(), func_.getFunctionType().getResults(),
      func_.getArguments(), kCustomSSDPostprocessing,
      CustomOption(&builder, custom_option_buffer));
  builder.create<func::ReturnOp>(func_.getLoc(), op.getResults());

  return success();
}

LogicalResult ConvertSSDPostProcessFunc::CreateNMSCustomOptions(
    FuncOp func, DictionaryAttr attrs, std::string& custom_option_buffer) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePSutilsPSnms_utilsDTcc mht_4(mht_4_v, 291, "", "./tensorflow/compiler/mlir/lite/utils/nms_utils.cc", "ConvertSSDPostProcessFunc::CreateNMSCustomOptions");

  flexbuffers::Builder fbb;
  size_t start_map = fbb.StartMap();

  if (failed(AddIntAttr(func, attrs, "max_detections", &fbb)) ||
      failed(AddIntAttr(func, attrs, "max_classes_per_detection", &fbb)) ||
      failed(AddIntAttr(func, attrs, "num_classes", &fbb)) ||
      failed(AddFloatAttr(func, attrs, "nms_score_threshold", &fbb)) ||
      failed(AddFloatAttr(func, attrs, "nms_iou_threshold", &fbb)) ||
      failed(AddFloatAttr(func, attrs, "y_scale", &fbb)) ||
      failed(AddFloatAttr(func, attrs, "x_scale", &fbb)) ||
      failed(AddFloatAttr(func, attrs, "h_scale", &fbb)) ||
      failed(AddFloatAttr(func, attrs, "w_scale", &fbb)))
    return failure();
  auto use_regular_nms =
      attrs.get("use_regular_nms").dyn_cast_or_null<BoolAttr>();
  if (!use_regular_nms) {
    return func.emitError()
           << "use_regular_nms attribute is not set or not a bool";
  }
  fbb.Int("use_regular_nms", use_regular_nms.getValue());

  fbb.EndMap(start_map);
  fbb.Finish();
  custom_option_buffer.assign(fbb.GetBuffer().begin(), fbb.GetBuffer().end());
  return success();
}

LogicalResult ConvertSSDPostProcessFunc::AddIntAttr(
    FuncOp func, DictionaryAttr attrs, const std::string& attribute,
    flexbuffers::Builder* builder) {
   std::vector<std::string> mht_5_v;
   mht_5_v.push_back("attribute: \"" + attribute + "\"");
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePSutilsPSnms_utilsDTcc mht_5(mht_5_v, 325, "", "./tensorflow/compiler/mlir/lite/utils/nms_utils.cc", "ConvertSSDPostProcessFunc::AddIntAttr");

  auto int_attr = attrs.get(attribute).dyn_cast_or_null<IntegerAttr>();
  if (!int_attr) {
    return func.emitError()
           << attribute.c_str() << " attribute is not set or not an integer";
  }
  builder->Int(attribute.c_str(), int_attr.getInt());
  return success();
}

LogicalResult ConvertSSDPostProcessFunc::AddFloatAttr(
    FuncOp func, DictionaryAttr attrs, const std::string& attribute,
    flexbuffers::Builder* builder) {
   std::vector<std::string> mht_6_v;
   mht_6_v.push_back("attribute: \"" + attribute + "\"");
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePSutilsPSnms_utilsDTcc mht_6(mht_6_v, 341, "", "./tensorflow/compiler/mlir/lite/utils/nms_utils.cc", "ConvertSSDPostProcessFunc::AddFloatAttr");

  auto float_attr = attrs.get(attribute).dyn_cast_or_null<FloatAttr>();
  if (!float_attr) {
    return func.emitError()
           << attribute.c_str() << " attribute is not set or not a float";
  }
  builder->Float(attribute.c_str(), float_attr.getValue().convertToFloat());
  return success();
}

LogicalResult ConvertSSDPostProcessFunc::HasIntAttr(
    FuncOp func, DictionaryAttr attrs, const std::string& attribute) {
   std::vector<std::string> mht_7_v;
   mht_7_v.push_back("attribute: \"" + attribute + "\"");
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePSutilsPSnms_utilsDTcc mht_7(mht_7_v, 356, "", "./tensorflow/compiler/mlir/lite/utils/nms_utils.cc", "ConvertSSDPostProcessFunc::HasIntAttr");

  auto int_attr = attrs.get(attribute).dyn_cast_or_null<IntegerAttr>();
  if (!int_attr) {
    return func.emitWarning()
           << attribute.c_str() << " attribute is not set or not an integer";
  }
  return success();
}

LogicalResult ConvertSSDPostProcessFunc::HasFloatAttr(
    FuncOp func, DictionaryAttr attrs, const std::string& attribute) {
   std::vector<std::string> mht_8_v;
   mht_8_v.push_back("attribute: \"" + attribute + "\"");
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePSutilsPSnms_utilsDTcc mht_8(mht_8_v, 370, "", "./tensorflow/compiler/mlir/lite/utils/nms_utils.cc", "ConvertSSDPostProcessFunc::HasFloatAttr");

  auto float_attr = attrs.get(attribute).dyn_cast_or_null<FloatAttr>();
  if (!float_attr) {
    return func.emitWarning()
           << attribute.c_str() << " attribute is not set or not a float";
  }
  return success();
}

LogicalResult ConvertSSDPostProcessFunc::VerifySignature() {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePSutilsPSnms_utilsDTcc mht_9(mht_9_v, 382, "", "./tensorflow/compiler/mlir/lite/utils/nms_utils.cc", "ConvertSSDPostProcessFunc::VerifySignature");

  // Verify high-level function signature.
  if (func_.getNumArguments() != 3) {
    return func_.emitWarning()
           << "Invalid number of arguments to " << kCustomSSDPostprocessing
           << ": " << func_.getNumArguments();
  }
  if (func_.getFunctionType().getNumResults() != 4) {
    return func_.emitWarning()
           << "Invalid number of results from " << kCustomSSDPostprocessing
           << ": " << func_.getFunctionType().getNumResults();
  }

  auto attrs = attr_.getAttrs();
  if (failed(HasIntAttr(func_, attrs, "max_detections")) ||
      failed(HasIntAttr(func_, attrs, "max_classes_per_detection")) ||
      failed(HasIntAttr(func_, attrs, "num_classes")) ||
      failed(HasFloatAttr(func_, attrs, "nms_score_threshold")) ||
      failed(HasFloatAttr(func_, attrs, "nms_iou_threshold")) ||
      failed(HasFloatAttr(func_, attrs, "y_scale")) ||
      failed(HasFloatAttr(func_, attrs, "x_scale")) ||
      failed(HasFloatAttr(func_, attrs, "h_scale")) ||
      failed(HasFloatAttr(func_, attrs, "w_scale"))) {
    return failure();
  }
  return success();
}

}  // namespace TFL
}  // namespace mlir
