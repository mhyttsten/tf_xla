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
class MHTracer_DTPStensorflowPScompilerPSmlirPSlitePSpythonPSjax_to_tfl_flatbufferDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePSpythonPSjax_to_tfl_flatbufferDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSmlirPSlitePSpythonPSjax_to_tfl_flatbufferDTcc() {
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
#include "tensorflow/compiler/mlir/lite/python/jax_to_tfl_flatbuffer.h"

#include <memory>
#include <utility>

#include "absl/strings/str_join.h"
#include "absl/types/span.h"
#include "llvm/ADT/None.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Support/ToolOutputFile.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/IR/TypeUtilities.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Support/FileUtilities.h"  // from @llvm-project
#include "mlir/Transforms/ViewOpGraph.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/lite/common/tfl_pass_config.h"
#include "tensorflow/compiler/mlir/lite/python/tf_tfl_flatbuffer_helpers.h"
#include "tensorflow/compiler/mlir/lite/tf_tfl_passes.h"
#include "tensorflow/compiler/mlir/lite/tf_to_tfl_flatbuffer.h"
#include "tensorflow/compiler/mlir/lite/transforms/passes.h"
#include "tensorflow/compiler/mlir/tensorflow/translate/mlir_roundtrip_flags.h"
#include "tensorflow/compiler/mlir/xla/hlo_to_mlir_hlo.h"
#include "tensorflow/compiler/xla/service/hlo.pb.h"
#include "tensorflow/compiler/xla/service/hlo_parser.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/protobuf/graph_debug_info.pb.h"
#include "tensorflow/lite/toco/model_flags.pb.h"
#include "tensorflow/lite/toco/toco_flags.pb.h"
#include "tensorflow/lite/toco/types.pb.h"
#include "tensorflow/stream_executor/lib/statusor.h"

namespace tensorflow {
namespace {

// Error collector that simply ignores errors reported.
class NoOpErrorCollector : public tensorflow::protobuf::io::ErrorCollector {
 public:
  void AddError(int line, int column, const string& message) override {
   std::vector<std::string> mht_0_v;
   mht_0_v.push_back("message: \"" + message + "\"");
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePSpythonPSjax_to_tfl_flatbufferDTcc mht_0(mht_0_v, 232, "", "./tensorflow/compiler/mlir/lite/python/jax_to_tfl_flatbuffer.cc", "AddError");
}
};

bool LoadHloProto(const std::string& contents, xla::HloProto* hlo_proto) {
   std::vector<std::string> mht_1_v;
   mht_1_v.push_back("contents: \"" + contents + "\"");
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePSpythonPSjax_to_tfl_flatbufferDTcc mht_1(mht_1_v, 239, "", "./tensorflow/compiler/mlir/lite/python/jax_to_tfl_flatbuffer.cc", "LoadHloProto");

  tensorflow::protobuf::TextFormat::Parser parser;
  NoOpErrorCollector collector;
  parser.RecordErrorsTo(&collector);
  return hlo_proto->ParseFromString(contents) ||
         parser.ParseFromString(contents, hlo_proto) ||
         hlo_proto->mutable_hlo_module()->ParseFromString(contents) ||
         parser.ParseFromString(contents, hlo_proto->mutable_hlo_module());
}

mlir::OwningOpRef<mlir::ModuleOp> HloToMlirHloTranslateFunction(
    llvm::StringRef input, mlir::MLIRContext* context,
    bool import_all_computations) {
  xla::HloProto hlo_proto;
  string content(input.data(), input.size());
  if (!LoadHloProto(content, &hlo_proto)) {
    LOG(ERROR) << "Failed to load proto";
    return nullptr;
  }

  mlir::OwningOpRef<mlir::ModuleOp> module =
      mlir::ModuleOp::create(mlir::UnknownLoc::get(context));
  auto status = ConvertHloToMlirHlo(
      module.get(), hlo_proto.mutable_hlo_module(), import_all_computations);
  if (!status.ok()) {
    LOG(ERROR) << "Hlo module import failed: " << status;
    return nullptr;
  }

  return module;
}

mlir::OwningOpRef<mlir::ModuleOp> HloTextToMlirHloTranslateFunction(
    llvm::StringRef input, mlir::MLIRContext* context,
    bool import_all_computations) {
  xla::HloProto hlo_proto;
  string content(input.data(), input.size());

  auto hlo_module_error = xla::ParseAndReturnUnverifiedModule(content);
  if (!hlo_module_error.ok()) {
    LOG(ERROR) << "HLO Module loading failed: " << hlo_module_error.status();
    return nullptr;
  }

  auto hlo_module = std::move(hlo_module_error.ValueOrDie());
  mlir::OwningOpRef<mlir::ModuleOp> module =
      mlir::ModuleOp::create(mlir::UnknownLoc::get(context));
  auto status =
      ConvertHloToMlirHlo(*module, hlo_module.get(), import_all_computations);
  if (!status.ok()) {
    LOG(ERROR) << "HLO Module import failed: " << status;
    return nullptr;
  }

  return module;
}

}  // namespace
Status ConvertJaxToTFLiteFlatBuffer(const std::string& input,
                                    const toco::ModelFlags& model_flags,
                                    const toco::TocoFlags& toco_flags,
                                    string* result) {
   std::vector<std::string> mht_2_v;
   mht_2_v.push_back("input: \"" + input + "\"");
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePSpythonPSjax_to_tfl_flatbufferDTcc mht_2(mht_2_v, 304, "", "./tensorflow/compiler/mlir/lite/python/jax_to_tfl_flatbuffer.cc", "ConvertJaxToTFLiteFlatBuffer");

  mlir::MLIRContext context;
  mlir::quant::QuantizationSpecs quant_specs;

  // Parse input arrays.
  std::vector<string> node_names;
  std::vector<string> node_dtypes;
  std::vector<llvm::Optional<std::vector<int>>> node_shapes;
  std::vector<llvm::Optional<double>> node_mins;
  std::vector<llvm::Optional<double>> node_maxs;

  // Populate quantization specs.
  TF_RETURN_IF_ERROR(internal::PopulateQuantizationSpecs(
      model_flags, toco_flags, &quant_specs, &node_names, &node_dtypes,
      &node_shapes, &node_mins, &node_maxs));

  internal::WarningUnusedFlags(model_flags, toco_flags);

  // Register all custom ops, including user-specified custom ops.
  TF_RETURN_IF_ERROR(internal::RegisterAllCustomOps(toco_flags));

  mlir::TFL::PassConfig pass_config(quant_specs);
  bool emit_builtin_tflite_ops = !toco_flags.force_select_tf_ops();
  pass_config.emit_builtin_tflite_ops = emit_builtin_tflite_ops;
  pass_config.enable_tflite_variables =
      toco_flags.enable_tflite_resource_variables();
  pass_config.unfold_batch_matmul = toco_flags.unfold_batchmatmul();
  pass_config.lower_tensor_list_ops = toco_flags.lower_tensor_list_ops();
  // Disable the unfolding of the 16x16 TF::BatchMatMulOp to avoid the
  // conversion to an unsupported 16x16 TFL::FullyConnectedOp.
  if (toco_flags.inference_type() == toco::IODataType::QUANTIZED_INT16) {
    pass_config.unfold_batch_matmul = false;
  }
  pass_config.unfold_large_splat_constant =
      toco_flags.unfold_large_splat_constant();
  pass_config.enable_hlo_to_tf_conversion = true;

  mlir::OwningOpRef<mlir::ModuleOp> module;
  if (model_flags.hlo_file_type() == toco::ModelFlags::HLO_TEXT) {
    module = HloTextToMlirHloTranslateFunction(input, &context, false);
  } else if (model_flags.hlo_file_type() == toco::ModelFlags::HLO_PROTO) {
    module = HloToMlirHloTranslateFunction(input, &context, false);
  } else {
    return errors::InvalidArgument("unknown hlo format type.");
  }

  // Set the input names.
  auto main_func = module->lookupSymbol<mlir::func::FuncOp>("main");
  if (!main_func) return errors::Internal("Failed to find the main function.");
  // Retrive input names from model flags.
  std::vector<std::string> input_names;
  for (const auto& input : model_flags.input_arrays()) {
    input_names.push_back(input.name());
  }

  const auto& inputs = absl::StrJoin(input_names, ",");
  mlir::OpBuilder builder(*module);
  llvm::SmallVector<mlir::NamedAttribute> attrs;
  attrs.push_back(
      builder.getNamedAttr("inputs", builder.getStringAttr(inputs)));
  // Jax wrapped the output nodes in a tuple, so it's pretty hard to us
  // to tell the output at this point, we will set the output at the export
  // phase.
  main_func->setAttr("tf.entry_function", builder.getDictionaryAttr(attrs));

  auto status = internal::ConvertMLIRToTFLiteFlatBuffer(
      model_flags, toco_flags, std::move(module), pass_config,
      /*saved_model_tags=*/{}, result,
      /*session=*/llvm::None);
  return status;
}

}  // namespace tensorflow
