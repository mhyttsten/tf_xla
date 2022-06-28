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
class MHTracer_DTPStensorflowPScompilerPSmlirPSlitePSquantizationPSlitePSquantize_modelDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePSquantizationPSlitePSquantize_modelDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSmlirPSlitePSquantizationPSlitePSquantize_modelDTcc() {
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

#include "tensorflow/compiler/mlir/lite/quantization/lite/quantize_model.h"

#include <string>

#include "absl/strings/string_view.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/Location.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Pass/PassManager.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/lite/common/tfl_pass_config.h"
#include "tensorflow/compiler/mlir/lite/flatbuffer_export.h"
#include "tensorflow/compiler/mlir/lite/flatbuffer_import.h"
#include "tensorflow/compiler/mlir/lite/ir/tfl_ops.h"
#include "tensorflow/compiler/mlir/lite/quantization/quantization_config.h"
#include "tensorflow/compiler/mlir/lite/tf_tfl_passes.h"
#include "tensorflow/compiler/mlir/lite/transforms/passes.h"
#include "tensorflow/compiler/mlir/lite/utils/convert_type.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/error_util.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/lite/schema/schema_generated.h"

namespace mlir {
namespace lite {

std::string TfLiteToMlir(const absl::string_view tflite_op_name) {
   std::vector<std::string> mht_0_v;
   mht_0_v.push_back("tflite_op_name: \"" + std::string(tflite_op_name.data(), tflite_op_name.size()) + "\"");
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePSquantizationPSlitePSquantize_modelDTcc mht_0(mht_0_v, 214, "", "./tensorflow/compiler/mlir/lite/quantization/lite/quantize_model.cc", "TfLiteToMlir");

  llvm::StringRef op_name(tflite_op_name.data(), tflite_op_name.size());
  return llvm::Twine("tfl.", op_name.lower()).str();
}

// TODO(fengliuai): check the result for `fully_quantize` flag.
TfLiteStatus QuantizeModel(
    const tflite::ModelT& input_model, const tflite::TensorType& input_type,
    const tflite::TensorType& output_type,
    const tflite::TensorType& inference_type,
    const std::unordered_set<std::string>& operator_names,
    bool disable_per_channel, bool fully_quantize,
    flatbuffers::FlatBufferBuilder* builder,
    tflite::ErrorReporter* error_reporter, bool verify_numeric,
    bool whole_model_verify, bool legacy_float_scale,
    const StringSet& denylisted_ops, const StringSet& denylisted_nodes) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePSquantizationPSlitePSquantize_modelDTcc mht_1(mht_1_v, 232, "", "./tensorflow/compiler/mlir/lite/quantization/lite/quantize_model.cc", "QuantizeModel");

  // Translate TFLite names to mlir op names.
  StringSet denylisted_mlir_op_names;
  for (const auto& entry : denylisted_ops) {
    denylisted_mlir_op_names.insert(TfLiteToMlir(entry));
  }

  DialectRegistry registry;
  registry.insert<mlir::TFL::TensorFlowLiteDialect>();
  MLIRContext context(registry);
  StatusScopedDiagnosticHandler statusHandler(&context,
                                              /*propagate=*/true);

  // Import input_model to a MLIR module
  flatbuffers::FlatBufferBuilder input_builder;
  flatbuffers::Offset<tflite::Model> input_model_location =
      tflite::Model::Pack(input_builder, &input_model);
  tflite::FinishModelBuffer(input_builder, input_model_location);

  std::string serialized_model(
      reinterpret_cast<const char*>(input_builder.GetBufferPointer()),
      input_builder.GetSize());

  OwningOpRef<mlir::ModuleOp> module = tflite::FlatBufferToMlir(
      serialized_model, &context, UnknownLoc::get(&context));
  if (!module) {
    error_reporter->Report("Couldn't import flatbuffer to MLIR.");
    return kTfLiteError;
  }

  // Apply quantization passes.
  PassManager pm(module->getContext(), OpPassManager::Nesting::Implicit);
  quant::QuantizationSpecs quant_specs;
  quant_specs.inference_type = tflite::TflTypeToTfType(inference_type);
  quant_specs.post_training_quantization = true;
  quant_specs.disable_per_channel = disable_per_channel;
  quant_specs.verify_numeric = verify_numeric;
  quant_specs.whole_model_verify = whole_model_verify;
  quant_specs.legacy_float_scale = legacy_float_scale;
  quant_specs.ops_blocklist = denylisted_mlir_op_names;
  quant_specs.nodes_blocklist = denylisted_nodes;

  llvm::dbgs() << "fully_quantize: " << fully_quantize
               << ", inference_type: " << quant_specs.inference_type
               << ", input_inference_type: " << input_type
               << ", output_inference_type: " << output_type << "\n";
  mlir::Builder mlir_builder(&context);
  mlir::Type input_mlir_type =
      tflite::ConvertElementType(input_type, mlir_builder);
  mlir::Type output_mlir_type =
      tflite::ConvertElementType(output_type, mlir_builder);

  if (fully_quantize) {
    input_mlir_type = tflite::ConvertElementType(inference_type, mlir_builder);
    output_mlir_type = input_mlir_type;
  }

  tensorflow::AddQuantizationPasses(quant_specs, pm);
  pm.addPass(TFL::CreateModifyIONodesPass(input_mlir_type, output_mlir_type));
  // If the first or final ops are not quantized, remove QDQ.
  pm.addPass(TFL::CreatePostQuantizeRemoveQDQPass());
  if (failed(pm.run(module.get()))) {
    const std::string& err = statusHandler.ConsumeStatus().error_message();
    error_reporter->Report("Failed to quantize: %s", err.c_str());
    return kTfLiteError;
  }

  // Export the results to the builder
  std::string result;
  tflite::FlatbufferExportOptions options;
  options.toco_flags.set_force_select_tf_ops(false);
  options.toco_flags.set_enable_select_tf_ops(true);
  options.toco_flags.set_allow_custom_ops(true);
  if (!tflite::MlirToFlatBufferTranslateFunction(module.get(), options,
                                                 &result)) {
    error_reporter->Report("Failed to export MLIR to flatbuffer.");
    return kTfLiteError;
  }
  builder->PushFlatBuffer(reinterpret_cast<const uint8_t*>(result.data()),
                          result.size());

  return kTfLiteOk;
}

}  // namespace lite
}  // namespace mlir
