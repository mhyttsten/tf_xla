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
class MHTracer_DTPStensorflowPScompilerPSmlirPSlitePSquantizationPSlitePSquantize_weightsDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePSquantizationPSlitePSquantize_weightsDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSmlirPSlitePSquantizationPSlitePSquantize_weightsDTcc() {
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

#include "tensorflow/compiler/mlir/lite/quantization/lite/quantize_weights.h"

#include <string>
#include <utility>

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
#include "tensorflow/compiler/mlir/lite/quantization/lite/quantize_model.h"
#include "tensorflow/compiler/mlir/lite/quantization/quantization_config.h"
#include "tensorflow/compiler/mlir/lite/tf_tfl_passes.h"
#include "tensorflow/compiler/mlir/lite/transforms/passes.h"
#include "tensorflow/compiler/mlir/lite/utils/convert_type.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/error_util.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/lite/schema/schema_generated.h"

namespace mlir {
namespace lite {

namespace {

using llvm::StringRef;

// Convert op represented in TFLite builtin_code to its corresponding MLIR
// OperationName.
void TfLiteBuiltinOpToMlir(const BuiltinOperatorSet& tflite_builtin_codes,
                           StringSet& mlir_op_names) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePSquantizationPSlitePSquantize_weightsDTcc mht_0(mht_0_v, 222, "", "./tensorflow/compiler/mlir/lite/quantization/lite/quantize_weights.cc", "TfLiteBuiltinOpToMlir");

  for (const auto& entry : tflite_builtin_codes) {
    StringRef tflite_op_name = EnumNameBuiltinOperator(entry);
    std::string mlir_name = llvm::Twine("tfl.", tflite_op_name.lower()).str();
    mlir_op_names.insert(std::move(mlir_name));
  }
}

std::string TfLiteToMlir(absl::string_view tflite_op_name) {
   std::vector<std::string> mht_1_v;
   mht_1_v.push_back("tflite_op_name: \"" + std::string(tflite_op_name.data(), tflite_op_name.size()) + "\"");
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePSquantizationPSlitePSquantize_weightsDTcc mht_1(mht_1_v, 234, "", "./tensorflow/compiler/mlir/lite/quantization/lite/quantize_weights.cc", "TfLiteToMlir");

  StringRef op_name(tflite_op_name.data(), tflite_op_name.size());
  return op_name.lower();
}

std::unique_ptr<tflite::ModelT> CreateMutableModelFromFile(
    const tflite::Model* input_model) {
  auto copied_model = absl::make_unique<tflite::ModelT>();
  input_model->UnPackTo(copied_model.get(), nullptr);
  return copied_model;
}
}  // namespace

// TODO(b/214314076): Support MLIR model as an input for the C++ dynamic range
// quantization API
TfLiteStatus QuantizeWeights(
    flatbuffers::FlatBufferBuilder* builder, const tflite::Model* input_model,
    tflite::ErrorReporter* error_reporter,
    const tflite::TensorType& inference_type, const StringSet& denylisted_ops,
    const CustomOpMap& custom_op_map, int64_t minimum_elements_for_weights,
    bool disable_per_channel, bool weight_only_quantization,
    bool legacy_float_scale) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePSquantizationPSlitePSquantize_weightsDTcc mht_2(mht_2_v, 258, "", "./tensorflow/compiler/mlir/lite/quantization/lite/quantize_weights.cc", "QuantizeWeights");

  // Translate TFLite names to mlir op names.
  StringSet denylisted_mlir_op_names;
  for (auto& entry : denylisted_ops) {
    denylisted_mlir_op_names.insert(TfLiteToMlir(entry));
  }

  DialectRegistry registry;
  MLIRContext context(registry);
  StatusScopedDiagnosticHandler statusHandler(&context,
                                              /*propagate=*/true);

  // Import input_model to a MLIR module
  flatbuffers::FlatBufferBuilder input_builder;
  flatbuffers::Offset<tflite::Model> input_model_location = tflite::Model::Pack(
      input_builder, CreateMutableModelFromFile(input_model).get());
  tflite::FinishModelBuffer(input_builder, input_model_location);

  std::string serialized_model(
      reinterpret_cast<const char*>(input_builder.GetBufferPointer()),
      input_builder.GetSize());

  OwningOpRef<mlir::ModuleOp> module = tflite::FlatBufferToMlir(
      serialized_model, &context, UnknownLoc::get(&context));

  // Apply quantization passes.
  PassManager pm(module->getContext(), OpPassManager::Nesting::Implicit);
  quant::QuantizationSpecs quant_specs;
  quant_specs.inference_type = tflite::TflTypeToTfType(inference_type);
  quant_specs.weight_quantization = true;
  quant_specs.weight_only_quantization = weight_only_quantization;
  quant_specs.minimum_elements_for_weights = minimum_elements_for_weights;
  quant_specs.disable_per_channel = disable_per_channel;
  quant_specs.legacy_float_scale = legacy_float_scale;
  quant_specs.ops_blocklist = denylisted_mlir_op_names;
  for (const auto& entry : custom_op_map) {
    quant_specs.custom_map[entry.first].quantizable_input_indices =
        entry.second.quantizable_input_indices;
    quant_specs.custom_map[entry.first].is_weight_only =
        entry.second.is_weight_only;
    quant_specs.custom_map[entry.first].no_side_effect =
        entry.second.no_side_effect;
  }

  if (quant_specs.inference_type == tensorflow::DT_INT8)
    quant_specs.inference_type = tensorflow::DT_QINT8;

  if (!(quant_specs.inference_type == tensorflow::DT_HALF ||
        quant_specs.inference_type == tensorflow::DT_QINT8)) {
    error_reporter->Report(
        "Couldn't apply dynamic range quantization since unsupported "
        "inference_type is passed.");
    return kTfLiteError;
  }

  llvm::dbgs() << "weight_quantization: " << true
               << ", weight_only_quantization: "
               << quant_specs.weight_only_quantization << ", mlir_quantizer: "
               << quant_specs.enable_mlir_dynamic_range_quantizer
               << ", inference_type: " << quant_specs.inference_type << "\n";
  Builder mlir_builder(&context);

  tensorflow::AddDynamicRangeQuantizationPasses(quant_specs, pm);

  if (failed(pm.run(module.get()))) {
    absl::string_view err = statusHandler.ConsumeStatus().error_message();
    error_reporter->Report("Failed to quantize: %s", err);
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

TfLiteStatus QuantizeWeights(flatbuffers::FlatBufferBuilder* builder,
                             const tflite::Model* input_model,
                             int64_t weights_min_num_elements,
                             bool use_hybrid_evaluation) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePSquantizationPSlitePSquantize_weightsDTcc mht_3(mht_3_v, 351, "", "./tensorflow/compiler/mlir/lite/quantization/lite/quantize_weights.cc", "QuantizeWeights");

  tflite::StderrReporter error_reporter;
  return QuantizeWeights(
      builder, input_model, &error_reporter,
      /*inference_type=*/tflite::TensorType_INT8,
      /*denylisted_ops=*/{},
      /*custom_op_map=*/{},
      /*minimum_elements_for_weights=*/weights_min_num_elements,
      /*disable_per_channel=*/false,
      /*weight_only_quantization=*/!use_hybrid_evaluation,
      /*legacy_float_scale=*/true);
}

// In MLIR use_updated_hybrid_scheme = true means per-channel operation.
TfLiteStatus QuantizeWeights(flatbuffers::FlatBufferBuilder* builder,
                             const tflite::Model* input_model,
                             BufferType quant_type,
                             bool use_updated_hybrid_scheme) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePSquantizationPSlitePSquantize_weightsDTcc mht_4(mht_4_v, 371, "", "./tensorflow/compiler/mlir/lite/quantization/lite/quantize_weights.cc", "QuantizeWeights");

  tflite::StderrReporter error_reporter;
  tflite::TensorType inference_type;
  switch (quant_type) {
    case BufferType::QUANTIZED_FLOAT16:
      inference_type = tflite::TensorType_FLOAT16;
      break;
    default:
      inference_type = tflite::TensorType_INT8;
  }
  return QuantizeWeights(builder, input_model, &error_reporter, inference_type,
                         /*denylisted_ops=*/{},
                         /*custom_op_map=*/{},
                         /*minimum_elements_for_weights=*/1024,
                         /*disable_per_channel=*/!use_updated_hybrid_scheme,
                         /*weight_only_quantization=*/false,
                         /*legacy_float_scale=*/true);
}

TfLiteStatus QuantizeWeights(flatbuffers::FlatBufferBuilder* builder,
                             const tflite::Model* input_model,
                             int64_t weights_min_num_elements,
                             const CustomOpMap& custom_op_map,
                             bool use_updated_hybrid_scheme,
                             const BuiltinOperatorSet& op_denylist) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePSquantizationPSlitePSquantize_weightsDTcc mht_5(mht_5_v, 398, "", "./tensorflow/compiler/mlir/lite/quantization/lite/quantize_weights.cc", "QuantizeWeights");

  tflite::StderrReporter error_reporter;
  const tflite::TensorType inference_type = tflite::TensorType_INT8;

  StringSet mlir_op_denylist;
  TfLiteBuiltinOpToMlir(op_denylist, mlir_op_denylist);

  return QuantizeWeights(
      builder, input_model, &error_reporter, inference_type,
      /*denylisted_ops=*/mlir_op_denylist,
      /*custom_op_map=*/custom_op_map,
      /*minimum_elements_for_weights=*/weights_min_num_elements,
      /*disable_per_channel=*/!use_updated_hybrid_scheme,
      /*weight_only_quantization=*/false,
      /*legacy_float_scale=*/true);
}

}  // namespace lite
}  // namespace mlir
