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
class MHTracer_DTPStensorflowPScompilerPSmlirPSlitePSpythonPSsaved_model_to_tfl_flatbufferDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePSpythonPSsaved_model_to_tfl_flatbufferDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSmlirPSlitePSpythonPSsaved_model_to_tfl_flatbufferDTcc() {
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
#include "tensorflow/compiler/mlir/lite/python/saved_model_to_tfl_flatbuffer.h"

#include <memory>
#include <utility>

#include "absl/types/span.h"
#include "llvm/ADT/None.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Support/ToolOutputFile.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/IR/TypeUtilities.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Support/FileUtilities.h"  // from @llvm-project
#include "mlir/Transforms/ViewOpGraph.h"  // from @llvm-project
#include "tensorflow/cc/saved_model/loader.h"
#include "tensorflow/compiler/mlir/lite/common/tfl_pass_config.h"
#include "tensorflow/compiler/mlir/lite/python/tf_tfl_flatbuffer_helpers.h"
#include "tensorflow/compiler/mlir/lite/tf_tfl_passes.h"
#include "tensorflow/compiler/mlir/lite/tf_to_tfl_flatbuffer.h"
#include "tensorflow/compiler/mlir/lite/transforms/passes.h"
#include "tensorflow/compiler/mlir/tensorflow/translate/import_model.h"
#include "tensorflow/compiler/mlir/tensorflow/translate/mlir_roundtrip_flags.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/protobuf/graph_debug_info.pb.h"
#include "tensorflow/lite/toco/model_flags.pb.h"
#include "tensorflow/lite/toco/toco_flags.pb.h"
#include "tensorflow/lite/toco/types.pb.h"
#include "tensorflow/stream_executor/lib/statusor.h"

namespace tensorflow {

Status HandleInputOutputArraysWithModule(
    const toco::ModelFlags& model_flags,
    mlir::OwningOpRef<mlir::ModuleOp>* module) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePSpythonPSsaved_model_to_tfl_flatbufferDTcc mht_0(mht_0_v, 223, "", "./tensorflow/compiler/mlir/lite/python/saved_model_to_tfl_flatbuffer.cc", "HandleInputOutputArraysWithModule");

  mlir::func::FuncOp entry_function = nullptr;
  for (auto func : module->get().getOps<mlir::func::FuncOp>()) {
    if (auto tf_attrs =
            func->getAttrOfType<mlir::DictionaryAttr>("tf.entry_function")) {
      // TODO(b/184697652): There could be multiple entry functions. Let's
      // handle such cases if there are any needs for that.
      if (entry_function != nullptr) {
        return errors::InvalidArgument(
            "There should be only one tf.entry_function");
      }
      entry_function = func;
    }
  }
  if (entry_function == nullptr) {
    return errors::InvalidArgument("no tf.entry_function found");
  }

  // Get the list of input Op names from the function attribute.
  mlir::DictionaryAttr tf_attrs =
      entry_function->getAttrOfType<mlir::DictionaryAttr>("tf.entry_function");
  llvm::SmallVector<llvm::StringRef, 4> function_input_names;
  function_input_names.reserve(model_flags.input_arrays().size());
  auto input_attr = tf_attrs.get("inputs");
  if (!input_attr) {
    return errors::InvalidArgument("no inputs attribute found");
  }
  auto input_names = input_attr.cast<mlir::StringAttr>().getValue();
  input_names.split(function_input_names, ",", /*MaxSplit=*/-1,
                    /*KeepEmpty=*/false);
  const int function_input_names_size = function_input_names.size();
  if (function_input_names_size != model_flags.input_arrays().size()) {
    return errors::InvalidArgument(
        "input array size mismatch: got ", function_input_names.size(),
        ", expected: ", model_flags.input_arrays().size());
  }
  llvm::StringSet<> function_input_names_set;
  function_input_names_set.insert(function_input_names.begin(),
                                  function_input_names.end());
  for (const auto& input_array : model_flags.input_arrays()) {
    if (function_input_names_set.count(input_array.name()) == 0) {
      return errors::InvalidArgument("input array name (", input_array.name(),
                                     ") does not exist in the given graph");
    }
  }

  // Get the list of output Op names from the function attribute.
  llvm::SmallVector<llvm::StringRef, 4> function_output_names;
  function_output_names.reserve(model_flags.output_arrays().size());
  auto output_attr = tf_attrs.get("outputs");
  if (!output_attr) {
    return errors::InvalidArgument("no outputs attribute found");
  }
  auto output_names = output_attr.cast<mlir::StringAttr>().getValue();
  output_names.split(function_output_names, ",", /*MaxSplit=*/-1,
                     /*KeepEmpty=*/false);
  const int function_output_names_size = function_output_names.size();
  if (function_output_names_size != model_flags.output_arrays().size()) {
    return errors::InvalidArgument(
        "output array size mismatch: got ", function_output_names.size(),
        ", expected: ", model_flags.output_arrays().size());
  }
  llvm::StringSet<> function_output_names_set;
  function_output_names_set.insert(function_output_names.begin(),
                                   function_output_names.end());
  for (const auto& output_array : model_flags.output_arrays()) {
    if (function_output_names_set.count(output_array) == 0) {
      return errors::InvalidArgument("output array name (", output_array,
                                     ") does not exist in the given graph");
    }
  }
  return Status::OK();
}

Status ConvertSavedModelToTFLiteFlatBuffer(const toco::ModelFlags& model_flags,
                                           const toco::TocoFlags& toco_flags,
                                           string* result) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePSpythonPSsaved_model_to_tfl_flatbufferDTcc mht_1(mht_1_v, 302, "", "./tensorflow/compiler/mlir/lite/python/saved_model_to_tfl_flatbuffer.cc", "ConvertSavedModelToTFLiteFlatBuffer");

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

  auto& saved_model_tags = model_flags.saved_model_tags();
  auto& saved_model_exported_names = model_flags.saved_model_exported_names();
  std::unordered_set<std::string> tags(saved_model_tags.begin(),
                                       saved_model_tags.end());
  auto exported_names_in_vector = std::vector<std::string>(
      saved_model_exported_names.begin(), saved_model_exported_names.end());
  absl::Span<std::string> exported_names(exported_names_in_vector);

  if (exported_names.empty()) {
    return errors::Unimplemented("Need at least one exported name.");
  }

  tensorflow::GraphImportConfig specs;
  specs.upgrade_legacy = true;

  std::vector<std::string> custom_opdefs(toco_flags.custom_opdefs().begin(),
                                         toco_flags.custom_opdefs().end());
  auto bundle = std::make_unique<tensorflow::SavedModelBundle>();
  TF_ASSIGN_OR_RETURN(
      auto module,
      ImportSavedModel(
          model_flags.saved_model_dir(), model_flags.saved_model_version(),
          tags, absl::MakeSpan(custom_opdefs), exported_names, specs,
          !toco_flags.enable_tflite_resource_variables(), &context, &bundle));

  if (!model_flags.input_arrays().empty() ||
      !model_flags.output_arrays().empty()) {
    TF_RETURN_IF_ERROR(HandleInputOutputArraysWithModule(model_flags, &module));
  }

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
  pass_config.enable_dynamic_update_slice =
      toco_flags.enable_dynamic_update_slice();
  pass_config.preserve_assert_op = toco_flags.preserve_assert_op();
  pass_config.guarantee_all_funcs_one_use =
      toco_flags.guarantee_all_funcs_one_use();

  // TODO(b/153507667): Pass the session object when importing logic is removed.
  auto status = internal::ConvertMLIRToTFLiteFlatBuffer(
      model_flags, toco_flags, std::move(module), pass_config, tags, result,
      bundle ? bundle->GetSession() : nullptr);
  return status;
}

}  // namespace tensorflow
