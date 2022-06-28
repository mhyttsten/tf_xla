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
class MHTracer_DTPStensorflowPScompilerPSmlirPSpythonPSmlirDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSmlirPSpythonPSmlirDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSmlirPSpythonPSmlirDTcc() {
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

#include "tensorflow/compiler/mlir/python/mlir.h"

#include <string>
#include <type_traits>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_set.h"
#include "absl/container/inlined_vector.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/strings/str_split.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/InitAllPasses.h"  // from @llvm-project
#include "mlir/Parser/Parser.h"  // from @llvm-project
#include "mlir/Pass/PassManager.h"  // from @llvm-project
#include "mlir/Pass/PassRegistry.h"  // from @llvm-project
#include "tensorflow/c/eager/c_api.h"
#include "tensorflow/c/eager/tfe_context_internal.h"
#include "tensorflow/c/tf_status.h"
#include "tensorflow/c/tf_status_helper.h"
#include "tensorflow/compiler/mlir/hlo/include/mlir-hlo/Dialect/lhlo/transforms/register_passes.h"
#include "tensorflow/compiler/mlir/hlo/include/mlir-hlo/Dialect/mhlo/transforms/register_passes.h"
#include "tensorflow/compiler/mlir/lite/ir/tfl_ops.h"
#include "tensorflow/compiler/mlir/tensorflow/dialect_registration.h"
#include "tensorflow/compiler/mlir/tensorflow/transforms/passes.h"
#include "tensorflow/compiler/mlir/tensorflow/transforms/tf_saved_model_passes.h"
#include "tensorflow/compiler/mlir/tensorflow/translate/import_model.h"
#include "tensorflow/compiler/mlir/tensorflow/translate/mlir_roundtrip_flags.h"
#include "tensorflow/compiler/mlir/tensorflow/translate/tf_mlir_translate.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/error_util.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/import_utils.h"
#include "tensorflow/compiler/mlir/tools/kernel_gen/ir/tf_framework_ops.h"
#include "tensorflow/compiler/mlir/tosa/tf_passes.h"
#include "tensorflow/compiler/mlir/tosa/tf_tfl_passes.h"
#include "tensorflow/compiler/mlir/tosa/tfl_passes.h"
#include "tensorflow/compiler/mlir/tosa/transforms/passes.h"
#include "tensorflow/compiler/mlir/xla/transforms/passes.h"
#include "tensorflow/compiler/mlir/xla/transforms/xla_passes.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/core/common_runtime/eager/context.h"
#include "tensorflow/core/common_runtime/function_body.h"
#include "tensorflow/core/common_runtime/function_def_utils.h"
#include "tensorflow/core/framework/function.h"
#include "tensorflow/core/framework/function.pb.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/tensor_shape.pb.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {

namespace {
// All the passes we will make available to Python by default.
// TODO(tf): this should be sharded instead of being monolithic like that.
static void RegisterPasses() {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSpythonPSmlirDTcc mht_0(mht_0_v, 243, "", "./tensorflow/compiler/mlir/python/mlir.cc", "RegisterPasses");

  static bool unique_registration = [] {
    mlir::registerAllPasses();
    mlir::registerTensorFlowPasses();
    mlir::TFDevice::registerTensorFlowDevicePasses();
    mlir::mhlo::registerAllMhloPasses();
    mlir::lmhlo::registerAllLmhloPasses();
    // These are in compiler/mlir/xla and not part of the above MHLO
    // passes.
    mlir::mhlo::registerXlaPasses();
    mlir::mhlo::registerTfXlaPasses();
    mlir::mhlo::registerLegalizeTFPass();
    mlir::mhlo::registerLegalizeTFControlFlowPass();
    mlir::mhlo::registerLegalizeTfTypesPassPass();
    mlir::tosa::registerLegalizeTosaPasses();
    mlir::tosa::registerTFtoTOSALegalizationPipeline();
    mlir::tosa::registerTFLtoTOSALegalizationPipeline();
    mlir::tosa::registerTFTFLtoTOSALegalizationPipeline();
    mlir::tf_saved_model::registerTensorFlowSavedModelPasses();
    return true;
  }();
  (void)unique_registration;
}

// Runs pass pipeline `pass_pipeline` on `module` if `pass_pipeline` is not
// empty.
std::string RunPassPipelineOnModule(mlir::ModuleOp module,
                                    const std::string &pass_pipeline,
                                    bool show_debug_info, TF_Status *status) {
   std::vector<std::string> mht_1_v;
   mht_1_v.push_back("pass_pipeline: \"" + pass_pipeline + "\"");
   MHTracer_DTPStensorflowPScompilerPSmlirPSpythonPSmlirDTcc mht_1(mht_1_v, 275, "", "./tensorflow/compiler/mlir/python/mlir.cc", "RunPassPipelineOnModule");

  RegisterPasses();
  if (!pass_pipeline.empty()) {
    mlir::PassManager pm(module.getContext());
    std::string error;
    llvm::raw_string_ostream error_stream(error);
    if (failed(mlir::parsePassPipeline(pass_pipeline, pm, error_stream))) {
      TF_SetStatus(status, TF_INVALID_ARGUMENT,
                   ("Invalid pass_pipeline: " + error_stream.str()).c_str());
      return "// error";
    }

    mlir::StatusScopedDiagnosticHandler statusHandler(module.getContext());
    if (failed(pm.run(module))) {
      Set_TF_Status_from_Status(status, statusHandler.ConsumeStatus());
      return "// error";
    }
  }
  return MlirModuleToString(module, show_debug_info);
}

}  // anonymous namespace

static std::string ImportGraphDefImpl(const std::string &proto,
                                      const std::string &pass_pipeline,
                                      bool show_debug_info,
                                      GraphDebugInfo &debug_info,
                                      GraphImportConfig &specs,
                                      TF_Status *status) {
   std::vector<std::string> mht_2_v;
   mht_2_v.push_back("proto: \"" + proto + "\"");
   mht_2_v.push_back("pass_pipeline: \"" + pass_pipeline + "\"");
   MHTracer_DTPStensorflowPScompilerPSmlirPSpythonPSmlirDTcc mht_2(mht_2_v, 308, "", "./tensorflow/compiler/mlir/python/mlir.cc", "ImportGraphDefImpl");

  GraphDef graphdef;
  auto s = tensorflow::LoadProtoFromBuffer(proto, &graphdef);
  if (!s.ok()) {
    Set_TF_Status_from_Status(status, s);
    return "// error";
  }
  mlir::MLIRContext context;
  auto module = ConvertGraphdefToMlir(graphdef, debug_info, specs, &context);
  if (!module.ok()) {
    Set_TF_Status_from_Status(status, module.status());
    return "// error";
  }

  return RunPassPipelineOnModule(module->get(), pass_pipeline, show_debug_info,
                                 status);
}

std::string ImportFunction(const std::string &functiondef_proto,
                           const std::string &pass_pipeline,
                           bool show_debug_info, TFE_Context *tfe_context,
                           TF_Status *status) {
   std::vector<std::string> mht_3_v;
   mht_3_v.push_back("functiondef_proto: \"" + functiondef_proto + "\"");
   mht_3_v.push_back("pass_pipeline: \"" + pass_pipeline + "\"");
   MHTracer_DTPStensorflowPScompilerPSmlirPSpythonPSmlirDTcc mht_3(mht_3_v, 334, "", "./tensorflow/compiler/mlir/python/mlir.cc", "ImportFunction");

  FunctionDef functiondef;
  auto s = tensorflow::LoadProtoFromBuffer(functiondef_proto, &functiondef);
  if (!s.ok()) {
    Set_TF_Status_from_Status(status, s);
    return "// error";
  }

  const std::string &function_name = functiondef.signature().name();
  EagerContext *cpp_context = ContextFromInterface(unwrap(tfe_context));
  FunctionLibraryDefinition &flib_def = *cpp_context->FuncLibDef();
  const tensorflow::FunctionDef *fdef = flib_def.Find(function_name);
  if (fdef == nullptr) {
    s = tensorflow::errors::NotFound("Cannot find function ", function_name);
    Set_TF_Status_from_Status(status, s);
    return "// error";
  }

  std::unique_ptr<tensorflow::FunctionBody> fbody;
  s = FunctionDefToBodyHelper(*fdef, tensorflow::AttrSlice(), &flib_def,
                              &fbody);
  if (!s.ok()) {
    Set_TF_Status_from_Status(status, s);
    return "// error";
  }

  mlir::MLIRContext context;
  auto module = ConvertFunctionToMlir(fbody.get(), flib_def, &context);
  if (!module.ok()) {
    Set_TF_Status_from_Status(status, module.status());
    return "// error";
  }

  return RunPassPipelineOnModule(module->get(), pass_pipeline, show_debug_info,
                                 status);
}

std::string ImportGraphDef(const std::string &proto,
                           const std::string &pass_pipeline,
                           bool show_debug_info, TF_Status *status) {
   std::vector<std::string> mht_4_v;
   mht_4_v.push_back("proto: \"" + proto + "\"");
   mht_4_v.push_back("pass_pipeline: \"" + pass_pipeline + "\"");
   MHTracer_DTPStensorflowPScompilerPSmlirPSpythonPSmlirDTcc mht_4(mht_4_v, 378, "", "./tensorflow/compiler/mlir/python/mlir.cc", "ImportGraphDef");

  GraphDebugInfo debug_info;
  GraphImportConfig specs;
  return ImportGraphDefImpl(proto, pass_pipeline, show_debug_info, debug_info,
                            specs, status);
}

std::string ImportGraphDef(const std::string &proto,
                           const std::string &pass_pipeline,
                           bool show_debug_info, absl::string_view input_names,
                           absl::string_view input_data_types,
                           absl::string_view input_data_shapes,
                           absl::string_view output_names, TF_Status *status) {
   std::vector<std::string> mht_5_v;
   mht_5_v.push_back("proto: \"" + proto + "\"");
   mht_5_v.push_back("pass_pipeline: \"" + pass_pipeline + "\"");
   mht_5_v.push_back("input_names: \"" + std::string(input_names.data(), input_names.size()) + "\"");
   mht_5_v.push_back("input_data_types: \"" + std::string(input_data_types.data(), input_data_types.size()) + "\"");
   mht_5_v.push_back("input_data_shapes: \"" + std::string(input_data_shapes.data(), input_data_shapes.size()) + "\"");
   mht_5_v.push_back("output_names: \"" + std::string(output_names.data(), output_names.size()) + "\"");
   MHTracer_DTPStensorflowPScompilerPSmlirPSpythonPSmlirDTcc mht_5(mht_5_v, 399, "", "./tensorflow/compiler/mlir/python/mlir.cc", "ImportGraphDef");

  std::vector<string> node_names = absl::StrSplit(input_names, ',');
  std::vector<string> node_dtypes = absl::StrSplit(input_data_types, ',');
  std::vector<string> node_shapes_str = absl::StrSplit(input_data_shapes, ':');
  std::vector<std::vector<int>> node_shapes;

  std::vector<int> dims;
  for (const string &shape_str : node_shapes_str) {
    dims.clear();
    if (!shape_str.empty()) {
      for (const auto &dim_str : absl::StrSplit(shape_str, ',')) {
        int size;
        if (absl::SimpleAtoi(dim_str, &size)) {
          dims.push_back(size);
        } else {
          auto s = tensorflow::errors::InvalidArgument(
              "Invalid Shape Specified.", dim_str);
          Set_TF_Status_from_Status(status, s);
          return "// error";
        }
      }
    }
    node_shapes.push_back(dims);
  }
  std::vector<string> output_nodes = absl::StrSplit(output_names, ',');

  GraphDebugInfo debug_info;
  GraphImportConfig specs;

  // Set the output to the output nodes.
  specs.outputs = output_nodes;

  // Set the input values to specs.input.
  std::vector<std::string> used_node_dtypes;
  if (node_dtypes.empty() ||
      (node_dtypes.size() == 1 && node_dtypes[0].empty())) {
    // Mark all the node dtypes Invalid, so the importer can handle them by
    // using the type from the graph.
    used_node_dtypes.resize(node_names.size(), DataType_Name(DT_INVALID));
  } else if (node_names.size() == node_dtypes.size()) {
    for (const auto &dtype : node_dtypes) {
      if (dtype.empty()) {
        used_node_dtypes.push_back(DataType_Name(DT_INVALID));
      } else if (dtype != DataType_Name(DT_INVALID)) {
        used_node_dtypes.push_back(dtype);

        // Use '' if you want to use the type from graph.
      } else {
        auto s = tensorflow::errors::InvalidArgument(
            "DT_INVALID isn't a valid input data type");
        Set_TF_Status_from_Status(status, s);
        return "// error";
      }
    }

    // Unmatched input node array and data type sizes.
  } else {
    auto s = tensorflow::errors::InvalidArgument(
        "Length of input node array and data type doesn't match");
    Set_TF_Status_from_Status(status, s);
    return "// error";
  }

  // Unmatched input node array and data shapes sizes.
  if (node_names.size() != node_shapes.size()) {
    auto s = tensorflow::errors::InvalidArgument(
        "Length of input node array and data shape doesn't match");
    Set_TF_Status_from_Status(status, s);
    return "// error";
  }
  for (unsigned i = 0, e = node_names.size(); i < e; i++) {
    const string &name = node_names[i];
    if (name.empty()) continue;

    auto it_inserted_pair = specs.inputs.insert({name, {}});
    ArrayInfo &info = it_inserted_pair.first->second;
    for (auto &dim : node_shapes[i]) {
      info.shape.add_dim()->set_size(dim);
    }
  }
  return ImportGraphDefImpl(proto, pass_pipeline, show_debug_info, debug_info,
                            specs, status);
}

std::string ExperimentalConvertSavedModelToMlir(
    const std::string &saved_model_path, const std::string &exported_names_str,
    bool show_debug_info, TF_Status *status) {
   std::vector<std::string> mht_6_v;
   mht_6_v.push_back("saved_model_path: \"" + saved_model_path + "\"");
   mht_6_v.push_back("exported_names_str: \"" + exported_names_str + "\"");
   MHTracer_DTPStensorflowPScompilerPSmlirPSpythonPSmlirDTcc mht_6(mht_6_v, 490, "", "./tensorflow/compiler/mlir/python/mlir.cc", "ExperimentalConvertSavedModelToMlir");

  // Load the saved model into a SavedModelV2Bundle.

  tensorflow::SavedModelV2Bundle bundle;
  auto load_status =
      tensorflow::SavedModelV2Bundle::Load(saved_model_path, &bundle);
  if (!load_status.ok()) {
    Set_TF_Status_from_Status(status, load_status);
    return "// error";
  }

  // Convert the SavedModelV2Bundle to an MLIR module.

  std::vector<string> exported_names =
      absl::StrSplit(exported_names_str, ',', absl::SkipEmpty());
  mlir::MLIRContext context;
  auto module_or = ConvertSavedModelToMlir(
      &bundle, &context, absl::Span<std::string>(exported_names));
  if (!module_or.status().ok()) {
    Set_TF_Status_from_Status(status, module_or.status());
    return "// error";
  }

  return MlirModuleToString(*module_or.ConsumeValueOrDie(), show_debug_info);
}

std::string ExperimentalConvertSavedModelV1ToMlirLite(
    const std::string &saved_model_path, const std::string &exported_names_str,
    const std::string &tags, bool upgrade_legacy, bool show_debug_info,
    TF_Status *status) {
   std::vector<std::string> mht_7_v;
   mht_7_v.push_back("saved_model_path: \"" + saved_model_path + "\"");
   mht_7_v.push_back("exported_names_str: \"" + exported_names_str + "\"");
   mht_7_v.push_back("tags: \"" + tags + "\"");
   MHTracer_DTPStensorflowPScompilerPSmlirPSpythonPSmlirDTcc mht_7(mht_7_v, 525, "", "./tensorflow/compiler/mlir/python/mlir.cc", "ExperimentalConvertSavedModelV1ToMlirLite");

  std::unordered_set<string> tag_set =
      absl::StrSplit(tags, ',', absl::SkipEmpty());

  std::vector<string> exported_names =
      absl::StrSplit(exported_names_str, ',', absl::SkipEmpty());
  mlir::MLIRContext context;

  tensorflow::MLIRImportOptions import_options;
  import_options.upgrade_legacy = upgrade_legacy;
  auto module_or = SavedModelSignatureDefsToMlirImportLite(
      saved_model_path, tag_set, absl::Span<std::string>(exported_names),
      &context, import_options);
  if (!module_or.status().ok()) {
    Set_TF_Status_from_Status(status, module_or.status());
    return "// error";
  }

  return MlirModuleToString(*module_or.ValueOrDie(), show_debug_info);
}

std::string ExperimentalConvertSavedModelV1ToMlir(
    const std::string &saved_model_path, const std::string &exported_names_str,
    const std::string &tags, bool lift_variables, bool upgrade_legacy,
    bool show_debug_info, TF_Status *status) {
   std::vector<std::string> mht_8_v;
   mht_8_v.push_back("saved_model_path: \"" + saved_model_path + "\"");
   mht_8_v.push_back("exported_names_str: \"" + exported_names_str + "\"");
   mht_8_v.push_back("tags: \"" + tags + "\"");
   MHTracer_DTPStensorflowPScompilerPSmlirPSpythonPSmlirDTcc mht_8(mht_8_v, 555, "", "./tensorflow/compiler/mlir/python/mlir.cc", "ExperimentalConvertSavedModelV1ToMlir");

  // Load the saved model into a SavedModelBundle.

  std::unordered_set<string> tag_set =
      absl::StrSplit(tags, ',', absl::SkipEmpty());

  tensorflow::SavedModelBundle bundle;
  auto load_status =
      tensorflow::LoadSavedModel({}, {}, saved_model_path, tag_set, &bundle);
  if (!load_status.ok()) {
    Set_TF_Status_from_Status(status, load_status);
    return "// error";
  }

  // Convert the SavedModelBundle to an MLIR module.
  std::vector<string> exported_names =
      absl::StrSplit(exported_names_str, ',', absl::SkipEmpty());
  mlir::MLIRContext context;
  tensorflow::MLIRImportOptions import_options;
  import_options.upgrade_legacy = upgrade_legacy;
  auto module_or =
      ConvertSavedModelV1ToMlir(bundle, absl::Span<std::string>(exported_names),
                                &context, import_options, lift_variables);
  if (!module_or.status().ok()) {
    Set_TF_Status_from_Status(status, module_or.status());
    return "// error";
  }

  // Run the tf standard pipeline by default and then, run passes that lift
  // variables if the flag is set on the module.
  mlir::OwningOpRef<mlir::ModuleOp> module = module_or.ConsumeValueOrDie();
  mlir::PassManager pm(&context);
  std::string error;
  llvm::raw_string_ostream error_stream(error);

  mlir::TF::StandardPipelineOptions tf_options;
  mlir::TF::CreateTFStandardPipeline(pm, tf_options);

  mlir::StatusScopedDiagnosticHandler diagnostic_handler(&context);
  if (failed(pm.run(*module))) {
    Set_TF_Status_from_Status(status, diagnostic_handler.ConsumeStatus());
    return "// error";
  }
  return MlirModuleToString(*module, show_debug_info);
}

std::string ExperimentalRunPassPipeline(const std::string &mlir_txt,
                                        const std::string &pass_pipeline,
                                        bool show_debug_info,
                                        TF_Status *status) {
   std::vector<std::string> mht_9_v;
   mht_9_v.push_back("mlir_txt: \"" + mlir_txt + "\"");
   mht_9_v.push_back("pass_pipeline: \"" + pass_pipeline + "\"");
   MHTracer_DTPStensorflowPScompilerPSmlirPSpythonPSmlirDTcc mht_9(mht_9_v, 609, "", "./tensorflow/compiler/mlir/python/mlir.cc", "ExperimentalRunPassPipeline");

  RegisterPasses();
  mlir::DialectRegistry registry;
  mlir::RegisterAllTensorFlowDialects(registry);
  mlir::MLIRContext context(registry);
  mlir::OwningOpRef<mlir::ModuleOp> module;
  {
    mlir::StatusScopedDiagnosticHandler diagnostic_handler(&context);
    module = mlir::parseSourceString<mlir::ModuleOp>(mlir_txt, &context);
    if (!module) {
      Set_TF_Status_from_Status(status, diagnostic_handler.ConsumeStatus());
      return "// error";
    }
  }

  // Run the pass_pipeline on the module.
  mlir::PassManager pm(&context);
  std::string error;
  llvm::raw_string_ostream error_stream(error);
  if (failed(mlir::parsePassPipeline(pass_pipeline, pm, error_stream))) {
    TF_SetStatus(status, TF_INVALID_ARGUMENT,
                 ("Invalid pass_pipeline: " + error_stream.str()).c_str());
    return "// error";
  }

  mlir::StatusScopedDiagnosticHandler diagnostic_handler(&context);
  if (failed(pm.run(*module))) {
    Set_TF_Status_from_Status(status, diagnostic_handler.ConsumeStatus());
    return "// error";
  }
  return MlirModuleToString(*module, show_debug_info);
}

}  // namespace tensorflow
