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
class MHTracer_DTPStensorflowPScompilerPSmlirPSlitePStf_tfl_passesDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePStf_tfl_passesDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSmlirPSlitePStf_tfl_passesDTcc() {
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

#include "tensorflow/compiler/mlir/lite/tf_tfl_passes.h"

#include <string>

#include "llvm/ADT/Optional.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Pass/PassManager.h"  // from @llvm-project
#include "mlir/Transforms/Passes.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/hlo/include/mlir-hlo/Dialect/mhlo/transforms/passes.h"
#include "tensorflow/compiler/mlir/lite/quantization/quantization_config.h"
#include "tensorflow/compiler/mlir/lite/quantization/quantization_passes.h"
#include "tensorflow/compiler/mlir/lite/quantization/tensorflow/passes.h"
#include "tensorflow/compiler/mlir/lite/transforms/passes.h"
#include "tensorflow/compiler/mlir/lite/utils/fake_quant_utils.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops_a_m.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_saved_model.h"
#include "tensorflow/compiler/mlir/tensorflow/transforms/passes.h"
#include "tensorflow/compiler/mlir/tensorflow/transforms/tf_saved_model_passes.h"
#include "tensorflow/compiler/mlir/tensorflow/translate/tf_mlir_translate.h"

namespace mlir {
/// Create a pass to convert from the TFExecutor to the TF control dialect.
std::unique_ptr<OperationPass<FuncOp>>
CreateTFExecutorToControlDialectConversion();
}  // namespace mlir

namespace tensorflow {
namespace {
// Data layout supported by TFLite.
const char kTFLiteDataLayout[] = "NHWC";
}  // namespace

void AddQuantizationPasses(const mlir::quant::QuantizationSpecs& quant_specs,
                           mlir::OpPassManager& pass_manager) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePStf_tfl_passesDTcc mht_0(mht_0_v, 222, "", "./tensorflow/compiler/mlir/lite/tf_tfl_passes.cc", "AddQuantizationPasses");

  pass_manager.addNestedPass<mlir::func::FuncOp>(
      mlir::TFL::CreatePrepareQuantizePass(quant_specs));
  if (quant_specs.default_ranges.first.hasValue() ||
      quant_specs.default_ranges.second.hasValue()) {
    pass_manager.addNestedPass<mlir::func::FuncOp>(
        mlir::TFL::CreateDefaultQuantParamsPass(
            quant_specs.default_ranges.first.getValueOr(0.0),
            quant_specs.default_ranges.second.getValueOr(0.0),
            quant_specs.IsSignedInferenceType()));
  }
  pass_manager.addNestedPass<mlir::func::FuncOp>(
      mlir::TFL::CreateQuantizePass(quant_specs));
  bool emit_quant_adaptor_ops =
      quant_specs.inference_type != quant_specs.inference_input_type;
  pass_manager.addNestedPass<mlir::func::FuncOp>(
      mlir::TFL::CreatePostQuantizePass(emit_quant_adaptor_ops));
  pass_manager.addNestedPass<mlir::func::FuncOp>(
      mlir::TFL::CreateOptimizeOpOrderPass());
}

void AddDynamicRangeQuantizationPasses(
    const mlir::quant::QuantizationSpecs& quant_specs,
    mlir::OpPassManager& pass_manager) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePStf_tfl_passesDTcc mht_1(mht_1_v, 248, "", "./tensorflow/compiler/mlir/lite/tf_tfl_passes.cc", "AddDynamicRangeQuantizationPasses");

  pass_manager.addNestedPass<mlir::func::FuncOp>(
      mlir::TFL::CreatePrepareDynamicRangeQuantizePass(quant_specs));
  pass_manager.addNestedPass<mlir::func::FuncOp>(
      mlir::TFL::CreateQuantizePass(quant_specs));
  bool emit_quant_adaptor_ops =
      quant_specs.inference_type != quant_specs.inference_input_type;
  pass_manager.addNestedPass<mlir::func::FuncOp>(
      mlir::TFL::CreatePostQuantizePass(emit_quant_adaptor_ops,
                                        quant_specs.custom_map));
  pass_manager.addNestedPass<mlir::func::FuncOp>(
      mlir::TFL::CreateOptimizeOpOrderPass());
}

void AddConvertHloToTfPass(std::string entry_function_name,
                           mlir::OpPassManager* pass_manager) {
   std::vector<std::string> mht_2_v;
   mht_2_v.push_back("entry_function_name: \"" + entry_function_name + "\"");
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePStf_tfl_passesDTcc mht_2(mht_2_v, 267, "", "./tensorflow/compiler/mlir/lite/tf_tfl_passes.cc", "AddConvertHloToTfPass");

  // Legalize jax random to tflite custom op.
  // The CreateLegalizeJaxRandom Pass has to stay at because we need to replace
  // the random function body before being inlined.
  pass_manager->addNestedPass<mlir::func::FuncOp>(
      mlir::TFL::CreateLegalizeJaxRandomPass());

  // Canonicalize, CSE etc.
  pass_manager->addNestedPass<mlir::func::FuncOp>(
      mlir::createCanonicalizerPass());
  pass_manager->addNestedPass<mlir::func::FuncOp>(mlir::createCSEPass());
  // DCE for private symbols.
  pass_manager->addPass(mlir::createSymbolDCEPass());

  pass_manager->addPass(mlir::TF::CreateStripNoinlineAttributePass());
  // Add inline pass.
  pass_manager->addPass(mlir::createInlinerPass());

  // Expands mhlo.tuple ops.
  pass_manager->addPass(
      mlir::mhlo::CreateExpandHloTuplesPass(entry_function_name));
  // Flatten tuples for control flows.
  pass_manager->addNestedPass<mlir::func::FuncOp>(
      mlir::mhlo::createFlattenTuplePass());

  // TF dialect passes
  pass_manager->addNestedPass<mlir::func::FuncOp>(
      mlir::TF::CreateLegalizeHloToTfPass());

  // Canonicalization after TF legalization.
  pass_manager->addNestedPass<mlir::func::FuncOp>(
      mlir::createCanonicalizerPass());
}

// This is the early part of the conversion in isolation. This enables a caller
// to inject more information in the middle of the conversion before resuming
// it.
void AddPreVariableFreezingTFToTFLConversionPasses(
    const mlir::TFL::PassConfig& pass_config,
    mlir::OpPassManager* pass_manager) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePStf_tfl_passesDTcc mht_3(mht_3_v, 309, "", "./tensorflow/compiler/mlir/lite/tf_tfl_passes.cc", "AddPreVariableFreezingTFToTFLConversionPasses");

  if (pass_config.enable_hlo_to_tf_conversion) {
    // TODO(b/194747383): We need to valid that indeed the "main" func is
    // presented.
    AddConvertHloToTfPass("main", pass_manager);
  }
  // This pass wraps all the tf.FakeQuant ops in a custom op so they are not
  // folded before being converted to tfl.quantize and tfl.dequantize ops.
  auto wrapped_ops = mlir::TFL::AllTfFakeQuantOps();
  pass_manager->addNestedPass<mlir::func::FuncOp>(
      mlir::TFL::CreateRaiseCustomOpsPass(wrapped_ops));

  mlir::TF::StandardPipelineOptions standard_pipeline_options;
  standard_pipeline_options.enable_inliner = false;
  standard_pipeline_options.form_clusters = pass_config.form_clusters;
  mlir::TF::CreateTFStandardPipeline(*pass_manager, standard_pipeline_options);
  pass_manager->addNestedPass<mlir::func::FuncOp>(
      mlir::TF::CreateDeviceIndexSelectorPass());

  // Add canonicalize pass to remove no-op session initializer pass.
  pass_manager->addPass(mlir::createCanonicalizerPass());

  if (pass_config.guarantee_all_funcs_one_use) {
    pass_manager->addPass(mlir::TF::CreateGuaranteeAllFuncsOneUsePass());
  }
  if (pass_config.shape_inference) {
    pass_manager->addPass(mlir::TF::CreateTFShapeInferencePass());
  }

  // Keep this pass after the shape inference pass, which couldn't do shape
  // inference for non-tf ops.
  if (!pass_config.quant_specs.serialized_quant_stats.empty()) {
    pass_manager->addNestedPass<mlir::func::FuncOp>(
        mlir::quant::CreateImportQuantStatsPassForTFControlDialect(
            pass_config.quant_specs.serialized_quant_stats));
  }

  pass_manager->addPass(mlir::TF::CreateTFFunctionalControlFlowToRegions());

  // The conversion pipeline has to follow the following orders:
  // 1) Saved model related optimization like decompose resource ops
  // 2) Convert composite functions like lstm/rnns, along with proper function
  // inlining & dce.
  // 3) Lower static tensor list pass.

  // This decomposes resource ops like ResourceGather into read-variable op
  // followed by gather. This is used when the saved model import path is used
  // during which resources dont get frozen in the python layer.
  pass_manager->addNestedPass<mlir::func::FuncOp>(
      mlir::TFDevice::CreateDecomposeResourceOpsPass());

  pass_manager->addPass(mlir::TF::CreateTFRegionControlFlowToFunctional());
}

// This is the later part of the conversion in isolation. This enables a caller
// to resume the conversion after injecting more information in the middle of
// it.
void AddPostVariableFreezingTFToTFLConversionPasses(
    llvm::StringRef saved_model_dir, const toco::TocoFlags& toco_flags,
    const mlir::TFL::PassConfig& pass_config,
    mlir::OpPassManager* pass_manager) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePStf_tfl_passesDTcc mht_4(mht_4_v, 372, "", "./tensorflow/compiler/mlir/lite/tf_tfl_passes.cc", "AddPostVariableFreezingTFToTFLConversionPasses");

  // Note:
  // We need to fuse composite ops before LowerStaticTensorList pass.
  // The tensorflow list is not supported right now by that pass.
  // Enable fusing composite ops that can be lowered to built-in TFLite ops.
  if (pass_config.emit_builtin_tflite_ops &&
      toco_flags.tf_quantization_mode().empty()) {
    pass_manager->addPass(mlir::TFL::CreatePrepareCompositeFunctionsPass());
  }

  pass_manager->addPass(mlir::createInlinerPass());
  pass_manager->addPass(mlir::createSymbolDCEPass());

  if (pass_config.lower_tensor_list_ops &&
      toco_flags.tf_quantization_mode().empty()) {
    // TODO(haoliang): Add this pass by default.
    pass_manager->addPass(mlir::TFL::CreateLowerStaticTensorListPass(
        /*allow_tensorlist_pass_through=*/toco_flags.force_select_tf_ops() ||
            toco_flags.enable_select_tf_ops(),
        /*default_to_single_batch=*/
        toco_flags.default_to_single_batch_in_tensor_list_ops(),
        /*enable_dynamic_update_slice=*/
        toco_flags.enable_dynamic_update_slice()));
  }

  // This pass does resource analysis of saved model global tensors and marks
  // those deemed read-only as immutable.
  pass_manager->addPass(
      mlir::tf_saved_model::CreateOptimizeGlobalTensorsPass());

  if (pass_config.shape_inference) {
    // Add a shape inference pass to optimize away the unnecessary casts.
    pass_manager->addPass(mlir::TF::CreateTFShapeInferencePass());
  }

  // Legalize while early to allow further constant folding.
  // TODO(jpienaar): This may not actually matter as we do canonicalization
  // after the legalize below, for now it needs to be below the above passes
  // that work on TF dialect and before inliner so that the function calls in
  // body and cond are inlined for optimization.
  pass_manager->addPass(mlir::TFL::CreateLegalizeTFWhilePass());

  // Add function inlining pass. Both TF and TFLite dialects are opted into
  // function inliner interface.
  pass_manager->addPass(mlir::createInlinerPass());
  // Reduce operands of TFL::While without changing the outcome.
  // It needs to stay here because:
  // 1. WhileOps are in TFL dialect.
  // 2. The body and cond are inlined.
  // 3. We need to do this before while canonicalization, otherwise it would be
  //   difficult to find dependencies.
  pass_manager->addNestedPass<mlir::func::FuncOp>(
      mlir::TFL::CreateReduceWhileOperandsPass());
  // Canonicalization includes const folding, which is utilized here to optimize
  // away ops that can't get constant folded after PrepareTF pass. For example,
  // tf.Conv2D is split into tf.Transpose and tfl.Conv2D.
  pass_manager->addNestedPass<mlir::func::FuncOp>(
      mlir::createCanonicalizerPass());
  pass_manager->addNestedPass<mlir::func::FuncOp>(mlir::createCSEPass());
  // This pass does dead code elimination based on symbol visibility.
  pass_manager->addPass(mlir::createSymbolDCEPass());

  if (!pass_config.disable_variable_freezing) {
    // This pass 'freezes' immutable global tensors and inlines them as tf
    // constant ops.
    pass_manager->addPass(mlir::tf_saved_model::CreateFreezeGlobalTensorsPass(
        /*allow_mutable_tensors=*/pass_config.enable_tflite_variables));
  }

  if (!saved_model_dir.empty()) {
    // This pass 'freezes' tf saved model asset ops and inlines as string values
    // in a format of the tf constant op.
    pass_manager->addPass(
        mlir::tf_saved_model::CreateFreezeAssetsPass(saved_model_dir.str()));
  }
  // For TF Quantization, convert unsupported ops to Flex ops before other
  // conversion passes.
  if (!toco_flags.tf_quantization_mode().empty()) {
    pass_manager->addNestedPass<mlir::func::FuncOp>(
        mlir::TF::CreateFallbackToFlexOpsPass(
            toco_flags.tf_quantization_mode()));
  }
  // The below passes only make sense if Builtin TFLite ops are enabled
  // for emission.
  if (pass_config.emit_builtin_tflite_ops) {
    // Run shape inference after variables are converted to constants.
    if (pass_config.shape_inference) {
      pass_manager->addPass(mlir::TF::CreateTFShapeInferencePass());
    }
    // Force layout supported by TFLite, this will transpose the data
    // to match 'kTFLiteDataLayout'
    mlir::TF::LayoutOptimizationPipelineOptions layout_optimization_options;
    layout_optimization_options.force_data_format = kTFLiteDataLayout;
    layout_optimization_options.skip_fold_transpose_in_ops = true;
    mlir::TF::CreateLayoutOptimizationPipeline(
        pass_manager->nest<mlir::func::FuncOp>(), layout_optimization_options);
    // Prepare for TFLite dialect, rerun canonicalization, and then legalize to
    // the TFLite dialect.
    pass_manager->addNestedPass<mlir::func::FuncOp>(
        mlir::TFL::CreatePrepareTFPass(pass_config.unfold_batch_matmul,
                                       /*allow_bf16_and_f16_type_legalization=*/
                                       !pass_config.runtime_verification,
                                       toco_flags.use_fake_quant_num_bits()));
    pass_manager->addNestedPass<mlir::func::FuncOp>(
        mlir::createCanonicalizerPass());
    if (pass_config.shape_inference) {
      // Add a shape inference pass to optimize away the unnecessary casts.
      // This also fixes the unranked shapes due to TF ops constant folding.
      // TODO(fengliuai): remove this pass if TableGen patterns have a better
      // to control the shapes for the intermediate results.
      pass_manager->addPass(mlir::TF::CreateTFShapeInferencePass());
    }

    // Inline function calls that left in the graph after folding functional
    // control flow ops (IfOp, CaseOp).
    pass_manager->addPass(mlir::createInlinerPass());

    // This pass removes the asset file dependencies in hash table use cases.
    pass_manager->addNestedPass<mlir::func::FuncOp>(
        mlir::TF::CreateInitTextFileToImportPass(saved_model_dir.str()));

    pass_manager->addNestedPass<mlir::func::FuncOp>(
        mlir::TFL::CreateLegalizeTFPass(pass_config.runtime_verification,
                                        pass_config.preserve_assert_op));
    pass_manager->addPass(mlir::TFL::CreateAnalyzeVariablesPass());
    pass_manager->addPass(mlir::TFL::CreateLegalizeVariablesPass());
    pass_manager->addPass(mlir::TFL::CreateLegalizeHashTablesPass());
    pass_manager->addNestedPass<mlir::func::FuncOp>(
        mlir::TFL::CreateOptimizePass(/*enable_canonicalization=*/true));
    // This pass operates on TensorFlow ops but is triggered after legalization
    // so that it can target constants introduced once TensorFlow Identity ops
    // are removed during legalization.
    pass_manager->addPass(mlir::TFL::CreateOptimizeFunctionalOpsPass());
    std::vector<std::string> empty_wrapped_ops({});
    pass_manager->addNestedPass<mlir::func::FuncOp>(
        mlir::TFL::CreateRaiseCustomOpsPass(empty_wrapped_ops));
    pass_manager->addPass(mlir::createSymbolDCEPass());
    pass_manager->addNestedPass<mlir::func::FuncOp>(
        mlir::createCanonicalizerPass());
    pass_manager->addNestedPass<mlir::func::FuncOp>(mlir::createCSEPass());

    // Run quantization after all the floating point model conversion is
    // completed. Add either full integer quantization or dynamic range
    // quantization passes based on quant_specs.
    if (pass_config.quant_specs.RunPropagationAndRewriteQuantizationPasses()) {
      AddQuantizationPasses(pass_config.quant_specs, *pass_manager);
    } else if (pass_config.quant_specs
                   .RunAndRewriteDynamicRangeQuantizationPasses()) {
      AddDynamicRangeQuantizationPasses(pass_config.quant_specs, *pass_manager);
    }
    pass_manager->addPass(mlir::createCanonicalizerPass());

    // This pass should be always at the end of the model
    // conversion (even after quantization). Some TFL ops like unidirectional
    // sequence lstm will have stateful operands and some optimization passes
    // will merge those operands if they have identical values & types. However,
    // it's not desired by TFL. This pass serves as a "fix" pass to split the
    // merged inputs until we have 1st class variable support or reuse
    // tf.variable to model this.
    pass_manager->addNestedPass<mlir::func::FuncOp>(
        mlir::TFL::CreateSplitMergedOperandsPass());

    // Add CallOnceOp when there is a session initializer function in tf saved
    // model dialect.
    pass_manager->addPass(
        mlir::TFL::CreateInsertCallOnceOpFromSessionInitializerPass());
  }
  if (pass_config.unfold_large_splat_constant) {
    pass_manager->addPass(mlir::TFL::CreateUnfoldLargeSplatConstantPass());
  }
  if (pass_config.outline_tf_while) {
    pass_manager->addPass(mlir::TFL::CreateWhileOutlinePass());
  }
  if (pass_config.runtime_verification) {
    pass_manager->addNestedPass<mlir::func::FuncOp>(
        mlir::TFL::CreateRuntimeVerifyPass());
  }
}

void AddTFToTFLConversionPasses(llvm::StringRef saved_model_dir,
                                const toco::TocoFlags& toco_flags,
                                const mlir::TFL::PassConfig& pass_config,
                                mlir::OpPassManager* pass_manager) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePStf_tfl_passesDTcc mht_5(mht_5_v, 557, "", "./tensorflow/compiler/mlir/lite/tf_tfl_passes.cc", "AddTFToTFLConversionPasses");

  AddPreVariableFreezingTFToTFLConversionPasses(pass_config, pass_manager);
  AddPostVariableFreezingTFToTFLConversionPasses(saved_model_dir, toco_flags,
                                                 pass_config, pass_manager);
}
void AddTFToTFLConversionPasses(const mlir::TFL::PassConfig& pass_config,
                                mlir::OpPassManager* pass_manager) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePStf_tfl_passesDTcc mht_6(mht_6_v, 566, "", "./tensorflow/compiler/mlir/lite/tf_tfl_passes.cc", "AddTFToTFLConversionPasses");

  const toco::TocoFlags toco_flags;
  AddTFToTFLConversionPasses(/*saved_model_dir=*/"", toco_flags, pass_config,
                             pass_manager);
}

}  // namespace tensorflow

namespace mlir {
namespace TFL {

struct StandardPipelineOptions
    : public PassPipelineOptions<StandardPipelineOptions> {
  // TODO(b/150915052): All the tf_tfl_translate_cl flags should
  // move inside this.
};

// NOLINTNEXTLINE
// This creates the standard pass pipeline for TF->TFLite. This
// represents a std configuration for TFLite, for use with APIs like
// tensorflow/python/pywrap_mlir.py::experimental_run_pass_pipeline
// This does not yet include quantization passes.
void CreateTFLStandardPipeline(OpPassManager& pm,
                               const StandardPipelineOptions& options) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePStf_tfl_passesDTcc mht_7(mht_7_v, 592, "", "./tensorflow/compiler/mlir/lite/tf_tfl_passes.cc", "CreateTFLStandardPipeline");

  OpPassManager& func_pm = pm.nest<FuncOp>();

  // tf_executor dialect passes - Cleaning up the IR.
  mlir::TF::StandardPipelineOptions standard_pipeline_options;
  mlir::TF::CreateTFStandardPipeline(func_pm, standard_pipeline_options);

  // This is needed for control flow support with TF TensorList.
  pm.addPass(mlir::TFL::CreateLowerStaticTensorListPass(
      /*allow_tensorlist_pass_through=*/false,
      /*default_to_single_batch=*/false,
      /*enable_dynamic_update_slice=*/false));

  // Saved model pass to mark global tensors immutable.
  pm.addPass(mlir::tf_saved_model::CreateOptimizeGlobalTensorsPass());
  // Op fusion pass.
  pm.addPass(mlir::TFL::CreatePrepareCompositeFunctionsPass());

  pm.addNestedPass<mlir::func::FuncOp>(mlir::TFL::CreateLegalizeTFWhilePass());

  pm.addPass(mlir::createInlinerPass());

  // Canonicalize, CSE etc.
  pm.addNestedPass<mlir::func::FuncOp>(mlir::createCanonicalizerPass());
  pm.addNestedPass<mlir::func::FuncOp>(mlir::createCSEPass());
  // DCE for private symbols.
  pm.addPass(mlir::createSymbolDCEPass());

  // freeze global tensors.
  pm.addPass(mlir::tf_saved_model::CreateFreezeGlobalTensorsPass());

  // TFLite dialect passes.
  pm.addPass(mlir::TFL::CreatePrepareTFPass(
      /*unfold_batch_matmul=*/true,
      /*allow_bf16_and_f16_type_legalization=*/false));
  pm.addNestedPass<mlir::func::FuncOp>(mlir::createCanonicalizerPass());
  pm.addPass(
      mlir::TFL::CreateLegalizeTFPass(/*run_tfl_runtime_verification=*/true,
                                      /*preserve_assert_op=*/false));
  pm.addPass(mlir::TFL::CreateLegalizeHashTablesPass());
  pm.addPass(mlir::TFL::CreateOptimizePass(/*enable_canonicalization=*/true));
  pm.addPass(mlir::TFL::CreateOptimizeFunctionalOpsPass());
  pm.addPass(mlir::createSymbolDCEPass());

  // Canonicalize, CSE etc.
  pm.addNestedPass<mlir::func::FuncOp>(mlir::createCanonicalizerPass());
  pm.addNestedPass<mlir::tf_saved_model::SessionInitializerOp>(
      mlir::createCanonicalizerPass());
  pm.addNestedPass<mlir::func::FuncOp>(mlir::createCSEPass());

  // Pass for stateful operands like LSTM.
  pm.addPass(mlir::TFL::CreateSplitMergedOperandsPass());

  pm.addPass(mlir::TFL::CreateWhileOutlinePass());

  pm.addNestedPass<mlir::func::FuncOp>(mlir::TFL::CreateRuntimeVerifyPass());
}

// Registers a pass pipeline for the standard TFL passes.
static mlir::PassPipelineRegistration<StandardPipelineOptions> pipeline(
    "tfl-standard-pipeline",
    "Run the standard passes involved in transforming/optimizing the TF "
    "program to TFLite after "
    "importing into MLIR.",
    CreateTFLStandardPipeline);

}  // namespace TFL
}  // namespace mlir
