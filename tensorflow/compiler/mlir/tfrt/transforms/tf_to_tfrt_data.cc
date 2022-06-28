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
class MHTracer_DTPStensorflowPScompilerPSmlirPStfrtPStransformsPStf_to_tfrt_dataDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSmlirPStfrtPStransformsPStf_to_tfrt_dataDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSmlirPStfrtPStransformsPStf_to_tfrt_dataDTcc() {
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

// This file implements lowering of TF dialect to TFRT data kernels.
#include "tensorflow/compiler/mlir/tfrt/transforms/tf_to_tfrt_data.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/OperationSupport.h"  // from @llvm-project
#include "mlir/IR/Types.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Pass/PassManager.h"  // from @llvm-project
#include "mlir/Transforms/DialectConversion.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_types.h"
#include "tensorflow/compiler/mlir/tensorflow/transforms/passes.h"
#include "tensorflow/compiler/mlir/tensorflow/translate/import_model.h"
#include "tensorflow/compiler/mlir/tensorflow/translate/mlir_roundtrip_flags.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/error_util.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/protobuf/graph_debug_info.pb.h"
#include "tfrt/basic_kernels/opdefs/basic_kernels.h"  // from @tf_runtime
#include "tfrt/bef_converter/mlir_to_bef.h"  // from @tf_runtime
#include "tfrt/data/opdefs/data_ops.h"  // from @tf_runtime
#include "tfrt/data/opdefs/types.h"  // from @tf_runtime

#define DEBUG_TYPE "tf-to-tfrt-data"

namespace tensorflow {
namespace {

bool isIntScalar(Type t, size_t width) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStfrtPStransformsPStf_to_tfrt_dataDTcc mht_0(mht_0_v, 213, "", "./tensorflow/compiler/mlir/tfrt/transforms/tf_to_tfrt_data.cc", "isIntScalar");

  if (auto ttype = t.dyn_cast<RankedTensorType>()) {
    if (ttype.hasStaticShape() && ttype.getNumElements() == 1 &&
        ttype.getRank() == 0 && ttype.getElementType().isSignlessInteger(width))
      return true;
  }
  return false;
}

// Converts `value_attr` from a TF Const node to the required type attr type `U`
template <typename T>
T ConstAttrToTypeAttr(ElementsAttr value_attr) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStfrtPStransformsPStf_to_tfrt_dataDTcc mht_1(mht_1_v, 227, "", "./tensorflow/compiler/mlir/tfrt/transforms/tf_to_tfrt_data.cc", "ConstAttrToTypeAttr");

  if (T type_attr = value_attr.dyn_cast<T>()) {
    return type_attr;
  } else if (auto v = value_attr.dyn_cast<SplatElementsAttr>()) {
    return v.getSplatValue<Attribute>().dyn_cast<T>();
  }
  return T(nullptr);
}

template <typename T>
LogicalResult ReplaceConst(TF::ConstOp &op, ConversionPatternRewriter &rewriter,
                           Type type) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStfrtPStransformsPStf_to_tfrt_dataDTcc mht_2(mht_2_v, 241, "", "./tensorflow/compiler/mlir/tfrt/transforms/tf_to_tfrt_data.cc", "ReplaceConst");

  IntegerAttr newAttr = ConstAttrToTypeAttr<IntegerAttr>(op.value());

  if (!newAttr) {
    return failure();
  }

  auto tfrtConst = rewriter.create<T>(op.getLoc(), type, newAttr);
  rewriter.replaceOp(op.getOperation(), tfrtConst.getResult());
  return success();
}

mlir::Type CreateDatasetType(mlir::Builder *builder) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStfrtPStransformsPStf_to_tfrt_dataDTcc mht_3(mht_3_v, 256, "", "./tensorflow/compiler/mlir/tfrt/transforms/tf_to_tfrt_data.cc", "CreateDatasetType");

  return builder->getType<tfrt::data::DatasetType>();
}

// A helper class for converting data-specific types and attributes
class DataConverter : public mlir::TypeConverter {
 public:
  explicit DataConverter(mlir::MLIRContext *context) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStfrtPStransformsPStf_to_tfrt_dataDTcc mht_4(mht_4_v, 266, "", "./tensorflow/compiler/mlir/tfrt/transforms/tf_to_tfrt_data.cc", "DataConverter");

    addConversion([](Type type) { return type; });
    addConversion([context](TensorType type) {
      mlir::Builder builder(context);
      // tf.data datasets are represented by DT_VARIANT tensors in TF.
      // TODO(rachelim): Identify datasets more accurately.
      if (type.getElementType().dyn_cast<TF::VariantType>()) {
        return CreateDatasetType(&builder);
      }
      return type.dyn_cast<Type>();
    });
  }
};  // namespace

struct ConstOpConversion : public mlir::OpConversionPattern<TF::ConstOp> {
  explicit ConstOpConversion(MLIRContext *context)
      : OpConversionPattern<TF::ConstOp>(context) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStfrtPStransformsPStf_to_tfrt_dataDTcc mht_5(mht_5_v, 285, "", "./tensorflow/compiler/mlir/tfrt/transforms/tf_to_tfrt_data.cc", "ConstOpConversion");
}

  LogicalResult matchAndRewrite(
      TF::ConstOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStfrtPStransformsPStf_to_tfrt_dataDTcc mht_6(mht_6_v, 292, "", "./tensorflow/compiler/mlir/tfrt/transforms/tf_to_tfrt_data.cc", "matchAndRewrite");

    if (isIntScalar(op.getType(), 64)) {
      return ReplaceConst<tfrt::compiler::ConstantI64Op>(op, rewriter,
                                                         rewriter.getI64Type());
    }
    if (isIntScalar(op.getType(), 1)) {
      return ReplaceConst<tfrt::compiler::ConstantI1Op>(op, rewriter,
                                                        rewriter.getI1Type());
    }
    // TODO(rachelim): Support converting other const types.
    return failure();
  }
};

struct ReturnOpConversion
    : public mlir::OpConversionPattern<mlir::func::ReturnOp> {
  explicit ReturnOpConversion(MLIRContext *context)
      : OpConversionPattern<mlir::func::ReturnOp>(context) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStfrtPStransformsPStf_to_tfrt_dataDTcc mht_7(mht_7_v, 312, "", "./tensorflow/compiler/mlir/tfrt/transforms/tf_to_tfrt_data.cc", "ReturnOpConversion");
}

  LogicalResult matchAndRewrite(
      mlir::func::ReturnOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStfrtPStransformsPStf_to_tfrt_dataDTcc mht_8(mht_8_v, 319, "", "./tensorflow/compiler/mlir/tfrt/transforms/tf_to_tfrt_data.cc", "matchAndRewrite");

    rewriter.replaceOpWithNewOp<tfrt::compiler::ReturnOp>(
        op, adaptor.getOperands());
    return success();
  }
};

class RangeDatasetOpConversion
    : public OpConversionPattern<TF::RangeDatasetOp> {
 public:
  explicit RangeDatasetOpConversion(MLIRContext *context)
      : OpConversionPattern<TF::RangeDatasetOp>(context),
        builder_(context),
        dataset_type_(CreateDatasetType(&builder_)) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStfrtPStransformsPStf_to_tfrt_dataDTcc mht_9(mht_9_v, 335, "", "./tensorflow/compiler/mlir/tfrt/transforms/tf_to_tfrt_data.cc", "RangeDatasetOpConversion");
}

  LogicalResult matchAndRewrite(
      TF::RangeDatasetOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStfrtPStransformsPStf_to_tfrt_dataDTcc mht_10(mht_10_v, 342, "", "./tensorflow/compiler/mlir/tfrt/transforms/tf_to_tfrt_data.cc", "matchAndRewrite");

    if (op.output_types().size() != 1) {
      // Range dataset should only have one output type.
      return failure();
    }
    if (auto output_type = op.output_types().begin()->cast<TypeAttr>()) {
      rewriter.replaceOpWithNewOp<tfrt::data::RangeDatasetOp>(
          op, dataset_type_, adaptor.start(), adaptor.stop(), adaptor.step(),
          output_type);
      return success();
    }
    return failure();
  }

 private:
  mlir::Builder builder_;
  mlir::Type dataset_type_;
};

class BatchDatasetV2OpConversion
    : public OpConversionPattern<TF::BatchDatasetV2Op> {
 public:
  explicit BatchDatasetV2OpConversion(MLIRContext *context)
      : OpConversionPattern<TF::BatchDatasetV2Op>(context),
        builder_(context),
        dataset_type_(CreateDatasetType(&builder_)) {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStfrtPStransformsPStf_to_tfrt_dataDTcc mht_11(mht_11_v, 370, "", "./tensorflow/compiler/mlir/tfrt/transforms/tf_to_tfrt_data.cc", "BatchDatasetV2OpConversion");
}

  LogicalResult matchAndRewrite(
      TF::BatchDatasetV2Op op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStfrtPStransformsPStf_to_tfrt_dataDTcc mht_12(mht_12_v, 377, "", "./tensorflow/compiler/mlir/tfrt/transforms/tf_to_tfrt_data.cc", "matchAndRewrite");

    // Since TFRT's BatchDataset doesn't have a drop_remainder=True option,
    // we only convert this op if its drop_remainder input is statically known
    // to be false.
    auto drop_remainder_op = op.drop_remainder().getDefiningOp<TF::ConstOp>();
    if (!drop_remainder_op) return failure();
    BoolAttr drop_remainder_val =
        ConstAttrToTypeAttr<BoolAttr>(drop_remainder_op.value());
    if (!drop_remainder_val || drop_remainder_val.getValue()) {
      return failure();
    }

    // TODO(b/155892156): Support converting non-unary BatchDataset
    if (op.output_types().size() != 1) return failure();

    // TODO(b/155892156): Support converting BatchDataset with unknown rank
    auto output_shape = op.output_shapes()[0].cast<TF::ShapeAttr>();
    if (!output_shape.hasRank()) {
      return failure();
    }

    if (output_shape.getRank() >= 2) {  // Input is a tensor
      rewriter.replaceOpWithNewOp<tfrt::data::BatchDatasetTensorOp>(
          op, dataset_type_, adaptor.input_dataset(), adaptor.batch_size(),
          /*same_input_metadata=*/rewriter.getBoolAttr(false));
      return success();
    }

    auto output_type = op.output_types()[0].cast<TypeAttr>().getValue();

    if (output_type.isInteger(32)) {
      rewriter.replaceOpWithNewOp<tfrt::data::BatchDatasetI32Op>(
          op, dataset_type_, adaptor.input_dataset(), adaptor.batch_size(),
          /*same_input_metadata=*/rewriter.getBoolAttr(false));
      return success();
    }
    if (output_type.isInteger(64)) {
      rewriter.replaceOpWithNewOp<tfrt::data::BatchDatasetI64Op>(
          op, dataset_type_, adaptor.input_dataset(), adaptor.batch_size(),
          /*same_input_metadata=*/rewriter.getBoolAttr(false));
      return success();
    }
    return failure();
  }

 private:
  mlir::Builder builder_;
  mlir::Type dataset_type_;
};

// This rewrite converts a tf.data function that returns a tf.data dataset (in
// the TF dialect) to the equivalent function in the TFRT and Data dialects that
// returns a `!tfrt.dataset`.
//
// For now, this can only lower a RangeDataset op and its inputs. As we add more
// native TFRT datasets, we add the corresponding lowering pattern here.
class TFToTFRTDataRewritePass
    : public mlir::PassWrapper<TFToTFRTDataRewritePass,
                               mlir::OperationPass<mlir::ModuleOp>> {
  llvm::StringRef getArgument() const final {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStfrtPStransformsPStf_to_tfrt_dataDTcc mht_13(mht_13_v, 439, "", "./tensorflow/compiler/mlir/tfrt/transforms/tf_to_tfrt_data.cc", "getArgument");
 return "tf-to-tfrt-data"; }
  llvm::StringRef getDescription() const final {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStfrtPStransformsPStf_to_tfrt_dataDTcc mht_14(mht_14_v, 443, "", "./tensorflow/compiler/mlir/tfrt/transforms/tf_to_tfrt_data.cc", "getDescription");

    return "Convert Tensorflow dialect to TFRT's data dialect.";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
   std::vector<std::string> mht_15_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStfrtPStransformsPStf_to_tfrt_dataDTcc mht_15(mht_15_v, 450, "", "./tensorflow/compiler/mlir/tfrt/transforms/tf_to_tfrt_data.cc", "getDependentDialects");

    registry.insert<tfrt::compiler::TFRTDialect, tfrt::data::DataDialect>();
  }

  void runOnOperation() override {
   std::vector<std::string> mht_16_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStfrtPStransformsPStf_to_tfrt_dataDTcc mht_16(mht_16_v, 457, "", "./tensorflow/compiler/mlir/tfrt/transforms/tf_to_tfrt_data.cc", "runOnOperation");

    auto module = getOperation();
    auto *context = &getContext();
    mlir::ConversionTarget target(*context);
    DataConverter data_converter(context);
    target.addIllegalDialect<TF::TensorFlowDialect>();
    target.addLegalDialect<tfrt::data::DataDialect>();
    target.addLegalDialect<tfrt::compiler::TFRTDialect>();
    target.addDynamicallyLegalOp<mlir::func::FuncOp>(
        [&data_converter](FuncOp op) {
          return data_converter.isSignatureLegal(op.getFunctionType());
        });
    mlir::RewritePatternSet patterns(&getContext());
    patterns.add<RangeDatasetOpConversion, BatchDatasetV2OpConversion,
                 ConstOpConversion, ReturnOpConversion>(context);
    mlir::populateFunctionOpInterfaceTypeConversionPattern<FuncOp>(
        patterns, data_converter);

    auto result =
        mlir::applyPartialConversion(module, target, std::move(patterns));
    if (failed(result)) {
      signalPassFailure();
    }
  }
};

// Creates a pipeline of passes that converts MLIR TF Executor dialect to
// Hex and Data dialect.
void CreateTFExecutorToTFRTDataPipeline(mlir::OpPassManager &pm) {
   std::vector<std::string> mht_17_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStfrtPStransformsPStf_to_tfrt_dataDTcc mht_17(mht_17_v, 488, "", "./tensorflow/compiler/mlir/tfrt/transforms/tf_to_tfrt_data.cc", "CreateTFExecutorToTFRTDataPipeline");

  // Prune unused operations.
  pm.addPass(mlir::tf_executor::CreateTFExecutorGraphPruningPass());

  // Run the TF standard pipeline
  mlir::TF::StandardPipelineOptions tf_options;
  tf_options.enable_inliner = true;
  mlir::TF::CreateTFStandardPipeline(pm, tf_options);

  // After all the standard passes, lower to TFRT Data.
  pm.addPass(CreateTFToTFRTDataConversionPass());
}

Status TFDataGraphDefToTFDataMLIR(
    const GraphDef &graph_def, mlir::MLIRContext *mlir_ctx,
    mlir::OwningOpRef<mlir::ModuleOp> *module_ref) {
   std::vector<std::string> mht_18_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStfrtPStransformsPStf_to_tfrt_dataDTcc mht_18(mht_18_v, 506, "", "./tensorflow/compiler/mlir/tfrt/transforms/tf_to_tfrt_data.cc", "TFDataGraphDefToTFDataMLIR");

  // Import to TF dialect
  string output_node;
  for (const auto &node : graph_def.node()) {
    if (node.op() == "_Retval") {
      output_node = node.input(0);
      VLOG(2) << "Output node: " << output_node;
      break;
    }
  }
  auto import_config = tensorflow::GraphImportConfig();
  import_config.outputs.push_back(std::move(output_node));
  import_config.prune_unused_nodes = true;
  TF_ASSIGN_OR_RETURN(*module_ref, ConvertGraphdefToMlir(
                                       graph_def, tensorflow::GraphDebugInfo(),
                                       std::move(import_config), mlir_ctx));

  return Status::OK();
}

Status CompileTFDataMLIRToBEF(mlir::ModuleOp module,
                              tfrt::BefBuffer *bef_buffer) {
   std::vector<std::string> mht_19_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStfrtPStransformsPStf_to_tfrt_dataDTcc mht_19(mht_19_v, 530, "", "./tensorflow/compiler/mlir/tfrt/transforms/tf_to_tfrt_data.cc", "CompileTFDataMLIRToBEF");

  VLOG(1) << "TF Dialect: " << MlirModuleToString(module);

  mlir::PassManager pm(module.getContext());
  CreateTFExecutorToTFRTDataPipeline(pm);

  mlir::StatusScopedDiagnosticHandler diag_handler(module.getContext());
  if (mlir::failed(pm.run(module)))
    return diag_handler.Combine(
        errors::Internal("failed to lower TF Dialect to TFRT Data dialect."));

  VLOG(1) << "TFRT Dialect: " << MlirModuleToString(module);

  *bef_buffer =
      tfrt::ConvertMLIRToBEF(module, /*disable_optional_sections=*/false);
  if (bef_buffer->empty())
    return diag_handler.Combine(
        errors::Internal("failed to convert MLIR to BEF."));

  return Status::OK();
}

}  // namespace

std::unique_ptr<mlir::Pass> CreateTFToTFRTDataConversionPass() {
  return std::make_unique<TFToTFRTDataRewritePass>();
}

Status TFDataGraphDefToHostBEF(const GraphDef &graph_def,
                               tfrt::BefBuffer *bef) {
   std::vector<std::string> mht_20_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStfrtPStransformsPStf_to_tfrt_dataDTcc mht_20(mht_20_v, 562, "", "./tensorflow/compiler/mlir/tfrt/transforms/tf_to_tfrt_data.cc", "TFDataGraphDefToHostBEF");

  mlir::MLIRContext mlir_ctx;
  mlir::OwningOpRef<mlir::ModuleOp> module_ref;
  TF_RETURN_IF_ERROR(
      TFDataGraphDefToTFDataMLIR(graph_def, &mlir_ctx, &module_ref));
  TF_RETURN_IF_ERROR(CompileTFDataMLIRToBEF(module_ref.get(), bef));

  return Status::OK();
}

static mlir::PassRegistration<TFToTFRTDataRewritePass> pass;

}  // namespace tensorflow
