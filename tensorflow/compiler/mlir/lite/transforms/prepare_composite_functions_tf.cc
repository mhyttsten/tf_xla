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
class MHTracer_DTPStensorflowPScompilerPSmlirPSlitePStransformsPSprepare_composite_functions_tfDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePStransformsPSprepare_composite_functions_tfDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSmlirPSlitePStransformsPSprepare_composite_functions_tfDTcc() {
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

#include <string>

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/None.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/IR/Location.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/IR/Operation.h"  // from @llvm-project
#include "mlir/IR/SymbolTable.h"  // from @llvm-project
#include "mlir/IR/Visitors.h"  // from @llvm-project
#include "mlir/Interfaces/CallInterfaces.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/lite/ir/tfl_ops.h"
#include "tensorflow/compiler/mlir/lite/transforms/passes.h"
#include "tensorflow/compiler/mlir/lite/utils/lstm_utils.h"
#include "tensorflow/compiler/mlir/lite/utils/nms_utils.h"
#include "tensorflow/compiler/mlir/lite/utils/perception_ops_utils.h"
#include "tensorflow/compiler/mlir/lite/utils/tftext_utils.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_attributes.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"

// The cmd line flag to turn on/off Tf.Text API fusion.
// NOLINTNEXTLINE
static llvm::cl::opt<bool> fuse_tftext_flag(
    "tfl-fuse-tftext", llvm::cl::value_desc("bool"),
    llvm::cl::desc("Fuse TF.Text API ops when it's true"),
    llvm::cl::init(false));

namespace mlir {
namespace TFL {
namespace {

constexpr char kTFAPIImplements[] = "tf.api_implements";
constexpr char kTFTextAPIPrefix[] = "tftext:";
constexpr char kCustomSSDPostprocessing[] = "TFLite_Detection_PostProcess";
constexpr char kTfNMSPadded[] = "non_max_suppression_padded_v2";
constexpr char kCustomMaxUnpooling[] = "addons:MaxUnpooling2D";
constexpr char kCustomDenseImageWarp[] = "addons:DenseImageWarp";
constexpr char kTFLFusableOp[] = "tfl_fusable_op";

using mlir::TF::FuncAttr;

inline OpaqueElementsAttr CustomOption(OpBuilder* builder,
                                       const std::string& content) {
   std::vector<std::string> mht_0_v;
   mht_0_v.push_back("content: \"" + content + "\"");
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePStransformsPSprepare_composite_functions_tfDTcc mht_0(mht_0_v, 242, "", "./tensorflow/compiler/mlir/lite/transforms/prepare_composite_functions_tf.cc", "CustomOption");

  ShapedType type = RankedTensorType::get(
      {static_cast<int64_t>(content.size())}, builder->getIntegerType(8));
  return OpaqueElementsAttr::get(builder->getContext()->getLoadedDialect("tfl"),
                                 type,
                                 StringRef(content.data(), content.size()));
}

LogicalResult CreateTflFusableOpCustomOptions(
    ArrayRef<std::pair<StringRef, Attribute>> attrs, OpBuilder* builder,
    std::string& custom_option_buffer) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePStransformsPSprepare_composite_functions_tfDTcc mht_1(mht_1_v, 255, "", "./tensorflow/compiler/mlir/lite/transforms/prepare_composite_functions_tf.cc", "CreateTflFusableOpCustomOptions");

  // There is something worth noting in the ordering of the custom op option:
  // At the MLIR level, all the option is ordered alphabetcially, so there is
  // no way for us to retrieve the original order, so please make sure you are
  // reading custom option from dictionary rather than depending on the order.
  flexbuffers::Builder fbb;
  size_t start_map = fbb.StartMap();

  for (auto attr : attrs) {
    if (auto float_attr = attr.second.dyn_cast_or_null<FloatAttr>()) {
      fbb.Float(attr.first.data(), float_attr.getValue().convertToFloat());
    } else if (auto int_attr = attr.second.dyn_cast_or_null<IntegerAttr>()) {
      fbb.Int(attr.first.data(), int_attr.getInt());
    } else if (auto bool_attr = attr.second.dyn_cast_or_null<BoolAttr>()) {
      fbb.Bool(attr.first.data(), bool_attr.getValue());
    } else {
      // TODO(b/201482289): support other data types.
      return failure();
    }
  }

  fbb.EndMap(start_map);
  fbb.Finish();
  custom_option_buffer.assign(fbb.GetBuffer().begin(), fbb.GetBuffer().end());
  return success();
}

// Convert func annotated with `tfl_fusable_op` attribute to tfl custom op.
LogicalResult ConvertTflFusableOp(
    FuncOp func, StringRef custom_op_name,
    ArrayRef<std::pair<StringRef, Attribute>> attrs) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePStransformsPSprepare_composite_functions_tfDTcc mht_2(mht_2_v, 288, "", "./tensorflow/compiler/mlir/lite/transforms/prepare_composite_functions_tf.cc", "ConvertTflFusableOp");

  func.eraseBody();
  func.addEntryBlock();

  OpBuilder builder(func.getBody());
  std::string custom_option_buffer;
  if (failed(CreateTflFusableOpCustomOptions(attrs, &builder,
                                             custom_option_buffer))) {
    return failure();
  }

  auto tfl_fusable_op = builder.create<TFL::CustomOp>(
      func->getLoc(), func.getFunctionType().getResults(), func.getArguments(),
      custom_op_name, CustomOption(&builder, custom_option_buffer));
  builder.create<func::ReturnOp>(func->getLoc(), tfl_fusable_op.getResults());
  return success();
}

// Abstracts the conversion of the embedded lookup composite function.
class ConvertEmbeddedLookupFunc {
 public:
  explicit ConvertEmbeddedLookupFunc(FuncOp func) : func_(func) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePStransformsPSprepare_composite_functions_tfDTcc mht_3(mht_3_v, 312, "", "./tensorflow/compiler/mlir/lite/transforms/prepare_composite_functions_tf.cc", "ConvertEmbeddedLookupFunc");
}

  void RewriteFunc() {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePStransformsPSprepare_composite_functions_tfDTcc mht_4(mht_4_v, 317, "", "./tensorflow/compiler/mlir/lite/transforms/prepare_composite_functions_tf.cc", "RewriteFunc");

    func_->setAttr(kTFImplements,
                   StringAttr::get(func_.getContext(), "embedding_lookup"));
    Value lookup = func_.getArgument(1);
    Value value = func_.getArgument(0);
    auto output_type = func_.getFunctionType().getResult(0);

    OpBuilder builder(func_.getBody());
    auto op = builder.create<mlir::TFL::EmbeddingLookupOp>(
        func_.getLoc(), output_type, lookup, value);

    builder.create<mlir::func::ReturnOp>(func_.getLoc(), op.getResult());
  }

  LogicalResult VerifySignature() {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePStransformsPSprepare_composite_functions_tfDTcc mht_5(mht_5_v, 334, "", "./tensorflow/compiler/mlir/lite/transforms/prepare_composite_functions_tf.cc", "VerifySignature");

    if (func_.getNumArguments() != 2) {
      return func_.emitWarning()
             << "Invalid number of arguments in the embedding "
                "matmul composite function";
    }
    if (func_.getFunctionType().getNumResults() != 1) {
      return func_.emitWarning() << "Invalid number of results in the "
                                    "embedding matmul composite function";
    }
    return success();
  }

 private:
  FuncOp func_;
};

// This pass uses mechanisms listed in RFC:
// https://github.com/tensorflow/community/pull/113
// It prepares composite functions that are attributed to indicate
// a specific interface (LSTM, SVDF, Embedding lookup etc.) by replacing the
// body with the corresponding fused TFLite op. The replacement need not always
// be a fused op, though that is the primary use case.
class PrepareCompositeFunctionsPass
    : public PassWrapper<PrepareCompositeFunctionsPass,
                         OperationPass<ModuleOp>> {
  void getDependentDialects(DialectRegistry& registry) const override {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePStransformsPSprepare_composite_functions_tfDTcc mht_6(mht_6_v, 363, "", "./tensorflow/compiler/mlir/lite/transforms/prepare_composite_functions_tf.cc", "getDependentDialects");

    registry.insert<TFL::TensorFlowLiteDialect>();
  }

 public:
  explicit PrepareCompositeFunctionsPass() {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePStransformsPSprepare_composite_functions_tfDTcc mht_7(mht_7_v, 371, "", "./tensorflow/compiler/mlir/lite/transforms/prepare_composite_functions_tf.cc", "PrepareCompositeFunctionsPass");
}

  StringRef getArgument() const final {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePStransformsPSprepare_composite_functions_tfDTcc mht_8(mht_8_v, 376, "", "./tensorflow/compiler/mlir/lite/transforms/prepare_composite_functions_tf.cc", "getArgument");

    // This is the argument used to refer to the pass in
    // the textual format (on the commandline for example).
    return "tfl-prepare-composite-funcs-tf";
  }
  StringRef getDescription() const final {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePStransformsPSprepare_composite_functions_tfDTcc mht_9(mht_9_v, 384, "", "./tensorflow/compiler/mlir/lite/transforms/prepare_composite_functions_tf.cc", "getDescription");

    // This is a brief description of the pass.
    return "Prepares composite functions in Tensorflow dialect of MLIR";
  }

 private:
  // TODO(b/160915525): Consolidate FuncAttr and StringAttr into one.
  void ConvertTFImplements(FuncOp func, StringAttr attr);
  void ConvertTFImplementsWithAttributes(FuncOp func, FuncAttr attr);
  void ConvertTFAPIImplements(FuncOp func, StringAttr attr, ModuleOp module);
  void runOnOperation() override;
};

LogicalResult CheckFusableLayerNormalizedLstmCellSimple(FuncOp lstm_func) {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePStransformsPSprepare_composite_functions_tfDTcc mht_10(mht_10_v, 400, "", "./tensorflow/compiler/mlir/lite/transforms/prepare_composite_functions_tf.cc", "CheckFusableLayerNormalizedLstmCellSimple");

  for (int i = 0; i < 5; ++i) {
    auto input = lstm_func.getArgument(i);
    auto input_type = input.getType().dyn_cast_or_null<RankedTensorType>();
    if (!input_type) {
      lstm_func.emitWarning(
          "we cannot fuse this lstm func because all the inputs have not "
          "ranked tensor type.");
      return failure();
    }
  }

  return success();
}

LogicalResult CheckFusableLstmCellSimple(FuncOp lstm_func) {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePStransformsPSprepare_composite_functions_tfDTcc mht_11(mht_11_v, 418, "", "./tensorflow/compiler/mlir/lite/transforms/prepare_composite_functions_tf.cc", "CheckFusableLstmCellSimple");

  for (int i = 0; i < 4; ++i) {
    auto input = lstm_func.getArgument(i);
    auto input_type = input.getType().dyn_cast_or_null<RankedTensorType>();
    if (!input_type) {
      lstm_func.emitWarning(
          "we cannot fuse this lstm func because all the inputs have not "
          "ranked tensor type.");
      return failure();
    }
  }

  return success();
}

LogicalResult CheckOutputConsumer(
    Operation* call_op, int expected_num_outputs,
    llvm::DenseSet<int> expected_consumer_indices) {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePStransformsPSprepare_composite_functions_tfDTcc mht_12(mht_12_v, 438, "", "./tensorflow/compiler/mlir/lite/transforms/prepare_composite_functions_tf.cc", "CheckOutputConsumer");

  const int num_results = call_op->getNumResults();
  if (num_results != expected_num_outputs) return failure();

  for (int i = 0; i < expected_num_outputs; ++i) {
    auto it = expected_consumer_indices.find(i);
    if (it == expected_consumer_indices.end()) {
      // Unexpected consumer.
      if (!call_op->getResult(i).use_empty()) return failure();
    }
  }
  return success();
}

LogicalResult CheckFusableKerasLstm(FuncOp lstm_func, ModuleOp module) {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePStransformsPSprepare_composite_functions_tfDTcc mht_13(mht_13_v, 455, "", "./tensorflow/compiler/mlir/lite/transforms/prepare_composite_functions_tf.cc", "CheckFusableKerasLstm");

  for (auto func : module.getOps<FuncOp>()) {
    if (func == lstm_func) continue;
    auto result = func.walk([&](CallOpInterface op) {
      if (dyn_cast<FuncOp>(op.resolveCallable()) == lstm_func) {
        // Keras LSTM have 5 outputs.
        // We should make sure only the first or the second output are
        // consumed.
        if (failed(CheckOutputConsumer(op.getOperation(), 5, {0, 1})))
          return WalkResult::interrupt();
      }
      return WalkResult::advance();
    });

    if (result.wasInterrupted()) return failure();
  }
  // Current UnidirectionalSequenceLSTMOp doesn't support mask input.
  if (lstm_func.getNumArguments() == 7) return failure();

  // We should know the batch size in advance for the lstm fusion.
  // A good indicator of batch size is both cell state and input state (indices
  // 1 & 2) have fixed shape and other input tenors should have ranked tensor
  // types.
  for (int i = 0; i < 6; ++i) {
    auto input = lstm_func.getArgument(i);
    auto input_type = input.getType().dyn_cast_or_null<RankedTensorType>();
    if (!input_type) {
      lstm_func.emitWarning(
          "we cannot fuse this lstm func because all the inputs have not "
          "ranked tensor type.");
      return failure();
    }
    switch (i) {
      case 1:  // output_init_state
      case 2:  // hidden_init_state
        if (!input_type.hasStaticShape()) {
          lstm_func.emitWarning(
              "we cannot fuse this lstm func because the batch size is not "
              "fixed, please consider setting fixed batch size like "
              "https://github.com/tensorflow/tensorflow/blob/master/tensorflow/"
              "lite/examples/experimental_new_converter/"
              "Keras_LSTM_fusion_Codelab.ipynb");
          return failure();
        }
        break;
      case 3:  // wiehgt
      case 4:  // recurrent_kernel
      case 5:  // bias
        if (!input_type.hasStaticShape()) {
          lstm_func.emitWarning(
              "we cannot fuse this lstm func because the weight & bias are not "
              "fixed, please consider setting fixed batch size like "
              "https://github.com/tensorflow/tensorflow/blob/master/tensorflow/"
              "lite/examples/experimental_new_converter/"
              "Keras_LSTM_fusion_Codelab.ipynb");
          return failure();
        }
        break;
      default:
        // No op.
        break;
    }
  }

  return success();
}

void PrepareCompositeFunctionsPass::ConvertTFImplements(FuncOp func,
                                                        StringAttr attr) {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePStransformsPSprepare_composite_functions_tfDTcc mht_14(mht_14_v, 526, "", "./tensorflow/compiler/mlir/lite/transforms/prepare_composite_functions_tf.cc", "PrepareCompositeFunctionsPass::ConvertTFImplements");

  if (attr.getValue() == "embedding_matmul") {
    // Convert the composite embedding_matmul function body to a
    // TFLite fused embedding_lookup op.
    ConvertEmbeddedLookupFunc convert_embedded_lookup(func);
    if (failed(convert_embedded_lookup.VerifySignature())) return;
    func.eraseBody();
    func.addEntryBlock();
    convert_embedded_lookup.RewriteFunc();
  } else if (attr.getValue() == mlir::TFL::kLstmCellSimple) {
    // Check if the lstm cell simple can be fused, if not, we just don't do
    // anything.
    if (failed(CheckFusableLstmCellSimple(func))) return;
    func.eraseBody();
    func.addEntryBlock();
    ConvertLSTMCellSimpleToFusedLSTM convert_lstm_cell_simple(func);
    if (failed(convert_lstm_cell_simple.RewriteFunc())) {
      return signalPassFailure();
    }
  } else if (attr.getValue() == mlir::TFL::kLayerNormalizedLstmCellSimple) {
    // Check if the layer normalized lstm cell simple can be fused, if not, we
    // just don't do anything.
    if (failed(CheckFusableLayerNormalizedLstmCellSimple(func))) return;
    func.eraseBody();
    func.addEntryBlock();
    ConvertLayerNormalizedLSTMCellSimpleToFusedLSTM
        convert_layer_norm_lstm_cell_simple(func);
    if (failed(convert_layer_norm_lstm_cell_simple.RewriteFunc())) {
      return signalPassFailure();
    }
  } else if (attr.getValue() == kTfNMSPadded) {
    ConvertNMSPaddedFunc convert_nms_padded(func);
    if (failed(convert_nms_padded.VerifySignature())) return;
    func.eraseBody();
    func.addEntryBlock();
    convert_nms_padded.RewriteFunc();
  } else if (attr.getValue() == kCustomDenseImageWarp) {
    ConvertDenseImageWarpFunc image_warping(func);
    if (failed(image_warping.VerifySignature())) return;
    if (failed(image_warping.RewriteFunc())) {
      return signalPassFailure();
    }
  }
}

void PrepareCompositeFunctionsPass::ConvertTFImplementsWithAttributes(
    FuncOp func, FuncAttr attr) {
   std::vector<std::string> mht_15_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePStransformsPSprepare_composite_functions_tfDTcc mht_15(mht_15_v, 575, "", "./tensorflow/compiler/mlir/lite/transforms/prepare_composite_functions_tf.cc", "PrepareCompositeFunctionsPass::ConvertTFImplementsWithAttributes");

  StringRef api_name = attr.getName().getLeafReference().getValue();
  bool enable_fuse_tftext =
      fuse_tftext_flag || IsTFTextRegistered(tensorflow::OpRegistry::Global());
  if (api_name.startswith(kTFTextAPIPrefix) && enable_fuse_tftext) {
    if (failed(ConvertTFTextAPI(func, api_name, attr))) {
      return signalPassFailure();
    }
  } else if (api_name == kCustomSSDPostprocessing) {
    ConvertSSDPostProcessFunc convert_ssd_postprocess(func, attr);
    if (failed(convert_ssd_postprocess.VerifySignature())) return;
    if (failed(convert_ssd_postprocess.RewriteFunc())) {
      return signalPassFailure();
    }
  } else if (api_name == kCustomMaxUnpooling) {
    ConvertMaxUnpoolingFunc max_unpooling(func, attr);
    if (failed(max_unpooling.VerifySignature())) return;
    if (failed(max_unpooling.RewriteFunc())) {
      return signalPassFailure();
    }
  } else {
    // We will look for the `tfl_fusable_op` attribute and fuse as a custom op.
    DictionaryAttr dict_attr = attr.getAttrs();

    SmallVector<std::pair<StringRef, Attribute>, 4> attributes;
    bool tfl_fusable_op = false;
    for (auto attr_item : dict_attr) {
      // Push other attributes except the TFLFusableOp.
      if (attr_item.getName() == kTFLFusableOp &&
          attr_item.getValue().dyn_cast<BoolAttr>().getValue()) {
        tfl_fusable_op = true;
      } else {
        attributes.push_back({attr_item.getName(), attr_item.getValue()});
      }
    }

    if (!tfl_fusable_op) return;

    if (failed(ConvertTflFusableOp(func, api_name, attributes))) {
      func->emitError(absl::StrCat("failed to fuse for op: ", api_name.str()));
      return signalPassFailure();
    }
  }
}

void PrepareCompositeFunctionsPass::ConvertTFAPIImplements(FuncOp func,
                                                           StringAttr attr,
                                                           ModuleOp module) {
   std::vector<std::string> mht_16_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePStransformsPSprepare_composite_functions_tfDTcc mht_16(mht_16_v, 625, "", "./tensorflow/compiler/mlir/lite/transforms/prepare_composite_functions_tf.cc", "PrepareCompositeFunctionsPass::ConvertTFAPIImplements");

  // Keras lstm tf.api_implements usually has attribute like "lstm_abcde91...".
  // TODO(b/147436982): we need to make sure that only the
  // outputs(full sequence) is used, not the last_output, not the new_states.
  // We will discard everything except the outputs.
  // And the outputs is in the shape of [batch, time, units].
  if (attr.getValue().startswith("lstm_")) {
    // Check if the keras lstm can be fused, if not, we just don't do anything.
    if (failed(CheckFusableKerasLstm(func, module))) return;
    func.eraseBody();
    func.addEntryBlock();
    OpBuilder builder(func.getBody());
    if (failed(ConvertKerasLSTMLayer(func, &builder)))
      return signalPassFailure();
  }
}

void PrepareCompositeFunctionsPass::runOnOperation() {
   std::vector<std::string> mht_17_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePStransformsPSprepare_composite_functions_tfDTcc mht_17(mht_17_v, 645, "", "./tensorflow/compiler/mlir/lite/transforms/prepare_composite_functions_tf.cc", "PrepareCompositeFunctionsPass::runOnOperation");

  auto module = getOperation();
  for (auto func : module.getOps<FuncOp>()) {
    // We have three kinds of implements:
    // 1) tf._implements, with string attributes.
    // 2) tf._implements, with proto attributes.
    // 3) tf.api_implements.
    // We need to handle them separately.
    auto tf_implements_attr_str =
        func->getAttrOfType<StringAttr>(kTFImplements);
    if (tf_implements_attr_str) {
      ConvertTFImplements(func, tf_implements_attr_str);
      continue;
    }

    auto tf_implements_attr = func->getAttrOfType<FuncAttr>(kTFImplements);
    if (tf_implements_attr) {
      ConvertTFImplementsWithAttributes(func, tf_implements_attr);
      continue;
    }

    auto tf_api_implements_attr =
        func->getAttrOfType<StringAttr>(kTFAPIImplements);
    if (tf_api_implements_attr) {
      // TODO(b/147536816): Keras lstm should set up the correct attributes.
      ConvertTFAPIImplements(func, tf_api_implements_attr, module);
    }
  }
}
}  // namespace

std::unique_ptr<OperationPass<ModuleOp>> CreatePrepareCompositeFunctionsPass() {
  return std::make_unique<PrepareCompositeFunctionsPass>();
}

static PassRegistration<PrepareCompositeFunctionsPass> pass;

}  // namespace TFL
}  // namespace mlir
