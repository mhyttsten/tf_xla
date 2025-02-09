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
class MHTracer_DTPStensorflowPScompilerPSmlirPSlitePSflatbuffer_translateDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePSflatbuffer_translateDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSmlirPSlitePSflatbuffer_translateDTcc() {
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

#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/ToolOutputFile.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"  // from @llvm-project
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/Dialect/Quant/QuantOps.h"  // from @llvm-project
#include "mlir/Dialect/Quant/QuantTypes.h"  // from @llvm-project
#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/IR/Location.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/IR/Operation.h"  // from @llvm-project
#include "mlir/IR/Types.h"  // from @llvm-project
#include "mlir/IR/Value.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "mlir/Tools/mlir-translate/Translation.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/lite/flatbuffer_export.h"
#include "tensorflow/compiler/mlir/lite/flatbuffer_import.h"
#include "tensorflow/compiler/mlir/lite/ir/tfl_ops.h"
#include "tensorflow/compiler/mlir/tensorflow/dialect_registration.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "tensorflow/compiler/mlir/tensorflow/translate/mlir_roundtrip_flags.h"

using llvm::cl::opt;

// Commandline flag to enable the control of flatbuffer import.
bool use_external_constant;

// Commandline flag to enable graph pruning.
bool experimental_prune_unreachable_nodes_unconditionally;

// NOLINTNEXTLINE
static opt<bool, true> use_external_constant_flag(
    "use-external-constant",
    llvm::cl::desc("Use external constant during flatbuffer import"),
    llvm::cl::location(use_external_constant), llvm::cl::init(false));

// TODO(b/147111261): After the importer supports generic custom ops, we should
// change the flag to a more lightwise flag, e.g.
// "import_custom_ops_as_side_effect_free_ops", and let the MLIR DCE to prune
// the operations.
// NOLINTNEXTLINE
static opt<bool, true> experimental_prune_unreachable_nodes_unconditionally_flg(
    "experimental-prune-unreachable-nodes-unconditionally",
    llvm::cl::desc("Prune nodes that are not ancestors of the output nodes."),
    llvm::cl::location(experimental_prune_unreachable_nodes_unconditionally),
    llvm::cl::init(false));

// NOLINTNEXTLINE
static opt<std::string> input_arrays_flag(
    "input-arrays",
    llvm::cl::desc(
        "List of input tensors, if different from the default inputs"),
    llvm::cl::init(""));

// NOLINTNEXTLINE
static opt<std::string> output_arrays_flag(
    "output-arrays",
    llvm::cl::desc(
        "List of output tensors, if different from the default outputs"),
    llvm::cl::init(""));
using llvm::cl::opt;

// These command line flags enable control of the translation implementation.
bool emit_builtin_tflite_ops;
bool emit_custom_ops;
bool emit_select_tf_ops;
bool lower_tensor_list_ops;
bool strip_debug_info;

// NOLINTNEXTLINE
static opt<bool, true> emit_builtin_tflite_ops_flag(
    "emit-builtin-tflite-ops",
    llvm::cl::desc(
        "Emit TFLite built in operations in the generated TFLite model"),
    llvm::cl::location(emit_builtin_tflite_ops), llvm::cl::init(true));

// NOLINTNEXTLINE
static opt<bool, true> emit_select_tf_ops_flag(
    "emit-select-tf-ops",
    llvm::cl::desc(
        "Emit Select TF operations (Flex ops) in the generated TFLite model"),
    llvm::cl::location(emit_select_tf_ops), llvm::cl::init(false));

// NOLINTNEXTLINE
static opt<bool, true> emit_custom_ops_flag(
    "emit-custom-ops",
    llvm::cl::desc("Emit Custom operations in the generated TFLite model"),
    llvm::cl::location(emit_custom_ops), llvm::cl::init(false));

// NOLINTNEXTLINE
static opt<bool, true> lower_tensor_list_ops_flag(
    "lower-tensor-list-ops",
    llvm::cl::desc("Lower the TensorList ops within the TFLite dialect"),
    llvm::cl::location(lower_tensor_list_ops), llvm::cl::init(false));

// NOLINTNEXTLINE
static opt<bool, true> strip_debug_info_flag(
    "strip-debug-info", llvm::cl::desc("Strip debug info during export"),
    llvm::cl::location(strip_debug_info), llvm::cl::init(false));

namespace mlir {
namespace {
static OwningOpRef<mlir::ModuleOp> FlatBufferFileToMlirTrans(
    llvm::SourceMgr* source_mgr, MLIRContext* context,
    bool use_external_constant,
    bool experimental_prune_unreachable_nodes_unconditionally) {
  const llvm::MemoryBuffer* input =
      source_mgr->getMemoryBuffer(source_mgr->getMainFileID());
  std::string error;
  auto loc =
      mlir::FileLineColLoc::get(context, input->getBufferIdentifier(), 0, 0);

  // Parses input/output names from command line options.
  std::vector<std::string> inputs;
  std::vector<std::string> outputs;
  // Use output parser since we only have tensor names.
  if (!tensorflow::ParseOutputArrayInfo(input_arrays_flag, &inputs).ok()) {
    return emitError(loc, "parsing input array info failed ")
               << input_arrays_flag,
           nullptr;
  }
  if (!tensorflow::ParseOutputArrayInfo(output_arrays_flag, &outputs).ok()) {
    return emitError(loc, "parsing output array info failed ")
               << output_arrays_flag,
           nullptr;
  }
  return tflite::FlatBufferToMlir(
      absl::string_view(input->getBufferStart(), input->getBufferSize()),
      context, loc, use_external_constant, inputs, outputs,
      experimental_prune_unreachable_nodes_unconditionally);
}

static LogicalResult MlirToFlatBufferFileTranslateFunction(
    ModuleOp module, llvm::raw_ostream& output) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePSflatbuffer_translateDTcc mht_0(mht_0_v, 324, "", "./tensorflow/compiler/mlir/lite/flatbuffer_translate.cc", "MlirToFlatBufferFileTranslateFunction");

  std::string serialized_flatbuffer;
  std::unique_ptr<tensorflow::OpOrArgNameMapper> op_or_arg_name_mapper;
  if (strip_debug_info) {
    op_or_arg_name_mapper =
        std::make_unique<tensorflow::OpOrArgStripNameMapper>();
  } else {
    op_or_arg_name_mapper =
        std::make_unique<tensorflow::OpOrArgLocNameMapper>();
  }
  tflite::FlatbufferExportOptions options;
  options.toco_flags.set_force_select_tf_ops(!emit_builtin_tflite_ops);
  options.toco_flags.set_enable_select_tf_ops(emit_select_tf_ops);
  options.toco_flags.set_allow_custom_ops(emit_custom_ops);
  options.op_or_arg_name_mapper = op_or_arg_name_mapper.get();
  if (!tflite::MlirToFlatBufferTranslateFunction(module, options,
                                                 &serialized_flatbuffer))
    return mlir::failure();

  output << serialized_flatbuffer;
  return success();
}
}  // namespace

static TranslateToMLIRRegistration FlatBufferFileToMlirTransReg(
    "tflite-flatbuffer-to-mlir",
    [](llvm::SourceMgr& source_mgr, MLIRContext* context) {
      return FlatBufferFileToMlirTrans(
          &source_mgr, context, use_external_constant,
          experimental_prune_unreachable_nodes_unconditionally);
    });

static TranslateFromMLIRRegistration MLIRToFlatBufferTranslate(
    "mlir-to-tflite-flatbuffer", MlirToFlatBufferFileTranslateFunction,
    [](DialectRegistry& registry) {
      registry.insert<quant::QuantizationDialect>();
      mlir::RegisterAllTensorFlowDialects(registry);
      registry.insert<TFL::TensorFlowLiteDialect>();
      registry.insert<arith::ArithmeticDialect>();
      registry.insert<func::FuncDialect>();
    });
}  // namespace mlir
