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
class MHTracer_DTPStensorflowPScompilerPSmlirPSlitePSmlir_tflite_runnerDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePSmlir_tflite_runnerDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSmlirPSlitePSmlir_tflite_runnerDTcc() {
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

// Tool to run a TFLite computation from a MLIR input using the TFLite
// interpreter.

#include <stdio.h>

#include <algorithm>
#include <memory>
#include <string>
#include <vector>

#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SMLoc.h"
#include "llvm/Support/SourceMgr.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"  // from @llvm-project
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/Dialect.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/Parser/Parser.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/lite/flatbuffer_export.h"
#include "tensorflow/compiler/mlir/lite/flatbuffer_export_flags.h"
#include "tensorflow/compiler/mlir/lite/ir/tfl_ops.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/lite/delegates/flex/delegate.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model.h"
#include "tensorflow/lite/optional_debug_tools.h"

using llvm::cl::desc;
using llvm::cl::init;
using llvm::cl::opt;

// NOLINTNEXTLINE
static opt<std::string> input_filename(llvm::cl::Positional,
                                       desc("<input file>"), init("-"));

// NOLINTNEXTLINE
static opt<bool> dump_state("dump-interpreter-state",
                            desc("dump interpreter state post execution"),
                            init(false));

// TODO(jpienaar): Move these functions to some debug utils.
static std::string TfLiteTensorDimString(const TfLiteTensor& tensor) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePSmlir_tflite_runnerDTcc mht_0(mht_0_v, 233, "", "./tensorflow/compiler/mlir/lite/mlir_tflite_runner.cc", "TfLiteTensorDimString");

  auto begin = tensor.dims ? tensor.dims->data : nullptr;
  auto end = tensor.dims ? tensor.dims->data + tensor.dims->size : nullptr;
  return absl::StrJoin(begin, end, ", ");
}

template <typename T>
static std::string TfLiteTypedTensorString(const TfLiteTensor& tensor) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePSmlir_tflite_runnerDTcc mht_1(mht_1_v, 243, "", "./tensorflow/compiler/mlir/lite/mlir_tflite_runner.cc", "TfLiteTypedTensorString");

  const T* data = reinterpret_cast<T*>(tensor.data.raw);
  if (!data) return "<null>";
  int count = tensor.bytes / sizeof(T);
  return absl::StrJoin(data, data + count, ", ");
}

// TODO(jpienaar): This really feels like something that should exist already.
static std::string TfLiteTensorString(const TfLiteTensor& tensor) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePSmlir_tflite_runnerDTcc mht_2(mht_2_v, 254, "", "./tensorflow/compiler/mlir/lite/mlir_tflite_runner.cc", "TfLiteTensorString");

  switch (tensor.type) {
    case kTfLiteInt32:
      return TfLiteTypedTensorString<int32_t>(tensor);
    case kTfLiteUInt32:
      return TfLiteTypedTensorString<uint32_t>(tensor);
    case kTfLiteInt64:
      return TfLiteTypedTensorString<int64_t>(tensor);
    case kTfLiteFloat32:
      return TfLiteTypedTensorString<float>(tensor);
    default:
      LOG(QFATAL) << "Unsupported type: " << TfLiteTypeGetName(tensor.type);
  }
}

int main(int argc, char** argv) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePSmlir_tflite_runnerDTcc mht_3(mht_3_v, 272, "", "./tensorflow/compiler/mlir/lite/mlir_tflite_runner.cc", "main");

  llvm::InitLLVM y(argc, argv);
  llvm::cl::ParseCommandLineOptions(argc, argv, "MLIR TFLite runner\n");

  auto file_or_err = llvm::MemoryBuffer::getFileOrSTDIN(input_filename.c_str());
  if (std::error_code error = file_or_err.getError()) {
    LOG(ERROR) << argv[0] << ": could not open input file '" << input_filename
               << "': " << error.message() << "\n";
    return 1;
  }

  // Load the MLIR module.
  mlir::DialectRegistry registry;
  registry.insert<mlir::TF::TensorFlowDialect, mlir::TFL::TensorFlowLiteDialect,
                  mlir::arith::ArithmeticDialect, mlir::func::FuncDialect>();
  mlir::MLIRContext context(registry);

  llvm::SourceMgr source_mgr;
  source_mgr.AddNewSourceBuffer(std::move(*file_or_err), llvm::SMLoc());
  mlir::OwningOpRef<mlir::ModuleOp> module(
      mlir::parseSourceFile<mlir::ModuleOp>(source_mgr, &context));
  if (!module) return 1;

  // TODO(jpienaar): Expand to support inputs.
  mlir::func::FuncOp main = module->lookupSymbol<mlir::func::FuncOp>("main");
  QCHECK(main) << "No 'main' function specified.";
  if (main.getFunctionType().getNumInputs() != 0)
    LOG(QFATAL) << "NYI: Only nullary functions supported.";

  // Convert to flatbuffer.
  std::string serialized_flatbuffer;
  tflite::FlatbufferExportOptions options;
  options.toco_flags.set_force_select_tf_ops(!emit_builtin_tflite_ops);
  options.toco_flags.set_enable_select_tf_ops(emit_select_tf_ops);
  options.toco_flags.set_allow_custom_ops(emit_custom_ops);
  if (!tflite::MlirToFlatBufferTranslateFunction(module.get(), options,
                                                 &serialized_flatbuffer))
    return 1;

  // Create TFLite interpreter & invoke converted program.
  std::unique_ptr<tflite::FlatBufferModel> model =
      tflite::FlatBufferModel::BuildFromBuffer(serialized_flatbuffer.c_str(),
                                               serialized_flatbuffer.size());
  tflite::ops::builtin::BuiltinOpResolver builtins;
  std::unique_ptr<tflite::Interpreter> interpreter;
  QCHECK(tflite::InterpreterBuilder(*model, builtins)(&interpreter) ==
         kTfLiteOk);
  QCHECK(interpreter->AllocateTensors() == kTfLiteOk);
  QCHECK(interpreter->Invoke() == kTfLiteOk);

  // Print the resulting outputs.
  // TODO(jpienaar): Allow specifying output stream/file.
  QCHECK(interpreter->outputs().size() ==
         main.getFunctionType().getNumResults());
  for (int index : interpreter->outputs()) {
    const auto& out = *interpreter->tensor(index);
    // Print name if named.
    if (out.name) fprintf(stdout, "%s: ", out.name);
    // Print tensor result.
    fprintf(stdout, "Tensor<type: %s, shape: %s, values: %s>\n",
            TfLiteTypeGetName(out.type), TfLiteTensorDimString(out).c_str(),
            TfLiteTensorString(out).c_str());
  }

  if (dump_state) tflite::PrintInterpreterState(interpreter.get());

  return 0;
}
