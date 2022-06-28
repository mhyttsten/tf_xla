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
class MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPStranslatePStf_mlir_translate_registrationDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPStranslatePStf_mlir_translate_registrationDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPStranslatePStf_mlir_translate_registrationDTcc() {
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

// This file wraps TensorFlow Graph(Def) to MLIR module conversion into passes
// to satisfy the API of MLIR pass registration. In order to do this, the
// command-line option header is pulled in.

#include <memory>

#include "llvm/Support/FileSystem.h"
#include "llvm/Support/MemoryBuffer.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/Tools/mlir-translate/Translation.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/dialect_registration.h"
#include "tensorflow/compiler/mlir/tensorflow/translate/export_graphdef.h"
#include "tensorflow/compiler/mlir/tensorflow/translate/mlir_roundtrip_flags.h"
#include "tensorflow/compiler/mlir/tensorflow/translate/tf_mlir_translate.h"
#include "tensorflow/compiler/mlir/tensorflow/translate/tf_mlir_translate_cl.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/stream_executor/lib/statusor.h"

namespace mlir {
using stream_executor::port::Status;
using stream_executor::port::StatusOr;

namespace {
inline absl::string_view StringRefToView(llvm::StringRef ref) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPStranslatePStf_mlir_translate_registrationDTcc mht_0(mht_0_v, 209, "", "./tensorflow/compiler/mlir/tensorflow/translate/tf_mlir_translate_registration.cc", "StringRefToView");

  return {ref.data(), ref.size()};
}
}  // namespace

static OwningOpRef<mlir::ModuleOp> GraphdefToMlirTranslateFunction(
    llvm::StringRef input, MLIRContext* context) {
  auto module_or = tensorflow::GraphdefToMlirTranslateFunction(
      input, debug_info_file, input_arrays, input_dtypes, input_shapes,
      output_arrays, control_output_arrays, prune_unused_nodes,
      convert_legacy_fed_inputs, graph_as_function, upgrade_legacy,
      enable_shape_inference, unconditionally_use_set_output_shapes, context);
  if (!module_or.status().ok()) return nullptr;
  return module_or.ConsumeValueOrDie();
}

static TranslateToMLIRRegistration GraphdefToMlirTranslate(
    "graphdef-to-mlir", GraphdefToMlirTranslateFunction);

static OwningOpRef<mlir::ModuleOp> GraphdefToSplattedMlirTranslateFunction(
    llvm::StringRef input, MLIRContext* context) {
  auto module_or = tensorflow::GraphdefToSplattedMlirTranslateFunction(
      input, debug_info_file, input_arrays, input_dtypes, input_shapes,
      output_arrays, control_output_arrays, prune_unused_nodes,
      convert_legacy_fed_inputs, graph_as_function, upgrade_legacy,
      enable_shape_inference, unconditionally_use_set_output_shapes, context);
  if (!module_or.status().ok()) return nullptr;
  return module_or.ConsumeValueOrDie();
}

static TranslateToMLIRRegistration GraphdefToSplattedMlirTranslate(
    "graphdef-to-splatted-mlir", GraphdefToSplattedMlirTranslateFunction);

static LogicalResult MlirToGraphdefTranslateFunction(
    ModuleOp module, llvm::raw_ostream& output) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPStranslatePStf_mlir_translate_registrationDTcc mht_1(mht_1_v, 246, "", "./tensorflow/compiler/mlir/tensorflow/translate/tf_mlir_translate_registration.cc", "MlirToGraphdefTranslateFunction");

  if (!module) return failure();

  // TODO(fengliuai): Add exporter flags.
  tensorflow::GraphExportConfig confs;
  confs.export_entry_func_to_flib = export_entry_func_to_flib;
  StatusOr<std::unique_ptr<tensorflow::GraphDef>> graphdef_or(
      tensorflow::ConvertMlirToGraphdef(module, confs));
  if (!graphdef_or.status().ok()) {
    LOG(ERROR) << "Graph export failed: " << graphdef_or.status();
    return mlir::failure();
  }

  output << graphdef_or.ValueOrDie()->DebugString();
  return success();
}

static TranslateFromMLIRRegistration mlir_to_graphdef_translate(
    "mlir-to-graphdef", MlirToGraphdefTranslateFunction,
    [](DialectRegistry& registry) {
      mlir::RegisterAllTensorFlowDialects(registry);
    });

}  // namespace mlir
