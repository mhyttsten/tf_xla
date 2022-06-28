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
class MHTracer_DTPStensorflowPScompilerPSmlirPSlitePSexperimentalPStacPStflite_import_exportDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePSexperimentalPStacPStflite_import_exportDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSmlirPSlitePSexperimentalPStacPStflite_import_exportDTcc() {
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
#include "tensorflow/compiler/mlir/lite/experimental/tac/tflite_import_export.h"

#include <string>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/ToolOutputFile.h"
#include "mlir/Support/FileUtilities.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/lite/experimental/tac/common/targets.h"
#include "tensorflow/compiler/mlir/lite/experimental/tac/common/utils.h"
#include "tensorflow/compiler/mlir/lite/experimental/tac/execution_metadata_exporter.h"
#include "tensorflow/compiler/mlir/lite/experimental/tac/hardwares/target_hardware.h"
#include "tensorflow/compiler/mlir/lite/experimental/tac/utils/utils.h"

namespace mlir {
namespace TFL {
namespace tac {
namespace {
void AttachCostPerDevice(mlir::ModuleOp module,
                         llvm::ArrayRef<std::string> device_specs) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePSexperimentalPStacPStflite_import_exportDTcc mht_0(mht_0_v, 205, "", "./tensorflow/compiler/mlir/lite/experimental/tac/tflite_import_export.cc", "AttachCostPerDevice");

  std::set<std::string> processed_device_specs;
  for (const auto& device_spec : device_specs) {
    processed_device_specs.insert(
        mlir::TFL::tac::GetCanonicalHardwareName(device_spec));
  }
  processed_device_specs.insert("CPU");

  module.walk([&](mlir::Operation* op) {
    if (!mlir::TFL::tac::IsNonConstOp(op) &&
        !llvm::isa<func::ReturnOp, FuncOp, CallOpInterface>(op))
      return;

    // Attach cost per target.
    // Unsupported op will have negative values.
    mlir::SmallVector<mlir::NamedAttribute, 4> device_costs;
    for (const auto& device : processed_device_specs) {
      auto* target_hardware = mlir::TFL::tac::GetTargetHardware(device);
      float cost = -1;
      if (target_hardware->IsOpSupported(op)) {
        cost = target_hardware->GetOpCost(op);
      }

      mlir::StringAttr device_identifier =
          mlir::StringAttr::get(module.getContext(), device);
      auto float_type = mlir::FloatType::getF32(module.getContext());
      auto float_attr =
          mlir::FloatAttr::get(float_type, static_cast<float>(cost));
      device_costs.push_back({device_identifier, float_attr});
    }

    op->setAttr("per_device_costs",
                mlir::DictionaryAttr::get(module.getContext(), device_costs));
  });
}

}  // namespace

//////////// Importer ////////////
absl::StatusOr<OwningOpRef<mlir::ModuleOp>> TfLiteImporter::Import() {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePSexperimentalPStacPStflite_import_exportDTcc mht_1(mht_1_v, 247, "", "./tensorflow/compiler/mlir/lite/experimental/tac/tflite_import_export.cc", "TfLiteImporter::Import");

  source_mgr_handler_ = std::make_unique<mlir::SourceMgrDiagnosticHandler>(
      source_mgr_, &context_);
  return ImportFlatbufferOrMlir(options_.file_name, options_.input_mlir,
                                &source_mgr_, &context_);
}

//////////// Exporter ////////////
absl::Status TfLiteExporter::Export(mlir::ModuleOp module) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePSexperimentalPStacPStflite_import_exportDTcc mht_2(mht_2_v, 258, "", "./tensorflow/compiler/mlir/lite/experimental/tac/tflite_import_export.cc", "TfLiteExporter::Export");

  // return absl::OkStatus();
  if (options_.export_runtime_metadata) {
    // Run the cost model for each device/op.
    AttachCostPerDevice(module, options_.target_hardware_backends);

    // We will export the runtime metadata with the same name under the same
    // directory except with a different extention ".rtmeta".
    llvm::SmallString<128> metadata_filename(options_.output_file_name);
    const char kRuntimeMetadataName[] = "rtmeta";
    llvm::sys::path::replace_extension(metadata_filename, kRuntimeMetadataName);

    std::string error_msg;
    auto output = mlir::openOutputFile(metadata_filename, &error_msg);
    if (output == nullptr) {
      return absl::InvalidArgumentError(
          absl::StrCat("cannot open output file: ", error_msg));
    }
    auto result = tflite::ExportRuntimeMetadata(module);
    if (!result) {
      return absl::InvalidArgumentError("Cannot export runtime metadata.");
    }
    output->os() << result;
    output->keep();
  }

  return mlir::TFL::tac::ExportFlatbufferOrMlir(options_.output_file_name,
                                                options_.output_mlir, module);
}

}  // namespace tac
}  // namespace TFL
}  // namespace mlir
