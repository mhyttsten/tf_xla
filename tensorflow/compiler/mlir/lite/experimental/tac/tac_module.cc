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
class MHTracer_DTPStensorflowPScompilerPSmlirPSlitePSexperimentalPStacPStac_moduleDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePSexperimentalPStacPStac_moduleDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSmlirPSlitePSexperimentalPStacPStac_moduleDTcc() {
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
#include "tensorflow/compiler/mlir/lite/experimental/tac/tac_module.h"

#include <memory>
#include <string>

#include "absl/status/status.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/Transforms/Passes.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/lite/experimental/tac/hardwares/target_hardware.h"
#include "tensorflow/compiler/mlir/lite/experimental/tac/transforms/passes.h"
#include "tensorflow/compiler/mlir/lite/transforms/passes.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/error_util.h"

namespace mlir {
namespace TFL {
namespace tac {
namespace {
// TODO(b/177376459): We should make this configureable.
void AddExportTFLPass(mlir::OpPassManager* pass_manager, bool enable_inliner) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePSexperimentalPStacPStac_moduleDTcc mht_0(mht_0_v, 202, "", "./tensorflow/compiler/mlir/lite/experimental/tac/tac_module.cc", "AddExportTFLPass");

  if (enable_inliner) pass_manager->addPass(mlir::createInlinerPass());
  pass_manager->addPass(mlir::createSymbolDCEPass());
  pass_manager->addNestedPass<mlir::func::FuncOp>(
      mlir::createCanonicalizerPass());
  pass_manager->addNestedPass<mlir::func::FuncOp>(mlir::createCSEPass());
}
}  // namespace

// TODO(b/177376459): We should make this configureable.
void TacModule::AddTACPass(mlir::OpPassManager* pass_manager,
                           llvm::ArrayRef<std::string> device_specs) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePSexperimentalPStacPStac_moduleDTcc mht_1(mht_1_v, 216, "", "./tensorflow/compiler/mlir/lite/experimental/tac/tac_module.cc", "TacModule::AddTACPass");

  pass_manager->addPass(mlir::TFL::tac::CreateTargetAnnotationPass(this));
  pass_manager->addPass(mlir::TFL::tac::CreateRaiseTargetSubgraphsPass());
  pass_manager->addPass(mlir::TFL::tac::CreateFoldConstantsToSubgraphPass(
      /*fold_all_constants=*/false));
  pass_manager->addPass(
      mlir::TFL::tac::CreateAlternativeSubgraphPass(device_specs));
  if (options_.legalize_to_tflite_ops) {
    // After we creat the alternative subgraph, we can still do canonicalization
    // legalization & other optimizations as long as we're not inlining the
    // function.
    // And in fact, we probably need to do the proper legalization, for the
    // compute cost to work. (in case we added some TF ops)
    pass_manager->addPass(mlir::TFL::CreatePrepareTFPass(
        /*unfold_batch_matmul=*/true,
        /*allow_bf16_and_f16_type_legalization=*/false));
    pass_manager->addNestedPass<mlir::func::FuncOp>(
        mlir::createCanonicalizerPass());
    pass_manager->addPass(
        mlir::TFL::CreateLegalizeTFPass(/*run_tfl_runtime_verification=*/true));
    pass_manager->addPass(
        mlir::TFL::CreateOptimizePass(/*enable_canonicalization=*/true));
  }

  pass_manager->addPass(mlir::TFL::tac::CreateComputeCostPass());
  pass_manager->addPass(mlir::TFL::tac::CreatePickSubgraphsPass());
  // After this pass, we may consider add a pass to merge small functions into
  // large functions (and maybe other metadata as well).
}

const tac::TargetHardware* TacModule::GetTargetHardware(
    const std::string& hardware_name) const {
   std::vector<std::string> mht_2_v;
   mht_2_v.push_back("hardware_name: \"" + hardware_name + "\"");
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePSexperimentalPStacPStac_moduleDTcc mht_2(mht_2_v, 251, "", "./tensorflow/compiler/mlir/lite/experimental/tac/tac_module.cc", "TacModule::GetTargetHardware");

  for (auto& hardware : backends_) {
    if (GetHardwareName(hardware.get()) == hardware_name) return hardware.get();
  }
  return nullptr;
}

absl::Status TacModule::RunTacPasses(mlir::ModuleOp* module, bool debug_mode) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePSexperimentalPStacPStac_moduleDTcc mht_3(mht_3_v, 261, "", "./tensorflow/compiler/mlir/lite/experimental/tac/tac_module.cc", "TacModule::RunTacPasses");

  mlir::PassManager pm(module->getContext(),
                       mlir::OpPassManager::Nesting::Implicit);
  AddTACPass(&pm, options_.hardware_backends);
  if (!debug_mode) {
    AddExportTFLPass(&pm, options_.enable_inliner);
  }

  mlir::StatusScopedDiagnosticHandler statusHandler(module->getContext(),
                                                    /*propagate=*/true);
  if (failed(pm.run(*module))) {
    return absl::InternalError("conversion error");
  }
  return absl::OkStatus();
}

std::vector<std::unique_ptr<tac::TargetHardware>>
TacModule::InstantiateBackends() {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePSexperimentalPStacPStac_moduleDTcc mht_4(mht_4_v, 281, "", "./tensorflow/compiler/mlir/lite/experimental/tac/tac_module.cc", "TacModule::InstantiateBackends");

  std::vector<std::unique_ptr<tac::TargetHardware>> backends;
  for (const auto& hardware_name : options_.hardware_backends) {
    auto factory = tac::GetTargetHardwareFactory(hardware_name);
    backends.emplace_back(factory());
    backends.back()->Init();
  }
  return backends;
}

absl::Status TacModule::Run() {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePSexperimentalPStacPStac_moduleDTcc mht_5(mht_5_v, 294, "", "./tensorflow/compiler/mlir/lite/experimental/tac/tac_module.cc", "TacModule::Run");

  // Construct all backends.
  backends_ = InstantiateBackends();
  const_backends_.resize(backends_.size());
  for (const auto& backend : backends_)
    const_backends_.emplace_back(backend.get());

  if (!importer_) {
    return absl::Status(absl::StatusCode::kFailedPrecondition,
                        "Null Importer provided");
  }
  if (!exporter_) {
    return absl::Status(absl::StatusCode::kFailedPrecondition,
                        "Null Exporter provided");
  }

  auto module_status = importer_->Import();
  if (!module_status.ok()) {
    return module_status.status();
  }
  auto module = module_status->get();
  auto* context = module->getContext();
  context->appendDialectRegistry(registry_);
  context->loadAllAvailableDialects();

  // Run TAC passes.
  auto status = RunTacPasses(&module, options_.debug_mode);

  if (!status.ok()) {
    return status;
  }

  return exporter_->Export(module);
}

void TacModule::RegisterExtraDialects(mlir::DialectRegistry& registry) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePSexperimentalPStacPStac_moduleDTcc mht_6(mht_6_v, 332, "", "./tensorflow/compiler/mlir/lite/experimental/tac/tac_module.cc", "TacModule::RegisterExtraDialects");

  registry.appendTo(registry_);
}
}  // namespace tac
}  // namespace TFL
}  // namespace mlir
