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
class MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPStransformsPSinit_text_file_to_import_test_passDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPStransformsPSinit_text_file_to_import_test_passDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPStransformsPSinit_text_file_to_import_test_passDTcc() {
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

#include <memory>

#include "llvm/Support/Casting.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/ToolOutputFile.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/IR/OperationSupport.h"  // from @llvm-project
#include "mlir/IR/PatternMatch.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Pass/PassManager.h"  // from @llvm-project
#include "mlir/Support/FileUtilities.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "tensorflow/compiler/mlir/tensorflow/transforms/passes.h"
#include "tensorflow/compiler/mlir/tensorflow/transforms/test_passes_detail.h"
#include "tensorflow/core/platform/path.h"
#include "tensorflow/core/platform/stringpiece.h"

namespace mlir {
namespace TF {
namespace {

// InitTextFileToImportTestPass generates a temporary file and run the
// InitTextFileToImportPass for testing purpose.
class InitTextFileToImportTestPass
    : public tf_test::InitTextFileToImportTestPassBase<
          InitTextFileToImportTestPass> {
 public:
  explicit InitTextFileToImportTestPass() {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPStransformsPSinit_text_file_to_import_test_passDTcc mht_0(mht_0_v, 213, "", "./tensorflow/compiler/mlir/tensorflow/transforms/init_text_file_to_import_test_pass.cc", "InitTextFileToImportTestPass");
}

  StringRef getArgument() const final {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPStransformsPSinit_text_file_to_import_test_passDTcc mht_1(mht_1_v, 218, "", "./tensorflow/compiler/mlir/tensorflow/transforms/init_text_file_to_import_test_pass.cc", "getArgument");

    return "tf-init-text-file-to-import-test";
  }

  StringRef getDescription() const final {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPStransformsPSinit_text_file_to_import_test_passDTcc mht_2(mht_2_v, 225, "", "./tensorflow/compiler/mlir/tensorflow/transforms/init_text_file_to_import_test_pass.cc", "getDescription");

    return "generate a temporary file and invoke InitTextFileToImportPass";
  }

 private:
  void runOnOperation() override;
};

void InitTextFileToImportTestPass::runOnOperation() {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPStransformsPSinit_text_file_to_import_test_passDTcc mht_3(mht_3_v, 236, "", "./tensorflow/compiler/mlir/tensorflow/transforms/init_text_file_to_import_test_pass.cc", "InitTextFileToImportTestPass::runOnOperation");

  ModuleOp module = getOperation();

  // Create a temporary vocab file.
  int fd;
  SmallString<256> filename;
  std::error_code error_code =
      llvm::sys::fs::createTemporaryFile("text", "vocab", fd, filename);
  if (error_code) return signalPassFailure();

  llvm::ToolOutputFile temp_file(filename, fd);
  temp_file.os() << "apple\n";
  temp_file.os() << "banana\n";
  temp_file.os() << "grape";
  temp_file.os().flush();

  // Replace filename constant ops to use the temporary file.
  MLIRContext* context = &getContext();

  for (FuncOp func : module.getOps<FuncOp>()) {
    llvm::SmallVector<arith::ConstantOp, 4> constant_ops(
        func.getOps<arith::ConstantOp>());
    for (auto op : constant_ops) {
      ShapedType shaped_type =
          RankedTensorType::get({1}, StringType::get(context));

      DenseStringElementsAttr attr;
      if (!matchPattern(op.getOperation(), m_Constant(&attr))) {
        continue;
      }

      ArrayRef<StringRef> values = attr.getRawStringData();
      if (values.size() != 1 || values[0] != "%FILE_PLACEHOLDER") {
        continue;
      }

      op.setValueAttr(DenseStringElementsAttr::get(shaped_type, {filename}));
    }
  }

  // Run the lowering pass.
  PassManager pm(context);
  pm.addNestedPass<FuncOp>(CreateInitTextFileToImportPass(""));
  if (failed(pm.run(module))) return signalPassFailure();
}

// InitTextFileToImportSavedModelTestPass mimicks a temporary saved model and
// run the InitTextFileToImportPass for testing purpose.
class InitTextFileToImportSavedModelTestPass
    : public tf_test::InitTextFileToImportSavedModelTestPassBase<
          InitTextFileToImportSavedModelTestPass> {
 public:
  explicit InitTextFileToImportSavedModelTestPass() {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPStransformsPSinit_text_file_to_import_test_passDTcc mht_4(mht_4_v, 291, "", "./tensorflow/compiler/mlir/tensorflow/transforms/init_text_file_to_import_test_pass.cc", "InitTextFileToImportSavedModelTestPass");
}

 private:
  void runOnOperation() override;
};

void InitTextFileToImportSavedModelTestPass::runOnOperation() {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPStransformsPSinit_text_file_to_import_test_passDTcc mht_5(mht_5_v, 300, "", "./tensorflow/compiler/mlir/tensorflow/transforms/init_text_file_to_import_test_pass.cc", "InitTextFileToImportSavedModelTestPass::runOnOperation");

  ModuleOp module = getOperation();

  // Create a temporary saved model's asset file.
  SmallString<256> tempdir;
  std::error_code error_code =
      llvm::sys::fs::createUniqueDirectory("saved-model", tempdir);
  if (error_code) return signalPassFailure();
  error_code =
      llvm::sys::fs::create_directories(Twine(tempdir) + "/assets", false);
  if (error_code) return signalPassFailure();

  std::string filename = std::string(tempdir) + "/assets/tokens.txt";

  std::string error_message;
  auto temp_file = openOutputFile(filename, &error_message);
  if (!error_message.empty()) return;
  temp_file->os() << "apple\n";
  temp_file->os() << "banana\n";
  temp_file->os() << "grape";
  temp_file->os().flush();

  // Run the lowering pass.
  MLIRContext* context = &getContext();
  PassManager pm(context);
  pm.addNestedPass<FuncOp>(
      CreateInitTextFileToImportPass(std::string(tempdir)));
  if (failed(pm.run(module))) return signalPassFailure();
}

}  // namespace
}  // namespace TF

namespace tf_test {
std::unique_ptr<OperationPass<ModuleOp>> CreateInitTextFileToImportTestPass() {
  return std::make_unique<TF::InitTextFileToImportTestPass>();
}
std::unique_ptr<OperationPass<ModuleOp>>
CreateInitTextFileToImportSavedModelTestPass() {
  return std::make_unique<TF::InitTextFileToImportSavedModelTestPass>();
}
}  // namespace tf_test

}  // namespace mlir
