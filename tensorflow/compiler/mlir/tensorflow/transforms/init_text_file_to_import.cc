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
class MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPStransformsPSinit_text_file_to_importDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPStransformsPSinit_text_file_to_importDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPStransformsPSinit_text_file_to_importDTcc() {
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

#include <numeric>

#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/MemoryBuffer.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/IR/OperationSupport.h"  // from @llvm-project
#include "mlir/IR/PatternMatch.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Support/FileUtilities.h"  // from @llvm-project
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "tensorflow/compiler/mlir/tensorflow/transforms/passes_detail.h"
#include "tensorflow/core/lib/io/path.h"

namespace mlir {
namespace TF {
namespace {

static constexpr int kTextFileIndex_WholeLine = -2;
static constexpr int kTextFileIndex_LineNumber = -1;

// InitTextFileToImportPass converts InitializeTableFromTextFileV2Op to the
// corresponding LookupTableImportV2Op if possible.
class InitTextFileToImportPass
    : public InitTextFileToImportPassBase<InitTextFileToImportPass> {
 public:
  InitTextFileToImportPass() {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPStransformsPSinit_text_file_to_importDTcc mht_0(mht_0_v, 213, "", "./tensorflow/compiler/mlir/tensorflow/transforms/init_text_file_to_import.cc", "InitTextFileToImportPass");
}
  InitTextFileToImportPass(const InitTextFileToImportPass&) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPStransformsPSinit_text_file_to_importDTcc mht_1(mht_1_v, 217, "", "./tensorflow/compiler/mlir/tensorflow/transforms/init_text_file_to_import.cc", "InitTextFileToImportPass");
}
  explicit InitTextFileToImportPass(std::string saved_model_dir) {
   std::vector<std::string> mht_2_v;
   mht_2_v.push_back("saved_model_dir: \"" + saved_model_dir + "\"");
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPStransformsPSinit_text_file_to_importDTcc mht_2(mht_2_v, 222, "", "./tensorflow/compiler/mlir/tensorflow/transforms/init_text_file_to_import.cc", "InitTextFileToImportPass");

    saved_model_dir_ = saved_model_dir;
  }

 private:
  void runOnOperation() override;
};

class ConvertInitializeTableFromTextFileV2
    : public OpRewritePattern<InitializeTableFromTextFileV2Op> {
 public:
  explicit ConvertInitializeTableFromTextFileV2(mlir::MLIRContext* context,
                                                StringRef saved_model_dir)
      : OpRewritePattern<InitializeTableFromTextFileV2Op>(context),
        saved_model_dir_(saved_model_dir) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPStransformsPSinit_text_file_to_importDTcc mht_3(mht_3_v, 239, "", "./tensorflow/compiler/mlir/tensorflow/transforms/init_text_file_to_import.cc", "ConvertInitializeTableFromTextFileV2");
}

  LogicalResult matchAndRewrite(InitializeTableFromTextFileV2Op op,
                                PatternRewriter& rewriter) const override {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPStransformsPSinit_text_file_to_importDTcc mht_4(mht_4_v, 245, "", "./tensorflow/compiler/mlir/tensorflow/transforms/init_text_file_to_import.cc", "matchAndRewrite");

    // Now, this pattern matching only supports the following case, which is
    // commonly used among inference use cases:
    //
    // tf.lookup.TextFileInitializer(
    //   "test.txt", tf.string, tf.lookup.TextFileIndex.WHOLE_LINE,
    //   tf.int64, tf.lookup.TextFileIndex.LINE_NUMBER, delimiter=" ")
    //
    // In the above case, the delimiter will be not used since the key is just a
    // whole line and value is a line number.
    if (op.key_index() != kTextFileIndex_WholeLine ||
        op.value_index() != kTextFileIndex_LineNumber) {
      return failure();
    }

    // Try to find filename from constant op.
    DenseStringElementsAttr filename_attr;
    if (!matchPattern(op.filename().getDefiningOp(),
                      m_Constant(&filename_attr))) {
      return failure();
    }

    if (filename_attr.getRawStringData().size() != 1) {
      return failure();
    }
    std::string filename = filename_attr.getRawStringData()[0].str();

    if (!saved_model_dir_.empty()) {
      filename = tensorflow::io::JoinPath(
          saved_model_dir_.str(),
          tensorflow::io::JoinPath("assets",
                                   tensorflow::io::Basename(filename)));
    }

    // Read the content of the file.
    std::string error_message;
    auto file = openInputFile(filename, &error_message);
    if (!file) {
      return op.emitOpError("failed to open vocabulary file")
             << " (" << filename << "): " << error_message;
    }

    // Splits into lines.
    SmallVector<StringRef, 8> lines;
    file->getBuffer().split(lines, "\n", -1, false);
    // The resize method is used since split operator puts tail value in the end
    // without splitting the leftovers.
    if (op.vocab_size() != -1) lines.resize(op.vocab_size());

    // Map each line to line number, starting from zero.
    SmallVector<int64_t, 8> line_nums;
    line_nums.resize(lines.size());
    std::iota(line_nums.begin(), line_nums.end(), 0);

    // Create constant ops for keys an values.
    Value key_constant_tensor = rewriter.create<arith::ConstantOp>(
        op.getLoc(),
        DenseStringElementsAttr::get(
            RankedTensorType::get(static_cast<int64_t>(lines.size()),
                                  StringType::get(rewriter.getContext())),
            lines));

    Value value_constant_tensor = rewriter.create<arith::ConstantOp>(
        op.getLoc(), rewriter.getI64TensorAttr(line_nums));

    // Replace the given op with LookupTableImportV2Op.
    rewriter.create<LookupTableImportV2Op>(op.getLoc(), op.table_handle(),
                                           key_constant_tensor,
                                           value_constant_tensor);
    rewriter.eraseOp(op);
    return success();
  }

 private:
  StringRef saved_model_dir_;
};

void InitTextFileToImportPass::runOnOperation() {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPStransformsPSinit_text_file_to_importDTcc mht_5(mht_5_v, 325, "", "./tensorflow/compiler/mlir/tensorflow/transforms/init_text_file_to_import.cc", "InitTextFileToImportPass::runOnOperation");

  RewritePatternSet patterns(&getContext());
  MLIRContext* context = &getContext();
  FuncOp func = getOperation();

  patterns.add<ConvertInitializeTableFromTextFileV2>(
      context, StringRef(saved_model_dir_));
  (void)applyPatternsAndFoldGreedily(func, std::move(patterns));
}

}  // namespace

// Replace InitializeTableFromTextFileV2Ops with LookupTableImportV2Ops.
std::unique_ptr<OperationPass<FuncOp>> CreateInitTextFileToImportPass(
    std::string saved_model_dir) {
  return std::make_unique<InitTextFileToImportPass>(saved_model_dir);
}


}  // namespace TF
}  // namespace mlir
