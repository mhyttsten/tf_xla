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
class MHTracer_DTPStensorflowPScompilerPSmlirPStfrtPSruntime_fallbackPSruntime_fallback_combineDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSmlirPStfrtPSruntime_fallbackPSruntime_fallback_combineDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSmlirPStfrtPSruntime_fallbackPSruntime_fallback_combineDTcc() {
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

//===----------------------------------------------------------------------===//
//
// This file implements a set of simple combiners for optimizing operations in
// the Runtime Fallback dialect.
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "tensorflow/compiler/mlir/tfrt/runtime_fallback/runtime_fallback_ops.h"

// This optimizes the following scenario:
// %tft0, %c2 = "tfd.move_dht_to_tft"(%dht0, %c1)
//     : (!dht.host_tensor, !tfrt.chain) -> (!tfd.tf_tensor, !tfrt.chain)
// %dht1, %c3 = "tfd.convert_tft_to_dht"(%tft0, %c2)
//     : (!tfd.tf_tensor, !tfrt.chain) -> (!dht.host_tensor, !tfrt.chain)
// some_op %dht1, %c3
//
// becomes
// some_op %dht0, %c1

struct SimplifyDoubleConversion
    : public mlir::OpRewritePattern<mlir::tfd::ConvertTftToDhtOp> {
  // We register this pattern to match every tfd.move_dht_to_tft op.
  // The "benefit" is used by the framework to order the patterns and process
  // them in order of profitability.
  explicit SimplifyDoubleConversion(mlir::MLIRContext* context)
      : mlir::OpRewritePattern<mlir::tfd::ConvertTftToDhtOp>(context,
                                                             /*benefit=*/1) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStfrtPSruntime_fallbackPSruntime_fallback_combineDTcc mht_0(mht_0_v, 213, "", "./tensorflow/compiler/mlir/tfrt/runtime_fallback/runtime_fallback_combine.cc", "SimplifyDoubleConversion");
}

  // This method attempts to match a pattern and rewrite it. The rewriter
  // argument is the orchestrator of the sequence of rewrites. The pattern is
  // expected to interact with it to perform any changes to the IR from here.
  mlir::LogicalResult matchAndRewrite(
      mlir::tfd::ConvertTftToDhtOp op,
      mlir::PatternRewriter& rewriter) const override {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStfrtPSruntime_fallbackPSruntime_fallback_combineDTcc mht_1(mht_1_v, 223, "", "./tensorflow/compiler/mlir/tfrt/runtime_fallback/runtime_fallback_combine.cc", "matchAndRewrite");

    // Look through the inputs of the ConvertTftToDhtOp.
    mlir::Value convert_op_input_0 = op.getOperand(0);
    mlir::Value convert_op_input_1 = op.getOperand(1);
    mlir::tfd::MoveDhtToTftOp move_input_op_0 =
        llvm::dyn_cast_or_null<mlir::tfd::MoveDhtToTftOp>(
            convert_op_input_0.getDefiningOp());
    mlir::tfd::MoveDhtToTftOp move_input_op_1 =
        llvm::dyn_cast_or_null<mlir::tfd::MoveDhtToTftOp>(
            convert_op_input_1.getDefiningOp());

    // The inputs should be MoveDhtToTftOp.
    if (!move_input_op_0 || !move_input_op_1) return mlir::failure();
    // Both inputs are the same MoveDhtToTftOp.
    if (move_input_op_0 != move_input_op_1) return mlir::failure();

    // Use the rewriter to replace the ConvertTftToDhtOp's users with the
    // operands of MoveDhtToTftOp.
    rewriter.replaceOp(
        op, {move_input_op_0.getOperand(0), move_input_op_0.getOperand(1)});
    return mlir::success();
  }
};

// Register rewrite pattern as "canonicalization" patterns on the MoveDhtToTftOp
// so that they can be picked up by the Canonicalization framework.
void mlir::tfd::ConvertTftToDhtOp::getCanonicalizationPatterns(
    RewritePatternSet& results, MLIRContext* context) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStfrtPSruntime_fallbackPSruntime_fallback_combineDTcc mht_2(mht_2_v, 253, "", "./tensorflow/compiler/mlir/tfrt/runtime_fallback/runtime_fallback_combine.cc", "mlir::tfd::ConvertTftToDhtOp::getCanonicalizationPatterns");

  results.add<SimplifyDoubleConversion>(context);
}
