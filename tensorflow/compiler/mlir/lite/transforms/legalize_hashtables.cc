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
class MHTracer_DTPStensorflowPScompilerPSmlirPSlitePStransformsPSlegalize_hashtablesDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePStransformsPSlegalize_hashtablesDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSmlirPSlitePStransformsPSlegalize_hashtablesDTcc() {
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

#include "llvm/ADT/None.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/IR/Operation.h"  // from @llvm-project
#include "mlir/IR/PatternMatch.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/lite/ir/tfl_ops.h"
#include "tensorflow/compiler/mlir/lite/transforms/passes.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops_n_z.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_saved_model.h"

namespace mlir {
namespace TFL {
namespace {
// This file has Legalize hash tables pass which is responsible for:
// - Converting static hash table ops to the TFLite equivalent ops.
//
// There are needs to fall back to Flex for the following cases:
// - Mutable hash table cases
// - Other resource operators consuming a hash table resource tensor

class LegalizeHashTableOpPattern : public OpRewritePattern<TF::HashTableV2Op> {
 public:
  using OpRewritePattern<TF::HashTableV2Op>::OpRewritePattern;

  LogicalResult matchAndRewrite(TF::HashTableV2Op hashtable_op,
                                PatternRewriter& rewriter) const override {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePStransformsPSlegalize_hashtablesDTcc mht_0(mht_0_v, 218, "", "./tensorflow/compiler/mlir/lite/transforms/legalize_hashtables.cc", "matchAndRewrite");

    auto output_type = RankedTensorType::get(
        {1}, TF::ResourceType::get(rewriter.getContext()));

    // Hash the shared name to generate integer hash table id. The TFLite
    // native resource design is based on integer keys to identify the
    // corresponding resource objects.
    auto table_id =
        static_cast<int32_t>(::llvm::hash_value(hashtable_op.shared_name()));
    auto key_dtype = hashtable_op.key_dtype();
    auto value_dtype = hashtable_op.value_dtype();

    rewriter.replaceOpWithNewOp<TFL::HashtableOp>(
        hashtable_op, output_type, table_id, key_dtype, value_dtype);
    return success();
  }
};

class LegalizeHashTableFindOpPattern
    : public OpRewritePattern<TF::LookupTableFindV2Op> {
 public:
  using OpRewritePattern<TF::LookupTableFindV2Op>::OpRewritePattern;

  LogicalResult matchAndRewrite(TF::LookupTableFindV2Op find_op,
                                PatternRewriter& rewriter) const override {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePStransformsPSlegalize_hashtablesDTcc mht_1(mht_1_v, 245, "", "./tensorflow/compiler/mlir/lite/transforms/legalize_hashtables.cc", "matchAndRewrite");

    auto handle_op = find_op.table_handle().getDefiningOp();
    if (handle_op == nullptr) return failure();
    auto hashtable_op = llvm::dyn_cast<TFL::HashtableOp>(handle_op);
    if (hashtable_op == nullptr) return failure();
    rewriter.replaceOpWithNewOp<TFL::HashtableFindOp>(
        find_op, find_op->getResultTypes(), find_op.table_handle(),
        find_op.keys(), find_op.default_value());
    return success();
  }
};

class LegalizeHashTableImportOpPattern
    : public OpRewritePattern<TF::LookupTableImportV2Op> {
 public:
  using OpRewritePattern<TF::LookupTableImportV2Op>::OpRewritePattern;

  LogicalResult matchAndRewrite(TF::LookupTableImportV2Op import_op,
                                PatternRewriter& rewriter) const override {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePStransformsPSlegalize_hashtablesDTcc mht_2(mht_2_v, 266, "", "./tensorflow/compiler/mlir/lite/transforms/legalize_hashtables.cc", "matchAndRewrite");

    auto handle_op = import_op.table_handle().getDefiningOp();
    if (handle_op == nullptr) return failure();
    auto hashtable_op = llvm::dyn_cast<TFL::HashtableOp>(handle_op);
    if (hashtable_op == nullptr) return failure();
    rewriter.replaceOpWithNewOp<TFL::HashtableImportOp>(
        import_op, import_op->getResultTypes(), import_op.table_handle(),
        import_op.keys(), import_op.values());
    return success();
  }
};

class LegalizeHashTableSizeOpPattern
    : public OpRewritePattern<TF::LookupTableSizeV2Op> {
 public:
  using OpRewritePattern<TF::LookupTableSizeV2Op>::OpRewritePattern;

  LogicalResult matchAndRewrite(TF::LookupTableSizeV2Op size_op,
                                PatternRewriter& rewriter) const override {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePStransformsPSlegalize_hashtablesDTcc mht_3(mht_3_v, 287, "", "./tensorflow/compiler/mlir/lite/transforms/legalize_hashtables.cc", "matchAndRewrite");

    auto handle_op = size_op.table_handle().getDefiningOp();
    if (handle_op == nullptr) return failure();
    auto hashtable_op = llvm::dyn_cast<TFL::HashtableOp>(handle_op);
    if (hashtable_op == nullptr) return failure();
    rewriter.replaceOpWithNewOp<TFL::HashtableSizeOp>(
        size_op, size_op->getResultTypes(), size_op.table_handle());
    return success();
  }
};

template <typename T>
std::vector<T> GetAllOps(mlir::ModuleOp* module) {
  std::vector<T> ops;
  module->walk([&](T op) { ops.emplace_back(op); });
  return ops;
}

bool checkWhetherGraphHasValidStaticLookupTables(ModuleOp module) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePStransformsPSlegalize_hashtablesDTcc mht_4(mht_4_v, 308, "", "./tensorflow/compiler/mlir/lite/transforms/legalize_hashtables.cc", "checkWhetherGraphHasValidStaticLookupTables");

  auto hashtables = GetAllOps<TF::HashTableV2Op>(&module);
  // No needs to run the legalization patterns.
  if (hashtables.empty()) {
    return false;
  }

  for (auto hashtable : hashtables) {
    auto key_dtype = hashtable.key_dtype();
    auto value_dtype = hashtable.value_dtype();

    // Only allow string -> int64 and int64 -> string mappings due to kernel
    // capability.
    if (!((key_dtype.isa<TF::StringType>() && value_dtype.isa<IntegerType>() &&
           value_dtype.cast<IntegerType>().getWidth() == 64) ||
          (value_dtype.isa<TF::StringType>() && key_dtype.isa<IntegerType>() &&
           key_dtype.cast<IntegerType>().getWidth() == 64))) {
      return false;
    }

    for (auto& use : hashtable->getUses()) {
      Operation* user = use.getOwner();

      // Allow consuming hash table ops that can be covered by TensorFlow Lite
      // hash table kernels.
      if (auto find_op = llvm::dyn_cast<TF::LookupTableFindV2Op>(user))
        continue;
      if (auto import_op = llvm::dyn_cast<TF::LookupTableImportV2Op>(user))
        continue;
      if (auto size_op = llvm::dyn_cast<TF::LookupTableSizeV2Op>(user))
        continue;

      return false;
    }
  }
  return true;
}

// Pass which legalizes TF hash tables only when they are covered by the
// TensorFlow Lite hash table kernels.
class LegalizeHashTables
    : public PassWrapper<LegalizeHashTables, OperationPass<ModuleOp>> {
  void getDependentDialects(DialectRegistry& registry) const override {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePStransformsPSlegalize_hashtablesDTcc mht_5(mht_5_v, 353, "", "./tensorflow/compiler/mlir/lite/transforms/legalize_hashtables.cc", "getDependentDialects");

    registry.insert<TensorFlowLiteDialect>();
  }

 public:
  LegalizeHashTables() = default;
  LegalizeHashTables(const LegalizeHashTables&) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePStransformsPSlegalize_hashtablesDTcc mht_6(mht_6_v, 362, "", "./tensorflow/compiler/mlir/lite/transforms/legalize_hashtables.cc", "LegalizeHashTables");
}

  StringRef getArgument() const final {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePStransformsPSlegalize_hashtablesDTcc mht_7(mht_7_v, 367, "", "./tensorflow/compiler/mlir/lite/transforms/legalize_hashtables.cc", "getArgument");

    // This is the argument used to refer to the pass in
    // the textual format (on the commandline for example).
    return "tfl-legalize-hashtables-tf";
  }
  StringRef getDescription() const final {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePStransformsPSlegalize_hashtablesDTcc mht_8(mht_8_v, 375, "", "./tensorflow/compiler/mlir/lite/transforms/legalize_hashtables.cc", "getDescription");

    // This is a brief description of the pass.
    return "Legalize TensorFlow hash tables to TensorFlow Lite dialect";
  }

  void runOnOperation() override {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePStransformsPSlegalize_hashtablesDTcc mht_9(mht_9_v, 383, "", "./tensorflow/compiler/mlir/lite/transforms/legalize_hashtables.cc", "runOnOperation");

    auto module = getOperation();

    if (!checkWhetherGraphHasValidStaticLookupTables(module)) {
      return;
    }

    RewritePatternSet patterns(&getContext());
    patterns
        .add<LegalizeHashTableOpPattern, LegalizeHashTableFindOpPattern,
             LegalizeHashTableImportOpPattern, LegalizeHashTableSizeOpPattern>(
            &getContext());
    if (failed(applyPatternsAndFoldGreedily(module, std::move(patterns)))) {
      signalPassFailure();
      return;
    }
  }
};

}  // namespace

std::unique_ptr<OperationPass<ModuleOp>> CreateLegalizeHashTablesPass() {
  return std::make_unique<LegalizeHashTables>();
}

static PassRegistration<LegalizeHashTables> pass;

}  // namespace TFL
}  // namespace mlir
