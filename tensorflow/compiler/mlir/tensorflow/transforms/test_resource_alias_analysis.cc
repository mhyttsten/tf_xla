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
class MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPStransformsPStest_resource_alias_analysisDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPStransformsPStest_resource_alias_analysisDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPStransformsPStest_resource_alias_analysisDTcc() {
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

#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Pass/PassManager.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "mlir/Transforms/Passes.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/analysis/resource_alias_analysis.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"

namespace mlir {
namespace tf_test {
namespace {

// A pass that annotates each operation with a resource type result with the
// aliasing values for each such result. Each value is assigned a unique ID, and
// that ID is used to annotate the operations.
struct TestResourceAliasAnalysis
    : public TF::PerFunctionAggregateAnalysisConsumerPass<
          TestResourceAliasAnalysis, TF::ResourceAliasAnalysis> {
  StringRef getArgument() const final {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPStransformsPStest_resource_alias_analysisDTcc mht_0(mht_0_v, 211, "", "./tensorflow/compiler/mlir/tensorflow/transforms/test_resource_alias_analysis.cc", "getArgument");

    return "tf-test-resource-alias-analysis";
  }

  StringRef getDescription() const final {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPStransformsPStest_resource_alias_analysisDTcc mht_1(mht_1_v, 218, "", "./tensorflow/compiler/mlir/tensorflow/transforms/test_resource_alias_analysis.cc", "getDescription");

    return "Add remarks based on resource alias analysis result, for testing "
           "purpose.";
  }

  void runOnFunction(FuncOp func,
                     const TF::ResourceAliasAnalysis::Info& analysis) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPStransformsPStest_resource_alias_analysisDTcc mht_2(mht_2_v, 227, "", "./tensorflow/compiler/mlir/tensorflow/transforms/test_resource_alias_analysis.cc", "runOnFunction");

    int64_t next_id = 0;
    llvm::SmallDenseMap<Value, int64_t, 8> ids;

    auto assign_id = [&](Value value) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPStransformsPStest_resource_alias_analysisDTcc mht_3(mht_3_v, 234, "", "./tensorflow/compiler/mlir/tensorflow/transforms/test_resource_alias_analysis.cc", "lambda");

      if (ids.find(value) == ids.end()) ids.insert({value, next_id++});
    };

    auto get_id = [&](Value value) -> int64_t {
      auto it = ids.find(value);
      assert(it != ids.end());
      return it->second;
    };

    auto print_aliases = [&](InFlightDiagnostic& diag, Value value) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPStransformsPStest_resource_alias_analysisDTcc mht_4(mht_4_v, 247, "", "./tensorflow/compiler/mlir/tensorflow/transforms/test_resource_alias_analysis.cc", "lambda");

      diag << ", ID " << get_id(value) << " : ";
      if (analysis.IsUnknownResource(value)) {
        diag << "Unknown";
      } else {
        auto aliases = llvm::to_vector<4>(analysis.GetResourceAliases(value));
        llvm::sort(aliases,
                   [&](Value v1, Value v2) { return get_id(v1) < get_id(v2); });
        llvm::interleaveComma(aliases, diag,
                              [&](Value v) { diag << get_id(v); });
      }
    };

    // Assign a unique ID to each value seen in this function.
    func.walk([&](Operation* op) {
      // For all attached regions, assign ID to the region arguments.
      for (Region& region : op->getRegions()) {
        for (auto region_arg : TF::filter_resources(region.getArguments()))
          assign_id(region_arg);
      }

      // Assign ID for all results.
      for (auto result : TF::filter_resources(op->getResults()))
        assign_id(result);
    });

    // Now walk each operation, and annotate it wil remarks for aliases for
    // each resource type result
    func.walk([&](Operation* op) {
      // For all attached regions, assign ID to the region arguments.
      for (Region& region : op->getRegions()) {
        for (auto region_arg : TF::filter_resources(region.getArguments())) {
          InFlightDiagnostic diag = op->emitRemark("Region #")
                                    << region.getRegionNumber() << ", Arg #"
                                    << region_arg.getArgNumber();
          print_aliases(diag, region_arg);
        }
      }

      for (auto result : TF::filter_resources(op->getResults())) {
        InFlightDiagnostic diag = op->emitRemark("Result #")
                                  << result.getResultNumber();
        print_aliases(diag, result);
      }
    });
  }
};

}  // anonymous namespace

std::unique_ptr<OperationPass<ModuleOp>> CreateTestResourceAliasAnalysisPass() {
  return std::make_unique<TestResourceAliasAnalysis>();
}

}  // namespace tf_test
}  // namespace mlir
