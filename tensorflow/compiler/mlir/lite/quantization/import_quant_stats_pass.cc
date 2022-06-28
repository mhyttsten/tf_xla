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
class MHTracer_DTPStensorflowPScompilerPSmlirPSlitePSquantizationPSimport_quant_stats_passDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePSquantizationPSimport_quant_stats_passDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSmlirPSlitePSquantizationPSimport_quant_stats_passDTcc() {
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

#include "absl/memory/memory.h"
#include "absl/strings/str_split.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/Support/Regex.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/Dialect/Quant/FakeQuantSupport.h"  // from @llvm-project
#include "mlir/Dialect/Quant/QuantOps.h"  // from @llvm-project
#include "mlir/IR/AffineExpr.h"  // from @llvm-project
#include "mlir/IR/AffineMap.h"  // from @llvm-project
#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/IR/Location.h"  // from @llvm-project
#include "mlir/IR/PatternMatch.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/lite/quantization/quantization_info.pb.h"
#include "tensorflow/compiler/mlir/lite/quantization/quantization_passes.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/import_utils.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/location_utils.h"

// NOLINTNEXTLINE
static llvm::cl::opt<std::string> quantize_stats(
    "quant-test-stats", llvm::cl::value_desc("string"),
    llvm::cl::desc("serialized quant info string. Only used in tests"),
    llvm::cl::init(""));

//===----------------------------------------------------------------------===//
// The Pass to import quantization stats to the ops in a function. This requires
// a custom method to retrieve the unique name of the operation.

namespace mlir {
namespace quant {

using QuantParamsEntry = QuantizationInfo::QuantParams;

namespace {
class ImportQuantStatsPass
    : public PassWrapper<ImportQuantStatsPass, OperationPass<FuncOp>> {
 public:
  explicit ImportQuantStatsPass(OperationToName op_to_name)
      : op_to_name_(op_to_name) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePSquantizationPSimport_quant_stats_passDTcc mht_0(mht_0_v, 231, "", "./tensorflow/compiler/mlir/lite/quantization/import_quant_stats_pass.cc", "ImportQuantStatsPass");
}

  StringRef getArgument() const final {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePSquantizationPSimport_quant_stats_passDTcc mht_1(mht_1_v, 236, "", "./tensorflow/compiler/mlir/lite/quantization/import_quant_stats_pass.cc", "getArgument");

    // This is the argument used to refer to the pass in
    // the textual format (on the commandline for example).
    return "quant-import-stats";
  }
  StringRef getDescription() const final {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePSquantizationPSimport_quant_stats_passDTcc mht_2(mht_2_v, 244, "", "./tensorflow/compiler/mlir/lite/quantization/import_quant_stats_pass.cc", "getDescription");

    // This is a brief description of the pass.
    return "Import quantization stats to the model";
  }

  void runOnOperation() override;

  void getDependentDialects(DialectRegistry &registry) const override {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePSquantizationPSimport_quant_stats_passDTcc mht_3(mht_3_v, 254, "", "./tensorflow/compiler/mlir/lite/quantization/import_quant_stats_pass.cc", "getDependentDialects");

    registry.insert<quant::QuantizationDialect>();
  }

  // Parses the serialized quant stats protobuf and initialize the internal
  // data structure. This method must be called after the pass is created.
  bool ParseQuantStats(const std::string &stats_str);

 private:
  void ImportAsStatsOps(OpBuilder b, Operation *op, int index,
                        const QuantParamsEntry &info);

  void InsertStatsOpAtResult(OpBuilder b, Value res, ElementsAttr layer_stats,
                             ElementsAttr axis_stats, IntegerAttr axis);

  // If the index is out of range, this method returns false. Otherwise it
  // returns true if the value is a float tensor.
  bool IsQuantizableResult(Operation *op, int index) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePSquantizationPSimport_quant_stats_passDTcc mht_4(mht_4_v, 274, "", "./tensorflow/compiler/mlir/lite/quantization/import_quant_stats_pass.cc", "IsQuantizableResult");

    if (index < 0 || index >= static_cast<int>(op->getNumResults()))
      return false;
    Value res = op->getResult(index);
    return res.getType().isa<ShapedType>() &&
           res.getType().cast<ShapedType>().getElementType().isa<FloatType>();
  }

  // A method to retrieve the name for the given op.
  OperationToName op_to_name_;

  // We split the normal names and regex names, since the former can use hash
  // map to lookup and the latter needs to iterate all the regex to find the
  // match.
  // The `int` in the following two containers are to specify the result index
  // of the given op. -1 indicates all the floating-point results.
  llvm::StringMap<std::pair<int, const QuantParamsEntry>> name_to_info_;
  llvm::StringMap<std::pair<int, const QuantParamsEntry>> regex_to_info_;
};
}  // namespace

bool ImportQuantStatsPass::ParseQuantStats(const std::string &stats_str) {
   std::vector<std::string> mht_5_v;
   mht_5_v.push_back("stats_str: \"" + stats_str + "\"");
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePSquantizationPSimport_quant_stats_passDTcc mht_5(mht_5_v, 299, "", "./tensorflow/compiler/mlir/lite/quantization/import_quant_stats_pass.cc", "ImportQuantStatsPass::ParseQuantStats");

  QuantizationInfo quant_stats;
  if (!tensorflow::LoadProtoFromBuffer(stats_str, &quant_stats).ok()) {
    return true;
  }

  for (const auto &entry : quant_stats.entries()) {
    if (!entry.name().empty()) {
      std::vector<std::string> name_and_port =
          absl::StrSplit(entry.name(), ':');
      int port = name_and_port.size() == 2 ? std::stoi(name_and_port[1]) : -1;
      name_to_info_.insert({name_and_port[0], {port, entry}});
    } else if (!entry.name_regex().empty()) {
      std::vector<std::string> name_and_port =
          absl::StrSplit(entry.name_regex(), ':');
      int port = name_and_port.size() == 2 ? std::stoi(name_and_port[1]) : -1;
      regex_to_info_.insert({name_and_port[0], {port, entry}});
    }
  }
  return false;
}

void ImportQuantStatsPass::InsertStatsOpAtResult(OpBuilder b, Value res,
                                                 ElementsAttr layer_stats,
                                                 ElementsAttr axis_stats,
                                                 IntegerAttr axis) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePSquantizationPSimport_quant_stats_passDTcc mht_6(mht_6_v, 327, "", "./tensorflow/compiler/mlir/lite/quantization/import_quant_stats_pass.cc", "ImportQuantStatsPass::InsertStatsOpAtResult");

  auto stats_op = b.create<quant::StatisticsOp>(b.getUnknownLoc(), res,
                                                layer_stats, axis_stats, axis);
  res.replaceAllUsesWith(stats_op);
  stats_op.getOperation()->replaceUsesOfWith(stats_op, res);
}

void ImportQuantStatsPass::ImportAsStatsOps(OpBuilder b, Operation *op,
                                            int index,
                                            const QuantParamsEntry &info) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePSquantizationPSimport_quant_stats_passDTcc mht_7(mht_7_v, 339, "", "./tensorflow/compiler/mlir/lite/quantization/import_quant_stats_pass.cc", "ImportQuantStatsPass::ImportAsStatsOps");

  if (info.params_size() == 0) return;

  SmallVector<APFloat, 4> min_maxs;
  min_maxs.reserve(info.params_size() * 2);
  for (const auto &param : info.params()) {
    llvm::APFloat min(param.min_max().min());
    llvm::APFloat max(param.min_max().max());
    min_maxs.push_back(min);
    min_maxs.push_back(max);
  }
  // The layer stats contain only the first min/max pairs.
  ElementsAttr layer_stats = DenseFPElementsAttr::get(
      RankedTensorType::get({2}, b.getF32Type()), {min_maxs[0], min_maxs[1]});
  ElementsAttr axis_stats;
  IntegerAttr axis;

  if (info.params_size() > 1) {
    SmallVector<int64_t, 4> axis_stats_shape{info.params_size(), 2};
    axis_stats = DenseFPElementsAttr::get(
        RankedTensorType::get(axis_stats_shape, b.getF32Type()), min_maxs);
    axis = b.getI64IntegerAttr(info.meta().quantize_axis());
  }

  b.setInsertionPointAfter(op);
  if (IsQuantizableResult(op, index)) {
    InsertStatsOpAtResult(b, op->getResult(index), layer_stats, axis_stats,
                          axis);
  } else {
    for (int i = 0, e = op->getNumResults(); i < e; ++i) {
      if (IsQuantizableResult(op, i)) {
        InsertStatsOpAtResult(b, op->getResult(i), layer_stats, axis_stats,
                              axis);
      }
    }
  }
}

void ImportQuantStatsPass::runOnOperation() {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePSquantizationPSimport_quant_stats_passDTcc mht_8(mht_8_v, 380, "", "./tensorflow/compiler/mlir/lite/quantization/import_quant_stats_pass.cc", "ImportQuantStatsPass::runOnOperation");

  FuncOp func = getOperation();
  OpBuilder builder(func);

  func.walk([&](Operation *op) {
    if (op->hasTrait<OpTrait::IsTerminator>()) return;
    auto op_name = op_to_name_(op);

    // Check the named info collection first.
    auto it = name_to_info_.find(op_name);
    if (it != name_to_info_.end()) {
      ImportAsStatsOps(builder, op, it->second.first, it->second.second);
      return;
    }

    // Iterate all the regex names and matches the first one.
    for (auto &regex : regex_to_info_) {
      if (llvm::Regex(regex.first()).match(op_name)) {
        ImportAsStatsOps(builder, op, regex.second.first, regex.second.second);
        break;
      }
    }
  });
}

// Creates an instance of the default quant parameters pass.
std::unique_ptr<OperationPass<FuncOp>> CreateImportQuantStatsPass(
    OperationToName op_to_name, const std::string &stats_str) {
  auto pass = absl::make_unique<ImportQuantStatsPass>(op_to_name);
  if (pass->ParseQuantStats(stats_str)) return nullptr;
  return pass;
}

// Creates an instance pass to import quantization stats to the operations in
// the function. A custom method to get the name from the op is used because
// different dialect ops might have different ways to assign the name.
std::unique_ptr<OperationPass<FuncOp>>
CreateImportQuantStatsPassForTFControlDialect(const std::string &stats_str) {
  auto get_name_func = [](Operation *op) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePSquantizationPSimport_quant_stats_passDTcc mht_9(mht_9_v, 421, "", "./tensorflow/compiler/mlir/lite/quantization/import_quant_stats_pass.cc", "lambda");

    Location loc = tensorflow::GetLocationWithoutOpType(op->getLoc());
    if (auto name = loc.dyn_cast<NameLoc>()) {
      return name.getName().strref();
    } else if (auto fused_name = loc.dyn_cast<FusedLoc>()) {
      for (auto sub_loc : fused_name.getLocations()) {
        if (auto named_sub_loc = sub_loc.dyn_cast<NameLoc>()) {
          return named_sub_loc.getName().strref();
        }
      }
    }
    return llvm::StringRef("");
  };

  return CreateImportQuantStatsPass(get_name_func, stats_str);
}

// Registers this pass with default values, only for test
static PassRegistration<ImportQuantStatsPass> pass([] {
  return CreateImportQuantStatsPassForTFControlDialect(quantize_stats);
});

}  // namespace quant
}  // namespace mlir
