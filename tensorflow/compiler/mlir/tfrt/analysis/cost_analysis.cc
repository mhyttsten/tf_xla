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
class MHTracer_DTPStensorflowPScompilerPSmlirPStfrtPSanalysisPScost_analysisDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSmlirPStfrtPSanalysisPScost_analysisDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSmlirPStfrtPSanalysisPScost_analysisDTcc() {
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
#include "tensorflow/compiler/mlir/tfrt/analysis/cost_analysis.h"

#include "absl/container/flat_hash_map.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"

namespace tensorflow {
namespace tfrt_compiler {
namespace {

constexpr int64_t kDefaultCheapCost = 1;

int64_t GetRankedTensorSize(mlir::TensorType type) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStfrtPSanalysisPScost_analysisDTcc mht_0(mht_0_v, 195, "", "./tensorflow/compiler/mlir/tfrt/analysis/cost_analysis.cc", "GetRankedTensorSize");

  auto shape = type.getShape();

  int64_t size = 1;
  for (int64_t dim : shape) {
    // For unknown dimensions, use 1 as the size because it is usually the batch
    // dimension.
    //
    // TODO(chky): Find out a better default number for this case.
    size *= std::max(kDefaultCheapCost, dim);
  }

  return size;
}

int64_t InferTensorSize(const CostContext& context, mlir::TensorType type) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStfrtPSanalysisPScost_analysisDTcc mht_1(mht_1_v, 213, "", "./tensorflow/compiler/mlir/tfrt/analysis/cost_analysis.cc", "InferTensorSize");

  if (type.hasRank()) return GetRankedTensorSize(type);
  return context.default_unranked_tensor_size;
}

// The cost function for tf.LookupTableFindV2.
int64_t InferLookupTableFindV2Cost(const CostContext& context,
                                   mlir::TF::LookupTableFindV2Op op) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStfrtPSanalysisPScost_analysisDTcc mht_2(mht_2_v, 223, "", "./tensorflow/compiler/mlir/tfrt/analysis/cost_analysis.cc", "InferLookupTableFindV2Cost");

  // tf.LookupTableFindV2 ops are usually more costly than tf.AddV2 with the
  // same input size, as it involves more operations like hashing, map lookup,
  // etc.
  constexpr int64_t kLookupTableFindCostScale = 8;
  constexpr int64_t kLookupTableFindStringKeyCostScale = 16;

  auto value_type = op.values().getType().cast<mlir::TensorType>();
  auto key_type = op.keys().getType().cast<mlir::TensorType>();

  int64_t output_size = InferTensorSize(context, value_type);

  int64_t cost = kLookupTableFindCostScale * output_size;

  if (key_type.getElementType().isa<mlir::TF::StringType>())
    cost *= kLookupTableFindStringKeyCostScale;

  return cost;
}

// The cost function for tf.GatherV2.
int64_t InferGatherV2Cost(const CostContext& context, mlir::TF::GatherV2Op op) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStfrtPSanalysisPScost_analysisDTcc mht_3(mht_3_v, 247, "", "./tensorflow/compiler/mlir/tfrt/analysis/cost_analysis.cc", "InferGatherV2Cost");

  return InferTensorSize(context,
                         op.output().getType().cast<mlir::TensorType>());
}

// The cost function for tf.SparseSegmentSumOp.
template <typename OpType>
int64_t InferSparseSegmentOpCost(const CostContext& context, OpType op) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStfrtPSanalysisPScost_analysisDTcc mht_4(mht_4_v, 257, "", "./tensorflow/compiler/mlir/tfrt/analysis/cost_analysis.cc", "InferSparseSegmentOpCost");

  return InferTensorSize(
      context, op.output().getType().template cast<mlir::TensorType>());
}

// CostFunctionRegistry is a map from op names to their cost functions.
using CostFunctionRegistry = absl::flat_hash_map<std::string, CostFunction>;

void RegisterCostFunction(CostFunctionRegistry& registry,
                          absl::string_view op_name,
                          CostFunction cost_function) {
   std::vector<std::string> mht_5_v;
   mht_5_v.push_back("op_name: \"" + std::string(op_name.data(), op_name.size()) + "\"");
   MHTracer_DTPStensorflowPScompilerPSmlirPStfrtPSanalysisPScost_analysisDTcc mht_5(mht_5_v, 271, "", "./tensorflow/compiler/mlir/tfrt/analysis/cost_analysis.cc", "RegisterCostFunction");

  auto r = registry.try_emplace(op_name, std::move(cost_function));
  assert(r.second);
  (void)r;
}

template <typename OpType, typename F>
void RegisterCostFunction(CostFunctionRegistry& registry, F f) {
  RegisterCostFunction(
      registry, OpType::getOperationName().str(),
      [f = std::move(f)](const CostContext& context, mlir::Operation* op) {
        return f(context, llvm::cast<OpType>(op));
      });
}

CostFunctionRegistry& GetCostFunctionRegistry() {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStfrtPSanalysisPScost_analysisDTcc mht_6(mht_6_v, 289, "", "./tensorflow/compiler/mlir/tfrt/analysis/cost_analysis.cc", "GetCostFunctionRegistry");

  static auto* const registry = []() {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStfrtPSanalysisPScost_analysisDTcc mht_7(mht_7_v, 293, "", "./tensorflow/compiler/mlir/tfrt/analysis/cost_analysis.cc", "lambda");

    auto* registry = new CostFunctionRegistry;
    // TODO(chky): Find a more scalable way to register cost functions. One
    // option is to incorporate it is TF MLIR ODS.
    RegisterCostFunction<mlir::TF::GatherV2Op>(*registry, InferGatherV2Cost);
    RegisterCostFunction<mlir::TF::SparseSegmentSumOp>(
        *registry, InferSparseSegmentOpCost<mlir::TF::SparseSegmentSumOp>);
    RegisterCostFunction<mlir::TF::SparseSegmentMeanOp>(
        *registry, InferSparseSegmentOpCost<mlir::TF::SparseSegmentMeanOp>);
    RegisterCostFunction<mlir::TF::SparseSegmentSqrtNOp>(
        *registry, InferSparseSegmentOpCost<mlir::TF::SparseSegmentSqrtNOp>);
    RegisterCostFunction<mlir::TF::LookupTableFindV2Op>(
        *registry, InferLookupTableFindV2Cost);
    return registry;
  }();
  return *registry;
}

}  // namespace

void RegisterCostFunction(absl::string_view op_name,
                          CostFunction cost_function) {
   std::vector<std::string> mht_8_v;
   mht_8_v.push_back("op_name: \"" + std::string(op_name.data(), op_name.size()) + "\"");
   MHTracer_DTPStensorflowPScompilerPSmlirPStfrtPSanalysisPScost_analysisDTcc mht_8(mht_8_v, 318, "", "./tensorflow/compiler/mlir/tfrt/analysis/cost_analysis.cc", "RegisterCostFunction");

  RegisterCostFunction(GetCostFunctionRegistry(), op_name,
                       std::move(cost_function));
}

void CostAnalysis::AnalyzeArguments(mlir::func::FuncOp func_op) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStfrtPSanalysisPScost_analysisDTcc mht_9(mht_9_v, 326, "", "./tensorflow/compiler/mlir/tfrt/analysis/cost_analysis.cc", "CostAnalysis::AnalyzeArguments");

  // Use the max size among function inputs as the default size of dynamic
  // shaped tensors in the function.
  for (auto arg : func_op.getArguments()) {
    auto type = arg.getType().cast<mlir::TensorType>();
    if (type.hasRank()) {
      max_arg_size_ = std::max(max_arg_size_, GetRankedTensorSize(type));
    }
  }
}

void CostAnalysis::AnalyzeBlock(mlir::Block* block) {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStfrtPSanalysisPScost_analysisDTcc mht_10(mht_10_v, 340, "", "./tensorflow/compiler/mlir/tfrt/analysis/cost_analysis.cc", "CostAnalysis::AnalyzeBlock");

  for (auto& op : *block) {
    EvaluateCost(&op);
  }
}

void CostAnalysis::EvaluateCost(mlir::Operation* op) {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStfrtPSanalysisPScost_analysisDTcc mht_11(mht_11_v, 349, "", "./tensorflow/compiler/mlir/tfrt/analysis/cost_analysis.cc", "CostAnalysis::EvaluateCost");

  if (!llvm::isa<mlir::TF::TensorFlowDialect>(op->getDialect())) {
    cost_map_[op] = max_arg_size_;
    return;
  }

  // These ops are cheap regardless of their input sizes.
  //
  // TODO(chky): Find a more scalable way to figure out cheap ops.
  if (llvm::isa<mlir::TF::ShapeOp, mlir::TF::StridedSliceOp,
                mlir::TF::ReshapeOp, mlir::TF::ExpandDimsOp>(op)) {
    cost_map_[op] = kDefaultCheapCost;
    return;
  }

  // Try to use its cost function if it is registered.
  const auto& registry = GetCostFunctionRegistry();
  auto iter = registry.find(op->getName().getStringRef().str());
  if (iter != registry.end()) {
    CostContext context;
    context.default_unranked_tensor_size = max_arg_size_;
    cost_map_[op] = iter->second(context, op);
    return;
  }

  // For other ops, use the sum of input sizes as its cost.
  int64_t cost = kDefaultCheapCost;
  for (auto operand : op->getOperands()) {
    auto type = operand.getType().cast<mlir::TensorType>();
    if (type.hasRank()) {
      cost += GetRankedTensorSize(type);
    } else {
      // For unranked tensors, use the max size among the input tensors. This is
      // because the only dynamic information of the function should be the
      // input, so the size of dynamic tensors should be usually capped by
      // inputs' sizes.
      cost += max_arg_size_;
    }
  }

  cost_map_[op] = cost;
}

}  // namespace tfrt_compiler
}  // namespace tensorflow
