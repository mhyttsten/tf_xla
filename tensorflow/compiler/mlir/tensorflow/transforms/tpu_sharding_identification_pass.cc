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
class MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPStransformsPStpu_sharding_identification_passDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPStransformsPStpu_sharding_identification_passDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPStransformsPStpu_sharding_identification_passDTcc() {
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
#include <string>

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/Optional.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Casting.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/IR/Block.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/Operation.h"  // from @llvm-project
#include "mlir/IR/Value.h"  // from @llvm-project
#include "mlir/Interfaces/CallInterfaces.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Pass/PassRegistry.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_device.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "tensorflow/compiler/mlir/tensorflow/transforms/passes_detail.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/xla_sharding_util.h"
#include "tensorflow/compiler/xla/client/sharding_builder.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"

namespace mlir {
namespace TFTPU {
namespace {

constexpr char kReplicateSharding[] = "";
constexpr char kShardingAttr[] = "mhlo.sharding";
constexpr char kUseSpmdAttr[] = "use_spmd_for_xla_partitioning";
constexpr char kAliasingAttr[] = "tf.aliasing_output";

struct TPUShardingIdentificationPass
    : public TF::TPUShardingIdentificationPassBase<
          TPUShardingIdentificationPass> {
  void runOnOperation() final;
};

// Returns XLA sharding from TPUPartitionedInput op connected to a
// `tf_device.cluster_func` operand value. If value is a resource type then
// TPUPartitionedInput op will be connected to a ReadVariable op that feeds into
// a `tf_device.cluster_func`.
llvm::Optional<llvm::StringRef> GetXlaShardingFromOperand(Value value) {
  Value value_to_visit = value;
  if (auto read_var = value_to_visit.getDefiningOp<TF::ReadVariableOp>())
    value_to_visit = read_var.resource();

  if (auto partitioned_input =
          value_to_visit.getDefiningOp<TF::TPUPartitionedInputOp>())
    return partitioned_input._XlaSharding();

  return llvm::None;
}

// Given a `tf_device.cluster_func` operand value return true iff it a device
// variable that should default to MAXIMAL sharding. Device variables that are
// per-replica or distributed default to MAXIMAL sharding, which corresponds to
// arguments of the `tf_device.replicate`. Otherwise the variable is broadcast,
// which corresponds to edges that are implicitly captured by the `replicate`.
bool IsMaximalVariable(Value value) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPStransformsPStpu_sharding_identification_passDTcc mht_0(mht_0_v, 249, "", "./tensorflow/compiler/mlir/tensorflow/transforms/tpu_sharding_identification_pass.cc", "IsMaximalVariable");

  auto read_var = value.getDefiningOp<TF::ReadVariableOp>();
  return read_var && read_var->getParentOfType<tf_device::ReplicateOp>();
}

// Verify whether the given sharding can be applied to the given (tensor) type.
// (A bad sharding might mean failing tf.Split ops if the graph later executes
//  on CPU)
// If the sharding is incorrect, return failure. If it's good, or if we can't
// verify it, return success.
LogicalResult VerifySharding(Type type, StringRef sharding_string) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPStransformsPStpu_sharding_identification_passDTcc mht_1(mht_1_v, 262, "", "./tensorflow/compiler/mlir/tensorflow/transforms/tpu_sharding_identification_pass.cc", "VerifySharding");

  xla::OpSharding sharding;
  if (!sharding.ParseFromString(sharding_string.str())) {
    // Some test cases use \01\02\03 as sharding, to test propagation. Treat
    // a non-proto sharding as valid, and don't verify further.
    return success();
  }
  if (sharding.type() != xla::OpSharding::OTHER) {
    // We currently only verify shardings that actually break a tensor apart.
    return success();
  }
  if (RankedTensorType ranked_type = type.dyn_cast<RankedTensorType>()) {
    if (ranked_type.getRank() < sharding.tile_assignment_dimensions_size()) {
      return failure();
    }
  }
  return success();
}

// Verify sharding for all arguments and return values.
LogicalResult VerifyShardings(
    mlir::func::FuncOp func,
    const llvm::SmallVectorImpl<llvm::StringRef>& sharding_for_args,
    const llvm::SmallVectorImpl<llvm::StringRef>& sharding_for_rets) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPStransformsPStpu_sharding_identification_passDTcc mht_2(mht_2_v, 288, "", "./tensorflow/compiler/mlir/tensorflow/transforms/tpu_sharding_identification_pass.cc", "VerifyShardings");

  Block& function_block = func.front();
  for (auto sharding_and_arg :
       llvm::zip(sharding_for_args, function_block.getArguments())) {
    StringRef sharding = std::get<0>(sharding_and_arg);
    BlockArgument arg = std::get<1>(sharding_and_arg);
    if (failed(VerifySharding(arg.getType(), sharding))) return failure();
  }
  Operation* terminator = function_block.getTerminator();
  for (auto sharding_and_retval :
       llvm::zip(sharding_for_rets, terminator->getOpOperands())) {
    StringRef sharding = std::get<0>(sharding_and_retval);
    OpOperand& retval = std::get<1>(sharding_and_retval);
    if (failed(VerifySharding(retval.get().getType(), sharding)))
      return failure();
  }
  return success();
}

// Returns XLA sharding from a XlaSharding op connected to an argument value. If
// value is a resource type then XlaSharding op will be connected to a
// ReadVariable op. XlaSharding op may be direct user of inputs but it may also
// be followed by an Identity op and, in the case where bfloat16 type is used,
// Cast op may be added right after the input.
//
// TODO(hongjunchoi): Add logic to parse XlaSharding op inside control flow (If,
// Case, While) ops and Caller return values.
// TODO(hongjunchoi): Consider explicitly checking op patterns to detect sharded
// inputs.
llvm::Optional<llvm::StringRef> GetXlaShardingFromArg(Value value) {
  llvm::SmallPtrSet<Value, 4> visited_values;
  llvm::SmallVector<Value, 4> values_to_visit{value};
  while (!values_to_visit.empty()) {
    llvm::SmallVector<Value, 4> next_values_to_visit;
    for (Value value_to_visit : values_to_visit) {
      if (!visited_values.insert(value_to_visit).second) continue;

      for (auto& use : value_to_visit.getUses()) {
        Operation* owner = use.getOwner();
        if (auto sharding = llvm::dyn_cast<TF::XlaShardingOp>(owner))
          return sharding._XlaSharding();

        if (llvm::isa<TF::IdentityOp, TF::CastOp, TF::ReadVariableOp>(owner)) {
          next_values_to_visit.push_back(use.getOwner()->getResult(0));
          continue;
        }

        if (auto call_op = llvm::dyn_cast<CallOpInterface>(owner)) {
          FuncOp func = llvm::dyn_cast<FuncOp>(call_op.resolveCallable());
          if (!func) continue;
          next_values_to_visit.push_back(
              func.getArgument(use.getOperandNumber()));
        }
      }
    }

    values_to_visit.swap(next_values_to_visit);
  }

  return llvm::None;
}

// Extracts sharding configurations for all inputs by parsing XlaSharding/
// TPUPartitionedInput op connected to the operands/arguments. If argument to
// the `cluster_func` directly feeds into another function call op, then
// recursively walk the function definition to find the connected XlaSharding
// op.
void IdentifyXlaShardingForComputationInputs(
    StringRef logical_core_0_sharding, bool use_spmd,
    bool infer_from_computation, tf_device::ClusterFuncOp cluster_func,
    FuncOp func, Builder* builder,
    llvm::SmallVectorImpl<llvm::StringRef>& sharding_for_args) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPStransformsPStpu_sharding_identification_passDTcc mht_3(mht_3_v, 362, "", "./tensorflow/compiler/mlir/tensorflow/transforms/tpu_sharding_identification_pass.cc", "IdentifyXlaShardingForComputationInputs");

  // Look up function definition from module.
  Block& function_block = func.front();

  sharding_for_args.reserve(function_block.getNumArguments());

  // Iterate through operands of `cluster_func`.
  // The computation operand can either be:
  //   1) a TPUPartitionedInput Op if the input has a non-resource type;
  //   2) a ReadVariableOp else.
  //
  // Replicate sharding is used if `use_spmd` is set.
  //
  // Iterate through input arguments to the entry block of
  // tf_device.ClusterFunc. For input ops, look for XlaSharding ops.
  // XlaSharding ops can:
  //   1) Directly follow the input argument if input argument has non-resource
  //      types.
  //   2) Follow ReadVariableOp if the input type is of resource type.
  //   3) Follow IdentityOp or CastOp after above cases (1), (2).
  //
  // Sharding configurations are added to the tf_device.ClusterFunc as an
  // attribute and the function as an argument attribute.
  for (auto operand_and_arg :
       llvm::zip(cluster_func.operands(), function_block.getArguments())) {
    Value operand = std::get<0>(operand_and_arg);
    BlockArgument arg = std::get<1>(operand_and_arg);

    if (auto operand_sharding = GetXlaShardingFromOperand(operand)) {
      sharding_for_args.push_back(operand_sharding.getValue());
      continue;
    }

    if (infer_from_computation) {
      auto arg_sharding = GetXlaShardingFromArg(arg);
      if (arg_sharding) {
        sharding_for_args.push_back(arg_sharding.getValue());
        continue;
      }
    }

    if (use_spmd && !IsMaximalVariable(operand)) {
      // If XLA SPMD is enabled, host variables or non-variable per-replica
      // inputs should take on replicate sharding, so that every device gets the
      // whole tensor(s) (and can slice them up later). Exclude device
      // variables, which always should take maximal sharding.
      sharding_for_args.push_back(kReplicateSharding);
      continue;
    }

    // Otherwise, default to maximal sharding core 0.
    sharding_for_args.push_back(logical_core_0_sharding);
  }
}

// Returns XLA sharding from TPUPartitionedOutput or TPUPartitionedInput (via
// AssignVariableOp/resource write) op connected to a `tf_device.cluster_func`
// result value.
llvm::Optional<llvm::StringRef> GetXlaShardingFromResult(Value value) {
  if (!value.hasOneUse()) return llvm::None;

  Operation* user = *value.getUsers().begin();
  if (auto partitioned_output =
          llvm::dyn_cast<TF::TPUPartitionedOutputOp>(user))
    return partitioned_output._XlaSharding();

  if (auto assign_var = llvm::dyn_cast<TF::AssignVariableOp>(user))
    if (auto partitioned_input =
            assign_var.resource().getDefiningOp<TF::TPUPartitionedInputOp>())
      return partitioned_input._XlaSharding();

  return llvm::None;
}

// Looks up arg->retval aliases for every argument, and builds a reverse map.
void ExtractAliases(FuncOp func, llvm::SmallVectorImpl<int>& aliases) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPStransformsPStpu_sharding_identification_passDTcc mht_4(mht_4_v, 440, "", "./tensorflow/compiler/mlir/tensorflow/transforms/tpu_sharding_identification_pass.cc", "ExtractAliases");

  aliases.resize(func.getNumResults(), -1);
  for (int i = 0; i < func.getNumArguments(); i++) {
    if (auto v = func.getArgAttrOfType<mlir::IntegerAttr>(i, kAliasingAttr)) {
      int retval_index = v.getInt();
      if (retval_index >= 0 && retval_index < aliases.size()) {
        aliases[retval_index] = i;
      }
    }
  }
}

// Returns XLA sharding from argument connected via tf.aliasing_output.
llvm::Optional<StringRef> GetXlaShardingFromAlias(
    Value value, llvm::SmallVectorImpl<int>& aliases,
    const llvm::SmallVectorImpl<llvm::StringRef>& sharding_for_args) {
  int retval_index = value.cast<OpResult>().getResultNumber();
  if (retval_index >= 0 && retval_index < aliases.size()) {
    int arg_index = aliases[retval_index];
    if (arg_index >= 0 && arg_index < sharding_for_args.size()) {
      return sharding_for_args[arg_index];
    }
  }
  return llvm::None;
}

// Returns XLA sharding from XlaSharding op connected to a result value.
// XlaSharding op may be directly connected to output but it may also be
// followed by Identity or simple arithmetic ops. In case where bfloat16 type is
// used, we might see a Cast op.
//
// TODO(hongjunchoi): Add logic to parse XlaSharding op inside control flow (If,
// Case, While) ops and Caller argument values.
// TODO(hongjunchoi): Consider explicitly checking op patterns to detect sharded
// inputs.
llvm::Optional<StringRef> GetXlaShardingFromRetval(Value value) {
  llvm::SmallPtrSet<Value, 4> visited_values;
  llvm::SmallVector<Value, 4> values_to_visit;
  values_to_visit.push_back(value);

  while (!values_to_visit.empty()) {
    Value value_to_visit = values_to_visit.pop_back_val();

    if (!visited_values.insert(value_to_visit).second) {
      continue;
    }

    Operation* def = value_to_visit.getDefiningOp();
    if (!def) {
      continue;
    }

    if (auto sharding = llvm::dyn_cast_or_null<TF::XlaShardingOp>(def))
      return sharding._XlaSharding();

    if (  // Cast, real/imag, etc.
        def->hasTrait<mlir::OpTrait::SameOperandsAndResultShape>() ||
        // Exp, ceil, etc.
        def->hasTrait<mlir::OpTrait::SameOperandsAndResultType>() ||
        // Identity
        def->hasTrait<mlir::OpTrait::TF::OperandsSameAsResultsTypeOrRef>() ||
        // AddV2, Sub, etc.
        (def->hasTrait<
             mlir::OpTrait::TF::SameOperandsAndResultElementTypeResolveRef>() &&
         def->hasTrait<mlir::OpTrait::TF::CwiseBinary>())) {
      for (auto operand : def->getOperands()) {
        values_to_visit.push_back(operand);
      }
      continue;
    }

    if (auto call_op = llvm::dyn_cast_or_null<CallOpInterface>(def)) {
      FuncOp func = llvm::dyn_cast<FuncOp>(call_op.resolveCallable());
      if (!func) continue;
      value_to_visit = func.front().getTerminator()->getOperand(
          value_to_visit.cast<OpResult>().getResultNumber());
      values_to_visit.push_back(value_to_visit);
      continue;
    }
  }

  return llvm::None;
}

// Extracts sharding configurations for all outputs by parsing XlaSharding/
// TPUPartitionedOutput op connected to the retvals/results.
void IdentifyXlaShardingForComputationOutputs(
    StringRef logical_core_0_sharding, bool use_spmd,
    bool infer_from_computation, tf_device::ClusterFuncOp cluster_func,
    FuncOp func, Builder* builder,
    const llvm::SmallVectorImpl<llvm::StringRef>& sharding_for_args,
    llvm::SmallVectorImpl<llvm::StringRef>& sharding_for_rets) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPStransformsPStpu_sharding_identification_passDTcc mht_5(mht_5_v, 534, "", "./tensorflow/compiler/mlir/tensorflow/transforms/tpu_sharding_identification_pass.cc", "IdentifyXlaShardingForComputationOutputs");

  Block& function_block = func.front();
  Operation* terminator = function_block.getTerminator();
  sharding_for_rets.reserve(terminator->getNumOperands());

  llvm::SmallVector<int, 8> aliases;  // maps return value index to arg index
  ExtractAliases(func, aliases);

  // Iterate through results of `cluster_func`. For output ops, look for
  // TPUPartitionedOutput ops.
  //
  // Replicate sharding is used if `use_spmd` is set.
  //
  // Iterate through operands of the terminator. If the preceding op is
  // XlaShardingOp, then the provided sharding configuration is added to the
  // tf_device.ClusterFunc as an attribute and the function as a result
  // attribute.
  for (auto result_and_retval :
       llvm::zip(cluster_func.results(), terminator->getOpOperands())) {
    Value result = std::get<0>(result_and_retval);
    OpOperand& retval = std::get<1>(result_and_retval);

    if (auto result_sharding = GetXlaShardingFromResult(result)) {
      sharding_for_rets.push_back(result_sharding.getValue());
      continue;
    }

    if (auto from_alias =
            GetXlaShardingFromAlias(result, aliases, sharding_for_args)) {
      sharding_for_rets.push_back(from_alias.getValue());
      continue;
    }

    if (infer_from_computation) {
      if (auto retval_sharding = GetXlaShardingFromRetval(retval.get())) {
        sharding_for_rets.push_back(retval_sharding.getValue());
        continue;
      }
    }

    if (use_spmd) {
      // If XLA SPMD is enabled, we default to replicate sharding. This way,
      // all devices get the whole tensor(s), but if there's an XlaSharding op
      // deeper in the function, they can use dynamic-slice to slice off their
      // part of the computation.
      sharding_for_rets.push_back(kReplicateSharding);
      continue;
    }

    // Otherwise, default to maximal sharding core 0.
    sharding_for_rets.push_back(logical_core_0_sharding);
  }
}

// Extracts input/output sharding configuration of `cluster_func` by parsing
// XlaSharding ops inside the `cluster_func`.
void IdentifyXlaShardingForTPUComputation(
    Builder* builder, tf_device::ClusterFuncOp cluster_func) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPStransformsPStpu_sharding_identification_passDTcc mht_6(mht_6_v, 594, "", "./tensorflow/compiler/mlir/tensorflow/transforms/tpu_sharding_identification_pass.cc", "IdentifyXlaShardingForTPUComputation");

  // Look up function definition from module.
  FuncOp func = cluster_func->getParentOfType<ModuleOp>().lookupSymbol<FuncOp>(
      cluster_func.func());

  // By default inputs/outputs have maximal sharding and are assigned to logical
  // core 0 if no sharding is defined.
  const std::string logical_core_0_sharding =
      xla::sharding_builder::AssignDevice(0).SerializeAsString();

  bool use_spmd = false;
  if (auto use_spmd_attr = cluster_func->getAttrOfType<BoolAttr>(kUseSpmdAttr))
    use_spmd = use_spmd_attr.getValue();

  llvm::SmallVector<llvm::StringRef, 8> sharding_for_args;
  IdentifyXlaShardingForComputationInputs(logical_core_0_sharding, use_spmd,
                                          /*infer_from_computation=*/true,
                                          cluster_func, func, builder,
                                          sharding_for_args);

  llvm::SmallVector<llvm::StringRef, 8> sharding_for_rets;
  IdentifyXlaShardingForComputationOutputs(
      logical_core_0_sharding, use_spmd, /*infer_from_computation=*/true,
      cluster_func, func, builder, sharding_for_args, sharding_for_rets);

  auto has_maximal_sharding = [](llvm::StringRef sharding_string) -> bool {
    xla::OpSharding sharding;
    sharding.ParseFromString(sharding_string.str());
    return sharding.type() == xla::OpSharding::MAXIMAL;
  };

  // XLA SPMD only supports cases where all inputs/outputs exist on every
  // partition (sharded or replicated). If any of the inputs/outputs have
  // maximal sharding, then fallback to MPMD. Also fall back if any of the
  // shardings aren't compatible with the rank of their tensor.
  if ((use_spmd && (absl::c_any_of(sharding_for_args, has_maximal_sharding) ||
                    absl::c_any_of(sharding_for_rets, has_maximal_sharding))) ||
      failed(VerifyShardings(func, sharding_for_args, sharding_for_rets))) {
    LOG(WARNING) << "XLA SPMD only supports cases where all inputs/outputs "
                    "exist on every partition (sharded or replicated). If any "
                    "of the inputs/outputs have maximal sharding, then "
                    "fallback to MPMD.";
    sharding_for_args.clear();
    sharding_for_rets.clear();
    cluster_func->setAttr(kUseSpmdAttr, builder->getBoolAttr(false));

    IdentifyXlaShardingForComputationInputs(
        logical_core_0_sharding,
        /*use_spmd=*/false, /*infer_from_computation=*/false, cluster_func,
        func, builder, sharding_for_args);
    IdentifyXlaShardingForComputationOutputs(
        logical_core_0_sharding,
        /*use_spmd=*/false, /*infer_from_computation=*/false, cluster_func,
        func, builder, sharding_for_args, sharding_for_rets);
  }

  // Update sharding on function arguments and returns.
  Block& function_block = func.front();
  for (auto sharding_and_arg :
       llvm::zip(sharding_for_args, function_block.getArguments())) {
    StringRef sharding = std::get<0>(sharding_and_arg);
    BlockArgument arg = std::get<1>(sharding_and_arg);
    func.setArgAttr(arg.getArgNumber(), kShardingAttr,
                    builder->getStringAttr(sharding));
  }

  Operation* terminator = function_block.getTerminator();
  for (auto sharding_and_retval :
       llvm::zip(sharding_for_rets, terminator->getOpOperands())) {
    StringRef sharding = std::get<0>(sharding_and_retval);
    OpOperand& retval = std::get<1>(sharding_and_retval);
    func.setResultAttr(retval.getOperandNumber(), kShardingAttr,
                       builder->getStringAttr(sharding));
  }

  // Update input/output sharding attributes on tf_device.cluster_func op.
  cluster_func->setAttr(tensorflow::kInputShardingAttr,
                        builder->getStrArrayAttr(sharding_for_args));
  cluster_func->setAttr(tensorflow::kOutputShardingAttr,
                        builder->getStrArrayAttr(sharding_for_rets));
}

void TPUShardingIdentificationPass::runOnOperation() {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPStransformsPStpu_sharding_identification_passDTcc mht_7(mht_7_v, 679, "", "./tensorflow/compiler/mlir/tensorflow/transforms/tpu_sharding_identification_pass.cc", "TPUShardingIdentificationPass::runOnOperation");

  Builder builder(getOperation().getContext());

  getOperation().walk([&](tf_device::ClusterFuncOp cluster_func) {
    IdentifyXlaShardingForTPUComputation(&builder, cluster_func);
  });
}

}  // anonymous namespace

std::unique_ptr<OperationPass<ModuleOp>> CreateTPUShardingIdentificationPass() {
  return std::make_unique<TPUShardingIdentificationPass>();
}

}  // namespace TFTPU
}  // namespace mlir
