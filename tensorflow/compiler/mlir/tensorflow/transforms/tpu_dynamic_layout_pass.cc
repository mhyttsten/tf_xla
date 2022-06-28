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
class MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPStransformsPStpu_dynamic_layout_passDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPStransformsPStpu_dynamic_layout_passDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPStransformsPStpu_dynamic_layout_passDTcc() {
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

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Casting.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/IR/Location.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/IR/Operation.h"  // from @llvm-project
#include "mlir/IR/OperationSupport.h"  // from @llvm-project
#include "mlir/IR/TypeUtilities.h"  // from @llvm-project
#include "mlir/IR/Types.h"  // from @llvm-project
#include "mlir/IR/UseDefLists.h"  // from @llvm-project
#include "mlir/IR/Value.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Pass/PassRegistry.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/analysis/side_effect_analysis.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_device.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_types.h"
#include "tensorflow/compiler/mlir/tensorflow/transforms/passes.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/tpu_rewrite_device_util.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/xla_sharding_util.h"
#include "tensorflow/core/protobuf/tpu/compile_metadata.pb.h"
#include "tensorflow/core/util/device_name_utils.h"

namespace mlir {
namespace TFTPU {

namespace {

constexpr char kDeviceAttr[] = "device";
constexpr char kDeviceCPU[] = "CPU";
constexpr char kFuncDeviceAttr[] = "tf.device";

struct TPUDynamicLayoutPass
    : public TF::PerFunctionAggregateAnalysisConsumerPass<
          TPUDynamicLayoutPass, TF::ResourceAliasAnalysis> {
  void runOnFunction(
      FuncOp func,
      const TF::ResourceAliasAnalysis::Info& resource_alias_analysis);

  StringRef getArgument() const final {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPStransformsPStpu_dynamic_layout_passDTcc mht_0(mht_0_v, 232, "", "./tensorflow/compiler/mlir/tensorflow/transforms/tpu_dynamic_layout_pass.cc", "getArgument");
 return "tf-tpu-dynamic-layout-pass"; }

  StringRef getDescription() const final {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPStransformsPStpu_dynamic_layout_passDTcc mht_1(mht_1_v, 237, "", "./tensorflow/compiler/mlir/tensorflow/transforms/tpu_dynamic_layout_pass.cc", "getDescription");

    return "Inserts TPU layout ops to determine layout at run time.";
  }
};

// Checks if the input producer op is supported in this transform. Right now, we
// only check if it is a tf.IteratorGetNext where resource input is coming from
// a VarHandle on CPU or a function argument assigned to CPU.
bool IsSupportedInputOp(
    Operation* op,
    const TF::ResourceAliasAnalysis::Info& resource_alias_analysis) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPStransformsPStpu_dynamic_layout_passDTcc mht_2(mht_2_v, 250, "", "./tensorflow/compiler/mlir/tensorflow/transforms/tpu_dynamic_layout_pass.cc", "IsSupportedInputOp");

  TF::IteratorGetNextOp iterator_op = llvm::dyn_cast<TF::IteratorGetNextOp>(op);
  if (!iterator_op) return false;

  Value resource_iterator = iterator_op.iterator();

  if (resource_alias_analysis.IsUnknownResource(resource_iterator))
    return false;
  llvm::SmallSetVector<Value, 8> aliases =
      resource_alias_analysis.GetResourceAliases(resource_iterator);

  auto is_generator = [](Value val) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPStransformsPStpu_dynamic_layout_passDTcc mht_3(mht_3_v, 264, "", "./tensorflow/compiler/mlir/tensorflow/transforms/tpu_dynamic_layout_pass.cc", "lambda");

    if (val.isa<BlockArgument>()) return true;
    Operation* definition = val.getDefiningOp();
    return definition->getNumOperands() == 0 &&
           definition->getNumResults() == 1;
  };

  // Check all generator aliases (ops or function argument) are on CPU.
  FuncOp func = iterator_op->getParentOfType<FuncOp>();
  return llvm::all_of(aliases, [&](Value alias) {
    // Ignore non-generator aliases.
    if (!is_generator(alias)) return true;

    StringAttr device;
    if (auto arg = alias.dyn_cast<BlockArgument>()) {
      device = func.getArgAttrOfType<mlir::StringAttr>(arg.getArgNumber(),
                                                       kFuncDeviceAttr);
    } else {
      device = alias.getDefiningOp()->getAttrOfType<StringAttr>(kDeviceAttr);
    }

    if (!device) return false;
    tensorflow::DeviceNameUtils::ParsedName parsed_device;
    if (!tensorflow::DeviceNameUtils::ParseFullName(device.getValue().str(),
                                                    &parsed_device)) {
      return false;
    }
    return parsed_device.has_type && parsed_device.type == kDeviceCPU;
  });
}

OpBuilder CreateBuilderAfterOp(Operation* op) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPStransformsPStpu_dynamic_layout_passDTcc mht_4(mht_4_v, 298, "", "./tensorflow/compiler/mlir/tensorflow/transforms/tpu_dynamic_layout_pass.cc", "CreateBuilderAfterOp");

  return OpBuilder(op->getBlock(), ++Block::iterator(op));
}

// Builds a TPUGetLayoutOp with the given compile op and input index.
TF::TPUGetLayoutOp BuildGetLayout(const int64_t execute_arg_index,
                                  Value compilation_key,
                                  tf_device::LaunchOp compile_launch,
                                  OpBuilder* builder) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPStransformsPStpu_dynamic_layout_passDTcc mht_5(mht_5_v, 309, "", "./tensorflow/compiler/mlir/tensorflow/transforms/tpu_dynamic_layout_pass.cc", "BuildGetLayout");

  return builder->create<TF::TPUGetLayoutOp>(
      compile_launch.getLoc(),
      llvm::ArrayRef<Type>{RankedTensorType::get({ShapedType::kDynamicSize},
                                                 builder->getIntegerType(64))},
      llvm::ArrayRef<Value>{compilation_key},
      llvm::ArrayRef<NamedAttribute>{
          builder->getNamedAttr("index",
                                builder->getI64IntegerAttr(execute_arg_index)),
          builder->getNamedAttr("is_output", builder->getBoolAttr(false))});
}

// Builds a TPUCopyWithLayoutOp with the given get_layout op and input.
TF::TPUCopyWithLayoutOp BuildCopyWithLayout(tf_device::LaunchOp execute_launch,
                                            tf_device::LaunchOp compile_launch,
                                            TF::TPUGetLayoutOp get_layout,
                                            Value input, OpBuilder* builder) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPStransformsPStpu_dynamic_layout_passDTcc mht_6(mht_6_v, 328, "", "./tensorflow/compiler/mlir/tensorflow/transforms/tpu_dynamic_layout_pass.cc", "BuildCopyWithLayout");

  return builder->create<TF::TPUCopyWithLayoutOp>(
      execute_launch.getLoc(), llvm::ArrayRef<Type>{input.getType()},
      llvm::ArrayRef<Value>{input, get_layout.layout()});
}

// Performs transformation for a non-replicated input.
void HandleInput(Value input, const int64_t execute_arg_index,
                 TF::TPUExecuteOp execute, tf_device::LaunchOp execute_launch,
                 tf_device::LaunchOp compile_launch) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPStransformsPStpu_dynamic_layout_passDTcc mht_7(mht_7_v, 340, "", "./tensorflow/compiler/mlir/tensorflow/transforms/tpu_dynamic_layout_pass.cc", "HandleInput");

  OpBuilder builder = CreateBuilderAfterOp(compile_launch);
  auto get_layout = BuildGetLayout(execute_arg_index, execute.key(),
                                   compile_launch, &builder);
  builder.setInsertionPoint(execute_launch);
  auto copy_with_layout = BuildCopyWithLayout(execute_launch, compile_launch,
                                              get_layout, input, &builder);
  copy_with_layout->setAttr(kDeviceAttr, execute_launch.deviceAttr());
  execute.setOperand(execute_arg_index, copy_with_layout);
}

// Performs transformation for replicated inputs. Returns true if this is a
// supported case (thus transform happened).
bool HandleReplicatedInputs(
    const int64_t execute_arg_index, Value compilation_key,
    tf_device::LaunchOp execute_launch, tf_device::LaunchOp compile_launch,
    mlir::BlockArgument replicate_arg, tf_device::ReplicateOp replicate,
    const TF::ResourceAliasAnalysis::Info& resource_alias_analysis) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPStransformsPStpu_dynamic_layout_passDTcc mht_8(mht_8_v, 360, "", "./tensorflow/compiler/mlir/tensorflow/transforms/tpu_dynamic_layout_pass.cc", "HandleReplicatedInputs");

  // We need to know the devices to copy to.
  if (!replicate.devices()) return false;

  MutableArrayRef<OpOperand> inputs =
      replicate.GetOperandsForBlockArgument(replicate_arg);
  for (auto entry : llvm::enumerate(inputs)) {
    auto input_op = entry.value().get().getDefiningOp();
    if (!input_op || !IsSupportedInputOp(input_op, resource_alias_analysis))
      return false;
  }
  OpBuilder builder = CreateBuilderAfterOp(compile_launch);
  auto get_layout = BuildGetLayout(execute_arg_index, compilation_key,
                                   compile_launch, &builder);
  builder.setInsertionPoint(replicate);
  for (auto entry : llvm::enumerate(inputs)) {
    auto copy_with_layout =
        BuildCopyWithLayout(execute_launch, compile_launch, get_layout,
                            entry.value().get(), &builder);

    auto device_list = replicate.devices()
                           .getValue()
                           .get(execute_launch.getDevice())
                           .cast<ArrayAttr>();
    copy_with_layout->setAttr(kDeviceAttr,
                              device_list.getValue()[entry.index()]);

    entry.value().set(copy_with_layout);
  }
  return true;
}

// Performs transformation on a compile and associated execute(s) ops. The
// compile should not have other uses.
void HandleCompileAndExecutes(
    tf_device::LaunchOp compile_launch,
    llvm::MutableArrayRef<tf_device::LaunchOp> execute_launches,
    const TF::ResourceAliasAnalysis::Info& resource_alias_analysis) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPStransformsPStpu_dynamic_layout_passDTcc mht_9(mht_9_v, 400, "", "./tensorflow/compiler/mlir/tensorflow/transforms/tpu_dynamic_layout_pass.cc", "HandleCompileAndExecutes");

  auto compile =
      llvm::cast<TF::_TPUCompileMlirOp>(compile_launch.GetBody().front());
  tensorflow::tpu::TPUCompileMetadataProto metadata;
  metadata.ParseFromString(compile.metadata().str());
  llvm::SmallVector<llvm::SmallVector<int64_t, 4>, 4> input_mappings =
      tensorflow::GetMetadataArgumentMapping(metadata);

  bool metadata_updated = false;
  auto maybe_replicate =
      execute_launches.front()->getParentOfType<tf_device::ReplicateOp>();

  for (auto execute_and_input_mapping :
       llvm::zip(execute_launches, input_mappings)) {
    auto& execute_launch = std::get<0>(execute_and_input_mapping);
    auto execute =
        llvm::cast<TF::TPUExecuteOp>(execute_launch.GetBody().front());
    const auto& input_mapping = std::get<1>(execute_and_input_mapping);

    for (auto& input_and_idx : llvm::enumerate(execute.args())) {
      Value input = input_and_idx.value();
      const int64_t execute_arg_index = input_and_idx.index();
      if (auto block_arg = input.dyn_cast<BlockArgument>()) {
        // For a block argument, consider transforms only when it is a
        // replicated input (defining ops will be outside the replicate node).
        if (maybe_replicate != block_arg.getParentRegion()->getParentOp() ||
            !HandleReplicatedInputs(execute_arg_index, execute.key(),
                                    execute_launch, compile_launch, block_arg,
                                    maybe_replicate, resource_alias_analysis)) {
          continue;
        }
      } else {
        // For an op output, consider transforms only when 1) there is no
        // replication or 2) it is outside the replicate node that encloses the
        // execute node. (Because if the op is inside replicate, it is probably
        // not on the host.)
        auto* input_op = input.getDefiningOp();
        if (maybe_replicate &&
            maybe_replicate.body().isAncestor(input_op->getParentRegion())) {
          continue;
        }
        if (!IsSupportedInputOp(input_op, resource_alias_analysis)) continue;
        HandleInput(input, execute_arg_index, execute, execute_launch,
                    compile_launch);
      }

      metadata.mutable_args(input_mapping[execute_arg_index])
          ->set_unrestricted_layout(true);
      metadata_updated = true;
    }
  }

  if (metadata_updated)
    compile->setAttr("metadata", StringAttr::get(compile.getContext(),
                                                 metadata.SerializeAsString()));
}

void TPUDynamicLayoutPass::runOnFunction(
    FuncOp func,
    const TF::ResourceAliasAnalysis::Info& resource_alias_analysis) {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPStransformsPStpu_dynamic_layout_passDTcc mht_10(mht_10_v, 462, "", "./tensorflow/compiler/mlir/tensorflow/transforms/tpu_dynamic_layout_pass.cc", "TPUDynamicLayoutPass::runOnFunction");

  func.walk([&](TF::_TPUCompileMlirOp compile) {
    // Detect tf._TPUCompileMlir -> tf.TPUExecute(s).
    auto compile_launch =
        llvm::dyn_cast<tf_device::LaunchOp>(compile->getParentOp());
    if (!compile_launch || !compile_launch.WrapsSingleOp()) return;

    llvm::SmallVector<tf_device::LaunchOp, 4> execute_launches;
    execute_launches.reserve(compile_launch.getNumResults() - 1);
    for (Value program_result : llvm::drop_begin(compile_launch.results(), 1)) {
      if (!program_result.hasOneUse()) return;
      Operation* user = *program_result.user_begin();
      auto execute = llvm::dyn_cast<TF::TPUExecuteOp>(user);
      if (!execute) return;
      auto execute_launch =
          llvm::dyn_cast<tf_device::LaunchOp>(execute->getParentOp());
      if (!execute_launch || !execute_launch.WrapsSingleOp()) return;
      execute_launches.push_back(execute_launch);
    }

    HandleCompileAndExecutes(compile_launch, execute_launches,
                             resource_alias_analysis);
  });
}

}  // namespace

std::unique_ptr<OperationPass<ModuleOp>> CreateTPUDynamicLayoutPass() {
  return std::make_unique<TPUDynamicLayoutPass>();
}

}  // namespace TFTPU
}  // namespace mlir
