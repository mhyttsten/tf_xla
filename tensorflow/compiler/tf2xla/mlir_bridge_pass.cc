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
class MHTracer_DTPStensorflowPScompilerPStf2xlaPSmlir_bridge_passDTcc {
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
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSmlir_bridge_passDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPStf2xlaPSmlir_bridge_passDTcc() {
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

#include "tensorflow/compiler/tf2xla/mlir_bridge_pass.h"

#include <string>

#include "absl/base/call_once.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_structs.h"
#include "tensorflow/compiler/mlir/tensorflow/transforms/bridge.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/device_util.h"
#include "tensorflow/core/common_runtime/device_set.h"
#include "tensorflow/core/framework/metrics.h"
#include "tensorflow/core/lib/monitoring/gauge.h"
#include "tensorflow/core/public/session_options.h"
#include "tensorflow/core/util/device_name_utils.h"

namespace tensorflow {

// Values for the label 'PassState'
constexpr char kEnabled[] = "kEnabled";
constexpr char kDisabled[] = "kDisabled";

auto* mlir_bridge_gauge_v1 = monitoring::Gauge<bool, 0>::New(
    "/tensorflow/config/experimental/enable_mlir_bridge_gauge_v1",
    "Tracks usage of the MLIR-based TF2XLA bridge among TF1 models");
auto* mlir_bridge_gauge_v2 = monitoring::Gauge<bool, 0>::New(
    "/tensorflow/config/experimental/enable_mlir_bridge_gauge_v2",
    "Tracks usage of the MLIR-based TF2XLA bridge among TF2 models");

namespace {

constexpr char kTPUReplicateAttr[] = "_tpu_replicate";

bool HasTPUDevice(mlir::ModuleOp module) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSmlir_bridge_passDTcc mht_0(mht_0_v, 216, "", "./tensorflow/compiler/tf2xla/mlir_bridge_pass.cc", "HasTPUDevice");

  mlir::TF::RuntimeDevices devices;
  if (failed(GetDevicesFromOp(module.getOperation(), &devices))) return false;
  return absl::c_any_of(
      devices.device_names(),
      [](const tensorflow::DeviceNameUtils::ParsedName& device) {
        return device.has_type && device.type == "TPU";
      });
}

bool HasTPUOp(mlir::ModuleOp module) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSmlir_bridge_passDTcc mht_1(mht_1_v, 229, "", "./tensorflow/compiler/tf2xla/mlir_bridge_pass.cc", "HasTPUOp");

  auto walk_result = module.walk([&](mlir::Operation* op) {
    // TODO(jiancai): we should check "_replication_info" attribute here instead
    // once the migration to unified compilation and replication markers is
    // done. See b/220150965 for more details.
    auto replicate_attr =
        op->getAttrOfType<mlir::StringAttr>(kTPUReplicateAttr);
    if (replicate_attr) return mlir::WalkResult::interrupt();
    return mlir::WalkResult::advance();
  });
  return walk_result.wasInterrupted();
}

// Checks that the module has both - TPU devices in its device list and contains
// TPU ops (identifed by `_tpu_replicate` attribute on ops).
bool HasTPUDevicesAndOps(mlir::ModuleOp module) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSmlir_bridge_passDTcc mht_2(mht_2_v, 247, "", "./tensorflow/compiler/tf2xla/mlir_bridge_pass.cc", "HasTPUDevicesAndOps");

  return HasTPUDevice(module) && HasTPUOp(module);
}

bool HasTPUDevice(const DeviceSet& device_set) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSmlir_bridge_passDTcc mht_3(mht_3_v, 254, "", "./tensorflow/compiler/tf2xla/mlir_bridge_pass.cc", "HasTPUDevice");

  for (const Device* device : device_set.devices()) {
    if (!device) continue;
    const DeviceNameUtils::ParsedName& name = device->parsed_name();
    if (name.has_type && name.type == "TPU") return true;
  }
  return false;
}

// Check if the `graph` has parameter serverjobs and resource variable arguments
// that are on parameter servers
bool HasPsWithResourceVariable(const Graph& graph) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSmlir_bridge_passDTcc mht_4(mht_4_v, 268, "", "./tensorflow/compiler/tf2xla/mlir_bridge_pass.cc", "HasPsWithResourceVariable");

  // Check parameter serverjobs and resource variable arguments that are
  // on parameter servers.
  const std::string jobType = "ps";
  const std::string nodeType = "_Arg";
  const std::string attrKey = "T";
  for (const Node* node : graph.nodes()) {
    if (node->type_string() == nodeType) {
      auto device_name = node->assigned_device_name();
      DeviceNameUtils::ParsedName device;
      if (DeviceNameUtils::ParseFullName(device_name, &device) &&
          device.has_job && device.job == jobType) {
        for (const auto& attr : node->attrs()) {
          auto attr_key = attr.first;
          auto attr_value = attr.second;
          if (attr_key == attrKey &&
              attr_value.value_case() == AttrValue::kType &&
              attr_value.type() == DT_RESOURCE) {
            return true;
            break;
          }
        }
      }
    }
  }
  return false;
}

// Check that graph has tf.StatefulPartitionedCall op with _XlaMustCompile.
bool HasQualifiedNonTPUOp(const Graph& graph) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSmlir_bridge_passDTcc mht_5(mht_5_v, 300, "", "./tensorflow/compiler/tf2xla/mlir_bridge_pass.cc", "HasQualifiedNonTPUOp");

  const std::string kStatefulPartitionedCallOp = "StatefulPartitionedCall";
  const std::string kXlaMustCompile = "_XlaMustCompile";
  for (const Node* node : graph.nodes()) {
    auto node_op = node->type_string();
    if (node_op == kStatefulPartitionedCallOp) {
      auto attr = node->attrs().FindByString(kXlaMustCompile);
      if (attr != nullptr && attr->b() == true) {
        return true;
      }
    }
  }
  return false;
}

// Check if non TPU pipeline should be used
bool EnableNonTpuBridge(const Graph& graph) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSmlir_bridge_passDTcc mht_6(mht_6_v, 319, "", "./tensorflow/compiler/tf2xla/mlir_bridge_pass.cc", "EnableNonTpuBridge");

  // Remark that this is staging change. It will be expanded later for further
  // check based on the requirement.
  return HasPsWithResourceVariable(graph) && HasQualifiedNonTPUOp(graph);
}

}  // namespace

// Analyzes the user requested policy as well as the contents of the graph and
// function_library_definition to determine whether the MLIR Bridge should be
// run.
//
// If the user explicitly requests the bridge be enabled or disabled, this
// function will respect the request. If the user does not explicitly request
// enabled or disabled, it will decide whether or not to run the bridge.
//
// The config_proto param is a required input for all TF1 graphs but it is
// redundant for TF2 graphs.
MlirOptimizationPassState MlirBridgePass::GetPassState(
    const DeviceSet* device_set, const ConfigProto& config_proto,
    const Graph& graph,
    const FunctionLibraryDefinition& function_library) const {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSmlir_bridge_passDTcc mht_7(mht_7_v, 343, "", "./tensorflow/compiler/tf2xla/mlir_bridge_pass.cc", "MlirBridgePass::GetPassState");

  // Skip MLIR TF XLA Bridge if no TPU devices found and the non TPU graph is
  // not qualified.
  if (device_set && !HasTPUDevice(*device_set)) {
    return EnableNonTpuBridge(graph) ? MlirOptimizationPassState::Enabled
                                     : MlirOptimizationPassState::Disabled;
  }

  // We set `uses_uninitialized_resource_args` to false here because the first
  // phase of the bridge is not affected by uninitialized resource args.
  MlirBridgeRolloutPolicy policy =
      GetMlirBridgeRolloutPolicy(graph, &function_library, config_proto,
                                 /*uses_uninitialized_resource_args=*/false);
  switch (policy) {
    case MlirBridgeRolloutPolicy::kEnabledByUser:
      return MlirOptimizationPassState::Enabled;
    case MlirBridgeRolloutPolicy::kEnabledAfterGraphAnalysis:
      return MlirOptimizationPassState::FallbackEnabled;
    case MlirBridgeRolloutPolicy::kEnabledAfterGraphAnalysisSafeModeFallback:
      return MlirOptimizationPassState::FallbackEnabled;
    case MlirBridgeRolloutPolicy::kDisabledByUser:
      VLOG(1) << "Skipping MLIR TPU Bridge, MLIR TPU bridge disabled by user. "
                 "Old bridge will evaluate.";
      metrics::UpdateTfMlirBridgeFirstPhaseCounter("tpu", "v2", true,
                                                   "disabled_by_user");
      return MlirOptimizationPassState::Disabled;
    case MlirBridgeRolloutPolicy::kDisabledAfterGraphAnalysis:
      VLOG(1) << "Skipping MLIR TPU Bridge, MLIR TPU bridge disabled because "
                 "graph has unsupported features. Old bridge will evaluate.";
      metrics::UpdateTfMlirBridgeFirstPhaseCounter("tpu", "v2", true,
                                                   "invalid_graph");
      return MlirOptimizationPassState::Disabled;
  }
}

// This runs the first phase of the "bridge", transforming the graph in a form
// that can be executed with delegation of some computations to an accelerator.
// This builds on the model of XLA where a subset of the graph is encapsulated
// and attached to a "compile" operation, whose result is fed to an "execute"
// operation. The kernel for these operations is responsible to lower the
// encapsulated graph to a particular device.
Status MlirBridgePass::Run(const ConfigProto& config_proto,
                           mlir::ModuleOp module, const Graph& graph,
                           const FunctionLibraryDefinition& function_library) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSmlir_bridge_passDTcc mht_8(mht_8_v, 389, "", "./tensorflow/compiler/tf2xla/mlir_bridge_pass.cc", "MlirBridgePass::Run");

  static absl::once_flag flag;
  absl::call_once(flag, UpdateLogVerbosityIfDefined, "TF_DEBUG_LOG_VERBOSITY");

  // Check if there are TPU devices or TPU ops. If not, then check if the
  // non TPU graph is qualified to run TF XLA Bridge.
  // This check needs to precede GetPassState for instrumentation purposes.
  if (!HasTPUDevicesAndOps(module)) {
    if (EnableNonTpuBridge(graph)) {
      VLOG(1) << "No TPU devices or TPU ops found, "
              << "this non TPU graph is qualified to run MLIR TF XLA Bridge";
      return mlir::TF::RunTFXLABridge(module, VLOG_IS_ON(1));
    } else {
      VLOG(1) << " Skipping MLIR TF XLA Bridge,"
              << " no TPU devices or TPU ops found, and this non TPU graph"
              << " is not qualified to run MLIR TF XLA Bridge.";
      return Status::OK();
    }
  }

  // Set device_set to nullptr here as the device specific checks are performed
  // based on the devices in the module.
  auto pass_state = GetPassState(/*device_set=*/nullptr, config_proto, graph,
                                 function_library);

  if (pass_state == MlirOptimizationPassState::Disabled) {
    // Currently the logging for handling the disabled case is in GetPassState
    // because it is called directly before run() and run() will not be called
    // if the pass is disabled.  This logic is here defenseively in case the
    // calling pass logic changes.
    VLOG(1) << "MlirBridgePass is disabled and will not run.";
    return Status::OK();
  }

  bool fallback_enabled = false;
  if (pass_state == MlirOptimizationPassState::FallbackEnabled)
    fallback_enabled = true;

  VLOG(1) << "Running MLIR TPU Bridge";

  mlir_bridge_gauge_v2->GetCell()->Set(true);
  return mlir::TFTPU::TPUBridge(module, /*enable_logging=*/VLOG_IS_ON(1),
                                fallback_enabled);
}

MlirOptimizationPassState MlirBridgeV1CompatPass::GetPassState(
    const DeviceSet* device_set, const ConfigProto& config_proto,
    const Graph& graph,
    const FunctionLibraryDefinition& function_library) const {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSmlir_bridge_passDTcc mht_9(mht_9_v, 440, "", "./tensorflow/compiler/tf2xla/mlir_bridge_pass.cc", "MlirBridgeV1CompatPass::GetPassState");

  // Skip MLIR TPU Bridge if no TPU devices found.
  if (device_set && !HasTPUDevice(*device_set))
    return MlirOptimizationPassState::Disabled;

  // Do not run the bridge if it's enabled by the graph analysis,
  // only run if it's enabled by the user explicitly.
  // We set `uses_uninitialized_resource_args` to false here because the first
  // phase of the bridge is not affected by uninitialized resource args.
  MlirBridgeRolloutPolicy policy = GetMlirBridgeRolloutPolicy(
      graph, /*function_library=*/&function_library, config_proto,
      /*uses_uninitialized_resource_args=*/false);
  switch (policy) {
    case MlirBridgeRolloutPolicy::kEnabledByUser:
      return MlirOptimizationPassState::Enabled;
    case MlirBridgeRolloutPolicy::kEnabledAfterGraphAnalysisSafeModeFallback:
      return MlirOptimizationPassState::FallbackEnabled;
    case MlirBridgeRolloutPolicy::kEnabledAfterGraphAnalysis:
      return MlirOptimizationPassState::FallbackEnabled;
    case MlirBridgeRolloutPolicy::kDisabledByUser:
      VLOG(1) << "Skipping MLIR TPU Bridge V1 Compat, MLIR TPU bridge disabled "
                 "by user. Old bridge will evaluate.";
      metrics::UpdateTfMlirBridgeFirstPhaseCounter("tpu", "v1", true,
                                                   "disabled_by_user");
      return MlirOptimizationPassState::Disabled;
    case MlirBridgeRolloutPolicy::kDisabledAfterGraphAnalysis:
      VLOG(1) << "Skipping MLIR TPU Bridge V1 Compat, MLIR TPU bridge disabled "
                 "because graph has unsupported features. Old bridge will "
                 "evaluate.";
      metrics::UpdateTfMlirBridgeFirstPhaseCounter("tpu", "v1", true,
                                                   "invalid_graph");
      return MlirOptimizationPassState::Disabled;
  }
}

Status MlirBridgeV1CompatPass::Run(const GraphOptimizationPassOptions& options,
                                   mlir::ModuleOp module) {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSmlir_bridge_passDTcc mht_10(mht_10_v, 479, "", "./tensorflow/compiler/tf2xla/mlir_bridge_pass.cc", "MlirBridgeV1CompatPass::Run");

  static absl::once_flag flag;
  absl::call_once(flag, UpdateLogVerbosityIfDefined, "TF_DEBUG_LOG_VERBOSITY");

  // Skip function graphs as MlirBridgePass will be used instead.
  if (options.is_function_graph) return Status::OK();

  // Skip MLIR TPU Bridge if no TPU devices or TPU ops found.
  if (!HasTPUDevicesAndOps(module)) {
    VLOG(1) << "Skipping MLIR TPU Bridge V1 Compat, no TPU devices or TPU ops "
               "found";
    return Status::OK();
  }

  MlirOptimizationPassState pass_state =
      GetPassState(/*device_set=*/nullptr, options.session_options->config,
                   **options.graph, *options.flib_def);

  // Set device_set to nullptr here as the device specific checks are performed
  // based on the devices in the module.
  if (pass_state == MlirOptimizationPassState::Disabled) {
    // Currently the logging for handling the disabled case is in GetPassState
    // because it is called directly before run() and run() will not be called
    // if the pass is disabled.  This logic is here defenseively in case the
    // calling pass logic changes.
    VLOG(1) << "Skipping MLIR TPU Bridge V1 Compat, session flag not enabled";
    mlir_bridge_gauge_v1->GetCell()->Set(false);
    return Status::OK();
  }

  VLOG(1) << "Running MLIR TPU Bridge V1 Compat";

  bool fallback_enabled = true;
  if (pass_state == MlirOptimizationPassState::Enabled)
    fallback_enabled = false;

  mlir_bridge_gauge_v1->GetCell()->Set(true);

  return mlir::TFTPU::TPUBridgeV1Compat(
      module, /*enable_logging=*/VLOG_IS_ON(1), fallback_enabled);
}

}  // namespace tensorflow
