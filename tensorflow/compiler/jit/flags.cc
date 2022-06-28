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
class MHTracer_DTPStensorflowPScompilerPSjitPSflagsDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSjitPSflagsDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSjitPSflagsDTcc() {
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

/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/jit/flags.h"

#include <mutex>  // NOLINT
#include <vector>

#include "absl/base/call_once.h"
#include "absl/strings/numbers.h"
#include "absl/strings/str_split.h"
#include "absl/strings/strip.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/dump_graph.h"
#include "tensorflow/compiler/xla/parse_flags_from_env.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/util/command_line_flags.h"

namespace tensorflow {
namespace {

BuildXlaOpsPassFlags* build_ops_flags;
MarkForCompilationPassFlags* mark_for_compilation_flags;
XlaDeviceFlags* device_flags;
XlaOpsCommonFlags* ops_flags;
IntroduceFloatingPointJitterPassFlags* jitter_flags;
MlirCommonFlags* mlir_flags;
JitRtFlags* jitrt_flags;
std::vector<Flag>* jitrt_flag_list;

std::vector<Flag>* flag_list;
absl::once_flag flags_init;

bool SetterForXlaAutoJitFlag(const string& value) {
   std::vector<std::string> mht_0_v;
   mht_0_v.push_back("value: \"" + value + "\"");
   MHTracer_DTPStensorflowPScompilerPSjitPSflagsDTcc mht_0(mht_0_v, 215, "", "./tensorflow/compiler/jit/flags.cc", "SetterForXlaAutoJitFlag");

  int32_t opt_level;
  // We need to use the mark_for_compilation_flags directly here instead of
  // going via GetMarkForCompilationPassFlags() to avoid infinite recursion. The
  // latter will try to setup and parse flags, which would bring us back to this
  // setter.
  if (absl::SimpleAtoi(value, &opt_level)) {
    mark_for_compilation_flags->xla_auto_jit_flag
        .optimization_level_single_gpu = opt_level;
    mark_for_compilation_flags->xla_auto_jit_flag.optimization_level_general =
        opt_level;
    return true;
  }

  if (value == "fusible") {
    mark_for_compilation_flags->xla_auto_jit_flag
        .optimization_level_single_gpu = 1;
    mark_for_compilation_flags->xla_auto_jit_flag.optimization_level_general =
        1;
    mark_for_compilation_flags->tf_xla_ops_to_cluster = "FUSIBLE";
    return true;
  }

  absl::string_view value_sv(value);
  if (!absl::ConsumePrefix(&value_sv, "single-gpu(") ||
      !absl::ConsumeSuffix(&value_sv, ")") ||
      !absl::SimpleAtoi(value_sv, &opt_level)) {
    return false;
  }

  mark_for_compilation_flags->xla_auto_jit_flag.optimization_level_single_gpu =
      opt_level;
  return true;
}

void AppendMarkForCompilationPassFlagsInternal(std::vector<Flag>* flag_list) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSjitPSflagsDTcc mht_1(mht_1_v, 253, "", "./tensorflow/compiler/jit/flags.cc", "AppendMarkForCompilationPassFlagsInternal");

  std::vector<Flag> new_flags = {
      Flag("tf_xla_auto_jit", SetterForXlaAutoJitFlag, "0",
           "Control compilation of operators into XLA computations on CPU and "
           "GPU devices.  0 = use ConfigProto setting; -1 = off; 1 = on for "
           "things very likely to be improved; 2 = on for everything; "
           "(experimental) fusible = only for Tensorflow operations that XLA "
           "knows how to fuse.  "
           "If set to single-gpu(<N>) then this resolves to <N> for single-GPU "
           "graphs (graphs that have at least one node placed on a GPU and no "
           "more than one GPU is in use through the entire graph) and 0 "
           "otherwise.  Experimental."),
      Flag("tf_xla_min_cluster_size",
           &mark_for_compilation_flags->tf_xla_min_cluster_size,
           "Minimum number of operators in an XLA compilation. Ignored for "
           "operators placed on an XLA device or operators explicitly marked "
           "for compilation."),
      Flag("tf_xla_max_cluster_size",
           &mark_for_compilation_flags->tf_xla_max_cluster_size,
           "Maximum number of operators in an XLA compilation."),
      Flag(
          "tf_xla_ops_to_cluster",
          &mark_for_compilation_flags->tf_xla_ops_to_cluster,
          "(experimental) "
          "Limit the operations clustered by XLA to these operations. "
          "If multiple, separate them with commas. Shortcuts: "
          " PW: All point-wise operations."
          " RED: All reduction operations."
          " MISC: Mixed operations."
          " PWRED: TF operations that get converted to PW+RED operation in XLA."
          " REDUCEWINDOW: TF operations like MaxPool/AvgPool that get "
          "converted to ReduceWindow in XLA."
          " REDUCEWINDOWPW: Operation that get converted to ReduceWindow + PW "
          "(LRN, LRNGrad)."
          " BN: TF FusedBatchNorm* operations."
          " FUSIBLE: All TF operations that XLA can fuse (All the above). "
          "You can also put any TF operation name, e.g. 'FUSIBLE,MatMul'."),
      Flag("tf_xla_clustering_debug",
           &mark_for_compilation_flags->tf_xla_clustering_debug,
           "Dump graphs during XLA compilation."),
      Flag("tf_xla_cpu_global_jit",
           &mark_for_compilation_flags->tf_xla_cpu_global_jit,
           "Enables global JIT compilation for CPU via SessionOptions."),
      Flag("tf_xla_clustering_fuel",
           &mark_for_compilation_flags->tf_xla_clustering_fuel,
           "Places an artificial limit on the number of ops marked as "
           "eligible for clustering."),
      Flag("tf_xla_disable_deadness_safety_checks_for_debugging",
           &mark_for_compilation_flags
                ->tf_xla_disable_deadness_safety_checks_for_debugging,
           "Disable deadness related safety checks when clustering (this is "
           "unsound)."),
      Flag("tf_xla_disable_resource_variable_safety_checks_for_debugging",
           &mark_for_compilation_flags
                ->tf_xla_disable_resource_variable_safety_checks_for_debugging,
           "Disable resource variables related safety checks when clustering "
           "(this is unsound)."),
      Flag("tf_xla_deterministic_cluster_names",
           &mark_for_compilation_flags->tf_xla_deterministic_cluster_names,
           "Causes the function names assigned by auto clustering to be "
           "deterministic from run to run."),
      Flag("tf_xla_persistent_cache_directory",
           &mark_for_compilation_flags->tf_xla_persistent_cache_directory,
           "If non-empty, JIT-compiled executables are saved to and loaded "
           "from the specified file system directory path. Empty by default."),
      Flag("tf_xla_disable_strict_signature_checks",
           &mark_for_compilation_flags->tf_xla_disable_strict_signature_checks,
           "If true, entires loaded into the XLA compile cache will not have "
           "their signatures checked strictly. Defaults to false."),
      Flag("tf_xla_persistent_cache_prefix",
           &mark_for_compilation_flags->tf_xla_persistent_cache_prefix,
           "Specifies the persistance cache prefix. Default is "
           "\"xla_compile_cache\"")};
  flag_list->insert(flag_list->end(), new_flags.begin(), new_flags.end());
}

void AllocateAndParseJitRtFlags() {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSjitPSflagsDTcc mht_2(mht_2_v, 332, "", "./tensorflow/compiler/jit/flags.cc", "AllocateAndParseJitRtFlags");

  jitrt_flags = new JitRtFlags;
  jitrt_flags->always_specialize = false;
  jitrt_flags->cost_driven_async_parallel_for = false;
  jitrt_flags->vectorize = false;
  jitrt_flag_list = new std::vector<Flag>({
      Flag("always_specialize", &jitrt_flags->always_specialize, ""),
      Flag("cost_driven_async_parallel_for",
           &jitrt_flags->cost_driven_async_parallel_for, ""),
      Flag("vectorize", &jitrt_flags->vectorize, ""),
  });
  xla::ParseFlagsFromEnvAndDieIfUnknown("TF_JITRT_FLAGS", *jitrt_flag_list);
}

void AllocateAndParseFlags() {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPSjitPSflagsDTcc mht_3(mht_3_v, 349, "", "./tensorflow/compiler/jit/flags.cc", "AllocateAndParseFlags");

  build_ops_flags = new BuildXlaOpsPassFlags;
  build_ops_flags->tf_xla_enable_lazy_compilation = true;
  build_ops_flags->tf_xla_print_cluster_outputs = false;
  build_ops_flags->tf_xla_check_cluster_input_numerics = false;
  build_ops_flags->tf_xla_check_cluster_output_numerics = false;
  build_ops_flags->tf_xla_disable_constant_folding = false;

  mark_for_compilation_flags = new MarkForCompilationPassFlags;
  mark_for_compilation_flags->xla_auto_jit_flag.optimization_level_single_gpu =
      0;
  mark_for_compilation_flags->xla_auto_jit_flag.optimization_level_general = 0;
  mark_for_compilation_flags->tf_xla_min_cluster_size = 4;
  mark_for_compilation_flags->tf_xla_max_cluster_size =
      std::numeric_limits<int32>::max();
  mark_for_compilation_flags->tf_xla_clustering_debug = false;
  mark_for_compilation_flags->tf_xla_cpu_global_jit = false;
  mark_for_compilation_flags->tf_xla_clustering_fuel =
      std::numeric_limits<int64_t>::max();
  mark_for_compilation_flags
      ->tf_xla_disable_deadness_safety_checks_for_debugging = false;
  mark_for_compilation_flags
      ->tf_xla_disable_resource_variable_safety_checks_for_debugging = false;
  mark_for_compilation_flags->tf_xla_deterministic_cluster_names = false;
  mark_for_compilation_flags->tf_xla_persistent_cache_directory = "";
  mark_for_compilation_flags->tf_xla_disable_strict_signature_checks = false;
  mark_for_compilation_flags->tf_xla_persistent_cache_prefix =
      "xla_compile_cache";

  device_flags = new XlaDeviceFlags;
  device_flags->tf_xla_compile_on_demand = false;
  device_flags->tf_xla_enable_xla_devices = false;

  ops_flags = new XlaOpsCommonFlags;
  ops_flags->tf_xla_always_defer_compilation = false;
  ops_flags->tf_xla_async_compilation = false;

  jitter_flags = new IntroduceFloatingPointJitterPassFlags;
  jitter_flags->jitter_amount = 1e-5;

  // The `enable_mlir_bridge` flag allows the user to explicitly request that
  // their program is (or isn't) compiled using the MLIR-based TF-to-XLA bridge.
  //
  // The `enable_mlir_bridge_is_explicit` variable tracks whether or not the
  // user has made an explicit request. That is, if this variable is set to
  // true, the program honors the user's request as per `enable_mlir_bridge`; if
  // it's set to false, the default behavior is used (which may run either
  // bridge, on a per-graph basis).
  bool enable_mlir_bridge = false;
  bool enable_mlir_bridge_is_explicit = false;
  bool mlir_bridge_safe_mode = false;
  bool enable_mlir_merge_control_flow_pass = true;
  bool enable_mlir_convert_control_to_data_outputs_pass = false;
  auto setter_for_jitter_tensor_names = [](string sequence) {
   std::vector<std::string> mht_4_v;
   mht_4_v.push_back("sequence: \"" + sequence + "\"");
   MHTracer_DTPStensorflowPScompilerPSjitPSflagsDTcc mht_4(mht_4_v, 406, "", "./tensorflow/compiler/jit/flags.cc", "lambda");

    jitter_flags->tensor_names = absl::StrSplit(sequence, ',');
    return true;
  };
  // Dump graphs in TFG dialect.
  bool use_tfg_graph_dumper = false;

  flag_list = new std::vector<Flag>(
      {Flag("tf_xla_enable_lazy_compilation",
            &build_ops_flags->tf_xla_enable_lazy_compilation, ""),
       Flag("tf_xla_print_cluster_outputs",
            &build_ops_flags->tf_xla_print_cluster_outputs,
            "If true then insert Print nodes to print out values produced by "
            "XLA clusters."),
       Flag("tf_xla_check_cluster_input_numerics",
            &build_ops_flags->tf_xla_check_cluster_input_numerics,
            "If true then insert CheckNumerics nodes to check all cluster "
            "inputs."),
       Flag("tf_xla_check_cluster_output_numerics",
            &build_ops_flags->tf_xla_check_cluster_output_numerics,
            "If true then insert CheckNumerics nodes to check all cluster "
            "outputs."),
       Flag("tf_xla_disable_constant_folding",
            &build_ops_flags->tf_xla_disable_constant_folding,
            "If true then disables constant folding on TF graph before XLA "
            "compilation."),

       Flag("tf_xla_compile_on_demand", &device_flags->tf_xla_compile_on_demand,
            "Switch a device into 'on-demand' mode, where instead of "
            "autoclustering ops are compiled one by one just-in-time."),

       Flag("tf_xla_enable_xla_devices",
            &device_flags->tf_xla_enable_xla_devices,
            "Generate XLA_* devices, where placing a computation on such a "
            "device"
            "forces compilation by XLA. Deprecated."),

       Flag("tf_xla_always_defer_compilation",
            &ops_flags->tf_xla_always_defer_compilation, ""),
       Flag("tf_xla_async_compilation", &ops_flags->tf_xla_async_compilation,
            "When lazy compilation is enabled, asynchronous compilation starts "
            "the cluster compilation in the background, and the fallback path "
            "is executed until the compilation has finished."),

       Flag("tf_introduce_floating_point_jitter_to_tensors",
            setter_for_jitter_tensor_names, "",
            "The Tensors to add the jitter to.  The tensors are named in the "
            "TensorId format of <node name>:<output idx>."),
       Flag("tf_introduce_floating_point_jitter_amount",
            &jitter_flags->jitter_amount,
            "The amount of jitter to introduce.  This amount is added to each "
            "element in the tensors named in `tensor_names."),

       Flag("tf_mlir_enable_mlir_bridge", &enable_mlir_bridge,
            "Enables experimental MLIR-Based TensorFlow Compiler Bridge.",
            &enable_mlir_bridge_is_explicit),
       Flag("tf_mlir_enable_merge_control_flow_pass",
            &enable_mlir_merge_control_flow_pass,
            "Enables MergeControlFlow pass for MLIR-Based TensorFlow Compiler "
            "Bridge."),
       Flag("tf_mlir_enable_convert_control_to_data_outputs_pass",
            &enable_mlir_convert_control_to_data_outputs_pass,
            "Enables `tf-executor-convert-control-to-data-outputs` pass for "
            "MLIR-Based TensorFlow Compiler Bridge."),
       Flag(
           "tf_mlir_bridge_safe_mode", &mlir_bridge_safe_mode,
           "When tf_mlir_enable_mlir_bridge is true, this field can enable "
           "the MLIR bridge's safe mode. When the MLIR bridge is in safe mode, "
           "it only runs for graphs that use features MLIR bridge currently "
           "supports."),
       Flag("tf_dump_graphs_in_tfg", &use_tfg_graph_dumper,
            "When tf_dump_graphs_in_tfg is true, graphs after transformations "
            "are dumped in MLIR TFG dialect and not in GraphDef")});

  AppendMarkForCompilationPassFlagsInternal(flag_list);
  xla::ParseFlagsFromEnvAndDieIfUnknown("TF_XLA_FLAGS", *flag_list);

  mlir_flags = new MlirCommonFlags;
  if (!enable_mlir_bridge_is_explicit) {
    mlir_flags->tf_mlir_enable_mlir_bridge =
        (mlir_bridge_safe_mode)
            ? ConfigProto::Experimental::
                  MLIR_BRIDGE_ROLLOUT_SAFE_MODE_FALLBACK_ENABLED
            : ConfigProto::Experimental::MLIR_BRIDGE_ROLLOUT_UNSPECIFIED;
  } else if (enable_mlir_bridge) {
    mlir_flags->tf_mlir_enable_mlir_bridge =
        (mlir_bridge_safe_mode)
            ? ConfigProto::Experimental::MLIR_BRIDGE_ROLLOUT_SAFE_MODE_ENABLED
            : ConfigProto::Experimental::MLIR_BRIDGE_ROLLOUT_ENABLED;
  } else {
    mlir_flags->tf_mlir_enable_mlir_bridge =
        ConfigProto::Experimental::MLIR_BRIDGE_ROLLOUT_DISABLED;
  }
  mlir_flags->tf_mlir_enable_merge_control_flow_pass =
      enable_mlir_merge_control_flow_pass;
  mlir_flags->tf_mlir_enable_convert_control_to_data_outputs_pass =
      enable_mlir_convert_control_to_data_outputs_pass;

  if (use_tfg_graph_dumper) {
    UseMlirForGraphDump(MlirDumpConfig{}.elide_large_attributes().emit_dialect(
        MlirDumpConfig::Dialect::kTFG));
  }

  AllocateAndParseJitRtFlags();
}

void ResetFlags() {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScompilerPSjitPSflagsDTcc mht_5(mht_5_v, 515, "", "./tensorflow/compiler/jit/flags.cc", "ResetFlags");

  delete build_ops_flags;
  delete mark_for_compilation_flags;
  delete device_flags;
  delete ops_flags;
  delete jitter_flags;
  delete mlir_flags;
  delete flag_list;
  delete jitrt_flags;
  delete jitrt_flag_list;
  AllocateAndParseFlags();
}

}  // namespace

bool SetXlaAutoJitFlagFromFlagString(const string& value) {
   std::vector<std::string> mht_6_v;
   mht_6_v.push_back("value: \"" + value + "\"");
   MHTracer_DTPStensorflowPScompilerPSjitPSflagsDTcc mht_6(mht_6_v, 534, "", "./tensorflow/compiler/jit/flags.cc", "SetXlaAutoJitFlagFromFlagString");

  absl::call_once(flags_init, &AllocateAndParseFlags);
  return SetterForXlaAutoJitFlag(value);
}

BuildXlaOpsPassFlags* GetBuildXlaOpsPassFlags() {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScompilerPSjitPSflagsDTcc mht_7(mht_7_v, 542, "", "./tensorflow/compiler/jit/flags.cc", "GetBuildXlaOpsPassFlags");

  absl::call_once(flags_init, &AllocateAndParseFlags);
  return build_ops_flags;
}

MarkForCompilationPassFlags* GetMarkForCompilationPassFlags() {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScompilerPSjitPSflagsDTcc mht_8(mht_8_v, 550, "", "./tensorflow/compiler/jit/flags.cc", "GetMarkForCompilationPassFlags");

  absl::call_once(flags_init, &AllocateAndParseFlags);
  return mark_for_compilation_flags;
}

XlaDeviceFlags* GetXlaDeviceFlags() {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScompilerPSjitPSflagsDTcc mht_9(mht_9_v, 558, "", "./tensorflow/compiler/jit/flags.cc", "GetXlaDeviceFlags");

  absl::call_once(flags_init, &AllocateAndParseFlags);
  return device_flags;
}

const XlaOpsCommonFlags& GetXlaOpsCommonFlags() {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScompilerPSjitPSflagsDTcc mht_10(mht_10_v, 566, "", "./tensorflow/compiler/jit/flags.cc", "GetXlaOpsCommonFlags");

  absl::call_once(flags_init, &AllocateAndParseFlags);
  return *ops_flags;
}

const IntroduceFloatingPointJitterPassFlags&
GetIntroduceFloatingPointJitterPassFlags() {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScompilerPSjitPSflagsDTcc mht_11(mht_11_v, 575, "", "./tensorflow/compiler/jit/flags.cc", "GetIntroduceFloatingPointJitterPassFlags");

  absl::call_once(flags_init, &AllocateAndParseFlags);
  return *jitter_flags;
}

MlirCommonFlags* GetMlirCommonFlags() {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScompilerPSjitPSflagsDTcc mht_12(mht_12_v, 583, "", "./tensorflow/compiler/jit/flags.cc", "GetMlirCommonFlags");

  absl::call_once(flags_init, &AllocateAndParseFlags);
  return mlir_flags;
}

void ResetJitCompilerFlags() {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPScompilerPSjitPSflagsDTcc mht_13(mht_13_v, 591, "", "./tensorflow/compiler/jit/flags.cc", "ResetJitCompilerFlags");
 ResetFlags(); }

const JitRtFlags& GetJitRtFlags() {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPScompilerPSjitPSflagsDTcc mht_14(mht_14_v, 596, "", "./tensorflow/compiler/jit/flags.cc", "GetJitRtFlags");

  absl::call_once(flags_init, &AllocateAndParseFlags);
  return *jitrt_flags;
}

ConfigProto::Experimental::MlirBridgeRollout GetMlirBridgeRolloutState(
    absl::optional<const ConfigProto> config_proto) {
   std::vector<std::string> mht_15_v;
   MHTracer_DTPStensorflowPScompilerPSjitPSflagsDTcc mht_15(mht_15_v, 605, "", "./tensorflow/compiler/jit/flags.cc", "GetMlirBridgeRolloutState");

  // TF1 graphs that do not override Sessions's ConfigProto and TF2 graphs
  // can enable/disable the graph via tf_mlir_enable_mlir_bridge.
  auto tf_mlir_enable_mlir_bridge =
      GetMlirCommonFlags()->tf_mlir_enable_mlir_bridge;
  if (tf_mlir_enable_mlir_bridge !=
      ConfigProto::Experimental::MLIR_BRIDGE_ROLLOUT_UNSPECIFIED) {
    return tf_mlir_enable_mlir_bridge;
  }

  // If a ConfigProto was not passed in, we can assume the caller is
  // checking if TF2 graph should have the bridge enabled / disabled. In that
  // case, we have already checked tf_mlir_enable_mlir_bridge so it is safe to
  // return UNSPECIFIED here.
  if (!config_proto.has_value()) {
    return ConfigProto::Experimental::MLIR_BRIDGE_ROLLOUT_UNSPECIFIED;
  }

  // TF1 graphs that do override Session's ConfigProto and set
  // ConfigProto's enable_mlir_bridge or mlir_bridge_rollout fields will not
  // update tf_mlir_enable_mlir_bridge so check their values.

  // ConfigProto's enable_mlir_bridge defaults to false so only respect it
  // when it is true.
  if (config_proto.value().experimental().enable_mlir_bridge()) {
    return ConfigProto::Experimental::MLIR_BRIDGE_ROLLOUT_ENABLED;
  }
  return config_proto.value().experimental().mlir_bridge_rollout();
}

void AppendMarkForCompilationPassFlags(std::vector<Flag>* flag_list) {
   std::vector<std::string> mht_16_v;
   MHTracer_DTPStensorflowPScompilerPSjitPSflagsDTcc mht_16(mht_16_v, 638, "", "./tensorflow/compiler/jit/flags.cc", "AppendMarkForCompilationPassFlags");

  absl::call_once(flags_init, &AllocateAndParseFlags);
  AppendMarkForCompilationPassFlagsInternal(flag_list);
}

static std::atomic<bool> xla_compilation_disabled(false);

void DisableXlaCompilation() {
   std::vector<std::string> mht_17_v;
   MHTracer_DTPStensorflowPScompilerPSjitPSflagsDTcc mht_17(mht_17_v, 648, "", "./tensorflow/compiler/jit/flags.cc", "DisableXlaCompilation");
 xla_compilation_disabled = true; }

bool FailOnXlaCompilation() {
   std::vector<std::string> mht_18_v;
   MHTracer_DTPStensorflowPScompilerPSjitPSflagsDTcc mht_18(mht_18_v, 653, "", "./tensorflow/compiler/jit/flags.cc", "FailOnXlaCompilation");
 return xla_compilation_disabled; }

}  // namespace tensorflow
