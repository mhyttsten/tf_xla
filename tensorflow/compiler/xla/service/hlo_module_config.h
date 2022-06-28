/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_HLO_MODULE_CONFIG_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_HLO_MODULE_CONFIG_H_
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
class MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_module_configDTh {
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
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_module_configDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_module_configDTh() {
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


#include <string>

#include "absl/container/flat_hash_map.h"
#include "absl/strings/string_view.h"
#include "absl/types/optional.h"
#include "tensorflow/compiler/xla/debug_options_flags.h"
#include "tensorflow/compiler/xla/service/computation_layout.h"
#include "tensorflow/compiler/xla/service/computation_placer.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/compiler/xla/xla.pb.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"

namespace xla {

enum class FusionConfigCollection {
  kOff,      // Do not collect configuration.
  kPerEdge,  // Collect per-edge configuration.
  kPerNode,  // Collect per-node configuration.
};

// This class gathers all settings and values which affect the compiled
// executable outside of the HLO code itself. This include layouts of inputs and
// outputs to the module and settings such as HLO profiling. Together the
// HloModule and HloModuleConfig unambiguously determine a particular
// executable.
class HloModuleConfig {
 public:
  // Represents a pair of input and output of the entry computation that can be
  // considered as the original and updated values of a variable maintained by
  // the caller, and that can be transparently sharded by XLA as an internal
  // optimization. If sharded, XLA will create separate sharding/unsharding
  // programs, and the caller is responsible to call the XLA-generated
  // sharding/unsharding programs before and after the sharded main program.
  //
  // If the variable is not updated and there is not a corresponding output, use
  // {-1} as the output_shape_index.
  //
  // The sharding/unsharding programs will include all the input/output pairs in
  // shardable_value_update_pairs() as a flat tuple in their inputs/outputs,
  // sorted by (input_parameter_number, parameter_shape_index).
  //
  // A typical usage pattern is to shard the variables first, then repeatedly
  // invoke the main program, and finally invoke the unsharding program before
  // they are used in full-shape.
  struct ShardableValueUpdatePair {
    int64_t input_parameter_number;
    ShapeIndex parameter_shape_index;
    ShapeIndex output_shape_index;
  };

  // A configuration can be created either with, or without an entry
  // ComputationLayout. The default ctor creates it without -- in this case
  // accessing entry_computation_layout will CHECK-fail. The ctor accepting a
  // ProgramShape creates a computation layout using this shape.
  // The layouts in the ProgramShape will be reset to default unless
  // ignore_layouts is set to false.
  HloModuleConfig() {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_module_configDTh mht_0(mht_0_v, 244, "", "./tensorflow/compiler/xla/service/hlo_module_config.h", "HloModuleConfig");
 debug_options_ = DefaultDebugOptionsIgnoringFlags(); }

  explicit HloModuleConfig(const ProgramShape& program_shape,
                           bool ignore_layouts = true);

  explicit HloModuleConfig(ComputationLayout entry_computation_layout);

  // Checks if this config has an entry computation layout already.
  bool has_entry_computation_layout() const {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_module_configDTh mht_1(mht_1_v, 255, "", "./tensorflow/compiler/xla/service/hlo_module_config.h", "has_entry_computation_layout");

    return entry_computation_layout_.has_value();
  }

  // Sets the entry_computation_layout's parameter and result shapes for this
  // config, according to the given program shape. The parameters and result
  // are set to default layout.
  void SetDefaultComputationLayout(const ProgramShape& program_shape);

  // Same as above but if the given program contains layout for parameters or
  // result, the entry_computation_layout's layout is updated accordingly.
  void SetComputationLayoutIfExists(const ProgramShape& program_shape);

  // Returns a constant reference to the layout of the entry computation.
  // Assumes the layout was set.
  const ComputationLayout& entry_computation_layout() const {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_module_configDTh mht_2(mht_2_v, 273, "", "./tensorflow/compiler/xla/service/hlo_module_config.h", "entry_computation_layout");

    CHECK(entry_computation_layout_.has_value());
    return *entry_computation_layout_;
  }

  // Returns a mutable pointer to the layout of the entry computation.
  // Assumes the layout was set.
  ComputationLayout* mutable_entry_computation_layout() {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_module_configDTh mht_3(mht_3_v, 283, "", "./tensorflow/compiler/xla/service/hlo_module_config.h", "mutable_entry_computation_layout");

    CHECK(entry_computation_layout_.has_value());
    return &(*entry_computation_layout_);
  }

  // Returns whether to enable HLO-level profiling.
  bool hlo_profiling_enabled() const {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_module_configDTh mht_4(mht_4_v, 292, "", "./tensorflow/compiler/xla/service/hlo_module_config.h", "hlo_profiling_enabled");

    return debug_options_.xla_hlo_profile();
  }

  bool cpu_traceme_enabled() const {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_module_configDTh mht_5(mht_5_v, 299, "", "./tensorflow/compiler/xla/service/hlo_module_config.h", "cpu_traceme_enabled");

    return debug_options_.xla_cpu_enable_xprof_traceme();
  }

  // Sets/returns the module seed set during execution.
  void set_seed(uint64_t seed) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_module_configDTh mht_6(mht_6_v, 307, "", "./tensorflow/compiler/xla/service/hlo_module_config.h", "set_seed");
 seed_ = seed; }
  uint64_t seed() const {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_module_configDTh mht_7(mht_7_v, 311, "", "./tensorflow/compiler/xla/service/hlo_module_config.h", "seed");
 return seed_; }

  // Set the launch id of the program. Launch id identifies a set of programs
  // that should be launched together.
  void set_launch_id(uint64_t launch_id) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_module_configDTh mht_8(mht_8_v, 318, "", "./tensorflow/compiler/xla/service/hlo_module_config.h", "set_launch_id");
 launch_id_ = launch_id; }

  int32_t launch_id() const {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_module_configDTh mht_9(mht_9_v, 323, "", "./tensorflow/compiler/xla/service/hlo_module_config.h", "launch_id");
 return launch_id_; }

  void set_replica_count(int64_t replica_count) {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_module_configDTh mht_10(mht_10_v, 328, "", "./tensorflow/compiler/xla/service/hlo_module_config.h", "set_replica_count");

    replica_count_ = replica_count;
  }
  int64_t replica_count() const {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_module_configDTh mht_11(mht_11_v, 334, "", "./tensorflow/compiler/xla/service/hlo_module_config.h", "replica_count");
 return replica_count_; }

  void set_num_partitions(int64_t num_partitions) {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_module_configDTh mht_12(mht_12_v, 339, "", "./tensorflow/compiler/xla/service/hlo_module_config.h", "set_num_partitions");

    num_partitions_ = num_partitions;
  }
  int64_t num_partitions() const {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_module_configDTh mht_13(mht_13_v, 345, "", "./tensorflow/compiler/xla/service/hlo_module_config.h", "num_partitions");
 return num_partitions_; }

  const std::vector<bool> param_requires_broadcast_via_collectives() const {
    return param_requires_broadcast_via_collectives_;
  }
  void set_param_requires_broadcast_via_collectives(
      const std::vector<bool> require_broadcast) {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_module_configDTh mht_14(mht_14_v, 354, "", "./tensorflow/compiler/xla/service/hlo_module_config.h", "set_param_requires_broadcast_via_collectives");

    param_requires_broadcast_via_collectives_ = std::move(require_broadcast);
  }

  void set_use_spmd_partitioning(bool use_spmd_partitioning) {
   std::vector<std::string> mht_15_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_module_configDTh mht_15(mht_15_v, 361, "", "./tensorflow/compiler/xla/service/hlo_module_config.h", "set_use_spmd_partitioning");

    use_spmd_partitioning_ = use_spmd_partitioning;
  }
  bool use_spmd_partitioning() const {
   std::vector<std::string> mht_16_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_module_configDTh mht_16(mht_16_v, 367, "", "./tensorflow/compiler/xla/service/hlo_module_config.h", "use_spmd_partitioning");
 return use_spmd_partitioning_; }

  void set_use_auto_spmd_partitioning(bool use_auto_spmd_partitioning) {
   std::vector<std::string> mht_17_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_module_configDTh mht_17(mht_17_v, 372, "", "./tensorflow/compiler/xla/service/hlo_module_config.h", "set_use_auto_spmd_partitioning");

    use_auto_spmd_partitioning_ = use_auto_spmd_partitioning;
    if (use_auto_spmd_partitioning) {
      // TODO(yuemmawang) Remove this warning once auto sharding is thoroughly
      // tested with fleetwide models.
      LOG(WARNING) << "Warning: Using auto_spmd_partitioning. It is "
                      "experimental and may "
                      "contain bugs!";
      LOG(INFO) << "Overwriting use_spmd_partitioning to true, because "
                   "use_auto_spmd_partitioning is true.";
      set_use_spmd_partitioning(true);
    }
  }
  bool use_auto_spmd_partitioning() const {
   std::vector<std::string> mht_18_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_module_configDTh mht_18(mht_18_v, 388, "", "./tensorflow/compiler/xla/service/hlo_module_config.h", "use_auto_spmd_partitioning");

    return use_auto_spmd_partitioning_;
  }

  // If enabled, deduplicate equivalent hlos into function calls to reduce code
  // size.
  void set_deduplicate_hlo(bool deduplicate_hlo) {
   std::vector<std::string> mht_19_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_module_configDTh mht_19(mht_19_v, 397, "", "./tensorflow/compiler/xla/service/hlo_module_config.h", "set_deduplicate_hlo");

    deduplicate_hlo_ = deduplicate_hlo;
  }

  void set_device_type(const std::string& device_type) {
   std::vector<std::string> mht_20_v;
   mht_20_v.push_back("device_type: \"" + device_type + "\"");
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_module_configDTh mht_20(mht_20_v, 405, "", "./tensorflow/compiler/xla/service/hlo_module_config.h", "set_device_type");

    device_type_ = device_type;
  }

  bool deduplicate_hlo() const {
   std::vector<std::string> mht_21_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_module_configDTh mht_21(mht_21_v, 412, "", "./tensorflow/compiler/xla/service/hlo_module_config.h", "deduplicate_hlo");
 return deduplicate_hlo_; }

  // Return a string which unambiguously represents all the fields of this data
  // structure. Used for generating a cache key for storing the compiled
  // executable.
  std::string compilation_cache_key() const;

  std::string device_type() const {
   std::vector<std::string> mht_22_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_module_configDTh mht_22(mht_22_v, 422, "", "./tensorflow/compiler/xla/service/hlo_module_config.h", "device_type");
 return device_type_; }

  const DebugOptions& debug_options() const {
   std::vector<std::string> mht_23_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_module_configDTh mht_23(mht_23_v, 427, "", "./tensorflow/compiler/xla/service/hlo_module_config.h", "debug_options");
 return debug_options_; }

  void set_debug_options(const DebugOptions& debug_options) {
   std::vector<std::string> mht_24_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_module_configDTh mht_24(mht_24_v, 432, "", "./tensorflow/compiler/xla/service/hlo_module_config.h", "set_debug_options");

    debug_options_ = debug_options;
  }

  // Sets/returns the number of intra op threads for this module.
  void set_intra_op_parallelism_threads(
      const int intra_op_parallelism_threads) {
   std::vector<std::string> mht_25_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_module_configDTh mht_25(mht_25_v, 441, "", "./tensorflow/compiler/xla/service/hlo_module_config.h", "set_intra_op_parallelism_threads");

    intra_op_parallelism_threads_ = intra_op_parallelism_threads;
  }
  int64_t intra_op_parallelism_threads() const {
   std::vector<std::string> mht_26_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_module_configDTh mht_26(mht_26_v, 447, "", "./tensorflow/compiler/xla/service/hlo_module_config.h", "intra_op_parallelism_threads");

    return intra_op_parallelism_threads_;
  }

  // Checks if this config has a static device assignment.
  bool has_static_device_assignment() const {
   std::vector<std::string> mht_27_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_module_configDTh mht_27(mht_27_v, 455, "", "./tensorflow/compiler/xla/service/hlo_module_config.h", "has_static_device_assignment");

    return static_device_assignment_.has_value();
  }

  // Getter and setter of the compile-time known device assignment.
  const DeviceAssignment& static_device_assignment() const {
   std::vector<std::string> mht_28_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_module_configDTh mht_28(mht_28_v, 463, "", "./tensorflow/compiler/xla/service/hlo_module_config.h", "static_device_assignment");

    CHECK(static_device_assignment_.has_value());
    return *static_device_assignment_;
  }
  void set_static_device_assignment(const DeviceAssignment& device_assignment) {
   std::vector<std::string> mht_29_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_module_configDTh mht_29(mht_29_v, 470, "", "./tensorflow/compiler/xla/service/hlo_module_config.h", "set_static_device_assignment");

    static_device_assignment_ = device_assignment;
  }

  const std::vector<ShardableValueUpdatePair> shardable_value_update_pairs()
      const {
    return shardable_value_update_pairs_;
  }
  void set_shardable_value_update_pairs(
      std::vector<ShardableValueUpdatePair> pairs) {
   std::vector<std::string> mht_30_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_module_configDTh mht_30(mht_30_v, 482, "", "./tensorflow/compiler/xla/service/hlo_module_config.h", "set_shardable_value_update_pairs");

    shardable_value_update_pairs_ = std::move(pairs);
  }

  // Whether input and output buffers are aliased if the associated parameter is
  // passed-through XLA modules without being changed.
  bool alias_passthrough_params() const {
   std::vector<std::string> mht_31_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_module_configDTh mht_31(mht_31_v, 491, "", "./tensorflow/compiler/xla/service/hlo_module_config.h", "alias_passthrough_params");
 return alias_passthrough_params_; }
  void set_alias_passthrough_params(bool alias_passthrough_params) {
   std::vector<std::string> mht_32_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_module_configDTh mht_32(mht_32_v, 495, "", "./tensorflow/compiler/xla/service/hlo_module_config.h", "set_alias_passthrough_params");

    alias_passthrough_params_ = alias_passthrough_params;
  }

  bool content_aware_computation_sorting() const {
   std::vector<std::string> mht_33_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_module_configDTh mht_33(mht_33_v, 502, "", "./tensorflow/compiler/xla/service/hlo_module_config.h", "content_aware_computation_sorting");

    return content_aware_computation_sorting_;
  }
  void set_content_aware_computation_sorting(
      bool content_aware_computation_sorting) {
   std::vector<std::string> mht_34_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_module_configDTh mht_34(mht_34_v, 509, "", "./tensorflow/compiler/xla/service/hlo_module_config.h", "set_content_aware_computation_sorting");

    content_aware_computation_sorting_ = content_aware_computation_sorting;
  }

  FusionConfigCollection fusion_config_collection() const {
   std::vector<std::string> mht_35_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_module_configDTh mht_35(mht_35_v, 516, "", "./tensorflow/compiler/xla/service/hlo_module_config.h", "fusion_config_collection");

    return fusion_config_collection_;
  }
  void set_fusion_config_collection(
      FusionConfigCollection fusion_config_collection) {
   std::vector<std::string> mht_36_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_module_configDTh mht_36(mht_36_v, 523, "", "./tensorflow/compiler/xla/service/hlo_module_config.h", "set_fusion_config_collection");

    fusion_config_collection_ = fusion_config_collection;
  }

  const std::vector<std::vector<bool>>& fusion_config() const {
   std::vector<std::string> mht_37_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_module_configDTh mht_37(mht_37_v, 530, "", "./tensorflow/compiler/xla/service/hlo_module_config.h", "fusion_config");

    return fusion_config_;
  }
  std::vector<std::vector<bool>>* mutable_fusion_config() {
   std::vector<std::string> mht_38_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_module_configDTh mht_38(mht_38_v, 536, "", "./tensorflow/compiler/xla/service/hlo_module_config.h", "mutable_fusion_config");

    return &fusion_config_;
  }

  const std::vector<std::vector<int64_t>>& dot_config() const {
   std::vector<std::string> mht_39_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_module_configDTh mht_39(mht_39_v, 543, "", "./tensorflow/compiler/xla/service/hlo_module_config.h", "dot_config");

    return dot_config_;
  }

  std::vector<std::vector<int64_t>>* mutable_dot_config() {
   std::vector<std::string> mht_40_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_module_configDTh mht_40(mht_40_v, 550, "", "./tensorflow/compiler/xla/service/hlo_module_config.h", "mutable_dot_config");

    return &dot_config_;
  }

  const std::vector<std::vector<std::vector<int64_t>>>& layout_config() const {
   std::vector<std::string> mht_41_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_module_configDTh mht_41(mht_41_v, 557, "", "./tensorflow/compiler/xla/service/hlo_module_config.h", "layout_config");

    return layout_config_;
  }

  std::vector<std::vector<std::vector<int64_t>>>* mutable_layout_config() {
   std::vector<std::string> mht_42_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_module_configDTh mht_42(mht_42_v, 564, "", "./tensorflow/compiler/xla/service/hlo_module_config.h", "mutable_layout_config");

    return &layout_config_;
  }

  const std::vector<std::vector<bool>>& phase_ordering_config() const {
   std::vector<std::string> mht_43_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_module_configDTh mht_43(mht_43_v, 571, "", "./tensorflow/compiler/xla/service/hlo_module_config.h", "phase_ordering_config");

    return phase_ordering_config_;
  }

  std::vector<std::vector<bool>>* mutable_phase_ordering_config() {
   std::vector<std::string> mht_44_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_module_configDTh mht_44(mht_44_v, 578, "", "./tensorflow/compiler/xla/service/hlo_module_config.h", "mutable_phase_ordering_config");

    return &phase_ordering_config_;
  }

  const absl::flat_hash_map<std::string, std::string>& flag_config() const {
   std::vector<std::string> mht_45_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_module_configDTh mht_45(mht_45_v, 585, "", "./tensorflow/compiler/xla/service/hlo_module_config.h", "flag_config");

    return flag_config_;
  }

  absl::flat_hash_map<std::string, std::string>* mutable_flag_config() {
   std::vector<std::string> mht_46_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_module_configDTh mht_46(mht_46_v, 592, "", "./tensorflow/compiler/xla/service/hlo_module_config.h", "mutable_flag_config");

    return &flag_config_;
  }

  const int phase_index() const {
   std::vector<std::string> mht_47_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_module_configDTh mht_47(mht_47_v, 599, "", "./tensorflow/compiler/xla/service/hlo_module_config.h", "phase_index");
 return phase_index_; }
  void set_phase_index(const int phase_index) {
   std::vector<std::string> mht_48_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_module_configDTh mht_48(mht_48_v, 603, "", "./tensorflow/compiler/xla/service/hlo_module_config.h", "set_phase_index");
 phase_index_ = phase_index; }

  void set_allow_spmd_sharding_propagation_to_output(
      bool allow_spmd_sharding_propagation_to_output) {
   std::vector<std::string> mht_49_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_module_configDTh mht_49(mht_49_v, 609, "", "./tensorflow/compiler/xla/service/hlo_module_config.h", "set_allow_spmd_sharding_propagation_to_output");

    allow_spmd_sharding_propagation_to_output_ =
        allow_spmd_sharding_propagation_to_output;
  }
  bool allow_spmd_sharding_propagation_to_output() const {
   std::vector<std::string> mht_50_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_module_configDTh mht_50(mht_50_v, 616, "", "./tensorflow/compiler/xla/service/hlo_module_config.h", "allow_spmd_sharding_propagation_to_output");

    return allow_spmd_sharding_propagation_to_output_;
  }

  const std::vector<uint64_t>& memory_space_assignment_config() const {
   std::vector<std::string> mht_51_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_module_configDTh mht_51(mht_51_v, 623, "", "./tensorflow/compiler/xla/service/hlo_module_config.h", "memory_space_assignment_config");

    return memory_space_assignment_config_;
  }

  std::vector<uint64_t>* mutable_memory_space_assignment_config() {
   std::vector<std::string> mht_52_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_module_configDTh mht_52(mht_52_v, 630, "", "./tensorflow/compiler/xla/service/hlo_module_config.h", "mutable_memory_space_assignment_config");

    return &memory_space_assignment_config_;
  }

  int64_t GetAnalysisAllowance(absl::string_view pass_name) const {
   std::vector<std::string> mht_53_v;
   mht_53_v.push_back("pass_name: \"" + std::string(pass_name.data(), pass_name.size()) + "\"");
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_module_configDTh mht_53(mht_53_v, 638, "", "./tensorflow/compiler/xla/service/hlo_module_config.h", "GetAnalysisAllowance");

    auto it = analysis_allowance_map_.find(pass_name);
    if (it == analysis_allowance_map_.end()) {
      return -1;
    }
    return (*it).second;
  }

  void SetAnalysisAllowance(absl::string_view pass_name, int64_t allowance) {
   std::vector<std::string> mht_54_v;
   mht_54_v.push_back("pass_name: \"" + std::string(pass_name.data(), pass_name.size()) + "\"");
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_module_configDTh mht_54(mht_54_v, 650, "", "./tensorflow/compiler/xla/service/hlo_module_config.h", "SetAnalysisAllowance");

    analysis_allowance_map_[pass_name] = allowance;
  }

 private:
  // If you add new members, be sure to update compilation_cache_key.

  absl::optional<ComputationLayout> entry_computation_layout_;

  // Module/graph-level seed handle.
  uint64_t seed_ = 0;

  // Program id that identifies a set of program to be launched together.
  int32_t launch_id_ = 0;

  // The number of replicas (data parallelism) to compile this binary for.
  int64_t replica_count_ = 1;

  // The number of partitions (model parallelism) to compile this binary for.
  int64_t num_partitions_ = 1;

  // Whether to broadcast args across all replicas. One entry per arg.
  std::vector<bool> param_requires_broadcast_via_collectives_;

  // Whether to use SPMD (true) or MPMD (false) when num_partitions_ > 0 and XLA
  // needs to partition the module.
  bool use_spmd_partitioning_ = false;

  // Whether to automatically generate XLA shardings for SPMD partitioner.
  bool use_auto_spmd_partitioning_ = false;

  // If enabled, deduplicate equivalent hlos into function calls to reduce code
  // size.
  bool deduplicate_hlo_ = false;

  // The target maximum parallelism at which to partition HLOs for parallel
  // execution on the CPU backend.
  int64_t intra_op_parallelism_threads_ = -1;

  std::string device_type_;

  DebugOptions debug_options_;

  // Compile-time known device assignment.
  absl::optional<DeviceAssignment> static_device_assignment_;

  std::vector<ShardableValueUpdatePair> shardable_value_update_pairs_;

  bool alias_passthrough_params_ = false;

  bool content_aware_computation_sorting_ = true;

  FusionConfigCollection fusion_config_collection_ =
      FusionConfigCollection::kOff;

  // TODO(b/155665133): Consolidate fusion, dot, and layout config into a proto
  // similar to backend config.

  // Custom fusion configuration, where fusion_config_[c][v] control if node v
  // in computation c must be fused to all its consumers (true) or not (false).
  std::vector<std::vector<bool>> fusion_config_;

  // Custom dot canonicalization configuration, where dot_config_[v] control
  // how to convert dot operation v (sorted topologically and by computation) to
  // convolution.
  std::vector<std::vector<int64_t>> dot_config_;

  // Layout configuration, where layout_config_[v][i] controls the layout
  // decision i of operation v.
  std::vector<std::vector<std::vector<int64_t>>> layout_config_;

  // Memory Space Assignment configuration, where
  // memory_space_assignment_config_ controls the order of buffer intervals
  // of this hlo module.
  std::vector<uint64_t> memory_space_assignment_config_;

  // Phase ordering configuration, where phase_ordering_config[v][i] controls
  // whether a specific pass with index i (e.g. 0 = DCE, 1 = CSE, etc.) is
  // inserted after pass v in pipeline. See tuning::PhaseOrderingConfig for
  // details on what indices (i) correspond to which passes.
  std::vector<std::vector<bool>> phase_ordering_config_;
  // Index (v) corresponding to current passes being added for phase ordering.
  // This is the variable that stores state to allow us to use the same
  // config across functions during compilation.
  int phase_index_ = 0;

  // Flag configuration to use instead of global flags. This allows multiple
  // HLO modules to be compiled in parallel with different flag values.
  absl::flat_hash_map<std::string, std::string> flag_config_;

  // Allows sharding propagation to propagate to the outputs. This changes the
  // output shape of the computation (which is undesirable), but it can be used
  // to allow to run partial compilation to determine what would be the output
  // sharding of a computation if XLA would be allowed to propagate the sharding
  // which can be used by higher level framework as a way to query intermediate
  // sharding of operations when multiple computation would be chained and
  // merged together.
  bool allow_spmd_sharding_propagation_to_output_ = false;

  // Each Hlo analysis is allowed at least a constant number of
  // abstract cost units, before it is considered for early termination.
  absl::flat_hash_map<absl::string_view, int64_t> analysis_allowance_map_;
};

}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_HLO_MODULE_CONFIG_H_
