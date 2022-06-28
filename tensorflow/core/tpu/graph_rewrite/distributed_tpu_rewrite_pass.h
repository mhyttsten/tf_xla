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

// Rewrites TPUReplicate nodes into replicated computations on TPU.
//
// To represent a distributed TPU computation, we use the
// TPUReplicate operator, that describes a subgraph (represented as a
// Tensorflow function) to replicate across a TPU pod.
//
// Model parallelism and data parallelism:
// ---------------------------------------
// We support two different kinds of parallelism on TPU:
// * data parallelism (replication), or parallelization across batches, and
// * model parallelism, or parallelization within a batch.
//
// The function passed to a TPUReplicate operator is replicated many
// times across a TPU pod (data parallelism). The `num_replicas` attribute
// controls how many replicas of the computation to create. Replicas are mostly
// independent; replicas can only communicate using the CrossReplicaSum
// operator, which is typically used to communicate gradients during training.
//
// Each replica may optionally use more than one TPU core (model
// parallelism). The `num_cores_per_replica` attribute controls how many cores
// there are per replica. For each core, there is a virtual TPU_REPLICATED_CORE
// device that is only valid within replicated TPU computations (e.g.,
// TPU_REPLICATED_CORE:0, TPU_REPLICATED_CORE:1, etc.); each TPU_REPLICATED_CORE
// device corresponds to one TPU core in every replica.
// Each replica has runs its own copy of the computation assigned to each
// TPU_REPLICATED_CORE device.
//
// The Python code is responsible for providing a device_assignment that
// describes how the replicated logical cores map to physical cores on the TPU
// topology.
//
// Inputs to TPUReplicate:
// ------------------------------
// The TPUReplicate operator takes three kinds of inputs, in the
// following order:
// * per-replica inputs. If there are three per-replica inputs (A, B, C) and two
//   replicas, the first six arguments to TPUReplicate will be:
//   A0 B0 C0 A1 B1 C1
//   where Ai is the A input to the i-th replica.
// * distributed inputs. These inputs follow the per-replica inputs.
//   If there are two distributed inputs (E, F) and two replicas, the following
//   arguments to TPUReplicate will be: E F.
//   But there is local E and F on each replica.
// * broadcast inputs. These inputs follow the distributed inputs. All
//   replicas receive a copy of each of these inputs.
// * variables. Resource variables accessed by the computation follow the
//   broadcast inputs.
//
// For example, for a computation with two replicas, three per-replica inputs
// (A, B, C), two distributed inputs(E, F), two broadcast inputs (X, Y), and two
// variables (V, W), the arguments to TPUReplicate will be:
// A0 B0 C0 A1 B1 C1 E F X Y V W
// and each replica will receive the following arguments:
// A B C E F X Y V W
//
// Distributed TPU compilation requires that the shapes of all operators
// be known statically at compilation time, before any nodes have executed.
// Shapes are determined using shape information emitted by InferShapes. It
// is not possible to replicate Tensorflow operators with unknown or dynamic
// shapes for TPU at present.
//
// Graph rewrite:
// --------------
// Compilation replaces TPUReplicate operators with:
// * a single TPUCompile node that compiles the computations,
// * one TPUExecute node for each TPU device in the system that
//   executes the relevant computation,
// * one ReadVariableOp for each variable accessed by the replicated
//   computation,
// * one AssignVariableOp for each variable accessed by the replicated
//   computation. An assignment is built even if a variable is only read by the
//   computation. We do not know which variables are written until we apply the
//   XlaCompiler to the computation, but that does not happen until after the
//   rewrite. Conservatively, we write back the values of all variables after
//   the computation completes.
//   TODO(phawkins): only write back variables that the computation may write.
// * one Shape node for each Tensor or Variable input to the computation whose
//   shape is not statically known at rewrite time. The input shapes are fed
//   to the TPUCompile node.
//
// To ensure that the reads and writes seem to happen at the right time in the
// graph execution, we add control edges from all predecessors of the original
// TPUReplicate operator to each of the ReadVariableOp operators.
// Similarly, we add control edges from all of the AssignVariableOp operators to
// all of the successors of the TPUReplicate operator.
//
// The TPUReplicate rewrite must run before placement, since resource
// variable inputs will have DT_RESOURCE, which cannot be sent across devices,
// leading to objections from the placer. The rewrite rewrites the resource
// accesses into explicit ReadVariableOp and AssignVariableOp operators that the
// placer is free to colocate with the variables.

#ifndef TENSORFLOW_CORE_TPU_GRAPH_REWRITE_DISTRIBUTED_TPU_REWRITE_PASS_H_
#define TENSORFLOW_CORE_TPU_GRAPH_REWRITE_DISTRIBUTED_TPU_REWRITE_PASS_H_
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
class MHTracer_DTPStensorflowPScorePStpuPSgraph_rewritePSdistributed_tpu_rewrite_passDTh {
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
   MHTracer_DTPStensorflowPScorePStpuPSgraph_rewritePSdistributed_tpu_rewrite_passDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePStpuPSgraph_rewritePSdistributed_tpu_rewrite_passDTh() {
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
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/container/node_hash_map.h"
#include "absl/types/span.h"
#include "tensorflow/compiler/jit/shape_inference.h"
#include "tensorflow/compiler/xla/service/computation_placer.h"
#include "tensorflow/core/common_runtime/optimization_registry.h"
#include "tensorflow/core/framework/function.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/stream_executor/tpu/tpu_topology.h"

namespace tensorflow {

// Replaces clusters assigned to TPU_SYSTEM devices with
// TPUCompile and TPUExecute nodes assigned to the corresponding
// TPU devices.
class DistributedTPURewritePass : public GraphOptimizationPass {
 public:
  static void SetDistributedTpuRewritePassOptions(
      bool distribute_vars, bool allow_xla_spmd_partition,
      bool replicate_inputs_outputs_by_default_for_xla_spmd,
      bool enable_cross_replica_sharding_mirrored_variables,
      bool enable_automatic_model_parallelism, bool enable_xla_param_broadcast,
      bool enable_multicore_locking, bool use_nd_sharding_ops);

  Status Run(const GraphOptimizationPassOptions& options) override;

  // The following methods are public only for the use of unit tests.

  // See comment at the top of the file for how the inputs are ordered.
  // Encapsulates the different TPU replicated node input and output
  // information, and provide common APIs over them.
  class ParameterInfo {
   public:
    ParameterInfo() {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePStpuPSgraph_rewritePSdistributed_tpu_rewrite_passDTh mht_0(mht_0_v, 317, "", "./tensorflow/core/tpu/graph_rewrite/distributed_tpu_rewrite_pass.h", "ParameterInfo");
}
    ParameterInfo(int64_t num_replicas, int64_t num_per_replica_args,
                  int64_t num_distributed_args, int64_t num_broadcast_args,
                  int64_t num_variables, int64_t num_guaranteed_constants,
                  int64_t num_retvals_per_replica)
        : num_replicas_(num_replicas),
          num_per_replica_args_(num_per_replica_args),
          num_distributed_args_(num_distributed_args),
          num_broadcast_args_(num_broadcast_args),
          num_variables_(num_variables),
          num_guaranteed_constants_(num_guaranteed_constants),
          num_retvals_per_replica_(num_retvals_per_replica) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePStpuPSgraph_rewritePSdistributed_tpu_rewrite_passDTh mht_1(mht_1_v, 331, "", "./tensorflow/core/tpu/graph_rewrite/distributed_tpu_rewrite_pass.h", "ParameterInfo");
}

    int64_t NumReplicas() const {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePStpuPSgraph_rewritePSdistributed_tpu_rewrite_passDTh mht_2(mht_2_v, 336, "", "./tensorflow/core/tpu/graph_rewrite/distributed_tpu_rewrite_pass.h", "NumReplicas");
 return num_replicas_; }

    int64_t NumPerReplicaArgs() const {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePStpuPSgraph_rewritePSdistributed_tpu_rewrite_passDTh mht_3(mht_3_v, 341, "", "./tensorflow/core/tpu/graph_rewrite/distributed_tpu_rewrite_pass.h", "NumPerReplicaArgs");
 return num_per_replica_args_; }

    int64_t NumDistributedArgs() const {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePStpuPSgraph_rewritePSdistributed_tpu_rewrite_passDTh mht_4(mht_4_v, 346, "", "./tensorflow/core/tpu/graph_rewrite/distributed_tpu_rewrite_pass.h", "NumDistributedArgs");
 return num_distributed_args_; }

    int64_t NumBroadcastArgs() const {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePStpuPSgraph_rewritePSdistributed_tpu_rewrite_passDTh mht_5(mht_5_v, 351, "", "./tensorflow/core/tpu/graph_rewrite/distributed_tpu_rewrite_pass.h", "NumBroadcastArgs");
 return num_broadcast_args_; }

    int64_t NumVariables() const {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePStpuPSgraph_rewritePSdistributed_tpu_rewrite_passDTh mht_6(mht_6_v, 356, "", "./tensorflow/core/tpu/graph_rewrite/distributed_tpu_rewrite_pass.h", "NumVariables");
 return num_variables_; }

    int64_t NumGuaranteedConstants() const {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePStpuPSgraph_rewritePSdistributed_tpu_rewrite_passDTh mht_7(mht_7_v, 361, "", "./tensorflow/core/tpu/graph_rewrite/distributed_tpu_rewrite_pass.h", "NumGuaranteedConstants");
 return num_guaranteed_constants_; }

    int64_t NumRetvalsPerReplica() const {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScorePStpuPSgraph_rewritePSdistributed_tpu_rewrite_passDTh mht_8(mht_8_v, 366, "", "./tensorflow/core/tpu/graph_rewrite/distributed_tpu_rewrite_pass.h", "NumRetvalsPerReplica");
 return num_retvals_per_replica_; }

    bool IsPerReplicaArg(int64_t index) const {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScorePStpuPSgraph_rewritePSdistributed_tpu_rewrite_passDTh mht_9(mht_9_v, 371, "", "./tensorflow/core/tpu/graph_rewrite/distributed_tpu_rewrite_pass.h", "IsPerReplicaArg");

      return index < num_per_replica_args_;
    }

    bool IsDistributedArg(int64_t index) const {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScorePStpuPSgraph_rewritePSdistributed_tpu_rewrite_passDTh mht_10(mht_10_v, 378, "", "./tensorflow/core/tpu/graph_rewrite/distributed_tpu_rewrite_pass.h", "IsDistributedArg");

      return index >= num_per_replica_args_ &&
             index < (num_per_replica_args_ + num_distributed_args_);
    }

    bool IsBroadcastArg(int64_t index) const {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScorePStpuPSgraph_rewritePSdistributed_tpu_rewrite_passDTh mht_11(mht_11_v, 386, "", "./tensorflow/core/tpu/graph_rewrite/distributed_tpu_rewrite_pass.h", "IsBroadcastArg");

      return (index >= num_per_replica_args_ + num_distributed_args_) &&
             index < (num_per_replica_args_ + num_distributed_args_ +
                      num_broadcast_args_);
    }

    bool IsVariableArg(int64_t index) const {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScorePStpuPSgraph_rewritePSdistributed_tpu_rewrite_passDTh mht_12(mht_12_v, 395, "", "./tensorflow/core/tpu/graph_rewrite/distributed_tpu_rewrite_pass.h", "IsVariableArg");

      return index >= (num_per_replica_args_ + num_distributed_args_ +
                       num_broadcast_args_) &&
             index < (num_per_replica_args_ + num_distributed_args_ +
                      num_broadcast_args_ + num_variables_);
    }

    bool IsConstantArg(int64_t index) const {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPScorePStpuPSgraph_rewritePSdistributed_tpu_rewrite_passDTh mht_13(mht_13_v, 405, "", "./tensorflow/core/tpu/graph_rewrite/distributed_tpu_rewrite_pass.h", "IsConstantArg");

      return index >= (num_per_replica_args_ + num_distributed_args_ +
                       num_broadcast_args_ + num_variables_) &&
             index < (num_per_replica_args_ + num_distributed_args_ +
                      num_broadcast_args_ + num_variables_ +
                      num_guaranteed_constants_);
    }

    // Returns the number of inputs which has been received by the host.
    int64_t NumInputsFromHost() const {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPScorePStpuPSgraph_rewritePSdistributed_tpu_rewrite_passDTh mht_14(mht_14_v, 417, "", "./tensorflow/core/tpu/graph_rewrite/distributed_tpu_rewrite_pass.h", "NumInputsFromHost");

      return num_replicas_ * num_per_replica_args_ + num_distributed_args_ +
             num_broadcast_args_ + num_variables_ + num_guaranteed_constants_;
    }

    // Returns the number of inputs which will be sent to each replica.
    int64_t NumInputsToEachReplica() const {
   std::vector<std::string> mht_15_v;
   MHTracer_DTPStensorflowPScorePStpuPSgraph_rewritePSdistributed_tpu_rewrite_passDTh mht_15(mht_15_v, 426, "", "./tensorflow/core/tpu/graph_rewrite/distributed_tpu_rewrite_pass.h", "NumInputsToEachReplica");

      return num_per_replica_args_ + num_distributed_args_ +
             num_broadcast_args_ + num_variables_ + num_guaranteed_constants_;
    }

    // Returns the total number of output values returned to the host (for all
    // replicas).
    int64_t NumOutputsToHost() const {
   std::vector<std::string> mht_16_v;
   MHTracer_DTPStensorflowPScorePStpuPSgraph_rewritePSdistributed_tpu_rewrite_passDTh mht_16(mht_16_v, 436, "", "./tensorflow/core/tpu/graph_rewrite/distributed_tpu_rewrite_pass.h", "NumOutputsToHost");

      return num_replicas_ * num_retvals_per_replica_;
    }

    // Returns the position of the first per-replica argument, within the set
    // of all hosts arguments.
    // Broadcast arguments follow the distributed arguments.
    int64_t FirstBroadcastArgFromHost() const {
   std::vector<std::string> mht_17_v;
   MHTracer_DTPStensorflowPScorePStpuPSgraph_rewritePSdistributed_tpu_rewrite_passDTh mht_17(mht_17_v, 446, "", "./tensorflow/core/tpu/graph_rewrite/distributed_tpu_rewrite_pass.h", "FirstBroadcastArgFromHost");

      return num_replicas_ * num_per_replica_args_ + num_distributed_args_;
    }

    // Indices of mirrored variables across replicas, which should be
    // categorized as per_replica_args.
    const std::set<int64_t>& mirrored_variable_indices() const {
   std::vector<std::string> mht_18_v;
   MHTracer_DTPStensorflowPScorePStpuPSgraph_rewritePSdistributed_tpu_rewrite_passDTh mht_18(mht_18_v, 455, "", "./tensorflow/core/tpu/graph_rewrite/distributed_tpu_rewrite_pass.h", "mirrored_variable_indices");

      return mirrored_variable_indices_;
    }
    std::set<int64_t>* mutable_mirrored_variable_indices() {
   std::vector<std::string> mht_19_v;
   MHTracer_DTPStensorflowPScorePStpuPSgraph_rewritePSdistributed_tpu_rewrite_passDTh mht_19(mht_19_v, 461, "", "./tensorflow/core/tpu/graph_rewrite/distributed_tpu_rewrite_pass.h", "mutable_mirrored_variable_indices");

      return &mirrored_variable_indices_;
    }

   private:
    int64_t num_replicas_ = 1;
    int64_t num_per_replica_args_ = 0;
    int64_t num_distributed_args_ = 0;
    int64_t num_broadcast_args_ = 0;
    int64_t num_variables_ = 0;
    int64_t num_guaranteed_constants_ = 0;
    int64_t num_retvals_per_replica_ = 0;
    std::set<int64_t> mirrored_variable_indices_;
  };

  // Mapping from TPUReplicate cluster name to tpu device names. Value is a
  // mapping from [replica][core] to a TF device name.
  typedef absl::flat_hash_map<string, std::vector<std::vector<string>>>
      TPUReplicateDeviceNamesMapping;

  // Determines which devices to use to run the computation.
  // Inputs:
  // * num_tpus_per_task: the number of TPU devices attached to each task
  // * tpu_devices: a [task][device] collection of TPU devices
  // * num_replicas: the number of replicas requested
  // * num_cores_per_replica: the number of cores in each computation instance
  // * topology_attr: the topology TPUReplicate attribute
  // * device_assignment_attr: the device_assignment TPUReplicate attribute
  // Outputs:
  // * tf_device_assignment: a mapping from [replica][core] to a TF device name
  // * devices_to_lock: a flat array of integer indices corresponding to devices
  //   that are used in this computation. They will be locked before the
  //   TPUExecute kernels are run, to ensure that the kernels from concurrent
  //   multi-core executions are enqueued consistently, i.e., all kernels from
  //   computation A before any kernel from computation B, thus preventing
  //   deadlock.
  // * xla_device_assignment: a mapping from [replica][core] to a linearized TPU
  //   coordinate.
  // TODO(phawkins): change tf_device_assignment to an xla::Array2D.
  static Status BuildDeviceAssignment(
      const tpu::TpuTopologyExternal& topology, int num_tpus_per_task,
      const std::vector<std::vector<Device*>>& tpu_devices, int num_replicas,
      int num_cores_per_replica, const string& topology_attr,
      absl::Span<const int> device_assignment_attr,
      std::vector<std::vector<string>>* tf_device_assignment,
      std::vector<int>* devices_to_lock,
      std::unique_ptr<xla::DeviceAssignment>* xla_device_assignment);

  // Returns the `computation` graph attached to TPUReplicate operator
  // `node`. `flr` is a FunctionLibraryRuntime to use when
  // instantiating the function body. Sets `*arg_types` and
  // `*retval_types` to the argument/return types of the function.
  static Status GetComputationForTPUReplicateOp(const NameAttrList& function,
                                                FunctionLibraryRuntime* flr,
                                                Graph* computation,
                                                DataTypeVector* arg_types,
                                                DataTypeVector* retval_types);

  // Returns the shapes of the argument tensors and return values of the
  // TPUReplicate operator `node` using the _output_shapes,
  // _output_handle_shapes, and _output_handle_types annotations on the input
  // nodes. Expects inputs in the following order (see comment at top of file):
  // * num_replicas * num_per_replica_args per-replica inputs,
  // * num_broadcast_args broadcast inputs,
  // * num_variables variable inputs.
  // Returns an error if the input shapes to `node` are not statically known.
  // Also verifies that all replicas have identical input shapes for their
  // per-replica inputs.
  static Status GetArgAndRetvalShapes(
      const GraphShapeInfo& shape_info, const Node& node,
      const ParameterInfo& params_info, std::vector<InferredShape>* arg_shapes,
      std::vector<InferredShape>* retval_shapes);

  // Assigns arguments and return values to cores. The assignment is represented
  // as an XLA op sharding, so that an argument can be replicated across cores.
  // `arg_sharding` and `retval_sharding` are vectors of shardings indexed by
  // argument/retval number.
  // `arg_fast_mem` is vector of fast_mem indication which is indexed by
  // argument number.
  static Status AssignArgsAndRetvalsToCores(
      int num_cores_per_replica, const ParameterInfo& params_info,
      const DataTypeVector& arg_types,
      const std::vector<InferredShape>& arg_shapes,
      const DataTypeVector& retval_types,
      const std::vector<InferredShape>& retval_shapes, const Graph& graph,
      const Node* replicate_node, FunctionLibraryRuntime* flr,
      bool allow_parameter_replication_for_spmd,
      std::vector<::xla::OpSharding>* arg_sharding,
      std::vector<bool>* arg_fast_mem,
      std::vector<::xla::OpSharding>* retval_sharding,
      std::vector<std::string>* arg_names);

  // Populates `*variables` with the "variables" inputs to `index`-th output of
  // `node`.
  struct VariableInput {
    Node* node;
    int index;

    // Type of the variable's value. Note that this is different to the type of
    // the output of 'variable', which is always DT_RESOURCE.
    DataType dtype;
  };
  static Status FindVariableInputs(const Node& node,
                                   const NameRangeMap& input_range_map,
                                   std::vector<VariableInput>* variables);

  // Populates '*guaranteed_constants' with the "guaranteed_constants" inputs
  // to 'node'.
  static Status FindGuaranteedConstantInputs(
      const Node& node, const NameRangeMap& input_range_map,
      std::vector<Node*>* guaranteed_constants);

  // Builds Shape nodes that compute the shapes of arguments whose shapes are
  // not statically known.
  static Status BuildDynamicShapeNodes(
      const Node& replicate_node, const std::vector<InferredShape>& arg_shapes,
      const ParameterInfo& params_info,
      const std::vector<Node*>& variable_reads, Graph* graph,
      std::vector<Node*>* dynamic_shape_nodes);

  // Builds a TPUCompile node that compiles the computation in
  // `function_names`. calls `nodes`.
  // TODO(b/33943292): at present, for model parallelism with Send/Recv to work
  // the `nodes` must correspond to the computations assigned to TPU:0,
  // TPU:1, ... in order since XLA hard-codes the chip IDs in the generated
  // executables.
  static Status BuildCompileNode(
      const Node* replicate_node, const NameAttrList& function,
      uint64 library_fingerprint, const ParameterInfo& params_info,
      const std::vector<InferredShape>& arg_shapes,
      const DataTypeVector& arg_types,
      const std::vector<Node*>& guaranteed_constant_nodes,
      const string& session_handle,
      const std::vector<::xla::OpSharding>& arg_sharding,
      const std::vector<bool>& arg_fast_mem,
      const std::vector<std::string>& arg_names,
      const std::vector<::xla::OpSharding>& retval_sharding,
      int num_cores_per_replica, const string& compile_device,
      const xla::DeviceAssignment* xla_device_assignment,
      const std::vector<Node*>& dynamic_shape_nodes, Graph* graph,
      Node** compile_node, int64_t autotuner_thresh);

  // Builds a TPUCompileSucceededAssert node that verifies that compilation
  // succeeded and replaces the TPUCompilationStatus node in the graph.
  static Status BuildCompilationStatusReturnNodes(
      Node* replicate_node, Node* compile_node,
      absl::Span<const int> devices_to_lock, Node** control_after_compilation,
      Node** multilock_acquire, Graph* graph);

  // Builds ReadVariableOp nodes that read `variables`, with a control
  // edges that ensure they happen after `control_predecessor`.
  static Status BuildVariableReads(absl::Span<const VariableInput> variables,
                                   Node* control_predecessor, Graph* graph,
                                   std::vector<Node*>* variable_reads);

  // Returns true if graph or functions contain resource write op, otherwise
  // return false.
  // TODO(b/137048563): Recognize unused resource rewrite op.
  static bool ContainsResourceWriteOp(const Graph& graph,
                                      const FunctionLibraryDefinition& fld);
  // Struct that describes a variable value to be written back from TPUExecute.
  struct VariableWrite {
    // A node:output pair containing a boolean tensor that determines whether
    // the value should be written back.
    Node* predicate;
    int predicate_output;

    // A node:output pair containing the value to be written back.
    Node* value;
    int value_output;
  };

  // Builds AssignVariableOp nodes that write `variables` with the values from
  // `variable_writes`, with control edges that ensure the writes happen before
  // `control_successor`.
  static Status BuildVariableWrites(
      absl::Span<const VariableInput> variables, Node* control_successor,
      absl::Span<const VariableWrite> variable_writes, Graph* graph);

  // Builds TPUExecute operators assigned to each TPU device
  // involved in the computation.
  // Arguments:
  // * `params_info` is the structure containing the information about the
  //    TPUReplicate node inputs and outputs.
  // * `num_tasks` is the number of TensorFlow tasks in the slice.
  // * `num_cores_per_replica` is the number of cores which are dedicated to
  //    each replica.
  // * `replicate_node` is the original TPUReplicate node.
  // * `arg_names` are the names of the arguments to the computation function
  //    passed as argument to TPUReplicate, including per-replica,
  //    broadcast, and variable arguments.
  // * `arg_types` are the corresponding types of the arguments.
  // * `arg_shapes` are the corresponding shapes (and handle types/shapes, if
  //    applicable).
  // * `arg_shardings` and `retval_shardings` are mappings from
  //    arguments/return indices to shardings, as returned by
  //    `AssignArgsAndRetvalsToCores`.
  // * `pod_devices` lists the devices to assign to each core of each replica.
  // * `variable_reads` is a vectors of ReadVariableOp operators, one for each
  //    variable argument to the computation.
  // * The execute operators will have a control edge from
  //   `control_predecessor` and another control edge to `control_successor`.
  // Populates '*variable_writes' with information about variable values to
  // write back.
  static Status BuildExecuteNodes(
      const ParameterInfo& params_info, int num_tasks,
      int num_cores_per_replica, const Node& replicate_node,
      const std::vector<std::string>& arg_names,
      const DataTypeVector& arg_types,
      const std::vector<InferredShape>& arg_shapes,
      const DataTypeVector& retval_types,
      const std::vector<::xla::OpSharding>& arg_shardings,
      const std::vector<::xla::OpSharding>& retval_shardings,
      const std::vector<std::vector<string>>& tpu_device_names,
      Node* compile_node, const std::vector<Node*>& variable_reads,
      Node* control_predecessor, Node* control_successor,
      Node* multilock_acquire, std::vector<VariableWrite>* variable_writes,
      Graph* graph);

  // Connects the compile node to all the host transfer nodes, and removes the
  // key placeholder node that was previously standing in for it.
  // Arguments:
  // * `compile_node` is the TPUCompile node that has been added to the graph.
  // * `key_placeholder_node` is the placeholder node to send the key to all the
  // host
  // * transfer nodes in the original graph.
  // * `graph` is the graph being rewritten.
  static Status ConnectHostComputeNodes(Node* compile_node,
                                        Node* key_placeholder_node,
                                        Graph* graph);

  // Map from a Node in an outside_compilation cluster in the original graph to
  // the list of Nodes, one for each replica, that it is expanded into during
  // replication.
  typedef absl::node_hash_map<Node*, std::vector<Node*>> NodeToNodeReplicasMap;

  // Map from the name of an outside_compilation cluster to the model-parallel
  // core index that the HostCompute Op should be placed on in that cluster.
  typedef std::map<string, int> HostComputeCoreMap;

  // Map from the name of an outside_compilation cluster to the list of Nodes
  // that should run on the host for that cluster.
  typedef std::map<string, std::vector<Node*>> OutsideCompilationNodeMap;

  // Copies the outside_compilation nodes in a cluster to create replica
  // replica_index.
  static Status CopyOutsideCompilationNodes(
      int replica_index, const std::vector<Node*>& outside_compilation_nodes,
      const DeviceNameUtils::ParsedName& tpu_device,
      const DeviceNameUtils::ParsedName& partial_device,
      NodeToNodeReplicasMap* node_images, Graph* graph);

  // Replicates all the nodes in outside_compilation clusters in a compiled
  // computation.
  static Status ReplicateOutsideCompilationNodes(
      const std::vector<std::vector<string>>& tf_device_assignment,
      const HostComputeCoreMap& host_compute_core,
      const OutsideCompilationNodeMap& outside_compilation_nodes,
      NodeToNodeReplicasMap* node_images, Graph* graph);

  // Lifts the edges between original outside_compilation nodes in a cluster
  // onto their replicas.
  static Status CopyOutsideCompilationEdges(
      const std::vector<Node*>& outside_compilation_nodes,
      const NodeToNodeReplicasMap& node_images,
      const std::unordered_map<string, Node*> outside_compilation_inputs,
      Graph* graph);

  // Lifts all the edges in outside_compilation clusters in a compiled
  // computation to their replicas.
  static Status ReplicateOutsideCompilationEdges(
      const OutsideCompilationNodeMap& outside_compilation_nodes,
      const NodeToNodeReplicasMap& node_images,
      const std::unordered_map<string, Node*> outside_compilation_inputs,
      Graph* graph);

  // Removes all the original outside_compilation nodes from the graph,
  // following replication.
  static Status RemoveOutsideCompilationNodes(
      const NodeToNodeReplicasMap& node_images, Graph* graph);

  // Lowers outside compilation functional nodes (If/While/function call).
  // Otherwise, when we have multiple workers, device placer will not be able to
  // place nodes if outside compilation has DT_RESOURCE inputs (e.g. a
  // DT_RESOURCE input fed into multiple While nodes on different devices).
  static Status LowerOutsideCompilationFunctionalNodes(
      Graph* g, FunctionLibraryDefinition& flib_def,
      const TPUReplicateDeviceNamesMapping& tpu_replicate_device_names_mapping);

  // Parses the 'host_compute_core' attribute on replicate_node to get the
  // replicated core id of each outside_compilation cluster.
  static Status ParseHostComputeCores(
      const Node& replicate_node,
      const OutsideCompilationNodeMap& outside_compilation_nodes,
      HostComputeCoreMap* host_compute_core);

  // Gets the physical topology information about the TPU system.
  static Status GetDeviceTopology(
      const DeviceSet& device_set, const Node& replicate_node,
      int* num_replicas, int* num_cores_per_replica, int* num_tasks,
      std::vector<std::vector<string>>* tf_device_assignment,
      std::vector<int>* devices_to_lock,
      std::unique_ptr<xla::DeviceAssignment>* xla_device_assignment,
      string* tpu_compilation_device);

  // Gets the types of args, retvals, and parameters.
  static Status GetIOTypes(
      int num_replicas, const Node& replicate_node, FunctionLibraryRuntime* flr,
      Graph* graph, NameRangeMap* input_name_map, const NameAttrList** function,
      std::unique_ptr<Graph>* computation, DataTypeVector* arg_types,
      DataTypeVector* retval_types, ParameterInfo* params_info);

  // Find known constants and deals with variable reads.
  static Status DealWithConstantsAndVariables(
      const Node& replicate_node, const NameRangeMap& input_name_map,
      Graph* graph, Node* host_transfer_sequencer, Node* control_before,
      Node* control_after, absl::Span<const VariableInput> variable_nodes,
      std::vector<Node*>* guaranteed_constant_nodes,
      std::vector<Node*>* variable_reads);

  // Adds NoOp nodes for sequencing computation and variable reads/writes.
  static Status BuildSequencingNodes(const string& tpu_compilation_device,
                                     const Node& replicate_node, Graph* graph,
                                     Node** host_transfer_sequencer,
                                     Node** control_before,
                                     Node** control_after);

  // Performs the pass's rewrite on a TPUReplicate node `node`.
  static Status RewriteTPUReplicateNode(
      const string& session_handle, const DeviceSet& device_set,
      Node* replicate_node, FunctionLibraryDefinition* flib_def,
      FunctionLibraryRuntime* flr, Node* host_compute_key_placeholder_node,
      const OutsideCompilationNodeMap& outside_compilation_nodes,
      const std::vector<Node*>& head_tail_outside_compilation_nodes,
      NodeToNodeReplicasMap* outside_compilation_node_images, Graph* graph,
      const GraphShapeInfo& shape_info,
      TPUReplicateDeviceNamesMapping* tpu_replicate_device_names_mapping,
      int64_t autotuner_thresh);

  // Performs host training loop optimization. For example, when TPUExecute
  // node is inside a while loop, then model weight variables can be sharded
  // in XLA preferred layout and then unsharded only at the very last iteration
  // to reduce the number of all_gather.
  static Status PerformHostTrainingLoopOptimization(
      Graph* graph, FunctionLibraryDefinition* flib_def,
      FunctionLibraryRuntime* flr);

  // Heuristically place some nodes with unassigned devices on TPUs for
  // performance reasons.
  static Status PlaceUnassignedDeviceNodesOnTPUIfPossible(Graph* graph);

  // Updates the head and tail outside compiled nodes so that nodes have the
  // correct device and removes the replication and outside compilation
  // attributes so that these nodes do not trigger further graph optimization
  // passes.
  static Status UpdateHeadTailOutsideCompilation(
      const std::vector<std::vector<string>>& tf_device_assignment,
      const std::vector<Node*>& head_tail_outside_compilation_nodes);

 private:
  static bool distribute_vars_;
  static bool allow_xla_spmd_partition_;
  static bool replicate_inputs_outputs_by_default_for_xla_spmd_;
  static bool enable_cross_replica_sharding_mirrored_variables_;
  static bool enable_automatic_model_parallelism_;
  static bool enable_xla_param_broadcast_;
  static bool enable_multicore_locking_;
  static bool use_nd_sharding_ops_;
};

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_TPU_GRAPH_REWRITE_DISTRIBUTED_TPU_REWRITE_PASS_H_
