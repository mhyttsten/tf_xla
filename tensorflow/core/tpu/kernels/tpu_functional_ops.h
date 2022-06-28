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

#ifndef TENSORFLOW_CORE_TPU_KERNELS_TPU_FUNCTIONAL_OPS_H_
#define TENSORFLOW_CORE_TPU_KERNELS_TPU_FUNCTIONAL_OPS_H_
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
class MHTracer_DTPStensorflowPScorePStpuPSkernelsPStpu_functional_opsDTh {
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
   MHTracer_DTPStensorflowPScorePStpuPSkernelsPStpu_functional_opsDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePStpuPSkernelsPStpu_functional_opsDTh() {
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


#include "absl/base/call_once.h"
#include "tensorflow/compiler/jit/shape_inference.h"
#include "tensorflow/core/common_runtime/device_mgr.h"
#include "tensorflow/core/common_runtime/optimization_registry.h"
#include "tensorflow/core/framework/function.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/lib/core/threadpool.h"
#include "tensorflow/core/tpu/kernels/tpu_ordinal_selector.h"
#include "tensorflow/core/tpu/tpu_api.h"
#include "tensorflow/core/tpu/tpu_ops_c_api.h"
#include "tensorflow/core/util/reffed_status_callback.h"
#include "absl/container/flat_hash_map.h"

namespace tensorflow {
// Holds node's shape information for Concat/Split.
using EdgeShapes = absl::flat_hash_map<const Edge*, std::vector<int>>;
using GroupedEdges =
    absl::flat_hash_map<std::string, std::vector<const Edge*>>;

// Contains attrs "T", "sharding", "_tpu_replicate" for each XlaSharding op that
// we find as part of searching for inputs to models that are replicated.
using XlaShardingInfoMap = absl::flat_hash_map<
    std::string, std::tuple<tensorflow::DataType, std::string, std::string>>;

// Contains attrs "T", and a pointer to tpu_replicated_metadata for ctrl dep
// for each TpuReplicatedInput op that we find as part of searching for inputs
// to models that are replicated.
using TpuReplicatedInputInfoMap =
    absl::flat_hash_map<std::string,
                           std::tuple<tensorflow::DataType, Node*>>;

namespace tpu_functional_internal {

// Helper functions for graph rewrites.
GroupedEdges GroupTensorsForInputPacking(
    const EdgeShapes& tpu_input_shapes,
    const absl::flat_hash_map<const Edge*, DataType>& tpu_input_dtypes,
    bool input_shape_opt, bool group_tensors_for_packing);
GroupedEdges GroupTensorsForOutputPacking(Graph* graph,
                                          EdgeShapes& tpu_output_shapes,
                                          GraphShapeInfo* shape_info);

Status CreateConcatAndSplitNodesForInputTensor(
    Graph* graph, const string& cluster_name, EdgeShapes* tpu_input_shapes,
    const absl::flat_hash_map<std::string, std::vector<const Edge*>>&
        grouped_input_edges,
    int32_t minimum_input_tensors_packing, bool xla_spmd_input_sharded,
    const XlaShardingInfoMap& xla_sharding_info,
    const TpuReplicatedInputInfoMap& tpu_replicated_input_info);
Status CreateConcatAndSplitNodesForOutputTensor(
    Graph* graph, const string& cluster_name, EdgeShapes* tpu_output_shapes,
    GraphShapeInfo* tpu_inferred_info, GroupedEdges shape_to_output,
    int32_t minimum_output_tensors_packing);

Status InsertReshapeNodePairs(Graph* graph, const string& cluster_name,
                              EdgeShapes* tpu_input_shapes,
                              int num_cores_per_replica);

}  // namespace tpu_functional_internal

typedef FunctionLibraryRuntime::Handle FHandle;

// A `TPUPartitionedCallOp` asynchronously executes a function on exactly one
// TPU core and potentially across multiple other devices, but within a single
// process. The kernel places and partitions the function's underlying graph,
// executing each of the partitioned subgraphs as a function.
//
// The core on which the TPU computation is executed must be specified via the
// `device_ordinal` input. Different invocations of this op may specify
// different device ordinals, making it possible to map TPU computations to
// different cores at runtime. Currently, macro-substitution of device ordinals
// is only supported for the following whitelisted ops:
//   * TPUExecute
//   * InfeedEnqueue
//   * InfeedEnqueueTuple
//
// Attempting to compute a TPUPartitionedCallOp whose function body has a
// non-whitelisted node bearing an attribute named "device_ordinal" will result
// in an error.
//
// TODO(akshayka): This class duplicates most of the logic of
// `PartitionedCallOp`; once that class and this one have evolved to stable
// states, and if at that time they remain sufficiently similar, either unify
// them in one op or set up an inheritance structure that allows for code reuse.
class TPUPartitionedCallOp : public AsyncOpKernel {
 public:
  explicit TPUPartitionedCallOp(OpKernelConstruction* ctx)
      : AsyncOpKernel(ctx),
        pool_(ctx->env(), "InitializeVarOnTPUPool", 1),
        library_runtime_(nullptr) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("f", &func_));
    // If the importer has set the original function name, it means the function
    // attribute is referring to a rewritten function, but we need to use the
    // original function name in order to find it in the function library.
    std::string orig_f;
    if (ctx->GetAttr("_orig_f", &orig_f).ok()) {
      func_.set_name(orig_f);
    }
    auto status = ctx->GetAttr("autotuner_thresh", &autotuner_thresh_);
    if (!status.ok()) {
      autotuner_thresh_ = 0;
    }
    tensorflow::tpu::OpsApiFn()->TfTpu_GetTpuPartitionedCallParamsFn(
        &runtime_params_);
  }

  ~TPUPartitionedCallOp() override {}

  void ComputeAsync(OpKernelContext* ctx, DoneCallback done) override;

 private:
  struct DeviceAndFHandle {
    std::string device;
    FHandle handle;

    // The FLD passed to `library_runtime_` as an overlay function library for
    // instantiation of function `handle`. This is a snapshot of the currrent
    // `flib_def_`. Since `flib_def_` can be changed concurrently by another
    // graph rewrite when executing `handle`, we need to make sure each
    // `handle` uses a different FLD to avoid races. See b/181149591.
    std::unique_ptr<FunctionLibraryDefinition> flib_def;
  };

  // This method is thread-safe.
  Status GetTpuCoreOrdinal(OpKernelContext* ctx, uint64 input_hash,
                           int64_t* ordinal_selector_req_id,
                           int32_t* core_ordinal);

  // Helper to create and initialize a TPU variable given a CPU variable
  // var: the CPU variable created by the user
  // ndef: the node def of the corresponding TPU var handle that we created
  // device_ordinal: TPU device ordinal on which to initialize this variable
  Status InitializeVarOnTPU(OpKernelContext* ctx,
                            const core::RefCountPtr<Var>& var, NodeDef* ndef,
                            int device_ordinal, bool fast_mem)
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu_);

  // Helper to create and initialize partitioned TPU variables given a CPU
  // variable with XLA sharding annotation.
  // var: the CPU variable created by the user.
  // ndefs: the node def of the corresponding TPU var handle on all the logical
  //   cores.
  // split_dim: the partition dimension of the variable. If -1, the variable is
  //   replicated.
  // device_ordinal: The index of the TPU core that is scheduled to run
  //   the computation. In the case of XLA SPMD, it is the "primary" core, which
  //   is the smallest index of all the cores.
  Status InitializeShardedVarOnTPU(OpKernelContext* ctx,
                                   const core::RefCountPtr<Var>& var,
                                   std::vector<NodeDef>& ndefs, int split_dim,
                                   int device_ordinal)
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu_);

  // Check if any of the immediate successors of node has attribute
  // "_tpu_replicate".
  bool IsInputToTPUReplicate(Node* node) ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu_);

  // Replace an _Arg node of type DT_RESOURCE by a VarHandleOp on TPU
  Status ReplaceResourceArgsWithVarHandleOps(Graph* graph, OpKernelContext* ctx,
                                             int device_ordinal,
                                             int num_cores_per_replica,
                                             bool enable_spmd_xla_partitioning)
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu_);

  // Replace a _Arg node indicates a variable on CPU host by sharded/replicated
  // variables on all logical TPU devices.
  Status ReplaceAndPartitionXLAShardingVariable(
      Graph* graph, OpKernelContext* ctx, int device_ordinal,
      ResourceHandle& handle, Node* variable, int num_cores_per_replica)
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu_);

  Status ShardInputsWithXlaSharding(Graph* graph, int num_cores_per_replica,
                                    OpKernelContext* ctx)
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu_);

  // Rewrite the graph for input and output optimiazations.
  // TODO(ylc): Move this function to Graph optimization pass.
  Status OptimizeTpuInputOutputTensors(
      Graph* graph, bool enable_spmd_xla_partitioning,
      int num_cores_per_replica,
      std::map<std::string, std::vector<int>>& named_input_shapes,
      OpKernelContext* ctx) ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu_);

  Status InferShapesWithResourceVar(Graph* graph, OpKernelContext* ctx,
                                    std::map<int, InferredShape>& arg_shapes,
                                    GraphShapeInfo* tpu_inferred_info);

  // Copies the graph backing `func_` into `graph`.
  Status GetGraphFromFunction(Graph* graph, int device_ordinal,
                              int* num_core_per_replica,
                              bool* use_spmd_for_xla_partitioning)
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu_);

  // Places the graph carried by `optimization_options` and runs graph
  // optimization passes (pre-placement, post-placement, and post-rewrite).
  Status PlacementHelper(
      const DeviceSet& device_set,
      const GraphOptimizationPassOptions& optimization_options,
      const string& function_name);
  // Partitions `graph`, populates `subgraphs` with the partitions, and runs
  // the post-partitioning graph optimization passes.
  Status PartitionHelper(
      const DeviceSet& device_set,
      const GraphOptimizationPassOptions& optimization_options, Graph* graph,
      std::unordered_map<std::string, std::unique_ptr<Graph>>* subgraphs);

  // Adds and instantiates a function backed by `graph` with name
  // `function_name` on device `target_device`, storing the handle in `handle`.
  // If `out_flib_def` is not null, it will be set to a copy of `flib_def_` and
  // used for instantiation.
  Status InstantiatePartition(
      const Graph& graph, const string& function_name,
      const string& target_device, FHandle* handle,
      std::unique_ptr<FunctionLibraryDefinition>* out_flib_def)
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu_);
  // Adds and instantiates functions for each subgraph in `subgraphs` after
  // rewriting nodes' `device_ordinal` attributes to match `replica_id` when
  // num_cores_per_replica == 1.
  Status InstantiateFunctionsFromSubgraphs(
      const DeviceSet& device_set, int replica_id, uint64 cache_hash,
      int num_cores_per_replica,
      std::unordered_map<std::string, std::unique_ptr<Graph>> subgraphs)
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu_);

  // Rewrites `graph` such that the device ordinal attributes of all whitelisted
  // nodes (see `IsSupportedTPUOp`) are set to `device_ordinal`;
  // `*modified` is set to true if the graph is modified in the process (i.e.,
  // if it contains a whitelisted node), otherwise is unmodified.
  //
  // Returns an error if
  //   (1) the graph contains a non-whitelisted node that carries an attribute
  //       with name "device_ordinal", or
  //   (2) the set of device ordinals found among the graph's nodes has
  //       cardinality greater than 1.
  Status SetDeviceOrdinal(const DeviceSet& device_set, int device_ordinal,
                          Graph* graph, bool* modified)
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu_);

  void ExecuteRemoteFunction(const FunctionLibraryRuntime::Options& opts,
                             FHandle handle, OpKernelContext* ctx,
                             ReffedStatusCallback* done)
      ABSL_LOCKS_EXCLUDED(mu_);
  void ExecuteLocalFunction(const FunctionLibraryRuntime::Options& opts,
                            const OpInputList& arguments, FHandle handle,
                            OpKernelContext* ctx, ReffedStatusCallback* done)
      ABSL_LOCKS_EXCLUDED(mu_);
  void ExecuteFunctions(const std::vector<DeviceAndFHandle>& functions,
                        OpKernelContext* ctx, int device_ordinal,
                        int64_t ordinal_selector_req_id, DoneCallback done)
      ABSL_LOCKS_EXCLUDED(mu_);

  Status ShouldUseRemoteExecutionForFn(const std::string& target_device,
                                       bool* remote_execution) {
   std::vector<std::string> mht_0_v;
   mht_0_v.push_back("target_device: \"" + target_device + "\"");
   MHTracer_DTPStensorflowPScorePStpuPSkernelsPStpu_functional_opsDTh mht_0(mht_0_v, 442, "", "./tensorflow/core/tpu/kernels/tpu_functional_ops.h", "ShouldUseRemoteExecutionForFn");

    DeviceNameUtils::ParsedName target_device_parsed;
    DeviceNameUtils::ParsedName local_device_parsed;

    if (!DeviceNameUtils::ParseFullOrLocalName(target_device,
                                               &target_device_parsed)) {
      return errors::InvalidArgument("Cannot parse target device ",
                                     target_device);
    }
    if (!DeviceNameUtils::ParseFullOrLocalName(local_device_name_,
                                               &local_device_parsed)) {
      return errors::InvalidArgument("Cannot parse local device ",
                                     local_device_name_);
    }

    if (DeviceNameUtils::AreCompatibleDevNames(target_device_parsed,
                                               local_device_parsed)) {
      *remote_execution = false;
    } else {
      *remote_execution = true;
    }
    return Status::OK();
  }

  // Init once flagas.
  absl::once_flag once_;
  absl::once_flag ordinal_selector_once_;

  // Device manager and device set.
  const DeviceMgr* device_mgr_;
  DeviceSet device_set_;

  // Threadpool.
  thread::ThreadPool pool_;

  // `func_` is the original function supplied to this OpKernel.
  NameAttrList func_;
  string local_device_name_;
  // Maps from cache key to their corresponding functions, which are
  // represented as (device, handle) pairs.
  gtl::FlatMap<uint64, std::vector<DeviceAndFHandle>> partition_cache_
      ABSL_GUARDED_BY(mu_);

  // A set contains seen ordinals. Used by variable initialization on TPU.
  absl::flat_hash_set<int> seen_ordinals_;

  // Record the indices of the _Arg with type DT_RESOURCE that goes
  // into a TPU Op.
  std::vector<bool> replaced_input_indices_;

  absl::Mutex mu_;
  // Function shards are added to the `flib_def_`, and later on it'll create
  // a copy of `flib_def_` to pass to `library_runtime_` as an overlay function
  // library for instantiation.
  std::unique_ptr<FunctionLibraryDefinition> flib_def_;
  FunctionLibraryRuntime* library_runtime_;

  // Used to uniquify function names in `flib_def_`.
  uint32 suffix_ = 0;

  // Minimum number of run steps (batches) necessary to trigger xla autotuner.
  int autotuner_thresh_ = 0;

  // TPU core selection.
  std::shared_ptr<tpu::TPUOrdinalSelector> ordinal_selector_;

  // Maps input hash to TF fingerprint.
  absl::flat_hash_map<uint64, uint64> inputs_to_fingerprint_;

  // List of TPU devices
  std::vector<Device*> tpu_devices_;

  TpuPartitionedCall_Params runtime_params_;
};

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_TPU_KERNELS_TPU_FUNCTIONAL_OPS_H_
