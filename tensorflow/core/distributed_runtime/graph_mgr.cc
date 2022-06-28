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
class MHTracer_DTPStensorflowPScorePSdistributed_runtimePSgraph_mgrDTcc {
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
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSgraph_mgrDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSdistributed_runtimePSgraph_mgrDTcc() {
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

/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/distributed_runtime/graph_mgr.h"

#include <chrono>  // NOLINT(build/c++11)
#include <vector>

#include "tensorflow/core/common_runtime/build_graph_options.h"
#include "tensorflow/core/common_runtime/constant_folding.h"
#include "tensorflow/core/common_runtime/debugger_state_interface.h"
#include "tensorflow/core/common_runtime/device.h"
#include "tensorflow/core/common_runtime/device_mgr.h"
#include "tensorflow/core/common_runtime/function.h"
#include "tensorflow/core/common_runtime/graph_constructor.h"
#include "tensorflow/core/common_runtime/graph_optimizer.h"
#include "tensorflow/core/common_runtime/memory_types.h"
#include "tensorflow/core/common_runtime/optimization_registry.h"
#include "tensorflow/core/common_runtime/process_util.h"
#include "tensorflow/core/common_runtime/rendezvous_util.h"
#include "tensorflow/core/common_runtime/step_stats_collector.h"
#include "tensorflow/core/distributed_runtime/rendezvous_mgr_interface.h"
#include "tensorflow/core/framework/cancellation.h"
#include "tensorflow/core/framework/collective.h"
#include "tensorflow/core/framework/log_memory.h"
#include "tensorflow/core/framework/metrics.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/framework/node_def_util.h"
#include "tensorflow/core/framework/versions.pb.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/graph/graph_partition.h"
#include "tensorflow/core/graph/validate.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/strings/stringprintf.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/tracing.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/profiler/lib/connected_traceme.h"
#include "tensorflow/core/profiler/lib/traceme_encode.h"
#include "tensorflow/core/protobuf/worker.pb.h"
#include "tensorflow/core/util/env_var.h"

namespace tensorflow {

GraphMgr::GraphMgr(const WorkerEnv* worker_env, const DeviceMgr* device_mgr)
    : worker_env_(worker_env), device_mgr_(device_mgr), table_(5) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSgraph_mgrDTcc mht_0(mht_0_v, 229, "", "./tensorflow/core/distributed_runtime/graph_mgr.cc", "GraphMgr::GraphMgr");

  // The default value of sync_on_finish will be flipped soon and this
  // environment variable will be removed as well.
  Status status =
      ReadBoolFromEnvVar("TF_SYNC_ON_FINISH", true, &sync_on_finish_);
  if (!status.ok()) {
    LOG(ERROR) << status.error_message();
  }
}

GraphMgr::~GraphMgr() {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSgraph_mgrDTcc mht_1(mht_1_v, 242, "", "./tensorflow/core/distributed_runtime/graph_mgr.cc", "GraphMgr::~GraphMgr");

  for (const auto& p : table_) p.second->Unref();
}

GraphMgr::Item::~Item() {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSgraph_mgrDTcc mht_2(mht_2_v, 249, "", "./tensorflow/core/distributed_runtime/graph_mgr.cc", "GraphMgr::Item::~Item");

  for (const auto& unit : this->units) {
    CHECK_NOTNULL(unit.device);
    if (!graph_mgr->skip_cost_models_) {
      graph_mgr->cost_model_manager_.RemoveCostModelForGraph(unit.graph.get());
    }
    delete unit.root;
    unit.device->op_segment()->RemoveHold(this->session);
  }
}

// NOTE: node->device_name() is not set by GraphConstructor.  We
// expects that NodeDef in GraphDef given to workers fully specifies
// device names.
static string SplitByDevice(const Node* node) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSgraph_mgrDTcc mht_3(mht_3_v, 266, "", "./tensorflow/core/distributed_runtime/graph_mgr.cc", "SplitByDevice");

  return node->assigned_device_name();
}

// Validates "gdef" device specifications.
static Status ValidateGraphDefForDevices(const GraphDef& gdef) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSgraph_mgrDTcc mht_4(mht_4_v, 274, "", "./tensorflow/core/distributed_runtime/graph_mgr.cc", "ValidateGraphDefForDevices");

  DeviceNameUtils::ParsedName parsed;
  for (const auto& ndef : gdef.node()) {
    if (!DeviceNameUtils::ParseFullName(ndef.device(), &parsed)) {
      return errors::InvalidArgument("Missing device name in: ",
                                     FormatNodeDefForError(ndef));
    }
  }
  return Status::OK();
}

Status GraphMgr::DecorateAndPublishGraphForDebug(
    const DebugOptions& debug_options, Graph* graph, Device* device) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSgraph_mgrDTcc mht_5(mht_5_v, 289, "", "./tensorflow/core/distributed_runtime/graph_mgr.cc", "GraphMgr::DecorateAndPublishGraphForDebug");

  std::unique_ptr<DebugGraphDecoratorInterface> decorator;
  TF_RETURN_IF_ERROR(
      DebugGraphDecoratorRegistry::CreateDecorator(debug_options, &decorator));
  TF_RETURN_IF_ERROR(decorator->DecorateGraph(graph, device));
  TF_RETURN_IF_ERROR(decorator->PublishGraph(*graph, device->name()));
  return Status::OK();
}

// Creates executors given a graph definition "gdef" of a "session".
// If a node in "gdef" is shared by other graphs in "session", the
// same op kernel is reused. E.g., typically a params node is shared
// by multiple graphs in a session.
//
// If "gdef" is assigned to multiple devices, extra nodes (e.g.,
// send/recv nodes) maybe added. The extra nodes' name are generated
// by calling "new_name(old_name)".
//
// "executors" are filled with one executor per device if success and
// the caller takes the ownership of returned executors.
Status GraphMgr::InitItem(const string& handle, const GraphDef& gdef,
                          const GraphOptions& graph_options,
                          const DebugOptions& debug_options,
                          const ConfigProto& config_proto,
                          int64_t collective_graph_key, WorkerSession* session,
                          DistributedFunctionLibraryRuntime* cluster_flr,
                          Item* item) {
   std::vector<std::string> mht_6_v;
   mht_6_v.push_back("handle: \"" + handle + "\"");
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSgraph_mgrDTcc mht_6(mht_6_v, 319, "", "./tensorflow/core/distributed_runtime/graph_mgr.cc", "GraphMgr::InitItem");

  item->session = handle;
  item->collective_graph_key = collective_graph_key;
  item->lib_def.reset(
      new FunctionLibraryDefinition(OpRegistry::Global(), gdef.library()));

  TF_RETURN_IF_ERROR(ValidateGraphDefForDevices(gdef));

  // We don't explicitly Validate the graph def because ConvertGraphDefToGraph
  // does that below.
  item->proc_flr.reset(new ProcessFunctionLibraryRuntime(
      device_mgr_, worker_env_->env, /*config=*/&config_proto,
      gdef.versions().producer(), item->lib_def.get(),
      graph_options.optimizer_options(), worker_env_->compute_pool, cluster_flr,
      /*session_metadata=*/nullptr,
      Rendezvous::Factory{
          [this, session](const int64_t step_id, const DeviceMgr*,
                          Rendezvous** r) -> Status {
            auto* remote_r = this->worker_env_->rendezvous_mgr->Find(step_id);
            TF_RETURN_IF_ERROR(remote_r->Initialize(session));
            *r = remote_r;
            return Status::OK();
          },
          [this](const int64_t step_id) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSgraph_mgrDTcc mht_7(mht_7_v, 345, "", "./tensorflow/core/distributed_runtime/graph_mgr.cc", "lambda");

            this->worker_env_->rendezvous_mgr->Cleanup(step_id);
            return Status::OK();
          }}));

  // Constructs the graph out of "gdef".
  Graph graph(OpRegistry::Global());
  GraphConstructorOptions opts;
  opts.allow_internal_ops = true;
  opts.expect_device_spec = true;
  opts.validate_nodes = true;
  TF_RETURN_IF_ERROR(ConvertGraphDefToGraph(opts, gdef, &graph));

  // Splits "graph" into multiple subgraphs by device names.
  std::unordered_map<string, GraphDef> partitions;
  PartitionOptions popts;
  popts.node_to_loc = SplitByDevice;
  popts.new_name = [this](const string& prefix) {
   std::vector<std::string> mht_8_v;
   mht_8_v.push_back("prefix: \"" + prefix + "\"");
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSgraph_mgrDTcc mht_8(mht_8_v, 366, "", "./tensorflow/core/distributed_runtime/graph_mgr.cc", "lambda");

    mutex_lock l(mu_);
    return strings::StrCat(prefix, "_G", next_id_++);
  };
  popts.get_incarnation = [this](const string& name) -> int64 {
    Device* device = nullptr;
    Status s = device_mgr_->LookupDevice(name, &device);
    if (s.ok()) {
      return device->attributes().incarnation();
    } else {
      return PartitionOptions::kIllegalIncarnation;
    }
  };
  popts.flib_def = item->lib_def.get();
  popts.control_flow_added = true;
  popts.scheduling_for_recvs = graph_options.enable_recv_scheduling();
  TF_RETURN_IF_ERROR(Partition(popts, &graph, &partitions));
  if (popts.scheduling_for_recvs) {
    TF_RETURN_IF_ERROR(AddControlEdges(popts, &partitions));
  }

  std::unordered_map<string, std::unique_ptr<Graph>> partition_graphs;
  for (auto& partition : partitions) {
    std::unique_ptr<Graph> device_graph(new Graph(OpRegistry::Global()));
    GraphConstructorOptions device_opts;
    // There are internal operations (e.g., send/recv) that we now allow.
    device_opts.allow_internal_ops = true;
    device_opts.expect_device_spec = true;
    TF_RETURN_IF_ERROR(ConvertGraphDefToGraph(
        device_opts, std::move(partition.second), device_graph.get()));
    partition_graphs.emplace(partition.first, std::move(device_graph));
  }

  GraphOptimizationPassOptions optimization_options;
  optimization_options.flib_def = item->lib_def.get();
  optimization_options.partition_graphs = &partition_graphs;
  TF_RETURN_IF_ERROR(OptimizationPassRegistry::Global()->RunGrouping(
      OptimizationPassRegistry::POST_PARTITIONING, optimization_options));

  LocalExecutorParams params;

  item->units.reserve(partitions.size());
  item->graph_mgr = this;
  const auto& optimizer_opts = graph_options.optimizer_options();
  GraphOptimizer optimizer(optimizer_opts);
  for (auto& p : partition_graphs) {
    const string& device_name = p.first;
    std::unique_ptr<Graph>& subgraph = p.second;
    item->units.resize(item->units.size() + 1);
    ExecutionUnit* unit = &(item->units.back());

    // Find the device.
    Status s = device_mgr_->LookupDevice(device_name, &unit->device);
    if (!s.ok()) {
      // Remove the empty unit from the item as the item destructor wants all
      // units to have valid devices.
      item->units.pop_back();
      return s;
    }

    // Give the device an opportunity to rewrite its subgraph.
    TF_RETURN_IF_ERROR(unit->device->MaybeRewriteGraph(&subgraph));

    // Top-level nodes in the graph uses the op segment to cache
    // kernels. Therefore, as long as the executor is alive, we need
    // to ensure the kernels cached for the session are alive.
    auto opseg = unit->device->op_segment();
    opseg->AddHold(handle);

    // Function library runtime.
    FunctionLibraryRuntime* lib = item->proc_flr->GetFLR(unit->device->name());
    if (lib == nullptr) {
      return errors::InvalidArgument("Cannot find FLR for device: ",
                                     unit->device->name());
    }

    // Construct the root executor for the subgraph.
    params.device = unit->device;
    params.function_library = lib;
    params.create_kernel =
        [handle, lib, opseg](const std::shared_ptr<const NodeProperties>& props,
                             OpKernel** kernel) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSgraph_mgrDTcc mht_9(mht_9_v, 450, "", "./tensorflow/core/distributed_runtime/graph_mgr.cc", "lambda");

          // NOTE(mrry): We must not share function kernels (implemented
          // using `CallOp`) between subgraphs, because `CallOp::handle_`
          // is tied to a particular subgraph. Even if the function itself
          // is stateful, the `CallOp` that invokes it is not.
          if (!OpSegment::ShouldOwnKernel(lib, props->node_def.op())) {
            return lib->CreateKernel(props, kernel);
          }
          auto create_fn = [lib, &props](OpKernel** kernel) {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSgraph_mgrDTcc mht_10(mht_10_v, 461, "", "./tensorflow/core/distributed_runtime/graph_mgr.cc", "lambda");

            return lib->CreateKernel(props, kernel);
          };
          // Kernels created for subgraph nodes need to be cached.  On
          // cache miss, create_fn() is invoked to create a kernel based
          // on the function library here + global op registry.
          return opseg->FindOrCreate(handle, props->node_def.name(), kernel,
                                     create_fn);
        };
    params.delete_kernel = [lib](OpKernel* kernel) {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSgraph_mgrDTcc mht_11(mht_11_v, 473, "", "./tensorflow/core/distributed_runtime/graph_mgr.cc", "lambda");

      if (kernel && !OpSegment::ShouldOwnKernel(lib, kernel->type_string())) {
        delete kernel;
      }
    };

    optimizer.Optimize(lib, worker_env_->env, params.device, &subgraph,
                       GraphOptimizer::Options());

    // TensorFlow Debugger (tfdbg) inserts debug nodes in the graph.
    if (!debug_options.debug_tensor_watch_opts().empty()) {
      TF_RETURN_IF_ERROR(DecorateAndPublishGraphForDebug(
          debug_options, subgraph.get(), params.device));
    }

    TF_RETURN_IF_ERROR(
        EnsureMemoryTypes(DeviceType(unit->device->device_type()),
                          unit->device->name(), subgraph.get()));
    unit->graph = std::move(subgraph);
    unit->build_cost_model = graph_options.build_cost_model();
    if (unit->build_cost_model > 0) {
      skip_cost_models_ = false;
    }
    TF_RETURN_IF_ERROR(NewLocalExecutor(params, *unit->graph, &unit->root));
  }
  return Status::OK();
}

Status GraphMgr::Register(const string& handle, const GraphDef& gdef,
                          const GraphOptions& graph_options,
                          const DebugOptions& debug_options,
                          const ConfigProto& config_proto,
                          int64_t collective_graph_key, WorkerSession* session,
                          DistributedFunctionLibraryRuntime* cluster_flr,
                          string* graph_handle) {
   std::vector<std::string> mht_12_v;
   mht_12_v.push_back("handle: \"" + handle + "\"");
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSgraph_mgrDTcc mht_12(mht_12_v, 511, "", "./tensorflow/core/distributed_runtime/graph_mgr.cc", "GraphMgr::Register");

  Item* item = new Item;
  Status s = InitItem(handle, gdef, graph_options, debug_options, config_proto,
                      collective_graph_key, session, cluster_flr, item);
  if (!s.ok()) {
    item->Unref();
    return s;
  }

  // Inserts one item into table_.
  {
    mutex_lock l(mu_);
    *graph_handle =
        strings::Printf("%016llx", static_cast<long long>(++next_id_));
    item->handle = *graph_handle;
    CHECK(table_.insert({*graph_handle, item}).second);
  }
  return Status::OK();
}

Status GraphMgr::Deregister(const string& handle) {
   std::vector<std::string> mht_13_v;
   mht_13_v.push_back("handle: \"" + handle + "\"");
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSgraph_mgrDTcc mht_13(mht_13_v, 535, "", "./tensorflow/core/distributed_runtime/graph_mgr.cc", "GraphMgr::Deregister");

  Item* item = nullptr;
  // Removes one item from table_.
  {
    mutex_lock l(mu_);
    auto iter = table_.find(handle);
    if (iter == table_.end()) {
      return errors::Aborted("Graph handle is not found: ", handle,
                             ". Possibly, this worker just restarted.");
    }
    item = iter->second;
    table_.erase(iter);
  }
  item->Unref();
  return Status::OK();
}

Status GraphMgr::DeregisterAll() {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSgraph_mgrDTcc mht_14(mht_14_v, 555, "", "./tensorflow/core/distributed_runtime/graph_mgr.cc", "GraphMgr::DeregisterAll");

  std::vector<Item*> items;
  // Removes all items from table_.
  {
    mutex_lock l(mu_);
    for (const auto& entry : table_) {
      items.push_back(entry.second);
    }
    table_.clear();
  }
  for (auto item : items) {
    item->Unref();
  }
  return Status::OK();
}

Status GraphMgr::SendInputs(const int64_t step_id, const NamedTensors& in) {
   std::vector<std::string> mht_15_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSgraph_mgrDTcc mht_15(mht_15_v, 574, "", "./tensorflow/core/distributed_runtime/graph_mgr.cc", "GraphMgr::SendInputs");

  Rendezvous* rendezvous = worker_env_->rendezvous_mgr->Find(step_id);
  std::vector<string> keys;
  std::vector<Tensor> tensors_to_send;
  keys.reserve(in.size());
  tensors_to_send.reserve(in.size());
  size_t input_size = 0;
  for (const auto& p : in) {
    keys.push_back(p.first);
    tensors_to_send.push_back(p.second);
    input_size += p.second.AllocatedBytes();
  }
  metrics::RecordGraphInputTensors(input_size);
  Status s =
      SendTensorsToRendezvous(rendezvous, nullptr, {}, keys, tensors_to_send);
  rendezvous->Unref();
  return s;
}

Status GraphMgr::RecvOutputs(const int64_t step_id, NamedTensors* out) {
   std::vector<std::string> mht_16_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSgraph_mgrDTcc mht_16(mht_16_v, 596, "", "./tensorflow/core/distributed_runtime/graph_mgr.cc", "GraphMgr::RecvOutputs");

  Rendezvous* rendezvous = worker_env_->rendezvous_mgr->Find(step_id);
  Status s = RecvOutputsFromRendezvous(rendezvous, out, Rendezvous::Args());
  rendezvous->Unref();
  if (!s.ok()) {
    // Failing to fetch the outputs should not be possible, so rewrite the error
    // status to an INTERNAL error.
    s = errors::Internal("Failed to fetch outputs for step ", step_id,
                         ". (Original error message: ", s.error_message(), ")");
  }
  size_t output_size = 0;
  for (auto& p : *out) {
    output_size += p.second.AllocatedBytes();
  }
  metrics::RecordGraphOutputTensors(output_size);
  return s;
}

void GraphMgr::RecvOutputsAsync(const int64_t step_id, NamedTensors* out,
                                StatusCallback done) {
   std::vector<std::string> mht_17_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSgraph_mgrDTcc mht_17(mht_17_v, 618, "", "./tensorflow/core/distributed_runtime/graph_mgr.cc", "GraphMgr::RecvOutputsAsync");

  Rendezvous* rendezvous = worker_env_->rendezvous_mgr->Find(step_id);
  std::vector<string> keys;
  std::vector<Tensor>* received_keys = new std::vector<Tensor>;
  keys.reserve(out->size());
  received_keys->reserve(out->size());
  for (const auto& p : *out) {
    keys.push_back(p.first);
    received_keys->push_back(p.second);
  }
  RecvOutputsFromRendezvousAsync(
      rendezvous, nullptr, {}, keys, received_keys,
      [done, rendezvous, received_keys, out, keys](const Status s) {
   std::vector<std::string> mht_18_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSgraph_mgrDTcc mht_18(mht_18_v, 633, "", "./tensorflow/core/distributed_runtime/graph_mgr.cc", "lambda");

        rendezvous->Unref();
        size_t output_size = 0;
        for (int i = 0, end = keys.size(); i < end; ++i) {
          (*out)[keys[i]] = (*received_keys)[i];
          output_size += (*out)[keys[i]].AllocatedBytes();
        }
        metrics::RecordGraphOutputTensors(output_size);
        delete received_keys;
        done(s);
      });
}

void GraphMgr::ExecuteAsync(
    const string& handle, const int64_t step_id, const ExecutorOpts& opts,
    const NamedTensors& in, WorkerSession* session,
    StepStatsCollector* collector, MutableRunGraphResponseWrapper* response,
    CancellationManager* cancellation_manager,
    CoordinationServiceAgent* coordination_service_agent, StatusCallback done) {
   std::vector<std::string> mht_19_v;
   mht_19_v.push_back("handle: \"" + handle + "\"");
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSgraph_mgrDTcc mht_19(mht_19_v, 655, "", "./tensorflow/core/distributed_runtime/graph_mgr.cc", "GraphMgr::ExecuteAsync");

  const uint64 start_time_usecs = Env::Default()->NowMicros();
  profiler::TraceMeProducer activity(
      // To TraceMeConsumers in ExecutorState::Process/Finish or RunGraphDone.
      [step_id] {
        return profiler::TraceMeEncode(
            "RunGraph", {{"id", step_id}, {"_r", 1} /*root_event*/});
      },
      profiler::ContextType::kTfExecutor, step_id,
      profiler::TraceMeLevel::kInfo);
  // Lookup an item. Holds one ref while executing.
  Item* item = nullptr;
  {
    mutex_lock l(mu_);
    auto iter = table_.find(handle);
    if (iter != table_.end()) {
      item = iter->second;
      item->Ref();
    }
  }

  if (item == nullptr) {
    done(errors::Aborted("Graph handle is not found: ", handle));
    return;
  }

  CostGraphDef* cost_graph = nullptr;
  if (response != nullptr) {
    cost_graph = response->mutable_cost_graph();
    if (opts.record_partition_graphs()) {
      for (const ExecutionUnit& unit : item->units) {
        GraphDef graph_def;
        unit.graph->ToGraphDef(&graph_def);
        response->AddPartitionGraph(graph_def);
      }
    }
  }

  RemoteRendezvous* rendezvous = worker_env_->rendezvous_mgr->Find(step_id);
  Status s = rendezvous->Initialize(session);
  CollectiveExecutor::Handle* ce_handle =
      item->collective_graph_key != BuildGraphOptions::kNoCollectiveGraphKey
          ? new CollectiveExecutor::Handle(
                worker_env_->collective_executor_mgr->FindOrCreate(step_id),
                true)
          : nullptr;
  // Sends values specified by the caller.
  size_t input_size = 0;
  if (s.ok()) {
    std::vector<string> keys;
    std::vector<Tensor> tensors_to_send;
    keys.reserve(in.size());
    tensors_to_send.reserve(in.size());
    for (auto& p : in) {
      keys.push_back(p.first);
      tensors_to_send.push_back(p.second);
      input_size += p.second.AllocatedBytes();
    }
    s = SendTensorsToRendezvous(rendezvous, nullptr, {}, keys, tensors_to_send);
  }

  if (!s.ok()) {
    done(s);
    delete ce_handle;
    item->Unref();
    rendezvous->Unref();
    return;
  }

  StartParallelExecutors(
      handle, step_id, item, rendezvous, ce_handle, collector, cost_graph,
      cancellation_manager, session, start_time_usecs,
      coordination_service_agent,
      [item, rendezvous, ce_handle, done, start_time_usecs, input_size,
       step_id](const Status& s) {
        profiler::TraceMeConsumer activity(
            // From TraceMeProducer in GraphMgr::ExecuteAsync.
            [step_id] {
              return profiler::TraceMeEncode("RunGraphDone", {{"id", step_id}});
            },
            profiler::ContextType::kTfExecutor, step_id,
            profiler::TraceMeLevel::kInfo);
        done(s);
        metrics::RecordGraphInputTensors(input_size);
        metrics::UpdateGraphExecTime(Env::Default()->NowMicros() -
                                     start_time_usecs);
        rendezvous->Unref();
        item->Unref();
        delete ce_handle;
      });
}

void GraphMgr::StartParallelExecutors(
    const string& handle, int64_t step_id, Item* item, Rendezvous* rendezvous,
    CollectiveExecutor::Handle* ce_handle, StepStatsCollector* collector,
    CostGraphDef* cost_graph, CancellationManager* cancellation_manager,
    WorkerSession* session, int64_t start_time_usecs,
    CoordinationServiceAgent* coordination_service_agent, StatusCallback done) {
   std::vector<std::string> mht_20_v;
   mht_20_v.push_back("handle: \"" + handle + "\"");
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSgraph_mgrDTcc mht_20(mht_20_v, 756, "", "./tensorflow/core/distributed_runtime/graph_mgr.cc", "GraphMgr::StartParallelExecutors");

  const int num_units = item->units.size();
  CHECK_GE(num_units, 1);
  ScopedStepContainer* step_container = new ScopedStepContainer(
      step_id,
      [this](const string& name) { device_mgr_->ClearContainers({name}); });
  // NOTE: Transfer one ref of rendezvous and item.
  ExecutorBarrier* barrier =
      new ExecutorBarrier(num_units, rendezvous,
                          [this, item, collector, cost_graph, step_container,
                           done](const Status& s) {
                            BuildCostModel(item, collector, cost_graph);
                            done(s);
                            delete step_container;
                          });
  Executor::Args args;
  args.step_id = step_id;
  args.rendezvous = rendezvous;
  args.collective_executor = ce_handle ? ce_handle->get() : nullptr;
  args.cancellation_manager = cancellation_manager;
  args.stats_collector = collector;
  args.step_container = step_container;
  args.sync_on_finish = sync_on_finish_;
  args.start_time_usecs = start_time_usecs;
  args.coordination_service_agent = coordination_service_agent;

  if (LogMemory::IsEnabled()) {
    LogMemory::RecordStep(args.step_id, handle);
  }
  thread::ThreadPool* pool = worker_env_->compute_pool;
  using std::placeholders::_1;
  // Line below is equivalent to this code, but does one less indirect call:
  //  args.runner = [pool](std::function<void()> fn) { pool->Schedule(fn); };
  auto default_runner = std::bind(&thread::ThreadPool::Schedule, pool, _1);
  for (const auto& unit : item->units) {
    // TODO(zhengxq): if the device picks its own threadpool, we need to assign
    //     less threads to the main compute pool by default.
    thread::ThreadPool* device_thread_pool =
        unit.device->tensorflow_device_thread_pool();
    if (!device_thread_pool) {
      args.runner = default_runner;
    } else {
      args.runner =
          std::bind(&thread::ThreadPool::Schedule, device_thread_pool, _1);
    }
    unit.root->RunAsync(args, barrier->Get());
  }
}

void GraphMgr::BuildCostModel(Item* item, StepStatsCollector* collector,
                              CostGraphDef* cost_graph) {
   std::vector<std::string> mht_21_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSgraph_mgrDTcc mht_21(mht_21_v, 809, "", "./tensorflow/core/distributed_runtime/graph_mgr.cc", "GraphMgr::BuildCostModel");

  if (collector && !skip_cost_models_) {
    // Build the cost model
    std::unordered_map<string, const Graph*> device_to_graph;
    for (const auto& unit : item->units) {
      if (unit.build_cost_model > 0) {
        device_to_graph[unit.device->name()] = unit.graph.get();
      }
    }
    collector->BuildCostModel(&cost_model_manager_, device_to_graph);

    if (cost_graph != nullptr) {
      for (const auto& unit : item->units) {
        cost_model_manager_.AddToCostGraphDef(unit.graph.get(), cost_graph)
            .IgnoreError();
      }
    }
  }
}

}  // end namespace tensorflow
