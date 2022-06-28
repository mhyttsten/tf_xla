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
class MHTracer_DTPStensorflowPScorePSkernelsPSpartitioned_function_opsDTcc {
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
   MHTracer_DTPStensorflowPScorePSkernelsPSpartitioned_function_opsDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSkernelsPSpartitioned_function_opsDTcc() {
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
#include "tensorflow/core/kernels/partitioned_function_ops.h"

#include "absl/strings/match.h"
#include "tensorflow/core/common_runtime/function.h"
#include "tensorflow/core/common_runtime/input_colocation_exemption_registry.h"
#include "tensorflow/core/common_runtime/rendezvous_mgr.h"
#include "tensorflow/core/framework/attr_value.pb.h"
#include "tensorflow/core/framework/function.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/lib/random/random.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/profiler/lib/traceme.h"
#include "tensorflow/core/protobuf/config.pb.h"
#include "tensorflow/core/protobuf/rewriter_config.pb.h"
#include "tensorflow/core/util/ptr_util.h"
#ifndef IS_MOBILE_PLATFORM
#include "tensorflow/core/grappler/optimizers/meta_optimizer.h"
#endif

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
#include "tensorflow/stream_executor/stream.h"
#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM

namespace tensorflow {

PartitionedCallOp::PartitionedCallOp(OpKernelConstruction* ctx)
    : AsyncOpKernel(ctx),
      func_(new NameAttrList),
      config_proto_(new ConfigProto),
      shared_rendezvous_(false) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSpartitioned_function_opsDTcc mht_0(mht_0_v, 217, "", "./tensorflow/core/kernels/partitioned_function_ops.cc", "PartitionedCallOp::PartitionedCallOp");

  OP_REQUIRES_OK(
      ctx, ctx->GetAttr(FunctionLibraryDefinition::kFuncAttr, func_.get()));
  string deprecated_config_serialized;
  OP_REQUIRES_OK(ctx, ctx->GetAttr("config", &deprecated_config_serialized));
  string config_proto_serialized;
  OP_REQUIRES_OK(ctx, ctx->GetAttr("config_proto", &config_proto_serialized));
  OP_REQUIRES(
      ctx,
      deprecated_config_serialized.empty() || config_proto_serialized.empty(),
      errors::InvalidArgument("Provided both 'config' and 'config_proto' but "
                              "only one should be provided.  Note the "
                              "'config' option is deprecated."));
  if (!deprecated_config_serialized.empty()) {
    OP_REQUIRES(ctx,
                config_proto_->mutable_graph_options()
                    ->mutable_rewrite_options()
                    ->ParseFromString(deprecated_config_serialized),
                errors::InvalidArgument("Unable to parse config string as "
                                        "tensorflow::RewriteOptions proto."));
  } else {
    OP_REQUIRES(
        ctx, config_proto_->ParseFromString(config_proto_serialized),
        errors::InvalidArgument("Unable to parse config_proto string as "
                                "tensorflow::ConfigProto proto."));
  }
  OP_REQUIRES_OK(ctx, ctx->GetAttr("executor_type", &executor_type_));
}

PartitionedCallOp::~PartitionedCallOp() {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSpartitioned_function_opsDTcc mht_1(mht_1_v, 249, "", "./tensorflow/core/kernels/partitioned_function_ops.cc", "PartitionedCallOp::~PartitionedCallOp");

  for (const auto& it : handles_) {
    Status status = it.first->ReleaseHandle(it.second);
    if (!status.ok()) {
      LOG(INFO) << "Ignoring error while destructing PartitionedCallOp: "
                << status.ToString();
    }
  }
}

void PartitionedCallOp::ComputeAsync(OpKernelContext* ctx, DoneCallback done) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSpartitioned_function_opsDTcc mht_2(mht_2_v, 262, "", "./tensorflow/core/kernels/partitioned_function_ops.cc", "PartitionedCallOp::ComputeAsync");

  FunctionLibraryRuntime* lib = ctx->function_library();
  OP_REQUIRES_ASYNC(ctx, lib != nullptr,
                    errors::Internal("No function library is provided."), done);

  // The function body's graph is placed and partitioned the first time
  // `ComputeAsync` is invoked; every subsequent invocation calls each
  // of the function shards yielded by partitioning.
  //
  // The partitioning step yields a set of devices on which to run the
  // function, and exactly one function shard is created for each device
  // Inputs and outputs are pinned to the local device, for simplicity.
  //
  // TODO(akshayka): Support re-sharding the function on subsequent calls,
  // via, e.g., virtual device annotations and a list of device names
  // supplied through an attribute.
  //
  // TODO(akshayka): Add a fastpath for functions that execute on a single
  // device.
  FunctionLibraryRuntime::Handle handle;
  // If we are instantiating the function, we can efficiently extract the
  // inputs while instantiating. Else, we extract them separately below.
  std::vector<Tensor> inputs;
  bool inputs_extracted = false;
  {
    mutex_lock l(mu_);
    auto it = handles_.find(lib);
    if (it == handles_.end()) {
      OP_REQUIRES_OK_ASYNC(ctx, Instantiate(lib, ctx, &inputs, &handle), done);
      inputs_extracted = true;
      handles_[lib] = handle;
    } else {
      handle = it->second;
    }
  }

  if (!inputs_extracted) {
    OpInputList args;
    OP_REQUIRES_OK_ASYNC(ctx, ctx->input_list("args", &args), done);
    inputs.reserve(args.size());
    for (const Tensor& tensor : args) {
      inputs.push_back(tensor);
    }
  }

  RunFunction(handle, inputs, lib, ctx, done);
}

Status PartitionedCallOp::FillOutputDevices(
    const FunctionLibraryRuntime& lib, const Device& cpu_device,
    AttrSlice attrs, FunctionLibraryRuntime::InstantiateOptions* opts) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSpartitioned_function_opsDTcc mht_3(mht_3_v, 315, "", "./tensorflow/core/kernels/partitioned_function_ops.cc", "PartitionedCallOp::FillOutputDevices");

  const FunctionLibraryDefinition* flib = lib.GetFunctionLibraryDefinition();
  const FunctionDef* fdef = flib->Find(func_->name());
  if (fdef == nullptr) {
    return errors::NotFound("Failed to find definition for function \"",
                            func_->name(), "\"");
  }
  auto func_attrs = fdef->attr();
  auto attr = func_attrs.find(FunctionLibraryDefinition::kSharedRendezvousAttr);
  if (attr != func_attrs.end() && attr->second.b()) {
    shared_rendezvous_ = true;
  }

  bool is_type_list;
  for (const OpDef::ArgDef& ret_def : fdef->signature().output_arg()) {
    DataTypeVector dtypes;
    TF_RETURN_IF_ERROR(ArgNumType(attrs, ret_def, &is_type_list, &dtypes));
    for (DataType dtype : dtypes) {
      if (dtype == DT_RESOURCE) {
        // Resource memory type is HOST_MEMORY, however the actual resource
        // might be allocated on a device. We leave output device for resource
        // outputs empty, and rely on a Placer and colocation constraints to
        // infer correct placement for the function output.
        opts->output_devices.push_back("");
      } else {
        opts->output_devices.push_back(opts->target);
      }
    }
  }
  return Status::OK();
}

Status PartitionedCallOp::Instantiate(FunctionLibraryRuntime* lib,
                                      OpKernelContext* ctx,
                                      std::vector<Tensor>* inputs,
                                      FunctionLibraryRuntime::Handle* handle) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSpartitioned_function_opsDTcc mht_4(mht_4_v, 353, "", "./tensorflow/core/kernels/partitioned_function_ops.cc", "PartitionedCallOp::Instantiate");

  FunctionLibraryRuntime::InstantiateOptions opts;
  const auto* config = (ctx->function_library())
                           ? ctx->function_library()->config_proto()
                           : nullptr;
  if (config) {
    opts.config_proto = *config;
  }

#ifndef IS_MOBILE_PLATFORM
  // Android tf library does not include grappler.
  grappler::GrapplerItem::OptimizationOptions optimization_options;
  // Tensorflow 2.0 in eager mode with automatic control dependencies will
  // prune all nodes that are not in the transitive fanin of the fetch nodes.
  // However because the function will be executed via FunctionLibraryRuntime,
  // and current function implementation does not prune stateful and dataset
  // ops, we rely on Grappler to do the correct graph pruning.
  optimization_options.allow_pruning_stateful_and_dataset_ops = true;

  // All the nested function calls will be executed and optimized via
  // PartitionedCallOp, there is no need to optimize functions now.
  optimization_options.optimize_function_library = false;

  opts.optimize_graph_fn =
      std::bind(grappler::OptimizeGraph, std::placeholders::_1,
                std::placeholders::_2, std::placeholders::_3,
                std::placeholders::_4, std::placeholders::_5, *config_proto_,
                func_->name(), optimization_options, std::placeholders::_6);
#endif

  // In some contexts like running the graph to evaluate constants,
  // the FLR won't have any device.
  opts.target = lib->device() == nullptr ? "" : lib->device()->name();
  opts.is_multi_device_function = true;
  opts.graph_collector = ctx->graph_collector();
  opts.executor_type = executor_type_;

  OpInputList args;
  TF_RETURN_IF_ERROR(ctx->input_list("args", &args));
  Device* cpu_device;
  TF_RETURN_IF_ERROR(lib->device_mgr()->LookupDevice("CPU:0", &cpu_device));

  inputs->reserve(args.size());
  for (const Tensor& tensor : args) {
    inputs->push_back(tensor);
    DataType dtype = tensor.dtype();
    if (dtype == DT_RESOURCE) {
      const ResourceHandle& handle = tensor.flat<ResourceHandle>()(0);
      opts.input_devices.push_back(handle.device());
    } else {
      opts.input_devices.push_back(opts.target);
    }
  }

  TF_RETURN_IF_ERROR(
      FillOutputDevices(*lib, *cpu_device, AttrSlice(&func_->attr()), &opts));

  TF_RETURN_IF_ERROR(
      lib->Instantiate(func_->name(), AttrSlice(&func_->attr()), opts, handle));
  return Status::OK();
}

void PartitionedCallOp::RunFunction(FunctionLibraryRuntime::Handle handle,
                                    const std::vector<Tensor>& inputs,
                                    FunctionLibraryRuntime* lib,
                                    OpKernelContext* ctx, DoneCallback done) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSpartitioned_function_opsDTcc mht_5(mht_5_v, 421, "", "./tensorflow/core/kernels/partitioned_function_ops.cc", "PartitionedCallOp::RunFunction");

  FunctionLibraryRuntime::Options run_opts;
  ResourceMgr* resource_mgr = lib->device()->resource_manager();
  ScopedStepContainer* step_container = new ScopedStepContainer(
      run_opts.step_id, [resource_mgr](const string& name) {
        resource_mgr->Cleanup(name).IgnoreError();
      });
  run_opts.step_container = step_container;
  run_opts.cancellation_manager = ctx->cancellation_manager();
  run_opts.stats_collector = ctx->stats_collector();
  run_opts.collective_executor = ctx->collective_executor();
  // TODO(akshayka): Consider selecting a runner on a per-device basis,
  // i.e., using device-specific threadpools when available.
  run_opts.runner = ctx->runner();
  run_opts.run_all_kernels_inline = ctx->run_all_kernels_inline();
  run_opts.source_device =
      lib->device() == nullptr ? "" : lib->device()->name();
  run_opts.allow_dead_tensors = true;
  if (shared_rendezvous_) {
    run_opts.rendezvous = ctx->rendezvous();
  }

  std::vector<Tensor>* rets = new std::vector<Tensor>;
  const string& func_name = func_->name();
  profiler::TraceMe trace_me("PartitionedCallOp");
  lib->Run(run_opts, handle, inputs, rets,
           [rets, done = std::move(done), ctx, func_name,
            step_container](const Status& status) {
             if (!status.ok()) {
               const string function_and_msg =
                   strings::StrCat(errors::FormatFunctionForError(func_name),
                                   " ", status.error_message());
               ctx->SetStatus(
                   errors::CreateWithUpdatedMessage(status, function_and_msg));
             } else {
               for (int i = 0; i < rets->size(); ++i) {
                 ctx->set_output(i, (*rets)[i]);
               }
             }
             delete rets;
             delete step_container;
             done();
           });
}

REGISTER_KERNEL_BUILDER(Name("PartitionedCall").Device(DEVICE_CPU),
                        PartitionedCallOp);
REGISTER_KERNEL_BUILDER(Name("StatefulPartitionedCall").Device(DEVICE_CPU),
                        PartitionedCallOp);
REGISTER_KERNEL_BUILDER(Name("PartitionedCall").Device(DEVICE_DEFAULT),
                        PartitionedCallOp);
REGISTER_KERNEL_BUILDER(Name("StatefulPartitionedCall").Device(DEVICE_DEFAULT),
                        PartitionedCallOp);

REGISTER_INPUT_COLOCATION_EXEMPTION("PartitionedCall");
REGISTER_INPUT_COLOCATION_EXEMPTION("StatefulPartitionedCall");

}  // namespace tensorflow
