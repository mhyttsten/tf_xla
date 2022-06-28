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
class MHTracer_DTPStensorflowPScorePScommon_runtimePSgraph_runnerDTcc {
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
   MHTracer_DTPStensorflowPScorePScommon_runtimePSgraph_runnerDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePScommon_runtimePSgraph_runnerDTcc() {
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

// TODO(skyewm): this is necessary to make the single_threaded_cpu_device.h
// include work. Some other include must be including eigen without defining
// this. Consider defining in this in a BUILD rule.
#define EIGEN_USE_THREADS

#include "tensorflow/core/common_runtime/graph_runner.h"

#include "tensorflow/core/common_runtime/device.h"
#include "tensorflow/core/common_runtime/device_factory.h"
#include "tensorflow/core/common_runtime/executor.h"
#include "tensorflow/core/common_runtime/graph_constructor.h"
#include "tensorflow/core/common_runtime/memory_types.h"
#include "tensorflow/core/common_runtime/rendezvous_mgr.h"
#include "tensorflow/core/common_runtime/single_threaded_cpu_device.h"
#include "tensorflow/core/framework/log_memory.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor_util.h"
#include "tensorflow/core/framework/versions.pb.h"
#include "tensorflow/core/graph/algorithm.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/graph/node_builder.h"
#include "tensorflow/core/graph/subgraph.h"
#include "tensorflow/core/lib/core/threadpool.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/public/session_options.h"

namespace tensorflow {

namespace {

// A simple rendezvous class.
// Assumes a single sender and a single receiver, no duplicate sends, and no
// sends of dead tensors.
class SimpleRendezvous : public RendezvousInterface {
 public:
  explicit SimpleRendezvous() {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSgraph_runnerDTcc mht_0(mht_0_v, 221, "", "./tensorflow/core/common_runtime/graph_runner.cc", "SimpleRendezvous");
}

  Status Send(const ParsedKey& parsed, const Args& send_args, const Tensor& val,
              const bool is_dead) override {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSgraph_runnerDTcc mht_1(mht_1_v, 227, "", "./tensorflow/core/common_runtime/graph_runner.cc", "Send");

    if (is_dead) {
      return errors::Internal("Send of a dead tensor");
    }

    mutex_lock l(mu_);
    string edge_name(parsed.edge_name);
    if (table_.count(edge_name) > 0) {
      return errors::Internal("Send of an already sent tensor");
    }
    table_[edge_name] = val;
    return Status::OK();
  }

  void RecvAsync(const ParsedKey& parsed, const Args& recv_args,
                 DoneCallback done) override {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSgraph_runnerDTcc mht_2(mht_2_v, 245, "", "./tensorflow/core/common_runtime/graph_runner.cc", "RecvAsync");

    Tensor tensor;
    Status status = Status::OK();
    {
      string key(parsed.edge_name);
      mutex_lock l(mu_);
      if (table_.count(key) <= 0) {
        status = errors::Internal("Did not find key ", key);
      } else {
        tensor = table_[key];
      }
    }
    done(status, Args{}, recv_args, tensor, false);
  }

  void StartAbort(const Status& status) override {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSgraph_runnerDTcc mht_3(mht_3_v, 263, "", "./tensorflow/core/common_runtime/graph_runner.cc", "StartAbort");
}

 private:
  typedef std::unordered_map<string, Tensor> Table;

  mutex mu_;
  Table table_ TF_GUARDED_BY(mu_);
};

}  // namespace

GraphRunner::GraphRunner(Env* env)
    : device_deleter_(NewSingleThreadedCpuDevice(env)),
      device_(device_deleter_.get()) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSgraph_runnerDTcc mht_4(mht_4_v, 279, "", "./tensorflow/core/common_runtime/graph_runner.cc", "GraphRunner::GraphRunner");
}
GraphRunner::GraphRunner(Device* device) : device_(device) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSgraph_runnerDTcc mht_5(mht_5_v, 283, "", "./tensorflow/core/common_runtime/graph_runner.cc", "GraphRunner::GraphRunner");
}

GraphRunner::~GraphRunner() {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSgraph_runnerDTcc mht_6(mht_6_v, 288, "", "./tensorflow/core/common_runtime/graph_runner.cc", "GraphRunner::~GraphRunner");
}

Status GraphRunner::Run(Graph* graph, FunctionLibraryRuntime* function_library,
                        const NamedTensorList& inputs,
                        const std::vector<string>& output_names,
                        std::vector<Tensor>* outputs) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSgraph_runnerDTcc mht_7(mht_7_v, 296, "", "./tensorflow/core/common_runtime/graph_runner.cc", "GraphRunner::Run");

  if (device_ == nullptr) {
    return errors::NotFound("Cannot find a device for GraphRunner.");
  }

  if (function_library && function_library->device() &&
      function_library->device()->device_type() != device_->device_type()) {
    // Mismatch between function_library's device_type and device_'s
    // device_type.
    // TODO(matthewmurray) Can we create a new FunctionLibraryRuntime that is
    // identical to function_library except that it uses the given 'device_'?
    VLOG(1) << "Cannot run on: " << device_->device_type()
            << " with a function library for a "
            << function_library->device()->device_type() << " device.";
    function_library = nullptr;
  }

  // TODO(vrv): Instead of copying the entire graph, consider modifying
  // the existing graph, and then removing those removed edges.
  // prior to returning.
  std::unique_ptr<Graph> graph_to_run(new Graph(graph->op_registry()));
  CopyGraph(*graph, graph_to_run.get());

  SimpleRendezvous rendez;

  // Extract the input names and keys, and feed in the inputs.
  std::vector<string> input_names;
  for (const auto& in : inputs) {
    const string& tensor_name = in.first;
    input_names.emplace_back(tensor_name);
    string full_key = Rendezvous::CreateKey("/device:CPU:0", 1, "/device:CPU:1",
                                            tensor_name, FrameAndIter(0, 0));
    Rendezvous::ParsedKey parsed;
    TF_RETURN_IF_ERROR(Rendezvous::ParseKey(full_key, &parsed));
    TF_RETURN_IF_ERROR(rendez.Send(parsed, Rendezvous::Args(), in.second,
                                   false /* is_dead */));
  }

  // Call RewriteGraphForExecution
  subgraph::RewriteGraphMetadata metadata;
  TF_RETURN_IF_ERROR(subgraph::RewriteGraphForExecution(
      graph_to_run.get(), input_names, output_names, {} /* target nodes */,
      device_->attributes(), false /* use_function_convention */, &metadata));

  // Create the local executor and the Rendezvous for fetching back the
  // constants.

  // Run operators on the local thread. We should not need concurrency here; we
  // should not be running expensive operators.
  auto runner = [](Executor::Args::Closure c) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSgraph_runnerDTcc mht_8(mht_8_v, 348, "", "./tensorflow/core/common_runtime/graph_runner.cc", "lambda");
 c(); };

  LocalExecutorParams params;
  // The ownership of the output tensors are bound to this device's lifetime.
  params.device = device_;
  params.function_library = function_library;
  const int producer = graph_to_run->versions().producer();
  params.create_kernel = [this, function_library, producer](
                             const std::shared_ptr<const NodeProperties>& props,
                             OpKernel** kernel) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSgraph_runnerDTcc mht_9(mht_9_v, 360, "", "./tensorflow/core/common_runtime/graph_runner.cc", "lambda");

    return CreateNonCachedKernel(device_, function_library, props, producer,
                                 kernel);
  };
  params.delete_kernel = [](OpKernel* kernel) {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSgraph_runnerDTcc mht_10(mht_10_v, 367, "", "./tensorflow/core/common_runtime/graph_runner.cc", "lambda");
 delete kernel; };

  Executor* executor;
  TF_RETURN_IF_ERROR(NewLocalExecutor(params, *graph_to_run, &executor));
  std::unique_ptr<Executor> executor_unref(executor);

  Executor::Args args;
  // NOTE: we could take a step id as an argument, but currently
  // there is no need since we never trace the running of a graph
  // called via this method.
  args.step_id = LogMemory::CONSTANT_FOLDING_STEP_ID;
  args.runner = runner;
  args.rendezvous = &rendez;
  // NOTE: Use of graph runner is limited to single-device executions
  // so a CollectiveExecutor should never be required.
  args.collective_executor = nullptr;

  CancellationManager cancellation_manager;
  args.cancellation_manager = &cancellation_manager;

  // Run the graph.
  TF_RETURN_IF_ERROR(executor->Run(args));

  outputs->resize(output_names.size());
  for (size_t i = 0; i < output_names.size(); ++i) {
    const string& output_key =
        Rendezvous::CreateKey("/device:CPU:0", 1, "/device:CPU:1",
                              output_names[i], FrameAndIter(0, 0));
    Rendezvous::ParsedKey parsed;
    TF_RETURN_IF_ERROR(Rendezvous::ParseKey(output_key, &parsed));
    bool is_dead;
    Tensor output_tensor;
    TF_RETURN_IF_ERROR(
        rendez.Recv(parsed, Rendezvous::Args(), &output_tensor, &is_dead));
    // Does a deep copy so that ownership of the tensor isn't tied to the
    // allocator of the cpu device we created above. The allocator could be
    // deleted along with the device.
    (*outputs)[i] = tensor::DeepCopy(output_tensor);
  }

  return Status::OK();
}

}  // namespace tensorflow
