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
class MHTracer_DTPStensorflowPScorePStpuPStpu_global_initDTcc {
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
   MHTracer_DTPStensorflowPScorePStpuPStpu_global_initDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePStpuPStpu_global_initDTcc() {
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
#include "tensorflow/core/tpu/tpu_global_init.h"

#include <algorithm>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/memory/memory.h"
#include "absl/strings/string_view.h"
#include "tensorflow/cc/framework/scope.h"
#include "tensorflow/cc/ops/tpu_configuration_ops.h"
#include "tensorflow/core/common_runtime/device.h"
#include "tensorflow/core/common_runtime/device_factory.h"
#include "tensorflow/core/common_runtime/device_mgr.h"
#include "tensorflow/core/common_runtime/device_set.h"
#include "tensorflow/core/common_runtime/graph_constructor.h"
#include "tensorflow/core/common_runtime/graph_runner.h"
#include "tensorflow/core/common_runtime/optimization_registry.h"
#include "tensorflow/core/common_runtime/session_factory.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/public/session.h"
#include "tensorflow/core/public/session_options.h"
#include "tensorflow/core/tpu/graph_rewrite/distributed_tpu_configuration_rewrite_pass.h"
#include "tensorflow/core/tpu/graph_rewrite/distributed_tpu_rewrite_helpers.h"
#include "tensorflow/core/tpu/tpu_defs.h"
#include "tensorflow/core/util/device_name_utils.h"

namespace tensorflow {

namespace {

ABSL_CONST_INIT static absl::Mutex global_init_tpu_mutex(absl::kConstInit);
static tpu::TopologyProto* global_tpu_topology
    ABSL_GUARDED_BY(global_init_tpu_mutex) = nullptr;

constexpr char kTaskSpec[] = "/job:localhost/replica:0/task:0";

Status CreateDeviceMgr(Env* env, std::unique_ptr<DeviceMgr>* device_mgr) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePStpuPStpu_global_initDTcc mht_0(mht_0_v, 225, "", "./tensorflow/core/tpu/tpu_global_init.cc", "CreateDeviceMgr");

  SessionOptions session_options;
  session_options.env = env;
  std::vector<std::unique_ptr<Device>> devices;
  DeviceFactory* device_factory = DeviceFactory::GetFactory(DEVICE_TPU_SYSTEM);
  if (device_factory == nullptr) {
    return errors::Internal("Unable to initialize DeviceFactory.");
  }
  TF_RETURN_IF_ERROR(
      device_factory->CreateDevices(session_options, kTaskSpec, &devices));
  *device_mgr = absl::make_unique<DynamicDeviceMgr>(std::move(devices));
  return Status::OK();
}

void DeviceSetFromDeviceMgr(const DeviceMgr& device_mgr,
                            DeviceSet* device_set) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePStpuPStpu_global_initDTcc mht_1(mht_1_v, 243, "", "./tensorflow/core/tpu/tpu_global_init.cc", "DeviceSetFromDeviceMgr");

  int devices_added = 0;
  for (auto d : device_mgr.ListDevices()) {
    device_set->AddDevice(d);
    if (devices_added == 0) {
      device_set->set_client_device(d);
    }
    ++devices_added;
  }
}

const std::string GetTPUSystemDevice(absl::string_view job_name) {
   std::vector<std::string> mht_2_v;
   mht_2_v.push_back("job_name: \"" + std::string(job_name.data(), job_name.size()) + "\"");
   MHTracer_DTPStensorflowPScorePStpuPStpu_global_initDTcc mht_2(mht_2_v, 258, "", "./tensorflow/core/tpu/tpu_global_init.cc", "GetTPUSystemDevice");

  if (job_name.empty()) {
    return DeviceNameUtils::LocalName(DEVICE_TPU_SYSTEM, 0);
  } else {
    return absl::StrCat("/job:", job_name, "/device:TPU_SYSTEM:0");
  }
}

Status ConstructDistributedInitializationGraph(absl::string_view job_name,
                                               const DeviceSet& device_set,
                                               Graph* graph_to_run) {
   std::vector<std::string> mht_3_v;
   mht_3_v.push_back("job_name: \"" + std::string(job_name.data(), job_name.size()) + "\"");
   MHTracer_DTPStensorflowPScorePStpuPStpu_global_initDTcc mht_3(mht_3_v, 272, "", "./tensorflow/core/tpu/tpu_global_init.cc", "ConstructDistributedInitializationGraph");

  std::unique_ptr<Graph> graph(new Graph(OpRegistry::Global()));
  GraphOptimizationPassOptions options;
  options.graph = &graph;
  options.device_set = &device_set;
  {
    Scope scope = Scope::NewRootScope();
    auto init_op = ops::ConfigureDistributedTPU(
        scope.WithOpName("InitializeTPUSystemGlobally")
            .WithDevice(GetTPUSystemDevice(job_name)),
        ops::ConfigureDistributedTPU::IsGlobalInit(true));
    TF_RETURN_IF_ERROR(scope.ToGraph(options.graph->get()));
  }
  DistributedTPUConfigurationRewritePass rewriter;
  TF_RETURN_IF_ERROR(rewriter.Run(options));

  // Graph doesn't update the node-def's after adding edges, which causes
  // node-def validation to fail in the executor. So we explicitly do a
  // round-trip through GraphDef, so that node-defs are updated.
  TF_RETURN_IF_ERROR(
      ConvertGraphDefToGraph({}, graph->ToGraphDefDebug(), graph_to_run));

  return Status::OK();
}

Status InitializeFromSession(absl::string_view session_target,
                             const Graph* graph_to_run,
                             std::vector<Tensor>* outputs) {
   std::vector<std::string> mht_4_v;
   mht_4_v.push_back("session_target: \"" + std::string(session_target.data(), session_target.size()) + "\"");
   MHTracer_DTPStensorflowPScorePStpuPStpu_global_initDTcc mht_4(mht_4_v, 303, "", "./tensorflow/core/tpu/tpu_global_init.cc", "InitializeFromSession");

  tensorflow::SessionOptions s_opts;
  s_opts.target = std::string(session_target);

  std::unique_ptr<tensorflow::Session> sess(tensorflow::NewSession(s_opts));

  GraphDef g_def;
  graph_to_run->ToGraphDef(&g_def);

  TF_RETURN_IF_ERROR(sess->Create(g_def));
  TF_RETURN_IF_ERROR(
      sess->Run({}, {"InitializeTPUSystemGlobally:0"}, {}, outputs));

  return Status::OK();
}

}  // namespace

Status InitializeTPUSystemGlobally(absl::string_view job_name,
                                   absl::string_view session_target,
                                   const DeviceSet& device_set, Env* env,
                                   tpu::TopologyProto* tpu_topology) {
   std::vector<std::string> mht_5_v;
   mht_5_v.push_back("job_name: \"" + std::string(job_name.data(), job_name.size()) + "\"");
   mht_5_v.push_back("session_target: \"" + std::string(session_target.data(), session_target.size()) + "\"");
   MHTracer_DTPStensorflowPScorePStpuPStpu_global_initDTcc mht_5(mht_5_v, 329, "", "./tensorflow/core/tpu/tpu_global_init.cc", "InitializeTPUSystemGlobally");

  VLOG(1) << "InitializeTpuSystemGlobally";

  absl::MutexLock lock(&global_init_tpu_mutex);
  if (global_tpu_topology != nullptr) {
    *tpu_topology = *global_tpu_topology;
    return Status::OK();
  }

  std::unique_ptr<Graph> graph_to_run(new Graph(OpRegistry::Global()));

  DeviceNameUtils::ParsedName system_spec;
  Device* tpu_system_device;

  std::string task_spec =
      job_name.empty() ? kTaskSpec
                       : absl::StrCat("/job:", job_name, "/replica:0/task:0");
  // Placed here, much before usage, to get a sane error if TPU_SYSTEM_DEVICE
  // hasn't been linked in. Otherwise we may get a cryptic error down the line.
  TF_RETURN_IF_ERROR(DistributedTPURewriteHelpers::GetSystemDevice(
      task_spec, device_set, &system_spec, &tpu_system_device));

  TF_RETURN_IF_ERROR(ConstructDistributedInitializationGraph(
      job_name, device_set, graph_to_run.get()));

  std::vector<Tensor> outputs;
  // Being a bit conservative here to run non-distributed initialization with
  // graph runner.
  // TODO(hthu): Re-evaluate the choice of using session for running the
  // initialization graph given that we need to a session in distributed
  // initialization anyway.
  if (session_target.empty()) {
    GraphRunner graph_runner(tpu_system_device);
    TF_RETURN_IF_ERROR(graph_runner.Run(graph_to_run.get(), nullptr, {},
                                        {"InitializeTPUSystemGlobally:0"},
                                        &outputs));
  } else {
    TF_RETURN_IF_ERROR(
        InitializeFromSession(session_target, graph_to_run.get(), &outputs));
  }

  if (outputs.empty()) {
    return errors::Internal("No output from running TPU initialization.");
  }

  global_tpu_topology = new tpu::TopologyProto();
  if (!global_tpu_topology->ParseFromString(outputs[0].scalar<tstring>()())) {
    return errors::Internal(
        "Unable to parse output from running TPU initialization as "
        "TopologyProto proto.");
  }

  *tpu_topology = *global_tpu_topology;
  return Status::OK();
}

// NOTE: Session would have been the obvious first choice to run the graph
// here, but instead we use a GraphRunner because Session creates a global
// EigenThreadPool based on the SessionOptions it receives the first time it
// runs. This means that we need to create the right options and pass it to this
// API to make it work correctly. We felt it was an onerous restriction to place
// on the API, so we went with the current approach.
Status InitializeTPUSystemGlobally(Env* env, tpu::TopologyProto* tpu_topology) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePStpuPStpu_global_initDTcc mht_6(mht_6_v, 394, "", "./tensorflow/core/tpu/tpu_global_init.cc", "InitializeTPUSystemGlobally");

  std::unique_ptr<DeviceMgr> device_mgr;
  TF_RETURN_IF_ERROR(CreateDeviceMgr(env, &device_mgr));
  DeviceSet device_set;
  DeviceSetFromDeviceMgr(*device_mgr, &device_set);

  return InitializeTPUSystemGlobally(/*job_name=*/absl::string_view(),
                                     /*session_target=*/absl::string_view(),
                                     device_set, env, tpu_topology);
}

Status InitializeTPUSystemGlobally() {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePStpuPStpu_global_initDTcc mht_7(mht_7_v, 408, "", "./tensorflow/core/tpu/tpu_global_init.cc", "InitializeTPUSystemGlobally");

  tensorflow::tpu::TopologyProto tpu_topology;
  return InitializeTPUSystemGlobally(tensorflow::Env::Default(), &tpu_topology);
}

}  // namespace tensorflow
