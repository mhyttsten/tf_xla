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
class MHTracer_DTPStensorflowPScorePSdistributed_runtimePSrpcbench_testDTcc {
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
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSrpcbench_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSdistributed_runtimePSrpcbench_testDTcc() {
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

#include <cstdio>
#include <functional>
#include <string>
#include <vector>

#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/distributed_runtime/rpc/grpc_session.h"
#include "tensorflow/core/distributed_runtime/server_lib.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/graph/default_device.h"
#include "tensorflow/core/graph/graph_def_builder.h"
#include "tensorflow/core/lib/core/threadpool.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/lib/strings/stringprintf.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/platform/test_benchmark.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/protobuf/cluster.pb.h"
#include "tensorflow/core/protobuf/tensorflow_server.pb.h"
#include "tensorflow/core/public/session.h"

namespace tensorflow {

static const int kWorkers = 60;
static thread::ThreadPool* worker_threads;

void MakeGRPCCluster(const SessionOptions& options, int n,
                     std::vector<string>* workers,
                     std::vector<DeviceAttributes>* devices) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSrpcbench_testDTcc mht_0(mht_0_v, 215, "", "./tensorflow/core/distributed_runtime/rpcbench_test.cc", "MakeGRPCCluster");

  CHECK_GE(n, 1);

  workers->clear();
  std::vector<int> port(n);
  for (int i = 0; i < n; ++i) {
    port[i] = testing::PickUnusedPortOrDie();
    workers->push_back(strings::StrCat("grpc://localhost:", port[i]));
  }

  int num_cpus = 1;
  int num_gpus = 0;
  auto iter = options.config.device_count().find("CPU");
  if (iter != options.config.device_count().end()) {
    num_cpus = iter->second;
  }
  iter = options.config.device_count().find("GPU");
  if (iter != options.config.device_count().end()) {
    num_gpus = iter->second;
  }

  worker_threads = new thread::ThreadPool(Env::Default(), "worker_threads", n);
  for (int worker_idx = 0; worker_idx < n; ++worker_idx) {
    worker_threads->Schedule([worker_idx, n, num_cpus, num_gpus, &port] {
      ServerDef server;
      server.set_protocol("grpc");
      server.set_job_name("localhost");
      server.set_task_index(worker_idx);

      auto job_def = server.mutable_cluster()->add_job();
      job_def->set_name("localhost");
      for (int i = 0; i < n; i++) {
        (*(job_def->mutable_tasks()))[i] =
            strings::StrCat("localhost:", port[i]);
      }

      auto config = server.mutable_default_session_config();
      (*config->mutable_device_count())["CPU"] = num_cpus;
      (*config->mutable_device_count())["GPU"] = num_gpus;

      std::unique_ptr<ServerInterface> svr;
      TF_CHECK_OK(NewServer(server, &svr));
      TF_CHECK_OK(svr->Start());
      TF_CHECK_OK(svr->Join());
    });
  }

  // Get attributes for all devices.
  LOG(ERROR) << "W '" << (*workers)[0] << "'";
  SessionOptions options_copy(options);
  options_copy.target = (*workers)[0];
  std::unique_ptr<GrpcSession> session;
  TF_CHECK_OK(GrpcSession::Create(options_copy, &session));
  TF_CHECK_OK(session->ListDevices(devices));
}

struct Cluster {
  SessionOptions options;
  std::vector<string> workers;
  std::vector<DeviceAttributes> devices;  // One per process

  Cluster() {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSrpcbench_testDTcc mht_1(mht_1_v, 279, "", "./tensorflow/core/distributed_runtime/rpcbench_test.cc", "Cluster");

    (*options.config.mutable_device_count())["CPU"] = 1;
    options.config.set_intra_op_parallelism_threads(1);
    options.config.set_inter_op_parallelism_threads(1);
    MakeGRPCCluster(options, kWorkers, &workers, &devices);
    LOG(ERROR) << "C " << workers.size() << " " << devices.size() << " "
               << workers[0] << " " << workers[1];
    options.target = workers[0];
  }
};

static const Cluster* GetCluster() {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSrpcbench_testDTcc mht_2(mht_2_v, 293, "", "./tensorflow/core/distributed_runtime/rpcbench_test.cc", "GetCluster");

  static Cluster* result = new Cluster;
  return result;
}

// Make a program with specified number of stages and "width" ops per stage.
GraphDef CreateGraphDef(int num_stages, int width, int tensor_size,
                        bool use_multiple_devices, const Cluster* cluster) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSrpcbench_testDTcc mht_3(mht_3_v, 303, "", "./tensorflow/core/distributed_runtime/rpcbench_test.cc", "CreateGraphDef");

  CHECK_GE(cluster->devices.size(), width);

  using namespace ::tensorflow::ops;  // NOLINT(build/namespaces)

  Scope s = Scope::NewRootScope();

  // x is from the feed.
  Output x = Const(s.WithOpName("x"), 0.0f, {tensor_size, 1});

  // Create stages.
  std::vector<Output> last_stage;
  last_stage.push_back(x);
  for (int i = 0; i < num_stages; i++) {
    std::vector<Output> this_stage;
    for (int j = 0; j < width; j++) {
      Output combine = AddN(
          s.WithDevice(cluster->devices[use_multiple_devices ? j : 0].name()),
          last_stage);
      this_stage.push_back(combine);
    }
    last_stage = this_stage;
  }

  // Create output.
  /* Output y =*/AddN(s.WithOpName("y"), last_stage);

  GraphDef def;
  TF_CHECK_OK(s.ToGraphDef(&def));
  return def;
}

string DebugString(const Tensor& x, const Tensor& y, int tensor_size) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSrpcbench_testDTcc mht_4(mht_4_v, 338, "", "./tensorflow/core/distributed_runtime/rpcbench_test.cc", "DebugString");

  CHECK_EQ(x.NumElements(), tensor_size);
  CHECK_EQ(y.NumElements(), tensor_size);
  auto x_flat = x.flat<float>();
  auto y_flat = y.flat<float>();
  // Just print the first couple of elements of each tensor
  CHECK_GE(tensor_size, 2);
  return strings::Printf("x = [%8.6f %8.6f] y = [%8.6f %8.6f]", x_flat(0),
                         x_flat(1), y_flat(0), y_flat(1));
}

// TODO: Support sharding and depth.
static void BM_Helper(::testing::benchmark::State& state, int width,
                      int num_stages, int tensor_size,
                      bool use_multiple_devices) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSrpcbench_testDTcc mht_5(mht_5_v, 355, "", "./tensorflow/core/distributed_runtime/rpcbench_test.cc", "BM_Helper");

  const Cluster* cluster = GetCluster();

  // Creates a session.
  std::unique_ptr<Session> session(NewSession(cluster->options));
  GraphDef def = CreateGraphDef(num_stages, width, tensor_size,
                                use_multiple_devices, cluster);
  graph::SetDefaultDevice(cluster->devices[0].name(), &def);

  TF_CHECK_OK(session->Create(def));

  // Randomly initialize the input.
  Tensor x(DT_FLOAT, TensorShape({tensor_size, 1}));

  state.SetLabel(
      strings::StrCat(def.node_size(), " nodes; ",
                      use_multiple_devices ? "Multi device" : "Single device",
                      "; tensor bytes/send: ", tensor_size * sizeof(float)));

  std::vector<Tensor> outputs;

  // Do a few warmup iterations.
  for (int i = 0; i < 3; i++) {
    outputs.clear();
    TF_CHECK_OK(session->Run({{"x", x}}, {"y:0"}, {}, &outputs));
    CHECK_EQ(size_t{1}, outputs.size());

    if (i == 0) {
      // Print out x, and y.
      const Tensor& y = outputs[0];
      VLOG(1) << DebugString(x, y, tensor_size);
    }
  }

  // Iterations.
  for (auto s : state) {
    outputs.clear();
    TF_CHECK_OK(session->Run({{"x", x}}, {"y:0"}, {}, &outputs));
    CHECK_EQ(size_t{1}, outputs.size());
  }
  TF_CHECK_OK(session->Close());
}
static void BM_ShardedProgram(::testing::benchmark::State& state) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSrpcbench_testDTcc mht_6(mht_6_v, 400, "", "./tensorflow/core/distributed_runtime/rpcbench_test.cc", "BM_ShardedProgram");

  const int width = state.range(0);
  const int num_stages = state.range(1);

  BM_Helper(state, width, num_stages, 2 /*tensor_size*/, true /*multi-device*/);
}
BENCHMARK(BM_ShardedProgram)
    ->ArgPair(1, 1)
    ->ArgPair(1, 3)
    ->ArgPair(1, 5)
    ->ArgPair(1, 15)
    ->ArgPair(1, 60)
    ->ArgPair(15, 1)
    ->ArgPair(15, 3)
    ->ArgPair(15, 5)
    ->ArgPair(30, 1)
    ->ArgPair(30, 2)
    ->ArgPair(30, 3)
    ->ArgPair(30, 5)
    ->ArgPair(60, 1)
    ->ArgPair(60, 3)
    ->ArgPair(60, 5);

static void BM_RPC(::testing::benchmark::State& state) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSrpcbench_testDTcc mht_7(mht_7_v, 426, "", "./tensorflow/core/distributed_runtime/rpcbench_test.cc", "BM_RPC");

  const int width = state.range(0);
  const int tensor_size = state.range(1);

  BM_Helper(state, width, 2 /*num_stages*/, tensor_size, true /*multi-device*/);
}
BENCHMARK(BM_RPC)->ArgPair(30, 2)->ArgPair(30, 1000)->ArgPair(30, 100000);

static void BM_SingleDevice(::testing::benchmark::State& state) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSrpcbench_testDTcc mht_8(mht_8_v, 437, "", "./tensorflow/core/distributed_runtime/rpcbench_test.cc", "BM_SingleDevice");

  const int width = state.range(0);
  const int num_stages = state.range(1);

  BM_Helper(state, width, num_stages, 2 /*tensor_size*/,
            false /*not multi-device*/);
}
BENCHMARK(BM_SingleDevice)
    ->ArgPair(1, 1)
    ->ArgPair(30, 2)
    ->ArgPair(60, 5)
    ->ArgPair(4, 10000)
    ->ArgPair(1, 1000000);

}  // namespace tensorflow
