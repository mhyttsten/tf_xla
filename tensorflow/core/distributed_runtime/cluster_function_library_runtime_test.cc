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
class MHTracer_DTPStensorflowPScorePSdistributed_runtimePScluster_function_library_runtime_testDTcc {
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
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePScluster_function_library_runtime_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSdistributed_runtimePScluster_function_library_runtime_testDTcc() {
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
#include "tensorflow/core/distributed_runtime/cluster_function_library_runtime.h"

#include "tensorflow/core/common_runtime/function_testlib.h"
#include "tensorflow/core/distributed_runtime/rpc/grpc_channel.h"
#include "tensorflow/core/distributed_runtime/rpc/grpc_testlib.h"
#include "tensorflow/core/distributed_runtime/rpc/grpc_worker_cache.h"
#include "tensorflow/core/distributed_runtime/worker_session.h"
#include "tensorflow/core/framework/function_testlib.h"
#include "tensorflow/core/framework/tensor_testutil.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/util/equal_graph_def.h"

namespace tensorflow {

class ClusterFunctionLibraryRuntimeTest : public ::testing::Test {
 public:
  ClusterFunctionLibraryRuntimeTest() {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePScluster_function_library_runtime_testDTcc mht_0(mht_0_v, 201, "", "./tensorflow/core/distributed_runtime/cluster_function_library_runtime_test.cc", "ClusterFunctionLibraryRuntimeTest");

    SessionOptions options;
    TF_CHECK_OK(test::TestCluster::MakeTestCluster(options, 2, &cluster_));
    GrpcChannelSpec spec;
    TF_CHECK_OK(spec.AddHostPortsJob("localhost", cluster_->targets()));
    ChannelCreationFunction channel_func =
        ConvertToChannelCreationFunction(NewHostPortGrpcChannel);
    grpc_worker_env_.reset(CreateGrpcWorkerEnv());
    std::shared_ptr<GrpcChannelCache> channel_cache(
        NewGrpcChannelCache(spec, channel_func));
    std::unique_ptr<WorkerCacheInterface> worker_cache(
        NewGrpcWorkerCache(channel_cache, grpc_worker_env_.get()));

    worker_session_.reset(new WorkerSession(
        "cluster_test_session", "/job:localhost/replica:0/task:0",
        std::move(worker_cache), std::unique_ptr<DeviceMgr>(),
        std::unique_ptr<GraphMgr>(), nullptr));

    cluster_flr_.reset(new ClusterFunctionLibraryRuntime(worker_session_.get(),
                                                         true, nullptr));
  }

  Status ConstructFunctionGraphHelper(
      const OpDef& sig, test::function::Attrs attrs,
      const FunctionLibraryRuntime::InstantiateOptions& options,
      const FunctionLibraryDefinition& lib_def, GraphDef* g,
      std::vector<string>* send_keys, std::vector<string>* recv_keys) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePScluster_function_library_runtime_testDTcc mht_1(mht_1_v, 230, "", "./tensorflow/core/distributed_runtime/cluster_function_library_runtime_test.cc", "ConstructFunctionGraphHelper");

    return ClusterFunctionLibraryRuntime::ConstructFunctionGraph(
        sig, attrs, options, lib_def, g, send_keys, recv_keys);
  }

  void Instantiate(const string& function_name,
                   const FunctionLibraryDefinition& lib_def,
                   test::function::Attrs attrs,
                   const FunctionLibraryRuntime::InstantiateOptions& options,
                   FunctionLibraryRuntime::LocalHandle* local_handle,
                   FunctionLibraryRuntime::DoneCallback done) {
   std::vector<std::string> mht_2_v;
   mht_2_v.push_back("function_name: \"" + function_name + "\"");
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePScluster_function_library_runtime_testDTcc mht_2(mht_2_v, 244, "", "./tensorflow/core/distributed_runtime/cluster_function_library_runtime_test.cc", "Instantiate");

    cluster_flr_->Instantiate(function_name, lib_def, attrs, options,
                              local_handle, done);
  }

  Status InstantiateAndRun(
      const string& function_name, const FunctionLibraryDefinition& lib_def,
      test::function::Attrs attrs,
      const FunctionLibraryRuntime::InstantiateOptions& options,
      const std::vector<Tensor>& args, std::vector<Tensor*> rets) {
   std::vector<std::string> mht_3_v;
   mht_3_v.push_back("function_name: \"" + function_name + "\"");
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePScluster_function_library_runtime_testDTcc mht_3(mht_3_v, 257, "", "./tensorflow/core/distributed_runtime/cluster_function_library_runtime_test.cc", "InstantiateAndRun");

    FunctionLibraryRuntime::LocalHandle handle;
    Status status;
    Notification instantiate_done;
    cluster_flr_->Instantiate(function_name, lib_def, attrs, options, &handle,
                              [&status, &instantiate_done](const Status& s) {
                                status = s;
                                instantiate_done.Notify();
                              });
    instantiate_done.WaitForNotification();
    if (!status.ok()) {
      return status;
    }

    Notification done;
    FunctionLibraryRuntime::Options opts;
    std::vector<Tensor> out;
    cluster_flr_->Run(opts, handle, args, &out,
                      [&status, &done](const Status& s) {
                        status = s;
                        done.Notify();
                      });
    done.WaitForNotification();
    if (!status.ok()) {
      return status;
    }
    CHECK_EQ(rets.size(), out.size());
    for (size_t i = 0; i < rets.size(); ++i) {
      *rets[i] = out[i];
    }

    return Status::OK();
  }

 protected:
  std::unique_ptr<test::TestCluster> cluster_;
  std::unique_ptr<WorkerSession> worker_session_;
  std::unique_ptr<ClusterFunctionLibraryRuntime> cluster_flr_;
  std::unique_ptr<GrpcWorkerEnv> grpc_worker_env_;
};

TEST_F(ClusterFunctionLibraryRuntimeTest, ConstructFunctionGraph) {
  GraphDef actual;
  std::vector<string> send_keys, recv_keys;
  FunctionDefLibrary proto;
  *(proto.add_function()) = test::function::Swap();
  FunctionLibraryDefinition lib_def(OpRegistry::Global(), proto);

  FunctionLibraryRuntime::InstantiateOptions instantiate_opts;
  instantiate_opts.target = "/job:a/replica:0/task:0/device:CPU:0";
  TF_CHECK_OK(ConstructFunctionGraphHelper(
      test::function::Swap().signature(), {{"T", DT_FLOAT}}, instantiate_opts,
      lib_def, &actual, &send_keys, &recv_keys));
  GraphDef expected;
  protobuf::TextFormat::ParseFromString(R"(
node {
  name: "_recv_i0_0"
  op: "_Recv"
  device: "/job:a/replica:0/task:0/device:CPU:0"
  attr {
    key: "client_terminated"
    value {
      b: true
    }
  }
  attr {
    key: "recv_device"
    value {
      s: "/job:a/replica:0/task:0/device:CPU:0"
    }
  }
  attr {
    key: "send_device"
    value {
      s: "/job:a/replica:0/task:0/device:CPU:0"
    }
  }
  attr {
    key: "send_device_incarnation"
    value {
      i: 1
    }
  }
  attr {
    key: "tensor_name"
    value {
      s: "i0"
    }
  }
  attr {
    key: "tensor_type"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "_recv_i1_1"
  op: "_Recv"
  device: "/job:a/replica:0/task:0/device:CPU:0"
  attr {
    key: "client_terminated"
    value {
      b: true
    }
  }
  attr {
    key: "recv_device"
    value {
      s: "/job:a/replica:0/task:0/device:CPU:0"
    }
  }
  attr {
    key: "send_device"
    value {
      s: "/job:a/replica:0/task:0/device:CPU:0"
    }
  }
  attr {
    key: "send_device_incarnation"
    value {
      i: 1
    }
  }
  attr {
    key: "tensor_name"
    value {
      s: "i1"
    }
  }
  attr {
    key: "tensor_type"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "Func/Swap/input/_0"
  op: "Identity"
  input: "_recv_i0_0"
  device: "/job:a/replica:0/task:0/device:CPU:0"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "Func/Swap/input/_1"
  op: "Identity"
  input: "_recv_i1_1"
  device: "/job:a/replica:0/task:0/device:CPU:0"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "Swap/o0"
  op: "Identity"
  input: "Func/Swap/input/_1"
  device: "/job:a/replica:0/task:0/device:CPU:0"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "Swap/o1"
  op: "Identity"
  input: "Func/Swap/input/_0"
  device: "/job:a/replica:0/task:0/device:CPU:0"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "Func/Swap/output/_2"
  op: "Identity"
  input: "Swap/o0"
  device: "/job:a/replica:0/task:0/device:CPU:0"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "Func/Swap/output/_3"
  op: "Identity"
  input: "Swap/o1"
  device: "/job:a/replica:0/task:0/device:CPU:0"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "_send_o0_0"
  op: "_Send"
  input: "Func/Swap/output/_2"
  device: "/job:a/replica:0/task:0/device:CPU:0"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "client_terminated"
    value {
      b: true
    }
  }
  attr {
    key: "recv_device"
    value {
      s: "/job:a/replica:0/task:0/device:CPU:0"
    }
  }
  attr {
    key: "send_device"
    value {
      s: "/job:a/replica:0/task:0/device:CPU:0"
    }
  }
  attr {
    key: "send_device_incarnation"
    value {
      i: 1
    }
  }
  attr {
    key: "tensor_name"
    value {
      s: "o0"
    }
  }
}
node {
  name: "_send_o1_1"
  op: "_Send"
  input: "Func/Swap/output/_3"
  device: "/job:a/replica:0/task:0/device:CPU:0"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "client_terminated"
    value {
      b: true
    }
  }
  attr {
    key: "recv_device"
    value {
      s: "/job:a/replica:0/task:0/device:CPU:0"
    }
  }
  attr {
    key: "send_device"
    value {
      s: "/job:a/replica:0/task:0/device:CPU:0"
    }
  }
  attr {
    key: "send_device_incarnation"
    value {
      i: 1
    }
  }
  attr {
    key: "tensor_name"
    value {
      s: "o1"
    }
  }
}
)",
                                        &expected);
  TF_EXPECT_GRAPH_EQ(expected, actual);
}

// Disabling the following two tests since there seem to be some issues with
// GRPC bringing up multiple processes as sub-processes.
// More info at: https://github.com/grpc/grpc/issues/10142.
// TODO(rohanj): Enable tests when the grpc bug is fixed.
TEST_F(ClusterFunctionLibraryRuntimeTest, DISABLED_InstantiateAndRun) {
  FunctionDefLibrary proto;
  *(proto.add_function()) = test::function::XTimesTwoInt32();
  FunctionLibraryDefinition lib_def(OpRegistry::Global(), proto);
  FunctionLibraryRuntime::InstantiateOptions instantiate_opts;
  instantiate_opts.target = "/job:localhost/replica:0/task:1/cpu:0";

  Tensor y;
  auto x = test::AsTensor<int32>({1, 2, 3, 4});
  TF_EXPECT_OK(InstantiateAndRun("XTimesTwoInt32", lib_def, {},
                                 instantiate_opts, {x}, {&y}));
  test::ExpectTensorEqual<int32>(y, test::AsTensor<int32>({2, 4, 6, 8}));
}

TEST_F(ClusterFunctionLibraryRuntimeTest,
       DISABLED_InstantiateAndRunAttrSubstitution) {
  FunctionDefLibrary proto;
  *(proto.add_function()) = test::function::Swap();
  FunctionLibraryDefinition lib_def(OpRegistry::Global(), proto);
  FunctionLibraryRuntime::InstantiateOptions instantiate_opts;
  instantiate_opts.target = "/job:localhost/replica:0/task:1/cpu:0";
  Tensor y1, y2;
  auto x1 = test::AsTensor<float>({1, 2, 3, 4});
  auto x2 = test::AsTensor<float>({4, 3, 2, 1});
  TF_EXPECT_OK(InstantiateAndRun("Swap", lib_def, {{"T", DT_FLOAT}},
                                 instantiate_opts, {x1, x2}, {&y1, &y2}));
  test::ExpectTensorEqual<float>(y1, test::AsTensor<float>({4, 3, 2, 1}));
  test::ExpectTensorEqual<float>(y2, test::AsTensor<float>({1, 2, 3, 4}));
}

}  // namespace tensorflow
