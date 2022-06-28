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
class MHTracer_DTPStensorflowPScorePScommon_runtimePSkernel_benchmark_testlibDTcc {
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
   MHTracer_DTPStensorflowPScorePScommon_runtimePSkernel_benchmark_testlibDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePScommon_runtimePSkernel_benchmark_testlibDTcc() {
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

/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/common_runtime/kernel_benchmark_testlib.h"

#include <vector>

#include "tensorflow/core/common_runtime/device.h"
#include "tensorflow/core/common_runtime/device_factory.h"
#include "tensorflow/core/common_runtime/device_mgr.h"
#include "tensorflow/core/common_runtime/executor_factory.h"
#include "tensorflow/core/common_runtime/function.h"
#include "tensorflow/core/common_runtime/local_device.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/op_segment.h"
#include "tensorflow/core/framework/versions.pb.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/kernels/ops_util.h"
#include "tensorflow/core/lib/core/notification.h"
#include "tensorflow/core/lib/core/threadpool.h"
#include "tensorflow/core/lib/gtl/cleanup.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/platform/byte_order.h"
#include "tensorflow/core/platform/cpu_info.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/public/session_options.h"
#include "tensorflow/core/public/version.h"
#include "tensorflow/core/util/device_name_utils.h"

namespace tensorflow {
namespace test {

// TODO(hongm): Convert `g` and `init` to using std::unique_ptr.
Benchmark::Benchmark(const string& device, Graph* g,
                     const SessionOptions* options, Graph* init,
                     Rendezvous* rendez, const char* executor_type,
                     bool old_benchmark_api) {
   std::vector<std::string> mht_0_v;
   mht_0_v.push_back("device: \"" + device + "\"");
   mht_0_v.push_back("executor_type: \"" + (executor_type == nullptr ? std::string("nullptr") : std::string((char*)executor_type)) + "\"");
   MHTracer_DTPStensorflowPScorePScommon_runtimePSkernel_benchmark_testlibDTcc mht_0(mht_0_v, 222, "", "./tensorflow/core/common_runtime/kernel_benchmark_testlib.cc", "Benchmark::Benchmark");

  auto cleanup = gtl::MakeCleanup([g, init]() {
    delete g;
    delete init;
  });

  SessionOptions default_options;
  if (!options) {
    options = &default_options;
  }

  CHECK(!old_benchmark_api) << "Expected new API only";

  string t = absl::AsciiStrToUpper(device);
  // Allow NewDevice to allocate a new threadpool with different number of
  // threads for each new benchmark.
  LocalDevice::set_use_global_threadpool(false);

  device_mgr_ = absl::make_unique<StaticDeviceMgr>(
      DeviceFactory::NewDevice(t, *options, "/job:localhost/replica:0/task:0"));
  device_ = device_mgr_->ListDevices()[0];
  CHECK(device_) << "Could not create a " << device << " device";

  pool_ =
      new thread::ThreadPool(options->env, "blocking", port::MaxParallelism());

  auto runner = [this](std::function<void()> closure) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSkernel_benchmark_testlibDTcc mht_1(mht_1_v, 251, "", "./tensorflow/core/common_runtime/kernel_benchmark_testlib.cc", "lambda");

    pool_->Schedule(closure);
  };

  if (rendez == nullptr) {
    rendez_ = NewLocalRendezvous();
  } else {
    rendez_ = rendez;
  }

  const int graph_def_version = g->versions().producer();

  flib_def_ = absl::make_unique<FunctionLibraryDefinition>(g->flib_def());

  pflr_ = std::unique_ptr<ProcessFunctionLibraryRuntime>(
      new ProcessFunctionLibraryRuntime(
          device_mgr_.get(), Env::Default(), nullptr, graph_def_version,
          flib_def_.get(), OptimizerOptions(), pool_, nullptr, nullptr,
          Rendezvous::Factory()));

  flr_ = pflr_->GetFLR(device_->name());

  LocalExecutorParams params;
  params.device = device_;
  params.function_library = flr_;
  params.create_kernel = [this, graph_def_version](
                             const std::shared_ptr<const NodeProperties>& props,
                             OpKernel** kernel) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSkernel_benchmark_testlibDTcc mht_2(mht_2_v, 281, "", "./tensorflow/core/common_runtime/kernel_benchmark_testlib.cc", "lambda");

    return CreateNonCachedKernel(device_, flr_, props, graph_def_version,
                                 kernel);
  };
  params.delete_kernel = [](OpKernel* kernel) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSkernel_benchmark_testlibDTcc mht_3(mht_3_v, 288, "", "./tensorflow/core/common_runtime/kernel_benchmark_testlib.cc", "lambda");

    DeleteNonCachedKernel(kernel);
  };

  if (init) {
    std::unique_ptr<Executor> init_exec;
    TF_CHECK_OK(NewExecutor(executor_type, params, *init, &init_exec));
    Executor::Args args;
    args.rendezvous = rendez_;
    args.runner = runner;
    TF_CHECK_OK(init_exec->Run(args));
  }

  TF_CHECK_OK(NewExecutor(executor_type, params, *g, &exec_));
}

Benchmark::Benchmark(const string& device, Graph* g, bool old_benchmark_api)
    : Benchmark(device, g, nullptr, nullptr, nullptr, "", old_benchmark_api) {
   std::vector<std::string> mht_4_v;
   mht_4_v.push_back("device: \"" + device + "\"");
   MHTracer_DTPStensorflowPScorePScommon_runtimePSkernel_benchmark_testlibDTcc mht_4(mht_4_v, 309, "", "./tensorflow/core/common_runtime/kernel_benchmark_testlib.cc", "Benchmark::Benchmark");
}

Benchmark::~Benchmark() {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSkernel_benchmark_testlibDTcc mht_5(mht_5_v, 314, "", "./tensorflow/core/common_runtime/kernel_benchmark_testlib.cc", "Benchmark::~Benchmark");

  if (device_) {
    rendez_->Unref();
    // We delete `exec_` before `device_mgr_` because the `exec_` destructor may
    // run kernel destructors that may attempt to access state borrowed from
    // `device_mgr_`, such as the resource manager.
    exec_.reset();
    pflr_.reset();
    device_mgr_.reset();
    delete pool_;
  }
}

void Benchmark::Run(benchmark::State& state) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSkernel_benchmark_testlibDTcc mht_6(mht_6_v, 330, "", "./tensorflow/core/common_runtime/kernel_benchmark_testlib.cc", "Benchmark::Run");

  RunWithRendezvousArgs({}, {}, state);
}

string GetRendezvousKey(const Node* node) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSkernel_benchmark_testlibDTcc mht_7(mht_7_v, 337, "", "./tensorflow/core/common_runtime/kernel_benchmark_testlib.cc", "GetRendezvousKey");

  string send_device;
  TF_CHECK_OK(GetNodeAttr(node->attrs(), "send_device", &send_device));
  string recv_device;
  TF_CHECK_OK(GetNodeAttr(node->attrs(), "recv_device", &recv_device));
  string tensor_name;
  TF_CHECK_OK(GetNodeAttr(node->attrs(), "tensor_name", &tensor_name));
  uint64 send_device_incarnation;
  TF_CHECK_OK(
      GetNodeAttr(node->attrs(), "send_device_incarnation",
                  reinterpret_cast<int64_t*>(&send_device_incarnation)));
  return Rendezvous::CreateKey(send_device, send_device_incarnation,
                               recv_device, tensor_name, FrameAndIter(0, 0));
}

void Benchmark::RunWithRendezvousArgs(
    const std::vector<std::pair<string, Tensor>>& inputs,
    const std::vector<string>& outputs, benchmark::State& state) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSkernel_benchmark_testlibDTcc mht_8(mht_8_v, 357, "", "./tensorflow/core/common_runtime/kernel_benchmark_testlib.cc", "Benchmark::RunWithRendezvousArgs");

  if (!device_ || state.max_iterations == 0) {
    return;
  }
  Tensor unused;  // In benchmark, we don't care the return value.
  bool is_dead;

  // Warm up
  Executor::Args args;
  args.rendezvous = rendez_;
  args.runner = [this](std::function<void()> closure) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSkernel_benchmark_testlibDTcc mht_9(mht_9_v, 370, "", "./tensorflow/core/common_runtime/kernel_benchmark_testlib.cc", "lambda");

    pool_->Schedule(closure);
  };
  static const int kWarmupRuns = 3;
  for (int i = 0; i < kWarmupRuns; ++i) {
    for (const auto& p : inputs) {
      Rendezvous::ParsedKey parsed;
      TF_CHECK_OK(Rendezvous::ParseKey(p.first, &parsed));
      TF_CHECK_OK(rendez_->Send(parsed, Rendezvous::Args(), p.second, false));
    }
    TF_CHECK_OK(exec_->Run(args));
    for (const string& key : outputs) {
      Rendezvous::ParsedKey parsed;
      TF_CHECK_OK(Rendezvous::ParseKey(key, &parsed));
      TF_CHECK_OK(rendez_->Recv(parsed, Rendezvous::Args(), &unused, &is_dead));
    }
  }
  TF_CHECK_OK(device_->Sync());
  VLOG(3) << kWarmupRuns << " warmup runs done.";

  // Benchmark loop. Timer starts automatically at the beginning of the loop
  // and ends automatically after the last iteration.
  for (auto s : state) {
    for (const auto& p : inputs) {
      Rendezvous::ParsedKey parsed;
      TF_CHECK_OK(Rendezvous::ParseKey(p.first, &parsed));
      TF_CHECK_OK(rendez_->Send(parsed, Rendezvous::Args(), p.second, false));
    }
    TF_CHECK_OK(exec_->Run(args));
    for (const string& key : outputs) {
      Rendezvous::ParsedKey parsed;
      TF_CHECK_OK(Rendezvous::ParseKey(key, &parsed));
      TF_CHECK_OK(rendez_->Recv(parsed, Rendezvous::Args(), &unused, &is_dead));
    }
  }
  TF_CHECK_OK(device_->Sync());
}

}  // end namespace test
}  // end namespace tensorflow
