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
class MHTracer_DTPStensorflowPScorePScommon_runtimePSeagerPSkernel_and_device_testDTcc {
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
   MHTracer_DTPStensorflowPScorePScommon_runtimePSeagerPSkernel_and_device_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePScommon_runtimePSeagerPSkernel_and_device_testDTcc() {
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

#include "tensorflow/core/common_runtime/eager/kernel_and_device.h"

#include <memory>
#include <vector>

#include "absl/memory/memory.h"
#include "absl/types/optional.h"
#include "tensorflow/cc/client/client_session.h"
#include "tensorflow/cc/framework/ops.h"
#include "tensorflow/cc/framework/scope.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/common_runtime/device_factory.h"
#include "tensorflow/core/common_runtime/device_mgr.h"
#include "tensorflow/core/common_runtime/eager/attr_builder.h"
#include "tensorflow/core/common_runtime/function.h"
#include "tensorflow/core/common_runtime/process_function_library_runtime.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/platform/test_benchmark.h"
#include "tensorflow/core/protobuf/config.pb.h"
#include "tensorflow/core/public/version.h"
#include "tensorflow/core/util/ptr_util.h"

namespace tensorflow {
namespace {

class TestEnv {
 public:
  TestEnv() : flib_def_(OpRegistry::Global(), {}) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSeagerPSkernel_and_device_testDTcc mht_0(mht_0_v, 213, "", "./tensorflow/core/common_runtime/eager/kernel_and_device_test.cc", "TestEnv");

    std::vector<std::unique_ptr<Device>> devices;
    devices.push_back(
        DeviceFactory::NewDevice("CPU", {}, "/job:a/replica:0/task:0"));
    cpu_device_ = devices.back().get();
    device_mgr_ = absl::make_unique<StaticDeviceMgr>(std::move(devices));
    OptimizerOptions opts;
    pflr_ = tensorflow::MakeUnique<ProcessFunctionLibraryRuntime>(
        device_mgr_.get(), Env::Default(), /*config=*/nullptr,
        TF_GRAPH_DEF_VERSION, &flib_def_, opts,
        /*default_thread_pool=*/nullptr);

    flr_ = pflr_->GetFLR("/job:a/replica:0/task:0/device:CPU:0");
    CHECK(flr_ != nullptr);
  }

  FunctionLibraryRuntime* function_library_runtime() const {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSeagerPSkernel_and_device_testDTcc mht_1(mht_1_v, 232, "", "./tensorflow/core/common_runtime/eager/kernel_and_device_test.cc", "function_library_runtime");
 return flr_; }
  ProcessFunctionLibraryRuntime* pflr() const {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSeagerPSkernel_and_device_testDTcc mht_2(mht_2_v, 236, "", "./tensorflow/core/common_runtime/eager/kernel_and_device_test.cc", "pflr");
 return pflr_.get(); }
  Device* cpu_device() {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSeagerPSkernel_and_device_testDTcc mht_3(mht_3_v, 240, "", "./tensorflow/core/common_runtime/eager/kernel_and_device_test.cc", "cpu_device");
 return cpu_device_; }

 private:
  FunctionLibraryDefinition flib_def_;
  std::unique_ptr<DeviceMgr> device_mgr_;
  FunctionLibraryRuntime* flr_;
  std::unique_ptr<ProcessFunctionLibraryRuntime> pflr_;
  Device* cpu_device_;
};

void BM_CreateGraph(::testing::benchmark::State& state) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSeagerPSkernel_and_device_testDTcc mht_4(mht_4_v, 253, "", "./tensorflow/core/common_runtime/eager/kernel_and_device_test.cc", "BM_CreateGraph");

  for (auto s : state) {
    Scope root = Scope::NewRootScope();
    auto C = ops::Const(root, {{1.0, 2.0}, {3.0, 4.0}});
    auto M = ops::MatMul(root, C, C);
    TF_CHECK_OK(root.status());
  }
}
BENCHMARK(BM_CreateGraph);

void BM_RunGraph(::testing::benchmark::State& state) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSeagerPSkernel_and_device_testDTcc mht_5(mht_5_v, 266, "", "./tensorflow/core/common_runtime/eager/kernel_and_device_test.cc", "BM_RunGraph");

  Scope root = Scope::NewRootScope();
  auto C = ops::Const(root, {{1.0, 2.0}, {3.0, 4.0}});
  auto M = ops::MatMul(root, C, C);
  SessionOptions opts;
  opts.config.set_inter_op_parallelism_threads(1);
  opts.config.set_intra_op_parallelism_threads(1);
  ClientSession sess(root, opts);
  std::vector<Tensor> outputs;
  for (auto s : state) {
    outputs.clear();
    TF_CHECK_OK(sess.Run({M}, &outputs));
  }
}
BENCHMARK(BM_RunGraph);

void BM_CreateAndDestroySession(::testing::benchmark::State& state) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSeagerPSkernel_and_device_testDTcc mht_6(mht_6_v, 285, "", "./tensorflow/core/common_runtime/eager/kernel_and_device_test.cc", "BM_CreateAndDestroySession");

  Scope root = Scope::NewRootScope();
  auto C = ops::Const(root, {{1.0, 2.0}, {3.0, 4.0}});
  auto M = ops::MatMul(root, C, C);
  for (auto s : state) {
    ClientSession sess(root);
  }
}
BENCHMARK(BM_CreateAndDestroySession);

void BM_KernelAndDeviceInit(::testing::benchmark::State& state) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSeagerPSkernel_and_device_testDTcc mht_7(mht_7_v, 298, "", "./tensorflow/core/common_runtime/eager/kernel_and_device_test.cc", "BM_KernelAndDeviceInit");

  NodeDef ndef(AttrBuilder("MatMul")
                   .Set("T", DT_FLOAT)
                   .Set("transpose_a", false)
                   .Set("transpose_b", false)
                   .NumInputs(2)
                   .BuildNodeDef());
  TestEnv env;
  KernelAndDeviceOp k(nullptr, false, env.function_library_runtime(), nullptr,
                      nullptr, env.cpu_device());
  for (auto s : state) {
    TF_CHECK_OK(k.Init({}, ndef, nullptr));
  }
}
BENCHMARK(BM_KernelAndDeviceInit);

void BM_KernelAndDeviceRun(::testing::benchmark::State& state) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSeagerPSkernel_and_device_testDTcc mht_8(mht_8_v, 317, "", "./tensorflow/core/common_runtime/eager/kernel_and_device_test.cc", "BM_KernelAndDeviceRun");

  Tensor t(Input({{1.0f, 2.0f}, {3.0f, 4.0f}}).tensor());
  gtl::InlinedVector<TensorValue, 4> inputs;
  inputs.push_back(TensorValue(&t));
  inputs.push_back(TensorValue(&t));
  std::vector<EagerKernelRet> outputs;
  NodeDef ndef(AttrBuilder("MatMul")
                   .Set("T", DT_FLOAT)
                   .Set("transpose_a", false)
                   .Set("transpose_b", false)
                   .NumInputs(inputs.size())
                   .BuildNodeDef());
  TestEnv env;
  KernelAndDeviceOp k(nullptr, false, env.function_library_runtime(), nullptr,
                      nullptr, env.cpu_device());
  TF_CHECK_OK(k.Init({}, ndef, nullptr));
  const EagerKernelArgs args(std::move(inputs));
  for (auto s : state) {
    TF_CHECK_OK(k.Run(nullptr, args, &outputs, nullptr, absl::nullopt,
                      absl::nullopt, nullptr));
  }
}
BENCHMARK(BM_KernelAndDeviceRun);
}  // namespace
}  // namespace tensorflow
