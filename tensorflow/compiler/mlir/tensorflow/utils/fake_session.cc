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
class MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPSutilsPSfake_sessionDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPSutilsPSfake_sessionDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPSutilsPSfake_sessionDTcc() {
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
#include "tensorflow/compiler/mlir/tensorflow/utils/fake_session.h"

#include "absl/strings/match.h"
#include "llvm/Support/CommandLine.h"
#include "tensorflow/core/common_runtime/device_mgr.h"
#include "tensorflow/core/common_runtime/threadpool_device.h"
#include "tensorflow/core/framework/allocator.h"
#include "tensorflow/core/framework/device_attributes.pb.h"
#include "tensorflow/core/framework/device_factory.h"
#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/framework/resource_var.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/graph/types.h"
#include "tensorflow/core/platform/threadpool_options.h"
#include "tensorflow/core/protobuf/error_codes.pb.h"
#include "tensorflow/core/public/session_options.h"

namespace mlir {
namespace TF {
namespace test_util {
namespace {
using ::tensorflow::Status;
using ::tensorflow::Tensor;

const char kDeviceNamePrefix[] = "/job:worker/replica:0/task:1";
const char kDeviceName[] = "/job:worker/replica:0/task:1/device:CPU:0";

// Struct holding options for FakeSession which are configuered through
// command line flags.
struct FakeSessionOptions {
  llvm::cl::opt<bool> fail_to_fetch_local_device_manager{
      "fail-to-fetch-local-device-manager",
      llvm::cl::desc("Fail to fetch local device manager."),
      llvm::cl::init(false)};
};
FakeSessionOptions* kSessionOptions = []() {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPSutilsPSfake_sessionDTcc mht_0(mht_0_v, 219, "", "./tensorflow/compiler/mlir/tensorflow/utils/fake_session.cc", "lambda");
 return new FakeSessionOptions; }();
}  // namespace

FakeSession::FakeSession() {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPSutilsPSfake_sessionDTcc mht_1(mht_1_v, 225, "", "./tensorflow/compiler/mlir/tensorflow/utils/fake_session.cc", "FakeSession::FakeSession");

  // We don't initialize devices in constructor as it causes some
  // global initialization fiasco between tests and code in TF.
}

void FakeSession::Initialize() {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPSutilsPSfake_sessionDTcc mht_2(mht_2_v, 233, "", "./tensorflow/compiler/mlir/tensorflow/utils/fake_session.cc", "FakeSession::Initialize");

  if (initialized_) return;
  BuildDeviceManager();
  InitVariables();
  initialized_ = true;
}

void FakeSession::BuildDeviceManager() {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPSutilsPSfake_sessionDTcc mht_3(mht_3_v, 243, "", "./tensorflow/compiler/mlir/tensorflow/utils/fake_session.cc", "FakeSession::BuildDeviceManager");

  auto device =
      tensorflow::DeviceFactory::NewDevice("CPU", {}, kDeviceNamePrefix);
  device_mgr_ =
      absl::make_unique<tensorflow::StaticDeviceMgr>(std::move(device));
}

void FakeSession::InitVariables() {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPSutilsPSfake_sessionDTcc mht_4(mht_4_v, 253, "", "./tensorflow/compiler/mlir/tensorflow/utils/fake_session.cc", "FakeSession::InitVariables");

  tensorflow::Device* device = nullptr;
  auto status = device_mgr_->LookupDevice(kDeviceName, &device);
  if (status != Status::OK()) return;
  auto container = device->resource_manager()->default_container();

  // Create 2 resources and initialize them with dummy values.
  (void)device->resource_manager()->Create(
      container, "var1", new tensorflow::Var(tensorflow::DataType::DT_FLOAT));
  (void)device->resource_manager()->Create(
      container, "var2", new tensorflow::Var(tensorflow::DataType::DT_FLOAT));
}

Status FakeSession::Create(const tensorflow::GraphDef& graph) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPSutilsPSfake_sessionDTcc mht_5(mht_5_v, 269, "", "./tensorflow/compiler/mlir/tensorflow/utils/fake_session.cc", "FakeSession::Create");

  return tensorflow::errors::Unimplemented("not available");
}
Status FakeSession::Extend(const tensorflow::GraphDef& graph) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPSutilsPSfake_sessionDTcc mht_6(mht_6_v, 275, "", "./tensorflow/compiler/mlir/tensorflow/utils/fake_session.cc", "FakeSession::Extend");

  return tensorflow::errors::Unimplemented("not available");
}

Status FakeSession::Close() {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPSutilsPSfake_sessionDTcc mht_7(mht_7_v, 282, "", "./tensorflow/compiler/mlir/tensorflow/utils/fake_session.cc", "FakeSession::Close");

  return tensorflow::errors::Unimplemented("not available");
}

Status FakeSession::ListDevices(
    std::vector<tensorflow::DeviceAttributes>* response) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPSutilsPSfake_sessionDTcc mht_8(mht_8_v, 290, "", "./tensorflow/compiler/mlir/tensorflow/utils/fake_session.cc", "FakeSession::ListDevices");

  return tensorflow::errors::Unimplemented("not available");
}

Status FakeSession::LocalDeviceManager(
    const tensorflow::DeviceMgr** deviceMgrPtr) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPSutilsPSfake_sessionDTcc mht_9(mht_9_v, 298, "", "./tensorflow/compiler/mlir/tensorflow/utils/fake_session.cc", "FakeSession::LocalDeviceManager");

  Initialize();
  if (kSessionOptions->fail_to_fetch_local_device_manager)
    return Status(tensorflow::error::UNKNOWN, "No Local Device Manager");
  *deviceMgrPtr = device_mgr_.get();
  return Status::OK();
}

Status FakeSession::Run(
    const std::vector<std::pair<std::string, Tensor>>& inputs,
    const std::vector<std::string>& output_names,
    const std::vector<std::string>& target_nodes,
    std::vector<Tensor>* outputs) {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPSutilsPSfake_sessionDTcc mht_10(mht_10_v, 313, "", "./tensorflow/compiler/mlir/tensorflow/utils/fake_session.cc", "FakeSession::Run");

  tensorflow::RunMetadata run_metadata;
  return Run(tensorflow::RunOptions(), inputs, output_names, target_nodes,
             outputs, &run_metadata);
}

Status FakeSession::Run(
    const tensorflow::RunOptions& run_options,
    const std::vector<std::pair<std::string, Tensor>>& inputs,
    const std::vector<std::string>& output_names,
    const std::vector<std::string>& target_nodes, std::vector<Tensor>* outputs,
    tensorflow::RunMetadata* run_metadata) {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPSutilsPSfake_sessionDTcc mht_11(mht_11_v, 327, "", "./tensorflow/compiler/mlir/tensorflow/utils/fake_session.cc", "FakeSession::Run");

  return Run(run_options, inputs, output_names, target_nodes, outputs,
             run_metadata, tensorflow::thread::ThreadPoolOptions());
}

Status FakeSession::Run(
    const tensorflow::RunOptions& run_options,
    const std::vector<std::pair<std::string, Tensor>>& inputs,
    const std::vector<std::string>& output_names,
    const std::vector<std::string>& target_nodes, std::vector<Tensor>* outputs,
    tensorflow::RunMetadata* run_metadata,
    const tensorflow::thread::ThreadPoolOptions& thread_pool_options) {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPSutilsPSfake_sessionDTcc mht_12(mht_12_v, 341, "", "./tensorflow/compiler/mlir/tensorflow/utils/fake_session.cc", "FakeSession::Run");

  Initialize();
  for (const std::string& output_name : output_names) {
    Tensor output;
    if (output_name == "dense/bias") {
      Tensor t = Tensor(tensorflow::DT_FLOAT, tensorflow::TensorShape({50}));
      t.flat<float>().setZero();
      outputs->push_back(t);
    } else if (output_name == "dense/kernel") {
      Tensor t =
          Tensor(tensorflow::DT_FLOAT, tensorflow::TensorShape({100, 50}));
      t.flat<float>().setZero();
      outputs->push_back(t);
    } else if (output_name == "var1") {
      Tensor t = Tensor(tensorflow::DT_RESOURCE, tensorflow::TensorShape({1}));
      t.scalar<tensorflow::ResourceHandle>()().set_name("var1");
      t.scalar<tensorflow::ResourceHandle>()().set_device(kDeviceName);

      outputs->push_back(t);
    } else if (output_name == "var2") {
      Tensor t = Tensor(tensorflow::DT_RESOURCE, tensorflow::TensorShape({1}));
      t.scalar<tensorflow::ResourceHandle>()().set_name("var2");
      t.scalar<tensorflow::ResourceHandle>()().set_device(kDeviceName);

      outputs->push_back(t);
    } else if (output_name == "var3") {
      Tensor t = Tensor(tensorflow::DT_RESOURCE, tensorflow::TensorShape({1}));
      t.scalar<tensorflow::ResourceHandle>()().set_name("var3");
      t.scalar<tensorflow::ResourceHandle>()().set_device(kDeviceName);

      outputs->push_back(t);
    } else if (output_name == "invalid_var") {
      Tensor t = Tensor(tensorflow::DT_RESOURCE, tensorflow::TensorShape({1}));
      t.scalar<tensorflow::ResourceHandle>()().set_name("invalid_var");
      t.scalar<tensorflow::ResourceHandle>()().set_device("invalid_device");

      outputs->push_back(t);
    } else if (absl::StartsWith(output_name, "var")) {
      return Status(tensorflow::error::NOT_FOUND,
                    "Can't find variable " + output_name + " in session");
    } else {
      // Create a scalar float tensor.
      Tensor t = Tensor(tensorflow::DT_FLOAT, tensorflow::TensorShape({}));
      t.flat<float>()(0) = 1.0f;
      outputs->push_back(t);
    }
  }
  return Status::OK();
}

}  // namespace test_util
}  // namespace TF
}  // namespace mlir
