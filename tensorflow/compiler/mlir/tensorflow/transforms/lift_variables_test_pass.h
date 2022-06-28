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

#ifndef TENSORFLOW_COMPILER_MLIR_TENSORFLOW_TRANSFORMS_LIFT_VARIABLES_TEST_PASS_H_
#define TENSORFLOW_COMPILER_MLIR_TENSORFLOW_TRANSFORMS_LIFT_VARIABLES_TEST_PASS_H_
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
class MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPStransformsPSlift_variables_test_passDTh {
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
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPStransformsPSlift_variables_test_passDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPStransformsPSlift_variables_test_passDTh() {
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


#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Pass/PassManager.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/transforms/lift_variables.h"
#include "tensorflow/core/common_runtime/device_mgr.h"
#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/platform/threadpool_options.h"
#include "tensorflow/core/public/session.h"

namespace mlir {
namespace tf_saved_model {

using ::tensorflow::DeviceMgr;
using ::tensorflow::Session;
using ::tensorflow::Status;
using ::tensorflow::Tensor;

// FakeSession is for testing only.
class FakeSession : public tensorflow::Session {
 public:
  FakeSession() {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPStransformsPSlift_variables_test_passDTh mht_0(mht_0_v, 211, "", "./tensorflow/compiler/mlir/tensorflow/transforms/lift_variables_test_pass.h", "FakeSession");
}
  ~FakeSession() override = default;

  Status Create(const tensorflow::GraphDef& graph) override {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPStransformsPSlift_variables_test_passDTh mht_1(mht_1_v, 217, "", "./tensorflow/compiler/mlir/tensorflow/transforms/lift_variables_test_pass.h", "Create");

    return tensorflow::errors::Unimplemented("not available");
  }
  Status Extend(const tensorflow::GraphDef& graph) override {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPStransformsPSlift_variables_test_passDTh mht_2(mht_2_v, 223, "", "./tensorflow/compiler/mlir/tensorflow/transforms/lift_variables_test_pass.h", "Extend");

    return tensorflow::errors::Unimplemented("not available");
  }

  Status Close() override {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPStransformsPSlift_variables_test_passDTh mht_3(mht_3_v, 230, "", "./tensorflow/compiler/mlir/tensorflow/transforms/lift_variables_test_pass.h", "Close");

    return tensorflow::errors::Unimplemented("not available");
  }

  Status ListDevices(
      std::vector<tensorflow::DeviceAttributes>* response) override {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPStransformsPSlift_variables_test_passDTh mht_4(mht_4_v, 238, "", "./tensorflow/compiler/mlir/tensorflow/transforms/lift_variables_test_pass.h", "ListDevices");

    return tensorflow::errors::Unimplemented("not available");
  }

  Status LocalDeviceManager(
      const tensorflow::DeviceMgr** deviceMgrPtr) override {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPStransformsPSlift_variables_test_passDTh mht_5(mht_5_v, 246, "", "./tensorflow/compiler/mlir/tensorflow/transforms/lift_variables_test_pass.h", "LocalDeviceManager");

    // This method returns a null device manager without making an error.
    // Users of this method will be notified since it will have a fake data.
    *deviceMgrPtr = nullptr;
    return Status::OK();
  }

  Status Run(const std::vector<std::pair<std::string, Tensor>>& inputs,
             const std::vector<std::string>& output_names,
             const std::vector<std::string>& target_nodes,
             std::vector<Tensor>* outputs) override {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPStransformsPSlift_variables_test_passDTh mht_6(mht_6_v, 259, "", "./tensorflow/compiler/mlir/tensorflow/transforms/lift_variables_test_pass.h", "Run");

    tensorflow::RunMetadata run_metadata;
    return Run(tensorflow::RunOptions(), inputs, output_names, target_nodes,
               outputs, &run_metadata);
  }

  Status Run(const tensorflow::RunOptions& run_options,
             const std::vector<std::pair<std::string, Tensor>>& inputs,
             const std::vector<std::string>& output_names,
             const std::vector<std::string>& target_nodes,
             std::vector<Tensor>* outputs,
             tensorflow::RunMetadata* run_metadata) override {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPStransformsPSlift_variables_test_passDTh mht_7(mht_7_v, 273, "", "./tensorflow/compiler/mlir/tensorflow/transforms/lift_variables_test_pass.h", "Run");

    return Run(run_options, inputs, output_names, target_nodes, outputs,
               run_metadata, tensorflow::thread::ThreadPoolOptions());
  }

  Status Run(const tensorflow::RunOptions& run_options,
             const std::vector<std::pair<std::string, Tensor>>& inputs,
             const std::vector<std::string>& output_names,
             const std::vector<std::string>& target_nodes,
             std::vector<Tensor>* outputs,
             tensorflow::RunMetadata* run_metadata,
             const tensorflow::thread::ThreadPoolOptions& thread_pool_options)
      override {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPStransformsPSlift_variables_test_passDTh mht_8(mht_8_v, 288, "", "./tensorflow/compiler/mlir/tensorflow/transforms/lift_variables_test_pass.h", "Run");

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
      } else {
        // Create a scalar float tensor.
        Tensor t = Tensor(tensorflow::DT_FLOAT, tensorflow::TensorShape({}));
        t.flat<float>()(0) = 1.0f;
        outputs->push_back(t);
      }
    }
    return Status::OK();
  }
};

// This pass is only available in the tf-opt binary for testing.
class LiftVariablesTestPass
    : public PassWrapper<LiftVariablesTestPass, OperationPass<ModuleOp>> {
 public:
  LiftVariablesTestPass() {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPStransformsPSlift_variables_test_passDTh mht_9(mht_9_v, 318, "", "./tensorflow/compiler/mlir/tensorflow/transforms/lift_variables_test_pass.h", "LiftVariablesTestPass");
 session_ = new FakeSession(); }

  ~LiftVariablesTestPass() override {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPStransformsPSlift_variables_test_passDTh mht_10(mht_10_v, 323, "", "./tensorflow/compiler/mlir/tensorflow/transforms/lift_variables_test_pass.h", "~LiftVariablesTestPass");
 delete session_; }

  void runOnOperation() override {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPStransformsPSlift_variables_test_passDTh mht_11(mht_11_v, 328, "", "./tensorflow/compiler/mlir/tensorflow/transforms/lift_variables_test_pass.h", "runOnOperation");

    ModuleOp module = getOperation();
    if (failed(LiftVariables(module, session_))) signalPassFailure();
  }

 private:
  Session* session_;
};

// This pass is only available in the tf-opt binary for testing.
class LiftVariablesInvalidSessionTestPass
    : public PassWrapper<LiftVariablesInvalidSessionTestPass,
                         OperationPass<ModuleOp>> {
 public:
  void runOnOperation() override {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPStransformsPSlift_variables_test_passDTh mht_12(mht_12_v, 345, "", "./tensorflow/compiler/mlir/tensorflow/transforms/lift_variables_test_pass.h", "runOnOperation");

    ModuleOp module = getOperation();
    // Pass an invalid session argument, which is a nullptr.
    if (failed(LiftVariables(module, /*session=*/nullptr))) signalPassFailure();
  }
};

}  // namespace tf_saved_model
}  // namespace mlir

#endif  // TENSORFLOW_COMPILER_MLIR_TENSORFLOW_TRANSFORMS_LIFT_VARIABLES_TEST_PASS_H_
