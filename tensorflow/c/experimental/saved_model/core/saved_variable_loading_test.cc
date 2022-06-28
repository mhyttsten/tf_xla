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
class MHTracer_DTPStensorflowPScPSexperimentalPSsaved_modelPScorePSsaved_variable_loading_testDTcc {
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
   MHTracer_DTPStensorflowPScPSexperimentalPSsaved_modelPScorePSsaved_variable_loading_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScPSexperimentalPSsaved_modelPScorePSsaved_variable_loading_testDTcc() {
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

#include <memory>
#include <vector>

#include "tensorflow/c/eager/immediate_execution_tensor_handle.h"
#include "tensorflow/c/experimental/saved_model/core/revived_types/constant.h"
#include "tensorflow/c/experimental/saved_model/core/saved_model_utils.h"
#include "tensorflow/c/experimental/saved_model/core/test_utils.h"
#include "tensorflow/c/tensor_interface.h"
#include "tensorflow/core/common_runtime/device_mgr.h"
#include "tensorflow/core/common_runtime/eager/context.h"
#include "tensorflow/core/common_runtime/eager/tensor_handle.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor.pb.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/protobuf/saved_object_graph.pb.h"

namespace tensorflow {
namespace {

class SavedVariableLoadingTest
    : public ::testing::TestWithParam<
          std::tuple<DataType, std::vector<int64_t>>> {
 public:
  SavedVariableLoadingTest() {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScPSexperimentalPSsaved_modelPScorePSsaved_variable_loading_testDTcc mht_0(mht_0_v, 212, "", "./tensorflow/c/experimental/saved_model/core/saved_variable_loading_test.cc", "SavedVariableLoadingTest");

    SessionOptions options;
    options.config.mutable_device_count()->insert({"CPU", 3});
    std::vector<std::unique_ptr<Device>> devices;
    TF_CHECK_OK(DeviceFactory::AddDevices(
        options, "/job:localhost/replica:0/task:0", &devices));
    device_mgr_ = absl::make_unique<StaticDeviceMgr>(std::move(devices));
    ctx_ = testing::CreateTestingEagerContext(device_mgr_.get());
  }

  EagerContext* context() {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScPSexperimentalPSsaved_modelPScorePSsaved_variable_loading_testDTcc mht_1(mht_1_v, 225, "", "./tensorflow/c/experimental/saved_model/core/saved_variable_loading_test.cc", "context");
 return ctx_.get(); }

 private:
  std::unique_ptr<StaticDeviceMgr> device_mgr_;
  EagerContextPtr ctx_;
};

// Sanity check that constructing a tensorflow::Variable from a SavedVariable
// 1. does not cause an error
// 2. preserves dtype and shape.
TEST_P(SavedVariableLoadingTest, LoadSavedVariableSuccessful) {
  auto& test_params = GetParam();
  DataType dtype = std::get<0>(test_params);
  TensorShape shape(std::get<1>(test_params));

  SavedVariable saved_variable;
  saved_variable.set_dtype(dtype);
  shape.AsProto(saved_variable.mutable_shape());

  std::unique_ptr<Variable> var;
  TF_EXPECT_OK(internal::LoadSavedVariable(context(), saved_variable, &var));
  EXPECT_EQ(var->dtype(), dtype);
  EXPECT_EQ(var->shape(), shape);
}

// Verify that a device specified in the SavedVariable is kept.
TEST_P(SavedVariableLoadingTest, LoadSavedVariableWithDevice) {
  auto& test_params = GetParam();
  DataType dtype = std::get<0>(test_params);
  TensorShape shape(std::get<1>(test_params));

  SavedVariable saved_variable;
  saved_variable.set_dtype(dtype);
  saved_variable.set_device("/job:localhost/replica:0/task:0/device:CPU:1"),
      shape.AsProto(saved_variable.mutable_shape());

  std::unique_ptr<Variable> var;
  TF_ASSERT_OK(internal::LoadSavedVariable(context(), saved_variable, &var));
  EXPECT_EQ(down_cast<TensorHandle*>(var->handle())->resource_device()->name(),
            "/job:localhost/replica:0/task:0/device:CPU:1");
}

// Verify load failure if a non-existing device is specified.
TEST_P(SavedVariableLoadingTest, LoadSavedVariableWithInvalidDevice) {
  auto& test_params = GetParam();
  DataType dtype = std::get<0>(test_params);
  TensorShape shape(std::get<1>(test_params));

  SavedVariable saved_variable;
  saved_variable.set_dtype(dtype);
  saved_variable.set_device("/job:localhost/replica:0/task:0/device:CPU:99"),
      shape.AsProto(saved_variable.mutable_shape());

  std::unique_ptr<Variable> var;
  ASSERT_NE(Status::OK(),
            internal::LoadSavedVariable(context(), saved_variable, &var));
}

// Assigning and reading values should yield
// consistent results.
TEST_P(SavedVariableLoadingTest, AssignAndReadVariableSuccesful) {
  auto& test_params = GetParam();
  DataType dtype = std::get<0>(test_params);
  std::vector<int64_t> shape_vector = std::get<1>(test_params);
  TensorShape shape(shape_vector);

  // Create the variable.
  Status status;
  std::unique_ptr<Variable> var;
  TF_EXPECT_OK(Variable::CreateUninitialized(context(), dtype, shape,
                                             absl::nullopt, nullptr, {}, &var));

  // Create a TensorHandle
  ImmediateTensorHandlePtr expected_handle =
      testing::CreateTensorHandle(context(), dtype, shape_vector, 42);
  AbstractTensorPtr expected_tensor(expected_handle->Resolve(&status));
  TF_EXPECT_OK(status) << status.error_message();

  // Assign the tensorhandle to the variable.
  TF_EXPECT_OK(var->Assign(expected_handle.get()));

  // Read back the value from the variable
  ImmediateTensorHandlePtr output_handle;
  TF_EXPECT_OK(var->ReadValue(&output_handle));
  AbstractTensorPtr output_tensor(output_handle->Resolve(&status));
  TF_EXPECT_OK(status) << status.error_message();

  // Check that output_tensor == expected_tensor
  EXPECT_EQ(output_tensor->Type(), expected_tensor->Type());
  EXPECT_EQ(output_tensor->NumElements(), expected_tensor->NumElements());
  testing::CheckBufferDataIsEqual(
      output_tensor->Type(), output_tensor->NumElements(),
      output_tensor->Data(), expected_tensor->Data());
}

// Test against combinations of SavedVariables of
// 1. Varying dtypes
// 2. Varying shapes
INSTANTIATE_TEST_SUITE_P(
    SavedVariableIntegerDtypesTest, SavedVariableLoadingTest,
    ::testing::Combine(
        ::testing::ValuesIn(testing::DataTypeSetToVector(kDataTypeIsInteger)),
        ::testing::ValuesIn(testing::InterestingShapes())));

INSTANTIATE_TEST_SUITE_P(
    SavedVariableFloatingDtypesTest, SavedVariableLoadingTest,
    ::testing::Combine(::testing::Values(DT_FLOAT, DT_DOUBLE),
                       ::testing::ValuesIn(testing::InterestingShapes())));

}  // namespace
}  // namespace tensorflow
