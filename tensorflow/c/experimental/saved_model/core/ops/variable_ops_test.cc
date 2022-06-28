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
class MHTracer_DTPStensorflowPScPSexperimentalPSsaved_modelPScorePSopsPSvariable_ops_testDTcc {
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
   MHTracer_DTPStensorflowPScPSexperimentalPSsaved_modelPScorePSopsPSvariable_ops_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScPSexperimentalPSsaved_modelPScorePSopsPSvariable_ops_testDTcc() {
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

#include "tensorflow/c/experimental/saved_model/core/ops/variable_ops.h"

#include <memory>

#include "tensorflow/c/eager/immediate_execution_tensor_handle.h"
#include "tensorflow/c/experimental/saved_model/core/test_utils.h"
#include "tensorflow/c/tensor_interface.h"
#include "tensorflow/core/common_runtime/device_mgr.h"
#include "tensorflow/core/common_runtime/eager/context.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {
namespace {

ImmediateTensorHandlePtr CreateScalarTensorHandle(EagerContext* context,
                                                  float value) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScPSexperimentalPSsaved_modelPScorePSopsPSvariable_ops_testDTcc mht_0(mht_0_v, 203, "", "./tensorflow/c/experimental/saved_model/core/ops/variable_ops_test.cc", "CreateScalarTensorHandle");

  AbstractTensorPtr tensor(context->CreateFloatScalar(value));
  ImmediateTensorHandlePtr handle(context->CreateLocalHandle(tensor.get()));
  return handle;
}

class VariableOpsTest : public ::testing::Test {
 public:
  VariableOpsTest()
      : device_mgr_(testing::CreateTestingDeviceMgr()),
        ctx_(testing::CreateTestingEagerContext(device_mgr_.get())) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScPSexperimentalPSsaved_modelPScorePSopsPSvariable_ops_testDTcc mht_1(mht_1_v, 216, "", "./tensorflow/c/experimental/saved_model/core/ops/variable_ops_test.cc", "VariableOpsTest");
}

  EagerContext* context() {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScPSexperimentalPSsaved_modelPScorePSopsPSvariable_ops_testDTcc mht_2(mht_2_v, 221, "", "./tensorflow/c/experimental/saved_model/core/ops/variable_ops_test.cc", "context");
 return ctx_.get(); }

 private:
  std::unique_ptr<StaticDeviceMgr> device_mgr_;
  EagerContextPtr ctx_;
};

// Sanity check for variable creation
TEST_F(VariableOpsTest, CreateVariableSuccessful) {
  // Create a DT_Resource TensorHandle that points to a scalar DT_FLOAT tensor
  ImmediateTensorHandlePtr handle;
  TF_EXPECT_OK(internal::CreateUninitializedResourceVariable(
      context(), DT_FLOAT, {}, nullptr, &handle));
  // The created TensorHandle should be a DT_Resource
  EXPECT_EQ(handle->DataType(), DT_RESOURCE);
}

// Sanity check for variable destruction
TEST_F(VariableOpsTest, DestroyVariableSuccessful) {
  // Create a DT_Resource TensorHandle that points to a scalar DT_FLOAT tensor
  ImmediateTensorHandlePtr handle;
  TF_EXPECT_OK(internal::CreateUninitializedResourceVariable(
      context(), DT_FLOAT, {}, nullptr, &handle));

  // Destroy the variable
  TF_EXPECT_OK(internal::DestroyResource(context(), handle.get()));
}

// Sanity check for handle assignment and reading
TEST_F(VariableOpsTest, AssignVariableAndReadSuccessful) {
  // Create a DT_Resource TensorHandle that points to a scalar DT_FLOAT tensor
  ImmediateTensorHandlePtr variable;
  TF_EXPECT_OK(internal::CreateUninitializedResourceVariable(
      context(), DT_FLOAT, {}, nullptr, &variable));

  // Create a Scalar float TensorHandle with value 42, and assign it to
  // the variable.
  ImmediateTensorHandlePtr my_value = CreateScalarTensorHandle(context(), 42.0);
  TF_EXPECT_OK(internal::AssignVariable(context(), variable.get(), DT_FLOAT,
                                        my_value.get()));

  // Read back the value from the variable, and check that it is 42.
  ImmediateTensorHandlePtr read_value_handle;
  TF_EXPECT_OK(internal::ReadVariable(context(), variable.get(), DT_FLOAT,
                                      &read_value_handle));
  Status status;
  AbstractTensorPtr read_value(read_value_handle->Resolve(&status));
  TF_EXPECT_OK(status);
  EXPECT_FLOAT_EQ(42.0, *static_cast<float*>(read_value->Data()));
}

}  // namespace
}  // namespace tensorflow
