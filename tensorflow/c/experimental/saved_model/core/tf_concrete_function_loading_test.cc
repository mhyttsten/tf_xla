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
class MHTracer_DTPStensorflowPScPSexperimentalPSsaved_modelPScorePStf_concrete_function_loading_testDTcc {
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
   MHTracer_DTPStensorflowPScPSexperimentalPSsaved_modelPScorePStf_concrete_function_loading_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScPSexperimentalPSsaved_modelPScorePStf_concrete_function_loading_testDTcc() {
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
#include <unordered_map>

#include "tensorflow/c/eager/immediate_execution_tensor_handle.h"
#include "tensorflow/c/experimental/saved_model/core/revived_types/tensorhandle_convertible.h"
#include "tensorflow/c/experimental/saved_model/core/revived_types/tf_concrete_function.h"
#include "tensorflow/c/experimental/saved_model/core/saved_model_utils.h"
#include "tensorflow/c/experimental/saved_model/core/test_utils.h"
#include "tensorflow/c/experimental/saved_model/core/tf_concrete_function_test_protos.h"
#include "tensorflow/core/framework/function.pb.h"
#include "tensorflow/core/framework/op_def.pb.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/protobuf/error_codes.pb.h"
#include "tensorflow/core/protobuf/saved_object_graph.pb.h"

namespace tensorflow {
namespace {

class SavedConcreteFunctionLoadingTest : public ::testing::Test {
 public:
  SavedConcreteFunctionLoadingTest()
      : device_mgr_(testing::CreateTestingDeviceMgr()),
        ctx_(testing::CreateTestingEagerContext(device_mgr_.get())) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScPSexperimentalPSsaved_modelPScorePStf_concrete_function_loading_testDTcc mht_0(mht_0_v, 210, "", "./tensorflow/c/experimental/saved_model/core/tf_concrete_function_loading_test.cc", "SavedConcreteFunctionLoadingTest");
}

  EagerContext* context() {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScPSexperimentalPSsaved_modelPScorePStf_concrete_function_loading_testDTcc mht_1(mht_1_v, 215, "", "./tensorflow/c/experimental/saved_model/core/tf_concrete_function_loading_test.cc", "context");
 return ctx_.get(); }

 private:
  std::unique_ptr<StaticDeviceMgr> device_mgr_;
  EagerContextPtr ctx_;
};

class DummyCapture : public TensorHandleConvertible {
 public:
  DummyCapture(ImmediateExecutionContext* ctx, int8_t value)
      : TensorHandleConvertible(
            testing::CreateTensorHandle(ctx, DT_FLOAT, {2, 4}, value)) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScPSexperimentalPSsaved_modelPScorePStf_concrete_function_loading_testDTcc mht_2(mht_2_v, 229, "", "./tensorflow/c/experimental/saved_model/core/tf_concrete_function_loading_test.cc", "DummyCapture");
}
};

FunctionDef FuncDefWithNumInputsOutputs(int num_inputs, int num_outputs) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScPSexperimentalPSsaved_modelPScorePStf_concrete_function_loading_testDTcc mht_3(mht_3_v, 235, "", "./tensorflow/c/experimental/saved_model/core/tf_concrete_function_loading_test.cc", "FuncDefWithNumInputsOutputs");

  FunctionDef func;
  OpDef* signature = func.mutable_signature();
  for (int i = 0; i < num_inputs; ++i) {
    signature->add_input_arg();
  }
  for (int i = 0; i < num_outputs; ++i) {
    signature->add_output_arg();
  }
  return func;
}

// A SavedConcreteFunction whose canonicalized input signature
// has less inputs than its corresponding FunctionDef should cause an error.
TEST_F(SavedConcreteFunctionLoadingTest, TooFewInputsInSavedConcreteFunction) {
  // `saved` has 1 input
  SavedConcreteFunction saved;
  *saved.mutable_canonicalized_input_signature() =
      testing::SingleArgInputSignature();
  *saved.mutable_output_signature() = testing::ZeroReturnOutputSignature();

  // `func` has 2 inputs
  FunctionDef func = FuncDefWithNumInputsOutputs(2, 0);

  std::unique_ptr<TFConcreteFunction> result;
  Status status =
      internal::LoadTFConcreteFunction(saved, &func, {}, context(), &result);
  EXPECT_EQ(status.code(), error::FAILED_PRECONDITION)
      << status.error_message();
}

// A SavedConcreteFunction whose canonicalized input signature length +
// captures is less than its corresponding FunctionDef should cause an error.
TEST_F(SavedConcreteFunctionLoadingTest,
       TooFewInputsWithCapturesInSavedConcreteFunction) {
  // `saved` has 1 input, and 1 capture, for a total of 2 inputs
  SavedConcreteFunction saved;
  *saved.mutable_canonicalized_input_signature() =
      testing::SingleArgInputSignature();
  *saved.mutable_output_signature() = testing::ZeroReturnOutputSignature();
  saved.add_bound_inputs(5);

  // `func` has 3 inputs
  FunctionDef func = FuncDefWithNumInputsOutputs(3, 0);

  std::unordered_map<int, std::unique_ptr<TensorHandleConvertible>> captures;
  captures[5] = std::make_unique<DummyCapture>(context(), 10);

  std::unique_ptr<TFConcreteFunction> result;
  Status status = internal::LoadTFConcreteFunction(saved, &func, captures,
                                                   context(), &result);
  EXPECT_EQ(status.code(), error::FAILED_PRECONDITION)
      << status.error_message();
}

// A SavedConcreteFunction whose canonicalized input signature
// has more inputs than its corresponding FunctionDef should cause an error.
TEST_F(SavedConcreteFunctionLoadingTest, TooManyInputsInSavedConcreteFunction) {
  // `saved` has 3 inputs
  SavedConcreteFunction saved;
  *saved.mutable_canonicalized_input_signature() =
      testing::ThreeArgInputSignature();
  *saved.mutable_output_signature() = testing::ZeroReturnOutputSignature();

  // `func` has 2 inputs
  FunctionDef func = FuncDefWithNumInputsOutputs(2, 0);

  std::unique_ptr<TFConcreteFunction> result;
  Status status =
      internal::LoadTFConcreteFunction(saved, &func, {}, context(), &result);
  EXPECT_EQ(status.code(), error::FAILED_PRECONDITION)
      << status.error_message();
}

// A SavedConcreteFunction whose canonicalized input signature
// has the same number of inputs than its corresponding FunctionDef, but has
// additional captures should cause an error.
TEST_F(SavedConcreteFunctionLoadingTest,
       TooManyInputsWithCaptureInSavedConcreteFunction) {
  // `saved` has 3 inputs, and 1 capture, for a total of 4 inputs.
  SavedConcreteFunction saved;
  *saved.mutable_canonicalized_input_signature() =
      testing::ThreeArgInputSignature();
  *saved.mutable_output_signature() = testing::ZeroReturnOutputSignature();
  saved.add_bound_inputs(5);

  // `func` has 3 inputs.
  FunctionDef func = FuncDefWithNumInputsOutputs(3, 0);

  std::unordered_map<int, std::unique_ptr<TensorHandleConvertible>> captures;
  captures[5] = std::make_unique<DummyCapture>(context(), 10);

  std::unique_ptr<TFConcreteFunction> result;
  Status status = internal::LoadTFConcreteFunction(saved, &func, captures,
                                                   context(), &result);
  EXPECT_EQ(status.code(), error::FAILED_PRECONDITION)
      << status.error_message();
}

// A SavedConcreteFunction whose capture refers to an index not in the capture
// map should cause an error.
TEST_F(SavedConcreteFunctionLoadingTest, ImproperCaptureIndex) {
  // `saved` has 3 inputs, 1 capture, for a total of 4 inputs
  SavedConcreteFunction saved;
  *saved.mutable_canonicalized_input_signature() =
      testing::ThreeArgInputSignature();
  *saved.mutable_output_signature() = testing::ZeroReturnOutputSignature();
  // Capture is at index "10"
  saved.add_bound_inputs(10);

  // `func` has 4 inputs
  FunctionDef func = FuncDefWithNumInputsOutputs(4, 0);

  // `captures` only has a capture for index 5
  std::unordered_map<int, std::unique_ptr<TensorHandleConvertible>> captures;
  captures[5] = std::make_unique<DummyCapture>(context(), 10);

  std::unique_ptr<TFConcreteFunction> result;
  Status status = internal::LoadTFConcreteFunction(saved, &func, captures,
                                                   context(), &result);
  EXPECT_EQ(status.code(), error::FAILED_PRECONDITION)
      << status.error_message();
}

// A SavedConcreteFunction whose outputs are fewer than its corresponding
// functiondef should cause an error.
TEST_F(SavedConcreteFunctionLoadingTest, TooFewOutputsInSavedConcreteFunction) {
  // `saved` has 0 inputs, 1 output
  SavedConcreteFunction saved;
  *saved.mutable_canonicalized_input_signature() =
      testing::ZeroArgInputSignature();
  *saved.mutable_output_signature() = testing::SingleReturnOutputSignature();

  // `func` has 0 inputs, 2 outputs
  FunctionDef func = FuncDefWithNumInputsOutputs(0, 2);

  std::unique_ptr<TFConcreteFunction> result;
  Status status =
      internal::LoadTFConcreteFunction(saved, &func, {}, context(), &result);
  EXPECT_EQ(status.code(), error::FAILED_PRECONDITION)
      << status.error_message();
}

// A SavedConcreteFunction whose outputs exceed its corresponding functiondef
// should cause an error.
TEST_F(SavedConcreteFunctionLoadingTest,
       TooManyOutputsInSavedConcreteFunction) {
  // `saved` has 1 input, 3 outputs
  SavedConcreteFunction saved;
  *saved.mutable_canonicalized_input_signature() =
      testing::SingleArgInputSignature();
  *saved.mutable_output_signature() = testing::ThreeReturnOutputSignature();

  // `func` has 1 input, 2 outputs
  FunctionDef func = FuncDefWithNumInputsOutputs(1, 2);

  std::unique_ptr<TFConcreteFunction> result;
  Status status =
      internal::LoadTFConcreteFunction(saved, &func, {}, context(), &result);
  EXPECT_EQ(status.code(), error::FAILED_PRECONDITION)
      << status.error_message();
}

// A SavedConcreteFunction whose (inputs + captures) = functiondef inputs,
// and whose outputs = functiondef outputs should successfully load.
TEST_F(SavedConcreteFunctionLoadingTest, SuccessfulLoad) {
  // `saved` has 1 input, 2 captures, 3 outputs
  SavedConcreteFunction saved;
  *saved.mutable_canonicalized_input_signature() =
      testing::SingleArgInputSignature();
  *saved.mutable_output_signature() = testing::ThreeReturnOutputSignature();
  saved.add_bound_inputs(2);
  saved.add_bound_inputs(5);

  // `func` has 3 inputs, 3 outputs
  FunctionDef func = FuncDefWithNumInputsOutputs(3, 3);

  std::unordered_map<int, std::unique_ptr<TensorHandleConvertible>> captures;
  captures[2] = std::make_unique<DummyCapture>(context(), 1);
  captures[5] = std::make_unique<DummyCapture>(context(), 10);

  std::unique_ptr<TFConcreteFunction> result;
  Status status = internal::LoadTFConcreteFunction(saved, &func, captures,
                                                   context(), &result);
  TF_EXPECT_OK(status) << status.error_message();
}

// A TFConcreteFunction should register functiondefs on creation, and
// remove them upon deletion.
TEST_F(SavedConcreteFunctionLoadingTest, RegistersAndRemovesFunctionDefs) {
  std::string func_name = "FooBarBazWombatFunction";

  SavedConcreteFunction saved;
  *saved.mutable_canonicalized_input_signature() =
      testing::ZeroArgInputSignature();
  *saved.mutable_output_signature() = testing::ZeroReturnOutputSignature();
  FunctionDef func = FuncDefWithNumInputsOutputs(0, 0);
  *func.mutable_signature()->mutable_name() = func_name;

  {
    std::unique_ptr<TFConcreteFunction> result;
    Status status =
        internal::LoadTFConcreteFunction(saved, &func, {}, context(), &result);
    TF_EXPECT_OK(status) << status.error_message();
    // The function should be registered with context.
    EXPECT_TRUE(context()->FindFunctionByName(func_name));
  }

  // After `result's` destructor runs, the function should no longer be
  // registered with context.
  EXPECT_FALSE(context()->FindFunctionByName(func_name));
}

}  // namespace
}  // namespace tensorflow
