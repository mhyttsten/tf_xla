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
class MHTracer_DTPStensorflowPSccPSframeworkPSwhile_gradients_testDTcc {
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
   MHTracer_DTPStensorflowPSccPSframeworkPSwhile_gradients_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSccPSframeworkPSwhile_gradients_testDTcc() {
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

#include "tensorflow/cc/client/client_session.h"
#include "tensorflow/cc/framework/gradients.h"
#include "tensorflow/cc/framework/testutil.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/cc/ops/while_loop.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/tensor_testutil.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {

namespace {

class WhileGradientsTest : public ::testing::Test {
 protected:
  WhileGradientsTest() : scope_(Scope::NewRootScope()) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSccPSframeworkPSwhile_gradients_testDTcc mht_0(mht_0_v, 201, "", "./tensorflow/cc/framework/while_gradients_test.cc", "WhileGradientsTest");
}

  void Init(int num_inputs, DataType dtype = DT_INT32) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSccPSframeworkPSwhile_gradients_testDTcc mht_1(mht_1_v, 206, "", "./tensorflow/cc/framework/while_gradients_test.cc", "Init");

    for (int i = 0; i < num_inputs; ++i) {
      inputs_.push_back(ops::Placeholder(scope_, dtype));
    }
  }

  void CreateLoop(const ops::CondGraphBuilderFn& cond,
                  const ops::BodyGraphBuilderFn& body,
                  const std::vector<Output>* inputs = nullptr) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSccPSframeworkPSwhile_gradients_testDTcc mht_2(mht_2_v, 217, "", "./tensorflow/cc/framework/while_gradients_test.cc", "CreateLoop");

    if (inputs == nullptr) inputs = &inputs_;
    TF_ASSERT_OK(ops::BuildWhileLoop(scope_, *inputs, cond, body, "test_loop",
                                     &outputs_));
  }

  void CreateBackprop() {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPSccPSframeworkPSwhile_gradients_testDTcc mht_3(mht_3_v, 226, "", "./tensorflow/cc/framework/while_gradients_test.cc", "CreateBackprop");

    TF_ASSERT_OK(
        AddSymbolicGradients(scope_, outputs_, inputs_, &grad_outputs_));
    ASSERT_EQ(grad_outputs_.size(), inputs_.size());
  }

  template <typename T>
  void Run(const std::vector<Input::Initializer>& input_values,
           const std::vector<T>& expected_grad_values) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPSccPSframeworkPSwhile_gradients_testDTcc mht_4(mht_4_v, 237, "", "./tensorflow/cc/framework/while_gradients_test.cc", "Run");

    Run<T>(ClientSession(scope_), input_values, expected_grad_values);
  }

  template <typename T>
  void Run(const ClientSession& session,
           const std::vector<Input::Initializer>& input_values,
           const std::vector<T>& expected_grad_values,
           const RunOptions& run_options = RunOptions(),
           RunMetadata* run_metadata = nullptr) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPSccPSframeworkPSwhile_gradients_testDTcc mht_5(mht_5_v, 249, "", "./tensorflow/cc/framework/while_gradients_test.cc", "Run");

    DCHECK_EQ(input_values.size(), inputs_.size());
    ClientSession::FeedType feeds;
    for (int i = 0; i < inputs_.size(); ++i) {
      feeds.emplace(inputs_[i], input_values[i]);
    }

    std::vector<Operation> run_outputs;
    std::vector<Tensor> out_tensors;
    TF_ASSERT_OK(session.Run(run_options, feeds, grad_outputs_, run_outputs,
                             &out_tensors, run_metadata));
    ASSERT_EQ(out_tensors.size(), grad_outputs_.size());

    DCHECK_EQ(expected_grad_values.size(), out_tensors.size());
    for (int i = 0; i < out_tensors.size(); ++i) {
      test::ExpectTensorEqual<T>(
          out_tensors[i], test::AsTensor<T>({expected_grad_values[i]}, {}));
    }
  }

  Scope scope_;
  std::vector<Output> inputs_;
  std::vector<Output> outputs_;
  std::vector<Output> grad_outputs_;
};

TEST_F(WhileGradientsTest, Basic) {
  // Create loop: while (i < 10) i += 1
  Init(1);
  CreateLoop(
      [](const Scope& s, const std::vector<Output>& inputs, Output* output) {
        *output = ops::Less(s, inputs[0], 10);
        return s.status();
      },
      [](const Scope& s, const std::vector<Output>& inputs,
         std::vector<Output>* outputs) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPSccPSframeworkPSwhile_gradients_testDTcc mht_6(mht_6_v, 287, "", "./tensorflow/cc/framework/while_gradients_test.cc", "lambda");

        // Use AddN, rather than Add, because the gradient function doesn't
        // depend on the input shapes, and thus we do not need to store
        // intermediate values in a stack.
        outputs->push_back(ops::AddN(s, {inputs[0], 1}));
        return s.status();
      });
  CreateBackprop();

  Run<int>({1}, {1});
  Run<int>({11}, {1});
}

TEST_F(WhileGradientsTest, MultipleLoopVars) {
  // Create loop: while (i < 10) i += j; j += 1; k = k
  Init(3);
  CreateLoop(
      [](const Scope& s, const std::vector<Output>& inputs, Output* output) {
        *output = ops::Less(s, inputs[0], 10);
        return s.status();
      },
      [](const Scope& s, const std::vector<Output>& inputs,
         std::vector<Output>* outputs) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPSccPSframeworkPSwhile_gradients_testDTcc mht_7(mht_7_v, 312, "", "./tensorflow/cc/framework/while_gradients_test.cc", "lambda");

        outputs->push_back(ops::AddN(s, {inputs[0], inputs[1]}));
        outputs->push_back(ops::AddN(s, {inputs[1], 1}));
        outputs->push_back(inputs[2]);
        return s.status();
      });
  CreateBackprop();

  // The following execution traces illustrate why we expect dF/dj to be 5:
  //
  //  i  j  k
  // ---------
  //  0  1  2 <-- initial values
  //  1  2  2
  //  3  3  2
  //  6  4  2
  // 10  5  2 <-- while output values
  // outputs sum = 17
  //
  //  i  j  k
  // ---------
  //  0  2  2 <-- initial values (add 1 to j)
  //  2  3  2
  //  5  4  2
  //  9  5  2
  // 14  6  2 <-- while output values
  // outputs sum = 22
  //
  // Calculate the "slope" between j=1 and j=2:
  // 22 - 17 = 5 => dF/dj = 5
  Run<int>({0, 1, 2}, {1, 5, 1});

  Run<int>({1, 1, 0}, {1, 5, 1});
  Run<int>({0, 0, 0}, {1, 6, 1});
}

TEST_F(WhileGradientsTest, Chaining) {
  Init(2, DT_DOUBLE);

  // Multiply each input by 2 before passing to while loop to make sure chaining
  // works properly
  std::vector<Output> loop_inputs = {ops::Multiply(scope_, inputs_[0], 2.0),
                                     ops::Multiply(scope_, inputs_[1], 2.0)};

  // Create loop: while (i > 0 && j > 0) i -= 1
  CreateLoop(
      [](const Scope& s, const std::vector<Output>& inputs, Output* output) {
        *output = ops::LogicalAnd(s, ops::Greater(s, inputs[0], 0.0),
                                  ops::Greater(s, inputs[1], 0.0));
        return s.status();
      },
      [](const Scope& s, const std::vector<Output>& inputs,
         std::vector<Output>* outputs) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPSccPSframeworkPSwhile_gradients_testDTcc mht_8(mht_8_v, 367, "", "./tensorflow/cc/framework/while_gradients_test.cc", "lambda");

        outputs->push_back(ops::AddN(s, {inputs[0], -1.0}));
        outputs->push_back(inputs[1]);
        return s.status();
      },
      &loop_inputs);

  // Take negative of first output to make sure chaining works properly
  outputs_[0] = ops::Neg(scope_, outputs_[0]);

  CreateBackprop();

  Run<double>({1.0, 1.0}, {-2.0, 2.0});
  Run<double>({0.0, 0.0}, {-2.0, 2.0});
}

TEST_F(WhileGradientsTest, MultipleDevices) {
  // Make sure loop is created on cpu0
  scope_ = scope_.WithDevice("/cpu:0");

  // Create loop: while (i < 10) i += j
  Init(2);
  CreateLoop(
      [](const Scope& s, const std::vector<Output>& inputs, Output* output) {
        *output = ops::Less(s, inputs[0], 10);
        return s.status();
      },
      [](const Scope& s, const std::vector<Output>& inputs,
         std::vector<Output>* outputs) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPSccPSframeworkPSwhile_gradients_testDTcc mht_9(mht_9_v, 398, "", "./tensorflow/cc/framework/while_gradients_test.cc", "lambda");

        // Place body on cpu1
        Scope cpu1_scope = s.WithDevice("/cpu:1");
        outputs->push_back(ops::AddN(cpu1_scope, {inputs[0], inputs[1]}));
        outputs->push_back(inputs[1]);
        return cpu1_scope.status();
      });

  // Build gradient graph on cpu1
  Scope cpu1_scope = scope_.WithDevice("/cpu:1");
  TF_ASSERT_OK(
      AddSymbolicGradients(cpu1_scope, outputs_, inputs_, &grad_outputs_));
  ASSERT_EQ(grad_outputs_.size(), inputs_.size());

  // Run with two CPU devices and output partition graphs
  SessionOptions session_options;
  (*session_options.config.mutable_device_count())["CPU"] = 2;
  RunOptions run_options;
  run_options.set_output_partition_graphs(true);
  RunMetadata run_metadata;
  Run<int>(ClientSession(scope_, session_options), {0, 1}, {1, 11}, run_options,
           &run_metadata);

  // Check that at least one node ran on each device
  ASSERT_EQ(run_metadata.partition_graphs().size(), 2);
  for (const GraphDef& partition_graph : run_metadata.partition_graphs()) {
    EXPECT_GE(partition_graph.node().size(), 1);
  }
}

}  // namespace
}  // namespace tensorflow
