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
class MHTracer_DTPStensorflowPSccPSopsPSwhile_loop_testDTcc {
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
   MHTracer_DTPStensorflowPSccPSopsPSwhile_loop_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSccPSopsPSwhile_loop_testDTcc() {
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

#include "tensorflow/cc/ops/while_loop.h"
#include "tensorflow/cc/client/client_session.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/framework/tensor_testutil.h"
#include "tensorflow/core/graph/while_context.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {

namespace {

class WhileLoopTest : public ::testing::Test {
 protected:
  WhileLoopTest() : scope_(Scope::NewRootScope()) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSccPSopsPSwhile_loop_testDTcc mht_0(mht_0_v, 199, "", "./tensorflow/cc/ops/while_loop_test.cc", "WhileLoopTest");
}

  void Init(int num_inputs, DataType dtype = DT_INT32) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSccPSopsPSwhile_loop_testDTcc mht_1(mht_1_v, 204, "", "./tensorflow/cc/ops/while_loop_test.cc", "Init");

    for (int i = 0; i < num_inputs; ++i) {
      inputs_.push_back(ops::Placeholder(scope_, dtype));
    }
  }

  void CreateLoop(const ops::CondGraphBuilderFn& cond,
                  const ops::BodyGraphBuilderFn& body,
                  error::Code error_code = error::OK,
                  const string& error_msg = "") {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSccPSopsPSwhile_loop_testDTcc mht_2(mht_2_v, 216, "", "./tensorflow/cc/ops/while_loop_test.cc", "CreateLoop");

    Status s =
        ops::BuildWhileLoop(scope_, inputs_, cond, body, kFrameName, &outputs_);
    EXPECT_EQ(s.code(), error_code);
    EXPECT_EQ(s.error_message(), error_msg);
  }

  template <typename T>
  void Run(const std::vector<Input::Initializer>& input_values,
           const std::vector<T>& expected_output_values) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPSccPSopsPSwhile_loop_testDTcc mht_3(mht_3_v, 228, "", "./tensorflow/cc/ops/while_loop_test.cc", "Run");

    ClientSession session(scope_);

    DCHECK_EQ(input_values.size(), inputs_.size());
    ClientSession::FeedType feeds;
    for (int i = 0; i < inputs_.size(); ++i) {
      feeds.emplace(inputs_[i], input_values[i]);
    }

    std::vector<Tensor> out_tensors;
    TF_ASSERT_OK(session.Run(feeds, outputs_, &out_tensors));
    ASSERT_EQ(out_tensors.size(), outputs_.size());

    DCHECK_EQ(expected_output_values.size(), out_tensors.size());
    for (int i = 0; i < out_tensors.size(); ++i) {
      test::ExpectTensorEqual<T>(
          out_tensors[i], test::AsTensor<T>({expected_output_values[i]}, {}));
    }
  }

  Scope scope_;
  std::vector<Output> inputs_;
  std::vector<Output> outputs_;

  static const char* const kFrameName;
};

const char* const WhileLoopTest::kFrameName = "test_loop";

Status LessThanTenCond(const Scope& s, const std::vector<Output>& inputs,
                       Output* output) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPSccPSopsPSwhile_loop_testDTcc mht_4(mht_4_v, 261, "", "./tensorflow/cc/ops/while_loop_test.cc", "LessThanTenCond");

  *output = ops::Less(s, inputs[0], 10);
  return s.status();
}

Status AddOneBody(const Scope& s, const std::vector<Output>& inputs,
                  std::vector<Output>* outputs) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPSccPSopsPSwhile_loop_testDTcc mht_5(mht_5_v, 270, "", "./tensorflow/cc/ops/while_loop_test.cc", "AddOneBody");

  outputs->push_back(ops::Add(s, inputs[0], 1));
  return s.status();
}

TEST_F(WhileLoopTest, Basic) {
  // Create loop: while (i < 10) i += 1
  Init(1);
  CreateLoop(LessThanTenCond, AddOneBody);

  // Verify some output invariants
  WhileContext* while_ctx;
  for (int i = 0; i < outputs_.size(); ++i) {
    Node* node = outputs_[i].node();
    ASSERT_TRUE(node->IsExit()) << "Output node " << i << ":\n"
                                << node->DebugString();
    ASSERT_TRUE(node->while_ctx() != nullptr) << i;
    if (i == 0) {
      while_ctx = node->while_ctx();
      EXPECT_EQ(while_ctx->frame_name(), kFrameName);
    } else {
      EXPECT_EQ(node->while_ctx(), while_ctx) << i;
    }
  }

  // Run the loop and test we get the expected results
  Run<int>({1}, {10});
  Run<int>({11}, {11});
}

TEST_F(WhileLoopTest, WrongCondOutputType) {
  Init(1);
  CreateLoop(
      [](const Scope& s, const std::vector<Output>& inputs, Output* output) {
        *output = ops::Placeholder(s, DT_FLOAT);
        return s.status();
      },
      AddOneBody, error::INVALID_ARGUMENT,
      "BuildWhileLoop: 'cond' argument must return a boolean output, got "
      "float");
}

// TODO(skyewm): test bad cond output shape

TEST_F(WhileLoopTest, NullCondOutputNode) {
  Init(1);
  // TODO(skyewm): improve error message
  CreateLoop(
      [](const Scope& s, const std::vector<Output>& inputs, Output* output) {
        *output = {nullptr, 0};
        return s.status();
      },
      AddOneBody, error::INVALID_ARGUMENT, "Node is null");
}

TEST_F(WhileLoopTest, InvalidCondOutputIndex) {
  Init(1);
  CreateLoop(
      [](const Scope& s, const std::vector<Output>& inputs, Output* output) {
        auto less = ops::Less(s, inputs[0], 10);
        *output = {less.node(), 100};
        return s.status();
      },
      AddOneBody, error::OUT_OF_RANGE,
      "Node 'cond/Less' (type: 'Less', num of outputs: 1) does not have output "
      "100");
}

TEST_F(WhileLoopTest, UnsetCondOutput) {
  Init(1);
  CreateLoop([](const Scope& s, const std::vector<Output>& inputs,
                Output* output) { return s.status(); },
             AddOneBody, error::INVALID_ARGUMENT, "Node is null");
}

// TODO(skyewm): test bad body output type
// TODO(skyewm): test bad body output shape

TEST_F(WhileLoopTest, NullBodyOutputNode) {
  Init(1);
  // TODO(skyewm): improve error message
  CreateLoop(LessThanTenCond,
             [](const Scope& s, const std::vector<Output>& inputs,
                std::vector<Output>* outputs) {
               outputs->push_back({nullptr, 0});
               return s.status();
             },
             error::INVALID_ARGUMENT, "Node is null");
}

TEST_F(WhileLoopTest, InvalidBodyOutputIndex) {
  Init(1);
  CreateLoop(LessThanTenCond,
             [](const Scope& s, const std::vector<Output>& inputs,
                std::vector<Output>* outputs) {
               auto add = ops::Add(s, inputs[0], 1);
               outputs->emplace_back(add.node(), 100);
               return s.status();
             },
             error::OUT_OF_RANGE,
             "Node 'body/Add' (type: 'Add', num of outputs: 1) does not have "
             "output 100");
}

TEST_F(WhileLoopTest, UnsetBodyOutputs) {
  Init(1);
  CreateLoop(
      LessThanTenCond,
      [](const Scope& s, const std::vector<Output>& inputs,
         std::vector<Output>* outputs) { return s.status(); },
      error::INVALID_ARGUMENT,
      "BuildWhileLoop: 'body' argument expected to return 1 output(s), got 0");
}

}  // namespace
}  // namespace tensorflow
