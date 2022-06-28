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
class MHTracer_DTPStensorflowPScPSwhile_loop_testDTcc {
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
   MHTracer_DTPStensorflowPScPSwhile_loop_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScPSwhile_loop_testDTcc() {
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

#include "tensorflow/c/c_api.h"
#include "tensorflow/c/c_test_util.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/strcat.h"
#include "tensorflow/core/platform/test.h"

using tensorflow::GraphDef;

namespace {

class CApiWhileLoopTest : public ::testing::Test {
 protected:
  CApiWhileLoopTest() : s_(TF_NewStatus()), graph_(TF_NewGraph()) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScPSwhile_loop_testDTcc mht_0(mht_0_v, 197, "", "./tensorflow/c/while_loop_test.cc", "CApiWhileLoopTest");
}

  ~CApiWhileLoopTest() override {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScPSwhile_loop_testDTcc mht_1(mht_1_v, 202, "", "./tensorflow/c/while_loop_test.cc", "~CApiWhileLoopTest");

    TF_DeleteGraph(graph_);
    TF_DeleteStatus(s_);
  }

  void Init(int ninputs) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScPSwhile_loop_testDTcc mht_2(mht_2_v, 210, "", "./tensorflow/c/while_loop_test.cc", "Init");

    DCHECK(inputs_.empty());
    DCHECK_GT(ninputs, 0);

    for (int i = 0; i < ninputs; ++i) {
      TF_Operation* placeholder = Placeholder(
          graph_, s_, ::tensorflow::strings::StrCat("p", i).c_str());
      DCHECK_EQ(TF_OK, TF_GetCode(s_)) << TF_Message(s_);
      inputs_.push_back({placeholder, 0});
    }

    original_graph_description_ = GraphDebugString();

    params_.reset(new TF_WhileParams(
        TF_NewWhile(graph_, &inputs_[0], inputs_.size(), s_)));
    ASSERT_EQ(TF_OK, TF_GetCode(s_)) << TF_Message(s_);
    ASSERT_EQ(original_graph_description_, GraphDebugString())
        << "TF_NewWhile() altered graph";

    params_->name = "test_loop";

    // Initialize outputs_ so we can easily detect errors/bugs
    outputs_.resize(ninputs, {nullptr, -1});
  }

  void ExpectOK() {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScPSwhile_loop_testDTcc mht_3(mht_3_v, 238, "", "./tensorflow/c/while_loop_test.cc", "ExpectOK");

    TF_FinishWhile(params_.get(), s_, &outputs_[0]);
    EXPECT_EQ(TF_OK, TF_GetCode(s_)) << TF_Message(s_);
  }

  void ExpectError(TF_Code expected_code, const string& expected_msg) {
   std::vector<std::string> mht_4_v;
   mht_4_v.push_back("expected_msg: \"" + expected_msg + "\"");
   MHTracer_DTPStensorflowPScPSwhile_loop_testDTcc mht_4(mht_4_v, 247, "", "./tensorflow/c/while_loop_test.cc", "ExpectError");

    TF_FinishWhile(params_.get(), s_, &outputs_[0]);
    EXPECT_EQ(expected_code, TF_GetCode(s_));
    EXPECT_EQ(expected_msg, TF_Message(s_));
    // TODO(skyewm): this assert is currently broken. Fix or remove guarantee.
    // ASSERT_EQ(original_graph_description_, GraphDebugString()) <<
    //     "TF_FinishWhile() altered graph on error";
  }

  void Run(std::initializer_list<int> input_values) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScPSwhile_loop_testDTcc mht_5(mht_5_v, 259, "", "./tensorflow/c/while_loop_test.cc", "Run");

    Run(outputs_, input_values);
  }

  void Run(const std::vector<TF_Output>& run_outputs,
           std::initializer_list<int> input_values) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScPSwhile_loop_testDTcc mht_6(mht_6_v, 267, "", "./tensorflow/c/while_loop_test.cc", "Run");

    DCHECK_EQ(inputs_.size(), input_values.size());
    std::vector<std::pair<TF_Operation*, TF_Tensor*>> inputs(inputs_.size());
    int i = 0;
    for (int v : input_values) {
      inputs[i] = {inputs_[i].oper, Int32Tensor(v)};
      ++i;
    }
    // TODO(skyewm): use std::make_unique or absl::make_unique when possible.
    csession_.reset(new CSession(graph_, s_));
    csession_->SetInputs(inputs);
    csession_->SetOutputs(run_outputs);
    csession_->Run(s_);
    ASSERT_EQ(TF_OK, TF_GetCode(s_)) << TF_Message(s_);
  }

  void ExpectOutputValue(int idx, int expected_value) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScPSwhile_loop_testDTcc mht_7(mht_7_v, 286, "", "./tensorflow/c/while_loop_test.cc", "ExpectOutputValue");

    TF_Tensor* out = csession_->output_tensor(idx);
    ASSERT_TRUE(out != nullptr);
    EXPECT_EQ(TF_INT32, TF_TensorType(out));
    EXPECT_EQ(0, TF_NumDims(out));
    ASSERT_EQ(sizeof(int32_t), TF_TensorByteSize(out));
    int32_t* data = static_cast<int32_t*>(TF_TensorData(out));
    EXPECT_EQ(expected_value, *data);
  }

  // Create a valid conditional graph. Useful for testing unrelated errors.
  void CreateCondGraph() {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScPSwhile_loop_testDTcc mht_8(mht_8_v, 300, "", "./tensorflow/c/while_loop_test.cc", "CreateCondGraph");

    TF_Operation* one = ScalarConst(1, params_->cond_graph, s_);
    TF_Operation* less_than =
        LessThan(params_->cond_inputs[0], {one, 0}, params_->cond_graph, s_);
    DCHECK_EQ(TF_OK, TF_GetCode(s_)) << TF_Message(s_);
    params_->cond_output = {less_than, 0};
  }

  string GraphDebugString() const {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScPSwhile_loop_testDTcc mht_9(mht_9_v, 311, "", "./tensorflow/c/while_loop_test.cc", "GraphDebugString");

    TF_Buffer* buf = TF_NewBuffer();
    TF_GraphToGraphDef(graph_, buf, s_);
    DCHECK_EQ(TF_OK, TF_GetCode(s_)) << TF_Message(s_);
    GraphDef def;
    bool success = def.ParseFromArray(buf->data, buf->length);
    DCHECK(success);
    TF_DeleteBuffer(buf);
    return def.DebugString();
  }

  TF_Status* s_;
  TF_Graph* graph_;
  std::vector<TF_Output> inputs_;   // The inputs to the while loop
  std::vector<TF_Output> outputs_;  // The final outputs of the while loop
  std::unique_ptr<TF_WhileParams> params_;
  std::unique_ptr<CSession> csession_;

 private:
  // Used to verify that errors don't change graph_
  string original_graph_description_;
};

TEST_F(CApiWhileLoopTest, BasicLoop) {
  Init(2);

  // Validate TF_WhileParams returned by TF_NewWhile()
  EXPECT_TRUE(params_->body_graph != nullptr);
  EXPECT_TRUE(params_->cond_graph != nullptr);

  EXPECT_EQ(params_->ninputs, 2);

  ASSERT_TRUE(params_->cond_inputs != nullptr);
  ASSERT_TRUE(params_->cond_inputs[0].oper != nullptr);
  EXPECT_TRUE(params_->cond_inputs[1].oper != nullptr);

  ASSERT_TRUE(params_->body_inputs != nullptr);
  EXPECT_TRUE(params_->body_inputs[0].oper != nullptr);
  EXPECT_TRUE(params_->body_inputs[1].oper != nullptr);

  ASSERT_TRUE(params_->body_outputs != nullptr);

  // Create loop: while (input1 < input2) input1 += input2 + 1
  TF_Operation* less_than =
      LessThan(params_->cond_inputs[0], params_->cond_inputs[1],
               params_->cond_graph, s_);
  ASSERT_EQ(TF_OK, TF_GetCode(s_)) << TF_Message(s_);
  params_->cond_output = {less_than, 0};

  TF_Operation* add1 = Add(params_->body_inputs[0], params_->body_inputs[1],
                           params_->body_graph, s_, "add1");
  ASSERT_EQ(TF_OK, TF_GetCode(s_)) << TF_Message(s_);
  TF_Operation* one = ScalarConst(1, params_->body_graph, s_);
  ASSERT_EQ(TF_OK, TF_GetCode(s_)) << TF_Message(s_);
  TF_Operation* add2 = Add(add1, one, params_->body_graph, s_, "add2");
  ASSERT_EQ(TF_OK, TF_GetCode(s_)) << TF_Message(s_);
  params_->body_outputs[0] = {add2, 0};
  params_->body_outputs[1] = params_->body_inputs[1];

  // Finalize while loop
  ExpectOK();

  // Validate while loop outputs returned by TF_FinishWhile()
  EXPECT_TRUE(outputs_[0].oper != nullptr);
  EXPECT_GE(outputs_[0].index, 0);
  EXPECT_TRUE(outputs_[1].oper != nullptr);
  EXPECT_GE(outputs_[1].index, 0);

  // Check that cond and body inputs are not present
  for (int i = 0; i < params_->ninputs; ++i) {
    string cond_name =
        ::tensorflow::strings::StrCat(params_->name, "/cond/cond_input", i);
    string body_name =
        ::tensorflow::strings::StrCat(params_->name, "/body/body_input", i);
    EXPECT_TRUE(TF_GraphOperationByName(graph_, cond_name.c_str()) == nullptr);
    EXPECT_TRUE(TF_GraphOperationByName(graph_, body_name.c_str()) == nullptr);
  }

  // Run the graph
  Run({-9, 2});
  ExpectOutputValue(0, 3);
  ExpectOutputValue(1, 2);
}

TEST_F(CApiWhileLoopTest, NestedLoop) {
  Init(2);
  // Create nested loop:
  //  while (input1 < 6) {
  //    inner_input1 = input1
  //    while (inner_input1 < 3) {
  //      input2 += 1
  //      inner_input1 += 2
  //    }
  //    input1 += input2
  //  }
  //
  // Expected execution with initial values input1 = input2 = 0:
  //
  // outer inner               inner_
  // step# step# input1 input2 input1
  // ------------------------------------
  //   0     0     0      0      0
  //   0     1     0      1      2
  //   0     2     0      2      4
  //   0     -     2      2      -
  //   1     0     2      2      2
  //   1     1     2      3      4
  //   1     -     5      3      -
  //   2     0     5      3      5
  //   2     -     8      3      -

  // Create outer cond graph
  TF_Operation* six = ScalarConst(6, params_->cond_graph, s_);
  ASSERT_EQ(TF_OK, TF_GetCode(s_)) << TF_Message(s_);
  TF_Operation* less_than =
      LessThan(params_->cond_inputs[0], {six, 0}, params_->cond_graph, s_);
  ASSERT_EQ(TF_OK, TF_GetCode(s_)) << TF_Message(s_);
  params_->cond_output = {less_than, 0};

  // Create outer body graph
  // Init inner graph
  TF_Output inner_inputs[] = {params_->body_inputs[0], params_->body_inputs[1]};
  TF_WhileParams inner_params =
      TF_NewWhile(params_->body_graph, inner_inputs, 2, s_);
  ASSERT_EQ(TF_OK, TF_GetCode(s_)) << TF_Message(s_);
  inner_params.name = "inner_loop";

  // Create inner cond graph
  TF_Operation* three = ScalarConst(3, inner_params.cond_graph, s_);
  ASSERT_EQ(TF_OK, TF_GetCode(s_)) << TF_Message(s_);
  TF_Operation* inner_less_than = LessThan(
      inner_params.cond_inputs[0], {three, 0}, inner_params.cond_graph, s_);
  ASSERT_EQ(TF_OK, TF_GetCode(s_)) << TF_Message(s_);
  inner_params.cond_output = {inner_less_than, 0};

  // Create inner body graph
  TF_Operation* one = ScalarConst(1, inner_params.body_graph, s_, "one");
  ASSERT_EQ(TF_OK, TF_GetCode(s_)) << TF_Message(s_);
  TF_Operation* two = ScalarConst(2, inner_params.body_graph, s_, "two");
  ASSERT_EQ(TF_OK, TF_GetCode(s_)) << TF_Message(s_);

  TF_Operation* input2_add =
      Add(inner_params.body_inputs[1].oper, one, inner_params.body_graph, s_);
  ASSERT_EQ(TF_OK, TF_GetCode(s_)) << TF_Message(s_);
  inner_params.body_outputs[1] = {input2_add, 0};

  TF_Operation* inner_input1_add = Add(inner_params.body_inputs[0].oper, two,
                                       inner_params.body_graph, s_, "add2");
  ASSERT_EQ(TF_OK, TF_GetCode(s_)) << TF_Message(s_);
  inner_params.body_outputs[0] = {inner_input1_add, 0};

  // Finalize inner graph
  TF_Output inner_outputs[2] = {{nullptr, -1}};
  TF_FinishWhile(&inner_params, s_, inner_outputs);
  ASSERT_EQ(TF_OK, TF_GetCode(s_)) << TF_Message(s_);

  TF_Operation* input1_add =
      Add(params_->body_inputs[0], inner_outputs[1], params_->body_graph, s_);
  ASSERT_EQ(TF_OK, TF_GetCode(s_)) << TF_Message(s_);
  params_->body_outputs[0] = {input1_add, 0};

  params_->body_outputs[1] = inner_outputs[1];

  // Finalize outer graph
  ExpectOK();

  // Check for a few expected nodes
  const char* node_name = "test_loop/cond/scalar";
  EXPECT_TRUE(TF_GraphOperationByName(graph_, node_name) != nullptr);
  node_name = "test_loop/body/add";
  EXPECT_TRUE(TF_GraphOperationByName(graph_, node_name) != nullptr);
  node_name = "test_loop/body/inner_loop/body/one";
  EXPECT_TRUE(TF_GraphOperationByName(graph_, node_name) != nullptr);
  node_name = "test_loop/body/inner_loop/cond/less_than";
  EXPECT_TRUE(TF_GraphOperationByName(graph_, node_name) != nullptr);

  // Run the graph
  Run({0, 0});
  ExpectOutputValue(0, 8);
  ExpectOutputValue(1, 3);
}

TEST_F(CApiWhileLoopTest, UnsetCondOutput) {
  Init(1);
  params_->body_outputs[0] = params_->body_inputs[0];
  ExpectError(TF_INVALID_ARGUMENT,
              "TF_WhileParams `cond_output` field isn't set");
}

TEST_F(CApiWhileLoopTest, WrongCondOutputType) {
  Init(1);
  params_->cond_output = params_->cond_inputs[0];
  params_->body_outputs[0] = params_->body_inputs[0];
  ExpectError(TF_INVALID_ARGUMENT,
              "BuildWhileLoop: 'cond' argument must return a boolean output, "
              "got int32");
}

TEST_F(CApiWhileLoopTest, InvalidCondOutputNode) {
  Init(1);
  // Try to reuse node from parent graph
  params_->cond_output = inputs_[0];
  params_->body_outputs[0] = params_->body_inputs[0];
  // TODO(skyewm): this error message could be more informative. Add explicit
  // checks for this case in the while loop implementation?
  ExpectError(TF_INVALID_ARGUMENT,
              "Requested return tensor 'p0:0' not found in graph def");
}

TEST_F(CApiWhileLoopTest, InvalidCondOutputIndex) {
  Init(1);
  CreateCondGraph();
  params_->cond_output.index = 100;
  params_->body_outputs[0] = params_->body_inputs[0];
  ExpectError(TF_INVALID_ARGUMENT,
              "Invalid return output 100 of node 'less_than', which has 1 "
              "output(s)");
}

// TODO(skyewm): test bad cond output shape

TEST_F(CApiWhileLoopTest, UnsetBodyOutput) {
  Init(1);
  CreateCondGraph();
  ExpectError(TF_INVALID_ARGUMENT,
              "TF_WhileParams `body_outputs[0]` field isn't set");
}

// TODO(skyewm): enable this when it works (currently doesn't error)
// TEST_F(CApiWhileLoopTest, WrongBodyOutputType) {
//   Init(1);
//   CreateCondGraph();
//   TF_Operation* double_scalar =
//       ScalarConst(1.0, params_->body_graph, s_, "double_scalar");
//   params_->body_outputs[0] = {double_scalar, 0};
//   ExpectError(TF_INVALID_ARGUMENT, "bad body output type");
// }

TEST_F(CApiWhileLoopTest, InvalidBodyOutputNode) {
  Init(1);
  CreateCondGraph();
  // Try to reuse node from parent graph
  params_->body_outputs[0] = inputs_[0];
  // TODO(skyewm): this error message could be more informative. Add explicit
  // checks for this case in the while loop implementation?
  ExpectError(TF_INVALID_ARGUMENT,
              "Requested return tensor 'p0:0' not found in graph def");
}

// TODO(skyewm): enable this when it works (currently segfaults!)
// TEST_F(CApiWhileLoopTest, InvalidBodyOutputIndex) {
//   Init(1);
//   CreateCondGraph();
//   params_->body_outputs[0] = params_->body_inputs[0];
//   params_->body_outputs[0].index = 100;
//   ExpectError(TF_INVALID_ARGUMENT,
//               "Invalid return output 100 of node 'less_than', which has 1 "
//               "output(s)");
// }

// TODO(skyewm): test bad body output shape

TEST_F(CApiWhileLoopTest, NullName) {
  Init(1);
  CreateCondGraph();
  params_->body_outputs[0] = params_->body_inputs[0];
  params_->name = nullptr;
  ExpectError(TF_INVALID_ARGUMENT, "TF_WhileParams `name` field is null");
}

TEST_F(CApiWhileLoopTest, WrongGraph) {
  Init(1);
  CreateCondGraph();
  // Set body output to output from outer graph
  params_->body_outputs[0] = inputs_[0];
  // TODO(skyewm): improve error message
  ExpectError(TF_INVALID_ARGUMENT,
              "Requested return tensor 'p0:0' not found in graph def");
}

TEST_F(CApiWhileLoopTest, BadTypes) {
  Init(1);
  CreateCondGraph();
  // Op that has a float input + output
  TF_OperationDescription* desc = TF_NewOperation(
      params_->body_graph, "FakeQuantWithMinMaxArgs", "float_op");
  TF_AddInput(desc, params_->body_inputs[0]);
  TF_FinishOperation(desc, s_);
  ASSERT_EQ(TF_INVALID_ARGUMENT, TF_GetCode(s_));
  string msg(TF_Message(s_));
  EXPECT_NE(msg.find("Input 'inputs' passed int32 expected float while "
                     "building NodeDef 'float_op'"),
            msg.npos);
  TF_AbortWhile(params_.get());
}

// This is a basic test to make sure the C++ gradient code can handle while
// loops created by the C API (which calls the C++ API under the hood). There
// are more while loop gradient tests in cc/framework/while_gradients_test.cc.
TEST_F(CApiWhileLoopTest, Gradients) {
  Init(1);

  // Create loop: while (i < 10) i += 1
  TF_Operation* ten = ScalarConst(10, params_->cond_graph, s_);
  TF_Operation* less_than =
      LessThan(params_->cond_inputs[0], {ten, 0}, params_->cond_graph, s_);
  DCHECK_EQ(TF_OK, TF_GetCode(s_)) << TF_Message(s_);
  params_->cond_output = {less_than, 0};

  TF_Operation* one = ScalarConst(1, params_->body_graph, s_);
  TF_Operation* add =
      Add(params_->body_inputs[0], {one, 0}, params_->body_graph, s_);
  ASSERT_EQ(TF_OK, TF_GetCode(s_)) << TF_Message(s_);
  params_->body_outputs[0] = {add, 0};

  ExpectOK();

  // Create backprop graph
  TF_Output grad_output;
  TF_AddGradients(graph_, outputs_.data(), outputs_.size(), inputs_.data(), 1,
                  nullptr, s_, &grad_output);
  ASSERT_EQ(TF_OK, TF_GetCode(s_)) << TF_Message(s_);

  // Run gradient
  Run({grad_output}, {0});
  ExpectOutputValue(0, 1);
}

}  // namespace
