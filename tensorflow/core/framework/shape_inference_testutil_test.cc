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
class MHTracer_DTPStensorflowPScorePSframeworkPSshape_inference_testutil_testDTcc {
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
   MHTracer_DTPStensorflowPScorePSframeworkPSshape_inference_testutil_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSframeworkPSshape_inference_testutil_testDTcc() {
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

/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/framework/shape_inference_testutil.h"

#include "tensorflow/core/framework/node_def_builder.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {
namespace shape_inference {

namespace {

#define EXPECT_CONTAINS(str, substr)                              \
  do {                                                            \
    string s = (str);                                             \
    EXPECT_TRUE(absl::StrContains(s, substr)) << "String: " << s; \
  } while (false)

static OpShapeInferenceFn* global_fn_ptr = nullptr;
REGISTER_OP("OpOneOut")
    .Input("inputs: N * T")
    .Output("o1: T")
    .Attr("N: int >= 1")
    .Attr("T: numbertype")
    .SetShapeFn([](InferenceContext* c) { return (*global_fn_ptr)(c); });
REGISTER_OP("OpTwoOut")
    .Input("inputs: N * T")
    .Output("o1: T")
    .Output("o2: T")
    .Attr("N: int >= 1")
    .Attr("T: numbertype")
    .SetShapeFn([](InferenceContext* c) { return (*global_fn_ptr)(c); });

string RunInferShapes(const string& op_name, const string& ins,
                      const string& expected_outs, OpShapeInferenceFn fn) {
   std::vector<std::string> mht_0_v;
   mht_0_v.push_back("op_name: \"" + op_name + "\"");
   mht_0_v.push_back("ins: \"" + ins + "\"");
   mht_0_v.push_back("expected_outs: \"" + expected_outs + "\"");
   MHTracer_DTPStensorflowPScorePSframeworkPSshape_inference_testutil_testDTcc mht_0(mht_0_v, 223, "", "./tensorflow/core/framework/shape_inference_testutil_test.cc", "RunInferShapes");

  ShapeInferenceTestOp op(op_name);
  const int num_inputs = 1 + std::count(ins.begin(), ins.end(), ';');
  std::vector<NodeDefBuilder::NodeOut> src_list;
  src_list.reserve(num_inputs);
  for (int i = 0; i < num_inputs; ++i) src_list.emplace_back("a", 0, DT_FLOAT);
  NodeDef node_def;
  TF_CHECK_OK(NodeDefBuilder("dummy", op_name)
                  .Input(src_list)
                  .Attr("N", num_inputs)
                  .Finalize(&op.node_def));
  global_fn_ptr = &fn;
  return ShapeInferenceTestutil::InferShapes(op, ins, expected_outs)
      .error_message();
}

}  // namespace

TEST(ShapeInferenceTestutilTest, Failures) {
  auto fn_copy_input_0 = [](InferenceContext* c) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSshape_inference_testutil_testDTcc mht_1(mht_1_v, 245, "", "./tensorflow/core/framework/shape_inference_testutil_test.cc", "lambda");

    c->set_output(0, c->input(0));
    return Status::OK();
  };
  auto fn_copy_input_2 = [](InferenceContext* c) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSshape_inference_testutil_testDTcc mht_2(mht_2_v, 252, "", "./tensorflow/core/framework/shape_inference_testutil_test.cc", "lambda");

    c->set_output(0, c->input(2));
    return Status::OK();
  };
  auto fn_output_unknown_shapes = [](InferenceContext* c) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSshape_inference_testutil_testDTcc mht_3(mht_3_v, 259, "", "./tensorflow/core/framework/shape_inference_testutil_test.cc", "lambda");

    for (int i = 0; i < c->num_outputs(); ++i) {
      c->set_output(i, c->UnknownShape());
    }
    return Status::OK();
  };
  auto fn_output_1_2 = [](InferenceContext* c) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSshape_inference_testutil_testDTcc mht_4(mht_4_v, 268, "", "./tensorflow/core/framework/shape_inference_testutil_test.cc", "lambda");

    c->set_output(0, c->Matrix(1, 2));
    return Status::OK();
  };
  auto fn_output_u_2 = [](InferenceContext* c) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSshape_inference_testutil_testDTcc mht_5(mht_5_v, 275, "", "./tensorflow/core/framework/shape_inference_testutil_test.cc", "lambda");

    c->set_output(0, c->Matrix(InferenceContext::kUnknownDim, 2));
    return Status::OK();
  };
  const string& op = "OpOneOut";

  EXPECT_EQ("Shape inference should have returned error",
            RunInferShapes(op, "[1];[2];[1]", "e", fn_copy_input_0));
  EXPECT_CONTAINS(RunInferShapes(op, "[1];[2];[1]", "[1];[2]", fn_copy_input_0),
                  "wrong number of outputs");
  auto error_message = ShapeInferenceTestutil::InferShapes(
                           ShapeInferenceTestOp("NoSuchOp"), "", "")
                           .error_message();
  EXPECT_TRUE(
      absl::StartsWith(error_message, "Op type not registered 'NoSuchOp'"));

  // Wrong shape error messages.
  EXPECT_CONTAINS(RunInferShapes(op, "[1];[2];[1]", "?", fn_copy_input_0),
                  "expected to not match");
  EXPECT_CONTAINS(RunInferShapes(op, "[1];[2];[1]", "in2", fn_copy_input_0),
                  "should have matched one of (in2)");
  EXPECT_CONTAINS(RunInferShapes(op, "[1];[2];[1]", "in1|in2", fn_copy_input_0),
                  "should have matched one of (in1|in2)");
  EXPECT_CONTAINS(RunInferShapes(op, "[1];[2];[1]", "[1]", fn_copy_input_2),
                  "but was expected to not match");
  EXPECT_CONTAINS(RunInferShapes(op, "[1];[2];[1]", "in0|in1", fn_output_1_2),
                  "Output 0 should have matched an input shape");
  EXPECT_EQ("Output 0 expected to be unknown. Output shape was [1,2]",
            RunInferShapes(op, "[1];[2];[1]", "?", fn_output_1_2));
  EXPECT_EQ("Output 0 expected rank 3 but was 2. Output shape was [1,2]",
            RunInferShapes(op, "[1];[2];[1]", "[1,2,3]", fn_output_1_2));
  EXPECT_EQ(
      "Output 0 expected rank 2 but was ?. Output shape was ?",
      RunInferShapes(op, "[1];[2];[1]", "[1,2]", fn_output_unknown_shapes));

  // Wrong shape error messages on the second output.
  EXPECT_EQ("Output 1 expected rank 3 but was ?. Output shape was ?",
            RunInferShapes("OpTwoOut", "[1];[2];[1]", "?;[1,2,3]",
                           fn_output_unknown_shapes));

  // Wrong dimension error messages.
  EXPECT_EQ("Output dim 0,1 expected to be 3 but was 2. Output shape was [1,2]",
            RunInferShapes(op, "[1];[2];[1]", "[1,3]", fn_output_1_2));
  EXPECT_EQ("Output dim 0,0 expected to be 2 but was 1. Output shape was [1,2]",
            RunInferShapes(op, "[1];[2];[1]", "[2,2]", fn_output_1_2));
  EXPECT_EQ(
      "Output dim 0,0 expected to be unknown but was 1. Output shape was [1,2]",
      RunInferShapes(op, "[1];[2];[1]", "[?,2]", fn_output_1_2));
  EXPECT_EQ("Output dim 0,1 expected to be 1 but was 2. Output shape was [?,2]",
            RunInferShapes(op, "[1];[2];[1]", "[?,1]", fn_output_u_2));
  EXPECT_EQ("Output dim 0,0 expected to be 1 but was ?. Output shape was [?,2]",
            RunInferShapes(op, "[0,1,?];[2];[1]", "[1,2]", fn_output_u_2));
  auto fn = [](InferenceContext* c) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSshape_inference_testutil_testDTcc mht_6(mht_6_v, 330, "", "./tensorflow/core/framework/shape_inference_testutil_test.cc", "lambda");

    c->set_output(0, c->MakeShape({c->Dim(c->input(0), 1), c->MakeDim(2),
                                   c->UnknownDim(), c->Dim(c->input(2), 0)}));
    return Status::OK();
  };
  const string ins = "[0,1,?];[2];[1]";
  EXPECT_CONTAINS(RunInferShapes(op, ins, "[?,2,?,d2_0]", fn),
                  "Output dim 0,0 expected to be an unknown");
  EXPECT_CONTAINS(RunInferShapes(op, ins, "[0,2,?,d2_0]", fn),
                  "Output dim 0,0 expected to be 0 but matched input d0_1.");
  EXPECT_CONTAINS(
      RunInferShapes(op, ins, "[d0_0,2,?,d2_0]", fn),
      "dim 0,0 matched input d0_1, but should have matched one of (d0_0).");
  EXPECT_CONTAINS(RunInferShapes(op, ins, "[x,2,?,d2_0]", fn),
                  "Output dim 0,0: the expected dimension value 'x' failed to "
                  "parse as int64.");
  EXPECT_CONTAINS(RunInferShapes(op, ins, "[d0_0|d0_2,2,?,d2_0]", fn),
                  "dim 0,0 matched input d0_1, but should have matched one of "
                  "(d0_0|d0_2).");
  EXPECT_CONTAINS(RunInferShapes(op, ins, "[d0_1,?,?,d0_0|d2_0]", fn),
                  ("Output dim 0,1 expected to be unknown but was 2. "
                   "Output shape was [1,2,?,1]"));
  EXPECT_EQ(
      "Output dim 0,2 expected to be 8 but was ?. Output shape was [1,2,?,1]",
      RunInferShapes(op, ins, "[d0_1,2,8,d0_0|d2_0]", fn));
  EXPECT_CONTAINS(RunInferShapes(op, ins, "[d0_1,2,d0_1|d2_0,d0_0|d2_0]", fn),
                  "expected to match");
  EXPECT_EQ("",  // OK, no error.
            RunInferShapes(op, ins, "[d0_1,2,?,d0_0|d2_0]", fn));
}

}  // namespace shape_inference
}  // namespace tensorflow
