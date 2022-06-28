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
class MHTracer_DTPStensorflowPStoolsPSgraph_transformsPStransform_graph_testDTcc {
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
   MHTracer_DTPStensorflowPStoolsPSgraph_transformsPStransform_graph_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPStoolsPSgraph_transformsPStransform_graph_testDTcc() {
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

/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/tools/graph_transforms/transform_graph.h"
#include "tensorflow/cc/ops/const_op.h"
#include "tensorflow/cc/ops/image_ops.h"
#include "tensorflow/cc/ops/nn_ops.h"
#include "tensorflow/cc/ops/sendrecv_ops.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/framework/tensor_testutil.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/platform/test_benchmark.h"
#include "tensorflow/core/public/session.h"
#include "tensorflow/tools/graph_transforms/transform_utils.h"

namespace tensorflow {
namespace graph_transforms {

// Declared here so we don't have to expose it in the public header.
Status ShouldIgnoreErrors(const TransformFuncParameters& transform_params,
                          bool* ignore_errors);

namespace {
Status test_empty_graph_transform(const GraphDef& graph_def,
                                  const TransformFuncContext& context,
                                  GraphDef* result) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPStoolsPSgraph_transformsPStransform_graph_testDTcc mht_0(mht_0_v, 210, "", "./tensorflow/tools/graph_transforms/transform_graph_test.cc", "test_empty_graph_transform");

  result->Clear();
  return Status::OK();
}
}  // namespace

REGISTER_GRAPH_TRANSFORM("test_empty_graph_transform",
                         test_empty_graph_transform);

class TransformGraphTest : public ::testing::Test {
 protected:
  void TestConstantFolding() {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPStoolsPSgraph_transformsPStransform_graph_testDTcc mht_1(mht_1_v, 224, "", "./tensorflow/tools/graph_transforms/transform_graph_test.cc", "TestConstantFolding");

    auto root = tensorflow::Scope::NewRootScope();
    using namespace ::tensorflow::ops;  // NOLINT(build/namespaces)

    const int width = 100;

    Tensor a_data(DT_FLOAT, TensorShape({width}));
    test::FillIota<float>(&a_data, 1.0f);
    Output a_const =
        Const(root.WithOpName("a_expect_removed"), Input::Initializer(a_data));

    Tensor b_data(DT_FLOAT, TensorShape({width}));
    test::FillIota<float>(&b_data, 1.0f);
    Output b_const =
        Const(root.WithOpName("b_expect_removed"), Input::Initializer(b_data));

    Output add = Add(root.WithOpName("add_expect_removed"), a_const, b_const);

    Output placeholder =
        Placeholder(root.WithOpName("placeholder_expect_remains"), DT_FLOAT);

    Output mul =
        Mul(root.WithOpName("output_expect_remains"), add, placeholder);

    GraphDef graph_def;
    TF_ASSERT_OK(root.ToGraphDef(&graph_def));
    string graph_def_serialized;
    graph_def.SerializeToString(&graph_def_serialized);
    const string dir = testing::TmpDir();
    const string in_filename_pb = io::JoinPath(dir, "in_graphdef.pb");
    const string out_filename_pb = io::JoinPath(dir, "out_graphdef.pb");
    TF_ASSERT_OK(WriteStringToFile(Env::Default(), in_filename_pb,
                                   graph_def_serialized));

    std::vector<string> args = {"some_binary",
                                "--in_graph=" + in_filename_pb,
                                "--out_graph=" + out_filename_pb,
                                "--inputs=placeholder_expect_remains",
                                "--outputs=output_expect_remains",
                                "--transforms=fold_constants"};
    const int argc = 6;
    EXPECT_EQ(argc, args.size());
    char* argv[argc];
    std::vector<char*> char_strings;
    for (int i = 0; i < argc; ++i) {
      string arg = args[i];
      char* char_string = new char[arg.size() + 1];
      std::copy_n(arg.c_str(), arg.size() + 1, char_string);
      argv[i] = char_string;
      char_strings.push_back(char_string);
    }
    ParseFlagsAndTransformGraph(argc, argv, false);
    for (char* char_string : char_strings) {
      delete[] char_string;
    }

    GraphDef out_graph_def;
    TF_EXPECT_OK(
        ReadBinaryProto(Env::Default(), out_filename_pb, &out_graph_def));

    std::map<string, const NodeDef*> out_node_map;
    graph_transforms::MapNamesToNodes(out_graph_def, &out_node_map);

    for (const NodeDef& node : out_graph_def.node()) {
      const int occurrence_count = out_node_map.count(node.name());
      if (str_util::EndsWith(node.name(), "expect_removed")) {
        EXPECT_EQ(0, occurrence_count) << "node.name()=" << node.name();
      }
      if (str_util::EndsWith(node.name(), "expect_remains")) {
        EXPECT_EQ(1, occurrence_count) << "node.name()=" << node.name();
      }
    }
  }

  void TestTransformRegistration() {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPStoolsPSgraph_transformsPStransform_graph_testDTcc mht_2(mht_2_v, 301, "", "./tensorflow/tools/graph_transforms/transform_graph_test.cc", "TestTransformRegistration");

    auto root = tensorflow::Scope::NewRootScope();
    using namespace ::tensorflow::ops;  // NOLINT(build/namespaces)
    Output placeholder =
        Placeholder(root.WithOpName("placeholder_expect_remains"), DT_FLOAT);
    GraphDef graph_def;
    TF_ASSERT_OK(root.ToGraphDef(&graph_def));
    EXPECT_EQ(1, graph_def.node().size());
    TF_ASSERT_OK(TransformGraph({}, {}, {{"test_empty_graph_transform", {}}},
                                &graph_def));
    EXPECT_EQ(0, graph_def.node().size());

    TF_ASSERT_OK(root.ToGraphDef(&graph_def));
    Status no_such_status =
        TransformGraph({}, {}, {{"test_no_such_transform", {}}}, &graph_def);
    EXPECT_TRUE(absl::StrContains(no_such_status.ToString(), "not recognized"));
  }

  void TestParseTransformParameters() {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPStoolsPSgraph_transformsPStransform_graph_testDTcc mht_3(mht_3_v, 322, "", "./tensorflow/tools/graph_transforms/transform_graph_test.cc", "TestParseTransformParameters");

    TransformParameters params_list;

    TF_EXPECT_OK(ParseTransformParameters("foo", &params_list));
    EXPECT_EQ(1, params_list.size());
    EXPECT_EQ("foo", params_list[0].first);
    EXPECT_TRUE(params_list[0].second.empty());

    TF_EXPECT_OK(ParseTransformParameters("foo bar", &params_list));
    EXPECT_EQ(2, params_list.size());
    EXPECT_EQ("foo", params_list[0].first);
    EXPECT_TRUE(params_list[0].second.empty());
    EXPECT_EQ("bar", params_list[1].first);
    EXPECT_TRUE(params_list[1].second.empty());

    TF_EXPECT_OK(ParseTransformParameters("foo() bar()", &params_list));
    EXPECT_EQ(2, params_list.size());
    EXPECT_EQ("foo", params_list[0].first);
    EXPECT_TRUE(params_list[0].second.empty());
    EXPECT_EQ("bar", params_list[1].first);
    EXPECT_TRUE(params_list[1].second.empty());

    TF_EXPECT_OK(
        ParseTransformParameters("foo(bob_something=sue)", &params_list));
    EXPECT_EQ(1, params_list.size());
    EXPECT_EQ("foo", params_list[0].first);
    EXPECT_EQ(1, params_list[0].second.count("bob_something"));
    EXPECT_EQ(1, params_list[0].second["bob_something"].size());
    EXPECT_EQ("sue", params_list[0].second["bob_something"][0]);

    TF_EXPECT_OK(ParseTransformParameters("bar(a=1, b=2, a=3)", &params_list));
    EXPECT_EQ(1, params_list.size());
    EXPECT_EQ("bar", params_list[0].first);
    EXPECT_EQ(1, params_list[0].second.count("a"));
    EXPECT_EQ(2, params_list[0].second["a"].size());
    EXPECT_EQ("1", params_list[0].second["a"][0]);
    EXPECT_EQ("3", params_list[0].second["a"][1]);
    EXPECT_EQ(1, params_list[0].second.count("b"));
    EXPECT_EQ(1, params_list[0].second["b"].size());
    EXPECT_EQ("2", params_list[0].second["b"][0]);

    TF_EXPECT_OK(ParseTransformParameters("bar(a=\"1\", b=\"1,2,3\", a=3)",
                                          &params_list));
    EXPECT_EQ(1, params_list.size());
    EXPECT_EQ("bar", params_list[0].first);
    EXPECT_EQ(1, params_list[0].second.count("a"));
    EXPECT_EQ(2, params_list[0].second["a"].size());
    EXPECT_EQ("1", params_list[0].second["a"][0]);
    EXPECT_EQ("3", params_list[0].second["a"][1]);
    EXPECT_EQ(1, params_list[0].second.count("b"));
    EXPECT_EQ(1, params_list[0].second["b"].size());
    EXPECT_EQ("1,2,3", params_list[0].second["b"][0]);
  }

  void TestParseEscapedNewline() {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPStoolsPSgraph_transformsPStransform_graph_testDTcc mht_4(mht_4_v, 379, "", "./tensorflow/tools/graph_transforms/transform_graph_test.cc", "TestParseEscapedNewline");

    // This sequence of characters caused an infinite loop in the parser, which
    // is responsible for the hang mentioned in
    // https://github.com/tensorflow/tensorflow/issues/7150
    TransformParameters params_list;
    ParseTransformParameters("\\\n", &params_list).IgnoreError();
    EXPECT_EQ(0, params_list.size());
  }

  void TestParseExtraSpaces() {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPStoolsPSgraph_transformsPStransform_graph_testDTcc mht_5(mht_5_v, 391, "", "./tensorflow/tools/graph_transforms/transform_graph_test.cc", "TestParseExtraSpaces");

    TransformParameters params_list;
    ParseTransformParameters(" ", &params_list).IgnoreError();
    EXPECT_EQ(0, params_list.size());

    TF_EXPECT_OK(ParseTransformParameters("  foo bar \\\n", &params_list));
    EXPECT_EQ(2, params_list.size());
    EXPECT_EQ("foo", params_list[0].first);
    EXPECT_TRUE(params_list[0].second.empty());
    EXPECT_EQ("bar", params_list[1].first);
    EXPECT_TRUE(params_list[1].second.empty());
  }

  void TestShouldIgnoreErrors() {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPStoolsPSgraph_transformsPStransform_graph_testDTcc mht_6(mht_6_v, 407, "", "./tensorflow/tools/graph_transforms/transform_graph_test.cc", "TestShouldIgnoreErrors");

    bool ignore_errors;
    TF_EXPECT_OK(
        ShouldIgnoreErrors({{"ignore_errors", {"true"}}}, &ignore_errors));
    EXPECT_TRUE(ignore_errors);

    TF_EXPECT_OK(
        ShouldIgnoreErrors({{"ignore_errors", {"false"}}}, &ignore_errors));
    EXPECT_FALSE(ignore_errors);

    TF_EXPECT_OK(ShouldIgnoreErrors({}, &ignore_errors));
    EXPECT_FALSE(ignore_errors);

    EXPECT_FALSE(
        ShouldIgnoreErrors({{"ignore_errors", {"foo"}}}, &ignore_errors).ok());
  }
};

TEST_F(TransformGraphTest, TestConstantFolding) { TestConstantFolding(); }

TEST_F(TransformGraphTest, TestTransformRegistration) {
  TestTransformRegistration();
}

TEST_F(TransformGraphTest, TestParseTransformParameters) {
  TestParseTransformParameters();
}

TEST_F(TransformGraphTest, TestParseEscapedNewline) {
  TestParseEscapedNewline();
}

TEST_F(TransformGraphTest, TestShouldIgnoreErrors) { TestShouldIgnoreErrors(); }

}  // namespace graph_transforms
}  // namespace tensorflow
