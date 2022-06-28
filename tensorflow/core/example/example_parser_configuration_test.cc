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
class MHTracer_DTPStensorflowPScorePSexamplePSexample_parser_configuration_testDTcc {
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
   MHTracer_DTPStensorflowPScorePSexamplePSexample_parser_configuration_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSexamplePSexample_parser_configuration_testDTcc() {
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
#include "tensorflow/core/example/example_parser_configuration.h"

#include <memory>

#include "tensorflow/core/framework/attr_value.pb.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/framework/tensor_testutil.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/platform/protobuf.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/public/session_options.h"
#include "tensorflow/core/util/example_proto_helper.h"

namespace tensorflow {
namespace {

void ReadFileToStringOrDie(Env* env, const string& filename, string* output) {
   std::vector<std::string> mht_0_v;
   mht_0_v.push_back("filename: \"" + filename + "\"");
   MHTracer_DTPStensorflowPScorePSexamplePSexample_parser_configuration_testDTcc mht_0(mht_0_v, 202, "", "./tensorflow/core/example/example_parser_configuration_test.cc", "ReadFileToStringOrDie");

  TF_CHECK_OK(ReadFileToString(env, filename, output));
}

std::unique_ptr<Session> CreateSession() {
  SessionOptions options;
  (*options.config.mutable_device_count())["CPU"] = 2;
  return std::unique_ptr<Session>(NewSession(options));
}

class ExtractExampleParserConfigurationTest : public ::testing::Test {
 protected:
  void SetUp() override {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSexamplePSexample_parser_configuration_testDTcc mht_1(mht_1_v, 217, "", "./tensorflow/core/example/example_parser_configuration_test.cc", "SetUp");

    string proto_string;
    string filename =
        io::JoinPath(testing::TensorFlowSrcRoot(),
                     "core/example/testdata/parse_example_graph_def.pbtxt");
    ReadFileToStringOrDie(Env::Default(), filename, &proto_string);
    protobuf::TextFormat::ParseFromString(proto_string, &graph_def_);
    session_ = CreateSession();
    TF_CHECK_OK(session_->Create(graph_def_));
  }

  NodeDef* parse_example_node() {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSexamplePSexample_parser_configuration_testDTcc mht_2(mht_2_v, 231, "", "./tensorflow/core/example/example_parser_configuration_test.cc", "parse_example_node");

    for (auto& node : *graph_def_.mutable_node()) {
      if (node.name() == "ParseExample/ParseExample") {
        return &node;
      }
    }
    return nullptr;
  }

  GraphDef graph_def_;
  std::unique_ptr<Session> session_;
};

TEST_F(ExtractExampleParserConfigurationTest, OpNotFound) {
  std::vector<FixedLenFeature> dense_vec;
  std::vector<VarLenFeature> sparse_vec;
  Status status = ExtractExampleParserConfiguration(
      graph_def_, "BlarseExample/ParseExample", session_.get(), &dense_vec,
      &sparse_vec);

  EXPECT_EQ(status.code(), error::INVALID_ARGUMENT);
}

TEST_F(ExtractExampleParserConfigurationTest, InconsistentAttrNsparse) {
  std::vector<FixedLenFeature> dense_vec;
  std::vector<VarLenFeature> sparse_vec;

  NodeDef* node = parse_example_node();
  auto mutable_attr = node->mutable_attr();
  (*mutable_attr)["Nsparse"].set_i(3);

  Status status = ExtractExampleParserConfiguration(
      graph_def_, "ParseExample/ParseExample", session_.get(), &dense_vec,
      &sparse_vec);

  EXPECT_EQ(status.code(), error::INVALID_ARGUMENT);
}

TEST_F(ExtractExampleParserConfigurationTest, InconsistentAttrNdense) {
  std::vector<FixedLenFeature> dense_vec;
  std::vector<VarLenFeature> sparse_vec;

  NodeDef* node = parse_example_node();
  auto mutable_attr = node->mutable_attr();
  (*mutable_attr)["Ndense"].set_i(2);

  Status status = ExtractExampleParserConfiguration(
      graph_def_, "ParseExample/ParseExample", session_.get(), &dense_vec,
      &sparse_vec);

  EXPECT_EQ(status.code(), error::INVALID_ARGUMENT);
}

TEST_F(ExtractExampleParserConfigurationTest, Basic) {
  std::vector<FixedLenFeature> dense_vec;
  std::vector<VarLenFeature> sparse_vec;
  Status status = ExtractExampleParserConfiguration(
      graph_def_, "ParseExample/ParseExample", session_.get(), &dense_vec,
      &sparse_vec);

  EXPECT_EQ(Status::OK(), status);
  EXPECT_EQ(2, sparse_vec.size());
  EXPECT_EQ(3, dense_vec.size());

  EXPECT_EQ("sf0", sparse_vec[0].key);
  EXPECT_EQ(DT_STRING, sparse_vec[0].dtype);
  EXPECT_EQ("ParseExample/ParseExample:0",
            sparse_vec[0].indices_output_tensor_name);
  EXPECT_EQ("ParseExample/ParseExample:2",
            sparse_vec[0].values_output_tensor_name);
  EXPECT_EQ("ParseExample/ParseExample:4",
            sparse_vec[0].shapes_output_tensor_name);

  EXPECT_EQ("sf1", sparse_vec[1].key);
  EXPECT_EQ(DT_STRING, sparse_vec[1].dtype);
  EXPECT_EQ("ParseExample/ParseExample:1",
            sparse_vec[1].indices_output_tensor_name);
  EXPECT_EQ("ParseExample/ParseExample:3",
            sparse_vec[1].values_output_tensor_name);
  EXPECT_EQ("ParseExample/ParseExample:5",
            sparse_vec[1].shapes_output_tensor_name);

  EXPECT_EQ("x", dense_vec[0].key);
  EXPECT_EQ(DT_FLOAT, dense_vec[0].dtype);
  EXPECT_EQ("ParseExample/ParseExample:6",
            dense_vec[0].values_output_tensor_name);

  EXPECT_EQ("y", dense_vec[1].key);
  EXPECT_EQ(DT_FLOAT, dense_vec[1].dtype);
  EXPECT_EQ("ParseExample/ParseExample:7",
            dense_vec[1].values_output_tensor_name);

  EXPECT_EQ("z", dense_vec[2].key);
  EXPECT_EQ(DT_FLOAT, dense_vec[2].dtype);
  EXPECT_EQ("ParseExample/ParseExample:8",
            dense_vec[2].values_output_tensor_name);
}

static const char kExampleParseConfigurationProto[] = R"( feature_map {
  key: "x"
  value {
    fixed_len_feature {
      dtype: DT_FLOAT
      shape {
        dim {
          size: 1
        }
      }
      default_value {
        dtype: DT_FLOAT
        tensor_shape {
          dim {
            size: 1
          }
        }
        float_val: 33.0
      }
      values_output_tensor_name: "ParseExample/ParseExample:3"
    }
  }
}
feature_map {
  key: "y"
  value {
    var_len_feature {
      dtype: DT_STRING
      values_output_tensor_name: "ParseExample/ParseExample:1"
      indices_output_tensor_name: "ParseExample/ParseExample:0"
      shapes_output_tensor_name: "ParseExample/ParseExample:2"
    }
  }
}
)";

class ExampleParserConfigurationProtoToFeatureVectorsTest
    : public ::testing::Test {
 protected:
  void SetUp() override {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSexamplePSexample_parser_configuration_testDTcc mht_3(mht_3_v, 371, "", "./tensorflow/core/example/example_parser_configuration_test.cc", "SetUp");

    CHECK(protobuf::TextFormat::ParseFromString(kExampleParseConfigurationProto,
                                                &config_proto_));
  }
  ExampleParserConfiguration config_proto_;
};

TEST_F(ExampleParserConfigurationProtoToFeatureVectorsTest, Basic) {
  std::vector<FixedLenFeature> fixed_len_features;
  std::vector<VarLenFeature> var_len_features;
  TF_ASSERT_OK(ExampleParserConfigurationProtoToFeatureVectors(
      config_proto_, &fixed_len_features, &var_len_features));
  ASSERT_EQ(1, fixed_len_features.size());
  ASSERT_EQ(1, var_len_features.size());

  const FixedLenFeature& f = fixed_len_features[0];
  ASSERT_EQ(DT_FLOAT, f.dtype);
  ASSERT_EQ("x", f.key);
  ASSERT_EQ("ParseExample/ParseExample:3", f.values_output_tensor_name);

  TensorShape expected_shape({1});
  ASSERT_EQ(expected_shape.dims(), f.shape.dims());
  ASSERT_EQ(1, f.shape.dim_size(0));

  Tensor expected_default(DT_FLOAT, TensorShape({1}));
  test::FillIota<float>(&expected_default, 33.0);
  test::ExpectTensorEqual<float>(expected_default, f.default_value);

  const VarLenFeature& v = var_len_features[0];
  ASSERT_EQ(DT_STRING, v.dtype);
  ASSERT_EQ("ParseExample/ParseExample:0", v.indices_output_tensor_name);
  ASSERT_EQ("ParseExample/ParseExample:1", v.values_output_tensor_name);
  ASSERT_EQ("ParseExample/ParseExample:2", v.shapes_output_tensor_name);
}

}  // namespace
}  // namespace tensorflow
