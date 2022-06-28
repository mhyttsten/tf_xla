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
class MHTracer_DTPStensorflowPSccPSsaved_modelPSsaved_model_bundle_testDTcc {
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
   MHTracer_DTPStensorflowPSccPSsaved_modelPSsaved_model_bundle_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSccPSsaved_modelPSsaved_model_bundle_testDTcc() {
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

#include "tensorflow/cc/saved_model/constants.h"
#include "tensorflow/cc/saved_model/loader.h"
#include "tensorflow/cc/saved_model/metrics.h"
#include "tensorflow/cc/saved_model/reader.h"
#include "tensorflow/cc/saved_model/signature_constants.h"
#include "tensorflow/cc/saved_model/tag_constants.h"
#include "tensorflow/core/example/example.pb.h"
#include "tensorflow/core/example/feature.pb.h"
#include "tensorflow/core/framework/tensor_testutil.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/protobuf/meta_graph.pb.h"

namespace tensorflow {
namespace {

constexpr char kTestDataPbTxt[] =
    "cc/saved_model/testdata/half_plus_two_pbtxt/00000123";
constexpr char kTestDataMainOp[] =
    "cc/saved_model/testdata/half_plus_two_main_op/00000123";
constexpr char kTestDataSharded[] =
    "cc/saved_model/testdata/half_plus_two/00000123";
constexpr char kTestDataInitOpV2[] =
    "cc/saved_model/testdata/half_plus_two_v2/00000123";
constexpr char kTestDataV2DebugInfo[] =
    "cc/saved_model/testdata/x_plus_y_v2_debuginfo";
constexpr char kTestFuzzGeneratedNegativeShape[] =
    "cc/saved_model/testdata/fuzz_generated/negative_shape";
constexpr char kTestFuzzGeneratedConstWithNoValue[] =
    "cc/saved_model/testdata/fuzz_generated/const_with_no_value";
constexpr char kTestFuzzGeneratedBadNodeAttr[] =
    "cc/saved_model/testdata/fuzz_generated/bad_node_attr";
constexpr char kTestCyclicModule[] = "cc/saved_model/testdata/CyclicModule";
constexpr char kTestSimpleV1Model[] = "cc/saved_model/testdata/SimpleV1Model";

class LoaderTest : public ::testing::Test {
 protected:
  LoaderTest() {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSccPSsaved_modelPSsaved_model_bundle_testDTcc mht_0(mht_0_v, 225, "", "./tensorflow/cc/saved_model/saved_model_bundle_test.cc", "LoaderTest");
}

  string MakeSerializedExample(float x) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSccPSsaved_modelPSsaved_model_bundle_testDTcc mht_1(mht_1_v, 230, "", "./tensorflow/cc/saved_model/saved_model_bundle_test.cc", "MakeSerializedExample");

    tensorflow::Example example;
    auto* feature_map = example.mutable_features()->mutable_feature();
    (*feature_map)["x"].mutable_float_list()->add_value(x);
    return example.SerializeAsString();
  }

  void ValidateAssets(const string& export_dir,
                      const SavedModelBundle& bundle) {
   std::vector<std::string> mht_2_v;
   mht_2_v.push_back("export_dir: \"" + export_dir + "\"");
   MHTracer_DTPStensorflowPSccPSsaved_modelPSsaved_model_bundle_testDTcc mht_2(mht_2_v, 242, "", "./tensorflow/cc/saved_model/saved_model_bundle_test.cc", "ValidateAssets");

    const string asset_directory =
        io::JoinPath(export_dir, kSavedModelAssetsDirectory);
    const string asset_filename = "foo.txt";
    const string asset_filepath = io::JoinPath(asset_directory, asset_filename);
    TF_EXPECT_OK(Env::Default()->FileExists(asset_filepath));

    std::vector<Tensor> path_outputs;
    TF_ASSERT_OK(
        bundle.session->Run({}, {"filename_tensor:0"}, {}, &path_outputs));
    ASSERT_EQ(1, path_outputs.size());

    test::ExpectTensorEqual<tstring>(
        test::AsTensor<tstring>({"foo.txt"}, TensorShape({})), path_outputs[0]);
  }

  void CheckSavedModelBundle(const string& export_dir,
                             const SavedModelBundle& bundle) {
   std::vector<std::string> mht_3_v;
   mht_3_v.push_back("export_dir: \"" + export_dir + "\"");
   MHTracer_DTPStensorflowPSccPSsaved_modelPSsaved_model_bundle_testDTcc mht_3(mht_3_v, 263, "", "./tensorflow/cc/saved_model/saved_model_bundle_test.cc", "CheckSavedModelBundle");

    ValidateAssets(export_dir, bundle);
    // Retrieve the regression signature from meta graph def.
    const auto& signature_def = bundle.GetSignatures().at("regress_x_to_y");

    const string input_name = signature_def.inputs().at(kRegressInputs).name();
    const string output_name =
        signature_def.outputs().at(kRegressOutputs).name();

    std::vector<tstring> serialized_examples;
    for (float x : {0, 1, 2, 3}) {
      serialized_examples.push_back(MakeSerializedExample(x));
    }

    // Validate the half plus two behavior.
    Tensor input =
        test::AsTensor<tstring>(serialized_examples, TensorShape({4}));
    std::vector<Tensor> outputs;
    TF_ASSERT_OK(bundle.session->Run({{input_name, input}}, {output_name}, {},
                                     &outputs));
    ASSERT_EQ(outputs.size(), 1);
    test::ExpectTensorEqual<float>(
        outputs[0],
        test::AsTensor<float>({2, 2.5, 3, 3.5}, TensorShape({4, 1})));
  }
};

// Test for resource leaks related to TensorFlow session closing requirements
// when loading and unloading large numbers of SavedModelBundles.
// TODO(sukritiramesh): Increase run iterations and move outside of the test
// suite.
TEST_F(LoaderTest, ResourceLeakTest) {
  SavedModelBundle bundle;
  SessionOptions session_options;
  RunOptions run_options;

  const string export_dir =
      io::JoinPath(testing::TensorFlowSrcRoot(), kTestDataSharded);
  for (int i = 0; i < 100; ++i) {
    TF_ASSERT_OK(LoadSavedModel(session_options, run_options, export_dir,
                                {kSavedModelTagServe}, &bundle));
    CheckSavedModelBundle(export_dir, bundle);
  }
}

TEST_F(LoaderTest, TagMatch) {
  SavedModelBundle bundle;
  SessionOptions session_options;
  RunOptions run_options;

  const string export_dir =
      io::JoinPath(testing::TensorFlowSrcRoot(), kTestDataSharded);
  TF_ASSERT_OK(LoadSavedModel(session_options, run_options, export_dir,
                              {kSavedModelTagServe}, &bundle));
  CheckSavedModelBundle(export_dir, bundle);
}

TEST_F(LoaderTest, ReadMetaGraphFromSavedModel) {
  SavedModelBundle bundle;
  SessionOptions session_options;
  RunOptions run_options;

  const string export_dir =
      io::JoinPath(testing::TensorFlowSrcRoot(), kTestDataSharded);
  TF_ASSERT_OK(LoadSavedModel(session_options, run_options, export_dir,
                              {kSavedModelTagServe}, &bundle));
  MetaGraphDef actual_metagraph;
  TF_ASSERT_OK(ReadMetaGraphDefFromSavedModel(export_dir, {kSavedModelTagServe},
                                              &actual_metagraph));
  EXPECT_EQ(actual_metagraph.DebugString(),
            bundle.meta_graph_def.DebugString());
}

TEST_F(LoaderTest, RestoreSession) {
  SavedModelBundle bundle;
  SessionOptions session_options;
  RunOptions run_options;

  const string export_dir =
      io::JoinPath(testing::TensorFlowSrcRoot(), kTestDataSharded);
  TF_ASSERT_OK(LoadSavedModel(session_options, run_options, export_dir,
                              {kSavedModelTagServe}, &bundle));

  SavedModelBundle actual_bundle;
  const std::unordered_set<std::string> tags = {kSavedModelTagServe};
  TF_ASSERT_OK(ReadMetaGraphDefFromSavedModel(export_dir, tags,
                                              &actual_bundle.meta_graph_def));
  TF_ASSERT_OK(LoadMetagraphIntoSession(
      session_options, actual_bundle.meta_graph_def, &actual_bundle.session));
  TF_ASSERT_OK(RestoreSession(run_options, actual_bundle.meta_graph_def,
                              export_dir, &actual_bundle.session));
  CheckSavedModelBundle(export_dir, actual_bundle);
}

TEST_F(LoaderTest, NoTagMatch) {
  SavedModelBundle bundle;
  RunOptions run_options;
  SessionOptions session_options;

  const string export_dir =
      io::JoinPath(testing::TensorFlowSrcRoot(), kTestDataSharded);
  Status st = LoadSavedModel(session_options, run_options, export_dir,
                             {"missing-tag"}, &bundle);
  EXPECT_FALSE(st.ok());
  EXPECT_TRUE(absl::StrContains(
      st.error_message(),
      "Could not find meta graph def matching supplied tags: { missing-tag }"))
      << st.error_message();
}

TEST_F(LoaderTest, NoTagMatchMultiple) {
  SavedModelBundle bundle;
  RunOptions run_options;
  SessionOptions session_options;

  const string export_dir =
      io::JoinPath(testing::TensorFlowSrcRoot(), kTestDataSharded);
  Status st = LoadSavedModel(session_options, run_options, export_dir,
                             {kSavedModelTagServe, "missing-tag"}, &bundle);
  EXPECT_FALSE(st.ok());
  EXPECT_TRUE(absl::StrContains(
      st.error_message(),
      "Could not find meta graph def matching supplied tags: "))
      << st.error_message();
}

TEST_F(LoaderTest, SessionCreationFailure) {
  SavedModelBundle bundle;
  // Use invalid SessionOptions to cause session creation to fail.  Default
  // options work, so provide an invalid value for the target field.
  SessionOptions session_options;
  constexpr char kInvalidTarget[] = "invalid target";
  session_options.target = kInvalidTarget;
  RunOptions run_options;

  const string export_dir =
      io::JoinPath(testing::TensorFlowSrcRoot(), kTestDataSharded);
  Status st = LoadSavedModel(session_options, run_options, export_dir,
                             {kSavedModelTagServe}, &bundle);
  EXPECT_FALSE(st.ok());
  EXPECT_TRUE(absl::StrContains(st.error_message(), kInvalidTarget))
      << st.error_message();
}

TEST_F(LoaderTest, PbtxtFormat) {
  SavedModelBundle bundle;
  SessionOptions session_options;
  RunOptions run_options;

  const string export_dir =
      io::JoinPath(testing::TensorFlowSrcRoot(), kTestDataPbTxt);
  TF_ASSERT_OK(LoadSavedModel(session_options, run_options, export_dir,
                              {kSavedModelTagServe}, &bundle));
  CheckSavedModelBundle(export_dir, bundle);
}

TEST_F(LoaderTest, MainOpFormat) {
  SavedModelBundle bundle;
  SessionOptions session_options;
  RunOptions run_options;

  const string export_dir =
      io::JoinPath(testing::TensorFlowSrcRoot(), kTestDataMainOp);
  TF_ASSERT_OK(LoadSavedModel(session_options, run_options, export_dir,
                              {kSavedModelTagServe}, &bundle));
  CheckSavedModelBundle(export_dir, bundle);
}

TEST_F(LoaderTest, InvalidExportPath) {
  SavedModelBundle bundle;
  RunOptions run_options;
  SessionOptions session_options;

  const string export_dir =
      io::JoinPath(testing::TensorFlowSrcRoot(), "missing-path");
  Status st = LoadSavedModel(session_options, run_options, export_dir,
                             {kSavedModelTagServe}, &bundle);
  EXPECT_FALSE(st.ok());
}

TEST_F(LoaderTest, MaybeSavedModelDirectory) {
  // Valid SavedModel directory.
  const string export_dir =
      io::JoinPath(testing::TensorFlowSrcRoot(), kTestDataSharded);
  EXPECT_TRUE(MaybeSavedModelDirectory(export_dir));

  // Directory that does not exist.
  const string missing_export_dir =
      io::JoinPath(testing::TensorFlowSrcRoot(), "missing-path");
  EXPECT_FALSE(MaybeSavedModelDirectory(missing_export_dir));

  // Directory that exists but is an invalid SavedModel location.
  const string invalid_export_dir =
      io::JoinPath(testing::TensorFlowSrcRoot(), "cc/saved_model");
  EXPECT_FALSE(MaybeSavedModelDirectory(invalid_export_dir));
}

TEST_F(LoaderTest, SavedModelInitOpV2Format) {
  SavedModelBundle bundle;
  SessionOptions session_options;
  RunOptions run_options;

  const string export_dir =
      io::JoinPath(testing::TensorFlowSrcRoot(), kTestDataInitOpV2);
  TF_ASSERT_OK(LoadSavedModel(session_options, run_options, export_dir,
                              {kSavedModelTagServe}, &bundle));
  CheckSavedModelBundle(export_dir, bundle);
}

TEST_F(LoaderTest, SavedModelV2DebugInfo) {
  SavedModelBundle bundle;
  SessionOptions session_options;
  RunOptions run_options;

  const string export_dir =
      io::JoinPath(testing::TensorFlowSrcRoot(), kTestDataV2DebugInfo);
  TF_ASSERT_OK(LoadSavedModel(session_options, run_options, export_dir,
                              {kSavedModelTagServe}, &bundle));

  // This SavedModel has debug info, so we should have loaded it.
  EXPECT_NE(bundle.debug_info.get(), nullptr);
}

TEST_F(LoaderTest, NegativeShapeDimension) {
  SavedModelBundle bundle;
  RunOptions run_options;
  SessionOptions session_options;

  const string export_dir = io::JoinPath(testing::TensorFlowSrcRoot(),
                                         kTestFuzzGeneratedNegativeShape);
  Status st = LoadSavedModel(session_options, run_options, export_dir,
                             {kSavedModelTagServe}, &bundle);
  EXPECT_FALSE(st.ok());
  EXPECT_NE(
      st.error_message().find("initializes from a tensor with -1 elements"),
      std::string::npos);
}

TEST_F(LoaderTest, ConstNoValue) {
  SavedModelBundle bundle;
  RunOptions run_options;
  SessionOptions session_options;

  const string export_dir = io::JoinPath(testing::TensorFlowSrcRoot(),
                                         kTestFuzzGeneratedConstWithNoValue);
  Status st = LoadSavedModel(session_options, run_options, export_dir,
                             {kSavedModelTagServe}, &bundle);
  EXPECT_FALSE(st.ok());
  EXPECT_NE(
      st.error_message().find("constant tensor but no value has been provided"),
      std::string::npos);
}

TEST_F(LoaderTest, BadNodeAttr) {
  SavedModelBundle bundle;
  RunOptions run_options;
  SessionOptions session_options;

  const string export_dir =
      io::JoinPath(testing::TensorFlowSrcRoot(), kTestFuzzGeneratedBadNodeAttr);
  Status st = LoadSavedModel(session_options, run_options, export_dir,
                             {kSavedModelTagServe}, &bundle);
  EXPECT_FALSE(st.ok());
  EXPECT_NE(
      st.error_message().find("constant tensor but no value has been provided"),
      std::string::npos);
}

TEST_F(LoaderTest, UpdateMetricsV2) {
  SavedModelBundle bundle;
  SessionOptions session_options;
  RunOptions run_options;
  const string kCCLoadLabel = "cc_load";

  const int read_count_v2 = metrics::SavedModelRead("2").value();
  const int api_count = metrics::SavedModelReadApi(kCCLoadLabel).value();
  const string export_dir =
      io::JoinPath(testing::TensorFlowSrcRoot(), kTestCyclicModule);
  TF_ASSERT_OK(LoadSavedModel(session_options, run_options, export_dir,
                              {kSavedModelTagServe}, &bundle));

  EXPECT_EQ(metrics::SavedModelRead("2").value(), read_count_v2 + 1);
  EXPECT_EQ(metrics::SavedModelReadApi(kCCLoadLabel).value(), api_count + 1);
}

TEST_F(LoaderTest, UpdateMetricsV1) {
  SavedModelBundle bundle;
  SessionOptions session_options;
  RunOptions run_options;
  const string kCCLoadLabel = "cc_load";

  const int read_count_v1 = metrics::SavedModelRead("1").value();
  const int read_count_v2 = metrics::SavedModelRead("2").value();

  const int api_count = metrics::SavedModelReadApi(kCCLoadLabel).value();
  const string export_dir =
      io::JoinPath(testing::TensorFlowSrcRoot(), kTestSimpleV1Model);
  TF_ASSERT_OK(LoadSavedModel(session_options, run_options, export_dir,
                              {kSavedModelTagServe}, &bundle));

  EXPECT_EQ(metrics::SavedModelRead("1").value(), read_count_v1 + 1);
  EXPECT_EQ(metrics::SavedModelRead("2").value(), read_count_v2);
  EXPECT_EQ(metrics::SavedModelReadApi(kCCLoadLabel).value(), api_count + 1);
}

}  // namespace
}  // namespace tensorflow
