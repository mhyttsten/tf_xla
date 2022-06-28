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
class MHTracer_DTPStensorflowPScorePSutilPSexample_proto_helper_testDTcc {
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
   MHTracer_DTPStensorflowPScorePSutilPSexample_proto_helper_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSutilPSexample_proto_helper_testDTcc() {
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
#include "tensorflow/core/util/example_proto_helper.h"

#include "tensorflow/core/example/example.pb.h"
#include "tensorflow/core/example/feature.pb.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {
namespace {

constexpr char kDenseInt64Key[] = "dense_int64";
constexpr char kDenseFloatKey[] = "dense_float";
constexpr char kDenseStringKey[] = "dense_string";

constexpr char kSparseInt64Key[] = "sparse_int64";
constexpr char kSparseFloatKey[] = "sparse_float";
constexpr char kSparseStringKey[] = "sparse_string";

// Note that this method is also extensively tested by the python unit test:
// tensorflow/python/kernel_tests/parsing_ops_test.py
class SingleExampleProtoToTensorsTest : public ::testing::Test {
 protected:
  void SetUp() override {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSutilPSexample_proto_helper_testDTcc mht_0(mht_0_v, 206, "", "./tensorflow/core/util/example_proto_helper_test.cc", "SetUp");

    // Setup dense feature configuration.
    FixedLenFeature int64_dense_config;
    int64_dense_config.key = kDenseInt64Key;
    int64_dense_config.dtype = DT_INT64;
    int64_dense_config.shape = TensorShape({1});
    int64_dense_config.default_value = Tensor(DT_INT64, TensorShape({1}));
    int64_dense_config.default_value.scalar<int64_t>()() = 0;
    dense_vec_.push_back(int64_dense_config);

    FixedLenFeature float_dense_config;
    float_dense_config.key = kDenseFloatKey;
    float_dense_config.dtype = DT_FLOAT;
    float_dense_config.shape = TensorShape({1});
    float_dense_config.default_value = Tensor(DT_FLOAT, TensorShape({1}));
    float_dense_config.default_value.scalar<float>()() = 0.0;
    dense_vec_.push_back(float_dense_config);

    FixedLenFeature string_dense_config;
    string_dense_config.key = kDenseStringKey;
    string_dense_config.dtype = DT_STRING;
    string_dense_config.shape = TensorShape({1});
    string_dense_config.default_value = Tensor(DT_STRING, TensorShape({1}));
    string_dense_config.default_value.scalar<tstring>()() = "default";
    dense_vec_.push_back(string_dense_config);

    // Setup sparse feature configuration.
    VarLenFeature int64_sparse_config;
    int64_sparse_config.key = kSparseInt64Key;
    int64_sparse_config.dtype = DT_INT64;
    sparse_vec_.push_back(int64_sparse_config);

    VarLenFeature float_sparse_config;
    float_sparse_config.key = kSparseFloatKey;
    float_sparse_config.dtype = DT_FLOAT;
    sparse_vec_.push_back(float_sparse_config);

    VarLenFeature string_sparse_config;
    string_sparse_config.key = kSparseStringKey;
    string_sparse_config.dtype = DT_STRING;
    sparse_vec_.push_back(string_sparse_config);
  }

  std::vector<FixedLenFeature> dense_vec_;
  std::vector<VarLenFeature> sparse_vec_;
};

TEST_F(SingleExampleProtoToTensorsTest, SparseOnlyTrivial) {
  Example ex;
  // Set up a feature for each of our supported types.
  (*ex.mutable_features()->mutable_feature())[kSparseInt64Key]
      .mutable_int64_list()
      ->add_value(42);
  (*ex.mutable_features()->mutable_feature())[kSparseFloatKey]
      .mutable_float_list()
      ->add_value(4.2);
  (*ex.mutable_features()->mutable_feature())[kSparseStringKey]
      .mutable_bytes_list()
      ->add_value("forty-two");

  std::vector<Tensor*> output_dense_values(0);
  std::vector<std::vector<Tensor>> output_sparse_values_tmp(3);
  for (int i = 0; i < 3; ++i) {
    output_sparse_values_tmp[i] = std::vector<Tensor>(1);
  }

  std::vector<FixedLenFeature> empty_dense_vec;
  TF_EXPECT_OK(SingleExampleProtoToTensors(ex, "", 0, empty_dense_vec,
                                           sparse_vec_, &output_dense_values,
                                           &output_sparse_values_tmp));

  const std::vector<Tensor>& int64_tensor_vec = output_sparse_values_tmp[0];
  EXPECT_EQ(1, int64_tensor_vec.size());
  EXPECT_EQ(42, int64_tensor_vec[0].vec<int64_t>()(0));

  const std::vector<Tensor>& float_tensor_vec = output_sparse_values_tmp[1];
  EXPECT_EQ(1, float_tensor_vec.size());
  EXPECT_NEAR(4.2, float_tensor_vec[0].vec<float>()(0), 0.001);

  const std::vector<Tensor>& string_tensor_vec = output_sparse_values_tmp[2];
  EXPECT_EQ(1, string_tensor_vec.size());
  EXPECT_EQ("forty-two", string_tensor_vec[0].vec<tstring>()(0));
}

TEST_F(SingleExampleProtoToTensorsTest, SparseOnlyEmpty) {
  Example empty;
  std::vector<Tensor*> output_dense_values(0);
  std::vector<std::vector<Tensor>> output_sparse_values_tmp(3);
  for (int i = 0; i < 3; ++i) {
    output_sparse_values_tmp[i] = std::vector<Tensor>(1);
  }

  std::vector<FixedLenFeature> empty_dense_vec;
  TF_EXPECT_OK(SingleExampleProtoToTensors(empty, "", 0, empty_dense_vec,
                                           sparse_vec_, &output_dense_values,
                                           &output_sparse_values_tmp));

  // Each feature will still have a tensor vector, however the tensor
  // in the vector will be empty.
  const std::vector<Tensor>& int64_tensor_vec = output_sparse_values_tmp[0];
  EXPECT_EQ(1, int64_tensor_vec.size());
  EXPECT_EQ(0, int64_tensor_vec[0].vec<int64_t>().size());

  const std::vector<Tensor>& float_tensor_vec = output_sparse_values_tmp[1];
  EXPECT_EQ(1, float_tensor_vec.size());
  EXPECT_EQ(0, float_tensor_vec[0].vec<float>().size());

  const std::vector<Tensor>& string_tensor_vec = output_sparse_values_tmp[2];
  EXPECT_EQ(1, string_tensor_vec.size());
  EXPECT_EQ(0, string_tensor_vec[0].vec<tstring>().size());
}

TEST_F(SingleExampleProtoToTensorsTest, DenseOnlyTrivial) {
  Example ex;
  // Set up a feature for each of our supported types.
  (*ex.mutable_features()->mutable_feature())[kDenseInt64Key]
      .mutable_int64_list()
      ->add_value(42);
  (*ex.mutable_features()->mutable_feature())[kDenseFloatKey]
      .mutable_float_list()
      ->add_value(4.2);
  (*ex.mutable_features()->mutable_feature())[kDenseStringKey]
      .mutable_bytes_list()
      ->add_value("forty-two");

  std::vector<Tensor*> output_dense_values(3);
  Tensor int64_dense_output(DT_INT64, TensorShape({1, 1}));
  output_dense_values[0] = &int64_dense_output;

  Tensor float_dense_output(DT_FLOAT, TensorShape({1, 1}));
  output_dense_values[1] = &float_dense_output;

  Tensor str_dense_output(DT_STRING, TensorShape({1, 1}));
  output_dense_values[2] = &str_dense_output;

  std::vector<VarLenFeature> empty_sparse_vec;
  std::vector<std::vector<Tensor>> output_sparse_values_tmp;
  TF_EXPECT_OK(SingleExampleProtoToTensors(
      ex, "", 0, dense_vec_, empty_sparse_vec, &output_dense_values,
      &output_sparse_values_tmp));
  EXPECT_TRUE(output_sparse_values_tmp.empty());

  EXPECT_EQ(1, int64_dense_output.matrix<int64_t>().size());
  EXPECT_EQ(42, int64_dense_output.matrix<int64_t>()(0, 0));

  EXPECT_EQ(1, float_dense_output.matrix<float>().size());
  EXPECT_NEAR(4.2, float_dense_output.matrix<float>()(0, 0), 0.001);

  EXPECT_EQ(1, str_dense_output.matrix<tstring>().size());
  EXPECT_EQ("forty-two", str_dense_output.matrix<tstring>()(0, 0));
}

TEST_F(SingleExampleProtoToTensorsTest, DenseOnlyDefaults) {
  std::vector<Tensor*> output_dense_values(3);
  Tensor int64_dense_output(DT_INT64, TensorShape({1, 1}));
  output_dense_values[0] = &int64_dense_output;

  Tensor float_dense_output(DT_FLOAT, TensorShape({1, 1}));
  output_dense_values[1] = &float_dense_output;

  Tensor str_dense_output(DT_STRING, TensorShape({1, 1}));
  output_dense_values[2] = &str_dense_output;

  Example empty;

  std::vector<VarLenFeature> empty_sparse_vec;
  std::vector<std::vector<Tensor>> output_sparse_values_tmp;
  TF_EXPECT_OK(SingleExampleProtoToTensors(
      empty, "", 0, dense_vec_, empty_sparse_vec, &output_dense_values,
      &output_sparse_values_tmp));

  EXPECT_EQ(1, int64_dense_output.matrix<int64_t>().size());
  EXPECT_EQ(0, int64_dense_output.matrix<int64_t>()(0, 0));

  EXPECT_EQ(1, float_dense_output.matrix<float>().size());
  EXPECT_NEAR(0.0, float_dense_output.matrix<float>()(0, 0), 0.001);

  EXPECT_EQ(1, str_dense_output.matrix<tstring>().size());
  EXPECT_EQ("default", str_dense_output.matrix<tstring>()(0, 0));
}

}  // namespace
}  // namespace tensorflow
