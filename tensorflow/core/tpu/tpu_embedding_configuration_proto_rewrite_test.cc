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
class MHTracer_DTPStensorflowPScorePStpuPStpu_embedding_configuration_proto_rewrite_testDTcc {
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
   MHTracer_DTPStensorflowPScorePStpuPStpu_embedding_configuration_proto_rewrite_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePStpuPStpu_embedding_configuration_proto_rewrite_testDTcc() {
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

/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/tpu/tpu_embedding_configuration_proto_rewrite.h"

#include <string>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/strings/string_view.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/casts.h"
#include "tensorflow/core/platform/protobuf.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/platform/status_matchers.h"
#include "tensorflow/core/protobuf/error_codes.pb.h"
#include "tensorflow/core/protobuf/tpu/tpu_embedding_configuration.pb.h"

namespace tensorflow {
namespace {

Status ParseTextProto(absl::string_view text_proto,
                      protobuf::Message* parsed_proto) {
   std::vector<std::string> mht_0_v;
   mht_0_v.push_back("text_proto: \"" + std::string(text_proto.data(), text_proto.size()) + "\"");
   MHTracer_DTPStensorflowPScorePStpuPStpu_embedding_configuration_proto_rewrite_testDTcc mht_0(mht_0_v, 207, "", "./tensorflow/core/tpu/tpu_embedding_configuration_proto_rewrite_test.cc", "ParseTextProto");

  protobuf::TextFormat::Parser parser;
  // Attempt to parse as text.
  protobuf::io::ArrayInputStream input_stream(text_proto.data(),
                                              text_proto.size());
  if (parser.Parse(&input_stream, parsed_proto)) {
    return Status::OK();
  }
  parsed_proto->Clear();
  return errors::InvalidArgument("Could not parse text proto: ", text_proto);
}

TEST(TPUEmbeddingConfigurationProtoRewriteTest, FillFeatureDescriptor) {
  const std::string config_str = R"pb(
    table_descriptor {
      name: "T0"
      vocabulary_size: 35324928
      dimension: 128
      num_features: 3
      optimization_parameters {
        adagrad {}
        learning_rate { constant: 0.1 }
      }
    }
    table_descriptor {
      name: "T1"
      vocabulary_size: 3122176
      dimension: 128
      num_features: 2
      optimization_parameters {
        adagrad {}
        learning_rate { constant: 0.1 }
      }
    }
    mode: TRAINING
    batch_size_per_tensor_core: 256
    num_hosts: 16
    num_tensor_cores: 128
    pipeline_execution_with_tensor_core: true
  )pb";
  tpu::TPUEmbeddingConfiguration tpu_embedding_config;
  TF_ASSERT_OK(ParseTextProto(config_str, &tpu_embedding_config));
  TF_ASSERT_OK(
      PopulateMissingFieldsInTPUEmbeddingConfig(&tpu_embedding_config));

  EXPECT_EQ(tpu_embedding_config.feature_descriptor_size(), 2);
  const auto& feature_0 = tpu_embedding_config.feature_descriptor(0);
  EXPECT_EQ(feature_0.table_id(), 0);
  EXPECT_THAT(feature_0.input_shape(), ::testing::ElementsAre(256 * 3));
  const auto& feature_1 = tpu_embedding_config.feature_descriptor(1);
  EXPECT_EQ(feature_1.table_id(), 1);
  EXPECT_THAT(feature_1.input_shape(), ::testing::ElementsAre(256 * 2));
}

TEST(TPUEmbeddingConfigurationProtoRewriteTest, FillBatchSizeAndNumFeatures) {
  const std::string config_str = R"pb(
    table_descriptor {
      name: "T0"
      vocabulary_size: 35324928
      dimension: 128
      optimization_parameters {
        adagrad {}
        learning_rate { constant: 0.1 }
      }
    }
    table_descriptor {
      name: "T1"
      vocabulary_size: 3122176
      dimension: 128
      optimization_parameters {
        adagrad {}
        learning_rate { constant: 0.1 }
      }
    }
    feature_descriptor {
      name: "F0"
      table_id: 0
      input_shape: [ 100, 5 ]
    }
    feature_descriptor {
      name: "F1"
      table_id: 1
      input_shape: [ 200, 5, 20 ]
    }
    feature_descriptor {
      name: "F2"
      table_id: 0
      input_shape: [ 50 ]
    }
    feature_descriptor {
      name: "F3"
      table_id: 0
      input_shape: [ 100, 2, 3 ]
    }
    mode: TRAINING
    num_hosts: 16
    num_tensor_cores: 128
    pipeline_execution_with_tensor_core: true
  )pb";
  tpu::TPUEmbeddingConfiguration tpu_embedding_config;
  TF_ASSERT_OK(ParseTextProto(config_str, &tpu_embedding_config));
  TF_ASSERT_OK(
      PopulateMissingFieldsInTPUEmbeddingConfig(&tpu_embedding_config));

  EXPECT_EQ(tpu_embedding_config.batch_size_per_tensor_core(), 50);
  const auto& table_0 = tpu_embedding_config.table_descriptor(0);
  EXPECT_EQ(table_0.num_features(), 23);
  const auto& table_1 = tpu_embedding_config.table_descriptor(1);
  EXPECT_EQ(table_1.num_features(), 400);
}

TEST(TPUEmbeddingConfigurationProtoRewriteTest, InvalidBatchSizeOrNumFeatures) {
  const std::string config_str = R"pb(
    table_descriptor {
      name: "T0"
      vocabulary_size: 35324928
      dimension: 128
      num_features: 3
      optimization_parameters {
        adagrad {}
        learning_rate { constant: 0.1 }
      }
    }
    feature_descriptor {
      table_id: 0
      input_shape: [ 768 ]
    }
    mode: TRAINING
    batch_size_per_tensor_core: 256
    num_hosts: 16
    num_tensor_cores: 128
    pipeline_execution_with_tensor_core: true
  )pb";
  tpu::TPUEmbeddingConfiguration tpu_embedding_config;
  TF_ASSERT_OK(ParseTextProto(config_str, &tpu_embedding_config));
  {
    tpu::TPUEmbeddingConfiguration invalid_config = tpu_embedding_config;
    invalid_config.clear_feature_descriptor();
    invalid_config.clear_batch_size_per_tensor_core();
    EXPECT_THAT(
        PopulateMissingFieldsInTPUEmbeddingConfig(&invalid_config),
        tensorflow::testing::StatusIs(
            tensorflow::error::INVALID_ARGUMENT,
            ::testing::HasSubstr("Invalid batch_size_per_tensor_core")));
  }
  {
    tpu::TPUEmbeddingConfiguration invalid_config = tpu_embedding_config;
    invalid_config.clear_feature_descriptor();
    invalid_config.mutable_table_descriptor(0)->clear_num_features();
    EXPECT_THAT(PopulateMissingFieldsInTPUEmbeddingConfig(&invalid_config),
                tensorflow::testing::StatusIs(
                    tensorflow::error::INVALID_ARGUMENT,
                    ::testing::HasSubstr("Invalid num_features")));
  }
  {
    tpu::TPUEmbeddingConfiguration invalid_config = tpu_embedding_config;
    EXPECT_THAT(
        PopulateMissingFieldsInTPUEmbeddingConfig(&invalid_config),
        tensorflow::testing::StatusIs(
            tensorflow::error::INVALID_ARGUMENT,
            ::testing::HasSubstr(
                "The batch_size_per_tensor_core field must NOT be populated")));
  }
  {
    tpu::TPUEmbeddingConfiguration invalid_config = tpu_embedding_config;
    invalid_config.clear_batch_size_per_tensor_core();
    EXPECT_THAT(PopulateMissingFieldsInTPUEmbeddingConfig(&invalid_config),
                tensorflow::testing::StatusIs(
                    tensorflow::error::INVALID_ARGUMENT,
                    ::testing::HasSubstr("The TableDescriptor.num_features "
                                         "field must NOT be populated")));
  }
}

TEST(TPUEmbeddingConfigurationProtoRewriteTest, InvalidFeatureDescriptor) {
  const std::string config_str = R"pb(
    table_descriptor {
      name: "T0"
      vocabulary_size: 35324928
      dimension: 128
      optimization_parameters {
        adagrad {}
        learning_rate { constant: 0.1 }
      }
    }
    table_descriptor {
      name: "T1"
      vocabulary_size: 3122176
      dimension: 128
      optimization_parameters {
        adagrad {}
        learning_rate { constant: 0.1 }
      }
    }
    feature_descriptor {
      name: "F1"
      table_id: 0
      input_shape: [ 768 ]
    }
    feature_descriptor {
      name: "F2"
      table_id: 1
      input_shape: [ 512 ]
    }
    mode: TRAINING
    num_hosts: 16
    num_tensor_cores: 128
    pipeline_execution_with_tensor_core: true
  )pb";
  tpu::TPUEmbeddingConfiguration tpu_embedding_config;
  TF_ASSERT_OK(ParseTextProto(config_str, &tpu_embedding_config));
  {
    tpu::TPUEmbeddingConfiguration invalid_config = tpu_embedding_config;
    invalid_config.mutable_feature_descriptor(0)->set_table_id(2);
    EXPECT_THAT(PopulateMissingFieldsInTPUEmbeddingConfig(&invalid_config),
                tensorflow::testing::StatusIs(
                    tensorflow::error::INVALID_ARGUMENT,
                    ::testing::HasSubstr("Invalid table_id")));
  }
  {
    tpu::TPUEmbeddingConfiguration invalid_config = tpu_embedding_config;
    invalid_config.mutable_feature_descriptor(0)->clear_input_shape();
    EXPECT_THAT(
        PopulateMissingFieldsInTPUEmbeddingConfig(&invalid_config),
        tensorflow::testing::StatusIs(
            tensorflow::error::INVALID_ARGUMENT,
            ::testing::HasSubstr("The input_shape field cannot be empty")));
  }
  {
    tpu::TPUEmbeddingConfiguration invalid_config = tpu_embedding_config;
    invalid_config.mutable_feature_descriptor(0)->set_input_shape(0, -5);
    EXPECT_THAT(
        PopulateMissingFieldsInTPUEmbeddingConfig(&invalid_config),
        tensorflow::testing::StatusIs(
            tensorflow::error::INVALID_ARGUMENT,
            ::testing::HasSubstr("The input_shape dimension sizes must all")));
  }
  {
    tpu::TPUEmbeddingConfiguration invalid_config = tpu_embedding_config;
    invalid_config.mutable_feature_descriptor(1)->set_table_id(0);
    EXPECT_THAT(PopulateMissingFieldsInTPUEmbeddingConfig(&invalid_config),
                tensorflow::testing::StatusIs(
                    tensorflow::error::INVALID_ARGUMENT,
                    ::testing::HasSubstr(
                        "No feature_descriptor fields found for table: T1")));
  }
}

}  // namespace
}  // namespace tensorflow
