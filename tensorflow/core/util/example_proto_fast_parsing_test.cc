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
class MHTracer_DTPStensorflowPScorePSutilPSexample_proto_fast_parsing_testDTcc {
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
   MHTracer_DTPStensorflowPScorePSutilPSexample_proto_fast_parsing_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSutilPSexample_proto_fast_parsing_testDTcc() {
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

#include <utility>

#include "tensorflow/core/util/example_proto_fast_parsing.h"

#include "tensorflow/core/example/example.pb.h"
#include "tensorflow/core/example/feature.pb.h"
#include "tensorflow/core/lib/random/philox_random.h"
#include "tensorflow/core/lib/random/simple_philox.h"
#include "tensorflow/core/platform/protobuf.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/platform/test_benchmark.h"
#include "tensorflow/core/util/example_proto_fast_parsing_test.pb.h"

namespace tensorflow {
namespace example {
namespace {

constexpr char kDenseInt64Key[] = "dense_int64";
constexpr char kDenseFloatKey[] = "dense_float";
constexpr char kDenseStringKey[] = "dense_string";

constexpr char kSparseInt64Key[] = "sparse_int64";
constexpr char kSparseFloatKey[] = "sparse_float";
constexpr char kSparseStringKey[] = "sparse_string";

string SerializedToReadable(string serialized) {
   std::vector<std::string> mht_0_v;
   mht_0_v.push_back("serialized: \"" + serialized + "\"");
   MHTracer_DTPStensorflowPScorePSutilPSexample_proto_fast_parsing_testDTcc mht_0(mht_0_v, 211, "", "./tensorflow/core/util/example_proto_fast_parsing_test.cc", "SerializedToReadable");

  string result;
  result += '"';
  for (char c : serialized)
    result += strings::StrCat("\\x", strings::Hex(c, strings::kZeroPad2));
  result += '"';
  return result;
}

template <class T>
string Serialize(const T& example) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSutilPSexample_proto_fast_parsing_testDTcc mht_1(mht_1_v, 224, "", "./tensorflow/core/util/example_proto_fast_parsing_test.cc", "Serialize");

  string serialized;
  example.SerializeToString(&serialized);
  return serialized;
}

// Tests that serialized gets parsed identically by TestFastParse(..)
// and the regular Example.ParseFromString(..).
void TestCorrectness(const string& serialized) {
   std::vector<std::string> mht_2_v;
   mht_2_v.push_back("serialized: \"" + serialized + "\"");
   MHTracer_DTPStensorflowPScorePSutilPSexample_proto_fast_parsing_testDTcc mht_2(mht_2_v, 236, "", "./tensorflow/core/util/example_proto_fast_parsing_test.cc", "TestCorrectness");

  Example example;
  Example fast_example;
  EXPECT_TRUE(example.ParseFromString(serialized));
  example.DiscardUnknownFields();
  EXPECT_TRUE(TestFastParse(serialized, &fast_example));
  EXPECT_EQ(example.DebugString(), fast_example.DebugString());
  if (example.DebugString() != fast_example.DebugString()) {
    LOG(ERROR) << "Bad serialized: " << SerializedToReadable(serialized);
  }
}

// Fast parsing does not differentiate between EmptyExample and EmptyFeatures
// TEST(FastParse, EmptyExample) {
//   Example example;
//   TestCorrectness(example);
// }

TEST(FastParse, IgnoresPrecedingUnknownTopLevelFields) {
  ExampleWithExtras example;
  (*example.mutable_features()->mutable_feature())["age"]
      .mutable_int64_list()
      ->add_value(13);
  example.set_extra1("some_str");
  example.set_extra2(123);
  example.set_extra3(234);
  example.set_extra4(345);
  example.set_extra5(4.56);
  example.add_extra6(5.67);
  example.add_extra6(6.78);
  (*example.mutable_extra7()->mutable_feature())["extra7"]
      .mutable_int64_list()
      ->add_value(1337);

  Example context;
  (*context.mutable_features()->mutable_feature())["zipcode"]
      .mutable_int64_list()
      ->add_value(94043);

  TestCorrectness(strings::StrCat(Serialize(example), Serialize(context)));
}

TEST(FastParse, IgnoresTrailingUnknownTopLevelFields) {
  Example example;
  (*example.mutable_features()->mutable_feature())["age"]
      .mutable_int64_list()
      ->add_value(13);

  ExampleWithExtras context;
  (*context.mutable_features()->mutable_feature())["zipcode"]
      .mutable_int64_list()
      ->add_value(94043);
  context.set_extra1("some_str");
  context.set_extra2(123);
  context.set_extra3(234);
  context.set_extra4(345);
  context.set_extra5(4.56);
  context.add_extra6(5.67);
  context.add_extra6(6.78);
  (*context.mutable_extra7()->mutable_feature())["extra7"]
      .mutable_int64_list()
      ->add_value(1337);

  TestCorrectness(strings::StrCat(Serialize(example), Serialize(context)));
}

TEST(FastParse, SingleInt64WithContext) {
  Example example;
  (*example.mutable_features()->mutable_feature())["age"]
      .mutable_int64_list()
      ->add_value(13);

  Example context;
  (*context.mutable_features()->mutable_feature())["zipcode"]
      .mutable_int64_list()
      ->add_value(94043);

  TestCorrectness(strings::StrCat(Serialize(example), Serialize(context)));
}

TEST(FastParse, DenseInt64WithContext) {
  Example example;
  (*example.mutable_features()->mutable_feature())["age"]
      .mutable_int64_list()
      ->add_value(0);

  Example context;
  (*context.mutable_features()->mutable_feature())["age"]
      .mutable_int64_list()
      ->add_value(15);

  string serialized = Serialize(example) + Serialize(context);

  {
    Example deserialized;
    EXPECT_TRUE(deserialized.ParseFromString(serialized));
    EXPECT_EQ(deserialized.DebugString(), context.DebugString());
    // Whoa! Last EQ is very surprising, but standard deserialization is what it
    // is and Servo team requested to replicate this 'feature'.
    // In future we should return error.
  }
  TestCorrectness(serialized);
}

TEST(FastParse, NonPacked) {
  TestCorrectness(
      "\x0a\x0e\x0a\x0c\x0a\x03\x61\x67\x65\x12\x05\x1a\x03\x0a\x01\x0d");
}

TEST(FastParse, Packed) {
  TestCorrectness(
      "\x0a\x0d\x0a\x0b\x0a\x03\x61\x67\x65\x12\x04\x1a\x02\x08\x0d");
}

TEST(FastParse, ValueBeforeKeyInMap) {
  TestCorrectness("\x0a\x12\x0a\x10\x12\x09\x0a\x07\x0a\x05value\x0a\x03key");
}

TEST(FastParse, EmptyFeatures) {
  Example example;
  example.mutable_features();
  TestCorrectness(Serialize(example));
}

void TestCorrectnessJson(const string& json) {
   std::vector<std::string> mht_3_v;
   mht_3_v.push_back("json: \"" + json + "\"");
   MHTracer_DTPStensorflowPScorePSutilPSexample_proto_fast_parsing_testDTcc mht_3(mht_3_v, 364, "", "./tensorflow/core/util/example_proto_fast_parsing_test.cc", "TestCorrectnessJson");

  auto resolver = protobuf::util::NewTypeResolverForDescriptorPool(
      "type.googleapis.com", protobuf::DescriptorPool::generated_pool());
  string serialized;
  auto s = protobuf::util::JsonToBinaryString(
      resolver, "type.googleapis.com/tensorflow.Example", json, &serialized);
  EXPECT_TRUE(s.ok()) << s;
  delete resolver;
  TestCorrectness(serialized);
}

TEST(FastParse, JsonUnivalent) {
  TestCorrectnessJson(
      "{'features': {"
      "  'feature': {'age': {'int64_list': {'value': [0]} }}, "
      "  'feature': {'flo': {'float_list': {'value': [1.1]} }}, "
      "  'feature': {'byt': {'bytes_list': {'value': ['WW8='] }}}"
      "}}");
}

TEST(FastParse, JsonMultivalent) {
  TestCorrectnessJson(
      "{'features': {"
      "  'feature': {'age': {'int64_list': {'value': [0, 13, 23]} }}, "
      "  'feature': {'flo': {'float_list': {'value': [1.1, 1.2, 1.3]} }}, "
      "  'feature': {'byt': {'bytes_list': {'value': ['WW8=', 'WW8K'] }}}"
      "}}");
}

TEST(FastParse, SingleInt64) {
  Example example;
  (*example.mutable_features()->mutable_feature())["age"]
      .mutable_int64_list()
      ->add_value(13);
  TestCorrectness(Serialize(example));
}

static string ExampleWithSomeFeatures() {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSutilPSexample_proto_fast_parsing_testDTcc mht_4(mht_4_v, 404, "", "./tensorflow/core/util/example_proto_fast_parsing_test.cc", "ExampleWithSomeFeatures");

  Example example;

  (*example.mutable_features()->mutable_feature())[""];

  (*example.mutable_features()->mutable_feature())["empty_bytes_list"]
      .mutable_bytes_list();
  (*example.mutable_features()->mutable_feature())["empty_float_list"]
      .mutable_float_list();
  (*example.mutable_features()->mutable_feature())["empty_int64_list"]
      .mutable_int64_list();

  BytesList* bytes_list =
      (*example.mutable_features()->mutable_feature())["bytes_list"]
          .mutable_bytes_list();
  bytes_list->add_value("bytes1");
  bytes_list->add_value("bytes2");

  FloatList* float_list =
      (*example.mutable_features()->mutable_feature())["float_list"]
          .mutable_float_list();
  float_list->add_value(1.0);
  float_list->add_value(2.0);

  Int64List* int64_list =
      (*example.mutable_features()->mutable_feature())["int64_list"]
          .mutable_int64_list();
  int64_list->add_value(3);
  int64_list->add_value(270);
  int64_list->add_value(86942);

  return Serialize(example);
}

TEST(FastParse, SomeFeatures) { TestCorrectness(ExampleWithSomeFeatures()); }

static void AddDenseFeature(const char* feature_name, DataType dtype,
                            PartialTensorShape shape, bool variable_length,
                            size_t elements_per_stride,
                            FastParseExampleConfig* out_config) {
   std::vector<std::string> mht_5_v;
   mht_5_v.push_back("feature_name: \"" + (feature_name == nullptr ? std::string("nullptr") : std::string((char*)feature_name)) + "\"");
   MHTracer_DTPStensorflowPScorePSutilPSexample_proto_fast_parsing_testDTcc mht_5(mht_5_v, 447, "", "./tensorflow/core/util/example_proto_fast_parsing_test.cc", "AddDenseFeature");

  out_config->dense.emplace_back();
  auto& new_feature = out_config->dense.back();
  new_feature.feature_name = feature_name;
  new_feature.dtype = dtype;
  new_feature.shape = std::move(shape);
  new_feature.default_value = Tensor(dtype, {});
  new_feature.variable_length = variable_length;
  new_feature.elements_per_stride = elements_per_stride;
}

static void AddSparseFeature(const char* feature_name, DataType dtype,
                             FastParseExampleConfig* out_config) {
   std::vector<std::string> mht_6_v;
   mht_6_v.push_back("feature_name: \"" + (feature_name == nullptr ? std::string("nullptr") : std::string((char*)feature_name)) + "\"");
   MHTracer_DTPStensorflowPScorePSutilPSexample_proto_fast_parsing_testDTcc mht_6(mht_6_v, 463, "", "./tensorflow/core/util/example_proto_fast_parsing_test.cc", "AddSparseFeature");

  out_config->sparse.emplace_back();
  auto& new_feature = out_config->sparse.back();
  new_feature.feature_name = feature_name;
  new_feature.dtype = dtype;
}

TEST(FastParse, StatsCollection) {
  const size_t kNumExamples = 13;
  std::vector<tstring> serialized(kNumExamples, ExampleWithSomeFeatures());

  FastParseExampleConfig config_dense;
  AddDenseFeature("bytes_list", DT_STRING, {2}, false, 2, &config_dense);
  AddDenseFeature("float_list", DT_FLOAT, {2}, false, 2, &config_dense);
  AddDenseFeature("int64_list", DT_INT64, {3}, false, 3, &config_dense);
  config_dense.collect_feature_stats = true;

  FastParseExampleConfig config_varlen;
  AddDenseFeature("bytes_list", DT_STRING, {-1}, true, 1, &config_varlen);
  AddDenseFeature("float_list", DT_FLOAT, {-1}, true, 1, &config_varlen);
  AddDenseFeature("int64_list", DT_INT64, {-1}, true, 1, &config_varlen);
  config_varlen.collect_feature_stats = true;

  FastParseExampleConfig config_sparse;
  AddSparseFeature("bytes_list", DT_STRING, &config_sparse);
  AddSparseFeature("float_list", DT_FLOAT, &config_sparse);
  AddSparseFeature("int64_list", DT_INT64, &config_sparse);
  config_sparse.collect_feature_stats = true;

  FastParseExampleConfig config_mixed;
  AddDenseFeature("bytes_list", DT_STRING, {2}, false, 2, &config_mixed);
  AddDenseFeature("float_list", DT_FLOAT, {-1}, true, 1, &config_mixed);
  AddSparseFeature("int64_list", DT_INT64, &config_mixed);
  config_mixed.collect_feature_stats = true;

  for (const FastParseExampleConfig& config :
       {config_dense, config_varlen, config_sparse, config_mixed}) {
    {
      Result result;
      TF_CHECK_OK(FastParseExample(config, serialized, {}, nullptr, &result));
      EXPECT_EQ(kNumExamples, result.feature_stats.size());
      for (const PerExampleFeatureStats& stats : result.feature_stats) {
        EXPECT_EQ(7, stats.features_count);
        EXPECT_EQ(7, stats.feature_values_count);
      }
    }

    {
      Result result;
      TF_CHECK_OK(FastParseSingleExample(config, serialized[0], &result));
      EXPECT_EQ(1, result.feature_stats.size());
      EXPECT_EQ(7, result.feature_stats[0].features_count);
      EXPECT_EQ(7, result.feature_stats[0].feature_values_count);
    }
  }
}

string RandStr(random::SimplePhilox* rng) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePSutilPSexample_proto_fast_parsing_testDTcc mht_7(mht_7_v, 523, "", "./tensorflow/core/util/example_proto_fast_parsing_test.cc", "RandStr");

  static const char key_char_lookup[] =
      "0123456789{}~`!@#$%^&*()"
      "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
      "abcdefghijklmnopqrstuvwxyz";
  auto len = 1 + rng->Rand32() % 200;
  string str;
  str.reserve(len);
  while (len-- > 0) {
    str.push_back(
        key_char_lookup[rng->Rand32() % (sizeof(key_char_lookup) /
                                         sizeof(key_char_lookup[0]))]);
  }
  return str;
}

void Fuzz(random::SimplePhilox* rng) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScorePSutilPSexample_proto_fast_parsing_testDTcc mht_8(mht_8_v, 542, "", "./tensorflow/core/util/example_proto_fast_parsing_test.cc", "Fuzz");

  // Generate keys.
  auto num_keys = 1 + rng->Rand32() % 100;
  std::unordered_set<string> unique_keys;
  for (auto i = 0; i < num_keys; ++i) {
    unique_keys.emplace(RandStr(rng));
  }

  // Generate serialized example.
  Example example;
  string serialized_example;
  auto num_concats = 1 + rng->Rand32() % 4;
  std::vector<Feature::KindCase> feat_types(
      {Feature::kBytesList, Feature::kFloatList, Feature::kInt64List});
  std::vector<string> all_keys(unique_keys.begin(), unique_keys.end());
  while (num_concats--) {
    example.Clear();
    auto num_active_keys = 1 + rng->Rand32() % all_keys.size();

    // Generate features.
    for (auto i = 0; i < num_active_keys; ++i) {
      auto fkey = all_keys[rng->Rand32() % all_keys.size()];
      auto ftype_idx = rng->Rand32() % feat_types.size();
      auto num_features = 1 + rng->Rand32() % 5;
      switch (static_cast<Feature::KindCase>(feat_types[ftype_idx])) {
        case Feature::kBytesList: {
          BytesList* bytes_list =
              (*example.mutable_features()->mutable_feature())[fkey]
                  .mutable_bytes_list();
          while (num_features--) {
            bytes_list->add_value(RandStr(rng));
          }
          break;
        }
        case Feature::kFloatList: {
          FloatList* float_list =
              (*example.mutable_features()->mutable_feature())[fkey]
                  .mutable_float_list();
          while (num_features--) {
            float_list->add_value(rng->RandFloat());
          }
          break;
        }
        case Feature::kInt64List: {
          Int64List* int64_list =
              (*example.mutable_features()->mutable_feature())[fkey]
                  .mutable_int64_list();
          while (num_features--) {
            int64_list->add_value(rng->Rand64());
          }
          break;
        }
        default: {
          LOG(QFATAL);
          break;
        }
      }
    }
    serialized_example += example.SerializeAsString();
  }

  // Test correctness.
  TestCorrectness(serialized_example);
}

TEST(FastParse, FuzzTest) {
  const uint64 seed = 1337;
  random::PhiloxRandom philox(seed);
  random::SimplePhilox rng(&philox);
  auto num_runs = 200;
  while (num_runs--) {
    LOG(INFO) << "runs left: " << num_runs;
    Fuzz(&rng);
  }
}

TEST(TestFastParseExample, Empty) {
  Result result;
  FastParseExampleConfig config;
  config.sparse.push_back({"test", DT_STRING});
  Status status =
      FastParseExample(config, gtl::ArraySlice<tstring>(),
                       gtl::ArraySlice<tstring>(), nullptr, &result);
  EXPECT_TRUE(status.ok()) << status;
}

}  // namespace
}  // namespace example
}  // namespace tensorflow
