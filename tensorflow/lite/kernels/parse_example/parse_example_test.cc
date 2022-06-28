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
class MHTracer_DTPStensorflowPSlitePSkernelsPSparse_examplePSparse_example_testDTcc {
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
   MHTracer_DTPStensorflowPSlitePSkernelsPSparse_examplePSparse_example_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePSkernelsPSparse_examplePSparse_example_testDTcc() {
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

/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/lite/kernels/parse_example/parse_example.h"

#include <cstdint>
#include <initializer_list>

#include "flatbuffers/flexbuffers.h"  // from @flatbuffers
#include "tensorflow/core/example/feature_util.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/platform/protobuf.h"
#include "tensorflow/core/platform/tstring.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/core/api/op_resolver.h"
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/interpreter_builder.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/kernels/test_util.h"
#include "tensorflow/lite/model_builder.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/string_util.h"

namespace tflite {
namespace ops {
namespace custom {

namespace tf = ::tensorflow;

const char* kNodeDefTxt = R"pb(
  name: "ParseExample/ParseExample"
  op: "ParseExample"
  input: "serialized"
  input: "ParseExample/ParseExample/names"
  input: "ParseExample/ParseExample/dense_keys_0"
  input: "ParseExample/Const"
  attr {
    key: "Ndense"
    value { i: 1 }
  }
  attr {
    key: "Nsparse"
    value { i: 0 }
  }
  attr {
    key: "Tdense"
    value { list { type: DT_FLOAT } }
  }
  attr {
    key: "dense_shapes"
    value { list { shape { dim { size: 2 } } } }
  }
  attr {
    key: "sparse_types"
    value { list { type: DT_FLOAT } }
  }
)pb";

const char* kNodeDefTxt2 = R"pb(
  name: "ParseExample/ParseExample"
  op: "ParseExample"
  input: "serialized"
  input: "ParseExample/ParseExample/names"
  input: "ParseExample/ParseExample/sparse_keys_0"
  attr {
    key: "Ndense"
    value { i: 0 }
  }
  attr {
    key: "Nsparse"
    value { i: 1 }
  }
  attr {
    key: "Tdense"
    value {}
  }
  attr {
    key: "dense_shapes"
    value {}
  }
  attr {
    key: "sparse_types"
    value { list { type: DT_FLOAT } }
  }
)pb";

const char* kNodeDefTxt3 = R"pb(
  name: "ParseExample/ParseExample"
  op: "ParseExample"
  input: "serialized"
  input: "ParseExample/ParseExample/names"
  input: "ParseExample/ParseExample/sparse_keys_0"
  attr {
    key: "Ndense"
    value { i: 1 }
  }
  attr {
    key: "Nsparse"
    value { i: 0 }
  }
  attr {
    key: "Tdense"
    value { list { type: DT_STRING } }
  }
  attr {
    key: "dense_shapes"
    value { list { shape { dim { size: 1 } } } }
  }
  attr {
    key: "sparse_types"
    value { list { type: DT_FLOAT } }
  }
)pb";

const char* kNodeDefTxt4 = R"pb(
  name: "ParseExample/ParseExample"
  op: "ParseExample"
  input: "serialized"
  input: "ParseExample/ParseExample/names"
  input: "ParseExample/ParseExample/sparse_keys_0"
  attr {
    key: "Ndense"
    value { i: 0 }
  }
  attr {
    key: "Nsparse"
    value { i: 1 }
  }
  attr {
    key: "Tdense"
    value {}
  }
  attr {
    key: "dense_shapes"
    value {}
  }
  attr {
    key: "sparse_types"
    value { list { type: DT_STRING } }
  }
)pb";

const char* kNodeDefTxt5 = R"pb(
  name: "ParseExample/ParseExample"
  op: "ParseExample"
  input: "serialized"
  input: "ParseExample/ParseExample/names"
  input: "ParseExample/ParseExample/dense_keys_0"
  input: "ParseExample/Const"
  attr {
    key: "Ndense"
    value { i: 1 }
  }
  attr {
    key: "Nsparse"
    value { i: 0 }
  }
  attr {
    key: "Tdense"
    value { list { type: DT_FLOAT } }
  }
  attr {
    key: "dense_shapes"
    value {}
  }
  attr {
    key: "sparse_types"
    value { list { type: DT_FLOAT } }
  }
)pb";

template <typename DefaultType>
class ParseExampleOpModel : public SingleOpModel {
 public:
  ParseExampleOpModel(std::vector<std::string> serialized_examples,
                      std::vector<std::string> sparse_keys,
                      std::vector<std::string> dense_keys,
                      std::initializer_list<DefaultType> dense_defaults,
                      std::vector<TensorType> dense_types,
                      std::vector<TensorType> sparse_types,
                      const char* text_def, int dense_size = 2) {
   std::vector<std::string> mht_0_v;
   mht_0_v.push_back("text_def: \"" + (text_def == nullptr ? std::string("nullptr") : std::string((char*)text_def)) + "\"");
   MHTracer_DTPStensorflowPSlitePSkernelsPSparse_examplePSparse_example_testDTcc mht_0(mht_0_v, 362, "", "./tensorflow/lite/kernels/parse_example/parse_example_test.cc", "ParseExampleOpModel");

    // Example
    const int input_size = serialized_examples.size();
    auto input_tensor_data = TensorData(TensorType_STRING, {input_size});
    string_indices_.push_back(AddInput(input_tensor_data));
    // Names
    string_indices_.push_back(
        AddConstInput<std::string>(TensorData(TensorType_STRING, {0}), {""}));
    std::for_each(sparse_keys.begin(), sparse_keys.end(), [&](auto&&) {
      string_indices_.push_back(AddInput(TensorData(TensorType_STRING, {1})));
    });
    std::for_each(dense_keys.begin(), dense_keys.end(), [&](auto&&) {
      string_indices_.push_back(AddInput(TensorData(TensorType_STRING, {1})));
    });
    if (dense_size > 0) {
      dense_defaults_ = AddConstInput<DefaultType>(
          TensorData(dense_types[0], {dense_size}), dense_defaults);
    }
    if (!sparse_keys.empty()) {
      for (int i = 0; i < sparse_keys.size(); i++) {
        sparse_indices_outputs_.push_back(AddOutput(TensorType_INT64));
      }
      for (int i = 0; i < sparse_keys.size(); i++) {
        sparse_values_outputs_.push_back(AddOutput(sparse_types[i]));
      }
      for (int i = 0; i < sparse_keys.size(); i++) {
        sparse_shapes_outputs_.push_back(AddOutput({TensorType_INT64, {2}}));
      }
    }
    for (int i = 0; i < dense_keys.size(); i++) {
      dense_outputs_.push_back(AddOutput({dense_types[i], {dense_size}}));
    }

    tf::NodeDef nodedef;
    tf::protobuf::TextFormat::Parser parser;
    tf::protobuf::io::ArrayInputStream input_stream(text_def, strlen(text_def));
    if (!parser.Parse(&input_stream, &nodedef)) {
      abort();
    }
    std::string serialized_nodedef;
    nodedef.SerializeToString(&serialized_nodedef);
    flexbuffers::Builder fbb;
    fbb.Vector([&]() {
      fbb.String(nodedef.op());
      fbb.String(serialized_nodedef);
    });
    fbb.Finish();
    const auto buffer = fbb.GetBuffer();
    SetCustomOp("ParseExample", buffer, Register_PARSE_EXAMPLE);
    BuildInterpreter({{input_size}});
    int idx = 0;
    PopulateStringTensor(string_indices_[idx++], serialized_examples);
    PopulateStringTensor(string_indices_[idx++], {""});
    for (const auto& key : sparse_keys) {
      PopulateStringTensor(string_indices_[idx++], {key});
    }
    for (const auto& key : dense_keys) {
      PopulateStringTensor(string_indices_[idx++], {key});
    }
  }

  void ResizeInputTensor(std::vector<std::vector<int>> input_shapes) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSparse_examplePSparse_example_testDTcc mht_1(mht_1_v, 426, "", "./tensorflow/lite/kernels/parse_example/parse_example_test.cc", "ResizeInputTensor");

    for (size_t i = 0; i < input_shapes.size(); ++i) {
      const int input_idx = interpreter_->inputs()[i];
      if (input_idx == kTfLiteOptionalTensor) continue;
      const auto& shape = input_shapes[i];
      if (shape.empty()) continue;
      CHECK(interpreter_->ResizeInputTensor(input_idx, shape) == kTfLiteOk);
    }
  }

  template <typename T>
  std::vector<T> GetSparseIndicesOutput(int i) {
    return ExtractVector<T>(sparse_indices_outputs_[i]);
  }

  template <typename T>
  std::vector<T> GetSparseValuesOutput(int i) {
    return ExtractVector<T>(sparse_values_outputs_[i]);
  }

  template <typename T>
  std::vector<T> GetSparseShapesOutput(int i) {
    return ExtractVector<T>(sparse_shapes_outputs_[i]);
  }

  template <typename T>
  std::vector<T> GetDenseOutput(int i) {
    return ExtractVector<T>(dense_outputs_[i]);
  }

  std::vector<std::string> GetStringOutput(int i) {
    auto* t = interpreter_->tensor(i);
    int count = GetStringCount(t);
    std::vector<std::string> v;
    for (int i = 0; i < count; ++i) {
      auto ref = GetString(t, i);
      v.emplace_back(ref.str, ref.len);
    }
    return v;
  }

  int DenseDefaults() {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSparse_examplePSparse_example_testDTcc mht_2(mht_2_v, 470, "", "./tensorflow/lite/kernels/parse_example/parse_example_test.cc", "DenseDefaults");
 return dense_defaults_; }

  int SparseValuesOutputs(int i) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSparse_examplePSparse_example_testDTcc mht_3(mht_3_v, 475, "", "./tensorflow/lite/kernels/parse_example/parse_example_test.cc", "SparseValuesOutputs");
 return sparse_values_outputs_[i]; }

  int DenseOutputs(int i) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSparse_examplePSparse_example_testDTcc mht_4(mht_4_v, 480, "", "./tensorflow/lite/kernels/parse_example/parse_example_test.cc", "DenseOutputs");
 return dense_outputs_[i]; }

  std::vector<int> dense_outputs_;
  std::vector<int> sparse_indices_outputs_;
  std::vector<int> sparse_shapes_outputs_;
  std::vector<int> sparse_values_outputs_;
  std::vector<int> string_indices_;
  int dense_defaults_ = -1;
};

TEST(ParseExampleOpsTest, SimpleTest) {
  tf::Example example;
  tf::AppendFeatureValues<float>({1.5f, 1.5f}, "time", &example);
  tf::AppendFeatureValues<float>({1.0f, 1.0f}, "num", &example);
  ParseExampleOpModel<float> m({example.SerializeAsString()}, {}, {"time"},
                               {0.f, 0.f}, {TensorType_FLOAT32}, {},
                               kNodeDefTxt);
  ASSERT_EQ(m.InvokeUnchecked(), kTfLiteOk);
  EXPECT_THAT(m.GetDenseOutput<float>(0),
              ElementsAreArray(ArrayFloatNear({1.5f, 1.5f})));
}

TEST(ParseExampleOpsTest, SparseTest) {
  tf::Example example;
  tf::AppendFeatureValues<float>({1.5f}, "time", &example);
  ParseExampleOpModel<float> m({example.SerializeAsString()}, {"time"}, {}, {},
                               {}, {TensorType_FLOAT32}, kNodeDefTxt2, 0);
  ASSERT_EQ(m.InvokeUnchecked(), kTfLiteOk);
  EXPECT_THAT(m.GetSparseIndicesOutput<int64_t>(0),
              ElementsAreArray(ArrayFloatNear({0, 0})));
  EXPECT_THAT(m.GetSparseValuesOutput<float>(0),
              ElementsAreArray(ArrayFloatNear({1.5f})));
  EXPECT_THAT(m.GetSparseShapesOutput<int64_t>(0),
              ElementsAreArray(ArrayFloatNear({1, 1})));
}

TEST(ParseExampleOpsTest, SimpleBytesTest) {
  tf::Example example;
  const std::string test_data = "simpletest";
  tf::AppendFeatureValues<tensorflow::tstring>({test_data}, "time", &example);
  tf::AppendFeatureValues<float>({1.0f, 1.0f}, "num", &example);
  std::string default_value = "missing";
  ParseExampleOpModel<std::string> m({example.SerializeAsString()}, {},
                                     {"time"}, {default_value},
                                     {TensorType_STRING}, {}, kNodeDefTxt3, 1);
  m.PopulateStringTensor(m.DenseDefaults(), {default_value});
  ASSERT_EQ(m.InvokeUnchecked(), kTfLiteOk);
  std::vector<string> c = m.GetStringOutput(m.DenseOutputs(0));
  EXPECT_EQ(1, c.size());
  EXPECT_EQ(test_data, c[0]);
}

TEST(ParseExampleOpsTest, SparseBytesTest) {
  tf::Example example;
  const std::string test_data = "simpletest";
  tf::AppendFeatureValues<tensorflow::tstring>({test_data, test_data}, "time",
                                               &example);
  tf::AppendFeatureValues<float>({1.0f, 1.0f}, "num", &example);
  ParseExampleOpModel<std::string> m({example.SerializeAsString()}, {"time"},
                                     {}, {}, {}, {TensorType_STRING},
                                     kNodeDefTxt4, 0);
  ASSERT_EQ(m.InvokeUnchecked(), kTfLiteOk);
  EXPECT_THAT(m.GetSparseIndicesOutput<int64_t>(0),
              testing::ElementsAreArray({0, 0, 0, 1}));
  auto values = m.GetStringOutput(m.SparseValuesOutputs(0));
  EXPECT_EQ(2, values.size());
  EXPECT_EQ(test_data, values[0]);
  EXPECT_EQ(test_data, values[1]);
  EXPECT_THAT(m.GetSparseShapesOutput<int64_t>(0),
              testing::ElementsAreArray({1, 2}));
}

TEST(ParseExampleOpsTest, ResizeTest) {
  const int num_tests = 3;
  std::vector<tf::Example> examples(num_tests);
  std::vector<std::vector<float>> expected(num_tests);
  std::vector<std::vector<std::string>> inputs(num_tests);
  std::vector<int> sizes;
  for (int i = 0; i < num_tests; ++i) {
    float val = i;
    std::initializer_list<float> floats = {val + val / 10.f, -val - val / 10.f};
    tf::AppendFeatureValues<float>({val, val}, "num", &examples[i]);
    tf::AppendFeatureValues<float>(floats, "time", &examples[i]);
    sizes.push_back((num_tests - i) * 2);
    for (int j = 0; j < sizes.back(); ++j) {
      inputs[i].push_back(examples[i].SerializeAsString());
      expected[i].insert(expected[i].end(), floats.begin(), floats.end());
    }
  }

  ParseExampleOpModel<float> m(inputs[0], {}, {"time"}, {0.f, 0.f},
                               {TensorType_FLOAT32}, {}, kNodeDefTxt);
  ASSERT_EQ(m.InvokeUnchecked(), kTfLiteOk);
  EXPECT_THAT(m.GetDenseOutput<float>(0),
              ElementsAreArray(ArrayFloatNear(expected[0])));

  for (int i = 1; i < num_tests; ++i) {
    m.ResizeInputTensor({{sizes[i]}});
    m.AllocateAndDelegate(false);
    m.PopulateStringTensor(0, inputs[i]);
    ASSERT_EQ(m.InvokeUnchecked(), kTfLiteOk);
    EXPECT_THAT(m.GetDenseOutput<float>(0),
                ElementsAreArray(ArrayFloatNear(expected[i])));
  }
}

TEST(ParseExampleOpsTest, ResizeMissingInfoTest) {
  const int num_tests = 3;
  std::vector<tf::Example> examples(num_tests);
  std::vector<std::vector<float>> expected(num_tests);
  std::vector<std::vector<std::string>> inputs(num_tests);
  std::vector<int> sizes;
  for (int i = 0; i < num_tests; ++i) {
    float val = i;
    std::initializer_list<float> floats = {val + val / 10.f, -val - val / 10.f};
    tf::AppendFeatureValues<float>({val, val}, "num", &examples[i]);
    tf::AppendFeatureValues<float>(floats, "time", &examples[i]);
    sizes.push_back((num_tests - i) * 2);
    for (int j = 0; j < sizes.back(); ++j) {
      inputs[i].push_back(examples[i].SerializeAsString());
      expected[i].insert(expected[i].end(), floats.begin(), floats.end());
    }
  }

  ParseExampleOpModel<float> m(inputs[0], {}, {"time"}, {0.f, 0.f},
                               {TensorType_FLOAT32}, {}, kNodeDefTxt5);
  ASSERT_EQ(m.InvokeUnchecked(), kTfLiteOk);
  EXPECT_THAT(m.GetDenseOutput<float>(0),
              ElementsAreArray(ArrayFloatNear(expected[0])));

  for (int i = 1; i < num_tests; ++i) {
    m.ResizeInputTensor({{sizes[i]}});
    m.AllocateAndDelegate(false);
    m.PopulateStringTensor(0, inputs[i]);
    ASSERT_EQ(m.InvokeUnchecked(), kTfLiteOk);
    EXPECT_THAT(m.GetDenseOutput<float>(0),
                ElementsAreArray(ArrayFloatNear(expected[i])));
  }
}

}  // namespace custom
}  // namespace ops
}  // namespace tflite
