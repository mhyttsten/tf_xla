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
class MHTracer_DTPStensorflowPSlitePStoolsPSbenchmarkPSbenchmark_testDTcc {
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
   MHTracer_DTPStensorflowPSlitePStoolsPSbenchmarkPSbenchmark_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePStoolsPSbenchmarkPSbenchmark_testDTcc() {
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

/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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
#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/algorithm/algorithm.h"
#include "absl/memory/memory.h"
#include "absl/strings/str_format.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/string_util.h"
#include "tensorflow/lite/testing/util.h"
#include "tensorflow/lite/tools/benchmark/benchmark_performance_options.h"
#include "tensorflow/lite/tools/benchmark/benchmark_tflite_model.h"
#include "tensorflow/lite/tools/command_line_flags.h"
#include "tensorflow/lite/tools/delegates/delegate_provider.h"
#include "tensorflow/lite/tools/logging.h"

namespace {
const std::string* g_fp32_model_path = nullptr;
const std::string* g_int8_model_path = nullptr;
const std::string* g_string_model_path = nullptr;
}  // namespace

namespace tflite {
namespace benchmark {
namespace {

enum class ModelGraphType { FP32, INT8, STRING };

BenchmarkParams CreateParams(int32_t num_runs, float min_secs, float max_secs,
                             ModelGraphType graph_type = ModelGraphType::FP32) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePStoolsPSbenchmarkPSbenchmark_testDTcc mht_0(mht_0_v, 218, "", "./tensorflow/lite/tools/benchmark/benchmark_test.cc", "CreateParams");

  BenchmarkParams params = BenchmarkTfLiteModel::DefaultParams();
  params.Set<int32_t>("num_runs", num_runs);
  params.Set<float>("min_secs", min_secs);
  params.Set<float>("max_secs", max_secs);

  if (graph_type == ModelGraphType::INT8) {
    params.Set<std::string>("graph", *g_int8_model_path);
  } else if (graph_type == ModelGraphType::STRING) {
    params.Set<std::string>("graph", *g_string_model_path);
  } else {
    // by default, simply use the fp32 one.
    params.Set<std::string>("graph", *g_fp32_model_path);
  }
  return params;
}

BenchmarkParams CreateParams() {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSlitePStoolsPSbenchmarkPSbenchmark_testDTcc mht_1(mht_1_v, 238, "", "./tensorflow/lite/tools/benchmark/benchmark_test.cc", "CreateParams");
 return CreateParams(2, 1.0f, 150.0f); }
BenchmarkParams CreateFp32Params() {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSlitePStoolsPSbenchmarkPSbenchmark_testDTcc mht_2(mht_2_v, 242, "", "./tensorflow/lite/tools/benchmark/benchmark_test.cc", "CreateFp32Params");

  return CreateParams(2, 1.0f, 150.0f, ModelGraphType::FP32);
}
BenchmarkParams CreateInt8Params() {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPSlitePStoolsPSbenchmarkPSbenchmark_testDTcc mht_3(mht_3_v, 248, "", "./tensorflow/lite/tools/benchmark/benchmark_test.cc", "CreateInt8Params");

  return CreateParams(2, 1.0f, 150.0f, ModelGraphType::INT8);
}
BenchmarkParams CreateStringParams() {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPSlitePStoolsPSbenchmarkPSbenchmark_testDTcc mht_4(mht_4_v, 254, "", "./tensorflow/lite/tools/benchmark/benchmark_test.cc", "CreateStringParams");

  return CreateParams(2, 1.0f, 150.0f, ModelGraphType::STRING);
}

std::string CreateFilePath(const std::string& file_name) {
   std::vector<std::string> mht_5_v;
   mht_5_v.push_back("file_name: \"" + file_name + "\"");
   MHTracer_DTPStensorflowPSlitePStoolsPSbenchmarkPSbenchmark_testDTcc mht_5(mht_5_v, 262, "", "./tensorflow/lite/tools/benchmark/benchmark_test.cc", "CreateFilePath");

  return std::string(getenv("TEST_TMPDIR")) + file_name;
}

void WriteInputLayerValueFile(const std::string& file_path,
                              ModelGraphType graph_type, int num_elements,
                              char file_value = 'a') {
   std::vector<std::string> mht_6_v;
   mht_6_v.push_back("file_path: \"" + file_path + "\"");
   MHTracer_DTPStensorflowPSlitePStoolsPSbenchmarkPSbenchmark_testDTcc mht_6(mht_6_v, 272, "", "./tensorflow/lite/tools/benchmark/benchmark_test.cc", "WriteInputLayerValueFile");

  std::ofstream file(file_path);
  int bytes = 0;
  switch (graph_type) {
    case ModelGraphType::FP32:
      bytes = 4 * num_elements;
      break;
    case ModelGraphType::INT8:
      bytes = num_elements;
      break;
    default:
      LOG(WARNING) << absl::StrFormat(
          "ModelGraphType(enum_value:%d) is not known.", graph_type);
      LOG(WARNING) << "The size of the ModelGraphType will be 1 byte in tests.";
      bytes = num_elements;
      break;
  }
  std::vector<char> buffer(bytes, file_value);
  file.write(buffer.data(), bytes);
}

void CheckInputTensorValue(const TfLiteTensor* input_tensor,
                           char expected_value) {
   std::vector<std::string> mht_7_v;
   mht_7_v.push_back("expected_value: '" + std::string(1, expected_value) + "'");
   MHTracer_DTPStensorflowPSlitePStoolsPSbenchmarkPSbenchmark_testDTcc mht_7(mht_7_v, 298, "", "./tensorflow/lite/tools/benchmark/benchmark_test.cc", "CheckInputTensorValue");

  ASSERT_THAT(input_tensor, testing::NotNull());
  EXPECT_TRUE(std::all_of(
      input_tensor->data.raw, input_tensor->data.raw + input_tensor->bytes,
      [expected_value](char c) { return c == expected_value; }));
}

void CheckInputTensorValue(const TfLiteTensor* input_tensor,
                           int tensor_dim_index,
                           const std::string& expected_value) {
   std::vector<std::string> mht_8_v;
   mht_8_v.push_back("expected_value: \"" + expected_value + "\"");
   MHTracer_DTPStensorflowPSlitePStoolsPSbenchmarkPSbenchmark_testDTcc mht_8(mht_8_v, 311, "", "./tensorflow/lite/tools/benchmark/benchmark_test.cc", "CheckInputTensorValue");

  StringRef tensor_value = GetString(input_tensor, tensor_dim_index);
  EXPECT_TRUE(absl::equal(tensor_value.str, tensor_value.str + tensor_value.len,
                          expected_value.c_str(),
                          expected_value.c_str() + expected_value.length()));
}

class TestBenchmark : public BenchmarkTfLiteModel {
 public:
  explicit TestBenchmark(BenchmarkParams params)
      : BenchmarkTfLiteModel(std::move(params)) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPSlitePStoolsPSbenchmarkPSbenchmark_testDTcc mht_9(mht_9_v, 324, "", "./tensorflow/lite/tools/benchmark/benchmark_test.cc", "TestBenchmark");
}
  const tflite::Interpreter* GetInterpreter() {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPSlitePStoolsPSbenchmarkPSbenchmark_testDTcc mht_10(mht_10_v, 328, "", "./tensorflow/lite/tools/benchmark/benchmark_test.cc", "GetInterpreter");
 return interpreter_.get(); }

  void Prepare() {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPSlitePStoolsPSbenchmarkPSbenchmark_testDTcc mht_11(mht_11_v, 333, "", "./tensorflow/lite/tools/benchmark/benchmark_test.cc", "Prepare");

    PrepareInputData();
    ResetInputsAndOutputs();
  }

  const TfLiteTensor* GetInputTensor(int index) {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPSlitePStoolsPSbenchmarkPSbenchmark_testDTcc mht_12(mht_12_v, 341, "", "./tensorflow/lite/tools/benchmark/benchmark_test.cc", "GetInputTensor");

    return index >= interpreter_->inputs().size()
               ? nullptr
               : interpreter_->input_tensor(index);
  }
};

TEST(BenchmarkTest, DoesntCrashFp32Model) {
  ASSERT_THAT(g_fp32_model_path, testing::NotNull());

  TestBenchmark benchmark(CreateFp32Params());
  benchmark.Run();
}

TEST(BenchmarkTest, DoesntCrashInt8Model) {
  ASSERT_THAT(g_int8_model_path, testing::NotNull());

  TestBenchmark benchmark(CreateInt8Params());
  benchmark.Run();
}

TEST(BenchmarkTest, DoesntCrashStringModel) {
  ASSERT_THAT(g_int8_model_path, testing::NotNull());

  TestBenchmark benchmark(CreateStringParams());
  benchmark.Run();
}

class TestMultiRunStatsRecorder : public MultiRunStatsRecorder {
 public:
  void OutputStats() override {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPSlitePStoolsPSbenchmarkPSbenchmark_testDTcc mht_13(mht_13_v, 374, "", "./tensorflow/lite/tools/benchmark/benchmark_test.cc", "OutputStats");

    MultiRunStatsRecorder::OutputStats();

    // Check results have been sorted according to avg. latency in increasing
    // order, and the incomplete runs are at the back of the results.
    double pre_avg_latency = -1e6;
    bool has_incomplete = false;  // ensure complete/incomplete are not mixed.
    for (const auto& result : results_) {
      const auto current_avg_latency = result.metrics.inference_time_us().avg();
      if (result.completed) {
        EXPECT_GE(current_avg_latency, pre_avg_latency);
        EXPECT_FALSE(has_incomplete);
      } else {
        EXPECT_EQ(0, result.metrics.inference_time_us().count());
        has_incomplete = true;
      }
      pre_avg_latency = current_avg_latency;
    }
  }
};

TEST(BenchmarkTest, DoesntCrashMultiPerfOptions) {
  ASSERT_THAT(g_fp32_model_path, testing::NotNull());

  TestBenchmark benchmark(CreateFp32Params());
  BenchmarkPerformanceOptions all_options_benchmark(
      &benchmark, absl::make_unique<TestMultiRunStatsRecorder>());
  all_options_benchmark.Run();
}

TEST(BenchmarkTest, DoesntCrashMultiPerfOptionsWithProfiling) {
  ASSERT_THAT(g_fp32_model_path, testing::NotNull());

  BenchmarkParams params = CreateFp32Params();
  params.Set<bool>("enable_op_profiling", true);
  TestBenchmark benchmark(std::move(params));
  BenchmarkPerformanceOptions all_options_benchmark(&benchmark);
  all_options_benchmark.Run();
}

TEST(BenchmarkTest, DoesntCrashWithExplicitInputFp32Model) {
  ASSERT_THAT(g_fp32_model_path, testing::NotNull());

  // Note: the following input-related params are *specific* to model
  // 'g_fp32_model_path' which is specified as 'lite:testdata/multi_add.bin for
  // the test.
  BenchmarkParams params = CreateFp32Params();
  params.Set<std::string>("input_layer", "a,b,c,d");
  params.Set<std::string>("input_layer_shape",
                          "1,8,8,3:1,8,8,3:1,8,8,3:1,8,8,3");
  params.Set<std::string>("input_layer_value_range", "d,1,10:b,0,100");
  TestBenchmark benchmark(std::move(params));
  benchmark.Run();
}

TEST(BenchmarkTest, DoesntCrashWithExplicitInputInt8Model) {
  ASSERT_THAT(g_int8_model_path, testing::NotNull());

  // Note: the following input-related params are *specific* to model
  // 'g_int8_model_path' which is specified as
  // 'lite:testdata/add_quantized_int8.bin for the test.
  int a_min = 1;
  int a_max = 10;
  BenchmarkParams params = CreateInt8Params();
  params.Set<std::string>("input_layer", "a");
  params.Set<std::string>("input_layer_shape", "1,8,8,3");
  params.Set<std::string>("input_layer_value_range",
                          absl::StrFormat("a,%d,%d", a_min, a_max));
  TestBenchmark benchmark(std::move(params));
  benchmark.Run();

  auto input_tensor = benchmark.GetInputTensor(0);
  ASSERT_THAT(input_tensor, testing::NotNull());
  EXPECT_TRUE(std::all_of(
      input_tensor->data.raw, input_tensor->data.raw + input_tensor->bytes,
      [a_min, a_max](int i) { return a_min <= i && i <= a_max; }));
}

TEST(BenchmarkTest, DoesntCrashWithExplicitInputValueFilesFp32Model) {
  ASSERT_THAT(g_fp32_model_path, testing::NotNull());
  char file_value_b = 'b';
  const std::string file_path_b = CreateFilePath("fp32_binary_b");
  WriteInputLayerValueFile(file_path_b, ModelGraphType::FP32, 192,
                           file_value_b);
  char file_value_d = 'd';
  const std::string file_path_d = CreateFilePath("fp32_binary_d");
  WriteInputLayerValueFile(file_path_d, ModelGraphType::FP32, 192,
                           file_value_d);

  // Note: the following input-related params are *specific* to model
  // 'g_fp32_model_path' which is specified as 'lite:testdata/multi_add.bin for
  // the test.
  BenchmarkParams params = CreateFp32Params();
  params.Set<std::string>("input_layer", "a,b,c,d");
  params.Set<std::string>("input_layer_shape",
                          "1,8,8,3:1,8,8,3:1,8,8,3:1,8,8,3");
  params.Set<std::string>("input_layer_value_files",
                          "d:" + file_path_d + ",b:" + file_path_b);
  TestBenchmark benchmark(std::move(params));
  benchmark.Run();

  CheckInputTensorValue(benchmark.GetInputTensor(1), file_value_b);
  CheckInputTensorValue(benchmark.GetInputTensor(3), file_value_d);
}

TEST(BenchmarkTest, DoesntCrashWithExplicitInputValueFilesInt8Model) {
  ASSERT_THAT(g_int8_model_path, testing::NotNull());
  const std::string file_path = CreateFilePath("int8_binary");
  char file_value = 'a';
  WriteInputLayerValueFile(file_path, ModelGraphType::INT8, 192, file_value);

  // Note: the following input-related params are *specific* to model
  // 'g_int8_model_path' which is specified as
  // 'lite:testdata/add_quantized_int8.bin for the test.
  BenchmarkParams params = CreateInt8Params();
  params.Set<std::string>("input_layer", "a");
  params.Set<std::string>("input_layer_shape", "1,8,8,3");
  params.Set<std::string>("input_layer_value_files", "a:" + file_path);
  TestBenchmark benchmark(std::move(params));
  benchmark.Run();

  CheckInputTensorValue(benchmark.GetInputTensor(0), file_value);
}

TEST(BenchmarkTest, DoesntCrashWithExplicitInputValueFilesStringModel) {
  ASSERT_THAT(g_string_model_path, testing::NotNull());
  const std::string file_path = CreateFilePath("string_binary");
  const std::string string_value_0 = "abcd";
  const std::string string_value_1 = "12345";
  const std::string string_value_2 = "a1b2c3d4e5";
  std::ofstream file(file_path);
  // Store the terminating null-character ('\0') at the end of the returned
  // value by std::string::c_str().
  file.write(string_value_0.c_str(), string_value_0.length() + 1);
  file.write(string_value_1.c_str(), string_value_1.length() + 1);
  file.write(string_value_2.c_str(), string_value_2.length() + 1);
  file.close();

  // Note: the following input-related params are *specific* to model
  // 'g_string_model_path' which is specified as
  // 'lite:testdata/string_input_model.bin for the test.
  BenchmarkParams params = CreateStringParams();
  params.Set<std::string>("input_layer", "a");
  params.Set<std::string>("input_layer_shape", "1,3");
  params.Set<std::string>("input_layer_value_files", "a:" + file_path);
  TestBenchmark benchmark(std::move(params));
  benchmark.Run();

  auto input_tensor = benchmark.GetInputTensor(0);
  ASSERT_THAT(input_tensor, testing::NotNull());
  EXPECT_EQ(GetStringCount(input_tensor), 3);
  CheckInputTensorValue(input_tensor, 0, string_value_0);
  CheckInputTensorValue(input_tensor, 1, string_value_1);
  CheckInputTensorValue(input_tensor, 2, string_value_2);
}

class ScopedCommandlineArgs {
 public:
  explicit ScopedCommandlineArgs(const std::vector<std::string>& actual_args) {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPSlitePStoolsPSbenchmarkPSbenchmark_testDTcc mht_14(mht_14_v, 535, "", "./tensorflow/lite/tools/benchmark/benchmark_test.cc", "ScopedCommandlineArgs");

    argc_ = actual_args.size() + 1;
    argv_ = new char*[argc_];
    const std::string program_name = "benchmark_model";
    int buffer_size = program_name.length() + 1;
    for (const auto& arg : actual_args) buffer_size += arg.length() + 1;
    buffer_ = new char[buffer_size];
    auto next_start = program_name.copy(buffer_, program_name.length());
    buffer_[next_start++] = '\0';
    argv_[0] = buffer_;
    for (int i = 0; i < actual_args.size(); ++i) {
      const auto& arg = actual_args[i];
      argv_[i + 1] = buffer_ + next_start;
      next_start += arg.copy(argv_[i + 1], arg.length());
      buffer_[next_start++] = '\0';
    }
  }
  ~ScopedCommandlineArgs() {
   std::vector<std::string> mht_15_v;
   MHTracer_DTPStensorflowPSlitePStoolsPSbenchmarkPSbenchmark_testDTcc mht_15(mht_15_v, 555, "", "./tensorflow/lite/tools/benchmark/benchmark_test.cc", "~ScopedCommandlineArgs");

    delete[] argv_;
    delete[] buffer_;
  }

  int argc() const {
   std::vector<std::string> mht_16_v;
   MHTracer_DTPStensorflowPSlitePStoolsPSbenchmarkPSbenchmark_testDTcc mht_16(mht_16_v, 563, "", "./tensorflow/lite/tools/benchmark/benchmark_test.cc", "argc");
 return argc_; }

  char** argv() const {
   std::vector<std::string> mht_17_v;
   MHTracer_DTPStensorflowPSlitePStoolsPSbenchmarkPSbenchmark_testDTcc mht_17(mht_17_v, 568, "", "./tensorflow/lite/tools/benchmark/benchmark_test.cc", "argv");
 return argv_; }

 private:
  char* buffer_;  // the buffer for all arguments.
  int argc_;
  char** argv_;  // Each char* element points to each argument.
};

TEST(BenchmarkTest, RunWithCorrectFlags) {
  ASSERT_THAT(g_fp32_model_path, testing::NotNull());
  TestBenchmark benchmark(CreateFp32Params());
  ScopedCommandlineArgs scoped_argv({"--num_threads=4"});
  auto status = benchmark.Run(scoped_argv.argc(), scoped_argv.argv());
  EXPECT_EQ(kTfLiteOk, status);
}

TEST(BenchmarkTest, RunWithWrongFlags) {
  ASSERT_THAT(g_fp32_model_path, testing::NotNull());
  TestBenchmark benchmark(CreateFp32Params());
  ScopedCommandlineArgs scoped_argv({"--num_threads=str"});
  auto status = benchmark.Run(scoped_argv.argc(), scoped_argv.argv());
  EXPECT_EQ(kTfLiteError, status);
}

TEST(BenchmarkTest, RunWithUseCaching) {
  ASSERT_THAT(g_fp32_model_path, testing::NotNull());
  TestBenchmark benchmark(CreateFp32Params());
  ScopedCommandlineArgs scoped_argv({"--use_caching=false"});
  auto status = benchmark.Run(scoped_argv.argc(), scoped_argv.argv());
  EXPECT_EQ(kTfLiteOk, status);
}

class MaxDurationWorksTestListener : public BenchmarkListener {
  void OnBenchmarkEnd(const BenchmarkResults& results) override {
   std::vector<std::string> mht_18_v;
   MHTracer_DTPStensorflowPSlitePStoolsPSbenchmarkPSbenchmark_testDTcc mht_18(mht_18_v, 604, "", "./tensorflow/lite/tools/benchmark/benchmark_test.cc", "OnBenchmarkEnd");

    const int64_t num_actual_runs = results.inference_time_us().count();
    TFLITE_LOG(INFO) << "number of actual runs: " << num_actual_runs;
    EXPECT_GE(num_actual_runs, 1);
    EXPECT_LT(num_actual_runs, 100000000);
  }
};

TEST(BenchmarkTest, MaxDurationWorks) {
  ASSERT_THAT(g_fp32_model_path, testing::NotNull());
  TestBenchmark benchmark(CreateParams(100000000 /* num_runs */,
                                       1000000.0f /* min_secs */,
                                       0.001f /* max_secs */));
  MaxDurationWorksTestListener listener;
  benchmark.AddListener(&listener);
  benchmark.Run();
}

TEST(BenchmarkTest, ParametersArePopulatedWhenInputShapeIsNotSpecified) {
  ASSERT_THAT(g_fp32_model_path, testing::NotNull());

  TestBenchmark benchmark(CreateParams());
  benchmark.Init();
  benchmark.Prepare();

  auto interpreter = benchmark.GetInterpreter();
  auto inputs = interpreter->inputs();
  ASSERT_GE(inputs.size(), 1);
  auto input_tensor = interpreter->tensor(inputs[0]);

  // Copy input tensor to a vector
  std::vector<char> input_bytes(input_tensor->data.raw,
                                input_tensor->data.raw + input_tensor->bytes);

  benchmark.Prepare();

  // Expect data is not the same.
  EXPECT_EQ(input_bytes.size(), input_tensor->bytes);
  EXPECT_FALSE(absl::equal(input_bytes.begin(), input_bytes.end(),
                           input_tensor->data.raw,
                           input_tensor->data.raw + input_tensor->bytes));
}

}  // namespace
}  // namespace benchmark
}  // namespace tflite

int main(int argc, char** argv) {
   std::vector<std::string> mht_19_v;
   MHTracer_DTPStensorflowPSlitePStoolsPSbenchmarkPSbenchmark_testDTcc mht_19(mht_19_v, 654, "", "./tensorflow/lite/tools/benchmark/benchmark_test.cc", "main");

  std::string fp32_model_path, int8_model_path, string_model_path;
  std::vector<tflite::Flag> flags = {
      tflite::Flag::CreateFlag("fp32_graph", &fp32_model_path,
                               "Path to a fp32 model file."),
      tflite::Flag::CreateFlag("int8_graph", &int8_model_path,
                               "Path to a int8 model file."),
      tflite::Flag::CreateFlag("string_graph", &string_model_path,
                               "Path to a string model file."),
  };

  g_fp32_model_path = &fp32_model_path;
  g_int8_model_path = &int8_model_path;
  g_string_model_path = &string_model_path;

  const bool parse_result =
      tflite::Flags::Parse(&argc, const_cast<const char**>(argv), flags);
  if (!parse_result) {
    std::cerr << tflite::Flags::Usage(argv[0], flags);
    return 1;
  }

  ::tflite::LogToStderr();
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
