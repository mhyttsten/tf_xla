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
class MHTracer_DTPStensorflowPSlitePSmodel_testDTcc {
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
   MHTracer_DTPStensorflowPSlitePSmodel_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePSmodel_testDTcc() {
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
#include "tensorflow/lite/model.h"

#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include <fstream>
#include <iterator>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <gtest/gtest.h>
#include "flatbuffers/flatbuffers.h"  // from @flatbuffers
#include "tensorflow/lite/allocation.h"
#include "tensorflow/lite/core/api/error_reporter.h"
#include "tensorflow/lite/core/api/op_resolver.h"
#include "tensorflow/lite/core/api/verifier.h"
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/interpreter_builder.h"
#include "tensorflow/lite/interpreter_test_util.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model_builder.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/string_type.h"
#include "tensorflow/lite/string_util.h"
#include "tensorflow/lite/testing/util.h"

// Comparison for TfLiteRegistration. Since TfLiteRegistration is a C object,
// we must declare this in global namespace, so argument-dependent operator
// lookup works.
inline bool operator==(const TfLiteRegistration& a,
                       const TfLiteRegistration& b) {
  return a.invoke == b.invoke && a.init == b.init && a.prepare == b.prepare &&
         a.free == b.free;
}

namespace tflite {

// Provide a dummy operation that does nothing.
namespace {
void* dummy_init(TfLiteContext*, const char*, size_t) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePSmodel_testDTcc mht_0(mht_0_v, 227, "", "./tensorflow/lite/model_test.cc", "dummy_init");
 return nullptr; }
void dummy_free(TfLiteContext*, void*) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSlitePSmodel_testDTcc mht_1(mht_1_v, 231, "", "./tensorflow/lite/model_test.cc", "dummy_free");
}
TfLiteStatus dummy_resize(TfLiteContext*, TfLiteNode*) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSlitePSmodel_testDTcc mht_2(mht_2_v, 235, "", "./tensorflow/lite/model_test.cc", "dummy_resize");
 return kTfLiteOk; }
TfLiteStatus dummy_invoke(TfLiteContext*, TfLiteNode*) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPSlitePSmodel_testDTcc mht_3(mht_3_v, 239, "", "./tensorflow/lite/model_test.cc", "dummy_invoke");
 return kTfLiteOk; }
TfLiteRegistration dummy_reg = {dummy_init, dummy_free, dummy_resize,
                                dummy_invoke};
}  // namespace

// Provide a trivial resolver that returns a constant value no matter what
// op is asked for.
class TrivialResolver : public OpResolver {
 public:
  explicit TrivialResolver(TfLiteRegistration* constant_return = nullptr)
      : constant_return_(constant_return) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPSlitePSmodel_testDTcc mht_4(mht_4_v, 252, "", "./tensorflow/lite/model_test.cc", "TrivialResolver");
}
  // Find the op registration of a custom operator by op name.
  const TfLiteRegistration* FindOp(tflite::BuiltinOperator op,
                                   int version) const override {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPSlitePSmodel_testDTcc mht_5(mht_5_v, 258, "", "./tensorflow/lite/model_test.cc", "FindOp");

    return constant_return_;
  }
  // Find the op registration of a custom operator by op name.
  const TfLiteRegistration* FindOp(const char* op, int version) const override {
   std::vector<std::string> mht_6_v;
   mht_6_v.push_back("op: \"" + (op == nullptr ? std::string("nullptr") : std::string((char*)op)) + "\"");
   MHTracer_DTPStensorflowPSlitePSmodel_testDTcc mht_6(mht_6_v, 266, "", "./tensorflow/lite/model_test.cc", "FindOp");

    return constant_return_;
  }

 private:
  TfLiteRegistration* constant_return_;
};

TEST(BasicFlatBufferModel, TestNonExistentFiles) {
  ASSERT_TRUE(!FlatBufferModel::BuildFromFile("/tmp/tflite_model_1234"));
}

TEST(BasicFlatBufferModel, TestBufferAlignment) {
  // On 32-bit ARM buffers are required to be 4-bytes aligned, on other
  // platforms there is no alignment requirement.
  const uintptr_t kAlignment = 4;
  const uintptr_t kAlignmentBits = kAlignment - 1;

  // Use real model data so that we can be sure error is only from the
  // alignment requirement and not from bad data.
  std::ifstream fp("tensorflow/lite/testdata/empty_model.bin");
  ASSERT_TRUE(fp.good());
  std::string empty_model_data((std::istreambuf_iterator<char>(fp)),
                               std::istreambuf_iterator<char>());
  auto free_chars = [](char* p) {
   std::vector<std::string> mht_7_v;
   mht_7_v.push_back("p: \"" + (p == nullptr ? std::string("nullptr") : std::string((char*)p)) + "\"");
   MHTracer_DTPStensorflowPSlitePSmodel_testDTcc mht_7(mht_7_v, 294, "", "./tensorflow/lite/model_test.cc", "lambda");
 free(p); };
  std::unique_ptr<char, decltype(free_chars)> buffer(
      reinterpret_cast<char*>(malloc(empty_model_data.size() + kAlignment)),
      free_chars);

  // Check that aligned buffer works (no other errors in the test).
  char* aligned = reinterpret_cast<char*>(
      (reinterpret_cast<uintptr_t>(buffer.get()) + kAlignment) &
      ~kAlignmentBits);
  memcpy(aligned, empty_model_data.c_str(), empty_model_data.size());
  EXPECT_TRUE(
      FlatBufferModel::BuildFromBuffer(aligned, empty_model_data.size()));

  // Check unaligned buffer handling.
  char* unaligned =
      reinterpret_cast<char*>(reinterpret_cast<uintptr_t>(buffer.get()) | 0x1);
  memcpy(unaligned, empty_model_data.c_str(), empty_model_data.size());
#ifdef __arm__
  EXPECT_FALSE(
      FlatBufferModel::BuildFromBuffer(unaligned, empty_model_data.size()));
#else   // !__arm__
  EXPECT_TRUE(
      FlatBufferModel::BuildFromBuffer(unaligned, empty_model_data.size()));
#endif  // __arm__
}

// Make sure a model with nothing in it loads properly.
TEST(BasicFlatBufferModel, TestEmptyModels) {
  auto model = FlatBufferModel::BuildFromFile(
      "tensorflow/lite/testdata/empty_model.bin");
  ASSERT_TRUE(model);
  // Now try to build it into a model.
  std::unique_ptr<Interpreter> interpreter;
  ASSERT_EQ(InterpreterBuilder(*model, TrivialResolver())(&interpreter),
            kTfLiteOk);
  ASSERT_NE(interpreter, nullptr);
}

TEST(BasicFlatBufferModel, TestNullDestination) {
  auto model = FlatBufferModel::BuildFromFile(
      "tensorflow/lite/testdata/empty_model.bin");
  ASSERT_TRUE(model);
  // Test that building with null destination fails.
  ASSERT_NE(InterpreterBuilder(*model, TrivialResolver())(nullptr), kTfLiteOk);
}

// Make sure currently unsupported # of subgraphs are checked
// TODO(aselle): Replace this test when multiple subgraphs are supported.
TEST(BasicFlatBufferModel, TestZeroSubgraphs) {
  auto m = FlatBufferModel::BuildFromFile(
      "tensorflow/lite/testdata/0_subgraphs.bin");
  ASSERT_TRUE(m);
  std::unique_ptr<Interpreter> interpreter;
  ASSERT_NE(InterpreterBuilder(*m, TrivialResolver())(&interpreter), kTfLiteOk);
}

TEST(BasicFlatBufferModel, TestMultipleSubgraphs) {
  auto m = FlatBufferModel::BuildFromFile(
      "tensorflow/lite/testdata/2_subgraphs.bin");
  ASSERT_TRUE(m);
  std::unique_ptr<Interpreter> interpreter;
  ASSERT_EQ(InterpreterBuilder(*m, TrivialResolver())(&interpreter), kTfLiteOk);
  EXPECT_EQ(interpreter->subgraphs_size(), 2);
}

TEST(BasicFlatBufferModel, TestSubgraphName) {
  auto m = FlatBufferModel::BuildFromFile(
      "tensorflow/lite/testdata/"
      "2_subgraphs_dont_delegate_name.bin");
  ASSERT_TRUE(m);
  std::unique_ptr<Interpreter> interpreter;
  ASSERT_EQ(InterpreterBuilder(*m, TrivialResolver())(&interpreter), kTfLiteOk);
  EXPECT_EQ(interpreter->subgraphs_size(), 2);
  EXPECT_EQ(interpreter->subgraph(0)->GetName(), "");
  EXPECT_EQ(interpreter->subgraph(1)->GetName(), "VALIDATION:main");
}

// Test what happens if we cannot bind any of the ops.
TEST(BasicFlatBufferModel, TestModelWithoutNullRegistrations) {
  auto model = FlatBufferModel::BuildFromFile(
      "tensorflow/lite/testdata/test_model.bin");
  ASSERT_TRUE(model);
  // Check that we get an error code and interpreter pointer is reset.
  std::unique_ptr<Interpreter> interpreter(new Interpreter);
  ASSERT_NE(InterpreterBuilder(*model, TrivialResolver(nullptr))(&interpreter),
            kTfLiteOk);
  ASSERT_EQ(interpreter, nullptr);
}

// Make sure model is read to interpreter properly
TEST(BasicFlatBufferModel, TestModelInInterpreter) {
  auto model = FlatBufferModel::BuildFromFile(
      "tensorflow/lite/testdata/test_model.bin");
  ASSERT_TRUE(model);
  // Check that we get an error code and interpreter pointer is reset.
  std::unique_ptr<Interpreter> interpreter(new Interpreter);
  ASSERT_EQ(
      InterpreterBuilder(*model, TrivialResolver(&dummy_reg))(&interpreter),
      kTfLiteOk);
  ASSERT_NE(interpreter, nullptr);
  ASSERT_EQ(interpreter->tensors_size(), 4);
  ASSERT_EQ(interpreter->nodes_size(), 2);
  std::vector<int> inputs = {0, 1};
  std::vector<int> outputs = {2, 3};
  ASSERT_EQ(interpreter->inputs(), inputs);
  ASSERT_EQ(interpreter->outputs(), outputs);

  EXPECT_EQ(std::string(interpreter->GetInputName(0)), "input0");
  EXPECT_EQ(std::string(interpreter->GetInputName(1)), "input1");
  EXPECT_EQ(std::string(interpreter->GetOutputName(0)), "out1");
  EXPECT_EQ(std::string(interpreter->GetOutputName(1)), "out2");

  // Make sure all input tensors are correct
  TfLiteTensor* i0 = interpreter->tensor(0);
  ASSERT_EQ(i0->type, kTfLiteFloat32);
  ASSERT_NE(i0->data.raw, nullptr);  // mmapped
  ASSERT_EQ(i0->allocation_type, kTfLiteMmapRo);
  TfLiteTensor* i1 = interpreter->tensor(1);
  ASSERT_EQ(i1->type, kTfLiteFloat32);
  ASSERT_EQ(i1->data.raw, nullptr);
  ASSERT_EQ(i1->allocation_type, kTfLiteArenaRw);
  TfLiteTensor* o0 = interpreter->tensor(2);
  ASSERT_EQ(o0->type, kTfLiteFloat32);
  ASSERT_EQ(o0->data.raw, nullptr);
  ASSERT_EQ(o0->allocation_type, kTfLiteArenaRw);
  TfLiteTensor* o1 = interpreter->tensor(3);
  ASSERT_EQ(o1->type, kTfLiteFloat32);
  ASSERT_EQ(o1->data.raw, nullptr);
  ASSERT_EQ(o1->allocation_type, kTfLiteArenaRw);

  // Check op 0 which has inputs {0, 1} outputs {2}.
  {
    const std::pair<TfLiteNode, TfLiteRegistration>* node_and_reg0 =
        interpreter->node_and_registration(0);
    ASSERT_NE(node_and_reg0, nullptr);
    const TfLiteNode& node0 = node_and_reg0->first;
    const TfLiteRegistration& reg0 = node_and_reg0->second;
    TfLiteIntArray* desired_inputs = TfLiteIntArrayCreate(2);
    desired_inputs->data[0] = 0;
    desired_inputs->data[1] = 1;
    TfLiteIntArray* desired_outputs = TfLiteIntArrayCreate(1);
    desired_outputs->data[0] = 2;
    ASSERT_TRUE(TfLiteIntArrayEqual(node0.inputs, desired_inputs));
    ASSERT_TRUE(TfLiteIntArrayEqual(node0.outputs, desired_outputs));
    TfLiteIntArrayFree(desired_inputs);
    TfLiteIntArrayFree(desired_outputs);
    ASSERT_EQ(reg0, dummy_reg);
  }

  // Check op 1 which has inputs {2} outputs {3}.
  {
    const std::pair<TfLiteNode, TfLiteRegistration>* node_and_reg1 =
        interpreter->node_and_registration(1);
    ASSERT_NE(node_and_reg1, nullptr);
    const TfLiteNode& node1 = node_and_reg1->first;
    const TfLiteRegistration& reg1 = node_and_reg1->second;
    TfLiteIntArray* desired_inputs = TfLiteIntArrayCreate(1);
    TfLiteIntArray* desired_outputs = TfLiteIntArrayCreate(1);
    desired_inputs->data[0] = 2;
    desired_outputs->data[0] = 3;
    ASSERT_TRUE(TfLiteIntArrayEqual(node1.inputs, desired_inputs));
    ASSERT_TRUE(TfLiteIntArrayEqual(node1.outputs, desired_outputs));
    TfLiteIntArrayFree(desired_inputs);
    TfLiteIntArrayFree(desired_outputs);
    ASSERT_EQ(reg1, dummy_reg);
  }
}

TEST(BasicFlatBufferModel, TestWithNumThreads) {
  TestErrorReporter reporter;
  auto model = FlatBufferModel::BuildFromFile(
      "tensorflow/lite/testdata/test_model.bin", &reporter);
  ASSERT_TRUE(model);
  TrivialResolver resolver(&dummy_reg);
  InterpreterBuilder builder(*model, resolver);

  std::unique_ptr<Interpreter> interpreter;
  ASSERT_EQ(builder(&interpreter, /*num_threads=*/42), kTfLiteOk);
  ASSERT_NE(interpreter, nullptr);
  ASSERT_EQ(interpreter->subgraph(0)->context()->recommended_num_threads, 42);

  interpreter.reset();
  ASSERT_EQ(builder(&interpreter, 0), kTfLiteOk);
  ASSERT_NE(interpreter, nullptr);
  ASSERT_EQ(interpreter->subgraph(0)->context()->recommended_num_threads, 1);

  interpreter.reset();
  ASSERT_EQ(builder(&interpreter, -1), kTfLiteOk);
  ASSERT_NE(interpreter, nullptr);
  ASSERT_EQ(interpreter->subgraph(0)->context()->recommended_num_threads, -1);

  ASSERT_EQ(reporter.num_calls(), 0);
  interpreter.reset(new Interpreter);
  ASSERT_EQ(builder(&interpreter, -2), kTfLiteError);
  ASSERT_EQ(interpreter, nullptr);
  ASSERT_EQ(reporter.num_calls(), 1);
  ASSERT_PRED_FORMAT2(testing::IsSubstring,
                      "num_threads should be >= 0 or just -1",
                      reporter.error_messages());
}

TEST(BasicFlatBufferModel, TestSetNumThreads) {
  TestErrorReporter reporter;
  auto model = FlatBufferModel::BuildFromFile(
      "tensorflow/lite/testdata/test_model.bin", &reporter);
  ASSERT_TRUE(model);
  std::unique_ptr<Interpreter> interpreter;
  TrivialResolver resolver(&dummy_reg);
  InterpreterBuilder builder(*model, resolver);

  ASSERT_EQ(builder.SetNumThreads(42), kTfLiteOk);
  interpreter.reset();
  ASSERT_EQ(builder(&interpreter), kTfLiteOk);
  ASSERT_NE(interpreter, nullptr);

  ASSERT_EQ(builder.SetNumThreads(0), kTfLiteOk);
  interpreter.reset();
  ASSERT_EQ(builder(&interpreter), kTfLiteOk);
  ASSERT_NE(interpreter, nullptr);

  ASSERT_EQ(builder.SetNumThreads(-1), kTfLiteOk);
  interpreter.reset();
  ASSERT_EQ(builder(&interpreter), kTfLiteOk);
  ASSERT_NE(interpreter, nullptr);

  ASSERT_EQ(reporter.num_calls(), 0);
  ASSERT_EQ(builder.SetNumThreads(-2), kTfLiteError);
  interpreter.reset();
  ASSERT_EQ(builder(&interpreter), kTfLiteOk);
  ASSERT_NE(interpreter, nullptr);
  ASSERT_EQ(reporter.num_calls(), 1);
  ASSERT_PRED_FORMAT2(testing::IsSubstring,
                      "num_threads should be >= 0 or just -1",
                      reporter.error_messages());
}

TEST(BasicFlatBufferModel, TestSetNumThreadsWithMultipleSubgraphs) {
  TestErrorReporter reporter;
  auto model = FlatBufferModel::BuildFromFile(
      "tensorflow/lite/testdata/2_subgraphs.bin", &reporter);
  ASSERT_TRUE(model);
  std::unique_ptr<Interpreter> interpreter;
  TrivialResolver resolver(&dummy_reg);
  InterpreterBuilder builder(*model, resolver);

  ASSERT_EQ(builder.SetNumThreads(4), kTfLiteOk);
  interpreter.reset();
  ASSERT_EQ(builder(&interpreter), kTfLiteOk);
  ASSERT_NE(interpreter, nullptr);

  // Check that each subgraph has the expected number of threads set.
  for (int i = 0; i < interpreter->subgraphs_size(); ++i) {
    EXPECT_EQ(interpreter->subgraph(i)->context()->recommended_num_threads, 4);
  }
}

// Test that loading a model with TensorFlow ops fails when the flex delegate is
// not linked into the target.
TEST(FlexModel, FailureWithoutFlexDelegate) {
  auto model = FlatBufferModel::BuildFromFile(
      "tensorflow/lite/testdata/multi_add_flex.bin");
  ASSERT_TRUE(model);

  // Note that creation will succeed when using the BuiltinOpResolver, but
  // unless the appropriate delegate is linked into the target or the client
  // explicitly installs the delegate, execution will fail.
  std::unique_ptr<Interpreter> interpreter;
  ASSERT_EQ(InterpreterBuilder(*model,
                               ops::builtin::BuiltinOpResolver{})(&interpreter),
            kTfLiteOk);
  ASSERT_TRUE(interpreter);

  // As the flex ops weren't resolved implicitly by the flex delegate, runtime
  // allocation and execution will fail.
  ASSERT_EQ(interpreter->AllocateTensors(), kTfLiteUnresolvedOps);
}

// This tests on a flatbuffer that defines a shape of 2 to be a memory mapped
// buffer. But the buffer is provided to be only 1 element.
TEST(BasicFlatBufferModel, TestBrokenMmap) {
  ASSERT_FALSE(FlatBufferModel::BuildFromFile(
      "tensorflow/lite/testdata/test_model_broken.bin"));
}

TEST(BasicFlatBufferModel, TestNullModel) {
  // Check that we get an error code and interpreter pointer is reset.
  std::unique_ptr<Interpreter> interpreter(new Interpreter);
  ASSERT_NE(
      InterpreterBuilder(nullptr, TrivialResolver(&dummy_reg))(&interpreter),
      kTfLiteOk);
  ASSERT_EQ(interpreter.get(), nullptr);
}

// Mocks the verifier by setting the result in ctor.
class FakeVerifier : public tflite::TfLiteVerifier {
 public:
  explicit FakeVerifier(bool result) : result_(result) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPSlitePSmodel_testDTcc mht_8(mht_8_v, 593, "", "./tensorflow/lite/model_test.cc", "FakeVerifier");
}
  bool Verify(const char* data, int length,
              tflite::ErrorReporter* reporter) override {
   std::vector<std::string> mht_9_v;
   mht_9_v.push_back("data: \"" + (data == nullptr ? std::string("nullptr") : std::string((char*)data)) + "\"");
   MHTracer_DTPStensorflowPSlitePSmodel_testDTcc mht_9(mht_9_v, 599, "", "./tensorflow/lite/model_test.cc", "Verify");

    return result_;
  }

 private:
  bool result_;
};

TEST(BasicFlatBufferModel, TestWithTrueVerifier) {
  FakeVerifier verifier(true);
  ASSERT_TRUE(FlatBufferModel::VerifyAndBuildFromFile(
      "tensorflow/lite/testdata/test_model.bin", &verifier));
}

TEST(BasicFlatBufferModel, TestWithFalseVerifier) {
  FakeVerifier verifier(false);
  ASSERT_FALSE(FlatBufferModel::VerifyAndBuildFromFile(
      "tensorflow/lite/testdata/test_model.bin", &verifier));
}

TEST(BasicFlatBufferModel, TestWithNullVerifier) {
  ASSERT_TRUE(FlatBufferModel::VerifyAndBuildFromFile(
      "tensorflow/lite/testdata/test_model.bin", nullptr));
}

// This makes sure the ErrorReporter is marshalled from FlatBufferModel to
// the Interpreter.
TEST(BasicFlatBufferModel, TestCustomErrorReporter) {
  TestErrorReporter reporter;
  auto model = FlatBufferModel::BuildFromFile(
      "tensorflow/lite/testdata/empty_model.bin", &reporter);
  ASSERT_TRUE(model);

  std::unique_ptr<Interpreter> interpreter;
  TrivialResolver resolver;
  InterpreterBuilder(*model, resolver)(&interpreter);
  ASSERT_NE(interpreter->Invoke(), kTfLiteOk);
  ASSERT_EQ(reporter.num_calls(), 1);
}

// This makes sure the ErrorReporter is marshalled from FlatBufferModel to
// the Interpreter.
TEST(BasicFlatBufferModel, TestNullErrorReporter) {
  auto model = FlatBufferModel::BuildFromFile(
      "tensorflow/lite/testdata/empty_model.bin", nullptr);
  ASSERT_TRUE(model);

  std::unique_ptr<Interpreter> interpreter;
  TrivialResolver resolver;
  InterpreterBuilder(*model, resolver)(&interpreter);
  ASSERT_NE(interpreter->Invoke(), kTfLiteOk);
}

// Test that loading model directly from a Model flatbuffer works.
TEST(BasicFlatBufferModel, TestBuildFromModel) {
  TestErrorReporter reporter;
  FileCopyAllocation model_allocation(
      "tensorflow/lite/testdata/test_model.bin", &reporter);
  ASSERT_TRUE(model_allocation.valid());
  ::flatbuffers::Verifier verifier(
      reinterpret_cast<const uint8_t*>(model_allocation.base()),
      model_allocation.bytes());
  ASSERT_TRUE(VerifyModelBuffer(verifier));
  const Model* model_fb = ::tflite::GetModel(model_allocation.base());

  auto model = FlatBufferModel::BuildFromModel(model_fb);
  ASSERT_TRUE(model);

  std::unique_ptr<Interpreter> interpreter;
  ASSERT_EQ(
      InterpreterBuilder(*model, TrivialResolver(&dummy_reg))(&interpreter),
      kTfLiteOk);
  ASSERT_NE(interpreter, nullptr);
}

// Test that loading model directly from an Allocation works.
TEST(BasicFlatBufferModel, TestBuildFromAllocation) {
  TestErrorReporter reporter;
  std::unique_ptr<Allocation> model_allocation(new FileCopyAllocation(
      "tensorflow/lite/testdata/test_model.bin", &reporter));
  ASSERT_TRUE(model_allocation->valid());

  auto model =
      FlatBufferModel::BuildFromAllocation(std::move(model_allocation));
  ASSERT_TRUE(model);

  std::unique_ptr<Interpreter> interpreter;
  ASSERT_EQ(
      InterpreterBuilder(*model, TrivialResolver(&dummy_reg))(&interpreter),
      kTfLiteOk);
  ASSERT_NE(interpreter, nullptr);
}

TEST(BasicFlatBufferModel, TestBuildFromNullAllocation) {
  TestErrorReporter reporter;
  std::unique_ptr<Allocation> model_allocation;

  auto model =
      FlatBufferModel::BuildFromAllocation(std::move(model_allocation));
  ASSERT_FALSE(model);
}

TEST(BasicFlatBufferModel, TestBuildFromInvalidAllocation) {
  TestErrorReporter reporter;
  std::unique_ptr<Allocation> model_allocation(
      new MemoryAllocation(nullptr, 0, nullptr));

  auto model =
      FlatBufferModel::BuildFromAllocation(std::move(model_allocation));
  ASSERT_FALSE(model);
}

// Test reading the minimum runtime string from metadata in a Model flatbuffer.
TEST(BasicFlatBufferModel, TestReadRuntimeVersionFromModel) {
  // First read a model that doesn't have the runtime string.
  auto model1 = FlatBufferModel::BuildFromFile(
      "tensorflow/lite/testdata/test_model.bin");
  ASSERT_TRUE(model1);
  ASSERT_EQ(model1->GetMinimumRuntime(), "");

  // Read a model that has minimum runtime string populated.
  auto model2 = FlatBufferModel::BuildFromFile(
      "tensorflow/lite/testdata/test_min_runtime.bin");
  ASSERT_TRUE(model2);
  // Check that we have read the runtime string correctly.
  ASSERT_EQ(model2->GetMinimumRuntime(), "1.5.0");
}

// Test reading all metadata from the model
TEST(BasicFlatBufferModel, TestReadMetadataFromModel) {
  // First read a model that doesn't have the runtime string.
  auto model1 = FlatBufferModel::BuildFromFile(
      "tensorflow/lite/testdata/test_model.bin");
  ASSERT_TRUE(model1);
  std::map<std::string, std::string> metadata = model1->ReadAllMetadata();
  ASSERT_EQ(metadata.size(), 0);

  // Read a model that has reduced precision support mask populated
  auto model2 = FlatBufferModel::BuildFromFile(
      "tensorflow/lite/testdata/test_model_redux_precision.bin");
  ASSERT_TRUE(model2);
  // Check that we have read the runtime string correctly.
  metadata = model2->ReadAllMetadata();
  ASSERT_EQ(metadata["reduced_precision_support"], "fp16bf16accfp32");
}

TEST(BasicFlatBufferModel, TestReadMetadataFromContext) {
  const std::string reduced_precision_meta_key = "reduced_precision_support";
  // First read a model that doesn't have any metadata.
  auto model1 = FlatBufferModel::BuildFromFile(
      "tensorflow/lite/testdata/test_model.bin");
  ASSERT_TRUE(model1);
  std::unique_ptr<Interpreter> interpreter;
  TrivialResolver resolver(&dummy_reg);
  InterpreterBuilder builder1(*model1, resolver);
  interpreter.reset();
  ASSERT_EQ(builder1(&interpreter), kTfLiteOk);
  ASSERT_NE(interpreter, nullptr);

  const char* ptr = nullptr;
  size_t bytes;
  auto* context = interpreter->subgraph(0)->context();
  ASSERT_EQ(context->GetModelMetadata(
                context, reduced_precision_meta_key.c_str(), &ptr, &bytes),
            kTfLiteError);

  // This model has metadata mapped to kTfLiteReducedPrecisionKey.
  auto model2 = FlatBufferModel::BuildFromFile(
      "tensorflow/lite/testdata/test_model_redux_precision.bin");
  ASSERT_TRUE(model2);
  InterpreterBuilder builder2(*model2, resolver);
  interpreter.reset();
  ASSERT_EQ(builder2(&interpreter), kTfLiteOk);
  ASSERT_NE(interpreter, nullptr);

  context = interpreter->subgraph(0)->context();
  ASSERT_EQ(context->GetModelMetadata(
                context, reduced_precision_meta_key.c_str(), &ptr, &bytes),
            kTfLiteOk);
  ASSERT_EQ(std::string(ptr, bytes), "fp16bf16accfp32");
  ASSERT_EQ(context->GetModelMetadata(context, "unknown_key", &ptr, &bytes),
            kTfLiteError);
}

// The test model has the following tensor encoded in the TACO format:
// [[1, 0, 2, 3],
//  [0, 4, 0, 0],
//  [0, 0, 5, 0],
//  [0, 0, 0, 6]].
// TACO supports multiple encodings like CSR, CSC, etc. We chose to use the one
// similar to the blocked-CSR format with 2x2 row-major dense blocks.
TEST(BasicFlatBufferModel, TestParseModelWithSparseTensor) {
  // The model only has 1 sparse constant tensor.
  auto model = FlatBufferModel::BuildFromFile(
      "tensorflow/lite/testdata/sparse_tensor.bin");
  ASSERT_TRUE(model);

  std::unique_ptr<Interpreter> interpreter(new Interpreter);
  ASSERT_EQ(InterpreterBuilder(*model, TrivialResolver())(&interpreter),
            kTfLiteOk);
  ASSERT_NE(interpreter, nullptr);
  ASSERT_EQ(interpreter->tensors_size(), 2);
  TfLiteTensor* t1 = interpreter->tensor(0);
  ASSERT_EQ(t1->allocation_type, kTfLiteMmapRo);

  TfLiteIntArray* traversal_order = TfLiteIntArrayCreate(4);
  traversal_order->data[0] = 0;
  traversal_order->data[1] = 1;
  traversal_order->data[2] = 2;
  traversal_order->data[3] = 3;
  ASSERT_TRUE(
      TfLiteIntArrayEqual(t1->sparsity->traversal_order, traversal_order));
  TfLiteIntArrayFree(traversal_order);

  TfLiteIntArray* block_map = TfLiteIntArrayCreate(2);
  block_map->data[0] = 0;
  block_map->data[1] = 1;
  ASSERT_TRUE(TfLiteIntArrayEqual(t1->sparsity->block_map, block_map));
  TfLiteIntArrayFree(block_map);

  ASSERT_EQ(t1->sparsity->dim_metadata_size, 4);

  ASSERT_EQ(t1->sparsity->dim_metadata[0].format, kTfLiteDimDense);
  ASSERT_EQ(t1->sparsity->dim_metadata[0].dense_size, 2);
  ASSERT_EQ(t1->sparsity->dim_metadata[0].array_segments, nullptr);
  ASSERT_EQ(t1->sparsity->dim_metadata[0].array_indices, nullptr);

  ASSERT_EQ(t1->sparsity->dim_metadata[1].format, kTfLiteDimSparseCSR);
  ASSERT_EQ(t1->sparsity->dim_metadata[1].dense_size, 0);
  TfLiteIntArray* array_segments = TfLiteIntArrayCreate(3);
  array_segments->data[0] = 0;
  array_segments->data[1] = 2;
  array_segments->data[2] = 3;
  ASSERT_TRUE(TfLiteIntArrayEqual(t1->sparsity->dim_metadata[1].array_segments,
                                  array_segments));
  TfLiteIntArrayFree(array_segments);

  TfLiteIntArray* array_indices = TfLiteIntArrayCreate(3);
  array_indices->data[0] = 0;
  array_indices->data[1] = 1;
  array_indices->data[2] = 1;
  ASSERT_TRUE(TfLiteIntArrayEqual(t1->sparsity->dim_metadata[1].array_indices,
                                  array_indices));
  TfLiteIntArrayFree(array_indices);

  ASSERT_EQ(t1->sparsity->dim_metadata[2].format, kTfLiteDimDense);
  ASSERT_EQ(t1->sparsity->dim_metadata[2].dense_size, 2);
  ASSERT_EQ(t1->sparsity->dim_metadata[2].array_segments, nullptr);
  ASSERT_EQ(t1->sparsity->dim_metadata[2].array_indices, nullptr);

  ASSERT_EQ(t1->sparsity->dim_metadata[3].format, kTfLiteDimDense);
  ASSERT_EQ(t1->sparsity->dim_metadata[3].dense_size, 2);
  ASSERT_EQ(t1->sparsity->dim_metadata[3].array_segments, nullptr);
  ASSERT_EQ(t1->sparsity->dim_metadata[3].array_indices, nullptr);
}

// TODO(b/150072943): Add malformed model with sparse tensor tests.

// The models here have at least a node that uses the same tensor as input and
// output. This causes segfaults when trying to eval the operator, hence we try
// to prevent this scenario. The earliest place we can check this is in
// `AllocateTensors`, hence the test checks that `interpreter->AllocateTensors`
// detects these bad models.
TEST(BasicFlatBufferModel, TestHandleMalformedModelReuseTensor) {
  const auto model_path =
      "tensorflow/lite/testdata/add_shared_tensors.bin";

  std::unique_ptr<tflite::FlatBufferModel> model =
      FlatBufferModel::BuildFromFile(model_path);
  ASSERT_NE(model, nullptr);

  tflite::ops::builtin::BuiltinOpResolver resolver;
  InterpreterBuilder builder(*model, resolver);
  std::unique_ptr<Interpreter> interpreter;
  ASSERT_EQ(builder(&interpreter), kTfLiteOk);
  ASSERT_NE(interpreter, nullptr);
  ASSERT_NE(interpreter->AllocateTensors(), kTfLiteOk);
}

// The models here have a buffer index for a tensor pointing to a null buffer.
// This results in the tensor being interpreted as read-write, but the model
// assumes the tensor is read-only. As such, `interpreter->Invoke()` would
// segfault if no precondition check is added. The test checks that the
// precondition check exists.
TEST(BasicFlatBufferModel, TestHandleMalformedModelInvalidBuffer) {
  const auto model_path =
      "tensorflow/lite/testdata/segment_sum_invalid_buffer.bin";

  std::unique_ptr<tflite::FlatBufferModel> model =
      FlatBufferModel::BuildFromFile(model_path);
  ASSERT_NE(model, nullptr);

  tflite::ops::builtin::BuiltinOpResolver resolver;
  InterpreterBuilder builder(*model, resolver);
  std::unique_ptr<Interpreter> interpreter;
  ASSERT_EQ(builder(&interpreter), kTfLiteOk);
  ASSERT_NE(interpreter, nullptr);
  ASSERT_EQ(interpreter->AllocateTensors(), kTfLiteOk);
  ASSERT_NE(interpreter->Invoke(), kTfLiteOk);
}

TEST(TestAddDelegateOwnership, AddDelegateDoesNotTakeOwnership) {
  class TestDelegate : public TfLiteDelegate {
   public:
    TestDelegate(bool* destroyed, bool* prepared)
        : TfLiteDelegate(TfLiteDelegateCreate()),
          destroyed_(destroyed),
          prepared_(prepared) {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPSlitePSmodel_testDTcc mht_10(mht_10_v, 909, "", "./tensorflow/lite/model_test.cc", "TestDelegate");

      flags = kTfLiteDelegateFlagsNone;
      Prepare = [](TfLiteContext*, TfLiteDelegate* delegate) -> TfLiteStatus {
        *(static_cast<TestDelegate*>(delegate)->prepared_) = true;
        return kTfLiteOk;
      };
    }
    ~TestDelegate() {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPSlitePSmodel_testDTcc mht_11(mht_11_v, 919, "", "./tensorflow/lite/model_test.cc", "~TestDelegate");
 *destroyed_ = true; }

   private:
    bool* destroyed_;
    bool* prepared_;
  };

  // Construct a delegate with flags for indicating preparation/destruction.
  bool destroyed = false;
  bool prepared = false;
  {
    std::unique_ptr<TestDelegate> delegate(
        new TestDelegate(&destroyed, &prepared));
    {
      // Load a model.
      auto model = FlatBufferModel::BuildFromFile(
          "tensorflow/lite/testdata/empty_model.bin");
      ASSERT_TRUE(model);
      // Now try to build it into an interpreter.
      std::unique_ptr<Interpreter> interpreter;
      InterpreterBuilder builder(*model, TrivialResolver());
      builder.AddDelegate(delegate.get());  // Does not transfer ownership.
      // Loop to check we can construct multiple interpreters from one builder.
      for (int i = 0; i < 3; i++) {
        prepared = false;
        ASSERT_EQ(builder(&interpreter), kTfLiteOk);
        ASSERT_NE(interpreter, nullptr);

        // The delegate should be prepared as normal, and should be preserved.
        EXPECT_TRUE(prepared);
        EXPECT_FALSE(destroyed);

        // Interpreter interaction should not impact the delegate's validity.
        interpreter->AllocateTensors();
        interpreter->Invoke();
        EXPECT_FALSE(destroyed);
      }
    }
    EXPECT_NE(delegate, nullptr);
    EXPECT_FALSE(destroyed);
  }
  // Only after the delegate itself goes out of scope should the delegate be
  // destroyed.
  EXPECT_TRUE(destroyed);
}

// The model contains a while loop with a forwarding string input. This test
// makes sure that the dynamic tensor existence in the while subgraph's outputs
// is detected. If not, the while loop will be failed at handling the dynamic
// tensor handling as a static tensor.
TEST(BasicFlatBufferModel, TestHandleModelWithWhileOpContainsForwardingInput) {
  const auto model_path =
      "tensorflow/lite/testdata/while_op_with_forwarding_input.bin";

  std::unique_ptr<tflite::FlatBufferModel> model =
      FlatBufferModel::BuildFromFile(model_path);
  ASSERT_NE(model, nullptr);

  tflite::ops::builtin::BuiltinOpResolver resolver;
  InterpreterBuilder builder(*model, resolver);
  std::unique_ptr<Interpreter> interpreter;
  ASSERT_EQ(builder(&interpreter), kTfLiteOk);
  ASSERT_NE(interpreter, nullptr);
  ASSERT_EQ(interpreter->AllocateTensors(), kTfLiteOk);

  int32_t* tensor_data = interpreter->typed_tensor<int32_t>(0);
  tensor_data[0] = 20;

  auto tensor = interpreter->tensor(1);
  DynamicBuffer buf;
  buf.AddString("a", 1);
  buf.WriteToTensor(tensor, /*new_shape=*/nullptr);

  ASSERT_EQ(interpreter->Invoke(), kTfLiteOk);
}

// TODO(aselle): Add tests for serialization of builtin op data types.
// These tests will occur with the evaluation tests of individual operators,
// not here.

}  // namespace tflite
