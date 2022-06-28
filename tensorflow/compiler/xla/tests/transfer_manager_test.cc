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
class MHTracer_DTPStensorflowPScompilerPSxlaPStestsPStransfer_manager_testDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSxlaPStestsPStransfer_manager_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSxlaPStestsPStransfer_manager_testDTcc() {
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

#include <memory>
#include <string>
#include <vector>

#include "tensorflow/compiler/xla/layout_util.h"
#include "tensorflow/compiler/xla/literal.h"
#include "tensorflow/compiler/xla/service/generic_transfer_manager.h"
#include "tensorflow/compiler/xla/service/hlo_parser.h"
#include "tensorflow/compiler/xla/service/shaped_buffer.h"
#include "tensorflow/compiler/xla/service/stream_pool.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/compiler/xla/tests/literal_test_util.h"
#include "tensorflow/compiler/xla/tests/local_client_test_base.h"
#include "tensorflow/compiler/xla/tests/test_macros.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/stream_executor_no_cuda.h"
#include "tensorflow/core/platform/test_benchmark.h"
#include "tensorflow/stream_executor/device_memory_allocator.h"

namespace xla {
namespace {

class TransferManagerTest : public LocalClientTestBase {
 protected:
  TransferManagerTest()
      : shape_size_fn_([this](const Shape& shape) {
          return transfer_manager_->GetByteSizeRequirement(shape);
        }) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPStestsPStransfer_manager_testDTcc mht_0(mht_0_v, 215, "", "./tensorflow/compiler/xla/tests/transfer_manager_test.cc", "TransferManagerTest");

    stream_ptr_ = local_client_->mutable_backend()
                      ->BorrowStream(stream_executor_)
                      .ValueOrDie();
    stream_ = stream_ptr_.get();
  }

  ~TransferManagerTest() override = default;

  ScopedShapedBuffer AllocateDeviceBuffer(const Shape& shape) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPStestsPStransfer_manager_testDTcc mht_1(mht_1_v, 227, "", "./tensorflow/compiler/xla/tests/transfer_manager_test.cc", "AllocateDeviceBuffer");

    return transfer_manager_
        ->AllocateScopedShapedBuffer(
            shape, GetOrCreateAllocator(local_client_->platform()),
            /*device_ordinal=*/0)
        .ValueOrDie();
  }

 protected:
  StreamPool::Ptr stream_ptr_;
  se::Stream* stream_;

 private:
  std::function<int64_t(const Shape&)> shape_size_fn_;
};

XLA_TEST_F(TransferManagerTest, TransferR0U32) {
  Literal literal = LiteralUtil::CreateR0<uint32_t>(42);
  const Shape& shape = literal.shape();
  auto device_buffer = AllocateDeviceBuffer(shape);

  // Round trip literal through device.
  ASSERT_IS_OK(transfer_manager_->TransferLiteralToDevice(stream_, literal,
                                                          device_buffer));
  TF_ASSERT_OK_AND_ASSIGN(
      Literal result,
      transfer_manager_->TransferLiteralFromDevice(stream_, device_buffer));

  LiteralTestUtil::ExpectR0Equal<uint32_t>(42, result);
}

XLA_TEST_F(TransferManagerTest, TransferR1F32) {
  Literal literal =
      LiteralUtil::CreateR1<float>({1.25f, 2.5f, -17.0f, -20.125f});
  const Shape& shape = literal.shape();
  auto device_buffer = AllocateDeviceBuffer(shape);

  // Round trip literal through device.
  ASSERT_IS_OK(transfer_manager_->TransferLiteralToDevice(stream_, literal,
                                                          device_buffer));
  TF_ASSERT_OK_AND_ASSIGN(
      Literal result,
      transfer_manager_->TransferLiteralFromDevice(stream_, device_buffer));

  LiteralTestUtil::ExpectR1Equal<float>({1.25f, 2.5f, -17.0f, -20.125f},
                                        result);
}

XLA_TEST_F(TransferManagerTest, TransferR1F32AwkwardSizes) {
  // Test transferring R1s from 0 to kMaxR1Size. The goal is to find bugs
  // related to "awkwardly" sized R1s.
  constexpr int kMaxR1Size = (1 << 11);
  for (int i = 0; i < kMaxR1Size; ++i) {
    std::vector<float> inputs(i);
    std::iota(inputs.begin(), inputs.end(), 0);
    Literal literal = LiteralUtil::CreateR1<float>(inputs);
    const Shape& shape = literal.shape();
    auto device_buffer = AllocateDeviceBuffer(shape);

    // Round trip literal through device.
    ASSERT_IS_OK(transfer_manager_->TransferLiteralToDevice(stream_, literal,
                                                            device_buffer));
    TF_ASSERT_OK_AND_ASSIGN(
        Literal result,
        transfer_manager_->TransferLiteralFromDevice(stream_, device_buffer));

    LiteralTestUtil::ExpectR1Equal<float>(inputs, result);
  }
}

XLA_TEST_F(TransferManagerTest, TransferR1LargeF32) {
  std::vector<float> test_vector(1024 * 1024);
  std::iota(test_vector.begin(), test_vector.end(), 0);
  Literal literal = LiteralUtil::CreateR1<float>(test_vector);
  const Shape& shape = literal.shape();
  auto device_buffer = AllocateDeviceBuffer(shape);

  // Round trip literal through device.
  ASSERT_IS_OK(transfer_manager_->TransferLiteralToDevice(stream_, literal,
                                                          device_buffer));
  TF_ASSERT_OK_AND_ASSIGN(
      Literal result,
      transfer_manager_->TransferLiteralFromDevice(stream_, device_buffer));

  LiteralTestUtil::ExpectR1Equal<float>(test_vector, result);
}

XLA_TEST_F(TransferManagerTest, TransferR1LargeUnalignedF32) {
  std::vector<float> test_vector(1025);
  std::iota(test_vector.begin(), test_vector.end(), 0);
  Shape shape = ShapeUtil::MakeShape(F32, {1024});
  BorrowingLiteral literal(reinterpret_cast<const char*>(&test_vector[1]),
                           shape);
  auto device_buffer = AllocateDeviceBuffer(shape);

  // Round trip literal through device.
  ASSERT_IS_OK(transfer_manager_->TransferLiteralToDevice(stream_, literal,
                                                          device_buffer));
  TF_ASSERT_OK_AND_ASSIGN(
      Literal result,
      transfer_manager_->TransferLiteralFromDevice(stream_, device_buffer));

  std::vector<float> expected_output(1024);
  std::iota(expected_output.begin(), expected_output.end(), 1);
  LiteralTestUtil::ExpectR1Equal<float>(expected_output, result);
}

XLA_TEST_F(TransferManagerTest, TransferR1U8) {
  const char* test_string = "0123456789abcdef";
  Literal literal = LiteralUtil::CreateR1U8(test_string);
  const Shape& shape = literal.shape();
  auto device_buffer = AllocateDeviceBuffer(shape);

  // Round trip literal through device.
  ASSERT_IS_OK(transfer_manager_->TransferLiteralToDevice(stream_, literal,
                                                          device_buffer));
  TF_ASSERT_OK_AND_ASSIGN(
      Literal result,
      transfer_manager_->TransferLiteralFromDevice(stream_, device_buffer));

  EXPECT_EQ(result.GetR1U8AsString(), test_string);
}

XLA_TEST_F(TransferManagerTest, TransferR2F32) {
  Literal literal =
      LiteralUtil::CreateR2<float>({{1.0f, 2.0f, 3.0f}, {4.0f, 5.0f, 6.0f}});
  const Shape& shape = literal.shape();
  auto device_buffer = AllocateDeviceBuffer(shape);

  // Round trip literal through device.
  ASSERT_IS_OK(transfer_manager_->TransferLiteralToDevice(stream_, literal,
                                                          device_buffer));
  TF_ASSERT_OK_AND_ASSIGN(
      Literal result,
      transfer_manager_->TransferLiteralFromDevice(stream_, device_buffer));

  LiteralTestUtil::ExpectR2Equal<float>(
      {{1.0f, 2.0f, 3.0f}, {4.0f, 5.0f, 6.0f}}, result);
}

XLA_TEST_F(TransferManagerTest,
           TransferR2F32AndChangeLayoutTransferringToDevice) {
  Literal literal = LiteralUtil::CreateR2WithLayout<float>(
      {{1.0f, 2.0f, 3.0f}, {4.0f, 5.0f, 6.0f}}, LayoutUtil::MakeLayout({0, 1}));
  const Shape ondevice_shape =
      ShapeUtil::MakeShapeWithLayout(F32, {2, 3}, {1, 0});
  auto device_buffer = AllocateDeviceBuffer(ondevice_shape);

  // Round trip literal through device. Set the on-device layout to something
  // different than the literal layout.
  ASSERT_IS_OK(transfer_manager_->TransferLiteralToDevice(stream_, literal,
                                                          device_buffer));
  TF_ASSERT_OK_AND_ASSIGN(
      Literal result,
      transfer_manager_->TransferLiteralFromDevice(stream_, device_buffer));

  EXPECT_FALSE(
      LayoutUtil::Equal(result.shape().layout(), literal.shape().layout()));
  LiteralTestUtil::ExpectR2Equal<float>(
      {{1.0f, 2.0f, 3.0f}, {4.0f, 5.0f, 6.0f}}, result);
}

XLA_TEST_F(TransferManagerTest, TransferTuple) {
  Literal literal = LiteralUtil::MakeTupleFromSlices(
      {LiteralUtil::CreateR0<float>(123.0f),
       LiteralUtil::CreateR2<float>({{1.0f, 2.0f}, {4.0f, 5.0f}}),
       LiteralUtil::CreateR1<float>({44.0f, -10.0f, 3333333.3f})});
  auto device_buffer = AllocateDeviceBuffer(literal.shape());

  // Round trip literal through device.
  ASSERT_IS_OK(transfer_manager_->TransferLiteralToDevice(stream_, literal,
                                                          device_buffer));
  TF_ASSERT_OK_AND_ASSIGN(
      Literal result,
      transfer_manager_->TransferLiteralFromDevice(stream_, device_buffer));

  EXPECT_TRUE(LiteralTestUtil::Equal(literal, result));
}

XLA_TEST_F(TransferManagerTest, TransferEmptyTuple) {
  Literal literal = LiteralUtil::MakeTuple({});
  auto device_buffer = AllocateDeviceBuffer(literal.shape());

  // Round trip literal through device.
  ASSERT_IS_OK(transfer_manager_->TransferLiteralToDevice(stream_, literal,
                                                          device_buffer));
  TF_ASSERT_OK_AND_ASSIGN(
      Literal result,
      transfer_manager_->TransferLiteralFromDevice(stream_, device_buffer));

  EXPECT_TRUE(LiteralTestUtil::Equal(literal, result));
}

XLA_TEST_F(TransferManagerTest, TransferNestedTuple) {
  Literal literal = LiteralUtil::MakeTupleFromSlices(
      {LiteralUtil::CreateR0<float>(123.0f),
       LiteralUtil::MakeTupleFromSlices(
           {LiteralUtil::CreateR2<float>({{1.0f, 2.0f}, {4.0f, 5.0f}}),
            LiteralUtil::CreateR1<float>({44.0f, -10.0f, 3333333.3f})}),
       LiteralUtil::CreateR1<float>({-10.0f, 123.0f})});
  auto device_buffer = AllocateDeviceBuffer(literal.shape());

  // Round trip literal through device.
  ASSERT_IS_OK(transfer_manager_->TransferLiteralToDevice(stream_, literal,
                                                          device_buffer));
  TF_ASSERT_OK_AND_ASSIGN(
      Literal result,
      transfer_manager_->TransferLiteralFromDevice(stream_, device_buffer));

  EXPECT_TRUE(LiteralTestUtil::Equal(literal, result));
}

XLA_TEST_F(TransferManagerTest, TransferComplexValue) {
  Literal literal = LiteralUtil::CreateR1<complex64>(
      {complex64(1.0f, 2.0f), complex64(42.0f, -123.4f)});
  auto device_buffer = AllocateDeviceBuffer(literal.shape());

  // Round trip literal through device.
  ASSERT_IS_OK(transfer_manager_->TransferLiteralToDevice(stream_, literal,
                                                          device_buffer));
  TF_ASSERT_OK_AND_ASSIGN(
      Literal result,
      transfer_manager_->TransferLiteralFromDevice(stream_, device_buffer));

  EXPECT_TRUE(LiteralTestUtil::Equal(literal, result));
}

XLA_TEST_F(TransferManagerTest, TransferComplexValueInTuple) {
  Literal literal = LiteralUtil::MakeTupleFromSlices(
      {LiteralUtil::CreateR1<complex64>(
           {complex64(1.0f, 2.0f), complex64(42.0f, -123.4f)}),
       LiteralUtil::CreateR1<int32_t>({1, 2, 3, 4, 5, 6}),
       LiteralUtil::CreateR0<complex64>(complex64(0.3f, -0.4f))});
  auto device_buffer = AllocateDeviceBuffer(literal.shape());

  // Round trip literal through device.
  ASSERT_IS_OK(transfer_manager_->TransferLiteralToDevice(stream_, literal,
                                                          device_buffer));
  TF_ASSERT_OK_AND_ASSIGN(
      Literal result,
      transfer_manager_->TransferLiteralFromDevice(stream_, device_buffer));

  EXPECT_TRUE(LiteralTestUtil::Equal(literal, result));
}

XLA_TEST_F(TransferManagerTest, TransferTokenFromDevice) {
  // "Copy" a token from the device. The token has no physical representation
  // so no copying is actually performed, but it shouldn't fail.
  // TODO(b/110532604): Add transferring the token to device when this is
  // supported.
  auto device_buffer = AllocateDeviceBuffer(ShapeUtil::MakeTokenShape());
  TF_ASSERT_OK_AND_ASSIGN(
      Literal result,
      transfer_manager_->TransferLiteralFromDevice(stream_, device_buffer));
  EXPECT_TRUE(LiteralTestUtil::Equal(LiteralUtil::CreateToken(), result));
}

XLA_TEST_F(TransferManagerTest, MultiStreamRoundTripSoak) {
  const int64_t kIterationCount = 5000;
  Literal literal1 = LiteralUtil::MakeTupleFromSlices(
      {LiteralUtil::CreateR0<float>(123.0f),
       LiteralUtil::MakeTupleFromSlices(
           {LiteralUtil::CreateR2<float>({{1.0f, 2.0f}, {4.0f, 5.0f}}),
            LiteralUtil::CreateR1<float>({44.0f, -10.0f, 3333333.3f})}),
       LiteralUtil::CreateR1<float>({-10.0f, 123.0f})});
  Literal literal2 = LiteralUtil::MakeTupleFromSlices(
      {LiteralUtil::CreateR0<float>(456.0f),
       LiteralUtil::MakeTupleFromSlices(
           {LiteralUtil::CreateR2<float>({{5.0f, 7.0f}, {9.0f, 4.0f}}),
            LiteralUtil::CreateR1<float>({44.0f, -11.0f, 3333333.3f})}),
       LiteralUtil::CreateR1<float>({-98.0f, 153.0f})});

  auto device_buffer1 = AllocateDeviceBuffer(literal1.shape());
  auto device_buffer2 = AllocateDeviceBuffer(literal2.shape());

  auto stream1 = stream_;
  auto stream2 = stream_->GetOrCreateSubStream();

  Literal result1, result2;

  // Round trip literals through device in multiple streams asynchronously.
  for (int i = 0; i < kIterationCount; ++i) {
    ASSERT_IS_OK(transfer_manager_->TransferLiteralToDevice(stream1, literal1,
                                                            device_buffer1));
    ASSERT_IS_OK(transfer_manager_->TransferLiteralToDevice(stream2, literal2,
                                                            device_buffer2));
    TF_ASSERT_OK_AND_ASSIGN(
        Literal this_result1,
        transfer_manager_->TransferLiteralFromDevice(stream1, device_buffer1));
    TF_ASSERT_OK_AND_ASSIGN(
        Literal this_result2,
        transfer_manager_->TransferLiteralFromDevice(stream2, device_buffer2));
    result1 = std::move(this_result1);
    result2 = std::move(this_result2);
  }

  EXPECT_TRUE(LiteralTestUtil::Equal(literal1, result1));
  EXPECT_TRUE(LiteralTestUtil::Equal(literal2, result2));
}

XLA_TEST_F(TransferManagerTest, TransferDynamicShape) {
  TF_ASSERT_OK_AND_ASSIGN(
      Shape s, ParseShape("(s64[], s32[<=1048576,3], f32[<=1048576,48])"));

  Literal literal(s);
  literal.SetDynamicSize(/*dim_index=*/0, /*shape_index=*/{1},
                         /*size=*/1048574);
  literal.SetDynamicSize(/*dim_index=*/0, /*shape_index=*/{2},
                         /*size=*/1048575);
  ASSERT_IS_OK(MutableBorrowingLiteral(&literal, /*view_root=*/{0})
                   .Populate<int64_t>(
                       [](absl::Span<const int64_t> indices) { return 42; }));
  ASSERT_IS_OK(MutableBorrowingLiteral(&literal, /*view_root=*/{1})
                   .Populate<int32_t>([](absl::Span<const int64_t> indices) {
                     return indices[0] + indices[1];
                   }));
  ASSERT_IS_OK(MutableBorrowingLiteral(&literal, /*view_root=*/{2})
                   .Populate<float>([](absl::Span<const int64_t> indices) {
                     return indices[0] + indices[1];
                   }));

  // Round trip `literal` through device.
  ScopedShapedBuffer device_buffer = AllocateDeviceBuffer(literal.shape());
  ASSERT_IS_OK(transfer_manager_->TransferLiteralToDevice(stream_, literal,
                                                          device_buffer));
  TF_ASSERT_OK_AND_ASSIGN(
      Literal result,
      transfer_manager_->TransferLiteralFromDevice(stream_, device_buffer));

  // LiteralTestUtil::Equal doesn't compare dynamic shapes, so we need to check
  // them ourselves.
  EXPECT_EQ(literal.GetDynamicSize(/*dim_index=*/0, /*shape_index=*/{1}),
            result.GetDynamicSize(0, {1}));
  EXPECT_EQ(literal.GetDynamicSize(/*dim_index=*/0, /*shape_index=*/{2}),
            result.GetDynamicSize(0, {2}));
  EXPECT_TRUE(LiteralTestUtil::Equal(literal, result));
}

class TransferDeviceToHostBenchmark : public TransferManagerTest {
 public:
  using TransferManagerTest::TransferManagerTest;
  ~TransferDeviceToHostBenchmark() override {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPStestsPStransfer_manager_testDTcc mht_2(mht_2_v, 571, "", "./tensorflow/compiler/xla/tests/transfer_manager_test.cc", "~TransferDeviceToHostBenchmark");
}

  void Run(::testing::benchmark::State& state, int num_tuple_elements,
           int array_size) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPStestsPStransfer_manager_testDTcc mht_3(mht_3_v, 577, "", "./tensorflow/compiler/xla/tests/transfer_manager_test.cc", "Run");

    SetUp();

    std::vector<Literal> tuple_elements;
    for (int i = 0; i < num_tuple_elements; ++i) {
      tuple_elements.push_back(
          LiteralUtil::CreateR2F32Linspace(0.0f, 1.0f, array_size, array_size));
    }
    Literal literal = LiteralUtil::MakeTupleOwned(std::move(tuple_elements));
    auto device_buffer = AllocateDeviceBuffer(literal.shape());
    TF_CHECK_OK(transfer_manager_->TransferLiteralToDevice(stream_, literal,
                                                           device_buffer));
    for (auto s : state) {
      TF_ASSERT_OK_AND_ASSIGN(
          Literal result,
          transfer_manager_->TransferLiteralFromDevice(stream_, device_buffer));
    }
    TearDown();
  }

  void TestBody() override {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPStestsPStransfer_manager_testDTcc mht_4(mht_4_v, 600, "", "./tensorflow/compiler/xla/tests/transfer_manager_test.cc", "TestBody");
}
};

class TransferHostToDeviceBenchmark : public TransferManagerTest {
 public:
  using TransferManagerTest::TransferManagerTest;
  ~TransferHostToDeviceBenchmark() override {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPStestsPStransfer_manager_testDTcc mht_5(mht_5_v, 609, "", "./tensorflow/compiler/xla/tests/transfer_manager_test.cc", "~TransferHostToDeviceBenchmark");
}

  void Run(::testing::benchmark::State& state, int num_tuple_elements,
           int array_size) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPStestsPStransfer_manager_testDTcc mht_6(mht_6_v, 615, "", "./tensorflow/compiler/xla/tests/transfer_manager_test.cc", "Run");

    SetUp();

    std::vector<Literal> tuple_elements;
    for (int i = 0; i < num_tuple_elements; ++i) {
      tuple_elements.push_back(
          LiteralUtil::CreateR2F32Linspace(0.0f, 1.0f, array_size, array_size));
    }
    Literal literal = LiteralUtil::MakeTupleOwned(std::move(tuple_elements));
    auto device_buffer = AllocateDeviceBuffer(literal.shape());

    for (auto s : state) {
      TF_CHECK_OK(transfer_manager_->TransferLiteralToDevice(stream_, literal,
                                                             device_buffer));
    }
    TearDown();
  }

  void TestBody() override {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPStestsPStransfer_manager_testDTcc mht_7(mht_7_v, 636, "", "./tensorflow/compiler/xla/tests/transfer_manager_test.cc", "TestBody");
}
};

void BM_TransferDeviceToHost(::testing::benchmark::State& state) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPStestsPStransfer_manager_testDTcc mht_8(mht_8_v, 642, "", "./tensorflow/compiler/xla/tests/transfer_manager_test.cc", "BM_TransferDeviceToHost");

  const int num_tuple_elements = state.range(0);
  const int array_size = state.range(1);

  TransferDeviceToHostBenchmark bm;
  bm.Run(state, num_tuple_elements, array_size);
}

void BM_TransferHostToDevice(::testing::benchmark::State& state) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPStestsPStransfer_manager_testDTcc mht_9(mht_9_v, 653, "", "./tensorflow/compiler/xla/tests/transfer_manager_test.cc", "BM_TransferHostToDevice");

  const int num_tuple_elements = state.range(0);
  const int array_size = state.range(1);

  TransferHostToDeviceBenchmark bm;
  bm.Run(state, num_tuple_elements, array_size);
}

BENCHMARK(BM_TransferHostToDevice)
    ->ArgPair(1, 256)
    ->ArgPair(1, 257)
    ->ArgPair(100, 256)
    ->ArgPair(100, 257);

BENCHMARK(BM_TransferDeviceToHost)
    ->ArgPair(1, 256)
    ->ArgPair(1, 257)
    ->ArgPair(100, 256)
    ->ArgPair(100, 257);

int main(int argc, char** argv) {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPStestsPStransfer_manager_testDTcc mht_10(mht_10_v, 676, "", "./tensorflow/compiler/xla/tests/transfer_manager_test.cc", "main");

  ::testing::InitGoogleTest(&argc, argv);
  tensorflow::testing::RunBenchmarks();
  return RUN_ALL_TESTS();
}

}  // namespace
}  // namespace xla
