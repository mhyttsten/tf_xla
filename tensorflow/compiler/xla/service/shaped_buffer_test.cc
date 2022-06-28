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
class MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSshaped_buffer_testDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSshaped_buffer_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSshaped_buffer_testDTcc() {
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

#include "tensorflow/compiler/xla/service/shaped_buffer.h"

#include "absl/memory/memory.h"
#include "tensorflow/compiler/xla/service/platform_util.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/test.h"
#include "tensorflow/core/platform/stream_executor_no_cuda.h"
#include "tensorflow/core/platform/test_benchmark.h"
#include "tensorflow/core/util/ptr_util.h"
#include "tensorflow/stream_executor/device_memory_allocator.h"

namespace xla {
namespace {

TEST(ShapedBufferTest, ScopedShapeBufferAsShapedBufferB71629047) {
  TF_ASSERT_OK_AND_ASSIGN(auto* platform,
                          xla::PlatformUtil::GetDefaultPlatform());
  TF_ASSERT_OK_AND_ASSIGN(auto executors,
                          xla::PlatformUtil::GetStreamExecutors(platform));
  xla::se::StreamExecutorMemoryAllocator allocator(platform, executors);
  const xla::Shape shape = xla::ShapeUtil::MakeShape(xla::F32, {});
  const int kDeviceOrdinal = 0;
  auto scoped_buffer = absl::make_unique<xla::ScopedShapedBuffer>(
      shape, shape, &allocator, kDeviceOrdinal);
  std::unique_ptr<xla::ShapedBuffer> buffer = std::move(scoped_buffer);
  buffer = nullptr;
}

class TestAllocator : public se::DeviceMemoryAllocator {
 public:
  TestAllocator()
      : se::DeviceMemoryAllocator(
            PlatformUtil::GetDefaultPlatform().ValueOrDie()) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSshaped_buffer_testDTcc mht_0(mht_0_v, 217, "", "./tensorflow/compiler/xla/service/shaped_buffer_test.cc", "TestAllocator");
}

  ~TestAllocator() override {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSshaped_buffer_testDTcc mht_1(mht_1_v, 222, "", "./tensorflow/compiler/xla/service/shaped_buffer_test.cc", "~TestAllocator");

    if (!allocations_.empty()) {
      ADD_FAILURE() << "Some allocations not freed!";
    }
  }

  // Pull in two-arg overload of Allocate.
  using se::DeviceMemoryAllocator::Allocate;

  StatusOr<se::OwningDeviceMemory> Allocate(int device_ordinal, uint64_t size,
                                            bool /*retry_on_failure*/,
                                            int64_t /*memory_space*/) override {
    // By contract, we must return null if size == 0.
    if (size == 0) {
      return se::OwningDeviceMemory();
    }
    void* buf = malloc(size);
    allocations_.insert({device_ordinal, buf});
    return se::OwningDeviceMemory(se::DeviceMemoryBase(buf, size),
                                  device_ordinal, this);
  }

  Status Deallocate(int device_ordinal, se::DeviceMemoryBase mem) override {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSshaped_buffer_testDTcc mht_2(mht_2_v, 247, "", "./tensorflow/compiler/xla/service/shaped_buffer_test.cc", "Deallocate");

    if (mem.is_null()) {
      return Status::OK();
    }

    auto it = allocations_.find({device_ordinal, mem.opaque()});
    if (it == allocations_.end()) {
      ADD_FAILURE() << "Allocation not found (double free?)";
    } else {
      free(mem.opaque());
      allocations_.erase(it);
    }
    return Status::OK();
  }

  bool AllowsAsynchronousDeallocation() const override {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSshaped_buffer_testDTcc mht_3(mht_3_v, 265, "", "./tensorflow/compiler/xla/service/shaped_buffer_test.cc", "AllowsAsynchronousDeallocation");
 return false; }

  StatusOr<se::Stream*> GetStream(int device_ordinal) override {
    LOG(FATAL) << "Not implemented";
  }

 private:
  std::set<std::pair</*device_ordinal*/ int64_t, void*>> allocations_;
};

TEST(ScopedShapedBufferTest, TestMoveAssignmentOperator) {
  Shape s = ShapeUtil::MakeShape(F32, {1});
  TestAllocator allocator;
  ScopedShapedBuffer sb1(s, &allocator, /*device_ordinal=*/0);
  sb1.set_buffer(
      allocator.Allocate(/*device_ordinal=*/0, /*size=*/42).ValueOrDie(),
      /*index=*/{});

  ScopedShapedBuffer sb2(s, &allocator, /*device_ordinal=*/1);
  sb2.set_buffer(
      allocator.Allocate(/*device_ordinal=*/1, /*size=*/10).ValueOrDie(),
      /*index=*/{});

  sb1 = std::move(sb2);

  // TestAllocator's destructor checks that all memory was freed.
}

TEST(ScopedShapedBufferTest, TestTakeSubTree) {
  TestAllocator allocator;

  Shape s = ShapeUtil::MakeShape(F32, {1});
  s = xla::ShapeUtil::MakeTupleShape(std::vector<xla::Shape>(2, s));
  s = xla::ShapeUtil::MakeTupleShape(std::vector<xla::Shape>(3, s));

  ScopedShapedBuffer sb(s, &allocator, /*device_ordinal=*/0);
  sb.buffers().ForEachMutableElement(
      [&](const xla::ShapeIndex& index, se::DeviceMemoryBase* buffer) {
        TF_ASSERT_OK_AND_ASSIGN(
            se::OwningDeviceMemory m,
            allocator.Allocate(/*device_ordinal=*/0, /*size=*/77));
        *buffer = m.Release();
      });
  ShapeTree<se::DeviceMemoryBase> buffers = sb.buffers();

  // Takes a subtree out of 'sb', and verifies the buffers are as expected.
  xla::ShapeIndex subtree_index = {1};
  ScopedShapedBuffer output = sb.TakeSubTree(subtree_index);

  output.buffers().ForEachElement([&](const xla::ShapeIndex& sub_index,
                                      const se::DeviceMemoryBase& buffer) {
    xla::ShapeIndex orig_index = subtree_index;
    for (int i : sub_index) {
      orig_index.push_back(i);
    }
    EXPECT_TRUE(buffers.find(orig_index)->second.IsSameAs(buffer));
  });
  sb.buffers().ForEachElement([&](const xla::ShapeIndex& index,
                                  const se::DeviceMemoryBase& buffer) {
    if ((index.size() >= subtree_index.size()) &&
        ShapeIndexView(index).first(subtree_index.size()) == subtree_index) {
      EXPECT_TRUE(buffer.is_null());
    } else {
      EXPECT_TRUE(buffers.find(index)->second.IsSameAs(buffer));
    }
  });
}

TEST(ScopedShapedBufferTest, TestSubShapeTree) {
  Shape array_shape = ShapeUtil::MakeShape(F32, {1});
  Shape tuple_shape =
      xla::ShapeUtil::MakeTupleShape({array_shape, array_shape});
  TestAllocator allocator;
  ScopedShapedBuffer sb(tuple_shape, &allocator, /*device_ordinal=*/0);
  sb.buffers().ForEachMutableElement(
      [&](const xla::ShapeIndex& index, se::DeviceMemoryBase* buffer) {
        TF_ASSERT_OK_AND_ASSIGN(
            se::OwningDeviceMemory m,
            allocator.Allocate(/*device_ordinal=*/0, /*size=*/32));
        *buffer = m.Release();
      });
  auto ssb_statusor = sb.SubShapedBuffer({1});
  ASSERT_TRUE(ssb_statusor.ok());
  auto ssb = ssb_statusor.ConsumeValueOrDie();
  EXPECT_EQ(ssb.on_host_shape(), array_shape);
  EXPECT_EQ(ssb.on_device_shape(), array_shape);
}

// Test TakeSubTree with different depths (depth of ShapeTree) and fan-outs
// (cardinality of each non-leaf node's children).
void BM_TakeSubTree(::testing::benchmark::State& state) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSshaped_buffer_testDTcc mht_4(mht_4_v, 358, "", "./tensorflow/compiler/xla/service/shaped_buffer_test.cc", "BM_TakeSubTree");

  const int depth = state.range(0);
  const int fan_out = state.range(1);

  TestAllocator allocator;
  xla::Shape shape = xla::ShapeUtil::MakeShape(xla::F32, {32, 64, 128});
  for (int i = 0; i < depth; ++i) {
    std::vector<xla::Shape> shapes(fan_out, shape);
    shape = xla::ShapeUtil::MakeTupleShape(shapes);
  }
  xla::ScopedShapedBuffer shaped_buffer(shape, /*allocator=*/&allocator,
                                        /*device_ordinal=*/0);
  for (auto s : state) {
    // Extract a buffer from approximately the middle of the first level of the
    // tree.
    (void)shaped_buffer.TakeSubTree(/*index=*/{fan_out / 2}).release();
  }
}

BENCHMARK(BM_TakeSubTree)
    ->ArgPair(1, 4)
    ->ArgPair(1, 8)
    ->ArgPair(1, 32)
    ->ArgPair(1, 64)
    ->ArgPair(1, 128)
    ->ArgPair(1, 256)
    ->ArgPair(1, 512)
    ->ArgPair(2, 4)
    ->ArgPair(2, 8)
    ->ArgPair(2, 32)
    ->ArgPair(2, 64)
    ->ArgPair(2, 128);

}  // anonymous namespace
}  // namespace xla
