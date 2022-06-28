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
class MHTracer_DTPStensorflowPScorePScommon_runtimePSgpuPSpool_allocator_testDTcc {
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
   MHTracer_DTPStensorflowPScorePScommon_runtimePSgpuPSpool_allocator_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePScommon_runtimePSgpuPSpool_allocator_testDTcc() {
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

/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM

#include "tensorflow/core/common_runtime/pool_allocator.h"

#include "gpu_init.h"
#include "tensorflow/core/common_runtime/device/device_host_allocator.h"
#include "tensorflow/core/platform/stream_executor.h"
#include "tensorflow/core/platform/test.h"
namespace tensorflow {
namespace {

TEST(PoolAllocatorTest, ZeroSizeBuffers) {
  se::Platform* platform =
      se::MultiPlatformManager::PlatformWithName(GpuPlatformName())
          .ValueOrDie();
  PoolAllocator pool(
      2 /*pool_size_limit*/, false /*auto_resize*/,
      new DeviceHostAllocator(
          platform->GetExecutor(se::StreamExecutorConfig(/*ordinal=*/0))
              .ValueOrDie(),
          0 /*numa_node*/, {}, {}),
      new NoopRounder, "pool");

  EXPECT_EQ(nullptr, pool.AllocateRaw(4 /*alignment*/, 0 /*num_bytes*/));
  pool.DeallocateRaw(nullptr);  // Should not crash.
  EXPECT_EQ(0, pool.get_from_pool_count());
  EXPECT_EQ(0, pool.put_count());
  EXPECT_EQ(0, pool.allocated_count());
  EXPECT_EQ(0, pool.evicted_count());
}

TEST(PoolAllocatorTest, ZeroSizePool) {
  se::Platform* platform =
      se::MultiPlatformManager::PlatformWithName(GpuPlatformName())
          .ValueOrDie();
  PoolAllocator pool(
      0 /*pool_size_limit*/, false /*auto_resize*/,
      new DeviceHostAllocator(
          platform->GetExecutor(se::StreamExecutorConfig(/*ordinal=*/0))
              .ValueOrDie(),
          0 /*numa_node*/, {}, {}),
      new NoopRounder, "pool");

  EXPECT_EQ(0, pool.get_from_pool_count());
  EXPECT_EQ(0, pool.put_count());
  EXPECT_EQ(0, pool.allocated_count());
  EXPECT_EQ(0, pool.evicted_count());

  // All allocations should bypass the pool and return valid pointers.
  for (int i = 0; i < 3; ++i) {
    void* p0 = pool.AllocateRaw(4, 0);
    void* p4 = pool.AllocateRaw(4, 4);
    void* p12 = pool.AllocateRaw(4, 12);
    EXPECT_EQ(nullptr, p0);
    EXPECT_NE(nullptr, p4);
    EXPECT_NE(nullptr, p12);
    pool.DeallocateRaw(p0);
    pool.DeallocateRaw(p4);
    pool.DeallocateRaw(p12);
  }
  EXPECT_EQ(0, pool.get_from_pool_count());
  EXPECT_EQ(0, pool.put_count());
  EXPECT_EQ(0, pool.allocated_count());
  EXPECT_EQ(0, pool.evicted_count());
}

TEST(PoolAllocatorTest, Alignment) {
  se::Platform* platform =
      se::MultiPlatformManager::PlatformWithName(GpuPlatformName())
          .ValueOrDie();
  PoolAllocator pool(
      0 /*pool_size_limit*/, false /*auto_resize*/,
      new DeviceHostAllocator(
          platform->GetExecutor(se::StreamExecutorConfig(/*ordinal=*/0))
              .ValueOrDie(),
          0 /*numa_node*/, {}, {}),
      new NoopRounder, "pool");
  for (int i = 0; i < 16; ++i) {
    size_t alignment = 1 << i;
    void* p = pool.AllocateRaw(alignment, 111);
    EXPECT_TRUE(p != nullptr);
    EXPECT_EQ(0, reinterpret_cast<int64_t>(p) & (alignment - 1))
        << "ptr: " << p << " alignment " << alignment;
    // Intentionally don't deallocate, to test that destruction of
    // the PoolAllocator frees all pending memory.
  }
}

TEST(PoolAllocatorTest, AutoResize) {
  PoolAllocator pool(2 /*pool_size_limit*/, true /*auto_resize*/,
                     new BasicCPUAllocator(0 /*numa_node*/, {}, {}),
                     new NoopRounder, "pool");

  // Alloc/dealloc 10 sizes just a few times, confirming pool size
  // stays at 2.
  for (int i = 0; i < 10; ++i) {
    void* p = pool.AllocateRaw(4, 64 << i);
    pool.DeallocateRaw(p);
  }
  EXPECT_EQ(0, pool.get_from_pool_count());
  EXPECT_EQ(10, pool.allocated_count());
  EXPECT_EQ(10, pool.put_count());
  EXPECT_EQ(8, pool.evicted_count());
  EXPECT_EQ(2, pool.size_limit());

  // Then repeat 1200 times.  Pool size limit should jump to 100.
  for (int j = 0; j < 120; ++j) {
    for (int i = 0; i < 10; ++i) {
      void* p = pool.AllocateRaw(4, 64 << i);
      pool.DeallocateRaw(p);
    }
  }
  EXPECT_EQ(100, pool.size_limit());
}

TEST(PoolAllocatorTest, CudaHostAllocator) {
  int alloc_count = 0;
  int64_t alloc_size = 0;
  SubAllocator::Visitor alloc_visitor =
      [&alloc_count, &alloc_size](void* ptr, int numa_node, int64_t size) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSgpuPSpool_allocator_testDTcc mht_0(mht_0_v, 304, "", "./tensorflow/core/common_runtime/gpu/pool_allocator_test.cc", "lambda");

        ++alloc_count;
        alloc_size += size;
      };
  int free_count = 0;
  int64_t free_size = 0;
  SubAllocator::Visitor free_visitor =
      [&free_count, &free_size](void* ptr, int numa_node, int64_t size) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSgpuPSpool_allocator_testDTcc mht_1(mht_1_v, 314, "", "./tensorflow/core/common_runtime/gpu/pool_allocator_test.cc", "lambda");

        ++free_count;
        free_size += size;
      };
  se::Platform* platform =
      se::MultiPlatformManager::PlatformWithName(GpuPlatformName())
          .ValueOrDie();
  DeviceHostAllocator* sub_allocator = new DeviceHostAllocator(
      platform->GetExecutor(se::StreamExecutorConfig(/*ordinal=*/0))
          .ValueOrDie(),
      0 /*numa_node*/, {alloc_visitor}, {free_visitor});
  PoolAllocator pool(2 /*pool_size_limit*/, false /*auto_resize*/,
                     sub_allocator, new NoopRounder, "pool");
  EXPECT_EQ(0, alloc_count);
  EXPECT_EQ(0, alloc_size);
  EXPECT_EQ(0, free_count);
  EXPECT_EQ(0, free_size);

  // Repeatedly Get a 16-byte value, confirming that there's only
  // one real allocation.
  void* p1_16 = pool.AllocateRaw(4, 16);
  EXPECT_EQ(0, pool.get_from_pool_count());
  EXPECT_EQ(1, pool.allocated_count());
  EXPECT_NE(nullptr, p1_16);
  EXPECT_EQ(1, alloc_count);  // Underlying suballoc of 16 bytes
  // Each suballocation includes a 16B ChunkPrefix.
  static const int kChunkPrefixSize = 16;
  EXPECT_EQ(16 + (alloc_count * kChunkPrefixSize), alloc_size);
  pool.DeallocateRaw(p1_16);
  // Pool contents {16}
  EXPECT_EQ(1, pool.put_count());
  void* p2_16 = pool.AllocateRaw(4, 16);  // Get it again.
  EXPECT_EQ(1, pool.get_from_pool_count());
  EXPECT_EQ(1, pool.allocated_count());
  EXPECT_EQ(p1_16, p2_16);    // Same pointer value
  pool.DeallocateRaw(p2_16);  // Put it back.
  // Pool contents {16}
  EXPECT_EQ(2, pool.put_count());
  EXPECT_EQ(1, alloc_count);  // Underlying suballoc of 16 bytes
  EXPECT_EQ(16 + (alloc_count * kChunkPrefixSize), alloc_size);
  EXPECT_EQ(0, free_count);

  // Get two more values of different sizes.
  void* p3_4 = pool.AllocateRaw(4, 4);
  EXPECT_EQ(2, pool.allocated_count());
  EXPECT_NE(p1_16, p3_4);  // Different pointer value
  EXPECT_NE(nullptr, p3_4);
  pool.DeallocateRaw(p3_4);  // Put it back. Pool is now full.
  // Pool contents {4, 16}
  EXPECT_EQ(3, pool.put_count());
  void* p4_2 = pool.AllocateRaw(4, 2);  // Get a third size buffer.
  EXPECT_NE(nullptr, p4_2);
  EXPECT_EQ(0, pool.evicted_count());
  EXPECT_EQ(3, alloc_count);
  EXPECT_EQ(16 + 4 + 2 + (alloc_count * kChunkPrefixSize), alloc_size);
  EXPECT_EQ(0, free_count);

  // The pool is full: when we put back p4_2, the 16-byte buffer
  // should be evicted since it was least recently inserted.
  pool.DeallocateRaw(p4_2);
  // Pool contents {2, 4}
  EXPECT_EQ(4, pool.put_count());
  EXPECT_EQ(1, pool.evicted_count());
  EXPECT_EQ(3, alloc_count);
  EXPECT_EQ(16 + 4 + 2 + (alloc_count * kChunkPrefixSize), alloc_size);
  EXPECT_EQ(1, free_count);
  EXPECT_EQ(16 + (free_count * kChunkPrefixSize), free_size);

  // Re-getting and putting size 2 or 4 should not alter pool size or
  // num-evicted.
  void* p5_4 = pool.AllocateRaw(4, 4);
  EXPECT_NE(nullptr, p5_4);
  pool.DeallocateRaw(p5_4);
  void* p6_2 = pool.AllocateRaw(4, 2);
  EXPECT_NE(nullptr, p6_2);
  pool.DeallocateRaw(p6_2);
  EXPECT_EQ(3, pool.get_from_pool_count());
  EXPECT_EQ(6, pool.put_count());
  EXPECT_EQ(3, pool.allocated_count());
  EXPECT_EQ(1, pool.evicted_count());
  EXPECT_EQ(3, alloc_count);
  EXPECT_EQ(16 + 4 + 2 + (alloc_count * kChunkPrefixSize), alloc_size);
  EXPECT_EQ(1, free_count);
  EXPECT_EQ(16 + (free_count * kChunkPrefixSize), free_size);

  pool.Clear();
  EXPECT_EQ(0, pool.get_from_pool_count());
  EXPECT_EQ(0, pool.put_count());
  EXPECT_EQ(0, pool.allocated_count());
  EXPECT_EQ(0, pool.evicted_count());
  EXPECT_EQ(3, alloc_count);
  EXPECT_EQ(16 + 4 + 2 + (alloc_count * kChunkPrefixSize), alloc_size);
  EXPECT_EQ(3, free_count);
  EXPECT_EQ(16 + 4 + 2 + (free_count * kChunkPrefixSize), free_size);
}

TEST(PoolAllocatorTest, Pow2Rounder) {
  Pow2Rounder rounder;
  EXPECT_EQ(1, rounder.RoundUp(1));
  EXPECT_EQ(2, rounder.RoundUp(2));
  EXPECT_EQ(16, rounder.RoundUp(9));
  EXPECT_EQ(16, rounder.RoundUp(16));
  EXPECT_EQ(65536, rounder.RoundUp(41234));
  EXPECT_EQ(65536, rounder.RoundUp(65535));
  EXPECT_EQ(65536, rounder.RoundUp(65536));
}

TEST(PoolAllocatorTest, Name) {
  se::Platform* platform =
      se::MultiPlatformManager::PlatformWithName(GpuPlatformName())
          .ValueOrDie();
  PoolAllocator pool(
      2 /*pool_size_limit*/, false /*auto_resize*/,
      new DeviceHostAllocator(
          platform->GetExecutor(se::StreamExecutorConfig(/*ordinal=*/0))
              .ValueOrDie(),
          0 /*numa_node*/, {}, {}),
      new NoopRounder, "pool");
  EXPECT_EQ("pool", pool.Name());
}

}  // namespace
}  // namespace tensorflow

#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM
