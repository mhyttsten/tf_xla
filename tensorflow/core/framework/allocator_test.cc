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
class MHTracer_DTPStensorflowPScorePSframeworkPSallocator_testDTcc {
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
   MHTracer_DTPStensorflowPScorePSframeworkPSallocator_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSframeworkPSallocator_testDTcc() {
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

#include "tensorflow/core/framework/allocator.h"

#include <algorithm>
#include <vector>

#include "tensorflow/core/framework/typed_allocator.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/mem.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/platform/test_benchmark.h"
#include "tensorflow/core/profiler/lib/profiler_session.h"
#include "tensorflow/core/profiler/protobuf/memory_profile.pb.h"
#include "tensorflow/core/profiler/protobuf/xplane.pb.h"
#include "tensorflow/core/profiler/utils/xplane_schema.h"
#include "tensorflow/core/profiler/utils/xplane_visitor.h"

namespace tensorflow {

static void CheckStats(Allocator* a, int64_t num_allocs, int64_t bytes_in_use,
                       int64_t peak_bytes_in_use, int64_t largest_alloc_size) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSallocator_testDTcc mht_0(mht_0_v, 204, "", "./tensorflow/core/framework/allocator_test.cc", "CheckStats");

  absl::optional<AllocatorStats> stats = a->GetStats();
  EXPECT_TRUE(stats);
  if (!stats) {
    return;
  }
  LOG(INFO) << "Alloc stats: \n" << stats->DebugString();
#if defined(PLATFORM_GOOGLE) && defined(NDEBUG)
  // NOTE: allocator stats expectation depends on the system malloc,
  // and can vary as that changes.
  static const int64 kSlop = 5 * 1024;
  EXPECT_GT(stats->bytes_in_use, bytes_in_use - kSlop);
  EXPECT_LT(stats->bytes_in_use, bytes_in_use + kSlop);
  EXPECT_GT(stats->peak_bytes_in_use, peak_bytes_in_use - kSlop);
  EXPECT_LT(stats->peak_bytes_in_use, peak_bytes_in_use + kSlop);
  EXPECT_EQ(stats->num_allocs, num_allocs);
  EXPECT_EQ(stats->largest_alloc_size, largest_alloc_size);
#endif
}

TEST(AllocatorAttributesTest, AllCombos) {
  for (bool on_host : {false, true}) {
    for (bool nic_compatible : {false, true}) {
      for (bool gpu_compatible : {false, true}) {
        AllocatorAttributes aa;
        aa.set_on_host(on_host);
        aa.set_nic_compatible(nic_compatible);
        aa.set_gpu_compatible(gpu_compatible);
        EXPECT_EQ(on_host, aa.on_host());
        EXPECT_EQ(nic_compatible, aa.nic_compatible());
        EXPECT_EQ(gpu_compatible, aa.gpu_compatible());
      }
    }
  }
}

TEST(AllocatorAttributesTest, IsEqualOrLessRestrictiveThan) {
  AllocatorAttributes a, b;
  EXPECT_TRUE(a.IsEqualOrLessRestrictiveThan(b));
  EXPECT_TRUE(a.IsEqualOrLessRestrictiveThan(a));
  EXPECT_TRUE(b.IsEqualOrLessRestrictiveThan(b));

  b.set_gpu_compatible(true);
  // The set of flags in b is not a subset of those in a.
  EXPECT_TRUE(a.IsEqualOrLessRestrictiveThan(b));
  EXPECT_FALSE(b.IsEqualOrLessRestrictiveThan(a));
  EXPECT_TRUE(a.IsEqualOrLessRestrictiveThan(a));
  EXPECT_TRUE(b.IsEqualOrLessRestrictiveThan(b));

  a.set_nic_compatible(true);
  // Neither a nor b is a subset of the other.
  EXPECT_FALSE(a.IsEqualOrLessRestrictiveThan(b));
  EXPECT_FALSE(b.IsEqualOrLessRestrictiveThan(a));

  a.set_gpu_compatible(true);
  // The set of flags in b is a proper subset of those in a.
  EXPECT_TRUE(b.IsEqualOrLessRestrictiveThan(a));
  EXPECT_FALSE(a.IsEqualOrLessRestrictiveThan(b));
}

TEST(AllocatorAttributesTest, Merge) {
  AllocatorAttributes a, b;

  // Merging nic_compatible=True and nic_compatible=False results in
  // nic_compatible=True.
  EXPECT_EQ(a.value, 0);
  EXPECT_EQ(b.value, 0);
  EXPECT_FALSE(a.nic_compatible());
  EXPECT_FALSE(b.nic_compatible());
  b.set_nic_compatible(true);
  a.Merge(b);
  EXPECT_TRUE(a.nic_compatible());
  EXPECT_TRUE(b.nic_compatible());

  // a.Merge(b) does not change b.
  EXPECT_EQ(a.scope_id, 0);
  EXPECT_EQ(b.scope_id, 0);
  a.scope_id = 1;
  a.Merge(b);
  EXPECT_EQ(a.scope_id, 1);
  EXPECT_EQ(b.scope_id, 0);

  // If a.scope_id=1 and b.scope_id=0, then b.Merge(a) results in b.scope_id=1.
  a.scope_id = 1;
  b.scope_id = 0;
  b.Merge(a);
  EXPECT_EQ(a.scope_id, 1);
  EXPECT_EQ(b.scope_id, 1);

  // If a.scope_id and b.scope_id are same, then merge leaves them unchanged.
  a.scope_id = 2;
  b.scope_id = 2;
  a.Merge(b);
  EXPECT_EQ(a.scope_id, 2);
  EXPECT_EQ(b.scope_id, 2);
}

TEST(AllocatorAttributesDeathTest, MergeDifferentScopeIds) {
  AllocatorAttributes a, b;
  // If a.scope_id and b.scope_id are both positive but different, then
  // a.Merge(b) should cause a CHECK failure.
  a.scope_id = 3;
  b.scope_id = 4;
  EXPECT_DEATH({ a.Merge(b); }, "");
}

TEST(CPUAllocatorTest, Simple) {
  EnableCPUAllocatorStats();
  Allocator* a = cpu_allocator();
  std::vector<void*> ptrs;
  for (int s = 1; s < 1024; s++) {
    void* raw = a->AllocateRaw(1, s);
    ptrs.push_back(raw);
  }
  std::sort(ptrs.begin(), ptrs.end());
  CheckStats(a, 1023, 552640, 552640, 1024);
  for (size_t i = 0; i < ptrs.size(); i++) {
    if (i > 0) {
      CHECK_NE(ptrs[i], ptrs[i - 1]);  // No dups
    }
    a->DeallocateRaw(ptrs[i]);
  }
  CheckStats(a, 1023, 0, 552640, 1024);
  float* t1 = TypedAllocator::Allocate<float>(a, 1024, {});
  double* t2 = TypedAllocator::Allocate<double>(a, 1048576, {});
  CheckStats(a, 1025, 1048576 * sizeof(double) + 1024 * sizeof(float),
             1048576 * sizeof(double) + 1024 * sizeof(float),
             1048576 * sizeof(double));

  TypedAllocator::Deallocate(a, t1, 1024);
  TypedAllocator::Deallocate(a, t2, 1048576);

  CheckStats(a, 1025, 0, 1048576 * sizeof(double) + 1024 * sizeof(float),
             1048576 * sizeof(double));
  CHECK(a->ClearStats());
  CheckStats(a, 0, 0, 0, 0);
  DisableCPUAllocatorStats();
}

// Define a struct that we will use to observe behavior in the unit tests
struct TestStruct {
  int x;  // not used just want to make sure sizeof(TestStruct) > 1
};

TEST(CPUAllocatorTest, CheckStructSize) { CHECK_GT(sizeof(TestStruct), 1); }

TEST(CPUAllocatorTest, AllocateOverflowMaxSizeT) {
  Allocator* a = cpu_allocator();

  // The maximum size_t value will definitely overflow.
  size_t count_to_allocate = std::numeric_limits<size_t>::max();
  TestStruct* const test_pointer =
      TypedAllocator::Allocate<TestStruct>(a, count_to_allocate, {});

  CHECK_EQ(test_pointer, reinterpret_cast<TestStruct*>(NULL));
}

TEST(CPUAllocatorTest, AllocateOverflowSmallest) {
  Allocator* a = cpu_allocator();

  // count_to_allocate is the smallest count that will cause overflow.
  const size_t count_to_allocate =
      (std::numeric_limits<size_t>::max() / sizeof(TestStruct)) + 1;
  TestStruct* const test_pointer =
      TypedAllocator::Allocate<TestStruct>(a, count_to_allocate, {});

  CHECK_EQ(test_pointer, reinterpret_cast<TestStruct*>(NULL));
}

TEST(CPUAllocatorTest, Sizes) {
  Allocator* a = cpu_allocator();

  EXPECT_EQ(false, a->TracksAllocationSizes());
}

TEST(CPUAllocatorTest, ProfilerReporting) {
  // TODO(b/196611863): Make debugging work even without GetAllocatedSize.
  void* p = port::AlignedMalloc(8, 1);
  const std::size_t alloc_size = port::MallocExtension_GetAllocatedSize(p);
  port::AlignedFree(p);
  if (alloc_size == 0) {
    LOG(WARNING) << "Skipping Memory Debugging test. It requires "
                 << "port::MallocExtension_GetAllocatedSize to work.";
    return;
  }

  EnableCPUAllocatorStats();
  Allocator* a = cpu_allocator();

  // Allocate something before profiling starts
  void* p1 = a->AllocateRaw(1, 16);

  // Start profiling
  std::unique_ptr<ProfilerSession> profiler =
      tensorflow::ProfilerSession::Create(
          tensorflow::ProfilerSession::DefaultOptions());

  // Profiled allocations
  void* p2 = a->AllocateRaw(1, 32);
  a->DeallocateRaw(p1);

  // Get profiling results
  tensorflow::profiler::XSpace xspace;
  EXPECT_EQ(tensorflow::Status::OK(), profiler->CollectData(&xspace));

  // Validate the output
  ASSERT_EQ(xspace.planes_size(), 1) << "XSpace: " << xspace.DebugString();
  const auto& plane = xspace.planes(0);
  ::tensorflow::profiler::XPlaneVisitor xplane(&plane);

  ASSERT_EQ(plane.name(), ::tensorflow::profiler::kHostThreadsPlaneName)
      << "XSpace: " << xspace.DebugString();
  ASSERT_EQ(plane.event_metadata_size(), 2)
      << "XSpace: " << xspace.DebugString();

  const auto& line = plane.lines(0);
  ASSERT_EQ(line.events_size(), 2) << "XSpace: " << xspace.DebugString();
  const auto& events = line.events();

  ::tensorflow::profiler::XEventVisitor e0(&xplane, &line, &events[0]);
  EXPECT_EQ(e0.Name(), "MemoryAllocation")
      << "XSpace: " << xspace.DebugString();
  {
    absl::optional<std::string> bytes_allocated, peak_bytes_in_use,
        requested_bytes, allocation_bytes;
    e0.ForEachStat([&](const ::tensorflow::profiler::XStatVisitor& stat) {
      LOG(ERROR) << "STAT " << stat.Name() << ": " << stat.ToString();
      if (stat.Name() == "bytes_allocated") {
        bytes_allocated = stat.ToString();
      } else if (stat.Name() == "peak_bytes_in_use") {
        peak_bytes_in_use = stat.ToString();
      } else if (stat.Name() == "requested_bytes") {
        requested_bytes = stat.ToString();
      } else if (stat.Name() == "allocation_bytes") {
        allocation_bytes = stat.ToString();
      }
    });
    ASSERT_TRUE(bytes_allocated && peak_bytes_in_use && requested_bytes &&
                allocation_bytes)
        << "XSpace: " << xspace.DebugString();
    EXPECT_EQ(*bytes_allocated, "48") << "XSpace: " << xspace.DebugString();
    EXPECT_EQ(*peak_bytes_in_use, "48") << "XSpace: " << xspace.DebugString();
    EXPECT_EQ(*requested_bytes, "32") << "XSpace: " << xspace.DebugString();
    EXPECT_EQ(*allocation_bytes, "32") << "XSpace: " << xspace.DebugString();
  }

  ::tensorflow::profiler::XEventVisitor e1(&xplane, &line, &events[1]);
  EXPECT_EQ(e1.Name(), "MemoryDeallocation")
      << "XSpace: " << xspace.DebugString();
  {
    absl::optional<std::string> bytes_allocated, peak_bytes_in_use,
        allocation_bytes;
    e1.ForEachStat([&](const ::tensorflow::profiler::XStatVisitor& stat) {
      if (stat.Name() == "bytes_allocated") {
        bytes_allocated = stat.ToString();
      } else if (stat.Name() == "peak_bytes_in_use") {
        peak_bytes_in_use = stat.ToString();
      } else if (stat.Name() == "allocation_bytes") {
        allocation_bytes = stat.ToString();
      }
    });
    ASSERT_TRUE(bytes_allocated && peak_bytes_in_use && allocation_bytes)
        << "XSpace: " << xspace.DebugString();
    EXPECT_EQ(*bytes_allocated, "32") << "XSpace: " << xspace.DebugString();
    EXPECT_EQ(*peak_bytes_in_use, "48") << "XSpace: " << xspace.DebugString();
    EXPECT_EQ(*allocation_bytes, "16") << "XSpace: " << xspace.DebugString();
  }

  // Cleanup
  a->DeallocateRaw(p2);
  DisableCPUAllocatorStats();
}

namespace {

AllocatorAttributes DeviceAllocatorAttribute() {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSallocator_testDTcc mht_1(mht_1_v, 482, "", "./tensorflow/core/framework/allocator_test.cc", "DeviceAllocatorAttribute");

  AllocatorAttributes attr;
  attr.value |= (0x1 << 24);
  return attr;
}

bool HasDeviceAllocatorAttribute(const AllocatorAttributes& attr) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSallocator_testDTcc mht_2(mht_2_v, 491, "", "./tensorflow/core/framework/allocator_test.cc", "HasDeviceAllocatorAttribute");

  return attr.value & (0x1 << 24);
}

}  // namespace

TEST(CustomAllocatorAttributes, TestSetterAndGetter) {
  AllocatorAttributes attr = DeviceAllocatorAttribute();
  EXPECT_TRUE(HasDeviceAllocatorAttribute(attr));
  EXPECT_FALSE(HasDeviceAllocatorAttribute(AllocatorAttributes()));
}

static void BM_Allocation(::testing::benchmark::State& state) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSallocator_testDTcc mht_3(mht_3_v, 506, "", "./tensorflow/core/framework/allocator_test.cc", "BM_Allocation");

  const int arg = state.range(0);

  Allocator* a = cpu_allocator();
  // Exercise a few different allocation sizes
  std::vector<int> sizes = {256, 4096, 16384, 524288, 512, 1048576};
  int size_index = 0;

  if (arg) EnableCPUAllocatorStats();
  for (auto s : state) {
    int bytes = sizes[size_index++ % sizes.size()];
    void* p = a->AllocateRaw(1, bytes);
    a->DeallocateRaw(p);
  }
  if (arg) DisableCPUAllocatorStats();
}
BENCHMARK(BM_Allocation)->Arg(0)->Arg(1);

}  // namespace tensorflow
