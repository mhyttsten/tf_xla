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
class MHTracer_DTPStensorflowPScPSexperimentalPSstream_executorPSstream_executor_testDTcc {
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
   MHTracer_DTPStensorflowPScPSexperimentalPSstream_executorPSstream_executor_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScPSexperimentalPSstream_executorPSstream_executor_testDTcc() {
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

Licensed under the Apache License, Version 2.0(the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/
#include "tensorflow/c/experimental/stream_executor/stream_executor.h"

#include "tensorflow/c/experimental/stream_executor/stream_executor_internal.h"
#include "tensorflow/c/experimental/stream_executor/stream_executor_test_util.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/protobuf/error_codes.pb.h"
#include "tensorflow/stream_executor/event.h"
#include "tensorflow/stream_executor/multi_platform_manager.h"
#include "tensorflow/stream_executor/stream.h"
#include "tensorflow/stream_executor/stream_executor_pimpl.h"
#include "tensorflow/stream_executor/timer.h"

namespace stream_executor {
namespace {

/*** Registration tests ***/
TEST(StreamExecutor, SuccessfulRegistration) {
  auto plugin_init = [](SE_PlatformRegistrationParams* const params,
                        TF_Status* const status) -> void {
    TF_SetStatus(status, TF_OK, "");
    test_util::PopulateDefaultPlatformRegistrationParams(params);
  };
  std::string device_type, platform_name;
  port::Status status =
      InitStreamExecutorPlugin(plugin_init, &device_type, &platform_name);
  TF_ASSERT_OK(status);
  port::StatusOr<Platform*> maybe_platform =
      MultiPlatformManager::PlatformWithName("MY_DEVICE");
  TF_ASSERT_OK(maybe_platform.status());
  Platform* platform = maybe_platform.ConsumeValueOrDie();
  ASSERT_EQ(platform->Name(), test_util::kDeviceName);
  ASSERT_EQ(platform->VisibleDeviceCount(), test_util::kDeviceCount);

  port::StatusOr<StreamExecutor*> maybe_executor =
      platform->ExecutorForDevice(0);
  TF_ASSERT_OK(maybe_executor.status());
}

TEST(StreamExecutor, NameNotSet) {
  auto plugin_init = [](SE_PlatformRegistrationParams* const params,
                        TF_Status* const status) -> void {
    TF_SetStatus(status, TF_OK, "");
    test_util::PopulateDefaultPlatformRegistrationParams(params);
    params->platform->name = nullptr;
  };

  std::string device_type, platform_name;
  port::Status status =
      InitStreamExecutorPlugin(plugin_init, &device_type, &platform_name);
  ASSERT_EQ(status.code(), tensorflow::error::FAILED_PRECONDITION);
  ASSERT_EQ(status.error_message(), "'name' field in SP_Platform must be set.");
}

TEST(StreamExecutor, InvalidNameWithSemicolon) {
  auto plugin_init = [](SE_PlatformRegistrationParams* const params,
                        TF_Status* const status) -> void {
    TF_SetStatus(status, TF_OK, "");
    test_util::PopulateDefaultPlatformRegistrationParams(params);
    params->platform->name = "INVALID:NAME";
  };

  std::string device_type, platform_name;
  port::Status status =
      InitStreamExecutorPlugin(plugin_init, &device_type, &platform_name);
  ASSERT_EQ(status.code(), tensorflow::error::FAILED_PRECONDITION);
  EXPECT_THAT(
      status.error_message(),
      testing::ContainsRegex("Device name/type 'INVALID:NAME' must match"));
}

TEST(StreamExecutor, InvalidNameWithSlash) {
  auto plugin_init = [](SE_PlatformRegistrationParams* const params,
                        TF_Status* const status) -> void {
    TF_SetStatus(status, TF_OK, "");
    test_util::PopulateDefaultPlatformRegistrationParams(params);
    params->platform->name = "INVALID/";
  };

  std::string device_type, platform_name;
  port::Status status =
      InitStreamExecutorPlugin(plugin_init, &device_type, &platform_name);
  ASSERT_EQ(status.code(), tensorflow::error::FAILED_PRECONDITION);
  EXPECT_THAT(status.error_message(),
              testing::ContainsRegex("Device name/type 'INVALID/' must match"));
}

TEST(StreamExecutor, CreateDeviceNotSet) {
  auto plugin_init = [](SE_PlatformRegistrationParams* const params,
                        TF_Status* const status) -> void {
    TF_SetStatus(status, TF_OK, "");
    test_util::PopulateDefaultPlatformRegistrationParams(params);
    params->platform_fns->create_device = nullptr;
  };

  std::string device_type, platform_name;
  port::Status status =
      InitStreamExecutorPlugin(plugin_init, &device_type, &platform_name);
  ASSERT_EQ(status.code(), tensorflow::error::FAILED_PRECONDITION);
  ASSERT_EQ(status.error_message(),
            "'create_device' field in SP_PlatformFns must be set.");
}

TEST(StreamExecutor, UnifiedMemoryAllocateNotSet) {
  auto plugin_init = [](SE_PlatformRegistrationParams* const params,
                        TF_Status* const status) -> void {
    TF_SetStatus(status, TF_OK, "");
    test_util::PopulateDefaultPlatformRegistrationParams(params);
    params->platform->supports_unified_memory = true;
  };

  std::string device_type, platform_name;
  port::Status status =
      InitStreamExecutorPlugin(plugin_init, &device_type, &platform_name);
  ASSERT_EQ(status.code(), tensorflow::error::FAILED_PRECONDITION);
  ASSERT_EQ(
      status.error_message(),
      "'unified_memory_allocate' field in SP_StreamExecutor must be set.");
}

/*** StreamExecutor behavior tests ***/
class StreamExecutorTest : public ::testing::Test {
 protected:
  StreamExecutorTest() {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScPSexperimentalPSstream_executorPSstream_executor_testDTcc mht_0(mht_0_v, 307, "", "./tensorflow/c/experimental/stream_executor/stream_executor_test.cc", "StreamExecutorTest");
}
  void SetUp() override {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScPSexperimentalPSstream_executorPSstream_executor_testDTcc mht_1(mht_1_v, 311, "", "./tensorflow/c/experimental/stream_executor/stream_executor_test.cc", "SetUp");

    test_util::PopulateDefaultPlatform(&platform_, &platform_fns_);
    test_util::PopulateDefaultDeviceFns(&device_fns_);
    test_util::PopulateDefaultStreamExecutor(&se_);
    test_util::PopulateDefaultTimerFns(&timer_fns_);
  }
  void TearDown() override {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScPSexperimentalPSstream_executorPSstream_executor_testDTcc mht_2(mht_2_v, 320, "", "./tensorflow/c/experimental/stream_executor/stream_executor_test.cc", "TearDown");
}

  StreamExecutor* GetExecutor(int ordinal) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScPSexperimentalPSstream_executorPSstream_executor_testDTcc mht_3(mht_3_v, 325, "", "./tensorflow/c/experimental/stream_executor/stream_executor_test.cc", "GetExecutor");

    if (!cplatform_) {
      cplatform_ = absl::make_unique<CPlatform>(
          platform_, test_util::DestroyPlatform, platform_fns_,
          test_util::DestroyPlatformFns, device_fns_, se_, timer_fns_);
    }
    port::StatusOr<StreamExecutor*> maybe_executor =
        cplatform_->ExecutorForDevice(ordinal);
    TF_CHECK_OK(maybe_executor.status());
    return maybe_executor.ConsumeValueOrDie();
  }
  SP_Platform platform_;
  SP_PlatformFns platform_fns_;
  SP_DeviceFns device_fns_;
  SP_StreamExecutor se_;
  SP_TimerFns timer_fns_;
  std::unique_ptr<CPlatform> cplatform_;
};

TEST_F(StreamExecutorTest, Allocate) {
  se_.allocate = [](const SP_Device* const device, uint64_t size,
                    int64_t memory_space, SP_DeviceMemoryBase* const mem) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScPSexperimentalPSstream_executorPSstream_executor_testDTcc mht_4(mht_4_v, 349, "", "./tensorflow/c/experimental/stream_executor/stream_executor_test.cc", "lambda");

    mem->struct_size = SP_DEVICE_MEMORY_BASE_STRUCT_SIZE;
    mem->opaque = malloc(size);
    mem->size = size;
  };
  se_.deallocate = [](const SP_Device* const device,
                      SP_DeviceMemoryBase* const mem) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScPSexperimentalPSstream_executorPSstream_executor_testDTcc mht_5(mht_5_v, 358, "", "./tensorflow/c/experimental/stream_executor/stream_executor_test.cc", "lambda");

    EXPECT_EQ(mem->size, 2 * sizeof(int));
    free(mem->opaque);
    mem->opaque = nullptr;
    mem->size = 0;
  };
  StreamExecutor* executor = GetExecutor(0);
  DeviceMemory<int> mem = executor->AllocateArray<int>(2);
  ASSERT_NE(mem.opaque(), nullptr);
  ASSERT_EQ(mem.size(), 2 * sizeof(int));
  executor->Deallocate(&mem);
  ASSERT_EQ(mem.opaque(), nullptr);
}

TEST_F(StreamExecutorTest, HostMemoryAllocate) {
  static bool allocate_called = false;
  static bool deallocate_called = false;
  se_.host_memory_allocate = [](const SP_Device* const device, uint64_t size) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScPSexperimentalPSstream_executorPSstream_executor_testDTcc mht_6(mht_6_v, 378, "", "./tensorflow/c/experimental/stream_executor/stream_executor_test.cc", "lambda");

    allocate_called = true;
    return malloc(size);
  };
  se_.host_memory_deallocate = [](const SP_Device* const device, void* mem) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScPSexperimentalPSstream_executorPSstream_executor_testDTcc mht_7(mht_7_v, 385, "", "./tensorflow/c/experimental/stream_executor/stream_executor_test.cc", "lambda");

    free(mem);
    deallocate_called = true;
  };
  StreamExecutor* executor = GetExecutor(0);
  ASSERT_FALSE(allocate_called);
  void* mem = executor->HostMemoryAllocate(8);
  ASSERT_NE(mem, nullptr);
  ASSERT_TRUE(allocate_called);
  ASSERT_FALSE(deallocate_called);
  executor->HostMemoryDeallocate(mem);
  ASSERT_TRUE(deallocate_called);
}

TEST_F(StreamExecutorTest, UnifiedMemoryAllocate) {
  static bool allocate_called = false;
  static bool deallocate_called = false;
  se_.unified_memory_allocate = [](const SP_Device* const device,
                                   uint64_t size) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScPSexperimentalPSstream_executorPSstream_executor_testDTcc mht_8(mht_8_v, 406, "", "./tensorflow/c/experimental/stream_executor/stream_executor_test.cc", "lambda");

    allocate_called = true;
    return malloc(size);
  };
  se_.unified_memory_deallocate = [](const SP_Device* const device, void* mem) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScPSexperimentalPSstream_executorPSstream_executor_testDTcc mht_9(mht_9_v, 413, "", "./tensorflow/c/experimental/stream_executor/stream_executor_test.cc", "lambda");

    free(mem);
    deallocate_called = true;
  };
  StreamExecutor* executor = GetExecutor(0);
  ASSERT_FALSE(allocate_called);
  void* mem = executor->UnifiedMemoryAllocate(8);
  ASSERT_NE(mem, nullptr);
  ASSERT_TRUE(allocate_called);
  ASSERT_FALSE(deallocate_called);
  executor->UnifiedMemoryDeallocate(mem);
  ASSERT_TRUE(deallocate_called);
}

TEST_F(StreamExecutorTest, GetAllocatorStats) {
  se_.get_allocator_stats = [](const SP_Device* const device,
                               SP_AllocatorStats* const stat) -> TF_Bool {
    stat->struct_size = SP_ALLOCATORSTATS_STRUCT_SIZE;
    stat->bytes_in_use = 123;
    return true;
  };

  StreamExecutor* executor = GetExecutor(0);
  absl::optional<AllocatorStats> optional_stats = executor->GetAllocatorStats();
  ASSERT_TRUE(optional_stats.has_value());
  AllocatorStats stats = optional_stats.value();
  ASSERT_EQ(stats.bytes_in_use, 123);
}

TEST_F(StreamExecutorTest, DeviceMemoryUsage) {
  se_.device_memory_usage = [](const SP_Device* const device,
                               int64_t* const free,
                               int64_t* const total) -> TF_Bool {
    *free = 45;
    *total = 7;
    return true;
  };

  StreamExecutor* executor = GetExecutor(0);
  int64_t free = 0;
  int64_t total = 0;
  executor->DeviceMemoryUsage(&free, &total);
  ASSERT_EQ(free, 45);
  ASSERT_EQ(total, 7);
}

TEST_F(StreamExecutorTest, CreateStream) {
  static bool stream_created = false;
  static bool stream_deleted = false;
  se_.create_stream = [](const SP_Device* const device, SP_Stream* stream,
                         TF_Status* const status) -> void {
    *stream = new SP_Stream_st(14);
    stream_created = true;
  };
  se_.destroy_stream = [](const SP_Device* const device,
                          SP_Stream stream) -> void {
    auto custom_stream = static_cast<SP_Stream_st*>(stream);
    ASSERT_EQ(custom_stream->stream_id, 14);
    delete custom_stream;
    stream_deleted = true;
  };

  StreamExecutor* executor = GetExecutor(0);
  ASSERT_FALSE(stream_created);
  Stream* stream = new Stream(executor);
  stream->Init();
  ASSERT_TRUE(stream->ok());
  ASSERT_TRUE(stream_created);
  ASSERT_FALSE(stream_deleted);
  delete stream;
  ASSERT_TRUE(stream_deleted);
}

TEST_F(StreamExecutorTest, CreateStreamDependency) {
  static bool create_stream_dependency_called = false;
  se_.create_stream_dependency = [](const SP_Device* const device,
                                    SP_Stream dependent, SP_Stream other,
                                    TF_Status* const status) {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScPSexperimentalPSstream_executorPSstream_executor_testDTcc mht_10(mht_10_v, 493, "", "./tensorflow/c/experimental/stream_executor/stream_executor_test.cc", "lambda");

    TF_SetStatus(status, TF_OK, "");
    create_stream_dependency_called = true;
  };

  StreamExecutor* executor = GetExecutor(0);
  Stream dependent(executor);
  dependent.Init();
  Stream other(executor);
  other.Init();
  ASSERT_FALSE(create_stream_dependency_called);
  dependent.ThenWaitFor(&other);
  ASSERT_TRUE(create_stream_dependency_called);
}

TEST_F(StreamExecutorTest, StreamStatus) {
  static bool status_ok = true;
  se_.get_stream_status = [](const SP_Device* const device, SP_Stream stream,
                             TF_Status* const status) -> void {
    if (status_ok) {
      TF_SetStatus(status, TF_OK, "");
    } else {
      TF_SetStatus(status, TF_INTERNAL, "Test error");
    }
  };

  StreamExecutor* executor = GetExecutor(0);
  Stream stream(executor);
  stream.Init();
  ASSERT_TRUE(stream.ok());
  TF_ASSERT_OK(stream.RefreshStatus());
  status_ok = false;
  auto updated_status = stream.RefreshStatus();
  ASSERT_FALSE(stream.ok());
  ASSERT_EQ(updated_status.error_message(), "Test error");
}

TEST_F(StreamExecutorTest, CreateEvent) {
  static bool event_created = false;
  static bool event_deleted = false;
  se_.create_event = [](const SP_Device* const device, SP_Event* event,
                        TF_Status* const status) -> void {
    *event = new SP_Event_st(123);
    event_created = true;
  };
  se_.destroy_event = [](const SP_Device* const device,
                         SP_Event event) -> void {
    auto custom_event = static_cast<SP_Event_st*>(event);
    ASSERT_EQ(custom_event->event_id, 123);
    delete custom_event;
    event_deleted = true;
  };

  StreamExecutor* executor = GetExecutor(0);
  ASSERT_FALSE(event_created);
  Event* event = new Event(executor);
  event->Init();
  ASSERT_TRUE(event_created);
  ASSERT_FALSE(event_deleted);
  delete event;
  ASSERT_TRUE(event_deleted);
}

TEST_F(StreamExecutorTest, PollForEventStatus) {
  static SE_EventStatus event_status = SE_EVENT_COMPLETE;
  se_.create_event = [](const SP_Device* const device, SP_Event* event,
                        TF_Status* const status) -> void {
    *event = new SP_Event_st(123);
  };
  se_.destroy_event = [](const SP_Device* const device,
                         SP_Event event) -> void { delete event; };
  se_.get_event_status = [](const SP_Device* const device,
                            SP_Event event) -> SE_EventStatus {
    EXPECT_EQ(event->event_id, 123);
    return event_status;
  };

  StreamExecutor* executor = GetExecutor(0);
  Event event(executor);
  event.Init();
  ASSERT_EQ(event.PollForStatus(), Event::Status::kComplete);
  event_status = SE_EVENT_ERROR;
  ASSERT_EQ(event.PollForStatus(), Event::Status::kError);
}

TEST_F(StreamExecutorTest, RecordAndWaitForEvent) {
  static bool record_called = false;
  static bool wait_called = false;
  se_.create_stream = [](const SP_Device* const device, SP_Stream* stream,
                         TF_Status* const status) -> void {
    *stream = new SP_Stream_st(1);
  };
  se_.destroy_stream = [](const SP_Device* const device,
                          SP_Stream stream) -> void { delete stream; };
  se_.create_event = [](const SP_Device* const device, SP_Event* event,
                        TF_Status* const status) -> void {
    *event = new SP_Event_st(2);
  };
  se_.destroy_event = [](const SP_Device* const device,
                         SP_Event event) -> void { delete event; };
  se_.record_event = [](const SP_Device* const device, SP_Stream stream,
                        SP_Event event, TF_Status* const status) {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScPSexperimentalPSstream_executorPSstream_executor_testDTcc mht_11(mht_11_v, 597, "", "./tensorflow/c/experimental/stream_executor/stream_executor_test.cc", "lambda");

    EXPECT_EQ(stream->stream_id, 1);
    EXPECT_EQ(event->event_id, 2);
    TF_SetStatus(status, TF_OK, "");
    record_called = true;
  };
  se_.wait_for_event = [](const SP_Device* const device, SP_Stream stream,
                          SP_Event event, TF_Status* const status) {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScPSexperimentalPSstream_executorPSstream_executor_testDTcc mht_12(mht_12_v, 607, "", "./tensorflow/c/experimental/stream_executor/stream_executor_test.cc", "lambda");

    EXPECT_EQ(stream->stream_id, 1);
    EXPECT_EQ(event->event_id, 2);
    TF_SetStatus(status, TF_OK, "");
    wait_called = true;
  };

  StreamExecutor* executor = GetExecutor(0);
  Event event(executor);
  event.Init();
  Stream stream(executor);
  stream.Init();
  ASSERT_FALSE(record_called);
  stream.ThenRecordEvent(&event);
  ASSERT_TRUE(record_called);
  ASSERT_FALSE(wait_called);
  stream.ThenWaitFor(&event);
  ASSERT_TRUE(wait_called);
}

TEST_F(StreamExecutorTest, CreateTimer) {
  static bool timer_created = false;
  static bool timer_deleted = false;
  se_.create_timer = [](const SP_Device* const device, SP_Timer* timer,
                        TF_Status* const status) -> void {
    *timer = new SP_Timer_st(25);
    timer_created = true;
  };
  se_.destroy_timer = [](const SP_Device* const device,
                         SP_Timer timer) -> void {
    auto custom_timer = static_cast<SP_Timer_st*>(timer);
    EXPECT_EQ(custom_timer->timer_id, 25);
    delete custom_timer;
    timer_deleted = true;
  };

  StreamExecutor* executor = GetExecutor(0);
  ASSERT_FALSE(timer_created);
  Stream stream(executor);
  stream.Init();
  Timer* timer = new Timer(executor);
  stream.InitTimer(timer);
  ASSERT_TRUE(stream.ok());
  ASSERT_TRUE(timer_created);
  ASSERT_FALSE(timer_deleted);
  delete timer;
  ASSERT_TRUE(timer_deleted);
}

TEST_F(StreamExecutorTest, StartTimer) {
  static bool start_called = false;
  static bool stop_called = false;
  static TF_Code start_timer_status = TF_OK;
  static TF_Code stop_timer_status = TF_OK;
  se_.create_timer = [](const SP_Device* const device, SP_Timer* timer,
                        TF_Status* const status) -> void {
    *timer = new SP_Timer_st(7);
  };
  se_.destroy_timer = [](const SP_Device* const device,
                         SP_Timer timer) -> void { delete timer; };
  se_.start_timer = [](const SP_Device* const device, SP_Stream stream,
                       SP_Timer timer, TF_Status* const status) {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPScPSexperimentalPSstream_executorPSstream_executor_testDTcc mht_13(mht_13_v, 671, "", "./tensorflow/c/experimental/stream_executor/stream_executor_test.cc", "lambda");

    TF_SetStatus(status, start_timer_status, "");
    EXPECT_EQ(timer->timer_id, 7);
    start_called = true;
  };
  se_.stop_timer = [](const SP_Device* const device, SP_Stream stream,
                      SP_Timer timer, TF_Status* const status) {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPScPSexperimentalPSstream_executorPSstream_executor_testDTcc mht_14(mht_14_v, 680, "", "./tensorflow/c/experimental/stream_executor/stream_executor_test.cc", "lambda");

    TF_SetStatus(status, stop_timer_status, "");
    EXPECT_EQ(timer->timer_id, 7);
    stop_called = true;
  };
  StreamExecutor* executor = GetExecutor(0);
  Stream stream(executor);
  stream.Init();
  Timer timer(executor);
  stream.InitTimer(&timer);

  // Check both start and stop succeed
  ASSERT_FALSE(start_called);
  stream.ThenStartTimer(&timer);
  ASSERT_TRUE(start_called);
  ASSERT_FALSE(stop_called);
  stream.ThenStopTimer(&timer);
  ASSERT_TRUE(stop_called);

  // Check start timer fails
  ASSERT_TRUE(stream.ok());
  start_timer_status = TF_UNKNOWN;
  stream.ThenStartTimer(&timer);
  ASSERT_FALSE(stream.ok());

  // Check stop timer fails
  start_timer_status = TF_OK;
  stop_timer_status = TF_UNKNOWN;
  Stream stream2(executor);
  stream2.Init();
  Timer timer2(executor);
  stream2.InitTimer(&timer2);
  stream2.ThenStartTimer(&timer2);
  ASSERT_TRUE(stream2.ok());
  stream2.ThenStopTimer(&timer2);
  ASSERT_FALSE(stream2.ok());
}

TEST_F(StreamExecutorTest, TimerFns) {
  se_.create_timer = [](const SP_Device* const device, SP_Timer* timer,
                        TF_Status* const status) -> void {
    *timer = new SP_Timer_st(25000);
  };
  se_.destroy_timer = [](const SP_Device* const device,
                         SP_Timer timer) -> void { delete timer; };

  StreamExecutor* executor = GetExecutor(0);
  Stream stream(executor);
  stream.Init();
  Timer timer(executor);
  stream.InitTimer(&timer);
  // Our test nanoseconds function just returns value
  // passed to SP_Timer_st constructor.
  ASSERT_EQ(timer.Nanoseconds(), 25000);
  ASSERT_EQ(timer.Microseconds(), 25);
}

TEST_F(StreamExecutorTest, MemcpyToHost) {
  se_.create_stream = [](const SP_Device* const device, SP_Stream* stream,
                         TF_Status* const status) -> void {
    *stream = new SP_Stream_st(14);
  };
  se_.destroy_stream = [](const SP_Device* const device,
                          SP_Stream stream) -> void { delete stream; };

  se_.memcpy_dtoh = [](const SP_Device* const device, SP_Stream stream,
                       void* host_dst,
                       const SP_DeviceMemoryBase* const device_src,
                       uint64_t size, TF_Status* const status) {
   std::vector<std::string> mht_15_v;
   MHTracer_DTPStensorflowPScPSexperimentalPSstream_executorPSstream_executor_testDTcc mht_15(mht_15_v, 751, "", "./tensorflow/c/experimental/stream_executor/stream_executor_test.cc", "lambda");

    TF_SetStatus(status, TF_OK, "");
    EXPECT_EQ(stream->stream_id, 14);
    std::memcpy(host_dst, device_src->opaque, size);
  };

  StreamExecutor* executor = GetExecutor(0);
  Stream stream(executor);
  stream.Init();
  size_t size = sizeof(int);
  int src_data = 34;
  int dst_data = 2;
  DeviceMemoryBase device_src(&src_data, size);
  Stream& stream_ref = stream.ThenMemcpy(&dst_data, device_src, size);
  ASSERT_EQ(dst_data, 34);
  ASSERT_EQ(stream_ref.implementation(), stream.implementation());
}

TEST_F(StreamExecutorTest, MemcpyFromHost) {
  se_.memcpy_htod = [](const SP_Device* const device, SP_Stream stream,
                       SP_DeviceMemoryBase* const device_dst,
                       const void* host_src, uint64_t size,
                       TF_Status* const status) {
   std::vector<std::string> mht_16_v;
   MHTracer_DTPStensorflowPScPSexperimentalPSstream_executorPSstream_executor_testDTcc mht_16(mht_16_v, 776, "", "./tensorflow/c/experimental/stream_executor/stream_executor_test.cc", "lambda");

    TF_SetStatus(status, TF_OK, "");
    std::memcpy(device_dst->opaque, host_src, size);
  };

  StreamExecutor* executor = GetExecutor(0);
  Stream stream(executor);
  stream.Init();
  size_t size = sizeof(int);
  int src_data = 18;
  int dst_data = 0;
  DeviceMemoryBase device_dst(&dst_data, size);
  stream.ThenMemcpy(&device_dst, &src_data, size);
  ASSERT_EQ(dst_data, 18);
}

TEST_F(StreamExecutorTest, MemcpyDeviceToDevice) {
  se_.memcpy_dtod = [](const SP_Device* const device, SP_Stream stream,
                       SP_DeviceMemoryBase* const device_dst,
                       const SP_DeviceMemoryBase* const device_src,
                       uint64_t size, TF_Status* const status) {
   std::vector<std::string> mht_17_v;
   MHTracer_DTPStensorflowPScPSexperimentalPSstream_executorPSstream_executor_testDTcc mht_17(mht_17_v, 799, "", "./tensorflow/c/experimental/stream_executor/stream_executor_test.cc", "lambda");

    TF_SetStatus(status, TF_OK, "");
    std::memcpy(device_dst->opaque, device_src->opaque, size);
  };

  StreamExecutor* executor = GetExecutor(0);
  Stream stream(executor);
  stream.Init();
  size_t size = sizeof(int);
  int src_data = 18;
  int dst_data = 0;
  DeviceMemoryBase device_dst(&dst_data, size);
  DeviceMemoryBase device_src(&src_data, size);
  stream.ThenMemcpy(&device_dst, device_src, size);
  ASSERT_EQ(dst_data, 18);
}

TEST_F(StreamExecutorTest, SyncMemcpyToHost) {
  se_.sync_memcpy_dtoh = [](const SP_Device* const device, void* host_dst,
                            const SP_DeviceMemoryBase* const device_src,
                            uint64_t size, TF_Status* const status) {
   std::vector<std::string> mht_18_v;
   MHTracer_DTPStensorflowPScPSexperimentalPSstream_executorPSstream_executor_testDTcc mht_18(mht_18_v, 822, "", "./tensorflow/c/experimental/stream_executor/stream_executor_test.cc", "lambda");

    TF_SetStatus(status, TF_OK, "");
    std::memcpy(host_dst, device_src->opaque, size);
  };

  StreamExecutor* executor = GetExecutor(0);
  size_t size = sizeof(int);
  int src_data = 34;
  int dst_data = 2;
  DeviceMemoryBase device_src(&src_data, size);
  TF_ASSERT_OK(executor->SynchronousMemcpyD2H(device_src, size, &dst_data));
  ASSERT_EQ(dst_data, 34);
}

TEST_F(StreamExecutorTest, SyncMemcpyFromHost) {
  se_.sync_memcpy_htod =
      [](const SP_Device* const device, SP_DeviceMemoryBase* const device_dst,
         const void* host_src, uint64_t size, TF_Status* const status) {
   std::vector<std::string> mht_19_v;
   MHTracer_DTPStensorflowPScPSexperimentalPSstream_executorPSstream_executor_testDTcc mht_19(mht_19_v, 842, "", "./tensorflow/c/experimental/stream_executor/stream_executor_test.cc", "lambda");

        TF_SetStatus(status, TF_OK, "");
        std::memcpy(device_dst->opaque, host_src, size);
      };

  StreamExecutor* executor = GetExecutor(0);
  size_t size = sizeof(int);
  int src_data = 18;
  int dst_data = 0;
  DeviceMemoryBase device_dst(&dst_data, size);
  TF_ASSERT_OK(executor->SynchronousMemcpyH2D(&src_data, size, &device_dst));
  ASSERT_EQ(dst_data, 18);
}

TEST_F(StreamExecutorTest, SyncMemcpyDeviceToDevice) {
  se_.sync_memcpy_dtod = [](const SP_Device* const device,
                            SP_DeviceMemoryBase* const device_dst,
                            const SP_DeviceMemoryBase* const device_src,
                            uint64_t size, TF_Status* const status) {
   std::vector<std::string> mht_20_v;
   MHTracer_DTPStensorflowPScPSexperimentalPSstream_executorPSstream_executor_testDTcc mht_20(mht_20_v, 863, "", "./tensorflow/c/experimental/stream_executor/stream_executor_test.cc", "lambda");

    TF_SetStatus(status, TF_OK, "");
    std::memcpy(device_dst->opaque, device_src->opaque, size);
  };

  StreamExecutor* executor = GetExecutor(0);
  size_t size = sizeof(int);
  int src_data = 18;
  int dst_data = 0;
  DeviceMemoryBase device_dst(&dst_data, size);
  DeviceMemoryBase device_src(&src_data, size);
  ASSERT_TRUE(executor->SynchronousMemcpy(&device_dst, device_src, size));
  ASSERT_EQ(dst_data, 18);
}

TEST_F(StreamExecutorTest, BlockHostForEvent) {
  static bool block_host_for_event_called = false;
  se_.create_event = [](const SP_Device* const device, SP_Event* event,
                        TF_Status* const status) {
   std::vector<std::string> mht_21_v;
   MHTracer_DTPStensorflowPScPSexperimentalPSstream_executorPSstream_executor_testDTcc mht_21(mht_21_v, 884, "", "./tensorflow/c/experimental/stream_executor/stream_executor_test.cc", "lambda");

    *event = new SP_Event_st(357);
  };
  se_.destroy_event = [](const SP_Device* const device, SP_Event event) {
   std::vector<std::string> mht_22_v;
   MHTracer_DTPStensorflowPScPSexperimentalPSstream_executorPSstream_executor_testDTcc mht_22(mht_22_v, 890, "", "./tensorflow/c/experimental/stream_executor/stream_executor_test.cc", "lambda");

    delete event;
  };
  se_.block_host_for_event = [](const SP_Device* const device, SP_Event event,
                                TF_Status* const status) -> void {
    ASSERT_EQ(event->event_id, 357);
    TF_SetStatus(status, TF_OK, "");
    block_host_for_event_called = true;
  };

  StreamExecutor* executor = GetExecutor(0);
  Stream stream(executor);
  stream.Init();
  ASSERT_FALSE(block_host_for_event_called);
  TF_ASSERT_OK(stream.BlockHostUntilDone());
  ASSERT_TRUE(block_host_for_event_called);
}

TEST_F(StreamExecutorTest, BlockHostUntilDone) {
  static bool block_host_until_done_called = false;
  se_.create_stream = [](const SP_Device* const device, SP_Stream* stream,
                         TF_Status* const status) {
   std::vector<std::string> mht_23_v;
   MHTracer_DTPStensorflowPScPSexperimentalPSstream_executorPSstream_executor_testDTcc mht_23(mht_23_v, 914, "", "./tensorflow/c/experimental/stream_executor/stream_executor_test.cc", "lambda");

    *stream = new SP_Stream_st(58);
  };
  se_.destroy_stream = [](const SP_Device* const device, SP_Stream stream) {
   std::vector<std::string> mht_24_v;
   MHTracer_DTPStensorflowPScPSexperimentalPSstream_executorPSstream_executor_testDTcc mht_24(mht_24_v, 920, "", "./tensorflow/c/experimental/stream_executor/stream_executor_test.cc", "lambda");

    delete stream;
  };
  se_.block_host_until_done = [](const SP_Device* const device,
                                 SP_Stream stream,
                                 TF_Status* const status) -> void {
    ASSERT_EQ(stream->stream_id, 58);
    TF_SetStatus(status, TF_OK, "");
    block_host_until_done_called = true;
  };

  StreamExecutor* executor = GetExecutor(0);
  Stream stream(executor);
  stream.Init();
  ASSERT_FALSE(block_host_until_done_called);
  TF_ASSERT_OK(stream.BlockHostUntilDone());
  ASSERT_TRUE(block_host_until_done_called);
}

TEST_F(StreamExecutorTest, SynchronizeAllActivity) {
  static bool synchronize_all_called = false;
  se_.synchronize_all_activity = [](const SP_Device* const device,
                                    TF_Status* const status) {
   std::vector<std::string> mht_25_v;
   MHTracer_DTPStensorflowPScPSexperimentalPSstream_executorPSstream_executor_testDTcc mht_25(mht_25_v, 945, "", "./tensorflow/c/experimental/stream_executor/stream_executor_test.cc", "lambda");

    TF_SetStatus(status, TF_OK, "");
    synchronize_all_called = true;
  };

  StreamExecutor* executor = GetExecutor(0);
  ASSERT_FALSE(synchronize_all_called);
  ASSERT_TRUE(executor->SynchronizeAllActivity());
  ASSERT_TRUE(synchronize_all_called);
}

TEST_F(StreamExecutorTest, HostCallbackOk) {
  se_.host_callback = [](const SP_Device* const device, SP_Stream stream,
                         SE_StatusCallbackFn const callback_fn,
                         void* const callback_arg) -> TF_Bool {
    TF_Status* status = TF_NewStatus();
    callback_fn(callback_arg, status);
    bool ok = TF_GetCode(status) == TF_OK;
    TF_DeleteStatus(status);
    return ok;
  };
  StreamExecutor* executor = GetExecutor(0);
  Stream stream(executor);
  stream.Init();
  std::function<port::Status()> callback = []() -> port::Status {
    return port::Status::OK();
  };
  stream.ThenDoHostCallbackWithStatus(callback);
  ASSERT_TRUE(stream.ok());
}

TEST_F(StreamExecutorTest, HostCallbackError) {
  se_.host_callback = [](const SP_Device* const device, SP_Stream stream,
                         SE_StatusCallbackFn const callback_fn,
                         void* const callback_arg) -> TF_Bool {
    TF_Status* status = TF_NewStatus();
    callback_fn(callback_arg, status);
    bool ok = TF_GetCode(status) == TF_OK;
    TF_DeleteStatus(status);
    return ok;
  };
  StreamExecutor* executor = GetExecutor(0);
  Stream stream(executor);
  stream.Init();
  std::function<port::Status()> callback = []() -> port::Status {
    return port::UnimplementedError("Unimplemented");
  };
  stream.ThenDoHostCallbackWithStatus(callback);
  ASSERT_FALSE(stream.ok());
}

TEST_F(StreamExecutorTest, DeviceDescription) {
  static const char* hardware_name = "TestName";
  static const char* vendor = "TestVendor";
  static const char* pci_bus_id = "TestPCIBusId";
  platform_fns_.create_device = [](const SP_Platform* platform,
                                   SE_CreateDeviceParams* params,
                                   TF_Status* status) {
   std::vector<std::string> mht_26_v;
   MHTracer_DTPStensorflowPScPSexperimentalPSstream_executorPSstream_executor_testDTcc mht_26(mht_26_v, 1005, "", "./tensorflow/c/experimental/stream_executor/stream_executor_test.cc", "lambda");

    params->device->hardware_name = hardware_name;
    params->device->device_vendor = vendor;
    params->device->pci_bus_id = pci_bus_id;
  };

  device_fns_.get_numa_node = [](const SP_Device* device) {
   std::vector<std::string> mht_27_v;
   MHTracer_DTPStensorflowPScPSexperimentalPSstream_executorPSstream_executor_testDTcc mht_27(mht_27_v, 1014, "", "./tensorflow/c/experimental/stream_executor/stream_executor_test.cc", "lambda");
 return 123; };
  device_fns_.get_memory_bandwidth = [](const SP_Device* device) -> int64_t {
    return 54;
  };
  device_fns_.get_gflops = [](const SP_Device* device) -> double { return 32; };

  StreamExecutor* executor = GetExecutor(0);
  const DeviceDescription& description = executor->GetDeviceDescription();
  ASSERT_EQ(description.name(), "TestName");
  ASSERT_EQ(description.device_vendor(), "TestVendor");
  ASSERT_EQ(description.pci_bus_id(), "TestPCIBusId");
  ASSERT_EQ(description.numa_node(), 123);
  ASSERT_EQ(description.memory_bandwidth(), 54);
}

TEST_F(StreamExecutorTest, DeviceDescriptionNumaNodeNotSet) {
  static const char* hardware_name = "TestName";
  static const char* vendor = "TestVendor";
  static const char* pci_bus_id = "TestPCIBusId";
  platform_fns_.create_device = [](const SP_Platform* platform,
                                   SE_CreateDeviceParams* params,
                                   TF_Status* status) {
   std::vector<std::string> mht_28_v;
   MHTracer_DTPStensorflowPScPSexperimentalPSstream_executorPSstream_executor_testDTcc mht_28(mht_28_v, 1038, "", "./tensorflow/c/experimental/stream_executor/stream_executor_test.cc", "lambda");

    params->device->hardware_name = hardware_name;
    params->device->device_vendor = vendor;
    params->device->pci_bus_id = pci_bus_id;
  };

  device_fns_.get_memory_bandwidth = [](const SP_Device* device) -> int64_t {
    return 54;
  };
  device_fns_.get_gflops = [](const SP_Device* device) -> double { return 32; };

  StreamExecutor* executor = GetExecutor(0);
  const DeviceDescription& description = executor->GetDeviceDescription();
  ASSERT_EQ(description.name(), "TestName");
  ASSERT_EQ(description.device_vendor(), "TestVendor");
  ASSERT_EQ(description.pci_bus_id(), "TestPCIBusId");
  ASSERT_EQ(description.numa_node(), -1);
  ASSERT_EQ(description.memory_bandwidth(), 54);
}

TEST_F(StreamExecutorTest, MemZero) {
  se_.create_stream = [](const SP_Device* const device, SP_Stream* stream,
                         TF_Status* const status) -> void {
    *stream = new SP_Stream_st(14);
  };
  se_.destroy_stream = [](const SP_Device* const device,
                          SP_Stream stream) -> void { delete stream; };

  se_.mem_zero = [](const SP_Device* device, SP_Stream stream,
                    SP_DeviceMemoryBase* location, uint64_t size,
                    TF_Status* status) {
   std::vector<std::string> mht_29_v;
   MHTracer_DTPStensorflowPScPSexperimentalPSstream_executorPSstream_executor_testDTcc mht_29(mht_29_v, 1071, "", "./tensorflow/c/experimental/stream_executor/stream_executor_test.cc", "lambda");

    TF_SetStatus(status, TF_OK, "");
    EXPECT_EQ(stream->stream_id, 14);
    std::memset(location->opaque, 0, size);
  };

  StreamExecutor* executor = GetExecutor(0);
  Stream stream(executor);
  stream.Init();
  size_t size = sizeof(int);
  int data = 2;
  DeviceMemoryBase device_data(&data, size);
  Stream& stream_ref = stream.ThenMemZero(&device_data, size);
  ASSERT_EQ(data, 0);
  ASSERT_EQ(stream_ref.implementation(), stream.implementation());
}

TEST_F(StreamExecutorTest, Memset32) {
  se_.create_stream = [](const SP_Device* const device, SP_Stream* stream,
                         TF_Status* const status) -> void {
    *stream = new SP_Stream_st(14);
  };
  se_.destroy_stream = [](const SP_Device* const device,
                          SP_Stream stream) -> void { delete stream; };

  se_.memset32 = [](const SP_Device* device, SP_Stream stream,
                    SP_DeviceMemoryBase* location, uint32_t pattern,
                    uint64_t size, TF_Status* status) {
   std::vector<std::string> mht_30_v;
   MHTracer_DTPStensorflowPScPSexperimentalPSstream_executorPSstream_executor_testDTcc mht_30(mht_30_v, 1101, "", "./tensorflow/c/experimental/stream_executor/stream_executor_test.cc", "lambda");

    TF_SetStatus(status, TF_OK, "");
    EXPECT_EQ(stream->stream_id, 14);
    EXPECT_EQ(size % 4, 0);
    auto ptr = static_cast<uint32_t*>(location->opaque);
    for (int i = 0; i < size / 4; i++) {
      *(ptr + i) = pattern;
    }
  };

  StreamExecutor* executor = GetExecutor(0);
  Stream stream(executor);
  stream.Init();
  size_t size = sizeof(int);
  int data = 2;
  DeviceMemoryBase device_data(&data, size);
  Stream& stream_ref = stream.ThenMemset32(&device_data, 18, size);
  ASSERT_EQ(data, 18);
  ASSERT_EQ(stream_ref.implementation(), stream.implementation());
}

}  // namespace
}  // namespace stream_executor
