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
class MHTracer_DTPStensorflowPScPSexperimentalPSstream_executorPSstream_executor_test_utilDTcc {
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
   MHTracer_DTPStensorflowPScPSexperimentalPSstream_executorPSstream_executor_test_utilDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScPSexperimentalPSstream_executorPSstream_executor_test_utilDTcc() {
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
#include "tensorflow/c/experimental/stream_executor/stream_executor_test_util.h"

#include "tensorflow/c/experimental/stream_executor/stream_executor.h"

namespace stream_executor {
namespace test_util {

/*** Functions for creating SP_StreamExecutor ***/
void Allocate(const SP_Device* const device, uint64_t size,
              int64_t memory_space, SP_DeviceMemoryBase* const mem) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScPSexperimentalPSstream_executorPSstream_executor_test_utilDTcc mht_0(mht_0_v, 193, "", "./tensorflow/c/experimental/stream_executor/stream_executor_test_util.cc", "Allocate");
}
void Deallocate(const SP_Device* const device, SP_DeviceMemoryBase* const mem) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScPSexperimentalPSstream_executorPSstream_executor_test_utilDTcc mht_1(mht_1_v, 197, "", "./tensorflow/c/experimental/stream_executor/stream_executor_test_util.cc", "Deallocate");

}
void* HostMemoryAllocate(const SP_Device* const device, uint64_t size) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScPSexperimentalPSstream_executorPSstream_executor_test_utilDTcc mht_2(mht_2_v, 202, "", "./tensorflow/c/experimental/stream_executor/stream_executor_test_util.cc", "HostMemoryAllocate");

  return nullptr;
}
void HostMemoryDeallocate(const SP_Device* const device, void* mem) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScPSexperimentalPSstream_executorPSstream_executor_test_utilDTcc mht_3(mht_3_v, 208, "", "./tensorflow/c/experimental/stream_executor/stream_executor_test_util.cc", "HostMemoryDeallocate");
}
TF_Bool GetAllocatorStats(const SP_Device* const device,
                          SP_AllocatorStats* const stats) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScPSexperimentalPSstream_executorPSstream_executor_test_utilDTcc mht_4(mht_4_v, 213, "", "./tensorflow/c/experimental/stream_executor/stream_executor_test_util.cc", "GetAllocatorStats");

  return true;
}
TF_Bool DeviceMemoryUsage(const SP_Device* const device, int64_t* const free,
                          int64_t* const total) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScPSexperimentalPSstream_executorPSstream_executor_test_utilDTcc mht_5(mht_5_v, 220, "", "./tensorflow/c/experimental/stream_executor/stream_executor_test_util.cc", "DeviceMemoryUsage");

  return true;
}
void CreateStream(const SP_Device* const device, SP_Stream* stream,
                  TF_Status* const status) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScPSexperimentalPSstream_executorPSstream_executor_test_utilDTcc mht_6(mht_6_v, 227, "", "./tensorflow/c/experimental/stream_executor/stream_executor_test_util.cc", "CreateStream");

  *stream = nullptr;
}
void DestroyStream(const SP_Device* const device, SP_Stream stream) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScPSexperimentalPSstream_executorPSstream_executor_test_utilDTcc mht_7(mht_7_v, 233, "", "./tensorflow/c/experimental/stream_executor/stream_executor_test_util.cc", "DestroyStream");
}
void CreateStreamDependency(const SP_Device* const device, SP_Stream dependent,
                            SP_Stream other, TF_Status* const status) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScPSexperimentalPSstream_executorPSstream_executor_test_utilDTcc mht_8(mht_8_v, 238, "", "./tensorflow/c/experimental/stream_executor/stream_executor_test_util.cc", "CreateStreamDependency");
}
void GetStreamStatus(const SP_Device* const device, SP_Stream stream,
                     TF_Status* const status) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScPSexperimentalPSstream_executorPSstream_executor_test_utilDTcc mht_9(mht_9_v, 243, "", "./tensorflow/c/experimental/stream_executor/stream_executor_test_util.cc", "GetStreamStatus");
}
void CreateEvent(const SP_Device* const device, SP_Event* event,
                 TF_Status* const status) {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScPSexperimentalPSstream_executorPSstream_executor_test_utilDTcc mht_10(mht_10_v, 248, "", "./tensorflow/c/experimental/stream_executor/stream_executor_test_util.cc", "CreateEvent");

  *event = nullptr;
}
void DestroyEvent(const SP_Device* const device, SP_Event event) {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScPSexperimentalPSstream_executorPSstream_executor_test_utilDTcc mht_11(mht_11_v, 254, "", "./tensorflow/c/experimental/stream_executor/stream_executor_test_util.cc", "DestroyEvent");
}
SE_EventStatus GetEventStatus(const SP_Device* const device, SP_Event event) {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScPSexperimentalPSstream_executorPSstream_executor_test_utilDTcc mht_12(mht_12_v, 258, "", "./tensorflow/c/experimental/stream_executor/stream_executor_test_util.cc", "GetEventStatus");

  return SE_EVENT_UNKNOWN;
}
void RecordEvent(const SP_Device* const device, SP_Stream stream,
                 SP_Event event, TF_Status* const status) {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPScPSexperimentalPSstream_executorPSstream_executor_test_utilDTcc mht_13(mht_13_v, 265, "", "./tensorflow/c/experimental/stream_executor/stream_executor_test_util.cc", "RecordEvent");
}
void WaitForEvent(const SP_Device* const device, SP_Stream stream,
                  SP_Event event, TF_Status* const status) {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPScPSexperimentalPSstream_executorPSstream_executor_test_utilDTcc mht_14(mht_14_v, 270, "", "./tensorflow/c/experimental/stream_executor/stream_executor_test_util.cc", "WaitForEvent");
}
void CreateTimer(const SP_Device* const device, SP_Timer* timer,
                 TF_Status* const status) {
   std::vector<std::string> mht_15_v;
   MHTracer_DTPStensorflowPScPSexperimentalPSstream_executorPSstream_executor_test_utilDTcc mht_15(mht_15_v, 275, "", "./tensorflow/c/experimental/stream_executor/stream_executor_test_util.cc", "CreateTimer");
}
void DestroyTimer(const SP_Device* const device, SP_Timer timer) {
   std::vector<std::string> mht_16_v;
   MHTracer_DTPStensorflowPScPSexperimentalPSstream_executorPSstream_executor_test_utilDTcc mht_16(mht_16_v, 279, "", "./tensorflow/c/experimental/stream_executor/stream_executor_test_util.cc", "DestroyTimer");
}
void StartTimer(const SP_Device* const device, SP_Stream stream, SP_Timer timer,
                TF_Status* const status) {
   std::vector<std::string> mht_17_v;
   MHTracer_DTPStensorflowPScPSexperimentalPSstream_executorPSstream_executor_test_utilDTcc mht_17(mht_17_v, 284, "", "./tensorflow/c/experimental/stream_executor/stream_executor_test_util.cc", "StartTimer");
}
void StopTimer(const SP_Device* const device, SP_Stream stream, SP_Timer timer,
               TF_Status* const status) {
   std::vector<std::string> mht_18_v;
   MHTracer_DTPStensorflowPScPSexperimentalPSstream_executorPSstream_executor_test_utilDTcc mht_18(mht_18_v, 289, "", "./tensorflow/c/experimental/stream_executor/stream_executor_test_util.cc", "StopTimer");
}
void MemcpyDToH(const SP_Device* const device, SP_Stream stream, void* host_dst,
                const SP_DeviceMemoryBase* const device_src, uint64_t size,
                TF_Status* const status) {
   std::vector<std::string> mht_19_v;
   MHTracer_DTPStensorflowPScPSexperimentalPSstream_executorPSstream_executor_test_utilDTcc mht_19(mht_19_v, 295, "", "./tensorflow/c/experimental/stream_executor/stream_executor_test_util.cc", "MemcpyDToH");
}
void MemcpyHToD(const SP_Device* const device, SP_Stream stream,
                SP_DeviceMemoryBase* const device_dst, const void* host_src,
                uint64_t size, TF_Status* const status) {
   std::vector<std::string> mht_20_v;
   MHTracer_DTPStensorflowPScPSexperimentalPSstream_executorPSstream_executor_test_utilDTcc mht_20(mht_20_v, 301, "", "./tensorflow/c/experimental/stream_executor/stream_executor_test_util.cc", "MemcpyHToD");
}
void SyncMemcpyDToH(const SP_Device* const device, void* host_dst,
                    const SP_DeviceMemoryBase* const device_src, uint64_t size,
                    TF_Status* const status) {
   std::vector<std::string> mht_21_v;
   MHTracer_DTPStensorflowPScPSexperimentalPSstream_executorPSstream_executor_test_utilDTcc mht_21(mht_21_v, 307, "", "./tensorflow/c/experimental/stream_executor/stream_executor_test_util.cc", "SyncMemcpyDToH");
}
void SyncMemcpyHToD(const SP_Device* const device,
                    SP_DeviceMemoryBase* const device_dst, const void* host_src,
                    uint64_t size, TF_Status* const status) {
   std::vector<std::string> mht_22_v;
   MHTracer_DTPStensorflowPScPSexperimentalPSstream_executorPSstream_executor_test_utilDTcc mht_22(mht_22_v, 313, "", "./tensorflow/c/experimental/stream_executor/stream_executor_test_util.cc", "SyncMemcpyHToD");
}
void BlockHostForEvent(const SP_Device* const device, SP_Event event,
                       TF_Status* const status) {
   std::vector<std::string> mht_23_v;
   MHTracer_DTPStensorflowPScPSexperimentalPSstream_executorPSstream_executor_test_utilDTcc mht_23(mht_23_v, 318, "", "./tensorflow/c/experimental/stream_executor/stream_executor_test_util.cc", "BlockHostForEvent");
}
void SynchronizeAllActivity(const SP_Device* const device,
                            TF_Status* const status) {
   std::vector<std::string> mht_24_v;
   MHTracer_DTPStensorflowPScPSexperimentalPSstream_executorPSstream_executor_test_utilDTcc mht_24(mht_24_v, 323, "", "./tensorflow/c/experimental/stream_executor/stream_executor_test_util.cc", "SynchronizeAllActivity");
}
TF_Bool HostCallback(const SP_Device* const device, SP_Stream stream,
                     SE_StatusCallbackFn const callback_fn,
                     void* const callback_arg) {
   std::vector<std::string> mht_25_v;
   MHTracer_DTPStensorflowPScPSexperimentalPSstream_executorPSstream_executor_test_utilDTcc mht_25(mht_25_v, 329, "", "./tensorflow/c/experimental/stream_executor/stream_executor_test_util.cc", "HostCallback");

  return true;
}

void MemZero(const SP_Device* device, SP_Stream stream,
             SP_DeviceMemoryBase* location, uint64_t size, TF_Status* status) {
   std::vector<std::string> mht_26_v;
   MHTracer_DTPStensorflowPScPSexperimentalPSstream_executorPSstream_executor_test_utilDTcc mht_26(mht_26_v, 337, "", "./tensorflow/c/experimental/stream_executor/stream_executor_test_util.cc", "MemZero");
}

void Memset(const SP_Device* device, SP_Stream stream,
            SP_DeviceMemoryBase* location, uint8_t pattern, uint64_t size,
            TF_Status* status) {
   std::vector<std::string> mht_27_v;
   MHTracer_DTPStensorflowPScPSexperimentalPSstream_executorPSstream_executor_test_utilDTcc mht_27(mht_27_v, 344, "", "./tensorflow/c/experimental/stream_executor/stream_executor_test_util.cc", "Memset");
}

void Memset32(const SP_Device* device, SP_Stream stream,
              SP_DeviceMemoryBase* location, uint32_t pattern, uint64_t size,
              TF_Status* status) {
   std::vector<std::string> mht_28_v;
   MHTracer_DTPStensorflowPScPSexperimentalPSstream_executorPSstream_executor_test_utilDTcc mht_28(mht_28_v, 351, "", "./tensorflow/c/experimental/stream_executor/stream_executor_test_util.cc", "Memset32");
}

void PopulateDefaultStreamExecutor(SP_StreamExecutor* se) {
   std::vector<std::string> mht_29_v;
   MHTracer_DTPStensorflowPScPSexperimentalPSstream_executorPSstream_executor_test_utilDTcc mht_29(mht_29_v, 356, "", "./tensorflow/c/experimental/stream_executor/stream_executor_test_util.cc", "PopulateDefaultStreamExecutor");

  *se = {SP_STREAMEXECUTOR_STRUCT_SIZE};
  se->allocate = Allocate;
  se->deallocate = Deallocate;
  se->host_memory_allocate = HostMemoryAllocate;
  se->host_memory_deallocate = HostMemoryDeallocate;
  se->get_allocator_stats = GetAllocatorStats;
  se->device_memory_usage = DeviceMemoryUsage;
  se->create_stream = CreateStream;
  se->destroy_stream = DestroyStream;
  se->create_stream_dependency = CreateStreamDependency;
  se->get_stream_status = GetStreamStatus;
  se->create_event = CreateEvent;
  se->destroy_event = DestroyEvent;
  se->get_event_status = GetEventStatus;
  se->record_event = RecordEvent;
  se->wait_for_event = WaitForEvent;
  se->create_timer = CreateTimer;
  se->destroy_timer = DestroyTimer;
  se->start_timer = StartTimer;
  se->stop_timer = StopTimer;
  se->memcpy_dtoh = MemcpyDToH;
  se->memcpy_htod = MemcpyHToD;
  se->sync_memcpy_dtoh = SyncMemcpyDToH;
  se->sync_memcpy_htod = SyncMemcpyHToD;
  se->block_host_for_event = BlockHostForEvent;
  se->synchronize_all_activity = SynchronizeAllActivity;
  se->host_callback = HostCallback;
  se->mem_zero = MemZero;
  se->memset = Memset;
  se->memset32 = Memset32;
}

void PopulateDefaultDeviceFns(SP_DeviceFns* device_fns) {
   std::vector<std::string> mht_30_v;
   MHTracer_DTPStensorflowPScPSexperimentalPSstream_executorPSstream_executor_test_utilDTcc mht_30(mht_30_v, 392, "", "./tensorflow/c/experimental/stream_executor/stream_executor_test_util.cc", "PopulateDefaultDeviceFns");

  *device_fns = {SP_DEVICE_FNS_STRUCT_SIZE};
}

/*** Functions for creating SP_TimerFns ***/
uint64_t Nanoseconds(SP_Timer timer) {
   std::vector<std::string> mht_31_v;
   MHTracer_DTPStensorflowPScPSexperimentalPSstream_executorPSstream_executor_test_utilDTcc mht_31(mht_31_v, 400, "", "./tensorflow/c/experimental/stream_executor/stream_executor_test_util.cc", "Nanoseconds");
 return timer->timer_id; }

void PopulateDefaultTimerFns(SP_TimerFns* timer_fns) {
   std::vector<std::string> mht_32_v;
   MHTracer_DTPStensorflowPScPSexperimentalPSstream_executorPSstream_executor_test_utilDTcc mht_32(mht_32_v, 405, "", "./tensorflow/c/experimental/stream_executor/stream_executor_test_util.cc", "PopulateDefaultTimerFns");

  timer_fns->nanoseconds = Nanoseconds;
}

/*** Functions for creating SP_Platform ***/
void CreateTimerFns(const SP_Platform* platform, SP_TimerFns* timer_fns,
                    TF_Status* status) {
   std::vector<std::string> mht_33_v;
   MHTracer_DTPStensorflowPScPSexperimentalPSstream_executorPSstream_executor_test_utilDTcc mht_33(mht_33_v, 414, "", "./tensorflow/c/experimental/stream_executor/stream_executor_test_util.cc", "CreateTimerFns");

  TF_SetStatus(status, TF_OK, "");
  PopulateDefaultTimerFns(timer_fns);
}
void DestroyTimerFns(const SP_Platform* platform, SP_TimerFns* timer_fns) {
   std::vector<std::string> mht_34_v;
   MHTracer_DTPStensorflowPScPSexperimentalPSstream_executorPSstream_executor_test_utilDTcc mht_34(mht_34_v, 421, "", "./tensorflow/c/experimental/stream_executor/stream_executor_test_util.cc", "DestroyTimerFns");
}

void CreateStreamExecutor(const SP_Platform* platform,
                          SE_CreateStreamExecutorParams* params,
                          TF_Status* status) {
   std::vector<std::string> mht_35_v;
   MHTracer_DTPStensorflowPScPSexperimentalPSstream_executorPSstream_executor_test_utilDTcc mht_35(mht_35_v, 428, "", "./tensorflow/c/experimental/stream_executor/stream_executor_test_util.cc", "CreateStreamExecutor");

  TF_SetStatus(status, TF_OK, "");
  PopulateDefaultStreamExecutor(params->stream_executor);
}
void DestroyStreamExecutor(const SP_Platform* platform, SP_StreamExecutor* se) {
   std::vector<std::string> mht_36_v;
   MHTracer_DTPStensorflowPScPSexperimentalPSstream_executorPSstream_executor_test_utilDTcc mht_36(mht_36_v, 435, "", "./tensorflow/c/experimental/stream_executor/stream_executor_test_util.cc", "DestroyStreamExecutor");

}
void GetDeviceCount(const SP_Platform* platform, int* device_count,
                    TF_Status* status) {
   std::vector<std::string> mht_37_v;
   MHTracer_DTPStensorflowPScPSexperimentalPSstream_executorPSstream_executor_test_utilDTcc mht_37(mht_37_v, 441, "", "./tensorflow/c/experimental/stream_executor/stream_executor_test_util.cc", "GetDeviceCount");

  TF_SetStatus(status, TF_OK, "");
  *device_count = kDeviceCount;
}
void CreateDevice(const SP_Platform* platform, SE_CreateDeviceParams* params,
                  TF_Status* status) {
   std::vector<std::string> mht_38_v;
   MHTracer_DTPStensorflowPScPSexperimentalPSstream_executorPSstream_executor_test_utilDTcc mht_38(mht_38_v, 449, "", "./tensorflow/c/experimental/stream_executor/stream_executor_test_util.cc", "CreateDevice");

  TF_SetStatus(status, TF_OK, "");
  params->device->struct_size = {SP_DEVICE_STRUCT_SIZE};
}
void DestroyDevice(const SP_Platform* platform, SP_Device* device) {
   std::vector<std::string> mht_39_v;
   MHTracer_DTPStensorflowPScPSexperimentalPSstream_executorPSstream_executor_test_utilDTcc mht_39(mht_39_v, 456, "", "./tensorflow/c/experimental/stream_executor/stream_executor_test_util.cc", "DestroyDevice");
}

void CreateDeviceFns(const SP_Platform* platform,
                     SE_CreateDeviceFnsParams* params, TF_Status* status) {
   std::vector<std::string> mht_40_v;
   MHTracer_DTPStensorflowPScPSexperimentalPSstream_executorPSstream_executor_test_utilDTcc mht_40(mht_40_v, 462, "", "./tensorflow/c/experimental/stream_executor/stream_executor_test_util.cc", "CreateDeviceFns");

  TF_SetStatus(status, TF_OK, "");
  params->device_fns->struct_size = {SP_DEVICE_FNS_STRUCT_SIZE};
}
void DestroyDeviceFns(const SP_Platform* platform, SP_DeviceFns* device_fns) {
   std::vector<std::string> mht_41_v;
   MHTracer_DTPStensorflowPScPSexperimentalPSstream_executorPSstream_executor_test_utilDTcc mht_41(mht_41_v, 469, "", "./tensorflow/c/experimental/stream_executor/stream_executor_test_util.cc", "DestroyDeviceFns");
}

void PopulateDefaultPlatform(SP_Platform* platform,
                             SP_PlatformFns* platform_fns) {
   std::vector<std::string> mht_42_v;
   MHTracer_DTPStensorflowPScPSexperimentalPSstream_executorPSstream_executor_test_utilDTcc mht_42(mht_42_v, 475, "", "./tensorflow/c/experimental/stream_executor/stream_executor_test_util.cc", "PopulateDefaultPlatform");

  *platform = {SP_PLATFORM_STRUCT_SIZE};
  platform->name = kDeviceName;
  platform->type = kDeviceType;
  platform_fns->get_device_count = GetDeviceCount;
  platform_fns->create_device = CreateDevice;
  platform_fns->destroy_device = DestroyDevice;
  platform_fns->create_device_fns = CreateDeviceFns;
  platform_fns->destroy_device_fns = DestroyDeviceFns;
  platform_fns->create_stream_executor = CreateStreamExecutor;
  platform_fns->destroy_stream_executor = DestroyStreamExecutor;
  platform_fns->create_timer_fns = CreateTimerFns;
  platform_fns->destroy_timer_fns = DestroyTimerFns;
}

/*** Functions for creating SE_PlatformRegistrationParams ***/
void DestroyPlatform(SP_Platform* platform) {
   std::vector<std::string> mht_43_v;
   MHTracer_DTPStensorflowPScPSexperimentalPSstream_executorPSstream_executor_test_utilDTcc mht_43(mht_43_v, 494, "", "./tensorflow/c/experimental/stream_executor/stream_executor_test_util.cc", "DestroyPlatform");
}
void DestroyPlatformFns(SP_PlatformFns* platform_fns) {
   std::vector<std::string> mht_44_v;
   MHTracer_DTPStensorflowPScPSexperimentalPSstream_executorPSstream_executor_test_utilDTcc mht_44(mht_44_v, 498, "", "./tensorflow/c/experimental/stream_executor/stream_executor_test_util.cc", "DestroyPlatformFns");
}

void PopulateDefaultPlatformRegistrationParams(
    SE_PlatformRegistrationParams* const params) {
   std::vector<std::string> mht_45_v;
   MHTracer_DTPStensorflowPScPSexperimentalPSstream_executorPSstream_executor_test_utilDTcc mht_45(mht_45_v, 504, "", "./tensorflow/c/experimental/stream_executor/stream_executor_test_util.cc", "PopulateDefaultPlatformRegistrationParams");

  PopulateDefaultPlatform(params->platform, params->platform_fns);
  params->destroy_platform = DestroyPlatform;
  params->destroy_platform_fns = DestroyPlatformFns;
}

}  // namespace test_util
}  // namespace stream_executor
