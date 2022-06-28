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

// Declares the XlaInterpreterExecutor class, which is a CPU-only implementation
// of the StreamExecutor interface. For now, this is used for testing and to
// examine the performance of host-based StreamExecutor code.
#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_INTERPRETER_EXECUTOR_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_INTERPRETER_EXECUTOR_H_
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
class MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSinterpreterPSexecutorDTh {
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
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSinterpreterPSexecutorDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSinterpreterPSexecutorDTh() {
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


#include <functional>
#include <memory>

#include "absl/types/span.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/stream_executor/blas.h"
#include "tensorflow/stream_executor/device_description.h"
#include "tensorflow/stream_executor/device_memory.h"
#include "tensorflow/stream_executor/device_options.h"
#include "tensorflow/stream_executor/event.h"
#include "tensorflow/stream_executor/host/host_stream.h"
#include "tensorflow/stream_executor/host/host_timer.h"
#include "tensorflow/stream_executor/kernel.h"
#include "tensorflow/stream_executor/kernel_spec.h"
#include "tensorflow/stream_executor/launch_dim.h"
#include "tensorflow/stream_executor/plugin.h"
#include "tensorflow/stream_executor/rng.h"
#include "tensorflow/stream_executor/stream.h"
#include "tensorflow/stream_executor/stream_executor.h"
#include "tensorflow/stream_executor/stream_executor_internal.h"
#include "tensorflow/stream_executor/timer.h"

namespace stream_executor {
namespace interpreter {

using Args = absl::Span<const DeviceMemoryBase>;

class XlaInterpreterExecutor : public internal::StreamExecutorInterface {
 public:
  explicit XlaInterpreterExecutor(const PluginConfig &plugin_config);
  ~XlaInterpreterExecutor() override;

  port::Status Init(int device_ordinal, DeviceOptions device_options) override {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSinterpreterPSexecutorDTh mht_0(mht_0_v, 224, "", "./tensorflow/compiler/xla/service/interpreter/executor.h", "Init");

    return port::Status::OK();
  }

  port::Status GetKernel(const MultiKernelLoaderSpec &spec,
                         KernelBase *kernel) override {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSinterpreterPSexecutorDTh mht_1(mht_1_v, 232, "", "./tensorflow/compiler/xla/service/interpreter/executor.h", "GetKernel");

    return port::UnimplementedError("Not Implemented");
  }
  port::Status Launch(Stream *stream, const ThreadDim &thread_dims,
                      const BlockDim &block_dims, const KernelBase &kernel,
                      const KernelArgsArrayBase &args) override {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSinterpreterPSexecutorDTh mht_2(mht_2_v, 240, "", "./tensorflow/compiler/xla/service/interpreter/executor.h", "Launch");

    return port::UnimplementedError("Not Implemented");
  }

  DeviceMemoryBase Allocate(uint64_t size, int64_t memory_space) override;
  void *GetSubBuffer(DeviceMemoryBase *parent, uint64_t offset_bytes,
                     uint64_t size_bytes) override;
  void Deallocate(DeviceMemoryBase *mem) override;

  void *HostMemoryAllocate(uint64_t size) override {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSinterpreterPSexecutorDTh mht_3(mht_3_v, 252, "", "./tensorflow/compiler/xla/service/interpreter/executor.h", "HostMemoryAllocate");
 return new char[size]; }
  void HostMemoryDeallocate(void *mem) override {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSinterpreterPSexecutorDTh mht_4(mht_4_v, 256, "", "./tensorflow/compiler/xla/service/interpreter/executor.h", "HostMemoryDeallocate");

    delete[] static_cast<char *>(mem);
  }
  bool HostMemoryRegister(void *mem, uint64_t size) override {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSinterpreterPSexecutorDTh mht_5(mht_5_v, 262, "", "./tensorflow/compiler/xla/service/interpreter/executor.h", "HostMemoryRegister");
 return true; }
  bool HostMemoryUnregister(void *mem) override {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSinterpreterPSexecutorDTh mht_6(mht_6_v, 266, "", "./tensorflow/compiler/xla/service/interpreter/executor.h", "HostMemoryUnregister");
 return true; }

  bool Memcpy(Stream *stream, void *host_dst, const DeviceMemoryBase &dev_src,
              uint64_t size) override;
  bool Memcpy(Stream *stream, DeviceMemoryBase *dev_dst, const void *host_src,
              uint64_t size) override;
  bool MemcpyDeviceToDevice(Stream *stream, DeviceMemoryBase *pop_dst,
                            const DeviceMemoryBase &host_src,
                            uint64_t size) override {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSinterpreterPSexecutorDTh mht_7(mht_7_v, 277, "", "./tensorflow/compiler/xla/service/interpreter/executor.h", "MemcpyDeviceToDevice");

    return false;
  }

  port::Status MemZero(Stream *stream, DeviceMemoryBase *location,
                       uint64_t size) override {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSinterpreterPSexecutorDTh mht_8(mht_8_v, 285, "", "./tensorflow/compiler/xla/service/interpreter/executor.h", "MemZero");

    return port::InternalError("Interpreter can not memzero");
  }
  port::Status Memset(Stream *stream, DeviceMemoryBase *location,
                      uint8_t pattern, uint64_t size) override {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSinterpreterPSexecutorDTh mht_9(mht_9_v, 292, "", "./tensorflow/compiler/xla/service/interpreter/executor.h", "Memset");

    return port::InternalError("Interpreter can not memset");
  }
  port::Status Memset32(Stream *stream, DeviceMemoryBase *location,
                        uint32_t pattern, uint64_t size) override {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSinterpreterPSexecutorDTh mht_10(mht_10_v, 299, "", "./tensorflow/compiler/xla/service/interpreter/executor.h", "Memset32");

    return port::InternalError("Interpreter can not memset");
  }

  // No "synchronize all activity" implemented for this platform at the moment.
  bool SynchronizeAllActivity() override {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSinterpreterPSexecutorDTh mht_11(mht_11_v, 307, "", "./tensorflow/compiler/xla/service/interpreter/executor.h", "SynchronizeAllActivity");
 return true; }
  port::Status SynchronousMemZero(DeviceMemoryBase *location,
                                  uint64_t size) override {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSinterpreterPSexecutorDTh mht_12(mht_12_v, 312, "", "./tensorflow/compiler/xla/service/interpreter/executor.h", "SynchronousMemZero");

    return port::InternalError("Interpreter can not memzero");
  }

  port::Status SynchronousMemSet(DeviceMemoryBase *location, int value,
                                 uint64_t size) override {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSinterpreterPSexecutorDTh mht_13(mht_13_v, 320, "", "./tensorflow/compiler/xla/service/interpreter/executor.h", "SynchronousMemSet");

    return port::InternalError("Interpreter can not memset");
  }

  port::Status SynchronousMemcpy(DeviceMemoryBase *dev_dst,
                                 const void *host_src, uint64_t size) override;
  port::Status SynchronousMemcpy(void *host_dst,
                                 const DeviceMemoryBase &dev_src,
                                 uint64_t size) override;
  port::Status SynchronousMemcpyDeviceToDevice(DeviceMemoryBase *pop_dst,
                                               const DeviceMemoryBase &pop_src,
                                               uint64_t size) override {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSinterpreterPSexecutorDTh mht_14(mht_14_v, 334, "", "./tensorflow/compiler/xla/service/interpreter/executor.h", "SynchronousMemcpyDeviceToDevice");

    return port::Status{port::error::UNIMPLEMENTED, ""};
  }

  bool HostCallback(Stream *stream,
                    std::function<port::Status()> callback) override;

  port::Status AllocateEvent(Event *event) override {
   std::vector<std::string> mht_15_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSinterpreterPSexecutorDTh mht_15(mht_15_v, 344, "", "./tensorflow/compiler/xla/service/interpreter/executor.h", "AllocateEvent");

    return port::Status::OK();
  }

  port::Status DeallocateEvent(Event *event) override {
   std::vector<std::string> mht_16_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSinterpreterPSexecutorDTh mht_16(mht_16_v, 351, "", "./tensorflow/compiler/xla/service/interpreter/executor.h", "DeallocateEvent");

    return port::Status::OK();
  }

  port::Status RecordEvent(Stream *stream, Event *event) override {
   std::vector<std::string> mht_17_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSinterpreterPSexecutorDTh mht_17(mht_17_v, 358, "", "./tensorflow/compiler/xla/service/interpreter/executor.h", "RecordEvent");

    return port::Status{port::error::UNIMPLEMENTED, "RecordEvent"};
  }

  port::Status WaitForEvent(Stream *stream, Event *event) override {
   std::vector<std::string> mht_18_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSinterpreterPSexecutorDTh mht_18(mht_18_v, 365, "", "./tensorflow/compiler/xla/service/interpreter/executor.h", "WaitForEvent");

    return port::Status{port::error::UNIMPLEMENTED, "WaitForEvent"};
  }

  Event::Status PollForEventStatus(Event *event) override {
   std::vector<std::string> mht_19_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSinterpreterPSexecutorDTh mht_19(mht_19_v, 372, "", "./tensorflow/compiler/xla/service/interpreter/executor.h", "PollForEventStatus");

    return Event::Status::kError;
  }

  bool AllocateStream(Stream *stream) override {
   std::vector<std::string> mht_20_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSinterpreterPSexecutorDTh mht_20(mht_20_v, 379, "", "./tensorflow/compiler/xla/service/interpreter/executor.h", "AllocateStream");
 return true; }
  void DeallocateStream(Stream *stream) override {
   std::vector<std::string> mht_21_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSinterpreterPSexecutorDTh mht_21(mht_21_v, 383, "", "./tensorflow/compiler/xla/service/interpreter/executor.h", "DeallocateStream");
}
  bool CreateStreamDependency(Stream *dependent, Stream *other) override;

  bool AllocateTimer(Timer *timer) override {
   std::vector<std::string> mht_22_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSinterpreterPSexecutorDTh mht_22(mht_22_v, 389, "", "./tensorflow/compiler/xla/service/interpreter/executor.h", "AllocateTimer");
 return true; }
  void DeallocateTimer(Timer *timer) override {
   std::vector<std::string> mht_23_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSinterpreterPSexecutorDTh mht_23(mht_23_v, 393, "", "./tensorflow/compiler/xla/service/interpreter/executor.h", "DeallocateTimer");
}
  bool StartTimer(Stream *stream, Timer *timer) override;
  bool StopTimer(Stream *stream, Timer *timer) override;

  port::Status BlockHostUntilDone(Stream *stream) override;

  int PlatformDeviceCount() override {
   std::vector<std::string> mht_24_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSinterpreterPSexecutorDTh mht_24(mht_24_v, 402, "", "./tensorflow/compiler/xla/service/interpreter/executor.h", "PlatformDeviceCount");
 return 1; }

  bool DeviceMemoryUsage(int64_t *free, int64_t *total) const override {
   std::vector<std::string> mht_25_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSinterpreterPSexecutorDTh mht_25(mht_25_v, 407, "", "./tensorflow/compiler/xla/service/interpreter/executor.h", "DeviceMemoryUsage");

    return false;
  }

  port::StatusOr<std::unique_ptr<DeviceDescription>> CreateDeviceDescription()
      const override {
    return CreateDeviceDescription(0);
  }

  static port::StatusOr<std::unique_ptr<DeviceDescription>>
  CreateDeviceDescription(int device_ordinal);

  port::Status EnablePeerAccessTo(StreamExecutorInterface *other) override {
   std::vector<std::string> mht_26_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSinterpreterPSexecutorDTh mht_26(mht_26_v, 422, "", "./tensorflow/compiler/xla/service/interpreter/executor.h", "EnablePeerAccessTo");

    return port::Status::OK();
  }

  bool CanEnablePeerAccessTo(StreamExecutorInterface *other) override {
   std::vector<std::string> mht_27_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSinterpreterPSexecutorDTh mht_27(mht_27_v, 429, "", "./tensorflow/compiler/xla/service/interpreter/executor.h", "CanEnablePeerAccessTo");

    return true;
  }

  std::unique_ptr<internal::EventInterface> CreateEventImplementation()
      override {
    return nullptr;
  }

  std::unique_ptr<internal::KernelInterface> CreateKernelImplementation()
      override {
    return nullptr;
  }

  std::unique_ptr<internal::StreamInterface> GetStreamImplementation()
      override {
    return std::unique_ptr<internal::StreamInterface>(
        new host::HostStream(/*thread_stack_size=*/0));
  }

  std::unique_ptr<internal::TimerInterface> GetTimerImplementation() override {
    return std::unique_ptr<internal::TimerInterface>(new host::HostTimer());
  }

 private:
  DeviceMemoryBase AllocateSingleOutput(const xla::Shape &shape);

  port::StatusOr<DeviceMemoryBase> AllocateOutputBuffer(
      const xla::Shape &shape);

  const PluginConfig plugin_config_;
};

}  // namespace interpreter
}  // namespace stream_executor

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_INTERPRETER_EXECUTOR_H_
