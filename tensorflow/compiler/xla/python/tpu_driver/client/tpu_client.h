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

#ifndef TENSORFLOW_COMPILER_XLA_PYTHON_TPU_DRIVER_CLIENT_TPU_CLIENT_H_
#define TENSORFLOW_COMPILER_XLA_PYTHON_TPU_DRIVER_CLIENT_TPU_CLIENT_H_
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
class MHTracer_DTPStensorflowPScompilerPSxlaPSpythonPStpu_driverPSclientPStpu_clientDTh {
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
   MHTracer_DTPStensorflowPScompilerPSxlaPSpythonPStpu_driverPSclientPStpu_clientDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSxlaPSpythonPStpu_driverPSclientPStpu_clientDTh() {
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


#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "absl/synchronization/notification.h"
#include "absl/types/span.h"
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "tensorflow/compiler/xla/client/executable_build_options.h"
#include "tensorflow/compiler/xla/executable_run_options.h"
#include "tensorflow/compiler/xla/pjrt/pjrt_client.h"
#include "tensorflow/compiler/xla/python/tpu_driver/tpu_driver.h"
#include "tensorflow/compiler/xla/python/tpu_driver/tpu_driver.pb.h"
#include "tensorflow/compiler/xla/shape.h"
#include "tensorflow/compiler/xla/status.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/core/platform/casts.h"
#include "tensorflow/core/platform/threadpool.h"

namespace xla {

inline const char* TpuPlatform() {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpythonPStpu_driverPSclientPStpu_clientDTh mht_0(mht_0_v, 212, "", "./tensorflow/compiler/xla/python/tpu_driver/client/tpu_client.h", "TpuPlatform");

  static constexpr char kTpuPlatform[] = "tpu";
  return kTpuPlatform;
}

class PyTpuClient;

class TpuDevice : public PjRtDevice {
 public:
  TpuDevice(int id, int process_index, const std::array<int, 3>& coords,
            int core_on_chip);

  const std::array<int, 3>& coords() const { return coords_; }
  int core_on_chip() const {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpythonPStpu_driverPSclientPStpu_clientDTh mht_1(mht_1_v, 228, "", "./tensorflow/compiler/xla/python/tpu_driver/client/tpu_client.h", "core_on_chip");
 return core_on_chip_; }

  std::string DebugString() const override;

  static xla::StatusOr<std::vector<std::shared_ptr<xla::PjRtDevice>>>
  GetTpuDevices(const tpu_driver::SystemInfo& system_info);

  PjRtClient* client() const override {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpythonPStpu_driverPSclientPStpu_clientDTh mht_2(mht_2_v, 238, "", "./tensorflow/compiler/xla/python/tpu_driver/client/tpu_client.h", "client");
 return nullptr; }
  PyTpuClient* tpu_client() const {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpythonPStpu_driverPSclientPStpu_clientDTh mht_3(mht_3_v, 242, "", "./tensorflow/compiler/xla/python/tpu_driver/client/tpu_client.h", "tpu_client");
 return tpu_client_; }
  void set_tpu_client(PyTpuClient* tpu_client) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpythonPStpu_driverPSclientPStpu_clientDTh mht_4(mht_4_v, 246, "", "./tensorflow/compiler/xla/python/tpu_driver/client/tpu_client.h", "set_tpu_client");
 tpu_client_ = tpu_client; }

  bool IsAddressable() const override {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpythonPStpu_driverPSclientPStpu_clientDTh mht_5(mht_5_v, 251, "", "./tensorflow/compiler/xla/python/tpu_driver/client/tpu_client.h", "IsAddressable");
 return false; }

  int id() const override {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpythonPStpu_driverPSclientPStpu_clientDTh mht_6(mht_6_v, 256, "", "./tensorflow/compiler/xla/python/tpu_driver/client/tpu_client.h", "id");
 return id_; }

  int process_index() const override {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpythonPStpu_driverPSclientPStpu_clientDTh mht_7(mht_7_v, 261, "", "./tensorflow/compiler/xla/python/tpu_driver/client/tpu_client.h", "process_index");
 return process_index_; }

  int local_hardware_id() const override {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpythonPStpu_driverPSclientPStpu_clientDTh mht_8(mht_8_v, 266, "", "./tensorflow/compiler/xla/python/tpu_driver/client/tpu_client.h", "local_hardware_id");
 return -1; }

  absl::string_view device_kind() const override {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpythonPStpu_driverPSclientPStpu_clientDTh mht_9(mht_9_v, 271, "", "./tensorflow/compiler/xla/python/tpu_driver/client/tpu_client.h", "device_kind");
 return device_kind_; }

  Status TransferToInfeed(const LiteralSlice& literal) override {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpythonPStpu_driverPSclientPStpu_clientDTh mht_10(mht_10_v, 276, "", "./tensorflow/compiler/xla/python/tpu_driver/client/tpu_client.h", "TransferToInfeed");

    return Unimplemented("Infeed not yet implemented via this API");
  }

  Status TransferFromOutfeed(MutableBorrowingLiteral literal) override {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpythonPStpu_driverPSclientPStpu_clientDTh mht_11(mht_11_v, 283, "", "./tensorflow/compiler/xla/python/tpu_driver/client/tpu_client.h", "TransferFromOutfeed");

    return Unimplemented("Outfeed not yet implemented via this API");
  }

 private:
  const int id_;
  const int process_index_;
  const std::array<int, 3> coords_;
  const std::string device_kind_ = "Cloud TPU";
  // Index of the core of the same chip.
  int core_on_chip_;
  PyTpuClient* tpu_client_;
};

// Encapsulates the state of Python session with XLA.
class PyTpuClient : public std::enable_shared_from_this<PyTpuClient> {
 public:
  // Initializes a local XLA client for `platform_name`. Returns an error if no
  // such platform exists, or if the platform has no visible devices.
  static StatusOr<std::shared_ptr<PyTpuClient>> Get(const std::string& worker);

  explicit PyTpuClient(std::string platform_name,
                       std::unique_ptr<tpu_driver::TpuDriver> driver,
                       std::vector<std::shared_ptr<PjRtDevice>> devices,
                       int process_index);
  virtual ~PyTpuClient() = default;

  PyTpuClient(const PyTpuClient&) = delete;
  PyTpuClient(PyTpuClient&&) = delete;
  PyTpuClient& operator=(const PyTpuClient&) = delete;
  PyTpuClient& operator=(PyTpuClient&&) = delete;

  Status TransferToInfeed(const LiteralSlice& literal, int device_id);
  StatusOr<Literal> TransferFromOutfeed(const Shape& shape, int device_id);

  virtual StatusOr<DeviceAssignment> GetDefaultDeviceAssignment(
      int num_replicas, int num_partitions) const;

  int device_count() const {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpythonPStpu_driverPSclientPStpu_clientDTh mht_12(mht_12_v, 324, "", "./tensorflow/compiler/xla/python/tpu_driver/client/tpu_client.h", "device_count");
 return devices_.size(); }
  int local_device_count() const {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpythonPStpu_driverPSclientPStpu_clientDTh mht_13(mht_13_v, 328, "", "./tensorflow/compiler/xla/python/tpu_driver/client/tpu_client.h", "local_device_count");
 return local_devices_.size(); }
  const std::vector<std::shared_ptr<PjRtDevice>>& devices() {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpythonPStpu_driverPSclientPStpu_clientDTh mht_14(mht_14_v, 332, "", "./tensorflow/compiler/xla/python/tpu_driver/client/tpu_client.h", "devices");
 return devices_; }
  const std::vector<std::shared_ptr<PjRtDevice>>& local_devices() {
   std::vector<std::string> mht_15_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpythonPStpu_driverPSclientPStpu_clientDTh mht_15(mht_15_v, 336, "", "./tensorflow/compiler/xla/python/tpu_driver/client/tpu_client.h", "local_devices");

    return local_devices_;
  }
  const std::map<int, std::shared_ptr<PjRtDevice>>& id_to_device() const {
   std::vector<std::string> mht_16_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpythonPStpu_driverPSclientPStpu_clientDTh mht_16(mht_16_v, 342, "", "./tensorflow/compiler/xla/python/tpu_driver/client/tpu_client.h", "id_to_device");

    return id_to_device_;
  }
  int process_index() const {
   std::vector<std::string> mht_17_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpythonPStpu_driverPSclientPStpu_clientDTh mht_17(mht_17_v, 348, "", "./tensorflow/compiler/xla/python/tpu_driver/client/tpu_client.h", "process_index");
 return process_index_; }
  const absl::string_view platform_name() const {
   std::vector<std::string> mht_18_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpythonPStpu_driverPSclientPStpu_clientDTh mht_18(mht_18_v, 352, "", "./tensorflow/compiler/xla/python/tpu_driver/client/tpu_client.h", "platform_name");
 return platform_name_; }
  const absl::string_view platform_version() const {
   std::vector<std::string> mht_19_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpythonPStpu_driverPSclientPStpu_clientDTh mht_19(mht_19_v, 356, "", "./tensorflow/compiler/xla/python/tpu_driver/client/tpu_client.h", "platform_version");
 return platform_version_; }

  StatusOr<Shape> ChooseCompactLayoutForShape(Shape subshape) {
    return Unimplemented("ChooseCompactLayoutForShape not implemented.");
  }

  // Returns a bad status containing `caller_name` if `device_id` doesn't
  // correspond to a valid device at the POD-slice boundary.
  Status CheckDeviceId(int device_id, absl::string_view caller_name);

  tpu_driver::TpuDriver* driver() {
   std::vector<std::string> mht_20_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpythonPStpu_driverPSclientPStpu_clientDTh mht_20(mht_20_v, 369, "", "./tensorflow/compiler/xla/python/tpu_driver/client/tpu_client.h", "driver");
 return driver_.get(); }

  tensorflow::thread::ThreadPool* GetThreadPool() {
   std::vector<std::string> mht_21_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpythonPStpu_driverPSclientPStpu_clientDTh mht_21(mht_21_v, 374, "", "./tensorflow/compiler/xla/python/tpu_driver/client/tpu_client.h", "GetThreadPool");
 return pool_.get(); }

 protected:
  std::string platform_name_;
  std::string platform_version_;
  std::unique_ptr<tpu_driver::TpuDriver> driver_;

  // Includes all devices, including non-local devices on multi-host platforms.
  std::vector<std::shared_ptr<PjRtDevice>> devices_;
  // Maps Device::id() to the corresponding Device. Includes all devices.
  std::map<int, std::shared_ptr<PjRtDevice>> id_to_device_;
  // Local devices indexed by local device ordinal.
  std::vector<std::shared_ptr<PjRtDevice>> local_devices_;
  int process_index_;

  // A thread pool for scheduling core executions in parallel.
  std::unique_ptr<tensorflow::thread::ThreadPool> pool_;
};

// Manages a buffer shared amongst multiple users. Buffers are asynchronously
// deallocated after the last use.
struct TpuSharedBuffer final {
 public:
  TpuSharedBuffer(tpu_driver::TpuDriver* driver,
                  std::unique_ptr<tpu_driver::BufferHandle> handle,
                  std::vector<std::shared_ptr<tpu_driver::Event>> wait_for_use,
                  std::shared_ptr<PjRtDevice> src_device)
      : driver(driver),
        device(std::move(src_device)),
        handle(std::move(handle)),
        wait_for_use(std::move(wait_for_use)) {
   std::vector<std::string> mht_22_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpythonPStpu_driverPSclientPStpu_clientDTh mht_22(mht_22_v, 407, "", "./tensorflow/compiler/xla/python/tpu_driver/client/tpu_client.h", "TpuSharedBuffer");
}

  ~TpuSharedBuffer() {
   std::vector<std::string> mht_23_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpythonPStpu_driverPSclientPStpu_clientDTh mht_23(mht_23_v, 412, "", "./tensorflow/compiler/xla/python/tpu_driver/client/tpu_client.h", "~TpuSharedBuffer");

    std::vector<tpu_driver::Event*> events;
    for (const auto& e : wait_for_use) {
      events.push_back(e.get());
    }
    driver->Deallocate(std::move(handle), events);
  }

  tpu_driver::TpuDriver* const driver;
  const std::shared_ptr<PjRtDevice> device;

  std::unique_ptr<tpu_driver::BufferHandle> handle;
  std::vector<std::shared_ptr<tpu_driver::Event>> wait_for_use;
};

// Holds a reference from Python to one or more device buffers.
// A PyTpuBuffer can be either valid or invalid. An invalid buffer is one that
// has never been initialized, or a buffer that has been deleted (e.g., by
// calling Delete). We allow PyTpuBuffer objects to outlive the underlying
// device buffers so we can decouple buffer lifetimes from the corresponding
// Python references if needed.
// Thread-safe.
class PyTpuBuffer {
 public:
  // `tuple_shape` can be at most a one-level tuple combining non-tuple leaves.
  static StatusOr<std::unique_ptr<PyTpuBuffer>> FromLiterals(
      std::vector<BorrowingLiteral> leaves_literals, const Shape& tuple_shape,
      std::shared_ptr<void> leaves_reference,
      std::shared_ptr<PyTpuClient> client, std::shared_ptr<PjRtDevice> device);

  // Supports nested tuple creation.
  static StatusOr<std::unique_ptr<PyTpuBuffer>> MakeTuple(
      absl::Span<PyTpuBuffer* const> buffers,
      std::shared_ptr<PyTpuClient> client, std::shared_ptr<PjRtDevice> device);

  PyTpuBuffer() = delete;
  PyTpuBuffer(Shape on_host_shape,
              std::shared_ptr<TpuSharedBuffer> device_buffer,
              std::vector<std::shared_ptr<TpuSharedBuffer>> child_buffers,
              std::shared_ptr<PyTpuClient> client);

  PyTpuBuffer(const PyTpuBuffer&) = delete;
  PyTpuBuffer(PyTpuBuffer&&) = delete;
  PyTpuBuffer& operator=(const PyTpuBuffer&) = delete;
  PyTpuBuffer& operator=(PyTpuBuffer&&) = delete;

  const Shape& on_host_shape() const {
   std::vector<std::string> mht_24_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpythonPStpu_driverPSclientPStpu_clientDTh mht_24(mht_24_v, 461, "", "./tensorflow/compiler/xla/python/tpu_driver/client/tpu_client.h", "on_host_shape");
 return on_host_shape_; }
  std::shared_ptr<PjRtDevice> device() const { return device_; }
  const absl::string_view platform_name() const {
   std::vector<std::string> mht_25_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpythonPStpu_driverPSclientPStpu_clientDTh mht_25(mht_25_v, 466, "", "./tensorflow/compiler/xla/python/tpu_driver/client/tpu_client.h", "platform_name");

    return client_->platform_name();
  }
  std::shared_ptr<PyTpuClient> client() const { return client_; }

  // Returns the buffer's value as a tuple DAG of Python arrays. If the value
  // has previously been prefetched to the host, then returns the prefetched
  // version, otherwise copies the buffer to the host. Blocks until the
  // value is ready.
  StatusOr<std::shared_ptr<Literal>> ToLiteral();

  // Initiates a copy of the buffer to the host. Does not block waiting for
  // the transfer to complete. The value can be retrieved by a later call to
  // ToLiteral().
  Status CopyToHostAsync();

  // Returns the associated device buffer. Returns a nullptr if the buffer is
  // invalid.
  std::shared_ptr<TpuSharedBuffer> DeviceBuffer() const;

  // Deletes the device memory associated with this buffer, leaving it in an
  // invalid state.
  void Delete();

  // Destructures a tuple-valued PyTpuBuffer into its constituent elements.
  StatusOr<std::vector<std::unique_ptr<PyTpuBuffer>>> DestructureTuple();

  // Copies the buffer to target device `dst_device` and returns a PyTpuBuffer
  // object holding the context to the target device buffer.
  StatusOr<std::unique_ptr<PyTpuBuffer>> CopyToDevice(
      std::shared_ptr<PjRtDevice> dst_device);

  // Blocks the host until the buffer's value has been computed and is ready for
  // immediate use on the device. Useful in particular for timing benchmarks.
  Status BlockHostUntilReady();

  // Allocates uninitialized buffers on device `device_id`. If `shape` is a
  // tuple, the returned buffer corresponds to the root tuple buffer.
  static StatusOr<std::unique_ptr<PyTpuBuffer>> AllocateBuffer(
      const Shape& shape, std::shared_ptr<PyTpuClient> client,
      std::shared_ptr<PjRtDevice> device);

 private:
  // Initializes a just allocated device buffer. The returned event will be
  // placed into the buffer's `wait_for_use` list.
  using BufferInitializer = std::function<std::shared_ptr<tpu_driver::Event>(
      tpu_driver::BufferHandle*)>;
  // Allocates and optionally initializes a non-tuple buffer on the device.
  static StatusOr<std::unique_ptr<PyTpuBuffer>> CreateBuffer(
      const Shape& non_tuple_shape,
      absl::optional<BufferInitializer> initializer,
      std::shared_ptr<PyTpuClient> client, std::shared_ptr<PjRtDevice> device);

  const std::shared_ptr<PyTpuClient> client_;
  const Shape on_host_shape_;
  const std::shared_ptr<PjRtDevice> device_;

  // If this is a tuple, `device_buffer_` stores the tuple buffer and
  // `child_buffers_` stores the child buffers; else, `device_buffer_` stores
  // the data content and `child_buffers_` is empty.
  mutable absl::Mutex mu_;
  std::shared_ptr<TpuSharedBuffer> device_buffer_ ABSL_GUARDED_BY(mu_);
  std::vector<std::shared_ptr<TpuSharedBuffer>> child_buffers_
      ABSL_GUARDED_BY(mu_);
  // The cached value of the buffer on the host, produced either from a call to
  // CopyToHost or from a call to ToLiteral. Once a value has been fetched to
  // the host, it persists Delete() is called or the PyTpuBuffer is destroyed.
  struct HostValue {
    absl::Mutex mutex;
    absl::Notification ready;
    int pending_ops;
    // status and value are valid for reading only after `ready` has been
    // notified.
    Status status;
    std::shared_ptr<Literal> value;
  };
  std::shared_ptr<HostValue> host_value_ ABSL_GUARDED_BY(mu_);
};

// Represents a compiled computation that can be executed given handles to
// device-allocated literals. Wraps an XLA LocalExecutable.
class PyTpuExecutable {
 public:
  static StatusOr<std::unique_ptr<PyTpuExecutable>> Compile(
      const XlaComputation& computation,
      absl::optional<std::vector<Shape>> argument_layouts,
      const ExecutableBuildOptions* build_options,
      std::shared_ptr<PyTpuClient> client, bool tuple_arguments);

  static StatusOr<std::unique_ptr<PyTpuExecutable>> CompileMlir(
      mlir::ModuleOp module,
      absl::optional<std::vector<Shape>> argument_layouts,
      const ExecutableBuildOptions* build_options,
      std::shared_ptr<PyTpuClient> client, bool tuple_arguments);

  PyTpuExecutable(
      std::unique_ptr<tpu_driver::CompiledProgramHandle> compiled_program,
      DeviceAssignment device_assignment, std::shared_ptr<PyTpuClient> client,
      xla::Shape result_shape, bool tuple_arguments);
  virtual ~PyTpuExecutable() {
   std::vector<std::string> mht_26_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpythonPStpu_driverPSclientPStpu_clientDTh mht_26(mht_26_v, 568, "", "./tensorflow/compiler/xla/python/tpu_driver/client/tpu_client.h", "~PyTpuExecutable");

    for (auto it = executables_.begin(); it != executables_.end(); ++it) {
      client_->driver()->UnloadProgram(std::move(it->second), {});
    }
  }

  PyTpuExecutable(const PyTpuExecutable&) = delete;
  PyTpuExecutable(PyTpuExecutable&&) = delete;
  PyTpuExecutable& operator=(const PyTpuExecutable&) = delete;
  PyTpuExecutable& operator=(PyTpuExecutable&&) = delete;

  std::shared_ptr<PyTpuClient> client() const { return client_; }

  int num_replicas() const {
   std::vector<std::string> mht_27_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpythonPStpu_driverPSclientPStpu_clientDTh mht_27(mht_27_v, 584, "", "./tensorflow/compiler/xla/python/tpu_driver/client/tpu_client.h", "num_replicas");
 return device_assignment_.replica_count(); }
  int num_partitions() const {
   std::vector<std::string> mht_28_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpythonPStpu_driverPSclientPStpu_clientDTh mht_28(mht_28_v, 588, "", "./tensorflow/compiler/xla/python/tpu_driver/client/tpu_client.h", "num_partitions");
 return device_assignment_.computation_count(); }

  int64_t SizeOfGeneratedCodeInBytes() const {
   std::vector<std::string> mht_29_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpythonPStpu_driverPSclientPStpu_clientDTh mht_29(mht_29_v, 593, "", "./tensorflow/compiler/xla/python/tpu_driver/client/tpu_client.h", "SizeOfGeneratedCodeInBytes");

    CHECK_GE(executables_.size(), 1);
    return executables_.begin()->second->size_in_bytes();
  }

  const DeviceAssignment& device_assignment() const {
   std::vector<std::string> mht_30_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpythonPStpu_driverPSclientPStpu_clientDTh mht_30(mht_30_v, 601, "", "./tensorflow/compiler/xla/python/tpu_driver/client/tpu_client.h", "device_assignment");

    return device_assignment_;
  }

  const std::vector<std::pair<int, int>>& local_logical_device_ids() const {
    return local_logical_device_ids_;
  }

  const std::vector<std::shared_ptr<PjRtDevice>>& local_devices() const {
   std::vector<std::string> mht_31_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpythonPStpu_driverPSclientPStpu_clientDTh mht_31(mht_31_v, 612, "", "./tensorflow/compiler/xla/python/tpu_driver/client/tpu_client.h", "local_devices");

    return local_devices_;
  }

  // TODO(power): Both Execute and ExecutePerOnLocalDevices block and wait
  // inside for computation to finish. Coordinate with JAX code change to see if
  // we can make both Execute and ExecutePerReplica non-blocking.
  StatusOr<std::vector<std::unique_ptr<PyTpuBuffer>>> Execute(
      absl::Span<PyTpuBuffer* const> argument_handles);

  // Execute on local devices. Takes a sequence of argument lists (one argument
  // list per local device) and returns a tuple of results (one result per local
  // device). The number of argument lists must be equal to the local device
  // count.
  StatusOr<std::vector<std::vector<std::unique_ptr<PyTpuBuffer>>>>
  ExecuteOnLocalDevices(
      absl::Span<const std::vector<PyTpuBuffer*>> argument_handles);

  StatusOr<std::vector<std::vector<std::unique_ptr<PyTpuBuffer>>>>
  ExecuteShardedOnLocalDevices(
      absl::Span<const std::vector<PyTpuBuffer*>> args);

  void Delete() {
   std::vector<std::string> mht_32_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpythonPStpu_driverPSclientPStpu_clientDTh mht_32(mht_32_v, 637, "", "./tensorflow/compiler/xla/python/tpu_driver/client/tpu_client.h", "Delete");
 executables_.clear(); }

 private:
  struct ExecuteResult {
    std::unique_ptr<PyTpuBuffer> buffer;
    std::shared_ptr<tpu_driver::Event> on_execute_finished;
  };

  ExecuteResult ExecuteHelper(
      absl::Span<const std::vector<PyTpuBuffer*>> all_core_arguments,
      absl::Span<PyTpuBuffer* const> this_core_arguments, int replica,
      int partition, const RunId& run_id);

  std::shared_ptr<PyTpuClient> const client_;
  std::map<int, std::unique_ptr<tpu_driver::LoadedProgramHandle>> executables_;
  const DeviceAssignment device_assignment_;
  const bool tuple_arguments_;

  // The replica and partition indices of device_assignment_ to be run by this
  // client. On single-host platforms without partitioning, this is all replicas
  // (i.e. local_logical_device_ids_[i] = (i, 0)), but this may not be the case
  // on multi-host platforms.
  // If there are 4 replicas and 2 partitions on a single host platform, size of
  // local_logical_device_ids_ is 4*2 = 8.
  std::vector<std::pair<int, int>> local_logical_device_ids_;

  // local_devices_[i] is the Device to which local_logical_device_ids_[i] is
  // assigned.
  // shared_ptrs instead of unique_ptrs to play well with the Python bindings
  // (see xla.cc).
  std::vector<std::shared_ptr<PjRtDevice>> local_devices_;

  xla::Shape result_shape_;
};

}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_PYTHON_TPU_DRIVER_CLIENT_TPU_CLIENT_H_
