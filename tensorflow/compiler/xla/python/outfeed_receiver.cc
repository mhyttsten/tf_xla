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
class MHTracer_DTPStensorflowPScompilerPSxlaPSpythonPSoutfeed_receiverDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSxlaPSpythonPSoutfeed_receiverDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSxlaPSpythonPSoutfeed_receiverDTcc() {
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

#include "tensorflow/compiler/xla/python/outfeed_receiver.h"

#include <sys/types.h>

#include <memory>
#include <queue>
#include <sstream>

#include "absl/container/flat_hash_map.h"
#include "absl/memory/memory.h"
#include "absl/strings/str_format.h"
#include "tensorflow/compiler/xla/client/sharding_builder.h"
#include "tensorflow/compiler/xla/client/xla_builder.h"
#include "tensorflow/compiler/xla/client/xla_computation.h"
#include "tensorflow/compiler/xla/pjrt/pjrt_client.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/core/profiler/lib/traceme.h"

// Implementation notes:
//
// Startup:
// -------
//
// The startup is initiated by a call from Python to StartOutfeedReceiver. For
// each local device there is one thread for listening for outfeeds from the
// device, one queue of received outfeeds, and one thread for invoking the
// Python callbacks.
//
// Framing protocol
// ----------------
//
// The outfeed mechanism has a single channel and the receiver must know
// exactly the shape and number of outfeed operations issued by the compiled
// code. This makes it hard to use outfeed in conditionals and loops and
// especially when outfeeding different-shaped data.
//
// To address this, when we compile the code we capture the shape of the
// data being outfed, and we generate a consumer ID (uint32_t) that is unique
// across the lifetime of the program to: the Python callable to callback to,
// the shape of the arguments, the keyword arguments to pass to the callable.
// Each outfeed payload is preceeded by a header (of shape u32[2]) with a
// special first value and the consumer ID. We maintain a registry of shapes
// by consumer ID. When receiving we lookup the shape by consumer ID, and then
// we read the payload.
//
// Back pressure:
// --------------
//
// We maintain a sum of the bytes from all the data waiting in the callback
// queues. The listening threads will wait for the sum to drop below a
// configurable threshold, default 256Mb. While the listening thread is waiting,
// on CPU and GPU the next outfeed operation from the device will block. On
// TPU there is a buffer, but eventually the TPU will also block.
//
// Shutdown:
// ---------
//
// The shutdown is initiated automatically when the last reference to the
// outfeed receiver object is dropped, and the Python garbage collector invokes
// the destructor.
//
// The shutdown sequence is implemented as follows:
// * we enqueue on all devices a computation that outfeeds a special header
//   with customer ID kOutfeedCidShutdown.
// * when each listening threads gets the shutdown header, it decrements
//   a counter of listening threads, and it
//   enqueues a special shutdown callback.
// * when each callback thread gets the shutdown callback marker, it terminates.
// * the shutdown code waits until all threads terminate.
//
// Since we currently keep the shape registry in the OutfeedReceiver, it is
// not safe to replace the OutfeedReceiver instance during the lifetime of
// the JAX program, or else previously cached jitted computations may refer
// to previously cached shapes. This can be solved, but for now we disallow
// replacing the OutfeedReceiver, and do not provide a Shutdown API to the
// Python program.

namespace xla {

// The header contains:
// 0. kOutfeedHeaderStart
// 1. consumer id
int constexpr kOutfeedHeaderWords = 2;
uint32_t constexpr kOutfeedHeaderStart = 271828;
// Special consumer IDs, without outfeed payload.
uint32_t constexpr kOutfeedCidShutdown = 0;

// Encapsulates data received from a device outfeed.
class OutfeedData {
 public:
  OutfeedData(PjRtDevice* device, uint32_t consumer_id, Shape shape)
      : device_(device),
        consumer_id_(consumer_id),
        shape_(shape),
        literal_(nullptr),
        literal_size_bytes_(0) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpythonPSoutfeed_receiverDTcc mht_0(mht_0_v, 280, "", "./tensorflow/compiler/xla/python/outfeed_receiver.cc", "OutfeedData");
}

  PjRtDevice* device() {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpythonPSoutfeed_receiverDTcc mht_1(mht_1_v, 285, "", "./tensorflow/compiler/xla/python/outfeed_receiver.cc", "device");
 return device_; }
  uint32_t consumer_id() const {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpythonPSoutfeed_receiverDTcc mht_2(mht_2_v, 289, "", "./tensorflow/compiler/xla/python/outfeed_receiver.cc", "consumer_id");
 return consumer_id_; }
  Shape shape() const {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpythonPSoutfeed_receiverDTcc mht_3(mht_3_v, 293, "", "./tensorflow/compiler/xla/python/outfeed_receiver.cc", "shape");
 return shape_; }
  std::unique_ptr<Literal> literal() {
    CHECK(literal_);
    return std::move(literal_);
  }

  void SetLiteral(std::unique_ptr<Literal> literal);

  ssize_t literal_size_bytes() const {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpythonPSoutfeed_receiverDTcc mht_4(mht_4_v, 304, "", "./tensorflow/compiler/xla/python/outfeed_receiver.cc", "literal_size_bytes");
 return literal_size_bytes_; }

  std::string DebugString() const;

 private:
  PjRtDevice* device_;
  uint32_t consumer_id_;
  Shape shape_;
  std::unique_ptr<Literal> literal_;
  ssize_t literal_size_bytes_;
};

void OutfeedData::SetLiteral(std::unique_ptr<Literal> literal) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpythonPSoutfeed_receiverDTcc mht_5(mht_5_v, 319, "", "./tensorflow/compiler/xla/python/outfeed_receiver.cc", "OutfeedData::SetLiteral");

  literal_ = std::move(literal);
  shape_ = literal_->shape();
  int total_size_bytes = 0;
  ShapeUtil::ForEachSubshape(
      shape_, [&](const Shape& literal_subshape, const ShapeIndex& index) {
        if (!literal_subshape.IsTuple()) {
          total_size_bytes += ShapeUtil::ByteSizeOf(literal_subshape, 8);
        }
      });
  literal_size_bytes_ = total_size_bytes;
}

std::string OutfeedData::DebugString() const {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpythonPSoutfeed_receiverDTcc mht_6(mht_6_v, 335, "", "./tensorflow/compiler/xla/python/outfeed_receiver.cc", "OutfeedData::DebugString");

  return absl::StrFormat("dev=%s; cons=%d; shape=%s", device_->DebugString(),
                         consumer_id_, shape_.ToString());
}

class OutfeedReceiverImpl {
 public:
  OutfeedReceiverImpl(OutfeedReceiver::Callback callback,
                      absl::Span<PjRtClient* const> clients,
                      ssize_t max_callback_queue_size_bytes);

  OutfeedReceiverImpl(const OutfeedReceiverImpl&) = delete;
  OutfeedReceiverImpl& operator=(const OutfeedReceiverImpl&) = delete;

  // Blocks until all data has been received from devices and all data
  // in the queue has been passed to Python.
  ~OutfeedReceiverImpl();

  void Start();

  StatusOr<XlaOp> AddOutfeedToBuilder(XlaBuilder* builder, XlaOp token,
                                      uint32_t consumer_id,
                                      std::vector<XlaOp> arrays);

 private:
  bool CallbackQueueHasSpace() ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu_) {
    return callback_queue_size_bytes_ < max_callback_queue_size_bytes_;
  }

  bool ShutdownDone() ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu_) {
    return (num_working_callback_threads_ == 0 && num_listening_threads_ == 0);
  }

  void CallbackThreadLoop(int device_idx);
  void DeviceListenerThreadLoop(int device_idx);

  // Enqueues to a device an outfeed operation with a shutdown consumer ID.
  Status SendShutdownOutfeedHeader(int device_idx);

  // Receives a raw Literal from a device outfeed.
  StatusOr<std::unique_ptr<Literal>> ReceiveRawFromOutfeed(PjRtDevice* device,
                                                           const Shape& shape);

  // Enqueues received data in the callbaback queue.
  void EnqueueReceivedData(uint32_t device_idx,
                           std::unique_ptr<OutfeedData> received)
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu_);

  // Shuts down the threads. See implementation notes at top of file.
  // It is not safe to restart an OutfeedReceiver after shutting down one.
  void Shutdown();

  OutfeedReceiver::Callback callback_;
  // The devices on which we are listening.
  std::vector<PjRtDevice*> devices_;
  // Maximum bytes capacity of the ensemble of callback queues.
  uint64_t max_callback_queue_size_bytes_;

  absl::Mutex mu_;
  // Registered shapes by consumer id.
  // The shape registry must be alive as long as the program exists.
  // Right now we tell the user to never restart after Shutdown.
  absl::flat_hash_map<uint32_t, Shape> shape_registry_ ABSL_GUARDED_BY(mu_);
  // How many bytes of Literal are in the ensemble of callback queues.
  uint64_t callback_queue_size_bytes_ ABSL_GUARDED_BY(mu_);
  // Threads listening.
  int num_listening_threads_ ABSL_GUARDED_BY(mu_);
  bool shutdown_started_ ABSL_GUARDED_BY(mu_);

  // How many callback threads are still working. Used for shutdown.
  int num_working_callback_threads_ ABSL_GUARDED_BY(mu_);

  std::vector<std::queue<std::unique_ptr<OutfeedData>>> callback_queues_
      ABSL_GUARDED_BY(mu_);
  // The threadpool must come last to ensure the queue exists
  // when the pool destructor is called.
  std::unique_ptr<tensorflow::thread::ThreadPool> threads_;
};

OutfeedReceiverImpl::OutfeedReceiverImpl(
    OutfeedReceiver::Callback callback, absl::Span<PjRtClient* const> clients,
    ssize_t max_callback_queue_size_bytes) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpythonPSoutfeed_receiverDTcc mht_7(mht_7_v, 419, "", "./tensorflow/compiler/xla/python/outfeed_receiver.cc", "OutfeedReceiverImpl::OutfeedReceiverImpl");

  callback_ = callback;
  max_callback_queue_size_bytes_ = max_callback_queue_size_bytes;
  for (const auto& client : clients) {
    for (auto device : client->addressable_devices()) {
      devices_.push_back(device);
    }
  }
  CHECK_GT(devices_.size(), 0);
  callback_queues_ =
      std::vector<std::queue<std::unique_ptr<OutfeedData>>>(devices_.size());

  callback_queue_size_bytes_ = 0;
  num_listening_threads_ = 0;
  num_working_callback_threads_ = 0;
  shutdown_started_ = false;
}

void OutfeedReceiverImpl::Start() {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpythonPSoutfeed_receiverDTcc mht_8(mht_8_v, 440, "", "./tensorflow/compiler/xla/python/outfeed_receiver.cc", "OutfeedReceiverImpl::Start");

  {
    absl::MutexLock lock(&mu_);
    CHECK(!shutdown_started_);
  }

  int num_threads = 2 * devices_.size();
  threads_ = absl::make_unique<tensorflow::thread::ThreadPool>(
      tensorflow::Env::Default(), "outfeed_receiver", num_threads);
  for (int device_idx = 0; device_idx < devices_.size(); ++device_idx) {
    threads_->Schedule(
        [this, device_idx]() { DeviceListenerThreadLoop(device_idx); });
    threads_->Schedule(
        [this, device_idx]() { CallbackThreadLoop(device_idx); });
  }
}

void OutfeedReceiverImpl::Shutdown() {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpythonPSoutfeed_receiverDTcc mht_9(mht_9_v, 460, "", "./tensorflow/compiler/xla/python/outfeed_receiver.cc", "OutfeedReceiverImpl::Shutdown");

  VLOG(2) << "Shutdown start";
  {
    absl::MutexLock lock(&mu_);
    CHECK(!shutdown_started_);
    shutdown_started_ = true;
  }
  for (int device_idx = 0; device_idx < devices_.size(); ++device_idx) {
    CHECK(SendShutdownOutfeedHeader(device_idx).ok());
  }
  VLOG(2) << "Shutdown waiting for listening and callback threads to stop";
  absl::MutexLock lock(&mu_);
  mu_.Await(absl::Condition(this, &OutfeedReceiverImpl::ShutdownDone));
  VLOG(2) << "Shutdown done";
}

OutfeedReceiverImpl::~OutfeedReceiverImpl() {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpythonPSoutfeed_receiverDTcc mht_10(mht_10_v, 479, "", "./tensorflow/compiler/xla/python/outfeed_receiver.cc", "OutfeedReceiverImpl::~OutfeedReceiverImpl");

  VLOG(2) << "~OutfeedReceiverImpl";
  Shutdown();
}

void OutfeedReceiverImpl::DeviceListenerThreadLoop(int device_idx) {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpythonPSoutfeed_receiverDTcc mht_11(mht_11_v, 487, "", "./tensorflow/compiler/xla/python/outfeed_receiver.cc", "OutfeedReceiverImpl::DeviceListenerThreadLoop");

  {
    absl::MutexLock lock(&mu_);
    ++num_listening_threads_;
  }
  PjRtDevice* device = devices_[device_idx];
  while (true) {
    Shape header_shape = ShapeUtil::MakeShape(U32, {kOutfeedHeaderWords});
    std::unique_ptr<Literal> header =
        ReceiveRawFromOutfeed(device, header_shape).ValueOrDie();
    absl::Span<uint32_t> header_data = header->data<uint32_t>();
    CHECK_EQ(header_data.size(), kOutfeedHeaderWords);
    CHECK_EQ(header_data[0], kOutfeedHeaderStart);
    uint32_t consumer_id = header_data[1];
    Shape shape;
    {
      absl::MutexLock lock(&mu_);
      auto registered_shape = shape_registry_.find(consumer_id);
      if (registered_shape == shape_registry_.end()) {
        LOG(FATAL)
            << "[" << device->DebugString()
            << "] Cannot find registered shape for consumer ID " << consumer_id
            << ". Perhaps the code was compiled with a different instance "
            << "of OutfeedReceiver.";
      }
      shape = registered_shape->second;
    }
    auto received = absl::make_unique<OutfeedData>(device, consumer_id, shape);
    VLOG(2) << "Listener received header " << received->DebugString();
    if (consumer_id == kOutfeedCidShutdown) {
      VLOG(2) << "[" << device->DebugString()
              << "] Listener received shutdown header";
      absl::MutexLock lock(&mu_);
      --num_listening_threads_;
      VLOG(2) << "[" << device->DebugString() << "] Enqueue shutdown callback";
      EnqueueReceivedData(device_idx, std::move(received));
      return;
    }
    std::unique_ptr<Literal> data =
        ReceiveRawFromOutfeed(device, shape).ValueOrDie();
    received->SetLiteral(std::move(data));
    absl::MutexLock lock(&mu_);
    EnqueueReceivedData(device_idx, std::move(received));
  }
}

void OutfeedReceiverImpl::EnqueueReceivedData(
    uint32_t device_idx, std::unique_ptr<OutfeedData> received)
    ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu_) {
  mu_.Await(absl::Condition(this, &OutfeedReceiverImpl::CallbackQueueHasSpace));
  ssize_t literal_size_bytes = received->literal_size_bytes();
  callback_queue_size_bytes_ += literal_size_bytes;
  VLOG(2) << "Listener enqueues data " << received->DebugString() << " of size "
          << literal_size_bytes << " bytes; "
          << (1 + callback_queues_[device_idx].size())
          << " callbacks in queue of total size " << callback_queue_size_bytes_
          << " bytes.\n";
  callback_queues_[device_idx].push(std::move(received));
}

StatusOr<std::unique_ptr<Literal>> OutfeedReceiverImpl::ReceiveRawFromOutfeed(
    PjRtDevice* device, const Shape& shape) {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpythonPSoutfeed_receiverDTcc mht_12(mht_12_v, 551, "", "./tensorflow/compiler/xla/python/outfeed_receiver.cc", "OutfeedReceiverImpl::ReceiveRawFromOutfeed");

  auto literal = std::make_unique<Literal>(shape);
  TF_RETURN_IF_ERROR(device->TransferFromOutfeed(literal.get()));
  return literal;
}

void OutfeedReceiverImpl::CallbackThreadLoop(int device_idx) {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpythonPSoutfeed_receiverDTcc mht_13(mht_13_v, 560, "", "./tensorflow/compiler/xla/python/outfeed_receiver.cc", "OutfeedReceiverImpl::CallbackThreadLoop");

  const PjRtDevice* device = devices_[device_idx];
  {
    absl::MutexLock lock(&mu_);
    num_working_callback_threads_++;
  }
  while (true) {
    std::unique_ptr<OutfeedData> received;
    {
      absl::MutexLock lock(&mu_);
      mu_.Await(absl::Condition(
          +[](std::queue<std::unique_ptr<OutfeedData>>* queue) {
            return !queue->empty();
          },
          &callback_queues_[device_idx]));
      received = std::move(callback_queues_[device_idx].front());
      callback_queues_[device_idx].pop();
      callback_queue_size_bytes_ -= received->literal_size_bytes();
      VLOG(2) << "[" << device->DebugString() << "] Dequeued callback for "
              << received->DebugString() << "; "
              << callback_queues_[device_idx].size()
              << " callbacks in queue of total size "
              << callback_queue_size_bytes_ << " bytes.\n";
    }
    if (received->consumer_id() == kOutfeedCidShutdown) {
      VLOG(2) << "[" << device->DebugString()
              << "] Callback loop received shutdown signal";
      {
        absl::MutexLock lock(&mu_);
        CHECK(callback_queues_[device_idx].empty());
        --num_working_callback_threads_;
      }
      VLOG(2) << "[" << device->DebugString() << "] Callback loop done";
      return;
    }
    {
      tensorflow::profiler::TraceMe traceme("OutfeedReceiver::Callback");
      callback_(received->device(), received->consumer_id(),
                received->literal());
    }
  }
}

Status OutfeedReceiverImpl::SendShutdownOutfeedHeader(int device_idx) {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpythonPSoutfeed_receiverDTcc mht_14(mht_14_v, 606, "", "./tensorflow/compiler/xla/python/outfeed_receiver.cc", "OutfeedReceiverImpl::SendShutdownOutfeedHeader");

  const PjRtDevice* device = devices_[device_idx];
  constexpr int consumer_id = kOutfeedCidShutdown;
  VLOG(2) << "[" << device->DebugString()
          << "] SendSpecialHeader cons=" << consumer_id;
  XlaBuilder builder(
      absl::StrFormat("special_outfeed_header_%d_%d", consumer_id, device_idx));
  XlaOp send =
      AddOutfeedToBuilder(&builder, CreateToken(&builder), consumer_id, {})
          .ValueOrDie();
  XlaComputation computation = builder.Build(send).ValueOrDie();

  CompileOptions compile_options;
  compile_options.executable_build_options.set_num_replicas(1);
  compile_options.executable_build_options.set_num_partitions(1);
  DeviceAssignment device_assignment(1, 1);
  device_assignment(0, 0) = device->id();
  compile_options.executable_build_options.set_device_assignment(
      device_assignment);

  TF_ASSIGN_OR_RETURN(std::unique_ptr<PjRtExecutable> executable,
                      devices_[device_idx]->client()->Compile(
                          computation, std::move(compile_options)));
  ExecuteOptions execute_options;
  TF_ASSIGN_OR_RETURN(
      std::vector<std::vector<std::unique_ptr<PjRtBuffer>>> output_buffers,
      executable->Execute({{}}, execute_options));
  return Status::OK();
}

StatusOr<XlaOp> OutfeedReceiverImpl::AddOutfeedToBuilder(
    XlaBuilder* builder, XlaOp token, uint32_t consumer_id,
    std::vector<XlaOp> arrays) {
   std::vector<std::string> mht_15_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpythonPSoutfeed_receiverDTcc mht_15(mht_15_v, 641, "", "./tensorflow/compiler/xla/python/outfeed_receiver.cc", "OutfeedReceiverImpl::AddOutfeedToBuilder");

  XlaOp data = Tuple(builder, std::move(arrays));
  Shape shape_with_layout = builder->GetShape(data).ValueOrDie();
  ShapeUtil::ForEachMutableSubshape(
      &shape_with_layout, [](Shape* subshape, const ShapeIndex&) {
        if (!subshape->has_layout()) {
          LayoutUtil::SetToDefaultLayout(subshape);
        }
      });
  VLOG(2) << "RegisterShape cons=" << consumer_id
          << "; shape=" << shape_with_layout.ToString();
  {
    absl::MutexLock lock(&mu_);
    auto found = shape_registry_.find(consumer_id);
    if (found != shape_registry_.end()) {
      if (!ShapeUtil::Equal(shape_with_layout, found->second)) {
        return InvalidArgument(
            "Shape %s does not match previous shape %s used "
            "for consumer id %d",
            shape_with_layout.DebugString(), found->second.DebugString(),
            consumer_id);
      }
    } else {
      shape_registry_.insert({consumer_id, shape_with_layout});
    }
  }

  std::vector<uint32_t> header{kOutfeedHeaderStart, consumer_id};
  XlaOp header_op = ConstantR1<uint32_t>(builder, header);
  // We assign the outfeed to the first device. This must match the sharding
  // for the paired infeed.
  builder->SetSharding(sharding_builder::AssignDevice(0));
  token = OutfeedWithToken(
      header_op, token, ShapeUtil::MakeShape(U32, {kOutfeedHeaderWords}), "");
  if (consumer_id != kOutfeedCidShutdown) {
    token = OutfeedWithToken(data, token, shape_with_layout, "");
  }
  builder->ClearSharding();
  return token;
}

OutfeedReceiver::OutfeedReceiver(Callback callback,
                                 absl::Span<PjRtClient* const> clients,
                                 ssize_t max_callback_queue_size_bytes) {
   std::vector<std::string> mht_16_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpythonPSoutfeed_receiverDTcc mht_16(mht_16_v, 687, "", "./tensorflow/compiler/xla/python/outfeed_receiver.cc", "OutfeedReceiver::OutfeedReceiver");

  p_impl_ = absl::make_unique<OutfeedReceiverImpl>(
      callback, clients, max_callback_queue_size_bytes);
}

OutfeedReceiver::~OutfeedReceiver() {
   std::vector<std::string> mht_17_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpythonPSoutfeed_receiverDTcc mht_17(mht_17_v, 695, "", "./tensorflow/compiler/xla/python/outfeed_receiver.cc", "OutfeedReceiver::~OutfeedReceiver");
}

void OutfeedReceiver::Start() {
   std::vector<std::string> mht_18_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpythonPSoutfeed_receiverDTcc mht_18(mht_18_v, 700, "", "./tensorflow/compiler/xla/python/outfeed_receiver.cc", "OutfeedReceiver::Start");
 p_impl_->Start(); }

StatusOr<XlaOp> OutfeedReceiver::AddOutfeedToBuilder(
    XlaBuilder* builder, XlaOp token, uint32_t consumer_id,
    std::vector<XlaOp> arrays) {
   std::vector<std::string> mht_19_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpythonPSoutfeed_receiverDTcc mht_19(mht_19_v, 707, "", "./tensorflow/compiler/xla/python/outfeed_receiver.cc", "OutfeedReceiver::AddOutfeedToBuilder");

  if (consumer_id == kOutfeedCidShutdown) {
    return InvalidArgument("Consumer ID cannot be a reserved value: %d",
                           consumer_id);
  }
  return p_impl_->AddOutfeedToBuilder(builder, token, consumer_id, arrays);
}

}  // namespace xla
