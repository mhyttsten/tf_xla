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
class MHTracer_DTPStensorflowPScompilerPSxlaPSpythonPSoutfeed_receiver_pyDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSxlaPSpythonPSoutfeed_receiver_pyDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSxlaPSpythonPSoutfeed_receiver_pyDTcc() {
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

#include "tensorflow/compiler/xla/python/outfeed_receiver_py.h"

#include <memory>

#include "absl/algorithm/container.h"
#include "absl/memory/memory.h"
#include "absl/synchronization/mutex.h"
#include "pybind11/functional.h"
#include "pybind11/pybind11.h"
#include "tensorflow/compiler/xla/client/xla_builder.h"
#include "tensorflow/compiler/xla/pjrt/pjrt_client.h"
#include "tensorflow/compiler/xla/python/outfeed_receiver.h"
#include "tensorflow/compiler/xla/python/py_client.h"
#include "tensorflow/compiler/xla/python/types.h"

namespace xla {

namespace py = pybind11;

namespace {

// A wrapper for OutfeedReceiver for use from Python, useful for ensuring
// that the GIL is released before destroying the OutfeedReceiver.
class OutfeedReceiverForPython {
 public:
  // A callback to Python takes: consumer id, received literal.
  using CallbackToPython =
      std::function<void(ClientAndPtr<PjRtDevice>, uint32_t, pybind11::object)>;

  OutfeedReceiverForPython(CallbackToPython callback_python,
                           std::vector<std::shared_ptr<PyClient>> clients,
                           ssize_t max_callback_queue_size_bytes)
      : callback_python_(std::move(callback_python)),
        clients_(std::move(clients)) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpythonPSoutfeed_receiver_pyDTcc mht_0(mht_0_v, 218, "", "./tensorflow/compiler/xla/python/outfeed_receiver_py.cc", "OutfeedReceiverForPython");

    OutfeedReceiver::Callback callback =
        [this](PjRtDevice* device, uint32_t consumer_id,
               std::shared_ptr<Literal> literal) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpythonPSoutfeed_receiver_pyDTcc mht_1(mht_1_v, 224, "", "./tensorflow/compiler/xla/python/outfeed_receiver_py.cc", "lambda");

          this->Callback(device, consumer_id, std::move(literal));
        };
    std::vector<PjRtClient*> client_ptrs(clients_.size());
    absl::c_transform(clients_, client_ptrs.begin(),
                      [](const std::shared_ptr<PyClient>& client) {
                        return client->pjrt_client();
                      });
    outfeed_receiver_ = absl::make_unique<OutfeedReceiver>(
        callback, client_ptrs, max_callback_queue_size_bytes);
  }
  OutfeedReceiverForPython(const OutfeedReceiverForPython&) = delete;
  OutfeedReceiverForPython& operator=(const OutfeedReceiverForPython&) = delete;

  ~OutfeedReceiverForPython() {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpythonPSoutfeed_receiver_pyDTcc mht_2(mht_2_v, 241, "", "./tensorflow/compiler/xla/python/outfeed_receiver_py.cc", "~OutfeedReceiverForPython");

    // This destructor is called from the Python GC. Release it for the duration
    // of the destruction, including the destruction of the OutfeedReceiver,
    // when we may actually have to wait for threads to end. During this time
    // we do not callback to Python (sometimes we get an exception
    // "std::runtime_error: scoped_acquire::dec_ref(): thread state must
    // be current!"").
    {
      absl::MutexLock lock(&mu_);
      outfeed_receiver_shutting_down_ = true;
    }
    py::gil_scoped_release gil_release;
    outfeed_receiver_ = nullptr;  // Shutdown the outfeed receiver.
  }

  void Start() {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpythonPSoutfeed_receiver_pyDTcc mht_3(mht_3_v, 259, "", "./tensorflow/compiler/xla/python/outfeed_receiver_py.cc", "Start");
 outfeed_receiver_->Start(); }

  StatusOr<XlaOp> AddOutfeed(XlaBuilder* builder, XlaOp token,
                             uint32_t consumer_id, std::vector<XlaOp> arrays) {
    return outfeed_receiver_->AddOutfeedToBuilder(builder, token, consumer_id,
                                                  arrays);
  }

  void Callback(PjRtDevice* device, uint32_t consumer_id,
                std::shared_ptr<Literal> literal) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpythonPSoutfeed_receiver_pyDTcc mht_4(mht_4_v, 271, "", "./tensorflow/compiler/xla/python/outfeed_receiver_py.cc", "Callback");

    {
      absl::MutexLock lock(&mu_);
      if (outfeed_receiver_shutting_down_) {
        VLOG(2) << "Ignoring unsafe callback to Python during shutdown";
        return;
      }
    }
    // We expect the number of clients to be small, so an O(n) search is fine.
    auto it = absl::c_find_if(
        clients_, [device](const std::shared_ptr<PyClient>& client) {
          return client->pjrt_client() == device->client();
        });
    CHECK(it != clients_.end());
    py::gil_scoped_acquire gil_acquire;  // Need GIL also for LiteralToPython
    py::object literal_python =
        LiteralToPython(std::move(literal)).ValueOrDie();
    // The callback_ should handle all exceptions in user-code. If we get
    // an exception here, it is a bug in the callback and we should stop.
    callback_python_(WrapWithClient<PjRtDevice>(*it, device), consumer_id,
                     std::move(literal_python));
  }

 private:
  CallbackToPython callback_python_;
  absl::Mutex mu_;
  bool outfeed_receiver_shutting_down_ ABSL_GUARDED_BY(mu_) = false;
  std::vector<std::shared_ptr<PyClient>> clients_;
  std::unique_ptr<OutfeedReceiver> outfeed_receiver_;
};

}  // namespace

void BuildOutfeedReceiverSubmodule(py::module* m) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpythonPSoutfeed_receiver_pyDTcc mht_5(mht_5_v, 307, "", "./tensorflow/compiler/xla/python/outfeed_receiver_py.cc", "BuildOutfeedReceiverSubmodule");

  py::module outfeed_receiver =
      m->def_submodule("outfeed_receiver", "Outfeed receiver");
  outfeed_receiver.def(
      "start",
      [](OutfeedReceiverForPython::CallbackToPython callback_to_python,
         std::vector<std::shared_ptr<PyClient>> clients,
         ssize_t max_callback_queue_size_bytes)
          -> std::unique_ptr<OutfeedReceiverForPython> {
        auto server = absl::make_unique<OutfeedReceiverForPython>(
            callback_to_python, clients, max_callback_queue_size_bytes);
        server->Start();
        return server;
      },
      py::arg("callback_to_python"), py::arg("backends"),
      py::arg("max_queue_size_bytes") = 256 * 1024 * 1024,
      R"(Starts a multithreaded outfeed receiver.

      There is one thread for each of the specified devices. When Python
      drops the last reference to the returned object, the receiver is shut
      down. The destructor will block until all data is received from
      devices.

      Args:
        * callback_to_python: a Python callback to call, with <consumer_id>
          and the data received.
        * backends: the list of backends to listen on.
        * max_queue_size_bytes: an optional integer to bound the maximum size
            of arrays in the callback queue. When this limit is reached the
            device listener pauses.
      )",
      py::call_guard<py::gil_scoped_release>());

  py::class_<OutfeedReceiverForPython> outfeed_receiver_class(
      outfeed_receiver, "OutfeedReceiverForPython");

  outfeed_receiver_class.def(
      "add_outfeed", &OutfeedReceiverForPython::AddOutfeed, py::arg("builder"),
      py::arg("token"), py::arg("consumer_id"), py::arg("arrays"),
      R"(Adds an outfeed into the given computation builder.

      Has the side-effect of registering the sent shape along with the consumer
      ID. Returns error if the outfeed shape is not compatible with previously
      used shape for the same consumer ID.)",
      py::call_guard<py::gil_scoped_release>());
}

}  // namespace xla
