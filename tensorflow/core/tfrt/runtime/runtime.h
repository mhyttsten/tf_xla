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
#ifndef TENSORFLOW_CORE_TFRT_RUNTIME_RUNTIME_H_
#define TENSORFLOW_CORE_TFRT_RUNTIME_RUNTIME_H_
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
class MHTracer_DTPStensorflowPScorePStfrtPSruntimePSruntimeDTh {
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
   MHTracer_DTPStensorflowPScorePStfrtPSruntimePSruntimeDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePStfrtPSruntimePSruntimeDTh() {
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

#include "tensorflow/core/tfrt/runtime/work_queue_interface.h"
#include "tfrt/host_context/resource_context.h"  // from @tf_runtime

namespace tfrt {
class CoreRuntime;
class ConcurrentWorkQueue;
}  // namespace tfrt

namespace tensorflow {
namespace tfrt_stub {

// This defines the runtime abstraction in tensorflow for TFRT. It is supposed
// to provide tensorflow specific functionalities that are implemented using
// TFRT. Currently, the only intended uses for this class are:
//  1) Creating the runtime instance with user specified dependencies (eg.
//  thread pool).
//  2) Creating tensors that can be used by the runtime.
//
// It is temporary and will be replaced by the official
// tensorflow::experimental::cc::Runtime when it lands.
class Runtime {
 public:
  ABSL_DEPRECATED("Use other Create() methods instead.")
  static std::unique_ptr<Runtime> Create();

  // Creates a runtime instance with specified threading configuration. Returns
  // null upon creation error.
  static std::unique_ptr<Runtime> Create(int num_inter_op_threads,
                                         int num_intra_op_threads = 0);

  // Creates a runtime instance with the specified work_queue. Returns null upon
  // creation error.
  static std::unique_ptr<Runtime> Create(
      std::unique_ptr<WorkQueueInterface> work_queue);

  ~Runtime();

  Runtime(Runtime&&) = default;
  Runtime& operator=(Runtime&&) = default;

  // TODO(tfrt-devs): Add methods for creating TFRT tensors.

  // TODO(chky): Make this method private as it should be only used by
  // tfrt::SavedModel. Simply making tfrt::SavedModel a friend class does not
  // work because the it resides in a different namespace. But we should
  // consider moving it to the same namespace.
  tfrt::CoreRuntime* core_runtime() const {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePStfrtPSruntimePSruntimeDTh mht_0(mht_0_v, 235, "", "./tensorflow/core/tfrt/runtime/runtime.h", "core_runtime");
 return core_runtime_.get(); }
  WorkQueueInterface* work_queue() const {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePStfrtPSruntimePSruntimeDTh mht_1(mht_1_v, 239, "", "./tensorflow/core/tfrt/runtime/runtime.h", "work_queue");
 return work_queue_; }

  // `AddCreateRuntimeResourceFn` allows the client to inject per model
  // resources that are related to system-wide concepts, such as devices, when
  // loading a SavedModel.
  //
  // A longer term plan is to use a Device concept for this purpose, so that
  // Runtime contains a vector of Devices. Since it will take some time to
  // iterate on the Device concept and integrate with the existing
  // `tfrt::Device` class, we use the callback function as a temporary solution.
  //
  // The argument `fn` should be thread-safe.
  void AddCreateRuntimeResourceFn(
      std::function<void(tfrt::ResourceContext*)> fn) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePStfrtPSruntimePSruntimeDTh mht_2(mht_2_v, 255, "", "./tensorflow/core/tfrt/runtime/runtime.h", "AddCreateRuntimeResourceFn");

    runtime_resource_fns_.emplace_back(std::move(fn));
  }

  // `CreateRuntimeResources` populates `resource_ctx` with runtime-related
  // resources.
  //
  // This function is thread-safe.
  void CreateRuntimeResources(tfrt::ResourceContext* resource_ctx) const {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePStfrtPSruntimePSruntimeDTh mht_3(mht_3_v, 266, "", "./tensorflow/core/tfrt/runtime/runtime.h", "CreateRuntimeResources");

    for (auto& fn : runtime_resource_fns_) {
      fn(resource_ctx);
    }
  }

 private:
  explicit Runtime(std::unique_ptr<tfrt::CoreRuntime> core_runtime,
                   WorkQueueInterface* work_queue);

  std::unique_ptr<tfrt::CoreRuntime> core_runtime_;
  WorkQueueInterface* work_queue_ = nullptr;
  std::vector<std::function<void(tfrt::ResourceContext*)>>
      runtime_resource_fns_;
};

}  // namespace tfrt_stub
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_TFRT_RUNTIME_RUNTIME_H_
