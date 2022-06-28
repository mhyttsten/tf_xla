/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_CORE_DISTRIBUTED_RUNTIME_SERVER_LIB_H_
#define TENSORFLOW_CORE_DISTRIBUTED_RUNTIME_SERVER_LIB_H_
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
class MHTracer_DTPStensorflowPScorePSdistributed_runtimePSserver_libDTh {
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
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSserver_libDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSdistributed_runtimePSserver_libDTh() {
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

#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/protobuf/tensorflow_server.pb.h"

namespace tensorflow {

class CoordinationServiceAgent;
class DeviceMgr;
class EagerContext;
class WorkerEnv;
class MasterEnv;

// This library supports a registration/factory-based mechanism for
// creating TensorFlow server objects. Each server implementation must
// have an accompanying implementation of ServerFactory, and create a
// static "registrar" object that calls `ServerFactory::Register()`
// with an instance of the factory class. See "rpc/grpc_server_lib.cc"
// for an example.

// Represents a single TensorFlow server that exports Master and Worker
// services.
class ServerInterface {
 public:
  ServerInterface() {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSserver_libDTh mht_0(mht_0_v, 213, "", "./tensorflow/core/distributed_runtime/server_lib.h", "ServerInterface");
}
  virtual ~ServerInterface() {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSserver_libDTh mht_1(mht_1_v, 217, "", "./tensorflow/core/distributed_runtime/server_lib.h", "~ServerInterface");
}

  // Starts the server running asynchronously. Returns OK on success, otherwise
  // returns an error.
  virtual Status Start() = 0;

  // Stops the server asynchronously. Returns OK on success, otherwise returns
  // an error.
  //
  // After calling `Stop()`, the caller may call `Join()` to block until the
  // server has stopped.
  virtual Status Stop() = 0;

  // Blocks until the server has stopped. Returns OK on success, otherwise
  // returns an error.
  virtual Status Join() = 0;

  // Returns a target string that can be used to connect to this server using
  // `tensorflow::NewSession()`.
  virtual const string target() const = 0;

  virtual WorkerEnv* worker_env() = 0;
  virtual MasterEnv* master_env() = 0;

  // Update the set of workers that can be reached by the server
  virtual Status UpdateServerDef(const ServerDef& server_def) = 0;

  // Functions to operate on service-specific properties.
  //
  // Add master eager context to local eager service in order to handle enqueue
  // requests from remote workers.
  virtual Status AddMasterEagerContextToEagerService(
      const tensorflow::uint64 context_id, EagerContext* context) = 0;
  // Set coordination service agent instance to coordination service RPC handler
  virtual Status SetCoordinationServiceAgentInstance(
      CoordinationServiceAgent* agent) = 0;
  // TODO(hanyangtay): Remove this method once gRPC server clean shutdown is
  // supported.
  virtual Status StopCoordinationService() = 0;

 private:
  TF_DISALLOW_COPY_AND_ASSIGN(ServerInterface);
};

class ServerFactory {
 public:
  struct Options {
    // Local DeviceMgr to use.
    tensorflow::DeviceMgr* local_device_mgr;
  };
  // Creates a new server based on the given `server_def`, and stores
  // it in `*out_server`. Returns OK on success, otherwise returns an
  // error.
  virtual Status NewServer(const ServerDef& server_def, const Options& options,
                           std::unique_ptr<ServerInterface>* out_server) = 0;

  // Returns true if and only if this factory can create a server
  // based on the given `server_def`.
  virtual bool AcceptsOptions(const ServerDef& server_def) = 0;

  virtual ~ServerFactory() {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSserver_libDTh mht_2(mht_2_v, 280, "", "./tensorflow/core/distributed_runtime/server_lib.h", "~ServerFactory");
}

  // For each `ServerFactory` subclass, an instance of that class must
  // be registered by calling this method.
  //
  // The `server_type` must be unique to the server factory.
  static void Register(const string& server_type, ServerFactory* factory);

  // Looks up a factory that can create a server based on the given
  // `server_def`, and stores it in `*out_factory`. Returns OK on
  // success, otherwise returns an error.
  static Status GetFactory(const ServerDef& server_def,
                           ServerFactory** out_factory);
};

// Creates a server based on the given `server_def`, and stores it in
// `*out_server`. Returns OK on success, otherwise returns an error.
Status NewServer(const ServerDef& server_def,
                 std::unique_ptr<ServerInterface>* out_server);
Status NewServerWithOptions(const ServerDef& server_def,
                            const ServerFactory::Options& options,
                            std::unique_ptr<ServerInterface>* out_server);

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_DISTRIBUTED_RUNTIME_SERVER_LIB_H_
