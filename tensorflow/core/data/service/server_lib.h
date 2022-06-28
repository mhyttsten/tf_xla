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

#ifndef TENSORFLOW_CORE_DATA_SERVICE_SERVER_LIB_H_
#define TENSORFLOW_CORE_DATA_SERVICE_SERVER_LIB_H_
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
class MHTracer_DTPStensorflowPScorePSdataPSservicePSserver_libDTh {
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
   MHTracer_DTPStensorflowPScorePSdataPSservicePSserver_libDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSdataPSservicePSserver_libDTh() {
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


#include "grpcpp/server.h"
#include "grpcpp/server_builder.h"
#include "tensorflow/core/data/service/data_transfer.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/profiler/rpc/profiler_service_impl.h"
#include "tensorflow/core/protobuf/service_config.pb.h"

namespace tensorflow {
namespace data {

// Forward declared because transitively depending on .grpc.pb.h files causes
// issues in the pywrap build.
class GrpcDispatcherImpl;
class GrpcWorkerImpl;

// A grpc server for the tf.data service.
class GrpcDataServerBase {
 public:
  // Constructs a tf.data server with the specified port. If the port is 0, the
  // server will find an available port in `Start()`. The chosen port can be
  // found by calling `BoundPort()`.
  GrpcDataServerBase(int requested_port, const std::string& protocol,
                     const std::string server_type);
  virtual ~GrpcDataServerBase() {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSdataPSservicePSserver_libDTh mht_0(mht_0_v, 211, "", "./tensorflow/core/data/service/server_lib.h", "~GrpcDataServerBase");
}

  // Starts the server running asynchronously.
  Status Start();

  // Stops the server. This will block until all outstanding requests complete.
  void Stop();

  // Blocks until the server stops.
  void Join();

  // Returns the port bound by the server. Only valid after calling Start().
  int BoundPort();

 protected:
  virtual void AddDataServiceToBuilder(::grpc::ServerBuilder& builder) = 0;
  void AddProfilerServiceToBuilder(::grpc::ServerBuilder& builder);
  // Starts the service. This will be called after building the service, so
  // bound_port() will return the actual bound port.
  virtual Status StartServiceInternal() = 0;
  virtual void StopServiceInternal() {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSdataPSservicePSserver_libDTh mht_1(mht_1_v, 234, "", "./tensorflow/core/data/service/server_lib.h", "StopServiceInternal");
}

  int bound_port() {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSdataPSservicePSserver_libDTh mht_2(mht_2_v, 239, "", "./tensorflow/core/data/service/server_lib.h", "bound_port");
 return bound_port_; }

  const int requested_port_;
  const std::string protocol_;
  const std::string server_type_;

 private:
  int bound_port_;
  bool started_ = false;
  bool stopped_ = false;

  std::unique_ptr<::grpc::Server> server_;
  // TensorFlow profiler service implementation.
  std::unique_ptr<grpc::ProfilerService::Service> profiler_service_ = nullptr;
};

class DispatchGrpcDataServer : public GrpcDataServerBase {
 public:
  explicit DispatchGrpcDataServer(const experimental::DispatcherConfig& config);
  ~DispatchGrpcDataServer() override;

  // Returns the number of workers registerd with the dispatcher.
  Status NumWorkers(int* num_workers);
  // Returns the number of active (non-finished) jobs running on the dispatcher.
  size_t NumActiveJobs();

 protected:
  void AddDataServiceToBuilder(::grpc::ServerBuilder& builder) override;
  Status StartServiceInternal() override;

 private:
  const experimental::DispatcherConfig config_;
  // Owned. We use a raw pointer because GrpcDispatcherImpl is forward-declared.
  GrpcDispatcherImpl* service_;
};

class WorkerGrpcDataServer : public GrpcDataServerBase {
 public:
  explicit WorkerGrpcDataServer(const experimental::WorkerConfig& config);
  ~WorkerGrpcDataServer() override;

  // Returns the number of tasks currently being executed by the worker.
  Status NumTasks(int* num_tasks);

 protected:
  void AddDataServiceToBuilder(::grpc::ServerBuilder& builder) override;
  Status StartServiceInternal() override;
  void StopServiceInternal() override;

 private:
  const experimental::WorkerConfig config_;
  // Owned. We use a raw pointer because GrpcWorkerImpl is forward-declared.
  GrpcWorkerImpl* service_;
  std::shared_ptr<DataTransferServer> transfer_server_;
};

// Creates a dispatch tf.data server and stores it in `out_server`.
Status NewDispatchServer(const experimental::DispatcherConfig& config,
                         std::unique_ptr<DispatchGrpcDataServer>& out_server);

// Creates a worker tf.data server and stores it in `out_server`.
Status NewWorkerServer(const experimental::WorkerConfig& config,
                       std::unique_ptr<WorkerGrpcDataServer>& out_server);

}  // namespace data
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_DATA_SERVICE_SERVER_LIB_H_
