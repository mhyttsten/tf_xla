/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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
#ifndef TENSORFLOW_CORE_DISTRIBUTED_RUNTIME_EAGER_REMOTE_TENSOR_HANDLE_DATA_H_
#define TENSORFLOW_CORE_DISTRIBUTED_RUNTIME_EAGER_REMOTE_TENSOR_HANDLE_DATA_H_
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
class MHTracer_DTPStensorflowPScorePSdistributed_runtimePSeagerPSremote_tensor_handle_dataDTh {
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
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSeagerPSremote_tensor_handle_dataDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSdistributed_runtimePSeagerPSremote_tensor_handle_dataDTh() {
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


#include "tensorflow/core/common_runtime/eager/context.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/lib/core/status.h"

namespace tensorflow {

// Remote Tensor Handle: A handle to a Tensor on a remote host. Note that only
// the shape is known.
class RemoteTensorHandleData {
 public:
  // Constructor for lazy remote handles. A lazy remote handle is created on
  // a remote worker with an op_id and an output_num. It doesn't control the
  // lifetime of a remote handle that it refers to. If it refers to a remote
  // function input, it's sent by a client which won't serialize it until
  // the corresponding remote tensor is ready. So the remote tensor should be
  // ready when we create a lazy remote handle. If it refers to a remote output,
  // it's not ready until the shape is set.
  RemoteTensorHandleData(int64_t op_id, int output_num, uint64 context_view_id,
                         bool is_ready);
  // Constructor for unshaped remote handles. It controls the lifetime of a
  // remote handel that it refers to.
  RemoteTensorHandleData(int64_t op_id, int output_num,
                         const string& remote_task, EagerContext* ctx);
  ~RemoteTensorHandleData();

  // A remote tensor handle does not have a Tensor object, hence it can only
  // support the shape requests.
  Status Shape(TensorShape* shape) const;
  Status NumDims(int* num_dims) const;
  Status Dim(int dim_index, int64_t* dim) const;
  Status NumElements(int64_t* num_elements) const;
  Status Unprotect() {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSeagerPSremote_tensor_handle_dataDTh mht_0(mht_0_v, 218, "", "./tensorflow/core/distributed_runtime/eager/remote_tensor_handle_data.h", "Unprotect");
 return Status::OK(); }

  bool IsReady() const;
  Status WaitReady(const char* caller) const;
  Status SetShape(const TensorShape& shape);
  Status SetShapeAndRemoteTask(const TensorShape& shape,
                               const string& remote_task);
  void Poison(Status status);
  Status IsPoisoned() const;

  string DebugString() const;

  // Return the op id and output num. If wait_util_ready is true, block until
  // the remote tensor is ready on a remote worker.
  Status OpIdAndOutputNum(const bool wait_util_ready, int64_t* op_id,
                          int32* output_num) const;

  uint64 context_view_id() const {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSeagerPSremote_tensor_handle_dataDTh mht_1(mht_1_v, 238, "", "./tensorflow/core/distributed_runtime/eager/remote_tensor_handle_data.h", "context_view_id");
 return context_view_id_; }

 private:
  mutable mutex mu_;
  bool is_ready_ TF_GUARDED_BY(mu_);
  Status is_poisoned_ TF_GUARDED_BY(mu_);
  TensorShape shape_ TF_GUARDED_BY(mu_);

  // IDs required when this class is representing a remote tensor handle.
  const int64_t op_id_;
  const int32 output_num_;
  string remote_task_ TF_GUARDED_BY(mu_);
  uint64 context_id_;
  uint64 context_view_id_;
  EagerContext* ctx_;
};

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_DISTRIBUTED_RUNTIME_EAGER_REMOTE_TENSOR_HANDLE_DATA_H_
