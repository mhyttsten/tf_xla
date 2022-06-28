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

#ifndef TENSORFLOW_CORE_DISTRIBUTED_RUNTIME_TENSOR_CODING_H_
#define TENSORFLOW_CORE_DISTRIBUTED_RUNTIME_TENSOR_CODING_H_
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
class MHTracer_DTPStensorflowPScorePSdistributed_runtimePStensor_codingDTh {
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
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePStensor_codingDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSdistributed_runtimePStensor_codingDTh() {
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


#include "tensorflow/core/framework/allocator.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/protobuf.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/protobuf/worker.pb.h"

namespace tensorflow {

class Allocator;
class DeviceBase;
class TensorProto;

// TensorResponse can be used as the destination of an RPC that returns
// a RecvTensorResponse.  It efficiently decodes the incoming data
// into Tensor contents as well as associated metadata.
class TensorResponse {
 public:
  TensorResponse() {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePStensor_codingDTh mht_0(mht_0_v, 206, "", "./tensorflow/core/distributed_runtime/tensor_coding.h", "TensorResponse");
}

  // Reset to initial state.
  void Clear();

  // Clear just tensor_ and meta_ members without setting allocation
  // related members.
  void ClearTensor();

  // Initialize memory allocation related members.
  void InitAlloc(DeviceBase* d, const AllocatorAttributes& aa);

  // Source provides a way for a particular RPC implementation to provide
  // received data to ParseFrom.
  class Source {
   public:
    virtual ~Source();

    // Return the stream that contains the data to be parsed.
    // Note that this method might be invoked more than once if
    // ParseFrom needs to fall back to a more expensive parsing method.
    // Every call must return a stream pointing at the beginning of
    // the serialized RecvTensorResponse.
    //
    // Note that a subsequent call to contents() invalidates previous
    // results of contents().
    //
    // Ownership of the returned stream is retained by the Source and
    // should not be deleted by the caller.
    virtual ::tensorflow::protobuf::io::ZeroCopyInputStream* contents() = 0;
  };

  // Parse the RecvTensorResponse encoded in the data yielded by
  // source->contents() into *this.
  Status ParseFrom(Source* source);

  // Initialize tensor from *response.
  // Leaves *response with unspecified contents.
  Status InitFrom(RecvTensorResponse* response);

  // Initialize tensor metadata from response and allocate
  // uninitialized backing storage for actual contents.
  void InitPartial(const RecvTensorResponse& response,
                   const AllocationAttributes& allocation_attr);

  // Return a reference to the parsed tensor.  The tensor will remain
  // live only until *this is destroyed or modified.
  const Tensor& tensor() const {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePStensor_codingDTh mht_1(mht_1_v, 256, "", "./tensorflow/core/distributed_runtime/tensor_coding.h", "tensor");
 return tensor_; }

  // Return a reference to the parsed tensor metadata (no contents).
  // The result will remain live only until *this is destroyed or
  // modified.
  const RecvTensorResponse& metadata() const {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePStensor_codingDTh mht_2(mht_2_v, 264, "", "./tensorflow/core/distributed_runtime/tensor_coding.h", "metadata");
 return meta_; }

  // Return pointer to the device hosting the tensor.
  DeviceBase* device() const {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePStensor_codingDTh mht_3(mht_3_v, 270, "", "./tensorflow/core/distributed_runtime/tensor_coding.h", "device");
 return device_; }

 private:
  bool ParseTensorSubmessage(protobuf::io::CodedInputStream* input,
                             TensorProto* tensor_meta);
  bool ParseFast(Source* source);
  bool ParseSlow(Source* source);

  bool on_host_ = false;
  DeviceBase* device_ = nullptr;
  AllocatorAttributes alloc_attrs_;
  Allocator* allocator_ = nullptr;
  bool already_used_ = false;
  Tensor tensor_;
  RecvTensorResponse meta_;
};

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_DISTRIBUTED_RUNTIME_TENSOR_CODING_H_
