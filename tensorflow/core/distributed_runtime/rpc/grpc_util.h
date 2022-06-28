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

#ifndef TENSORFLOW_CORE_DISTRIBUTED_RUNTIME_RPC_GRPC_UTIL_H_
#define TENSORFLOW_CORE_DISTRIBUTED_RUNTIME_RPC_GRPC_UTIL_H_
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
class MHTracer_DTPStensorflowPScorePSdistributed_runtimePSrpcPSgrpc_utilDTh {
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
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSrpcPSgrpc_utilDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSdistributed_runtimePSrpcPSgrpc_utilDTh() {
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

#include "grpcpp/grpcpp.h"
#include "grpcpp/impl/codegen/proto_utils.h"
#include "grpcpp/support/byte_buffer.h"
#include "tensorflow/core/distributed_runtime/error_payloads.h"
#include "tensorflow/core/distributed_runtime/tensor_coding.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/strings/stringprintf.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/protobuf.h"
#include "tensorflow/core/platform/stringpiece.h"
#include "tensorflow/core/protobuf/distributed_runtime_payloads.pb.h"

namespace tensorflow {

// Given the total number of RPC retries attempted, return a randomized
// amount of time to delay before retrying the request.
//
// The average computed backoff increases with the number of RPCs attempted.
// See implementation for details on the calculations.
int64_t ComputeBackoffMicroseconds(int current_retry_attempt,
                                   int64_t min_delay = 1000,
                                   int64_t max_delay = 10000000);

// Thin wrapper around ::grpc::ProtoBufferReader to give TensorResponse an
// efficient byte reader from which to decode a RecvTensorResponse.
class GrpcByteSource : public TensorResponse::Source {
 public:
  explicit GrpcByteSource(::grpc::ByteBuffer* buffer) : buffer_(buffer) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSrpcPSgrpc_utilDTh mht_0(mht_0_v, 218, "", "./tensorflow/core/distributed_runtime/rpc/grpc_util.h", "GrpcByteSource");
}
  ~GrpcByteSource() override {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSrpcPSgrpc_utilDTh mht_1(mht_1_v, 222, "", "./tensorflow/core/distributed_runtime/rpc/grpc_util.h", "~GrpcByteSource");
 DeleteStream(); }

  typedef ::grpc::ProtoBufferReader Reader;

  protobuf::io::ZeroCopyInputStream* contents() override {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSrpcPSgrpc_utilDTh mht_2(mht_2_v, 229, "", "./tensorflow/core/distributed_runtime/rpc/grpc_util.h", "contents");

    DeleteStream();
    stream_ = new (&space_) Reader(buffer_);
    return stream_;
  }

 private:
  void DeleteStream() {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSrpcPSgrpc_utilDTh mht_3(mht_3_v, 239, "", "./tensorflow/core/distributed_runtime/rpc/grpc_util.h", "DeleteStream");

    if (stream_) {
      stream_->~Reader();
    }
  }

  ::grpc::ByteBuffer* buffer_;  // Not owned
  Reader* stream_ = nullptr;    // Points into space_ if non-nullptr
  char space_[sizeof(Reader)];
};

constexpr char kStreamRemovedMessage[] = "Stream removed";

// Identify if the given grpc::Status corresponds to an HTTP stream removed
// error (see chttp2_transport.cc).
//
// When auto-reconnecting to a remote TensorFlow worker after it restarts, gRPC
// can return an UNKNOWN error code with a "Stream removed" error message.
// This should not be treated as an unrecoverable error.
//
// N.B. This is dependent on the error message from grpc remaining consistent.
inline bool IsStreamRemovedError(const ::grpc::Status& s) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSrpcPSgrpc_utilDTh mht_4(mht_4_v, 263, "", "./tensorflow/core/distributed_runtime/rpc/grpc_util.h", "IsStreamRemovedError");

  return !s.ok() && s.error_code() == ::grpc::StatusCode::UNKNOWN &&
         s.error_message() == kStreamRemovedMessage;
}

inline std::string SerializePayloads(const ::tensorflow::Status& s) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSrpcPSgrpc_utilDTh mht_5(mht_5_v, 271, "", "./tensorflow/core/distributed_runtime/rpc/grpc_util.h", "SerializePayloads");

  distributed_runtime::GrpcPayloadContainer container;
  s.ForEachPayload(
      [&container](tensorflow::StringPiece key, tensorflow::StringPiece value) {
        (*container.mutable_payloads())[std::string(key)] = std::string(value);
      });
  return container.SerializeAsString();
}

inline void InsertSerializedPayloads(::tensorflow::Status& s,
                                     std::string payloads) {
   std::vector<std::string> mht_6_v;
   mht_6_v.push_back("payloads: \"" + payloads + "\"");
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSrpcPSgrpc_utilDTh mht_6(mht_6_v, 285, "", "./tensorflow/core/distributed_runtime/rpc/grpc_util.h", "InsertSerializedPayloads");

  distributed_runtime::GrpcPayloadContainer container;
  if (container.ParseFromString(payloads)) {
    for (const auto& key_val : container.payloads()) {
      s.SetPayload(key_val.first, key_val.second);
    }
  } else {
    s.SetPayload(kGrpcPayloadsLost,
                 distributed_runtime::GrpcPayloadsLost().SerializeAsString());
  }
}

inline ::tensorflow::Status FromGrpcStatus(const ::grpc::Status& s) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSrpcPSgrpc_utilDTh mht_7(mht_7_v, 300, "", "./tensorflow/core/distributed_runtime/rpc/grpc_util.h", "FromGrpcStatus");

  if (s.ok()) {
    return Status::OK();
  } else {
    ::tensorflow::Status converted;
    // Convert "UNKNOWN" stream removed errors into unavailable, to allow
    // for retry upstream.
    if (IsStreamRemovedError(s)) {
      converted = Status(tensorflow::error::UNAVAILABLE, s.error_message());
    }
    converted = Status(static_cast<tensorflow::error::Code>(s.error_code()),
                       s.error_message());
    InsertSerializedPayloads(converted, s.error_details());
    return converted;
  }
}

inline ::grpc::Status ToGrpcStatus(const ::tensorflow::Status& s) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSrpcPSgrpc_utilDTh mht_8(mht_8_v, 320, "", "./tensorflow/core/distributed_runtime/rpc/grpc_util.h", "ToGrpcStatus");

  if (s.ok()) {
    return ::grpc::Status::OK;
  } else {
    if (s.error_message().size() > 3072 /* 3k bytes */) {
      // TODO(b/62947679): Remove truncation once the gRPC issue is resolved.
      string scratch =
          strings::Printf("%.3072s ... [truncated]", s.error_message().c_str());
      LOG(ERROR) << "Truncated error message: " << s;
      return ::grpc::Status(static_cast<::grpc::StatusCode>(s.code()), scratch,
                            SerializePayloads(s));
    }
    return ::grpc::Status(static_cast<::grpc::StatusCode>(s.code()),
                          s.error_message(), SerializePayloads(s));
  }
}

typedef std::shared_ptr<::grpc::Channel> SharedGrpcChannelPtr;

inline string GrpcIdKey() {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSrpcPSgrpc_utilDTh mht_9(mht_9_v, 342, "", "./tensorflow/core/distributed_runtime/rpc/grpc_util.h", "GrpcIdKey");
 return "tf-rpc"; }

// Serialize src and store in *dst.
::grpc::Status GrpcMaybeUnparseProto(const protobuf::Message& src,
                                     ::grpc::ByteBuffer* dst);

// Parse contents of src and initialize *dst with them.
bool GrpcMaybeParseProto(::grpc::ByteBuffer* src, protobuf::Message* dst);

// Specialization for TensorResponse
bool GrpcMaybeParseProto(::grpc::ByteBuffer* src, TensorResponse* dst);

// Copy string src to grpc buffer *dst.
::grpc::Status GrpcMaybeUnparseProto(const string& src,
                                     ::grpc::ByteBuffer* dst);

// Copy grpc buffer src to string *dst.
bool GrpcMaybeParseProto(::grpc::ByteBuffer* src, string* dst);

// Copy grpc buffer src to tstring *dst.
bool GrpcMaybeParseProto(::grpc::ByteBuffer* src, tstring* dst);

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_DISTRIBUTED_RUNTIME_RPC_GRPC_UTIL_H_
