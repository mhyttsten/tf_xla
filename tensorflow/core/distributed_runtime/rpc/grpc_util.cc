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
class MHTracer_DTPStensorflowPScorePSdistributed_runtimePSrpcPSgrpc_utilDTcc {
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
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSrpcPSgrpc_utilDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSdistributed_runtimePSrpcPSgrpc_utilDTcc() {
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

#include "tensorflow/core/distributed_runtime/rpc/grpc_util.h"
#include "tensorflow/core/distributed_runtime/tensor_coding.h"
#include "tensorflow/core/lib/random/random.h"

namespace tensorflow {

namespace {

double GenerateUniformRandomNumber() {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSrpcPSgrpc_utilDTcc mht_0(mht_0_v, 193, "", "./tensorflow/core/distributed_runtime/rpc/grpc_util.cc", "GenerateUniformRandomNumber");

  return random::New64() * (1.0 / std::numeric_limits<uint64>::max());
}

double GenerateUniformRandomNumberBetween(double a, double b) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSrpcPSgrpc_utilDTcc mht_1(mht_1_v, 200, "", "./tensorflow/core/distributed_runtime/rpc/grpc_util.cc", "GenerateUniformRandomNumberBetween");

  if (a == b) return a;
  DCHECK_LT(a, b);
  return a + GenerateUniformRandomNumber() * (b - a);
}

}  // namespace

int64_t ComputeBackoffMicroseconds(int current_retry_attempt, int64_t min_delay,
                                   int64_t max_delay) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSrpcPSgrpc_utilDTcc mht_2(mht_2_v, 212, "", "./tensorflow/core/distributed_runtime/rpc/grpc_util.cc", "ComputeBackoffMicroseconds");

  DCHECK_GE(current_retry_attempt, 0);

  // This function with the constants below is calculating:
  //
  // (0.4 * min_delay) + (random[0.6,1.0] * min_delay * 1.3^retries)
  //
  // Note that there is an extra truncation that occurs and is documented in
  // comments below.
  constexpr double kBackoffBase = 1.3;
  constexpr double kBackoffRandMult = 0.4;

  // This first term does not vary with current_retry_attempt or a random
  // number. It exists to ensure the final term is >= min_delay
  const double first_term = kBackoffRandMult * min_delay;

  // This is calculating min_delay * 1.3^retries
  double uncapped_second_term = min_delay;
  while (current_retry_attempt > 0 &&
         uncapped_second_term < max_delay - first_term) {
    current_retry_attempt--;
    uncapped_second_term *= kBackoffBase;
  }
  // Note that first_term + uncapped_second_term can exceed max_delay here
  // because of the final multiply by kBackoffBase.  We fix that problem with
  // the min() below.
  double second_term = std::min(uncapped_second_term, max_delay - first_term);

  // This supplies the random jitter to ensure that retried don't cause a
  // thundering herd problem.
  second_term *=
      GenerateUniformRandomNumberBetween(1.0 - kBackoffRandMult, 1.0);

  return std::max(static_cast<int64_t>(first_term + second_term), min_delay);
}

::grpc::Status GrpcMaybeUnparseProto(const protobuf::Message& src,
                                     grpc::ByteBuffer* dst) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSrpcPSgrpc_utilDTcc mht_3(mht_3_v, 252, "", "./tensorflow/core/distributed_runtime/rpc/grpc_util.cc", "GrpcMaybeUnparseProto");

  bool own_buffer;
  return ::grpc::GenericSerialize<::grpc::ProtoBufferWriter,
                                  protobuf::Message>(src, dst, &own_buffer);
}

// GrpcMaybeUnparseProto from a string simply copies the string to the
// ByteBuffer.
::grpc::Status GrpcMaybeUnparseProto(const string& src, grpc::ByteBuffer* dst) {
   std::vector<std::string> mht_4_v;
   mht_4_v.push_back("src: \"" + src + "\"");
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSrpcPSgrpc_utilDTcc mht_4(mht_4_v, 264, "", "./tensorflow/core/distributed_runtime/rpc/grpc_util.cc", "GrpcMaybeUnparseProto");

  ::grpc::Slice s(src.data(), src.size());
  ::grpc::ByteBuffer buffer(&s, 1);
  dst->Swap(&buffer);
  return ::grpc::Status::OK;
}

bool GrpcMaybeParseProto(::grpc::ByteBuffer* src, protobuf::Message* dst) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSrpcPSgrpc_utilDTcc mht_5(mht_5_v, 274, "", "./tensorflow/core/distributed_runtime/rpc/grpc_util.cc", "GrpcMaybeParseProto");

  ::grpc::ProtoBufferReader reader(src);
  return dst->ParseFromZeroCopyStream(&reader);
}

// Overload of GrpcParseProto so we can decode a TensorResponse without
// extra copying.  This overload is used by the RPCState class in
// grpc_state.h.
bool GrpcMaybeParseProto(::grpc::ByteBuffer* src, TensorResponse* dst) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSrpcPSgrpc_utilDTcc mht_6(mht_6_v, 285, "", "./tensorflow/core/distributed_runtime/rpc/grpc_util.cc", "GrpcMaybeParseProto");

  ::tensorflow::GrpcByteSource byte_source(src);
  auto s = dst->ParseFrom(&byte_source);
  return s.ok();
}

// GrpcMaybeParseProto simply copies bytes into the string.
bool GrpcMaybeParseProto(grpc::ByteBuffer* src, string* dst) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSrpcPSgrpc_utilDTcc mht_7(mht_7_v, 295, "", "./tensorflow/core/distributed_runtime/rpc/grpc_util.cc", "GrpcMaybeParseProto");

  dst->clear();
  dst->reserve(src->Length());
  std::vector<::grpc::Slice> slices;
  if (!src->Dump(&slices).ok()) {
    return false;
  }
  for (const ::grpc::Slice& s : slices) {
    dst->append(reinterpret_cast<const char*>(s.begin()), s.size());
  }
  return true;
}

// GrpcMaybeParseProto simply copies bytes into the tstring.
bool GrpcMaybeParseProto(grpc::ByteBuffer* src, tstring* dst) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSrpcPSgrpc_utilDTcc mht_8(mht_8_v, 312, "", "./tensorflow/core/distributed_runtime/rpc/grpc_util.cc", "GrpcMaybeParseProto");

  dst->clear();
  dst->reserve(src->Length());
  std::vector<::grpc::Slice> slices;
  if (!src->Dump(&slices).ok()) {
    return false;
  }
  for (const ::grpc::Slice& s : slices) {
    dst->append(reinterpret_cast<const char*>(s.begin()), s.size());
  }
  return true;
}

}  // namespace tensorflow
