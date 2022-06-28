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
class MHTracer_DTPStensorflowPScorePSdistributed_runtimePSrpcPSgrpc_util_testDTcc {
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
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSrpcPSgrpc_util_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSdistributed_runtimePSrpcPSgrpc_util_testDTcc() {
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

#include "grpcpp/grpcpp.h"
#include "tensorflow/core/distributed_runtime/error_payloads.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/platform/test_benchmark.h"
#include "tensorflow/core/protobuf/worker.pb.h"

namespace tensorflow {

namespace {
string ToString(const grpc::ByteBuffer& buf) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSrpcPSgrpc_util_testDTcc mht_0(mht_0_v, 197, "", "./tensorflow/core/distributed_runtime/rpc/grpc_util_test.cc", "ToString");

  std::vector<grpc::Slice> slices;
  CHECK(buf.Dump(&slices).ok());
  string result;
  for (const grpc::Slice& s : slices) {
    result.append(reinterpret_cast<const char*>(s.begin()), s.size());
  }
  return result;
}

// Return a ByteBuffer that contains str split up into num_slices slices.
grpc::ByteBuffer MakeBuffer(const string& str, int num_slices) {
   std::vector<std::string> mht_1_v;
   mht_1_v.push_back("str: \"" + str + "\"");
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSrpcPSgrpc_util_testDTcc mht_1(mht_1_v, 212, "", "./tensorflow/core/distributed_runtime/rpc/grpc_util_test.cc", "MakeBuffer");

  // Convert to a ByteBuffer.
  std::vector<::grpc::Slice> slices;
  const size_t per_slice = (str.size() + num_slices - 1) / num_slices;
  for (size_t pos = 0; pos < str.size();) {
    const size_t n = std::min(str.size() - pos, per_slice);
    slices.emplace_back(&str[pos], n);
    pos += n;
  }
  if (slices.empty()) {
    slices.emplace_back();
  }
  return ::grpc::ByteBuffer(&slices[0], slices.size());
}

// Make a proto with approximately the specified length.
CleanupAllRequest MakeProto(int size) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSrpcPSgrpc_util_testDTcc mht_2(mht_2_v, 231, "", "./tensorflow/core/distributed_runtime/rpc/grpc_util_test.cc", "MakeProto");

  int approx_size = 0;
  CleanupAllRequest proto;
  int index = 0;
  while (approx_size < size) {
    int item_size = std::min(size - approx_size, 1024);
    proto.add_container(string(item_size, 'a' + static_cast<char>(index % 26)));
    approx_size += item_size + 3;  // +3 for encoding overhead.
    index++;
  }
  return proto;
}
}  // namespace

TEST(PayloadSerialization, PayloadsAreTransmitted) {
  ::tensorflow::Status status = errors::InvalidArgument("invalid arg message");
  status.SetPayload("a", "\\xFF\\x02\\x03");
  ::tensorflow::Status status_recovered = FromGrpcStatus(ToGrpcStatus(status));

  ASSERT_TRUE(status_recovered.GetPayload("a").has_value());
  EXPECT_EQ(status_recovered.GetPayload("a").value(), "\\xFF\\x02\\x03");
}

TEST(PayloadSerialization, PayloadsCorrupted) {
  ::grpc::Status status(
      ::grpc::StatusCode::INVALID_ARGUMENT, "invalid arg message",
      "string that can not be serialized to the GrpcPayloadContainer proto");

  ::tensorflow::Status converted = FromGrpcStatus(status);
  EXPECT_TRUE(converted.GetPayload(kGrpcPayloadsLost).has_value());
}

TEST(GrpcProto, Unparse) {
  CleanupAllRequest proto;
  proto.add_container("hello");
  proto.add_container("world");
  grpc::ByteBuffer buf;
  ASSERT_TRUE(GrpcMaybeUnparseProto(proto, &buf).ok());
  CleanupAllRequest parsed;
  ASSERT_TRUE(parsed.ParseFromString(ToString(buf)));
  ASSERT_EQ(proto.DebugString(), parsed.DebugString());
}

TEST(GrpcProto, UnparseToString) {
  CleanupAllRequest proto;
  proto.add_container("hello");
  proto.add_container("world");
  string str;
  CHECK(proto.SerializeToString(&str));
  grpc::ByteBuffer buf;
  ASSERT_TRUE(GrpcMaybeUnparseProto(str, &buf).ok());
  CleanupAllRequest parsed;
  ASSERT_TRUE(parsed.ParseFromString(ToString(buf)));
  ASSERT_EQ(proto.DebugString(), parsed.DebugString());
}

TEST(GrpcProto, Parse) {
  // Test with serialization broken up into a bunch of slices.
  struct Case {
    int length;
    int slices;
  };
  for (Case c : std::vector<Case>{
           {0, 1},
           {20, 1},
           {100, 1},
           {1 << 20, 1},
           {100, 5},
           {10000, 50},
       }) {
    CleanupAllRequest proto = MakeProto(c.length);
    ::grpc::ByteBuffer src = MakeBuffer(proto.SerializeAsString(), c.slices);
    CleanupAllRequest parsed;
    ASSERT_TRUE(GrpcMaybeParseProto(&src, &parsed))
        << c.length << " " << c.slices;
    ASSERT_EQ(proto.DebugString(), parsed.DebugString());
  }
}

TEST(GrpcProto, ParseFromString) {
  // Test with serialization broken up into a bunch of slices.
  struct Case {
    int length;
    int slices;
  };
  for (Case c : std::vector<Case>{
           {0, 1},
           {20, 1},
           {100, 1},
           {1 << 20, 1},
           {100, 5},
           {10000, 50},
       }) {
    CleanupAllRequest proto = MakeProto(c.length);
    ::grpc::ByteBuffer src = MakeBuffer(proto.SerializeAsString(), c.slices);
    string parsed_str;
    CleanupAllRequest parsed;
    ASSERT_TRUE(GrpcMaybeParseProto(&src, &parsed_str))
        << c.length << " " << c.slices;
    ASSERT_TRUE(parsed.ParseFromString(parsed_str));
    ASSERT_EQ(proto.DebugString(), parsed.DebugString());
  }
}

static void BM_UnparseGrpc(::testing::benchmark::State& state) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSrpcPSgrpc_util_testDTcc mht_3(mht_3_v, 338, "", "./tensorflow/core/distributed_runtime/rpc/grpc_util_test.cc", "BM_UnparseGrpc");

  const int size = state.range(0);

  auto proto = MakeProto(size);
  for (auto s : state) {
    grpc::ByteBuffer buf;
    CHECK(GrpcMaybeUnparseProto(proto, &buf).ok());
  }
}
BENCHMARK(BM_UnparseGrpc)->Arg(1)->Arg(1 << 10)->Arg(1 << 20);

static void BM_UnparseString(::testing::benchmark::State& state) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSrpcPSgrpc_util_testDTcc mht_4(mht_4_v, 352, "", "./tensorflow/core/distributed_runtime/rpc/grpc_util_test.cc", "BM_UnparseString");

  const int size = state.range(0);

  auto proto = MakeProto(size);

  for (auto s : state) {
    string buf;
    proto.SerializeToString(&buf);
  }
}
BENCHMARK(BM_UnparseString)->Arg(1)->Arg(1 << 10)->Arg(1 << 20);

static void BM_ParseGrpc(::testing::benchmark::State& state) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSrpcPSgrpc_util_testDTcc mht_5(mht_5_v, 367, "", "./tensorflow/core/distributed_runtime/rpc/grpc_util_test.cc", "BM_ParseGrpc");

  const int size = state.range(0);
  const int num_slices = state.range(1);

  CleanupAllRequest proto = MakeProto(size);
  auto buf = MakeBuffer(proto.SerializeAsString(), num_slices);

  for (auto s : state) {
    CHECK(GrpcMaybeParseProto(&buf, &proto));
  }
}
BENCHMARK(BM_ParseGrpc)
    ->ArgPair(1, 1)
    ->ArgPair(1 << 10, 1)
    ->ArgPair(1 << 10, 4)
    ->ArgPair(1 << 20, 1)
    ->ArgPair(1 << 20, 4);

static void BM_ParseString(::testing::benchmark::State& state) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSrpcPSgrpc_util_testDTcc mht_6(mht_6_v, 388, "", "./tensorflow/core/distributed_runtime/rpc/grpc_util_test.cc", "BM_ParseString");

  const int size = state.range(0);

  CleanupAllRequest proto = MakeProto(size);
  string serial = proto.SerializeAsString();

  for (auto s : state) {
    CHECK(proto.ParseFromString(serial));
  }
}
BENCHMARK(BM_ParseString)->Arg(1)->Arg(1 << 10)->Arg(1 << 20);

}  // namespace tensorflow
