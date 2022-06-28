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
class MHTracer_DTPStensorflowPScorePSdistributed_runtimePStensor_coding_testDTcc {
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
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePStensor_coding_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSdistributed_runtimePStensor_coding_testDTcc() {
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

#include "tensorflow/core/distributed_runtime/tensor_coding.h"

#include "tensorflow/core/framework/device_attributes.pb.h"
#include "tensorflow/core/framework/device_base.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_testutil.h"
#include "tensorflow/core/lib/gtl/inlined_vector.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/protobuf.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/platform/test_benchmark.h"
#include "tensorflow/core/protobuf/worker.pb.h"
#include "tensorflow/core/public/session_options.h"

namespace tensorflow {

class DummyDevice : public DeviceBase {
 public:
  explicit DummyDevice(Env* env) : DeviceBase(env) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePStensor_coding_testDTcc mht_0(mht_0_v, 204, "", "./tensorflow/core/distributed_runtime/tensor_coding_test.cc", "DummyDevice");

    attr_.set_device_type("CPU");
  }

  const DeviceAttributes& attributes() const override {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePStensor_coding_testDTcc mht_1(mht_1_v, 211, "", "./tensorflow/core/distributed_runtime/tensor_coding_test.cc", "attributes");
 return attr_; }

  Allocator* GetAllocator(AllocatorAttributes attr) override {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePStensor_coding_testDTcc mht_2(mht_2_v, 216, "", "./tensorflow/core/distributed_runtime/tensor_coding_test.cc", "GetAllocator");

    return cpu_allocator();
  }

 private:
  DeviceAttributes attr_;
};

class StringSource : public TensorResponse::Source {
 public:
  explicit StringSource(const string* s, int block_size)
      : s_(s), stream_(nullptr), block_size_(block_size) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePStensor_coding_testDTcc mht_3(mht_3_v, 230, "", "./tensorflow/core/distributed_runtime/tensor_coding_test.cc", "StringSource");
}
  ~StringSource() override {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePStensor_coding_testDTcc mht_4(mht_4_v, 234, "", "./tensorflow/core/distributed_runtime/tensor_coding_test.cc", "~StringSource");
 DeleteStream(); }

  protobuf::io::ZeroCopyInputStream* contents() override {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePStensor_coding_testDTcc mht_5(mht_5_v, 239, "", "./tensorflow/core/distributed_runtime/tensor_coding_test.cc", "contents");

    DeleteStream();
    stream_ = new (&space_)
        protobuf::io::ArrayInputStream(s_->data(), s_->size(), block_size_);
    return stream_;
  }

  void DeleteStream() {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePStensor_coding_testDTcc mht_6(mht_6_v, 249, "", "./tensorflow/core/distributed_runtime/tensor_coding_test.cc", "DeleteStream");

    if (stream_) {
      stream_->~ArrayInputStream();
    }
  }

 private:
  const string* s_;
  protobuf::io::ArrayInputStream* stream_;
  char space_[sizeof(protobuf::io::ArrayInputStream)];
  int block_size_;
};

class TensorResponseTest : public ::testing::Test {
 public:
  void Validate(const Tensor& src, bool is_dead, bool use_tensor_content) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePStensor_coding_testDTcc mht_7(mht_7_v, 267, "", "./tensorflow/core/distributed_runtime/tensor_coding_test.cc", "Validate");

    RecvTensorResponse proto;
    proto.set_is_dead(is_dead);
    proto.set_send_start_micros(123456);
    if (use_tensor_content) {
      src.AsProtoTensorContent(proto.mutable_tensor());
    } else {
      src.AsProtoField(proto.mutable_tensor());
    }
    string encoded;
    proto.AppendToString(&encoded);

    StringSource source(&encoded, 1024);

    TensorResponse response;
    DummyDevice cpu_device(Env::Default());
    response.InitAlloc(&cpu_device, AllocatorAttributes());
    for (int i = 0; i < 2; i++) {  // Twice so we exercise reuse of "response"
      Status s = response.ParseFrom(&source);
      EXPECT_TRUE(s.ok());

      const RecvTensorResponse& meta = response.metadata();
      EXPECT_EQ(meta.is_dead(), is_dead);
      EXPECT_EQ(meta.send_start_micros(), 123456);

      const Tensor& result = response.tensor();
      EXPECT_EQ(result.dtype(), src.dtype());
      EXPECT_EQ(result.shape().DebugString(), src.shape().DebugString());
      EXPECT_EQ(result.DebugString(), src.DebugString());
    }
  }

  template <typename T>
  void DoTest(DataType dt) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePStensor_coding_testDTcc mht_8(mht_8_v, 303, "", "./tensorflow/core/distributed_runtime/tensor_coding_test.cc", "DoTest");

    gtl::InlinedVector<T, 4> v;
    LOG(ERROR) << "DT: " << static_cast<int>(dt);
    for (int elems = 0; elems <= 10000; elems++) {
      if (elems < 100 || (elems % 1000 == 0)) {
        Tensor a(dt, TensorShape({1, static_cast<int64_t>(v.size())}));
        test::FillValues<T>(&a, v);
        Validate(a, (elems == 0), true);
      }
      v.push_back(static_cast<T>(elems));
    }
  }
  void DoTestForStrings(DataType dt) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePStensor_coding_testDTcc mht_9(mht_9_v, 318, "", "./tensorflow/core/distributed_runtime/tensor_coding_test.cc", "DoTestForStrings");

    gtl::InlinedVector<tstring, 4> v;
    LOG(ERROR) << "DT: string";
    for (int elems = 0; elems <= 10000; elems++) {
      if (elems < 100 || (elems % 1000 == 0)) {
        Tensor a(dt, TensorShape({1, static_cast<int64_t>(v.size())}));
        test::FillValues<tstring>(&a, v);
        Validate(a, (elems == 0), true);
      }
      v.push_back(strings::StrCat("This is string ", elems));
    }
  }
};

TEST_F(TensorResponseTest, Simple) {
  DoTest<float>(DT_FLOAT);
  DoTest<double>(DT_DOUBLE);
  DoTest<int32>(DT_INT32);
  DoTest<uint16>(DT_UINT16);
  DoTest<uint8>(DT_UINT8);
  DoTest<int16>(DT_INT16);
  DoTest<int8>(DT_INT8);
  DoTest<complex64>(DT_COMPLEX64);
  DoTest<complex128>(DT_COMPLEX128);
  DoTest<int64_t>(DT_INT64);
  DoTest<bool>(DT_BOOL);
  DoTest<qint8>(DT_QINT8);
  DoTest<quint8>(DT_QUINT8);
  DoTest<qint16>(DT_QINT16);
  DoTest<quint16>(DT_QUINT16);
  DoTest<qint32>(DT_QINT32);
  DoTest<bfloat16>(DT_BFLOAT16);
  DoTest<Eigen::half>(DT_HALF);
}

TEST_F(TensorResponseTest, StringTensor) { DoTestForStrings(DT_STRING); }

string MakeFloatTensorTestCase(int num_elems) {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePStensor_coding_testDTcc mht_10(mht_10_v, 358, "", "./tensorflow/core/distributed_runtime/tensor_coding_test.cc", "MakeFloatTensorTestCase");

  std::vector<int8> v(num_elems);
  for (int i = 0; i < num_elems; i++) {
    v[i] = i % 10;
  }
  Tensor src(DT_INT8, TensorShape({1, static_cast<int64_t>(v.size())}));
  test::FillValues<int8>(&src, v);

  RecvTensorResponse proto;
  proto.set_is_dead(false);
  proto.set_send_start_micros(123456);
  src.AsProtoTensorContent(proto.mutable_tensor());
  string encoded;
  proto.AppendToString(&encoded);
  return encoded;
}

static void BM_TensorResponse(::testing::benchmark::State& state) {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePStensor_coding_testDTcc mht_11(mht_11_v, 378, "", "./tensorflow/core/distributed_runtime/tensor_coding_test.cc", "BM_TensorResponse");

  const int arg = state.range(0);

  string encoded = MakeFloatTensorTestCase(arg);
  DummyDevice cpu_device(Env::Default());
  size_t bytes = 0;
  for (auto i : state) {
    TensorResponse response;
    response.InitAlloc(&cpu_device, AllocatorAttributes());
    StringSource source(&encoded, -1);
    Status s = response.ParseFrom(&source);
    bytes = response.tensor().TotalBytes();
  }
  state.SetLabel(strings::StrCat("Bytes: ", bytes));
}
BENCHMARK(BM_TensorResponse)->Arg(0)->Arg(1000)->Arg(100000);

static void BM_TensorViaTensorProto(::testing::benchmark::State& state) {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePStensor_coding_testDTcc mht_12(mht_12_v, 398, "", "./tensorflow/core/distributed_runtime/tensor_coding_test.cc", "BM_TensorViaTensorProto");

  const int arg = state.range(0);

  std::string encoded = MakeFloatTensorTestCase(arg);
  size_t bytes = 0;
  for (auto s : state) {
    RecvTensorResponse r;
    r.ParseFromString(encoded);
    Tensor t;
    CHECK(t.FromProto(r.tensor()));
    bytes = t.TotalBytes();
  }
  state.SetLabel(strings::StrCat("Bytes: ", bytes));
}
BENCHMARK(BM_TensorViaTensorProto)->Arg(0)->Arg(1000)->Arg(100000);

}  // namespace tensorflow
