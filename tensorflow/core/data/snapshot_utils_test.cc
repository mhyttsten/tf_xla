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
class MHTracer_DTPStensorflowPScorePSdataPSsnapshot_utils_testDTcc {
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
   MHTracer_DTPStensorflowPScorePSdataPSsnapshot_utils_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSdataPSsnapshot_utils_testDTcc() {
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

#include "tensorflow/core/data/snapshot_utils.h"

#include "tensorflow/core/framework/tensor.pb.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/lib/io/compression.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/platform/test_benchmark.h"

namespace tensorflow {
namespace data {
namespace snapshot_util {
namespace {

void GenerateTensorVector(tensorflow::DataTypeVector& dtypes,
                          std::vector<Tensor>& tensors) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSdataPSsnapshot_utils_testDTcc mht_0(mht_0_v, 201, "", "./tensorflow/core/data/snapshot_utils_test.cc", "GenerateTensorVector");

  std::string tensor_data(1024, 'a');
  for (int i = 0; i < 10; ++i) {
    Tensor t(tensor_data.data());
    dtypes.push_back(t.dtype());
    tensors.push_back(t);
  }
}

void SnapshotRoundTrip(std::string compression_type, int version) {
   std::vector<std::string> mht_1_v;
   mht_1_v.push_back("compression_type: \"" + compression_type + "\"");
   MHTracer_DTPStensorflowPScorePSdataPSsnapshot_utils_testDTcc mht_1(mht_1_v, 214, "", "./tensorflow/core/data/snapshot_utils_test.cc", "SnapshotRoundTrip");

  // Generate ground-truth tensors for writing and reading.
  std::vector<Tensor> tensors;
  tensorflow::DataTypeVector dtypes;
  GenerateTensorVector(dtypes, tensors);

  std::string filename;
  EXPECT_TRUE(Env::Default()->LocalTempFilename(&filename));

  std::unique_ptr<Writer> writer;
  TF_ASSERT_OK(Writer::Create(tensorflow::Env::Default(), filename,
                              compression_type, version, dtypes, &writer));

  for (int i = 0; i < 100; ++i) {
    TF_ASSERT_OK(writer->WriteTensors(tensors));
  }
  TF_ASSERT_OK(writer->Close());

  std::unique_ptr<Reader> reader;
  TF_ASSERT_OK(Reader::Create(Env::Default(), filename, compression_type,
                              version, dtypes, &reader));

  for (int i = 0; i < 100; ++i) {
    std::vector<Tensor> read_tensors;
    TF_ASSERT_OK(reader->ReadTensors(&read_tensors));
    EXPECT_EQ(tensors.size(), read_tensors.size());
    for (int j = 0; j < read_tensors.size(); ++j) {
      TensorProto proto;
      TensorProto read_proto;

      tensors[j].AsProtoTensorContent(&proto);
      read_tensors[j].AsProtoTensorContent(&read_proto);

      std::string proto_serialized, read_proto_serialized;
      proto.AppendToString(&proto_serialized);
      read_proto.AppendToString(&read_proto_serialized);
      EXPECT_EQ(proto_serialized, read_proto_serialized);
    }
  }

  TF_ASSERT_OK(Env::Default()->DeleteFile(filename));
}

TEST(SnapshotUtilTest, CombinationRoundTripTest) {
  SnapshotRoundTrip(io::compression::kNone, 1);
  SnapshotRoundTrip(io::compression::kGzip, 1);
  SnapshotRoundTrip(io::compression::kSnappy, 1);

  SnapshotRoundTrip(io::compression::kNone, 2);
  SnapshotRoundTrip(io::compression::kGzip, 2);
  SnapshotRoundTrip(io::compression::kSnappy, 2);
}

void SnapshotReaderBenchmarkLoop(::testing::benchmark::State& state,
                                 std::string compression_type, int version) {
   std::vector<std::string> mht_2_v;
   mht_2_v.push_back("compression_type: \"" + compression_type + "\"");
   MHTracer_DTPStensorflowPScorePSdataPSsnapshot_utils_testDTcc mht_2(mht_2_v, 272, "", "./tensorflow/core/data/snapshot_utils_test.cc", "SnapshotReaderBenchmarkLoop");

  tensorflow::DataTypeVector dtypes;
  std::vector<Tensor> tensors;
  GenerateTensorVector(dtypes, tensors);

  std::string filename;
  EXPECT_TRUE(Env::Default()->LocalTempFilename(&filename));

  std::unique_ptr<Writer> writer;
  TF_ASSERT_OK(Writer::Create(tensorflow::Env::Default(), filename,
                              compression_type, version, dtypes, &writer));

  for (auto s : state) {
    writer->WriteTensors(tensors).IgnoreError();
  }
  TF_ASSERT_OK(writer->Close());

  std::unique_ptr<Reader> reader;
  TF_ASSERT_OK(Reader::Create(Env::Default(), filename, compression_type,
                              version, dtypes, &reader));

  for (auto s : state) {
    std::vector<Tensor> read_tensors;
    reader->ReadTensors(&read_tensors).IgnoreError();
  }

  TF_ASSERT_OK(Env::Default()->DeleteFile(filename));
}

void SnapshotCustomReaderNoneBenchmark(::testing::benchmark::State& state) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSdataPSsnapshot_utils_testDTcc mht_3(mht_3_v, 304, "", "./tensorflow/core/data/snapshot_utils_test.cc", "SnapshotCustomReaderNoneBenchmark");

  SnapshotReaderBenchmarkLoop(state, io::compression::kNone, 1);
}

void SnapshotCustomReaderGzipBenchmark(::testing::benchmark::State& state) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSdataPSsnapshot_utils_testDTcc mht_4(mht_4_v, 311, "", "./tensorflow/core/data/snapshot_utils_test.cc", "SnapshotCustomReaderGzipBenchmark");

  SnapshotReaderBenchmarkLoop(state, io::compression::kGzip, 1);
}

void SnapshotCustomReaderSnappyBenchmark(::testing::benchmark::State& state) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSdataPSsnapshot_utils_testDTcc mht_5(mht_5_v, 318, "", "./tensorflow/core/data/snapshot_utils_test.cc", "SnapshotCustomReaderSnappyBenchmark");

  SnapshotReaderBenchmarkLoop(state, io::compression::kSnappy, 1);
}

void SnapshotTFRecordReaderNoneBenchmark(::testing::benchmark::State& state) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePSdataPSsnapshot_utils_testDTcc mht_6(mht_6_v, 325, "", "./tensorflow/core/data/snapshot_utils_test.cc", "SnapshotTFRecordReaderNoneBenchmark");

  SnapshotReaderBenchmarkLoop(state, io::compression::kNone, 2);
}

void SnapshotTFRecordReaderGzipBenchmark(::testing::benchmark::State& state) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePSdataPSsnapshot_utils_testDTcc mht_7(mht_7_v, 332, "", "./tensorflow/core/data/snapshot_utils_test.cc", "SnapshotTFRecordReaderGzipBenchmark");

  SnapshotReaderBenchmarkLoop(state, io::compression::kGzip, 2);
}

BENCHMARK(SnapshotCustomReaderNoneBenchmark);
BENCHMARK(SnapshotCustomReaderGzipBenchmark);
BENCHMARK(SnapshotCustomReaderSnappyBenchmark);
BENCHMARK(SnapshotTFRecordReaderNoneBenchmark);
BENCHMARK(SnapshotTFRecordReaderGzipBenchmark);

void SnapshotWriterBenchmarkLoop(::testing::benchmark::State& state,
                                 std::string compression_type, int version) {
   std::vector<std::string> mht_8_v;
   mht_8_v.push_back("compression_type: \"" + compression_type + "\"");
   MHTracer_DTPStensorflowPScorePSdataPSsnapshot_utils_testDTcc mht_8(mht_8_v, 347, "", "./tensorflow/core/data/snapshot_utils_test.cc", "SnapshotWriterBenchmarkLoop");

  tensorflow::DataTypeVector dtypes;
  std::vector<Tensor> tensors;
  GenerateTensorVector(dtypes, tensors);

  std::string filename;
  EXPECT_TRUE(Env::Default()->LocalTempFilename(&filename));

  std::unique_ptr<Writer> writer;
  TF_ASSERT_OK(Writer::Create(tensorflow::Env::Default(), filename,
                              compression_type, version, dtypes, &writer));

  for (auto s : state) {
    writer->WriteTensors(tensors).IgnoreError();
  }
  writer->Close().IgnoreError();

  TF_ASSERT_OK(Env::Default()->DeleteFile(filename));
}

void SnapshotCustomWriterNoneBenchmark(::testing::benchmark::State& state) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScorePSdataPSsnapshot_utils_testDTcc mht_9(mht_9_v, 370, "", "./tensorflow/core/data/snapshot_utils_test.cc", "SnapshotCustomWriterNoneBenchmark");

  SnapshotWriterBenchmarkLoop(state, io::compression::kNone, 1);
}

void SnapshotCustomWriterGzipBenchmark(::testing::benchmark::State& state) {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScorePSdataPSsnapshot_utils_testDTcc mht_10(mht_10_v, 377, "", "./tensorflow/core/data/snapshot_utils_test.cc", "SnapshotCustomWriterGzipBenchmark");

  SnapshotWriterBenchmarkLoop(state, io::compression::kGzip, 1);
}

void SnapshotCustomWriterSnappyBenchmark(::testing::benchmark::State& state) {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScorePSdataPSsnapshot_utils_testDTcc mht_11(mht_11_v, 384, "", "./tensorflow/core/data/snapshot_utils_test.cc", "SnapshotCustomWriterSnappyBenchmark");

  SnapshotWriterBenchmarkLoop(state, io::compression::kSnappy, 1);
}

void SnapshotTFRecordWriterNoneBenchmark(::testing::benchmark::State& state) {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScorePSdataPSsnapshot_utils_testDTcc mht_12(mht_12_v, 391, "", "./tensorflow/core/data/snapshot_utils_test.cc", "SnapshotTFRecordWriterNoneBenchmark");

  SnapshotWriterBenchmarkLoop(state, io::compression::kNone, 2);
}

void SnapshotTFRecordWriterGzipBenchmark(::testing::benchmark::State& state) {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPScorePSdataPSsnapshot_utils_testDTcc mht_13(mht_13_v, 398, "", "./tensorflow/core/data/snapshot_utils_test.cc", "SnapshotTFRecordWriterGzipBenchmark");

  SnapshotWriterBenchmarkLoop(state, io::compression::kGzip, 2);
}

void SnapshotTFRecordWriterSnappyBenchmark(::testing::benchmark::State& state) {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPScorePSdataPSsnapshot_utils_testDTcc mht_14(mht_14_v, 405, "", "./tensorflow/core/data/snapshot_utils_test.cc", "SnapshotTFRecordWriterSnappyBenchmark");

  SnapshotWriterBenchmarkLoop(state, io::compression::kSnappy, 2);
}

BENCHMARK(SnapshotCustomWriterNoneBenchmark);
BENCHMARK(SnapshotCustomWriterGzipBenchmark);
BENCHMARK(SnapshotCustomWriterSnappyBenchmark);
BENCHMARK(SnapshotTFRecordWriterNoneBenchmark);
BENCHMARK(SnapshotTFRecordWriterGzipBenchmark);
BENCHMARK(SnapshotTFRecordWriterSnappyBenchmark);

}  // namespace
}  // namespace snapshot_util
}  // namespace data
}  // namespace tensorflow
