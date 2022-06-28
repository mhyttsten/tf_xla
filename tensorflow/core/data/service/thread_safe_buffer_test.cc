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
class MHTracer_DTPStensorflowPScorePSdataPSservicePSthread_safe_buffer_testDTcc {
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
   MHTracer_DTPStensorflowPScorePSdataPSservicePSthread_safe_buffer_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSdataPSservicePSthread_safe_buffer_testDTcc() {
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
#include "tensorflow/core/data/service/thread_safe_buffer.h"

#include <memory>
#include <tuple>
#include <vector>

#include "absl/strings/str_cat.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_testutil.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/status_matchers.h"
#include "tensorflow/core/platform/statusor.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/protobuf/error_codes.pb.h"

namespace tensorflow {
namespace data {
namespace {

using ::tensorflow::testing::IsOk;
using ::tensorflow::testing::StatusIs;
using ::testing::UnorderedElementsAreArray;

class ThreadSafeBufferTest
    : public ::testing::Test,
      public ::testing::WithParamInterface<std::tuple<size_t, size_t>> {
 protected:
  size_t GetBufferSize() const {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSdataPSservicePSthread_safe_buffer_testDTcc mht_0(mht_0_v, 214, "", "./tensorflow/core/data/service/thread_safe_buffer_test.cc", "GetBufferSize");
 return std::get<0>(GetParam()); }
  size_t GetNumOfElements() const {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSdataPSservicePSthread_safe_buffer_testDTcc mht_1(mht_1_v, 218, "", "./tensorflow/core/data/service/thread_safe_buffer_test.cc", "GetNumOfElements");
 return std::get<1>(GetParam()); }
};

std::vector<int> GetRange(const size_t range) {
  std::vector<int> result;
  for (int i = 0; i < range; ++i) {
    result.push_back(i);
  }
  return result;
}

INSTANTIATE_TEST_SUITE_P(VaryingBufferAndInputSizes, ThreadSafeBufferTest,
                         ::testing::Values(std::make_tuple(1, 2),
                                           std::make_tuple(2, 10),
                                           std::make_tuple(10, 2)));

TEST_P(ThreadSafeBufferTest, OneReaderAndOneWriter) {
  ThreadSafeBuffer<int> buffer(GetBufferSize());
  auto thread = absl::WrapUnique(Env::Default()->StartThread(
      /*thread_options=*/{}, /*name=*/"writer_thread", [this, &buffer]() {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSdataPSservicePSthread_safe_buffer_testDTcc mht_2(mht_2_v, 240, "", "./tensorflow/core/data/service/thread_safe_buffer_test.cc", "lambda");

        for (int i = 0; i < GetNumOfElements(); ++i) {
          ASSERT_THAT(buffer.Push(i), IsOk());
        }
      }));

  for (size_t i = 0; i < GetNumOfElements(); ++i) {
    TF_ASSERT_OK_AND_ASSIGN(int next, buffer.Pop());
    EXPECT_EQ(next, i);
  }
}

TEST_P(ThreadSafeBufferTest, OneReaderAndMultipleWriters) {
  ThreadSafeBuffer<int> buffer(GetBufferSize());
  std::vector<std::unique_ptr<Thread>> threads;
  for (int i = 0; i < GetNumOfElements(); ++i) {
    threads.push_back(absl::WrapUnique(Env::Default()->StartThread(
        /*thread_options=*/{}, /*name=*/absl::StrCat("writer_thread_", i),
        [&buffer, i] { ASSERT_THAT(buffer.Push(i), IsOk()); })));
  }

  std::vector<int> results;
  for (int i = 0; i < GetNumOfElements(); ++i) {
    TF_ASSERT_OK_AND_ASSIGN(int next, buffer.Pop());
    results.push_back(next);
  }
  EXPECT_THAT(results, UnorderedElementsAreArray(GetRange(GetNumOfElements())));
}

TEST_P(ThreadSafeBufferTest, MultipleReadersAndOneWriter) {
  ThreadSafeBuffer<int> buffer(GetBufferSize());
  mutex mu;
  std::vector<int> results;

  std::vector<std::unique_ptr<Thread>> threads;
  for (int i = 0; i < GetNumOfElements(); ++i) {
    threads.push_back(absl::WrapUnique(Env::Default()->StartThread(
        /*thread_options=*/{}, /*name=*/absl::StrCat("reader_thread_", i),
        [&buffer, &mu, &results]() {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSdataPSservicePSthread_safe_buffer_testDTcc mht_3(mht_3_v, 281, "", "./tensorflow/core/data/service/thread_safe_buffer_test.cc", "lambda");

          TF_ASSERT_OK_AND_ASSIGN(int next, buffer.Pop());
          mutex_lock l(mu);
          results.push_back(next);
        })));
  }

  for (int i = 0; i < GetNumOfElements(); ++i) {
    ASSERT_THAT(buffer.Push(i), IsOk());
  }

  // Wait for all threads to complete.
  threads.clear();
  EXPECT_THAT(results, UnorderedElementsAreArray(GetRange(GetNumOfElements())));
}

TEST_P(ThreadSafeBufferTest, MultipleReadersAndWriters) {
  ThreadSafeBuffer<int> buffer(GetBufferSize());
  mutex mu;
  std::vector<int> results;

  std::vector<std::unique_ptr<Thread>> threads;
  for (int i = 0; i < GetNumOfElements(); ++i) {
    threads.push_back(absl::WrapUnique(Env::Default()->StartThread(
        /*thread_options=*/{}, /*name=*/absl::StrCat("reader_thread_", i),
        [&buffer, &mu, &results]() {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSdataPSservicePSthread_safe_buffer_testDTcc mht_4(mht_4_v, 309, "", "./tensorflow/core/data/service/thread_safe_buffer_test.cc", "lambda");

          TF_ASSERT_OK_AND_ASSIGN(int next, buffer.Pop());
          mutex_lock l(mu);
          results.push_back(next);
        })));
  }

  for (int i = 0; i < GetNumOfElements(); ++i) {
    threads.push_back(absl::WrapUnique(Env::Default()->StartThread(
        /*thread_options=*/{}, /*name=*/absl::StrCat("writer_thread_", i),
        [&buffer, i]() {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSdataPSservicePSthread_safe_buffer_testDTcc mht_5(mht_5_v, 322, "", "./tensorflow/core/data/service/thread_safe_buffer_test.cc", "lambda");
 ASSERT_THAT(buffer.Push(i), IsOk()); })));
  }

  // Wait for all threads to complete.
  threads.clear();
  EXPECT_THAT(results, UnorderedElementsAreArray(GetRange(GetNumOfElements())));
}

TEST_P(ThreadSafeBufferTest, BlockReaderWhenBufferIsEmpty) {
  ThreadSafeBuffer<Tensor> buffer(GetBufferSize());

  // The buffer is empty, blocking the next `Pop` call.
  auto thread = absl::WrapUnique(Env::Default()->StartThread(
      /*thread_options=*/{}, /*name=*/"reader_thread", [&buffer]() {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePSdataPSservicePSthread_safe_buffer_testDTcc mht_6(mht_6_v, 338, "", "./tensorflow/core/data/service/thread_safe_buffer_test.cc", "lambda");

        TF_ASSERT_OK_AND_ASSIGN(Tensor tensor, buffer.Pop());
        test::ExpectEqual(tensor, Tensor("Test tensor"));
      }));

  // Pushing an element unblocks the `Pop` call.
  Env::Default()->SleepForMicroseconds(10000);
  ASSERT_THAT(buffer.Push(Tensor("Test tensor")), IsOk());
}

TEST_P(ThreadSafeBufferTest, BlockWriterWhenBufferIsFull) {
  ThreadSafeBuffer<Tensor> buffer(GetBufferSize());
  // Fills the buffer to block the next `Push` call.
  for (int i = 0; i < GetBufferSize(); ++i) {
    ASSERT_THAT(buffer.Push(Tensor("Test tensor")), IsOk());
  }

  uint64 push_time = 0;
  auto thread = absl::WrapUnique(Env::Default()->StartThread(
      /*thread_options=*/{}, /*name=*/"writer_thread", [&buffer, &push_time]() {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePSdataPSservicePSthread_safe_buffer_testDTcc mht_7(mht_7_v, 360, "", "./tensorflow/core/data/service/thread_safe_buffer_test.cc", "lambda");

        ASSERT_THAT(buffer.Push(Tensor("Test tensor")), IsOk());
        push_time = Env::Default()->NowMicros();
      }));

  // Popping an element unblocks the `Push` call.
  Env::Default()->SleepForMicroseconds(10000);
  uint64 pop_time = Env::Default()->NowMicros();
  ASSERT_THAT(buffer.Pop(), IsOk());
  thread.reset();
  EXPECT_LE(pop_time, push_time);
}

TEST_P(ThreadSafeBufferTest, CancelReaders) {
  ThreadSafeBuffer<int> buffer(GetBufferSize());
  std::vector<std::unique_ptr<Thread>> threads;

  for (int i = 0; i < GetNumOfElements(); ++i) {
    threads.push_back(absl::WrapUnique(Env::Default()->StartThread(
        /*thread_options=*/{}, /*name=*/absl::StrCat("reader_thread_", i),
        [&buffer]() {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScorePSdataPSservicePSthread_safe_buffer_testDTcc mht_8(mht_8_v, 383, "", "./tensorflow/core/data/service/thread_safe_buffer_test.cc", "lambda");
 EXPECT_THAT(buffer.Pop(), StatusIs(error::ABORTED)); })));
  }
  buffer.Cancel(errors::Aborted("Aborted"));
}

TEST_P(ThreadSafeBufferTest, CancelWriters) {
  ThreadSafeBuffer<Tensor> buffer(GetBufferSize());
  // Fills the buffer so subsequent pushes are all cancelled.
  for (int i = 0; i < GetBufferSize(); ++i) {
    ASSERT_THAT(buffer.Push(Tensor("Test tensor")), IsOk());
  }

  std::vector<std::unique_ptr<Thread>> threads;
  for (int i = 0; i < GetNumOfElements(); ++i) {
    threads.push_back(absl::WrapUnique(Env::Default()->StartThread(
        /*thread_options=*/{}, /*name=*/absl::StrCat("writer_thread_", i),
        [&buffer]() {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScorePSdataPSservicePSthread_safe_buffer_testDTcc mht_9(mht_9_v, 402, "", "./tensorflow/core/data/service/thread_safe_buffer_test.cc", "lambda");

          for (int i = 0; i < 100; ++i) {
            EXPECT_THAT(buffer.Push(Tensor("Test tensor")),
                        StatusIs(error::CANCELLED));
          }
        })));
  }
  buffer.Cancel(errors::Cancelled("Cancelled"));
}

TEST_P(ThreadSafeBufferTest, CancelMultipleTimes) {
  ThreadSafeBuffer<Tensor> buffer(GetBufferSize());
  buffer.Cancel(errors::Unknown("Unknown"));
  EXPECT_THAT(buffer.Push(Tensor("Test tensor")), StatusIs(error::UNKNOWN));
  buffer.Cancel(errors::DeadlineExceeded("Deadline exceeded"));
  EXPECT_THAT(buffer.Pop(), StatusIs(error::DEADLINE_EXCEEDED));
  buffer.Cancel(errors::ResourceExhausted("Resource exhausted"));
  EXPECT_THAT(buffer.Push(Tensor("Test tensor")),
              StatusIs(error::RESOURCE_EXHAUSTED));
}

}  // namespace
}  // namespace data
}  // namespace tensorflow
