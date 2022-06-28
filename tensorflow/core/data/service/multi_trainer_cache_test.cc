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
class MHTracer_DTPStensorflowPScorePSdataPSservicePSmulti_trainer_cache_testDTcc {
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
   MHTracer_DTPStensorflowPScorePSdataPSservicePSmulti_trainer_cache_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSdataPSservicePSmulti_trainer_cache_testDTcc() {
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

/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/core/data/service/multi_trainer_cache.h"

#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/memory/memory.h"
#include "absl/strings/str_cat.h"
#include "absl/time/time.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_testutil.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/random.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/platform/status_matchers.h"
#include "tensorflow/core/platform/statusor.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/protobuf/error_codes.pb.h"

namespace tensorflow {
namespace data {
namespace {

using ::tensorflow::testing::IsOkAndHolds;
using ::tensorflow::testing::StatusIs;
using ::testing::Gt;
using ::testing::HasSubstr;
using ::testing::Pointee;
using ::testing::UnorderedElementsAreArray;

class InfiniteRange : public CachableSequence<int64_t> {
 public:
  StatusOr<int64_t> GetNext() override { return next_++; }
  size_t GetElementSizeBytes(const int64_t& element) const override {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSdataPSservicePSmulti_trainer_cache_testDTcc mht_0(mht_0_v, 223, "", "./tensorflow/core/data/service/multi_trainer_cache_test.cc", "GetElementSizeBytes");

    return sizeof(element);
  }

 private:
  // No need to guard this variable because only one thread can write the cache.
  int64_t next_ = 0;
};

class TensorDataset : public CachableSequence<Tensor> {
 public:
  StatusOr<Tensor> GetNext() override { return Tensor("Test Tensor"); }
  size_t GetElementSizeBytes(const Tensor& element) const override {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSdataPSservicePSmulti_trainer_cache_testDTcc mht_1(mht_1_v, 238, "", "./tensorflow/core/data/service/multi_trainer_cache_test.cc", "GetElementSizeBytes");

    return element.TotalBytes();
  }
};

class SlowDataset : public CachableSequence<Tensor> {
 public:
  explicit SlowDataset(absl::Duration delay) : delay_(delay) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSdataPSservicePSmulti_trainer_cache_testDTcc mht_2(mht_2_v, 248, "", "./tensorflow/core/data/service/multi_trainer_cache_test.cc", "SlowDataset");
}

  StatusOr<Tensor> GetNext() override {
    Env::Default()->SleepForMicroseconds(absl::ToInt64Microseconds(delay_));
    return Tensor("Test Tensor");
  }

  size_t GetElementSizeBytes(const Tensor& element) const override {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSdataPSservicePSmulti_trainer_cache_testDTcc mht_3(mht_3_v, 258, "", "./tensorflow/core/data/service/multi_trainer_cache_test.cc", "GetElementSizeBytes");

    return element.TotalBytes();
  }

 private:
  absl::Duration delay_;
};

template <class T>
class ElementOrErrorDataset : public CachableSequence<T> {
 public:
  explicit ElementOrErrorDataset(const std::vector<StatusOr<T>>& elements)
      : elements_(elements) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSdataPSservicePSmulti_trainer_cache_testDTcc mht_4(mht_4_v, 273, "", "./tensorflow/core/data/service/multi_trainer_cache_test.cc", "ElementOrErrorDataset");
}

  StatusOr<T> GetNext() override {
    if (next_ >= elements_.size()) {
      return errors::OutOfRange("Out of range.");
    }

    return elements_[next_++];
  }

  size_t GetElementSizeBytes(const T& element) const override {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSdataPSservicePSmulti_trainer_cache_testDTcc mht_5(mht_5_v, 286, "", "./tensorflow/core/data/service/multi_trainer_cache_test.cc", "GetElementSizeBytes");

    return sizeof(element);
  }

 private:
  const std::vector<StatusOr<T>> elements_;
  int64_t next_ = 0;
};

template <>
size_t ElementOrErrorDataset<std::string>::GetElementSizeBytes(
    const std::string& element) const {
   std::vector<std::string> mht_6_v;
   mht_6_v.push_back("element: \"" + element + "\"");
   MHTracer_DTPStensorflowPScorePSdataPSservicePSmulti_trainer_cache_testDTcc mht_6(mht_6_v, 301, "", "./tensorflow/core/data/service/multi_trainer_cache_test.cc", "ElementOrErrorDataset<std::string>::GetElementSizeBytes");

  return element.size();
}

template <>
size_t ElementOrErrorDataset<Tensor>::GetElementSizeBytes(
    const Tensor& element) const {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePSdataPSservicePSmulti_trainer_cache_testDTcc mht_7(mht_7_v, 310, "", "./tensorflow/core/data/service/multi_trainer_cache_test.cc", "ElementOrErrorDataset<Tensor>::GetElementSizeBytes");

  return element.TotalBytes();
}

std::vector<int64_t> GetRange(const size_t range) {
  std::vector<int64_t> result;
  for (int64_t i = 0; i < range; ++i) {
    result.push_back(i);
  }
  return result;
}

bool SequenceIsIncreasing(const std::vector<int64_t> sequence) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScorePSdataPSservicePSmulti_trainer_cache_testDTcc mht_8(mht_8_v, 325, "", "./tensorflow/core/data/service/multi_trainer_cache_test.cc", "SequenceIsIncreasing");

  for (int i = 1; i < sequence.size(); ++i) {
    if (sequence[i - 1] > sequence[i - 1]) {
      return false;
    }
  }
  return true;
}

TEST(MultiTrainerCacheTest, GetFromOneTrainer) {
  const size_t num_elements = 10;
  MultiTrainerCache<int64_t> cache(
      /*max_cache_size_bytes=*/1024, absl::make_unique<InfiniteRange>());
  for (size_t i = 0; i < num_elements; ++i) {
    EXPECT_THAT(cache.Get("Trainer ID"), IsOkAndHolds(Pointee(i)));
  }
}

TEST(MultiTrainerCacheTest, GetFromMultipleTrainers) {
  const size_t num_elements = 10;
  const size_t num_trainers = 10;

  MultiTrainerCache<int64_t> cache(
      /*max_cache_size_bytes=*/1024, absl::make_unique<InfiniteRange>());
  for (size_t i = 0; i < num_elements; ++i) {
    // All the readers get the same element in one step.
    for (size_t j = 0; j < num_trainers; ++j) {
      const std::string trainer_id = absl::StrCat("Trainer ", j);
      EXPECT_THAT(cache.Get(trainer_id), IsOkAndHolds(Pointee(i)));
    }
  }
}

TEST(MultiTrainerCacheTest, SlowTrainersSkipData) {
  MultiTrainerCache<int64_t> cache(
      /*max_cache_size_bytes=*/5 * sizeof(int64_t),
      absl::make_unique<InfiniteRange>());
  EXPECT_THAT(cache.Get("Fast trainer 1"), IsOkAndHolds(Pointee(0)));
  EXPECT_THAT(cache.Get("Fast trainer 2"), IsOkAndHolds(Pointee(0)));
  EXPECT_THAT(cache.Get("Slow trainer 1"), IsOkAndHolds(Pointee(0)));
  EXPECT_THAT(cache.Get("Slow trainer 2"), IsOkAndHolds(Pointee(0)));

  for (int i = 1; i < 20; ++i) {
    EXPECT_THAT(cache.Get("Fast trainer 1"), IsOkAndHolds(Pointee(i)));
    EXPECT_THAT(cache.Get("Fast trainer 2"), IsOkAndHolds(Pointee(i)));
  }

  // When 19 is cached, 14 must have been discarded.
  EXPECT_THAT(cache.Get("Slow trainer 1"), IsOkAndHolds(Pointee(Gt(14))));
  EXPECT_THAT(cache.Get("Slow trainer 2"), IsOkAndHolds(Pointee(Gt(14))));

  for (int i = 20; i < 100; ++i) {
    EXPECT_THAT(cache.Get("Fast trainer 1"), IsOkAndHolds(Pointee(i)));
    EXPECT_THAT(cache.Get("Fast trainer 2"), IsOkAndHolds(Pointee(i)));
  }

  // When 99 is cached, 94 must have been discarded.
  EXPECT_THAT(cache.Get("Slow trainer 1"), IsOkAndHolds(Pointee(Gt(94))));
  EXPECT_THAT(cache.Get("Slow trainer 2"), IsOkAndHolds(Pointee(Gt(94))));
}

TEST(MultiTrainerCacheTest, NewTrainersStartLate) {
  MultiTrainerCache<int64_t> cache(
      /*max_cache_size_bytes=*/5 * sizeof(int64_t),
      absl::make_unique<InfiniteRange>());
  for (int i = 0; i < 100; ++i) {
    EXPECT_THAT(cache.Get("Old trainer"), IsOkAndHolds(Pointee(i)));
  }

  // New trainers start to read after the first trainer has finished.
  for (int j = 0; j < 100; ++j) {
    EXPECT_THAT(cache.Get(absl::StrCat("New trainer ", j)),
                IsOkAndHolds(Pointee(Gt(94))));
  }
}

TEST(MultiTrainerCacheTest, AlternateTrainerExtendsCache) {
  // The cache size is smaller than one int64_t.
  MultiTrainerCache<int64_t> cache(
      /*max_cache_size_bytes=*/sizeof(int64_t),
      absl::make_unique<InfiniteRange>());
  EXPECT_THAT(cache.Get("Trainer 1"), IsOkAndHolds(Pointee(0)));
  EXPECT_THAT(cache.Get("Trainer 1"), IsOkAndHolds(Pointee(1)));
  EXPECT_THAT(cache.Get("Trainer 1"), IsOkAndHolds(Pointee(2)));

  // When 2 is cached, 0 must have been discarded.
  EXPECT_THAT(cache.Get("Trainer 2"), IsOkAndHolds(Pointee(Gt(0))));
  EXPECT_THAT(cache.Get("Trainer 2"), IsOkAndHolds(Pointee(Gt(1))));
  EXPECT_THAT(cache.Get("Trainer 2"), IsOkAndHolds(Pointee(Gt(2))));

  // When 3 is cached, 1 must have been discarded.
  EXPECT_THAT(cache.Get("Trainer 1"), IsOkAndHolds(Pointee(Gt(1))));
  EXPECT_THAT(cache.Get("Trainer 1"), IsOkAndHolds(Pointee(Gt(2))));
  EXPECT_THAT(cache.Get("Trainer 1"), IsOkAndHolds(Pointee(Gt(3))));

  // When 4 is cached, 2 must have been discarded.
  EXPECT_THAT(cache.Get("Trainer 2"), IsOkAndHolds(Pointee(Gt(2))));
  EXPECT_THAT(cache.Get("Trainer 2"), IsOkAndHolds(Pointee(Gt(3))));
  EXPECT_THAT(cache.Get("Trainer 2"), IsOkAndHolds(Pointee(Gt(4))));

  // When 5 is cached, 3 must have been discarded.
  EXPECT_THAT(cache.Get("Trainer 3"), IsOkAndHolds(Pointee(Gt(3))));
  EXPECT_THAT(cache.Get("Trainer 3"), IsOkAndHolds(Pointee(Gt(4))));
  EXPECT_THAT(cache.Get("Trainer 3"), IsOkAndHolds(Pointee(Gt(5))));
}

TEST(MultiTrainerCacheTest, ConcurrentReaders) {
  size_t num_trainers = 10;
  size_t num_elements_to_read = 200;
  MultiTrainerCache<int64_t> cache(
      /*max_cache_size_bytes=*/3 * sizeof(int64_t),
      absl::make_unique<InfiniteRange>());

  std::vector<std::vector<int64_t>> results;
  std::vector<std::unique_ptr<Thread>> reader_threads;
  results.reserve(num_trainers);
  for (size_t i = 0; i < num_trainers; ++i) {
    results.emplace_back();
    std::vector<int64_t>& result = results.back();
    reader_threads.push_back(absl::WrapUnique(Env::Default()->StartThread(
        /*thread_options=*/{}, /*name=*/absl::StrCat("Trainer_", i),
        [&cache, num_elements_to_read, &result]() {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScorePSdataPSservicePSmulti_trainer_cache_testDTcc mht_9(mht_9_v, 449, "", "./tensorflow/core/data/service/multi_trainer_cache_test.cc", "lambda");

          for (size_t i = 0; i < num_elements_to_read; ++i) {
            // Randomly slows down some trainers.
            if (random::New64() % 5 == 0) {
              Env::Default()->SleepForMicroseconds(2000);
            }
            TF_ASSERT_OK_AND_ASSIGN(std::shared_ptr<const int64_t> next,
                                    cache.Get(absl::StrCat("Trainer_", i)));
            result.push_back(*next);
          }
        })));
  }

  for (auto& thread : reader_threads) {
    thread.reset();
  }

  // Verifies all trainers can read `num_elements_to_read` elements.
  EXPECT_EQ(results.size(), num_trainers);
  for (const std::vector<int64_t>& result : results) {
    EXPECT_EQ(result.size(), num_elements_to_read);
    EXPECT_TRUE(SequenceIsIncreasing(result));
  }
}

TEST(MultiTrainerCacheTest, ConcurrentReadersFromOneTrainer) {
  size_t num_trainers = 10;
  size_t num_elements_to_read = 100;
  MultiTrainerCache<int64_t> cache(
      /*max_cache_size_bytes=*/3 * sizeof(int64_t),
      absl::make_unique<InfiniteRange>());

  mutex mu;
  std::vector<int64_t> results;  // Guarded by `mu`.
  std::vector<std::unique_ptr<Thread>> reader_threads;
  for (size_t i = 0; i < num_trainers; ++i) {
    reader_threads.push_back(absl::WrapUnique(Env::Default()->StartThread(
        /*thread_options=*/{}, /*name=*/absl::StrCat("Thread_", i),
        [&cache, num_elements_to_read, &results, &mu]() {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScorePSdataPSservicePSmulti_trainer_cache_testDTcc mht_10(mht_10_v, 490, "", "./tensorflow/core/data/service/multi_trainer_cache_test.cc", "lambda");

          for (size_t i = 0; i < num_elements_to_read; ++i) {
            // Randomly slows down some trainers.
            if (random::New64() % 5 == 0) {
              Env::Default()->SleepForMicroseconds(1000);
            }
            TF_ASSERT_OK_AND_ASSIGN(std::shared_ptr<const int64_t> next,
                                    cache.Get("Trainer ID"));
            mutex_lock l(mu);
            results.push_back(*next);
          }
        })));
  }

  for (auto& thread : reader_threads) {
    thread.reset();
  }
  // Verifies the readers have read all elements because they have the same
  // trainer ID.
  EXPECT_THAT(results, UnorderedElementsAreArray(GetRange(1000)));
}

TEST(MultiTrainerCacheTest, Cancel) {
  size_t num_trainers = 10;
  MultiTrainerCache<Tensor> cache(
      /*max_cache_size_bytes=*/1000, absl::make_unique<TensorDataset>());
  EXPECT_FALSE(cache.IsCancelled());

  mutex mu;
  Status status;  // Guarded by `mu`.
  std::vector<std::unique_ptr<Thread>> reader_threads;
  for (size_t i = 0; i < num_trainers; ++i) {
    reader_threads.push_back(absl::WrapUnique(Env::Default()->StartThread(
        /*thread_options=*/{}, /*name=*/absl::StrCat("Trainer_", i),
        [&cache, &status, &mu]() {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScorePSdataPSservicePSmulti_trainer_cache_testDTcc mht_11(mht_11_v, 527, "", "./tensorflow/core/data/service/multi_trainer_cache_test.cc", "lambda");

          for (int j = 0; true; ++j) {
            StatusOr<std::shared_ptr<const Tensor>> tensor =
                cache.Get(absl::StrCat("Trainer_", j % 1000));
            {
              mutex_lock l(mu);
              status = tensor.status();
            }
            if (!tensor.status().ok()) {
              return;
            }
            test::ExpectEqual(*tensor.ValueOrDie(), Tensor("Test Tensor"));
          }
        })));
  }

  Env::Default()->SleepForMicroseconds(1000000);
  cache.Cancel(errors::Cancelled("Cancelled"));
  for (auto& thread : reader_threads) {
    thread.reset();
  }

  mutex_lock l(mu);
  EXPECT_THAT(status, StatusIs(error::CANCELLED));
  EXPECT_THAT(cache.Get("New trainer"), StatusIs(error::CANCELLED));
  EXPECT_TRUE(cache.IsCancelled());
}

TEST(MultiTrainerCacheTest, Errors) {
  auto elements = absl::make_unique<ElementOrErrorDataset<std::string>>(
      std::vector<StatusOr<std::string>>{
          std::string("First element"),
          errors::Cancelled("Cancelled"),
          std::string("Second element"),
          errors::InvalidArgument("InvalidArgument"),
          std::string("Third element"),
          errors::Unavailable("Unavailable"),
      });
  MultiTrainerCache<std::string> cache(
      /*max_cache_size_bytes=*/1000, std::move(elements));

  EXPECT_THAT(cache.Get("Trainer ID"),
              IsOkAndHolds(Pointee(std::string("First element"))));
  EXPECT_THAT(cache.Get("Trainer ID"), StatusIs(error::CANCELLED));
  EXPECT_THAT(cache.Get("Trainer ID"),
              IsOkAndHolds(Pointee(std::string("Second element"))));
  EXPECT_THAT(cache.Get("Trainer ID"), StatusIs(error::INVALID_ARGUMENT));
  EXPECT_THAT(cache.Get("Trainer ID"),
              IsOkAndHolds(Pointee(std::string("Third element"))));
  EXPECT_THAT(cache.Get("Trainer ID"), StatusIs(error::UNAVAILABLE));

  // Errors are not stored in the cache.
  EXPECT_THAT(cache.Get("New Trainer"),
              IsOkAndHolds(Pointee(std::string("First element"))));
  EXPECT_THAT(cache.Get("New Trainer"),
              IsOkAndHolds(Pointee(std::string("Second element"))));
  EXPECT_THAT(cache.Get("New Trainer"),
              IsOkAndHolds(Pointee(std::string("Third element"))));
}

TEST(MultiTrainerCacheTest, CacheSizeIsTooSmall) {
  // The cache size is smaller than one int64_t.
  MultiTrainerCache<Tensor> cache(
      /*max_cache_size_bytes=*/1, absl::make_unique<TensorDataset>());
  EXPECT_THAT(cache.Get("Trainer ID"),
              StatusIs(error::INVALID_ARGUMENT,
                       HasSubstr("tf.data service element size is larger than "
                                 "cache size in bytes.")));
}

TEST(MultiTrainerCacheTest, TrainerIDMustBeNonEmpty) {
  MultiTrainerCache<Tensor> cache(
      /*max_cache_size_bytes=*/1000, absl::make_unique<TensorDataset>());
  EXPECT_THAT(
      cache.Get(""),
      StatusIs(
          error::INTERNAL,
          "tf.data service multi-trainer cache trainer ID must be non-empty."));
}

}  // namespace
}  // namespace data
}  // namespace tensorflow
