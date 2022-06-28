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
class MHTracer_DTPStensorflowPScorePSutilPSpresized_cuckoo_map_testDTcc {
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
   MHTracer_DTPStensorflowPScorePSutilPSpresized_cuckoo_map_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSutilPSpresized_cuckoo_map_testDTcc() {
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

#include <array>

#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/fingerprint.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/platform/test_benchmark.h"
#include "tensorflow/core/util/presized_cuckoo_map.h"

namespace tensorflow {
namespace {

TEST(PresizedCuckooMapTest, MultiplyHigh) {
  struct Testcase {
    uint64 x;
    uint64 y;
    uint64 result;
  };
  std::array<Testcase, 7> testcases{
      {{0, 0, 0},
       {0xffffffff, 0xffffffff, 0},
       {0x2, 0xf000000000000000, 1},
       {0x3, 0xf000000000000000, 2},
       {0x3, 0xf000000000000001, 2},
       {0x3, 0xffffffffffffffff, 2},
       {0xffffffffffffffff, 0xffffffffffffffff, 0xfffffffffffffffe}}};
  for (auto &tc : testcases) {
    EXPECT_EQ(tc.result, presized_cuckoo_map::multiply_high_u64(tc.x, tc.y));
  }
}

TEST(PresizedCuckooMapTest, Basic) {
  PresizedCuckooMap<int> pscm(1000);
  EXPECT_TRUE(pscm.InsertUnique(1, 2));
  int out;
  EXPECT_TRUE(pscm.Find(1, &out));
  EXPECT_EQ(out, 2);
}

TEST(PresizedCuckooMapTest, Prefetch) {
  PresizedCuckooMap<int64_t> pscm(2);
  EXPECT_TRUE(pscm.InsertUnique(1, 2));
  // Works for both present and absent keys.
  pscm.PrefetchKey(1);
  pscm.PrefetchKey(2);
}

TEST(PresizedCuckooMapTest, TooManyItems) {
  static constexpr int kTableSize = 1000;
  PresizedCuckooMap<int> pscm(kTableSize);
  for (uint64 i = 0; i < kTableSize; i++) {
    uint64 key =
        Fingerprint64(string(reinterpret_cast<char *>(&i), sizeof(int64_t)));
    ASSERT_TRUE(pscm.InsertUnique(key, i));
  }
  // Try to over-fill the table.  A few of these
  // inserts will succeed, but should start failing.
  uint64 failed_at = 0;
  for (uint64 i = kTableSize; i < (2 * kTableSize); i++) {
    uint64 key =
        Fingerprint64(string(reinterpret_cast<char *>(&i), sizeof(int64_t)));
    if (!pscm.InsertUnique(key, i)) {
      failed_at = i;
      break;
    }
  }
  // Requirement 1:  Table must return failure when it's full.
  EXPECT_NE(failed_at, 0);

  // Requirement 2:  Table must preserve all items inserted prior
  // to the failure.
  for (uint64 i = 0; i < failed_at; i++) {
    int out;
    uint64 key =
        Fingerprint64(string(reinterpret_cast<char *>(&i), sizeof(int64_t)));
    EXPECT_TRUE(pscm.Find(key, &out));
    EXPECT_EQ(out, i);
  }
}

TEST(PresizedCuckooMapTest, ZeroSizeMap) {
  PresizedCuckooMap<int> pscm(0);
  int out;
  for (uint64 i = 0; i < 100; i++) {
    EXPECT_FALSE(pscm.Find(i, &out));
  }
}

TEST(PresizedCuckooMapTest, RepeatedClear) {
  PresizedCuckooMap<int> pscm(2);
  int out;
  for (int i = 0; i < 100; ++i) {
    pscm.InsertUnique(0, 0);
    pscm.InsertUnique(1, 1);
    EXPECT_TRUE(pscm.Find(0, &out));
    EXPECT_EQ(0, out);
    EXPECT_TRUE(pscm.Find(1, &out));
    EXPECT_EQ(1, out);
    pscm.Clear(2);
    EXPECT_FALSE(pscm.Find(0, &out));
    EXPECT_FALSE(pscm.Find(1, &out));
  }
}

void RunFill(int64_t table_size) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSutilPSpresized_cuckoo_map_testDTcc mht_0(mht_0_v, 288, "", "./tensorflow/core/util/presized_cuckoo_map_test.cc", "RunFill");

  PresizedCuckooMap<int> pscm(table_size);
  for (int64_t i = 0; i < table_size; i++) {
    uint64 key =
        Fingerprint64(string(reinterpret_cast<char *>(&i), sizeof(int64_t)));
    EXPECT_TRUE(pscm.InsertUnique(key, i));
  }
  for (int64_t i = 0; i < table_size; i++) {
    uint64 key =
        Fingerprint64(string(reinterpret_cast<char *>(&i), sizeof(int64_t)));
    int out;
    EXPECT_TRUE(pscm.Find(key, &out));
    EXPECT_EQ(out, i);
  }
}

TEST(PresizedCuckooMapTest, Fill) {
  for (int64_t table_size = 10; table_size <= 5000000; table_size *= 71) {
    RunFill(table_size);
  }
}

TEST(PresizedCuckooMapTest, Duplicates) {
  static constexpr int kSmallTableSize = 1000;
  PresizedCuckooMap<int> pscm(kSmallTableSize);

  for (uint64 i = 0; i < kSmallTableSize; i++) {
    uint64 key =
        Fingerprint64(string(reinterpret_cast<char *>(&i), sizeof(uint64)));
    EXPECT_TRUE(pscm.InsertUnique(key, i));
  }

  for (uint64 i = 0; i < kSmallTableSize; i++) {
    uint64 key =
        Fingerprint64(string(reinterpret_cast<char *>(&i), sizeof(uint64)));
    EXPECT_FALSE(pscm.InsertUnique(key, i));
  }
}

static void CalculateKeys(uint64 num, std::vector<uint64> *dst) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSutilPSpresized_cuckoo_map_testDTcc mht_1(mht_1_v, 330, "", "./tensorflow/core/util/presized_cuckoo_map_test.cc", "CalculateKeys");

  dst->resize(num);
  for (uint64 i = 0; i < num; i++) {
    uint64 key =
        Fingerprint64(string(reinterpret_cast<char *>(&i), sizeof(uint64)));
    dst->at(i) = key;
  }
}

void BM_CuckooFill(::testing::benchmark::State &state) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSutilPSpresized_cuckoo_map_testDTcc mht_2(mht_2_v, 342, "", "./tensorflow/core/util/presized_cuckoo_map_test.cc", "BM_CuckooFill");

  const int arg = state.range(0);

  uint64 table_size = arg;
  std::vector<uint64> calculated_keys;
  CalculateKeys(table_size, &calculated_keys);
  for (auto s : state) {
    PresizedCuckooMap<int> pscm(table_size);
    for (uint64 i = 0; i < table_size; i++) {
      pscm.InsertUnique(calculated_keys[i], i);
    }
  }
}

BENCHMARK(BM_CuckooFill)->Arg(1000)->Arg(10000000);

void BM_CuckooRead(::testing::benchmark::State &state) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSutilPSpresized_cuckoo_map_testDTcc mht_3(mht_3_v, 361, "", "./tensorflow/core/util/presized_cuckoo_map_test.cc", "BM_CuckooRead");

  const int arg = state.range(0);

  uint64 table_size = arg;
  std::vector<uint64> calculated_keys;
  CalculateKeys(table_size, &calculated_keys);
  PresizedCuckooMap<int> pscm(table_size);
  for (uint64 i = 0; i < table_size; i++) {
    pscm.InsertUnique(calculated_keys[i], i);
  }

  int i = 0;
  for (auto s : state) {
    // Avoid using '%', which is expensive.
    uint64 key_index = i;
    ++i;
    if (i == table_size) i = 0;

    int out = 0;
    pscm.Find(calculated_keys[key_index], &out);
    tensorflow::testing::DoNotOptimize(out);
  }
}

BENCHMARK(BM_CuckooRead)->Arg(1000)->Arg(10000000);

}  // namespace
}  // namespace tensorflow
