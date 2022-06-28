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
class MHTracer_DTPStensorflowPScorePSlibPSrandomPSweighted_picker_testDTcc {
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
   MHTracer_DTPStensorflowPScorePSlibPSrandomPSweighted_picker_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSlibPSrandomPSweighted_picker_testDTcc() {
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

/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/lib/random/weighted_picker.h"

#include <string.h>
#include <vector>

#include "tensorflow/core/lib/random/simple_philox.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/platform/test_benchmark.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {
namespace random {

static void TestPicker(SimplePhilox* rnd, int size);
static void CheckUniform(SimplePhilox* rnd, WeightedPicker* picker, int trials);
static void CheckSkewed(SimplePhilox* rnd, WeightedPicker* picker, int trials);
static void TestPickAt(int items, const int32* weights);

TEST(WeightedPicker, Simple) {
  PhiloxRandom philox(testing::RandomSeed(), 17);
  SimplePhilox rnd(&philox);

  {
    VLOG(0) << "======= Zero-length picker";
    WeightedPicker picker(0);
    EXPECT_EQ(picker.Pick(&rnd), -1);
  }

  {
    VLOG(0) << "======= Singleton picker";
    WeightedPicker picker(1);
    EXPECT_EQ(picker.Pick(&rnd), 0);
    EXPECT_EQ(picker.Pick(&rnd), 0);
    EXPECT_EQ(picker.Pick(&rnd), 0);
  }

  {
    VLOG(0) << "======= Grown picker";
    WeightedPicker picker(0);
    for (int i = 0; i < 10; i++) {
      picker.Append(1);
    }
    CheckUniform(&rnd, &picker, 100000);
  }

  {
    VLOG(0) << "======= Grown picker with zero weights";
    WeightedPicker picker(1);
    picker.Resize(10);
    EXPECT_EQ(picker.Pick(&rnd), 0);
    EXPECT_EQ(picker.Pick(&rnd), 0);
    EXPECT_EQ(picker.Pick(&rnd), 0);
  }

  {
    VLOG(0) << "======= Shrink picker and check weights";
    WeightedPicker picker(1);
    picker.Resize(10);
    EXPECT_EQ(picker.Pick(&rnd), 0);
    EXPECT_EQ(picker.Pick(&rnd), 0);
    EXPECT_EQ(picker.Pick(&rnd), 0);
    for (int i = 0; i < 10; i++) {
      picker.set_weight(i, i);
    }
    EXPECT_EQ(picker.total_weight(), 45);
    picker.Resize(5);
    EXPECT_EQ(picker.total_weight(), 10);
    picker.Resize(2);
    EXPECT_EQ(picker.total_weight(), 1);
    picker.Resize(1);
    EXPECT_EQ(picker.total_weight(), 0);
  }
}

TEST(WeightedPicker, BigWeights) {
  PhiloxRandom philox(testing::RandomSeed() + 1, 17);
  SimplePhilox rnd(&philox);
  VLOG(0) << "======= Check uniform with big weights";
  WeightedPicker picker(2);
  picker.SetAllWeights(2147483646L / 3);  // (2^31 - 2) / 3
  CheckUniform(&rnd, &picker, 100000);
}

TEST(WeightedPicker, Deterministic) {
  VLOG(0) << "======= Testing deterministic pick";
  static const int32 weights[] = {1, 0, 200, 5, 42};
  TestPickAt(TF_ARRAYSIZE(weights), weights);
}

TEST(WeightedPicker, Randomized) {
  PhiloxRandom philox(testing::RandomSeed() + 10, 17);
  SimplePhilox rnd(&philox);
  TestPicker(&rnd, 1);
  TestPicker(&rnd, 2);
  TestPicker(&rnd, 3);
  TestPicker(&rnd, 4);
  TestPicker(&rnd, 7);
  TestPicker(&rnd, 8);
  TestPicker(&rnd, 9);
  TestPicker(&rnd, 10);
  TestPicker(&rnd, 100);
}

static void TestPicker(SimplePhilox* rnd, int size) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSlibPSrandomPSweighted_picker_testDTcc mht_0(mht_0_v, 290, "", "./tensorflow/core/lib/random/weighted_picker_test.cc", "TestPicker");

  VLOG(0) << "======= Testing size " << size;

  // Check that empty picker returns -1
  {
    WeightedPicker picker(size);
    picker.SetAllWeights(0);
    for (int i = 0; i < 100; i++) EXPECT_EQ(picker.Pick(rnd), -1);
  }

  // Create zero weights array
  std::vector<int32> weights(size);
  for (int elem = 0; elem < size; elem++) {
    weights[elem] = 0;
  }

  // Check that singleton picker always returns the same element
  for (int elem = 0; elem < size; elem++) {
    WeightedPicker picker(size);
    picker.SetAllWeights(0);
    picker.set_weight(elem, elem + 1);
    for (int i = 0; i < 100; i++) EXPECT_EQ(picker.Pick(rnd), elem);
    weights[elem] = 10;
    picker.SetWeightsFromArray(size, &weights[0]);
    for (int i = 0; i < 100; i++) EXPECT_EQ(picker.Pick(rnd), elem);
    weights[elem] = 0;
  }

  // Check that uniform picker generates elements roughly uniformly
  {
    WeightedPicker picker(size);
    CheckUniform(rnd, &picker, 100000);
  }

  // Check uniform picker that was grown piecemeal
  if (size / 3 > 0) {
    WeightedPicker picker(size / 3);
    while (picker.num_elements() != size) {
      picker.Append(1);
    }
    CheckUniform(rnd, &picker, 100000);
  }

  // Check that skewed distribution works
  if (size <= 10) {
    // When picker grows one element at a time
    WeightedPicker picker(size);
    int32_t weight = 1;
    for (int elem = 0; elem < size; elem++) {
      picker.set_weight(elem, weight);
      weights[elem] = weight;
      weight *= 2;
    }
    CheckSkewed(rnd, &picker, 1000000);

    // When picker is created from an array
    WeightedPicker array_picker(0);
    array_picker.SetWeightsFromArray(size, &weights[0]);
    CheckSkewed(rnd, &array_picker, 1000000);
  }
}

static void CheckUniform(SimplePhilox* rnd, WeightedPicker* picker,
                         int trials) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSlibPSrandomPSweighted_picker_testDTcc mht_1(mht_1_v, 356, "", "./tensorflow/core/lib/random/weighted_picker_test.cc", "CheckUniform");

  const int size = picker->num_elements();
  int* count = new int[size];
  memset(count, 0, sizeof(count[0]) * size);
  for (int i = 0; i < size * trials; i++) {
    const int elem = picker->Pick(rnd);
    EXPECT_GE(elem, 0);
    EXPECT_LT(elem, size);
    count[elem]++;
  }
  const int expected_min = int(0.9 * trials);
  const int expected_max = int(1.1 * trials);
  for (int i = 0; i < size; i++) {
    EXPECT_GE(count[i], expected_min);
    EXPECT_LE(count[i], expected_max);
  }
  delete[] count;
}

static void CheckSkewed(SimplePhilox* rnd, WeightedPicker* picker, int trials) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSlibPSrandomPSweighted_picker_testDTcc mht_2(mht_2_v, 378, "", "./tensorflow/core/lib/random/weighted_picker_test.cc", "CheckSkewed");

  const int size = picker->num_elements();
  int* count = new int[size];
  memset(count, 0, sizeof(count[0]) * size);
  for (int i = 0; i < size * trials; i++) {
    const int elem = picker->Pick(rnd);
    EXPECT_GE(elem, 0);
    EXPECT_LT(elem, size);
    count[elem]++;
  }

  for (int i = 0; i < size - 1; i++) {
    LOG(INFO) << i << ": " << count[i];
    const float ratio = float(count[i + 1]) / float(count[i]);
    EXPECT_GE(ratio, 1.6f);
    EXPECT_LE(ratio, 2.4f);
  }
  delete[] count;
}

static void TestPickAt(int items, const int32* weights) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSlibPSrandomPSweighted_picker_testDTcc mht_3(mht_3_v, 401, "", "./tensorflow/core/lib/random/weighted_picker_test.cc", "TestPickAt");

  WeightedPicker picker(items);
  picker.SetWeightsFromArray(items, weights);
  int weight_index = 0;
  for (int i = 0; i < items; ++i) {
    for (int j = 0; j < weights[i]; ++j) {
      int pick = picker.PickAt(weight_index);
      EXPECT_EQ(pick, i);
      ++weight_index;
    }
  }
  EXPECT_EQ(weight_index, picker.total_weight());
}

static void BM_Create(::testing::benchmark::State& state) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSlibPSrandomPSweighted_picker_testDTcc mht_4(mht_4_v, 418, "", "./tensorflow/core/lib/random/weighted_picker_test.cc", "BM_Create");

  int arg = state.range(0);
  for (auto s : state) {
    WeightedPicker p(arg);
  }
}
BENCHMARK(BM_Create)->Range(1, 1024);

static void BM_CreateAndSetWeights(::testing::benchmark::State& state) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSlibPSrandomPSweighted_picker_testDTcc mht_5(mht_5_v, 429, "", "./tensorflow/core/lib/random/weighted_picker_test.cc", "BM_CreateAndSetWeights");

  int arg = state.range(0);
  std::vector<int32> weights(arg);
  for (int i = 0; i < arg; i++) {
    weights[i] = i * 10;
  }
  for (auto s : state) {
    WeightedPicker p(arg);
    p.SetWeightsFromArray(arg, &weights[0]);
  }
}
BENCHMARK(BM_CreateAndSetWeights)->Range(1, 1024);

static void BM_Pick(::testing::benchmark::State& state) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePSlibPSrandomPSweighted_picker_testDTcc mht_6(mht_6_v, 445, "", "./tensorflow/core/lib/random/weighted_picker_test.cc", "BM_Pick");

  int arg = state.range(0);
  PhiloxRandom philox(301, 17);
  SimplePhilox rnd(&philox);
  WeightedPicker p(arg);
  int result = 0;
  for (auto s : state) {
    result += p.Pick(&rnd);
  }
  VLOG(4) << result;  // Dummy use
}
BENCHMARK(BM_Pick)->Range(1, 1024);

}  // namespace random
}  // namespace tensorflow
