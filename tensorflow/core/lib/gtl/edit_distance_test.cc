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
class MHTracer_DTPStensorflowPScorePSlibPSgtlPSedit_distance_testDTcc {
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
   MHTracer_DTPStensorflowPScorePSlibPSgtlPSedit_distance_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSlibPSgtlPSedit_distance_testDTcc() {
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

#include "tensorflow/core/lib/gtl/edit_distance.h"

#include <cctype>
#include <vector>
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/platform/test_benchmark.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {
namespace gtl {
namespace {

class LevenshteinDistanceTest : public ::testing::Test {
 protected:
  std::vector<char> empty_;
  std::string s1_;
  std::string s1234_;
  std::string s567_;
  std::string kilo_;
  std::string kilogram_;
  std::string mother_;
  std::string grandmother_;
  std::string lower_;
  std::string upper_;
  std::vector<char> ebab_;
  std::vector<char> abcd_;

  void SetUp() override {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSlibPSgtlPSedit_distance_testDTcc mht_0(mht_0_v, 213, "", "./tensorflow/core/lib/gtl/edit_distance_test.cc", "SetUp");

    s1_ = "1";
    s1234_ = "1234";
    s567_ = "567";
    kilo_ = "kilo";
    kilogram_ = "kilogram";
    mother_ = "mother";
    grandmother_ = "grandmother";
    lower_ = "lower case";
    upper_ = "UPPER case";
    ebab_ = {'e', 'b', 'a', 'b'};
    abcd_ = {'a', 'b', 'c', 'd'};
  }
};

TEST_F(LevenshteinDistanceTest, BothEmpty) {
  ASSERT_EQ(LevenshteinDistance(empty_, empty_, std::equal_to<char>()), 0);
}

TEST_F(LevenshteinDistanceTest, Symmetry) {
  ASSERT_EQ(LevenshteinDistance(ebab_, abcd_, std::equal_to<char>()), 3);
  ASSERT_EQ(LevenshteinDistance(abcd_, ebab_, std::equal_to<char>()), 3);
}

TEST_F(LevenshteinDistanceTest, OneEmpty) {
  ASSERT_EQ(LevenshteinDistance(s1234_, empty_, std::equal_to<char>()), 4);
  ASSERT_EQ(LevenshteinDistance(empty_, s567_, std::equal_to<char>()), 3);
}

TEST_F(LevenshteinDistanceTest, SingleElement) {
  ASSERT_EQ(LevenshteinDistance(s1234_, s1_, std::equal_to<char>()), 3);
  ASSERT_EQ(LevenshteinDistance(s1_, s1234_, std::equal_to<char>()), 3);
}

TEST_F(LevenshteinDistanceTest, Prefix) {
  ASSERT_EQ(LevenshteinDistance(kilo_, kilogram_, std::equal_to<char>()), 4);
  ASSERT_EQ(LevenshteinDistance(kilogram_, kilo_, std::equal_to<char>()), 4);
}

TEST_F(LevenshteinDistanceTest, Suffix) {
  ASSERT_EQ(LevenshteinDistance(mother_, grandmother_, std::equal_to<char>()),
            5);
  ASSERT_EQ(LevenshteinDistance(grandmother_, mother_, std::equal_to<char>()),
            5);
}

TEST_F(LevenshteinDistanceTest, DifferentComparisons) {
  ASSERT_EQ(LevenshteinDistance(lower_, upper_, std::equal_to<char>()), 5);
  ASSERT_EQ(LevenshteinDistance(upper_, lower_, std::equal_to<char>()), 5);
  ASSERT_EQ(
      LevenshteinDistance(gtl::ArraySlice<char>(lower_.data(), lower_.size()),
                          gtl::ArraySlice<char>(upper_.data(), upper_.size()),
                          std::equal_to<char>()),
      5);
  auto no_case_cmp = [](char c1, char c2) {
   std::vector<std::string> mht_1_v;
   mht_1_v.push_back("c1: '" + std::string(1, c1) + "'");
   mht_1_v.push_back("c2: '" + std::string(1, c2) + "'");
   MHTracer_DTPStensorflowPScorePSlibPSgtlPSedit_distance_testDTcc mht_1(mht_1_v, 272, "", "./tensorflow/core/lib/gtl/edit_distance_test.cc", "lambda");

    return std::tolower(c1) == std::tolower(c2);
  };
  ASSERT_EQ(LevenshteinDistance(lower_, upper_, no_case_cmp), 3);
  ASSERT_EQ(LevenshteinDistance(upper_, lower_, no_case_cmp), 3);
}

TEST_F(LevenshteinDistanceTest, Vectors) {
  ASSERT_EQ(
      LevenshteinDistance(std::string("algorithm"), std::string("altruistic"),
                          std::equal_to<char>()),
      6);
}

static void BM_EditDistanceHelper(::testing::benchmark::State& state, int len,
                                  bool completely_different) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSlibPSgtlPSedit_distance_testDTcc mht_2(mht_2_v, 290, "", "./tensorflow/core/lib/gtl/edit_distance_test.cc", "BM_EditDistanceHelper");

  string a =
      "The quick brown fox jumped over the lazy dog and on and on and on"
      " Every good boy deserves fudge.  In fact, this is a very long sentence  "
      " w/many bytes..";
  while (a.size() < static_cast<size_t>(len)) {
    a = a + a;
  }
  string b = a;
  if (completely_different) {
    for (size_t i = 0; i < b.size(); i++) {
      b[i]++;
    }
  }
  for (auto s : state) {
    LevenshteinDistance(gtl::ArraySlice<char>(a.data(), len),
                        gtl::ArraySlice<char>(b.data(), len),
                        std::equal_to<char>());
  }
}

static void BM_EditDistanceSame(::testing::benchmark::State& state) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSlibPSgtlPSedit_distance_testDTcc mht_3(mht_3_v, 314, "", "./tensorflow/core/lib/gtl/edit_distance_test.cc", "BM_EditDistanceSame");

  BM_EditDistanceHelper(state, state.range(0), false);
}
static void BM_EditDistanceDiff(::testing::benchmark::State& state) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSlibPSgtlPSedit_distance_testDTcc mht_4(mht_4_v, 320, "", "./tensorflow/core/lib/gtl/edit_distance_test.cc", "BM_EditDistanceDiff");

  BM_EditDistanceHelper(state, state.range(0), true);
}

BENCHMARK(BM_EditDistanceSame)->Arg(5);
BENCHMARK(BM_EditDistanceSame)->Arg(50);
BENCHMARK(BM_EditDistanceSame)->Arg(200);
BENCHMARK(BM_EditDistanceSame)->Arg(1000);
BENCHMARK(BM_EditDistanceDiff)->Arg(5);
BENCHMARK(BM_EditDistanceDiff)->Arg(50);
BENCHMARK(BM_EditDistanceDiff)->Arg(200);
BENCHMARK(BM_EditDistanceDiff)->Arg(1000);

}  // namespace
}  // namespace gtl
}  // namespace tensorflow
