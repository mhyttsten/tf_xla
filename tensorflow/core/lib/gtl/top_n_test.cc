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
class MHTracer_DTPStensorflowPScorePSlibPSgtlPStop_n_testDTcc {
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
   MHTracer_DTPStensorflowPScorePSlibPSgtlPStop_n_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSlibPSgtlPStop_n_testDTcc() {
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

// Unit test for TopN.

#include "tensorflow/core/lib/gtl/top_n.h"

#include <string>
#include <vector>

#include "tensorflow/core/lib/random/simple_philox.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/platform/types.h"

namespace {

using tensorflow::string;
using tensorflow::gtl::TopN;
using tensorflow::random::PhiloxRandom;
using tensorflow::random::SimplePhilox;

// Move the contents from an owned raw pointer, returning by value.
// Objects are easier to manage by value.
template <class T>
T ConsumeRawPtr(T *p) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSlibPSgtlPStop_n_testDTcc mht_0(mht_0_v, 207, "", "./tensorflow/core/lib/gtl/top_n_test.cc", "ConsumeRawPtr");

  T tmp = std::move(*p);
  delete p;
  return tmp;
}

template <class Cmp>
void TestIntTopNHelper(size_t limit, size_t n_elements, const Cmp &cmp,
                       SimplePhilox *random, bool test_peek,
                       bool test_extract_unsorted) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSlibPSgtlPStop_n_testDTcc mht_1(mht_1_v, 219, "", "./tensorflow/core/lib/gtl/top_n_test.cc", "TestIntTopNHelper");

  LOG(INFO) << "Testing limit=" << limit << ", n_elements=" << n_elements
            << ", test_peek=" << test_peek
            << ", test_extract_unsorted=" << test_extract_unsorted;
  TopN<int, Cmp> top(limit, cmp);
  std::vector<int> shadow(n_elements);
  for (int i = 0; i != n_elements; ++i) shadow[i] = random->Uniform(limit);
  for (int e : shadow) top.push(e);
  std::sort(shadow.begin(), shadow.end(), cmp);
  size_t top_size = std::min(limit, n_elements);
  EXPECT_EQ(top_size, top.size());
  if (test_peek && top_size != 0) {
    EXPECT_EQ(shadow[top_size - 1], top.peek_bottom());
  }
  std::vector<int> v;
  if (test_extract_unsorted) {
    v = ConsumeRawPtr(top.ExtractUnsorted());
    std::sort(v.begin(), v.end(), cmp);
  } else {
    v = ConsumeRawPtr(top.Extract());
  }
  EXPECT_EQ(top_size, v.size());
  for (int i = 0; i != top_size; ++i) {
    VLOG(1) << "Top element " << v[i];
    EXPECT_EQ(shadow[i], v[i]);
  }
}

template <class Cmp>
void TestIntTopN(size_t limit, size_t n_elements, const Cmp &cmp,
                 SimplePhilox *random) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSlibPSgtlPStop_n_testDTcc mht_2(mht_2_v, 252, "", "./tensorflow/core/lib/gtl/top_n_test.cc", "TestIntTopN");

  // Test peek_bottom() and Extract()
  TestIntTopNHelper(limit, n_elements, cmp, random, true, false);
  // Test Extract()
  TestIntTopNHelper(limit, n_elements, cmp, random, false, false);
  // Test peek_bottom() and ExtractUnsorted()
  TestIntTopNHelper(limit, n_elements, cmp, random, true, true);
  // Test ExtractUnsorted()
  TestIntTopNHelper(limit, n_elements, cmp, random, false, true);
}

TEST(TopNTest, Misc) {
  PhiloxRandom philox(1, 1);
  SimplePhilox random(&philox);

  TestIntTopN(0, 5, std::greater<int>(), &random);
  TestIntTopN(32, 0, std::greater<int>(), &random);
  TestIntTopN(6, 6, std::greater<int>(), &random);
  TestIntTopN(6, 6, std::less<int>(), &random);
  TestIntTopN(1000, 999, std::greater<int>(), &random);
  TestIntTopN(1000, 1000, std::greater<int>(), &random);
  TestIntTopN(1000, 1001, std::greater<int>(), &random);
  TestIntTopN(2300, 28393, std::less<int>(), &random);
  TestIntTopN(30, 100, std::greater<int>(), &random);
  TestIntTopN(100, 30, std::less<int>(), &random);
  TestIntTopN(size_t(-1), 3, std::greater<int>(), &random);
  TestIntTopN(size_t(-1), 0, std::greater<int>(), &random);
  TestIntTopN(0, 5, std::greater<int>(), &random);
}

TEST(TopNTest, String) {
  LOG(INFO) << "Testing strings";

  TopN<string> top(3);
  EXPECT_TRUE(top.empty());
  top.push("abracadabra");
  top.push("waldemar");
  EXPECT_EQ(2, top.size());
  EXPECT_EQ("abracadabra", top.peek_bottom());
  top.push("");
  EXPECT_EQ(3, top.size());
  EXPECT_EQ("", top.peek_bottom());
  top.push("top");
  EXPECT_EQ(3, top.size());
  EXPECT_EQ("abracadabra", top.peek_bottom());
  top.push("Google");
  top.push("test");
  EXPECT_EQ(3, top.size());
  EXPECT_EQ("test", top.peek_bottom());
  TopN<string> top2(top);
  TopN<string> top3(5);
  top3 = top;
  EXPECT_EQ("test", top3.peek_bottom());
  {
    std::vector<string> s = ConsumeRawPtr(top.Extract());
    EXPECT_EQ(s[0], "waldemar");
    EXPECT_EQ(s[1], "top");
    EXPECT_EQ(s[2], "test");
  }

  top2.push("zero");
  EXPECT_EQ(top2.peek_bottom(), "top");

  {
    std::vector<string> s = ConsumeRawPtr(top2.Extract());
    EXPECT_EQ(s[0], "zero");
    EXPECT_EQ(s[1], "waldemar");
    EXPECT_EQ(s[2], "top");
  }
  {
    std::vector<string> s = ConsumeRawPtr(top3.Extract());
    EXPECT_EQ(s[0], "waldemar");
    EXPECT_EQ(s[1], "top");
    EXPECT_EQ(s[2], "test");
  }

  TopN<string> top4(3);
  // Run this test twice to check Reset():
  for (int i = 0; i < 2; ++i) {
    top4.push("abcd");
    top4.push("ijkl");
    top4.push("efgh");
    top4.push("mnop");
    std::vector<string> s = ConsumeRawPtr(top4.Extract());
    EXPECT_EQ(s[0], "mnop");
    EXPECT_EQ(s[1], "ijkl");
    EXPECT_EQ(s[2], "efgh");
    top4.Reset();
  }
}

// Test that pointers aren't leaked from a TopN if we use the 2-argument version
// of push().
TEST(TopNTest, Ptr) {
  LOG(INFO) << "Testing 2-argument push()";
  TopN<string *> topn(3);
  for (int i = 0; i < 8; ++i) {
    string *dropped = nullptr;
    topn.push(new string(std::to_string(i)), &dropped);
    delete dropped;
  }

  for (int i = 8; i > 0; --i) {
    string *dropped = nullptr;
    topn.push(new string(std::to_string(i)), &dropped);
    delete dropped;
  }

  std::vector<string *> extract = ConsumeRawPtr(topn.Extract());
  for (auto &temp : extract) {
    delete temp;
  }
  extract.clear();
}

struct PointeeGreater {
  template <typename T>
  bool operator()(const T &a, const T &b) const {
    return *a > *b;
  }
};

TEST(TopNTest, MoveOnly) {
  using StrPtr = std::unique_ptr<string>;
  TopN<StrPtr, PointeeGreater> topn(3);
  for (int i = 0; i < 8; ++i) topn.push(StrPtr(new string(std::to_string(i))));
  for (int i = 8; i > 0; --i) topn.push(StrPtr(new string(std::to_string(i))));

  std::vector<StrPtr> extract = ConsumeRawPtr(topn.Extract());
  EXPECT_EQ(extract.size(), 3);
  EXPECT_EQ(*(extract[0]), "8");
  EXPECT_EQ(*(extract[1]), "7");
  EXPECT_EQ(*(extract[2]), "7");
}

// Test that Nondestructive extracts do not need a Reset() afterwards,
// and that pointers aren't leaked from a TopN after calling them.
TEST(TopNTest, Nondestructive) {
  LOG(INFO) << "Testing Nondestructive extracts";
  TopN<int> top4(4);
  for (int i = 0; i < 8; ++i) {
    top4.push(i);
    std::vector<int> v = ConsumeRawPtr(top4.ExtractNondestructive());
    EXPECT_EQ(std::min(i + 1, 4), v.size());
    for (size_t j = 0; j < v.size(); ++j) EXPECT_EQ(i - j, v[j]);
  }

  TopN<int> top3(3);
  for (int i = 0; i < 8; ++i) {
    top3.push(i);
    std::vector<int> v = ConsumeRawPtr(top3.ExtractUnsortedNondestructive());
    std::sort(v.begin(), v.end(), std::greater<int>());
    EXPECT_EQ(std::min(i + 1, 3), v.size());
    for (size_t j = 0; j < v.size(); ++j) EXPECT_EQ(i - j, v[j]);
  }
}

struct ForbiddenCmp {
  bool operator()(int lhs, int rhs) const {
    LOG(FATAL) << "ForbiddenCmp called " << lhs << " " << rhs;
  }
};

TEST(TopNTest, ZeroLimit) {
  TopN<int, ForbiddenCmp> top(0);
  top.push(1);
  top.push(2);

  int dropped = -1;
  top.push(1, &dropped);
  top.push(2, &dropped);

  std::vector<int> v;
  top.ExtractNondestructive(&v);
  EXPECT_EQ(0, v.size());
}

TEST(TopNTest, Iteration) {
  TopN<int> top(4);
  for (int i = 0; i < 8; ++i) top.push(i);
  std::vector<int> actual(top.unsorted_begin(), top.unsorted_end());
  // Check that we have 4,5,6,7 as the top 4 (in some order, so we sort)
  std::sort(actual.begin(), actual.end());
  EXPECT_EQ(actual.size(), 4);
  EXPECT_EQ(actual[0], 4);
  EXPECT_EQ(actual[1], 5);
  EXPECT_EQ(actual[2], 6);
  EXPECT_EQ(actual[3], 7);
}
}  // namespace
