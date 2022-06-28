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
class MHTracer_DTPStensorflowPScorePSlibPSgtlPScleanup_testDTcc {
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
   MHTracer_DTPStensorflowPScorePSlibPSgtlPScleanup_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSlibPSgtlPScleanup_testDTcc() {
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

#include "tensorflow/core/lib/gtl/cleanup.h"

#include <functional>
#include <type_traits>

#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/platform/test_benchmark.h"

namespace tensorflow {
namespace {

using AnyCleanup = gtl::Cleanup<std::function<void()>>;

template <typename T1, typename T2>
void AssertTypeEq() {
  static_assert(std::is_same<T1, T2>::value, "unexpected type");
}

TEST(CleanupTest, BasicLambda) {
  string s = "active";
  {
    auto s_cleaner = gtl::MakeCleanup([&s] { s.assign("cleaned"); });
    EXPECT_EQ("active", s);
  }
  EXPECT_EQ("cleaned", s);
}

TEST(FinallyTest, NoCaptureLambda) {
  // Noncapturing lambdas are just structs and use aggregate initializers.
  // Make sure MakeCleanup is compatible with that kind of initialization.
  static string& s = *new string;
  s.assign("active");
  {
    auto s_cleaner = gtl::MakeCleanup([] { s.append(" clean"); });
    EXPECT_EQ("active", s);
  }
  EXPECT_EQ("active clean", s);
}

TEST(CleanupTest, Release) {
  string s = "active";
  {
    auto s_cleaner = gtl::MakeCleanup([&s] { s.assign("cleaned"); });
    EXPECT_EQ("active", s);
    s_cleaner.release();
  }
  EXPECT_EQ("active", s);  // no cleanup should have occurred.
}

TEST(FinallyTest, TypeErasedWithoutFactory) {
  string s = "active";
  {
    AnyCleanup s_cleaner([&s] { s.append(" clean"); });
    EXPECT_EQ("active", s);
  }
  EXPECT_EQ("active clean", s);
}

struct Appender {
  Appender(string* s, const string& msg) : s_(s), msg_(msg) {
   std::vector<std::string> mht_0_v;
   mht_0_v.push_back("msg: \"" + msg + "\"");
   MHTracer_DTPStensorflowPScorePSlibPSgtlPScleanup_testDTcc mht_0(mht_0_v, 245, "", "./tensorflow/core/lib/gtl/cleanup_test.cc", "Appender");
}
  void operator()() const { s_->append(msg_); }
  string* s_;
  string msg_;
};

TEST(CleanupTest, NonLambda) {
  string s = "active";
  {
    auto c = gtl::MakeCleanup(Appender(&s, " cleaned"));
    AssertTypeEq<decltype(c), gtl::Cleanup<Appender>>();
    EXPECT_EQ("active", s);
  }
  EXPECT_EQ("active cleaned", s);
}

TEST(CleanupTest, Assign) {
  string s = "0";
  {
    auto clean1 = gtl::MakeCleanup(Appender(&s, " 1"));
    auto clean2 = gtl::MakeCleanup(Appender(&s, " 2"));
    EXPECT_EQ("0", s);
    clean2 = std::move(clean1);
    EXPECT_EQ("0 2", s);
  }
  EXPECT_EQ("0 2 1", s);
}

TEST(CleanupTest, AssignAny) {
  // Check that implicit conversions can happen in assignment.
  string s = "0";
  {
    auto clean1 = gtl::MakeCleanup(Appender(&s, " 1"));
    AnyCleanup clean2 = gtl::MakeCleanup(Appender(&s, " 2"));
    EXPECT_EQ("0", s);
    clean2 = std::move(clean1);
    EXPECT_EQ("0 2", s);
  }
  EXPECT_EQ("0 2 1", s);
}

TEST(CleanupTest, AssignFromReleased) {
  string s = "0";
  {
    auto clean1 = gtl::MakeCleanup(Appender(&s, " 1"));
    auto clean2 = gtl::MakeCleanup(Appender(&s, " 2"));
    EXPECT_EQ("0", s);
    clean1.release();
    clean2 = std::move(clean1);
    EXPECT_EQ("0 2", s);
  }
  EXPECT_EQ("0 2", s);
}

TEST(CleanupTest, AssignToReleased) {
  string s = "0";
  {
    auto clean1 = gtl::MakeCleanup(Appender(&s, " 1"));
    auto clean2 = gtl::MakeCleanup(Appender(&s, " 2"));
    EXPECT_EQ("0", s);
    clean2.release();
    EXPECT_EQ("0", s);
    clean2 = std::move(clean1);
    EXPECT_EQ("0", s);
  }
  EXPECT_EQ("0 1", s);
}

TEST(CleanupTest, AssignToDefaultInitialized) {
  string s = "0";
  {
    auto clean1 = gtl::MakeCleanup(Appender(&s, " 1"));
    {
      AnyCleanup clean2;
      EXPECT_EQ("0", s);
      clean2 = std::move(clean1);
      EXPECT_EQ("0", s);
    }
    EXPECT_EQ("0 1", s);
  }
  EXPECT_EQ("0 1", s);
}

class CleanupReferenceTest : public ::testing::Test {
 public:
  struct F {
    int* cp;
    int* i;
    F(int* cp, int* i) : cp(cp), i(i) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSlibPSgtlPScleanup_testDTcc mht_1(mht_1_v, 336, "", "./tensorflow/core/lib/gtl/cleanup_test.cc", "F");
}
    F(const F& o) : cp(o.cp), i(o.i) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSlibPSgtlPScleanup_testDTcc mht_2(mht_2_v, 340, "", "./tensorflow/core/lib/gtl/cleanup_test.cc", "F");
 ++*cp; }
    F& operator=(const F& o) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSlibPSgtlPScleanup_testDTcc mht_3(mht_3_v, 344, "", "./tensorflow/core/lib/gtl/cleanup_test.cc", "=");

      cp = o.cp;
      i = o.i;
      ++*cp;
      return *this;
    }
    F(F&&) = default;
    F& operator=(F&&) = default;
    void operator()() const { ++*i; }
  };
  int copies_ = 0;
  int calls_ = 0;
  F f_ = F(&copies_, &calls_);

  static int g_calls;
  void SetUp() override {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSlibPSgtlPScleanup_testDTcc mht_4(mht_4_v, 362, "", "./tensorflow/core/lib/gtl/cleanup_test.cc", "SetUp");
 g_calls = 0; }
  static void CleanerFunction() {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSlibPSgtlPScleanup_testDTcc mht_5(mht_5_v, 366, "", "./tensorflow/core/lib/gtl/cleanup_test.cc", "CleanerFunction");
 ++g_calls; }
};
int CleanupReferenceTest::g_calls = 0;

TEST_F(CleanupReferenceTest, FunctionPointer) {
  {
    auto c = gtl::MakeCleanup(&CleanerFunction);
    AssertTypeEq<decltype(c), gtl::Cleanup<void (*)()>>();
    EXPECT_EQ(0, g_calls);
  }
  EXPECT_EQ(1, g_calls);
  // Test that a function reference decays to a function pointer.
  {
    auto c = gtl::MakeCleanup(CleanerFunction);
    AssertTypeEq<decltype(c), gtl::Cleanup<void (*)()>>();
    EXPECT_EQ(1, g_calls);
  }
  EXPECT_EQ(2, g_calls);
}

TEST_F(CleanupReferenceTest, AssignLvalue) {
  string s = "0";
  Appender app1(&s, "1");
  Appender app2(&s, "2");
  {
    auto c = gtl::MakeCleanup(app1);
    c.release();
    c = gtl::MakeCleanup(app2);
    EXPECT_EQ("0", s);
    app1();
    EXPECT_EQ("01", s);
  }
  EXPECT_EQ("012", s);
}

TEST_F(CleanupReferenceTest, FunctorLvalue) {
  // Test that MakeCleanup(lvalue) produces Cleanup<F>, not Cleanup<F&>.
  EXPECT_EQ(0, copies_);
  EXPECT_EQ(0, calls_);
  {
    auto c = gtl::MakeCleanup(f_);
    AssertTypeEq<decltype(c), gtl::Cleanup<F>>();
    EXPECT_EQ(1, copies_);
    EXPECT_EQ(0, calls_);
  }
  EXPECT_EQ(1, copies_);
  EXPECT_EQ(1, calls_);
  {
    auto c = gtl::MakeCleanup(f_);
    EXPECT_EQ(2, copies_);
    EXPECT_EQ(1, calls_);
    F f2 = c.release();  // release is a move.
    EXPECT_EQ(2, copies_);
    EXPECT_EQ(1, calls_);
    auto c2 = gtl::MakeCleanup(f2);  // copy
    EXPECT_EQ(3, copies_);
    EXPECT_EQ(1, calls_);
  }
  EXPECT_EQ(3, copies_);
  EXPECT_EQ(2, calls_);
}

TEST_F(CleanupReferenceTest, FunctorRvalue) {
  {
    auto c = gtl::MakeCleanup(std::move(f_));
    AssertTypeEq<decltype(c), gtl::Cleanup<F>>();
    EXPECT_EQ(0, copies_);
    EXPECT_EQ(0, calls_);
  }
  EXPECT_EQ(0, copies_);
  EXPECT_EQ(1, calls_);
}

TEST_F(CleanupReferenceTest, FunctorReferenceWrapper) {
  {
    auto c = gtl::MakeCleanup(std::cref(f_));
    AssertTypeEq<decltype(c), gtl::Cleanup<std::reference_wrapper<const F>>>();
    EXPECT_EQ(0, copies_);
    EXPECT_EQ(0, calls_);
  }
  EXPECT_EQ(0, copies_);
  EXPECT_EQ(1, calls_);
}

volatile int i;

void Incr(volatile int* ip) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePSlibPSgtlPScleanup_testDTcc mht_6(mht_6_v, 455, "", "./tensorflow/core/lib/gtl/cleanup_test.cc", "Incr");
 ++*ip; }
void Incr() {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePSlibPSgtlPScleanup_testDTcc mht_7(mht_7_v, 459, "", "./tensorflow/core/lib/gtl/cleanup_test.cc", "Incr");
 Incr(&i); }

void BM_Cleanup(::testing::benchmark::State& state) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScorePSlibPSgtlPScleanup_testDTcc mht_8(mht_8_v, 464, "", "./tensorflow/core/lib/gtl/cleanup_test.cc", "BM_Cleanup");

  for (auto s : state) {
    auto fin = gtl::MakeCleanup([] { Incr(); });
  }
}
BENCHMARK(BM_Cleanup);

void BM_AnyCleanup(::testing::benchmark::State& state) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScorePSlibPSgtlPScleanup_testDTcc mht_9(mht_9_v, 474, "", "./tensorflow/core/lib/gtl/cleanup_test.cc", "BM_AnyCleanup");

  for (auto s : state) {
    AnyCleanup fin = gtl::MakeCleanup([] { Incr(); });
  }
}
BENCHMARK(BM_AnyCleanup);

void BM_AnyCleanupNoFactory(::testing::benchmark::State& state) {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScorePSlibPSgtlPScleanup_testDTcc mht_10(mht_10_v, 484, "", "./tensorflow/core/lib/gtl/cleanup_test.cc", "BM_AnyCleanupNoFactory");

  for (auto s : state) {
    AnyCleanup fin([] { Incr(); });
  }
}
BENCHMARK(BM_AnyCleanupNoFactory);

void BM_CleanupBound(::testing::benchmark::State& state) {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScorePSlibPSgtlPScleanup_testDTcc mht_11(mht_11_v, 494, "", "./tensorflow/core/lib/gtl/cleanup_test.cc", "BM_CleanupBound");

  volatile int* ip = &i;
  for (auto s : state) {
    auto fin = gtl::MakeCleanup([ip] { Incr(ip); });
  }
}
BENCHMARK(BM_CleanupBound);

void BM_AnyCleanupBound(::testing::benchmark::State& state) {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScorePSlibPSgtlPScleanup_testDTcc mht_12(mht_12_v, 505, "", "./tensorflow/core/lib/gtl/cleanup_test.cc", "BM_AnyCleanupBound");

  volatile int* ip = &i;
  for (auto s : state) {
    AnyCleanup fin = gtl::MakeCleanup([ip] { Incr(ip); });
  }
}
BENCHMARK(BM_AnyCleanupBound);

void BM_AnyCleanupNoFactoryBound(::testing::benchmark::State& state) {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPScorePSlibPSgtlPScleanup_testDTcc mht_13(mht_13_v, 516, "", "./tensorflow/core/lib/gtl/cleanup_test.cc", "BM_AnyCleanupNoFactoryBound");

  volatile int* ip = &i;
  for (auto s : state) {
    AnyCleanup fin([ip] { Incr(ip); });
  }
}
BENCHMARK(BM_AnyCleanupNoFactoryBound);

}  // namespace
}  // namespace tensorflow
