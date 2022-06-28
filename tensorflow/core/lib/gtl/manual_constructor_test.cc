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
class MHTracer_DTPStensorflowPScorePSlibPSgtlPSmanual_constructor_testDTcc {
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
   MHTracer_DTPStensorflowPScorePSlibPSgtlPSmanual_constructor_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSlibPSgtlPSmanual_constructor_testDTcc() {
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

#include "tensorflow/core/lib/gtl/manual_constructor.h"

#include <stdint.h>

#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {
namespace {

static int constructor_count_ = 0;

template <int kSize>
struct TestN {
  TestN() {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSlibPSgtlPSmanual_constructor_testDTcc mht_0(mht_0_v, 199, "", "./tensorflow/core/lib/gtl/manual_constructor_test.cc", "TestN");
 ++constructor_count_; }
  ~TestN() {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSlibPSgtlPSmanual_constructor_testDTcc mht_1(mht_1_v, 203, "", "./tensorflow/core/lib/gtl/manual_constructor_test.cc", "~TestN");
 --constructor_count_; }
  char a[kSize];
};

typedef TestN<1> Test1;
typedef TestN<2> Test2;
typedef TestN<3> Test3;
typedef TestN<4> Test4;
typedef TestN<5> Test5;
typedef TestN<9> Test9;
typedef TestN<15> Test15;

}  // namespace

namespace {

TEST(ManualConstructorTest, Sizeof) {
  CHECK_EQ(sizeof(ManualConstructor<Test1>), sizeof(Test1));
  CHECK_EQ(sizeof(ManualConstructor<Test2>), sizeof(Test2));
  CHECK_EQ(sizeof(ManualConstructor<Test3>), sizeof(Test3));
  CHECK_EQ(sizeof(ManualConstructor<Test4>), sizeof(Test4));
  CHECK_EQ(sizeof(ManualConstructor<Test5>), sizeof(Test5));
  CHECK_EQ(sizeof(ManualConstructor<Test9>), sizeof(Test9));
  CHECK_EQ(sizeof(ManualConstructor<Test15>), sizeof(Test15));

  CHECK_EQ(constructor_count_, 0);
  ManualConstructor<Test1> mt[4];
  CHECK_EQ(sizeof(mt), 4);
  CHECK_EQ(constructor_count_, 0);
  mt[0].Init();
  CHECK_EQ(constructor_count_, 1);
  mt[0].Destroy();
}

TEST(ManualConstructorTest, Alignment) {
  // We want to make sure that ManualConstructor aligns its memory properly
  // on a word barrier.  Otherwise, it might be unexpectedly slow, since
  // memory access will be unaligned.

  struct {
    char a;
    ManualConstructor<void*> b;
  } test1;
  struct {
    char a;
    void* b;
  } control1;

  // TODO(bww): Make these tests more direct with C++11 alignment_of<T>::value.
  EXPECT_EQ(reinterpret_cast<char*>(test1.b.get()) - &test1.a,
            reinterpret_cast<char*>(&control1.b) - &control1.a);
  EXPECT_EQ(reinterpret_cast<intptr_t>(test1.b.get()) % sizeof(control1.b), 0);

  struct {
    char a;
    ManualConstructor<long double> b;
  } test2;
  struct {
    char a;
    long double b;
  } control2;

  EXPECT_EQ(reinterpret_cast<char*>(test2.b.get()) - &test2.a,
            reinterpret_cast<char*>(&control2.b) - &control2.a);
#ifdef __x86_64__
  EXPECT_EQ(reinterpret_cast<intptr_t>(test2.b.get()) % 16, 0);
#endif
}

TEST(ManualConstructorTest, DefaultInitialize) {
  struct X {
    X() : x(123) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSlibPSgtlPSmanual_constructor_testDTcc mht_2(mht_2_v, 277, "", "./tensorflow/core/lib/gtl/manual_constructor_test.cc", "X");
}
    int x;
  };
  union {
    ManualConstructor<X> x;
    ManualConstructor<int> y;
  } u;
  *u.y = -1;
  u.x.Init();  // should default-initialize u.x
  EXPECT_EQ(123, u.x->x);
}

TEST(ManualConstructorTest, ZeroInitializePOD) {
  union {
    ManualConstructor<int> x;
    ManualConstructor<int> y;
  } u;
  *u.y = -1;
  u.x.Init();  // should not zero-initialize u.x
  EXPECT_EQ(-1, *u.y);
}

}  // namespace
}  // namespace tensorflow
