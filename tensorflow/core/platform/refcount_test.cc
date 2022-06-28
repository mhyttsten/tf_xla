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
class MHTracer_DTPStensorflowPScorePSplatformPSrefcount_testDTcc {
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
   MHTracer_DTPStensorflowPScorePSplatformPSrefcount_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSplatformPSrefcount_testDTcc() {
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

#include "tensorflow/core/platform/refcount.h"

#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/platform/threadpool.h"

namespace tensorflow {
namespace core {
namespace {

class RefTest : public ::testing::Test {
 public:
  RefTest() {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSplatformPSrefcount_testDTcc mht_0(mht_0_v, 196, "", "./tensorflow/core/platform/refcount_test.cc", "RefTest");

    constructed_ = 0;
    destroyed_ = 0;
  }

  static int constructed_;
  static int destroyed_;
};

int RefTest::constructed_;
int RefTest::destroyed_;

class MyRef : public RefCounted {
 public:
  MyRef() {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSplatformPSrefcount_testDTcc mht_1(mht_1_v, 213, "", "./tensorflow/core/platform/refcount_test.cc", "MyRef");
 RefTest::constructed_++; }
  ~MyRef() override {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSplatformPSrefcount_testDTcc mht_2(mht_2_v, 217, "", "./tensorflow/core/platform/refcount_test.cc", "~MyRef");
 RefTest::destroyed_++; }
};

TEST_F(RefTest, New) {
  MyRef* ref = new MyRef;
  ASSERT_EQ(1, constructed_);
  ASSERT_EQ(0, destroyed_);
  ref->Unref();
  ASSERT_EQ(1, constructed_);
  ASSERT_EQ(1, destroyed_);
}

TEST_F(RefTest, RefUnref) {
  MyRef* ref = new MyRef;
  ASSERT_EQ(1, constructed_);
  ASSERT_EQ(0, destroyed_);
  ref->Ref();
  ASSERT_EQ(0, destroyed_);
  ref->Unref();
  ASSERT_EQ(0, destroyed_);
  ref->Unref();
  ASSERT_EQ(1, destroyed_);
}

TEST_F(RefTest, RefCountOne) {
  MyRef* ref = new MyRef;
  ASSERT_TRUE(ref->RefCountIsOne());
  ref->Unref();
}

TEST_F(RefTest, RefCountNotOne) {
  MyRef* ref = new MyRef;
  ref->Ref();
  ASSERT_FALSE(ref->RefCountIsOne());
  ref->Unref();
  ref->Unref();
}

TEST_F(RefTest, ConstRefUnref) {
  const MyRef* cref = new MyRef;
  ASSERT_EQ(1, constructed_);
  ASSERT_EQ(0, destroyed_);
  cref->Ref();
  ASSERT_EQ(0, destroyed_);
  cref->Unref();
  ASSERT_EQ(0, destroyed_);
  cref->Unref();
  ASSERT_EQ(1, destroyed_);
}

TEST_F(RefTest, ReturnOfUnref) {
  MyRef* ref = new MyRef;
  ref->Ref();
  EXPECT_FALSE(ref->Unref());
  EXPECT_TRUE(ref->Unref());
}

TEST_F(RefTest, ScopedUnref) {
  { ScopedUnref unref(new MyRef); }
  EXPECT_EQ(destroyed_, 1);
}

TEST_F(RefTest, ScopedUnref_Nullptr) {
  { ScopedUnref unref(nullptr); }
  EXPECT_EQ(destroyed_, 0);
}

class ObjType : public WeakRefCounted {};

TEST(WeakPtr, SingleThread) {
  auto obj = new ObjType();
  WeakPtr<ObjType> weakptr(obj);

  ASSERT_TRUE(obj->RefCountIsOne());
  EXPECT_EQ(obj->WeakRefCount(), 1);
  EXPECT_NE(weakptr.GetNewRef(), nullptr);

  obj->Unref();
  EXPECT_EQ(weakptr.GetNewRef(), nullptr);
}

TEST(WeakPtr, MultiThreadedWeakRef) {
  // Exercise 100 times to make sure both branches of fn are hit.
  std::atomic<int> hit_destructed{0};

  auto env = Env::Default();

  for (int i = 0; i < 100; i++) {
    auto obj = new ObjType();
    WeakPtr<ObjType> weakptr(obj);

    bool obj_destructed = false;
    EXPECT_EQ(obj->WeakRefCount(), 1);

    auto fn = [&]() {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSplatformPSrefcount_testDTcc mht_3(mht_3_v, 314, "", "./tensorflow/core/platform/refcount_test.cc", "lambda");

      auto ref = weakptr.GetNewRef();
      if (ref != nullptr) {
        EXPECT_EQ(ref.get(), obj);
        EXPECT_EQ(ref->WeakRefCount(), 1);
        EXPECT_GE(ref->RefCount(), 1);
      } else {
        hit_destructed++;
        EXPECT_TRUE(obj_destructed);
      }
    };

    auto t1 = env->StartThread(ThreadOptions{}, "thread-1", fn);
    auto t2 = env->StartThread(ThreadOptions{}, "thread-2", fn);

    env->SleepForMicroseconds(10);
    obj_destructed = true;  // This shall run before weakref is purged.
    obj->Unref();

    delete t1;
    delete t2;

    EXPECT_EQ(weakptr.GetNewRef(), nullptr);
  }
  if (hit_destructed == 0) {
    LOG(WARNING) << "The destructed weakref test branch is not exercised.";
  }
  if (hit_destructed == 200) {
    LOG(WARNING) << "The valid weakref test branch is not exercised.";
  }
}

TEST(WeakPtr, NotifyCalled) {
  auto obj = new ObjType();
  int num_calls1 = 0;
  int num_calls2 = 0;

  auto notify_fn1 = [&num_calls1]() {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSplatformPSrefcount_testDTcc mht_4(mht_4_v, 354, "", "./tensorflow/core/platform/refcount_test.cc", "lambda");
 num_calls1++; };
  auto notify_fn2 = [&num_calls2]() {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSplatformPSrefcount_testDTcc mht_5(mht_5_v, 358, "", "./tensorflow/core/platform/refcount_test.cc", "lambda");
 num_calls2++; };
  WeakPtr<ObjType> weakptr1(obj, notify_fn1);
  WeakPtr<ObjType> weakptr2(obj, notify_fn2);

  ASSERT_TRUE(obj->RefCountIsOne());
  EXPECT_EQ(obj->WeakRefCount(), 2);
  EXPECT_NE(weakptr1.GetNewRef(), nullptr);
  EXPECT_NE(weakptr2.GetNewRef(), nullptr);

  EXPECT_EQ(num_calls1, 0);
  EXPECT_EQ(num_calls2, 0);
  obj->Unref();
  EXPECT_EQ(weakptr1.GetNewRef(), nullptr);
  EXPECT_EQ(weakptr2.GetNewRef(), nullptr);
  EXPECT_EQ(num_calls1, 1);
  EXPECT_EQ(num_calls2, 1);
}

TEST(WeakPtr, MoveTargetNotCalled) {
  auto obj = new ObjType();
  int num_calls1 = 0;
  int num_calls2 = 0;
  int num_calls3 = 0;

  auto notify_fn1 = [&num_calls1]() {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePSplatformPSrefcount_testDTcc mht_6(mht_6_v, 385, "", "./tensorflow/core/platform/refcount_test.cc", "lambda");
 num_calls1++; };
  auto notify_fn2 = [&num_calls2]() {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePSplatformPSrefcount_testDTcc mht_7(mht_7_v, 389, "", "./tensorflow/core/platform/refcount_test.cc", "lambda");
 num_calls2++; };
  auto notify_fn3 = [&num_calls3]() {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScorePSplatformPSrefcount_testDTcc mht_8(mht_8_v, 393, "", "./tensorflow/core/platform/refcount_test.cc", "lambda");
 num_calls3++; };
  WeakPtr<ObjType> weakptr1(obj, notify_fn1);
  WeakPtr<ObjType> weakptr2(obj, notify_fn2);
  WeakPtr<ObjType> weakptr3(WeakPtr<ObjType>(obj, notify_fn3));

  weakptr2 = std::move(weakptr1);

  ASSERT_TRUE(obj->RefCountIsOne());
  EXPECT_EQ(obj->WeakRefCount(), 2);
  EXPECT_NE(weakptr2.GetNewRef(), nullptr);
  EXPECT_NE(weakptr3.GetNewRef(), nullptr);

  EXPECT_EQ(num_calls1, 0);
  EXPECT_EQ(num_calls2, 0);
  EXPECT_EQ(num_calls3, 0);
  obj->Unref();
  EXPECT_EQ(weakptr2.GetNewRef(), nullptr);
  EXPECT_EQ(weakptr3.GetNewRef(), nullptr);
  EXPECT_EQ(num_calls1, 1);
  EXPECT_EQ(num_calls2, 0);
  EXPECT_EQ(num_calls3, 1);
}

TEST(WeakPtr, DestroyedNotifyNotCalled) {
  auto obj = new ObjType();
  int num_calls = 0;
  auto notify_fn = [&num_calls]() {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScorePSplatformPSrefcount_testDTcc mht_9(mht_9_v, 422, "", "./tensorflow/core/platform/refcount_test.cc", "lambda");
 num_calls++; };
  { WeakPtr<ObjType> weakptr(obj, notify_fn); }
  ASSERT_TRUE(obj->RefCountIsOne());
  EXPECT_EQ(obj->WeakRefCount(), 0);

  EXPECT_EQ(num_calls, 0);
  obj->Unref();
  EXPECT_EQ(num_calls, 0);
}

}  // namespace
}  // namespace core
}  // namespace tensorflow
