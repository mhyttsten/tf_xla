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
class MHTracer_DTPStensorflowPSccPSexperimentalPSlibtfPStestsPSobject_testDTcc {
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
   MHTracer_DTPStensorflowPSccPSexperimentalPSlibtfPStestsPSobject_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSccPSexperimentalPSlibtfPStestsPSobject_testDTcc() {
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
#include "tensorflow/cc/experimental/libtf/object.h"

#include <cstdint>

#include "tensorflow/cc/experimental/libtf/value.h"
#include "tensorflow/cc/experimental/libtf/value_iostream.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/statusor.h"
#include "tensorflow/core/platform/test.h"

namespace tf {
namespace libtf {

TEST(ObjectTest, TestDictionary) {
  Dictionary foo;
  foo.Set(String("a"), Integer(33));
  foo.Set(String("b"), Integer(33));
  EXPECT_EQ(foo.Get<Integer>(String("b"))->get(), 33);
}

TEST(ObjectTest, TestTuple) {
  Tuple foo(String("a"), Integer(33), Float(10.f));
  EXPECT_EQ(foo.size(), 3);
  EXPECT_EQ(foo.Get<Integer>(1)->get(), 33);
}

TEST(ObjectTest, TestList) {
  List l;
  EXPECT_EQ(l.size(), 0);
  l.append(Integer(3));
  EXPECT_EQ(l.Get<Integer>(0)->get(), 3);
  EXPECT_EQ(l.size(), 1);
}

TaggedValue AddIntegers(TaggedValue args_, TaggedValue kwargs_) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSccPSexperimentalPSlibtfPStestsPSobject_testDTcc mht_0(mht_0_v, 218, "", "./tensorflow/cc/experimental/libtf/tests/object_test.cc", "AddIntegers");

  auto& args = args_.tuple();
  // auto& kwargs = kwargs_.dict();
  return TaggedValue(args[0].i64() + args[1].i64());
}

TEST(ObjectTest, TestCast) {
  Integer i(3);
  auto result = Cast<String>(i);
  ASSERT_TRUE(!result.ok());
}

TEST(ObjectTest, TestCall) {
  TaggedValue add_func(AddIntegers);
  Callable add(add_func);
  TF_ASSERT_OK_AND_ASSIGN(Integer i,
                          add.Call<Integer>(Integer(1), Integer(10)));
  EXPECT_EQ(i.get(), 11);

  TF_ASSERT_OK_AND_ASSIGN(
      Integer i2, add.Call<Integer>(1, Integer(10), KeywordArg("foo") = 3));
  EXPECT_EQ(i2.get(), 11);
}

TEST(ObjectTest, MakeObject) {
  // TaggedValue func(f);
  Object parent;
  parent.Set(String("test3"), Integer(3));
  Object child;
  child.Set(String("test1"), Integer(1));
  child.Set(String("test2"), Integer(2));
  child.Set(Object::ParentKey(), parent);
  EXPECT_EQ(child.Get<Integer>(String("test1"))->get(), 1);
  EXPECT_EQ(child.Get<Integer>(String("test2"))->get(), 2);
  EXPECT_EQ(child.Get<Integer>(String("test3"))->get(), 3);
  ASSERT_FALSE(child.Get<Integer>(String("test4")).status().ok());
  TF_ASSERT_OK(child.Get(String("test3")).status());
}

TEST(ObjectTest, CallFunctionOnObject) {
  Object module;
  module.Set(String("add"), Callable(TaggedValue(AddIntegers)));
  TF_ASSERT_OK_AND_ASSIGN(Callable method, module.Get<Callable>(String("add")));

  TF_ASSERT_OK_AND_ASSIGN(Integer val, method.Call<Integer>(1, 2));
  EXPECT_EQ(val.get(), 3);
}

TEST(ObjectTest, Capsule) {
  Object obj;
  int* hundred = new int(100);
  Handle capsule =
      Handle(TaggedValue::Capsule(static_cast<void*>(hundred), [](void* p) {
        delete static_cast<int*>(p);
      }));
  obj.Set(String("hundred"), capsule);
  EXPECT_EQ(*static_cast<int*>(
                obj.Get<internal::Capsule>(String("hundred"))->cast<int*>()),
            100);
}

None AppendIntegerToList(List a, Integer b) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSccPSexperimentalPSlibtfPStestsPSobject_testDTcc mht_1(mht_1_v, 282, "", "./tensorflow/cc/experimental/libtf/tests/object_test.cc", "AppendIntegerToList");

  a.append(b);
  return None();
}
Integer AddIntegersTyped(Integer a, Integer b) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSccPSexperimentalPSlibtfPStestsPSobject_testDTcc mht_2(mht_2_v, 289, "", "./tensorflow/cc/experimental/libtf/tests/object_test.cc", "AddIntegersTyped");

  return Integer(a.get() + b.get());
}
Integer ReturnFive() {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPSccPSexperimentalPSlibtfPStestsPSobject_testDTcc mht_3(mht_3_v, 295, "", "./tensorflow/cc/experimental/libtf/tests/object_test.cc", "ReturnFive");
 return Integer(5); }

TEST(TypeUneraseCallTest, TestCallable) {
  // Add two integers.
  Callable add(TFLIB_CALLABLE_ADAPTOR(AddIntegersTyped));
  auto res = add.Call<Integer>(Integer(3), Integer(1));
  EXPECT_EQ(res->get(), 4);
}

TEST(TypeUneraseCallTest, TestAppend) {
  // Append some indices to a list.
  Callable append(TFLIB_CALLABLE_ADAPTOR(AppendIntegerToList));
  List l;
  TF_ASSERT_OK(append.Call<None>(l, Integer(3)).status());
  TF_ASSERT_OK(append.Call<None>(l, Integer(6)).status());
  EXPECT_EQ(l.size(), 2);
  EXPECT_EQ(l.Get<Integer>(0)->get(), 3);
  EXPECT_EQ(l.Get<Integer>(1)->get(), 6);
}

TEST(TypeUneraseCallTest, TestCallableWrongArgs) {
  // Try variants of wrong argument types.
  Callable append(TFLIB_CALLABLE_ADAPTOR(AddIntegersTyped));
  ASSERT_FALSE(append.Call<None>(Object(), Integer(3)).ok());
  ASSERT_FALSE(append.Call<None>(Object(), Object()).ok());
  // Try variants of wrong numbers of arguments.
  ASSERT_FALSE(append.Call().ok());
  ASSERT_FALSE(append.Call(Integer(3)).ok());
  ASSERT_FALSE(append.Call(Integer(3), Integer(4), Integer(5)).ok());
}

Handle Polymorph(Handle a) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPSccPSexperimentalPSlibtfPStestsPSobject_testDTcc mht_4(mht_4_v, 329, "", "./tensorflow/cc/experimental/libtf/tests/object_test.cc", "Polymorph");

  auto i = Cast<Integer>(a);
  if (i.ok()) {
    return Integer(i->get() * 2);
  }
  auto f = Cast<Float>(a);
  if (f.ok()) {
    return Float(f->get() * 2.f);
  }
  return None();
}

TEST(TypeUneraseCallTest, TestCallableGeneric) {
  Callable f(TFLIB_CALLABLE_ADAPTOR(Polymorph));
  EXPECT_EQ(f.Call<Float>(Float(.2))->get(), .4f);
  EXPECT_EQ(Cast<Float>(*f.Call(Float(.2)))->get(), .4f);
  EXPECT_EQ(f.Call<Integer>(Integer(3))->get(), 6);
}

TEST(TypeUneraseCallTest, TestLambda) {
  // Test a trivial lambda that doubles an integer.
  Callable c(
      TFLIB_CALLABLE_ADAPTOR([](Integer a) { return Integer(a.get() * 2); }));
  EXPECT_EQ(c.Call<Integer>(Integer(3))->get(), 6);
  // Testa lambda that has captured state (call count).
  int call_count = 0;
  Callable f(TFLIB_CALLABLE_ADAPTOR([&call_count](Integer a, Integer b) {
    call_count++;
    return Integer(a.get() + b.get());
  }));
  EXPECT_EQ(f.Call<Integer>(Integer(3), Integer(-1))->get(), 2);
  EXPECT_EQ(f.Call<Integer>(Integer(3), Integer(-3))->get(), 0);
  EXPECT_EQ(call_count, 2);
}

}  // namespace libtf
}  // namespace tf
