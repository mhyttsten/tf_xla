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
class MHTracer_DTPStensorflowPSccPSexperimentalPSlibtfPStestsPSvalue_testDTcc {
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
   MHTracer_DTPStensorflowPSccPSexperimentalPSlibtfPStestsPSvalue_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSccPSexperimentalPSlibtfPStestsPSvalue_testDTcc() {
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
#include "tensorflow/cc/experimental/libtf/value.h"

#include <cstdint>

#include "tensorflow/cc/experimental/libtf/value_iostream.h"
#include "tensorflow/core/platform/test.h"

namespace tf {
namespace libtf {
namespace impl {

TEST(ValueTest, TestBasic) {
  TaggedValue valuef(3.f);
  TaggedValue valuei(int64_t(3));
  TaggedValue list = TaggedValue::List();
  TaggedValue tuple = TaggedValue::Tuple();
  tuple.tuple().push_back(TaggedValue(int64_t(310)));
  list.list().push_back(valuei);
  list.list().push_back(valuef);
  list.list().push_back(tuple);
  std::stringstream stream;
  stream << list;
  ASSERT_EQ(stream.str(), "[3, 3, (310, ), ]");
}

TEST(ValueTest, TestString) {
  TaggedValue value1a("string1");
  std::string s = "string";
  s += "1";
  TaggedValue value1b(s.c_str());
  // Verify that interned the pointers are the same.
  ASSERT_EQ(value1b.s(), value1a.s());
  TaggedValue value2("string2");
  ASSERT_NE(value1a.s(), value2.s());
  ASSERT_STREQ(value1a.s(), "string1");
  ASSERT_STREQ(value2.s(), "string2");
}

TEST(Test1, TestDict) {
  TaggedValue s1("test1");
  TaggedValue s2("test2");
  TaggedValue d = TaggedValue::Dict();
  d.dict()[s2] = TaggedValue(6.f);
  std::stringstream stream;
  stream << d;
  ASSERT_EQ(stream.str(), "{test2: 6, }");
}

namespace {
TaggedValue add(TaggedValue args, TaggedValue kwargs) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSccPSexperimentalPSlibtfPStestsPSvalue_testDTcc mht_0(mht_0_v, 233, "", "./tensorflow/cc/experimental/libtf/tests/value_test.cc", "add");

  if (args.type() == TaggedValue::TUPLE) {
    return TaggedValue(args.tuple()[0].f32() + args.tuple()[1].f32());
  }
  return TaggedValue::None();
}
}  // namespace
TEST(Test1, TestFunctionCall) {
  TaggedValue f32 = TaggedValue(add);
  TaggedValue args = TaggedValue::Tuple();
  args.tuple().emplace_back(TaggedValue(1.f));
  args.tuple().emplace_back(TaggedValue(2.f));
  TaggedValue c = f32.func()(args, TaggedValue::None()).ValueOrDie();
  ASSERT_EQ(c, TaggedValue(3.f));
}

namespace {
int alloc_count = 0;
class Cool {
 public:
  Cool() {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSccPSexperimentalPSlibtfPStestsPSvalue_testDTcc mht_1(mht_1_v, 256, "", "./tensorflow/cc/experimental/libtf/tests/value_test.cc", "Cool");
 alloc_count++; }
  ~Cool() {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSccPSexperimentalPSlibtfPStestsPSvalue_testDTcc mht_2(mht_2_v, 260, "", "./tensorflow/cc/experimental/libtf/tests/value_test.cc", "~Cool");
 alloc_count--; }
};
}  // namespace

TEST(Test1, TestCapsule) {
  TaggedValue test_moved, test_copy;
  ASSERT_EQ(alloc_count, 0);
  void* ptr_value = new Cool();
  {
    TaggedValue capsule =
        TaggedValue::Capsule(static_cast<void*>(ptr_value),
                             [](void* x) { delete static_cast<Cool*>(x); });
    ASSERT_EQ(alloc_count, 1);
    ASSERT_EQ(capsule.capsule(), ptr_value);
    test_moved = std::move(capsule);
    ASSERT_EQ(capsule.type(), TaggedValue::NONE);  // NOLINT
    test_copy = test_moved;
    ASSERT_EQ(test_moved.capsule(), ptr_value);
    ASSERT_EQ(test_copy.capsule(), ptr_value);
  }
  ASSERT_EQ(alloc_count, 1);
  test_moved = TaggedValue::None();
  ASSERT_EQ(alloc_count, 1);
  test_copy = TaggedValue(3.f);
  ASSERT_EQ(alloc_count, 0);
}

}  // namespace impl
}  // namespace libtf
}  // namespace tf
