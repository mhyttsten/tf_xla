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
class MHTracer_DTPStensorflowPScorePSplatformPSstatusor_testDTcc {
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
   MHTracer_DTPStensorflowPScorePSplatformPSstatusor_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSplatformPSstatusor_testDTcc() {
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

/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

// Unit tests for StatusOr

#include "tensorflow/core/platform/statusor.h"

#include <memory>
#include <type_traits>

#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/platform/test_benchmark.h"

namespace tensorflow {
namespace {

class Base1 {
 public:
  virtual ~Base1() {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSplatformPSstatusor_testDTcc mht_0(mht_0_v, 202, "", "./tensorflow/core/platform/statusor_test.cc", "~Base1");
}
  int pad_;
};

class Base2 {
 public:
  virtual ~Base2() {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSplatformPSstatusor_testDTcc mht_1(mht_1_v, 211, "", "./tensorflow/core/platform/statusor_test.cc", "~Base2");
}
  int yetotherpad_;
};

class Derived : public Base1, public Base2 {
 public:
  ~Derived() override {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSplatformPSstatusor_testDTcc mht_2(mht_2_v, 220, "", "./tensorflow/core/platform/statusor_test.cc", "~Derived");
}
  int evenmorepad_;
};

class CopyNoAssign {
 public:
  explicit CopyNoAssign(int value) : foo_(value) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSplatformPSstatusor_testDTcc mht_3(mht_3_v, 229, "", "./tensorflow/core/platform/statusor_test.cc", "CopyNoAssign");
}
  CopyNoAssign(const CopyNoAssign& other) : foo_(other.foo_) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSplatformPSstatusor_testDTcc mht_4(mht_4_v, 233, "", "./tensorflow/core/platform/statusor_test.cc", "CopyNoAssign");
}
  int foo_;

 private:
  const CopyNoAssign& operator=(const CopyNoAssign&);
};

class NoDefaultConstructor {
 public:
  explicit NoDefaultConstructor(int foo);
};

static_assert(!std::is_default_constructible<NoDefaultConstructor>(),
              "Should not be default-constructible.");

StatusOr<std::unique_ptr<int>> ReturnUniquePtr() {
  // Uses implicit constructor from T&&
  return std::unique_ptr<int>(new int(0));
}

TEST(StatusOr, ElementType) {
  static_assert(std::is_same<StatusOr<int>::element_type, int>(), "");
  static_assert(std::is_same<StatusOr<char>::element_type, char>(), "");
}

TEST(StatusOr, NullPointerStatusOr) {
  // As a very special case, null-plain-pointer StatusOr used to be an
  // error. Test that it no longer is.
  StatusOr<int*> null_status(nullptr);
  EXPECT_TRUE(null_status.ok());
  EXPECT_EQ(null_status.ValueOrDie(), nullptr);
}

TEST(StatusOr, TestNoDefaultConstructorInitialization) {
  // Explicitly initialize it with an error code.
  StatusOr<NoDefaultConstructor> statusor(tensorflow::errors::Cancelled(""));
  EXPECT_FALSE(statusor.ok());
  EXPECT_EQ(statusor.status().code(), tensorflow::error::CANCELLED);

  // Default construction of StatusOr initializes it with an UNKNOWN error code.
  StatusOr<NoDefaultConstructor> statusor2;
  EXPECT_FALSE(statusor2.ok());
  EXPECT_EQ(statusor2.status().code(), tensorflow::error::UNKNOWN);
}

TEST(StatusOr, TestMoveOnlyInitialization) {
  StatusOr<std::unique_ptr<int>> thing(ReturnUniquePtr());
  ASSERT_TRUE(thing.ok());
  EXPECT_EQ(0, *thing.ValueOrDie());
  int* previous = thing.ValueOrDie().get();

  thing = ReturnUniquePtr();
  EXPECT_TRUE(thing.ok());
  EXPECT_EQ(0, *thing.ValueOrDie());
  EXPECT_NE(previous, thing.ValueOrDie().get());
}

TEST(StatusOr, TestMoveOnlyStatusCtr) {
  StatusOr<std::unique_ptr<int>> thing(tensorflow::errors::Cancelled(""));
  ASSERT_FALSE(thing.ok());
}

TEST(StatusOr, TestMoveOnlyValueExtraction) {
  StatusOr<std::unique_ptr<int>> thing(ReturnUniquePtr());
  ASSERT_TRUE(thing.ok());
  std::unique_ptr<int> ptr = thing.ConsumeValueOrDie();
  EXPECT_EQ(0, *ptr);

  thing = std::move(ptr);
  ptr = std::move(thing.ValueOrDie());
  EXPECT_EQ(0, *ptr);
}

TEST(StatusOr, TestMoveOnlyConversion) {
  StatusOr<std::unique_ptr<const int>> const_thing(ReturnUniquePtr());
  EXPECT_TRUE(const_thing.ok());
  EXPECT_EQ(0, *const_thing.ValueOrDie());

  // Test rvalue converting assignment
  const int* const_previous = const_thing.ValueOrDie().get();
  const_thing = ReturnUniquePtr();
  EXPECT_TRUE(const_thing.ok());
  EXPECT_EQ(0, *const_thing.ValueOrDie());
  EXPECT_NE(const_previous, const_thing.ValueOrDie().get());
}

TEST(StatusOr, TestMoveOnlyVector) {
  // Sanity check that StatusOr<MoveOnly> works in vector.
  std::vector<StatusOr<std::unique_ptr<int>>> vec;
  vec.push_back(ReturnUniquePtr());
  vec.resize(2);
  auto another_vec = std::move(vec);
  EXPECT_EQ(0, *another_vec[0].ValueOrDie());
  EXPECT_EQ(tensorflow::error::UNKNOWN, another_vec[1].status().code());
}

TEST(StatusOr, TestMoveWithValuesAndErrors) {
  StatusOr<std::string> status_or(std::string(1000, '0'));
  StatusOr<std::string> value1(std::string(1000, '1'));
  StatusOr<std::string> value2(std::string(1000, '2'));
  StatusOr<std::string> error1(Status(tensorflow::error::UNKNOWN, "error1"));
  StatusOr<std::string> error2(Status(tensorflow::error::UNKNOWN, "error2"));

  ASSERT_TRUE(status_or.ok());
  EXPECT_EQ(std::string(1000, '0'), status_or.ValueOrDie());

  // Overwrite the value in status_or with another value.
  status_or = std::move(value1);
  ASSERT_TRUE(status_or.ok());
  EXPECT_EQ(std::string(1000, '1'), status_or.ValueOrDie());

  // Overwrite the value in status_or with an error.
  status_or = std::move(error1);
  ASSERT_FALSE(status_or.ok());
  EXPECT_EQ("error1", status_or.status().error_message());

  // Overwrite the error in status_or with another error.
  status_or = std::move(error2);
  ASSERT_FALSE(status_or.ok());
  EXPECT_EQ("error2", status_or.status().error_message());

  // Overwrite the error with a value.
  status_or = std::move(value2);
  ASSERT_TRUE(status_or.ok());
  EXPECT_EQ(std::string(1000, '2'), status_or.ValueOrDie());
}

TEST(StatusOr, TestCopyWithValuesAndErrors) {
  StatusOr<std::string> status_or(std::string(1000, '0'));
  StatusOr<std::string> value1(std::string(1000, '1'));
  StatusOr<std::string> value2(std::string(1000, '2'));
  StatusOr<std::string> error1(Status(tensorflow::error::UNKNOWN, "error1"));
  StatusOr<std::string> error2(Status(tensorflow::error::UNKNOWN, "error2"));

  ASSERT_TRUE(status_or.ok());
  EXPECT_EQ(std::string(1000, '0'), status_or.ValueOrDie());

  // Overwrite the value in status_or with another value.
  status_or = value1;
  ASSERT_TRUE(status_or.ok());
  EXPECT_EQ(std::string(1000, '1'), status_or.ValueOrDie());

  // Overwrite the value in status_or with an error.
  status_or = error1;
  ASSERT_FALSE(status_or.ok());
  EXPECT_EQ("error1", status_or.status().error_message());

  // Overwrite the error in status_or with another error.
  status_or = error2;
  ASSERT_FALSE(status_or.ok());
  EXPECT_EQ("error2", status_or.status().error_message());

  // Overwrite the error with a value.
  status_or = value2;
  ASSERT_TRUE(status_or.ok());
  EXPECT_EQ(std::string(1000, '2'), status_or.ValueOrDie());

  // Verify original values unchanged.
  EXPECT_EQ(std::string(1000, '1'), value1.ValueOrDie());
  EXPECT_EQ("error1", error1.status().error_message());
  EXPECT_EQ("error2", error2.status().error_message());
  EXPECT_EQ(std::string(1000, '2'), value2.ValueOrDie());
}

TEST(StatusOr, TestDefaultCtor) {
  StatusOr<int> thing;
  EXPECT_FALSE(thing.ok());
  EXPECT_EQ(thing.status().code(), tensorflow::error::UNKNOWN);
}

TEST(StatusOrDeathTest, TestDefaultCtorValue) {
  StatusOr<int> thing;
  EXPECT_DEATH(thing.ValueOrDie(), "");

  const StatusOr<int> thing2;
  EXPECT_DEATH(thing.ValueOrDie(), "");
}

TEST(StatusOr, TestStatusCtor) {
  StatusOr<int> thing(Status(tensorflow::error::CANCELLED, ""));
  EXPECT_FALSE(thing.ok());
  EXPECT_EQ(thing.status().code(), tensorflow::error::CANCELLED);
}

TEST(StatusOr, TestValueCtor) {
  const int kI = 4;
  const StatusOr<int> thing(kI);
  EXPECT_TRUE(thing.ok());
  EXPECT_EQ(kI, thing.ValueOrDie());
}

TEST(StatusOr, TestCopyCtorStatusOk) {
  const int kI = 4;
  const StatusOr<int> original(kI);
  const StatusOr<int> copy(original);
  EXPECT_EQ(copy.status(), original.status());
  EXPECT_EQ(original.ValueOrDie(), copy.ValueOrDie());
}

TEST(StatusOr, TestCopyCtorStatusNotOk) {
  StatusOr<int> original(Status(tensorflow::error::CANCELLED, ""));
  StatusOr<int> copy(original);
  EXPECT_EQ(copy.status(), original.status());
}

TEST(StatusOr, TestCopyCtorNonAssignable) {
  const int kI = 4;
  CopyNoAssign value(kI);
  StatusOr<CopyNoAssign> original(value);
  StatusOr<CopyNoAssign> copy(original);
  EXPECT_EQ(copy.status(), original.status());
  EXPECT_EQ(original.ValueOrDie().foo_, copy.ValueOrDie().foo_);
}

TEST(StatusOr, TestCopyCtorStatusOKConverting) {
  const int kI = 4;
  StatusOr<int> original(kI);
  StatusOr<double> copy(original);
  EXPECT_EQ(copy.status(), original.status());
  EXPECT_DOUBLE_EQ(original.ValueOrDie(), copy.ValueOrDie());
}

TEST(StatusOr, TestCopyCtorStatusNotOkConverting) {
  StatusOr<int> original(Status(tensorflow::error::CANCELLED, ""));
  StatusOr<double> copy(original);
  EXPECT_EQ(copy.status(), original.status());
}

TEST(StatusOr, TestAssignmentStatusOk) {
  const int kI = 4;
  StatusOr<int> source(kI);
  StatusOr<int> target;
  target = source;
  EXPECT_EQ(target.status(), source.status());
  EXPECT_EQ(source.ValueOrDie(), target.ValueOrDie());
}

TEST(StatusOr, TestAssignmentStatusNotOk) {
  StatusOr<int> source(Status(tensorflow::error::CANCELLED, ""));
  StatusOr<int> target;
  target = source;
  EXPECT_EQ(target.status(), source.status());
}

TEST(StatusOr, TestStatus) {
  StatusOr<int> good(4);
  EXPECT_TRUE(good.ok());
  StatusOr<int> bad(Status(tensorflow::error::CANCELLED, ""));
  EXPECT_FALSE(bad.ok());
  EXPECT_EQ(bad.status(), Status(tensorflow::error::CANCELLED, ""));
}

TEST(StatusOr, TestValue) {
  const int kI = 4;
  StatusOr<int> thing(kI);
  EXPECT_EQ(kI, thing.ValueOrDie());
}

TEST(StatusOr, TestValueConst) {
  const int kI = 4;
  const StatusOr<int> thing(kI);
  EXPECT_EQ(kI, thing.ValueOrDie());
}

TEST(StatusOrDeathTest, TestValueNotOk) {
  StatusOr<int> thing(Status(tensorflow::error::CANCELLED, "cancelled"));
  EXPECT_DEATH(thing.ValueOrDie(), "cancelled");
}

TEST(StatusOrDeathTest, TestValueNotOkConst) {
  const StatusOr<int> thing(Status(tensorflow::error::UNKNOWN, ""));
  EXPECT_DEATH(thing.ValueOrDie(), "");
}

TEST(StatusOr, TestPointerDefaultCtor) {
  StatusOr<int*> thing;
  EXPECT_FALSE(thing.ok());
  EXPECT_EQ(thing.status().code(), tensorflow::error::UNKNOWN);
}

TEST(StatusOrDeathTest, TestPointerDefaultCtorValue) {
  StatusOr<int*> thing;
  EXPECT_DEATH(thing.ValueOrDie(), "");
}

TEST(StatusOr, TestPointerStatusCtor) {
  StatusOr<int*> thing(Status(tensorflow::error::CANCELLED, ""));
  EXPECT_FALSE(thing.ok());
  EXPECT_EQ(thing.status(), Status(tensorflow::error::CANCELLED, ""));
}

TEST(StatusOr, TestPointerValueCtor) {
  const int kI = 4;
  StatusOr<const int*> thing(&kI);
  EXPECT_TRUE(thing.ok());
  EXPECT_EQ(&kI, thing.ValueOrDie());
}

TEST(StatusOr, TestPointerCopyCtorStatusOk) {
  const int kI = 0;
  StatusOr<const int*> original(&kI);
  StatusOr<const int*> copy(original);
  EXPECT_EQ(copy.status(), original.status());
  EXPECT_EQ(original.ValueOrDie(), copy.ValueOrDie());
}

TEST(StatusOr, TestPointerCopyCtorStatusNotOk) {
  StatusOr<int*> original(Status(tensorflow::error::CANCELLED, ""));
  StatusOr<int*> copy(original);
  EXPECT_EQ(copy.status(), original.status());
}

TEST(StatusOr, TestPointerCopyCtorStatusOKConverting) {
  Derived derived;
  StatusOr<Derived*> original(&derived);
  StatusOr<Base2*> copy(original);
  EXPECT_EQ(copy.status(), original.status());
  EXPECT_EQ(static_cast<const Base2*>(original.ValueOrDie()),
            copy.ValueOrDie());
}

TEST(StatusOr, TestPointerCopyCtorStatusNotOkConverting) {
  StatusOr<Derived*> original(Status(tensorflow::error::CANCELLED, ""));
  StatusOr<Base2*> copy(original);
  EXPECT_EQ(copy.status(), original.status());
}

TEST(StatusOr, TestPointerAssignmentStatusOk) {
  const int kI = 0;
  StatusOr<const int*> source(&kI);
  StatusOr<const int*> target;
  target = source;
  EXPECT_EQ(target.status(), source.status());
  EXPECT_EQ(source.ValueOrDie(), target.ValueOrDie());
}

TEST(StatusOr, TestPointerAssignmentStatusNotOk) {
  StatusOr<int*> source(Status(tensorflow::error::CANCELLED, ""));
  StatusOr<int*> target;
  target = source;
  EXPECT_EQ(target.status(), source.status());
}

TEST(StatusOr, TestPointerStatus) {
  const int kI = 0;
  StatusOr<const int*> good(&kI);
  EXPECT_TRUE(good.ok());
  StatusOr<const int*> bad(Status(tensorflow::error::CANCELLED, ""));
  EXPECT_EQ(bad.status(), Status(tensorflow::error::CANCELLED, ""));
}

TEST(StatusOr, TestPointerValue) {
  const int kI = 0;
  StatusOr<const int*> thing(&kI);
  EXPECT_EQ(&kI, thing.ValueOrDie());
}

TEST(StatusOr, TestPointerValueConst) {
  const int kI = 0;
  const StatusOr<const int*> thing(&kI);
  EXPECT_EQ(&kI, thing.ValueOrDie());
}

TEST(StatusOr, TestArrowOperator) {
  StatusOr<std::unique_ptr<int>> uptr = ReturnUniquePtr();
  EXPECT_EQ(*uptr->get(), 0);
}

TEST(StatusOr, TestArrowOperatorNotOk) {
  StatusOr<Base1> error(Status(tensorflow::error::CANCELLED, "cancelled"));
  EXPECT_DEATH(error->pad_++, "cancelled");
}

TEST(StatusOr, TestStarOperator) {
  StatusOr<std::unique_ptr<int>> uptr = ReturnUniquePtr();
  EXPECT_EQ(**uptr, 0);
}

TEST(StatusOr, TestStarOperatorDeath) {
  StatusOr<Base1> error(Status(tensorflow::error::CANCELLED, "cancelled"));
  EXPECT_DEATH(*error, "cancelled");
}

// NOTE(tucker): StatusOr does not support this kind
// of resize op.
// TEST(StatusOr, StatusOrVectorOfUniquePointerCanResize) {
//   using EvilType = std::vector<std::unique_ptr<int>>;
//   static_assert(std::is_copy_constructible<EvilType>::value, "");
//   std::vector<StatusOr<EvilType>> v(5);
//   v.reserve(v.capacity() + 10);
// }

TEST(StatusOrDeathTest, TestPointerValueNotOk) {
  StatusOr<int*> thing(Status(tensorflow::error::CANCELLED, "cancelled"));
  EXPECT_DEATH(thing.ValueOrDie(), "cancelled");
}

TEST(StatusOrDeathTest, TestPointerValueNotOkConst) {
  const StatusOr<int*> thing(Status(tensorflow::error::CANCELLED, "cancelled"));
  EXPECT_DEATH(thing.ValueOrDie(), "cancelled");
}

static StatusOr<int> MakeStatus() { return 100; }
// A factory to help us benchmark the various factory styles. All of
// the factory methods are marked as non-inlineable so as to more
// accurately simulate calling a factory for which you do not have
// visibility of implementation. Similarly, the value_ variable is
// marked volatile to prevent the compiler from getting too clever
// about detecting that the same value is used in all loop iterations.
template <typename T>
class BenchmarkFactory {
 public:
  // Construct a new factory. Allocate an object which will always
  // be the result of the factory methods.
  BenchmarkFactory() : value_(new T) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSplatformPSstatusor_testDTcc mht_5(mht_5_v, 650, "", "./tensorflow/core/platform/statusor_test.cc", "BenchmarkFactory");
}

  // Destroy this factory, including the result value.
  ~BenchmarkFactory() {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePSplatformPSstatusor_testDTcc mht_6(mht_6_v, 656, "", "./tensorflow/core/platform/statusor_test.cc", "~BenchmarkFactory");
 delete value_; }

  // A trivial factory that just returns the value. There is no status
  // object that could be returned to encapsulate an error
  T* TrivialFactory() TF_ATTRIBUTE_NOINLINE { return value_; }

  // A more sophisticated factory, which returns a status to indicate
  // the result of the operation. The factory result is populated into
  // the user provided pointer result.
  Status ArgumentFactory(T** result) TF_ATTRIBUTE_NOINLINE {
    *result = value_;
    return Status::OK();
  }

  Status ArgumentFactoryFail(T** result) TF_ATTRIBUTE_NOINLINE {
    *result = nullptr;
    return Status(tensorflow::error::CANCELLED, "");
  }

  Status ArgumentFactoryFailShortMsg(T** result) TF_ATTRIBUTE_NOINLINE {
    *result = nullptr;
    return Status(::tensorflow::error::INTERNAL, "");
  }

  Status ArgumentFactoryFailLongMsg(T** result) TF_ATTRIBUTE_NOINLINE {
    *result = nullptr;
    return Status(::tensorflow::error::INTERNAL,
                  "a big string of message junk that will never be read");
  }

  // A factory that returns a StatusOr<T*>. If the factory operation
  // is OK, then the StatusOr<T*> will hold a T*. Otherwise, it will
  // hold a status explaining the error.
  StatusOr<T*> StatusOrFactory() TF_ATTRIBUTE_NOINLINE {
    return static_cast<T*>(value_);
  }

  StatusOr<T*> StatusOrFactoryFail() TF_ATTRIBUTE_NOINLINE {
    return Status(tensorflow::error::CANCELLED, "");
  }

  StatusOr<T*> StatusOrFactoryFailShortMsg() TF_ATTRIBUTE_NOINLINE {
    return Status(::tensorflow::error::INTERNAL, "");
  }

  StatusOr<T*> StatusOrFactoryFailLongMsg() TF_ATTRIBUTE_NOINLINE {
    return Status(::tensorflow::error::INTERNAL,
                  "a big string of message junk that will never be read");
  }

 private:
  T* volatile value_;
  TF_DISALLOW_COPY_AND_ASSIGN(BenchmarkFactory);
};

// A simple type we use with the factory.
class BenchmarkType {
 public:
  BenchmarkType() {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePSplatformPSstatusor_testDTcc mht_7(mht_7_v, 717, "", "./tensorflow/core/platform/statusor_test.cc", "BenchmarkType");
}
  virtual ~BenchmarkType() {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScorePSplatformPSstatusor_testDTcc mht_8(mht_8_v, 721, "", "./tensorflow/core/platform/statusor_test.cc", "~BenchmarkType");
}
  virtual void DoWork() TF_ATTRIBUTE_NOINLINE {}

 private:
  TF_DISALLOW_COPY_AND_ASSIGN(BenchmarkType);
};

// Calibrate the amount of time spent just calling DoWork, since each of our
// tests will do this, we can subtract this out of benchmark results.
void BM_CalibrateWorkLoop(::testing::benchmark::State& state) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScorePSplatformPSstatusor_testDTcc mht_9(mht_9_v, 733, "", "./tensorflow/core/platform/statusor_test.cc", "BM_CalibrateWorkLoop");

  BenchmarkFactory<BenchmarkType> factory;
  BenchmarkType* result = factory.TrivialFactory();
  for (auto s : state) {
    if (result != nullptr) {
      result->DoWork();
    }
  }
}
BENCHMARK(BM_CalibrateWorkLoop);

// Measure the time taken to call into the factory, return the value,
// determine that it is OK, and invoke a trivial function.
void BM_TrivialFactory(::testing::benchmark::State& state) {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScorePSplatformPSstatusor_testDTcc mht_10(mht_10_v, 749, "", "./tensorflow/core/platform/statusor_test.cc", "BM_TrivialFactory");

  BenchmarkFactory<BenchmarkType> factory;
  for (auto s : state) {
    BenchmarkType* result = factory.TrivialFactory();
    if (result != nullptr) {
      result->DoWork();
    }
  }
}
BENCHMARK(BM_TrivialFactory);

// Measure the time taken to call into the factory, providing an
// out-param for the result, evaluating the status result and the
// result pointer, and invoking the trivial function.
void BM_ArgumentFactory(::testing::benchmark::State& state) {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScorePSplatformPSstatusor_testDTcc mht_11(mht_11_v, 766, "", "./tensorflow/core/platform/statusor_test.cc", "BM_ArgumentFactory");

  BenchmarkFactory<BenchmarkType> factory;
  for (auto s : state) {
    BenchmarkType* result = nullptr;
    Status status = factory.ArgumentFactory(&result);
    if (status.ok() && result != nullptr) {
      result->DoWork();
    }
  }
}
BENCHMARK(BM_ArgumentFactory);

// Measure the time to use the StatusOr<T*> factory, evaluate the result,
// and invoke the trivial function.
void BM_StatusOrFactory(::testing::benchmark::State& state) {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScorePSplatformPSstatusor_testDTcc mht_12(mht_12_v, 783, "", "./tensorflow/core/platform/statusor_test.cc", "BM_StatusOrFactory");

  BenchmarkFactory<BenchmarkType> factory;
  for (auto s : state) {
    StatusOr<BenchmarkType*> result = factory.StatusOrFactory();
    if (result.ok()) {
      result.ValueOrDie()->DoWork();
    }
  }
}
BENCHMARK(BM_StatusOrFactory);

// Measure the time taken to call into the factory, providing an
// out-param for the result, evaluating the status result and the
// result pointer, and invoking the trivial function.
void BM_ArgumentFactoryFail(::testing::benchmark::State& state) {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPScorePSplatformPSstatusor_testDTcc mht_13(mht_13_v, 800, "", "./tensorflow/core/platform/statusor_test.cc", "BM_ArgumentFactoryFail");

  BenchmarkFactory<BenchmarkType> factory;
  for (auto s : state) {
    BenchmarkType* result = nullptr;
    Status status = factory.ArgumentFactoryFail(&result);
    if (status.ok() && result != nullptr) {
      result->DoWork();
    }
  }
}
BENCHMARK(BM_ArgumentFactoryFail);

// Measure the time to use the StatusOr<T*> factory, evaluate the result,
// and invoke the trivial function.
void BM_StatusOrFactoryFail(::testing::benchmark::State& state) {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPScorePSplatformPSstatusor_testDTcc mht_14(mht_14_v, 817, "", "./tensorflow/core/platform/statusor_test.cc", "BM_StatusOrFactoryFail");

  BenchmarkFactory<BenchmarkType> factory;
  for (auto s : state) {
    StatusOr<BenchmarkType*> result = factory.StatusOrFactoryFail();
    if (result.ok()) {
      result.ValueOrDie()->DoWork();
    }
  }
}
BENCHMARK(BM_StatusOrFactoryFail);

// Measure the time taken to call into the factory, providing an
// out-param for the result, evaluating the status result and the
// result pointer, and invoking the trivial function.
void BM_ArgumentFactoryFailShortMsg(::testing::benchmark::State& state) {
   std::vector<std::string> mht_15_v;
   MHTracer_DTPStensorflowPScorePSplatformPSstatusor_testDTcc mht_15(mht_15_v, 834, "", "./tensorflow/core/platform/statusor_test.cc", "BM_ArgumentFactoryFailShortMsg");

  BenchmarkFactory<BenchmarkType> factory;
  for (auto s : state) {
    BenchmarkType* result = nullptr;
    Status status = factory.ArgumentFactoryFailShortMsg(&result);
    if (status.ok() && result != nullptr) {
      result->DoWork();
    }
  }
}
BENCHMARK(BM_ArgumentFactoryFailShortMsg);

// Measure the time to use the StatusOr<T*> factory, evaluate the result,
// and invoke the trivial function.
void BM_StatusOrFactoryFailShortMsg(::testing::benchmark::State& state) {
   std::vector<std::string> mht_16_v;
   MHTracer_DTPStensorflowPScorePSplatformPSstatusor_testDTcc mht_16(mht_16_v, 851, "", "./tensorflow/core/platform/statusor_test.cc", "BM_StatusOrFactoryFailShortMsg");

  BenchmarkFactory<BenchmarkType> factory;
  for (auto s : state) {
    StatusOr<BenchmarkType*> result = factory.StatusOrFactoryFailShortMsg();
    if (result.ok()) {
      result.ValueOrDie()->DoWork();
    }
  }
}
BENCHMARK(BM_StatusOrFactoryFailShortMsg);

// Measure the time taken to call into the factory, providing an
// out-param for the result, evaluating the status result and the
// result pointer, and invoking the trivial function.
void BM_ArgumentFactoryFailLongMsg(::testing::benchmark::State& state) {
   std::vector<std::string> mht_17_v;
   MHTracer_DTPStensorflowPScorePSplatformPSstatusor_testDTcc mht_17(mht_17_v, 868, "", "./tensorflow/core/platform/statusor_test.cc", "BM_ArgumentFactoryFailLongMsg");

  BenchmarkFactory<BenchmarkType> factory;
  for (auto s : state) {
    BenchmarkType* result = nullptr;
    Status status = factory.ArgumentFactoryFailLongMsg(&result);
    if (status.ok() && result != nullptr) {
      result->DoWork();
    }
  }
}
BENCHMARK(BM_ArgumentFactoryFailLongMsg);

// Measure the time to use the StatusOr<T*> factory, evaluate the result,
// and invoke the trivial function.
void BM_StatusOrFactoryFailLongMsg(::testing::benchmark::State& state) {
   std::vector<std::string> mht_18_v;
   MHTracer_DTPStensorflowPScorePSplatformPSstatusor_testDTcc mht_18(mht_18_v, 885, "", "./tensorflow/core/platform/statusor_test.cc", "BM_StatusOrFactoryFailLongMsg");

  BenchmarkFactory<BenchmarkType> factory;
  for (auto s : state) {
    StatusOr<BenchmarkType*> result = factory.StatusOrFactoryFailLongMsg();
    if (result.ok()) {
      result.ValueOrDie()->DoWork();
    }
  }
}
BENCHMARK(BM_StatusOrFactoryFailLongMsg);

}  // namespace
}  // namespace tensorflow
