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
class MHTracer_DTPStensorflowPScorePSlibPSgtlPSint_type_testDTcc {
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
   MHTracer_DTPStensorflowPScorePSlibPSgtlPSint_type_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSlibPSgtlPSint_type_testDTcc() {
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

// Unit test cases for IntType.

#include "tensorflow/core/lib/gtl/int_type.h"

#include <memory>
#include <unordered_map>

#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {

TF_LIB_GTL_DEFINE_INT_TYPE(Int8_IT, int8);
TF_LIB_GTL_DEFINE_INT_TYPE(UInt8_IT, uint8);
TF_LIB_GTL_DEFINE_INT_TYPE(Int16_IT, int16);
TF_LIB_GTL_DEFINE_INT_TYPE(UInt16_IT, uint16);
TF_LIB_GTL_DEFINE_INT_TYPE(Int32_IT, int32);
TF_LIB_GTL_DEFINE_INT_TYPE(Int64_IT, int64_t);
TF_LIB_GTL_DEFINE_INT_TYPE(UInt32_IT, uint32);
TF_LIB_GTL_DEFINE_INT_TYPE(UInt64_IT, uint64);
TF_LIB_GTL_DEFINE_INT_TYPE(Long_IT, long);  // NOLINT

template <typename IntType_Type>
class IntTypeTest : public ::testing::Test {};

// All tests below will be executed on all supported IntTypes.
typedef ::testing::Types<Int8_IT, UInt8_IT, Int16_IT, UInt16_IT, Int32_IT,
                         Int64_IT, UInt64_IT, Long_IT>
    SupportedIntTypes;

TYPED_TEST_SUITE(IntTypeTest, SupportedIntTypes);

TYPED_TEST(IntTypeTest, TestInitialization) {
  constexpr TypeParam a;
  constexpr TypeParam b(1);
  constexpr TypeParam c(b);
  EXPECT_EQ(0, a);  // default initialization to 0
  EXPECT_EQ(1, b);
  EXPECT_EQ(1, c);
}

TYPED_TEST(IntTypeTest, TestOperators) {
  TypeParam a(0);
  TypeParam b(1);
  TypeParam c(2);
  constexpr TypeParam d(3);
  constexpr TypeParam e(4);

  // On all EXPECT_EQ below, we use the accessor value() as to not invoke the
  // comparison operators which must themselves be tested.

  // -- UNARY OPERATORS --------------------------------------------------------
  EXPECT_EQ(0, (a++).value());
  EXPECT_EQ(2, (++a).value());
  EXPECT_EQ(2, (a--).value());
  EXPECT_EQ(0, (--a).value());

  EXPECT_EQ(true, !a);
  EXPECT_EQ(false, !b);
  static_assert(!d == false, "Unary operator! failed");

  EXPECT_EQ(a.value(), +a);
  static_assert(+d == d.value(), "Unary operator+ failed");
  EXPECT_EQ(-a.value(), -a);
  static_assert(-d == -d.value(), "Unary operator- failed");
  EXPECT_EQ(~a.value(), ~a);  // ~zero
  EXPECT_EQ(~b.value(), ~b);  // ~non-zero
  static_assert(~d == ~d.value(), "Unary operator~ failed");

  // -- ASSIGNMENT OPERATORS ---------------------------------------------------
  // We test all assignment operators using IntType and constant as arguments.
  // We also test the return from the operators.
  // From same IntType
  c = a = b;
  EXPECT_EQ(1, a.value());
  EXPECT_EQ(1, c.value());
  // From constant
  c = b = 2;
  EXPECT_EQ(2, b.value());
  EXPECT_EQ(2, c.value());
  // From same IntType
  c = a += b;
  EXPECT_EQ(3, a.value());
  EXPECT_EQ(3, c.value());
  c = a -= b;
  EXPECT_EQ(1, a.value());
  EXPECT_EQ(1, c.value());
  c = a *= b;
  EXPECT_EQ(2, a.value());
  EXPECT_EQ(2, c.value());
  c = a /= b;
  EXPECT_EQ(1, a.value());
  EXPECT_EQ(1, c.value());
  c = a <<= b;
  EXPECT_EQ(4, a.value());
  EXPECT_EQ(4, c.value());
  c = a >>= b;
  EXPECT_EQ(1, a.value());
  EXPECT_EQ(1, c.value());
  c = a %= b;
  EXPECT_EQ(1, a.value());
  EXPECT_EQ(1, c.value());
  // From constant
  c = a += 2;
  EXPECT_EQ(3, a.value());
  EXPECT_EQ(3, c.value());
  c = a -= 2;
  EXPECT_EQ(1, a.value());
  EXPECT_EQ(1, c.value());
  c = a *= 2;
  EXPECT_EQ(2, a.value());
  EXPECT_EQ(2, c.value());
  c = a /= 2;
  EXPECT_EQ(1, a.value());
  EXPECT_EQ(1, c.value());
  c = a <<= 2;
  EXPECT_EQ(4, a.value());
  EXPECT_EQ(4, c.value());
  c = a >>= 2;
  EXPECT_EQ(1, a.value());
  EXPECT_EQ(1, c.value());
  c = a %= 2;
  EXPECT_EQ(1, a.value());
  EXPECT_EQ(1, c.value());

  // -- COMPARISON OPERATORS ---------------------------------------------------
  a = 0;
  b = 1;

  EXPECT_FALSE(a == b);
  EXPECT_TRUE(a == 0);   // NOLINT
  EXPECT_FALSE(1 == a);  // NOLINT
  static_assert(d == d, "operator== failed");
  static_assert(d == 3, "operator== failed");
  static_assert(3 == d, "operator== failed");
  EXPECT_TRUE(a != b);
  EXPECT_TRUE(a != 1);   // NOLINT
  EXPECT_FALSE(0 != a);  // NOLINT
  static_assert(d != e, "operator!= failed");
  static_assert(d != 4, "operator!= failed");
  static_assert(4 != d, "operator!= failed");
  EXPECT_TRUE(a < b);
  EXPECT_TRUE(a < 1);   // NOLINT
  EXPECT_FALSE(0 < a);  // NOLINT
  static_assert(d < e, "operator< failed");
  static_assert(d < 4, "operator< failed");
  static_assert(3 < e, "operator< failed");
  EXPECT_TRUE(a <= b);
  EXPECT_TRUE(a <= 1);  // NOLINT
  EXPECT_TRUE(0 <= a);  // NOLINT
  static_assert(d <= e, "operator<= failed");
  static_assert(d <= 4, "operator<= failed");
  static_assert(3 <= e, "operator<= failed");
  EXPECT_FALSE(a > b);
  EXPECT_FALSE(a > 1);  // NOLINT
  EXPECT_FALSE(0 > a);  // NOLINT
  static_assert(e > d, "operator> failed");
  static_assert(e > 3, "operator> failed");
  static_assert(4 > d, "operator> failed");
  EXPECT_FALSE(a >= b);
  EXPECT_FALSE(a >= 1);  // NOLINT
  EXPECT_TRUE(0 >= a);   // NOLINT
  static_assert(e >= d, "operator>= failed");
  static_assert(e >= 3, "operator>= failed");
  static_assert(4 >= d, "operator>= failed");

  // -- BINARY OPERATORS -------------------------------------------------------
  a = 1;
  b = 3;
  EXPECT_EQ(4, (a + b).value());
  EXPECT_EQ(4, (a + 3).value());
  EXPECT_EQ(4, (1 + b).value());
  static_assert((d + e).value() == 7, "Binary operator+ failed");
  static_assert((d + 4).value() == 7, "Binary operator+ failed");
  static_assert((3 + e).value() == 7, "Binary operator+ failed");
  EXPECT_EQ(2, (b - a).value());
  EXPECT_EQ(2, (b - 1).value());
  EXPECT_EQ(2, (3 - a).value());
  static_assert((e - d).value() == 1, "Binary operator- failed");
  static_assert((e - 3).value() == 1, "Binary operator- failed");
  static_assert((4 - d).value() == 1, "Binary operator- failed");
  EXPECT_EQ(3, (a * b).value());
  EXPECT_EQ(3, (a * 3).value());
  EXPECT_EQ(3, (1 * b).value());
  static_assert((d * e).value() == 12, "Binary operator* failed");
  static_assert((d * 4).value() == 12, "Binary operator* failed");
  static_assert((3 * e).value() == 12, "Binary operator* failed");
  EXPECT_EQ(0, (a / b).value());
  EXPECT_EQ(0, (a / 3).value());
  EXPECT_EQ(0, (1 / b).value());
  static_assert((d / e).value() == 0, "Binary operator/ failed");
  static_assert((d / 4).value() == 0, "Binary operator/ failed");
  static_assert((3 / e).value() == 0, "Binary operator/ failed");
  EXPECT_EQ(8, (a << b).value());
  EXPECT_EQ(8, (a << 3).value());
  EXPECT_EQ(8, (1 << b).value());
  static_assert((d << e).value() == 48, "Binary operator<< failed");
  static_assert((d << 4).value() == 48, "Binary operator<< failed");
  static_assert((3 << e).value() == 48, "Binary operator<< failed");
  b = 8;
  EXPECT_EQ(4, (b >> a).value());
  EXPECT_EQ(4, (b >> 1).value());
  EXPECT_EQ(4, (8 >> a).value());
  static_assert((d >> e).value() == 0, "Binary operator>> failed");
  static_assert((d >> 4).value() == 0, "Binary operator>> failed");
  static_assert((3 >> e).value() == 0, "Binary operator>> failed");
  b = 3;
  a = 2;
  EXPECT_EQ(1, (b % a).value());
  EXPECT_EQ(1, (b % 2).value());
  EXPECT_EQ(1, (3 % a).value());
  static_assert((e % d).value() == 1, "Binary operator% failed");
  static_assert((e % 3).value() == 1, "Binary operator% failed");
  static_assert((4 % d).value() == 1, "Binary operator% failed");
}

TYPED_TEST(IntTypeTest, TestHashFunctor) {
  std::unordered_map<TypeParam, char, typename TypeParam::Hasher> map;
  TypeParam a(0);
  map[a] = 'c';
  EXPECT_EQ('c', map[a]);
  map[++a] = 'o';
  EXPECT_EQ('o', map[a]);

  TypeParam b(a);
  EXPECT_EQ(typename TypeParam::Hasher()(a), typename TypeParam::Hasher()(b));
}

// Tests the use of the templatized value accessor that performs static_casts.
// We use -1 to force casting in unsigned integers.
TYPED_TEST(IntTypeTest, TestValueAccessor) {
  constexpr typename TypeParam::ValueType i = -1;
  constexpr TypeParam int_type(i);
  EXPECT_EQ(i, int_type.value());
  static_assert(int_type.value() == i, "value() failed");
  // The use of the keyword 'template' (suggested by Clang) is only necessary
  // as this code is part of a template class.  Weird syntax though.  Good news
  // is that only int_type.value<int>() is needed in most code.
  EXPECT_EQ(static_cast<int>(i), int_type.template value<int>());
  EXPECT_EQ(static_cast<int8>(i), int_type.template value<int8>());
  EXPECT_EQ(static_cast<int16>(i), int_type.template value<int16>());
  EXPECT_EQ(static_cast<int32>(i), int_type.template value<int32>());
  EXPECT_EQ(static_cast<uint32>(i), int_type.template value<uint32>());
  EXPECT_EQ(static_cast<int64_t>(i), int_type.template value<int64_t>());
  EXPECT_EQ(static_cast<uint64>(i), int_type.template value<uint64>());
  EXPECT_EQ(static_cast<long>(i), int_type.template value<long>());  // NOLINT
  static_assert(int_type.template value<int>() == static_cast<int>(i),
                "value<Value>() failed");
}

TYPED_TEST(IntTypeTest, TestMove) {
  // Check that the int types have move constructor/assignment.
  // We do this by composing a struct with an int type and a unique_ptr. This
  // struct can't be copied due to the unique_ptr, so it must be moved.
  // If this compiles, it means that the int types have move operators.
  struct NotCopyable {
    TypeParam inttype;
    std::unique_ptr<int> ptr;

    static NotCopyable Make(int i) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSlibPSgtlPSint_type_testDTcc mht_0(mht_0_v, 444, "", "./tensorflow/core/lib/gtl/int_type_test.cc", "Make");

      NotCopyable f;
      f.inttype = TypeParam(i);
      f.ptr.reset(new int(i));
      return f;
    }
  };

  // Test move constructor.
  NotCopyable foo = NotCopyable::Make(123);
  EXPECT_EQ(123, foo.inttype);
  EXPECT_EQ(123, *foo.ptr);

  // Test move assignment.
  foo = NotCopyable::Make(321);
  EXPECT_EQ(321, foo.inttype);
  EXPECT_EQ(321, *foo.ptr);
}

}  // namespace tensorflow
