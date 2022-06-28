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

#ifndef TENSORFLOW_CORE_FRAMEWORK_TENSOR_TESTUTIL_H_
#define TENSORFLOW_CORE_FRAMEWORK_TENSOR_TESTUTIL_H_
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
class MHTracer_DTPStensorflowPScorePSframeworkPStensor_testutilDTh {
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
   MHTracer_DTPStensorflowPScorePSframeworkPStensor_testutilDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSframeworkPStensor_testutilDTh() {
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


#include <numeric>

#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/lib/gtl/array_slice.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {
namespace test {

// Constructs a scalar tensor with 'val'.
template <typename T>
Tensor AsScalar(const T& val) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSframeworkPStensor_testutilDTh mht_0(mht_0_v, 200, "", "./tensorflow/core/framework/tensor_testutil.h", "AsScalar");

  Tensor ret(DataTypeToEnum<T>::value, {});
  ret.scalar<T>()() = val;
  return ret;
}

// Constructs a flat tensor with 'vals'.
template <typename T>
Tensor AsTensor(gtl::ArraySlice<T> vals) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSframeworkPStensor_testutilDTh mht_1(mht_1_v, 211, "", "./tensorflow/core/framework/tensor_testutil.h", "AsTensor");

  Tensor ret(DataTypeToEnum<T>::value, {static_cast<int64_t>(vals.size())});
  std::copy_n(vals.data(), vals.size(), ret.flat<T>().data());
  return ret;
}

// Constructs a tensor of "shape" with values "vals".
template <typename T>
Tensor AsTensor(gtl::ArraySlice<T> vals, const TensorShape& shape) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSframeworkPStensor_testutilDTh mht_2(mht_2_v, 222, "", "./tensorflow/core/framework/tensor_testutil.h", "AsTensor");

  Tensor ret;
  CHECK(ret.CopyFrom(AsTensor(vals), shape));
  return ret;
}

// Fills in '*tensor' with 'vals'. E.g.,
//   Tensor x(&alloc, DT_FLOAT, TensorShape({2, 2}));
//   test::FillValues<float>(&x, {11, 21, 21, 22});
template <typename T>
void FillValues(Tensor* tensor, gtl::ArraySlice<T> vals) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSframeworkPStensor_testutilDTh mht_3(mht_3_v, 235, "", "./tensorflow/core/framework/tensor_testutil.h", "FillValues");

  auto flat = tensor->flat<T>();
  CHECK_EQ(flat.size(), vals.size());
  if (flat.size() > 0) {
    std::copy_n(vals.data(), vals.size(), flat.data());
  }
}

// Fills in '*tensor' with 'vals', converting the types as needed.
template <typename T, typename SrcType>
void FillValues(Tensor* tensor, std::initializer_list<SrcType> vals) {
  auto flat = tensor->flat<T>();
  CHECK_EQ(flat.size(), vals.size());
  if (flat.size() > 0) {
    size_t i = 0;
    for (auto itr = vals.begin(); itr != vals.end(); ++itr, ++i) {
      flat(i) = T(*itr);
    }
  }
}

// Fills in '*tensor' with a sequence of value of val, val+1, val+2, ...
//   Tensor x(&alloc, DT_FLOAT, TensorShape({2, 2}));
//   test::FillIota<float>(&x, 1.0);
template <typename T>
void FillIota(Tensor* tensor, const T& val) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSframeworkPStensor_testutilDTh mht_4(mht_4_v, 263, "", "./tensorflow/core/framework/tensor_testutil.h", "FillIota");

  auto flat = tensor->flat<T>();
  std::iota(flat.data(), flat.data() + flat.size(), val);
}

// Fills in '*tensor' with a sequence of value of fn(0), fn(1), ...
//   Tensor x(&alloc, DT_FLOAT, TensorShape({2, 2}));
//   test::FillFn<float>(&x, [](int i)->float { return i*i; });
template <typename T>
void FillFn(Tensor* tensor, std::function<T(int)> fn) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSframeworkPStensor_testutilDTh mht_5(mht_5_v, 275, "", "./tensorflow/core/framework/tensor_testutil.h", "FillFn");

  auto flat = tensor->flat<T>();
  for (int i = 0; i < flat.size(); ++i) flat(i) = fn(i);
}

// Expects "x" and "y" are tensors of the same type, same shape, and identical
// values (within 4 ULPs for floating point types unless explicitly disabled).
enum class Tolerance {
  kNone,
  kDefault,
};
void ExpectEqual(const Tensor& x, const Tensor& y,
                 Tolerance t = Tolerance ::kDefault);

// Expects "x" and "y" are tensors of the same (floating point) type,
// same shape and element-wise difference between x and y is no more
// than atol + rtol * abs(x). If atol or rtol is negative, the data type's
// epsilon * kSlackFactor is used.
void ExpectClose(const Tensor& x, const Tensor& y, double atol = -1.0,
                 double rtol = -1.0);

// Expects "x" and "y" are tensors of the same type T, same shape, and
// equal values. Consider using ExpectEqual above instead.
template <typename T>
void ExpectTensorEqual(const Tensor& x, const Tensor& y) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePSframeworkPStensor_testutilDTh mht_6(mht_6_v, 302, "", "./tensorflow/core/framework/tensor_testutil.h", "ExpectTensorEqual");

  EXPECT_EQ(x.dtype(), DataTypeToEnum<T>::value);
  ExpectEqual(x, y);
}

::testing::AssertionResult IsSameType(const Tensor& x, const Tensor& y);
::testing::AssertionResult IsSameShape(const Tensor& x, const Tensor& y);

template <typename T>
void ExpectTensorEqual(const Tensor& x, const Tensor& y,
                       std::function<bool(const T&, const T&)> is_equal) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePSframeworkPStensor_testutilDTh mht_7(mht_7_v, 315, "", "./tensorflow/core/framework/tensor_testutil.h", "ExpectTensorEqual");

  EXPECT_EQ(x.dtype(), DataTypeToEnum<T>::value);
  ASSERT_TRUE(IsSameType(x, y));
  ASSERT_TRUE(IsSameShape(x, y));

  const T* Tx = x.unaligned_flat<T>().data();
  const T* Ty = y.unaligned_flat<T>().data();
  auto size = x.NumElements();
  int max_failures = 10;
  int num_failures = 0;
  for (decltype(size) i = 0; i < size; ++i) {
    EXPECT_TRUE(is_equal(Tx[i], Ty[i])) << "i = " << (++num_failures, i);
    ASSERT_LT(num_failures, max_failures) << "Too many mismatches, giving up.";
  }
}

// Expects "x" and "y" are tensors of the same type T, same shape, and
// approximate equal values. Consider using ExpectClose above instead.
template <typename T>
void ExpectTensorNear(const Tensor& x, const Tensor& y, double atol) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScorePSframeworkPStensor_testutilDTh mht_8(mht_8_v, 337, "", "./tensorflow/core/framework/tensor_testutil.h", "ExpectTensorNear");

  EXPECT_EQ(x.dtype(), DataTypeToEnum<T>::value);
  ExpectClose(x, y, atol, /*rtol=*/0.0);
}

// For tensor_testutil_test only.
namespace internal_test {
::testing::AssertionResult IsClose(Eigen::half x, Eigen::half y,
                                   double atol = -1.0, double rtol = -1.0);
::testing::AssertionResult IsClose(float x, float y, double atol = -1.0,
                                   double rtol = -1.0);
::testing::AssertionResult IsClose(double x, double y, double atol = -1.0,
                                   double rtol = -1.0);
}  // namespace internal_test

}  // namespace test
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_FRAMEWORK_TENSOR_TESTUTIL_H_
