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
class MHTracer_DTPStensorflowPScorePSframeworkPStensor_testutilDTcc {
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
   MHTracer_DTPStensorflowPScorePSframeworkPStensor_testutilDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSframeworkPStensor_testutilDTcc() {
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

#include "tensorflow/core/framework/tensor_testutil.h"

#include <cmath>

#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {
namespace test {

::testing::AssertionResult IsSameType(const Tensor& x, const Tensor& y) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSframeworkPStensor_testutilDTcc mht_0(mht_0_v, 195, "", "./tensorflow/core/framework/tensor_testutil.cc", "IsSameType");

  if (x.dtype() != y.dtype()) {
    return ::testing::AssertionFailure()
           << "Tensors have different dtypes (" << x.dtype() << " vs "
           << y.dtype() << ")";
  }
  return ::testing::AssertionSuccess();
}

::testing::AssertionResult IsSameShape(const Tensor& x, const Tensor& y) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSframeworkPStensor_testutilDTcc mht_1(mht_1_v, 207, "", "./tensorflow/core/framework/tensor_testutil.cc", "IsSameShape");

  if (!x.IsSameSize(y)) {
    return ::testing::AssertionFailure()
           << "Tensors have different shapes (" << x.shape().DebugString()
           << " vs " << y.shape().DebugString() << ")";
  }
  return ::testing::AssertionSuccess();
}

template <typename T>
static ::testing::AssertionResult EqualFailure(const T& x, const T& y) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSframeworkPStensor_testutilDTcc mht_2(mht_2_v, 220, "", "./tensorflow/core/framework/tensor_testutil.cc", "EqualFailure");

  return ::testing::AssertionFailure()
         << std::setprecision(std::numeric_limits<T>::digits10 + 2) << x
         << " not equal to " << y;
}
static ::testing::AssertionResult IsEqual(float x, float y, Tolerance t) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSframeworkPStensor_testutilDTcc mht_3(mht_3_v, 228, "", "./tensorflow/core/framework/tensor_testutil.cc", "IsEqual");

  // We consider NaNs equal for testing.
  if (Eigen::numext::isnan(x) && Eigen::numext::isnan(y))
    return ::testing::AssertionSuccess();
  if (t == Tolerance::kNone) {
    if (x == y) return ::testing::AssertionSuccess();
  } else {
    if (::testing::internal::CmpHelperFloatingPointEQ<float>("", "", x, y))
      return ::testing::AssertionSuccess();
  }
  return EqualFailure(x, y);
}
static ::testing::AssertionResult IsEqual(double x, double y, Tolerance t) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSframeworkPStensor_testutilDTcc mht_4(mht_4_v, 243, "", "./tensorflow/core/framework/tensor_testutil.cc", "IsEqual");

  // We consider NaNs equal for testing.
  if (Eigen::numext::isnan(x) && Eigen::numext::isnan(y))
    return ::testing::AssertionSuccess();
  if (t == Tolerance::kNone) {
    if (x == y) return ::testing::AssertionSuccess();
  } else {
    if (::testing::internal::CmpHelperFloatingPointEQ<double>("", "", x, y))
      return ::testing::AssertionSuccess();
  }
  return EqualFailure(x, y);
}
static ::testing::AssertionResult IsEqual(Eigen::half x, Eigen::half y,
                                          Tolerance t) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSframeworkPStensor_testutilDTcc mht_5(mht_5_v, 259, "", "./tensorflow/core/framework/tensor_testutil.cc", "IsEqual");

  // We consider NaNs equal for testing.
  if (Eigen::numext::isnan(x) && Eigen::numext::isnan(y))
    return ::testing::AssertionSuccess();

  // Below is a reimplementation of CmpHelperFloatingPointEQ<Eigen::half>, which
  // we cannot use because Eigen::half is not default-constructible.

  if (Eigen::numext::isnan(x) || Eigen::numext::isnan(y))
    return EqualFailure(x, y);

  auto sign_and_magnitude_to_biased = [](uint16_t sam) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePSframeworkPStensor_testutilDTcc mht_6(mht_6_v, 273, "", "./tensorflow/core/framework/tensor_testutil.cc", "lambda");

    const uint16_t kSignBitMask = 0x8000;
    if (kSignBitMask & sam) return ~sam + 1;  // negative number.
    return kSignBitMask | sam;                // positive number.
  };

  auto xb = sign_and_magnitude_to_biased(Eigen::numext::bit_cast<uint16_t>(x));
  auto yb = sign_and_magnitude_to_biased(Eigen::numext::bit_cast<uint16_t>(y));
  if (t == Tolerance::kNone) {
    if (xb == yb) return ::testing::AssertionSuccess();
  } else {
    auto distance = xb >= yb ? xb - yb : yb - xb;
    const uint16_t kMaxUlps = 4;
    if (distance <= kMaxUlps) return ::testing::AssertionSuccess();
  }
  return EqualFailure(x, y);
}
template <typename T>
static ::testing::AssertionResult IsEqual(const T& x, const T& y, Tolerance t) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePSframeworkPStensor_testutilDTcc mht_7(mht_7_v, 294, "", "./tensorflow/core/framework/tensor_testutil.cc", "IsEqual");

  if (::testing::internal::CmpHelperEQ<T>("", "", x, y))
    return ::testing::AssertionSuccess();
  return EqualFailure(x, y);
}
template <typename T>
static ::testing::AssertionResult IsEqual(const std::complex<T>& x,
                                          const std::complex<T>& y,
                                          Tolerance t) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScorePSframeworkPStensor_testutilDTcc mht_8(mht_8_v, 305, "", "./tensorflow/core/framework/tensor_testutil.cc", "IsEqual");

  if (IsEqual(x.real(), y.real(), t) && IsEqual(x.imag(), y.imag(), t))
    return ::testing::AssertionSuccess();
  return EqualFailure(x, y);
}

template <typename T>
static void ExpectEqual(const Tensor& x, const Tensor& y,
                        Tolerance t = Tolerance::kDefault) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScorePSframeworkPStensor_testutilDTcc mht_9(mht_9_v, 316, "", "./tensorflow/core/framework/tensor_testutil.cc", "ExpectEqual");

  const T* Tx = x.unaligned_flat<T>().data();
  const T* Ty = y.unaligned_flat<T>().data();
  auto size = x.NumElements();
  int max_failures = 10;
  int num_failures = 0;
  for (decltype(size) i = 0; i < size; ++i) {
    EXPECT_TRUE(IsEqual(Tx[i], Ty[i], t)) << "i = " << (++num_failures, i);
    ASSERT_LT(num_failures, max_failures) << "Too many mismatches, giving up.";
  }
}

template <typename T>
static ::testing::AssertionResult IsClose(const T& x, const T& y, const T& atol,
                                          const T& rtol) {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScorePSframeworkPStensor_testutilDTcc mht_10(mht_10_v, 333, "", "./tensorflow/core/framework/tensor_testutil.cc", "IsClose");

  // We consider NaNs equal for testing.
  if (Eigen::numext::isnan(x) && Eigen::numext::isnan(y))
    return ::testing::AssertionSuccess();
  if (x == y) return ::testing::AssertionSuccess();  // Handle infinity.
  auto tolerance = atol + rtol * Eigen::numext::abs(x);
  if (Eigen::numext::abs(x - y) <= tolerance)
    return ::testing::AssertionSuccess();
  return ::testing::AssertionFailure() << x << " not close to " << y;
}

template <typename T>
static ::testing::AssertionResult IsClose(const std::complex<T>& x,
                                          const std::complex<T>& y,
                                          const T& atol, const T& rtol) {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScorePSframeworkPStensor_testutilDTcc mht_11(mht_11_v, 350, "", "./tensorflow/core/framework/tensor_testutil.cc", "IsClose");

  if (IsClose(x.real(), y.real(), atol, rtol) &&
      IsClose(x.imag(), y.imag(), atol, rtol))
    return ::testing::AssertionSuccess();
  return ::testing::AssertionFailure() << x << " not close to " << y;
}

// Return type can be different from T, e.g. float for T=std::complex<float>.
template <typename T>
static auto GetTolerance(double tolerance) {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScorePSframeworkPStensor_testutilDTcc mht_12(mht_12_v, 362, "", "./tensorflow/core/framework/tensor_testutil.cc", "GetTolerance");

  using Real = typename Eigen::NumTraits<T>::Real;
  auto default_tol = static_cast<Real>(5.0) * Eigen::NumTraits<T>::epsilon();
  auto result = tolerance < 0.0 ? default_tol : static_cast<Real>(tolerance);
  EXPECT_GE(result, static_cast<Real>(0));
  return result;
}

template <typename T>
static void ExpectClose(const Tensor& x, const Tensor& y, double atol,
                        double rtol) {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPScorePSframeworkPStensor_testutilDTcc mht_13(mht_13_v, 375, "", "./tensorflow/core/framework/tensor_testutil.cc", "ExpectClose");

  auto typed_atol = GetTolerance<T>(atol);
  auto typed_rtol = GetTolerance<T>(rtol);

  const T* Tx = x.unaligned_flat<T>().data();
  const T* Ty = y.unaligned_flat<T>().data();
  auto size = x.NumElements();
  int max_failures = 10;
  int num_failures = 0;
  for (decltype(size) i = 0; i < size; ++i) {
    EXPECT_TRUE(IsClose(Tx[i], Ty[i], typed_atol, typed_rtol))
        << "i = " << (++num_failures, i) << " Tx[i] = " << Tx[i]
        << " Ty[i] = " << Ty[i];
    ASSERT_LT(num_failures, max_failures)
        << "Too many mismatches (atol = " << atol << " rtol = " << rtol
        << "), giving up.";
  }
  EXPECT_EQ(num_failures, 0)
      << "Mismatches detected (atol = " << atol << " rtol = " << rtol << ").";
}

void ExpectEqual(const Tensor& x, const Tensor& y, Tolerance t) {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPScorePSframeworkPStensor_testutilDTcc mht_14(mht_14_v, 399, "", "./tensorflow/core/framework/tensor_testutil.cc", "ExpectEqual");

  ASSERT_TRUE(IsSameType(x, y));
  ASSERT_TRUE(IsSameShape(x, y));

  switch (x.dtype()) {
    case DT_FLOAT:
      return ExpectEqual<float>(x, y, t);
    case DT_DOUBLE:
      return ExpectEqual<double>(x, y, t);
    case DT_INT32:
      return ExpectEqual<int32>(x, y);
    case DT_UINT32:
      return ExpectEqual<uint32>(x, y);
    case DT_UINT16:
      return ExpectEqual<uint16>(x, y);
    case DT_UINT8:
      return ExpectEqual<uint8>(x, y);
    case DT_INT16:
      return ExpectEqual<int16>(x, y);
    case DT_INT8:
      return ExpectEqual<int8>(x, y);
    case DT_STRING:
      return ExpectEqual<tstring>(x, y);
    case DT_COMPLEX64:
      return ExpectEqual<complex64>(x, y, t);
    case DT_COMPLEX128:
      return ExpectEqual<complex128>(x, y, t);
    case DT_INT64:
      return ExpectEqual<int64_t>(x, y);
    case DT_UINT64:
      return ExpectEqual<uint64>(x, y);
    case DT_BOOL:
      return ExpectEqual<bool>(x, y);
    case DT_QINT8:
      return ExpectEqual<qint8>(x, y);
    case DT_QUINT8:
      return ExpectEqual<quint8>(x, y);
    case DT_QINT16:
      return ExpectEqual<qint16>(x, y);
    case DT_QUINT16:
      return ExpectEqual<quint16>(x, y);
    case DT_QINT32:
      return ExpectEqual<qint32>(x, y);
    case DT_BFLOAT16:
      return ExpectEqual<bfloat16>(x, y, t);
    case DT_HALF:
      return ExpectEqual<Eigen::half>(x, y, t);
    default:
      EXPECT_TRUE(false) << "Unsupported type : " << DataTypeString(x.dtype());
  }
}

void ExpectClose(const Tensor& x, const Tensor& y, double atol, double rtol) {
   std::vector<std::string> mht_15_v;
   MHTracer_DTPStensorflowPScorePSframeworkPStensor_testutilDTcc mht_15(mht_15_v, 454, "", "./tensorflow/core/framework/tensor_testutil.cc", "ExpectClose");

  ASSERT_TRUE(IsSameType(x, y));
  ASSERT_TRUE(IsSameShape(x, y));

  switch (x.dtype()) {
    case DT_HALF:
      return ExpectClose<Eigen::half>(x, y, atol, rtol);
    case DT_BFLOAT16:
      return ExpectClose<Eigen::bfloat16>(x, y, atol, rtol);
    case DT_FLOAT:
      return ExpectClose<float>(x, y, atol, rtol);
    case DT_DOUBLE:
      return ExpectClose<double>(x, y, atol, rtol);
    case DT_COMPLEX64:
      return ExpectClose<complex64>(x, y, atol, rtol);
    case DT_COMPLEX128:
      return ExpectClose<complex128>(x, y, atol, rtol);
    default:
      EXPECT_TRUE(false) << "Unsupported type : " << DataTypeString(x.dtype());
  }
}

::testing::AssertionResult internal_test::IsClose(Eigen::half x, Eigen::half y,
                                                  double atol, double rtol) {
   std::vector<std::string> mht_16_v;
   MHTracer_DTPStensorflowPScorePSframeworkPStensor_testutilDTcc mht_16(mht_16_v, 480, "", "./tensorflow/core/framework/tensor_testutil.cc", "internal_test::IsClose");

  return test::IsClose(x, y, GetTolerance<Eigen::half>(atol),
                       GetTolerance<Eigen::half>(rtol));
}
::testing::AssertionResult internal_test::IsClose(float x, float y, double atol,
                                                  double rtol) {
   std::vector<std::string> mht_17_v;
   MHTracer_DTPStensorflowPScorePSframeworkPStensor_testutilDTcc mht_17(mht_17_v, 488, "", "./tensorflow/core/framework/tensor_testutil.cc", "internal_test::IsClose");

  return test::IsClose(x, y, GetTolerance<float>(atol),
                       GetTolerance<float>(rtol));
}
::testing::AssertionResult internal_test::IsClose(double x, double y,
                                                  double atol, double rtol) {
   std::vector<std::string> mht_18_v;
   MHTracer_DTPStensorflowPScorePSframeworkPStensor_testutilDTcc mht_18(mht_18_v, 496, "", "./tensorflow/core/framework/tensor_testutil.cc", "internal_test::IsClose");

  return test::IsClose(x, y, GetTolerance<double>(atol),
                       GetTolerance<double>(rtol));
}

}  // end namespace test
}  // end namespace tensorflow
