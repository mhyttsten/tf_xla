/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_LITE_DELEGATES_GPU_COMMON_TESTING_FEATURE_PARITY_UTILS_H_
#define TENSORFLOW_LITE_DELEGATES_GPU_COMMON_TESTING_FEATURE_PARITY_UTILS_H_
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
class MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStestingPSfeature_parityPSutilsDTh {
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
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStestingPSfeature_parityPSutilsDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStestingPSfeature_parityPSutilsDTh() {
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


#include <stddef.h>

#include <cstdint>
#include <memory>
#include <optional>
#include <ostream>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/types/span.h"
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/model.h"
#include "tensorflow/lite/string_type.h"

namespace tflite {

// These two functions implement usability printing for TfLiteTensor dimensions
// and coordinates. By default dimensions are interpreted depending on the size:
// 1:Linear, 2:HW, 3: HWC, 4:BHWC. If there are more than 4 dimensions,
// absl::nullopt will be returned.
absl::optional<std::string> ShapeToString(TfLiteIntArray* shape);
absl::optional<std::string> CoordinateToString(TfLiteIntArray* shape,
                                               int linear);

template <typename TupleMatcher>
class TensorEqMatcher {
 public:
  TensorEqMatcher(const TupleMatcher& tuple_matcher, const TfLiteTensor& rhs)
      : tuple_matcher_(tuple_matcher), rhs_(rhs) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStestingPSfeature_parityPSutilsDTh mht_0(mht_0_v, 221, "", "./tensorflow/lite/delegates/gpu/common/testing/feature_parity/utils.h", "TensorEqMatcher");
}

  // Make TensorEqMatcher movable only (The copy operations are implicitly
  // deleted).
  TensorEqMatcher(TensorEqMatcher&& other) = default;
  TensorEqMatcher& operator=(TensorEqMatcher&& other) = default;

  template <typename T>
  operator testing::Matcher<T>() const {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStestingPSfeature_parityPSutilsDTh mht_1(mht_1_v, 232, "", "./tensorflow/lite/delegates/gpu/common/testing/feature_parity/utils.h", "testing::Matcher<T>");
  // NOLINT
    return testing::Matcher<T>(new Impl(tuple_matcher_, rhs_));
  }

  class Impl : public testing::MatcherInterface<TfLiteTensor> {
   public:
    typedef ::std::tuple<float, float> InnerMatcherArg;

    Impl(const TupleMatcher& tuple_matcher, const TfLiteTensor& rhs)
        : mono_tuple_matcher_(
              testing::SafeMatcherCast<InnerMatcherArg>(tuple_matcher)),
          rhs_(rhs) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStestingPSfeature_parityPSutilsDTh mht_2(mht_2_v, 246, "", "./tensorflow/lite/delegates/gpu/common/testing/feature_parity/utils.h", "Impl");
}

    // Make Impl movable only (The copy operations are implicitly deleted).
    Impl(Impl&& other) = default;
    Impl& operator=(Impl&& other) = default;

    // Define what gtest framework will print for the Expected field.
    void DescribeTo(std::ostream* os) const override {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStestingPSfeature_parityPSutilsDTh mht_3(mht_3_v, 256, "", "./tensorflow/lite/delegates/gpu/common/testing/feature_parity/utils.h", "DescribeTo");

      std::string shape;
      absl::optional<std::string> result = ShapeToString(rhs_.dims);
      if (result.has_value()) {
        shape = std::move(result.value());
      } else {
        shape = "[error: unsupported number of dimensions]";
      }
      *os << "tensor which has the shape of " << shape
          << ", where each value and its corresponding expected value ";
      mono_tuple_matcher_.DescribeTo(os);
    }

    bool MatchAndExplain(
        TfLiteTensor lhs,
        testing::MatchResultListener* listener) const override {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStestingPSfeature_parityPSutilsDTh mht_4(mht_4_v, 274, "", "./tensorflow/lite/delegates/gpu/common/testing/feature_parity/utils.h", "MatchAndExplain");

      // 1. Check that TfLiteTensor data type is supported.
      // Support for other data types will be added on demand.
      if (lhs.type != kTfLiteFloat32 || rhs_.type != kTfLiteFloat32) {
        *listener << "which data type is not float32, which is not currently "
                     "supported.";
        return false;
      }

      // 2. Check that dimensions' sizes match. Otherwise, we are not able to
      // compare tensors.
      if (lhs.dims->size != rhs_.dims->size) {
        *listener << "which is different from the expected shape of size "
                  << rhs_.dims->size;
        return false;
      }
      // 3. Check that dimensions' values are equal as well. We are not able to
      // compare tensors of different shapes, even if the total elements count
      // matches.
      bool dims_are_equal = true;
      for (int i = 0; i < lhs.dims->size; i++) {
        dims_are_equal &= lhs.dims->data[i] == rhs_.dims->data[i];
      }
      if (!dims_are_equal) {
        std::string shape;
        absl::optional<std::string> result = ShapeToString(rhs_.dims);
        if (result.has_value()) {
          shape = std::move(result.value());
        } else {
          shape = "[error: unsupported number of dimensions]";
        }
        *listener << "which is different from the expected shape " << shape;
        return false;
      }

      // 4. Proceed to data comparison. Iterate through elements as they lay
      // flat. If some pair of elements don't match, deduct the coordinate
      // basing on the dimensions, then return.
      absl::Span<float> lhs_span(lhs.data.f, lhs.bytes / sizeof(float));
      absl::Span<float> rhs_span(rhs_.data.f, rhs_.bytes / sizeof(float));

      auto left = lhs_span.begin();
      auto right = rhs_span.begin();
      for (size_t i = 0; i != lhs_span.size(); ++i, ++left, ++right) {
        if (listener->IsInterested()) {
          testing::StringMatchResultListener inner_listener;
          if (!mono_tuple_matcher_.MatchAndExplain({*left, *right},
                                                   &inner_listener)) {
            *listener << "where the value pair (";
            testing::internal::UniversalPrint(*left, listener->stream());
            *listener << ", ";
            testing::internal::UniversalPrint(*right, listener->stream());
            std::string coordinate;
            absl::optional<std::string> result =
                CoordinateToString(lhs.dims, i);
            if (result.has_value()) {
              coordinate = std::move(result.value());
            } else {
              coordinate = "[error: unsupported number of dimensions]";
            }
            *listener << ") with coordinate " << coordinate << " don't match";
            testing::internal::PrintIfNotEmpty(inner_listener.str(),
                                               listener->stream());
            return false;
          }
        } else {
          if (!mono_tuple_matcher_.Matches({*left, *right})) return false;
        }
      }

      return true;
    }

   private:
    const testing::Matcher<InnerMatcherArg> mono_tuple_matcher_;
    const TfLiteTensor rhs_;
  };

 private:
  const TupleMatcher tuple_matcher_;
  const TfLiteTensor rhs_;
};

// Builds interpreter for a model, allocates tensors.
absl::Status BuildInterpreter(const Model* model,
                              std::unique_ptr<Interpreter>* interpreter);

// Allocates tensors for a given interpreter.
absl::Status AllocateTensors(std::unique_ptr<Interpreter>* interpreter);

// Modifies graph with given delegate.
absl::Status ModifyGraphWithDelegate(std::unique_ptr<Interpreter>* interpreter,
                                     TfLiteDelegate* delegate);

// Initializes inputs with consequent values of some fixed range.
void InitializeInputs(int left, int right,
                      std::unique_ptr<Interpreter>* interpreter);

// Invokes a prebuilt interpreter.
absl::Status Invoke(std::unique_ptr<Interpreter>* interpreter);

// Usability structure, which is used to pass parameters data to parameterized
// tests.
struct TestParams {
  // A gtest name, which will be used for a generated tests.
  std::string name;

  // Function, which returns a TFLite model, associated with this test name.
  std::vector<uint8_t> model;
};

// Defines how the TestParams should be printed into the command line if
// something fails during testing.
std::ostream& operator<<(std::ostream& os, const TestParams& param);

}  // namespace tflite

// Gtest framework uses this function to describe TfLiteTensor if something
// fails. TfLiteTensor is defined in global namespace, same should be done for
// streaming operator.
std::ostream& operator<<(std::ostream& os, const TfLiteTensor& tensor);

// Defines a matcher to compare two TfLiteTensors pointwise using the given
// tuple matcher for comparing their values.
template <typename TupleMatcherT>
inline tflite::TensorEqMatcher<TupleMatcherT> TensorEq(
    const TupleMatcherT& matcher, const TfLiteTensor& rhs) {
  return tflite::TensorEqMatcher<TupleMatcherT>(matcher, rhs);
}

#endif  // TENSORFLOW_LITE_DELEGATES_GPU_COMMON_TESTING_FEATURE_PARITY_UTILS_H_
