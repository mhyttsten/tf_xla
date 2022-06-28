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

#ifndef TENSORFLOW_COMPILER_XLA_TESTS_TEST_UTILS_H_
#define TENSORFLOW_COMPILER_XLA_TESTS_TEST_UTILS_H_
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
class MHTracer_DTPStensorflowPScompilerPSxlaPStestsPStest_utilsDTh {
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
   MHTracer_DTPStensorflowPScompilerPSxlaPStestsPStest_utilsDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSxlaPStestsPStest_utilsDTh() {
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


#include <initializer_list>
#include <memory>
#include <random>

#include "absl/memory/memory.h"
#include "absl/types/span.h"
#include "tensorflow/compiler/xla/layout_util.h"
#include "tensorflow/compiler/xla/literal.h"
#include "tensorflow/compiler/xla/service/hlo_instructions.h"
#include "tensorflow/compiler/xla/service/hlo_module.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"

namespace xla {

// A class which generates pseudorandom numbers of a given type within a given
// range. Not cryptographically secure and likely not perfectly evenly
// distributed across the range but sufficient for most tests.
template <typename NativeT>
class PseudorandomGenerator {
 public:
  explicit PseudorandomGenerator(NativeT min_value, NativeT max_value,
                                 uint32_t seed)
      : min_(min_value), max_(max_value), generator_(seed) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPStestsPStest_utilsDTh mht_0(mht_0_v, 210, "", "./tensorflow/compiler/xla/tests/test_utils.h", "PseudorandomGenerator");
}

  // Get a pseudorandom value.
  NativeT get() {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPStestsPStest_utilsDTh mht_1(mht_1_v, 216, "", "./tensorflow/compiler/xla/tests/test_utils.h", "get");

    std::uniform_real_distribution<> distribution;
    return static_cast<NativeT>(min_ +
                                (max_ - min_) * distribution(generator_));
  }

 private:
  NativeT min_;
  NativeT max_;
  std::mt19937 generator_;
};

// Generates fake data in a literal of the given shape, or returns an error
// status if the element type is currently unhandled for fake data
// generation. See below for documentation of pseudo_random and use_large_range.
StatusOr<Literal> MakeFakeLiteral(const Shape& shape, bool pseudo_random = true,
                                  bool use_large_range = false);

// Generates a vector of arguments containing fake data. The number, shape and
// layout of the arguments is appropriate for given HLO module.
//
// A best-effort attempt is made to generate the data in a way which produce
// stable computation results across platforms. Specifically:
//
//  (1) Init values of reductions should be the identity of the reduction
//  computation.
//
//  (2) Indices of dynamic slices and update slices should be in bounds.
//
//  (3) Keys of key/value sorts should contain no duplicates.
//
// These constraints are best-effort only.
//
// If pseudo_random is true, the generated numbers will be generated
// deterministically in a pseudo random way unless the values are constrated to
// be e.g. init values as above. If pseudo_random is false, the returned values
// will be generated in a faster way that yields less interesting data, e.g. the
// values may all be just the same value.
//
// If use_large_range is false, the generated floating point numbers will be
// sampled from a small range of possible values. If use_large_range is true,
// the generated floating point numbers will be sampled from a uniform-log
// distribution of most possible floats, with a small chance to instead be
// sampled from a list of special floating point values (such as 0, inf, etc.).
//
// TODO(b/79942829): Make interesting argument generation fast enough that using
// pseudo_random does not save any noticeable amount of time so that the
// parameter can be removed.
StatusOr<std::vector<Literal>> MakeFakeArguments(HloModule* const module,
                                                 bool pseudo_random = true,
                                                 bool use_large_range = false);

// Overload which accepts a random number generator. This enables generation of
// different random values with sequential calls to MakeFakeArguments by reusing
// the same generator.
StatusOr<std::vector<Literal>> MakeFakeArguments(HloModule* const module,
                                                 std::minstd_rand0* engine,
                                                 bool use_large_range = false);

// Check that a given module satisfies various constraints before trying to
// execute it.
Status VerifyHloModule(HloModule* const module, bool layout_sensitive,
                       bool allow_mixed_precision);

// Creates a dot op with operands 'lhs' and 'rhs' that contracts dimension 1 of
// the LHS with dimension 0 of the RHS with no batch dimensions.
// Both LHS and the RHS must be of rank 2.
std::unique_ptr<HloDotInstruction> CreateCanonicalDot(const Shape& shape,
                                                      HloInstruction* lhs,
                                                      HloInstruction* rhs);
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_TESTS_TEST_UTILS_H_
