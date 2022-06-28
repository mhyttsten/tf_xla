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
class MHTracer_DTPStensorflowPScorePSplatformPSdenormal_testDTcc {
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
   MHTracer_DTPStensorflowPScorePSplatformPSdenormal_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSplatformPSdenormal_testDTcc() {
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
// Testing configuration of denormal state.
#include "tensorflow/core/platform/denormal.h"

#include <cstring>
#include <limits>

#include "tensorflow/core/platform/test.h"

namespace tensorflow {
namespace port {

TEST(DenormalStateTest, ConstructorAndAccessorsWork) {
  const bool flush_to_zero[] = {true, true, false, false};
  const bool denormals_are_zero[] = {true, false, true, false};
  for (int i = 0; i < 4; ++i) {
    const DenormalState state =
        DenormalState(flush_to_zero[i], denormals_are_zero[i]);
    EXPECT_EQ(state.flush_to_zero(), flush_to_zero[i]);
    EXPECT_EQ(state.denormals_are_zero(), denormals_are_zero[i]);
  }
}

// Convert a 32-bit float to its binary representation.
uint32_t bits(float x) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSplatformPSdenormal_testDTcc mht_0(mht_0_v, 207, "", "./tensorflow/core/platform/denormal_test.cc", "bits");

  uint32_t out;
  memcpy(&out, &x, sizeof(float));
  return out;
}

void CheckDenormalHandling(const DenormalState& state) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSplatformPSdenormal_testDTcc mht_1(mht_1_v, 216, "", "./tensorflow/core/platform/denormal_test.cc", "CheckDenormalHandling");

  // Notes:
  //  - In the following tests we need to compare binary representations because
  //    floating-point comparisons can trigger denormal flushing on SSE/ARM.
  //  - We also require the input value to be marked `volatile` to prevent the
  //    compiler from optimizing away any floating-point operations that might
  //    otherwise be expected to flush denormals.

  // The following is zero iff denormal outputs are flushed to zero.
  volatile float denormal_output = std::numeric_limits<float>::min();
  denormal_output *= 0.25f;
  if (state.flush_to_zero()) {
    EXPECT_EQ(bits(denormal_output), 0x0);
  } else {
    EXPECT_NE(bits(denormal_output), 0x0);
  }

  // The following is zero iff denormal inputs are flushed to zero.
  volatile float normal_output = std::numeric_limits<float>::denorm_min();
  normal_output *= std::numeric_limits<float>::max();
  if (state.denormals_are_zero()) {
    EXPECT_EQ(bits(normal_output), 0x0);
  } else {
    EXPECT_NE(bits(normal_output), 0x0);
  }
}

TEST(DenormalTest, GetAndSetStateWorkWithCorrectFlushing) {
  const DenormalState states[] = {
      DenormalState(/*flush_to_zero=*/true, /*denormals_are_zero=*/true),
      DenormalState(/*flush_to_zero=*/true, /*denormals_are_zero=*/false),
      DenormalState(/*flush_to_zero=*/false, /*denormals_are_zero=*/true),
      DenormalState(/*flush_to_zero=*/false, /*denormals_are_zero=*/false)};

  for (const DenormalState& state : states) {
    if (SetDenormalState(state)) {
      EXPECT_EQ(GetDenormalState(), state);
      CheckDenormalHandling(state);
    }
  }
}

TEST(ScopedRestoreFlushDenormalStateTest, RestoresState) {
  const DenormalState flush_denormals(/*flush_to_zero=*/true,
                                      /*denormals_are_zero=*/true);
  const DenormalState dont_flush_denormals(/*flush_to_zero=*/false,
                                           /*denormals_are_zero=*/false);

  // Only test if the platform supports setting the denormal state.
  const bool can_set_denormal_state = SetDenormalState(flush_denormals) &&
                                      SetDenormalState(dont_flush_denormals);
  if (can_set_denormal_state) {
    // Flush -> Don't Flush -> Flush.
    SetDenormalState(flush_denormals);
    {
      ScopedRestoreFlushDenormalState restore_state;
      SetDenormalState(dont_flush_denormals);
      EXPECT_EQ(GetDenormalState(), dont_flush_denormals);
    }
    EXPECT_EQ(GetDenormalState(), flush_denormals);

    // Don't Flush -> Flush -> Don't Flush.
    SetDenormalState(dont_flush_denormals);
    {
      ScopedRestoreFlushDenormalState restore_state;
      SetDenormalState(flush_denormals);
      EXPECT_EQ(GetDenormalState(), flush_denormals);
    }
    EXPECT_EQ(GetDenormalState(), dont_flush_denormals);
  }
}

TEST(ScopedFlushDenormalTest, SetsFlushingAndRestoresState) {
  const DenormalState flush_denormals(/*flush_to_zero=*/true,
                                      /*denormals_are_zero=*/true);
  const DenormalState dont_flush_denormals(/*flush_to_zero=*/false,
                                           /*denormals_are_zero=*/false);

  // Only test if the platform supports setting the denormal state.
  const bool can_set_denormal_state = SetDenormalState(flush_denormals) &&
                                      SetDenormalState(dont_flush_denormals);
  if (can_set_denormal_state) {
    SetDenormalState(dont_flush_denormals);
    {
      ScopedFlushDenormal scoped_flush_denormal;
      EXPECT_EQ(GetDenormalState(), flush_denormals);
    }
    EXPECT_EQ(GetDenormalState(), dont_flush_denormals);
  }
}

TEST(ScopedDontFlushDenormalTest, SetsNoFlushingAndRestoresState) {
  const DenormalState flush_denormals(/*flush_to_zero=*/true,
                                      /*denormals_are_zero=*/true);
  const DenormalState dont_flush_denormals(/*flush_to_zero=*/false,
                                           /*denormals_are_zero=*/false);

  // Only test if the platform supports setting the denormal state.
  const bool can_set_denormal_state = SetDenormalState(flush_denormals) &&
                                      SetDenormalState(dont_flush_denormals);
  if (can_set_denormal_state) {
    SetDenormalState(flush_denormals);
    {
      ScopedDontFlushDenormal scoped_dont_flush_denormal;
      EXPECT_EQ(GetDenormalState(), dont_flush_denormals);
    }
    EXPECT_EQ(GetDenormalState(), flush_denormals);
  }
}

}  // namespace port
}  // namespace tensorflow
