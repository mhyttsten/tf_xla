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
class MHTracer_DTPStensorflowPScompilerPSxlaPStestsPSround_trip_transfer_testDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSxlaPStestsPSround_trip_transfer_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSxlaPStestsPSround_trip_transfer_testDTcc() {
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

// Tests transferring literals of various shapes and values in and out of the
// XLA service.

#include <memory>
#include <numeric>
#include <vector>

#include "tensorflow/compiler/xla/array4d.h"
#include "tensorflow/compiler/xla/client/global_data.h"
#include "tensorflow/compiler/xla/client/local_client.h"
#include "tensorflow/compiler/xla/literal.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/compiler/xla/tests/client_library_test_base.h"
#include "tensorflow/compiler/xla/tests/literal_test_util.h"
#include "tensorflow/compiler/xla/tests/test_macros.h"
#include "tensorflow/core/platform/test.h"

namespace xla {
namespace {

class RoundTripTransferTest : public ClientLibraryTestBase {
 protected:
  void RoundTripTest(const Literal& original) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPStestsPSround_trip_transfer_testDTcc mht_0(mht_0_v, 207, "", "./tensorflow/compiler/xla/tests/round_trip_transfer_test.cc", "RoundTripTest");

    std::unique_ptr<GlobalData> data =
        client_->TransferToServer(original).ConsumeValueOrDie();
    Literal result = client_->Transfer(*data).ConsumeValueOrDie();
    EXPECT_TRUE(LiteralTestUtil::Equal(original, result));
  }
};

TEST_F(RoundTripTransferTest, R0S32) {
  RoundTripTest(LiteralUtil::CreateR0<int32_t>(42));
}

TEST_F(RoundTripTransferTest, R0F32) {
  RoundTripTest(LiteralUtil::CreateR0<float>(42.0));
}

TEST_F(RoundTripTransferTest, R1F32_Len0) {
  RoundTripTest(LiteralUtil::CreateR1<float>({}));
}

TEST_F(RoundTripTransferTest, R1F32_Len2) {
  RoundTripTest(LiteralUtil::CreateR1<float>({42.0, 64.0}));
}

TEST_F(RoundTripTransferTest, R1F32_Len256) {
  std::vector<float> values(256);
  std::iota(values.begin(), values.end(), 1.0);
  RoundTripTest(LiteralUtil::CreateR1<float>(values));
}

TEST_F(RoundTripTransferTest, R1F32_Len1024) {
  std::vector<float> values(1024);
  std::iota(values.begin(), values.end(), 1.0);
  RoundTripTest(LiteralUtil::CreateR1<float>(values));
}

TEST_F(RoundTripTransferTest, R1F32_Len1025) {
  std::vector<float> values(1025);
  std::iota(values.begin(), values.end(), 1.0);
  RoundTripTest(LiteralUtil::CreateR1<float>(values));
}

TEST_F(RoundTripTransferTest, R1F32_Len4096) {
  std::vector<float> values(4096);
  std::iota(values.begin(), values.end(), 1.0);
  RoundTripTest(LiteralUtil::CreateR1<float>(values));
}

TEST_F(RoundTripTransferTest, R2F32_Len10x0) {
  RoundTripTest(LiteralUtil::CreateR2FromArray2D<float>(Array2D<float>(10, 0)));
}

TEST_F(RoundTripTransferTest, R2F32_Len2x2) {
  RoundTripTest(LiteralUtil::CreateR2<float>({{42.0, 64.0}, {77.0, 88.0}}));
}

TEST_F(RoundTripTransferTest, R3F32) {
  RoundTripTest(
      LiteralUtil::CreateR3<float>({{{1.0, 2.0}, {1.0, 2.0}, {1.0, 2.0}},
                                    {{3.0, 4.0}, {3.0, 4.0}, {3.0, 4.0}}}));
}

TEST_F(RoundTripTransferTest, R4F32) {
  RoundTripTest(LiteralUtil::CreateR4<float>({{
      {{10, 11, 12, 13}, {14, 15, 16, 17}},
      {{18, 19, 20, 21}, {22, 23, 24, 25}},
      {{26, 27, 28, 29}, {30, 31, 32, 33}},
  }}));
}

TEST_F(RoundTripTransferTest, EmptyTuple) {
  RoundTripTest(LiteralUtil::MakeTuple({}));
}

TEST_F(RoundTripTransferTest, TupleOfR1F32) {
  RoundTripTest(
      LiteralUtil::MakeTupleFromSlices({LiteralUtil::CreateR1<float>({1, 2}),
                                        LiteralUtil::CreateR1<float>({3, 4})}));
}

TEST_F(RoundTripTransferTest, TupleOfR1F32_Len0_Len2) {
  RoundTripTest(
      LiteralUtil::MakeTupleFromSlices({LiteralUtil::CreateR1<float>({}),
                                        LiteralUtil::CreateR1<float>({3, 4})}));
}

TEST_F(RoundTripTransferTest, TupleOfR0F32AndR1S32) {
  RoundTripTest(LiteralUtil::MakeTupleFromSlices(
      {LiteralUtil::CreateR0<float>(1.0), LiteralUtil::CreateR1<int>({2, 3})}));
}

// Below two tests are added to identify the cost of large data transfers.
TEST_F(RoundTripTransferTest, R2F32_Large) {
  RoundTripTest(LiteralUtil::CreateR2F32Linspace(-1.0f, 1.0f, 512, 512));
}

TEST_F(RoundTripTransferTest, R4F32_Large) {
  Array4D<float> array4d(2, 2, 256, 256);
  array4d.FillWithMultiples(1.0f);
  RoundTripTest(LiteralUtil::CreateR4FromArray4D<float>(array4d));
}

}  // namespace
}  // namespace xla
