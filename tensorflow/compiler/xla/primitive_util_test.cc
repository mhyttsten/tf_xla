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
class MHTracer_DTPStensorflowPScompilerPSxlaPSprimitive_util_testDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSxlaPSprimitive_util_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSxlaPSprimitive_util_testDTcc() {
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

/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/primitive_util.h"

#include <numeric>
#include <string>

#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/compiler/xla/test.h"
#include "tensorflow/compiler/xla/test_helpers.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"

namespace xla {
namespace {

TEST(PrimitiveUtilTest, StringToPrimitiveType) {
  auto expect_ok_and_equal = [](const std::string& str,
                                PrimitiveType expected) {
   std::vector<std::string> mht_0_v;
   mht_0_v.push_back("str: \"" + str + "\"");
   MHTracer_DTPStensorflowPScompilerPSxlaPSprimitive_util_testDTcc mht_0(mht_0_v, 203, "", "./tensorflow/compiler/xla/primitive_util_test.cc", "lambda");

    TF_ASSERT_OK_AND_ASSIGN(PrimitiveType actual,
                            primitive_util::StringToPrimitiveType(str));
    EXPECT_EQ(expected, actual);
  };
  expect_ok_and_equal("f32", F32);
  expect_ok_and_equal("tuple", TUPLE);
  expect_ok_and_equal("pred", PRED);
  expect_ok_and_equal("s32", S32);

  EXPECT_IS_NOT_OK(primitive_util::StringToPrimitiveType("F32").status());
  EXPECT_IS_NOT_OK(primitive_util::StringToPrimitiveType("Pred").status());
  EXPECT_IS_NOT_OK(primitive_util::StringToPrimitiveType("preD").status());
}

TEST(PrimitiveUtilTest, FloatTypes) {
  EXPECT_EQ(primitive_util::SignificandWidth(F32), 24);
  EXPECT_EQ(primitive_util::SignificandWidth(BF16), 8);
  EXPECT_EQ(primitive_util::ExponentWidth(F32), 8);
  EXPECT_EQ(primitive_util::ExponentWidth(BF16), 8);
}

TEST(PrimitiveUtilTest, CastPreservesValues) {
  bool expecteds[PrimitiveType_ARRAYSIZE][PrimitiveType_ARRAYSIZE];
  expecteds[PRED][PRED] = true;
  expecteds[PRED][S8] = true;
  expecteds[PRED][S16] = true;
  expecteds[PRED][S32] = true;
  expecteds[PRED][S64] = true;
  expecteds[PRED][U8] = true;
  expecteds[PRED][U16] = true;
  expecteds[PRED][U32] = true;
  expecteds[PRED][U64] = true;
  expecteds[PRED][F16] = true;
  expecteds[PRED][F32] = true;
  expecteds[PRED][F64] = true;
  expecteds[PRED][C64] = true;
  expecteds[PRED][BF16] = true;
  expecteds[PRED][C128] = true;
  expecteds[S8][PRED] = false;
  expecteds[S8][S8] = true;
  expecteds[S8][S16] = true;
  expecteds[S8][S32] = true;
  expecteds[S8][S64] = true;
  expecteds[S8][U8] = false;
  expecteds[S8][U16] = false;
  expecteds[S8][U32] = false;
  expecteds[S8][U64] = false;
  expecteds[S8][F16] = true;
  expecteds[S8][F32] = true;
  expecteds[S8][F64] = true;
  expecteds[S8][C64] = true;
  expecteds[S8][BF16] = true;
  expecteds[S8][C128] = true;
  expecteds[S16][PRED] = false;
  expecteds[S16][S8] = false;
  expecteds[S16][S16] = true;
  expecteds[S16][S32] = true;
  expecteds[S16][S64] = true;
  expecteds[S16][U8] = false;
  expecteds[S16][U16] = false;
  expecteds[S16][U32] = false;
  expecteds[S16][U64] = false;
  expecteds[S16][F16] = false;
  expecteds[S16][F32] = true;
  expecteds[S16][F64] = true;
  expecteds[S16][C64] = true;
  expecteds[S16][BF16] = false;
  expecteds[S16][C128] = true;
  expecteds[S32][PRED] = false;
  expecteds[S32][S8] = false;
  expecteds[S32][S16] = false;
  expecteds[S32][S32] = true;
  expecteds[S32][S64] = true;
  expecteds[S32][U8] = false;
  expecteds[S32][U16] = false;
  expecteds[S32][U32] = false;
  expecteds[S32][U64] = false;
  expecteds[S32][F16] = false;
  expecteds[S32][F32] = false;
  expecteds[S32][F64] = true;
  expecteds[S32][C64] = false;
  expecteds[S32][BF16] = false;
  expecteds[S32][C128] = true;
  expecteds[S64][PRED] = false;
  expecteds[S64][S8] = false;
  expecteds[S64][S16] = false;
  expecteds[S64][S32] = false;
  expecteds[S64][S64] = true;
  expecteds[S64][U8] = false;
  expecteds[S64][U16] = false;
  expecteds[S64][U32] = false;
  expecteds[S64][U64] = false;
  expecteds[S64][F16] = false;
  expecteds[S64][F32] = false;
  expecteds[S64][F64] = false;
  expecteds[S64][C64] = false;
  expecteds[S64][BF16] = false;
  expecteds[S64][C128] = false;
  expecteds[U8][PRED] = false;
  expecteds[U8][S8] = false;
  expecteds[U8][S16] = true;
  expecteds[U8][S32] = true;
  expecteds[U8][S64] = true;
  expecteds[U8][U8] = true;
  expecteds[U8][U16] = true;
  expecteds[U8][U32] = true;
  expecteds[U8][U64] = true;
  expecteds[U8][F16] = true;
  expecteds[U8][F32] = true;
  expecteds[U8][F64] = true;
  expecteds[U8][C64] = true;
  expecteds[U8][BF16] = true;
  expecteds[U8][C128] = true;
  expecteds[U16][PRED] = false;
  expecteds[U16][S8] = false;
  expecteds[U16][S16] = false;
  expecteds[U16][S32] = true;
  expecteds[U16][S64] = true;
  expecteds[U16][U8] = false;
  expecteds[U16][U16] = true;
  expecteds[U16][U32] = true;
  expecteds[U16][U64] = true;
  expecteds[U16][F16] = false;
  expecteds[U16][F32] = true;
  expecteds[U16][F64] = true;
  expecteds[U16][C64] = true;
  expecteds[U16][BF16] = false;
  expecteds[U16][C128] = true;
  expecteds[U32][PRED] = false;
  expecteds[U32][S8] = false;
  expecteds[U32][S16] = false;
  expecteds[U32][S32] = false;
  expecteds[U32][S64] = true;
  expecteds[U32][U8] = false;
  expecteds[U32][U16] = false;
  expecteds[U32][U32] = true;
  expecteds[U32][U64] = true;
  expecteds[U32][F16] = false;
  expecteds[U32][F32] = false;
  expecteds[U32][F64] = true;
  expecteds[U32][C64] = false;
  expecteds[U32][BF16] = false;
  expecteds[U32][C128] = true;
  expecteds[U64][PRED] = false;
  expecteds[U64][S8] = false;
  expecteds[U64][S16] = false;
  expecteds[U64][S32] = false;
  expecteds[U64][S64] = false;
  expecteds[U64][U8] = false;
  expecteds[U64][U16] = false;
  expecteds[U64][U32] = false;
  expecteds[U64][U64] = true;
  expecteds[U64][F16] = false;
  expecteds[U64][F32] = false;
  expecteds[U64][F64] = false;
  expecteds[U64][C64] = false;
  expecteds[U64][BF16] = false;
  expecteds[U64][C128] = false;
  expecteds[F16][PRED] = false;
  expecteds[F16][S8] = false;
  expecteds[F16][S16] = false;
  expecteds[F16][S32] = false;
  expecteds[F16][S64] = false;
  expecteds[F16][U8] = false;
  expecteds[F16][U16] = false;
  expecteds[F16][U32] = false;
  expecteds[F16][U64] = false;
  expecteds[F16][F16] = true;
  expecteds[F16][F32] = true;
  expecteds[F16][F64] = true;
  expecteds[F16][C64] = true;
  expecteds[F16][BF16] = false;
  expecteds[F16][C128] = true;
  expecteds[F32][PRED] = false;
  expecteds[F32][S8] = false;
  expecteds[F32][S16] = false;
  expecteds[F32][S32] = false;
  expecteds[F32][S64] = false;
  expecteds[F32][U8] = false;
  expecteds[F32][U16] = false;
  expecteds[F32][U32] = false;
  expecteds[F32][U64] = false;
  expecteds[F32][F16] = false;
  expecteds[F32][F32] = true;
  expecteds[F32][F64] = true;
  expecteds[F32][C64] = true;
  expecteds[F32][BF16] = false;
  expecteds[F32][C128] = true;
  expecteds[F64][PRED] = false;
  expecteds[F64][S8] = false;
  expecteds[F64][S16] = false;
  expecteds[F64][S32] = false;
  expecteds[F64][S64] = false;
  expecteds[F64][U8] = false;
  expecteds[F64][U16] = false;
  expecteds[F64][U32] = false;
  expecteds[F64][U64] = false;
  expecteds[F64][F16] = false;
  expecteds[F64][F32] = false;
  expecteds[F64][F64] = true;
  expecteds[F64][C64] = false;
  expecteds[F64][BF16] = false;
  expecteds[F64][C128] = true;
  expecteds[C64][PRED] = false;
  expecteds[C64][S8] = false;
  expecteds[C64][S16] = false;
  expecteds[C64][S32] = false;
  expecteds[C64][S64] = false;
  expecteds[C64][U8] = false;
  expecteds[C64][U16] = false;
  expecteds[C64][U32] = false;
  expecteds[C64][U64] = false;
  expecteds[C64][F16] = false;
  expecteds[C64][F32] = false;
  expecteds[C64][F64] = false;
  expecteds[C64][C64] = true;
  expecteds[C64][BF16] = false;
  expecteds[C64][C128] = true;
  expecteds[BF16][PRED] = false;
  expecteds[BF16][S8] = false;
  expecteds[BF16][S16] = false;
  expecteds[BF16][S32] = false;
  expecteds[BF16][S64] = false;
  expecteds[BF16][U8] = false;
  expecteds[BF16][U16] = false;
  expecteds[BF16][U32] = false;
  expecteds[BF16][U64] = false;
  expecteds[BF16][F16] = false;
  expecteds[BF16][F32] = true;
  expecteds[BF16][F64] = true;
  expecteds[BF16][C64] = true;
  expecteds[BF16][BF16] = true;
  expecteds[BF16][C128] = true;
  expecteds[C128][PRED] = false;
  expecteds[C128][S8] = false;
  expecteds[C128][S16] = false;
  expecteds[C128][S32] = false;
  expecteds[C128][S64] = false;
  expecteds[C128][U8] = false;
  expecteds[C128][U16] = false;
  expecteds[C128][U32] = false;
  expecteds[C128][U64] = false;
  expecteds[C128][F16] = false;
  expecteds[C128][F32] = false;
  expecteds[C128][F64] = false;
  expecteds[C128][C64] = false;
  expecteds[C128][BF16] = false;
  expecteds[C128][C128] = true;

  for (int from_type_int = PrimitiveType_MIN;
       from_type_int < PrimitiveType_ARRAYSIZE; ++from_type_int) {
    auto from_type = static_cast<PrimitiveType>(from_type_int);
    if (!primitive_util::IsArrayType(from_type)) {
      continue;
    }
    for (int to_type_int = PrimitiveType_MIN;
         to_type_int < PrimitiveType_ARRAYSIZE; ++to_type_int) {
      auto to_type = static_cast<PrimitiveType>(to_type_int);
      if (!primitive_util::IsArrayType(to_type)) {
        continue;
      }
      bool expected = expecteds[from_type][to_type];
      bool actual = primitive_util::CastPreservesValues(from_type, to_type);
      EXPECT_EQ(expected, actual)
          << primitive_util::LowercasePrimitiveTypeName(from_type) << " -> "
          << primitive_util::LowercasePrimitiveTypeName(to_type);
    }
  }
}

}  // namespace
}  // namespace xla
