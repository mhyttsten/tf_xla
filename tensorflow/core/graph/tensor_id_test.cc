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
class MHTracer_DTPStensorflowPScorePSgraphPStensor_id_testDTcc {
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
   MHTracer_DTPStensorflowPScorePSgraphPStensor_id_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSgraphPStensor_id_testDTcc() {
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

#include "tensorflow/core/graph/tensor_id.h"
#include <vector>
#include "tensorflow/core/lib/random/simple_philox.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/platform/test_benchmark.h"

namespace tensorflow {
namespace {

string ParseHelper(const string& n) {
   std::vector<std::string> mht_0_v;
   mht_0_v.push_back("n: \"" + n + "\"");
   MHTracer_DTPStensorflowPScorePSgraphPStensor_id_testDTcc mht_0(mht_0_v, 196, "", "./tensorflow/core/graph/tensor_id_test.cc", "ParseHelper");
 return ParseTensorName(n).ToString(); }

TEST(TensorIdTest, ParseTensorName) {
  EXPECT_EQ(ParseHelper("W1"), "W1:0");
  EXPECT_EQ(ParseHelper("W1:0"), "W1:0");
  EXPECT_EQ(ParseHelper("weights:0"), "weights:0");
  EXPECT_EQ(ParseHelper("W1:1"), "W1:1");
  EXPECT_EQ(ParseHelper("W1:17"), "W1:17");
  EXPECT_EQ(ParseHelper("xyz1_17"), "xyz1_17:0");
  EXPECT_EQ(ParseHelper("^foo"), "^foo");
}

uint32 Skewed(random::SimplePhilox* rnd, int max_log) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSgraphPStensor_id_testDTcc mht_1(mht_1_v, 211, "", "./tensorflow/core/graph/tensor_id_test.cc", "Skewed");

  const uint32 space = 1 << (rnd->Rand32() % (max_log + 1));
  return rnd->Rand32() % space;
}

void BM_ParseTensorName(::testing::benchmark::State& state) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSgraphPStensor_id_testDTcc mht_2(mht_2_v, 219, "", "./tensorflow/core/graph/tensor_id_test.cc", "BM_ParseTensorName");

  const int arg = state.range(0);
  random::PhiloxRandom philox(301, 17);
  random::SimplePhilox rnd(&philox);
  std::vector<string> names;
  for (int i = 0; i < 100; i++) {
    string name;
    switch (arg) {
      case 0: {  // Generate random names
        size_t len = Skewed(&rnd, 4);
        while (name.size() < len) {
          name += rnd.OneIn(4) ? '0' : 'a';
        }
        if (rnd.OneIn(3)) {
          strings::StrAppend(&name, ":", rnd.Uniform(12));
        }
        break;
      }
      case 1:
        name = "W1";
        break;
      case 2:
        name = "t0003";
        break;
      case 3:
        name = "weights";
        break;
      case 4:
        name = "weights:17";
        break;
      case 5:
        name = "^weights";
        break;
      default:
        LOG(FATAL) << "Unexpected arg";
        break;
    }
    names.push_back(name);
  }

  TensorId id;
  int index = 0;
  int sum = 0;
  for (auto s : state) {
    id = ParseTensorName(names[index++ % names.size()]);
    sum += id.second;
  }
  VLOG(2) << sum;  // Prevent compiler from eliminating loop body
}
BENCHMARK(BM_ParseTensorName)->Arg(0)->Arg(1)->Arg(2)->Arg(3)->Arg(4)->Arg(5);

TEST(TensorIdTest, IsTensorIdControl) {
  string input = "^foo";
  TensorId tensor_id = ParseTensorName(input);
  EXPECT_TRUE(IsTensorIdControl(tensor_id));

  input = "foo";
  tensor_id = ParseTensorName(input);
  EXPECT_FALSE(IsTensorIdControl(tensor_id));

  input = "foo:2";
  tensor_id = ParseTensorName(input);
  EXPECT_FALSE(IsTensorIdControl(tensor_id));
}

TEST(TensorIdTest, PortZero) {
  for (string input : {"foo", "foo:0"}) {
    TensorId tensor_id = ParseTensorName(input);
    EXPECT_EQ("foo", tensor_id.node());
    EXPECT_EQ(0, tensor_id.index());
  }
}

}  // namespace
}  // namespace tensorflow
