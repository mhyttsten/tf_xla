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
class MHTracer_DTPStensorflowPScorePSlibPSrandomPSsimple_philox_testDTcc {
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
   MHTracer_DTPStensorflowPScorePSlibPSrandomPSsimple_philox_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSlibPSrandomPSsimple_philox_testDTcc() {
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

#include "tensorflow/core/lib/random/simple_philox.h"

#include <set>
#include <string>

#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {
namespace random {
namespace {

TEST(SimplePhiloxTest, FloatTest) {
  PhiloxRandom philox(7, 7);
  SimplePhilox gen(&philox);
  static const int kIters = 1000000;
  for (int i = 0; i < kIters; ++i) {
    float f = gen.RandFloat();
    EXPECT_LE(0.0f, f);
    EXPECT_GT(1.0f, f);
  }
  for (int i = 0; i < kIters; ++i) {
    double d = gen.RandDouble();
    EXPECT_LE(0.0, d);
    EXPECT_GT(1.0, d);
  }
}

static void DifferenceTest(const char *names, SimplePhilox *gen1,
                           SimplePhilox *gen2) {
   std::vector<std::string> mht_0_v;
   mht_0_v.push_back("names: \"" + (names == nullptr ? std::string("nullptr") : std::string((char*)names)) + "\"");
   MHTracer_DTPStensorflowPScorePSlibPSrandomPSsimple_philox_testDTcc mht_0(mht_0_v, 216, "", "./tensorflow/core/lib/random/simple_philox_test.cc", "DifferenceTest");

  static const int kIters = 100;
  bool different = false;
  for (int i = 0; i < kIters; ++i) {
    if (gen1->Rand32() != gen2->Rand32()) {
      different = true;
      break;
    }
  }
  CHECK(different) << "different seeds but same output!";
}

TEST(SimplePhiloxTest, DifferenceTest) {
  PhiloxRandom philox1(1, 1), philox2(17, 17);
  SimplePhilox gen1(&philox1), gen2(&philox2);

  DifferenceTest("SimplePhilox: different seeds", &gen1, &gen2);
}

TEST(SimplePhiloxTest, DifferenceTestCloseSeeds) {
  PhiloxRandom philox1(1, 1), philox2(2, 1);
  SimplePhilox gen1(&philox1), gen2(&philox2);

  DifferenceTest("SimplePhilox: close seeds", &gen1, &gen2);
}

TEST(SimplePhiloxTest, Regression_CloseSeedsAreDifferent) {
  const int kCount = 1000;

  // Two seeds differ only by the last bit.
  PhiloxRandom philox1(0, 1), philox2(1, 1);
  SimplePhilox gen1(&philox1), gen2(&philox2);

  std::set<uint32> first;
  std::set<uint32> all;
  for (int i = 0; i < kCount; ++i) {
    uint32 v = gen1.Rand32();
    first.insert(v);
    all.insert(v);
    all.insert(gen2.Rand32());
  }

  // Broken array initialization implementation (before 2009-08-18) using the
  // above seeds return <1000, 1007>, generating output that is >99% similar.
  // The fix returns <1000, 2000> for completely disjoint sets.
  EXPECT_EQ(kCount, first.size());
  EXPECT_EQ(2 * kCount, all.size());
}

TEST(SimplePhiloxTest, TestUniform) {
  PhiloxRandom philox(17, 17);
  SimplePhilox gen(&philox);

  uint32 range = 3 * (1L << 29);
  uint32 threshold = 1L << 30;

  size_t count = 0;
  static const int kTrials = 100000;
  for (int i = 0; i < kTrials; ++i) {
    uint32 rnd = gen.Uniform(range);
    if (rnd < threshold) {
      ++count;
    }
  }

  EXPECT_LT(fabs((threshold + 0.0) / range - (count + 0.0) / kTrials), 0.005);
}

TEST(SimplePhiloxTest, TestUniform64) {
  PhiloxRandom philox(17, 17);
  SimplePhilox gen(&philox);

  uint64 range = 3 * (1LL << 59);
  uint64 threshold = 1LL << 60;

  size_t count = 0;
  static const int kTrials = 100000;
  for (int i = 0; i < kTrials; ++i) {
    uint64 rnd = gen.Uniform64(range);
    if (rnd < threshold) {
      ++count;
    }
  }

  EXPECT_LT(fabs((threshold + 0.0) / range - (count + 0.0) / kTrials), 0.005);
}

}  // namespace
}  // namespace random
}  // namespace tensorflow
