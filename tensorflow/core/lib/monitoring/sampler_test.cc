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
class MHTracer_DTPStensorflowPScorePSlibPSmonitoringPSsampler_testDTcc {
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
   MHTracer_DTPStensorflowPScorePSlibPSmonitoringPSsampler_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSlibPSmonitoringPSsampler_testDTcc() {
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

/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/lib/monitoring/sampler.h"

#include "tensorflow/core/platform/test.h"

namespace tensorflow {
namespace monitoring {
namespace {

using histogram::Histogram;

void EqHistograms(const Histogram& expected,
                  const HistogramProto& actual_proto) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSlibPSmonitoringPSsampler_testDTcc mht_0(mht_0_v, 196, "", "./tensorflow/core/lib/monitoring/sampler_test.cc", "EqHistograms");

  Histogram actual;
  ASSERT_TRUE(actual.DecodeFromProto(actual_proto));

  EXPECT_EQ(expected.ToString(), actual.ToString());
}

auto* sampler_with_labels =
    Sampler<1>::New({"/tensorflow/test/sampler_with_labels",
                     "Sampler with one label.", "MyLabel"},
                    Buckets::Explicit({10.0, 20.0}));

TEST(LabeledSamplerTest, InitializedEmpty) {
  Histogram empty;
  EqHistograms(empty, sampler_with_labels->GetCell("Empty")->value());
}

TEST(LabeledSamplerTest, ExplicitBucketBoundaries) {
  // Sampler automatically adds DBL_MAX to the list of buckets.
  Histogram expected({10.0, 20.0, DBL_MAX});
  auto* cell = sampler_with_labels->GetCell("BucketBoundaries");
  sampler_with_labels->GetCell("AddedToCheckPreviousCellValidity");
  cell->Add(-1.0);
  expected.Add(-1.0);
  cell->Add(10.0);
  expected.Add(10.0);
  cell->Add(20.0);
  expected.Add(20.0);
  cell->Add(31.0);
  expected.Add(31.0);

  EqHistograms(expected, cell->value());
}

auto* init_sampler_without_labels =
    Sampler<0>::New({"/tensorflow/test/init_sampler_without_labels",
                     "Sampler without labels initialized as empty."},
                    Buckets::Explicit(std::vector<double>{1.5, 2.8}));

TEST(UnlabeledSamplerTest, InitializedEmpty) {
  Histogram empty;
  EqHistograms(empty, init_sampler_without_labels->GetCell()->value());
}

auto* sampler_without_labels =
    Sampler<0>::New({"/tensorflow/test/sampler_without_labels",
                     "Sampler without labels initialized as empty."},
                    Buckets::Explicit({1.5, 2.8}));

TEST(UnlabeledSamplerTest, ExplicitBucketBoundaries) {
  // Sampler automatically adds DBL_MAX to the list of buckets.
  Histogram expected({1.5, 2.8, DBL_MAX});
  auto* cell = sampler_without_labels->GetCell();
  cell->Add(-1.0);
  expected.Add(-1.0);
  cell->Add(2.0);
  expected.Add(2.0);
  cell->Add(31.0);
  expected.Add(31.0);

  EqHistograms(expected, cell->value());
}

auto* sampler_with_exponential =
    Sampler<1>::New({"/tensorflow/test/sampler_with_exponential",
                     "Sampler with exponential buckets.", "MyLabel"},
                    // So limits are {1, 2, 4}.
                    Buckets::Exponential(1, 2, 3));

TEST(ExponentialSamplerTest, ExponentialBucketBoundaries) {
  // Sampler automatically adds DBL_MAX to the list of buckets.
  Histogram expected({1.0, 2.0, 4.0, DBL_MAX});
  auto* cell = sampler_with_exponential->GetCell("BucketBoundaries");
  sampler_with_exponential->GetCell("AddedToCheckPreviousCellValidity");
  cell->Add(-1.0);
  expected.Add(-1.0);
  cell->Add(0.5);
  expected.Add(0.5);
  cell->Add(1.001);
  expected.Add(1.001);
  cell->Add(3.999);
  expected.Add(3.999);
  cell->Add(6.0);
  expected.Add(6.0);

  EqHistograms(expected, cell->value());
}

TEST(ExplicitSamplerTest, SameName) {
  auto* same_sampler = Sampler<1>::New({"/tensorflow/test/sampler_with_labels",
                                        "Sampler with one label.", "MyLabel"},
                                       Buckets::Explicit({10.0, 20.0}));
  EXPECT_TRUE(sampler_with_labels->GetStatus().ok());
  EXPECT_FALSE(same_sampler->GetStatus().ok());
  delete same_sampler;
}

}  // namespace
}  // namespace monitoring
}  // namespace tensorflow
