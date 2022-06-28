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
class MHTracer_DTPStensorflowPScorePSlibPShistogramPShistogram_testDTcc {
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
   MHTracer_DTPStensorflowPScorePSlibPShistogramPShistogram_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSlibPShistogramPShistogram_testDTcc() {
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

#include "tensorflow/core/lib/histogram/histogram.h"
#include <float.h>
#include "tensorflow/core/framework/summary.pb.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {
namespace histogram {

static void Validate(const Histogram& h) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSlibPShistogramPShistogram_testDTcc mht_0(mht_0_v, 194, "", "./tensorflow/core/lib/histogram/histogram_test.cc", "Validate");

  string s1 = h.ToString();
  LOG(ERROR) << s1;

  HistogramProto proto_with_zeroes;
  h.EncodeToProto(&proto_with_zeroes, true);
  Histogram h2;
  EXPECT_TRUE(h2.DecodeFromProto(proto_with_zeroes));
  string s2 = h2.ToString();
  LOG(ERROR) << s2;

  EXPECT_EQ(s1, s2);

  HistogramProto proto_no_zeroes;
  h.EncodeToProto(&proto_no_zeroes, false);
  LOG(ERROR) << proto_no_zeroes.DebugString();
  Histogram h3;
  EXPECT_TRUE(h3.DecodeFromProto(proto_no_zeroes));
  string s3 = h3.ToString();
  LOG(ERROR) << s3;

  EXPECT_EQ(s1, s3);
}

TEST(Histogram, Empty) {
  Histogram h;
  Validate(h);
}

TEST(Histogram, SingleValue) {
  Histogram h;
  h.Add(-3.0);
  Validate(h);
}

TEST(Histogram, CustomBuckets) {
  Histogram h({-10, -5, 0, 5, 10, 100, 1000, 10000, DBL_MAX});
  h.Add(-3.0);
  h.Add(4.99);
  h.Add(5.0);
  h.Add(1000.0);
  Validate(h);
}

TEST(Histogram, Median) {
  Histogram h({0, 10, 100, DBL_MAX});
  h.Add(-2);
  h.Add(-2);
  h.Add(0);
  double median = h.Median();
  EXPECT_EQ(median, -0.5);
}

TEST(Histogram, Percentile) {
  // 10%, 30%, 40%, 20%
  Histogram h({1, 2, 3, 4});
  // 10% first bucket
  h.Add(-1.0);
  // 30% second bucket
  h.Add(1.5);
  h.Add(1.5);
  h.Add(1.5);
  // 40% third bucket
  h.Add(2.5);
  h.Add(2.5);
  h.Add(2.5);
  h.Add(2.5);
  // 20% fourth bucket
  h.Add(3.5);
  h.Add(3.9);

  EXPECT_EQ(h.Percentile(0), -1.0);    // -1.0 = histo.min_
  EXPECT_EQ(h.Percentile(25), 1.5);    // 1.5 = remap(25, 10, 40, 1, 2)
  EXPECT_EQ(h.Percentile(50), 2.25);   // 2.25 = remap(50, 40, 80, 2, 3)
  EXPECT_EQ(h.Percentile(75), 2.875);  // 2.875 = remap(75, 40, 80, 2, 3)
  EXPECT_EQ(h.Percentile(90), 3.45);   // 3.45 = remap(90, 80, 100, 3, 3.9)
  EXPECT_EQ(h.Percentile(100), 3.9);   // 3.9 = histo.max_
}

TEST(Histogram, Basic) {
  Histogram h;
  for (int i = 0; i < 100; i++) {
    h.Add(i);
  }
  for (int i = 1000; i < 100000; i += 1000) {
    h.Add(i);
  }
  Validate(h);
}

TEST(ThreadSafeHistogram, Basic) {
  // Fill a normal histogram.
  Histogram h;
  for (int i = 0; i < 100; i++) {
    h.Add(i);
  }

  // Fill a thread-safe histogram with the same values.
  ThreadSafeHistogram tsh;
  for (int i = 0; i < 100; i++) {
    tsh.Add(i);
  }

  for (int i = 0; i < 2; ++i) {
    bool preserve_zero_buckets = (i == 0);
    HistogramProto h_proto;
    h.EncodeToProto(&h_proto, preserve_zero_buckets);
    HistogramProto tsh_proto;
    tsh.EncodeToProto(&tsh_proto, preserve_zero_buckets);

    // Let's decode from the proto of the other histogram type.
    Histogram h2;
    EXPECT_TRUE(h2.DecodeFromProto(tsh_proto));
    ThreadSafeHistogram tsh2;
    EXPECT_TRUE(tsh2.DecodeFromProto(h_proto));

    // Now let's reencode and check they match.
    EXPECT_EQ(h2.ToString(), tsh2.ToString());
  }

  EXPECT_EQ(h.Median(), tsh.Median());
  EXPECT_EQ(h.Percentile(40.0), tsh.Percentile(40.0));
  EXPECT_EQ(h.Average(), tsh.Average());
  EXPECT_EQ(h.StandardDeviation(), tsh.StandardDeviation());
  EXPECT_EQ(h.ToString(), tsh.ToString());
}

}  // namespace histogram
}  // namespace tensorflow
