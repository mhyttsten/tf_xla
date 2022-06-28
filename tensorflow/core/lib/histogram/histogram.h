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

#ifndef TENSORFLOW_CORE_LIB_HISTOGRAM_HISTOGRAM_H_
#define TENSORFLOW_CORE_LIB_HISTOGRAM_HISTOGRAM_H_
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
class MHTracer_DTPStensorflowPScorePSlibPShistogramPShistogramDTh {
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
   MHTracer_DTPStensorflowPScorePSlibPShistogramPShistogramDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSlibPShistogramPShistogramDTh() {
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


#include <string>
#include <vector>
#include "tensorflow/core/lib/gtl/array_slice.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/thread_annotations.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {

class HistogramProto;

namespace histogram {

class Histogram {
 public:
  // Create a histogram with a default set of bucket boundaries.
  // Buckets near zero cover very small ranges (e.g. 10^-12), and each
  // bucket range grows by ~10% as we head away from zero.  The
  // buckets cover the range from -DBL_MAX to DBL_MAX.
  Histogram();

  // Create a histogram with a custom set of bucket boundaries,
  // specified in "custom_bucket_limits[0..custom_bucket_limits.size()-1]"
  // REQUIRES: custom_bucket_limits[i] values are monotonically increasing.
  // REQUIRES: custom_bucket_limits is not empty()
  explicit Histogram(gtl::ArraySlice<double> custom_bucket_limits);

  // Restore the state of a histogram that was previously encoded
  // via Histogram::EncodeToProto.  Note that only the bucket boundaries
  // generated by EncodeToProto will be restored.
  bool DecodeFromProto(const HistogramProto& proto);

  ~Histogram() {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSlibPShistogramPShistogramDTh mht_0(mht_0_v, 221, "", "./tensorflow/core/lib/histogram/histogram.h", "~Histogram");
}

  void Clear();
  void Add(double value);

  // Save the current state of the histogram to "*proto".  If
  // "preserve_zero_buckets" is false, only non-zero bucket values and
  // ranges are saved, and the bucket boundaries of zero-valued buckets
  // are lost.
  void EncodeToProto(HistogramProto* proto, bool preserve_zero_buckets) const;

  // Return the median of the values in the histogram
  double Median() const;

  // Return the "p"th percentile [0.0..100.0] of the values in the
  // distribution
  double Percentile(double p) const;

  // Return the average value of the distribution
  double Average() const;

  // Return the standard deviation of values in the distribution
  double StandardDeviation() const;

  // Returns a multi-line human-readable string representing the histogram
  // contents.  Example output:
  //   Count: 4  Average: 251.7475  StdDev: 432.02
  //   Min: -3.0000  Median: 5.0000  Max: 1000.0000
  //   ------------------------------------------------------
  //   [      -5,       0 )       1  25.000%  25.000% #####
  //   [       0,       5 )       1  25.000%  50.000% #####
  //   [       5,      10 )       1  25.000%  75.000% #####
  //   [    1000,   10000 )       1  25.000% 100.000% #####
  std::string ToString() const;

 private:
  double min_;
  double max_;
  double num_;
  double sum_;
  double sum_squares_;

  std::vector<double> custom_bucket_limits_;
  gtl::ArraySlice<double> bucket_limits_;
  std::vector<double> buckets_;

  double Remap(double x, double x0, double x1, double y0, double y1) const;

  TF_DISALLOW_COPY_AND_ASSIGN(Histogram);
};

// Wrapper around a Histogram object that is thread safe.
//
// All methods hold a lock while delegating to a Histogram object owned by the
// ThreadSafeHistogram instance.
//
// See Histogram for documentation of the methods.
class ThreadSafeHistogram {
 public:
  ThreadSafeHistogram() {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSlibPShistogramPShistogramDTh mht_1(mht_1_v, 283, "", "./tensorflow/core/lib/histogram/histogram.h", "ThreadSafeHistogram");
}
  explicit ThreadSafeHistogram(gtl::ArraySlice<double> custom_bucket_limits)
      : histogram_(custom_bucket_limits) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSlibPShistogramPShistogramDTh mht_2(mht_2_v, 288, "", "./tensorflow/core/lib/histogram/histogram.h", "ThreadSafeHistogram");
}
  bool DecodeFromProto(const HistogramProto& proto);

  ~ThreadSafeHistogram() {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSlibPShistogramPShistogramDTh mht_3(mht_3_v, 294, "", "./tensorflow/core/lib/histogram/histogram.h", "~ThreadSafeHistogram");
}

  void Clear();

  // TODO(touts): It might be a good idea to provide a AddN(<many values>)
  // method to avoid grabbing/releasing the lock when adding many values.
  void Add(double value);

  void EncodeToProto(HistogramProto* proto, bool preserve_zero_buckets) const;
  double Median() const;
  double Percentile(double p) const;
  double Average() const;
  double StandardDeviation() const;
  std::string ToString() const;

 private:
  mutable mutex mu_;
  Histogram histogram_ TF_GUARDED_BY(mu_);
};

}  // namespace histogram
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_LIB_HISTOGRAM_HISTOGRAM_H_
