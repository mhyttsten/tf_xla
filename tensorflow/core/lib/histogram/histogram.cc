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
class MHTracer_DTPStensorflowPScorePSlibPShistogramPShistogramDTcc {
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
   MHTracer_DTPStensorflowPScorePSlibPShistogramPShistogramDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSlibPShistogramPShistogramDTcc() {
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
#include <math.h>
#include <vector>
#include "tensorflow/core/framework/summary.pb.h"

#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/types.h"
namespace tensorflow {
namespace histogram {

static std::vector<double>* InitDefaultBucketsInner() {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSlibPShistogramPShistogramDTcc mht_0(mht_0_v, 197, "", "./tensorflow/core/lib/histogram/histogram.cc", "InitDefaultBucketsInner");

  std::vector<double> buckets;
  std::vector<double> neg_buckets;
  // Make buckets whose range grows by 10% starting at 1.0e-12 up to 1.0e20
  double v = 1.0e-12;
  while (v < 1.0e20) {
    buckets.push_back(v);
    neg_buckets.push_back(-v);
    v *= 1.1;
  }
  buckets.push_back(DBL_MAX);
  neg_buckets.push_back(-DBL_MAX);
  std::reverse(neg_buckets.begin(), neg_buckets.end());
  std::vector<double>* result = new std::vector<double>;
  result->insert(result->end(), neg_buckets.begin(), neg_buckets.end());
  result->push_back(0.0);
  result->insert(result->end(), buckets.begin(), buckets.end());
  return result;
}

static gtl::ArraySlice<double> InitDefaultBuckets() {
  static std::vector<double>* default_bucket_limits = InitDefaultBucketsInner();
  return *default_bucket_limits;
}

Histogram::Histogram() : bucket_limits_(InitDefaultBuckets()) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSlibPShistogramPShistogramDTcc mht_1(mht_1_v, 225, "", "./tensorflow/core/lib/histogram/histogram.cc", "Histogram::Histogram");
 Clear(); }

// Create a histogram with a custom set of bucket limits,
// specified in "custom_buckets[0..custom_buckets.size()-1]"
Histogram::Histogram(gtl::ArraySlice<double> custom_bucket_limits)
    : custom_bucket_limits_(custom_bucket_limits.begin(),
                            custom_bucket_limits.end()),
      bucket_limits_(custom_bucket_limits_) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSlibPShistogramPShistogramDTcc mht_2(mht_2_v, 235, "", "./tensorflow/core/lib/histogram/histogram.cc", "Histogram::Histogram");

#ifndef NDEBUG
  DCHECK_GT(bucket_limits_.size(), size_t{0});
  // Verify that the bucket boundaries are strictly increasing
  for (size_t i = 1; i < bucket_limits_.size(); i++) {
    DCHECK_GT(bucket_limits_[i], bucket_limits_[i - 1]);
  }
#endif
  Clear();
}

bool Histogram::DecodeFromProto(const HistogramProto& proto) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSlibPShistogramPShistogramDTcc mht_3(mht_3_v, 249, "", "./tensorflow/core/lib/histogram/histogram.cc", "Histogram::DecodeFromProto");

  if ((proto.bucket_size() != proto.bucket_limit_size()) ||
      (proto.bucket_size() == 0)) {
    return false;
  }
  min_ = proto.min();
  max_ = proto.max();
  num_ = proto.num();
  sum_ = proto.sum();
  sum_squares_ = proto.sum_squares();
  custom_bucket_limits_.clear();
  custom_bucket_limits_.insert(custom_bucket_limits_.end(),
                               proto.bucket_limit().begin(),
                               proto.bucket_limit().end());
  bucket_limits_ = custom_bucket_limits_;
  buckets_.clear();
  buckets_.insert(buckets_.end(), proto.bucket().begin(), proto.bucket().end());
  return true;
}

void Histogram::Clear() {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSlibPShistogramPShistogramDTcc mht_4(mht_4_v, 272, "", "./tensorflow/core/lib/histogram/histogram.cc", "Histogram::Clear");

  min_ = bucket_limits_[bucket_limits_.size() - 1];
  max_ = -DBL_MAX;
  num_ = 0;
  sum_ = 0;
  sum_squares_ = 0;
  buckets_.resize(bucket_limits_.size());
  for (size_t i = 0; i < bucket_limits_.size(); i++) {
    buckets_[i] = 0;
  }
}

void Histogram::Add(double value) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSlibPShistogramPShistogramDTcc mht_5(mht_5_v, 287, "", "./tensorflow/core/lib/histogram/histogram.cc", "Histogram::Add");

  int b =
      std::upper_bound(bucket_limits_.begin(), bucket_limits_.end(), value) -
      bucket_limits_.begin();

  buckets_[b] += 1.0;
  if (min_ > value) min_ = value;
  if (max_ < value) max_ = value;
  num_++;
  sum_ += value;
  sum_squares_ += (value * value);
}

double Histogram::Median() const {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePSlibPShistogramPShistogramDTcc mht_6(mht_6_v, 303, "", "./tensorflow/core/lib/histogram/histogram.cc", "Histogram::Median");
 return Percentile(50.0); }

// Linearly map the variable x from [x0, x1] unto [y0, y1]
double Histogram::Remap(double x, double x0, double x1, double y0,
                        double y1) const {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePSlibPShistogramPShistogramDTcc mht_7(mht_7_v, 310, "", "./tensorflow/core/lib/histogram/histogram.cc", "Histogram::Remap");

  return y0 + (x - x0) / (x1 - x0) * (y1 - y0);
}

// Pick tight left-hand-side and right-hand-side bounds and then
// interpolate a histogram value at percentile p
double Histogram::Percentile(double p) const {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScorePSlibPShistogramPShistogramDTcc mht_8(mht_8_v, 319, "", "./tensorflow/core/lib/histogram/histogram.cc", "Histogram::Percentile");

  if (num_ == 0.0) return 0.0;

  double threshold = num_ * (p / 100.0);
  double cumsum_prev = 0;
  for (size_t i = 0; i < buckets_.size(); i++) {
    double cumsum = cumsum_prev + buckets_[i];

    // Find the first bucket whose cumsum >= threshold
    if (cumsum >= threshold) {
      // Prevent divide by 0 in remap which happens if cumsum == cumsum_prev
      // This should only get hit when p == 0, cumsum == 0, and cumsum_prev == 0
      if (cumsum == cumsum_prev) {
        continue;
      }

      // Calculate the lower bound of interpolation
      double lhs = (i == 0 || cumsum_prev == 0) ? min_ : bucket_limits_[i - 1];
      lhs = std::max(lhs, min_);

      // Calculate the upper bound of interpolation
      double rhs = bucket_limits_[i];
      rhs = std::min(rhs, max_);

      double weight = Remap(threshold, cumsum_prev, cumsum, lhs, rhs);
      return weight;
    }

    cumsum_prev = cumsum;
  }
  return max_;
}

double Histogram::Average() const {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScorePSlibPShistogramPShistogramDTcc mht_9(mht_9_v, 355, "", "./tensorflow/core/lib/histogram/histogram.cc", "Histogram::Average");

  if (num_ == 0.0) return 0;
  return sum_ / num_;
}

double Histogram::StandardDeviation() const {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScorePSlibPShistogramPShistogramDTcc mht_10(mht_10_v, 363, "", "./tensorflow/core/lib/histogram/histogram.cc", "Histogram::StandardDeviation");

  if (num_ == 0.0) return 0;
  double variance = (sum_squares_ * num_ - sum_ * sum_) / (num_ * num_);
  return sqrt(variance);
}

std::string Histogram::ToString() const {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScorePSlibPShistogramPShistogramDTcc mht_11(mht_11_v, 372, "", "./tensorflow/core/lib/histogram/histogram.cc", "Histogram::ToString");

  std::string r;
  char buf[200];
  snprintf(buf, sizeof(buf), "Count: %.0f  Average: %.4f  StdDev: %.2f\n", num_,
           Average(), StandardDeviation());
  r.append(buf);
  snprintf(buf, sizeof(buf), "Min: %.4f  Median: %.4f  Max: %.4f\n",
           (num_ == 0.0 ? 0.0 : min_), Median(), max_);
  r.append(buf);
  r.append("------------------------------------------------------\n");
  const double mult = num_ > 0 ? 100.0 / num_ : 0.0;
  double sum = 0;
  for (size_t b = 0; b < buckets_.size(); b++) {
    if (buckets_[b] <= 0.0) continue;
    sum += buckets_[b];
    snprintf(buf, sizeof(buf), "[ %10.2g, %10.2g ) %7.0f %7.3f%% %7.3f%% ",
             ((b == 0) ? -DBL_MAX : bucket_limits_[b - 1]),  // left
             bucket_limits_[b],                              // right
             buckets_[b],                                    // count
             mult * buckets_[b],                             // percentage
             mult * sum);                                    // cum percentage
    r.append(buf);

    // Add hash marks based on percentage; 20 marks for 100%.
    int marks = static_cast<int>(20 * (buckets_[b] / num_) + 0.5);
    r.append(marks, '#');
    r.push_back('\n');
  }
  return r;
}

void Histogram::EncodeToProto(HistogramProto* proto,
                              bool preserve_zero_buckets) const {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScorePSlibPShistogramPShistogramDTcc mht_12(mht_12_v, 407, "", "./tensorflow/core/lib/histogram/histogram.cc", "Histogram::EncodeToProto");

  proto->Clear();
  proto->set_min(min_);
  proto->set_max(max_);
  proto->set_num(num_);
  proto->set_sum(sum_);
  proto->set_sum_squares(sum_squares_);
  for (size_t i = 0; i < buckets_.size();) {
    double end = bucket_limits_[i];
    double count = buckets_[i];
    i++;
    if (!preserve_zero_buckets && count <= 0.0) {
      // Find run of empty buckets and collapse them into one
      while (i < buckets_.size() && buckets_[i] <= 0.0) {
        end = bucket_limits_[i];
        count = buckets_[i];
        i++;
      }
    }
    proto->add_bucket_limit(end);
    proto->add_bucket(count);
  }
  if (proto->bucket_size() == 0.0) {
    // It's easier when we restore if we always have at least one bucket entry
    proto->add_bucket_limit(DBL_MAX);
    proto->add_bucket(0.0);
  }
}

// ThreadSafeHistogram implementation.
bool ThreadSafeHistogram::DecodeFromProto(const HistogramProto& proto) {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPScorePSlibPShistogramPShistogramDTcc mht_13(mht_13_v, 440, "", "./tensorflow/core/lib/histogram/histogram.cc", "ThreadSafeHistogram::DecodeFromProto");

  mutex_lock l(mu_);
  return histogram_.DecodeFromProto(proto);
}

void ThreadSafeHistogram::Clear() {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPScorePSlibPShistogramPShistogramDTcc mht_14(mht_14_v, 448, "", "./tensorflow/core/lib/histogram/histogram.cc", "ThreadSafeHistogram::Clear");

  mutex_lock l(mu_);
  histogram_.Clear();
}

void ThreadSafeHistogram::Add(double value) {
   std::vector<std::string> mht_15_v;
   MHTracer_DTPStensorflowPScorePSlibPShistogramPShistogramDTcc mht_15(mht_15_v, 456, "", "./tensorflow/core/lib/histogram/histogram.cc", "ThreadSafeHistogram::Add");

  mutex_lock l(mu_);
  histogram_.Add(value);
}

void ThreadSafeHistogram::EncodeToProto(HistogramProto* proto,
                                        bool preserve_zero_buckets) const {
   std::vector<std::string> mht_16_v;
   MHTracer_DTPStensorflowPScorePSlibPShistogramPShistogramDTcc mht_16(mht_16_v, 465, "", "./tensorflow/core/lib/histogram/histogram.cc", "ThreadSafeHistogram::EncodeToProto");

  mutex_lock l(mu_);
  histogram_.EncodeToProto(proto, preserve_zero_buckets);
}

double ThreadSafeHistogram::Median() const {
   std::vector<std::string> mht_17_v;
   MHTracer_DTPStensorflowPScorePSlibPShistogramPShistogramDTcc mht_17(mht_17_v, 473, "", "./tensorflow/core/lib/histogram/histogram.cc", "ThreadSafeHistogram::Median");

  mutex_lock l(mu_);
  return histogram_.Median();
}

double ThreadSafeHistogram::Percentile(double p) const {
   std::vector<std::string> mht_18_v;
   MHTracer_DTPStensorflowPScorePSlibPShistogramPShistogramDTcc mht_18(mht_18_v, 481, "", "./tensorflow/core/lib/histogram/histogram.cc", "ThreadSafeHistogram::Percentile");

  mutex_lock l(mu_);
  return histogram_.Percentile(p);
}

double ThreadSafeHistogram::Average() const {
   std::vector<std::string> mht_19_v;
   MHTracer_DTPStensorflowPScorePSlibPShistogramPShistogramDTcc mht_19(mht_19_v, 489, "", "./tensorflow/core/lib/histogram/histogram.cc", "ThreadSafeHistogram::Average");

  mutex_lock l(mu_);
  return histogram_.Average();
}

double ThreadSafeHistogram::StandardDeviation() const {
   std::vector<std::string> mht_20_v;
   MHTracer_DTPStensorflowPScorePSlibPShistogramPShistogramDTcc mht_20(mht_20_v, 497, "", "./tensorflow/core/lib/histogram/histogram.cc", "ThreadSafeHistogram::StandardDeviation");

  mutex_lock l(mu_);
  return histogram_.StandardDeviation();
}

std::string ThreadSafeHistogram::ToString() const {
   std::vector<std::string> mht_21_v;
   MHTracer_DTPStensorflowPScorePSlibPShistogramPShistogramDTcc mht_21(mht_21_v, 505, "", "./tensorflow/core/lib/histogram/histogram.cc", "ThreadSafeHistogram::ToString");

  mutex_lock l(mu_);
  return histogram_.ToString();
}

}  // namespace histogram
}  // namespace tensorflow
