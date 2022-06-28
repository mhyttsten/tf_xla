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
class MHTracer_DTPStensorflowPScorePSlibPSmonitoringPSsamplerDTcc {
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
   MHTracer_DTPStensorflowPScorePSlibPSmonitoringPSsamplerDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSlibPSmonitoringPSsamplerDTcc() {
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

#include "tensorflow/core/lib/monitoring/sampler.h"

// clang-format off
// Required for IS_MOBILE_PLATFORM
#include "tensorflow/core/platform/platform.h"
// clang-format on

// We replace this implementation with a null implementation for mobile
// platforms.
#ifdef IS_MOBILE_PLATFORM
// Do nothing.
#else

namespace tensorflow {
namespace monitoring {
namespace {

class ExplicitBuckets : public Buckets {
 public:
  ~ExplicitBuckets() override = default;

  explicit ExplicitBuckets(std::vector<double> bucket_limits)
      : bucket_limits_(std::move(bucket_limits)) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSlibPSmonitoringPSsamplerDTcc mht_0(mht_0_v, 207, "", "./tensorflow/core/lib/monitoring/sampler.cc", "ExplicitBuckets");

    CHECK_GT(bucket_limits_.size(), 0);
    // Verify that the bucket boundaries are strictly increasing
    for (size_t i = 1; i < bucket_limits_.size(); i++) {
      CHECK_GT(bucket_limits_[i], bucket_limits_[i - 1]);
    }
    // We augment the bucket limits so that all boundaries are within [-DBL_MAX,
    // DBL_MAX].
    //
    // Since we use ThreadSafeHistogram, we don't have to explicitly add
    // -DBL_MAX, because it uses these limits as upper-bounds, so
    // bucket_count[0] is always the number of elements in
    // [-DBL_MAX, bucket_limits[0]).
    if (bucket_limits_.back() != DBL_MAX) {
      bucket_limits_.push_back(DBL_MAX);
    }
  }

  const std::vector<double>& explicit_bounds() const override {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSlibPSmonitoringPSsamplerDTcc mht_1(mht_1_v, 228, "", "./tensorflow/core/lib/monitoring/sampler.cc", "explicit_bounds");

    return bucket_limits_;
  }

 private:
  std::vector<double> bucket_limits_;

  TF_DISALLOW_COPY_AND_ASSIGN(ExplicitBuckets);
};

class ExponentialBuckets : public Buckets {
 public:
  ~ExponentialBuckets() override = default;

  ExponentialBuckets(double scale, double growth_factor, int bucket_count)
      : explicit_buckets_(
            ComputeBucketLimits(scale, growth_factor, bucket_count)) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSlibPSmonitoringPSsamplerDTcc mht_2(mht_2_v, 247, "", "./tensorflow/core/lib/monitoring/sampler.cc", "ExponentialBuckets");
}

  const std::vector<double>& explicit_bounds() const override {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSlibPSmonitoringPSsamplerDTcc mht_3(mht_3_v, 252, "", "./tensorflow/core/lib/monitoring/sampler.cc", "explicit_bounds");

    return explicit_buckets_.explicit_bounds();
  }

 private:
  static std::vector<double> ComputeBucketLimits(double scale,
                                                 double growth_factor,
                                                 int bucket_count) {
    CHECK_GT(bucket_count, 0);
    std::vector<double> bucket_limits;
    double bound = scale;
    for (int i = 0; i < bucket_count; i++) {
      bucket_limits.push_back(bound);
      bound *= growth_factor;
    }
    return bucket_limits;
  }

  ExplicitBuckets explicit_buckets_;

  TF_DISALLOW_COPY_AND_ASSIGN(ExponentialBuckets);
};

}  // namespace

// static
std::unique_ptr<Buckets> Buckets::Explicit(std::vector<double> bucket_limits) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSlibPSmonitoringPSsamplerDTcc mht_4(mht_4_v, 281, "", "./tensorflow/core/lib/monitoring/sampler.cc", "Buckets::Explicit");

  return std::unique_ptr<Buckets>(
      new ExplicitBuckets(std::move(bucket_limits)));
}

// static
std::unique_ptr<Buckets> Buckets::Explicit(
    std::initializer_list<double> bucket_limits) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSlibPSmonitoringPSsamplerDTcc mht_5(mht_5_v, 291, "", "./tensorflow/core/lib/monitoring/sampler.cc", "Buckets::Explicit");

  return std::unique_ptr<Buckets>(new ExplicitBuckets(bucket_limits));
}

// static
std::unique_ptr<Buckets> Buckets::Exponential(double scale,
                                              double growth_factor,
                                              int bucket_count) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePSlibPSmonitoringPSsamplerDTcc mht_6(mht_6_v, 301, "", "./tensorflow/core/lib/monitoring/sampler.cc", "Buckets::Exponential");

  return std::unique_ptr<Buckets>(
      new ExponentialBuckets(scale, growth_factor, bucket_count));
}

}  // namespace monitoring
}  // namespace tensorflow

#endif  // IS_MOBILE_PLATFORM
