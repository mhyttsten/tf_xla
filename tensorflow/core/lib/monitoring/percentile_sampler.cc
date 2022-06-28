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
class MHTracer_DTPStensorflowPScorePSlibPSmonitoringPSpercentile_samplerDTcc {
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
   MHTracer_DTPStensorflowPScorePSlibPSmonitoringPSpercentile_samplerDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSlibPSmonitoringPSpercentile_samplerDTcc() {
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

/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/lib/monitoring/percentile_sampler.h"

#include <algorithm>

// We replace this implementation with a null implementation for mobile
// platforms.
#ifdef IS_MOBILE_PLATFORM
// Do nothing.
#else

namespace tensorflow {
namespace monitoring {

void PercentileSamplerCell::Add(double sample) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSlibPSmonitoringPSpercentile_samplerDTcc mht_0(mht_0_v, 198, "", "./tensorflow/core/lib/monitoring/percentile_sampler.cc", "PercentileSamplerCell::Add");

  uint64 nstime = EnvTime::NowNanos();
  mutex_lock l(mu_);
  samples_[next_position_] = {nstime, sample};
  ++next_position_;
  if (TF_PREDICT_FALSE(next_position_ >= samples_.size())) {
    next_position_ = 0;
  }
  if (TF_PREDICT_FALSE(num_samples_ < samples_.size())) {
    ++num_samples_;
  }
  ++total_samples_;
  accumulator_ += sample;
}

Percentiles PercentileSamplerCell::value() const {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSlibPSmonitoringPSpercentile_samplerDTcc mht_1(mht_1_v, 216, "", "./tensorflow/core/lib/monitoring/percentile_sampler.cc", "PercentileSamplerCell::value");

  Percentiles pct_samples;
  pct_samples.unit_of_measure = unit_of_measure_;
  size_t total_samples;
  long double accumulator;
  std::vector<Sample> samples = GetSamples(&total_samples, &accumulator);
  if (!samples.empty()) {
    pct_samples.num_samples = samples.size();
    pct_samples.total_samples = total_samples;
    pct_samples.accumulator = accumulator;
    pct_samples.start_nstime = samples.front().nstime;
    pct_samples.end_nstime = samples.back().nstime;

    long double total = 0.0;
    for (auto& sample : samples) {
      total += sample.value;
    }
    pct_samples.mean = total / pct_samples.num_samples;
    long double total_sigma = 0.0;
    for (auto& sample : samples) {
      double delta = sample.value - pct_samples.mean;
      total_sigma += delta * delta;
    }
    pct_samples.stddev = std::sqrt(total_sigma / pct_samples.num_samples);

    std::sort(samples.begin(), samples.end());
    pct_samples.min_value = samples.front().value;
    pct_samples.max_value = samples.back().value;
    for (auto percentile : percentiles_) {
      size_t index = std::min<size_t>(
          static_cast<size_t>(percentile * pct_samples.num_samples / 100.0),
          pct_samples.num_samples - 1);
      PercentilePoint pct = {percentile, samples[index].value};
      pct_samples.points.push_back(pct);
    }
  }
  return pct_samples;
}

std::vector<PercentileSamplerCell::Sample> PercentileSamplerCell::GetSamples(
    size_t* total_samples, long double* accumulator) const {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSlibPSmonitoringPSpercentile_samplerDTcc mht_2(mht_2_v, 259, "", "./tensorflow/core/lib/monitoring/percentile_sampler.cc", "PercentileSamplerCell::GetSamples");

  mutex_lock l(mu_);
  std::vector<Sample> samples;
  if (num_samples_ == samples_.size()) {
    samples.insert(samples.end(), samples_.begin() + next_position_,
                   samples_.end());
  }
  samples.insert(samples.end(), samples_.begin(),
                 samples_.begin() + next_position_);
  *total_samples = total_samples_;
  *accumulator = accumulator_;
  return samples;
}

}  // namespace monitoring
}  // namespace tensorflow

#endif  // IS_MOBILE_PLATFORM
