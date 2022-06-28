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

#ifndef TENSORFLOW_CORE_KERNELS_RANGE_SAMPLER_H_
#define TENSORFLOW_CORE_KERNELS_RANGE_SAMPLER_H_
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
class MHTracer_DTPStensorflowPScorePSkernelsPSrange_samplerDTh {
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
   MHTracer_DTPStensorflowPScorePSkernelsPSrange_samplerDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSkernelsPSrange_samplerDTh() {
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


#include <vector>

#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/gtl/array_slice.h"
#include "tensorflow/core/lib/random/distribution_sampler.h"
#include "tensorflow/core/lib/random/random_distributions.h"
#include "tensorflow/core/lib/random/weighted_picker.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/thread_annotations.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {

class Env;

// Abstract subclass for sampling from the set of non-negative integers
// [0, range)
class RangeSampler {
 public:
  explicit RangeSampler(int64_t range) : range_(range) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSrange_samplerDTh mht_0(mht_0_v, 208, "", "./tensorflow/core/kernels/range_sampler.h", "RangeSampler");
 CHECK_GT(range_, 0); }
  virtual ~RangeSampler();

  // Sample a single value
  virtual int64_t Sample(random::SimplePhilox* rnd) const = 0;

  // The probability that a single call to Sample() returns the given value.
  // Assumes that value is in [0, range).  No range checking is done.
  virtual float Probability(int64_t value) const = 0;

  // Fill "batch" with samples from the distribution.
  // If unique=true, then we re-pick each element until we get a
  // value distinct from all previously picked values in the batch.
  void SampleBatch(random::SimplePhilox* rnd, bool unique,
                   gtl::MutableArraySlice<int64_t> batch) const;

  // Fill "batch" with samples from the distribution, and report
  // "expected counts".
  //
  // The "expected count" of a value is an estimate of the expected
  // number of occurrences of the value in the batch returned by a
  // call to this function with the given parameters.  If unique=true,
  // the expected count is an inclusion probability.  For details on
  // this estimation, see the comment to "ExpectedCountHelper" in the
  // .cc file.
  //
  // Expected counts for the elements of the returned "batch" are reported
  // in the aligned array "batch_expected_count".
  //
  // The user can optionally provide "extras", containing values in the range.
  // The expected counts for the extras are reported in the aligned array
  // "extras_expected_count".
  //
  // "batch_expected_count" must have size equal to 0 or to the size of "batch".
  // "extras" and "extras_expected_count" must have equal size.
  void SampleBatchGetExpectedCount(
      random::SimplePhilox* rnd, bool unique,
      gtl::MutableArraySlice<int64_t> batch,
      gtl::MutableArraySlice<float> batch_expected_count,
      gtl::ArraySlice<int64_t> extras,
      gtl::MutableArraySlice<float> extras_expected_count) const;

  // Same as SampleBatchGetExpectedCount (see above), but with avoided values.
  // We repick to avoid all of the values in "avoided_values".
  // "avoided_values" is only supported with unique=true.  If
  // unique=false, then avoided_values must be empty.
  virtual void SampleBatchGetExpectedCountAvoid(
      random::SimplePhilox* rnd, bool unique,
      gtl::MutableArraySlice<int64_t> batch,
      gtl::MutableArraySlice<float> batch_expected_count,
      gtl::ArraySlice<int64_t> extras,
      gtl::MutableArraySlice<float> extras_expected_count,
      gtl::ArraySlice<int64_t> avoided_values) const;

  // Does this sampler need to be updated with values, e.g. UnigramSampler
  virtual bool NeedsUpdates() const {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSrange_samplerDTh mht_1(mht_1_v, 266, "", "./tensorflow/core/kernels/range_sampler.h", "NeedsUpdates");
 return false; }

  // Updates the underlying distribution
  virtual void Update(gtl::ArraySlice<int64_t> values) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSrange_samplerDTh mht_2(mht_2_v, 272, "", "./tensorflow/core/kernels/range_sampler.h", "Update");

    LOG(FATAL) << "Update not supported for this sampler type.";
  }

  int64_t range() {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSrange_samplerDTh mht_3(mht_3_v, 279, "", "./tensorflow/core/kernels/range_sampler.h", "range");
 return range_; }

 protected:
  const int64_t range_;
};

// An AllSampler only samples batches of size equal to range.
// It returns the entire range.
// It cannot sample single values.
class AllSampler : public RangeSampler {
 public:
  explicit AllSampler(int64_t range);

  ~AllSampler() override {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSrange_samplerDTh mht_4(mht_4_v, 295, "", "./tensorflow/core/kernels/range_sampler.h", "~AllSampler");
}

  int64_t Sample(random::SimplePhilox* rnd) const override {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSrange_samplerDTh mht_5(mht_5_v, 300, "", "./tensorflow/core/kernels/range_sampler.h", "Sample");

    LOG(FATAL) << "Should not be called";
    return 0;
  }

  float Probability(int64_t value) const override {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSrange_samplerDTh mht_6(mht_6_v, 308, "", "./tensorflow/core/kernels/range_sampler.h", "Probability");

    LOG(FATAL) << "Should not be called";
    return 0;
  }

  void SampleBatchGetExpectedCountAvoid(
      random::SimplePhilox* rnd, bool unique,
      gtl::MutableArraySlice<int64_t> batch,
      gtl::MutableArraySlice<float> batch_expected_count,
      gtl::ArraySlice<int64_t> extras,
      gtl::MutableArraySlice<float> extras_expected_count,
      gtl::ArraySlice<int64_t> avoided_values) const override;
};

class UniformSampler : public RangeSampler {
 public:
  explicit UniformSampler(int64_t range);

  ~UniformSampler() override {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSrange_samplerDTh mht_7(mht_7_v, 329, "", "./tensorflow/core/kernels/range_sampler.h", "~UniformSampler");
}

  int64_t Sample(random::SimplePhilox* rnd) const override;

  float Probability(int64_t value) const override;

 private:
  const float inv_range_;
};

class LogUniformSampler : public RangeSampler {
 public:
  explicit LogUniformSampler(int64_t range);

  ~LogUniformSampler() override {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSrange_samplerDTh mht_8(mht_8_v, 346, "", "./tensorflow/core/kernels/range_sampler.h", "~LogUniformSampler");
}

  int64_t Sample(random::SimplePhilox* rnd) const override;

  float Probability(int64_t value) const override;

 private:
  const double log_range_;
};

// Thread-unsafe unigram sampler
class ThreadUnsafeUnigramSampler : public RangeSampler {
 public:
  explicit ThreadUnsafeUnigramSampler(int64_t range);
  ~ThreadUnsafeUnigramSampler() override {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSrange_samplerDTh mht_9(mht_9_v, 363, "", "./tensorflow/core/kernels/range_sampler.h", "~ThreadUnsafeUnigramSampler");
}

  int64_t Sample(random::SimplePhilox* rnd) const override;

  float Probability(int64_t value) const override;

  bool NeedsUpdates() const override {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSrange_samplerDTh mht_10(mht_10_v, 372, "", "./tensorflow/core/kernels/range_sampler.h", "NeedsUpdates");
 return true; }
  void Update(gtl::ArraySlice<int64_t> values) override;

 private:
  random::WeightedPicker picker_;
};

// Thread-safe unigram sampler
class UnigramSampler : public RangeSampler {
 public:
  explicit UnigramSampler(int64_t range);
  ~UnigramSampler() override {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSrange_samplerDTh mht_11(mht_11_v, 386, "", "./tensorflow/core/kernels/range_sampler.h", "~UnigramSampler");
}

  int64_t Sample(random::SimplePhilox* rnd) const override;

  float Probability(int64_t value) const override;

  // Overriding at a high level results in far fewer lock acquisitions.
  void SampleBatchGetExpectedCountAvoid(
      random::SimplePhilox* rnd, bool unique,
      gtl::MutableArraySlice<int64_t> batch,
      gtl::MutableArraySlice<float> batch_expected_count,
      gtl::ArraySlice<int64_t> extras,
      gtl::MutableArraySlice<float> extras_expected_count,
      gtl::ArraySlice<int64_t> avoided_values) const override;

  bool NeedsUpdates() const override {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSrange_samplerDTh mht_12(mht_12_v, 404, "", "./tensorflow/core/kernels/range_sampler.h", "NeedsUpdates");
 return true; }
  void Update(gtl::ArraySlice<int64_t> values) override;

 private:
  ThreadUnsafeUnigramSampler unsafe_sampler_ TF_GUARDED_BY(mu_);
  mutable mutex mu_;
};

// A unigram sampler that uses a fixed unigram distribution read from a
// file or passed in as an in-memory array instead of building up the
// distribution from data on the fly. There is also an option to skew the
// distribution by applying a distortion power to the weights.
class FixedUnigramSampler : public RangeSampler {
 public:
  // The vocab_file is assumed to be a CSV, with the last entry of each row a
  // value representing the counts or probabilities for the corresponding ID.
  FixedUnigramSampler(Env* env, int64_t range, const string& vocab_file,
                      float distortion, int32_t num_reserved_ids,
                      int32_t num_shards, int32_t shard);

  FixedUnigramSampler(int64_t range, const std::vector<float>& unigrams,
                      float distortion, int32_t num_reserved_ids,
                      int32_t num_shards, int32_t shard);

  float Probability(int64_t value) const override;

  int64_t Sample(random::SimplePhilox* rnd) const override;

 private:
  // Underlying distribution sampler.
  std::unique_ptr<random::DistributionSampler> dist_sampler_;
  // Weights for individual samples. The probability of a sample i is defined
  // as weights_.at(i) / total_weight_.
  std::vector<float> weights_;
  // The total weights of all samples.
  float total_weight_;
  // Sharding information of the sampler. The whole vocabulary is sharded
  // into num_shards_ smaller ranges and each sampler is responsible for one
  // such smaller range, identified by the shard number.
  int32 num_shards_;
  int32 shard_;

  // Fill the sampler with the appropriate number of reserved IDs.
  void FillReservedIds(int32_t num_reserved_ids);
  // Load IDs to sample from a CSV file. It is assumed that the last item of
  // each row contains a count or probability for the corresponding ID.
  Status LoadFromFile(Env* env, const string& vocab_file, float distortion);
  // Load from an in-memory array.
  void LoadFromUnigrams(const std::vector<float>& unigrams, float distortion);
};

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_KERNELS_RANGE_SAMPLER_H_
