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
class MHTracer_DTPStensorflowPScorePSkernelsPSrange_samplerDTcc {
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
   MHTracer_DTPStensorflowPScorePSkernelsPSrange_samplerDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSkernelsPSrange_samplerDTcc() {
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

#include "tensorflow/core/kernels/range_sampler.h"

#include <cmath>
#include <unordered_set>
#include <vector>

#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/gtl/map_util.h"
#include "tensorflow/core/lib/io/inputbuffer.h"
#include "tensorflow/core/lib/strings/numbers.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {

using gtl::ArraySlice;
using gtl::MutableArraySlice;

RangeSampler::~RangeSampler() {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSrange_samplerDTcc mht_0(mht_0_v, 205, "", "./tensorflow/core/kernels/range_sampler.cc", "RangeSampler::~RangeSampler");
}

void RangeSampler::SampleBatch(random::SimplePhilox* rnd, bool unique,
                               gtl::MutableArraySlice<int64_t> batch) const {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSrange_samplerDTcc mht_1(mht_1_v, 211, "", "./tensorflow/core/kernels/range_sampler.cc", "RangeSampler::SampleBatch");

  SampleBatchGetExpectedCount(
      rnd, unique, batch, gtl::MutableArraySlice<float>(),
      gtl::ArraySlice<int64_t>(), gtl::MutableArraySlice<float>());
}

void RangeSampler::SampleBatchGetExpectedCount(
    random::SimplePhilox* rnd, bool unique,
    gtl::MutableArraySlice<int64_t> batch,
    gtl::MutableArraySlice<float> batch_expected_count,
    gtl::ArraySlice<int64_t> extras,
    gtl::MutableArraySlice<float> extras_expected_count) const {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSrange_samplerDTcc mht_2(mht_2_v, 225, "", "./tensorflow/core/kernels/range_sampler.cc", "RangeSampler::SampleBatchGetExpectedCount");

  SampleBatchGetExpectedCountAvoid(rnd, unique, batch, batch_expected_count,
                                   extras, extras_expected_count,
                                   gtl::ArraySlice<int64_t>());
}

namespace {

// Approximates the expected count of a value in the output of SampleBatch.
//
// If unique=false, then this is (Probability(value) * batch_size)
//
// We use batch_size and num_tries, where num_tries is the observed number of
// tries it took to get batch_size unique values.
//
// Assuming (falsely) that the number of tries to get a batch of batch_size
// distinct values is _always_ num_tries, the probability that the value
// is in a batch is (1 - (1-p)^num_tries)
static float ExpectedCountHelper(float p, int batch_size, int num_tries) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSrange_samplerDTcc mht_3(mht_3_v, 246, "", "./tensorflow/core/kernels/range_sampler.cc", "ExpectedCountHelper");

  if (num_tries == batch_size) {
    // This shortcut will always be taken if unique=false
    return p * batch_size;
  }
  // numerically stable version of (1 - (1-p)^num_tries)
  return -std::expm1(num_tries * std::log1p(-p));
}

}  // namespace

void RangeSampler::SampleBatchGetExpectedCountAvoid(
    random::SimplePhilox* rnd, bool unique, MutableArraySlice<int64_t> batch,
    MutableArraySlice<float> batch_expected_count, ArraySlice<int64_t> extras,
    MutableArraySlice<float> extras_expected_count,
    ArraySlice<int64_t> avoided_values) const {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSrange_samplerDTcc mht_4(mht_4_v, 264, "", "./tensorflow/core/kernels/range_sampler.cc", "RangeSampler::SampleBatchGetExpectedCountAvoid");

  const int batch_size = batch.size();
  int num_tries;

  if (unique) {
    CHECK_LE(static_cast<int64_t>(batch_size + avoided_values.size()), range_);
    std::unordered_set<int64_t> used(batch_size);
    used.insert(avoided_values.begin(), avoided_values.end());
    int num_picked = 0;
    num_tries = 0;
    while (num_picked < batch_size) {
      num_tries++;
      CHECK_LT(num_tries, kint32max);
      int64_t value = Sample(rnd);
      if (gtl::InsertIfNotPresent(&used, value)) {
        batch[num_picked++] = value;
      }
    }
  } else {
    CHECK_EQ(avoided_values.size(), size_t{0})
        << "avoided_values only supported with unique=true";
    for (int i = 0; i < batch_size; i++) {
      batch[i] = Sample(rnd);
    }
    num_tries = batch_size;
  }
  // Compute the expected counts of the batch and the extra values
  if (!batch_expected_count.empty()) {
    CHECK_EQ(batch_size, batch_expected_count.size());
    for (int i = 0; i < batch_size; i++) {
      batch_expected_count[i] =
          ExpectedCountHelper(Probability(batch[i]), batch_size, num_tries);
    }
  }
  CHECK_EQ(extras.size(), extras_expected_count.size());
  for (size_t i = 0; i < extras.size(); i++) {
    extras_expected_count[i] =
        ExpectedCountHelper(Probability(extras[i]), batch_size, num_tries);
  }
}

AllSampler::AllSampler(int64_t range) : RangeSampler(range) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSrange_samplerDTcc mht_5(mht_5_v, 308, "", "./tensorflow/core/kernels/range_sampler.cc", "AllSampler::AllSampler");
}

void AllSampler::SampleBatchGetExpectedCountAvoid(
    random::SimplePhilox* rnd, bool unique, MutableArraySlice<int64_t> batch,
    MutableArraySlice<float> batch_expected_count, ArraySlice<int64_t> extras,
    MutableArraySlice<float> extras_expected_count,
    ArraySlice<int64_t> avoided_values) const {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSrange_samplerDTcc mht_6(mht_6_v, 317, "", "./tensorflow/core/kernels/range_sampler.cc", "AllSampler::SampleBatchGetExpectedCountAvoid");

  const int batch_size = batch.size();
  CHECK_EQ(range_, batch_size);
  for (int i = 0; i < batch_size; i++) {
    batch[i] = i;
  }
  if (!batch_expected_count.empty()) {
    CHECK_EQ(batch_size, batch_expected_count.size());
    for (int i = 0; i < batch_size; i++) {
      batch_expected_count[i] = 1;
    }
  }
  CHECK_EQ(size_t{0}, avoided_values.size());
  CHECK_EQ(extras.size(), extras_expected_count.size());
  for (size_t i = 0; i < extras.size(); i++) {
    extras_expected_count[i] = 1;
  }
}

UniformSampler::UniformSampler(int64_t range)
    : RangeSampler(range), inv_range_(1.0 / range) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSrange_samplerDTcc mht_7(mht_7_v, 340, "", "./tensorflow/core/kernels/range_sampler.cc", "UniformSampler::UniformSampler");
}

int64_t UniformSampler::Sample(random::SimplePhilox* rnd) const {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSrange_samplerDTcc mht_8(mht_8_v, 345, "", "./tensorflow/core/kernels/range_sampler.cc", "UniformSampler::Sample");

  return rnd->Uniform64(range_);
}

float UniformSampler::Probability(int64_t value) const {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSrange_samplerDTcc mht_9(mht_9_v, 352, "", "./tensorflow/core/kernels/range_sampler.cc", "UniformSampler::Probability");
 return inv_range_; }

LogUniformSampler::LogUniformSampler(int64_t range)
    : RangeSampler(range), log_range_(log1p(range)) {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSrange_samplerDTcc mht_10(mht_10_v, 358, "", "./tensorflow/core/kernels/range_sampler.cc", "LogUniformSampler::LogUniformSampler");
}

int64_t LogUniformSampler::Sample(random::SimplePhilox* rnd) const {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSrange_samplerDTcc mht_11(mht_11_v, 363, "", "./tensorflow/core/kernels/range_sampler.cc", "LogUniformSampler::Sample");

  const int64_t value =
      static_cast<int64_t>(exp(rnd->RandDouble() * log_range_)) - 1;
  DCHECK_GE(value, 0);
  // Mathematically, value should be <= range_, but might not be due to some
  // floating point roundoff, so we mod by range_.  In practice this case
  // happens never regardless of the value of range_, including and up to
  // DBL_MAX.  But we include it as a guarantee of the function's output.
  return value % range_;
}

float LogUniformSampler::Probability(int64_t value) const {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSrange_samplerDTcc mht_12(mht_12_v, 377, "", "./tensorflow/core/kernels/range_sampler.cc", "LogUniformSampler::Probability");

  // value is returned iff the call to UniformDouble(log_range_) in the
  // Sample() function returns a value between log(value + 1)
  // and log(value + 2).   The probability of this is:
  // (log(value + 2) - log(value + 1)) / log_range
  // To avoid two calls to log(), we compute this as follows:
  return (log((value + 2.0) / (value + 1.0))) / log_range_;
}

ThreadUnsafeUnigramSampler::ThreadUnsafeUnigramSampler(int64_t range)
    : RangeSampler(range), picker_(range) {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSrange_samplerDTcc mht_13(mht_13_v, 390, "", "./tensorflow/core/kernels/range_sampler.cc", "ThreadUnsafeUnigramSampler::ThreadUnsafeUnigramSampler");

  CHECK_LT(range, kint32max);
}

int64_t ThreadUnsafeUnigramSampler::Sample(random::SimplePhilox* rnd) const {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSrange_samplerDTcc mht_14(mht_14_v, 397, "", "./tensorflow/core/kernels/range_sampler.cc", "ThreadUnsafeUnigramSampler::Sample");

  return picker_.Pick(rnd);
}

float ThreadUnsafeUnigramSampler::Probability(int64_t value) const {
   std::vector<std::string> mht_15_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSrange_samplerDTcc mht_15(mht_15_v, 404, "", "./tensorflow/core/kernels/range_sampler.cc", "ThreadUnsafeUnigramSampler::Probability");

  return static_cast<float>(picker_.get_weight(value)) / picker_.total_weight();
}

void ThreadUnsafeUnigramSampler::Update(ArraySlice<int64_t> values) {
   std::vector<std::string> mht_16_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSrange_samplerDTcc mht_16(mht_16_v, 411, "", "./tensorflow/core/kernels/range_sampler.cc", "ThreadUnsafeUnigramSampler::Update");

  int num_updates = std::min(static_cast<int>(values.size()),
                             kint32max - picker_.total_weight());
  for (int i = 0; i < num_updates; i++) {
    const int64_t value = values[i];
    picker_.set_weight(value, picker_.get_weight(value) + 1);
  }
}

// Thread-safe unigram sampler
UnigramSampler::UnigramSampler(int64_t range)
    : RangeSampler(range), unsafe_sampler_(range) {
   std::vector<std::string> mht_17_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSrange_samplerDTcc mht_17(mht_17_v, 425, "", "./tensorflow/core/kernels/range_sampler.cc", "UnigramSampler::UnigramSampler");

  CHECK_LT(range, kint32max);
}

int64_t UnigramSampler::Sample(random::SimplePhilox* rnd) const {
   std::vector<std::string> mht_18_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSrange_samplerDTcc mht_18(mht_18_v, 432, "", "./tensorflow/core/kernels/range_sampler.cc", "UnigramSampler::Sample");

  tf_shared_lock lock(mu_);
  return unsafe_sampler_.Sample(rnd);
}

float UnigramSampler::Probability(int64_t value) const {
   std::vector<std::string> mht_19_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSrange_samplerDTcc mht_19(mht_19_v, 440, "", "./tensorflow/core/kernels/range_sampler.cc", "UnigramSampler::Probability");

  tf_shared_lock lock(mu_);
  return unsafe_sampler_.Probability(value);
}

// Overriding at a high level results in far fewer lock acquisitions.
void UnigramSampler::SampleBatchGetExpectedCountAvoid(
    random::SimplePhilox* rnd, bool unique, MutableArraySlice<int64_t> batch,
    MutableArraySlice<float> batch_expected_count, ArraySlice<int64_t> extras,
    MutableArraySlice<float> extras_expected_count,
    ArraySlice<int64_t> avoided_values) const {
   std::vector<std::string> mht_20_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSrange_samplerDTcc mht_20(mht_20_v, 453, "", "./tensorflow/core/kernels/range_sampler.cc", "UnigramSampler::SampleBatchGetExpectedCountAvoid");

  tf_shared_lock lock(mu_);
  unsafe_sampler_.SampleBatchGetExpectedCountAvoid(
      rnd, unique, batch, batch_expected_count, extras, extras_expected_count,
      avoided_values);
}

void UnigramSampler::Update(ArraySlice<int64_t> values) {
   std::vector<std::string> mht_21_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSrange_samplerDTcc mht_21(mht_21_v, 463, "", "./tensorflow/core/kernels/range_sampler.cc", "UnigramSampler::Update");

  mutex_lock lock(mu_);
  unsafe_sampler_.Update(values);
}

FixedUnigramSampler::FixedUnigramSampler(Env* env, int64_t range,
                                         const string& vocab_file,
                                         float distortion,
                                         int32_t num_reserved_ids,
                                         int32_t num_shards, int32_t shard)
    : RangeSampler(range),
      total_weight_(0.0),
      num_shards_(num_shards),
      shard_(shard) {
   std::vector<std::string> mht_22_v;
   mht_22_v.push_back("vocab_file: \"" + vocab_file + "\"");
   MHTracer_DTPStensorflowPScorePSkernelsPSrange_samplerDTcc mht_22(mht_22_v, 480, "", "./tensorflow/core/kernels/range_sampler.cc", "FixedUnigramSampler::FixedUnigramSampler");

  FillReservedIds(num_reserved_ids);
  // TODO(vanhoucke): make this non-crashing.
  TF_CHECK_OK(LoadFromFile(env, vocab_file, distortion));
  CHECK_EQ(range, weights_.size());
  dist_sampler_.reset(new random::DistributionSampler(weights_));
}

FixedUnigramSampler::FixedUnigramSampler(int64_t range,
                                         const std::vector<float>& unigrams,
                                         float distortion,
                                         int32_t num_reserved_ids,
                                         int32_t num_shards, int32_t shard)
    : RangeSampler(range),
      total_weight_(0.0),
      num_shards_(num_shards),
      shard_(shard) {
   std::vector<std::string> mht_23_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSrange_samplerDTcc mht_23(mht_23_v, 499, "", "./tensorflow/core/kernels/range_sampler.cc", "FixedUnigramSampler::FixedUnigramSampler");

  FillReservedIds(num_reserved_ids);
  LoadFromUnigrams(unigrams, distortion);
  // TODO(vanhoucke): make this non-crashing.
  CHECK_EQ(range, weights_.size());
  dist_sampler_.reset(new random::DistributionSampler(weights_));
}

float FixedUnigramSampler::Probability(int64_t value) const {
   std::vector<std::string> mht_24_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSrange_samplerDTcc mht_24(mht_24_v, 510, "", "./tensorflow/core/kernels/range_sampler.cc", "FixedUnigramSampler::Probability");

  if (value < 0 || static_cast<size_t>(value) >= weights_.size()) {
    return 0.0;
  }
  return weights_.at(value) / total_weight_;
}

int64_t FixedUnigramSampler::Sample(random::SimplePhilox* rnd) const {
   std::vector<std::string> mht_25_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSrange_samplerDTcc mht_25(mht_25_v, 520, "", "./tensorflow/core/kernels/range_sampler.cc", "FixedUnigramSampler::Sample");

  return dist_sampler_->Sample(rnd);
}

void FixedUnigramSampler::FillReservedIds(int32_t num_reserved_ids) {
   std::vector<std::string> mht_26_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSrange_samplerDTcc mht_26(mht_26_v, 527, "", "./tensorflow/core/kernels/range_sampler.cc", "FixedUnigramSampler::FillReservedIds");

  for (int32_t word_id = 0; word_id < num_reserved_ids; ++word_id) {
    if (word_id % num_shards_ == shard_) weights_.push_back(0.0);
  }
}

Status FixedUnigramSampler::LoadFromFile(Env* env, const string& vocab_file,
                                         float distortion) {
   std::vector<std::string> mht_27_v;
   mht_27_v.push_back("vocab_file: \"" + vocab_file + "\"");
   MHTracer_DTPStensorflowPScorePSkernelsPSrange_samplerDTcc mht_27(mht_27_v, 538, "", "./tensorflow/core/kernels/range_sampler.cc", "FixedUnigramSampler::LoadFromFile");

  std::unique_ptr<RandomAccessFile> file;
  TF_RETURN_IF_ERROR(env->NewRandomAccessFile(vocab_file, &file));

  io::InputBuffer in(file.get(), 262144 /*bytes*/);
  string line;
  int32_t word_id = weights_.size();
  while (in.ReadLine(&line).ok()) {
    // The vocabulary file should be in csv like format, with the last
    // field the weight associated with the word.
    std::vector<string> cols = str_util::Split(line, ',');
    if (cols.empty()) continue;
    // Skip entries that do not belong to this shard.
    if (word_id % num_shards_ == shard_) {
      float w = 0.0;
      if (!strings::safe_strtof(cols.at(cols.size() - 1), &w)) {
        return errors::InvalidArgument("Wrong vocabulary format at line: ",
                                       line);
      }
      w = std::pow(w, distortion);
      total_weight_ += w;
      weights_.push_back(w);
    }
    ++word_id;
  }
  return Status::OK();
}

void FixedUnigramSampler::LoadFromUnigrams(const std::vector<float>& unigrams,
                                           float distortion) {
   std::vector<std::string> mht_28_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSrange_samplerDTcc mht_28(mht_28_v, 570, "", "./tensorflow/core/kernels/range_sampler.cc", "FixedUnigramSampler::LoadFromUnigrams");

  int32_t word_id = weights_.size();
  for (float w : unigrams) {
    // Skip entries that do not belong to this shard.
    if (word_id % num_shards_ == shard_) {
      w = std::pow(w, distortion);
      total_weight_ += w;
      weights_.push_back(w);
    }
    ++word_id;
  }
}

}  // namespace tensorflow
