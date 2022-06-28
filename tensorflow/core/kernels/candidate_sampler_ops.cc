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
class MHTracer_DTPStensorflowPScorePSkernelsPScandidate_sampler_opsDTcc {
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
   MHTracer_DTPStensorflowPScorePSkernelsPScandidate_sampler_opsDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSkernelsPScandidate_sampler_opsDTcc() {
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

// See docs in ../ops/candidate_sampling_ops.cc.

#define EIGEN_USE_THREADS

#include <cfloat>
#include <unordered_map>
#include <vector>

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/kernels/range_sampler.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/util/guarded_philox_random.h"

namespace tensorflow {

class BaseCandidateSamplerOp : public OpKernel {
 public:
  explicit BaseCandidateSamplerOp(OpKernelConstruction* context)
      : OpKernel(context) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSkernelsPScandidate_sampler_opsDTcc mht_0(mht_0_v, 204, "", "./tensorflow/core/kernels/candidate_sampler_ops.cc", "BaseCandidateSamplerOp");

    OP_REQUIRES_OK(context, context->GetAttr("num_sampled", &num_sampled_));
    OP_REQUIRES_OK(context, context->GetAttr("num_true", &num_true_));
    OP_REQUIRES_OK(context, context->GetAttr("unique", &unique_));
    OP_REQUIRES_OK(context, generator_.Init(context));
  }

  void Compute(OpKernelContext* context) override {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSkernelsPScandidate_sampler_opsDTcc mht_1(mht_1_v, 214, "", "./tensorflow/core/kernels/candidate_sampler_ops.cc", "Compute");

    const Tensor& true_classes = context->input(0);
    OP_REQUIRES(context, true_classes.dims() == 2,
                errors::InvalidArgument("true_classes must be a matrix"));
    const int32_t batch_size = true_classes.dim_size(0);
    OP_REQUIRES(
        context, true_classes.dim_size(1) == num_true_,
        errors::InvalidArgument("true_classes must have "
                                "num_true columns, expected: ",
                                true_classes.dim_size(1), " was: ", num_true_));
    CHECK(sampler_) << "CandidateSamplerOp did not set sampler_";

    if (unique_) {
      OP_REQUIRES(context, num_sampled_ <= sampler_->range(),
                  errors::InvalidArgument("Sampler's range is too small."));
    }

    // Output candidates and expected_count.
    Tensor* out_sampled_candidates = nullptr;
    OP_REQUIRES_OK(context,
                   context->allocate_output(0, TensorShape({num_sampled_}),
                                            &out_sampled_candidates));

    Tensor* out_true_expected_count = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(
                                1, TensorShape({batch_size, num_true_}),
                                &out_true_expected_count));
    Tensor* out_sampled_expected_count = nullptr;
    OP_REQUIRES_OK(context,
                   context->allocate_output(2, TensorShape({num_sampled_}),
                                            &out_sampled_expected_count));

    gtl::ArraySlice<int64_t> true_candidate(
        true_classes.matrix<int64_t>().data(), batch_size * num_true_);
    gtl::MutableArraySlice<int64_t> sampled_candidate(
        out_sampled_candidates->vec<int64_t>().data(), num_sampled_);
    gtl::MutableArraySlice<float> true_expected_count(
        out_true_expected_count->matrix<float>().data(),
        batch_size * num_true_);
    gtl::MutableArraySlice<float> sampled_expected_count(
        out_sampled_expected_count->vec<float>().data(), num_sampled_);

    // Approximately conservatively estimate the number of samples required.
    // In cases where rejection sampling is used we may occasionally use more
    // samples than expected, which will result in reused random bits.
    const int64_t samples32 = 2048 * num_sampled_;

    // Pick sampled candidates.
    auto local_gen = generator_.ReserveSamples32(samples32);
    random::SimplePhilox random(&local_gen);
    sampler_->SampleBatchGetExpectedCount(&random, unique_, sampled_candidate,
                                          sampled_expected_count,
                                          true_candidate, true_expected_count);

    if (sampler_->NeedsUpdates()) {
      sampler_->Update(true_candidate);
    }
  }

 protected:
  void set_sampler(RangeSampler* sampler) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSkernelsPScandidate_sampler_opsDTcc mht_2(mht_2_v, 277, "", "./tensorflow/core/kernels/candidate_sampler_ops.cc", "set_sampler");
 sampler_.reset(sampler); }

 private:
  int32 num_true_;
  int32 num_sampled_;
  bool unique_;
  std::unique_ptr<RangeSampler> sampler_;
  GuardedPhiloxRandom generator_;
};

template <class RangeSamplerType>
class SimpleCandidateSamplerOp : public BaseCandidateSamplerOp {
 public:
  explicit SimpleCandidateSamplerOp(OpKernelConstruction* context)
      : BaseCandidateSamplerOp(context) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSkernelsPScandidate_sampler_opsDTcc mht_3(mht_3_v, 294, "", "./tensorflow/core/kernels/candidate_sampler_ops.cc", "SimpleCandidateSamplerOp");

    int64_t range_max;
    OP_REQUIRES_OK(context, context->GetAttr("range_max", &range_max));
    set_sampler(new RangeSamplerType(range_max));
  }
};

REGISTER_KERNEL_BUILDER(Name("UniformCandidateSampler").Device(DEVICE_CPU),
                        SimpleCandidateSamplerOp<UniformSampler>);

REGISTER_KERNEL_BUILDER(Name("LogUniformCandidateSampler").Device(DEVICE_CPU),
                        SimpleCandidateSamplerOp<LogUniformSampler>);

REGISTER_KERNEL_BUILDER(
    Name("LearnedUnigramCandidateSampler").Device(DEVICE_CPU),
    SimpleCandidateSamplerOp<UnigramSampler>);

REGISTER_KERNEL_BUILDER(
    Name("ThreadUnsafeUnigramCandidateSampler").Device(DEVICE_CPU),
    SimpleCandidateSamplerOp<ThreadUnsafeUnigramSampler>);

class AllCandidateSamplerOp : public BaseCandidateSamplerOp {
 public:
  explicit AllCandidateSamplerOp(OpKernelConstruction* context)
      : BaseCandidateSamplerOp(context) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSkernelsPScandidate_sampler_opsDTcc mht_4(mht_4_v, 321, "", "./tensorflow/core/kernels/candidate_sampler_ops.cc", "AllCandidateSamplerOp");

    int64_t range_max;
    OP_REQUIRES_OK(context, context->GetAttr("num_sampled", &range_max));
    set_sampler(new AllSampler(range_max));
  }
};

REGISTER_KERNEL_BUILDER(Name("AllCandidateSampler").Device(DEVICE_CPU),
                        AllCandidateSamplerOp);

class FixedUnigramCandidateSamplerOp : public BaseCandidateSamplerOp {
 public:
  explicit FixedUnigramCandidateSamplerOp(OpKernelConstruction* context)
      : BaseCandidateSamplerOp(context) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSkernelsPScandidate_sampler_opsDTcc mht_5(mht_5_v, 337, "", "./tensorflow/core/kernels/candidate_sampler_ops.cc", "FixedUnigramCandidateSamplerOp");

    int64_t range_max;
    OP_REQUIRES_OK(context, context->GetAttr("range_max", &range_max));
    string vocab_file;
    OP_REQUIRES_OK(context, context->GetAttr("vocab_file", &vocab_file));
    std::vector<float> unigrams;
    OP_REQUIRES_OK(context, context->GetAttr("unigrams", &unigrams));
    OP_REQUIRES(
        context, !vocab_file.empty() || !unigrams.empty(),
        errors::InvalidArgument("Must provide either vocab_file or unigrams."));
    OP_REQUIRES(context, vocab_file.empty() || unigrams.empty(),
                errors::InvalidArgument(
                    "Must only provide one of vocab_file and unigrams."));
    float distortion;
    OP_REQUIRES_OK(context, context->GetAttr("distortion", &distortion));
    int64_t num_reserved_ids;
    OP_REQUIRES_OK(context,
                   context->GetAttr("num_reserved_ids", &num_reserved_ids));
    int64_t num_shards;
    OP_REQUIRES_OK(context, context->GetAttr("num_shards", &num_shards));
    int64_t shard;
    OP_REQUIRES_OK(context, context->GetAttr("shard", &shard));

    if (!vocab_file.empty()) {
      set_sampler(new FixedUnigramSampler(context->env(), range_max, vocab_file,
                                          distortion, num_reserved_ids,
                                          num_shards, shard));
    } else {
      set_sampler(new FixedUnigramSampler(range_max, unigrams, distortion,
                                          num_reserved_ids, num_shards, shard));
    }
  }
};

REGISTER_KERNEL_BUILDER(Name("FixedUnigramCandidateSampler").Device(DEVICE_CPU),
                        FixedUnigramCandidateSamplerOp);

class ComputeAccidentalHitsOp : public OpKernel {
 public:
  explicit ComputeAccidentalHitsOp(OpKernelConstruction* context)
      : OpKernel(context) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePSkernelsPScandidate_sampler_opsDTcc mht_6(mht_6_v, 380, "", "./tensorflow/core/kernels/candidate_sampler_ops.cc", "ComputeAccidentalHitsOp");

    OP_REQUIRES_OK(context, context->GetAttr("num_true", &num_true_));
  }

  void Compute(OpKernelContext* context) override {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePSkernelsPScandidate_sampler_opsDTcc mht_7(mht_7_v, 387, "", "./tensorflow/core/kernels/candidate_sampler_ops.cc", "Compute");

    const Tensor& in_true_candidates = context->input(0);
    const TensorShape& in_true_candidates_shape = in_true_candidates.shape();
    OP_REQUIRES(context,
                TensorShapeUtils::IsMatrix(in_true_candidates_shape) &&
                    in_true_candidates_shape.dim_size(1) == num_true_,
                errors::InvalidArgument(
                    "true_candidates must be a batch_size * num_true matrix"));

    const int64_t batch_size = in_true_candidates_shape.dim_size(0);

    const Tensor& in_sampled_candidates = context->input(1);
    OP_REQUIRES(context,
                TensorShapeUtils::IsVector(in_sampled_candidates.shape()),
                errors::InvalidArgument(
                    "sampled_candidates must be a vector, which is typically "
                    "an output from CandidateSampler"));

    std::unordered_map<int64_t, int> sampled_candidate_to_pos;
    for (int64_t i = 0; i < in_sampled_candidates.dim_size(0); ++i) {
      sampled_candidate_to_pos[in_sampled_candidates.vec<int64_t>()(i)] = i;
    }

    // Produce output in the same format as UnpackSparseFeatures.
    std::vector<int> indices;
    std::vector<int64_t> ids;
    std::vector<float> weights;

    for (int64_t i = 0; i < batch_size; ++i) {
      for (int64_t j = 0; j < num_true_; ++j) {
        const int64_t true_candidate =
            in_true_candidates.matrix<int64_t>()(i, j);
        const auto look = sampled_candidate_to_pos.find(true_candidate);
        if (look != sampled_candidate_to_pos.end()) {
          indices.push_back(i);
          ids.push_back(look->second);
          weights.push_back(-FLT_MAX);
        }
      }
    }

    Tensor* out_indices = nullptr;
    OP_REQUIRES_OK(
        context,
        context->allocate_output(
            0, TensorShape({static_cast<int>(indices.size())}), &out_indices));
    Tensor* out_ids = nullptr;
    OP_REQUIRES_OK(
        context, context->allocate_output(
                     1, TensorShape({static_cast<int>(ids.size())}), &out_ids));
    Tensor* out_weights = nullptr;
    OP_REQUIRES_OK(
        context,
        context->allocate_output(
            2, TensorShape({static_cast<int>(weights.size())}), &out_weights));

    for (size_t i = 0; i < indices.size(); ++i) {
      out_indices->vec<int32>()(i) = indices[i];
      out_ids->vec<int64_t>()(i) = ids[i];
      out_weights->vec<float>()(i) = weights[i];
    }
  }

 private:
  int64_t num_true_;
};

REGISTER_KERNEL_BUILDER(Name("ComputeAccidentalHits").Device(DEVICE_CPU),
                        ComputeAccidentalHitsOp);

}  // namespace tensorflow
