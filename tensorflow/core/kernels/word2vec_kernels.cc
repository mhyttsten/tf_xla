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
class MHTracer_DTPStensorflowPScorePSkernelsPSword2vec_kernelsDTcc {
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
   MHTracer_DTPStensorflowPScorePSkernelsPSword2vec_kernelsDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSkernelsPSword2vec_kernelsDTcc() {
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

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/lib/core/stringpiece.h"
#include "tensorflow/core/lib/gtl/map_util.h"
#include "tensorflow/core/lib/random/distribution_sampler.h"
#include "tensorflow/core/lib/random/philox_random.h"
#include "tensorflow/core/lib/random/simple_philox.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/platform/thread_annotations.h"
#include "tensorflow/core/util/guarded_philox_random.h"

namespace tensorflow {

// Number of examples to precalculate.
const int kPrecalc = 3000;
// Number of words to read into a sentence before processing.
const int kSentenceSize = 1000;

namespace {

bool ScanWord(StringPiece* input, string* word) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSword2vec_kernelsDTcc mht_0(mht_0_v, 205, "", "./tensorflow/core/kernels/word2vec_kernels.cc", "ScanWord");

  str_util::RemoveLeadingWhitespace(input);
  StringPiece tmp;
  if (str_util::ConsumeNonWhitespace(input, &tmp)) {
    word->assign(tmp.data(), tmp.size());
    return true;
  } else {
    return false;
  }
}

}  // end namespace

class SkipgramOp : public OpKernel {
 public:
  explicit SkipgramOp(OpKernelConstruction* ctx)
      : OpKernel(ctx), rng_(&philox_) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSword2vec_kernelsDTcc mht_1(mht_1_v, 224, "", "./tensorflow/core/kernels/word2vec_kernels.cc", "SkipgramOp");

    string filename;
    OP_REQUIRES_OK(ctx, ctx->GetAttr("filename", &filename));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("batch_size", &batch_size_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("window_size", &window_size_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("min_count", &min_count_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("subsample", &subsample_));
    OP_REQUIRES_OK(ctx, Init(ctx->env(), filename));

    mutex_lock l(mu_);
    example_pos_ = corpus_size_;
    label_pos_ = corpus_size_;
    label_limit_ = corpus_size_;
    sentence_index_ = kSentenceSize;
    for (int i = 0; i < kPrecalc; ++i) {
      NextExample(&precalc_examples_[i].input, &precalc_examples_[i].label);
    }
  }

  void Compute(OpKernelContext* ctx) override {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSword2vec_kernelsDTcc mht_2(mht_2_v, 246, "", "./tensorflow/core/kernels/word2vec_kernels.cc", "Compute");

    Tensor words_per_epoch(DT_INT64, TensorShape({}));
    Tensor current_epoch(DT_INT32, TensorShape({}));
    Tensor total_words_processed(DT_INT64, TensorShape({}));
    Tensor examples(DT_INT32, TensorShape({batch_size_}));
    auto Texamples = examples.flat<int32>();
    Tensor labels(DT_INT32, TensorShape({batch_size_}));
    auto Tlabels = labels.flat<int32>();
    {
      mutex_lock l(mu_);
      for (int i = 0; i < batch_size_; ++i) {
        Texamples(i) = precalc_examples_[precalc_index_].input;
        Tlabels(i) = precalc_examples_[precalc_index_].label;
        precalc_index_++;
        if (precalc_index_ >= kPrecalc) {
          precalc_index_ = 0;
          for (int j = 0; j < kPrecalc; ++j) {
            NextExample(&precalc_examples_[j].input,
                        &precalc_examples_[j].label);
          }
        }
      }
      words_per_epoch.scalar<int64_t>()() = corpus_size_;
      current_epoch.scalar<int32>()() = current_epoch_;
      total_words_processed.scalar<int64_t>()() = total_words_processed_;
    }
    ctx->set_output(0, word_);
    ctx->set_output(1, freq_);
    ctx->set_output(2, words_per_epoch);
    ctx->set_output(3, current_epoch);
    ctx->set_output(4, total_words_processed);
    ctx->set_output(5, examples);
    ctx->set_output(6, labels);
  }

 private:
  struct Example {
    int32 input;
    int32 label;
  };

  int32 batch_size_ = 0;
  int32 window_size_ = 5;
  float subsample_ = 1e-3;
  int min_count_ = 5;
  int32 vocab_size_ = 0;
  Tensor word_;
  Tensor freq_;
  int64_t corpus_size_ = 0;
  std::vector<int32> corpus_;
  std::vector<Example> precalc_examples_;
  int precalc_index_ = 0;
  std::vector<int32> sentence_;
  int sentence_index_ = 0;

  mutex mu_;
  random::PhiloxRandom philox_ TF_GUARDED_BY(mu_);
  random::SimplePhilox rng_ TF_GUARDED_BY(mu_);
  int32 current_epoch_ TF_GUARDED_BY(mu_) = -1;
  int64_t total_words_processed_ TF_GUARDED_BY(mu_) = 0;
  int32 example_pos_ TF_GUARDED_BY(mu_);
  int32 label_pos_ TF_GUARDED_BY(mu_);
  int32 label_limit_ TF_GUARDED_BY(mu_);

  // {example_pos_, label_pos_} is the cursor for the next example.
  // example_pos_ wraps around at the end of corpus_. For each
  // example, we randomly generate [label_pos_, label_limit) for
  // labels.
  void NextExample(int32* example, int32* label)
      TF_EXCLUSIVE_LOCKS_REQUIRED(mu_) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSword2vec_kernelsDTcc mht_3(mht_3_v, 318, "", "./tensorflow/core/kernels/word2vec_kernels.cc", "NextExample");

    while (true) {
      if (label_pos_ >= label_limit_) {
        ++total_words_processed_;
        ++sentence_index_;
        if (sentence_index_ >= kSentenceSize) {
          sentence_index_ = 0;
          for (int i = 0; i < kSentenceSize; ++i, ++example_pos_) {
            if (example_pos_ >= corpus_size_) {
              ++current_epoch_;
              example_pos_ = 0;
            }
            if (subsample_ > 0) {
              int32_t word_freq = freq_.flat<int32>()(corpus_[example_pos_]);
              // See Eq. 5 in http://arxiv.org/abs/1310.4546
              float keep_prob =
                  (std::sqrt(word_freq / (subsample_ * corpus_size_)) + 1) *
                  (subsample_ * corpus_size_) / word_freq;
              if (rng_.RandFloat() > keep_prob) {
                i--;
                continue;
              }
            }
            sentence_[i] = corpus_[example_pos_];
          }
        }
        const int32_t skip = 1 + rng_.Uniform(window_size_);
        label_pos_ = std::max<int32>(0, sentence_index_ - skip);
        label_limit_ =
            std::min<int32>(kSentenceSize, sentence_index_ + skip + 1);
      }
      if (sentence_index_ != label_pos_) {
        break;
      }
      ++label_pos_;
    }
    *example = sentence_[sentence_index_];
    *label = sentence_[label_pos_++];
  }

  Status Init(Env* env, const string& filename) {
   std::vector<std::string> mht_4_v;
   mht_4_v.push_back("filename: \"" + filename + "\"");
   MHTracer_DTPStensorflowPScorePSkernelsPSword2vec_kernelsDTcc mht_4(mht_4_v, 362, "", "./tensorflow/core/kernels/word2vec_kernels.cc", "Init");

    string data;
    TF_RETURN_IF_ERROR(ReadFileToString(env, filename, &data));
    StringPiece input = data;
    string w;
    corpus_size_ = 0;
    std::unordered_map<string, int32> word_freq;
    while (ScanWord(&input, &w)) {
      ++(word_freq[w]);
      ++corpus_size_;
    }
    if (corpus_size_ < window_size_ * 10) {
      return errors::InvalidArgument(
          "The text file ", filename,
          " contains too little data: ", corpus_size_, " words");
    }
    typedef std::pair<string, int32> WordFreq;
    std::vector<WordFreq> ordered;
    for (const auto& p : word_freq) {
      if (p.second >= min_count_) ordered.push_back(p);
    }
    LOG(INFO) << "Data file: " << filename << " contains " << data.size()
              << " bytes, " << corpus_size_ << " words, " << word_freq.size()
              << " unique words, " << ordered.size()
              << " unique frequent words.";
    word_freq.clear();
    std::sort(ordered.begin(), ordered.end(),
              [](const WordFreq& x, const WordFreq& y) {
                return x.second > y.second;
              });
    vocab_size_ = static_cast<int32>(1 + ordered.size());
    Tensor word(DT_STRING, TensorShape({vocab_size_}));
    Tensor freq(DT_INT32, TensorShape({vocab_size_}));
    word.flat<tstring>()(0) = "UNK";
    static const int32_t kUnkId = 0;
    std::unordered_map<string, int32> word_id;
    int64_t total_counted = 0;
    for (std::size_t i = 0; i < ordered.size(); ++i) {
      const auto& w = ordered[i].first;
      auto id = i + 1;
      word.flat<tstring>()(id) = w;
      auto word_count = ordered[i].second;
      freq.flat<int32>()(id) = word_count;
      total_counted += word_count;
      word_id[w] = id;
    }
    freq.flat<int32>()(kUnkId) = corpus_size_ - total_counted;
    word_ = word;
    freq_ = freq;
    corpus_.reserve(corpus_size_);
    input = data;
    while (ScanWord(&input, &w)) {
      corpus_.push_back(gtl::FindWithDefault(word_id, w, kUnkId));
    }
    precalc_examples_.resize(kPrecalc);
    sentence_.resize(kSentenceSize);
    return Status::OK();
  }
};

REGISTER_KERNEL_BUILDER(Name("Skipgram").Device(DEVICE_CPU), SkipgramOp);

class NegTrainOp : public OpKernel {
 public:
  explicit NegTrainOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSword2vec_kernelsDTcc mht_5(mht_5_v, 429, "", "./tensorflow/core/kernels/word2vec_kernels.cc", "NegTrainOp");

    base_.Init(0, 0);

    OP_REQUIRES_OK(ctx, ctx->GetAttr("num_negative_samples", &num_samples_));

    std::vector<int32> vocab_count;
    OP_REQUIRES_OK(ctx, ctx->GetAttr("vocab_count", &vocab_count));

    std::vector<float> vocab_weights;
    vocab_weights.reserve(vocab_count.size());
    for (const auto& f : vocab_count) {
      float r = std::pow(static_cast<float>(f), 0.75f);
      vocab_weights.push_back(r);
    }
    sampler_ = new random::DistributionSampler(vocab_weights);
  }

  ~NegTrainOp() override {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSword2vec_kernelsDTcc mht_6(mht_6_v, 449, "", "./tensorflow/core/kernels/word2vec_kernels.cc", "~NegTrainOp");
 delete sampler_; }

  void Compute(OpKernelContext* ctx) override {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSword2vec_kernelsDTcc mht_7(mht_7_v, 454, "", "./tensorflow/core/kernels/word2vec_kernels.cc", "Compute");

    Tensor w_in = ctx->mutable_input(0, false);
    OP_REQUIRES(ctx, TensorShapeUtils::IsMatrix(w_in.shape()),
                errors::InvalidArgument("Must be a matrix"));
    Tensor w_out = ctx->mutable_input(1, false);
    OP_REQUIRES(ctx, w_in.shape() == w_out.shape(),
                errors::InvalidArgument("w_in.shape == w_out.shape"));
    const Tensor& examples = ctx->input(2);
    OP_REQUIRES(ctx, TensorShapeUtils::IsVector(examples.shape()),
                errors::InvalidArgument("Must be a vector"));
    const Tensor& labels = ctx->input(3);
    OP_REQUIRES(ctx, examples.shape() == labels.shape(),
                errors::InvalidArgument("examples.shape == labels.shape"));
    const Tensor& learning_rate = ctx->input(4);
    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(learning_rate.shape()),
                errors::InvalidArgument("Must be a scalar"));

    auto Tw_in = w_in.matrix<float>();
    auto Tw_out = w_out.matrix<float>();
    auto Texamples = examples.flat<int32>();
    auto Tlabels = labels.flat<int32>();
    auto lr = learning_rate.scalar<float>()();
    const int64_t vocab_size = w_in.dim_size(0);
    const int64_t dims = w_in.dim_size(1);
    const int64_t batch_size = examples.dim_size(0);
    OP_REQUIRES(ctx, vocab_size == sampler_->num(),
                errors::InvalidArgument("vocab_size mismatches: ", vocab_size,
                                        " vs. ", sampler_->num()));

    // Gradient accumulator for v_in.
    Tensor buf(DT_FLOAT, TensorShape({dims}));
    auto Tbuf = buf.flat<float>();

    // Scalar buffer to hold sigmoid(+/- dot).
    Tensor g_buf(DT_FLOAT, TensorShape({}));
    auto g = g_buf.scalar<float>();

    // The following loop needs 2 random 32-bit values per negative
    // sample.  We reserve 8 values per sample just in case the
    // underlying implementation changes.
    auto rnd = base_.ReserveSamples32(batch_size * num_samples_ * 8);
    random::SimplePhilox srnd(&rnd);

    for (int64_t i = 0; i < batch_size; ++i) {
      const int32_t example = Texamples(i);
      DCHECK(0 <= example && example < vocab_size) << example;
      const int32_t label = Tlabels(i);
      DCHECK(0 <= label && label < vocab_size) << label;
      auto v_in = Tw_in.chip<0>(example);

      // Positive: example predicts label.
      //   forward: x = v_in' * v_out
      //            l = log(sigmoid(x))
      //   backward: dl/dx = g = sigmoid(-x)
      //             dl/d(v_in) = g * v_out'
      //             dl/d(v_out) = v_in' * g
      {
        auto v_out = Tw_out.chip<0>(label);
        auto dot = (v_in * v_out).sum();
        g = (dot.exp() + 1.f).inverse();
        Tbuf = v_out * (g() * lr);
        v_out += v_in * (g() * lr);
      }

      // Negative samples:
      //   forward: x = v_in' * v_sample
      //            l = log(sigmoid(-x))
      //   backward: dl/dx = g = -sigmoid(x)
      //             dl/d(v_in) = g * v_out'
      //             dl/d(v_out) = v_in' * g
      for (int j = 0; j < num_samples_; ++j) {
        const int sample = sampler_->Sample(&srnd);
        if (sample == label) continue;  // Skip.
        auto v_sample = Tw_out.chip<0>(sample);
        auto dot = (v_in * v_sample).sum();
        g = -((-dot).exp() + 1.f).inverse();
        Tbuf += v_sample * (g() * lr);
        v_sample += v_in * (g() * lr);
      }

      // Applies the gradient on v_in.
      v_in += Tbuf;
    }
  }

 private:
  int32 num_samples_ = 0;
  random::DistributionSampler* sampler_ = nullptr;
  GuardedPhiloxRandom base_;
};

REGISTER_KERNEL_BUILDER(Name("NegTrain").Device(DEVICE_CPU), NegTrainOp);

}  // end namespace tensorflow
