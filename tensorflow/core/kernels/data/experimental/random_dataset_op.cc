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
class MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSrandom_dataset_opDTcc {
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
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSrandom_dataset_opDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSrandom_dataset_opDTcc() {
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
#include "tensorflow/core/kernels/data/experimental/random_dataset_op.h"

#include <string>
#include <utility>

#include "tensorflow/core/data/dataset_utils.h"
#include "tensorflow/core/data/split_utils.h"
#include "tensorflow/core/framework/dataset.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/partial_tensor_shape.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/lib/random/philox_random.h"
#include "tensorflow/core/lib/random/random.h"
#include "tensorflow/core/lib/random/random_distributions.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {
namespace data {
namespace experimental {

// Constants declared in random_dataset_op.h and used both here and in test
// cases.
/* static */ constexpr const char* const RandomDatasetOp::kDatasetType;
/* static */ constexpr const char* const RandomDatasetOp::kSeed;
/* static */ constexpr const char* const RandomDatasetOp::kSeed2;
/* static */ constexpr const char* const RandomDatasetOp::kOutputTypes;
/* static */ constexpr const char* const RandomDatasetOp::kOutputShapes;

class RandomDatasetOp::Dataset : public DatasetBase {
 public:
  Dataset(OpKernelContext* ctx, int64_t seed, int64_t seed2)
      : DatasetBase(DatasetContext(ctx)), seeds_(seed, seed2) {}

  std::unique_ptr<IteratorBase> MakeIteratorInternal(
      const string& prefix) const override {
    return absl::make_unique<Iterator>(
        Iterator::Params{this, strings::StrCat(prefix, "::Random")});
  }

  Status MakeSplitProviders(std::vector<std::unique_ptr<SplitProvider>>*
                                split_providers) const override {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSrandom_dataset_opDTcc mht_0(mht_0_v, 224, "", "./tensorflow/core/kernels/data/experimental/random_dataset_op.cc", "MakeSplitProviders");

    // We use kint64 to generate an effectively infinite number of "splits".
    // These splits aren't actually used during iteration.
    // TODO(aaudibert): Avoid sending dummy splits over RPC when using tf.data
    // service with RandomDataset.
    split_providers->push_back(
        absl::make_unique<IndexSplitProvider>(kint64max));
    return Status::OK();
  }

  const DataTypeVector& output_dtypes() const override {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSrandom_dataset_opDTcc mht_1(mht_1_v, 237, "", "./tensorflow/core/kernels/data/experimental/random_dataset_op.cc", "output_dtypes");

    static DataTypeVector* dtypes = new DataTypeVector({DT_INT64});
    return *dtypes;
  }

  const std::vector<PartialTensorShape>& output_shapes() const override {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSrandom_dataset_opDTcc mht_2(mht_2_v, 245, "", "./tensorflow/core/kernels/data/experimental/random_dataset_op.cc", "output_shapes");

    static std::vector<PartialTensorShape>* shapes =
        new std::vector<PartialTensorShape>({{}});
    return *shapes;
  }

  string DebugString() const override {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSrandom_dataset_opDTcc mht_3(mht_3_v, 254, "", "./tensorflow/core/kernels/data/experimental/random_dataset_op.cc", "DebugString");

    return strings::StrCat("RandomDatasetOp(", seeds_.first, ", ",
                           seeds_.second, ")::Dataset");
  }

  int64_t CardinalityInternal() const override {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSrandom_dataset_opDTcc mht_4(mht_4_v, 262, "", "./tensorflow/core/kernels/data/experimental/random_dataset_op.cc", "CardinalityInternal");
 return kInfiniteCardinality; }

  Status InputDatasets(std::vector<const DatasetBase*>* inputs) const override {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSrandom_dataset_opDTcc mht_5(mht_5_v, 267, "", "./tensorflow/core/kernels/data/experimental/random_dataset_op.cc", "InputDatasets");

    return Status::OK();
  }

  Status CheckExternalState() const override {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSrandom_dataset_opDTcc mht_6(mht_6_v, 274, "", "./tensorflow/core/kernels/data/experimental/random_dataset_op.cc", "CheckExternalState");
 return Status::OK(); }

 protected:
  Status AsGraphDefInternal(SerializationContext* ctx,
                            DatasetGraphDefBuilder* b,
                            Node** output) const override {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSrandom_dataset_opDTcc mht_7(mht_7_v, 282, "", "./tensorflow/core/kernels/data/experimental/random_dataset_op.cc", "AsGraphDefInternal");

    Node* seed = nullptr;
    Node* seed2 = nullptr;
    TF_RETURN_IF_ERROR(b->AddScalar(seeds_.first, &seed));
    TF_RETURN_IF_ERROR(b->AddScalar(seeds_.second, &seed2));
    TF_RETURN_IF_ERROR(b->AddDataset(this, {seed, seed2}, output));
    return Status::OK();
  }

 private:
  class Iterator : public DatasetIterator<Dataset> {
   public:
    explicit Iterator(const Params& params)
        : DatasetIterator<Dataset>(params),
          seeds_(MaybeOverrideSeeds(dataset()->seeds_)),
          parent_generator_(seeds_.first, seeds_.second),
          generator_(&parent_generator_) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSrandom_dataset_opDTcc mht_8(mht_8_v, 301, "", "./tensorflow/core/kernels/data/experimental/random_dataset_op.cc", "Iterator");
}

    Status GetNextInternal(IteratorContext* ctx,
                           std::vector<Tensor>* out_tensors,
                           bool* end_of_sequence) override {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSrandom_dataset_opDTcc mht_9(mht_9_v, 308, "", "./tensorflow/core/kernels/data/experimental/random_dataset_op.cc", "GetNextInternal");

      out_tensors->reserve(1);
      mutex_lock l(mu_);
      out_tensors->emplace_back(ctx->allocator({}), DT_INT64, TensorShape({}));
      out_tensors->back().scalar<int64_t>()() = Random();
      *end_of_sequence = false;
      return Status::OK();
    }

   protected:
    std::shared_ptr<model::Node> CreateNode(
        IteratorContext* ctx, model::Node::Args args) const override {
      return model::MakeSourceNode(std::move(args));
    }

    Status SaveInternal(SerializationContext* ctx,
                        IteratorStateWriter* writer) override {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSrandom_dataset_opDTcc mht_10(mht_10_v, 327, "", "./tensorflow/core/kernels/data/experimental/random_dataset_op.cc", "SaveInternal");

      mutex_lock l(mu_);
      TF_RETURN_IF_ERROR(writer->WriteScalar(full_name("num_random_samples"),
                                             num_random_samples_));
      return Status::OK();
    }

    Status RestoreInternal(IteratorContext* ctx,
                           IteratorStateReader* reader) override {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSrandom_dataset_opDTcc mht_11(mht_11_v, 338, "", "./tensorflow/core/kernels/data/experimental/random_dataset_op.cc", "RestoreInternal");

      mutex_lock l(mu_);
      TF_RETURN_IF_ERROR(reader->ReadScalar(full_name("num_random_samples"),
                                            &num_random_samples_));
      parent_generator_ = random::PhiloxRandom(seeds_.first, seeds_.second);
      generator_ =
          random::SingleSampleAdapter<random::PhiloxRandom>(&parent_generator_);
      generator_.Skip(num_random_samples_);
      return Status::OK();
    }

   private:
    random::SingleSampleAdapter<random::PhiloxRandom>::ResultType Random()
        TF_EXCLUSIVE_LOCKS_REQUIRED(mu_) {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSrandom_dataset_opDTcc mht_12(mht_12_v, 354, "", "./tensorflow/core/kernels/data/experimental/random_dataset_op.cc", "Random");

      num_random_samples_++;
      auto out = generator_();
      return out;
    }
    const std::pair<int64_t, int64_t> seeds_;
    mutex mu_;
    random::PhiloxRandom parent_generator_ TF_GUARDED_BY(mu_);
    random::SingleSampleAdapter<random::PhiloxRandom> generator_
        TF_GUARDED_BY(mu_);
    int64_t num_random_samples_ TF_GUARDED_BY(mu_) = 0;
  };

  const std::pair<int64_t, int64_t> seeds_;
};  // RandomDatasetOp::Dataset

RandomDatasetOp::RandomDatasetOp(OpKernelConstruction* ctx)
    : DatasetOpKernel(ctx) {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSrandom_dataset_opDTcc mht_13(mht_13_v, 374, "", "./tensorflow/core/kernels/data/experimental/random_dataset_op.cc", "RandomDatasetOp::RandomDatasetOp");
}

void RandomDatasetOp::MakeDataset(OpKernelContext* ctx, DatasetBase** output) {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSrandom_dataset_opDTcc mht_14(mht_14_v, 379, "", "./tensorflow/core/kernels/data/experimental/random_dataset_op.cc", "RandomDatasetOp::MakeDataset");

  int64_t seed;
  OP_REQUIRES_OK(ctx, ParseScalarArgument<int64_t>(ctx, "seed", &seed));

  int64_t seed2;
  OP_REQUIRES_OK(ctx, ParseScalarArgument<int64_t>(ctx, "seed2", &seed2));

  *output = new Dataset(ctx, seed, seed2);
}
namespace {

REGISTER_KERNEL_BUILDER(Name("RandomDataset").Device(DEVICE_CPU),
                        RandomDatasetOp);
REGISTER_KERNEL_BUILDER(Name("ExperimentalRandomDataset").Device(DEVICE_CPU),
                        RandomDatasetOp);

}  // namespace
}  // namespace experimental
}  // namespace data
}  // namespace tensorflow
