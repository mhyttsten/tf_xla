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
class MHTracer_DTPStensorflowPScorePSkernelsPSsdca_internalDTcc {
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
   MHTracer_DTPStensorflowPScorePSkernelsPSsdca_internalDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSkernelsPSsdca_internalDTcc() {
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

/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

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

#define EIGEN_USE_THREADS

#include "tensorflow/core/kernels/sdca_internal.h"

#include <limits>
#include <numeric>
#include <random>

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/lib/gtl/flatset.h"
#include "tensorflow/core/lib/math/math_util.h"
#include "tensorflow/core/lib/random/simple_philox.h"

#if defined(TENSORFLOW_USE_CUSTOM_CONTRACTION_KERNEL)
#include "tensorflow/core/kernels/eigen_contraction_kernel.h"
#endif

namespace tensorflow {
namespace sdca {

using UnalignedFloatVector = TTypes<const float>::UnalignedConstVec;
using UnalignedInt64Vector = TTypes<const int64_t>::UnalignedConstVec;

void FeatureWeightsDenseStorage::UpdateDenseDeltaWeights(
    const Eigen::ThreadPoolDevice& device,
    const Example::DenseVector& dense_vector,
    const std::vector<double>& normalized_bounded_dual_delta) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSsdca_internalDTcc mht_0(mht_0_v, 211, "", "./tensorflow/core/kernels/sdca_internal.cc", "FeatureWeightsDenseStorage::UpdateDenseDeltaWeights");

  const size_t num_weight_vectors = normalized_bounded_dual_delta.size();
  if (num_weight_vectors == 1) {
    deltas_.device(device) =
        deltas_ + dense_vector.RowAsMatrix() *
                      deltas_.constant(normalized_bounded_dual_delta[0]);
  } else {
    // Transform the dual vector into a column matrix.
    const Eigen::TensorMap<Eigen::Tensor<const double, 2, Eigen::RowMajor>>
        dual_matrix(normalized_bounded_dual_delta.data(), num_weight_vectors,
                    1);
    const Eigen::array<Eigen::IndexPair<int>, 1> product_dims = {
        Eigen::IndexPair<int>(1, 0)};
    // This computes delta_w += delta_vector / \lamdba * N.
    deltas_.device(device) =
        (deltas_.cast<double>() +
         dual_matrix.contract(dense_vector.RowAsMatrix().cast<double>(),
                              product_dims))
            .cast<float>();
  }
}

void FeatureWeightsSparseStorage::UpdateSparseDeltaWeights(
    const Eigen::ThreadPoolDevice& device,
    const Example::SparseFeatures& sparse_features,
    const std::vector<double>& normalized_bounded_dual_delta) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSsdca_internalDTcc mht_1(mht_1_v, 239, "", "./tensorflow/core/kernels/sdca_internal.cc", "FeatureWeightsSparseStorage::UpdateSparseDeltaWeights");

  for (int64_t k = 0; k < sparse_features.indices->size(); ++k) {
    const double feature_value =
        sparse_features.values == nullptr ? 1.0 : (*sparse_features.values)(k);
    auto it = indices_to_id_.find((*sparse_features.indices)(k));
    for (size_t l = 0; l < normalized_bounded_dual_delta.size(); ++l) {
      deltas_(l, it->second) +=
          feature_value * normalized_bounded_dual_delta[l];
    }
  }
}

void ModelWeights::UpdateDeltaWeights(
    const Eigen::ThreadPoolDevice& device, const Example& example,
    const std::vector<double>& normalized_bounded_dual_delta) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSsdca_internalDTcc mht_2(mht_2_v, 256, "", "./tensorflow/core/kernels/sdca_internal.cc", "ModelWeights::UpdateDeltaWeights");

  // Sparse weights.
  for (size_t j = 0; j < sparse_weights_.size(); ++j) {
    sparse_weights_[j].UpdateSparseDeltaWeights(
        device, example.sparse_features_[j], normalized_bounded_dual_delta);
  }

  // Dense weights.
  for (size_t j = 0; j < dense_weights_.size(); ++j) {
    dense_weights_[j].UpdateDenseDeltaWeights(
        device, *example.dense_vectors_[j], normalized_bounded_dual_delta);
  }
}

Status ModelWeights::Initialize(OpKernelContext* const context) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSsdca_internalDTcc mht_3(mht_3_v, 273, "", "./tensorflow/core/kernels/sdca_internal.cc", "ModelWeights::Initialize");

  OpInputList sparse_indices_inputs;
  TF_RETURN_IF_ERROR(
      context->input_list("sparse_indices", &sparse_indices_inputs));
  OpInputList sparse_weights_inputs;
  TF_RETURN_IF_ERROR(
      context->input_list("sparse_weights", &sparse_weights_inputs));
  if (sparse_indices_inputs.size() != sparse_weights_inputs.size())
    return errors::InvalidArgument(
        "sparse_indices and sparse_weights must have the same length, got ",
        sparse_indices_inputs.size(), " and ", sparse_weights_inputs.size());
  OpInputList dense_weights_inputs;
  TF_RETURN_IF_ERROR(
      context->input_list("dense_weights", &dense_weights_inputs));

  OpOutputList sparse_weights_outputs;
  TF_RETURN_IF_ERROR(context->output_list("out_delta_sparse_weights",
                                          &sparse_weights_outputs));
  if (sparse_weights_outputs.size() != sparse_weights_inputs.size())
    return errors::InvalidArgument(
        "out_delta_sparse_weights and sparse_weights must have the same "
        "length, got ",
        sparse_weights_outputs.size(), " and ", sparse_weights_inputs.size());

  OpOutputList dense_weights_outputs;
  TF_RETURN_IF_ERROR(
      context->output_list("out_delta_dense_weights", &dense_weights_outputs));
  if (dense_weights_outputs.size() != dense_weights_inputs.size())
    return errors::InvalidArgument(
        "out_delta_dense_weights and dense_weights must have the same length, "
        "got ",
        dense_weights_outputs.size(), " and ", dense_weights_inputs.size());

  for (int i = 0; i < sparse_weights_inputs.size(); ++i) {
    Tensor* delta_t;
    TF_RETURN_IF_ERROR(sparse_weights_outputs.allocate(
        i, sparse_weights_inputs[i].shape(), &delta_t));
    // Convert the input vector to a row matrix in internal representation.
    auto deltas = delta_t->shaped<float, 2>({1, delta_t->NumElements()});
    deltas.setZero();
    sparse_weights_.emplace_back(FeatureWeightsSparseStorage{
        sparse_indices_inputs[i].flat<int64_t>(),
        sparse_weights_inputs[i].shaped<float, 2>(
            {1, sparse_weights_inputs[i].NumElements()}),
        deltas});
  }

  // Reads in the weights, and allocates and initializes the delta weights.
  const auto initialize_weights =
      [&](const OpInputList& weight_inputs, OpOutputList* const weight_outputs,
          std::vector<FeatureWeightsDenseStorage>* const feature_weights) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSsdca_internalDTcc mht_4(mht_4_v, 326, "", "./tensorflow/core/kernels/sdca_internal.cc", "lambda");

        for (int i = 0; i < weight_inputs.size(); ++i) {
          Tensor* delta_t;
          TF_RETURN_IF_ERROR(
              weight_outputs->allocate(i, weight_inputs[i].shape(), &delta_t));
          // Convert the input vector to a row matrix in internal
          // representation.
          auto deltas = delta_t->shaped<float, 2>({1, delta_t->NumElements()});
          deltas.setZero();
          feature_weights->emplace_back(FeatureWeightsDenseStorage{
              weight_inputs[i].shaped<float, 2>(
                  {1, weight_inputs[i].NumElements()}),
              deltas});
        }
        return Status::OK();
      };

  return initialize_weights(dense_weights_inputs, &dense_weights_outputs,
                            &dense_weights_);
}

// Computes the example statistics for given example, and model. Defined here
// as we need definition of ModelWeights and Regularizations.
const ExampleStatistics Example::ComputeWxAndWeightedExampleNorm(
    const int num_loss_partitions, const ModelWeights& model_weights,
    const Regularizations& regularization, const int num_weight_vectors) const {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSsdca_internalDTcc mht_5(mht_5_v, 354, "", "./tensorflow/core/kernels/sdca_internal.cc", "Example::ComputeWxAndWeightedExampleNorm");

  ExampleStatistics result(num_weight_vectors);

  result.normalized_squared_norm =
      squared_norm_ / regularization.symmetric_l2();

  // Compute w \dot x and prev_w \dot x.
  // This is for sparse features contribution to the logit.
  for (size_t j = 0; j < sparse_features_.size(); ++j) {
    const Example::SparseFeatures& sparse_features = sparse_features_[j];
    const FeatureWeightsSparseStorage& sparse_weights =
        model_weights.sparse_weights()[j];

    for (int64_t k = 0; k < sparse_features.indices->size(); ++k) {
      const int64_t feature_index = (*sparse_features.indices)(k);
      const double feature_value = sparse_features.values == nullptr
                                       ? 1.0
                                       : (*sparse_features.values)(k);
      for (int l = 0; l < num_weight_vectors; ++l) {
        const float sparse_weight = sparse_weights.nominals(l, feature_index);
        const double feature_weight =
            sparse_weight +
            sparse_weights.deltas(l, feature_index) * num_loss_partitions;
        result.prev_wx[l] +=
            feature_value * regularization.Shrink(sparse_weight);
        result.wx[l] += feature_value * regularization.Shrink(feature_weight);
      }
    }
  }

  // Compute w \dot x and prev_w \dot x.
  // This is for dense features contribution to the logit.
  for (size_t j = 0; j < dense_vectors_.size(); ++j) {
    const Example::DenseVector& dense_vector = *dense_vectors_[j];
    const FeatureWeightsDenseStorage& dense_weights =
        model_weights.dense_weights()[j];

    const Eigen::Tensor<float, 2, Eigen::RowMajor> feature_weights =
        dense_weights.nominals() +
        dense_weights.deltas() *
            dense_weights.deltas().constant(num_loss_partitions);
    if (num_weight_vectors == 1) {
      const Eigen::Tensor<float, 0, Eigen::RowMajor> prev_prediction =
          (dense_vector.Row() *
           regularization.EigenShrinkVector(
               Eigen::TensorMap<Eigen::Tensor<const float, 1, Eigen::RowMajor>>(
                   dense_weights.nominals().data(),
                   dense_weights.nominals().dimension(1))))
              .sum();
      const Eigen::Tensor<float, 0, Eigen::RowMajor> prediction =
          (dense_vector.Row() *
           regularization.EigenShrinkVector(
               Eigen::TensorMap<Eigen::Tensor<const float, 1, Eigen::RowMajor>>(
                   feature_weights.data(), feature_weights.dimension(1))))
              .sum();
      result.prev_wx[0] += prev_prediction();
      result.wx[0] += prediction();
    } else {
      const Eigen::array<Eigen::IndexPair<int>, 1> product_dims = {
          Eigen::IndexPair<int>(1, 1)};
      const Eigen::Tensor<float, 2, Eigen::RowMajor> prev_prediction =
          regularization.EigenShrinkMatrix(dense_weights.nominals())
              .contract(dense_vector.RowAsMatrix(), product_dims);
      const Eigen::Tensor<float, 2, Eigen::RowMajor> prediction =
          regularization.EigenShrinkMatrix(feature_weights)
              .contract(dense_vector.RowAsMatrix(), product_dims);
      // The result of "tensor contraction" (multiplication)  in the code
      // above is of dimension num_weight_vectors * 1.
      for (int l = 0; l < num_weight_vectors; ++l) {
        result.prev_wx[l] += prev_prediction(l, 0);
        result.wx[l] += prediction(l, 0);
      }
    }
  }

  return result;
}

// Examples contains all the training examples that SDCA uses for a mini-batch.
Status Examples::SampleAdaptiveProbabilities(
    const int num_loss_partitions, const Regularizations& regularization,
    const ModelWeights& model_weights,
    const TTypes<float>::Matrix example_state_data,
    const std::unique_ptr<DualLossUpdater>& loss_updater,
    const int num_weight_vectors) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSsdca_internalDTcc mht_6(mht_6_v, 441, "", "./tensorflow/core/kernels/sdca_internal.cc", "Examples::SampleAdaptiveProbabilities");

  if (num_weight_vectors != 1) {
    return errors::InvalidArgument(
        "Adaptive SDCA only works with binary SDCA, "
        "where num_weight_vectors should be 1.");
  }
  // Compute the probabilities
  for (int example_id = 0; example_id < num_examples(); ++example_id) {
    const Example& example = examples_[example_id];
    const double example_weight = example.example_weight();
    float label = example.example_label();
    const Status conversion_status = loss_updater->ConvertLabel(&label);
    const ExampleStatistics example_statistics =
        example.ComputeWxAndWeightedExampleNorm(num_loss_partitions,
                                                model_weights, regularization,
                                                num_weight_vectors);
    const double kappa = example_state_data(example_id, 0) +
                         loss_updater->PrimalLossDerivative(
                             example_statistics.wx[0], label, 1.0);
    probabilities_[example_id] = example_weight *
                                 sqrt(examples_[example_id].squared_norm_ +
                                      regularization.symmetric_l2() *
                                          loss_updater->SmoothnessConstant()) *
                                 std::abs(kappa);
  }

  // Sample the index
  random::DistributionSampler sampler(probabilities_);
  GuardedPhiloxRandom generator;
  generator.Init(0, 0);
  auto local_gen = generator.ReserveSamples32(num_examples());
  random::SimplePhilox random(&local_gen);
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<> dis(0, 1);

  // We use a decay of 10: the probability of an example is divided by 10
  // once that example is picked. A good approximation of that is to only
  // keep a picked example with probability (1 / 10) ^ k where k is the
  // number of times we already picked that example. We add a num_retries
  // to avoid taking too long to sample. We then fill the sampled_index with
  // unseen examples sorted by probabilities.
  int id = 0;
  int num_retries = 0;
  while (id < num_examples() && num_retries < num_examples()) {
    int picked_id = sampler.Sample(&random);
    if (dis(gen) > MathUtil::IPow(0.1, sampled_count_[picked_id])) {
      num_retries++;
      continue;
    }
    sampled_count_[picked_id]++;
    sampled_index_[id++] = picked_id;
  }

  std::vector<std::pair<int, float>> examples_not_seen;
  examples_not_seen.reserve(num_examples());
  for (int i = 0; i < num_examples(); ++i) {
    if (sampled_count_[i] == 0)
      examples_not_seen.emplace_back(sampled_index_[i], probabilities_[i]);
  }
  std::sort(
      examples_not_seen.begin(), examples_not_seen.end(),
      [](const std::pair<int, float>& lhs, const std::pair<int, float>& rhs) {
        return lhs.second > rhs.second;
      });
  for (int i = id; i < num_examples(); ++i) {
    sampled_count_[i] = examples_not_seen[i - id].first;
  }
  return Status::OK();
}

void Examples::RandomShuffle() {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSsdca_internalDTcc mht_7(mht_7_v, 515, "", "./tensorflow/core/kernels/sdca_internal.cc", "Examples::RandomShuffle");

  std::iota(sampled_index_.begin(), sampled_index_.end(), 0);

  std::random_device rd;
  std::mt19937 rng(rd());
  std::shuffle(sampled_index_.begin(), sampled_index_.end(), rng);
}

// TODO(sibyl-Aix6ihai): Refactor/shorten this function.
Status Examples::Initialize(OpKernelContext* const context,
                            const ModelWeights& weights,
                            const int num_sparse_features,
                            const int num_sparse_features_with_values,
                            const int num_dense_features) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSsdca_internalDTcc mht_8(mht_8_v, 531, "", "./tensorflow/core/kernels/sdca_internal.cc", "Examples::Initialize");

  num_features_ = num_sparse_features + num_dense_features;

  OpInputList sparse_example_indices_inputs;
  TF_RETURN_IF_ERROR(context->input_list("sparse_example_indices",
                                         &sparse_example_indices_inputs));
  if (sparse_example_indices_inputs.size() != num_sparse_features)
    return errors::InvalidArgument(
        "Expected ", num_sparse_features,
        " tensors in sparse_example_indices but got ",
        sparse_example_indices_inputs.size());
  OpInputList sparse_feature_indices_inputs;
  TF_RETURN_IF_ERROR(context->input_list("sparse_feature_indices",
                                         &sparse_feature_indices_inputs));
  if (sparse_feature_indices_inputs.size() != num_sparse_features)
    return errors::InvalidArgument(
        "Expected ", num_sparse_features,
        " tensors in sparse_feature_indices but got ",
        sparse_feature_indices_inputs.size());
  OpInputList sparse_feature_values_inputs;
  if (num_sparse_features_with_values > 0) {
    TF_RETURN_IF_ERROR(context->input_list("sparse_feature_values",
                                           &sparse_feature_values_inputs));
    if (sparse_feature_values_inputs.size() != num_sparse_features_with_values)
      return errors::InvalidArgument(
          "Expected ", num_sparse_features_with_values,
          " tensors in sparse_feature_values but got ",
          sparse_feature_values_inputs.size());
  }

  const Tensor* example_weights_t;
  TF_RETURN_IF_ERROR(context->input("example_weights", &example_weights_t));
  auto example_weights = example_weights_t->flat<float>();

  if (example_weights.size() >= std::numeric_limits<int>::max()) {
    return errors::InvalidArgument(strings::Printf(
        "Too many examples in a mini-batch: %zu > %d", example_weights.size(),
        std::numeric_limits<int>::max()));
  }

  // The static_cast here is safe since num_examples can be at max an int.
  const int num_examples = static_cast<int>(example_weights.size());
  const Tensor* example_labels_t;
  TF_RETURN_IF_ERROR(context->input("example_labels", &example_labels_t));
  auto example_labels = example_labels_t->flat<float>();
  if (example_labels.size() != num_examples) {
    return errors::InvalidArgument("Expected ", num_examples,
                                   " example labels but got ",
                                   example_labels.size());
  }

  OpInputList dense_features_inputs;
  TF_RETURN_IF_ERROR(
      context->input_list("dense_features", &dense_features_inputs));

  examples_.clear();
  examples_.resize(num_examples);
  probabilities_.resize(num_examples);
  sampled_index_.resize(num_examples);
  sampled_count_.resize(num_examples);
  for (int example_id = 0; example_id < num_examples; ++example_id) {
    Example* const example = &examples_[example_id];
    example->sparse_features_.resize(num_sparse_features);
    example->dense_vectors_.resize(num_dense_features);
    example->example_weight_ = example_weights(example_id);
    example->example_label_ = example_labels(example_id);
  }
  const DeviceBase::CpuWorkerThreads& worker_threads =
      *context->device()->tensorflow_cpu_worker_threads();
  TF_RETURN_IF_ERROR(CreateSparseFeatureRepresentation(
      worker_threads, num_examples, num_sparse_features, weights,
      sparse_example_indices_inputs, sparse_feature_indices_inputs,
      sparse_feature_values_inputs, &examples_));
  TF_RETURN_IF_ERROR(CreateDenseFeatureRepresentation(
      worker_threads, num_examples, num_dense_features, weights,
      dense_features_inputs, &examples_));
  TF_RETURN_IF_ERROR(ComputeSquaredNormPerExample(
      worker_threads, num_examples, num_sparse_features, num_dense_features,
      &examples_));
  return Status::OK();
}

Status Examples::CreateSparseFeatureRepresentation(
    const DeviceBase::CpuWorkerThreads& worker_threads, const int num_examples,
    const int num_sparse_features, const ModelWeights& weights,
    const OpInputList& sparse_example_indices_inputs,
    const OpInputList& sparse_feature_indices_inputs,
    const OpInputList& sparse_feature_values_inputs,
    std::vector<Example>* const examples) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSsdca_internalDTcc mht_9(mht_9_v, 622, "", "./tensorflow/core/kernels/sdca_internal.cc", "Examples::CreateSparseFeatureRepresentation");

  mutex mu;
  Status result;  // Guarded by mu
  auto parse_partition = [&](const int64_t begin, const int64_t end) {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSsdca_internalDTcc mht_10(mht_10_v, 628, "", "./tensorflow/core/kernels/sdca_internal.cc", "lambda");

    // The static_cast here is safe since begin and end can be at most
    // num_examples which is an int.
    for (int i = static_cast<int>(begin); i < end; ++i) {
      auto example_indices =
          sparse_example_indices_inputs[i].template flat<int64_t>();
      auto feature_indices =
          sparse_feature_indices_inputs[i].template flat<int64_t>();
      if (example_indices.size() != feature_indices.size()) {
        mutex_lock l(mu);
        result = errors::InvalidArgument(
            "Found mismatched example_indices and feature_indices [",
            example_indices, "] vs [", feature_indices, "]");
        return;
      }

      // Parse features for each example. Features for a particular example
      // are at the offsets (start_id, end_id]
      int start_id = -1;
      int end_id = 0;
      for (int example_id = 0; example_id < num_examples; ++example_id) {
        start_id = end_id;
        while (end_id < example_indices.size() &&
               example_indices(end_id) == example_id) {
          ++end_id;
        }
        Example::SparseFeatures* const sparse_features =
            &(*examples)[example_id].sparse_features_[i];
        if (start_id < example_indices.size() &&
            example_indices(start_id) == example_id) {
          sparse_features->indices.reset(new UnalignedInt64Vector(
              &(feature_indices(start_id)), end_id - start_id));
          if (sparse_feature_values_inputs.size() > i) {
            auto feature_weights =
                sparse_feature_values_inputs[i].flat<float>();
            sparse_features->values.reset(new UnalignedFloatVector(
                &(feature_weights(start_id)), end_id - start_id));
          }
          // If features are non empty.
          if (end_id - start_id > 0) {
            // TODO(sibyl-Aix6ihai): Write this efficiently using vectorized
            // operations from eigen.
            for (int64_t k = 0; k < sparse_features->indices->size(); ++k) {
              const int64_t feature_index = (*sparse_features->indices)(k);
              if (!weights.SparseIndexValid(i, feature_index)) {
                mutex_lock l(mu);
                result = errors::InvalidArgument(
                    "Found sparse feature indices out of valid range: ",
                    (*sparse_features->indices)(k));
                return;
              }
            }
          }
        } else {
          // Add a Tensor that has size 0.
          sparse_features->indices.reset(
              new UnalignedInt64Vector(&(feature_indices(0)), 0));
          // If values exist for this feature group.
          if (sparse_feature_values_inputs.size() > i) {
            auto feature_weights =
                sparse_feature_values_inputs[i].flat<float>();
            sparse_features->values.reset(
                new UnalignedFloatVector(&(feature_weights(0)), 0));
          }
        }
      }
    }
  };
  // For each column, the cost of parsing it is O(num_examples). We use
  // num_examples here, as empirically Shard() creates the right amount of
  // threads based on the problem size.
  // TODO(sibyl-Aix6ihai): Tune this as a function of dataset size.
  const int64_t kCostPerUnit = num_examples;
  Shard(worker_threads.num_threads, worker_threads.workers, num_sparse_features,
        kCostPerUnit, parse_partition);
  return result;
}

Status Examples::CreateDenseFeatureRepresentation(
    const DeviceBase::CpuWorkerThreads& worker_threads, const int num_examples,
    const int num_dense_features, const ModelWeights& weights,
    const OpInputList& dense_features_inputs,
    std::vector<Example>* const examples) {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSsdca_internalDTcc mht_11(mht_11_v, 713, "", "./tensorflow/core/kernels/sdca_internal.cc", "Examples::CreateDenseFeatureRepresentation");

  mutex mu;
  Status result;  // Guarded by mu
  auto parse_partition = [&](const int64_t begin, const int64_t end) {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSsdca_internalDTcc mht_12(mht_12_v, 719, "", "./tensorflow/core/kernels/sdca_internal.cc", "lambda");

    // The static_cast here is safe since begin and end can be at most
    // num_examples which is an int.
    for (int i = static_cast<int>(begin); i < end; ++i) {
      auto dense_features = dense_features_inputs[i].template matrix<float>();
      for (int example_id = 0; example_id < num_examples; ++example_id) {
        (*examples)[example_id].dense_vectors_[i].reset(
            new Example::DenseVector{dense_features, example_id});
      }
      if (!weights.DenseIndexValid(i, dense_features.dimension(1) - 1)) {
        mutex_lock l(mu);
        result = errors::InvalidArgument(
            "More dense features than we have parameters for: ",
            dense_features.dimension(1));
        return;
      }
    }
  };
  // TODO(sibyl-Aix6ihai): Tune this as a function of dataset size.
  const int64_t kCostPerUnit = num_examples;
  Shard(worker_threads.num_threads, worker_threads.workers, num_dense_features,
        kCostPerUnit, parse_partition);
  return result;
}

Status Examples::ComputeSquaredNormPerExample(
    const DeviceBase::CpuWorkerThreads& worker_threads, const int num_examples,
    const int num_sparse_features, const int num_dense_features,
    std::vector<Example>* const examples) {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSsdca_internalDTcc mht_13(mht_13_v, 750, "", "./tensorflow/core/kernels/sdca_internal.cc", "Examples::ComputeSquaredNormPerExample");

  mutex mu;
  Status result;  // Guarded by mu
  // Compute norm of examples.
  auto compute_example_norm = [&](const int64_t begin, const int64_t end) {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSsdca_internalDTcc mht_14(mht_14_v, 757, "", "./tensorflow/core/kernels/sdca_internal.cc", "lambda");

    // The static_cast here is safe since begin and end can be at most
    // num_examples which is an int.
    gtl::FlatSet<int64_t> previous_indices;
    for (int example_id = static_cast<int>(begin); example_id < end;
         ++example_id) {
      double squared_norm = 0;
      Example* const example = &(*examples)[example_id];
      for (int j = 0; j < num_sparse_features; ++j) {
        const Example::SparseFeatures& sparse_features =
            example->sparse_features_[j];
        previous_indices.clear();
        for (int64_t k = 0; k < sparse_features.indices->size(); ++k) {
          const int64_t feature_index = (*sparse_features.indices)(k);
          if (previous_indices.insert(feature_index).second == false) {
            mutex_lock l(mu);
            result =
                errors::InvalidArgument("Duplicate index in sparse vector.");
            return;
          }
          const double feature_value = sparse_features.values == nullptr
                                           ? 1.0
                                           : (*sparse_features.values)(k);
          squared_norm += feature_value * feature_value;
        }
      }
      for (int j = 0; j < num_dense_features; ++j) {
        const Eigen::Tensor<float, 0, Eigen::RowMajor> sn =
            example->dense_vectors_[j]->Row().square().sum();
        squared_norm += sn();
      }
      example->squared_norm_ = squared_norm;
    }
  };
  // TODO(sibyl-Aix6ihai): Compute the cost optimally.
  const int64_t kCostPerUnit = num_dense_features + num_sparse_features;
  Shard(worker_threads.num_threads, worker_threads.workers, num_examples,
        kCostPerUnit, compute_example_norm);
  return result;
}

}  // namespace sdca
}  // namespace tensorflow
