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

#ifndef TENSORFLOW_CORE_KERNELS_SDCA_INTERNAL_H_
#define TENSORFLOW_CORE_KERNELS_SDCA_INTERNAL_H_
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
class MHTracer_DTPStensorflowPScorePSkernelsPSsdca_internalDTh {
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
   MHTracer_DTPStensorflowPScorePSkernelsPSsdca_internalDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSkernelsPSsdca_internalDTh() {
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


#define EIGEN_USE_THREADS

#include <stddef.h>
#include <algorithm>
#include <cmath>
#include <memory>
#include <new>
#include <unordered_map>
#include <utility>
#include <vector>

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/device_base.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/kernels/loss.h"
#include "tensorflow/core/lib/core/coding.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/core/stringpiece.h"
#include "tensorflow/core/lib/gtl/inlined_vector.h"
#include "tensorflow/core/lib/random/distribution_sampler.h"
#include "tensorflow/core/lib/strings/stringprintf.h"
#include "tensorflow/core/util/guarded_philox_random.h"
#include "tensorflow/core/util/work_sharder.h"

namespace tensorflow {

namespace sdca {

// Statistics computed with input (ModelWeights, Example).
struct ExampleStatistics {
  // Logits for each class.
  // For binary case, this should be a vector of length 1; while for multiclass
  // case, this vector has the same length as the number of classes, where each
  // value corresponds to one class.
  // Use InlinedVector to avoid heap allocation for small number of classes.
  gtl::InlinedVector<double, 1> wx;

  // Logits for each class, using the previous weights.
  gtl::InlinedVector<double, 1> prev_wx;

  // Sum of squared feature values occurring in the example divided by
  // L2 * sum(example_weights).
  double normalized_squared_norm = 0;

  // Num_weight_vectors equals to the number of classification classes in the
  // multiclass case; while for binary case, it is 1.
  ExampleStatistics(const int num_weight_vectors)
      : wx(num_weight_vectors, 0.0), prev_wx(num_weight_vectors, 0.0) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSsdca_internalDTh mht_0(mht_0_v, 240, "", "./tensorflow/core/kernels/sdca_internal.h", "ExampleStatistics");
}
};

class Regularizations {
 public:
  Regularizations() {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSsdca_internalDTh mht_1(mht_1_v, 248, "", "./tensorflow/core/kernels/sdca_internal.h", "Regularizations");
}

  // Initialize() must be called immediately after construction.
  Status Initialize(OpKernelConstruction* const context) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSsdca_internalDTh mht_2(mht_2_v, 254, "", "./tensorflow/core/kernels/sdca_internal.h", "Initialize");

    TF_RETURN_IF_ERROR(context->GetAttr("l1", &symmetric_l1_));
    TF_RETURN_IF_ERROR(context->GetAttr("l2", &symmetric_l2_));
    shrinkage_ = symmetric_l1_ / symmetric_l2_;
    return Status::OK();
  }

  // Proximal SDCA shrinking for L1 regularization.
  double Shrink(const double weight) const {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSsdca_internalDTh mht_3(mht_3_v, 265, "", "./tensorflow/core/kernels/sdca_internal.h", "Shrink");

    const double shrinked = std::max(std::abs(weight) - shrinkage_, 0.0);
    if (shrinked > 0.0) {
      return std::copysign(shrinked, weight);
    }
    return 0.0;
  }

  // Vectorized float variant of the above.
  Eigen::Tensor<float, 1, Eigen::RowMajor> EigenShrinkVector(
      const Eigen::Tensor<float, 1, Eigen::RowMajor> weights) const {
    // Proximal step on the weights which is sign(w)*|w - shrinkage|+.
    return weights.sign() * ((weights.abs() - weights.constant(shrinkage_))
                                 .cwiseMax(weights.constant(0.0)));
  }

  // Matrix float variant of the above.
  Eigen::Tensor<float, 2, Eigen::RowMajor> EigenShrinkMatrix(
      const Eigen::Tensor<float, 2, Eigen::RowMajor> weights) const {
    // Proximal step on the weights which is sign(w)*|w - shrinkage|+.
    return weights.sign() * ((weights.abs() - weights.constant(shrinkage_))
                                 .cwiseMax(weights.constant(0.0)));
  }

  float symmetric_l2() const {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSsdca_internalDTh mht_4(mht_4_v, 292, "", "./tensorflow/core/kernels/sdca_internal.h", "symmetric_l2");
 return symmetric_l2_; }

 private:
  float symmetric_l1_ = 0;
  float symmetric_l2_ = 0;

  // L1 divided by L2, pre-computed for use during weight shrinking.
  double shrinkage_ = 0;

  TF_DISALLOW_COPY_AND_ASSIGN(Regularizations);
};

class ModelWeights;

// Struct describing a single example.
class Example {
 public:
  // Compute matrix vector product between weights (a matrix) and features
  // (a vector). This method also computes the normalized example norm used
  // in SDCA update.
  // For multiclass case, num_weight_vectors equals to the number of classes;
  // while for binary case, it is 1.
  const ExampleStatistics ComputeWxAndWeightedExampleNorm(
      const int num_loss_partitions, const ModelWeights& model_weights,
      const Regularizations& regularization,
      const int num_weight_vectors) const;

  float example_label() const {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSsdca_internalDTh mht_5(mht_5_v, 322, "", "./tensorflow/core/kernels/sdca_internal.h", "example_label");
 return example_label_; }

  float example_weight() const {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSsdca_internalDTh mht_6(mht_6_v, 327, "", "./tensorflow/core/kernels/sdca_internal.h", "example_weight");
 return example_weight_; }

  double squared_norm() const {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSsdca_internalDTh mht_7(mht_7_v, 332, "", "./tensorflow/core/kernels/sdca_internal.h", "squared_norm");
 return squared_norm_; }

  // Sparse features associated with the example.
  // Indices and Values are the associated feature index, and values. Values
  // can be optionally absent, in which we case we implicitly assume a value of
  // 1.0f.
  struct SparseFeatures {
    std::unique_ptr<TTypes<const int64_t>::UnalignedConstVec> indices;
    std::unique_ptr<TTypes<const float>::UnalignedConstVec>
        values;  // nullptr encodes optional.
  };

  // A dense vector which is a row-slice of the underlying matrix.
  struct DenseVector {
    // Returns a row slice from the matrix.
    Eigen::TensorMap<Eigen::Tensor<const float, 1, Eigen::RowMajor>> Row()
        const {
      return Eigen::TensorMap<Eigen::Tensor<const float, 1, Eigen::RowMajor>>(
          data_matrix.data() + row_index * data_matrix.dimension(1),
          data_matrix.dimension(1));
    }

    // Returns a row slice as a 1 * F matrix, where F is the number of features.
    Eigen::TensorMap<Eigen::Tensor<const float, 2, Eigen::RowMajor>>
    RowAsMatrix() const {
      return Eigen::TensorMap<Eigen::Tensor<const float, 2, Eigen::RowMajor>>(
          data_matrix.data() + row_index * data_matrix.dimension(1), 1,
          data_matrix.dimension(1));
    }

    const TTypes<float>::ConstMatrix data_matrix;
    const int64_t row_index;
  };

 private:
  std::vector<SparseFeatures> sparse_features_;
  std::vector<std::unique_ptr<DenseVector>> dense_vectors_;

  float example_label_ = 0;
  float example_weight_ = 0;
  double squared_norm_ = 0;  // sum squared norm of the features.

  // Examples fills Example in a multi-threaded way.
  friend class Examples;

  // ModelWeights use each example for model update w += \alpha * x_{i};
  friend class ModelWeights;
};

// Weights related to features. For example, say you have two sets of sparse
// features i.e. age bracket and country, then FeatureWeightsDenseStorage hold
// the parameters for it. We keep track of the original weight passed in and the
// delta weight which the optimizer learns in each call to the optimizer.
class FeatureWeightsDenseStorage {
 public:
  FeatureWeightsDenseStorage(const TTypes<const float>::Matrix nominals,
                             TTypes<float>::Matrix deltas)
      : nominals_(nominals), deltas_(deltas) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSsdca_internalDTh mht_8(mht_8_v, 392, "", "./tensorflow/core/kernels/sdca_internal.h", "FeatureWeightsDenseStorage");

    CHECK_GT(deltas.rank(), 1);
  }

  // Check if a feature index is with-in the bounds.
  bool IndexValid(const int64_t index) const {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSsdca_internalDTh mht_9(mht_9_v, 400, "", "./tensorflow/core/kernels/sdca_internal.h", "IndexValid");

    return index >= 0 && index < deltas_.dimension(1);
  }

  // Nominals here are the original weight matrix.
  TTypes<const float>::Matrix nominals() const {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSsdca_internalDTh mht_10(mht_10_v, 408, "", "./tensorflow/core/kernels/sdca_internal.h", "nominals");
 return nominals_; }

  // Delta weights during mini-batch updates.
  TTypes<float>::Matrix deltas() const {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSsdca_internalDTh mht_11(mht_11_v, 414, "", "./tensorflow/core/kernels/sdca_internal.h", "deltas");
 return deltas_; }

  // Updates delta weights based on active dense features in the example and
  // the corresponding dual residual.
  void UpdateDenseDeltaWeights(
      const Eigen::ThreadPoolDevice& device,
      const Example::DenseVector& dense_vector,
      const std::vector<double>& normalized_bounded_dual_delta);

 private:
  // The nominal value of the weight for a feature (indexed by its id).
  const TTypes<const float>::Matrix nominals_;
  // The accumulated delta weight for a feature (indexed by its id).
  TTypes<float>::Matrix deltas_;
};

// Similar to FeatureWeightsDenseStorage, but the underlying weights are stored
// in an unordered map.
class FeatureWeightsSparseStorage {
 public:
  FeatureWeightsSparseStorage(const TTypes<const int64_t>::Vec indices,
                              const TTypes<const float>::Matrix nominals,
                              TTypes<float>::Matrix deltas)
      : nominals_(nominals), deltas_(deltas) {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSsdca_internalDTh mht_12(mht_12_v, 440, "", "./tensorflow/core/kernels/sdca_internal.h", "FeatureWeightsSparseStorage");

    // Create a map from sparse index to the dense index of the underlying
    // storage.
    for (int64_t j = 0; j < indices.size(); ++j) {
      indices_to_id_[indices(j)] = j;
    }
  }

  // Check if a feature index exists.
  bool IndexValid(const int64_t index) const {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSsdca_internalDTh mht_13(mht_13_v, 452, "", "./tensorflow/core/kernels/sdca_internal.h", "IndexValid");

    return indices_to_id_.find(index) != indices_to_id_.end();
  }

  // Nominal value at a particular feature index and class label.
  float nominals(const int class_id, const int64_t index) const {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSsdca_internalDTh mht_14(mht_14_v, 460, "", "./tensorflow/core/kernels/sdca_internal.h", "nominals");

    auto it = indices_to_id_.find(index);
    return nominals_(class_id, it->second);
  }

  // Delta weights during mini-batch updates.
  float deltas(const int class_id, const int64_t index) const {
   std::vector<std::string> mht_15_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSsdca_internalDTh mht_15(mht_15_v, 469, "", "./tensorflow/core/kernels/sdca_internal.h", "deltas");

    auto it = indices_to_id_.find(index);
    return deltas_(class_id, it->second);
  }

  // Updates delta weights based on active sparse features in the example and
  // the corresponding dual residual.
  void UpdateSparseDeltaWeights(
      const Eigen::ThreadPoolDevice& device,
      const Example::SparseFeatures& sparse_features,
      const std::vector<double>& normalized_bounded_dual_delta);

 private:
  // The nominal value of the weight for a feature (indexed by its id).
  const TTypes<const float>::Matrix nominals_;
  // The accumulated delta weight for a feature (indexed by its id).
  TTypes<float>::Matrix deltas_;
  // Map from feature index to an index to the dense vector.
  std::unordered_map<int64_t, int64_t> indices_to_id_;
};

// Weights in the model, wraps both current weights, and the delta weights
// for both sparse and dense features.
class ModelWeights {
 public:
  ModelWeights() {
   std::vector<std::string> mht_16_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSsdca_internalDTh mht_16(mht_16_v, 497, "", "./tensorflow/core/kernels/sdca_internal.h", "ModelWeights");
}

  bool SparseIndexValid(const int col, const int64_t index) const {
   std::vector<std::string> mht_17_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSsdca_internalDTh mht_17(mht_17_v, 502, "", "./tensorflow/core/kernels/sdca_internal.h", "SparseIndexValid");

    return sparse_weights_[col].IndexValid(index);
  }

  bool DenseIndexValid(const int col, const int64_t index) const {
   std::vector<std::string> mht_18_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSsdca_internalDTh mht_18(mht_18_v, 509, "", "./tensorflow/core/kernels/sdca_internal.h", "DenseIndexValid");

    return dense_weights_[col].IndexValid(index);
  }

  // Go through all the features present in the example, and update the
  // weights based on the dual delta.
  void UpdateDeltaWeights(
      const Eigen::ThreadPoolDevice& device, const Example& example,
      const std::vector<double>& normalized_bounded_dual_delta);

  Status Initialize(OpKernelContext* const context);

  const std::vector<FeatureWeightsSparseStorage>& sparse_weights() const {
   std::vector<std::string> mht_19_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSsdca_internalDTh mht_19(mht_19_v, 524, "", "./tensorflow/core/kernels/sdca_internal.h", "sparse_weights");

    return sparse_weights_;
  }

  const std::vector<FeatureWeightsDenseStorage>& dense_weights() const {
   std::vector<std::string> mht_20_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSsdca_internalDTh mht_20(mht_20_v, 531, "", "./tensorflow/core/kernels/sdca_internal.h", "dense_weights");

    return dense_weights_;
  }

 private:
  std::vector<FeatureWeightsSparseStorage> sparse_weights_;
  std::vector<FeatureWeightsDenseStorage> dense_weights_;

  TF_DISALLOW_COPY_AND_ASSIGN(ModelWeights);
};

// Examples contains all the training examples that SDCA uses for a mini-batch.
class Examples {
 public:
  Examples() {
   std::vector<std::string> mht_21_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSsdca_internalDTh mht_21(mht_21_v, 548, "", "./tensorflow/core/kernels/sdca_internal.h", "Examples");
}

  // Returns the Example at |example_index|.
  const Example& example(const int example_index) const {
   std::vector<std::string> mht_22_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSsdca_internalDTh mht_22(mht_22_v, 554, "", "./tensorflow/core/kernels/sdca_internal.h", "example");

    return examples_.at(example_index);
  }

  int sampled_index(const int id) const {
   std::vector<std::string> mht_23_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSsdca_internalDTh mht_23(mht_23_v, 561, "", "./tensorflow/core/kernels/sdca_internal.h", "sampled_index");
 return sampled_index_[id]; }

  // Adaptive SDCA in the current implementation only works for
  // binary classification, where the input argument for num_weight_vectors
  // is 1.
  Status SampleAdaptiveProbabilities(
      const int num_loss_partitions, const Regularizations& regularization,
      const ModelWeights& model_weights,
      const TTypes<float>::Matrix example_state_data,
      const std::unique_ptr<DualLossUpdater>& loss_updater,
      const int num_weight_vectors);

  void RandomShuffle();

  int num_examples() const {
   std::vector<std::string> mht_24_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSsdca_internalDTh mht_24(mht_24_v, 578, "", "./tensorflow/core/kernels/sdca_internal.h", "num_examples");
 return examples_.size(); }

  int num_features() const {
   std::vector<std::string> mht_25_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSsdca_internalDTh mht_25(mht_25_v, 583, "", "./tensorflow/core/kernels/sdca_internal.h", "num_features");
 return num_features_; }

  // Initialize() must be called immediately after construction.
  Status Initialize(OpKernelContext* const context, const ModelWeights& weights,
                    int num_sparse_features,
                    int num_sparse_features_with_values,
                    int num_dense_features);

 private:
  // Reads the input tensors, and builds the internal representation for sparse
  // features per example. This function modifies the |examples| passed in
  // to build the sparse representations.
  static Status CreateSparseFeatureRepresentation(
      const DeviceBase::CpuWorkerThreads& worker_threads, int num_examples,
      int num_sparse_features, const ModelWeights& weights,
      const OpInputList& sparse_example_indices_inputs,
      const OpInputList& sparse_feature_indices_inputs,
      const OpInputList& sparse_feature_values_inputs,
      std::vector<Example>* const examples);

  // Reads the input tensors, and builds the internal representation for dense
  // features per example. This function modifies the |examples| passed in
  // to build the sparse representations.
  static Status CreateDenseFeatureRepresentation(
      const DeviceBase::CpuWorkerThreads& worker_threads, int num_examples,
      int num_dense_features, const ModelWeights& weights,
      const OpInputList& dense_features_inputs,
      std::vector<Example>* const examples);

  // Computes squared example norm per example i.e |x|^2. This function modifies
  // the |examples| passed in and adds the squared norm per example.
  static Status ComputeSquaredNormPerExample(
      const DeviceBase::CpuWorkerThreads& worker_threads, int num_examples,
      int num_sparse_features, int num_dense_features,
      std::vector<Example>* const examples);

  // All examples in the batch.
  std::vector<Example> examples_;

  // Adaptive sampling variables.
  std::vector<float> probabilities_;
  std::vector<int> sampled_index_;
  std::vector<int> sampled_count_;

  int num_features_ = 0;

  TF_DISALLOW_COPY_AND_ASSIGN(Examples);
};

}  // namespace sdca
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_KERNELS_SDCA_INTERNAL_H_
