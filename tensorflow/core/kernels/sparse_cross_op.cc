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
class MHTracer_DTPStensorflowPScorePSkernelsPSsparse_cross_opDTcc {
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
   MHTracer_DTPStensorflowPScorePSkernelsPSsparse_cross_opDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSkernelsPSsparse_cross_opDTcc() {
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

// Contains OP to generate sparse crosses.
#include <assert.h>

#include <limits>
#include <string>
#include <vector>

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/kernel_def_builder.h"
#include "tensorflow/core/framework/op_def_builder.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/lib/core/stringpiece.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/platform/fingerprint.h"
#include "tensorflow/core/platform/strong_hash.h"
#include "tensorflow/core/util/work_sharder.h"

namespace tensorflow {

namespace {
// An interface that represents a column with batches.
template <typename InternalType>
class ColumnInterface {
 public:
  // Returns the number of features in the specified batch.
  virtual int64_t FeatureCount(int64_t batch) const = 0;

  // Returns the fingerprint of nth feature from the specified batch.
  virtual InternalType Feature(int64_t batch, int64_t n,
                               bool strong_hash) const = 0;

  virtual ~ColumnInterface() {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSsparse_cross_opDTcc mht_0(mht_0_v, 220, "", "./tensorflow/core/kernels/sparse_cross_op.cc", "~ColumnInterface");
}
};

// A column that is backed by a sparse tensor.
template <typename InternalType>
class SparseTensorColumn : public ColumnInterface<InternalType> {
 public:
  SparseTensorColumn(const Tensor& values, std::vector<int64_t> feature_counts,
                     std::vector<int64_t> feature_start_indices)
      : values_(values),
        feature_counts_(std::move(feature_counts)),
        feature_start_indices_(std::move(feature_start_indices)) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSsparse_cross_opDTcc mht_1(mht_1_v, 234, "", "./tensorflow/core/kernels/sparse_cross_op.cc", "SparseTensorColumn");

    CHECK_EQ(feature_counts_.size(), feature_start_indices_.size());
  }

  int64_t FeatureCount(int64_t batch) const override {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSsparse_cross_opDTcc mht_2(mht_2_v, 241, "", "./tensorflow/core/kernels/sparse_cross_op.cc", "FeatureCount");

    return feature_counts_[batch];
  }

  InternalType Feature(int64_t batch, int64_t n,
                       bool strong_hash) const override;

  ~SparseTensorColumn() override {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSsparse_cross_opDTcc mht_3(mht_3_v, 251, "", "./tensorflow/core/kernels/sparse_cross_op.cc", "~SparseTensorColumn");
}

 private:
  const Tensor& values_;
  std::vector<int64_t> feature_counts_;
  std::vector<int64_t> feature_start_indices_;
};

// A column that is backed by a sparse tensor.
template <typename InternalType>
class KeyedSparseTensorColumn : public ColumnInterface<InternalType> {
 public:
  KeyedSparseTensorColumn(const Tensor& values,
                          std::vector<int64_t> feature_counts,
                          std::vector<int64_t> feature_start_indices,
                          std::vector<int64_t> key)
      : values_(values),
        feature_counts_(std::move(feature_counts)),
        feature_start_indices_(std::move(feature_start_indices)) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSsparse_cross_opDTcc mht_4(mht_4_v, 272, "", "./tensorflow/core/kernels/sparse_cross_op.cc", "KeyedSparseTensorColumn");

    DCHECK_EQ(feature_counts_.size(), feature_start_indices_.size());
    std::memcpy(key_, key.data(), sizeof(key_));
  }

  int64_t FeatureCount(int64_t batch) const override {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSsparse_cross_opDTcc mht_5(mht_5_v, 280, "", "./tensorflow/core/kernels/sparse_cross_op.cc", "FeatureCount");

    return feature_counts_[batch];
  }

  InternalType Feature(int64_t batch, int64_t n,
                       bool strong_hash) const override;

  ~KeyedSparseTensorColumn() override {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSsparse_cross_opDTcc mht_6(mht_6_v, 290, "", "./tensorflow/core/kernels/sparse_cross_op.cc", "~KeyedSparseTensorColumn");
}

 private:
  const Tensor& values_;
  tensorflow::uint64 key_[2];
  std::vector<int64_t> feature_counts_;
  std::vector<int64_t> feature_start_indices_;
};

// InternalType is int64 only when using HashCrosser.
template <>
int64_t SparseTensorColumn<int64_t>::Feature(int64_t batch, int64_t n,
                                             bool strong_hash) const {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSsparse_cross_opDTcc mht_7(mht_7_v, 305, "", "./tensorflow/core/kernels/sparse_cross_op.cc", "SparseTensorColumn<int64_t>::Feature");

  const int64_t start = feature_start_indices_[batch];
  if (DT_STRING == values_.dtype())
    return Fingerprint64(values_.vec<tstring>().data()[start + n]);
  return values_.vec<int64_t>().data()[start + n];
}

template <>
int64_t KeyedSparseTensorColumn<int64_t>::Feature(int64_t batch, int64_t n,
                                                  bool strong_hash) const {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSsparse_cross_opDTcc mht_8(mht_8_v, 317, "", "./tensorflow/core/kernels/sparse_cross_op.cc", "KeyedSparseTensorColumn<int64_t>::Feature");

  const int64_t start = feature_start_indices_[batch];
  if (strong_hash) {
    if (DT_STRING == values_.dtype()) {
      return StrongKeyedHash(key_, values_.vec<tstring>()(start + n));
    }
    return StrongKeyedHash(
        key_,
        {reinterpret_cast<const char*>(&values_.vec<int64_t>()(start + n)),
         sizeof(values_.dtype())});
  }
  if (DT_STRING == values_.dtype())
    return Fingerprint64(values_.vec<tstring>()(start + n));
  return Fingerprint64(
      {reinterpret_cast<const char*>(&values_.vec<int64_t>()(start + n)),
       sizeof(values_.dtype())});
}

// InternalType is string or StringPiece when using StringCrosser.
template <>
tstring SparseTensorColumn<tstring>::Feature(int64_t batch, int64_t n,
                                             bool strong_hash) const {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSsparse_cross_opDTcc mht_9(mht_9_v, 341, "", "./tensorflow/core/kernels/sparse_cross_op.cc", "SparseTensorColumn<tstring>::Feature");

  const int64_t start = feature_start_indices_[batch];
  if (DT_STRING == values_.dtype())
    return values_.vec<tstring>().data()[start + n];
  return std::to_string(values_.vec<int64_t>().data()[start + n]);
}

template <>
tstring KeyedSparseTensorColumn<tstring>::Feature(int64_t batch, int64_t n,
                                                  bool strong_hash) const {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSsparse_cross_opDTcc mht_10(mht_10_v, 353, "", "./tensorflow/core/kernels/sparse_cross_op.cc", "KeyedSparseTensorColumn<tstring>::Feature");

  const int64_t start = feature_start_indices_[batch];
  if (DT_STRING == values_.dtype())
    return values_.vec<tstring>().data()[start + n];
  return std::to_string(values_.vec<int64_t>().data()[start + n]);
}

template <>
StringPiece SparseTensorColumn<StringPiece>::Feature(int64_t batch, int64_t n,
                                                     bool strong_hash) const {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSsparse_cross_opDTcc mht_11(mht_11_v, 365, "", "./tensorflow/core/kernels/sparse_cross_op.cc", "SparseTensorColumn<StringPiece>::Feature");

  const int64_t start = feature_start_indices_[batch];
  return values_.vec<tstring>().data()[start + n];
}

template <>
StringPiece KeyedSparseTensorColumn<StringPiece>::Feature(
    int64_t batch, int64_t n, bool strong_hash) const {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSsparse_cross_opDTcc mht_12(mht_12_v, 375, "", "./tensorflow/core/kernels/sparse_cross_op.cc", "KeyedSparseTensorColumn<StringPiece>::Feature");

  const int64_t start = feature_start_indices_[batch];
  return values_.vec<tstring>().data()[start + n];
}

// A column that is backed by a dense tensor.
template <typename InternalType>
class DenseTensorColumn : public ColumnInterface<InternalType> {
 public:
  explicit DenseTensorColumn(const Tensor& tensor) : tensor_(tensor) {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSsparse_cross_opDTcc mht_13(mht_13_v, 387, "", "./tensorflow/core/kernels/sparse_cross_op.cc", "DenseTensorColumn");
}

  int64_t FeatureCount(int64_t batch) const override {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSsparse_cross_opDTcc mht_14(mht_14_v, 392, "", "./tensorflow/core/kernels/sparse_cross_op.cc", "FeatureCount");

    return tensor_.dim_size(1);
  }

  InternalType Feature(int64_t batch, int64_t n,
                       bool strong_hash) const override;

  ~DenseTensorColumn() override {
   std::vector<std::string> mht_15_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSsparse_cross_opDTcc mht_15(mht_15_v, 402, "", "./tensorflow/core/kernels/sparse_cross_op.cc", "~DenseTensorColumn");
}

 private:
  const Tensor& tensor_;
};

// A column that is backed by a dense tensor.
template <typename InternalType>
class KeyedDenseTensorColumn : public ColumnInterface<InternalType> {
 public:
  explicit KeyedDenseTensorColumn(const Tensor& tensor,
                                  std::vector<int64_t> key)
      : tensor_(tensor) {
   std::vector<std::string> mht_16_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSsparse_cross_opDTcc mht_16(mht_16_v, 417, "", "./tensorflow/core/kernels/sparse_cross_op.cc", "KeyedDenseTensorColumn");

    std::memcpy(key_, key.data(), sizeof(key_));
  }

  int64_t FeatureCount(int64_t batch) const override {
   std::vector<std::string> mht_17_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSsparse_cross_opDTcc mht_17(mht_17_v, 424, "", "./tensorflow/core/kernels/sparse_cross_op.cc", "FeatureCount");

    return tensor_.dim_size(1);
  }

  InternalType Feature(int64_t batch, int64_t n,
                       bool strong_hash) const override;

  ~KeyedDenseTensorColumn() override {
   std::vector<std::string> mht_18_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSsparse_cross_opDTcc mht_18(mht_18_v, 434, "", "./tensorflow/core/kernels/sparse_cross_op.cc", "~KeyedDenseTensorColumn");
}

 private:
  const Tensor& tensor_;
  tensorflow::uint64 key_[2];
};

// InternalType is int64 only when using HashCrosser.
template <>
int64_t DenseTensorColumn<int64_t>::Feature(int64_t batch, int64_t n,
                                            bool strong_hash) const {
   std::vector<std::string> mht_19_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSsparse_cross_opDTcc mht_19(mht_19_v, 447, "", "./tensorflow/core/kernels/sparse_cross_op.cc", "DenseTensorColumn<int64_t>::Feature");

  if (DT_STRING == tensor_.dtype())
    return Fingerprint64(tensor_.matrix<tstring>()(batch, n));
  return tensor_.matrix<int64_t>()(batch, n);
}

template <>
int64_t KeyedDenseTensorColumn<int64_t>::Feature(int64_t batch, int64_t n,
                                                 bool strong_hash) const {
   std::vector<std::string> mht_20_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSsparse_cross_opDTcc mht_20(mht_20_v, 458, "", "./tensorflow/core/kernels/sparse_cross_op.cc", "KeyedDenseTensorColumn<int64_t>::Feature");

  if (strong_hash) {
    if (DT_STRING == tensor_.dtype()) {
      return StrongKeyedHash(key_, tensor_.matrix<tstring>()(batch, n));
    }
    return StrongKeyedHash(
        key_,
        {reinterpret_cast<const char*>(tensor_.matrix<int64_t>()(batch, n)),
         sizeof(tensor_.dtype())});
  }
  if (DT_STRING == tensor_.dtype())
    return Fingerprint64(tensor_.matrix<tstring>()(batch, n));
  return tensor_.matrix<int64_t>()(batch, n);
}

// Internal type is string or StringPiece when using StringCrosser.
template <>
tstring DenseTensorColumn<tstring>::Feature(int64_t batch, int64_t n,
                                            bool strong_hash) const {
   std::vector<std::string> mht_21_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSsparse_cross_opDTcc mht_21(mht_21_v, 479, "", "./tensorflow/core/kernels/sparse_cross_op.cc", "DenseTensorColumn<tstring>::Feature");

  if (DT_STRING == tensor_.dtype()) return tensor_.matrix<tstring>()(batch, n);
  return std::to_string(tensor_.matrix<int64_t>()(batch, n));
}

template <>
tstring KeyedDenseTensorColumn<tstring>::Feature(int64_t batch, int64_t n,
                                                 bool strong_hash) const {
   std::vector<std::string> mht_22_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSsparse_cross_opDTcc mht_22(mht_22_v, 489, "", "./tensorflow/core/kernels/sparse_cross_op.cc", "KeyedDenseTensorColumn<tstring>::Feature");

  if (DT_STRING == tensor_.dtype()) return tensor_.matrix<tstring>()(batch, n);
  return std::to_string(tensor_.matrix<int64_t>()(batch, n));
}

template <>
StringPiece DenseTensorColumn<StringPiece>::Feature(int64_t batch, int64_t n,
                                                    bool strong_hash) const {
   std::vector<std::string> mht_23_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSsparse_cross_opDTcc mht_23(mht_23_v, 499, "", "./tensorflow/core/kernels/sparse_cross_op.cc", "DenseTensorColumn<StringPiece>::Feature");

  return tensor_.matrix<tstring>()(batch, n);
}

template <>
StringPiece KeyedDenseTensorColumn<StringPiece>::Feature(
    int64_t batch, int64_t n, bool strong_hash) const {
   std::vector<std::string> mht_24_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSsparse_cross_opDTcc mht_24(mht_24_v, 508, "", "./tensorflow/core/kernels/sparse_cross_op.cc", "KeyedDenseTensorColumn<StringPiece>::Feature");

  return tensor_.matrix<tstring>()(batch, n);
}

// Updates Output tensors with sparse crosses.
template <typename OutType>
class OutputUpdater {
 public:
  OutputUpdater(const std::vector<int64_t>& output_start_indices,
                Tensor* indices_out, Tensor* values_out)
      : output_start_indices_(output_start_indices),
        indices_out_(indices_out),
        values_out_(values_out) {
   std::vector<std::string> mht_25_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSsparse_cross_opDTcc mht_25(mht_25_v, 523, "", "./tensorflow/core/kernels/sparse_cross_op.cc", "OutputUpdater");
}

  void Update(const int64_t batch_index, const int64_t cross_count,
              const OutType& cross) const {
   std::vector<std::string> mht_26_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSsparse_cross_opDTcc mht_26(mht_26_v, 529, "", "./tensorflow/core/kernels/sparse_cross_op.cc", "Update");

    const int64_t output_index =
        output_start_indices_[batch_index] + cross_count;

    auto indices_matrix = indices_out_->matrix<int64_t>();
    indices_matrix(output_index, 0) = batch_index;
    indices_matrix(output_index, 1) = cross_count;

    auto value_vec = values_out_->vec<OutType>();
    value_vec(output_index) = cross;
  }

 private:
  const std::vector<int64_t>& output_start_indices_;
  Tensor* indices_out_;
  Tensor* values_out_;
};

// Generates the sparse crosses as concatenation of strings.
template <typename InternalType>
class StringCrosser {
 public:
  StringCrosser(const std::vector<
                    std::unique_ptr<ColumnInterface<InternalType>>>& columns,
                const int64_t num_buckets_unused, const uint64 hash_key_unused,
                const tstring k_feature_separator)
      : columns_(columns), k_feature_separator_(k_feature_separator) {
   std::vector<std::string> mht_27_v;
   mht_27_v.push_back("k_feature_separator: \"" + (std::string)k_feature_separator + "\"");
   MHTracer_DTPStensorflowPScorePSkernelsPSsparse_cross_opDTcc mht_27(mht_27_v, 559, "", "./tensorflow/core/kernels/sparse_cross_op.cc", "StringCrosser");
}

  string Generate(const int64_t batch_index,
                  const std::vector<int>& permutation,
                  bool unused_strong_hash) const {
   std::vector<std::string> mht_28_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSsparse_cross_opDTcc mht_28(mht_28_v, 566, "", "./tensorflow/core/kernels/sparse_cross_op.cc", "Generate");

    gtl::InlinedVector<InternalType, 6> cross_vec(columns_.size());
    for (int i = 0; i < permutation.size(); i++) {
      cross_vec[i] = columns_[i]->Feature(batch_index, permutation[i], false);
    }
    // TODO(zakaria): this will copy the string twice, might effect
    // performance.
    return absl::StrJoin(cross_vec, k_feature_separator_);
  }

 private:
  const std::vector<std::unique_ptr<ColumnInterface<InternalType>>>& columns_;
  const tstring k_feature_separator_;
};

// Generates the sparse crosses as nested hash to avoid string manipulations.
class HashCrosser {
 public:
  HashCrosser(
      const std::vector<std::unique_ptr<ColumnInterface<int64_t>>>& columns,
      const int64_t num_buckets, const uint64 hash_key,
      const tstring k_feature_separator_unused)
      : columns_(columns), num_buckets_(num_buckets), hash_key_(hash_key) {
   std::vector<std::string> mht_29_v;
   mht_29_v.push_back("k_feature_separator_unused: \"" + (std::string)k_feature_separator_unused + "\"");
   MHTracer_DTPStensorflowPScorePSkernelsPSsparse_cross_opDTcc mht_29(mht_29_v, 592, "", "./tensorflow/core/kernels/sparse_cross_op.cc", "HashCrosser");
}

  int64_t Generate(const int64_t batch_index,
                   const std::vector<int>& permutation,
                   bool unused_strong_hash) const {
   std::vector<std::string> mht_30_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSsparse_cross_opDTcc mht_30(mht_30_v, 599, "", "./tensorflow/core/kernels/sparse_cross_op.cc", "Generate");

    // Do the fingerprint concatenation on uint64.
    uint64 hashed_output = hash_key_;
    for (size_t i = 0; i < permutation.size(); ++i) {
      uint64 hash_i = columns_[i]->Feature(batch_index, permutation[i], false);
      hashed_output = FingerprintCat64(hashed_output, hash_i);
    }
    // The return value is int64 based on the number of buckets.
    if (num_buckets_ > 0) {
      return hashed_output % num_buckets_;
    } else {
      // To prevent negative output we take modulo to max int64.
      return hashed_output % std::numeric_limits<int64_t>::max();
    }
  }

 private:
  const std::vector<std::unique_ptr<ColumnInterface<int64_t>>>& columns_;
  const int64_t num_buckets_;
  const uint64 hash_key_;
};

// Generates the sparse crosses as nested hash to avoid string manipulations.
class HashCrosserV2 {
 public:
  HashCrosserV2(
      const std::vector<std::unique_ptr<ColumnInterface<int64_t>>>& columns,
      const int64_t num_buckets, const uint64 hash_key_unused,
      const tstring k_feature_separator_unused)
      : columns_(columns), num_buckets_(num_buckets) {
   std::vector<std::string> mht_31_v;
   mht_31_v.push_back("k_feature_separator_unused: \"" + (std::string)k_feature_separator_unused + "\"");
   MHTracer_DTPStensorflowPScorePSkernelsPSsparse_cross_opDTcc mht_31(mht_31_v, 632, "", "./tensorflow/core/kernels/sparse_cross_op.cc", "HashCrosserV2");
}

  int64_t Generate(const int64_t batch_index,
                   const std::vector<int>& permutation,
                   bool strong_hash) const {
   std::vector<std::string> mht_32_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSsparse_cross_opDTcc mht_32(mht_32_v, 639, "", "./tensorflow/core/kernels/sparse_cross_op.cc", "Generate");

    // Do the fingerprint concatenation on uint64.
    uint64 hashed_output =
        columns_[0]->Feature(batch_index, permutation[0], strong_hash);
    for (size_t i = 1; i < permutation.size(); ++i) {
      uint64 hash_i =
          columns_[i]->Feature(batch_index, permutation[i], strong_hash);
      hashed_output = FingerprintCat64(hashed_output, hash_i);
    }
    // The return value is int64 based on the number of buckets.
    if (num_buckets_ > 0) {
      return hashed_output % num_buckets_;
    } else {
      // To prevent negative output we take modulo to max int64.
      return hashed_output % std::numeric_limits<int64_t>::max();
    }
  }

 private:
  const std::vector<std::unique_ptr<ColumnInterface<int64_t>>>& columns_;
  const int64_t num_buckets_;
};

// ProductIterator generates cartesian products based on indices.
template <typename InternalType>
class ProductIterator {
 public:
  explicit ProductIterator(
      const std::vector<std::unique_ptr<ColumnInterface<InternalType>>>&
          columns,
      int64_t batch_index)
      : columns_(columns), batch_index_(batch_index) {
   std::vector<std::string> mht_33_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSsparse_cross_opDTcc mht_33(mht_33_v, 673, "", "./tensorflow/core/kernels/sparse_cross_op.cc", "ProductIterator");

    next_permutation_.resize(columns_.size(), 0);
    // Sets has_next_ to false if any feature column has 0 features.
    has_next_ = true;
    for (int i = 0; i < columns_.size(); i++) {
      if (columns_[i]->FeatureCount(batch_index_) == 0) {
        has_next_ = false;
        break;
      }
    }
  }

  std::vector<int> Next() {
    std::vector<int> permutation(next_permutation_);

    // Generates next permutation, if available.
    bool carry = true;
    for (int i = next_permutation_.size() - 1; i >= 0; i--) {
      if (carry) {
        next_permutation_[i] = next_permutation_[i] + 1;
      }
      if (next_permutation_[i] == columns_[i]->FeatureCount(batch_index_)) {
        next_permutation_[i] = 0;
      } else {
        carry = false;
        break;
      }
    }
    has_next_ = !carry;
    return permutation;
  }

  bool HasNext() {
   std::vector<std::string> mht_34_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSsparse_cross_opDTcc mht_34(mht_34_v, 708, "", "./tensorflow/core/kernels/sparse_cross_op.cc", "HasNext");
 return has_next_; }

 private:
  bool has_next_;
  const std::vector<std::unique_ptr<ColumnInterface<InternalType>>>& columns_;
  const int64_t batch_index_;
  std::vector<int> next_permutation_;
};

template <bool HASHED_OUTPUT, typename InternalType>
struct CrossTraits;

template <typename InternalType>
struct CrossTraits<false, InternalType> {
  typedef StringCrosser<InternalType> Crosser;
  typedef StringCrosser<InternalType> CrosserV2;
  typedef OutputUpdater<tstring> Updater;
};

template <>
struct CrossTraits<true, int64_t> {
  typedef HashCrosser Crosser;
  typedef HashCrosserV2 CrosserV2;
  typedef OutputUpdater<int64_t> Updater;
};
}  // namespace

// Calculate the batch size from either the shapes input or the dense input.
int64_t CalculateBatchSize(const OpInputList& shapes_list_in,
                           const OpInputList& dense_list_in) {
   std::vector<std::string> mht_35_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSsparse_cross_opDTcc mht_35(mht_35_v, 740, "", "./tensorflow/core/kernels/sparse_cross_op.cc", "CalculateBatchSize");

  if (shapes_list_in.size() > 0) {
    return shapes_list_in[0].vec<int64_t>()(0);
  }

  if (dense_list_in.size() > 0) {
    return dense_list_in[0].dim_size(0);
  }

  return 0;
}

// Validates input tensors.
Status ValidateInput(const OpInputList& indices_list_in,
                     const OpInputList& values_list_in,
                     const OpInputList& shapes_list_in,
                     const OpInputList& dense_list_in,
                     const DataType& internal_type) {
   std::vector<std::string> mht_36_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSsparse_cross_opDTcc mht_36(mht_36_v, 760, "", "./tensorflow/core/kernels/sparse_cross_op.cc", "ValidateInput");

  const auto size = indices_list_in.size();
  // Only perform internal_type check for SparseCrossOp.
  // Check if the internal_type is not invalid before doing so.
  bool check_type = internal_type != DT_INVALID;
  // Validates indices_list_in OpInputList.
  for (int i = 0; i < size; i++) {
    if (check_type && indices_list_in[i].dtype() != DT_INT64) {
      return errors::InvalidArgument("Input indices should be of type ",
                                     DT_INT64, " but received ",
                                     indices_list_in[i].dtype());
    }
    if (!TensorShapeUtils::IsMatrix(indices_list_in[i].shape())) {
      return errors::InvalidArgument(
          "Input indices should be a matrix but received shape ",
          indices_list_in[i].shape().DebugString(), " at position ", i);
    }
    if (indices_list_in[i].shape().dim_size(1) != 2) {
      return errors::InvalidArgument("Expected D2 of index to be 2 got ",
                                     indices_list_in[i].shape().dim_size(1),
                                     " at position ", i);
    }
  }

  // Validates values_list_in OpInputList.
  if (values_list_in.size() != size) {
    return errors::InvalidArgument("Expected ", size, " input values, got ",
                                   values_list_in.size());
  }
  for (int i = 0; i < size; i++) {
    // Make sure to avoid the expected type to be string, but input values to be
    // int64.
    if (check_type && internal_type == DT_STRING &&
        values_list_in[i].dtype() == DT_INT64) {
      return errors::InvalidArgument("Input values should be of internal type ",
                                     internal_type, " but received ",
                                     values_list_in[i].dtype());
    }
    if (!TensorShapeUtils::IsVector(values_list_in[i].shape())) {
      return errors::InvalidArgument(
          "Input values should be a vector but received shape ",
          values_list_in[i].shape().DebugString(), " at position ", i);
    }
    if (indices_list_in[i].shape().dim_size(0) !=
        values_list_in[i].shape().dim_size(0)) {
      return errors::InvalidArgument(
          "Expected size of values to be ",
          indices_list_in[i].shape().dim_size(0), " got ",
          values_list_in[i].shape().dim_size(0), " at position ", i);
    }
  }

  // Validates shapes_list_in OpInputList
  if (shapes_list_in.size() != size) {
    return errors::InvalidArgument("Expected ", size, " input shapes, got ",
                                   shapes_list_in.size());
  }
  for (int i = 0; i < size; i++) {
    if (check_type && shapes_list_in[i].dtype() != DT_INT64) {
      return errors::InvalidArgument("Input shape should be of type ", DT_INT64,
                                     " but received ",
                                     shapes_list_in[i].dtype());
    }
    if (!TensorShapeUtils::IsVector(shapes_list_in[i].shape())) {
      return errors::InvalidArgument(
          "Input shapes should be a vector but received shape ",
          shapes_list_in[i].shape().DebugString(), " at position ", i);
    }

    if (shapes_list_in[i].vec<int64_t>().size() != 2) {
      return errors::InvalidArgument("shape should imply a 2D tensor, but got ",
                                     shapes_list_in[i].shape().DebugString(),
                                     " at position ", i);
    }
  }

  // Validates dense_list_in OpInputList
  for (int i = 0; i < dense_list_in.size(); ++i) {
    // Make sure to avoid the expected type to be string, but input values to be
    // int64.
    if (check_type && internal_type == DT_STRING &&
        dense_list_in[i].dtype() == DT_INT64) {
      return errors::InvalidArgument("Dense inputs should be of internal type ",
                                     internal_type, " but received ",
                                     dense_list_in[i].dtype());
    }
    if (!TensorShapeUtils::IsMatrix(dense_list_in[i].shape())) {
      return errors::InvalidArgument(
          "Dense inputs should be a matrix but received shape ",
          dense_list_in[i].shape().DebugString(), " at position ", i);
    }
  }

  // Validates batch sizes.  (Note: we do this after validating the input
  // shapes, because CalculateBatchSize() depends on inputs having valid
  // shapes).
  const auto batch_size = CalculateBatchSize(shapes_list_in, dense_list_in);
  for (int i = 0; i < size; i++) {
    if (shapes_list_in[i].vec<int64_t>()(0) != batch_size) {
      return errors::InvalidArgument(
          "Expected batch size ", batch_size, " got ",
          shapes_list_in[i].vec<int64_t>()(0), " at position ", i);
    }
  }
  for (int i = 0; i < dense_list_in.size(); ++i) {
    if (dense_list_in[i].dim_size(0) != batch_size) {
      return errors::InvalidArgument("Expected batch size ", batch_size,
                                     " got ", dense_list_in[i].dim_size(0),
                                     " at dense tensor ", i);
    }
  }

  return Status::OK();
}

// Extracts data about the features and populates feature data.
void ExtractFeatureData(
    const OpInputList& indices_list_in, int64_t batch_size,
    std::vector<std::vector<int64_t>>* feature_counts,
    std::vector<std::vector<int64_t>>* feature_start_indices) {
   std::vector<std::string> mht_37_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSsparse_cross_opDTcc mht_37(mht_37_v, 882, "", "./tensorflow/core/kernels/sparse_cross_op.cc", "ExtractFeatureData");

  gtl::InlinedVector<int64_t, 8> current_row(indices_list_in.size(), 0);
  for (int b = 0; b < batch_size; b++) {
    for (int i = 0; i < indices_list_in.size(); i++) {
      const auto indices = indices_list_in[i].matrix<int64_t>();
      int64_t feature_count = 0;
      int64_t start_index = current_row[i];
      // Loops until we reach next batch index for current feature column.
      while (current_row[i] < indices_list_in[i].dim_size(0) &&
             indices(current_row[i], 0) == b) {
        feature_count++;
        current_row[i]++;
      }
      (*feature_counts)[i].push_back(feature_count);
      (*feature_start_indices)[i].push_back(start_index);
    }
  }
}

// Returns number of crosses for a given batch_index
template <typename InternalType>
int64_t CrossCountByBatchIndex(
    const std::vector<std::unique_ptr<ColumnInterface<InternalType>>>& columns,
    int batch_index) {
   std::vector<std::string> mht_38_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSsparse_cross_opDTcc mht_38(mht_38_v, 908, "", "./tensorflow/core/kernels/sparse_cross_op.cc", "CrossCountByBatchIndex");

  int64_t cross_count = 1;
  for (int i = 0; i < columns.size(); i++) {
    const auto feature_count = columns[i]->FeatureCount(batch_index);
    // If one column is missing any feature, there won't be any cross.
    if (feature_count == 0) {
      return 0;
    }
    cross_count *= feature_count;
  }
  return cross_count;
}

// Generate the columns given the sparse and dense inputs.
template <typename InternalType>
std::vector<std::unique_ptr<ColumnInterface<InternalType>>>
GenerateColumnsFromInput(const OpInputList& indices_list_in,
                         const OpInputList& values_list_in,
                         const OpInputList& shapes_list_in,
                         const OpInputList& dense_list_in) {
  std::vector<std::unique_ptr<ColumnInterface<InternalType>>> columns;
  const int64_t batch_size = CalculateBatchSize(shapes_list_in, dense_list_in);
  const int64_t number_of_columns = shapes_list_in.size();

  std::vector<std::vector<int64_t>> feature_counts(number_of_columns,
                                                   std::vector<int64_t>());
  std::vector<std::vector<int64_t>> feature_start_indices(
      number_of_columns, std::vector<int64_t>());

  ExtractFeatureData(indices_list_in, batch_size, &feature_counts,
                     &feature_start_indices);

  columns.reserve(values_list_in.size());
  for (int i = 0; i < values_list_in.size(); ++i) {
    columns.emplace_back(new SparseTensorColumn<InternalType>(
        values_list_in[i], std::move(feature_counts[i]),
        std::move(feature_start_indices[i])));
  }
  for (int i = 0; i < dense_list_in.size(); ++i) {
    columns.emplace_back(new DenseTensorColumn<InternalType>(dense_list_in[i]));
  }

  return columns;
}

// Generate the columns given the sparse and dense inputs.
template <typename InternalType>
std::vector<std::unique_ptr<ColumnInterface<InternalType>>>
GenerateKeyedColumnsFromInput(const OpInputList& indices_list_in,
                              const OpInputList& values_list_in,
                              const OpInputList& shapes_list_in,
                              const OpInputList& dense_list_in,
                              std::vector<int64_t> keys) {
  std::vector<std::unique_ptr<ColumnInterface<InternalType>>> columns;
  const int64_t batch_size = CalculateBatchSize(shapes_list_in, dense_list_in);
  const int64_t number_of_columns = shapes_list_in.size();

  std::vector<std::vector<int64_t>> feature_counts(number_of_columns,
                                                   std::vector<int64_t>());
  std::vector<std::vector<int64_t>> feature_start_indices(
      number_of_columns, std::vector<int64_t>());

  ExtractFeatureData(indices_list_in, batch_size, &feature_counts,
                     &feature_start_indices);

  columns.reserve(values_list_in.size());
  for (int i = 0; i < values_list_in.size(); ++i) {
    columns.emplace_back(new KeyedSparseTensorColumn<InternalType>(
        values_list_in[i], std::move(feature_counts[i]),
        std::move(feature_start_indices[i]), keys));
  }
  for (int i = 0; i < dense_list_in.size(); ++i) {
    columns.emplace_back(
        new KeyedDenseTensorColumn<InternalType>(dense_list_in[i], keys));
  }

  return columns;
}

// Allocates output tensors with proper size and sets the shape tensor of
// the output SparseTensor.
// It also output_start_indices which contains the start indices for each
// input in the output SparseTensor.
template <typename InternalType>
Status CreateOutputTensors(
    const std::vector<std::unique_ptr<ColumnInterface<InternalType>>>& columns,
    int64_t batch_size, OpKernelContext* context, Tensor** indices_out,
    Tensor** values_out, Tensor** shape_out,
    std::vector<int64_t>* output_start_indices) {
   std::vector<std::string> mht_39_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSsparse_cross_opDTcc mht_39(mht_39_v, 999, "", "./tensorflow/core/kernels/sparse_cross_op.cc", "CreateOutputTensors");

  // Calculates dimensions for output tensors.
  int64_t cross_count_total = 0;
  int64_t max_cross_count = 0;
  for (int64_t b = 0; b < batch_size; b++) {
    // For each input, sets starting indices in output SparseTensor
    (*output_start_indices)[b] = cross_count_total;
    const auto cross_count = CrossCountByBatchIndex(columns, b);
    max_cross_count = std::max(max_cross_count, cross_count);
    cross_count_total += cross_count;
  }

  // Allocates tensors.
  TF_RETURN_IF_ERROR(context->allocate_output(
      0, TensorShape({cross_count_total, 2}), indices_out));
  TF_RETURN_IF_ERROR(context->allocate_output(
      1, TensorShape({cross_count_total}), values_out));
  TF_RETURN_IF_ERROR(context->allocate_output(2, TensorShape({2}), shape_out));

  // Sets shape.
  auto shape_vec = (*shape_out)->vec<int64_t>();
  shape_vec(0) = batch_size;
  shape_vec(1) = max_cross_count;

  return Status::OK();
}

template <bool HASHED_OUTPUT, typename InternalType>
class SparseCrossOp : public OpKernel {
 public:
  explicit SparseCrossOp(OpKernelConstruction* context) : OpKernel(context) {
   std::vector<std::string> mht_40_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSsparse_cross_opDTcc mht_40(mht_40_v, 1032, "", "./tensorflow/core/kernels/sparse_cross_op.cc", "SparseCrossOp");

    OP_REQUIRES_OK(context, context->GetAttr("num_buckets", &num_buckets_));
    // Read signed_hash_key_ as int64 since uint64 attributes are not
    // supported by REGISTER_OP.
    int64_t signed_hash_key_;
    OP_REQUIRES_OK(context, context->GetAttr("hash_key", &signed_hash_key_));
    hash_key_ = static_cast<uint64>(signed_hash_key_);
    OP_REQUIRES_OK(context, context->GetAttr("internal_type", &internal_type_));
  }

  void Compute(OpKernelContext* context) override {
   std::vector<std::string> mht_41_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSsparse_cross_opDTcc mht_41(mht_41_v, 1045, "", "./tensorflow/core/kernels/sparse_cross_op.cc", "Compute");

    OpInputList indices_list_in;
    OP_REQUIRES_OK(context, context->input_list("indices", &indices_list_in));
    OpInputList values_list_in;
    OP_REQUIRES_OK(context, context->input_list("values", &values_list_in));
    OpInputList shapes_list_in;
    OP_REQUIRES_OK(context, context->input_list("shapes", &shapes_list_in));
    OpInputList dense_list_in;
    OP_REQUIRES_OK(context,
                   context->input_list("dense_inputs", &dense_list_in));

    DataType internal_type = internal_type_;
    OP_REQUIRES_OK(
        context, ValidateInput(indices_list_in, values_list_in, shapes_list_in,
                               dense_list_in, internal_type));

    std::vector<std::unique_ptr<ColumnInterface<InternalType>>> columns =
        GenerateColumnsFromInput<InternalType>(indices_list_in, values_list_in,
                                               shapes_list_in, dense_list_in);

    const tstring k_feature_separator = "_X_";
    typename CrossTraits<HASHED_OUTPUT, InternalType>::Crosser crosser(
        columns, num_buckets_, hash_key_, k_feature_separator);
    Tensor* indices_out;
    Tensor* values_out;
    Tensor* shape_out;
    const int64_t batch_size =
        CalculateBatchSize(shapes_list_in, dense_list_in);
    std::vector<int64_t> output_start_indices(batch_size);
    OP_REQUIRES_OK(
        context,
        CreateOutputTensors(columns, batch_size, context, &indices_out,
                            &values_out, &shape_out, &output_start_indices));

    typename CrossTraits<HASHED_OUTPUT, InternalType>::Updater updater(
        output_start_indices, indices_out, values_out);
    auto do_work = [&columns, crosser, updater](int64_t begin, int64_t end) {
   std::vector<std::string> mht_42_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSsparse_cross_opDTcc mht_42(mht_42_v, 1084, "", "./tensorflow/core/kernels/sparse_cross_op.cc", "lambda");

      for (int b = begin; b < end; b++) {
        ProductIterator<InternalType> product_iterator(columns, b);
        int64_t cross_count = 0;
        while (product_iterator.HasNext()) {
          const auto permutation = product_iterator.Next();
          updater.Update(b, cross_count,
                         crosser.Generate(b, permutation, false));
          cross_count++;
        }
      }
    };

    auto* worker_threads = context->device()->tensorflow_cpu_worker_threads();
    // TODO(zakaria): optimize kCostPerUnit
    const int kCostPerUnit = 5000 * indices_list_in.size();
    Shard(worker_threads->num_threads, worker_threads->workers, batch_size,
          kCostPerUnit, do_work);
  }

 private:
  int64_t num_buckets_;
  uint64 hash_key_;
  DataType internal_type_;
};

class SparseCrossV2Op : public OpKernel {
 public:
  explicit SparseCrossV2Op(OpKernelConstruction* context) : OpKernel(context) {
   std::vector<std::string> mht_43_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSsparse_cross_opDTcc mht_43(mht_43_v, 1115, "", "./tensorflow/core/kernels/sparse_cross_op.cc", "SparseCrossV2Op");
}

  void Compute(OpKernelContext* context) override {
   std::vector<std::string> mht_44_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSsparse_cross_opDTcc mht_44(mht_44_v, 1120, "", "./tensorflow/core/kernels/sparse_cross_op.cc", "Compute");

    OpInputList indices_list_in;
    OP_REQUIRES_OK(context, context->input_list("indices", &indices_list_in));
    OpInputList values_list_in;
    OP_REQUIRES_OK(context, context->input_list("values", &values_list_in));
    OpInputList shapes_list_in;
    OP_REQUIRES_OK(context, context->input_list("shapes", &shapes_list_in));
    OpInputList dense_list_in;
    OP_REQUIRES_OK(context,
                   context->input_list("dense_inputs", &dense_list_in));

    // Set internal_type to invalid_type so that the check will be ignored.
    DataType internal_type = DT_INVALID;
    OP_REQUIRES_OK(
        context, ValidateInput(indices_list_in, values_list_in, shapes_list_in,
                               dense_list_in, internal_type));

    const Tensor* sep_t;
    OP_REQUIRES_OK(context, context->input("sep", &sep_t));
    const tstring separator = sep_t->scalar<tstring>()();

    std::vector<std::unique_ptr<ColumnInterface<tstring>>> columns =
        GenerateColumnsFromInput<tstring>(indices_list_in, values_list_in,
                                          shapes_list_in, dense_list_in);
    Tensor* indices_out;
    Tensor* values_out;
    Tensor* shape_out;
    const int64_t batch_size =
        CalculateBatchSize(shapes_list_in, dense_list_in);
    std::vector<int64_t> output_start_indices(batch_size);
    OP_REQUIRES_OK(
        context,
        CreateOutputTensors(columns, batch_size, context, &indices_out,
                            &values_out, &shape_out, &output_start_indices));
    StringCrosser<tstring> crosser(columns, 0, 0, separator);
    OutputUpdater<tstring> updater(output_start_indices, indices_out,
                                   values_out);
    auto do_work = [&columns, crosser, updater](int64_t begin, int64_t end) {
   std::vector<std::string> mht_45_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSsparse_cross_opDTcc mht_45(mht_45_v, 1160, "", "./tensorflow/core/kernels/sparse_cross_op.cc", "lambda");

      for (int b = begin; b < end; b++) {
        ProductIterator<tstring> product_iterator(columns, b);
        int64_t cross_count = 0;
        while (product_iterator.HasNext()) {
          const auto permutation = product_iterator.Next();
          updater.Update(b, cross_count,
                         crosser.Generate(b, permutation, false));
          cross_count++;
        }
      }
    };

    auto* worker_threads = context->device()->tensorflow_cpu_worker_threads();
    // TODO(zakaria): optimize kCostPerUnit
    const int kCostPerUnit = 5000 * indices_list_in.size();
    Shard(worker_threads->num_threads, worker_threads->workers, batch_size,
          kCostPerUnit, do_work);
  }
};

class SparseCrossHashedOp : public OpKernel {
 public:
  explicit SparseCrossHashedOp(OpKernelConstruction* context)
      : OpKernel(context) {
   std::vector<std::string> mht_46_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSsparse_cross_opDTcc mht_46(mht_46_v, 1187, "", "./tensorflow/core/kernels/sparse_cross_op.cc", "SparseCrossHashedOp");
}

  void Compute(OpKernelContext* context) override {
   std::vector<std::string> mht_47_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSsparse_cross_opDTcc mht_47(mht_47_v, 1192, "", "./tensorflow/core/kernels/sparse_cross_op.cc", "Compute");

    OpInputList indices_list_in;
    OP_REQUIRES_OK(context, context->input_list("indices", &indices_list_in));
    OpInputList values_list_in;
    OP_REQUIRES_OK(context, context->input_list("values", &values_list_in));
    OpInputList shapes_list_in;
    OP_REQUIRES_OK(context, context->input_list("shapes", &shapes_list_in));
    OpInputList dense_list_in;
    OP_REQUIRES_OK(context,
                   context->input_list("dense_inputs", &dense_list_in));

    // Set internal_type to invalid_type so that the check will be ignored.
    DataType internal_type = DT_INVALID;
    OP_REQUIRES_OK(
        context, ValidateInput(indices_list_in, values_list_in, shapes_list_in,
                               dense_list_in, internal_type));

    const Tensor* num_buckets_t;
    OP_REQUIRES_OK(context, context->input("num_buckets", &num_buckets_t));
    const int64_t num_buckets = num_buckets_t->scalar<int64_t>()();

    const Tensor* strong_hash_t;
    OP_REQUIRES_OK(context, context->input("strong_hash", &strong_hash_t));
    const bool strong_hash = strong_hash_t->scalar<bool>()();

    const Tensor* salt_t;
    OP_REQUIRES_OK(context, context->input("salt", &salt_t));
    const auto salt = salt_t->flat<int64_t>();
    std::vector<int64_t> key_{salt(0), salt(1)};

    std::vector<std::unique_ptr<ColumnInterface<int64_t>>> columns =
        GenerateKeyedColumnsFromInput<int64_t>(indices_list_in, values_list_in,
                                               shapes_list_in, dense_list_in,
                                               key_);
    Tensor* indices_out;
    Tensor* values_out;
    Tensor* shape_out;
    const int64_t batch_size =
        CalculateBatchSize(shapes_list_in, dense_list_in);
    std::vector<int64_t> output_start_indices(batch_size);
    OP_REQUIRES_OK(
        context,
        CreateOutputTensors(columns, batch_size, context, &indices_out,
                            &values_out, &shape_out, &output_start_indices));
    const tstring unused_sep;
    HashCrosserV2 crosser(columns, num_buckets, 0, unused_sep);
    OutputUpdater<int64_t> updater(output_start_indices, indices_out,
                                   values_out);
    auto do_work = [&columns, crosser, updater, strong_hash](int64_t begin,
                                                             int64_t end) {
   std::vector<std::string> mht_48_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSsparse_cross_opDTcc mht_48(mht_48_v, 1244, "", "./tensorflow/core/kernels/sparse_cross_op.cc", "lambda");

      for (int b = begin; b < end; b++) {
        ProductIterator<int64_t> product_iterator(columns, b);
        int64_t cross_count = 0;
        while (product_iterator.HasNext()) {
          const auto permutation = product_iterator.Next();
          updater.Update(b, cross_count,
                         crosser.Generate(b, permutation, strong_hash));
          cross_count++;
        }
      }
    };

    auto* worker_threads = context->device()->tensorflow_cpu_worker_threads();
    // TODO(zakaria): optimize kCostPerUnit
    const int kCostPerUnit = 5000 * indices_list_in.size();
    Shard(worker_threads->num_threads, worker_threads->workers, batch_size,
          kCostPerUnit, do_work);
  }
};

REGISTER_KERNEL_BUILDER(Name("SparseCross")
                            .Device(DEVICE_CPU)
                            .TypeConstraint<tstring>("out_type")
                            .TypeConstraint<tstring>("internal_type"),
                        SparseCrossOp<false, StringPiece>);

REGISTER_KERNEL_BUILDER(Name("SparseCross")
                            .Device(DEVICE_CPU)
                            .TypeConstraint<tstring>("out_type")
                            .TypeConstraint<int64_t>("internal_type"),
                        SparseCrossOp<false, tstring>);

REGISTER_KERNEL_BUILDER(Name("SparseCross")
                            .Device(DEVICE_CPU)
                            .TypeConstraint<int64_t>("out_type")
                            .TypeConstraint<tstring>("internal_type"),
                        SparseCrossOp<true, int64>);

REGISTER_KERNEL_BUILDER(Name("SparseCross")
                            .Device(DEVICE_CPU)
                            .TypeConstraint<int64_t>("out_type")
                            .TypeConstraint<int64_t>("internal_type"),
                        SparseCrossOp<true, int64>);

REGISTER_KERNEL_BUILDER(Name("SparseCrossV2").Device(DEVICE_CPU),
                        SparseCrossV2Op);

REGISTER_KERNEL_BUILDER(Name("SparseCrossHashed").Device(DEVICE_CPU),
                        SparseCrossHashedOp);

}  // namespace tensorflow
