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
class MHTracer_DTPStensorflowPScorePSkernelsPSragged_tensor_to_tensor_opDTcc {
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
   MHTracer_DTPStensorflowPScorePSkernelsPSragged_tensor_to_tensor_opDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSkernelsPSragged_tensor_to_tensor_opDTcc() {
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

#define EIGEN_USE_THREADS

#include <stddef.h>

#include <algorithm>
#include <string>
#include <vector>

#include "tensorflow/core/framework/kernel_def_builder.h"
#include "tensorflow/core/framework/numeric_types.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/tensor_shape.pb.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/kernels/broadcast_to_op.h"
#include "tensorflow/core/kernels/list_kernels.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/bfloat16.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/util/bcast.h"
#include "tensorflow/core/util/ragged_to_dense_util.h"

namespace tensorflow {

namespace {
typedef Eigen::ThreadPoolDevice CPUDevice;
using ::std::vector;

const int kShapeInputIndex = 0;
const int kValueInputIndex = 1;
const int kDefaultValueInputIndex = 2;
const int kFirstPartitionInputIndex = 3;

template <typename INDEX_TYPE>
class RaggedTensorToTensorBaseOp : public OpKernel {
 public:
  typedef
      typename ::tensorflow::TTypes<const INDEX_TYPE>::Flat RowPartitionTensor;

  explicit RaggedTensorToTensorBaseOp(OpKernelConstruction* context)
      : OpKernel(context) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSragged_tensor_to_tensor_opDTcc mht_0(mht_0_v, 231, "", "./tensorflow/core/kernels/ragged_tensor_to_tensor_op.cc", "RaggedTensorToTensorBaseOp");

    OP_REQUIRES_OK(context, GetRowPartitionTypes<OpKernelConstruction>(
                                context, &row_partition_types_));
    ragged_rank_ = GetRaggedRank(row_partition_types_);
  }

  // Returns the relationship between dimension and dimension + 1.
  RowPartitionType GetRowPartitionTypeByDimension(int dimension) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSragged_tensor_to_tensor_opDTcc mht_1(mht_1_v, 241, "", "./tensorflow/core/kernels/ragged_tensor_to_tensor_op.cc", "GetRowPartitionTypeByDimension");

    if (row_partition_types_[0] == RowPartitionType::FIRST_DIM_SIZE) {
      return row_partition_types_[dimension + 1];
    } else {
      return row_partition_types_[dimension];
    }
  }

  // Returns the relationship between dimension and dimension + 1.
  RowPartitionTensor GetRowPartitionTensor(OpKernelContext* c, int dimension) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSragged_tensor_to_tensor_opDTcc mht_2(mht_2_v, 253, "", "./tensorflow/core/kernels/ragged_tensor_to_tensor_op.cc", "GetRowPartitionTensor");

    if (row_partition_types_[0] == RowPartitionType::FIRST_DIM_SIZE) {
      return c->input(dimension + 1 + kFirstPartitionInputIndex)
          .flat<INDEX_TYPE>();
    } else {
      return c->input(dimension + kFirstPartitionInputIndex).flat<INDEX_TYPE>();
    }
  }

  Status GetMaxWidth(OpKernelContext* c, int dimension, INDEX_TYPE* result) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSragged_tensor_to_tensor_opDTcc mht_3(mht_3_v, 265, "", "./tensorflow/core/kernels/ragged_tensor_to_tensor_op.cc", "GetMaxWidth");

    const RowPartitionTensor row_partition_tensor =
        GetRowPartitionTensor(c, dimension - 1);
    switch (GetRowPartitionTypeByDimension(dimension - 1)) {
      case RowPartitionType::VALUE_ROWIDS:
        *result = GetMaxWidthValueRowID(row_partition_tensor);
        return Status::OK();
      case RowPartitionType::ROW_SPLITS:
        *result = GetMaxWidthRowSplit(row_partition_tensor);
        return Status::OK();
      default:
        return errors::InvalidArgument(
            "Cannot handle partition type ",
            RowPartitionTypeToString(
                GetRowPartitionTypeByDimension(dimension - 1)));
    }
  }

  static INDEX_TYPE GetMaxWidthRowSplit(const RowPartitionTensor& row_split) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSragged_tensor_to_tensor_opDTcc mht_4(mht_4_v, 286, "", "./tensorflow/core/kernels/ragged_tensor_to_tensor_op.cc", "GetMaxWidthRowSplit");

    const INDEX_TYPE tensor_length = row_split.size();
    if (tensor_length == 0 || tensor_length == 1) {
      return 0;
    }
    INDEX_TYPE max_width = 0;
    for (INDEX_TYPE i = 0; i < tensor_length - 1; ++i) {
      const INDEX_TYPE current_width = row_split(i + 1) - row_split(i);
      if (current_width > max_width) {
        max_width = current_width;
      }
    }
    return max_width;
  }

  static INDEX_TYPE GetMaxWidthValueRowID(
      const RowPartitionTensor& value_rowids) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSragged_tensor_to_tensor_opDTcc mht_5(mht_5_v, 305, "", "./tensorflow/core/kernels/ragged_tensor_to_tensor_op.cc", "GetMaxWidthValueRowID");

    const INDEX_TYPE index_length = value_rowids.size();
    if (index_length == 0) {
      return 0;
    }
    INDEX_TYPE first_equal_index = 0;
    INDEX_TYPE first_equal_index_value = value_rowids(0);
    INDEX_TYPE max_width = 0;
    for (INDEX_TYPE i = 1; i < index_length; ++i) {
      const INDEX_TYPE value = value_rowids(i);
      if (value != first_equal_index_value) {
        first_equal_index_value = value;
        max_width = std::max(i - first_equal_index, max_width);
        first_equal_index = i;
      }
    }
    return std::max(index_length - first_equal_index, max_width);
  }

  Status CalculateOutputSize(INDEX_TYPE first_dim, OpKernelContext* c,
                             vector<INDEX_TYPE>* result) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSragged_tensor_to_tensor_opDTcc mht_6(mht_6_v, 328, "", "./tensorflow/core/kernels/ragged_tensor_to_tensor_op.cc", "CalculateOutputSize");

    TensorShapeProto value_shape_proto;
    c->input(kValueInputIndex).shape().AsProto(&value_shape_proto);

    TensorShapeProto default_value_shape_proto;
    c->input(kDefaultValueInputIndex)
        .shape()
        .AsProto(&default_value_shape_proto);

    TensorShapeProto output_shape_proto;
    TF_RETURN_IF_ERROR(ValidateDefaultValueShape(default_value_shape_proto,
                                                 value_shape_proto));

    TensorShapeProto shape_proto;
    {
      PartialTensorShape partial_tensor_shape;
      TF_RETURN_IF_ERROR(TensorShapeFromTensor(c->input(kShapeInputIndex),
                                               &partial_tensor_shape));
      partial_tensor_shape.AsProto(&shape_proto);
    }

    TF_RETURN_IF_ERROR(CombineRaggedTensorToTensorShapes(
        ragged_rank_, shape_proto, value_shape_proto, &output_shape_proto));

    result->reserve(output_shape_proto.dim_size());
    for (const TensorShapeProto::Dim& dim : output_shape_proto.dim()) {
      // Note that this may be -1 (if dimension size is unknown).
      result->push_back(dim.size());
    }

    if ((*result)[0] < 0) {
      (*result)[0] = first_dim;
    }
    for (int i = 1; i <= ragged_rank_; ++i) {
      if ((*result)[i] < 0) {
        TF_RETURN_IF_ERROR(GetMaxWidth(c, i, &(*result)[i]));
      }
    }
    return Status::OK();
  }

  /**
   * The output_index represents the index in the output tensor
   * where the first element of a particular dimension would be written.
   * If it is -1, it indicates that the index is out of scope.
   * Example, given first_dimension = 10, first_dimension_output = 6,
   * and output_index_multiplier = 100:
   * result = [0 100 200 300 400 500 -1 -1 -1 -1]
   * If first_dimension_output = 11 instead, then:
   * result = [0 100 200 300 400 500 600 700 800 900]
   */
  void CalculateFirstParentOutputIndex(INDEX_TYPE first_dimension,
                                       INDEX_TYPE output_index_multiplier,
                                       INDEX_TYPE first_dimension_output,
                                       vector<INDEX_TYPE>* result) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSragged_tensor_to_tensor_opDTcc mht_7(mht_7_v, 385, "", "./tensorflow/core/kernels/ragged_tensor_to_tensor_op.cc", "CalculateFirstParentOutputIndex");

    const INDEX_TYPE min_dimension =
        std::min(first_dimension, first_dimension_output);
    result->reserve(first_dimension);
    int current_output_index = 0;
    for (INDEX_TYPE i = 0; i < min_dimension;
         ++i, current_output_index += output_index_multiplier) {
      result->push_back(current_output_index);
    }
    for (INDEX_TYPE i = min_dimension; i < first_dimension; ++i) {
      result->push_back(-1);
    }
    DCHECK_EQ(result->size(), first_dimension);
  }

  Status CalculateOutputIndexRowSplit(
      const RowPartitionTensor& row_split,
      const vector<INDEX_TYPE>& parent_output_index,
      INDEX_TYPE output_index_multiplier, INDEX_TYPE output_size,
      vector<INDEX_TYPE>* result) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSragged_tensor_to_tensor_opDTcc mht_8(mht_8_v, 407, "", "./tensorflow/core/kernels/ragged_tensor_to_tensor_op.cc", "CalculateOutputIndexRowSplit");

    INDEX_TYPE row_split_size = row_split.size();
    if (row_split_size > 0) {
      result->reserve(row_split(row_split_size - 1));
    }
    for (INDEX_TYPE i = 0; i < row_split_size - 1; ++i) {
      INDEX_TYPE row_length = row_split(i + 1) - row_split(i);
      INDEX_TYPE real_length = std::min(output_size, row_length);
      INDEX_TYPE parent_output_index_current = parent_output_index[i];

      if (parent_output_index_current == -1) {
        real_length = 0;
      }
      for (INDEX_TYPE j = 0; j < real_length; ++j) {
        result->push_back(parent_output_index_current);
        parent_output_index_current += output_index_multiplier;
      }
      for (INDEX_TYPE j = 0; j < row_length - real_length; ++j) {
        result->push_back(-1);
      }
    }
    if (row_split_size > 0 && result->size() != row_split(row_split_size - 1)) {
      return errors::InvalidArgument("Invalid row split size.");
    }

    return Status::OK();
  }

  // Calculate the output index of the first element of a list.
  // The parent_output_index is the same computation for the previous list.
  // -1 indicates an element or list that is out of range.
  // The output_index_multiplier is the number of output indices one moves
  // forward for each column.
  // E.g., given:
  // value_rowids:[0 1 2 2 2 3 5 5 6]
  // parent_output_index:[1000 1100 2000 2100 -1 3000 4000]
  // output_index_multiplier: 10
  // output_size: 2
  // You get:
  // result = [1000 1100 2000 2010 -1 2100 -1 -1 3000]
  // result[0] = parent_output_index[value_rowids[0]]
  // result[1] = parent_output_index[value_rowids[1]]
  // result[2] = parent_output_index[value_rowids[2]]
  // result[3] = parent_output_index[value_rowids[2] + 10]
  // result[4] = -1 because it is the third element the size is 2.
  // result[5] = parent_output_index[value_rowids[3]]
  // result[6] = -1 because parent_output_index[value_rowids[6]] == -1
  // result[7] = -1 because parent_output_index[value_rowids[6]] == -1
  // result[8] = parent_output_index[value_rowids[7]]
  Status CalculateOutputIndexValueRowID(
      const RowPartitionTensor& value_rowids,
      const vector<INDEX_TYPE>& parent_output_index,
      INDEX_TYPE output_index_multiplier, INDEX_TYPE output_size,
      vector<INDEX_TYPE>* result) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSragged_tensor_to_tensor_opDTcc mht_9(mht_9_v, 463, "", "./tensorflow/core/kernels/ragged_tensor_to_tensor_op.cc", "CalculateOutputIndexValueRowID");

    const INDEX_TYPE index_size = value_rowids.size();
    result->reserve(index_size);
    if (index_size == 0) {
      return Status::OK();
    }

    INDEX_TYPE current_output_column = 0;
    INDEX_TYPE current_value_rowid = value_rowids(0);

    if (current_value_rowid >= parent_output_index.size()) {
      return errors::InvalidArgument(
          "Got current_value_rowid=", current_value_rowid,
          " which is not less than ", parent_output_index.size());
    }

    INDEX_TYPE current_output_index = parent_output_index[current_value_rowid];
    result->push_back(current_output_index);
    for (INDEX_TYPE i = 1; i < index_size; ++i) {
      INDEX_TYPE next_value_rowid = value_rowids(i);
      if (next_value_rowid == current_value_rowid) {
        if (current_output_index >= 0) {
          ++current_output_column;
          if (current_output_column < output_size) {
            current_output_index += output_index_multiplier;
          } else {
            current_output_index = -1;
          }
        }
      } else {
        current_output_column = 0;
        current_value_rowid = next_value_rowid;

        if (next_value_rowid >= parent_output_index.size()) {
          return errors::InvalidArgument(
              "Got next_value_rowid=", next_value_rowid,
              " which is not less than ", parent_output_index.size());
        }

        current_output_index = parent_output_index[next_value_rowid];
      }
      result->push_back(current_output_index);
    }

    if (result->size() != value_rowids.size()) {
      return errors::InvalidArgument("Invalid row ids.");
    }

    return Status::OK();
  }

  Status CalculateOutputIndex(OpKernelContext* context, int dimension,
                              const vector<INDEX_TYPE>& parent_output_index,
                              INDEX_TYPE output_index_multiplier,
                              INDEX_TYPE output_size,
                              vector<INDEX_TYPE>* result) {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSragged_tensor_to_tensor_opDTcc mht_10(mht_10_v, 521, "", "./tensorflow/core/kernels/ragged_tensor_to_tensor_op.cc", "CalculateOutputIndex");

    const RowPartitionTensor row_partition_tensor =
        GetRowPartitionTensor(context, dimension);
    auto partition_type = GetRowPartitionTypeByDimension(dimension);
    switch (partition_type) {
      case RowPartitionType::VALUE_ROWIDS:
        return CalculateOutputIndexValueRowID(
            row_partition_tensor, parent_output_index, output_index_multiplier,
            output_size, result);
      case RowPartitionType::ROW_SPLITS:
        if (row_partition_tensor.size() - 1 > parent_output_index.size()) {
          return errors::InvalidArgument(
              "Row partition size is greater than output size: ",
              row_partition_tensor.size() - 1, " > ",
              parent_output_index.size());
        }
        return CalculateOutputIndexRowSplit(
            row_partition_tensor, parent_output_index, output_index_multiplier,
            output_size, result);
      default:
        return errors::InvalidArgument(
            "Unsupported partition type:",
            RowPartitionTypeToString(partition_type));
    }
  }

  Status GetFirstDimensionSize(OpKernelContext* context, INDEX_TYPE* result) {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSragged_tensor_to_tensor_opDTcc mht_11(mht_11_v, 550, "", "./tensorflow/core/kernels/ragged_tensor_to_tensor_op.cc", "GetFirstDimensionSize");

    const Tensor first_partition_tensor =
        context->input(kFirstPartitionInputIndex);
    if (row_partition_types_.empty()) {
      return errors::InvalidArgument("No row_partition_types given.");
    }
    const RowPartitionType first_partition_type = row_partition_types_[0];
    switch (first_partition_type) {
      case RowPartitionType::FIRST_DIM_SIZE:
        *result = first_partition_tensor.scalar<INDEX_TYPE>()();
        return Status::OK();
      case RowPartitionType::VALUE_ROWIDS:
        return errors::InvalidArgument(
            "Cannot handle VALUE_ROWIDS in first dimension.");
      case RowPartitionType::ROW_SPLITS:
        *result = first_partition_tensor.shape().dim_size(0) - 1;
        return Status::OK();
      default:
        return errors::InvalidArgument(
            "Cannot handle type ",
            RowPartitionTypeToString(first_partition_type));
    }
  }

  void Compute(OpKernelContext* context) override {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSragged_tensor_to_tensor_opDTcc mht_12(mht_12_v, 577, "", "./tensorflow/core/kernels/ragged_tensor_to_tensor_op.cc", "Compute");

    INDEX_TYPE first_dimension;
    const Tensor first_partition_tensor =
        context->input(kFirstPartitionInputIndex);
    OP_REQUIRES(context, first_partition_tensor.NumElements() > 0,
                errors::InvalidArgument("Invalid first partition input. Tensor "
                                        "requires at least one element."));
    OP_REQUIRES_OK(context, GetFirstDimensionSize(context, &first_dimension));
    vector<INDEX_TYPE> output_size;
    OP_REQUIRES_OK(context,
                   CalculateOutputSize(first_dimension, context, &output_size));
    vector<INDEX_TYPE> multiplier;
    multiplier.resize(ragged_rank_ + 1);

    multiplier[multiplier.size() - 1] = 1;
    for (int i = multiplier.size() - 2; i >= 0; --i) {
      multiplier[i] = multiplier[i + 1] * output_size[i + 1];
    }
    // Full size of the tensor.
    TensorShape output_shape;
    OP_REQUIRES_OK(context,
                   TensorShapeUtils::MakeShape(output_size, &output_shape));
    Tensor* output_tensor = nullptr;

    OP_REQUIRES_OK(context,
                   context->allocate_output(0, output_shape, &output_tensor));
    const INDEX_TYPE full_size = multiplier[0] * output_size[0];
    if (full_size > 0) {
      vector<INDEX_TYPE> output_index, new_output_index;
      int nvals = context->input(kValueInputIndex).shape().dim_size(0);
      output_index.reserve(nvals);
      new_output_index.reserve(nvals);

      CalculateFirstParentOutputIndex(first_dimension, multiplier[0],
                                      output_size[0], &output_index);
      for (int i = 1; i <= ragged_rank_; ++i) {
        OP_REQUIRES_OK(context, CalculateOutputIndex(
                                    context, i - 1, output_index, multiplier[i],
                                    output_size[i], &new_output_index));
        output_index.swap(new_output_index);
        new_output_index.clear();
      }

      SetOutput(context, ragged_rank_, output_index, output_tensor);
    }
  }
  virtual void SetOutput(OpKernelContext* context, int ragged_rank,
                         const vector<INDEX_TYPE>& output_index,
                         Tensor* output_tensor) = 0;

 private:
  vector<RowPartitionType> row_partition_types_;
  int ragged_rank_;
};

template <typename VALUE_TYPE, typename INDEX_TYPE>
void slow_copy_array(VALUE_TYPE* dst, const VALUE_TYPE* src, INDEX_TYPE size) {
  for (INDEX_TYPE index = 0; index < size; ++index) {
    dst[index] = src[index];
  }
}

template <typename VALUE_TYPE, typename INDEX_TYPE>
void copy_array(VALUE_TYPE* dst, const VALUE_TYPE* src, INDEX_TYPE size) {
  memcpy(dst, src, size * sizeof(VALUE_TYPE));
}

template <>
void copy_array<tstring, int64_t>(tstring* dst, const tstring* src,
                                  int64_t size) {
  slow_copy_array(dst, src, size);
}

template <>
void copy_array<tstring, int32>(tstring* dst, const tstring* src,
                                int32_t size) {
  slow_copy_array(dst, src, size);
}

// If we don't specialize for Eigen::half, we get:
// undefined behavior, destination object type 'Eigen::half'
// is not TriviallyCopyable
template <>
void copy_array<Eigen::half, int64_t>(Eigen::half* dst, const Eigen::half* src,
                                      int64_t size) {
  slow_copy_array(dst, src, size);
}

template <>
void copy_array<Eigen::half, int32>(Eigen::half* dst, const Eigen::half* src,
                                    int32_t size) {
  slow_copy_array(dst, src, size);
}

template <typename VALUE_TYPE, typename INDEX_TYPE>
class RaggedTensorToTensorOp : public RaggedTensorToTensorBaseOp<INDEX_TYPE> {
 public:
  explicit RaggedTensorToTensorOp(OpKernelConstruction* context)
      : RaggedTensorToTensorBaseOp<INDEX_TYPE>(context) {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSragged_tensor_to_tensor_opDTcc mht_13(mht_13_v, 678, "", "./tensorflow/core/kernels/ragged_tensor_to_tensor_op.cc", "RaggedTensorToTensorOp");
}

  void SetOutput(OpKernelContext* context, int ragged_rank,
                 const vector<INDEX_TYPE>& output_index,
                 Tensor* output_tensor) override {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSragged_tensor_to_tensor_opDTcc mht_14(mht_14_v, 685, "", "./tensorflow/core/kernels/ragged_tensor_to_tensor_op.cc", "SetOutput");

    // Note: it's ok to use OP_REQUIRES_OK (rather than TF_RETURN_IF_ERROR)
    // in this function, but only because it's the last thing we do before
    // returning from Compute().

    if (output_tensor->NumElements() == 0) return;

    const auto& values_tensor = context->input(kValueInputIndex);
    const VALUE_TYPE* values_base = values_tensor.flat<VALUE_TYPE>().data();
    const auto& default_value_tensor = context->input(kDefaultValueInputIndex);
    VALUE_TYPE* output_base = output_tensor->flat<VALUE_TYPE>().data();

    TensorShape element_shape = output_tensor->shape();
    element_shape.RemoveDimRange(0, ragged_rank + 1);
    int value_element_size = element_shape.num_elements();
    size_t output_index_size = output_index.size();

    // Broadcast the default value to value_element_size.  (We can skip this
    // if default_value_tensor.NumElements() == 1, since we use std::fill
    // when that's true.)
    const VALUE_TYPE* default_value =
        default_value_tensor.flat<VALUE_TYPE>().data();
    Tensor bcast_default;  // Temporary tensor for result of broadcast
    if (default_value_tensor.NumElements() != value_element_size &&
        default_value_tensor.NumElements() != 1) {
      const auto& src_shape = default_value_tensor.shape();
      BCast bcast(BCast::FromShape(src_shape), BCast::FromShape(element_shape),
                  /*fewer_dims_optimization=*/true);
      // Note: bcast should always be valid, since we rejected any incompatible
      // shapes when we called ValidateDefaultValueShape().
      OP_REQUIRES(context, bcast.IsValid(),
                  errors::InvalidArgument("Error broadcasting default_value"));
      OP_REQUIRES_OK(context,
                     context->allocate_temp(default_value_tensor.dtype(),
                                            element_shape, &bcast_default));
      const CPUDevice& device = context->eigen_device<CPUDevice>();
      functor::BroadcastTo<CPUDevice, VALUE_TYPE>()(
          device, context, bcast_default, element_shape, default_value_tensor,
          src_shape, bcast);
      default_value = bcast_default.flat<VALUE_TYPE>().data();
    }

    // Loop through the output_index vector, finding contiguous regions that
    // should be copied.  Once we find the end of a contiguous region, copy it
    // and add any necessary padding (with default_value).
    INDEX_TYPE src_start = 0;  // Start of contiguous region (in values)
    INDEX_TYPE dst_start = 0;  // Destination for contiguous region (in output)
    INDEX_TYPE dst_end = 0;    // Destination for contiguous region (in output)
    for (int src_i = 0; src_i <= output_index_size; ++src_i) {
      // dst_i is the destination where the value at src_i should be copied.
      INDEX_TYPE dst_i = src_i < output_index_size ? output_index[src_i] : -1;

      // If we're still in a contiguous region, then update dst_end go to the
      // next src_i.
      if (dst_i == dst_end) {
        ++dst_end;
        continue;
      }

      // We found the end of contiguous region.  This can be because we found
      // a gap (dst_i > dst_end), or a source value that shouldn't be copied
      // because it's out-of-bounds (dst_i == -1), or the end of the tensor
      // (dst_i = -1).
      if (dst_start < dst_end) {
        // Copy the contiguous region.
        const VALUE_TYPE* src = values_base + src_start * value_element_size;
        VALUE_TYPE* dst = output_base + dst_start * value_element_size;
        INDEX_TYPE nvals = (dst_end - dst_start) * value_element_size;
        copy_array<VALUE_TYPE, INDEX_TYPE>(dst, src, nvals);
      }

      // Add any necessary padding (w/ default_value).
      if (src_i >= output_index_size) {
        // We reached the end of values: pad to the end of output.
        size_t output_size = output_tensor->NumElements();
        dst_i = output_size / value_element_size;
      }
      if (dst_i > dst_end) {
        if (default_value_tensor.NumElements() == 1) {
          std::fill(output_base + dst_end * value_element_size,
                    output_base + dst_i * value_element_size, *default_value);
          dst_end = dst_i;
        } else {
          while (dst_i > dst_end) {
            VALUE_TYPE* dst = output_base + dst_end * value_element_size;
            copy_array<VALUE_TYPE, INDEX_TYPE>(dst, default_value,
                                               value_element_size);
            ++dst_end;
          }
        }
      }

      // Update indices.
      if (dst_i < 0) {
        // src_i should be skipped -- leave it out of the contiguous region.
        src_start = src_i + 1;
        dst_start = dst_end;
      } else {
        // src_i should be copied -- include it in the contiguous region.
        src_start = src_i;
        dst_start = dst_end;
        dst_end = dst_start + 1;
      }
    }
  }
};

#define REGISTER_CPU_KERNEL_INDEX_TYPE(value_type, index_type)       \
  REGISTER_KERNEL_BUILDER(Name("RaggedTensorToTensor")               \
                              .Device(DEVICE_CPU)                    \
                              .TypeConstraint<value_type>("T")       \
                              .TypeConstraint<index_type>("Tindex"), \
                          RaggedTensorToTensorOp<value_type, index_type>);

#define REGISTER_CPU_KERNEL(value_type)                \
  REGISTER_CPU_KERNEL_INDEX_TYPE(value_type, int64_t); \
  REGISTER_CPU_KERNEL_INDEX_TYPE(value_type, tensorflow::int32);

TF_CALL_POD_TYPES(REGISTER_CPU_KERNEL);
TF_CALL_string(REGISTER_CPU_KERNEL);
TF_CALL_QUANTIZED_TYPES(REGISTER_CPU_KERNEL);
TF_CALL_quint16(REGISTER_CPU_KERNEL);
TF_CALL_qint16(REGISTER_CPU_KERNEL);

#undef REGISTER_CPU_KERNEL

}  // namespace
}  // namespace tensorflow
