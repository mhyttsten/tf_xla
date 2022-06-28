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
class MHTracer_DTPStensorflowPScorePSkernelsPSload_and_remap_matrix_opDTcc {
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
   MHTracer_DTPStensorflowPScorePSkernelsPSload_and_remap_matrix_opDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSkernelsPSload_and_remap_matrix_opDTcc() {
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

#include <algorithm>
#include <string>
#include <unordered_map>
#include <vector>

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/kernel_def_builder.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/gtl/map_util.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/util/tensor_bundle/tensor_bundle.h"

namespace tensorflow {

namespace {
// Returning a Status instead of using OP_REQUIRES directly since that doesn't
// seem to work outside the main OpKernel functions.
Status RemapVectorToMap(
    const TTypes<const int64_t>::Vec& remapping, std::vector<bool>* id_present,
    std::unordered_map<int64_t, int64_t>* old_id_to_new_id) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSload_and_remap_matrix_opDTcc mht_0(mht_0_v, 208, "", "./tensorflow/core/kernels/load_and_remap_matrix_op.cc", "RemapVectorToMap");

  id_present->clear();
  id_present->resize(remapping.size(), false);
  for (int i = 0; i < remapping.size(); ++i) {
    const int64_t old_id = remapping(i);
    if (old_id < 0) continue;
    (*id_present)[i] = true;
    if (!gtl::InsertIfNotPresent(old_id_to_new_id, old_id, i)) {
      return errors::Unimplemented(
          strings::StrCat("Old ID ", old_id, " is mapped to both new ID ",
                          old_id_to_new_id->at(old_id), " and ", i,
                          ", which is not supported."));
    }
  }
  return Status::OK();
}
}  // anonymous namespace

// This op loads a rank-2 Tensor (matrix) from a TensorFlow checkpoint (V2) and
// swaps around the rows/columns according to row_remapping/col_remapping.
// "Missing" cells are initialized with values from initializing_values.
class LoadAndRemapMatrixOp : public OpKernel {
 public:
  explicit LoadAndRemapMatrixOp(OpKernelConstruction* context)
      : OpKernel(context) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSload_and_remap_matrix_opDTcc mht_1(mht_1_v, 235, "", "./tensorflow/core/kernels/load_and_remap_matrix_op.cc", "LoadAndRemapMatrixOp");

    OP_REQUIRES_OK(context, context->GetAttr("num_rows", &num_rows_));
    OP_REQUIRES_OK(context, context->GetAttr("num_cols", &num_cols_));
    OP_REQUIRES_OK(
        context, context->GetAttr("max_rows_in_memory", &max_rows_in_memory_));
  }

  void Compute(OpKernelContext* context) override {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSload_and_remap_matrix_opDTcc mht_2(mht_2_v, 245, "", "./tensorflow/core/kernels/load_and_remap_matrix_op.cc", "Compute");

    // Checks what we're remapping and inverts the relevant remapping Tensors to
    // be maps with key = old ID, value = new ID.
    std::unordered_map<int64_t, int64_t> old_row_to_new_row_map;
    std::vector<bool> row_id_present;
    const Tensor* row_remapping_t;
    OP_REQUIRES_OK(context, context->input("row_remapping", &row_remapping_t));
    OP_REQUIRES(
        context, row_remapping_t->dims() == 1,
        errors::InvalidArgument("The `row_remapping` tensor must be 1-D, got "
                                "a tensor of shape ",
                                row_remapping_t->shape().DebugString()));
    const auto row_remapping = row_remapping_t->vec<int64_t>();
    OP_REQUIRES(context, row_remapping.size() == num_rows_,
                errors::InvalidArgument(strings::StrCat(
                    "Size of row_remapping is ", row_remapping.size(),
                    " instead of being equal to num_rows=", num_rows_)));
    OP_REQUIRES_OK(context, RemapVectorToMap(row_remapping, &row_id_present,
                                             &old_row_to_new_row_map));

    // Calculates the min/max old row ID that we need to read, to save us from
    // reading some unnecessary slices of the old tensor.
    int64_t min_old_row = -1;
    int64_t max_old_row = -1;
    for (int i = 0; i < row_remapping.size(); ++i) {
      if (min_old_row < 0 ||
          (row_remapping(i) >= 0 && row_remapping(i) < min_old_row)) {
        min_old_row = row_remapping(i);
      }
      if (max_old_row < 0 ||
          (row_remapping(i) >= 0 && row_remapping(i) > max_old_row)) {
        max_old_row = row_remapping(i);
      }
    }

    // Processes the remapping for columns.
    std::unordered_map<int64_t, int64_t> old_col_to_new_col_map;
    std::vector<bool> col_id_present;
    const Tensor* col_remapping_t;
    OP_REQUIRES_OK(context, context->input("col_remapping", &col_remapping_t));
    const auto col_remapping = col_remapping_t->vec<int64_t>();
    // Note that we always "remap rows", even when the row vocabulary does
    // not change, because partitioning requires a mapping from partitioned
    // Variables to the full checkpoints we load.
    const bool remap_cols = col_remapping.size() > 0;
    if (remap_cols) {
      OP_REQUIRES(
          context, col_remapping.size() == num_cols_,
          errors::InvalidArgument(strings::StrCat(
              "Provided col_remapping, but its size is ", col_remapping.size(),
              " instead of being equal to num_cols=", num_cols_)));
      OP_REQUIRES_OK(context, RemapVectorToMap(col_remapping, &col_id_present,
                                               &old_col_to_new_col_map));
    } else {
      col_id_present.clear();
      col_id_present.resize(num_cols_, true);
    }

    // Processes the checkpoint source and the provided Tensor name.
    const Tensor* ckpt_path_t;
    OP_REQUIRES_OK(context, context->input("ckpt_path", &ckpt_path_t));
    OP_REQUIRES(
        context, ckpt_path_t->NumElements() == 1,
        errors::InvalidArgument("The `ckpt_path` tensor must have exactly one "
                                "element, got tensor of shape ",
                                ckpt_path_t->shape().DebugString()));
    const string& ckpt_path = ckpt_path_t->scalar<tstring>()();
    const Tensor* old_tensor_name_t;
    OP_REQUIRES_OK(context,
                   context->input("old_tensor_name", &old_tensor_name_t));
    const string& old_tensor_name = old_tensor_name_t->scalar<tstring>()();

    LOG(INFO) << "Processing checkpoint : " << ckpt_path;
    BundleReader reader(context->env(), ckpt_path);
    OP_REQUIRES_OK(context, reader.status());

    DataType tensor_type;
    TensorShape tensor_shape;
    OP_REQUIRES_OK(context, reader.LookupDtypeAndShape(
                                old_tensor_name, &tensor_type, &tensor_shape));
    OP_REQUIRES(context, tensor_type == DT_FLOAT,
                errors::InvalidArgument(strings::StrCat(
                    "Tensor ", old_tensor_name, " has invalid type ",
                    DataTypeString(tensor_type), " instead of expected type ",
                    DataTypeString(DT_FLOAT))));
    // This op is limited to loading Tensors of rank 2 (matrices).
    OP_REQUIRES(
        context, tensor_shape.dims() == 2,
        errors::InvalidArgument(strings::StrCat(
            "Tensor ", old_tensor_name, " has shape ",
            tensor_shape.DebugString(), " of invalid rank ",
            tensor_shape.dims(), " instead of expected shape of rank 2.")));

    if (!remap_cols) {
      // TODO(weiho): Consider relaxing this restriction to allow partial column
      // loading (even when no column remapping is specified) if there turns out
      // to be a use case for it.
      OP_REQUIRES(context, num_cols_ == tensor_shape.dim_size(1),
                  errors::InvalidArgument(strings::StrCat(
                      "Tensor ", old_tensor_name, " has shape ",
                      tensor_shape.DebugString(),
                      ", where the size of its 2nd dimension is ",
                      tensor_shape.dim_size(1),
                      " instead of being equal to num_cols=", num_cols_)));
    }

    // Uses TensorSlice to potentially load the old tensor in chunks in case
    // memory usage is a concern.
    std::vector<TensorSlice> tensor_slices;
    TensorSlice slice(tensor_shape.dims());
    if (min_old_row >= 0 && max_old_row >= 0) {
      int64_t row_start = min_old_row;
      // TODO(weiho): Given the list of old row IDs of interest (the keys of
      // old_row_to_new_row_map), we could also try something smarter to
      // find some minimal set of covering ranges for the list of old row IDs
      // such that the size of each range is less than max_rows_in_memory_.
      while (row_start <= max_old_row) {
        const int64_t slice_length =
            max_rows_in_memory_ <= 0
                // If max_rows_in_memory_ <= 0, we just load the entire chunk.
                ? max_old_row - row_start + 1
                : std::min(max_rows_in_memory_, max_old_row - row_start + 1);
        slice.set_start(0, row_start);
        slice.set_length(0, slice_length);
        tensor_slices.push_back(slice);
        row_start += slice_length;
      }
    }

    // Allocates the output matrix.
    Tensor* output_matrix_t = nullptr;
    OP_REQUIRES_OK(context,
                   context->allocate_output("output_matrix",
                                            TensorShape({num_rows_, num_cols_}),
                                            &output_matrix_t));
    auto output_matrix = output_matrix_t->matrix<float>();

    // Iterates through tensor slices and copies over values from the old tensor
    // to the output matrix.
    int64_t row_index = min_old_row;
    int64_t rows_copied = 0;
    Tensor loaded_tensor_t;
    for (const TensorSlice& tensor_slice : tensor_slices) {
      LOG(INFO) << "Loading slice " << tensor_slice.DebugString();
      TensorShape slice_shape;
      OP_REQUIRES_OK(context,
                     tensor_slice.SliceTensorShape(tensor_shape, &slice_shape));
      // Potentially re-allocates the tensor buffer since the last slice may
      // have fewer rows than the other slices.
      if (loaded_tensor_t.shape() != slice_shape) {
        loaded_tensor_t = Tensor(DT_FLOAT, slice_shape);
      }
      OP_REQUIRES_OK(context, reader.LookupSlice(old_tensor_name, tensor_slice,
                                                 &loaded_tensor_t));

      // Iterates through the old loaded tensor slice row-by-row.
      for (int row = 0; row < loaded_tensor_t.dim_size(0); ++row, ++row_index) {
        if (row_index % 500000 == min_old_row) {
          LOG(INFO) << "Processing old row " << row_index;
        }

        // If the old row ID is not found in old_row_to_new_row_map, continue
        // to the next row; otherwise, copy it to the output matrix.
        const int64_t* new_row_ptr =
            gtl::FindOrNull(old_row_to_new_row_map, row_index);
        if (new_row_ptr == nullptr) {
          continue;
        }
        ++rows_copied;
        const int64_t new_row = *new_row_ptr;

        // Copies over the row element-by-element, in case remapping is needed
        // along the column axis.
        const auto& loaded_tensor = loaded_tensor_t.matrix<float>();
        for (int old_col = 0; old_col < loaded_tensor_t.dim_size(1);
             ++old_col) {
          int64_t new_col = old_col;
          if (remap_cols) {
            const int64_t* new_col_ptr =
                gtl::FindOrNull(old_col_to_new_col_map, old_col);
            if (new_col_ptr == nullptr) {
              // Column remapping is specified, but this column is not found in
              // old_col_to_new_col_map, so we leave it uninitialized, to be
              // filled in with initializing_values later.
              continue;
            }
            new_col = *new_col_ptr;
          }

          OP_REQUIRES(context,
                      new_row < num_rows_ && new_col < num_cols_ &&
                          new_row >= 0 && new_col >= 0,
                      errors::Internal(strings::StrCat(
                          "new_row=", new_row, " and new_col=", new_col,
                          " should have been less than num_rows_=", num_rows_,
                          " and num_cols_=", num_cols_,
                          " and non-negative. This should never have happened "
                          "if the code were correct. Please file a bug.")));
          output_matrix(new_row, new_col) = loaded_tensor(row, old_col);
        }
      }
    }
    LOG(INFO) << "Copied " << rows_copied << " rows from old matrix (with "
              << tensor_shape.dim_size(0) << " rows) to new matrix (with "
              << num_rows_ << " rows).";

    // At this point, there are potentially whole rows/columns uninitialized
    // (corresponding to the indices where row_id_present/col_id_present are
    // false). We fill this in cell-by-cell using row_id_present and
    // col_id_present while dequeuing from the initializing_values vector.
    const Tensor* initializing_values_t;
    OP_REQUIRES_OK(
        context, context->input("initializing_values", &initializing_values_t));
    const auto initializing_values = initializing_values_t->flat<float>();
    int64_t initializing_values_index = 0;
    for (int i = 0; i < num_rows_; ++i) {
      for (int j = 0; j < num_cols_; ++j) {
        if (row_id_present[i] && col_id_present[j]) continue;
        OP_REQUIRES(
            context, initializing_values_index < initializing_values.size(),
            errors::InvalidArgument(
                "initializing_values contained ", initializing_values.size(),
                " elements, but more missing values remain."));
        output_matrix(i, j) = initializing_values(initializing_values_index);
        ++initializing_values_index;
      }
    }

    // Checks that we used all the given initializing values.
    OP_REQUIRES(
        context, initializing_values_index == initializing_values.size(),
        errors::InvalidArgument(
            "initializing_values contained ", initializing_values.size(),
            " elements, but only ", initializing_values_index,
            " elements were used to fill in missing values."));
  }

 private:
  int64_t num_rows_;
  int64_t num_cols_;
  int64_t max_rows_in_memory_;
};

REGISTER_KERNEL_BUILDER(Name("LoadAndRemapMatrix").Device(DEVICE_CPU),
                        LoadAndRemapMatrixOp);

}  // namespace tensorflow
