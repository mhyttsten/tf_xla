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
class MHTracer_DTPStensorflowPScorePSkernelsPSsave_restore_tensorDTcc {
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
   MHTracer_DTPStensorflowPScorePSkernelsPSsave_restore_tensorDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSkernelsPSsave_restore_tensorDTcc() {
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

#include "tensorflow/core/kernels/save_restore_tensor.h"

#include <memory>
#include <numeric>
#include <unordered_map>
#include <utility>
#include <vector>

#include "tensorflow/core/framework/bounds_check.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/lib/core/threadpool.h"
#include "tensorflow/core/lib/gtl/array_slice.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/lib/strings/stringprintf.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/util/tensor_bundle/tensor_bundle.h"
#include "tensorflow/core/util/tensor_slice_reader.h"
#include "tensorflow/core/util/tensor_slice_reader_cache.h"
#include "tensorflow/core/util/tensor_slice_writer.h"

namespace tensorflow {

void SaveTensors(
    OpKernelContext* context,
    checkpoint::TensorSliceWriter::CreateBuilderFunction builder_func,
    bool save_slices) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSsave_restore_tensorDTcc mht_0(mht_0_v, 215, "", "./tensorflow/core/kernels/save_restore_tensor.cc", "SaveTensors");

  const Tensor& filename_t = context->input(0);
  {
    const int64_t size = filename_t.NumElements();
    OP_REQUIRES(
        context, size == 1,
        errors::InvalidArgument(
            "Input 0 (filename) must be a string scalar; got a tensor of ",
            size, "elements"));
  }

  // Path, names, and slices if save_slices is true.
  const int kFixedInputs = save_slices ? 3 : 2;
  const Tensor& tensor_names_t = context->input(1);
  OP_REQUIRES(context,
              FastBoundsCheck(tensor_names_t.NumElements() + kFixedInputs,
                              std::numeric_limits<int>::max()),
              errors::InvalidArgument("Too many inputs to SaveTensors"));
  const int N = static_cast<int>(tensor_names_t.NumElements());
  const tstring* tensor_shapes_and_slices_ptr = nullptr;
  if (save_slices) {
    const Tensor& tensor_shapes_and_slices_t = context->input(2);
    OP_REQUIRES(
        context,
        tensor_shapes_and_slices_t.NumElements() == static_cast<int64_t>(N),
        errors::InvalidArgument("Expected ", N,
                                " elements for the tensor "
                                "shapes and slices but got ",
                                tensor_shapes_and_slices_t.NumElements()));
    tensor_shapes_and_slices_ptr =
        tensor_shapes_and_slices_t.flat<tstring>().data();
  }
  OP_REQUIRES(context, context->num_inputs() == N + kFixedInputs,
              errors::InvalidArgument("Expected totally ", N + kFixedInputs,
                                      " inputs as input #1 (which is a string "
                                      "tensor of saved names) contains ",
                                      N, " names, but received ",
                                      context->num_inputs(), " inputs"));

  VLOG(1) << "About to save tensors to file " << filename_t.flat<tstring>()(0)
          << "...";
  checkpoint::TensorSliceWriter writer(filename_t.flat<tstring>()(0),
                                       std::move(builder_func));

  Status s;
  auto tensor_names_flat = tensor_names_t.flat<tstring>();

  // Process tensors in sorted name order.  This allows us to avoid seeking
  // during restoration in the common case where we are restoring a full
  // checkpoint.
  // RestoreTensorsV2 was changed to sort by file offset, so this sorting isn't
  // strictly necessary anymore. However, restores with TF version <= 2.7 will
  // still benefit.
  std::vector<size_t> sorted_name_idx(tensor_names_flat.size());
  std::iota(sorted_name_idx.begin(), sorted_name_idx.end(), 0);
  std::sort(sorted_name_idx.begin(), sorted_name_idx.end(),
            [&tensor_names_flat](size_t a, size_t b) {
              return tensor_names_flat(a) < tensor_names_flat(b);
            });

  for (const size_t i : sorted_name_idx) {
    const string& name = tensor_names_flat(i);
    const Tensor& input = context->input(i + kFixedInputs);
    TensorShape shape(input.shape());
    TensorSlice slice(input.dims());
    if (save_slices && !tensor_shapes_and_slices_ptr[i].empty()) {
      const tstring& shape_spec = tensor_shapes_and_slices_ptr[i];
      TensorShape slice_shape;
      OP_REQUIRES_OK(context, checkpoint::ParseShapeAndSlice(
                                  shape_spec, &shape, &slice, &slice_shape));
      OP_REQUIRES(context, slice_shape.IsSameSize(input.shape()),
                  errors::InvalidArgument(
                      "Slice in shape_and_slice "
                      "specification does not match the "
                      "shape of the tensor to  save: ",
                      shape_spec, ", tensor: ", input.shape().DebugString()));
    }

#define WRITER_ADD(T)                                           \
  case DataTypeToEnum<T>::value:                                \
    s = writer.Add(name, shape, slice, input.flat<T>().data()); \
    break;

    switch (input.dtype()) {
      TF_CALL_SAVE_RESTORE_TYPES(WRITER_ADD)
      default:
        context->SetStatus(errors::Unimplemented("Saving data type ",
                                                 DataTypeString(input.dtype()),
                                                 " not yet supported"));
        return;
    }
#undef WRITER_ADD
    if (!s.ok()) {
      context->SetStatus(s);
      return;
    }
  }

  s = writer.Finish();
  if (!s.ok()) {
    context->SetStatus(s);
  }
}

void RestoreTensor(OpKernelContext* context,
                   checkpoint::TensorSliceReader::OpenTableFunction open_func,
                   int preferred_shard, bool restore_slice, int restore_index) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSsave_restore_tensorDTcc mht_1(mht_1_v, 324, "", "./tensorflow/core/kernels/save_restore_tensor.cc", "RestoreTensor");

  const Tensor& file_pattern_t = context->input(0);
  {
    const int64_t size = file_pattern_t.NumElements();
    OP_REQUIRES(
        context, size == 1,
        errors::InvalidArgument(
            "Input 0 (file_pattern) must be a string scalar; got a tensor of ",
            size, " elements"));
  }
  const string& file_pattern = file_pattern_t.flat<tstring>()(0);

  const Tensor& tensor_name_t = context->input(1);
  {
    const int64_t size = tensor_name_t.NumElements();
    OP_REQUIRES(context, size > restore_index,
                errors::InvalidArgument(
                    "Input 1 (file_pattern) must be a have at least ",
                    restore_index + 1, " elements"));
  }
  const string& tensor_name = tensor_name_t.flat<tstring>()(restore_index);

  // If we cannot find a cached reader we will allocate our own.
  std::unique_ptr<checkpoint::TensorSliceReader> allocated_reader;

  const checkpoint::TensorSliceReader* reader = nullptr;

  if (context->slice_reader_cache()) {
    reader = context->slice_reader_cache()->GetReader(file_pattern, open_func,
                                                      preferred_shard);
  }
  if (!reader) {
    allocated_reader.reset(new checkpoint::TensorSliceReader(
        file_pattern, open_func, preferred_shard));
    reader = allocated_reader.get();
  }
  OP_REQUIRES_OK(context, CHECK_NOTNULL(reader)->status());

  // Get the shape and type from the save file.
  DataType type;
  TensorShape saved_shape;
  OP_REQUIRES(
      context, reader->HasTensor(tensor_name, &saved_shape, &type),
      errors::NotFound("Tensor name \"", tensor_name,
                       "\" not found in checkpoint files ", file_pattern));
  OP_REQUIRES(
      context, type == context->expected_output_dtype(restore_index),
      errors::InvalidArgument("Expected to restore a tensor of type ",
                              DataTypeString(context->expected_output_dtype(0)),
                              ", got a tensor of type ", DataTypeString(type),
                              " instead: tensor_name = ", tensor_name));

  // Shape of the output and slice to load.
  TensorShape output_shape(saved_shape);
  TensorSlice slice_to_load(saved_shape.dims());
  if (restore_slice) {
    const tstring& shape_spec =
        context->input(2).flat<tstring>()(restore_index);
    if (!shape_spec.empty()) {
      TensorShape parsed_shape;
      OP_REQUIRES_OK(context, checkpoint::ParseShapeAndSlice(
                                  shape_spec, &parsed_shape, &slice_to_load,
                                  &output_shape));
      OP_REQUIRES(
          context, parsed_shape.IsSameSize(saved_shape),
          errors::InvalidArgument(
              "Shape in shape_and_slice spec does not match the shape in the "
              "save file: ",
              parsed_shape.DebugString(),
              ", save file shape: ", saved_shape.DebugString()));
    }
  }

  Tensor* t = nullptr;
  OP_REQUIRES_OK(context,
                 context->allocate_output(restore_index, output_shape, &t));

  if (output_shape.num_elements() == 0) return;

#define READER_COPY(T)                                                \
  case DataTypeToEnum<T>::value:                                      \
    OP_REQUIRES(context,                                              \
                reader->CopySliceData(tensor_name, slice_to_load,     \
                                      t->flat<T>().data()),           \
                errors::InvalidArgument("Error copying slice data")); \
    break;

  switch (type) {
    TF_CALL_SAVE_RESTORE_TYPES(READER_COPY)
    default:
      context->SetStatus(errors::Unimplemented(
          "Restoring data type ", DataTypeString(type), " not yet supported"));
  }
#undef READER_COPY
}

namespace {

// Tensors larger than this threshold will be restored from a thread-pool.
const int64_t kLargeShapeThreshold = 16 << 20;  // 16M

// A restore operation for a single tensor.  Small tensors may be restored
// directly from the op thread to improve read locality.  Large tensors can be
// restored from a thread pool: this requires creating a separate BundleReader
// for each restore.
struct RestoreOp {
  RestoreOp(OpKernelContext* context, int idx, const string& tensor_name,
            const string& shape_and_slice, const string& reader_prefix,
            DataType dtype)
      : context(context),
        idx(idx),
        tensor_name(tensor_name),
        shape_and_slice(shape_and_slice),
        reader_prefix(reader_prefix),
        dtype(dtype) {
   std::vector<std::string> mht_2_v;
   mht_2_v.push_back("tensor_name: \"" + tensor_name + "\"");
   mht_2_v.push_back("shape_and_slice: \"" + shape_and_slice + "\"");
   mht_2_v.push_back("reader_prefix: \"" + reader_prefix + "\"");
   MHTracer_DTPStensorflowPScorePSkernelsPSsave_restore_tensorDTcc mht_2(mht_2_v, 444, "", "./tensorflow/core/kernels/save_restore_tensor.cc", "RestoreOp");
}

  // Move-only. It does not make sense to "run()" a copied RestoreOp.
  RestoreOp(const RestoreOp&) = delete;
  RestoreOp& operator=(const RestoreOp&) = delete;
  RestoreOp(RestoreOp&&) = default;
  RestoreOp& operator=(RestoreOp&&) = default;

  bool should_run_in_pool(BundleReader* reader) const {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSsave_restore_tensorDTcc mht_3(mht_3_v, 455, "", "./tensorflow/core/kernels/save_restore_tensor.cc", "should_run_in_pool");

    TensorShape restored_full_shape;

    // Ignore status here; we'll catch the error later.
    if (!reader->LookupTensorShape(tensor_name, &restored_full_shape).ok()) {
      return false;
    }

    return restored_full_shape.num_elements() > kLargeShapeThreshold;
  }

  // Run this restore operation using a new BundleReader.
  void run_with_new_reader() {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSsave_restore_tensorDTcc mht_4(mht_4_v, 470, "", "./tensorflow/core/kernels/save_restore_tensor.cc", "run_with_new_reader");

    BundleReader reader(Env::Default(), reader_prefix);
    if (!reader.status().ok()) {
      status = reader.status();
      return;
    }

    status = run(&reader);
  }

  Status run(BundleReader* reader) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSsave_restore_tensorDTcc mht_5(mht_5_v, 483, "", "./tensorflow/core/kernels/save_restore_tensor.cc", "run");

    TensorShape restored_full_shape;
    TF_RETURN_IF_ERROR(
        reader->LookupTensorShape(tensor_name, &restored_full_shape));

    VLOG(1) << "Restoring tensor " << idx << " : " << tensor_name << " : "
            << restored_full_shape.num_elements();
    Tensor* restored_tensor;
    if (shape_and_slice.empty()) {
      // Lookup the full tensor.
      TF_RETURN_IF_ERROR(
          context->allocate_output(idx, restored_full_shape, &restored_tensor));
      TF_RETURN_IF_ERROR(reader->Lookup(tensor_name, restored_tensor));
    } else {
      // Lookup the slice.
      TensorShape parsed_full_shape;
      TensorSlice parsed_slice;
      TensorShape parsed_slice_shape;

      TF_RETURN_IF_ERROR(
          checkpoint::ParseShapeAndSlice(shape_and_slice, &parsed_full_shape,
                                         &parsed_slice, &parsed_slice_shape));

      if (!restored_full_shape.IsSameSize(parsed_full_shape)) {
        return errors::InvalidArgument(
            "tensor_name = ", tensor_name, "; shape in shape_and_slice spec ",
            parsed_full_shape.DebugString(),
            " does not match the shape stored in checkpoint: ",
            restored_full_shape.DebugString());
      }
      TF_RETURN_IF_ERROR(
          context->allocate_output(idx, parsed_slice_shape, &restored_tensor));
      TF_RETURN_IF_ERROR(
          reader->LookupSlice(tensor_name, parsed_slice, restored_tensor));
    }
    if (VLOG_IS_ON(5)) {
      if (restored_tensor->dtype() == DT_FLOAT) {
        const float* t_data = restored_tensor->flat<float>().data();
        float min = std::numeric_limits<float>::infinity();
        float max = -std::numeric_limits<float>::infinity();
        double avg = 0.0;
        for (int i = 0; i < restored_tensor->NumElements(); ++i) {
          if (t_data[i] < min) min = t_data[i];
          if (t_data[i] > max) max = t_data[i];
          avg += t_data[i];
        }
        VLOG(5) << " min " << min << " max " << max << " avg "
                << avg / restored_tensor->NumElements() << " total elts "
                << restored_tensor->NumElements();
      }
    }
    VLOG(1) << "Done restoring tensor " << idx << " : " << tensor_name << " : "
            << restored_full_shape.num_elements();
    return Status::OK();
  }

  OpKernelContext* context;
  int idx;
  string tensor_name;
  string shape_and_slice;
  string reader_prefix;
  DataType dtype;

  ::tensorflow::Status status;
};

}  // namespace

Status RestoreTensorsV2(OpKernelContext* context, const Tensor& prefix,
                        const Tensor& tensor_names,
                        const Tensor& shape_and_slices,
                        gtl::ArraySlice<DataType> dtypes) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSsave_restore_tensorDTcc mht_6(mht_6_v, 557, "", "./tensorflow/core/kernels/save_restore_tensor.cc", "RestoreTensorsV2");

  const string& prefix_string = prefix.scalar<tstring>()();

  const auto& tensor_names_flat = tensor_names.flat<tstring>();
  const auto& shape_and_slices_flat = shape_and_slices.flat<tstring>();

  std::vector<RestoreOp> restore_ops;
  restore_ops.reserve(tensor_names_flat.size());
  for (int i = 0; i < tensor_names_flat.size(); ++i) {
    restore_ops.push_back({context, i, tensor_names_flat(i),
                           shape_and_slices_flat(i), prefix_string, dtypes[i]});
  }

  BundleReader default_reader(Env::Default(), prefix_string);
  TF_RETURN_IF_ERROR(default_reader.status());

  TF_RETURN_IF_ERROR(default_reader.SortForSequentialAccess<RestoreOp>(
      restore_ops, [](const RestoreOp& op) { return op.tensor_name; }));

  std::vector<string> mismatched_errors;
  for (const RestoreOp& restore_op : restore_ops) {
    TensorShape restored_full_shape;
    DataType original_dtype;
    TF_RETURN_IF_ERROR(default_reader.LookupDtypeAndShape(
        restore_op.tensor_name, &original_dtype, &restored_full_shape));
    if (restore_op.dtype != original_dtype) {
      string error_msg = strings::StrCat(
          "tensor_name = ", restore_op.tensor_name, "; expected dtype ",
          DataTypeString(restore_op.dtype), " does not equal original dtype ",
          DataTypeString(original_dtype));
      mismatched_errors.emplace_back(error_msg);
    }
  }
  if (!mismatched_errors.empty()) {
    const string error_msg = absl::StrJoin(mismatched_errors, "\n");
    return errors::InvalidArgument(error_msg);
  }

  std::vector<RestoreOp*> pool_restore_ops;
  std::vector<RestoreOp*> direct_restore_ops;
  for (RestoreOp& restore_op : restore_ops) {
    if (restore_op.should_run_in_pool(&default_reader)) {
      pool_restore_ops.push_back(&restore_op);
    } else {
      direct_restore_ops.push_back(&restore_op);
    }
  }

  {
    // Schedule any threaded operations first, skipping thread pool creation if
    // we don't have any expensive operations.
    std::unique_ptr<thread::ThreadPool> reader_pool;
    if (!pool_restore_ops.empty()) {
      reader_pool.reset(
          new thread::ThreadPool(Env::Default(), "restore_tensors", 8));
      for (auto* op : pool_restore_ops) {
        reader_pool->Schedule([op]() { op->run_with_new_reader(); });
      }
    }

    // Read small tensors from the op thread
    for (auto* op : direct_restore_ops) {
      TF_RETURN_IF_ERROR(op->run(&default_reader));
    }
  }

  // Check status of pool ops; this must come after the pool shuts down.
  for (auto* op : pool_restore_ops) {
    TF_RETURN_IF_ERROR(op->status);
  }

  for (const RestoreOp& restore_op : restore_ops) {
    if (restore_op.dtype != context->mutable_output(restore_op.idx)->dtype()) {
      return errors::InvalidArgument(
          "tensor_name = ", restore_op.tensor_name, "; expected dtype ",
          DataTypeString(restore_op.dtype), " does not equal restored dtype ",
          DataTypeString(context->mutable_output(restore_op.idx)->dtype()));
    }
  }

  return Status::OK();
}

}  // namespace tensorflow
