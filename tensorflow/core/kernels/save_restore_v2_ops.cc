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
class MHTracer_DTPStensorflowPScorePSkernelsPSsave_restore_v2_opsDTcc {
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
   MHTracer_DTPStensorflowPScorePSkernelsPSsave_restore_v2_opsDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSkernelsPSsave_restore_v2_opsDTcc() {
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

// See docs in ../ops/io_ops.cc.

#include <string>
#include <vector>

#include "tensorflow/core/framework/bounds_check.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/kernels/checkpoint_callback_manager.h"
#include "tensorflow/core/kernels/save_restore_tensor.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/util/saved_tensor_slice_util.h"
#include "tensorflow/core/util/tensor_bundle/tensor_bundle.h"
#include "tensorflow/core/util/tensor_slice_reader.h"

namespace tensorflow {

namespace {

// Shared validations of the inputs to the SaveV2 and RestoreV2 ops.
void ValidateInputs(bool is_save_op, OpKernelContext* context,
                    const Tensor& prefix, const Tensor& tensor_names,
                    const Tensor& shape_and_slices) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSsave_restore_v2_opsDTcc mht_0(mht_0_v, 214, "", "./tensorflow/core/kernels/save_restore_v2_ops.cc", "ValidateInputs");

  const int kFixedInputs = 3;  // Prefix, tensor names, shape_and_slices.
  const int num_tensors = static_cast<int>(tensor_names.NumElements());
  OP_REQUIRES(
      context, prefix.NumElements() == 1,
      errors::InvalidArgument("Input prefix should have a single element, got ",
                              prefix.NumElements(), " instead."));
  OP_REQUIRES(context,
              TensorShapeUtils::IsVector(tensor_names.shape()) &&
                  TensorShapeUtils::IsVector(shape_and_slices.shape()),
              errors::InvalidArgument(
                  "Input tensor_names and shape_and_slices "
                  "should be an 1-D tensors, got ",
                  tensor_names.shape().DebugString(), " and ",
                  shape_and_slices.shape().DebugString(), " instead."));
  OP_REQUIRES(context,
              tensor_names.NumElements() == shape_and_slices.NumElements(),
              errors::InvalidArgument("tensor_names and shape_and_slices "
                                      "have different number of elements: ",
                                      tensor_names.NumElements(), " vs. ",
                                      shape_and_slices.NumElements()));
  OP_REQUIRES(context,
              FastBoundsCheck(tensor_names.NumElements() + kFixedInputs,
                              std::numeric_limits<int>::max()),
              errors::InvalidArgument("Too many inputs to the op"));
  OP_REQUIRES(
      context, shape_and_slices.NumElements() == num_tensors,
      errors::InvalidArgument("Expected ", num_tensors,
                              " elements in shapes_and_slices, but got ",
                              context->input(2).NumElements()));
  if (is_save_op) {
    OP_REQUIRES(context, context->num_inputs() == num_tensors + kFixedInputs,
                errors::InvalidArgument(
                    "Got ", num_tensors, " tensor names but ",
                    context->num_inputs() - kFixedInputs, " tensors."));
    OP_REQUIRES(context, context->num_inputs() == num_tensors + kFixedInputs,
                errors::InvalidArgument(
                    "Expected a total of ", num_tensors + kFixedInputs,
                    " inputs as input #1 (which is a string "
                    "tensor of saved names) contains ",
                    num_tensors, " names, but received ", context->num_inputs(),
                    " inputs"));
  }
}

}  // namespace

// Saves a list of named tensors using the tensor bundle library.
class SaveV2 : public OpKernel {
 public:
  explicit SaveV2(OpKernelConstruction* context) : OpKernel(context) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSsave_restore_v2_opsDTcc mht_1(mht_1_v, 267, "", "./tensorflow/core/kernels/save_restore_v2_ops.cc", "SaveV2");
}

  void Compute(OpKernelContext* context) override {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSsave_restore_v2_opsDTcc mht_2(mht_2_v, 272, "", "./tensorflow/core/kernels/save_restore_v2_ops.cc", "Compute");

    const Tensor& prefix = context->input(0);
    const Tensor& tensor_names = context->input(1);
    const Tensor& shape_and_slices = context->input(2);
    ValidateInputs(true /* is save op */, context, prefix, tensor_names,
                   shape_and_slices);
    if (!context->status().ok()) return;

    const int kFixedInputs = 3;  // Prefix, tensor names, shape_and_slices.
    const int num_tensors = static_cast<int>(tensor_names.NumElements());
    const string& prefix_string = prefix.scalar<tstring>()();
    const auto& tensor_names_flat = tensor_names.flat<tstring>();
    const auto& shape_and_slices_flat = shape_and_slices.flat<tstring>();

    BundleWriter writer(Env::Default(), prefix_string);
    OP_REQUIRES_OK(context, writer.status());
    VLOG(1) << "BundleWriter, prefix_string: " << prefix_string;

    for (int i = 0; i < num_tensors; ++i) {
      const string& tensor_name = tensor_names_flat(i);
      const Tensor& tensor = context->input(i + kFixedInputs);
      VLOG(2) << "Starting save of " << tensor_name;

      if (!shape_and_slices_flat(i).empty()) {
        const string& shape_spec = shape_and_slices_flat(i);
        TensorShape shape;
        TensorSlice slice(tensor.dims());
        TensorShape slice_shape;

        OP_REQUIRES_OK(context, checkpoint::ParseShapeAndSlice(
                                    shape_spec, &shape, &slice, &slice_shape));
        OP_REQUIRES(context, slice_shape.IsSameSize(tensor.shape()),
                    errors::InvalidArgument("Slice in shape_and_slice "
                                            "specification does not match the "
                                            "shape of the tensor to  save: ",
                                            shape_spec, ", tensor: ",
                                            tensor.shape().DebugString()));

        OP_REQUIRES_OK(context,
                       writer.AddSlice(tensor_name, shape, slice, tensor));
      } else {
        OP_REQUIRES_OK(context, writer.Add(tensor_name, tensor));
      }

      if (VLOG_IS_ON(5)) {
        if (tensor.dtype() == DT_FLOAT) {
          const float* t_data = tensor.flat<float>().data();
          float min = std::numeric_limits<float>::infinity();
          float max = -std::numeric_limits<float>::infinity();
          double avg = 0.0;
          for (int i = 0; i < tensor.NumElements(); ++i) {
            if (t_data[i] < min) min = t_data[i];
            if (t_data[i] > max) max = t_data[i];
            avg += t_data[i];
          }
          VLOG(5) << " min " << min << " max " << max << " avg "
                  << avg / tensor.NumElements() << " total elts "
                  << tensor.NumElements();
        }
      }

      VLOG(2) << "Done save of " << tensor_name;
    }
    OP_REQUIRES_OK(context, writer.Finish());
    VLOG(1) << "Done BundleWriter, prefix_string: " << prefix_string;

    ResourceMgr* resource_manager = context->resource_manager();
    if (resource_manager != nullptr) {
      checkpoint::CheckpointCallbackManager* checkpoint_callback_manager;
      OP_REQUIRES_OK(
          context,
          resource_manager
              ->LookupOrCreate<checkpoint::CheckpointCallbackManager>(
                  resource_manager->default_container(),
                  std::string(
                      checkpoint::kCheckpointCallbackManagerResourceName),
                  &checkpoint_callback_manager,
                  [](checkpoint::CheckpointCallbackManager** out) {
                    *out = new checkpoint::CheckpointCallbackManager();
                    return Status::OK();
                  }));
      checkpoint_callback_manager->Save(prefix_string);
      checkpoint_callback_manager->Unref();
    }
  }
};
REGISTER_KERNEL_BUILDER(Name("SaveV2").Device(DEVICE_CPU), SaveV2);

// Restores a list of named tensors from a tensor bundle (V2 checkpoint format).
class RestoreV2 : public OpKernel {
 public:
  explicit RestoreV2(OpKernelConstruction* context) : OpKernel(context) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSsave_restore_v2_opsDTcc mht_3(mht_3_v, 366, "", "./tensorflow/core/kernels/save_restore_v2_ops.cc", "RestoreV2");

    OP_REQUIRES_OK(context, context->GetAttr("dtypes", &dtypes_));
  }

  void Compute(OpKernelContext* context) override {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSsave_restore_v2_opsDTcc mht_4(mht_4_v, 373, "", "./tensorflow/core/kernels/save_restore_v2_ops.cc", "Compute");

    const Tensor& prefix = context->input(0);
    const Tensor& tensor_names = context->input(1);
    const Tensor& shape_and_slices = context->input(2);
    OP_REQUIRES(context, tensor_names.NumElements() == dtypes_.size(),
                errors::InvalidArgument("Got ", tensor_names.NumElements(),
                                        " tensor names, but ", dtypes_.size(),
                                        " expected dtypes."));
    ValidateInputs(false /* not save op */, context, prefix, tensor_names,
                   shape_and_slices);
    if (!context->status().ok()) return;

    const string& prefix_string = prefix.scalar<tstring>()();

    // Intention: we plan to use the RestoreV2 op as a backward-compatible
    // reader as we upgrade to the V2 format.  This allows transparent upgrade.
    // We here attempt to read a V1 checkpoint, if "prefix_string" does not
    // refer to a V2 checkpoint.
    Env* env = Env::Default();
    std::vector<string> paths;
    if (!env->GetMatchingPaths(MetaFilename(prefix_string), &paths).ok() ||
        paths.empty()) {
      // Cannot find V2's metadata file, so "prefix_string" does not point to a
      // V2 checkpoint.  Invokes the V1 read path instead.
      for (size_t i = 0; i < tensor_names.NumElements(); ++i) {
        RestoreTensor(context, &checkpoint::OpenTableTensorSliceReader,
                      /* preferred_shard */ -1, /* restore_slice */ true,
                      /* restore_index */ i);
        if (!context->status().ok()) {
          return;
        }
      }
      return;
    }
    // If found, invokes the V2 reader.
    OP_REQUIRES_OK(context, RestoreTensorsV2(context, prefix, tensor_names,
                                             shape_and_slices, dtypes_));

    ResourceMgr* resource_manager = context->resource_manager();
    if (resource_manager != nullptr) {
      checkpoint::CheckpointCallbackManager* checkpoint_callback_manager;
      OP_REQUIRES_OK(
          context,
          resource_manager
              ->LookupOrCreate<checkpoint::CheckpointCallbackManager>(
                  resource_manager->default_container(),
                  std::string(
                      checkpoint::kCheckpointCallbackManagerResourceName),
                  &checkpoint_callback_manager,
                  [](checkpoint::CheckpointCallbackManager** out) {
                    *out = new checkpoint::CheckpointCallbackManager();
                    return Status::OK();
                  }));
      checkpoint_callback_manager->Restore(prefix_string);
      checkpoint_callback_manager->Unref();
    }
  }

 private:
  // Expected dtypes of the to-restore tensors.
  std::vector<DataType> dtypes_;
};
REGISTER_KERNEL_BUILDER(Name("RestoreV2").Device(DEVICE_CPU), RestoreV2);

// The final step in saving sharded V2 checkpoints: merges metadata files.
class MergeV2Checkpoints : public OpKernel {
 public:
  explicit MergeV2Checkpoints(OpKernelConstruction* context)
      : OpKernel(context) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSsave_restore_v2_opsDTcc mht_5(mht_5_v, 444, "", "./tensorflow/core/kernels/save_restore_v2_ops.cc", "MergeV2Checkpoints");

    OP_REQUIRES_OK(context,
                   context->GetAttr("delete_old_dirs", &delete_old_dirs_));
  }

  void Compute(OpKernelContext* context) override {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSsave_restore_v2_opsDTcc mht_6(mht_6_v, 452, "", "./tensorflow/core/kernels/save_restore_v2_ops.cc", "Compute");

    const Tensor& checkpoint_prefixes = context->input(0);
    const Tensor& destination_prefix = context->input(1);
    OP_REQUIRES(context,
                TensorShapeUtils::IsVector(checkpoint_prefixes.shape()),
                errors::InvalidArgument(
                    "Input checkpoint_prefixes should be an 1-D tensor, got ",
                    checkpoint_prefixes.shape().DebugString(), " instead."));
    OP_REQUIRES(context, TensorShapeUtils::IsScalar(destination_prefix.shape()),
                errors::InvalidArgument(
                    "Input destination_prefix should be a scalar tensor, got ",
                    destination_prefix.shape().DebugString(), " instead."));

    const gtl::ArraySlice<tstring> input_prefixes =
        gtl::ArraySlice<tstring>(checkpoint_prefixes.flat<tstring>());
    Env* env = Env::Default();
    const string& merged_prefix = destination_prefix.scalar<tstring>()();
    OP_REQUIRES_OK(
        context, tensorflow::MergeBundles(env, input_prefixes, merged_prefix));

    if (delete_old_dirs_) {
      const string merged_dir(io::Dirname(merged_prefix));
      for (const string& input_prefix : input_prefixes) {
        const string dirname(io::Dirname(input_prefix));
        if (dirname == merged_dir) continue;
        Status status = env->DeleteDir(dirname);
        // For sharded save, only the first delete will go through and all
        // others will hit NotFound.  Use vlog to be less verbose.
        if (!status.ok()) VLOG(1) << status;
      }
    }
  }

 private:
  // On merge, whether or not to delete the input (temporary) directories.
  bool delete_old_dirs_;
};
REGISTER_KERNEL_BUILDER(Name("MergeV2Checkpoints").Device(DEVICE_CPU),
                        MergeV2Checkpoints);

}  // namespace tensorflow
