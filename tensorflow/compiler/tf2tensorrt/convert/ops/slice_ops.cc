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
class MHTracer_DTPStensorflowPScompilerPStf2tensorrtPSconvertPSopsPSslice_opsDTcc {
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
   MHTracer_DTPStensorflowPScompilerPStf2tensorrtPSconvertPSopsPSslice_opsDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPStf2tensorrtPSconvertPSopsPSslice_opsDTcc() {
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

/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/tf2tensorrt/convert/ops/slice_ops.h"

#if GOOGLE_CUDA && GOOGLE_TENSORRT

#include <bitset>
#include <vector>

#include "absl/container/inlined_vector.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "absl/types/optional.h"
#include "absl/types/span.h"
#include "tensorflow/compiler/tf2tensorrt/convert/convert_nodes.h"
#include "tensorflow/compiler/tf2tensorrt/convert/ops/layer_utils.h"
#include "tensorflow/compiler/tf2tensorrt/utils/trt_tensor_proxy.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/util/strided_slice_op.h"
#include "third_party/tensorrt/NvInfer.h"

namespace tensorflow {
namespace tensorrt {
namespace convert {

// Adds a set of operations to the network which set the parameters for the
// given "slice_layer" in order to handle dynamic input shape.
Status HandleDynamicStridedSliceInput(
    TRTNetworkBuilder* builder, nvinfer1::ISliceLayer* slice_layer,
    const StridedSliceShapeSpec& strided_slice_spec,
    const absl::InlinedVector<int64, 4>& dynamic_input_size_indices,
    nvinfer1::Dims begin_dims, nvinfer1::Dims stride_dims,
    nvinfer1::Dims end_dims);

Status ConvertStridedSliceHelper(
    OpConverterParams* params, const TRT_TensorOrWeights& input,
    const PartialTensorShape& input_dims, const SliceDims& begin,
    const SliceDims& stride, const SliceDims& end,
    absl::optional<nvinfer1::Dims> final_shape, absl::optional<int> op_instance,
    absl::optional<StridedSliceShapeSpec> strided_slice_spec) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPStf2tensorrtPSconvertPSopsPSslice_opsDTcc mht_0(mht_0_v, 225, "", "./tensorflow/compiler/tf2tensorrt/convert/ops/slice_ops.cc", "ConvertStridedSliceHelper");

  const auto& node_def = params->node_def;

  auto begin_dims = DimsAdapter::Create(begin, params->use_implicit_batch);
  auto stride_dims = DimsAdapter::Create(stride, params->use_implicit_batch);
  auto end_dims = DimsAdapter::Create(end, params->use_implicit_batch);
  TRT_ENSURE_OK(begin_dims);
  TRT_ENSURE_OK(stride_dims);
  TRT_ENSURE_OK(end_dims);

  // For each dimension, gather information about static vs dynamic dimension
  // and slice size.
  nvinfer1::Dims size_dims = begin_dims->AsTrtDims();
  absl::InlinedVector<int64, 4> static_input_size_indices;
  absl::InlinedVector<int64, 4> dynamic_input_size_indices;
  for (int i = 0; i < begin_dims->NumDims(); i++) {
    size_dims.d[i] = (std::abs(end_dims->dim(i) - begin_dims->dim(i)) +
                      std::abs(stride_dims->dim(i)) - 1) /
                     std::abs(stride_dims->dim(i));

    if (input_dims.dim_size(i) < 0) {
      // end_dims and begin_dims do not have valid information yet.
      dynamic_input_size_indices.push_back(i);
    } else {
      static_input_size_indices.push_back(i);
      if (end_dims->dim(i) < begin_dims->dim(i) && stride_dims->dim(i) > 0) {
        return errors::InvalidArgument(
            "\"size\" cannot be negative for StridedSlice");
      }
    }
  }

  if (!dynamic_input_size_indices.empty() && params->use_implicit_batch) {
    return errors::InvalidArgument(
        "In implicit batch mode, dynamic input size is not supported.");
  }

  if (params->validation_only) return Status::OK();

  StatusOr<TRTNetworkBuilder> builder = TRTNetworkBuilder::Create(
      params->converter->network(), params->weight_store);
  TRT_ENSURE_OK(builder);

  // VLOG(2) << "strided slice helper:"
  //         << " begin:" << DebugString(begin_dims)
  //         << "\n stride: " << DebugString(stride_dims)
  //         << "\n end: " << DebugString(end_dims)
  //         << "\n size: " << DebugString(size_dims)
  //         << "\n Dynamic indices: " <<
  //         DebugString(dynamic_input_size_indices)
  //         << "\n Static indices: " << DebugString(static_input_size_indices);
  // Create the slice operation. For dynamic dims, the inputs of the operations
  // may be reassigned later.
  StatusOr<nvinfer1::ISliceLayer*> slice =
      builder->Slice(input.tensor()->trt_tensor(), begin_dims->AsTrtDims(),
                     size_dims, stride_dims->AsTrtDims());
  TRT_ENSURE_PTR_OK(slice);

  // Handle dynamic input shapes.
  if (!dynamic_input_size_indices.empty()) {
    TRT_ENSURE(strided_slice_spec != absl::nullopt);
    TF_RETURN_IF_ERROR(HandleDynamicStridedSliceInput(
        &*builder, *slice, *strided_slice_spec, dynamic_input_size_indices,
        begin_dims->AsTrtDims(), stride_dims->AsTrtDims(),
        end_dims->AsTrtDims()));
  }

  params->converter->SetLayerName(*slice, params->node_def, "slice",
                                  op_instance);
  ITensorProxyPtr tensor = (*slice)->getOutput(0);

  // Reshape for shrink_axis.
  if (final_shape) {
    TF_RETURN_IF_ERROR(PrepareTensorForShape(
        params->converter, TRT_TensorOrWeights(tensor), *final_shape,
        /*validation_only=*/false, &tensor, node_def, op_instance));
  }
  params->outputs->push_back(TRT_TensorOrWeights(tensor));
  return Status::OK();
}

Status HandleDynamicStridedSliceInput(
    TRTNetworkBuilder* builder, nvinfer1::ISliceLayer* slice_layer,
    const StridedSliceShapeSpec& strided_slice_spec,
    const absl::InlinedVector<int64, 4>& dynamic_input_size_indices,
    nvinfer1::Dims begin_dims, nvinfer1::Dims stride_dims,
    nvinfer1::Dims end_dims) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPStf2tensorrtPSconvertPSopsPSslice_opsDTcc mht_1(mht_1_v, 314, "", "./tensorflow/compiler/tf2tensorrt/convert/ops/slice_ops.cc", "HandleDynamicStridedSliceInput");

  TRT_ENSURE(builder);
  TRT_ENSURE(slice_layer);

  nvinfer1::ITensor* input_tensor = slice_layer->getInput(0);
  TRT_ENSURE(input_tensor);

  // For each dynamic input dimension of the input, do some preprocessing based
  // on whether this dimension is set in "begin_mask" or "end_mask" and the sign
  // of the dimension's stride value.
  // When stride is negative:
  //   - If "begin_mask[dynamic_idx]" is set, then we need to adjust the slice
  //     start of dimension[i] to the dynamic size.
  //   - If "end_mask[dynamic_idx]" is set, it suffices to set
  //     end_dims[dynamic_idx] to -1.
  // When stride is positive:
  //   - If "begin_mask[dynamic_idx]" is set, it suffices to set
  //     begin_dims[dynamic_idx] to zero.
  //   - If "end_mask[dynamic_idx]" is set, we need to adjust slice end to the
  //     dynamic size of dimension "dynamic_idx".
  absl::InlinedVector<int64, 4> dynamic_begin_indices;
  absl::InlinedVector<int64, 4> dynamic_end_indices;
  const auto begin_mask = std::bitset<32>(strided_slice_spec.begin_dense_mask);
  const auto end_mask = std::bitset<32>(strided_slice_spec.end_dense_mask);
  for (int i = 0; i < dynamic_input_size_indices.size(); i++) {
    auto dynamic_idx = dynamic_input_size_indices[i];
    if (begin_mask[dynamic_idx]) {
      begin_dims.d[dynamic_idx] = 0;
      if (stride_dims.d[dynamic_idx] < 0) {
        dynamic_begin_indices.push_back(dynamic_idx);
      }
    }
    if (end_mask[dynamic_idx]) {
      end_dims.d[dynamic_idx] = stride_dims.d[dynamic_idx] > 0 ? 0 : -1;
      if (stride_dims.d[dynamic_idx] > 0) {
        dynamic_end_indices.push_back(dynamic_idx);
      }
    }
  }

  // VLOG(2) << " Dynamic begin indices: " << DebugString(dynamic_begin_indices)
  //         << " Dynamic end indices: " << DebugString(dynamic_end_indices);

  // Create ITensors for each of the begin/stride/end constants.
  StatusOr<nvinfer1::IConstantLayer*> begin_const = builder->Constant(
      std::vector<int>(begin_dims.d, begin_dims.d + begin_dims.nbDims));
  TRT_ENSURE_PTR_OK(begin_const);
  nvinfer1::ITensor* begin_tensor = (*begin_const)->getOutput(0);
  StatusOr<nvinfer1::IConstantLayer*> stride_const = builder->Constant(
      std::vector<int>(stride_dims.d, stride_dims.d + stride_dims.nbDims));
  TRT_ENSURE_PTR_OK(stride_const);
  StatusOr<nvinfer1::IConstantLayer*> end_const = builder->Constant(
      std::vector<int>(end_dims.d, end_dims.d + end_dims.nbDims));
  TRT_ENSURE_PTR_OK(end_const);
  nvinfer1::ITensor* end_tensor = (*end_const)->getOutput(0);

  // Make corrections based on the begin_mask/end_mask values.
  if (dynamic_end_indices.size() > 0) {
    StatusOr<nvinfer1::IGatherLayer*> dynamic_end_masked_tensor =
        builder->GetPartialShapeOf(input_tensor, dynamic_end_indices,
                                   /*sub_one=*/false);
    TRT_ENSURE_PTR_OK(dynamic_end_masked_tensor);
    StatusOr<nvinfer1::IElementWiseLayer*> end_corrected =
        builder->Add((*dynamic_end_masked_tensor)->getOutput(0), end_tensor);
    TRT_ENSURE_PTR_OK(end_corrected);
    end_tensor = (*end_corrected)->getOutput(0);
  }
  if (dynamic_begin_indices.size() > 0) {
    StatusOr<nvinfer1::IGatherLayer*> dynamic_begin_masked_tensor =
        builder->GetPartialShapeOf(input_tensor, dynamic_begin_indices,
                                   /*sub_one=*/true);
    TRT_ENSURE_PTR_OK(dynamic_begin_masked_tensor);

    // Add back the original "begin" values for static dimensions.
    StatusOr<nvinfer1::IElementWiseLayer*> begin_corrected = builder->Add(
        (*dynamic_begin_masked_tensor)->getOutput(0), begin_tensor);
    TRT_ENSURE_PTR_OK(begin_corrected);
    begin_tensor = (*begin_corrected)->getOutput(0);
  }

  // Calculate the final size of the slice dynamicaly.
  nvinfer1::ITensor* size_tensor;
  {
    StatusOr<nvinfer1::IElementWiseLayer*> num =
        builder->Sub(end_tensor, begin_tensor);
    TRT_ENSURE_PTR_OK(num);
    StatusOr<nvinfer1::IElementWiseLayer*> ceil_div = builder->AbsCeilDivInt(
        (*num)->getOutput(0), (*stride_const)->getOutput(0));
    TRT_ENSURE_PTR_OK(ceil_div);
    size_tensor = (*ceil_div)->getOutput(0);
  }

  slice_layer->setInput(1, *begin_tensor);
  slice_layer->setInput(2, *size_tensor);
  slice_layer->setInput(3, *(*stride_const)->getOutput(0));

  return Status::OK();
}

}  // namespace convert
}  // namespace tensorrt
}  // namespace tensorflow
#endif  // GOOGLE_CUDA && GOOGLE_TENSORRT
