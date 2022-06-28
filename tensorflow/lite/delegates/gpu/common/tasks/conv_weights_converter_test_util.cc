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
class MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStasksPSconv_weights_converter_test_utilDTcc {
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
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStasksPSconv_weights_converter_test_utilDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStasksPSconv_weights_converter_test_utilDTcc() {
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

#include "tensorflow/lite/delegates/gpu/common/tasks/conv_weights_converter_test_util.h"

#include <vector>

#include "tensorflow/lite/delegates/gpu/common/operations.h"
#include "tensorflow/lite/delegates/gpu/common/status.h"
#include "tensorflow/lite/delegates/gpu/common/task/testing_util.h"
#include "tensorflow/lite/delegates/gpu/common/task/weights_conversion.h"
#include "tensorflow/lite/delegates/gpu/common/tasks/conv_weights_converter.h"

namespace tflite {
namespace gpu {
namespace {
absl::Status ConvolutionWeightsConverterTest(
    const Tensor<OHWI, DataType::FLOAT32>& weights,
    const WeightsDescription& weight_desc, TestExecutionEnvironment* env,
    const OperationDef& op_def) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStasksPSconv_weights_converter_test_utilDTcc mht_0(mht_0_v, 201, "", "./tensorflow/lite/delegates/gpu/common/tasks/conv_weights_converter_test_util.cc", "ConvolutionWeightsConverterTest");

  // reinterpreting weights in OHWI as tensor in BHWC
  TensorFloat32 src_tensor;
  auto src_shape =
      BHWC(weights.shape.o, weights.shape.h, weights.shape.w, weights.shape.i);
  src_tensor.shape = src_shape;
  src_tensor.data.resize(src_shape.DimensionsProduct(), 2.0);
  for (int o = 0; o < weights.shape.o; ++o) {
    for (int y = 0; y < weights.shape.h; ++y) {
      for (int x = 0; x < weights.shape.w; ++x) {
        for (int i = 0; i < weights.shape.i; ++i) {
          const int f_index = weights.shape.LinearIndex({o, y, x, i});
          const int s_index = src_shape.LinearIndex({o, y, x, i});
          src_tensor.data[s_index] = weights.data[f_index];
        }
      }
    }
  }

  WeightsDescription weight_desc_copy = weight_desc;
  weight_desc_copy.type = DataType::FLOAT32;
  const int flt_count =
      GetTotalElementsCountForLayout(weight_desc_copy, weights.shape);
  DataType weights_type = DataType::FLOAT32;

  std::vector<uint8_t> weights_data(flt_count * SizeOf(weights_type));
  RearrangeWeights(weights, weight_desc_copy, absl::MakeSpan(weights_data));

  std::vector<TensorFloat32> dst_tensors;
  if (weight_desc_copy.layout ==
          WeightsLayout::k2DX4I4YIsSpatialIAndXIsOOGroupO4 ||
      weight_desc_copy.layout ==
          WeightsLayout::k2DX4O4YIsSpatialIAndXIsOOGroupI4) {
    dst_tensors.resize(4);
    const int dst_depth = AlignByN(DivideRoundUp(weights.shape.o, 4),
                                   weight_desc_copy.output_group_size);
    const int src_depth = DivideRoundUp(weights.shape.i, 4);
    const int kernel_x = weights.shape.w;
    const int kernel_y = weights.shape.h;
    int texture_width = dst_depth;
    int texture_height = src_depth * kernel_x * kernel_y;
    int sub_size = SizeOf(weights_type) * 4 * texture_width * texture_height;
    for (int i = 0; i < 4; ++i) {
      dst_tensors[i].shape = BHWC(1, texture_height, texture_width, 4);
      dst_tensors[i].data.resize(4 * texture_width * texture_height);
      memcpy(dst_tensors[i].data.data(), weights_data.data() + sub_size * i,
             sub_size);
    }
  } else {
    dst_tensors.resize(1);
    dst_tensors[0].shape = BHWC(1, 1, 1, flt_count);
    dst_tensors[0].data.resize(flt_count);
    memcpy(dst_tensors[0].data.data(), weights_data.data(),
           flt_count * SizeOf(weights_type));
  }

  std::vector<TensorFloat32> dst_tensors_gpu(dst_tensors.size());
  std::vector<TensorFloat32*> dst_ptrs;
  std::vector<BHWC> dst_shapes;
  for (int i = 0; i < dst_tensors.size(); ++i) {
    dst_shapes.push_back(dst_tensors[i].shape);
    dst_ptrs.push_back(&dst_tensors_gpu[i]);
  }

  auto converter = ConverterToConvWeights(op_def, weight_desc);
  RETURN_IF_ERROR(env->ExecuteGPUOperation(
      {src_tensor},
      absl::make_unique<ConverterToConvWeights>(std::move(converter)),
      dst_shapes, dst_ptrs));
  for (int i = 0; i < dst_tensors.size(); ++i) {
    RETURN_IF_ERROR(
        PointWiseNear(dst_tensors[i].data, dst_tensors_gpu[i].data, 0.0f));
  }
  return absl::OkStatus();
}

}  // namespace

absl::Status ConverterToConvWeights1x1OutX4Test(TestExecutionEnvironment* env) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStasksPSconv_weights_converter_test_utilDTcc mht_1(mht_1_v, 282, "", "./tensorflow/lite/delegates/gpu/common/tasks/conv_weights_converter_test_util.cc", "ConverterToConvWeights1x1OutX4Test");

  const int kSrcChannels = 8;
  const int kDstChannels = 32;
  auto weights_shape = OHWI(kDstChannels, 1, 1, kSrcChannels);
  WeightsDescription conv_weight_desc;
  conv_weight_desc.output_group_size = 4;

  Tensor<OHWI, DataType::FLOAT32> weights;
  weights.shape = weights_shape;
  weights.data.resize(weights_shape.DimensionsProduct());
  for (int i = 0; i < weights.data.size(); ++i) {
    weights.data[i] = half(static_cast<float>(i));
  }

  for (auto precision : env->GetSupportedPrecisions()) {
    auto data_type = DeduceDataTypeFromPrecision(precision);
    for (auto storage : env->GetSupportedStorages(data_type)) {
      for (auto weights_layout : {WeightsLayout::kOSpatialIOGroupI4O4,
                                  WeightsLayout::kOSpatialIOGroupO4I4}) {
        conv_weight_desc.layout = weights_layout;
        OperationDef op_def;
        op_def.precision = precision;
        op_def.src_tensors.push_back({data_type, storage, Layout::BHWC});
        op_def.dst_tensors.push_back(
            {data_type, TensorStorageType::BUFFER, Layout::UNKNOWN});
        RETURN_IF_ERROR(ConvolutionWeightsConverterTest(
            weights, conv_weight_desc, env, op_def));
      }
    }
  }
  return absl::OkStatus();
}

absl::Status ConverterToConvWeights1x1OutX4UnalignedTest(
    TestExecutionEnvironment* env) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStasksPSconv_weights_converter_test_utilDTcc mht_2(mht_2_v, 319, "", "./tensorflow/lite/delegates/gpu/common/tasks/conv_weights_converter_test_util.cc", "ConverterToConvWeights1x1OutX4UnalignedTest");

  const int kSrcChannels = 8;
  const int kDstChannels = 17;
  auto weights_shape = OHWI(kDstChannels, 1, 1, kSrcChannels);
  WeightsDescription conv_weight_desc;
  conv_weight_desc.output_group_size = 4;

  Tensor<OHWI, DataType::FLOAT32> weights;
  weights.shape = weights_shape;
  weights.data.resize(weights_shape.DimensionsProduct());
  for (int i = 0; i < weights.data.size(); ++i) {
    weights.data[i] = half(static_cast<float>(i));
  }

  for (auto precision : env->GetSupportedPrecisions()) {
    auto data_type = DeduceDataTypeFromPrecision(precision);
    for (auto storage : env->GetSupportedStorages(data_type)) {
      for (auto weights_layout : {WeightsLayout::kOSpatialIOGroupI4O4,
                                  WeightsLayout::kOSpatialIOGroupO4I4}) {
        conv_weight_desc.layout = weights_layout;
        OperationDef op_def;
        op_def.precision = precision;
        op_def.src_tensors.push_back({data_type, storage, Layout::BHWC});
        op_def.dst_tensors.push_back(
            {data_type, TensorStorageType::BUFFER, Layout::UNKNOWN});
        RETURN_IF_ERROR(ConvolutionWeightsConverterTest(
            weights, conv_weight_desc, env, op_def));
      }
    }
  }
  return absl::OkStatus();
}

absl::Status ConverterToConvWeights1x1OutX2Test(TestExecutionEnvironment* env) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStasksPSconv_weights_converter_test_utilDTcc mht_3(mht_3_v, 355, "", "./tensorflow/lite/delegates/gpu/common/tasks/conv_weights_converter_test_util.cc", "ConverterToConvWeights1x1OutX2Test");

  const int kSrcChannels = 7;
  const int kDstChannels = 37;
  auto weights_shape = OHWI(kDstChannels, 1, 1, kSrcChannels);
  WeightsDescription conv_weight_desc;
  conv_weight_desc.output_group_size = 2;

  Tensor<OHWI, DataType::FLOAT32> weights;
  weights.shape = weights_shape;
  weights.data.resize(weights_shape.DimensionsProduct());
  for (int i = 0; i < weights.data.size(); ++i) {
    weights.data[i] = half(static_cast<float>(i));
  }

  for (auto precision : env->GetSupportedPrecisions()) {
    auto data_type = DeduceDataTypeFromPrecision(precision);
    for (auto storage : env->GetSupportedStorages(data_type)) {
      for (auto weights_layout : {WeightsLayout::kOSpatialIOGroupI4O4,
                                  WeightsLayout::kOSpatialIOGroupO4I4}) {
        conv_weight_desc.layout = weights_layout;
        OperationDef op_def;
        op_def.precision = precision;
        op_def.src_tensors.push_back({data_type, storage, Layout::BHWC});
        op_def.dst_tensors.push_back(
            {data_type, TensorStorageType::BUFFER, Layout::UNKNOWN});
        RETURN_IF_ERROR(ConvolutionWeightsConverterTest(
            weights, conv_weight_desc, env, op_def));
      }
    }
  }
  return absl::OkStatus();
}

absl::Status ConverterToConvWeightsOutX2Test(TestExecutionEnvironment* env) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStasksPSconv_weights_converter_test_utilDTcc mht_4(mht_4_v, 391, "", "./tensorflow/lite/delegates/gpu/common/tasks/conv_weights_converter_test_util.cc", "ConverterToConvWeightsOutX2Test");

  const int kSrcChannels = 8;
  const int kDstChannels = 38;
  auto weights_shape = OHWI(kDstChannels, 3, 4, kSrcChannels);
  WeightsDescription conv_weight_desc;
  conv_weight_desc.output_group_size = 2;

  Tensor<OHWI, DataType::FLOAT32> weights;
  weights.shape = weights_shape;
  weights.data.resize(weights_shape.DimensionsProduct());
  for (int i = 0; i < weights.data.size(); ++i) {
    weights.data[i] = half(static_cast<float>(i));
  }

  for (auto precision : env->GetSupportedPrecisions()) {
    auto data_type = DeduceDataTypeFromPrecision(precision);
    for (auto storage : env->GetSupportedStorages(data_type)) {
      for (auto weights_layout : {WeightsLayout::kOSpatialIOGroupI4O4,
                                  WeightsLayout::kOSpatialIOGroupO4I4}) {
        conv_weight_desc.layout = weights_layout;
        OperationDef op_def;
        op_def.precision = precision;
        op_def.src_tensors.push_back({data_type, storage, Layout::BHWC});
        op_def.dst_tensors.push_back(
            {data_type, TensorStorageType::BUFFER, Layout::UNKNOWN});
        RETURN_IF_ERROR(ConvolutionWeightsConverterTest(
            weights, conv_weight_desc, env, op_def));
      }
    }
  }
  return absl::OkStatus();
}

absl::Status ConverterToConvTransposedWeights4x4Test(
    TestExecutionEnvironment* env) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStasksPSconv_weights_converter_test_utilDTcc mht_5(mht_5_v, 428, "", "./tensorflow/lite/delegates/gpu/common/tasks/conv_weights_converter_test_util.cc", "ConverterToConvTransposedWeights4x4Test");

  const int kSrcChannels = 7;
  const int kDstChannels = 11;
  auto weights_shape = OHWI(kDstChannels, 4, 4, kSrcChannels);
  WeightsDescription weight_desc;
  weight_desc.spatial_remap = {10, 11, 14, 15, 8, 9, 12, 13,
                               2,  3,  6,  7,  0, 1, 4,  5};

  Tensor<OHWI, DataType::FLOAT32> weights;
  weights.shape = weights_shape;
  weights.data.resize(weights_shape.DimensionsProduct());
  for (int i = 0; i < weights.data.size(); ++i) {
    weights.data[i] = half(static_cast<float>(i));
  }

  for (auto precision : env->GetSupportedPrecisions()) {
    auto data_type = DeduceDataTypeFromPrecision(precision);
    for (auto storage : env->GetSupportedStorages(data_type)) {
      for (auto weights_layout : {WeightsLayout::kOICustomSpatialI4O4,
                                  WeightsLayout::kOICustomSpatialO4I4}) {
        weight_desc.layout = weights_layout;
        OperationDef op_def;
        op_def.precision = precision;
        op_def.src_tensors.push_back({data_type, storage, Layout::BHWC});
        op_def.dst_tensors.push_back(
            {data_type, TensorStorageType::BUFFER, Layout::UNKNOWN});
        RETURN_IF_ERROR(
            ConvolutionWeightsConverterTest(weights, weight_desc, env, op_def));
      }
    }
  }
  return absl::OkStatus();
}

absl::Status ConverterToConvWeights4xTexturesTest(
    TestExecutionEnvironment* env) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStasksPSconv_weights_converter_test_utilDTcc mht_6(mht_6_v, 466, "", "./tensorflow/lite/delegates/gpu/common/tasks/conv_weights_converter_test_util.cc", "ConverterToConvWeights4xTexturesTest");

  const int src_channels = 9;
  const int dst_channels = 17;
  auto weights_shape = OHWI(dst_channels, 1, 1, src_channels);
  WeightsDescription conv_weight_desc;
  conv_weight_desc.output_group_size = 4;

  Tensor<OHWI, DataType::FLOAT32> weights;
  weights.shape = weights_shape;
  weights.data.resize(weights_shape.DimensionsProduct());
  for (int i = 0; i < weights.data.size(); ++i) {
    weights.data[i] = half(static_cast<float>(i));
  }

  for (auto precision : env->GetSupportedPrecisions()) {
    auto data_type = DeduceDataTypeFromPrecision(precision);
    for (auto storage : env->GetSupportedStorages(data_type)) {
      for (auto weights_layout :
           {WeightsLayout::k2DX4I4YIsSpatialIAndXIsOOGroupO4,
            WeightsLayout::k2DX4O4YIsSpatialIAndXIsOOGroupI4}) {
        conv_weight_desc.layout = weights_layout;
        OperationDef op_def;
        op_def.precision = precision;
        op_def.src_tensors.push_back({data_type, storage, Layout::BHWC});
        op_def.dst_tensors.push_back(
            {data_type, TensorStorageType::TEXTURE_2D, Layout::HWC});
        op_def.dst_tensors.push_back(
            {data_type, TensorStorageType::TEXTURE_2D, Layout::HWC});
        op_def.dst_tensors.push_back(
            {data_type, TensorStorageType::TEXTURE_2D, Layout::HWC});
        op_def.dst_tensors.push_back(
            {data_type, TensorStorageType::TEXTURE_2D, Layout::HWC});
        RETURN_IF_ERROR(ConvolutionWeightsConverterTest(
            weights, conv_weight_desc, env, op_def));
      }
    }
  }
  return absl::OkStatus();
}

}  // namespace gpu
}  // namespace tflite
