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
class MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPSselectorsPSsimple_selectorsDTcc {
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
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPSselectorsPSsimple_selectorsDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPSselectorsPSsimple_selectorsDTcc() {
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

/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/lite/delegates/gpu/common/selectors/simple_selectors.h"

#include <memory>
#include <set>

#include "absl/memory/memory.h"
#include "tensorflow/lite/delegates/gpu/common/status.h"
#include "tensorflow/lite/delegates/gpu/common/tasks/add.h"
#include "tensorflow/lite/delegates/gpu/common/tasks/concat_xy.h"
#include "tensorflow/lite/delegates/gpu/common/tasks/concat_z.h"
#include "tensorflow/lite/delegates/gpu/common/tasks/depthwise_conv.h"
#include "tensorflow/lite/delegates/gpu/common/tasks/gather.h"
#include "tensorflow/lite/delegates/gpu/common/tasks/lstm.h"
#include "tensorflow/lite/delegates/gpu/common/tasks/max_unpooling.h"
#include "tensorflow/lite/delegates/gpu/common/tasks/padding.h"
#include "tensorflow/lite/delegates/gpu/common/tasks/pooling.h"
#include "tensorflow/lite/delegates/gpu/common/tasks/prelu.h"
#include "tensorflow/lite/delegates/gpu/common/tasks/quantize_and_dequantize.h"
#include "tensorflow/lite/delegates/gpu/common/tasks/reduce.h"
#include "tensorflow/lite/delegates/gpu/common/tasks/relu.h"
#include "tensorflow/lite/delegates/gpu/common/tasks/resampler.h"
#include "tensorflow/lite/delegates/gpu/common/tasks/reshape.h"
#include "tensorflow/lite/delegates/gpu/common/tasks/reshapex4.h"
#include "tensorflow/lite/delegates/gpu/common/tasks/resize.h"
#include "tensorflow/lite/delegates/gpu/common/tasks/softmax.h"
#include "tensorflow/lite/delegates/gpu/common/tasks/softmax1x1.h"
#include "tensorflow/lite/delegates/gpu/common/tasks/space_to_depth.h"
#include "tensorflow/lite/delegates/gpu/common/tasks/split.h"
#include "tensorflow/lite/delegates/gpu/common/tasks/strided_slice.h"
#include "tensorflow/lite/delegates/gpu/common/tasks/tile.h"
#include "tensorflow/lite/delegates/gpu/common/tasks/transpose.h"
#include "tensorflow/lite/delegates/gpu/common/tasks/winograd.h"

namespace tflite {
namespace gpu {

std::unique_ptr<GPUOperation> SelectLSTM(const OperationDef& op_def,
                                         const GpuInfo& gpu_info) {
  return absl::make_unique<GPUOperation>(CreateLSTM(op_def, gpu_info));
}

std::unique_ptr<GPUOperation> SelectReLU(const ReLUAttributes& attr,
                                         const OperationDef& op_def) {
  return absl::make_unique<GPUOperation>(CreateReLU(op_def, attr));
}

std::unique_ptr<GPUOperation> SelectPReLU(const PReLUAttributes& attr,
                                          const GpuInfo& gpu_info,
                                          const OperationDef& op_def) {
  return absl::make_unique<GPUOperation>(CreatePReLU(gpu_info, op_def, attr));
}

std::unique_ptr<GPUOperation> SelectPooling(const Pooling2DAttributes& attr,
                                            const OperationDef& op_def) {
  return absl::make_unique<GPUOperation>(CreatePooling(op_def, attr));
}

std::unique_ptr<GPUOperation> SelectMaxUnpooling(
    const MaxUnpooling2DAttributes& attr, const OperationDef& op_def) {
  return absl::make_unique<GPUOperation>(CreateMaxUnpooling(op_def, attr));
}

void SelectAdd(const OperationDef& op_def, const std::vector<int>& channels,
               int dst_channels, std::unique_ptr<GPUOperation>* ptr) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPSselectorsPSsimple_selectorsDTcc mht_0(mht_0_v, 248, "", "./tensorflow/lite/delegates/gpu/common/selectors/simple_selectors.cc", "SelectAdd");

  GPUOperation operation = CreateAdd(op_def, channels, dst_channels);
  *ptr = absl::make_unique<GPUOperation>(std::move(operation));
}

absl::Status SelectGather(const GatherAttributes& attr,
                          const OperationDef& op_def,
                          std::unique_ptr<GPUOperation>* ptr) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPSselectorsPSsimple_selectorsDTcc mht_1(mht_1_v, 258, "", "./tensorflow/lite/delegates/gpu/common/selectors/simple_selectors.cc", "SelectGather");

  if (attr.axis != Axis::WIDTH) {
    return absl::UnimplementedError(
        "No gather for this axis. Only Width axis supported.");
  }
  GPUOperation operation = CreateGather(op_def, attr);
  *ptr = absl::make_unique<GPUOperation>(std::move(operation));
  return absl::OkStatus();
}

std::unique_ptr<GPUOperation> SelectResampler(const OperationDef& op_def) {
  GPUOperation operation = CreateResampler(op_def);
  return absl::make_unique<GPUOperation>(std::move(operation));
}

absl::Status SelectResize(const Resize2DAttributes& attr,
                          const OperationDef& op_def,
                          std::unique_ptr<GPUOperation>* ptr) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPSselectorsPSsimple_selectorsDTcc mht_2(mht_2_v, 278, "", "./tensorflow/lite/delegates/gpu/common/selectors/simple_selectors.cc", "SelectResize");

  Resize operation = CreateResize(op_def, attr);
  *ptr = absl::make_unique<Resize>(std::move(operation));
  return absl::OkStatus();
}

absl::Status SelectConcat(const ConcatAttributes& attr,
                          const std::vector<int>& channels,
                          const OperationDef& op_def, const GpuInfo& gpu_info,
                          std::unique_ptr<GPUOperation>* ptr) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPSselectorsPSsimple_selectorsDTcc mht_3(mht_3_v, 290, "", "./tensorflow/lite/delegates/gpu/common/selectors/simple_selectors.cc", "SelectConcat");

  switch (attr.axis) {
    case Axis::CHANNELS: {
      GPUOperation operation = CreateConcatZ(op_def, channels, gpu_info);
      *ptr = absl::make_unique<GPUOperation>(std::move(operation));
      return absl::OkStatus();
    }
    case Axis::BATCH:
    case Axis::DEPTH:
    case Axis::HEIGHT:
    case Axis::WIDTH: {
      GPUOperation operation = CreateConcatXY(op_def, attr);
      *ptr = absl::make_unique<GPUOperation>(std::move(operation));
      return absl::OkStatus();
    }
    default:
      return absl::UnimplementedError("No concat for this axis.");
  }
}

std::unique_ptr<GPUOperation> SelectDWConvolutionDynamicWeights(
    const DepthwiseConvolution2DAttributes& attr, const GpuInfo& gpu_info,
    const OperationDef& op_def) {
  return absl::make_unique<GPUOperation>(
      CreateDepthwiseConvolution2DDynamicWeights(gpu_info, op_def, attr));
}

void SelectReshape(int src_channels, int dst_channels,
                   const OperationDef& op_def,
                   std::unique_ptr<GPUOperation>* ptr) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPSselectorsPSsimple_selectorsDTcc mht_4(mht_4_v, 322, "", "./tensorflow/lite/delegates/gpu/common/selectors/simple_selectors.cc", "SelectReshape");

  if (src_channels % 4 == 0 && dst_channels % 4 == 0) {
    GPUOperation operation = CreateReshapex4(op_def);
    *ptr = absl::make_unique<GPUOperation>(std::move(operation));
  } else {
    GPUOperation operation = CreateReshape(op_def);
    *ptr = absl::make_unique<GPUOperation>(std::move(operation));
  }
}

void SelectSpaceToDepth(const SpaceToDepthAttributes& attr,
                        const OperationDef& op_def,
                        std::unique_ptr<GPUOperation>* ptr) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPSselectorsPSsimple_selectorsDTcc mht_5(mht_5_v, 337, "", "./tensorflow/lite/delegates/gpu/common/selectors/simple_selectors.cc", "SelectSpaceToDepth");

  GPUOperation operation = CreateSpaceToDepth(op_def, attr);
  *ptr = absl::make_unique<GPUOperation>(std::move(operation));
}

void SelectDepthToSpace(const SpaceToDepthAttributes& attr,
                        const OperationDef& op_def,
                        std::unique_ptr<GPUOperation>* ptr) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPSselectorsPSsimple_selectorsDTcc mht_6(mht_6_v, 347, "", "./tensorflow/lite/delegates/gpu/common/selectors/simple_selectors.cc", "SelectDepthToSpace");

  GPUOperation operation = CreateDepthToSpace(op_def, attr);
  *ptr = absl::make_unique<GPUOperation>(std::move(operation));
}

void SelectSplit(const SplitAttributes& attr, const OperationDef& op_def,
                 std::unique_ptr<GPUOperation>* ptr) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPSselectorsPSsimple_selectorsDTcc mht_7(mht_7_v, 356, "", "./tensorflow/lite/delegates/gpu/common/selectors/simple_selectors.cc", "SelectSplit");

  Split operation = CreateSplit(op_def, attr);
  *ptr = absl::make_unique<Split>(std::move(operation));
}

void SelectPadding(const PadAttributes& attr, const OperationDef& op_def,
                   std::unique_ptr<GPUOperation>* ptr) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPSselectorsPSsimple_selectorsDTcc mht_8(mht_8_v, 365, "", "./tensorflow/lite/delegates/gpu/common/selectors/simple_selectors.cc", "SelectPadding");

  GPUOperation operation = CreatePadding(op_def, attr);
  *ptr = absl::make_unique<GPUOperation>(std::move(operation));
}

void SelectStridedSlice(const SliceAttributes& attr, const OperationDef& op_def,
                        std::unique_ptr<GPUOperation>* ptr) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPSselectorsPSsimple_selectorsDTcc mht_9(mht_9_v, 374, "", "./tensorflow/lite/delegates/gpu/common/selectors/simple_selectors.cc", "SelectStridedSlice");

  StridedSlice operation = CreateStridedSlice(op_def, attr);
  *ptr = absl::make_unique<StridedSlice>(std::move(operation));
}

std::unique_ptr<GPUOperation> SelectReduce(const std::set<Axis>& axis_to_reduce,
                                           const BHWC& src_shape,
                                           OperationType op_type,
                                           const OperationDef& op_def,
                                           const GpuInfo& gpu_info) {
  return absl::make_unique<Reduce>(
      CreateReduce(axis_to_reduce, src_shape, op_type, op_def, gpu_info));
}

void SelectSoftmax(const BHWC& shape, const OperationDef& op_def,
                   std::unique_ptr<GPUOperation>* ptr) {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPSselectorsPSsimple_selectorsDTcc mht_10(mht_10_v, 392, "", "./tensorflow/lite/delegates/gpu/common/selectors/simple_selectors.cc", "SelectSoftmax");

  if (shape.w == 1 && shape.h == 1) {
    Softmax1x1 operation = CreateSoftmax1x1(op_def);
    *ptr = absl::make_unique<Softmax1x1>(std::move(operation));
  } else {
    GPUOperation operation = CreateSoftmax(op_def);
    *ptr = absl::make_unique<GPUOperation>(std::move(operation));
  }
}

std::unique_ptr<GPUOperation> SelectTile(const OperationDef& op_def,
                                         const BHWC& src_shape) {
  return absl::make_unique<GPUOperation>(CreateTile(op_def, src_shape.c));
}

void SelectTranspose(const TransposeAttributes& attr,
                     const OperationDef& op_def,
                     std::unique_ptr<GPUOperation>* ptr) {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPSselectorsPSsimple_selectorsDTcc mht_11(mht_11_v, 412, "", "./tensorflow/lite/delegates/gpu/common/selectors/simple_selectors.cc", "SelectTranspose");

  GPUOperation operation = CreateTranspose(op_def, attr);
  *ptr = absl::make_unique<GPUOperation>(std::move(operation));
}

std::unique_ptr<GPUOperation> SelectWinograd4x4To36(
    const GpuInfo& gpu_info, const Padding2D& padding,
    const OperationDef& op_def) {
  if (gpu_info.IsApple() || gpu_info.IsAMD()) {
    Winograd4x4To36 operation = CreateWinograd4x4To36(op_def, padding);
    return absl::make_unique<Winograd4x4To36>(std::move(operation));
  }
  return absl::make_unique<Winograd4x4To36TileX6>(
      CreateWinograd4x4To36TileX6(gpu_info, op_def, padding));
}

std::unique_ptr<GPUOperation> SelectWinograd36To4x4(
    const GpuInfo& gpu_info, const OperationDef& op_def,
    const tflite::gpu::Tensor<Linear, DataType::FLOAT32>& biases) {
  if (gpu_info.IsApple() || gpu_info.IsAMD()) {
    Winograd36To4x4 operation = CreateWinograd36To4x4(op_def, biases);
    return absl::make_unique<Winograd36To4x4>(std::move(operation));
  }
  return absl::make_unique<Winograd36To4x4Tile4x1>(
      CreateWinograd36To4x4Tile4x1(gpu_info, op_def, biases));
}

std::unique_ptr<GPUOperation> SelectQuantizeAndDequantize(
    const QuantizeAndDequantizeAttributes& attr, const OperationDef& op_def) {
  return absl::make_unique<GPUOperation>(
      CreateQuantizeAndDequantize(op_def, attr));
}

}  // namespace gpu
}  // namespace tflite
