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
class MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStasksPSstrided_sliceDTcc {
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
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStasksPSstrided_sliceDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStasksPSstrided_sliceDTcc() {
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

#include "tensorflow/lite/delegates/gpu/common/tasks/strided_slice.h"

#include <string>

#include "tensorflow/lite/delegates/gpu/common/task/work_group_picking.h"

namespace tflite {
namespace gpu {

namespace {
bool Is4Aligned(const SliceAttributes& attr) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStasksPSstrided_sliceDTcc mht_0(mht_0_v, 195, "", "./tensorflow/lite/delegates/gpu/common/tasks/strided_slice.cc", "Is4Aligned");

  return attr.strides.c == 1 && attr.starts.c % 4 == 0;
}

int4 GetOffset(const SliceAttributes& attr, int src_width, int src_height,
               int src_channels, int src_batch) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStasksPSstrided_sliceDTcc mht_1(mht_1_v, 203, "", "./tensorflow/lite/delegates/gpu/common/tasks/strided_slice.cc", "GetOffset");

  int4 offset;
  if (attr.strides.w > 0) {
    offset.x = attr.starts.w;
  } else {
    if (attr.ends.w > 0) {
      offset.x = attr.ends.w;
    } else {
      offset.x = src_width + attr.ends.w;
    }
  }
  if (attr.strides.h > 0) {
    offset.y = attr.starts.h;
  } else {
    if (attr.ends.h > 0) {
      offset.y = attr.ends.h;
    } else {
      offset.y = src_height + attr.ends.h;
    }
  }
  if (attr.strides.c > 0) {
    offset.z = attr.starts.c;
  } else {
    if (attr.ends.c > 0) {
      offset.z = attr.ends.c;
    } else {
      offset.z = src_channels + attr.ends.c;
    }
  }
  if (Is4Aligned(attr)) {
    offset.z /= 4;
  }
  if (attr.strides.b > 0) {
    offset.w = attr.starts.b;
  } else {
    if (attr.ends.b > 0) {
      offset.w = attr.ends.b;
    } else {
      offset.w = src_batch + attr.ends.b;
    }
  }
  return offset;
}

}  // namespace

StridedSlice::StridedSlice(const OperationDef& definition,
                           const SliceAttributes& attr)
    : GPUOperation(definition), attributes_(attr) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStasksPSstrided_sliceDTcc mht_2(mht_2_v, 254, "", "./tensorflow/lite/delegates/gpu/common/tasks/strided_slice.cc", "StridedSlice::StridedSlice");

  work_group_size_ = int3(8, 4, 1);
  code_ = GetStridedSliceCode(definition_, Is4Aligned(attributes_));
}

StridedSlice::StridedSlice(StridedSlice&& operation)
    : GPUOperation(std::move(operation)), attributes_(operation.attributes_) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStasksPSstrided_sliceDTcc mht_3(mht_3_v, 263, "", "./tensorflow/lite/delegates/gpu/common/tasks/strided_slice.cc", "StridedSlice::StridedSlice");
}

StridedSlice& StridedSlice::operator=(StridedSlice&& operation) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStasksPSstrided_sliceDTcc mht_4(mht_4_v, 268, "", "./tensorflow/lite/delegates/gpu/common/tasks/strided_slice.cc", "=");

  if (this != &operation) {
    attributes_ = operation.attributes_;
    GPUOperation::operator=(std::move(operation));
  }
  return *this;
}

std::string StridedSlice::GetStridedSliceCode(const OperationDef& op_def,
                                              bool alignedx4) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStasksPSstrided_sliceDTcc mht_5(mht_5_v, 280, "", "./tensorflow/lite/delegates/gpu/common/tasks/strided_slice.cc", "StridedSlice::GetStridedSliceCode");

  AddSrcTensor("src_tensor", op_def.src_tensors[0]);
  AddDstTensor("dst_tensor", op_def.dst_tensors[0]);
  args_.AddInt("offset_x");
  args_.AddInt("offset_y");
  args_.AddInt("offset_z");
  args_.AddInt("offset_b");
  args_.AddInt("stride_x");
  args_.AddInt("stride_y");
  args_.AddInt("stride_z");
  args_.AddInt("stride_b");

  const std::string batch_id =
      op_def.dst_tensors[0].HasAxis(Axis::BATCH) ? "B" : "0";
  std::string c;
  c += "MAIN_FUNCTION($0) {\n";
  if (op_def.dst_tensors[0].HasAxis(Axis::BATCH)) {
    c += "  int linear_id = GLOBAL_ID_0;\n";
    c += "  int X = linear_id / args.dst_tensor.Batch();\n";
    c += "  int B = linear_id % args.dst_tensor.Batch();\n";
    c += "  args.dst_tensor.SetBatchRef(B);\n";
  } else {
    c += "  int X = GLOBAL_ID_0;\n";
  }
  c += "  int Y = GLOBAL_ID_1;\n";
  c += "  int S = GLOBAL_ID_2;\n";
  c += "  if (X >= args.dst_tensor.Width() || Y >= args.dst_tensor.Height() || "
       "S >= args.dst_tensor.Slices()) { \n";
  c += "    return; \n";
  c += "  } \n";
  c += "  int s_x = X * args.stride_x + args.offset_x;\n";
  c += "  int s_y = Y * args.stride_y + args.offset_y;\n";
  if (op_def.src_tensors[0].HasAxis(Axis::BATCH)) {
    c += "  int s_b = " + batch_id + " * args.stride_b + args.offset_b;\n";
    c += "  args.src_tensor.SetBatchRef(s_b);\n";
  }
  if (alignedx4) {
    c += "  int s_z = S + args.offset_z;\n";
    c += "  args.src_tensor::type result = args.src_tensor.Read(s_x, s_y, "
         "s_z);\n";
  } else {
    c += "  args.src_tensor::type result;\n";
    const std::string postfixes[] = {"x", "y", "z", "w"};
    for (int i = 0; i < 4; ++i) {
      c += "  {\n";
      const std::string channel = "(S * 4 + " + std::to_string(i) + ")";
      c += "    int s_ch = min(" + channel +
           " * args.stride_z + args.offset_z, args.src_tensor.Channels() - "
           "1);\n";
      c += "    args.src_tensor.ReadPerChannel(result." + postfixes[i] +
           ", s_x, s_y, s_ch);\n";
      c += "  }\n";
    }
  }
  c += "  args.dst_tensor.Write(result, X, Y, S);\n";
  c += "}\n";
  return c;
}

absl::Status StridedSlice::BindArguments(ArgumentsBinder* args) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStasksPSstrided_sliceDTcc mht_6(mht_6_v, 342, "", "./tensorflow/lite/delegates/gpu/common/tasks/strided_slice.cc", "StridedSlice::BindArguments");

  int4 offset = GetOffset(attributes_, src_[0]->Width(), src_[0]->Height(),
                          src_[0]->Channels(), src_[0]->Batch());
  RETURN_IF_ERROR(args->SetInt("offset_x", offset.x));
  RETURN_IF_ERROR(args->SetInt("offset_y", offset.y));
  RETURN_IF_ERROR(args->SetInt("offset_z", offset.z));
  RETURN_IF_ERROR(args->SetInt("offset_b", offset.w));
  RETURN_IF_ERROR(args->SetInt("stride_x", attributes_.strides.w));
  RETURN_IF_ERROR(args->SetInt("stride_y", attributes_.strides.h));
  RETURN_IF_ERROR(args->SetInt("stride_z", attributes_.strides.c));
  RETURN_IF_ERROR(args->SetInt("stride_b", attributes_.strides.b));
  return absl::OkStatus();
}

int3 StridedSlice::GetGridSize() const {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStasksPSstrided_sliceDTcc mht_7(mht_7_v, 359, "", "./tensorflow/lite/delegates/gpu/common/tasks/strided_slice.cc", "StridedSlice::GetGridSize");

  const int grid_x = dst_[0]->Width() * dst_[0]->Batch();
  const int grid_y = dst_[0]->Height();
  const int grid_z = dst_[0]->Slices();
  return int3(grid_x, grid_y, grid_z);
}

StridedSlice CreateStridedSlice(const OperationDef& definition,
                                const SliceAttributes& attr) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStasksPSstrided_sliceDTcc mht_8(mht_8_v, 370, "", "./tensorflow/lite/delegates/gpu/common/tasks/strided_slice.cc", "CreateStridedSlice");

  return StridedSlice(definition, attr);
}

}  // namespace gpu
}  // namespace tflite
