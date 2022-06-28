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
class MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStasksPSresizeDTcc {
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
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStasksPSresizeDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStasksPSresizeDTcc() {
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

#include "tensorflow/lite/delegates/gpu/common/tasks/resize.h"

#include <string>
#include <utility>

#include "tensorflow/lite/delegates/gpu/common/operations.h"
#include "tensorflow/lite/delegates/gpu/common/task/work_group_picking.h"

namespace tflite {
namespace gpu {

Resize::Resize(const OperationDef& definition, const Resize2DAttributes& attr)
    : GPUOperation(definition), attr_(attr) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStasksPSresizeDTcc mht_0(mht_0_v, 197, "", "./tensorflow/lite/delegates/gpu/common/tasks/resize.cc", "Resize::Resize");

  code_ = GetResizeCode(definition_, attr_);
}

Resize::Resize(Resize&& operation)
    : GPUOperation(std::move(operation)), attr_(operation.attr_) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStasksPSresizeDTcc mht_1(mht_1_v, 205, "", "./tensorflow/lite/delegates/gpu/common/tasks/resize.cc", "Resize::Resize");
}

Resize& Resize::operator=(Resize&& operation) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStasksPSresizeDTcc mht_2(mht_2_v, 210, "", "./tensorflow/lite/delegates/gpu/common/tasks/resize.cc", "=");

  if (this != &operation) {
    attr_ = operation.attr_;
    GPUOperation::operator=(std::move(operation));
  }
  return *this;
}

std::string Resize::GetResizeCode(const OperationDef& op_def,
                                  const Resize2DAttributes& attr) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStasksPSresizeDTcc mht_3(mht_3_v, 222, "", "./tensorflow/lite/delegates/gpu/common/tasks/resize.cc", "Resize::GetResizeCode");

  AddSrcTensor("src_tensor", op_def.src_tensors[0]);
  AddDstTensor("dst_tensor", op_def.dst_tensors[0]);
  args_.AddFloat("scale_factor_x");
  args_.AddFloat("scale_factor_y");

  std::string c;
  c += "MAIN_FUNCTION($0) {\n";
  if (op_def.dst_tensors[0].HasAxis(Axis::BATCH)) {
    c += "  int linear_id = GLOBAL_ID_0;\n";
    c += "  int X = linear_id / args.dst_tensor.Batch();\n";
    c += "  int B = linear_id % args.dst_tensor.Batch();\n";
    c += "  args.src_tensor.SetBatchRef(B);\n";
    c += "  args.dst_tensor.SetBatchRef(B);\n";
  } else {
    c += "  int X = GLOBAL_ID_0;\n";
  }
  c += "  int Y = GLOBAL_ID_1;\n";
  c += "  int Z = GLOBAL_ID_2;\n";
  c += "  if (X >= args.dst_tensor.Width() || Y >= args.dst_tensor.Height() "
       "|| Z >= args.dst_tensor.Slices()) return;\n";
  if (attr.half_pixel_centers) {
    c += "  float f_coords_x = (INIT_FLOAT(X) + 0.5f) * args.scale_factor_x;\n";
    c += "  float f_coords_y = (INIT_FLOAT(Y) + 0.5f) * args.scale_factor_y;\n";
  } else {
    c += "  float f_coords_x = INIT_FLOAT(X) * args.scale_factor_x;\n";
    c += "  float f_coords_y = INIT_FLOAT(Y) * args.scale_factor_y;\n";
  }
  c += "  FLT4 r0;\n";
  if (attr.type == SamplingType::NEAREST) {
    if (attr.align_corners) {
      c += "  f_coords_x += 0.5f;";
      c += "  f_coords_y += 0.5f;";
    }
    c += "  args.src_tensor.ReadNearest(r0, f_coords_x, f_coords_y, Z);\n";
  } else {
    if (attr.half_pixel_centers) {
      c += "  f_coords_x -= 0.5f;";
      c += "  f_coords_y -= 0.5f;";
    }
    c += "  args.src_tensor.ReadBilinear(r0, f_coords_x, f_coords_y, Z);\n";
  }
  c += "  args.dst_tensor.Write(r0, X, Y, Z);\n";
  c += "}\n";
  return c;
}

absl::Status Resize::BindArguments(ArgumentsBinder* args) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStasksPSresizeDTcc mht_4(mht_4_v, 272, "", "./tensorflow/lite/delegates/gpu/common/tasks/resize.cc", "Resize::BindArguments");

  RETURN_IF_ERROR(args->SetFloat(
      "scale_factor_x",
      CalculateResizeScale(src_[0]->Width(), dst_[0]->Width(), attr_)));
  RETURN_IF_ERROR(args->SetFloat(
      "scale_factor_y",
      CalculateResizeScale(src_[0]->Height(), dst_[0]->Height(), attr_)));
  return absl::OkStatus();
}

int3 Resize::GetGridSize() const {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStasksPSresizeDTcc mht_5(mht_5_v, 285, "", "./tensorflow/lite/delegates/gpu/common/tasks/resize.cc", "Resize::GetGridSize");

  const int grid_x = dst_[0]->Width() * dst_[0]->Batch();
  const int grid_y = dst_[0]->Height();
  const int grid_z = dst_[0]->Slices();
  return int3(grid_x, grid_y, grid_z);
}

Resize CreateResize(const OperationDef& definition,
                    const Resize2DAttributes& attr) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStasksPSresizeDTcc mht_6(mht_6_v, 296, "", "./tensorflow/lite/delegates/gpu/common/tasks/resize.cc", "CreateResize");

  return Resize(definition, attr);
}

Resize3D::Resize3D(const OperationDef& definition,
                   const Resize3DAttributes& attr)
    : GPUOperation(definition), attr_(attr) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStasksPSresizeDTcc mht_7(mht_7_v, 305, "", "./tensorflow/lite/delegates/gpu/common/tasks/resize.cc", "Resize3D::Resize3D");

  code_ = GetResize3DCode(definition_, attr_);
}

Resize3D::Resize3D(Resize3D&& operation)
    : GPUOperation(std::move(operation)), attr_(operation.attr_) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStasksPSresizeDTcc mht_8(mht_8_v, 313, "", "./tensorflow/lite/delegates/gpu/common/tasks/resize.cc", "Resize3D::Resize3D");
}

Resize3D& Resize3D::operator=(Resize3D&& operation) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStasksPSresizeDTcc mht_9(mht_9_v, 318, "", "./tensorflow/lite/delegates/gpu/common/tasks/resize.cc", "=");

  if (this != &operation) {
    attr_ = operation.attr_;
    GPUOperation::operator=(std::move(operation));
  }
  return *this;
}

std::string Resize3D::GetResize3DCode(const OperationDef& op_def,
                                      const Resize3DAttributes& attr) {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStasksPSresizeDTcc mht_10(mht_10_v, 330, "", "./tensorflow/lite/delegates/gpu/common/tasks/resize.cc", "Resize3D::GetResize3DCode");

  AddSrcTensor("src_tensor", op_def.src_tensors[0]);
  AddDstTensor("dst_tensor", op_def.dst_tensors[0]);
  args_.AddFloat("scale_factor_x");
  args_.AddFloat("scale_factor_y");
  args_.AddFloat("scale_factor_z");

  std::string c;
  c += "MAIN_FUNCTION($0) {\n";
  if (op_def.dst_tensors[0].HasAxis(Axis::BATCH)) {
    c += "  int linear_id = GLOBAL_ID_0;\n";
    c += "  int X = linear_id / args.dst_tensor.Batch();\n";
    c += "  int B = linear_id % args.dst_tensor.Batch();\n";
    c += "  args.src_tensor.SetBatchRef(B);\n";
    c += "  args.dst_tensor.SetBatchRef(B);\n";
  } else {
    c += "  int X = GLOBAL_ID_0;\n";
  }
  c += "  int Y = GLOBAL_ID_1;\n";
  c += "  int linear_id_z = GLOBAL_ID_2;\n";
  c += "  int S = linear_id_z % args.dst_tensor.Slices();\n";
  c += "  int Z = linear_id_z / args.dst_tensor.Slices();\n";
  c += "  if (X >= args.dst_tensor.Width() || Y >= args.dst_tensor.Height() "
       "|| Z >= args.dst_tensor.Depth()) return;\n";
  if (attr.half_pixel_centers) {
    c += "  float f_coords_x = (INIT_FLOAT(X) + 0.5f) * args.scale_factor_x;\n";
    c += "  float f_coords_y = (INIT_FLOAT(Y) + 0.5f) * args.scale_factor_y;\n";
    c += "  float f_coords_z = (INIT_FLOAT(Z) + 0.5f) * args.scale_factor_z;\n";
  } else {
    c += "  float f_coords_x = INIT_FLOAT(X) * args.scale_factor_x;\n";
    c += "  float f_coords_y = INIT_FLOAT(Y) * args.scale_factor_y;\n";
    c += "  float f_coords_z = INIT_FLOAT(Z) * args.scale_factor_z;\n";
  }
  c += "  FLT4 r0;\n";
  if (attr.type == SamplingType::NEAREST) {
    if (attr.align_corners) {
      c += "  f_coords_x += 0.5f;";
      c += "  f_coords_y += 0.5f;";
      c += "  f_coords_z += 0.5f;";
    }
    c += "  args.src_tensor.ReadNearest(r0, f_coords_x, f_coords_y, "
         "f_coords_z, S);\n";
  } else {
    if (attr.half_pixel_centers) {
      c += "  f_coords_x -= 0.5f;";
      c += "  f_coords_y -= 0.5f;";
      c += "  f_coords_z -= 0.5f;";
    }
    c += "  args.src_tensor.ReadBilinear(r0, f_coords_x, f_coords_y, "
         "f_coords_z, S);\n";
  }
  c += "  args.dst_tensor.Write(r0, X, Y, Z, S);\n";
  c += "}\n";
  return c;
}

absl::Status Resize3D::BindArguments(ArgumentsBinder* args) {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStasksPSresizeDTcc mht_11(mht_11_v, 389, "", "./tensorflow/lite/delegates/gpu/common/tasks/resize.cc", "Resize3D::BindArguments");

  RETURN_IF_ERROR(args->SetFloat(
      "scale_factor_x",
      CalculateResizeScale(src_[0]->Width(), dst_[0]->Width(), attr_)));
  RETURN_IF_ERROR(args->SetFloat(
      "scale_factor_y",
      CalculateResizeScale(src_[0]->Height(), dst_[0]->Height(), attr_)));
  RETURN_IF_ERROR(args->SetFloat(
      "scale_factor_z",
      CalculateResizeScale(src_[0]->Depth(), dst_[0]->Depth(), attr_)));
  return absl::OkStatus();
}

int3 Resize3D::GetGridSize() const {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStasksPSresizeDTcc mht_12(mht_12_v, 405, "", "./tensorflow/lite/delegates/gpu/common/tasks/resize.cc", "Resize3D::GetGridSize");

  const int grid_x = dst_[0]->Width() * dst_[0]->Batch();
  const int grid_y = dst_[0]->Height();
  const int grid_z = dst_[0]->Slices() * dst_[0]->Depth();
  return int3(grid_x, grid_y, grid_z);
}

Resize3D CreateResize3D(const OperationDef& definition,
                        const Resize3DAttributes& attr) {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStasksPSresizeDTcc mht_13(mht_13_v, 416, "", "./tensorflow/lite/delegates/gpu/common/tasks/resize.cc", "CreateResize3D");

  return Resize3D(definition, attr);
}

}  // namespace gpu
}  // namespace tflite
