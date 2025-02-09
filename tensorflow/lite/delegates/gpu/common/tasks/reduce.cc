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
class MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStasksPSreduceDTcc {
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
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStasksPSreduceDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStasksPSreduceDTcc() {
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

#include "tensorflow/lite/delegates/gpu/common/tasks/reduce.h"

#include <set>
#include <string>
#include <utility>

#include "tensorflow/lite/delegates/gpu/common/status.h"
#include "tensorflow/lite/delegates/gpu/common/task/util.h"
#include "tensorflow/lite/delegates/gpu/common/task/work_group_picking.h"
#include "tensorflow/lite/delegates/gpu/common/util.h"

namespace tflite {
namespace gpu {

namespace {
int GetMaximumWGTotalSize(const GpuInfo& gpu_info) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStasksPSreduceDTcc mht_0(mht_0_v, 200, "", "./tensorflow/lite/delegates/gpu/common/tasks/reduce.cc", "GetMaximumWGTotalSize");

  // total_wg_size must be power of 2 and >= 4;
  int total_wg_size = 256;
  if (gpu_info.IsAdreno() && gpu_info.adreno_info.IsAdreno3xx()) {
    total_wg_size = 128;
  }
  if (gpu_info.IsMali()) {
    const MaliInfo& mali_info = gpu_info.mali_info;
    if (mali_info.IsMaliT6xx() || mali_info.IsMaliT7xx() ||
        mali_info.IsMaliT8xx()) {
      total_wg_size = 32;
    } else {
      total_wg_size = 64;
    }
  }
  return total_wg_size;
}

bool HasAxis(const std::vector<Axis>& axis, Axis a) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStasksPSreduceDTcc mht_1(mht_1_v, 221, "", "./tensorflow/lite/delegates/gpu/common/tasks/reduce.cc", "HasAxis");

  for (const auto& a2 : axis) {
    if (a2 == a) {
      return true;
    }
  }
  return false;
}

std::string MakeOp(OperationType op_type, const std::string& a,
                   const std::string& b) {
   std::vector<std::string> mht_2_v;
   mht_2_v.push_back("a: \"" + a + "\"");
   mht_2_v.push_back("b: \"" + b + "\"");
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStasksPSreduceDTcc mht_2(mht_2_v, 236, "", "./tensorflow/lite/delegates/gpu/common/tasks/reduce.cc", "MakeOp");

  if (op_type == OperationType::REDUCE_SUM || op_type == OperationType::MEAN) {
    return "((" + a + ") + (" + b + "))";
  } else if (op_type == OperationType::REDUCE_PRODUCT) {
    return "((" + a + ") * (" + b + "))";
  } else if (op_type == OperationType::REDUCE_MAXIMUM) {
    return "max(" + a + ", " + b + ")";
  } else if (op_type == OperationType::REDUCE_MINIMUM) {
    return "min(" + a + ", " + b + ")";
  }
  return "UnsupportedOperation";
}

// max_total_wg_size is pot
int3 GetMaximumPossibleWGSize(const std::vector<int>& ordered_sizes,
                              int max_total_wg_size) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStasksPSreduceDTcc mht_3(mht_3_v, 254, "", "./tensorflow/lite/delegates/gpu/common/tasks/reduce.cc", "GetMaximumPossibleWGSize");

  int3 wg_size = int3(1, 1, 1);
  int wg_size_total = 1;
  for (int i = ordered_sizes.size() - 1; i >= 0; i--) {
    const int wg_index = ordered_sizes.size() - 1 - i;
    if (wg_index >= 3) {
      return wg_size;
    }
    while (ordered_sizes[i] >= wg_size[wg_index] * 2) {
      wg_size_total *= 2;
      if (wg_size_total > max_total_wg_size) {
        return wg_size;
      }
      wg_size[wg_index] *= 2;
    }
  }
  return wg_size;
}

std::map<Axis, int> GetSizesFromShape(const std::set<Axis>& axis,
                                      const BHWC& shape) {
  std::map<Axis, int> result;
  for (auto a : axis) {
    result[a] = shape.get(a);
  }
  return result;
}

std::map<Axis, int> GetSizesFromShape(const std::set<Axis>& axis,
                                      const BHWDC& shape) {
  std::map<Axis, int> result;
  for (auto a : axis) {
    result[a] = shape.get(a);
  }
  return result;
}

DataType GetAccumType(DataType src_type) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStasksPSreduceDTcc mht_4(mht_4_v, 294, "", "./tensorflow/lite/delegates/gpu/common/tasks/reduce.cc", "GetAccumType");

  if (src_type == DataType::FLOAT32 || src_type == DataType::FLOAT16) {
    return DataType::FLOAT32;
  } else if (src_type == DataType::INT32 || src_type == DataType::INT16 ||
             src_type == DataType::INT8) {
    return DataType::INT32;
  } else if (src_type == DataType::UINT32 || src_type == DataType::UINT16 ||
             src_type == DataType::UINT8) {
    return DataType::UINT32;
  } else {
    return src_type;
  }
}

}  // namespace

Reduce::Reduce(const std::map<Axis, int>& axis_to_reduce, OperationType op_type,
               const OperationDef& definition, const GpuInfo& gpu_info)
    : GPUOperation(definition) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStasksPSreduceDTcc mht_5(mht_5_v, 315, "", "./tensorflow/lite/delegates/gpu/common/tasks/reduce.cc", "Reduce::Reduce");

  std::vector<Axis> ordered_axis_to_reduce;
  std::vector<int> ordered_sizes;
  for (const auto& a :
       {Axis::CHANNELS, Axis::DEPTH, Axis::HEIGHT, Axis::WIDTH, Axis::BATCH}) {
    auto it = axis_to_reduce.find(a);
    if (it != axis_to_reduce.end()) {
      ordered_axis_to_reduce.push_back(it->first);
      int reduction_size = it->second;
      if (a == Axis::CHANNELS) {
        reduction_size = DivideRoundUp(reduction_size, 4);
      }
      ordered_sizes.push_back(reduction_size);
    }
  }
  const int max_total_wg_size = GetMaximumWGTotalSize(gpu_info);
  int3 current_wg_size =
      GetMaximumPossibleWGSize(ordered_sizes, max_total_wg_size);
  int current_wg_size_total =
      current_wg_size.x * current_wg_size.y * current_wg_size.z;
  int threshold = max_total_wg_size / 4;
  if (gpu_info.IsApple()) {
    threshold = 16;
  }
  if (current_wg_size_total < threshold) {
    use_wg_reduction_ = false;
  } else {
    use_wg_reduction_ = true;
    work_group_size_ = current_wg_size;
  }
  code_ = GetReduceKernelCode(definition_, gpu_info, work_group_size_,
                              ordered_axis_to_reduce, op_type);
}

Reduce::Reduce(Reduce&& operation)
    : GPUOperation(std::move(operation)),
      use_wg_reduction_(operation.use_wg_reduction_) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStasksPSreduceDTcc mht_6(mht_6_v, 354, "", "./tensorflow/lite/delegates/gpu/common/tasks/reduce.cc", "Reduce::Reduce");
}

Reduce& Reduce::operator=(Reduce&& operation) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStasksPSreduceDTcc mht_7(mht_7_v, 359, "", "./tensorflow/lite/delegates/gpu/common/tasks/reduce.cc", "=");

  if (this != &operation) {
    use_wg_reduction_ = operation.use_wg_reduction_;
    GPUOperation::operator=(std::move(operation));
  }
  return *this;
}

std::string Reduce::GetReduceKernelCode(const OperationDef& op_def,
                                        const GpuInfo& gpu_info,
                                        const int3& work_group_size,
                                        const std::vector<Axis>& axis_to_reduce,
                                        OperationType op_type) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStasksPSreduceDTcc mht_8(mht_8_v, 374, "", "./tensorflow/lite/delegates/gpu/common/tasks/reduce.cc", "Reduce::GetReduceKernelCode");

  AddSrcTensor("src_tensor", op_def.src_tensors[0]);
  AddDstTensor("dst_tensor", op_def.dst_tensors[0]);
  args_.AddFloat("inv_multiplier_1");
  args_.AddFloat("inv_multiplier_2");

  std::set<Axis> axis_to_leave;
  const std::vector<Axis> all_axis = {Axis::WIDTH, Axis::HEIGHT, Axis::DEPTH,
                                      Axis::CHANNELS, Axis::BATCH};
  for (const auto& a : all_axis) {
    if (op_def.dst_tensors[0].HasAxis(a)) {
      if (!HasAxis(axis_to_reduce, a)) {
        axis_to_leave.insert(a);
      }
    }
  }
  const bool channels_reductin = HasAxis(axis_to_reduce, Axis::CHANNELS);
  int wg_dims = 0;
  if (use_wg_reduction_) {
    if (work_group_size.y == 1 && work_group_size.z == 1) {
      wg_dims = 1;
    } else if (work_group_size.z == 1) {
      wg_dims = 2;
    } else {
      wg_dims = 3;
    }
  }

  auto get_global_id = [&](int i) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStasksPSreduceDTcc mht_9(mht_9_v, 405, "", "./tensorflow/lite/delegates/gpu/common/tasks/reduce.cc", "lambda");

    if (use_wg_reduction_) {
      return "GROUP_ID_" + std::to_string(i);
    } else {
      return "GLOBAL_ID_" + std::to_string(i);
    }
  };

  auto accum_type = GetAccumType(op_def.src_tensors[0].data_type);
  const std::string accum_type_decl =
      GetTypeDeclaration(gpu_info, accum_type, 4);
  std::string read_as_template;
  if (accum_type == DataType::FLOAT32) {
    read_as_template = "<float>";
  } else if (accum_type == DataType::INT32) {
    read_as_template = "<int>";
  } else if (accum_type == DataType::UINT32) {
    read_as_template = "<uint>";
  }

  std::string c;
  const std::string wg_x = std::to_string(work_group_size.x);
  const std::string wg_y = std::to_string(work_group_size.y);
  const std::string wg_z = std::to_string(work_group_size.z);
  const int wg_total_size =
      work_group_size.x * work_group_size.y * work_group_size.z;
  c += "MAIN_FUNCTION($0) {\n";
  if (use_wg_reduction_) {
    c += "  __local " + accum_type_decl + " accum[" +
         std::to_string(wg_total_size) + "];\n";
    if (wg_dims == 1) {
      c += "  int local_x = LOCAL_ID_0;\n";
      c += "  int local_id = local_x;\n";
    } else if (wg_dims == 2) {
      c += "  int local_x = LOCAL_ID_0;\n";
      c += "  int local_y = LOCAL_ID_1;\n";
      c += "  int local_id = local_y * " + wg_x + " + local_x;\n";
    } else if (wg_dims == 3) {
      c += "  int local_x = LOCAL_ID_0;\n";
      c += "  int local_y = LOCAL_ID_1;\n";
      c += "  int local_z = LOCAL_ID_2;\n";
      c += "  int local_id = (local_z * " + wg_y + " + local_y) * " + wg_x +
           " + local_x;\n";
    }
  }
  if (axis_to_leave.count(Axis::WIDTH)) {
    if (axis_to_leave.count(Axis::BATCH)) {
      c += "  int linear_id = " + get_global_id(0) + ";\n";
      c += "  int DST_X = linear_id / args.dst_tensor.Batch();\n";
      c += "  int DST_B = linear_id % args.dst_tensor.Batch();\n";
    } else {
      c += "  int DST_X = " + get_global_id(0) + ";\n";
    }
  } else if (axis_to_leave.count(Axis::BATCH)) {
    c += "  int DST_B = " + get_global_id(0) + ";\n";
  }
  if (axis_to_leave.count(Axis::HEIGHT)) {
    if (axis_to_leave.count(Axis::DEPTH)) {
      c += "  int linear_id = " + get_global_id(1) + ";\n";
      c += "  int DST_Y = linear_id % args.dst_tensor.Height();\n";
      c += "  int DST_Z = linear_id / args.dst_tensor.Height();\n";
    } else {
      c += "  int DST_Y = " + get_global_id(1) + ";\n";
    }
  } else if (axis_to_leave.count(Axis::DEPTH)) {
    c += "  int DST_Z = " + get_global_id(1) + ";\n";
  }
  if (axis_to_leave.count(Axis::CHANNELS)) {
    c += "  int DST_S = " + get_global_id(2) + ";\n";
  }
  std::map<Axis, std::string> axis_to_selector = {
      {Axis::BATCH, "Batch()"},     {Axis::WIDTH, "Width()"},
      {Axis::HEIGHT, "Height()"},   {Axis::DEPTH, "Depth()"},
      {Axis::CHANNELS, "Slices()"},
  };
  std::map<Axis, std::string> axis_to_coord = {
      {Axis::BATCH, "B"}, {Axis::WIDTH, "X"},    {Axis::HEIGHT, "Y"},
      {Axis::DEPTH, "Z"}, {Axis::CHANNELS, "S"},
  };
  std::string dst_check;
  for (auto& axis : axis_to_leave) {
    if (!dst_check.empty()) {
      dst_check += " || ";
    }
    dst_check += "DST_" + axis_to_coord[axis] + " >= args.dst_tensor." +
                 axis_to_selector[axis];
  }
  if (!dst_check.empty()) {
    c += "  if (" + dst_check + ") return;\n";
  }
  std::map<Axis, std::string> src_coords;
  for (const auto& a : all_axis) {
    if (op_def.dst_tensors[0].HasAxis(a) && !HasAxis(axis_to_reduce, a)) {
      src_coords[a] = "DST_" + axis_to_coord[a];
    } else {
      src_coords[a] = "0";
    }
  }
  std::string src_coordinates;
  for (const auto& a : all_axis) {
    if (op_def.src_tensors[0].HasAxis(a)) {
      if (!src_coordinates.empty()) {
        src_coordinates += ", ";
      }
      src_coordinates += src_coords[a];
    }
  }
  if (op_type == OperationType::REDUCE_SUM || op_type == OperationType::MEAN) {
    c += "  " + accum_type_decl +
         " reducer = " + GetZeroValue(gpu_info, accum_type, 4) + ";\n";
  } else if (op_type == OperationType::REDUCE_PRODUCT) {
    c += "  " + accum_type_decl +
         " reducer = " + GetOneValue(gpu_info, accum_type, 4) + ";\n";
  } else if (op_type == OperationType::REDUCE_MAXIMUM ||
             op_type == OperationType::REDUCE_MINIMUM) {
    c += "  " + accum_type_decl + " reducer = args.src_tensor.Read" +
         read_as_template + "(" + src_coordinates + ");\n";
    if (channels_reductin) {
      c += "  reducer.y = reducer.x;\n";
      c += "  reducer.z = reducer.x;\n";
      c += "  reducer.w = reducer.x;\n";
    }
  }
  const std::vector<std::string> local_ids = {"local_x", "local_y", "local_z"};
  const std::vector<std::string> local_sizes = {wg_x, wg_y, wg_z};
  for (const auto& axis : axis_to_reduce) {
    if (axis == Axis::CHANNELS) {
      c += "  " + accum_type_decl + " mask;\n";
      const std::string one_or_zero_value =
          GetOneValue(gpu_info, accum_type, 1) + " : " +
          GetZeroValue(gpu_info, accum_type, 1);
      c += "  mask.x = (args.src_tensor.Slices() - 1) * 4 + 0 < "
           "args.src_tensor.Channels() ? " +
           one_or_zero_value + ";\n";
      c += "  mask.y = (args.src_tensor.Slices() - 1) * 4 + 1 < "
           "args.src_tensor.Channels() ? " +
           one_or_zero_value + ";\n";
      c += "  mask.z = (args.src_tensor.Slices() - 1) * 4 + 2 < "
           "args.src_tensor.Channels() ? " +
           one_or_zero_value + ";\n";
      c += "  mask.w = (args.src_tensor.Slices() - 1) * 4 + 3 < "
           "args.src_tensor.Channels() ? " +
           one_or_zero_value + ";\n";
    }
  }
  for (int i = 0; i < axis_to_reduce.size(); ++i) {
    const auto& axis = axis_to_reduce[i];
    const int index = axis_to_reduce.size() - 1 - i;
    const std::string first = index < wg_dims ? local_ids[index] : "0";
    const std::string step = index < wg_dims ? local_sizes[index] : "1";
    const std::string src_coord = "SRC_" + axis_to_coord[axis];
    src_coords[axis] = src_coord;
    c += "  for (int " + src_coord + " = " + first + "; " + src_coord +
         " < args.src_tensor." + axis_to_selector[axis] + "; " + src_coord +
         " += " + step + ") {\n";
    if (axis == Axis::CHANNELS) {
      c += "    bool last = SRC_S == args.src_tensor.Slices() - 1;\n";
      c += "    " + accum_type_decl +
           " mask_a = last ? mask : " + GetOneValue(gpu_info, accum_type, 4) +
           ";\n";
      if (op_type == OperationType::REDUCE_PRODUCT ||
          op_type == OperationType::REDUCE_MAXIMUM ||
          op_type == OperationType::REDUCE_MINIMUM) {
        c += "    " + accum_type_decl +
             " mask_b = " + GetOneValue(gpu_info, accum_type, 4) +
             " - mask_a;\n";
      }
    }
  }
  src_coordinates = "";
  for (const auto& a : all_axis) {
    if (op_def.src_tensors[0].HasAxis(a)) {
      if (!src_coordinates.empty()) {
        src_coordinates += ", ";
      }
      src_coordinates += src_coords[a];
    }
  }
  c += "    " + accum_type_decl + " src_val = args.src_tensor.Read" +
       read_as_template + "(" + src_coordinates + ");\n";
  if (channels_reductin) {
    if (op_type == OperationType::REDUCE_SUM ||
        op_type == OperationType::MEAN) {
      c += "    src_val = src_val * mask_a;\n";
    } else if (op_type == OperationType::REDUCE_PRODUCT) {
      c += "    src_val = src_val * mask_a + mask_b;\n";
    } else if (op_type == OperationType::REDUCE_MAXIMUM ||
               op_type == OperationType::REDUCE_MINIMUM) {
      c += "    src_val = src_val * mask_a + mask_b * src_val.x;\n";
    }
  }
  c += "    reducer = " + MakeOp(op_type, "reducer", "src_val") + ";\n";
  for (int i = 0; i < axis_to_reduce.size(); ++i) {
    c += "  }\n";
  }
  if (op_type == OperationType::MEAN) {
    c += "  reducer *= args.inv_multiplier_1;\n";
  }
  if (use_wg_reduction_) {
    c += "  accum[local_id] = reducer;\n";
    c += "  LOCAL_MEM_BARRIER;\n";
    const int total_size =
        work_group_size.x * work_group_size.y * work_group_size.z;
    int offset = 1;
    int reminder = total_size / 4;
    for (; reminder >= 8; reminder /= 4, offset *= 4) {
      c += "  if (local_id < " + std::to_string(reminder) + ") {\n";
      c += "    int t = local_id * " + std::to_string(offset * 4) + ";\n";
      c += "    " + accum_type_decl + " sum = accum[t + " +
           std::to_string(offset) + "];\n";
      c += "    sum = " +
           MakeOp(op_type, "sum",
                  "accum[t + " + std::to_string(offset * 2) + "]") +
           ";\n";
      c += "    sum = " +
           MakeOp(op_type, "sum",
                  "accum[t + " + std::to_string(offset * 3) + "]") +
           ";\n";
      c += "    accum[t] = " + MakeOp(op_type, "accum[t]", "sum") + ";\n";
      c += "  }\n";
      c += "  LOCAL_MEM_BARRIER;\n";
    }
    c += "  reducer = accum[0];\n";
    reminder *= 4;
    for (int i = 1; i < reminder; ++i) {
      c += "  reducer = " +
           MakeOp(op_type, "reducer",
                  "accum[" + std::to_string(offset * i) + "]") +
           ";\n";
    }
    if (op_type == OperationType::MEAN) {
      c += "  reducer *= args.inv_multiplier_2;\n";
    }
  }
  if (channels_reductin) {
    if (op_type == OperationType::REDUCE_SUM ||
        op_type == OperationType::MEAN) {
      c += "  reducer.x += reducer.y + reducer.z + reducer.w;\n";
    } else if (op_type == OperationType::REDUCE_PRODUCT) {
      c += "  reducer.x *= reducer.y * reducer.z * reducer.w;\n";
    } else if (op_type == OperationType::REDUCE_MAXIMUM) {
      c += "  reducer.x = max(reducer.x, reducer.y);\n";
      c += "  reducer.x = max(reducer.x, reducer.z);\n";
      c += "  reducer.x = max(reducer.x, reducer.w);\n";
    } else if (op_type == OperationType::REDUCE_MINIMUM) {
      c += "  reducer.x = min(reducer.x, reducer.y);\n";
      c += "  reducer.x = min(reducer.x, reducer.z);\n";
      c += "  reducer.x = min(reducer.x, reducer.w);\n";
    }
  }
  const std::string conversion = GetTypeConvertion(
      gpu_info, accum_type, op_def.src_tensors[0].data_type, 4);
  if (conversion.empty()) {
    c += "  args.src_tensor::type result = reducer;\n";
  } else {
    c += "  args.src_tensor::type result = " + conversion + "(reducer);\n";
  }
  std::string dst_coordinates;
  for (const auto& a : all_axis) {
    if (op_def.dst_tensors[0].HasAxis(a)) {
      if (!dst_coordinates.empty()) {
        dst_coordinates += ", ";
      }
      if (axis_to_leave.count(a)) {
        dst_coordinates += "DST_" + axis_to_coord[a];
      } else {
        dst_coordinates += "0";
      }
    }
  }
  c += "  args.dst_tensor.Write(result, " + dst_coordinates + ");\n";
  c += "}\n";
  return c;
}

absl::Status Reduce::BindArguments(ArgumentsBinder* args) {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStasksPSreduceDTcc mht_10(mht_10_v, 683, "", "./tensorflow/lite/delegates/gpu/common/tasks/reduce.cc", "Reduce::BindArguments");

  const double total_src_elements = 1.0 * src_[0]->Batch() * src_[0]->Width() *
                                    src_[0]->Height() * src_[0]->Depth() *
                                    src_[0]->Channels();
  const double total_dst_elements = 1.0 * dst_[0]->Batch() * dst_[0]->Width() *
                                    dst_[0]->Height() * dst_[0]->Depth() *
                                    dst_[0]->Channels();
  const double reduction_size = total_src_elements / total_dst_elements;
  if (use_wg_reduction_) {
    const double size_0 =
        work_group_size_.x * work_group_size_.y * work_group_size_.z;
    const double size_1 = reduction_size / size_0;
    RETURN_IF_ERROR(args->SetFloat("inv_multiplier_1", 1.0 / size_1));
    RETURN_IF_ERROR(args->SetFloat("inv_multiplier_2", 1.0 / size_0));
  } else {
    RETURN_IF_ERROR(args->SetFloat("inv_multiplier_1", 1.0 / reduction_size));
    RETURN_IF_ERROR(args->SetFloat("inv_multiplier_2", 1.0));
  }
  return absl::OkStatus();
}

int3 Reduce::GetGridSize() const {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStasksPSreduceDTcc mht_11(mht_11_v, 707, "", "./tensorflow/lite/delegates/gpu/common/tasks/reduce.cc", "Reduce::GetGridSize");

  int grid_x = dst_[0]->Width() * dst_[0]->Batch();
  int grid_y = dst_[0]->Height() * dst_[0]->Depth();
  int grid_z = dst_[0]->Slices();
  if (use_wg_reduction_) {
    grid_x *= work_group_size_.x;
    grid_y *= work_group_size_.y;
    grid_z *= work_group_size_.z;
  }
  return int3(grid_x, grid_y, grid_z);
}

void Reduce::GetPossibleKernelWorkGroups(TuningType tuning_type,
                                         const GpuInfo& gpu_info,
                                         const KernelInfo& kernel_info,
                                         std::vector<int3>* work_groups) const {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStasksPSreduceDTcc mht_12(mht_12_v, 725, "", "./tensorflow/lite/delegates/gpu/common/tasks/reduce.cc", "Reduce::GetPossibleKernelWorkGroups");

  if (use_wg_reduction_) {
    work_groups->push_back(work_group_size_);
  } else {
    GetPossibleWorkGroups(tuning_type, gpu_info, kernel_info, grid_size_,
                          work_groups);
  }
}

Reduce CreateReduce(const std::set<Axis>& axis_to_reduce, const BHWC& src_shape,
                    OperationType op_type, const OperationDef& definition,
                    const GpuInfo& gpu_info) {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStasksPSreduceDTcc mht_13(mht_13_v, 739, "", "./tensorflow/lite/delegates/gpu/common/tasks/reduce.cc", "CreateReduce");

  return Reduce(GetSizesFromShape(axis_to_reduce, src_shape), op_type,
                definition, gpu_info);
}

Reduce CreateReduce(const std::set<Axis>& axis_to_reduce,
                    const BHWDC& src_shape, OperationType op_type,
                    const OperationDef& definition, const GpuInfo& gpu_info) {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStasksPSreduceDTcc mht_14(mht_14_v, 749, "", "./tensorflow/lite/delegates/gpu/common/tasks/reduce.cc", "CreateReduce");

  return Reduce(GetSizesFromShape(axis_to_reduce, src_shape), op_type,
                definition, gpu_info);
}

}  // namespace gpu
}  // namespace tflite
