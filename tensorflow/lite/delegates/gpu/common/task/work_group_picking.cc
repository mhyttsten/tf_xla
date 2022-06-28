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
class MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStaskPSwork_group_pickingDTcc {
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
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStaskPSwork_group_pickingDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStaskPSwork_group_pickingDTcc() {
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

#include "tensorflow/lite/delegates/gpu/common/task/work_group_picking.h"

#include <algorithm>
#include <limits>
#include <set>
#include <vector>

#include "tensorflow/lite/delegates/gpu/common/util.h"

namespace tflite {
namespace gpu {

namespace {

std::vector<int2> Get2DWorkgroupsEqualTo128() {
  return {{128, 1}, {64, 2}, {32, 4}, {16, 8},
          {8, 16},  {4, 32}, {2, 64}, {1, 128}};
}

std::vector<int3> GenerateWorkGroupSizesXYMultipleOf(
    int multiplier, int3 grid, const KernelInfo& kernel_info,
    const GpuInfo& gpu_info, WorkGroupSizeAlignment z_alignment) {
  std::vector<int3> work_groups;
  work_groups.reserve(32);

  std::vector<int> possible_z_sizes = GetPossibleSizes(grid.z, z_alignment);

  for (int x = 1; x <= kernel_info.max_work_group_size; x *= 2) {
    for (int y = 1; y <= kernel_info.max_work_group_size; y *= 2) {
      int work_group_size_xy = x * y;
      if (work_group_size_xy % multiplier != 0 ||
          work_group_size_xy > kernel_info.max_work_group_size) {
        continue;
      }
      for (auto z : possible_z_sizes) {
        if (work_group_size_xy * z > kernel_info.max_work_group_size) {
          continue;
        }
        if (x <= gpu_info.GetMaxWorkGroupSizeForX() &&
            y <= gpu_info.GetMaxWorkGroupSizeForY() &&
            z <= gpu_info.GetMaxWorkGroupSizeForZ()) {
          work_groups.push_back({x, y, z});
        }
      }
    }
  }
  return work_groups;
}

std::vector<int3> GenerateWorkGroupSizesXMultipleOf(
    int multiplier, int3 grid, const KernelInfo& kernel_info,
    const GpuInfo& gpu_info, WorkGroupSizeAlignment z_alignment) {
  std::vector<int3> work_groups;
  work_groups.reserve(32);

  std::vector<int> possible_z_sizes = GetPossibleSizes(grid.z, z_alignment);
  std::vector<int> possible_y_sizes =
      GetPossibleSizes(grid.y, WorkGroupSizeAlignment::PRECISE);

  for (int x = multiplier;
       x <= kernel_info.max_work_group_size && x < grid.x + multiplier;
       x += multiplier) {
    for (auto y : possible_y_sizes) {
      for (auto z : possible_z_sizes) {
        if (x <= gpu_info.GetMaxWorkGroupSizeForX() &&
            y <= gpu_info.GetMaxWorkGroupSizeForY() &&
            z <= gpu_info.GetMaxWorkGroupSizeForZ() &&
            x * y * z <= kernel_info.max_work_group_size) {
          work_groups.push_back({x, y, z});
        }
      }
    }
  }
  return work_groups;
}

void GetWorkGroupsAlignedToGrid(const GpuInfo& gpu_info,
                                const KernelInfo& kernel_info, const int3& grid,
                                std::vector<int3>* work_groups) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStaskPSwork_group_pickingDTcc mht_0(mht_0_v, 263, "", "./tensorflow/lite/delegates/gpu/common/task/work_group_picking.cc", "GetWorkGroupsAlignedToGrid");

  int3 max_wg_size;
  max_wg_size.x = gpu_info.GetMaxWorkGroupSizeForX();
  max_wg_size.y = gpu_info.GetMaxWorkGroupSizeForY();
  max_wg_size.z = gpu_info.GetMaxWorkGroupSizeForZ();
  GenerateWorkGroupSizesAlignedToGrid(
      grid, max_wg_size, kernel_info.max_work_group_size, work_groups);
}

int GetPenalty(int grid_size, int group_size) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStaskPSwork_group_pickingDTcc mht_1(mht_1_v, 275, "", "./tensorflow/lite/delegates/gpu/common/task/work_group_picking.cc", "GetPenalty");

  const int reminder = grid_size % group_size;
  return reminder == 0 ? 0 : group_size - reminder;
}

int GetPenalty(int2 grid_size, int2 group_size) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStaskPSwork_group_pickingDTcc mht_2(mht_2_v, 283, "", "./tensorflow/lite/delegates/gpu/common/task/work_group_picking.cc", "GetPenalty");

  const int p_x = GetPenalty(grid_size.x, group_size.x);
  const int p_y = GetPenalty(grid_size.y, group_size.y);
  return p_x * grid_size.y + p_y * grid_size.x + p_x * p_y;
}

int GetMaxSizeWithMinPenalty(int size, int max_size) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStaskPSwork_group_pickingDTcc mht_3(mht_3_v, 292, "", "./tensorflow/lite/delegates/gpu/common/task/work_group_picking.cc", "GetMaxSizeWithMinPenalty");

  int best_size = 128;
  int min_penalty = GetPenalty(size, best_size);
  for (int i = 2; i * 128 <= max_size; ++i) {
    if (GetPenalty(size, i * 128) == min_penalty) {
      best_size = i * 128;
    }
  }
  return best_size;
}

int2 GetMaxSizeWithMinPenalty(int2 size, int max_size) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStaskPSwork_group_pickingDTcc mht_4(mht_4_v, 306, "", "./tensorflow/lite/delegates/gpu/common/task/work_group_picking.cc", "GetMaxSizeWithMinPenalty");

  std::vector<int2> base_groups = Get2DWorkgroupsEqualTo128();
  int min_penalty = std::numeric_limits<int>::max();
  for (const auto& group : base_groups) {
    min_penalty = std::min(GetPenalty(size, group), min_penalty);
  }
  for (const auto& group : base_groups) {
    for (int y = 1; y * group.y <= max_size; ++y) {
      int new_group_y = y * group.y;
      for (int x = 1; x * group.x <= max_size; ++x) {
        int new_group_x = x * group.x;
        if (new_group_x * new_group_y > max_size) {
          break;
        }
        if (GetPenalty(size, int2(new_group_x, new_group_y)) == min_penalty) {
          return int2(new_group_x, new_group_y);
        }
      }
    }
  }
  return int2(0, 0);
}

int GetBiggestDividerWithPriority(int number, int max_divider) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStaskPSwork_group_pickingDTcc mht_5(mht_5_v, 332, "", "./tensorflow/lite/delegates/gpu/common/task/work_group_picking.cc", "GetBiggestDividerWithPriority");

  if (number % 8 == 0 && 8 <= max_divider) {
    return 8;
  }
  if (number % 4 == 0 && 4 <= max_divider) {
    return 4;
  }
  if (number % 2 == 0 && 2 <= max_divider) {
    return 2;
  }
  for (int i = max_divider; i != 0; i--) {
    if (number % i == 0) {
      return i;
    }
  }
  return 1;
}

int GetBiggestDivider(int number, int max_divider) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStaskPSwork_group_pickingDTcc mht_6(mht_6_v, 353, "", "./tensorflow/lite/delegates/gpu/common/task/work_group_picking.cc", "GetBiggestDivider");

  for (int i = max_divider; i != 0; i--) {
    if (number % i == 0) {
      return i;
    }
  }
  return 1;
}

int GetOptimalSizeForApple(int grid_size) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStaskPSwork_group_pickingDTcc mht_7(mht_7_v, 365, "", "./tensorflow/lite/delegates/gpu/common/task/work_group_picking.cc", "GetOptimalSizeForApple");

  if (grid_size % 8 == 0 || grid_size % 8 >= 4 || grid_size >= 16) {
    return 8;
  }
  if (grid_size % 4 == 0 || grid_size % 4 >= 2 || grid_size >= 8) {
    return 4;
  }
  if (grid_size % 2 == 0 || grid_size >= 4) {
    return 2;
  }
  return 1;
}

int3 GetWorkGroupSizeForApple(const uint3& grid_size) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStaskPSwork_group_pickingDTcc mht_8(mht_8_v, 381, "", "./tensorflow/lite/delegates/gpu/common/task/work_group_picking.cc", "GetWorkGroupSizeForApple");

  int x_size = GetOptimalSizeForApple(grid_size.x);
  int y_size = GetOptimalSizeForApple(grid_size.y);
  int z_size = std::max(1, 32 / (x_size * y_size));
  z_size = std::min(z_size, static_cast<int>(grid_size.z));
  return {x_size, y_size, z_size};
}

}  // namespace

int3 GetWorkGroupXY128ConvLinear(const int3& grid) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStaskPSwork_group_pickingDTcc mht_9(mht_9_v, 394, "", "./tensorflow/lite/delegates/gpu/common/task/work_group_picking.cc", "GetWorkGroupXY128ConvLinear");

  int grid_z = GetBiggestDividerWithPriority(grid.z, 4);
  if (grid.x <= 128) {
    return int3(128, 1, grid_z);
  }
  int grid_x = GetMaxSizeWithMinPenalty(grid.x, 512 / grid_z);
  return {grid_x, 1, grid_z};
}

int3 GetWorkGroupXY128Conv(const int3& grid) {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStaskPSwork_group_pickingDTcc mht_10(mht_10_v, 406, "", "./tensorflow/lite/delegates/gpu/common/task/work_group_picking.cc", "GetWorkGroupXY128Conv");

  int grid_z = GetBiggestDividerWithPriority(grid.z, 4);
  if (grid.x <= 16 && grid.y <= 8) {
    return int3(16, 8, grid_z);
  }
  int2 grid_xy = GetMaxSizeWithMinPenalty(int2(grid.x, grid.y), 512 / grid_z);
  return int3(grid_xy.x, grid_xy.y, grid_z);
}

int3 GetWorkGroupXY128Simple(const int3& grid) {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStaskPSwork_group_pickingDTcc mht_11(mht_11_v, 418, "", "./tensorflow/lite/delegates/gpu/common/task/work_group_picking.cc", "GetWorkGroupXY128Simple");
 return int3(16, 8, 1); }

int3 GetWorkGroup(const int3& grid, int max_size) {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStaskPSwork_group_pickingDTcc mht_12(mht_12_v, 423, "", "./tensorflow/lite/delegates/gpu/common/task/work_group_picking.cc", "GetWorkGroup");

  int wg_z = GetBiggestDividerWithPriority(grid.z, 8);
  int wg_xy_size = max_size / wg_z;
  int wg_x = std::min(DivideRoundUp(grid.x, 2), wg_xy_size);
  int wg_y = std::min(wg_xy_size / wg_x, grid.y);
  return int3(wg_x, wg_y, wg_z);
}

int3 GetWorkGroupConv(const int3& grid, int max_size, int max_z_size) {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStaskPSwork_group_pickingDTcc mht_13(mht_13_v, 434, "", "./tensorflow/lite/delegates/gpu/common/task/work_group_picking.cc", "GetWorkGroupConv");

  int wg_z = GetBiggestDivider(grid.z, max_z_size);
  int wg_xy_size = std::min(256, max_size) / wg_z;
  int wg_x = std::min(grid.x, wg_xy_size);
  int wg_y = std::min(wg_xy_size / wg_x, grid.y);
  if (wg_y == grid.y && grid.y % 2 == 0) {
    wg_y = grid.y / 2;
  }
  return int3(wg_x, wg_y, wg_z);
}

void GetPossibleWorkGroupsXYMultipleOf(int multiplier, const GpuInfo& gpu_info,
                                       const KernelInfo& kernel_info,
                                       const int3& grid,
                                       WorkGroupSizeAlignment z_alignment,
                                       std::vector<int3>* work_groups) {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStaskPSwork_group_pickingDTcc mht_14(mht_14_v, 452, "", "./tensorflow/lite/delegates/gpu/common/task/work_group_picking.cc", "GetPossibleWorkGroupsXYMultipleOf");

  *work_groups = GenerateWorkGroupSizesXYMultipleOf(
      multiplier, grid, kernel_info, gpu_info, z_alignment);
}

void GetPossibleWorkGroupsXMultipleOf(int multiplier, const GpuInfo& gpu_info,
                                      const KernelInfo& kernel_info,
                                      const int3& grid,
                                      WorkGroupSizeAlignment z_alignment,
                                      std::vector<int3>* work_groups) {
   std::vector<std::string> mht_15_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStaskPSwork_group_pickingDTcc mht_15(mht_15_v, 464, "", "./tensorflow/lite/delegates/gpu/common/task/work_group_picking.cc", "GetPossibleWorkGroupsXMultipleOf");

  *work_groups = GenerateWorkGroupSizesXMultipleOf(
      multiplier, grid, kernel_info, gpu_info, z_alignment);
}

bool XY128RequiresMoreWorkGroupsThenXY128Linear(int width, int height) {
   std::vector<std::string> mht_16_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStaskPSwork_group_pickingDTcc mht_16(mht_16_v, 472, "", "./tensorflow/lite/delegates/gpu/common/task/work_group_picking.cc", "XY128RequiresMoreWorkGroupsThenXY128Linear");

  int planar_work_groups = DivideRoundUp(width * height, 128);
  auto base_work_groups = Get2DWorkgroupsEqualTo128();
  bool have_equal_work_groups = false;
  for (auto& work_group : base_work_groups) {
    int x_groups = DivideRoundUp(width, work_group.x);
    int y_groups = DivideRoundUp(height, work_group.y);
    int xy_groups = x_groups * y_groups;
    if (xy_groups == planar_work_groups) {
      have_equal_work_groups = true;
      break;
    }
  }
  return !have_equal_work_groups;
}

void GetPossibleWorkGroups(TuningType tuning_type, const GpuInfo& gpu_info,
                           const KernelInfo& kernel_info, const int3& grid,
                           std::vector<int3>* work_groups) {
   std::vector<std::string> mht_17_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStaskPSwork_group_pickingDTcc mht_17(mht_17_v, 493, "", "./tensorflow/lite/delegates/gpu/common/task/work_group_picking.cc", "GetPossibleWorkGroups");

  if (gpu_info.IsApple()) {
    work_groups->push_back(GetWorkGroupSizeForApple(grid));
    return;
  }
  switch (tuning_type) {
    case TuningType::kFast:
      work_groups->push_back(
          GetWorkGroup(grid, kernel_info.max_work_group_size));
      return;
    case TuningType::kExhaustive: {
      GetWorkGroupsAlignedToGrid(gpu_info, kernel_info, grid, work_groups);
      return;
    }
    default:
      work_groups->push_back({8, 4, 1});
      return;
  }
}

void GetPossibleWorkGroupsConv(TuningType tuning_type, const GpuInfo& gpu_info,
                               const KernelInfo& kernel_info, const int3& grid,
                               std::vector<int3>* work_groups) {
   std::vector<std::string> mht_18_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStaskPSwork_group_pickingDTcc mht_18(mht_18_v, 518, "", "./tensorflow/lite/delegates/gpu/common/task/work_group_picking.cc", "GetPossibleWorkGroupsConv");

  if (gpu_info.IsApple()) {
    work_groups->push_back(GetWorkGroupSizeForApple(grid));
    return;
  }
  switch (tuning_type) {
    case TuningType::kFast: {
      int max_z_size = 16;
      if (gpu_info.IsAdreno()) {
        max_z_size = gpu_info.adreno_info.IsAdreno3xx() ? 16 : 64;
      }
      max_z_size = std::min(max_z_size, gpu_info.GetMaxWorkGroupSizeForZ());
      work_groups->push_back(
          GetWorkGroupConv(grid, kernel_info.max_work_group_size, max_z_size));
      return;
    }
    case TuningType::kExhaustive: {
      GetWorkGroupsAlignedToGrid(gpu_info, kernel_info, grid, work_groups);
      return;
    }
    default:
      work_groups->push_back({8, 4, 1});
      return;
  }
}

int3 GetFirstSuitableWorkGroup(const std::vector<int3>& wgs, int max_wg_size) {
   std::vector<std::string> mht_19_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStaskPSwork_group_pickingDTcc mht_19(mht_19_v, 547, "", "./tensorflow/lite/delegates/gpu/common/task/work_group_picking.cc", "GetFirstSuitableWorkGroup");

  for (const auto& wg : wgs) {
    const int wg_size = wg.x * wg.y * wg.z;
    if (wg_size <= max_wg_size) {
      return wg;
    }
  }
  return {1, 1, 1};
}

}  // namespace gpu
}  // namespace tflite
