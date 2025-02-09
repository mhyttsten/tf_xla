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
class MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSglPSkernelsPSconcatDTcc {
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
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSglPSkernelsPSconcatDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSglPSkernelsPSconcatDTcc() {
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

#include "tensorflow/lite/delegates/gpu/gl/kernels/concat.h"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

#include "absl/memory/memory.h"
#include "tensorflow/lite/delegates/gpu/common/status.h"
#include "tensorflow/lite/delegates/gpu/common/types.h"
#include "tensorflow/lite/delegates/gpu/gl/variable.h"

namespace tflite {
namespace gpu {
namespace gl {
namespace {

class AlignedConcatByChannels : public NodeShader {
 public:
  static bool IsSupported(const GenerationContext& ctx) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSglPSkernelsPSconcatDTcc mht_0(mht_0_v, 205, "", "./tensorflow/lite/delegates/gpu/gl/kernels/concat.cc", "IsSupported");

    const auto& attr = absl::any_cast<const ConcatAttributes&>(ctx.op_attr);

    // Implementation supports concatenation by channels only.
    if (attr.axis != Axis::CHANNELS) return false;

    // Implementation supports concatenation of 2 tensors only.
    if (ctx.input_shapes.size() != 2) return false;

    // H and W must be the same for every concatenated tensor.
    for (int i = 1; i < ctx.input_shapes.size(); i++) {
      if (ctx.input_shapes[0][1] != ctx.input_shapes[i][1] ||
          ctx.input_shapes[0][2] != ctx.input_shapes[i][2]) {
        return false;
      }
    }

    // Channels must be aligned by 4 for every concatenated tensor.
    for (const auto& shape : ctx.input_shapes) {
      if (shape[3] % 4 != 0) return false;
    }

    return true;
  }

  absl::Status GenerateCode(const GenerationContext& ctx,
                            GeneratedCode* generated_code) const final {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSglPSkernelsPSconcatDTcc mht_1(mht_1_v, 234, "", "./tensorflow/lite/delegates/gpu/gl/kernels/concat.cc", "GenerateCode");

    if (!IsSupported(ctx)) {
      return absl::InvalidArgumentError(
          "This case is not supported by aligned concat");
    }

    // Shader below concatenates 2 tensors which channels are aligned by 4
    std::string source = R"(
      if (gid.z < $border$) {
        value_0 = $input_data_0[gid.x, gid.y, gid.z]$;
      } else {
        int z = gid.z - $border$;
        value_0 = $input_data_1[gid.x, gid.y, z]$;
      }
)";
    *generated_code = {
        /*parameters=*/{
            {"border", static_cast<int>(ctx.input_shapes[0][3]) / 4}},
        /*objects=*/{},
        /*shared_variables=*/{},
        /*workload=*/uint3(),
        /*workgroup=*/uint3(),
        /*source_code=*/std::move(source),
        /*input=*/IOStructure::ONLY_DEFINITIONS,
        /*output=*/IOStructure::AUTO,
    };
    return absl::OkStatus();
  }
};

class ConcatByAnyChannel : public NodeShader {
 public:
  static bool IsSupported(const GenerationContext& ctx) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSglPSkernelsPSconcatDTcc mht_2(mht_2_v, 269, "", "./tensorflow/lite/delegates/gpu/gl/kernels/concat.cc", "IsSupported");

    const auto& attr = absl::any_cast<const ConcatAttributes&>(ctx.op_attr);

    // Implementation supports concatenation by channels only.
    if (attr.axis != Axis::CHANNELS) return false;

    // Implementation supports concatenation of more that 1 tensors only.
    if (ctx.input_shapes.size() <= 1) return false;

    // H and W must be the same for every concatenated tensor.
    for (int i = 1; i < ctx.input_shapes.size(); i++) {
      if (ctx.input_shapes[0][1] != ctx.input_shapes[i][1] ||
          ctx.input_shapes[0][2] != ctx.input_shapes[i][2]) {
        return false;
      }
    }

    return true;
  }

  absl::Status GenerateCode(const GenerationContext& ctx,
                            GeneratedCode* generated_code) const final {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSglPSkernelsPSconcatDTcc mht_3(mht_3_v, 293, "", "./tensorflow/lite/delegates/gpu/gl/kernels/concat.cc", "GenerateCode");

    if (!IsSupported(ctx)) {
      return absl::UnimplementedError("This case is not supported by concat");
    }

    std::string code = DeclareVariables();

    // "already_written" is used to keep the amount of already joined channels
    int already_written = 0;
    // "t" is an id of the next temp* variable.
    // Generally, temp* variables are used in macros
    // READ_BUFFER_VEC4(buff, addr, var).
    // This macros instantiate the variable "var" and
    // reads the value from buffer "buff" by address "addr"
    int t = 0;
    for (int current_input_id = 0; current_input_id < ctx.input_shapes.size();
         current_input_id++) {
      // Start joining next inout tensor

      // Grab channels amount
      int in_ch = ctx.input_shapes[current_input_id][3];
      code += PrintStartMessage(current_input_id, in_ch, already_written);

      // Construct the buffer name associated with this tensor
      std::string input = "input_data_" + std::to_string(current_input_id);

      // "reminder" shows us how many cells in 4-element vector are left after
      // the last write. As example, if we join two tensors both with
      // 3 channels, after joining the first one we come to this line again
      // and, when joining the second tensor, the reminder value
      // will be equal to 1
      int reminder = already_written % 4;

      if (reminder == 0) {
        code += AlignedCase(in_ch, input);
      } else {
        code += UnalignedCase(reminder, in_ch, input, &t);
      }
      already_written += in_ch;
    }

    *generated_code = {
        /*parameters=*/{},
        /*objects=*/{},
        /*shared_variables=*/{},
        /*workload=*/
        uint3(static_cast<int>(ctx.output_shapes[0][2]),
              static_cast<int>(ctx.output_shapes[0][1]), 1),
        /*workgroup=*/uint3(),
        /*source_code=*/std::move(code),
        /*input=*/IOStructure::ONLY_DEFINITIONS,
        /*output=*/IOStructure::ONLY_DEFINITIONS,
    };
    return absl::OkStatus();
  }

 private:
  // Utility function
  std::string temp(int t) const {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSglPSkernelsPSconcatDTcc mht_4(mht_4_v, 354, "", "./tensorflow/lite/delegates/gpu/gl/kernels/concat.cc", "temp");
 return "temp" + std::to_string(t); }

  std::string DeclareVariables() const {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSglPSkernelsPSconcatDTcc mht_5(mht_5_v, 359, "", "./tensorflow/lite/delegates/gpu/gl/kernels/concat.cc", "DeclareVariables");

    // "val" is used to collect useful information before the next
    // upcoming write.
    return R"(
int z = gid.z;
vec4 val = vec4(0.0f);

)";
  }

  std::string PrintStartMessage(int current_input_id, int in_ch,
                                int already_written) const {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSglPSkernelsPSconcatDTcc mht_6(mht_6_v, 373, "", "./tensorflow/lite/delegates/gpu/gl/kernels/concat.cc", "PrintStartMessage");

    return "//              Joining " + std::to_string(current_input_id) +
           " tensor with " + std::to_string(in_ch) +
           " channels\n//  * * * *\\n// Already wrote " +
           std::to_string(already_written) + " elements\n\n";
  }

  std::string AlignedCase(int in_ch, const std::string& input) const {
   std::vector<std::string> mht_7_v;
   mht_7_v.push_back("input: \"" + input + "\"");
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSglPSkernelsPSconcatDTcc mht_7(mht_7_v, 384, "", "./tensorflow/lite/delegates/gpu/gl/kernels/concat.cc", "AlignedCase");

    std::string code;
    // This branch is for aligned reading and writing, when we can copy
    // all 4 components at once. Address of the first element to write
    // should be aligned.
    // Visual examples:
    // 1) when copy input_data_0
    //
    //       | * * * * | * * * @ | @ @ . . .
    //         ^
    // 2) when in the middle of joining process:
    //
    //       | X X X X | * * * @ | @ @ . . .
    //                   ^
    // Note that amount of * equals to the in_ch
    //
    // X - cells were written before
    // * - you are going to write into these cells
    // @ - you will fill these cells next cycles
    // ^ - first elem you start writing from
    int blocks_amount = DivideRoundUp<int>(in_ch, 4);
    code += "// Aligned case\n";
    code += "// I'm going to make " + std::to_string(blocks_amount) +
            " write(s)\n\n";
    for (int block = 0; block < blocks_amount; block++) {
      // Copy full 4-element vector
      code += "val = $" + input + "[gid.x, gid.y, " + std::to_string(block) +
              "]$;\n" +
              "$output_data_0[gid.x, gid.y, z] = val$;\n"
              // calculate next address to write
              + "z++; \n\n";
    }
    return code;
  }

  std::string UnalignedCase(int reminder, int in_ch, const std::string& input,
                            int* t) const {
   std::vector<std::string> mht_8_v;
   mht_8_v.push_back("input: \"" + input + "\"");
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSglPSkernelsPSconcatDTcc mht_8(mht_8_v, 424, "", "./tensorflow/lite/delegates/gpu/gl/kernels/concat.cc", "UnalignedCase");

    // This branch is for copying cell-by-cell. It will never start from the
    // first tensor input_data_0. This function is splitting in two stages:
    // 1) Copy the "leftovers" for the previous cells
    // 2) Copy all other
    // Visual examples:
    //
    //        Stage 1       Stage 2
    //        -----------   -------------------------
    // . . X | X  X  X *1 | *2 *2 *2  @ | @  @  . . .
    //               ^
    // . . X | X  X *1 *1 | *2 *2 *2 *2 | *2 *2 . . .
    //             ^
    // . . X | X *1 *1 *1 | *2  @  @  @ | @  @  . . .
    //           ^
    // Note that amount of * equals to the in_ch
    //
    // X - cells were written before
    // *1 - write there at the Stage 1
    // *2 - write there at the Stage 2
    // @ - you will fill these cells next cycles
    // ^ - first elem you start writing from

    std::string code = "// Unaligned case\n";

    // Variable "shift" showes how many "empty" cells are left after previous
    // write. Remember, that this case should is unaligned.
    // shift now can only be 1, 2 or 3
    int shift = 4 - reminder;
    if (shift > in_ch) {
      shift = in_ch;
    }
    code += "\n// Stage 1\n";
    code += "vec4 " + temp(*t) + " = $" + input + "[gid.x, gid.y, 0]$;\n";
    for (int i = 0; i < shift; i++) {
      // Note that reminder + i has implicitly added 1, cause
      // reminder by it's nature is an amount, not an index
      code += "val[" + std::to_string(reminder + i) + "] = " + temp(*t) + "[" +
              std::to_string(i) + "];\n";
    }
    // Rewrite previous value with updated last cells
    code += "$output_data_0[gid.x, gid.y, z - 1] = val$;\n";
    (*t)++;

    // "left_blocks" is equal to an amount of WRITE_BUFFER_VEC4 calls
    // which will are left for this input to be finally copied
    int left_blocks = (in_ch - shift) / 4;
    if ((in_ch - shift) % 4 != 0) {
      left_blocks++;
    }
    if (left_blocks) {
      code += "\n// Stage 2\n";
      for (int block = 0; block < left_blocks; block++) {
        for (int elem = 0; elem < 4; elem++) {
          if (shift % 4 == 0) {
            code += "vec4 " + temp(*t) + " = $" + input + "[gid.x, gid.y, " +
                    std::to_string(block + 1) + "]$;\n";
            (*t)++;
          }
          code += "val[" + std::to_string(elem) + "] = " + temp(*t - 1) + "[" +
                  std::to_string(shift % 4) + "];\n";
          if (shift == in_ch) {
            break;
          }
          shift++;
        }
        code += "$output_data_0[gid.x, gid.y, z] = val$;\n";
        code += "z++;\n";
      }
    } else {
      code += "// No Stage 2\n";
    }
    return code;
  }
};

class FlatConcatByHeight : public NodeShader {
 public:
  static bool IsSupported(const GenerationContext& ctx) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSglPSkernelsPSconcatDTcc mht_9(mht_9_v, 505, "", "./tensorflow/lite/delegates/gpu/gl/kernels/concat.cc", "IsSupported");

    const auto& attr = absl::any_cast<const ConcatAttributes&>(ctx.op_attr);

    // Implementation supports concatenation by height only.
    if (attr.axis != Axis::HEIGHT) return false;

    // Implementation supports concatenation of more that 1 tensors only.
    if (ctx.input_shapes.size() <= 1) return false;

    // C and W must be the same for every concatenated tensor.
    for (int i = 1; i < ctx.input_shapes.size(); i++) {
      if (ctx.input_shapes[0][3] != ctx.input_shapes[i][3] ||
          ctx.input_shapes[0][2] != ctx.input_shapes[i][2]) {
        return false;
      }
    }

    return true;
  }

  absl::Status GenerateCode(const GenerationContext& ctx,
                            GeneratedCode* generated_code) const final {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSglPSkernelsPSconcatDTcc mht_10(mht_10_v, 529, "", "./tensorflow/lite/delegates/gpu/gl/kernels/concat.cc", "GenerateCode");

    std::string code;
    std::vector<Variable> params;
    for (int i = 0, shift = 0; i < ctx.input_shapes.size();
         shift += ctx.input_shapes[i][1], i++) {
      code += "if (";
      if (i != 0) {
        code += "$input_data_" + std::to_string(i - 1) + "_h$ <= gid.y && ";
      }
      code +=
          "gid.y < " + std::to_string(shift + ctx.input_shapes[i][1]) + ") {\n";
      code += "if (gid.y - " + std::to_string(shift) + " >= $input_data_" +
              std::to_string(i) + "_h$) return;\n";
      code += "value_0 = $input_data_" + std::to_string(i) +
              "[gid.x, gid.y - " + std::to_string(shift) + ", gid.z]$;\n}\n";
      if (i != ctx.input_shapes.size() - 1) {
        code += " else ";
      }
      params.push_back({"input_data_" + std::to_string(i) + "_h",
                        static_cast<int>(ctx.input_shapes[i][1])});
    }

    *generated_code = {
        /*parameters=*/std::move(params),
        /*objects=*/{},
        /*shared_variables=*/{},
        /*workload=*/uint3(),
        /*workgroup=*/uint3(),
        /*source_code=*/std::move(code),
        /*input=*/IOStructure::ONLY_DEFINITIONS,
        /*output=*/IOStructure::AUTO,
    };
    return absl::OkStatus();
  }
};

class FlatConcatByWidth : public NodeShader {
 public:
  static bool IsSupported(const GenerationContext& ctx) {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSglPSkernelsPSconcatDTcc mht_11(mht_11_v, 570, "", "./tensorflow/lite/delegates/gpu/gl/kernels/concat.cc", "IsSupported");

    const auto& attr = absl::any_cast<const ConcatAttributes&>(ctx.op_attr);

    // Implementation supports concatenation by width only.
    if (attr.axis != Axis::WIDTH) return false;

    // Implementation supports concatenation of more that 1 tensors only.
    if (ctx.input_shapes.size() <= 1) return false;

    // C and H must be the same for every concatenated tensor.
    for (int i = 1; i < ctx.input_shapes.size(); i++) {
      if (ctx.input_shapes[0][3] != ctx.input_shapes[i][3] ||
          ctx.input_shapes[0][1] != ctx.input_shapes[i][1]) {
        return false;
      }
    }

    return true;
  }

  absl::Status GenerateCode(const GenerationContext& ctx,
                            GeneratedCode* generated_code) const final {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSglPSkernelsPSconcatDTcc mht_12(mht_12_v, 594, "", "./tensorflow/lite/delegates/gpu/gl/kernels/concat.cc", "GenerateCode");

    std::string code;
    std::vector<Variable> params;
    for (int i = 0, shift = 0; i < ctx.input_shapes.size();
         shift += ctx.input_shapes[i][2], i++) {
      code += "if (";
      if (i != 0) {
        code += "$input_data_" + std::to_string(i - 1) + "_w$ <= gid.x && ";
      }
      code +=
          "gid.x < " + std::to_string(shift + ctx.input_shapes[i][2]) + ") {\n";
      code += "if (gid.x - " + std::to_string(shift) + " >= $input_data_" +
              std::to_string(i) + "_w$) return;\n";
      code += "value_0 = $input_data_" + std::to_string(i) + "[gid.x - " +
              std::to_string(shift) + ", gid.y, gid.z]$;\n}\n";
      if (i != ctx.input_shapes.size() - 1) {
        code += " else ";
      }
      params.push_back({"input_data_" + std::to_string(i) + "_w",
                        static_cast<int>(ctx.input_shapes[i][2])});
    }

    *generated_code = {
        /*parameters=*/std::move(params),
        /*objects=*/{},
        /*shared_variables=*/{},
        /*workload=*/uint3(),
        /*workgroup=*/uint3(),
        /*source_code=*/std::move(code),
        /*input=*/IOStructure::ONLY_DEFINITIONS,
        /*output=*/IOStructure::AUTO,
    };
    return absl::OkStatus();
  }
};

class FlatConcat : public NodeShader {
 public:
  absl::Status GenerateCode(const GenerationContext& ctx,
                            GeneratedCode* generated_code) const final {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSglPSkernelsPSconcatDTcc mht_13(mht_13_v, 636, "", "./tensorflow/lite/delegates/gpu/gl/kernels/concat.cc", "GenerateCode");

    if (FlatConcatByHeight::IsSupported(ctx)) {
      return flat_concat_by_height_.GenerateCode(ctx, generated_code);
    }
    if (FlatConcatByWidth::IsSupported(ctx)) {
      return flat_concat_by_width_.GenerateCode(ctx, generated_code);
    }
    return absl::InvalidArgumentError(
        "This case is not supported by flat concat");
  }

 private:
  FlatConcatByHeight flat_concat_by_height_;
  FlatConcatByWidth flat_concat_by_width_;
};

}  // namespace

std::unique_ptr<NodeShader> NewAlignedConcatNodeShader() {
  return absl::make_unique<AlignedConcatByChannels>();
}

std::unique_ptr<NodeShader> NewConcatNodeShader() {
  return absl::make_unique<ConcatByAnyChannel>();
}

std::unique_ptr<NodeShader> NewFlatConcatNodeShader() {
  return absl::make_unique<FlatConcat>();
}

}  // namespace gl
}  // namespace gpu
}  // namespace tflite
