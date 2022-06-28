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
class MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPStestsPShlo_to_llvm_irDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPStestsPShlo_to_llvm_irDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPStestsPShlo_to_llvm_irDTcc() {
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

#include "llvm/Target/TargetMachine.h"
#include "tensorflow/compiler/xla/service/gpu/gpu_compiler.h"
#include "tensorflow/compiler/xla/service/gpu/gpu_device_info.h"
#include "tensorflow/compiler/xla/service/gpu/llvm_gpu_backend/gpu_backend_lib.h"
#if GOOGLE_CUDA
#include "tensorflow/compiler/xla/service/gpu/nvptx_compiler.h"
#include "tensorflow/compiler/xla/service/gpu/nvptx_helper.h"
#endif
#include "tensorflow/compiler/xla/service/gpu/target_constants.h"
#include "tensorflow/compiler/xla/service/hlo_module.h"
#include "tensorflow/compiler/xla/status.h"
#include "tensorflow/compiler/xla/tools/hlo_module_loader.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/util/command_line_flags.h"
#include "tensorflow/stream_executor/cuda/cuda_platform_id.h"

const char* const kUsage = R"(
This tool reads in an HloModule from a file, compiles it using the NVPTX
compiler and prints out the LLVM IR generated by the IR emitter.  The LLVM IR is
not optimized by the LLVM pass pipeline, so this tool can be used to unit test
the XLA GPU IR emitters.

Note that the LLVM IR does not contain the *full* module, but only parts that
will be code generated into PTX.  The NVPTX compiler also generates a
GpuExecutable on the side that is not printed.

When passed the parameter `--ptx`, the LLVM IR will be optimized and PTX
will be emitted and printed instead of the non-optimized LLVM.
By default SM 70 is targeted. But this can be changed with `--sm=SM`.)";

namespace {
xla::Status CompileAndPrintLlvmIr(const std::string& hlo_text,
                                  bool generate_ptx, int sm) {
   std::vector<std::string> mht_0_v;
   mht_0_v.push_back("hlo_text: \"" + hlo_text + "\"");
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPStestsPShlo_to_llvm_irDTcc mht_0(mht_0_v, 219, "", "./tensorflow/compiler/xla/service/gpu/tests/hlo_to_llvm_ir.cc", "CompileAndPrintLlvmIr");

  TF_ASSIGN_OR_RETURN(
      std::unique_ptr<xla::HloModule> hlo_module,
      xla::LoadModuleFromData(/*data=*/hlo_text, /*format=*/"hlo"));
  llvm::LLVMContext llvm_context;

  // For now we pretend we're compiling for V100.  This can be generalized
  // later.

  xla::gpu::GpuDeviceInfo gpu_device_info{};
  gpu_device_info.threads_per_block_limit = 1024;
  gpu_device_info.threads_per_warp = 32;
  gpu_device_info.shared_memory_per_block = 49152;
  gpu_device_info.core_count = 80;
  gpu_device_info.threads_per_core_limit = 2048;
  gpu_device_info.block_dim_limit_x = 2147483647;
  gpu_device_info.block_dim_limit_y = 65535;
  gpu_device_info.block_dim_limit_z = 65535;

  tensorflow::se::CudaComputeCapability cuda_compute_capability;
  cuda_compute_capability.major = sm / 10;
  cuda_compute_capability.minor = sm % 10;
  tensorflow::se::RocmComputeCapability rocm_compute_capability("gfx908");
  std::string target_triple = "nvptx64-nvidia-cuda";
  std::string datalayout = "nvptx64-nvidia-cuda";
  std::string platform_name = "CUDA";
  stream_executor::Platform::Id platform_id =
      stream_executor::cuda::kCudaPlatformId;
  TF_ASSIGN_OR_RETURN(std::unique_ptr<llvm::Module> llvm_module,
                      xla::gpu::CompileModuleToLlvmIr(
                          hlo_module.get(), &llvm_context,
                          /*target_triple=*/xla::gpu::nvptx::TargetTriple(),
                          /*data_layout=*/xla::gpu::nvptx::DataLayout(),
                          /*platform_name=*/platform_name,
                          /*platform_id=*/platform_id, gpu_device_info,
                          cuda_compute_capability, rocm_compute_capability,
                          /*pointer_size=*/8));

  if (!generate_ptx) {
    llvm_module->print(llvm::outs(), nullptr);
  } else {
#if GOOGLE_CUDA
    std::string libdevice_dir = xla::gpu::GetLibdeviceDir(hlo_module->config());
    TF_ASSIGN_OR_RETURN(std::string ptx,
                        xla::gpu::nvptx::CompileToPtx(
                            llvm_module.get(), cuda_compute_capability,
                            hlo_module->config(), libdevice_dir));
    std::cout << ptx << std::endl;
#else
    return {tensorflow::error::UNIMPLEMENTED,
            "Feature not yet implemented in ROCm"};
#endif
  }
  return xla::Status::OK();
}

xla::Status CompileAndPrintLlvmIrFromFile(const std::string& file_name,
                                          bool ptx, int sm) {
   std::vector<std::string> mht_1_v;
   mht_1_v.push_back("file_name: \"" + file_name + "\"");
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPStestsPShlo_to_llvm_irDTcc mht_1(mht_1_v, 280, "", "./tensorflow/compiler/xla/service/gpu/tests/hlo_to_llvm_ir.cc", "CompileAndPrintLlvmIrFromFile");

  std::string full_text;
  TF_RETURN_IF_ERROR(tensorflow::ReadFileToString(tensorflow::Env::Default(),
                                                  file_name, &full_text));

  std::vector<std::string> hlo_module_texts =
      absl::StrSplit(full_text, "// -----");
  for (const std::string& hlo_module_text : hlo_module_texts) {
    TF_RETURN_IF_ERROR(CompileAndPrintLlvmIr(hlo_module_text, ptx, sm));
  }

  return xla::Status::OK();
}
}  // namespace

int main(int argc, char** argv) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPStestsPShlo_to_llvm_irDTcc mht_2(mht_2_v, 298, "", "./tensorflow/compiler/xla/service/gpu/tests/hlo_to_llvm_ir.cc", "main");

  bool ptx = false;
  int sm = 70;
  std::vector<tensorflow::Flag> flag_list;
  xla::AppendDebugOptionsFlags(&flag_list);
  flag_list.emplace_back("ptx", &ptx,
                         "Print PTX instead of not optimized LLVM.");
  flag_list.emplace_back("sm", &sm,
                         "Specify the SM to target (useful only with --ptx).");
  // The usage string includes the message at the top of the file, the
  // DebugOptions flags and the flags defined above.
  const std::string kUsageString = absl::StrCat(
      kUsage, "\n\n", tensorflow::Flags::Usage(argv[0], flag_list));
  bool parse_ok = tensorflow::Flags::Parse(&argc, argv, flag_list);
  tensorflow::port::InitMain(kUsageString.c_str(), &argc, &argv);
  if (!parse_ok) {
    LOG(QFATAL) << kUsageString;
  }

  QCHECK(argc == 2) << "Must specify a single input file";
  TF_CHECK_OK(CompileAndPrintLlvmIrFromFile(argv[1], ptx, sm));

  return 0;
}
