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
class MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSclPSprogram_cacheDTcc {
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
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSclPSprogram_cacheDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSclPSprogram_cacheDTcc() {
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

#include "tensorflow/lite/delegates/gpu/cl/program_cache.h"

#include <cstdint>
#include <string>
#include <utility>

#include "flatbuffers/flatbuffers.h"  // from @flatbuffers
#include "tensorflow/lite/delegates/gpu/cl/cl_program.h"
#include "tensorflow/lite/delegates/gpu/cl/compiled_program_cache_generated.h"
#include "tensorflow/lite/delegates/gpu/cl/util.h"
#include "tensorflow/lite/delegates/gpu/common/status.h"
#include <farmhash.h>

namespace tflite {
namespace gpu {
namespace cl {
namespace {

// Farmhash Fingerprint
inline uint64_t CombineFingerprints(uint64_t l, uint64_t h) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSclPSprogram_cacheDTcc mht_0(mht_0_v, 204, "", "./tensorflow/lite/delegates/gpu/cl/program_cache.cc", "CombineFingerprints");

  // Murmur-inspired hashing.
  const uint64_t kMul = 0x9ddfea08eb382d69ULL;
  uint64_t a = (l ^ h) * kMul;
  a ^= (a >> 47);
  uint64_t b = (h ^ a) * kMul;
  b ^= (b >> 44);
  b *= kMul;
  b ^= (b >> 41);
  b *= kMul;
  return b;
}

uint64_t GetProgramFingerprint(const std::string& code,
                               const std::string& compiler_options) {
   std::vector<std::string> mht_1_v;
   mht_1_v.push_back("code: \"" + code + "\"");
   mht_1_v.push_back("compiler_options: \"" + compiler_options + "\"");
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSclPSprogram_cacheDTcc mht_1(mht_1_v, 223, "", "./tensorflow/lite/delegates/gpu/cl/program_cache.cc", "GetProgramFingerprint");

  const uint64_t code_fingerprint = ::util::Fingerprint64(code);
  const uint64_t options_fingerprint =
      ::util::Fingerprint64(compiler_options);
  return CombineFingerprints(code_fingerprint, options_fingerprint);
}

std::string GetDriverVersion(const CLDevice& device) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSclPSprogram_cacheDTcc mht_2(mht_2_v, 233, "", "./tensorflow/lite/delegates/gpu/cl/program_cache.cc", "GetDriverVersion");

  return device.GetPlatformVersion() + "_jet_version_0";
}

}  // namespace

ProgramCache::ProgramDescriptor::ProgramDescriptor(
    const std::string& code, const std::string& compiler_options)
    : fingerprint(GetProgramFingerprint(code, compiler_options)) {
   std::vector<std::string> mht_3_v;
   mht_3_v.push_back("code: \"" + code + "\"");
   mht_3_v.push_back("compiler_options: \"" + compiler_options + "\"");
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSclPSprogram_cacheDTcc mht_3(mht_3_v, 246, "", "./tensorflow/lite/delegates/gpu/cl/program_cache.cc", "ProgramCache::ProgramDescriptor::ProgramDescriptor");
}

ProgramCache::ProgramDescriptor::ProgramDescriptor(uint64_t fingerprints)
    : fingerprint(fingerprints) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSclPSprogram_cacheDTcc mht_4(mht_4_v, 252, "", "./tensorflow/lite/delegates/gpu/cl/program_cache.cc", "ProgramCache::ProgramDescriptor::ProgramDescriptor");
}

ProgramCache::ProgramCache(ProgramCache&& program_cache)
    : programs_(std::move(program_cache.programs_)) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSclPSprogram_cacheDTcc mht_5(mht_5_v, 258, "", "./tensorflow/lite/delegates/gpu/cl/program_cache.cc", "ProgramCache::ProgramCache");
}

ProgramCache& ProgramCache::operator=(ProgramCache&& program_cache) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSclPSprogram_cacheDTcc mht_6(mht_6_v, 263, "", "./tensorflow/lite/delegates/gpu/cl/program_cache.cc", "=");

  if (this != &program_cache) {
    programs_ = std::move(program_cache.programs_);
  }
  return *this;
}

absl::Status ProgramCache::GetOrCreateCLKernel(
    const std::string& code, const std::string& function_name,
    const std::vector<CompilerOptions>& compiler_options,
    const CLContext& context, const CLDevice& device, CLKernel* result,
    uint64_t* kernel_fingerprint) {
   std::vector<std::string> mht_7_v;
   mht_7_v.push_back("code: \"" + code + "\"");
   mht_7_v.push_back("function_name: \"" + function_name + "\"");
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSclPSprogram_cacheDTcc mht_7(mht_7_v, 279, "", "./tensorflow/lite/delegates/gpu/cl/program_cache.cc", "ProgramCache::GetOrCreateCLKernel");

  const std::string options =
      CompilerOptionsToString(device.GetInfo(), compiler_options);
  ProgramDescriptor desc(code, options);
  if (kernel_fingerprint) {
    *kernel_fingerprint = desc.fingerprint;
  }
  auto it = programs_.find(desc);
  if (it != programs_.end()) {
    return result->CreateFromProgram(it->second, function_name);
  }

  CLProgram program;
  RETURN_IF_ERROR(CreateCLProgram(code, options, context, device, &program));
  RETURN_IF_ERROR(result->CreateFromProgram(program, function_name));
  programs_.insert(std::make_pair(std::move(desc), std::move(program)));
  return absl::OkStatus();
}

absl::Status ProgramCache::GetOrCreateCLKernel(const std::string& code,
                                               const std::string& function_name,
                                               const CLContext& context,
                                               const CLDevice& device,
                                               CLKernel* result,
                                               uint64_t* kernel_fingerprint) {
   std::vector<std::string> mht_8_v;
   mht_8_v.push_back("code: \"" + code + "\"");
   mht_8_v.push_back("function_name: \"" + function_name + "\"");
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSclPSprogram_cacheDTcc mht_8(mht_8_v, 308, "", "./tensorflow/lite/delegates/gpu/cl/program_cache.cc", "ProgramCache::GetOrCreateCLKernel");

  return GetOrCreateCLKernel(code, function_name, {}, context, device, result,
                             kernel_fingerprint);
}

absl::Status ProgramCache::GetKernel(uint64_t fingerprint,
                                     const std::string& function_name,
                                     CLKernel* result) const {
   std::vector<std::string> mht_9_v;
   mht_9_v.push_back("function_name: \"" + function_name + "\"");
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSclPSprogram_cacheDTcc mht_9(mht_9_v, 319, "", "./tensorflow/lite/delegates/gpu/cl/program_cache.cc", "ProgramCache::GetKernel");

  ProgramDescriptor desc(fingerprint);
  auto it = programs_.find(desc);
  if (it == programs_.end()) {
    return absl::NotFoundError("No program with this fingerprint.");
  }
  return result->CreateFromProgram(it->second, function_name);
}

absl::Status ProgramCache::AddProgramBinary(const CLContext& context,
                                            const CLDevice& device,
                                            uint64_t fingerprint,
                                            absl::Span<const uint8_t> binary) {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSclPSprogram_cacheDTcc mht_10(mht_10_v, 334, "", "./tensorflow/lite/delegates/gpu/cl/program_cache.cc", "ProgramCache::AddProgramBinary");

  ProgramDescriptor desc(fingerprint);
  auto it = programs_.find(desc);
  if (it == programs_.end()) {
    CLProgram program;
    RETURN_IF_ERROR(
        CreateCLProgramFromBinary(context, device, binary, &program));
    programs_.insert(std::make_pair(std::move(desc), std::move(program)));
  }
  return absl::OkStatus();
}

absl::Status ProgramCache::GetProgramBinary(
    uint64_t fingerprint, std::vector<uint8_t>* program_binary) const {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSclPSprogram_cacheDTcc mht_11(mht_11_v, 350, "", "./tensorflow/lite/delegates/gpu/cl/program_cache.cc", "ProgramCache::GetProgramBinary");

  ProgramDescriptor desc(fingerprint);
  auto it = programs_.find(desc);
  if (it == programs_.end()) {
    return absl::NotFoundError("No program with this fingerprint.");
  }
  return it->second.GetBinary(program_binary);
}

absl::Status ProgramCache::AddSerializedCache(
    const CLContext& context, const CLDevice& device,
    absl::Span<const uint8_t> serialized_cache) {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSclPSprogram_cacheDTcc mht_12(mht_12_v, 364, "", "./tensorflow/lite/delegates/gpu/cl/program_cache.cc", "ProgramCache::AddSerializedCache");

  flatbuffers::Verifier verifier(serialized_cache.data(),
                                 serialized_cache.size());
  if (!data::VerifyCompiledCacheBuffer(verifier)) {
    return absl::InvalidArgumentError("Serialized model is corrupted.");
  }

  auto model = data::GetCompiledCache(serialized_cache.data());
  std::string platform_version(model->driver_version()->c_str(),
                               model->driver_version()->size());

  if (GetDriverVersion(device) != platform_version) {
    return absl::InvalidArgumentError(
        "OpenCL driver changed, cache invalid, should be regenerated");
  }

  for (auto serialized_program : *model->programs()) {
    auto binary_span = absl::MakeSpan(serialized_program->binary()->data(),
                                      serialized_program->binary()->size());
    RETURN_IF_ERROR(AddProgramBinary(
        context, device, serialized_program->fingerprint(), binary_span));
  }
  return absl::OkStatus();
}

absl::Status ProgramCache::GetSerializedCache(
    const CLDevice& device, std::vector<uint8_t>* serialized_cache) const {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSclPSprogram_cacheDTcc mht_13(mht_13_v, 393, "", "./tensorflow/lite/delegates/gpu/cl/program_cache.cc", "ProgramCache::GetSerializedCache");

  ::flatbuffers::FlatBufferBuilder builder;
  std::vector<flatbuffers::Offset<data::Program>> serialized_programs;
  for (auto& program : programs_) {
    std::vector<uint8_t> binary;
    RETURN_IF_ERROR(program.second.GetBinary(&binary));
    auto binary_offset = builder.CreateVector(binary);
    data::ProgramBuilder program_builder(builder);
    program_builder.add_fingerprint(program.first.fingerprint);
    program_builder.add_binary(binary_offset);
    serialized_programs.push_back(program_builder.Finish());
  }
  auto driver_version = builder.CreateString(GetDriverVersion(device));
  auto programs_s = builder.CreateVector(serialized_programs);
  data::CompiledCacheBuilder cache_builder(builder);
  cache_builder.add_driver_version(driver_version);
  cache_builder.add_programs(programs_s);
  data::FinishCompiledCacheBuffer(builder, cache_builder.Finish());
  size_t next_element = serialized_cache->size();
  serialized_cache->resize(serialized_cache->size() + builder.GetSize());
  std::memcpy(&(*serialized_cache)[next_element], builder.GetBufferPointer(),
              builder.GetSize());
  return absl::OkStatus();
}

}  // namespace cl
}  // namespace gpu
}  // namespace tflite
