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
class MHTracer_DTPStensorflowPSstream_executorPSkernel_specDTcc {
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
   MHTracer_DTPStensorflowPSstream_executorPSkernel_specDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSstream_executorPSkernel_specDTcc() {
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

/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/stream_executor/kernel_spec.h"
#include "absl/strings/string_view.h"

namespace stream_executor {

KernelLoaderSpec::KernelLoaderSpec(absl::string_view kernelname)
    : kernelname_(std::string(kernelname)) {
   std::vector<std::string> mht_0_v;
   mht_0_v.push_back("kernelname: \"" + std::string(kernelname.data(), kernelname.size()) + "\"");
   MHTracer_DTPStensorflowPSstream_executorPSkernel_specDTcc mht_0(mht_0_v, 192, "", "./tensorflow/stream_executor/kernel_spec.cc", "KernelLoaderSpec::KernelLoaderSpec");
}

OnDiskKernelLoaderSpec::OnDiskKernelLoaderSpec(absl::string_view filename,
                                               absl::string_view kernelname)
    : KernelLoaderSpec(kernelname), filename_(std::string(filename)) {
   std::vector<std::string> mht_1_v;
   mht_1_v.push_back("filename: \"" + std::string(filename.data(), filename.size()) + "\"");
   mht_1_v.push_back("kernelname: \"" + std::string(kernelname.data(), kernelname.size()) + "\"");
   MHTracer_DTPStensorflowPSstream_executorPSkernel_specDTcc mht_1(mht_1_v, 201, "", "./tensorflow/stream_executor/kernel_spec.cc", "OnDiskKernelLoaderSpec::OnDiskKernelLoaderSpec");
}

CudaPtxOnDisk::CudaPtxOnDisk(absl::string_view filename,
                             absl::string_view kernelname)
    : OnDiskKernelLoaderSpec(filename, kernelname) {
   std::vector<std::string> mht_2_v;
   mht_2_v.push_back("filename: \"" + std::string(filename.data(), filename.size()) + "\"");
   mht_2_v.push_back("kernelname: \"" + std::string(kernelname.data(), kernelname.size()) + "\"");
   MHTracer_DTPStensorflowPSstream_executorPSkernel_specDTcc mht_2(mht_2_v, 210, "", "./tensorflow/stream_executor/kernel_spec.cc", "CudaPtxOnDisk::CudaPtxOnDisk");
}

CudaCubinOnDisk::CudaCubinOnDisk(absl::string_view filename,
                                 absl::string_view kernelname)
    : OnDiskKernelLoaderSpec(filename, kernelname) {
   std::vector<std::string> mht_3_v;
   mht_3_v.push_back("filename: \"" + std::string(filename.data(), filename.size()) + "\"");
   mht_3_v.push_back("kernelname: \"" + std::string(kernelname.data(), kernelname.size()) + "\"");
   MHTracer_DTPStensorflowPSstream_executorPSkernel_specDTcc mht_3(mht_3_v, 219, "", "./tensorflow/stream_executor/kernel_spec.cc", "CudaCubinOnDisk::CudaCubinOnDisk");
}

CudaCubinInMemory::CudaCubinInMemory(const char *bytes,
                                     absl::string_view kernelname)
    : KernelLoaderSpec(kernelname), bytes_(bytes) {
   std::vector<std::string> mht_4_v;
   mht_4_v.push_back("bytes: \"" + (bytes == nullptr ? std::string("nullptr") : std::string((char*)bytes)) + "\"");
   mht_4_v.push_back("kernelname: \"" + std::string(kernelname.data(), kernelname.size()) + "\"");
   MHTracer_DTPStensorflowPSstream_executorPSkernel_specDTcc mht_4(mht_4_v, 228, "", "./tensorflow/stream_executor/kernel_spec.cc", "CudaCubinInMemory::CudaCubinInMemory");
}

bool CompareComputeCapability(const std::tuple<int, int> &lhs,
                              const std::tuple<int, int> &rhs) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPSstream_executorPSkernel_specDTcc mht_5(mht_5_v, 234, "", "./tensorflow/stream_executor/kernel_spec.cc", "CompareComputeCapability");

  return std::get<0>(lhs) < std::get<0>(rhs) ||
         (std::get<0>(lhs) == std::get<0>(rhs) &&
          std::get<1>(lhs) < std::get<1>(rhs));
}

const std::tuple<int, int> CudaPtxInMemory::kMinimumCapability{1, 0};

CudaPtxInMemory::CudaPtxInMemory(absl::string_view ptx,
                                 absl::string_view kernel_name,
                                 bool ptx_compressed)
    : KernelLoaderSpec(kernel_name),
      ptx_by_compute_capability_(CompareComputeCapability) {
   std::vector<std::string> mht_6_v;
   mht_6_v.push_back("ptx: \"" + std::string(ptx.data(), ptx.size()) + "\"");
   mht_6_v.push_back("kernel_name: \"" + std::string(kernel_name.data(), kernel_name.size()) + "\"");
   MHTracer_DTPStensorflowPSstream_executorPSkernel_specDTcc mht_6(mht_6_v, 251, "", "./tensorflow/stream_executor/kernel_spec.cc", "CudaPtxInMemory::CudaPtxInMemory");

  if (ptx_compressed) {
    // Lazy decompression. Put an empty string in decompressed_ptx_ showing that
    // the original ptx is compressed.
    decompressed_ptx_[ptx.data()] = "";
  }
  ptx_by_compute_capability_[kMinimumCapability] = ptx.data();
}

CudaPtxInMemory::CudaPtxInMemory(
    const std::initializer_list<CudaPtxInMemory::PtxSpec> &spec_list,
    absl::string_view kernel_name, bool ptx_compressed)
    : KernelLoaderSpec(kernel_name),
      ptx_by_compute_capability_(CompareComputeCapability) {
   std::vector<std::string> mht_7_v;
   mht_7_v.push_back("kernel_name: \"" + std::string(kernel_name.data(), kernel_name.size()) + "\"");
   MHTracer_DTPStensorflowPSstream_executorPSkernel_specDTcc mht_7(mht_7_v, 268, "", "./tensorflow/stream_executor/kernel_spec.cc", "CudaPtxInMemory::CudaPtxInMemory");

  for (const auto &spec : spec_list) {
    int major, minor;
    absl::string_view ptx;
    std::tie(major, minor, ptx) = spec;
    if (ptx_compressed) {
      // Lazy decompression. Put an empty string in decompressed_ptx_ showing
      // that the original ptx is compressed.
      decompressed_ptx_[ptx.data()] = "";
    }
    ptx_by_compute_capability_[std::tuple<int, int>{major, minor}] = ptx.data();
  }
}

std::string CudaPtxInMemory::DecompressPtx(const char *ptx) {
   std::vector<std::string> mht_8_v;
   mht_8_v.push_back("ptx: \"" + (ptx == nullptr ? std::string("nullptr") : std::string((char*)ptx)) + "\"");
   MHTracer_DTPStensorflowPSstream_executorPSkernel_specDTcc mht_8(mht_8_v, 286, "", "./tensorflow/stream_executor/kernel_spec.cc", "CudaPtxInMemory::DecompressPtx");

  // Get the length of the PTX string from the beginning of the buffer.
  uint64_t ptx_length = *reinterpret_cast<const uint64 *>(ptx);
  // Get the PTX string from the buffer with offset and length.
  std::string compressed_ptx(ptx + sizeof(uint64_t),
                             ptx + sizeof(uint64_t) + ptx_length);
  std::string decompressed_ptx;
  // Decompress the PTX string with bzip2.
  LOG(FATAL) << "bzip2 decompression is not supported yet.";
  return decompressed_ptx;
}

const char *CudaPtxInMemory::default_text() const {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPSstream_executorPSkernel_specDTcc mht_9(mht_9_v, 301, "", "./tensorflow/stream_executor/kernel_spec.cc", "CudaPtxInMemory::default_text");

  if (ptx_by_compute_capability_.empty()) {
    return nullptr;
  }

  absl::MutexLock lock(&mu_);

  auto ptx = ptx_by_compute_capability_.begin()->second;
  // Check if there is an entry in decompressed ptx table.
  auto decompressed_ptx_iter = decompressed_ptx_.find(ptx);
  if (decompressed_ptx_iter != decompressed_ptx_.end()) {
    // If the decompressed string is empty, which means the ptx hasn't been
    // decompressed, decompress it here.
    if (decompressed_ptx_iter->second.empty()) {
      decompressed_ptx_iter->second = DecompressPtx(ptx);
    }
    return decompressed_ptx_iter->second.c_str();
  }
  return ptx;
}

const char *CudaPtxInMemory::original_default_text() const {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPSstream_executorPSkernel_specDTcc mht_10(mht_10_v, 325, "", "./tensorflow/stream_executor/kernel_spec.cc", "CudaPtxInMemory::original_default_text");

  if (ptx_by_compute_capability_.empty()) {
    return nullptr;
  }

  return ptx_by_compute_capability_.begin()->second;
}

const char *CudaPtxInMemory::text(int compute_capability_major,
                                  int compute_capability_minor) const {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPSstream_executorPSkernel_specDTcc mht_11(mht_11_v, 337, "", "./tensorflow/stream_executor/kernel_spec.cc", "CudaPtxInMemory::text");

  std::tuple<int, int> capability{compute_capability_major,
                                  compute_capability_minor};

  auto ptx_iter = ptx_by_compute_capability_.find(capability);
  if (ptx_iter == ptx_by_compute_capability_.end()) {
    return nullptr;
  }

  absl::MutexLock lock(&mu_);

  // Check if there is an entry in decompressed ptx table.
  auto decompressed_ptx_iter = decompressed_ptx_.find(ptx_iter->second);
  if (decompressed_ptx_iter != decompressed_ptx_.end()) {
    // If the decompressed string is empty, which means the ptx hasn't been
    // decompressed, decompress it here.
    if (decompressed_ptx_iter->second.empty()) {
      decompressed_ptx_iter->second = DecompressPtx(ptx_iter->second);
    }
    return decompressed_ptx_iter->second.c_str();
  }
  return ptx_iter->second;
}

const char *CudaPtxInMemory::original_text(int compute_capability_major,
                                           int compute_capability_minor) const {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPSstream_executorPSkernel_specDTcc mht_12(mht_12_v, 365, "", "./tensorflow/stream_executor/kernel_spec.cc", "CudaPtxInMemory::original_text");

  std::tuple<int, int> capability{compute_capability_major,
                                  compute_capability_minor};

  auto ptx_iter = ptx_by_compute_capability_.find(capability);
  if (ptx_iter == ptx_by_compute_capability_.end()) {
    return nullptr;
  }

  return ptx_iter->second;
}

OpenCLTextOnDisk::OpenCLTextOnDisk(absl::string_view filename,
                                   absl::string_view kernelname)
    : OnDiskKernelLoaderSpec(filename, kernelname) {
   std::vector<std::string> mht_13_v;
   mht_13_v.push_back("filename: \"" + std::string(filename.data(), filename.size()) + "\"");
   mht_13_v.push_back("kernelname: \"" + std::string(kernelname.data(), kernelname.size()) + "\"");
   MHTracer_DTPStensorflowPSstream_executorPSkernel_specDTcc mht_13(mht_13_v, 384, "", "./tensorflow/stream_executor/kernel_spec.cc", "OpenCLTextOnDisk::OpenCLTextOnDisk");
}

OpenCLTextInMemory::OpenCLTextInMemory(absl::string_view text,
                                       absl::string_view kernelname)
    : KernelLoaderSpec(kernelname), text_(text) {
   std::vector<std::string> mht_14_v;
   mht_14_v.push_back("text: \"" + std::string(text.data(), text.size()) + "\"");
   mht_14_v.push_back("kernelname: \"" + std::string(kernelname.data(), kernelname.size()) + "\"");
   MHTracer_DTPStensorflowPSstream_executorPSkernel_specDTcc mht_14(mht_14_v, 393, "", "./tensorflow/stream_executor/kernel_spec.cc", "OpenCLTextInMemory::OpenCLTextInMemory");
}

OpenCLBinaryOnDisk::OpenCLBinaryOnDisk(absl::string_view filename,
                                       absl::string_view kernelname)
    : OnDiskKernelLoaderSpec(filename, kernelname) {
   std::vector<std::string> mht_15_v;
   mht_15_v.push_back("filename: \"" + std::string(filename.data(), filename.size()) + "\"");
   mht_15_v.push_back("kernelname: \"" + std::string(kernelname.data(), kernelname.size()) + "\"");
   MHTracer_DTPStensorflowPSstream_executorPSkernel_specDTcc mht_15(mht_15_v, 402, "", "./tensorflow/stream_executor/kernel_spec.cc", "OpenCLBinaryOnDisk::OpenCLBinaryOnDisk");
}

MultiKernelLoaderSpec *MultiKernelLoaderSpec::AddOpenCLTextOnDisk(
    absl::string_view filename, absl::string_view kernelname) {
   std::vector<std::string> mht_16_v;
   mht_16_v.push_back("filename: \"" + std::string(filename.data(), filename.size()) + "\"");
   mht_16_v.push_back("kernelname: \"" + std::string(kernelname.data(), kernelname.size()) + "\"");
   MHTracer_DTPStensorflowPSstream_executorPSkernel_specDTcc mht_16(mht_16_v, 410, "", "./tensorflow/stream_executor/kernel_spec.cc", "MultiKernelLoaderSpec::AddOpenCLTextOnDisk");

  CHECK(ocl_text_on_disk_ == nullptr);
  ocl_text_on_disk_.reset(new OpenCLTextOnDisk{filename, kernelname});
  return this;
}

MultiKernelLoaderSpec *MultiKernelLoaderSpec::AddOpenCLBinaryOnDisk(
    absl::string_view filename, absl::string_view kernelname) {
   std::vector<std::string> mht_17_v;
   mht_17_v.push_back("filename: \"" + std::string(filename.data(), filename.size()) + "\"");
   mht_17_v.push_back("kernelname: \"" + std::string(kernelname.data(), kernelname.size()) + "\"");
   MHTracer_DTPStensorflowPSstream_executorPSkernel_specDTcc mht_17(mht_17_v, 422, "", "./tensorflow/stream_executor/kernel_spec.cc", "MultiKernelLoaderSpec::AddOpenCLBinaryOnDisk");

  CHECK(ocl_binary_on_disk_ == nullptr);
  ocl_binary_on_disk_.reset(new OpenCLBinaryOnDisk{filename, kernelname});
  return this;
}

MultiKernelLoaderSpec *MultiKernelLoaderSpec::AddOpenCLTextInMemory(
    absl::string_view filename, absl::string_view kernelname) {
   std::vector<std::string> mht_18_v;
   mht_18_v.push_back("filename: \"" + std::string(filename.data(), filename.size()) + "\"");
   mht_18_v.push_back("kernelname: \"" + std::string(kernelname.data(), kernelname.size()) + "\"");
   MHTracer_DTPStensorflowPSstream_executorPSkernel_specDTcc mht_18(mht_18_v, 434, "", "./tensorflow/stream_executor/kernel_spec.cc", "MultiKernelLoaderSpec::AddOpenCLTextInMemory");

  CHECK(ocl_text_in_memory_ == nullptr);
  ocl_text_in_memory_.reset(new OpenCLTextInMemory{filename, kernelname});
  return this;
}

MultiKernelLoaderSpec *MultiKernelLoaderSpec::AddCudaPtxOnDisk(
    absl::string_view filename, absl::string_view kernelname) {
   std::vector<std::string> mht_19_v;
   mht_19_v.push_back("filename: \"" + std::string(filename.data(), filename.size()) + "\"");
   mht_19_v.push_back("kernelname: \"" + std::string(kernelname.data(), kernelname.size()) + "\"");
   MHTracer_DTPStensorflowPSstream_executorPSkernel_specDTcc mht_19(mht_19_v, 446, "", "./tensorflow/stream_executor/kernel_spec.cc", "MultiKernelLoaderSpec::AddCudaPtxOnDisk");

  CHECK(cuda_ptx_on_disk_ == nullptr);
  cuda_ptx_on_disk_.reset(new CudaPtxOnDisk{filename, kernelname});
  return this;
}

MultiKernelLoaderSpec *MultiKernelLoaderSpec::AddCudaCubinInMemory(
    const char *bytes, absl::string_view kernelname) {
   std::vector<std::string> mht_20_v;
   mht_20_v.push_back("bytes: \"" + (bytes == nullptr ? std::string("nullptr") : std::string((char*)bytes)) + "\"");
   mht_20_v.push_back("kernelname: \"" + std::string(kernelname.data(), kernelname.size()) + "\"");
   MHTracer_DTPStensorflowPSstream_executorPSkernel_specDTcc mht_20(mht_20_v, 458, "", "./tensorflow/stream_executor/kernel_spec.cc", "MultiKernelLoaderSpec::AddCudaCubinInMemory");

  CHECK(cuda_cubin_in_memory_ == nullptr);
  cuda_cubin_in_memory_.reset(new CudaCubinInMemory{bytes, kernelname});
  return this;
}

MultiKernelLoaderSpec *MultiKernelLoaderSpec::AddCudaCubinOnDisk(
    absl::string_view filename, absl::string_view kernelname) {
   std::vector<std::string> mht_21_v;
   mht_21_v.push_back("filename: \"" + std::string(filename.data(), filename.size()) + "\"");
   mht_21_v.push_back("kernelname: \"" + std::string(kernelname.data(), kernelname.size()) + "\"");
   MHTracer_DTPStensorflowPSstream_executorPSkernel_specDTcc mht_21(mht_21_v, 470, "", "./tensorflow/stream_executor/kernel_spec.cc", "MultiKernelLoaderSpec::AddCudaCubinOnDisk");

  CHECK(cuda_cubin_on_disk_ == nullptr);
  cuda_cubin_on_disk_.reset(new CudaCubinOnDisk{filename, kernelname});
  return this;
}

MultiKernelLoaderSpec *MultiKernelLoaderSpec::AddCudaPtxInMemory(
    absl::string_view ptx, absl::string_view kernelname) {
   std::vector<std::string> mht_22_v;
   mht_22_v.push_back("ptx: \"" + std::string(ptx.data(), ptx.size()) + "\"");
   mht_22_v.push_back("kernelname: \"" + std::string(kernelname.data(), kernelname.size()) + "\"");
   MHTracer_DTPStensorflowPSstream_executorPSkernel_specDTcc mht_22(mht_22_v, 482, "", "./tensorflow/stream_executor/kernel_spec.cc", "MultiKernelLoaderSpec::AddCudaPtxInMemory");

  CHECK(cuda_ptx_in_memory_ == nullptr);
  cuda_ptx_in_memory_.reset(
      new CudaPtxInMemory{ptx, kernelname, false /* ptx_compressed */});
  return this;
}

MultiKernelLoaderSpec *MultiKernelLoaderSpec::AddCudaCompressedPtxInMemory(
    absl::string_view ptx, absl::string_view kernelname) {
   std::vector<std::string> mht_23_v;
   mht_23_v.push_back("ptx: \"" + std::string(ptx.data(), ptx.size()) + "\"");
   mht_23_v.push_back("kernelname: \"" + std::string(kernelname.data(), kernelname.size()) + "\"");
   MHTracer_DTPStensorflowPSstream_executorPSkernel_specDTcc mht_23(mht_23_v, 495, "", "./tensorflow/stream_executor/kernel_spec.cc", "MultiKernelLoaderSpec::AddCudaCompressedPtxInMemory");

  CHECK(cuda_ptx_in_memory_ == nullptr);
  cuda_ptx_in_memory_.reset(
      new CudaPtxInMemory{ptx, kernelname, true /* ptx_compressed */});
  return this;
}

MultiKernelLoaderSpec *MultiKernelLoaderSpec::AddCudaPtxInMemory(
    std::initializer_list<CudaPtxInMemory::PtxSpec> spec_list,
    absl::string_view kernelname) {
   std::vector<std::string> mht_24_v;
   mht_24_v.push_back("kernelname: \"" + std::string(kernelname.data(), kernelname.size()) + "\"");
   MHTracer_DTPStensorflowPSstream_executorPSkernel_specDTcc mht_24(mht_24_v, 508, "", "./tensorflow/stream_executor/kernel_spec.cc", "MultiKernelLoaderSpec::AddCudaPtxInMemory");

  CHECK(cuda_ptx_in_memory_ == nullptr);
  cuda_ptx_in_memory_.reset(
      new CudaPtxInMemory{spec_list, kernelname, false /* ptx_compressed */});
  return this;
}

MultiKernelLoaderSpec *MultiKernelLoaderSpec::AddCudaCompressedPtxInMemory(
    std::initializer_list<CudaPtxInMemory::PtxSpec> spec_list,
    absl::string_view kernelname) {
   std::vector<std::string> mht_25_v;
   mht_25_v.push_back("kernelname: \"" + std::string(kernelname.data(), kernelname.size()) + "\"");
   MHTracer_DTPStensorflowPSstream_executorPSkernel_specDTcc mht_25(mht_25_v, 521, "", "./tensorflow/stream_executor/kernel_spec.cc", "MultiKernelLoaderSpec::AddCudaCompressedPtxInMemory");

  CHECK(cuda_ptx_in_memory_ == nullptr);
  cuda_ptx_in_memory_.reset(
      new CudaPtxInMemory{spec_list, kernelname, true /* ptx_compressed */});
  return this;
}

MultiKernelLoaderSpec::MultiKernelLoaderSpec(size_t arity) : arity_(arity) {
   std::vector<std::string> mht_26_v;
   MHTracer_DTPStensorflowPSstream_executorPSkernel_specDTcc mht_26(mht_26_v, 531, "", "./tensorflow/stream_executor/kernel_spec.cc", "MultiKernelLoaderSpec::MultiKernelLoaderSpec");
}

}  // namespace stream_executor
