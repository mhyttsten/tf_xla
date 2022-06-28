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

#ifndef TENSORFLOW_STREAM_EXECUTOR_TPU_C_API_CONVERSIONS_H_
#define TENSORFLOW_STREAM_EXECUTOR_TPU_C_API_CONVERSIONS_H_
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
class MHTracer_DTPStensorflowPSstream_executorPStpuPSc_api_conversionsDTh {
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
   MHTracer_DTPStensorflowPSstream_executorPStpuPSc_api_conversionsDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSstream_executorPStpuPSc_api_conversionsDTh() {
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


#include "absl/container/inlined_vector.h"
#include "tensorflow/compiler/xla/executable_run_options.h"
#include "tensorflow/compiler/xla/literal.h"
#include "tensorflow/compiler/xla/service/hlo_module.h"
#include "tensorflow/compiler/xla/service/hlo_module_config.h"
#include "tensorflow/compiler/xla/service/maybe_owning_device_memory.h"
#include "tensorflow/compiler/xla/service/service_executable_run_options.h"
#include "tensorflow/compiler/xla/service/shaped_buffer.h"
#include "tensorflow/compiler/xla/shape.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/stream_executor/device_memory.h"
#include "tensorflow/stream_executor/device_memory_allocator.h"
#include "tensorflow/stream_executor/tpu/c_api_decl.h"
#include "tensorflow/stream_executor/tpu/tpu_executor_c_api.h"

// APIs for converting between internal and external versions of
// XLA/StreamExecutor data structures.
namespace ApiConverter {

absl::Span<const float> MakeSpan(const FloatList& src_list);
void CreateVector(const absl::Span<const float> src, FloatList* dst);
void Destroy(FloatList* float_list);

// se::DeviceMemoryBase
SE_DeviceMemoryBase ToC(const stream_executor::DeviceMemoryBase& base);
void ToC(const stream_executor::DeviceMemoryBase& base,
         SE_DeviceMemoryBase* se_base);
stream_executor::DeviceMemoryBase FromC(const SE_DeviceMemoryBase& se_base);
void Destroy(SE_DeviceMemoryBase*);

// xla::Shape
xla::Shape FromC(const XLA_Shape* c_shape);
void ToC(const xla::Shape& xla_shape, XLA_Shape* c_shape);
void Destroy(XLA_Shape* c_shape);

// xla::Layout
xla::Layout FromC(const XLA_Layout* c_layout);
void ToC(const xla::Layout& xla_layout, XLA_Layout* c_layout);
void Destroy(XLA_Layout* c_layout);

// xla::Tile
xla::Tile FromC(const XLA_Tile* c_tile);
void ToC(const xla::Tile& xla_tile, XLA_Tile* c_tile);
void Destroy(XLA_Tile* c_tile);

// xla::ShapeIndex
XLA_ShapeIndex ToC(const xla::ShapeIndex& xla_shape);
xla::ShapeIndex FromC(XLA_ShapeIndex* c_shape);
void Destroy(XLA_ShapeIndex*);

// Literal
void ToC(const xla::LiteralSlice& literal, XLA_Literal* c_literal);
xla::MutableBorrowingLiteral FromC(XLA_Literal* c_literal);
void Destroy(XLA_Literal* c_literal);

// ShapedBuffer
void ToC(const xla::ShapedBuffer& buffer, XLA_ShapedBuffer* c_device_buffer);
xla::ShapedBuffer FromC(XLA_ShapedBuffer* c_buffer);
void Destroy(XLA_ShapedBuffer* c_buffer);

// se::DeviceMemoryBase
SE_DeviceMemoryBase ToC(const stream_executor::DeviceMemoryBase& base);
stream_executor::DeviceMemoryBase FromC(const SE_DeviceMemoryBase& se_base);
void Destroy(SE_DeviceMemoryBase*);

// Literal
void ToC(const xla::LiteralSlice& literal, XLA_Literal* c_literal);
xla::MutableBorrowingLiteral FromC(XLA_Literal* c_literal);
void Destroy(XLA_Literal* c_literal);

// ShapedBuffer
void ToC(const xla::ShapedBuffer& buffer, XLA_ShapedBuffer* c_device_buffer);
xla::ShapedBuffer FromC(XLA_ShapedBuffer* c_buffer);
void Destroy(XLA_ShapedBuffer* c_buffer);

// TpuEmbeddingEngineParametersData
struct TpuEmbeddingEngineParametersData {
  // Backing vector for struct
  std::array<std::vector<FloatListRef*>, 8> vectors;
  TpuEmbeddingEngineParameters c_params;
};

std::unique_ptr<TpuEmbeddingEngineParametersData> Create(int num_tables);

xla::MaybeOwningDeviceMemory FromC(
    SE_MaybeOwningDeviceMemory* se_mem,
    stream_executor::DeviceMemoryAllocator* allocator);

// DeviceMemoryAllocator
SE_DeviceMemoryAllocator ToC(stream_executor::DeviceMemoryAllocator* allocator);

// OwningDeviceMemory
SE_MaybeOwningDeviceMemory ToC(stream_executor::OwningDeviceMemory* mem);
// mem.HasOwnership() may be true if the buffer is aliased and shouldn't be
// released. 'aliased' should be true in this case. 'aliased' has no effect if
// 'mem' is unowned.
SE_MaybeOwningDeviceMemory ToC(xla::MaybeOwningDeviceMemory& mem, bool aliased);

// HloModule
XLA_HloModule ToC(const xla::HloModule& module);
xla::StatusOr<std::unique_ptr<xla::HloModule>> FromC(
    const XLA_HloModule& c_module);
void Destroy(XLA_HloModule* c_module);

// HloModuleConfig
XLA_HloModuleConfig ToC(const xla::HloModuleConfig& config);
xla::HloModuleConfig FromC(const XLA_HloModuleConfig& c_config);
void Destroy(XLA_HloModuleConfig* c_config);

// Helper for managing stack based C -> C++ conversions.
template <class CType>
struct StackHelper {
  explicit StackHelper() {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSstream_executorPStpuPSc_api_conversionsDTh mht_0(mht_0_v, 300, "", "./tensorflow/stream_executor/tpu/c_api_conversions.h", "StackHelper");
}

  template <class CppType>
  explicit StackHelper(const CppType& t) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSstream_executorPStpuPSc_api_conversionsDTh mht_1(mht_1_v, 306, "", "./tensorflow/stream_executor/tpu/c_api_conversions.h", "StackHelper");

    ::ApiConverter::ToC(t, &value);
  }
  ~StackHelper() {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSstream_executorPStpuPSc_api_conversionsDTh mht_2(mht_2_v, 312, "", "./tensorflow/stream_executor/tpu/c_api_conversions.h", "~StackHelper");
 ::ApiConverter::Destroy(&value); }

  template <class CppType>
  CppType AsCpp() const {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPSstream_executorPStpuPSc_api_conversionsDTh mht_3(mht_3_v, 318, "", "./tensorflow/stream_executor/tpu/c_api_conversions.h", "AsCpp");

    return ::ApiConverter::FromC(&value);
  }

  mutable CType value;
};

}  // namespace ApiConverter

#endif
