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
class MHTracer_DTPStensorflowPSstream_executorPStpuPSc_api_conversionsDTcc {
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
   MHTracer_DTPStensorflowPSstream_executorPStpuPSc_api_conversionsDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSstream_executorPStpuPSc_api_conversionsDTcc() {
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

#include "tensorflow/stream_executor/tpu/c_api_conversions.h"

#include <memory>
#include <vector>

#include "tensorflow/core/tpu/tpu_api.h"
#include "tensorflow/stream_executor/tpu/c_api_decl.h"
#include "tensorflow/stream_executor/tpu/c_api_defn.h"
#include "tensorflow/stream_executor/tpu/tpu_executor_c_api.h"
#include "tensorflow/stream_executor/tpu/tpu_platform_interface.h"

namespace ApiConverter {

xla::ShapedBuffer FromC(XLA_ShapedBuffer* c_buffer) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSstream_executorPStpuPSc_api_conversionsDTcc mht_0(mht_0_v, 198, "", "./tensorflow/stream_executor/tpu/c_api_conversions.cc", "FromC");

  xla::Shape xla_on_device_shape =
      ApiConverter::FromC(&c_buffer->on_device_shape);

  xla::ShapeTree<stream_executor::DeviceMemoryBase> xla_shape_tree(
      xla_on_device_shape);
  size_t i = 0;
  for (auto& pair : xla_shape_tree) {
    pair.second = ApiConverter::FromC(c_buffer->bases[i]);
    i++;
  }

  xla::ShapedBuffer xla_shaped_buffer(xla_on_device_shape,
                                      c_buffer->device_ordinal);
  xla_shaped_buffer.set_buffers(xla_shape_tree);
  return xla_shaped_buffer;
}

SE_MaybeOwningDeviceMemory ToC(xla::MaybeOwningDeviceMemory& mem,
                               bool aliased) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSstream_executorPStpuPSc_api_conversionsDTcc mht_1(mht_1_v, 220, "", "./tensorflow/stream_executor/tpu/c_api_conversions.cc", "ToC");

  SE_MaybeOwningDeviceMemory se_mem;
  se_mem.owned = mem.HasOwnership();
  se_mem.memory = ApiConverter::ToC(mem.AsDeviceMemoryBase());
  if (mem.HasOwnership()) {
    const stream_executor::OwningDeviceMemory* owned =
        mem.AsOwningDeviceMemory();
    se_mem.device_ordinal = owned->device_ordinal();
    se_mem.allocator = ApiConverter::ToC(owned->allocator());
    if (!aliased) {
      // Underlying buffer is owned by se_mem now.
      mem.Release()->Release();
    }
  } else {
    se_mem.allocator =
        ToC(static_cast<stream_executor::DeviceMemoryAllocator*>(nullptr));
    se_mem.device_ordinal = -1;
  }
  return se_mem;
}

xla::MaybeOwningDeviceMemory FromC(
    SE_MaybeOwningDeviceMemory* se_mem,
    stream_executor::DeviceMemoryAllocator* allocator) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSstream_executorPStpuPSc_api_conversionsDTcc mht_2(mht_2_v, 246, "", "./tensorflow/stream_executor/tpu/c_api_conversions.cc", "FromC");

  if (se_mem->owned) {
    return xla::MaybeOwningDeviceMemory(
        stream_executor::OwningDeviceMemory(ApiConverter::FromC(se_mem->memory),
                                            se_mem->device_ordinal, allocator));
  } else {
    return xla::MaybeOwningDeviceMemory(ApiConverter::FromC(se_mem->memory));
  }
}

SE_DeviceMemoryAllocator ToC(
    stream_executor::DeviceMemoryAllocator* allocator) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPSstream_executorPStpuPSc_api_conversionsDTcc mht_3(mht_3_v, 260, "", "./tensorflow/stream_executor/tpu/c_api_conversions.cc", "ToC");

  SE_DeviceMemoryAllocator se_allocator;
  if (allocator == nullptr) {
    se_allocator.ctx = nullptr;
    se_allocator.platform = nullptr;
    se_allocator.allocate = nullptr;
    se_allocator.deallocate = nullptr;
    return se_allocator;
  }
  // N.B. Platform is assumed to be the registered backend platform.
  se_allocator.platform = nullptr;
  se_allocator.ctx = allocator;
  se_allocator.allocate = [](void* ctx, int device_ordinal, uint64_t size,
                             bool retry_on_failure, int64_t memory_space,
                             SE_ScopedDeviceMemory* memory,
                             TF_Status* se_status) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPSstream_executorPStpuPSc_api_conversionsDTcc mht_4(mht_4_v, 278, "", "./tensorflow/stream_executor/tpu/c_api_conversions.cc", "lambda");

    auto allocation =
        reinterpret_cast<stream_executor::DeviceMemoryAllocator*>(ctx)
            ->Allocate(device_ordinal, size, retry_on_failure, memory_space);
    if (!allocation.ok()) {
      auto status = allocation.status();
      tensorflow::tpu::ExecutorApiFn()->TpuStatus_SetFn(
          se_status, status.code(), status.error_message().data(),
          status.error_message().size());
    } else {
      auto& scoped_memory = allocation.ValueOrDie();
      memory->wrapped = ApiConverter::ToC(scoped_memory.Release());
      memory->device_ordinal = scoped_memory.device_ordinal();
    }
  };

  se_allocator.deallocate = [](void* ctx, SE_DeviceMemoryBase* base,
                               int device_ordinal, TF_Status* se_status) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPSstream_executorPStpuPSc_api_conversionsDTcc mht_5(mht_5_v, 298, "", "./tensorflow/stream_executor/tpu/c_api_conversions.cc", "lambda");

    auto status = reinterpret_cast<stream_executor::DeviceMemoryAllocator*>(ctx)
                      ->Deallocate(device_ordinal, ApiConverter::FromC(*base));
    if (!status.ok()) {
      tensorflow::tpu::ExecutorApiFn()->TpuStatus_SetFn(
          se_status, status.code(), status.error_message().data(),
          status.error_message().size());
    }
  };
  return se_allocator;
}

SE_MaybeOwningDeviceMemory ToC(stream_executor::OwningDeviceMemory* mem) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPSstream_executorPStpuPSc_api_conversionsDTcc mht_6(mht_6_v, 313, "", "./tensorflow/stream_executor/tpu/c_api_conversions.cc", "ToC");

  SE_MaybeOwningDeviceMemory se_mem;
  se_mem.device_ordinal = mem->device_ordinal();
  se_mem.memory = ApiConverter::ToC(mem->Release());
  se_mem.allocator = ApiConverter::ToC(mem->allocator());
  se_mem.owned = true;
  return se_mem;
}

void ToC(const stream_executor::DeviceMemoryBase& base,
         SE_DeviceMemoryBase* se_base) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPSstream_executorPStpuPSc_api_conversionsDTcc mht_7(mht_7_v, 326, "", "./tensorflow/stream_executor/tpu/c_api_conversions.cc", "ToC");

  se_base->opaque = const_cast<void*>(base.opaque());
  se_base->payload = base.payload();
  se_base->size = base.size();
}

SE_DeviceMemoryBase ToC(const stream_executor::DeviceMemoryBase& base) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPSstream_executorPStpuPSc_api_conversionsDTcc mht_8(mht_8_v, 335, "", "./tensorflow/stream_executor/tpu/c_api_conversions.cc", "ToC");

  SE_DeviceMemoryBase se_base;
  ToC(base, &se_base);
  return se_base;
}

stream_executor::DeviceMemoryBase FromC(const SE_DeviceMemoryBase& se_base) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPSstream_executorPStpuPSc_api_conversionsDTcc mht_9(mht_9_v, 344, "", "./tensorflow/stream_executor/tpu/c_api_conversions.cc", "FromC");

  stream_executor::DeviceMemoryBase base(se_base.opaque, se_base.size);
  base.SetPayload(se_base.payload);
  return base;
}

// Helper functions for copying data to possibly-inlined C arrays.

// 'Src' and 'Dst' are allowed to be different types to make this usable with
// memory-identical types, e.g. int64_t and int64_t. This should not be used
// with types that require a static_cast.
template <typename Src, typename Dst, typename DstList>
static void CreateVectorBase(const absl::Span<Src> src, DstList* dst) {
  static_assert(sizeof(Src) == sizeof(Dst), "Mismatched types");
  dst->size = src.size();
  if (dst->size > TPU_C_API_MAX_INLINED) {
    dst->heap = new Dst[dst->size];
    memcpy(dst->heap, src.data(), dst->size * sizeof(Src));
  } else {
    memcpy(dst->inlined, src.data(), dst->size * sizeof(Src));
  }
}

static void CreateVector(const absl::Span<const int64_t> src, Int64List* dst) {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPSstream_executorPStpuPSc_api_conversionsDTcc mht_10(mht_10_v, 370, "", "./tensorflow/stream_executor/tpu/c_api_conversions.cc", "CreateVector");

  return CreateVectorBase<const int64_t, int64_t, Int64List>(src, dst);
}
void CreateVector(const absl::Span<const float> src, FloatList* dst) {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPSstream_executorPStpuPSc_api_conversionsDTcc mht_11(mht_11_v, 376, "", "./tensorflow/stream_executor/tpu/c_api_conversions.cc", "CreateVector");

  return CreateVectorBase<const float, float, FloatList>(src, dst);
}
static void CreateVector(const absl::Span<const bool> src, BoolList* dst) {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPSstream_executorPStpuPSc_api_conversionsDTcc mht_12(mht_12_v, 382, "", "./tensorflow/stream_executor/tpu/c_api_conversions.cc", "CreateVector");

  return CreateVectorBase<const bool, bool, BoolList>(src, dst);
}

static void CreateVector(const absl::Span<const xla::Tile> src, TileList* dst) {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPSstream_executorPStpuPSc_api_conversionsDTcc mht_13(mht_13_v, 389, "", "./tensorflow/stream_executor/tpu/c_api_conversions.cc", "CreateVector");

  dst->size = src.size();
  XLA_Tile* c_tiles;
  if (dst->size > TPU_C_API_MAX_INLINED) {
    dst->heap = new XLA_Tile[dst->size];
    c_tiles = dst->heap;
  } else {
    c_tiles = dst->inlined;
  }
  for (int i = 0; i < dst->size; ++i) {
    ToC(src[i], &c_tiles[i]);
  }
}

// Helper functions for creating a view of possibly-inlined C arrays.

// 'Src' and 'Dst' are allowed to be different types to make this usable with
// memory-identical types, e.g. int64_t and int64_t. This should not be used
// with types that require a static_cast.
template <typename Dst, typename Src, typename SrcList>
static absl::Span<const Dst> MakeSpanBase(const SrcList& src_list) {
  static_assert(sizeof(Src) == sizeof(Dst), "Mismatched types");
  const Src* src = src_list.size > TPU_C_API_MAX_INLINED ? src_list.heap
                                                         : &src_list.inlined[0];
  return absl::Span<const Dst>(reinterpret_cast<const Dst*>(src),
                               src_list.size);
}

static absl::Span<const int64_t> MakeSpan(const Int64List& src_list) {
  return MakeSpanBase<int64_t, int64_t, Int64List>(src_list);
}

absl::Span<const float> MakeSpan(const FloatList& src_list) {
  return MakeSpanBase<float, float, FloatList>(src_list);
}
static absl::Span<const bool> MakeSpan(const BoolList& src_list) {
  return MakeSpanBase<bool, bool, BoolList>(src_list);
}

void ToC(const xla::Shape& xla_shape, XLA_Shape* c_shape) {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPSstream_executorPStpuPSc_api_conversionsDTcc mht_14(mht_14_v, 431, "", "./tensorflow/stream_executor/tpu/c_api_conversions.cc", "ToC");

  c_shape->element_type = xla_shape.element_type();

  CreateVector(xla_shape.dimensions(), &c_shape->dimensions);
  CreateVector(xla_shape.dynamic_dimensions(), &c_shape->dynamic_dimensions);

  c_shape->ntuple_shapes = xla_shape.tuple_shapes_size();
  if (c_shape->ntuple_shapes > 0) {
    c_shape->tuple_shapes = new XLA_Shape[c_shape->ntuple_shapes];
    for (int i = 0; i < c_shape->ntuple_shapes; ++i) {
      ToC(xla_shape.tuple_shapes(i), &c_shape->tuple_shapes[i]);
    }
  }

  if (xla_shape.has_layout()) {
    ToC(xla_shape.layout(), &c_shape->layout);
  } else {
    c_shape->layout.format = xla::INVALID_FORMAT;
  }
}

xla::Shape FromC(const XLA_Shape* c_shape) {
   std::vector<std::string> mht_15_v;
   MHTracer_DTPStensorflowPSstream_executorPStpuPSc_api_conversionsDTcc mht_15(mht_15_v, 455, "", "./tensorflow/stream_executor/tpu/c_api_conversions.cc", "FromC");

  absl::Span<const int64_t> dims = MakeSpan(c_shape->dimensions);
  absl::Span<const bool> dynamic_dims = MakeSpan(c_shape->dynamic_dimensions);

  std::vector<xla::Shape> tuple_shapes;
  tuple_shapes.reserve(c_shape->ntuple_shapes);
  for (int i = 0; i < c_shape->ntuple_shapes; ++i) {
    tuple_shapes.push_back(FromC(&c_shape->tuple_shapes[i]));
  }

  xla::Shape result(static_cast<xla::PrimitiveType>(c_shape->element_type),
                    dims, dynamic_dims, std::move(tuple_shapes));
  if (c_shape->layout.format != xla::INVALID_FORMAT) {
    *result.mutable_layout() = FromC(&c_shape->layout);
  }
  return result;
}

void Destroy(XLA_Shape* c_shape) {
   std::vector<std::string> mht_16_v;
   MHTracer_DTPStensorflowPSstream_executorPStpuPSc_api_conversionsDTcc mht_16(mht_16_v, 476, "", "./tensorflow/stream_executor/tpu/c_api_conversions.cc", "Destroy");

  if (c_shape->dimensions.size > TPU_C_API_MAX_INLINED) {
    delete[] c_shape->dimensions.heap;
  }
  if (c_shape->dynamic_dimensions.size > TPU_C_API_MAX_INLINED) {
    delete[] c_shape->dynamic_dimensions.heap;
  }
  if (c_shape->ntuple_shapes > 0) {
    for (int i = 0; i < c_shape->ntuple_shapes; ++i) {
      Destroy(&c_shape->tuple_shapes[i]);
    }
    delete[] c_shape->tuple_shapes;
  }
  if (c_shape->layout.format != xla::INVALID_FORMAT) {
    Destroy(&c_shape->layout);
  }
}

void ToC(const xla::Layout& layout, XLA_Layout* c_layout) {
   std::vector<std::string> mht_17_v;
   MHTracer_DTPStensorflowPSstream_executorPStpuPSc_api_conversionsDTcc mht_17(mht_17_v, 497, "", "./tensorflow/stream_executor/tpu/c_api_conversions.cc", "ToC");

  c_layout->format = layout.format();
  CreateVector(layout.minor_to_major(), &c_layout->minor_to_major);
  c_layout->element_size_in_bits = layout.element_size_in_bits();
  c_layout->memory_space = layout.memory_space();
  CreateVector(layout.tiles(), &c_layout->tiles);
}

xla::Layout FromC(const XLA_Layout* c_layout) {
   std::vector<std::string> mht_18_v;
   MHTracer_DTPStensorflowPSstream_executorPStpuPSc_api_conversionsDTcc mht_18(mht_18_v, 508, "", "./tensorflow/stream_executor/tpu/c_api_conversions.cc", "FromC");

  absl::Span<const int64_t> minor_to_major = MakeSpan(c_layout->minor_to_major);
  absl::InlinedVector<xla::Tile, 1> tiles;
  const XLA_Tile* c_tiles = c_layout->tiles.size > TPU_C_API_MAX_INLINED
                                ? c_layout->tiles.heap
                                : c_layout->tiles.inlined;
  for (int i = 0; i < c_layout->tiles.size; ++i) {
    tiles.push_back(FromC(&c_tiles[i]));
  }
  return xla::Layout(minor_to_major, tiles, c_layout->element_size_in_bits,
                     c_layout->memory_space);
}

void Destroy(XLA_Layout* c_layout) {
   std::vector<std::string> mht_19_v;
   MHTracer_DTPStensorflowPSstream_executorPStpuPSc_api_conversionsDTcc mht_19(mht_19_v, 524, "", "./tensorflow/stream_executor/tpu/c_api_conversions.cc", "Destroy");

  if (c_layout->minor_to_major.size > TPU_C_API_MAX_INLINED) {
    delete[] c_layout->minor_to_major.heap;
  }
  if (c_layout->tiles.size > TPU_C_API_MAX_INLINED) {
    delete[] c_layout->tiles.heap;
  }
}

void ToC(const xla::Tile& tile, XLA_Tile* c_tile) {
   std::vector<std::string> mht_20_v;
   MHTracer_DTPStensorflowPSstream_executorPStpuPSc_api_conversionsDTcc mht_20(mht_20_v, 536, "", "./tensorflow/stream_executor/tpu/c_api_conversions.cc", "ToC");

  CreateVector(tile.dimensions(), &c_tile->dimensions);
}

xla::Tile FromC(const XLA_Tile* c_tile) {
   std::vector<std::string> mht_21_v;
   MHTracer_DTPStensorflowPSstream_executorPStpuPSc_api_conversionsDTcc mht_21(mht_21_v, 543, "", "./tensorflow/stream_executor/tpu/c_api_conversions.cc", "FromC");

  absl::Span<const int64_t> dims = MakeSpan(c_tile->dimensions);
  return xla::Tile(dims);
}

void Destroy(XLA_Tile* c_tile) {
   std::vector<std::string> mht_22_v;
   MHTracer_DTPStensorflowPSstream_executorPStpuPSc_api_conversionsDTcc mht_22(mht_22_v, 551, "", "./tensorflow/stream_executor/tpu/c_api_conversions.cc", "Destroy");

  if (c_tile->dimensions.size > TPU_C_API_MAX_INLINED) {
    delete[] c_tile->dimensions.heap;
  }
}

XLA_ShapeIndex ToC(const xla::ShapeIndex& xla_shape) {
   std::vector<std::string> mht_23_v;
   MHTracer_DTPStensorflowPSstream_executorPStpuPSc_api_conversionsDTcc mht_23(mht_23_v, 560, "", "./tensorflow/stream_executor/tpu/c_api_conversions.cc", "ToC");

  XLA_ShapeIndex c_shape;
  CHECK_LT(xla_shape.size(), 8);
  c_shape.count = xla_shape.size();
  for (int i = 0; i < xla_shape.size(); ++i) {
    c_shape.indices[i] = xla_shape[i];
  }
  return c_shape;
}

xla::ShapeIndex FromC(XLA_ShapeIndex* c_shape) {
   std::vector<std::string> mht_24_v;
   MHTracer_DTPStensorflowPSstream_executorPStpuPSc_api_conversionsDTcc mht_24(mht_24_v, 573, "", "./tensorflow/stream_executor/tpu/c_api_conversions.cc", "FromC");

  return xla::ShapeIndex(&c_shape->indices[0],
                         &c_shape->indices[c_shape->count]);
}

void ToC(const xla::LiteralSlice& literal, XLA_Literal* c_literal) {
   std::vector<std::string> mht_25_v;
   MHTracer_DTPStensorflowPSstream_executorPStpuPSc_api_conversionsDTcc mht_25(mht_25_v, 581, "", "./tensorflow/stream_executor/tpu/c_api_conversions.cc", "ToC");

  ApiConverter::ToC(literal.shape(), &c_literal->shape);
  auto shapes = xla::ShapeUtil::GetLeafShapes(literal.shape());
  c_literal->buffers = new char*[shapes.size()];
  c_literal->sizes = new size_t[shapes.size()];
  c_literal->count = shapes.size();
  for (int i = 0; i < shapes.size(); ++i) {
    c_literal->buffers[i] = reinterpret_cast<char*>(
        const_cast<void*>(literal.untyped_data(shapes[i].index)));
    c_literal->sizes[i] = literal.size_bytes(shapes[i].index);
  }
}

xla::MutableBorrowingLiteral FromC(XLA_Literal* c_literal) {
   std::vector<std::string> mht_26_v;
   MHTracer_DTPStensorflowPSstream_executorPStpuPSc_api_conversionsDTcc mht_26(mht_26_v, 597, "", "./tensorflow/stream_executor/tpu/c_api_conversions.cc", "FromC");

  xla::Shape shape = ApiConverter::FromC(&c_literal->shape);
  return xla::MutableBorrowingLiteral(
      absl::MakeSpan(c_literal->buffers, c_literal->count), shape);
}

void ToC(const xla::ShapedBuffer& buffer, XLA_ShapedBuffer* c_device_buffer) {
   std::vector<std::string> mht_27_v;
   MHTracer_DTPStensorflowPSstream_executorPStpuPSc_api_conversionsDTcc mht_27(mht_27_v, 606, "", "./tensorflow/stream_executor/tpu/c_api_conversions.cc", "ToC");

  ApiConverter::ToC(buffer.on_device_shape(),
                    &c_device_buffer->on_device_shape);
  c_device_buffer->device_ordinal = buffer.device_ordinal();
  absl::InlinedVector<SE_DeviceMemoryBase, 2> bases;
  for (auto& pair : buffer.buffers()) {
    bases.push_back(ApiConverter::ToC(pair.second));
  }
  c_device_buffer->count = bases.size();
  c_device_buffer->bases = new SE_DeviceMemoryBase[bases.size()];
  for (int i = 0; i < bases.size(); ++i) {
    c_device_buffer->bases[i] = bases[i];
  }
}

std::unique_ptr<TpuEmbeddingEngineParametersData> Create(int num_tables) {
  auto data = std::make_unique<TpuEmbeddingEngineParametersData>();
  data->c_params.num_tables = num_tables;
  for (int i = 0; i < 8; i++) {
    data->vectors[i].resize(num_tables);
    data->c_params.parameters[i] = data->vectors[i].data();
  }
  return data;
}

void Destroy(XLA_ShapeIndex* shape_index) {
   std::vector<std::string> mht_28_v;
   MHTracer_DTPStensorflowPSstream_executorPStpuPSc_api_conversionsDTcc mht_28(mht_28_v, 634, "", "./tensorflow/stream_executor/tpu/c_api_conversions.cc", "Destroy");
 delete[] shape_index; }
void Destroy(SE_DeviceMemoryBase*) {
   std::vector<std::string> mht_29_v;
   MHTracer_DTPStensorflowPSstream_executorPStpuPSc_api_conversionsDTcc mht_29(mht_29_v, 638, "", "./tensorflow/stream_executor/tpu/c_api_conversions.cc", "Destroy");
}

void Destroy(XLA_Literal* c_literal) {
   std::vector<std::string> mht_30_v;
   MHTracer_DTPStensorflowPSstream_executorPStpuPSc_api_conversionsDTcc mht_30(mht_30_v, 643, "", "./tensorflow/stream_executor/tpu/c_api_conversions.cc", "Destroy");

  delete[] c_literal->buffers;
  delete[] c_literal->sizes;
  ApiConverter::Destroy(&c_literal->shape);
}

void Destroy(XLA_ShapedBuffer* c_buffer) {
   std::vector<std::string> mht_31_v;
   MHTracer_DTPStensorflowPSstream_executorPStpuPSc_api_conversionsDTcc mht_31(mht_31_v, 652, "", "./tensorflow/stream_executor/tpu/c_api_conversions.cc", "Destroy");

  ApiConverter::Destroy(&c_buffer->on_device_shape);
  delete[] c_buffer->bases;
}

XLA_HloModule ToC(const xla::HloModule& module) {
   std::vector<std::string> mht_32_v;
   MHTracer_DTPStensorflowPSstream_executorPStpuPSc_api_conversionsDTcc mht_32(mht_32_v, 660, "", "./tensorflow/stream_executor/tpu/c_api_conversions.cc", "ToC");

  XLA_HloModule c_module;
  c_module.proto = stream_executor::tpu::SerializeProto(module.ToProto());
  c_module.module_config = ApiConverter::ToC(module.config());
  return c_module;
}

xla::StatusOr<std::unique_ptr<xla::HloModule>> FromC(
    const XLA_HloModule& c_module) {
  xla::HloModuleProto module_proto =
      stream_executor::tpu::DeserializeProto<xla::HloModuleProto>(
          c_module.proto);
  return xla::HloModule::CreateFromProto(
      module_proto, ApiConverter::FromC(c_module.module_config));
}

void Destroy(XLA_HloModule* c_module) {
   std::vector<std::string> mht_33_v;
   MHTracer_DTPStensorflowPSstream_executorPStpuPSc_api_conversionsDTcc mht_33(mht_33_v, 679, "", "./tensorflow/stream_executor/tpu/c_api_conversions.cc", "Destroy");

  stream_executor::tpu::SerializedProto_Free(c_module->proto);
  Destroy(&c_module->module_config);
}

static xla::HloModuleConfig ConfigWithLayout(
    const XLA_HloModuleConfig& se_config) {
   std::vector<std::string> mht_34_v;
   MHTracer_DTPStensorflowPSstream_executorPStpuPSc_api_conversionsDTcc mht_34(mht_34_v, 688, "", "./tensorflow/stream_executor/tpu/c_api_conversions.cc", "ConfigWithLayout");

  xla::ShapeLayout result_layout(
      FromC(&se_config.entry_computation_layout.result_layout));
  xla::ComputationLayout layout(result_layout);
  for (int i = 0; i < se_config.entry_computation_layout.parameter_count; ++i) {
    layout.add_parameter_layout(xla::ShapeLayout(
        FromC(&se_config.entry_computation_layout.parameter_layouts[i])));
  }
  return xla::HloModuleConfig(layout);
}

XLA_HloModuleConfig ToC(const xla::HloModuleConfig& config) {
   std::vector<std::string> mht_35_v;
   MHTracer_DTPStensorflowPSstream_executorPStpuPSc_api_conversionsDTcc mht_35(mht_35_v, 702, "", "./tensorflow/stream_executor/tpu/c_api_conversions.cc", "ToC");

  XLA_HloModuleConfig hlo_config;

  hlo_config.seed = config.seed();
  hlo_config.launch_id = config.launch_id();
  hlo_config.replica_count = config.replica_count();
  hlo_config.num_partitions = config.num_partitions();
  hlo_config.use_spmd_partitioning = config.use_spmd_partitioning();
  hlo_config.use_auto_spmd_partitioning = config.use_auto_spmd_partitioning();
  hlo_config.has_static_device_assignment =
      config.has_static_device_assignment();
  hlo_config.has_entry_computation_layout =
      config.has_entry_computation_layout();

  if (config.has_static_device_assignment()) {
    xla::DeviceAssignmentProto dev_proto;
    config.static_device_assignment().Serialize(&dev_proto).IgnoreError();
    hlo_config.static_device_assignment =
        stream_executor::tpu::SerializeProto(dev_proto);
  }

  hlo_config.debug_options =
      stream_executor::tpu::SerializeProto(config.debug_options());

  if (config.has_entry_computation_layout()) {
    const auto& layout = config.entry_computation_layout();
    ApiConverter::ToC(layout.result_layout().shape(),
                      &hlo_config.entry_computation_layout.result_layout);
    hlo_config.entry_computation_layout.parameter_layouts =
        new XLA_Shape[layout.parameter_count()];
    for (int i = 0; i < layout.parameter_count(); ++i) {
      ApiConverter::ToC(
          layout.parameter_layout(i).shape(),
          &hlo_config.entry_computation_layout.parameter_layouts[i]);
    }
    hlo_config.entry_computation_layout.parameter_count =
        layout.parameter_count();
  }
  return hlo_config;
}

xla::HloModuleConfig FromC(const XLA_HloModuleConfig& c_config) {
   std::vector<std::string> mht_36_v;
   MHTracer_DTPStensorflowPSstream_executorPStpuPSc_api_conversionsDTcc mht_36(mht_36_v, 746, "", "./tensorflow/stream_executor/tpu/c_api_conversions.cc", "FromC");

  xla::HloModuleConfig config = c_config.has_entry_computation_layout
                                    ? ConfigWithLayout(c_config)
                                    : xla::HloModuleConfig();
  config.set_launch_id(c_config.launch_id);
  config.set_seed(c_config.seed);
  config.set_replica_count(c_config.replica_count);
  config.set_num_partitions(c_config.num_partitions);
  config.set_use_spmd_partitioning(c_config.use_spmd_partitioning);
  config.set_use_auto_spmd_partitioning(c_config.use_auto_spmd_partitioning);
  if (c_config.has_static_device_assignment) {
    auto device_assignment = xla::DeviceAssignment::Deserialize(
        stream_executor::tpu::DeserializeProto<xla::DeviceAssignmentProto>(
            c_config.static_device_assignment));
    config.set_static_device_assignment(
        *(device_assignment.ConsumeValueOrDie()));
  }
  config.set_debug_options(
      stream_executor::tpu::DeserializeProto<xla::DebugOptions>(
          c_config.debug_options));
  return config;
}

void Destroy(XLA_HloModuleConfig* c_config) {
   std::vector<std::string> mht_37_v;
   MHTracer_DTPStensorflowPSstream_executorPStpuPSc_api_conversionsDTcc mht_37(mht_37_v, 772, "", "./tensorflow/stream_executor/tpu/c_api_conversions.cc", "Destroy");

  for (auto i = 0; i < c_config->entry_computation_layout.parameter_count;
       ++i) {
    ApiConverter::Destroy(
        &c_config->entry_computation_layout.parameter_layouts[i]);
  }
  delete[] c_config->entry_computation_layout.parameter_layouts;
  ApiConverter::Destroy(&c_config->entry_computation_layout.result_layout);
  if (c_config->has_static_device_assignment) {
    stream_executor::tpu::SerializedProto_Free(
        c_config->static_device_assignment);
  }
  stream_executor::tpu::SerializedProto_Free(c_config->debug_options);
}

void Destroy(FloatList* float_list) {
   std::vector<std::string> mht_38_v;
   MHTracer_DTPStensorflowPSstream_executorPStpuPSc_api_conversionsDTcc mht_38(mht_38_v, 790, "", "./tensorflow/stream_executor/tpu/c_api_conversions.cc", "Destroy");

  if (float_list->size > TPU_C_API_MAX_INLINED) {
    delete[] float_list->heap;
  }
}

}  // namespace ApiConverter
