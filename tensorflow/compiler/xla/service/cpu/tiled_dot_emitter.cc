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
class MHTracer_DTPStensorflowPScompilerPSxlaPSservicePScpuPStiled_dot_emitterDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePScpuPStiled_dot_emitterDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSxlaPSservicePScpuPStiled_dot_emitterDTcc() {
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

/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/service/cpu/tiled_dot_emitter.h"

#include "tensorflow/compiler/xla/service/cpu/vector_support_library.h"
#include "tensorflow/compiler/xla/service/hlo_module.h"
#include "tensorflow/compiler/xla/service/llvm_ir/kernel_support_library.h"
#include "tensorflow/compiler/xla/service/llvm_ir/llvm_util.h"

namespace xla {
namespace cpu {
namespace {

using ::int64_t;

// Provides tiled access to an in-memory rank 2 array.
class MemoryTile {
 public:
  // Constructs a MemoryTile that can operate on tiles consisting of
  // `tile_size_along_major_dim` vectors from the matrix `matrix`, starting at
  // `major_dim_offset` in the major dimension.  The tile size along the minor
  // dimension is the vector size, and that is implicitly determined by `vsl`.
  MemoryTile(VectorSupportLibrary* vsl, llvm::IRBuilder<>* b,
             llvm::Value* matrix, int64_t matrix_size_along_minor_dim,
             llvm::Value* major_dim_offset, int64_t tile_size_along_major_dim)
      : vsl_(vsl), b_(b) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePScpuPStiled_dot_emitterDTcc mht_0(mht_0_v, 208, "", "./tensorflow/compiler/xla/service/cpu/tiled_dot_emitter.cc", "MemoryTile");

    pointers_.reserve(tile_size_along_major_dim);
    for (int64_t i = 0; i < tile_size_along_major_dim; i++) {
      llvm::Value* total_offset =
          b->CreateMul(b->getInt64(matrix_size_along_minor_dim),
                       b->CreateAdd(b->getInt64(i), major_dim_offset));
      pointers_.push_back(vsl_->ComputeOffsetPointer(matrix, total_offset));
    }
  }

  // Load a tile consisting of `tile_size_along_major_dim` vectors from position
  // {major: `major_dim_offset`, minor: `minor_dim_offset`}.
  //
  // Note: `major_dim_offset` is a parameter to the constructor.
  std::vector<llvm::Value*> LoadTile(llvm::Value* minor_dim_offset) const {
    std::vector<llvm::Value*> result;
    result.reserve(pointers_.size());
    for (const auto& pointer : pointers_) {
      result.push_back(vsl_->LoadVector(pointer, minor_dim_offset));
    }
    return result;
  }

  // Stores `tile` to position {major: `major_dim_offset`, minor:
  // `minor_dim_offset`}.
  //
  // Note: `major_dim_offset` is a parameter to the constructor.
  void StoreTile(absl::Span<llvm::Value* const> tile,
                 llvm::Value* minor_dim_offset) const {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePScpuPStiled_dot_emitterDTcc mht_1(mht_1_v, 239, "", "./tensorflow/compiler/xla/service/cpu/tiled_dot_emitter.cc", "StoreTile");

    CHECK_EQ(tile.size(), pointers_.size());
    for (int64_t i = 0; i < pointers_.size(); i++) {
      vsl_->StoreVector(tile[i], pointers_[i], minor_dim_offset);
    }
  }

  // Loads a tile of size [`tile_size_along_major_dim`,
  // `tile_size_along_middle_dim`] from position {major: `major_dim_offset`,
  // minor: `minor_dim_offset`} and then broadcasts each element into a vector
  // of size vsl_.vector_size().  The (i,j)'th element of the return value is
  // the (i,j)'th element in the tile broadcasted into an LLVM vector.
  //
  // Note: `major_dim_offset` is a parameter to the constructor.
  std::vector<std::vector<llvm::Value*>> LoadBroadcastTile(
      llvm::Value* minor_dim_offset, int64_t tile_size_along_middle_dim) const {
    std::vector<std::vector<llvm::Value*>> result;
    result.resize(pointers_.size());
    for (int64_t i = 0; i < pointers_.size(); i++) {
      for (int64_t j = 0; j < tile_size_along_middle_dim; j++) {
        result[i].push_back(vsl_->LoadBroadcast(
            pointers_[i], b_->CreateAdd(minor_dim_offset, b_->getInt64(j))));
      }
    }
    return result;
  }

 private:
  VectorSupportLibrary* vsl_;
  llvm::IRBuilder<>* b_;
  std::vector<llvm::Value*> pointers_;
};

// The base class for the classes representing the GEMV emitter configurations.
//
// The IR emitted (modulo the LLVM values representing the input and output
// buffers) by the row major and column major GEMV emitters should be a function
// of their configuration.  This is important because their configuration is
// used as a key to cache the generated IR.
class GemvConfig {
 public:
  // Mixin for convenience.
  template <typename T>
  struct User {
   public:
    PrimitiveType scalar_type() const {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePScpuPStiled_dot_emitterDTcc mht_2(mht_2_v, 287, "", "./tensorflow/compiler/xla/service/cpu/tiled_dot_emitter.cc", "scalar_type");

      return derived().config().scalar_type();
    }
    int64_t tile_rows() const {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePScpuPStiled_dot_emitterDTcc mht_3(mht_3_v, 293, "", "./tensorflow/compiler/xla/service/cpu/tiled_dot_emitter.cc", "tile_rows");
 return derived().config().tile_rows(); }
    int64_t tile_cols() const {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePScpuPStiled_dot_emitterDTcc mht_4(mht_4_v, 297, "", "./tensorflow/compiler/xla/service/cpu/tiled_dot_emitter.cc", "tile_cols");
 return derived().config().tile_cols(); }
    int64_t m() const {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePScpuPStiled_dot_emitterDTcc mht_5(mht_5_v, 301, "", "./tensorflow/compiler/xla/service/cpu/tiled_dot_emitter.cc", "m");
 return derived().config().m(); }
    int64_t k() const {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePScpuPStiled_dot_emitterDTcc mht_6(mht_6_v, 305, "", "./tensorflow/compiler/xla/service/cpu/tiled_dot_emitter.cc", "k");
 return derived().config().k(); }
    int64_t has_addend() const {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePScpuPStiled_dot_emitterDTcc mht_7(mht_7_v, 309, "", "./tensorflow/compiler/xla/service/cpu/tiled_dot_emitter.cc", "has_addend");
 return derived().config().has_addend(); }

   private:
    const T& derived() const {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePScpuPStiled_dot_emitterDTcc mht_8(mht_8_v, 315, "", "./tensorflow/compiler/xla/service/cpu/tiled_dot_emitter.cc", "derived");
 return *static_cast<const T*>(this); }
  };

  PrimitiveType scalar_type() const {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePScpuPStiled_dot_emitterDTcc mht_9(mht_9_v, 321, "", "./tensorflow/compiler/xla/service/cpu/tiled_dot_emitter.cc", "scalar_type");
 return scalar_type_; }
  int64_t tile_rows() const {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePScpuPStiled_dot_emitterDTcc mht_10(mht_10_v, 325, "", "./tensorflow/compiler/xla/service/cpu/tiled_dot_emitter.cc", "tile_rows");
 return tile_rows_; }
  int64_t tile_cols() const {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePScpuPStiled_dot_emitterDTcc mht_11(mht_11_v, 329, "", "./tensorflow/compiler/xla/service/cpu/tiled_dot_emitter.cc", "tile_cols");
 return tile_cols_; }
  int64_t m() const {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePScpuPStiled_dot_emitterDTcc mht_12(mht_12_v, 333, "", "./tensorflow/compiler/xla/service/cpu/tiled_dot_emitter.cc", "m");
 return m_; }
  int64_t k() const {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePScpuPStiled_dot_emitterDTcc mht_13(mht_13_v, 337, "", "./tensorflow/compiler/xla/service/cpu/tiled_dot_emitter.cc", "k");
 return k_; }
  bool has_addend() const {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePScpuPStiled_dot_emitterDTcc mht_14(mht_14_v, 341, "", "./tensorflow/compiler/xla/service/cpu/tiled_dot_emitter.cc", "has_addend");
 return has_addend_; }

  std::string GetCacheKey() const {
   std::vector<std::string> mht_15_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePScpuPStiled_dot_emitterDTcc mht_15(mht_15_v, 346, "", "./tensorflow/compiler/xla/service/cpu/tiled_dot_emitter.cc", "GetCacheKey");

    return absl::StrCat(name_, "_", PrimitiveType_Name(scalar_type()), "_",
                        tile_rows(), "_", tile_cols(), "_", m(), "_", k(),
                        has_addend() ? "_with_addend" : "");
  }

 protected:
  explicit GemvConfig(std::string name, PrimitiveType scalar_type,
                      int64_t tile_rows, int64_t tile_cols, int64_t m,
                      int64_t k, bool has_addend)
      : name_(std::move(name)),
        scalar_type_(scalar_type),
        tile_rows_(tile_rows),
        tile_cols_(tile_cols),
        m_(m),
        k_(k),
        has_addend_(has_addend) {
   std::vector<std::string> mht_16_v;
   mht_16_v.push_back("name: \"" + name + "\"");
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePScpuPStiled_dot_emitterDTcc mht_16(mht_16_v, 366, "", "./tensorflow/compiler/xla/service/cpu/tiled_dot_emitter.cc", "GemvConfig");
}

 private:
  std::string name_;
  PrimitiveType scalar_type_;
  int64_t tile_rows_;
  int64_t tile_cols_;
  int64_t m_;
  int64_t k_;
  bool has_addend_;
};

// Computes a dot product between "[M,K]{0,1} lhs" with a [K,1] vector (the
// layout of the vector does not matter).  This implementation uses a tiling
// scheme to improve performance.
//
// We logically separate the LHS matrix into four segments:
//
//   +----------------------+---+
//   |                      |   |
//   |                      |   |
//   |         A            | B |
//   |                      |   |
//   |                      |   |
//   |                      |   |
//   +----------------------+---+
//   |         C            | D |
//   +----------------------+---+
//
// where A is the largest submatrix of the LHS that can be evenly divided into
// tiles.  For each tile in A, assuming tile_rows_ == tile_cols_ == 4, we have:
//
//   +---+---+---+---+       +--+--+--+--+
//   |M00|M10|M20|M30|       |V0|V1|V2|V3|
//   +---+---+---+---+       +--+--+--+--+
//   |M01|M11|M21|M31| and   |V0|V1|V2|V3|
//   +---+---+---+---+       +--+--+--+--+
//   |M02|M12|M22|M32|       |V0|V1|V2|V3|
//   +---+---+---+---+       +--+--+--+--+
//   |M03|M13|M23|M33|       |V0|V1|V2|V3|
//   +---+---+---+---+       +--+--+--+--+
//
// (Legend: rows are horizontal and columns are vertical; and each column is one
// llvm::Value of a vector type)
//
// where:
//
//   a. The left tile is from the column major left matrix.
//   b. The right tile is an elementwise broadcast of a [V0, V1, V2, V3]
//      vector loaded from the RHS vector.
//
// As we iterate through the column dimension, we compute the change to the
// result vector by an elementwise multiplication between the two tiles above
// followed by a reduction along the major dimension:
//
//                     +-----------------------------------+
//                     | M00*V0 + M10*V1 + M20*V2 + M30*V3 |
//                     +-----------------------------------+
//                     | M01*V0 + M11*V1 + M21*V2 + M31*V3 |
// Result[R:R+4] +=    +-----------------------------------+
//                     | M02*V0 + M12*V1 + M22*V2 + M32*V3 |
//                     +-----------------------------------+
//                     | M03*V0 + M13*V1 + M23*V2 + M33*V3 |
//                     +-----------------------------------+
//
// Where R is the starting row for the tile.
//
// We have an inner epilogue loop to deal with the "C" submatrix and an outer
// epilogue loop to deal with the B,D submatrix.
//
// TODO(sanjoy): We should investigate if using gather loads and scatter stores
// can be used here have the same inner loop for both column-major and row-major
// matrix-vector products.
class ColumnMajorMatrixVectorProductEmitter
    : public GemvConfig::User<ColumnMajorMatrixVectorProductEmitter> {
 public:
  class Config : public GemvConfig {
   public:
    explicit Config(PrimitiveType scalar_type, int64_t tile_rows,
                    int64_t tile_cols, int64_t m, int64_t k, bool has_addend)
        : GemvConfig(/*name=*/"col_major_gemv", scalar_type,
                     /*tile_rows=*/tile_rows, /*tile_cols=*/tile_cols, /*m=*/m,
                     /*k=*/k, /*has_addend=*/has_addend) {
   std::vector<std::string> mht_17_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePScpuPStiled_dot_emitterDTcc mht_17(mht_17_v, 451, "", "./tensorflow/compiler/xla/service/cpu/tiled_dot_emitter.cc", "Config");
}
  };

  ColumnMajorMatrixVectorProductEmitter(const Config& config, llvm::Value* lhs,
                                        llvm::Value* rhs, llvm::Value* addend,
                                        llvm::Value* result,
                                        llvm::IRBuilder<>* b)
      : config_(config),
        lhs_(lhs),
        rhs_(rhs),
        addend_(addend),
        result_(result),
        b_(b),
        ksl_(b_),
        vsl_(config.scalar_type(), /*vector_size=*/config.tile_rows(), b_, "") {
   std::vector<std::string> mht_18_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePScpuPStiled_dot_emitterDTcc mht_18(mht_18_v, 468, "", "./tensorflow/compiler/xla/service/cpu/tiled_dot_emitter.cc", "ColumnMajorMatrixVectorProductEmitter");

    CHECK(tile_rows() > 0 &&
          absl::has_single_bit(static_cast<uint64_t>(tile_rows())));
    CHECK(!has_addend() || addend != nullptr);
  }

  void Emit();

  const Config& config() const {
   std::vector<std::string> mht_19_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePScpuPStiled_dot_emitterDTcc mht_19(mht_19_v, 479, "", "./tensorflow/compiler/xla/service/cpu/tiled_dot_emitter.cc", "config");
 return config_; }

 private:
  void EmitOuterLoopBody(llvm::Value* column, int64_t column_count,
                         bool is_first_column);

  MemoryTile GetLhsMemoryTile(llvm::Value* column_start, int64_t column_count) {
   std::vector<std::string> mht_20_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePScpuPStiled_dot_emitterDTcc mht_20(mht_20_v, 488, "", "./tensorflow/compiler/xla/service/cpu/tiled_dot_emitter.cc", "GetLhsMemoryTile");

    return MemoryTile(&vsl_, b_, /*matrix=*/lhs_,
                      /*matrix_size_along_minor_dim=*/m(),
                      /*major_dim_offset=*/column_start,
                      /*tile_size_along_major_dim=*/column_count);
  }

  // Load a tile of values from the RHS.  For the RHS a "tile" is a contiguous
  // sequence of `count` values, each one broadcasted to the vector width.
  std::vector<llvm::Value*> LoadRhsTile(llvm::Value* offset, int64_t count) {
    llvm::Value* base_pointer = vsl_.ComputeOffsetPointer(rhs_, offset);
    std::vector<llvm::Value*> result;
    result.reserve(count);
    for (int64_t i = 0; i < count; i++) {
      result.push_back(vsl_.LoadBroadcast(base_pointer, i));
    }
    return result;
  }

  void EmitInnerLoopTiled(MemoryTile* lhs_memory_tile,
                          const std::vector<llvm::Value*>& rhs_tile,
                          int64_t columns, bool is_first_column);

  void EmitInnerLoopEpilogue(llvm::Value* current_tile_col, int64_t columns,
                             bool is_first_tiled_column);

  Config config_;
  llvm::Value* lhs_;
  llvm::Value* rhs_;
  llvm::Value* addend_;
  llvm::Value* result_;
  llvm::IRBuilder<>* b_;
  KernelSupportLibrary ksl_;
  VectorSupportLibrary vsl_;
};

void ColumnMajorMatrixVectorProductEmitter::EmitOuterLoopBody(
    llvm::Value* column, int64_t column_count, bool is_first_column) {
   std::vector<std::string> mht_21_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePScpuPStiled_dot_emitterDTcc mht_21(mht_21_v, 528, "", "./tensorflow/compiler/xla/service/cpu/tiled_dot_emitter.cc", "ColumnMajorMatrixVectorProductEmitter::EmitOuterLoopBody");

  MemoryTile lhs_memory_tile = GetLhsMemoryTile(/*column_start=*/column,
                                                /*column_count=*/column_count);

  std::vector<llvm::Value*> rhs_tile =
      LoadRhsTile(column, /*count=*/column_count);
  EmitInnerLoopTiled(&lhs_memory_tile, rhs_tile,
                     /*columns=*/column_count, is_first_column);
  EmitInnerLoopEpilogue(column, /*columns=*/column_count, is_first_column);
}

void ColumnMajorMatrixVectorProductEmitter::Emit() {
   std::vector<std::string> mht_22_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePScpuPStiled_dot_emitterDTcc mht_22(mht_22_v, 542, "", "./tensorflow/compiler/xla/service/cpu/tiled_dot_emitter.cc", "ColumnMajorMatrixVectorProductEmitter::Emit");

  // See the comment on the class declaration for the algorithm used here.
  int64_t column_remainder = k() % tile_cols();
  int64_t column_limit = k() - column_remainder;

  ksl_.For("dot.outer.tiled",
           /*start=*/0, /*end=*/column_limit, /*step=*/tile_cols(),
           [&](llvm::Value* column, bool is_first_column) {
             EmitOuterLoopBody(column, tile_cols(), is_first_column);
           });

  if (column_remainder != 0) {
    EmitOuterLoopBody(b_->getInt64(column_limit), column_remainder,
                      column_limit == 0);
  }
}

void ColumnMajorMatrixVectorProductEmitter::EmitInnerLoopTiled(
    MemoryTile* lhs_memory_tile, const std::vector<llvm::Value*>& rhs_tile,
    int64_t columns, bool is_first_column) {
   std::vector<std::string> mht_23_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePScpuPStiled_dot_emitterDTcc mht_23(mht_23_v, 564, "", "./tensorflow/compiler/xla/service/cpu/tiled_dot_emitter.cc", "ColumnMajorMatrixVectorProductEmitter::EmitInnerLoopTiled");

  int64_t row_limit = m() - (m() % tile_rows());

  ksl_.For("dot.inner.tiled", /*start=*/0, /*end=*/row_limit,
           /*step=*/tile_rows(), [&](llvm::Value* row) {
             std::vector<llvm::Value*> lhs_tile =
                 lhs_memory_tile->LoadTile(/*minor_dim_offset=*/row);
             llvm::Value* accumulator =
                 is_first_column ? (addend_ ? vsl_.LoadVector(addend_, row)
                                            : vsl_.GetZeroVector())
                                 : vsl_.LoadVector(result_, row);
             for (int i = 0; i < columns; i++) {
               accumulator = vsl_.MulAdd(lhs_tile[i], rhs_tile[i], accumulator);
             }
             vsl_.StoreVector(accumulator, result_, row);
           });
}

void ColumnMajorMatrixVectorProductEmitter::EmitInnerLoopEpilogue(
    llvm::Value* current_tile_col, int64_t columns,
    bool is_first_tiled_column) {
   std::vector<std::string> mht_24_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePScpuPStiled_dot_emitterDTcc mht_24(mht_24_v, 587, "", "./tensorflow/compiler/xla/service/cpu/tiled_dot_emitter.cc", "ColumnMajorMatrixVectorProductEmitter::EmitInnerLoopEpilogue");

  int64_t row_start = m() - (m() % tile_rows());
  if (row_start == m()) {
    return;
  }

  llvm::Value* columns_llvm = b_->getInt64(columns);

  // for (col = current_tile_col; col < (columns + current_tile_col); col++)
  //   for (row = row_start, row < m_; row++) {
  //     result[row] += lhs[row, col] * rhs[col]
  //     // Also take into account that if col is 0 then result[row] is not
  //     // initialized.
  //   }

  ksl_.For(
      "dot.inner.epilg.outer", /*start=*/current_tile_col,
      /*end=*/b_->CreateAdd(columns_llvm, current_tile_col),
      /*step=*/1, /*peel_first_iteration=*/false,
      [&](llvm::Value* col, llvm::Value* is_first_scalar_col) {
        llvm::Value* rhs_element = vsl_.LoadScalar(rhs_, col);
        llvm::Value* total_offset = b_->CreateMul(col, b_->getInt64(m()));
        llvm::Value* lhs_base_pointer =
            vsl_.ComputeOffsetPointer(lhs_, total_offset);
        ksl_.For(
            "dot.inner.epilg.inner", /*start=*/row_start, /*end=*/m(),
            /*step=*/1, [&](llvm::Value* scalar_row) {
              llvm::Value* product = vsl_.Mul(
                  vsl_.LoadScalar(lhs_base_pointer, scalar_row), rhs_element);
              llvm::Value* setting_result_first_time = b_->CreateAnd(
                  is_first_scalar_col, b_->getInt1(is_first_tiled_column));
              ksl_.If(
                  setting_result_first_time,
                  /*true_block_generator=*/
                  [&]() {
                    if (addend_) {
                      vsl_.StoreScalar(
                          vsl_.Add(vsl_.LoadScalar(addend_, scalar_row),
                                   product),
                          result_, scalar_row);
                    } else {
                      vsl_.StoreScalar(product, result_, scalar_row);
                    }
                  },
                  /*false_block_generator=*/
                  [&]() {
   std::vector<std::string> mht_25_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePScpuPStiled_dot_emitterDTcc mht_25(mht_25_v, 635, "", "./tensorflow/compiler/xla/service/cpu/tiled_dot_emitter.cc", "lambda");

                    vsl_.StoreScalar(
                        vsl_.Add(vsl_.LoadScalar(result_, scalar_row), product),
                        result_, scalar_row);
                  });
            });
      });
}

// Computes a dot product between "[M,K]{1,0} lhs" with a [K,1] vector (the
// layout of the vector does not matter).  This implementation uses a tiling
// scheme to improve performance.
//
// We logically separate the LHS matrix into four segments:
//
//   +----------------------+---+
//   |                      |   |
//   |                      |   |
//   |         A            | B |
//   |                      |   |
//   |                      |   |
//   |                      |   |
//   +----------------------+---+
//   |         C            | D |
//   +----------------------+---+
//
// where A is the largest submatrix of the LHS that can be evenly divided into
// tiles.  For each tile in A, assuming tile_rows_ == tile_cols_ == 4, we have:
//
//   +---+---+---+---+
//   |M00|M10|M20|M30|
//   +---+---+---+---+       +--+--+--+--+
//   |M01|M11|M21|M31| and   |V0|V1|V2|V3|
//   +---+---+---+---+       +--+--+--+--+
//   |M02|M12|M22|M32|
//   +---+---+---+---+
//   |M03|M13|M23|M33|
//   +---+---+---+---+
//
// (Legend: rows are horizontal and columns are vertical; and each row is one
// llvm::Value of a vector type)
//
// where:
//
//   a. The left tile is loaded from the row major left matrix.
//   b. The right vector is loaded from the RHS vector.
//
// We keep 4 vector accumulators accumulating the following four vector
// expressions as we iterate over the row dimension:
//
//   +------+------+------+------+
//   |M0I*V0|M1I*V1|M2I*V2|M3I*V3|  for I in [0,4)
//   +------+------+------+------+
//
// In the end we do a horizontal reduction over these 4 vector accumulators to
// get 4 values in the result vector.
//
// We have an inner epilogue loop to deal with the "B" sub-matrix and an outer
// epilogue loop to deal with the C,D submatrix.
class RowMajorMatrixVectorProductEmitter
    : public GemvConfig::User<RowMajorMatrixVectorProductEmitter> {
 public:
  class Config : public GemvConfig {
   public:
    explicit Config(PrimitiveType scalar_type, int64_t tile_rows,
                    int64_t tile_cols, int64_t m, int64_t k, bool has_addend)
        : GemvConfig(/*name=*/"row_major_gemv", scalar_type,
                     /*tile_rows=*/tile_rows, /*tile_cols=*/tile_cols, /*m=*/m,
                     /*k=*/k, /*has_addend=*/has_addend) {
   std::vector<std::string> mht_26_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePScpuPStiled_dot_emitterDTcc mht_26(mht_26_v, 706, "", "./tensorflow/compiler/xla/service/cpu/tiled_dot_emitter.cc", "Config");
}
  };

  RowMajorMatrixVectorProductEmitter(const Config& config, llvm::Value* lhs,
                                     llvm::Value* rhs, llvm::Value* addend,
                                     llvm::Value* result, llvm::IRBuilder<>* b)
      : config_(config),
        lhs_(lhs),
        rhs_(rhs),
        addend_(addend),
        result_(result),
        b_(b),
        ksl_(b_),
        vsl_(scalar_type(), /*vector_size=*/tile_cols(), b_, "") {
   std::vector<std::string> mht_27_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePScpuPStiled_dot_emitterDTcc mht_27(mht_27_v, 722, "", "./tensorflow/compiler/xla/service/cpu/tiled_dot_emitter.cc", "RowMajorMatrixVectorProductEmitter");

    CHECK(tile_cols() > 0 &&
          absl::has_single_bit(static_cast<uint64_t>(tile_cols())));
    CHECK(!has_addend() || addend != nullptr);
  }

  void Emit();

  const Config& config() const {
   std::vector<std::string> mht_28_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePScpuPStiled_dot_emitterDTcc mht_28(mht_28_v, 733, "", "./tensorflow/compiler/xla/service/cpu/tiled_dot_emitter.cc", "config");
 return config_; }

 private:
  MemoryTile GetLhsMemoryTile(llvm::Value* row_start, int64_t row_count) {
   std::vector<std::string> mht_29_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePScpuPStiled_dot_emitterDTcc mht_29(mht_29_v, 739, "", "./tensorflow/compiler/xla/service/cpu/tiled_dot_emitter.cc", "GetLhsMemoryTile");

    return MemoryTile(&vsl_, b_, /*matrix=*/lhs_,
                      /*matrix_size_along_minor_dim=*/k(),
                      /*major_dim_offset=*/row_start,
                      /*tile_size_along_major_dim=*/row_count);
  }

  void EmitOuterLoopBody(llvm::Value* row, int64_t row_count);

  void EmitInnerLoopTiled(MemoryTile* lhs_memory_tile, int64_t rows,
                          std::vector<VectorVariable>* vector_accumulators);

  void EmitInnerLoopEpilogue(llvm::Value* current_tile_row, int64_t rows,
                             std::vector<ScalarVariable>* scalar_accumulators);

  Config config_;
  llvm::Value* lhs_;
  llvm::Value* rhs_;
  llvm::Value* addend_;
  llvm::Value* result_;
  llvm::IRBuilder<>* b_;
  KernelSupportLibrary ksl_;
  VectorSupportLibrary vsl_;
};

void RowMajorMatrixVectorProductEmitter::EmitOuterLoopBody(llvm::Value* row,
                                                           int64_t row_count) {
   std::vector<std::string> mht_30_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePScpuPStiled_dot_emitterDTcc mht_30(mht_30_v, 768, "", "./tensorflow/compiler/xla/service/cpu/tiled_dot_emitter.cc", "RowMajorMatrixVectorProductEmitter::EmitOuterLoopBody");

  MemoryTile lhs_memory_tile = GetLhsMemoryTile(/*row_start=*/row,
                                                /*row_count=*/row_count);
  std::vector<VectorVariable> vector_accumulators;
  std::vector<ScalarVariable> scalar_accumulators;
  vector_accumulators.reserve(row_count);
  scalar_accumulators.reserve(row_count);
  for (int64_t i = 0; i < row_count; i++) {
    vector_accumulators.emplace_back(&vsl_, vsl_.GetZeroVector());
    scalar_accumulators.emplace_back(&vsl_, vsl_.GetZeroScalar());
  }
  EmitInnerLoopTiled(&lhs_memory_tile, /*rows=*/row_count,
                     &vector_accumulators);
  EmitInnerLoopEpilogue(/*current_tile_row=*/row, /*rows=*/row_count,
                        &scalar_accumulators);

  std::vector<llvm::Value*> accumulator_values;
  std::transform(
      vector_accumulators.begin(), vector_accumulators.end(),
      std::back_inserter(accumulator_values),
      [](const VectorVariable& vector_var) { return vector_var.Get(); });

  std::vector<llvm::Value*> horizontal_sums;
  if (row_count == vsl_.vector_size()) {
    if (addend_) {
      horizontal_sums = vsl_.ComputeHorizontalSums(
          std::move(accumulator_values), vsl_.LoadVector(addend_, row));
    } else {
      horizontal_sums =
          vsl_.ComputeHorizontalSums(std::move(accumulator_values));
    }
  } else {
    horizontal_sums = vsl_.ComputeHorizontalSums(std::move(accumulator_values));
  }

  for (int i = 0; i < row_count; i++) {
    llvm::Value* result_value =
        vsl_.Add(horizontal_sums[i], scalar_accumulators[i].Get());
    llvm::Value* offset = b_->CreateAdd(b_->getInt64(i), row);
    if (addend_ && row_count != vsl_.vector_size()) {
      result_value = vsl_.Add(vsl_.LoadScalar(addend_, offset), result_value);
    }
    vsl_.StoreScalar(result_value, result_, offset);
  }
}

void RowMajorMatrixVectorProductEmitter::Emit() {
   std::vector<std::string> mht_31_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePScpuPStiled_dot_emitterDTcc mht_31(mht_31_v, 817, "", "./tensorflow/compiler/xla/service/cpu/tiled_dot_emitter.cc", "RowMajorMatrixVectorProductEmitter::Emit");

  // See the comment on the class declaration for the algorithm used here.
  int64_t row_remainder = m() % tile_rows();
  int64_t row_limit = m() - row_remainder;

  ksl_.For("dot.outer.tiled",
           /*start=*/0, /*end=*/row_limit, /*step=*/tile_rows(),
           [&](llvm::Value* row) { EmitOuterLoopBody(row, tile_rows()); });

  if (row_remainder != 0) {
    EmitOuterLoopBody(b_->getInt64(row_limit), row_remainder);
  }
}

void RowMajorMatrixVectorProductEmitter::EmitInnerLoopTiled(
    MemoryTile* lhs_memory_tile, int64_t rows,
    std::vector<VectorVariable>* vector_accumulators) {
   std::vector<std::string> mht_32_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePScpuPStiled_dot_emitterDTcc mht_32(mht_32_v, 836, "", "./tensorflow/compiler/xla/service/cpu/tiled_dot_emitter.cc", "RowMajorMatrixVectorProductEmitter::EmitInnerLoopTiled");

  int64_t column_limit = k() - (k() % tile_cols());

  ksl_.For("dot.inner.tiled", /*start=*/0, /*end=*/column_limit,
           /*step=*/tile_cols(), [&](llvm::Value* col) {
             std::vector<llvm::Value*> lhs_tile =
                 lhs_memory_tile->LoadTile(/*minor_dim_offset=*/col);
             llvm::Value* rhs_value = vsl_.LoadVector(rhs_, col);
             for (int i = 0; i < rows; i++) {
               llvm::Value* old_sum = (*vector_accumulators)[i].Get();
               (*vector_accumulators)[i].Set(
                   vsl_.Add(old_sum, vsl_.Mul(rhs_value, lhs_tile[i])));
             }
           });
}

void RowMajorMatrixVectorProductEmitter::EmitInnerLoopEpilogue(
    llvm::Value* current_tile_row, int64_t rows,
    std::vector<ScalarVariable>* scalar_accumulators) {
   std::vector<std::string> mht_33_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePScpuPStiled_dot_emitterDTcc mht_33(mht_33_v, 857, "", "./tensorflow/compiler/xla/service/cpu/tiled_dot_emitter.cc", "RowMajorMatrixVectorProductEmitter::EmitInnerLoopEpilogue");

  int64_t column_start = k() - (k() % tile_cols());
  if (column_start == k()) {
    return;
  }

  for (int r = 0; r < rows; r++) {
    llvm::Value* total_offset = b_->CreateMul(
        b_->CreateAdd(b_->getInt64(r), current_tile_row), b_->getInt64(k()));
    llvm::Value* lhs_base_pointer =
        vsl_.ComputeOffsetPointer(lhs_, total_offset);
    ksl_.For("dot.inner.epilg.inner", /*start=*/column_start, /*end=*/k(),
             /*step=*/1, [&](llvm::Value* scalar_col) {
               llvm::Value* product =
                   vsl_.Mul(vsl_.LoadScalar(lhs_base_pointer, scalar_col),
                            vsl_.LoadScalar(rhs_, scalar_col));
               llvm::Value* old_value = (*scalar_accumulators)[r].Get();
               (*scalar_accumulators)[r].Set(vsl_.Add(old_value, product));
             });
  }
}

// This class implements a tiled matrix multiplication algorithm, intended for
// multiplying small matrices that don't need cache tiling.
//
// In the future this can be used as the innermost GEBP loop in a GEMM kernel as
// described in "Goto, Kazushige, and Robert A. Geijn. "Anatomy of
// high-performance matrix multiplication." ACM Transactions on Mathematical
// Software (TOMS) 34.3 (2008): 12.".
//
// This only supports canonical dot operations (i.e. where the lhs contraction
// dimension is 1 and the rhs contraction dimension is 0) over row major
// matrices.
class TiledSmallGemmEmitter {
 public:
  // Describe the dimensions of the kernel.
  class Dimensions {
   public:
    explicit Dimensions(int64_t m, int64_t k, int64_t n)
        : m_(m), k_(k), n_(n) {
   std::vector<std::string> mht_34_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePScpuPStiled_dot_emitterDTcc mht_34(mht_34_v, 899, "", "./tensorflow/compiler/xla/service/cpu/tiled_dot_emitter.cc", "Dimensions");
}

    int64_t m() const {
   std::vector<std::string> mht_35_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePScpuPStiled_dot_emitterDTcc mht_35(mht_35_v, 904, "", "./tensorflow/compiler/xla/service/cpu/tiled_dot_emitter.cc", "m");
 return m_; }
    int64_t k() const {
   std::vector<std::string> mht_36_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePScpuPStiled_dot_emitterDTcc mht_36(mht_36_v, 908, "", "./tensorflow/compiler/xla/service/cpu/tiled_dot_emitter.cc", "k");
 return k_; }
    int64_t n() const {
   std::vector<std::string> mht_37_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePScpuPStiled_dot_emitterDTcc mht_37(mht_37_v, 912, "", "./tensorflow/compiler/xla/service/cpu/tiled_dot_emitter.cc", "n");
 return n_; }

    std::string ToString() const {
   std::vector<std::string> mht_38_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePScpuPStiled_dot_emitterDTcc mht_38(mht_38_v, 917, "", "./tensorflow/compiler/xla/service/cpu/tiled_dot_emitter.cc", "ToString");

      return absl::StrCat(m(), "x", k(), "x", n());
    }

   private:
    const int64_t m_;
    const int64_t k_;
    const int64_t n_;
  };

  // Represents the configuration of the emitter.  The LLVM IR emitted by the
  // emitter, modulo the LLVM values holding the input and output buffers, must
  // be a function of the instance of `Config` passed to it.
  //
  // `dims` holds the matrix multiplication dimensions.
  //
  // `max_vectorization_width` is the maximum vector width (i.e. the width of
  // the largest vector register we will use).  This can be larger than the
  // largest vector register supported by the machine -- LLVM will legalize
  // these large vector widths into legally sized vectors.
  //
  // `max_vector_count` is the maximum number of vectors of size
  // `max_vectorization_width` that we will attempt to process at once.
  //
  // `min_vectorization_width` is the smallest vector width the emitter will use
  // -- below that it will devolve to using a scalar loop.
  //
  // The innermost reduction loop executes the matrix multiply in tiles of size
  // [`tile_size_m`, `tile_size_k`] from the LHS and [`tile_size_k`,
  // <vectorization width>] in the RHS.
  class Config {
   public:
    explicit Config(PrimitiveType scalar_type, Dimensions dims,
                    int64_t max_vectorization_width, int64_t max_vector_count,
                    int64_t min_vectorization_width, int64_t tile_size_m,
                    int64_t tile_size_k)
        : scalar_type_(scalar_type),
          dims_(dims),
          max_vectorization_width_(max_vectorization_width),
          max_vector_count_(max_vector_count),
          min_vectorization_width_(min_vectorization_width),
          tile_size_m_(tile_size_m),
          tile_size_k_(tile_size_k) {
   std::vector<std::string> mht_39_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePScpuPStiled_dot_emitterDTcc mht_39(mht_39_v, 962, "", "./tensorflow/compiler/xla/service/cpu/tiled_dot_emitter.cc", "Config");
}

    std::string GetCacheKey() const {
   std::vector<std::string> mht_40_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePScpuPStiled_dot_emitterDTcc mht_40(mht_40_v, 967, "", "./tensorflow/compiler/xla/service/cpu/tiled_dot_emitter.cc", "GetCacheKey");

      return absl::StrCat("gemm_", PrimitiveType_Name(scalar_type()), "_",
                          dims().ToString(), "_", max_vectorization_width(),
                          "_", min_vectorization_width(), "_", tile_size_m(),
                          "_", tile_size_k());
    }

    PrimitiveType scalar_type() const {
   std::vector<std::string> mht_41_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePScpuPStiled_dot_emitterDTcc mht_41(mht_41_v, 977, "", "./tensorflow/compiler/xla/service/cpu/tiled_dot_emitter.cc", "scalar_type");
 return scalar_type_; }
    Dimensions dims() const {
   std::vector<std::string> mht_42_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePScpuPStiled_dot_emitterDTcc mht_42(mht_42_v, 981, "", "./tensorflow/compiler/xla/service/cpu/tiled_dot_emitter.cc", "dims");
 return dims_; }
    int64_t max_vectorization_width() const {
   std::vector<std::string> mht_43_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePScpuPStiled_dot_emitterDTcc mht_43(mht_43_v, 985, "", "./tensorflow/compiler/xla/service/cpu/tiled_dot_emitter.cc", "max_vectorization_width");
 return max_vectorization_width_; }
    int64_t max_vector_count() const {
   std::vector<std::string> mht_44_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePScpuPStiled_dot_emitterDTcc mht_44(mht_44_v, 989, "", "./tensorflow/compiler/xla/service/cpu/tiled_dot_emitter.cc", "max_vector_count");
 return max_vector_count_; }
    int64_t min_vectorization_width() const {
   std::vector<std::string> mht_45_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePScpuPStiled_dot_emitterDTcc mht_45(mht_45_v, 993, "", "./tensorflow/compiler/xla/service/cpu/tiled_dot_emitter.cc", "min_vectorization_width");
 return min_vectorization_width_; }

    int64_t tile_size_m() const {
   std::vector<std::string> mht_46_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePScpuPStiled_dot_emitterDTcc mht_46(mht_46_v, 998, "", "./tensorflow/compiler/xla/service/cpu/tiled_dot_emitter.cc", "tile_size_m");
 return tile_size_m_; }
    int64_t tile_size_k() const {
   std::vector<std::string> mht_47_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePScpuPStiled_dot_emitterDTcc mht_47(mht_47_v, 1002, "", "./tensorflow/compiler/xla/service/cpu/tiled_dot_emitter.cc", "tile_size_k");
 return tile_size_k_; }

   private:
    PrimitiveType scalar_type_;
    Dimensions dims_;
    int64_t max_vectorization_width_;
    int64_t max_vector_count_;
    int64_t min_vectorization_width_;
    int64_t tile_size_m_;
    int64_t tile_size_k_;
  };

  // Creates an instance of TiledSmallGemmEmitter that matrix-multiplies
  // `lhs` with `rhs` and stores the result in `result`.
  explicit TiledSmallGemmEmitter(Config config, llvm::Value* lhs,
                                 llvm::Value* rhs, llvm::Value* result,
                                 llvm::IRBuilder<>* b)
      : lhs_(lhs),
        rhs_(rhs),
        result_(result),
        config_(config),
        b_(b),
        ksl_(b_) {
   std::vector<std::string> mht_48_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePScpuPStiled_dot_emitterDTcc mht_48(mht_48_v, 1027, "", "./tensorflow/compiler/xla/service/cpu/tiled_dot_emitter.cc", "TiledSmallGemmEmitter");

    CHECK(
        max_vectorization_width() > 0 &&
        absl::has_single_bit(static_cast<uint64_t>(max_vectorization_width())));
    CHECK_GT(max_vector_count(), 0);
    CHECK(
        min_vectorization_width() > 0 &&
        absl::has_single_bit(static_cast<uint64_t>(min_vectorization_width())));
    CHECK_GE(max_vectorization_width(), min_vectorization_width());
    CHECK_GT(tile_size_k(), 0);
  }

  void Emit();

 private:
  // The HandleResiduesOnX helpers split the iteration space for dimension X
  // into a multiple of the tile size on dimension X and an epilogue.  These
  // helpers ultimately call into `EmitTiledGemm` for emitting the
  // tiled GEMM kernel.

  void HandleResiduesOnN();
  void HandleResiduesOnK(VectorSupportLibrary* vsl, llvm::Value* n_start,
                         llvm::Value* n_end);
  void HandleResiduesOnM(VectorSupportLibrary* vsl, int64_t tile_size_k,
                         llvm::Value* k_start, llvm::Value* k_end,
                         llvm::Value* n_start, llvm::Value* n_end);

  // This emits a tiled GEMM kernel.  For a detailed description see the comment
  // on the implementation.
  void EmitTiledGemm(VectorSupportLibrary* vsl, int64_t tile_size_k,
                     llvm::Value* k_start, llvm::Value* k_end,
                     llvm::Value* n_start, llvm::Value* n_end,
                     int64_t tile_size_m, llvm::Value* m_start,
                     llvm::Value* m_end);

  llvm::Value* GetInt64(int64_t value) {
   std::vector<std::string> mht_49_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePScpuPStiled_dot_emitterDTcc mht_49(mht_49_v, 1065, "", "./tensorflow/compiler/xla/service/cpu/tiled_dot_emitter.cc", "GetInt64");
 return b_->getInt64(value); }

  Config config() const {
   std::vector<std::string> mht_50_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePScpuPStiled_dot_emitterDTcc mht_50(mht_50_v, 1070, "", "./tensorflow/compiler/xla/service/cpu/tiled_dot_emitter.cc", "config");
 return config_; }
  Dimensions dims() const {
   std::vector<std::string> mht_51_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePScpuPStiled_dot_emitterDTcc mht_51(mht_51_v, 1074, "", "./tensorflow/compiler/xla/service/cpu/tiled_dot_emitter.cc", "dims");
 return config().dims(); }

  int64_t max_vectorization_width() const {
   std::vector<std::string> mht_52_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePScpuPStiled_dot_emitterDTcc mht_52(mht_52_v, 1079, "", "./tensorflow/compiler/xla/service/cpu/tiled_dot_emitter.cc", "max_vectorization_width");

    return config().max_vectorization_width();
  }
  int64_t max_vector_count() const {
   std::vector<std::string> mht_53_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePScpuPStiled_dot_emitterDTcc mht_53(mht_53_v, 1085, "", "./tensorflow/compiler/xla/service/cpu/tiled_dot_emitter.cc", "max_vector_count");
 return config().max_vector_count(); }
  int64_t min_vectorization_width() const {
   std::vector<std::string> mht_54_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePScpuPStiled_dot_emitterDTcc mht_54(mht_54_v, 1089, "", "./tensorflow/compiler/xla/service/cpu/tiled_dot_emitter.cc", "min_vectorization_width");

    return config().min_vectorization_width();
  }
  int64_t tile_size_m() const {
   std::vector<std::string> mht_55_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePScpuPStiled_dot_emitterDTcc mht_55(mht_55_v, 1095, "", "./tensorflow/compiler/xla/service/cpu/tiled_dot_emitter.cc", "tile_size_m");
 return config().tile_size_m(); }
  int64_t tile_size_k() const {
   std::vector<std::string> mht_56_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePScpuPStiled_dot_emitterDTcc mht_56(mht_56_v, 1099, "", "./tensorflow/compiler/xla/service/cpu/tiled_dot_emitter.cc", "tile_size_k");
 return config().tile_size_k(); }
  PrimitiveType scalar_type() const {
   std::vector<std::string> mht_57_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePScpuPStiled_dot_emitterDTcc mht_57(mht_57_v, 1103, "", "./tensorflow/compiler/xla/service/cpu/tiled_dot_emitter.cc", "scalar_type");
 return config().scalar_type(); }

  llvm::Value* lhs_;
  llvm::Value* rhs_;
  llvm::Value* result_;
  Config config_;

  llvm::IRBuilder<>* b_;
  KernelSupportLibrary ksl_;
};

void TiledSmallGemmEmitter::Emit() {
   std::vector<std::string> mht_58_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePScpuPStiled_dot_emitterDTcc mht_58(mht_58_v, 1117, "", "./tensorflow/compiler/xla/service/cpu/tiled_dot_emitter.cc", "TiledSmallGemmEmitter::Emit");
 HandleResiduesOnN(); }

void TiledSmallGemmEmitter::HandleResiduesOnN() {
   std::vector<std::string> mht_59_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePScpuPStiled_dot_emitterDTcc mht_59(mht_59_v, 1122, "", "./tensorflow/compiler/xla/service/cpu/tiled_dot_emitter.cc", "TiledSmallGemmEmitter::HandleResiduesOnN");

  // We can only iterate the `n` dimension for an extent that is divisible by
  // the vectorization width.  So we emit an outer loop that first processes the
  // largest extent in `n` that is divisible by max_vectorization_width, then
  // the largest remaining extent that is divisible by max_vectorization_width /
  // 2 etc.

  int64_t current_vectorization_width =
      max_vector_count() * max_vectorization_width();
  int64_t current_vector_count = max_vector_count();

  int64_t n_start = 0;
  while (n_start != dims().n() &&
         current_vectorization_width >= min_vectorization_width()) {
    int64_t n_end = dims().n() - (dims().n() % current_vectorization_width);
    if (n_start != n_end) {
      VectorSupportLibrary vsl(scalar_type(), current_vectorization_width, b_,
                               "gemm");
      HandleResiduesOnK(&vsl, GetInt64(n_start), GetInt64(n_end));
      n_start = n_end;
    }
    if (current_vector_count == 1) {
      current_vectorization_width /= 2;
    } else {
      current_vector_count--;
      current_vectorization_width =
          current_vector_count * max_vectorization_width();
    }
  }

  if (n_start != dims().n()) {
    VectorSupportLibrary vsl(scalar_type(), 1, b_, "gemm");
    ksl_.For("epi.n", n_start, dims().n(), 1, [&](llvm::Value* n_i) {
      llvm::Value* n_i_next = b_->CreateAdd(n_i, b_->getInt64(1));
      HandleResiduesOnK(&vsl, n_i, n_i_next);
    });
  }
}

void TiledSmallGemmEmitter::HandleResiduesOnK(VectorSupportLibrary* vsl,
                                              llvm::Value* n_start,
                                              llvm::Value* n_end) {
   std::vector<std::string> mht_60_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePScpuPStiled_dot_emitterDTcc mht_60(mht_60_v, 1166, "", "./tensorflow/compiler/xla/service/cpu/tiled_dot_emitter.cc", "TiledSmallGemmEmitter::HandleResiduesOnK");

  int64_t k_start = 0;
  int64_t k_end = dims().k() - (dims().k() % tile_size_k());
  if (k_end != k_start) {
    HandleResiduesOnM(vsl, tile_size_k(), GetInt64(k_start), GetInt64(k_end),
                      n_start, n_end);
    k_start = k_end;
  }

  if (k_start != dims().k()) {
    HandleResiduesOnM(vsl, dims().k() - k_start, GetInt64(k_start),
                      GetInt64(dims().k()), n_start, n_end);
  }
}

void TiledSmallGemmEmitter::HandleResiduesOnM(
    VectorSupportLibrary* vsl, int64_t tile_size_k, llvm::Value* k_start,
    llvm::Value* k_end, llvm::Value* n_start, llvm::Value* n_end) {
   std::vector<std::string> mht_61_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePScpuPStiled_dot_emitterDTcc mht_61(mht_61_v, 1186, "", "./tensorflow/compiler/xla/service/cpu/tiled_dot_emitter.cc", "TiledSmallGemmEmitter::HandleResiduesOnM");

  const int64_t m_end = dims().m() - dims().m() % tile_size_m();
  EmitTiledGemm(vsl, tile_size_k, k_start, k_end, n_start, n_end, tile_size_m(),
                GetInt64(0), GetInt64(m_end));

  if (m_end != dims().m()) {
    EmitTiledGemm(vsl, tile_size_k, k_start, k_end, n_start, n_end,
                  dims().m() - m_end, GetInt64(m_end), GetInt64(dims().m()));
  }
}

// The loop structure is:
//
// Iterate over dimension M as m:
//   Iterate over dimension N as n:
//     Iterate over dimension K as k:
//       OutputTile[m,n] += Dot(LhsTile[m,k], RhsTile[k,n])
//
// I.e. a just a tiled version of a "naive" GEMM.
//
// The tiling scheme is as follows:
//
// Let the LHS be:
//
//   +----+----+----+
//   | a0 | b0 | c0 | .
//   +----+----+----+ .
//   | a1 | b1 | c1 | .
//   +----+----+----+
//     ..     ..
//
// and the RHS be:
//
//   +----+----+----+----+
//   | p0 | p1 | p2 | p3 | .
//   +----+----+----+----+ .
//   | q0 | q1 | q2 | q3 | .
//   +----+----+----+----+
//   | r0 | r1 | r2 | r3 | .
//   +----+----+----+----+ .
//     ......    ......
//
// and let tile_size_m=2, tile_size_k=3 and the vector width (implicitly denoted
// by `vsl`) be 4.  Then we want to matrix multiply this tile to get a [2,4]
// matrix that we can increment the result matrix by.
//
// First broadcast the rows row in LHS to 3 vectors of width 4, giving us a rank
// 3 array, L, of dimension [2,3,4]:
//
//       L[0,_,_]           *      L[1,_,_]
//                          *
//   +----+----+----+----+  *  +----+----+----+----+
//   | a0 | a0 | a0 | a0 |  *  | a1 | a1 | a1 | a1 |
//   +----+----+----+----+  *  +----+----+----+----+
//   | b0 | b0 | b0 | b0 |  *  | b1 | b1 | b1 | b1 |
//   +----+----+----+----+  *  +----+----+----+----+
//   | c0 | c0 | c0 | c0 |  *  | c1 | c1 | c1 | c1 |
//   +----+----+----+----+  *  +----+----+----+----+
//
//
// Then we FMA L[0,_,_] with the RHS to get the first row of the result and
// L[1,_,_] with the RHS to get the second row of the result.  For example,
// L[0,_,_] is computed as:
//
//   +----+----+----+----+   +----+----+----+----+
//   | a0 | a0 | a0 | a0 | * | p0 | p1 | p2 | p3 |   +
//   +----+----+----+----+   +----+----+----+----+
//
//   +----+----+----+----+   +----+----+----+----+
//   | b0 | b0 | b0 | b0 | * | q0 | q1 | q2 | q3 |   +
//   +----+----+----+----+   +----+----+----+----+
//
//   +----+----+----+----+   +----+----+----+----+
//   | c0 | c0 | c0 | c0 | * | r0 | r1 | r2 | r3 |
//   +----+----+----+----+   +----+----+----+----+
//
// to get:
//
//   +-------------------+-------------------+-------------------+---------
//   | a0*p0+b0*q0+c0*r0 | a0*p1+b0*q1+c0*r1 | a0*p2+b0*q2+c0*r2 |  ...
//   +-------------------+-------------------+-------------------+---------
void TiledSmallGemmEmitter::EmitTiledGemm(
    VectorSupportLibrary* vsl, int64_t tile_size_k, llvm::Value* k_start,
    llvm::Value* k_end, llvm::Value* n_start, llvm::Value* n_end,
    int64_t tile_size_m, llvm::Value* m_start, llvm::Value* m_end) {
   std::vector<std::string> mht_62_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePScpuPStiled_dot_emitterDTcc mht_62(mht_62_v, 1273, "", "./tensorflow/compiler/xla/service/cpu/tiled_dot_emitter.cc", "TiledSmallGemmEmitter::EmitTiledGemm");

  ksl_.For("dot.m", m_start, m_end, tile_size_m, [&](llvm::Value* m_i) {
    MemoryTile result_memory_tile(vsl, b_, /*matrix=*/result_,
                                  /*matrix_size_along_minor_dim=*/dims().n(),
                                  /*major_dim_offset=*/m_i,
                                  /*tile_size_along_major_dim=*/tile_size_m);
    MemoryTile lhs_memory_tile(vsl, b_, /*matrix=*/lhs_,
                               /*matrix_size_along_minor_dim=*/dims().k(),
                               /*major_dim_offset=*/m_i,
                               /*tile_size_along_major_dim=*/tile_size_m);
    ksl_.For(
        "dot.n", n_start, n_end, vsl->vector_size(), [&](llvm::Value* n_i) {
          TileVariable result_tile_var(vsl, result_memory_tile.LoadTile(n_i));
          ksl_.For("dot.k", k_start, k_end, tile_size_k, [&](llvm::Value* k_i) {
            MemoryTile rhs_memory_tile(vsl, b_, rhs_, dims().n(), k_i,
                                       tile_size_k);
            std::vector<std::vector<llvm::Value*>> lhs_tile =
                lhs_memory_tile.LoadBroadcastTile(k_i, tile_size_k);
            std::vector<llvm::Value*> rhs_tile = rhs_memory_tile.LoadTile(n_i);
            std::vector<llvm::Value*> result_tile = result_tile_var.Get();
            for (int64_t r_m_i = 0; r_m_i < tile_size_m; r_m_i++) {
              for (int64_t r_k_i = 0; r_k_i < tile_size_k; r_k_i++) {
                result_tile[r_m_i] =
                    vsl->MulAdd(lhs_tile[r_m_i][r_k_i], rhs_tile[r_k_i],
                                result_tile[r_m_i]);
              }
            }
            result_tile_var.Set(result_tile);
          });

          result_memory_tile.StoreTile(result_tile_var.Get(), n_i);
        });
  });
}

llvm::Type* GetPointerToElementType(llvm::Type* pointer_type) {
   std::vector<std::string> mht_63_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePScpuPStiled_dot_emitterDTcc mht_63(mht_63_v, 1311, "", "./tensorflow/compiler/xla/service/cpu/tiled_dot_emitter.cc", "GetPointerToElementType");

  llvm::Type* type = pointer_type->getPointerElementType();
  while (auto* array_type = llvm::dyn_cast<llvm::ArrayType>(type)) {
    type = array_type->getElementType();
  }

  return type->getPointerTo();
}

struct GemvBuffersWithCanonicalType {
  llvm::Value* lhs_canonicalized;
  llvm::Value* rhs_canonicalized;
  llvm::Value* addend_canonicalized;
  llvm::Value* result_canonicalized;
};

GemvBuffersWithCanonicalType GetGemvBuffersWithCanonicalType(
    llvm::Value* lhs, llvm::Value* rhs, llvm::Value* addend,
    llvm::Value* result, llvm::IRBuilder<>* b) {
   std::vector<std::string> mht_64_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePScpuPStiled_dot_emitterDTcc mht_64(mht_64_v, 1332, "", "./tensorflow/compiler/xla/service/cpu/tiled_dot_emitter.cc", "GetGemvBuffersWithCanonicalType");

  // We characterize a GEMV operation via M and K, since N is implicitly 1.
  // This means the GEMV that multiplies (say) [5,6] with [6,1] is implemented
  // by the same GEMV that multiplies [5,6] with [1,6].  However, the
  // `llvm::Types` for the inputs to the two GEMVs don't match (in a trivial
  // sense -- the in memory representations are the same) since they're computed
  // from the `xla::Shape`s.  Since we want to be able to call the same
  // `llvm::Function` for the two GEMVs we canonicalize the types of the GEMV
  // inputs here into the same type.
  GemvBuffersWithCanonicalType buffers_with_canonical_type;
  llvm::Type* lhs_type = lhs->getType();
  llvm::Type* rhs_type = rhs->getType();
  llvm::Type* addend_type = addend ? addend->getType() : nullptr;
  llvm::Type* result_type = result->getType();

  buffers_with_canonical_type.lhs_canonicalized =
      b->CreateBitCast(lhs, GetPointerToElementType(lhs_type));
  buffers_with_canonical_type.rhs_canonicalized =
      b->CreateBitCast(rhs, GetPointerToElementType(rhs_type));
  buffers_with_canonical_type.addend_canonicalized =
      addend ? b->CreateBitCast(addend, GetPointerToElementType(addend_type))
             : nullptr;
  buffers_with_canonical_type.result_canonicalized =
      b->CreateBitCast(result, GetPointerToElementType(result_type));

  return buffers_with_canonical_type;
}

}  // namespace

void EmitRowMajorGemv(PrimitiveType scalar_type, int64_t tile_rows,
                      int64_t tile_cols, int64_t m, int64_t k, llvm::Value* lhs,
                      llvm::Value* rhs, llvm::Value* addend,
                      llvm::Value* result, llvm::IRBuilder<>* b,
                      const HloModuleConfig& module_config) {
   std::vector<std::string> mht_65_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePScpuPStiled_dot_emitterDTcc mht_65(mht_65_v, 1369, "", "./tensorflow/compiler/xla/service/cpu/tiled_dot_emitter.cc", "EmitRowMajorGemv");

  RowMajorMatrixVectorProductEmitter::Config config(
      /*scalar_type=*/scalar_type,
      /*tile_rows=*/tile_rows, /*tile_cols=*/tile_cols,
      /*m=*/m, /*k=*/k, /*has_addend=*/addend != nullptr);

  GemvBuffersWithCanonicalType canonical_inputs =
      GetGemvBuffersWithCanonicalType(lhs, rhs, addend, result, b);

  KernelSupportLibrary::EmitAndCallOutlinedKernel(
      module_config, b, config.GetCacheKey(),
      canonical_inputs.lhs_canonicalized, canonical_inputs.rhs_canonicalized,
      canonical_inputs.addend_canonicalized,
      canonical_inputs.result_canonicalized,
      [&config, b, &canonical_inputs](llvm::Value* lhs, llvm::Value* rhs,
                                      llvm::Value* addend,
                                      llvm::Value* result) {
        RowMajorMatrixVectorProductEmitter emitter(config, lhs, rhs, addend,
                                                   result, b);
        emitter.Emit();
      });
}

void EmitColumnMajorGemv(PrimitiveType scalar_type, int64_t tile_rows,
                         int64_t tile_cols, int64_t m, int64_t k,
                         llvm::Value* lhs, llvm::Value* rhs,
                         llvm::Value* addend, llvm::Value* result,
                         llvm::IRBuilder<>* b,
                         const HloModuleConfig& module_config) {
   std::vector<std::string> mht_66_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePScpuPStiled_dot_emitterDTcc mht_66(mht_66_v, 1400, "", "./tensorflow/compiler/xla/service/cpu/tiled_dot_emitter.cc", "EmitColumnMajorGemv");

  ColumnMajorMatrixVectorProductEmitter::Config config(
      /*scalar_type=*/scalar_type,
      /*tile_rows=*/tile_rows, /*tile_cols=*/tile_cols,
      /*m=*/m, /*k=*/k, /*has_addend=*/addend != nullptr);

  GemvBuffersWithCanonicalType canonical_inputs =
      GetGemvBuffersWithCanonicalType(lhs, rhs, addend, result, b);

  KernelSupportLibrary::EmitAndCallOutlinedKernel(
      module_config, b, config.GetCacheKey(),
      canonical_inputs.lhs_canonicalized, canonical_inputs.rhs_canonicalized,
      canonical_inputs.addend_canonicalized,
      canonical_inputs.result_canonicalized,
      [&config, b, &canonical_inputs](llvm::Value* lhs, llvm::Value* rhs,
                                      llvm::Value* addend,
                                      llvm::Value* result) {
        ColumnMajorMatrixVectorProductEmitter emitter(config, lhs, rhs, addend,
                                                      result, b);
        emitter.Emit();
      });
}

void EmitSmallGemm(PrimitiveType scalar_type, int64_t m, int64_t k, int64_t n,
                   int64_t max_vectorization_width, int64_t max_vector_count,
                   int64_t min_vectorization_width, int64_t tile_size_m,
                   int64_t tile_size_k, llvm::Value* lhs, llvm::Value* rhs,
                   llvm::Value* result, llvm::IRBuilder<>* b,
                   const HloModuleConfig& module_config) {
   std::vector<std::string> mht_67_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePScpuPStiled_dot_emitterDTcc mht_67(mht_67_v, 1431, "", "./tensorflow/compiler/xla/service/cpu/tiled_dot_emitter.cc", "EmitSmallGemm");

  TiledSmallGemmEmitter::Config config(
      /*scalar_type=*/scalar_type,
      TiledSmallGemmEmitter::Dimensions{/*m=*/m, /*k=*/k, /*n=*/n},
      /*max_vectorization_width=*/max_vectorization_width,
      /*max_vector_count=*/max_vector_count,
      /*min_vectorization_width=*/min_vectorization_width,
      /*tile_size_m=*/tile_size_m, /*tile_size_k=*/tile_size_k);

  KernelSupportLibrary::EmitAndCallOutlinedKernel(
      module_config, b, config.GetCacheKey(), lhs, rhs, result,
      [&](llvm::Value* lhs, llvm::Value* rhs, llvm::Value* result) {
        TiledSmallGemmEmitter small_gemm_emitter(config, /*lhs=*/lhs,
                                                 /*rhs=*/rhs,
                                                 /*result=*/result, b);
        small_gemm_emitter.Emit();
      });
}

}  // namespace cpu
}  // namespace xla
