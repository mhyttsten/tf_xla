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

#ifndef TENSORFLOW_CORE_KERNELS_EIGEN_SPATIAL_CONVOLUTIONS_INL_H_
#define TENSORFLOW_CORE_KERNELS_EIGEN_SPATIAL_CONVOLUTIONS_INL_H_
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
class MHTracer_DTPStensorflowPScorePSkernelsPSeigen_spatial_convolutionsSLinlDTh {
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
   MHTracer_DTPStensorflowPScorePSkernelsPSeigen_spatial_convolutionsSLinlDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSkernelsPSeigen_spatial_convolutionsSLinlDTh() {
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


#include "tensorflow/core/kernels/eigen_convolution_helpers.h"

// Note this header is used in both TF and TFLite.
namespace Eigen {

namespace internal {

#if !EIGEN_ALTIVEC_USE_CUSTOM_PACK
// WARNING: Most of the code here implicitly assumes that the matrix is in
// ColMajor layout. This is guaranteed by the tensor contraction (see
// TensorContraction.h).
//
// Inside Eigen a tensor contraction is represented by a matrix multiplication.
// We don't want to actually extract image patches and reshape the result into
// a matrix (this involves allocating huge extra memory), so the patch
// extraction and reshape operations are implicit.
//
// TensorContractionInputMapper takes a matrix index and returns the coefficient
// (or the packet) of the "virtual tensor", that would be at that index if we
// were to actually reshape the result of patch extraction.
//
// TensorContractionSubMapper provides a similar view into the "virtual matrix"
// at the given vertical and horizontal offsets.
//
// "Virtual matrix" dimensions:
//   *0: kernelChannels * kernelRows * kernelCols;
//    1: out_height * out_width; * OTHERS (e.g batches, etc...)
//
// *) extracted patches are continuous in memory (innermost dimension assuming
//    col major layout)
//
// With this dimensions:
//   row - offset within a single patch (in code: patchId)
//   col - index of the extracted patch (in code: patchIndex)
//         patchIndex ∈ [0..num_patches * OTHERS] (batch and other dimensions)
//
// TODO(ezhulenev): Consolidate this part of the code with the image patch
// extraction code since they are both very similar.

template <typename NewDimension, Index Rows, Index Cols, typename ArgType,
          typename Device, typename Scalar_, typename Index,
          typename nocontract_t, typename contract_t, int Side, int packet_size,
          bool inner_dim_contiguous, bool inner_dim_reordered, int Alignment>
class TensorContractionInputMapper<
    Scalar_, Index, Side,
    TensorEvaluator<
        const TensorReshapingOp<NewDimension,
                                const TensorImagePatchOp<Rows, Cols, ArgType> >,
        Device>,
    nocontract_t, contract_t, packet_size, inner_dim_contiguous,
    inner_dim_reordered, Alignment> {
 public:
  typedef Scalar_ Scalar;

  typedef TensorContractionInputMapper<
      Scalar, Index, Side,
      TensorEvaluator<
          const TensorReshapingOp<
              NewDimension, const TensorImagePatchOp<Rows, Cols, ArgType> >,
          Device>,
      nocontract_t, contract_t, packet_size, inner_dim_contiguous,
      inner_dim_reordered, Alignment>
      Self;

  typedef TensorContractionSubMapper<
      Scalar, Index, Side,
      TensorEvaluator<
          const TensorReshapingOp<
              NewDimension, const TensorImagePatchOp<Rows, Cols, ArgType> >,
          Device>,
      nocontract_t, contract_t, packet_size, inner_dim_contiguous,
      inner_dim_reordered, Alignment>
      SubMapper;

  typedef SubMapper VectorMapper;
  typedef SubMapper LinearMapper;
  typedef typename packet_traits<Scalar>::type Packet;

  typedef TensorEvaluator<ArgType, Device> TensorEvaluatorT;

  EIGEN_DEVICE_FUNC
  TensorContractionInputMapper(
      const TensorEvaluator<
          const TensorReshapingOp<
              NewDimension, const TensorImagePatchOp<Rows, Cols, ArgType> >,
          Device>& tensor,
      const nocontract_t&, const nocontract_t&, const contract_t&,
      const contract_t&)
      : m_impl(tensor.impl().impl()) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSeigen_spatial_convolutionsSLinlDTh mht_0(mht_0_v, 276, "", "./tensorflow/core/kernels/eigen_spatial_convolutions-inl.h", "TensorContractionInputMapper");

    Index patch_rows;
    Index patch_depth;
    if (internal::traits<ArgType>::Layout == ColMajor) {
      patch_depth = tensor.impl().dimensions()[0];
      patch_rows = tensor.impl().dimensions()[1];
      m_patch_cols = tensor.impl().dimensions()[2];
      m_num_patches = tensor.impl().dimensions()[3];
    } else {
      const size_t NumDims = tensor.impl().dimensions().size();
      patch_depth = tensor.impl().dimensions()[NumDims - 1];
      patch_rows = tensor.impl().dimensions()[NumDims - 2];
      m_patch_cols = tensor.impl().dimensions()[NumDims - 3];
      m_num_patches = tensor.impl().dimensions()[NumDims - 4];
    }

    // Strides for navigating through the single patch.
    m_patch_row_stride = patch_depth;
    m_patch_col_stride = patch_rows * m_patch_row_stride;

    m_patch_row_inflate_strides = tensor.impl().rowInflateStride();
    m_patch_col_inflate_strides = tensor.impl().colInflateStride();

    m_colStride = patch_rows;

    m_outputRows = tensor.impl().outputRows();
    m_outputCols = tensor.impl().outputCols();
    m_row_strides = tensor.impl().userRowStride();
    m_col_strides = tensor.impl().userColStride();

    m_in_row_strides = tensor.impl().userInRowStride();
    m_in_col_strides = tensor.impl().userInColStride();

    if (internal::traits<ArgType>::Layout == ColMajor) {
      m_inputRows = tensor.impl().impl().dimensions()[1];
      m_inputCols = tensor.impl().impl().dimensions()[2];
    } else {
      const int NumDims = tensor.impl().impl().dimensions().size();
      m_inputRows = tensor.impl().impl().dimensions()[NumDims - 2];
      m_inputCols = tensor.impl().impl().dimensions()[NumDims - 3];
    }

    m_rowInputStride = patch_depth;
    m_colInputStride = patch_depth * m_inputRows;
    m_patchInputStride = patch_depth * m_inputRows * m_inputCols;

    m_rowPaddingTop = tensor.impl().rowPaddingTop();
    m_colPaddingLeft = tensor.impl().colPaddingLeft();

    m_fastPatchRowStride =
        internal::TensorIntDivisor<Index>(m_patch_row_stride);
    m_fastPatchColStride =
        internal::TensorIntDivisor<Index>(m_patch_col_stride);
    m_fastInputRowStride =
        internal::TensorIntDivisor<Index>(m_patch_row_inflate_strides);
    m_fastInputColStride =
        internal::TensorIntDivisor<Index>(m_patch_col_inflate_strides);
    m_fastNumPatches = internal::TensorIntDivisor<Index>(m_num_patches);
    m_fastColStride = internal::TensorIntDivisor<Index>(m_colStride);
    m_fastOutputRows = internal::TensorIntDivisor<Index>(m_outputRows);
    m_fastDimZero = internal::TensorIntDivisor<Index>(patch_depth);
  }

  EIGEN_DEVICE_FUNC
  TensorContractionInputMapper(const TensorContractionInputMapper& base_mapper)
      : m_impl(base_mapper.m_impl) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSeigen_spatial_convolutionsSLinlDTh mht_1(mht_1_v, 344, "", "./tensorflow/core/kernels/eigen_spatial_convolutions-inl.h", "TensorContractionInputMapper");

    m_patch_cols = base_mapper.m_patch_cols;
    m_num_patches = base_mapper.m_num_patches;

    m_patch_row_stride = base_mapper.m_patch_row_stride;
    m_patch_col_stride = base_mapper.m_patch_col_stride;

    m_patch_row_inflate_strides = base_mapper.m_patch_row_inflate_strides;
    m_patch_col_inflate_strides = base_mapper.m_patch_col_inflate_strides;

    m_colStride = base_mapper.m_colStride;

    m_rowInputStride = base_mapper.m_rowInputStride;
    m_colInputStride = base_mapper.m_colInputStride;
    m_patchInputStride = base_mapper.m_patchInputStride;

    m_inputRows = base_mapper.m_inputRows;
    m_inputCols = base_mapper.m_inputCols;

    m_outputRows = base_mapper.m_outputRows;
    m_outputCols = base_mapper.m_outputCols;
    m_row_strides = base_mapper.m_row_strides;
    m_col_strides = base_mapper.m_col_strides;

    m_in_row_strides = base_mapper.m_in_row_strides;
    m_in_col_strides = base_mapper.m_in_col_strides;

    m_rowPaddingTop = base_mapper.m_rowPaddingTop;
    m_colPaddingLeft = base_mapper.m_colPaddingLeft;

    m_fastPatchRowStride = base_mapper.m_fastPatchRowStride;
    m_fastPatchColStride = base_mapper.m_fastPatchColStride;
    m_fastInputRowStride = base_mapper.m_fastInputRowStride;
    m_fastInputColStride = base_mapper.m_fastInputColStride;
    m_fastNumPatches = base_mapper.m_fastNumPatches;
    m_fastColStride = base_mapper.m_fastColStride;
    m_fastOutputRows = base_mapper.m_fastOutputRows;
    m_fastDimZero = base_mapper.m_fastDimZero;
  }

  // If true, turns off some optimizations for loading packets since the image
  // patches are "non-standard" such as there are non-trivial strides or
  // inflations in the input.
  EIGEN_DEVICE_FUNC
  EIGEN_ALWAYS_INLINE bool nonStandardPatches() const {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSeigen_spatial_convolutionsSLinlDTh mht_2(mht_2_v, 391, "", "./tensorflow/core/kernels/eigen_spatial_convolutions-inl.h", "nonStandardPatches");

    return m_in_row_strides != 1 || m_in_col_strides != 1 ||
           m_patch_row_inflate_strides != 1 || m_patch_col_inflate_strides != 1;
  }

  EIGEN_DEVICE_FUNC
  EIGEN_STRONG_INLINE SubMapper getSubMapper(Index i, Index j) const {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSeigen_spatial_convolutionsSLinlDTh mht_3(mht_3_v, 400, "", "./tensorflow/core/kernels/eigen_spatial_convolutions-inl.h", "getSubMapper");

    return SubMapper(*this, i, j);
  }

  EIGEN_DEVICE_FUNC
  EIGEN_STRONG_INLINE LinearMapper getLinearMapper(Index i, Index j) const {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSeigen_spatial_convolutionsSLinlDTh mht_4(mht_4_v, 408, "", "./tensorflow/core/kernels/eigen_spatial_convolutions-inl.h", "getLinearMapper");

    return LinearMapper(*this, i, j);
  }

  EIGEN_DEVICE_FUNC
  EIGEN_ALWAYS_INLINE Scalar operator()(Index row) const {
    Index rowIndex, colIndex, otherIndex;
    computeBaseIndices(0, rowIndex, colIndex, otherIndex);
    return loadCoeff(row, rowIndex, colIndex, otherIndex);
  }

  // Load the coefficient at the patchIndex location instead of the usual
  // m_rowIndex,
  // m_colIndex, m_otherIndex. This is currently only used by the gpu code.
  // EIGEN_DEVICE_FUNC
  EIGEN_DEVICE_FUNC
  EIGEN_STRONG_INLINE Scalar operator()(Index row, Index patchIndex) const {
    Index rowIndex, colIndex, otherIndex;
    computeBaseIndices(patchIndex, rowIndex, colIndex, otherIndex);
    return loadCoeff(row, rowIndex, colIndex, otherIndex);
  }

  EIGEN_DEVICE_FUNC
  EIGEN_ALWAYS_INLINE Packet loadPacket(Index row) const {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSeigen_spatial_convolutionsSLinlDTh mht_5(mht_5_v, 434, "", "./tensorflow/core/kernels/eigen_spatial_convolutions-inl.h", "loadPacket");

    Index rowIndex, colIndex, otherIndex;
    computeBaseIndices(0, rowIndex, colIndex, otherIndex);
    return loadPacket(row, rowIndex, colIndex, otherIndex);
  }

  // Load the packet at the patchIndex location instead of the usual m_rowIndex,
  // m_colIndex, m_otherIndex. This is currently only used by the gpu code.
  EIGEN_DEVICE_FUNC
  EIGEN_ALWAYS_INLINE Packet loadPacket(Index row, Index patchIndex) const {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSeigen_spatial_convolutionsSLinlDTh mht_6(mht_6_v, 446, "", "./tensorflow/core/kernels/eigen_spatial_convolutions-inl.h", "loadPacket");

    Index rowIndex, colIndex, otherIndex;
    computeBaseIndices(patchIndex, rowIndex, colIndex, otherIndex);
    return loadPacket(row, rowIndex, colIndex, otherIndex);
  }

  EIGEN_DEVICE_FUNC
  EIGEN_ALWAYS_INLINE const TensorEvaluator<ArgType, Device>& impl() const {
    return m_impl;
  }

  EIGEN_DEVICE_FUNC
  EIGEN_ALWAYS_INLINE Index patchDepth() const {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSeigen_spatial_convolutionsSLinlDTh mht_7(mht_7_v, 461, "", "./tensorflow/core/kernels/eigen_spatial_convolutions-inl.h", "patchDepth");
 return m_rowInputStride; }
  EIGEN_DEVICE_FUNC
  EIGEN_ALWAYS_INLINE Index patchRows() const {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSeigen_spatial_convolutionsSLinlDTh mht_8(mht_8_v, 466, "", "./tensorflow/core/kernels/eigen_spatial_convolutions-inl.h", "patchRows");
 return m_colStride; }
  EIGEN_DEVICE_FUNC
  EIGEN_ALWAYS_INLINE Index patchCols() const {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSeigen_spatial_convolutionsSLinlDTh mht_9(mht_9_v, 471, "", "./tensorflow/core/kernels/eigen_spatial_convolutions-inl.h", "patchCols");
 return m_patch_cols; }

 private:
  friend class TensorContractionSubMapper<
      Scalar, Index, Side,
      TensorEvaluator<
          const TensorReshapingOp<
              NewDimension, const TensorImagePatchOp<Rows, Cols, ArgType> >,
          Device>,
      nocontract_t, contract_t, packet_size, inner_dim_contiguous,
      inner_dim_reordered, Alignment>;

  // Load coefficient from a patch specified by the "within patch offset"
  // (patchId) and the precomputed indices of the first element of the patch.
  EIGEN_DEVICE_FUNC
  EIGEN_STRONG_INLINE Scalar loadCoeff(Index patchId, Index rowIndex,
                                       Index colIndex, Index otherIndex) const {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSeigen_spatial_convolutionsSLinlDTh mht_10(mht_10_v, 490, "", "./tensorflow/core/kernels/eigen_spatial_convolutions-inl.h", "loadCoeff");

    // Find the offset of the element wrt the location of the first element.
    const Index patchOffset = patchId / m_fastDimZero;

    const Index colOffset = patchOffset / m_fastColStride;
    const Index inputCol = colIndex + colOffset * m_in_col_strides;
    const Index origInputCol =
        (m_patch_col_inflate_strides == 1)
            ? inputCol
            : ((inputCol >= 0) ? (inputCol / m_fastInputColStride) : 0);

    const Index rowOffset = patchOffset - colOffset * m_colStride;
    const Index inputRow = rowIndex + rowOffset * m_in_row_strides;
    const Index origInputRow =
        (m_patch_row_inflate_strides == 1)
            ? inputRow
            : ((inputRow >= 0) ? (inputRow / m_fastInputRowStride) : 0);
    if (origInputCol < 0 || origInputRow < 0 || origInputCol >= m_inputCols ||
        origInputRow >= m_inputRows ||
        (inputCol != origInputCol * m_patch_col_inflate_strides) ||
        (inputRow != origInputRow * m_patch_row_inflate_strides)) {
      return Scalar(0);
    }
    const Index depth = patchId - patchOffset * patchDepth();
    const Index inputIndex = depth + origInputRow * m_rowInputStride +
                             origInputCol * m_colInputStride + otherIndex;
    return m_impl.coeff(inputIndex);
  }

  // This is the same as loadCoeff(...), but optimized for all `inflate_strides`
  // and `in_strides` equal to 1 (template specialization without templates).
  EIGEN_DEVICE_FUNC
  EIGEN_STRONG_INLINE Scalar loadCoeffStandard(Index patchId, Index rowIndex,
                                               Index colIndex,
                                               Index otherIndex) const {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSeigen_spatial_convolutionsSLinlDTh mht_11(mht_11_v, 527, "", "./tensorflow/core/kernels/eigen_spatial_convolutions-inl.h", "loadCoeffStandard");

    eigen_assert(!nonStandardPatches());

    // Find the offset of the element wrt the location of the first element.
    const Index patchOffset = patchId / m_fastDimZero;
    const Index colOffset = patchOffset / m_fastColStride;
    const Index rowOffset = patchOffset - colOffset * m_colStride;
    const Index inputCol = colIndex + colOffset;
    const Index inputRow = rowIndex + rowOffset;
    if (inputCol < 0 || inputCol >= m_inputCols || inputRow < 0 ||
        inputRow >= m_inputRows) {
      return Scalar(0);
    }
    const Index depth = patchId - patchOffset * patchDepth();
    const Index inputIndex = depth + inputRow * m_rowInputStride +
                             inputCol * m_colInputStride + otherIndex;
    return m_impl.coeff(inputIndex);
  }

  // Load packet from a patch specified by the "within patch offset"
  // (patchId) and the precomputed indices of the first element of the patch.
  EIGEN_DEVICE_FUNC
  EIGEN_ALWAYS_INLINE Packet loadPacket(Index patchId, Index rowIndex,
                                        Index colIndex,
                                        Index otherIndex) const {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSeigen_spatial_convolutionsSLinlDTh mht_12(mht_12_v, 554, "", "./tensorflow/core/kernels/eigen_spatial_convolutions-inl.h", "loadPacket");

    const Index packetSize = internal::unpacket_traits<Packet>::size;
    EIGEN_STATIC_ASSERT(packetSize > 1, YOU_MADE_A_PROGRAMMING_MISTAKE)
    eigen_assert(patchId < patchDepth() * patchRows() * m_patch_cols);

    if (nonStandardPatches()) {
      return packetWithPossibleZero(patchId, rowIndex, colIndex, otherIndex);
    }
    typedef decltype(m_impl) TensorEvaluatorT;
    return loadPacketStandard<Packet, TensorEvaluatorT>(patchId, rowIndex,
                                                        colIndex, otherIndex);
  }

  // Helper function to load a 'partial' packet - this is the single column
  // part of a packet that is split across two columns. In the 'partial' packet,
  // the elements corresponding to the column (specified through colOffset) are
  // loaded and the rest of the elements are zero-filled into the 'partial'
  // packet. This function is called from loadPacketStandardFromTwoColumns().
  // This code path is exercised only when the packet type supports masked load
  // and when the partial packet load is available in the TensorEvaluator.
  EIGEN_DEVICE_FUNC
  EIGEN_ALWAYS_INLINE Packet loadPartialPacketStandard(
      Index rowIndex, Index colIndex, Index otherIndex, Index patchId,
      const Index span[], const Index patchOffsets[], Index colOffset) const {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSeigen_spatial_convolutionsSLinlDTh mht_13(mht_13_v, 580, "", "./tensorflow/core/kernels/eigen_spatial_convolutions-inl.h", "loadPartialPacketStandard");

    const Index inputCol = colIndex + colOffset;
    const Index rowOffsets[2] = {patchOffsets[0] - colOffset * m_colStride,
                                 patchOffsets[1] - colOffset * m_colStride};
    const Index inputRows[2] = {rowIndex + rowOffsets[0],
                                rowIndex + rowOffsets[1]};

    if (inputRows[0] >= m_inputRows || inputRows[1] < 0 ||
        inputCol >= m_inputCols || inputCol < 0) {
      // Partial packet is all zeros
      return internal::pset1<Packet>(Scalar(0));
    } else if (inputRows[0] >= 0 && inputRows[1] < m_inputRows) {
      // From inputIndex-span[0], we need to load elements starting from index
      // span[0] all the way upto (and including) span[1].
      const Index depth = patchId - patchOffsets[0] * patchDepth();
      const Index inputIndex = depth + inputRows[0] * m_rowInputStride +
                               inputCol * m_colInputStride + otherIndex;
      return m_impl.template partialPacket<Packet>(
          inputIndex - span[0], mask<Packet>(span[0], span[1] + 1));
    } else {
      // Using slow path for this partial packet.
      // We need to load elements starting from index span[0] all the way upto
      // (and including) span[1]. We split this load into 3 parts:
      // 0 : span[0]-1 - Zeros will be loaded for these indices
      // span[0] : span[1] - Elements will be loaded here for these indices
      // span[1]+1 : packetSize-1 - Zeross will be loaded for these indices
      const Index packetSize = internal::unpacket_traits<Packet>::size;
      EIGEN_ALIGN_MAX
      typename internal::remove_const<Scalar>::type values[packetSize];
      for (int i = 0; i < span[0]; ++i) values[i] = Scalar(0);
      for (int i = span[0]; i < span[1] + 1; ++i)
        values[i] =
            loadCoeff(patchId - span[0] + i, rowIndex, colIndex, otherIndex);
      for (int i = span[1] + 1; i < packetSize; ++i) values[i] = Scalar(0);
      return internal::pload<Packet>(values);
    }
  }

  // Helper function to load a packet that is split across two columns.
  // If required, this function is called from loadPacketStandard() when the
  // packet type supports masked load and when the partial packet load is
  // available in the TensorEvaluator.
  EIGEN_DEVICE_FUNC
  EIGEN_ALWAYS_INLINE Packet loadPacketStandardFromTwoColumns(
      Index patchId, Index rowIndex, Index colIndex, Index otherIndex,
      const Index patchOffsets[], const Index colOffsets[]) const {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSeigen_spatial_convolutionsSLinlDTh mht_14(mht_14_v, 628, "", "./tensorflow/core/kernels/eigen_spatial_convolutions-inl.h", "loadPacketStandardFromTwoColumns");

    eigen_assert(colOffsets[1] == colOffsets[0] + 1);
    const Index packetSize = internal::unpacket_traits<Packet>::size;

    // Packet to load will be split into 2 parts where each part spans a single
    // column. First determine where to split.
    const Index patchIdSplit =
        ((colOffsets[1] * m_colStride) * m_rowInputStride) - 1;
    const Index patchOffsetSplit = patchIdSplit / m_fastDimZero;

    // patchIds[i]:          patchId corresponding to partial packet i
    // spans[i]:             Start and end indices corresponding to the elements
    //                       to be loaded for partial packet i
    // patchOffsets2Cols[i]: patchOffsets corresponding to partial packet i
    const Index patchIds[2] = {patchId, patchIdSplit + 1};
    const Index spans[2][2] = {{0, patchIdSplit - patchId},
                               {patchIdSplit - patchId + 1, packetSize - 1}};
    const Index patchOffsets2Cols[2][2] = {
        {patchOffsets[0], patchOffsetSplit},
        {patchOffsetSplit + 1, patchOffsets[1]}};

    // Load partial packets and do bit-wise OR to generate required packet
    return internal::por<Packet>(
        loadPartialPacketStandard(rowIndex, colIndex, otherIndex, patchIds[0],
                                  spans[0], patchOffsets2Cols[0],
                                  colOffsets[0]),
        loadPartialPacketStandard(rowIndex, colIndex, otherIndex, patchIds[1],
                                  spans[1], patchOffsets2Cols[1],
                                  colOffsets[1]));
  }

  // Helper function to load a packet that is present in a single columns.
  // If required, this function is called from loadPacketStandard().
  EIGEN_DEVICE_FUNC
  EIGEN_ALWAYS_INLINE Packet loadPacketStandardFromSingleColumn(
      Index patchId, Index rowIndex, Index colIndex, Index otherIndex,
      const Index patchOffsets[], const Index colOffsets[],
      const Index inputCols[]) const {
   std::vector<std::string> mht_15_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSeigen_spatial_convolutionsSLinlDTh mht_15(mht_15_v, 668, "", "./tensorflow/core/kernels/eigen_spatial_convolutions-inl.h", "loadPacketStandardFromSingleColumn");

    eigen_assert(colOffsets[0] == colOffsets[1]);
    const Index rowOffsets[2] = {patchOffsets[0] - colOffsets[0] * m_colStride,
                                 patchOffsets[1] - colOffsets[1] * m_colStride};
    eigen_assert(rowOffsets[0] <= rowOffsets[1]);
    const Index inputRows[2] = {rowIndex + rowOffsets[0],
                                rowIndex + rowOffsets[1]};

    if (inputRows[0] >= m_inputRows || inputRows[1] < 0) {
      // all zeros
      return internal::pset1<Packet>(Scalar(0));  // all zeros
    }

    if (inputRows[0] >= 0 && inputRows[1] < m_inputRows) {
      // no padding
      const Index depth = patchId - patchOffsets[0] * patchDepth();
      const Index inputIndex = depth + inputRows[0] * m_rowInputStride +
                               inputCols[0] * m_colInputStride + otherIndex;
      return m_impl.template packet<Unaligned>(inputIndex);
    }
    return packetWithPossibleZero(patchId, rowIndex, colIndex, otherIndex);
  }

  // Load standard packet from a patch specified by the "within patch offset"
  // (patchId) and the precomputed indices of the first element of the patch.
  // This function will be called if partial packet loading is not available
  // for the TensorEvaluator or if the packet type does not support masked
  // load.
  template <typename PacketT, typename TensorEvaluatorT>
  EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE typename std::enable_if<
      !TensorEvaluatorHasPartialPacket<TensorEvaluatorT, PacketT, Index>::value,
      PacketT>::type
  loadPacketStandard(Index patchId, Index rowIndex, Index colIndex,
                     Index otherIndex) const {
   std::vector<std::string> mht_16_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSeigen_spatial_convolutionsSLinlDTh mht_16(mht_16_v, 704, "", "./tensorflow/core/kernels/eigen_spatial_convolutions-inl.h", "loadPacketStandard");

    const Index packetSize = internal::unpacket_traits<Packet>::size;
    EIGEN_STATIC_ASSERT(packetSize > 1, YOU_MADE_A_PROGRAMMING_MISTAKE)
    eigen_assert(patchId < patchDepth() * patchRows() * m_patch_cols);

    eigen_assert(!nonStandardPatches());

    if ((patchDepth() % packetSize) == 0) {
      return loadPacketFast(patchId, rowIndex, colIndex, otherIndex);
    }

    // Offsets and input calculation here are identical to
    // loadCoeffStandard(...), but repeated twice.
    const Index patchOffsets[2] = {patchId / m_fastDimZero,
                                   (patchId + packetSize - 1) / m_fastDimZero};
    const Index colOffsets[2] = {patchOffsets[0] / m_fastColStride,
                                 patchOffsets[1] / m_fastColStride};
    const Index inputCols[2] = {colIndex + colOffsets[0],
                                colIndex + colOffsets[1]};

    if (inputCols[0] >= m_inputCols || inputCols[1] < 0) {
      // all zeros
      return internal::pset1<Packet>(Scalar(0));
    }
    if (inputCols[0] == inputCols[1]) {
      return loadPacketStandardFromSingleColumn(patchId, rowIndex, colIndex,
                                                otherIndex, patchOffsets,
                                                colOffsets, inputCols);
    }
    return packetWithPossibleZero(patchId, rowIndex, colIndex, otherIndex);
  }

  // Load standard packet from a patch specified by the "within patch offset"
  // (patchId) and the precomputed indices of the first element of the patch.
  // This function will be called if partial packet loading is available for
  // the TensorEvaluator and if the packet type supports masked load.
  // The only difference between this and the other case is that if the packet
  // to load is split across two columns, then in this case instead of going to
  // the slow (element-by-element) load, we load two packets - each containing
  // elements from one of the columns (rest of the elements of the packets are
  // zeroes), and then combine these two packets to generate the required
  // packet. The idea is to enable fast load (if possible) of these 'partial'
  // packets.
  template <typename PacketT, typename TensorEvaluatorT>
  EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE typename std::enable_if<
      TensorEvaluatorHasPartialPacket<TensorEvaluatorT, PacketT, Index>::value,
      PacketT>::type
  loadPacketStandard(Index patchId, Index rowIndex, Index colIndex,
                     Index otherIndex) const {
   std::vector<std::string> mht_17_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSeigen_spatial_convolutionsSLinlDTh mht_17(mht_17_v, 755, "", "./tensorflow/core/kernels/eigen_spatial_convolutions-inl.h", "loadPacketStandard");

    const Index packetSize = internal::unpacket_traits<PacketT>::size;
    EIGEN_STATIC_ASSERT(packetSize > 1, YOU_MADE_A_PROGRAMMING_MISTAKE)
    eigen_assert(patchId < patchDepth() * patchRows() * m_patch_cols);

    eigen_assert(!nonStandardPatches());

    if ((patchDepth() % packetSize) == 0) {
      return loadPacketFast(patchId, rowIndex, colIndex, otherIndex);
    }

    // Offsets and input calculation here are identical to
    // loadCoeffStandard(...), but repeated twice.
    const Index patchOffsets[2] = {patchId / m_fastDimZero,
                                   (patchId + packetSize - 1) / m_fastDimZero};
    const Index colOffsets[2] = {patchOffsets[0] / m_fastColStride,
                                 patchOffsets[1] / m_fastColStride};
    const Index inputCols[2] = {colIndex + colOffsets[0],
                                colIndex + colOffsets[1]};

    if (inputCols[0] >= m_inputCols || inputCols[1] < 0) {
      // all zeros
      return internal::pset1<PacketT>(Scalar(0));
    }
    if (inputCols[0] == inputCols[1]) {
      return loadPacketStandardFromSingleColumn(patchId, rowIndex, colIndex,
                                                otherIndex, patchOffsets,
                                                colOffsets, inputCols);
    }
    if (inputCols[1] == inputCols[0] + 1) {
      return loadPacketStandardFromTwoColumns(
          patchId, rowIndex, colIndex, otherIndex, patchOffsets, colOffsets);
    }
    return packetWithPossibleZero(patchId, rowIndex, colIndex, otherIndex);
  }

  EIGEN_DEVICE_FUNC
  EIGEN_ALWAYS_INLINE Packet loadPacketFast(Index patchId, Index rowIndex,
                                            Index colIndex,
                                            Index otherIndex) const {
   std::vector<std::string> mht_18_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSeigen_spatial_convolutionsSLinlDTh mht_18(mht_18_v, 797, "", "./tensorflow/core/kernels/eigen_spatial_convolutions-inl.h", "loadPacketFast");

    const Index packetSize = internal::unpacket_traits<Packet>::size;
    EIGEN_STATIC_ASSERT(packetSize > 1, YOU_MADE_A_PROGRAMMING_MISTAKE)
    eigen_assert(patchId < patchDepth() * patchRows() * m_patch_cols);

    eigen_assert(!nonStandardPatches());
    eigen_assert((patchDepth() % packetSize) == 0);
    // Find the offset of the element wrt the location of the first element.
    const Index patchOffset = patchId / m_fastDimZero;
    eigen_assert((patchId + packetSize - 1) / m_fastDimZero == patchOffset);

    const Index colOffset = patchOffset / m_fastColStride;
    const Index rowOffset = patchOffset - colOffset * m_colStride;
    const Index inputCol = colIndex + colOffset;
    const Index inputRow = rowIndex + rowOffset;
    if (inputCol < 0 || inputRow < 0 || inputCol >= m_inputCols ||
        inputRow >= m_inputRows) {
      // all zeros
      return internal::pset1<Packet>(Scalar(0));
    }
    // no padding
    const Index depth = patchId - patchOffset * patchDepth();
    const Index inputIndex = depth + inputRow * m_rowInputStride +
                             inputCol * m_colInputStride + otherIndex;
    return m_impl.template packet<Unaligned>(inputIndex);
  }

  EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE Packet packetWithPossibleZero(
      Index patchId, Index rowIndex, Index colIndex, Index otherIndex) const {
   std::vector<std::string> mht_19_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSeigen_spatial_convolutionsSLinlDTh mht_19(mht_19_v, 828, "", "./tensorflow/core/kernels/eigen_spatial_convolutions-inl.h", "packetWithPossibleZero");

    const int packetSize = internal::unpacket_traits<Packet>::size;
    EIGEN_ALIGN_MAX
    typename internal::remove_const<Scalar>::type values[packetSize];
    for (int i = 0; i < packetSize; ++i) {
      values[i] = loadCoeff(patchId + i, rowIndex, colIndex, otherIndex);
    }
    Packet rslt = internal::pload<Packet>(values);
    return rslt;
  }

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE void computeBaseIndices(
      Index patchIndex, Index& rowIndex, Index& colIndex,
      Index& otherIndex) const {
   std::vector<std::string> mht_20_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSeigen_spatial_convolutionsSLinlDTh mht_20(mht_20_v, 844, "", "./tensorflow/core/kernels/eigen_spatial_convolutions-inl.h", "computeBaseIndices");

    const size_t NumInputDims = array_size<
        typename TensorEvaluator<ArgType, Device>::Dimensions>::value;
    otherIndex = (NumInputDims == 3) ? 0 : patchIndex / m_fastNumPatches;
    const Index patch2DIndex = (NumInputDims == 3)
                                   ? patchIndex
                                   : (patchIndex - otherIndex * m_num_patches);
    otherIndex *= m_patchInputStride;
    colIndex = patch2DIndex / m_fastOutputRows;
    rowIndex = patch2DIndex - colIndex * m_outputRows;
    colIndex = colIndex * m_col_strides - m_colPaddingLeft;
    rowIndex = rowIndex * m_row_strides - m_rowPaddingTop;
  }

  Index m_patch_cols;   // number of columns in the patch
  Index m_num_patches;  // number of patches to extract.

  // Strides for navigating through the single patch.
  Index m_patch_row_stride;
  Index m_patch_col_stride;
  internal::TensorIntDivisor<Index> m_fastPatchRowStride;
  internal::TensorIntDivisor<Index> m_fastPatchColStride;

  Index m_patch_row_inflate_strides;  // the strides for row inflation in the
                                      // image patch
  Index m_patch_col_inflate_strides;  // the strides for col inflation in the
                                      // image patch
  // Fast representation of inflation strides.
  internal::TensorIntDivisor<Index> m_fastInputRowStride;
  internal::TensorIntDivisor<Index> m_fastInputColStride;

  Index m_otherStride;
  Index m_colStride;
  internal::TensorIntDivisor<Index> m_fastNumPatches;
  internal::TensorIntDivisor<Index> m_fastColStride;

  Index m_rowInputStride;    // row stride in the input tensor
  Index m_colInputStride;    // col stride in the input tensor
  Index m_patchInputStride;  // patch stride in the input tensor

  Index m_inputRows;  // Number of rows in the input tensor
  Index m_inputCols;  // Number of cols in the input tensor

  Index m_outputRows;  // Number of convolution output rows
  Index m_outputCols;  // Number of convolution output column

  Index m_row_strides;  // User specified row stride
  Index m_col_strides;  // User specified col stride

  Index m_in_row_strides;  // User specified input row stride
  Index m_in_col_strides;  // User specified input col stride

  Index m_rowPaddingTop;   // Row padding
  Index m_colPaddingLeft;  // Column padding

  internal::TensorIntDivisor<Index> m_fastOutputRows;
  internal::TensorIntDivisor<Index> m_fastDimZero;

  const TensorEvaluator<ArgType, Device> m_impl;
};

template <typename NewDimension, Index Rows, Index Cols, typename ArgType,
          typename Device, typename Scalar, typename Index,
          typename nocontract_t, typename contract_t, int Side, int packet_size,
          bool inner_dim_contiguous, bool inner_dim_reordered, int Alignment>
class TensorContractionSubMapper<
    Scalar, Index, Side,
    TensorEvaluator<
        const TensorReshapingOp<NewDimension,
                                const TensorImagePatchOp<Rows, Cols, ArgType> >,
        Device>,
    nocontract_t, contract_t, packet_size, inner_dim_contiguous,
    inner_dim_reordered, Alignment> {
 public:
  typedef typename packet_traits<Scalar>::type Packet;
  typedef typename packet_traits<Scalar>::half HalfPacket;

  typedef TensorContractionInputMapper<
      Scalar, Index, Side,
      TensorEvaluator<
          const TensorReshapingOp<
              NewDimension, const TensorImagePatchOp<Rows, Cols, ArgType> >,
          Device>,
      nocontract_t, contract_t, packet_size, inner_dim_contiguous,
      inner_dim_reordered, Alignment>
      ParentMapper;

  typedef TensorContractionSubMapper<
      Scalar, Index, Side,
      TensorEvaluator<
          const TensorReshapingOp<
              NewDimension, const TensorImagePatchOp<Rows, Cols, ArgType> >,
          Device>,
      nocontract_t, contract_t, packet_size, inner_dim_contiguous,
      inner_dim_reordered, Alignment>
      Self;

  typedef Self LinearMapper;

  typedef typename ParentMapper::TensorEvaluatorT TensorEvaluatorT;

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE TensorContractionSubMapper(
      const ParentMapper& base_mapper, Index vert_offset, Index horiz_offset)
      : m_depth_offset(vert_offset),
        m_col_offset(horiz_offset),
        m_base_mapper(base_mapper) {
   std::vector<std::string> mht_21_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSeigen_spatial_convolutionsSLinlDTh mht_21(mht_21_v, 952, "", "./tensorflow/core/kernels/eigen_spatial_convolutions-inl.h", "TensorContractionSubMapper");

    m_base_mapper.computeBaseIndices(m_col_offset, m_rowIndex, m_colIndex,
                                     m_otherIndex);
  }
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE TensorContractionSubMapper(
      const Self& base_mapper, Index vert_offset, Index horiz_offset)
      : m_depth_offset(vert_offset + base_mapper.m_depth_offset),
        m_col_offset(horiz_offset + base_mapper.m_col_offset),
        m_base_mapper(base_mapper.m_base_mapper) {
   std::vector<std::string> mht_22_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSeigen_spatial_convolutionsSLinlDTh mht_22(mht_22_v, 963, "", "./tensorflow/core/kernels/eigen_spatial_convolutions-inl.h", "TensorContractionSubMapper");

    m_base_mapper.computeBaseIndices(m_col_offset, m_rowIndex, m_colIndex,
                                     m_otherIndex);
  }
  EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE Scalar operator()(Index i) const {
    return m_base_mapper.loadCoeff(i + m_depth_offset, m_rowIndex, m_colIndex,
                                   m_otherIndex);
  }
  EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE Scalar operator()(Index i,
                                                          Index j) const {
    return m_base_mapper(i + m_depth_offset, j + m_col_offset);
  }

  EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE Packet loadPacket(Index i) const {
   std::vector<std::string> mht_23_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSeigen_spatial_convolutionsSLinlDTh mht_23(mht_23_v, 979, "", "./tensorflow/core/kernels/eigen_spatial_convolutions-inl.h", "loadPacket");

    return m_base_mapper.loadPacket(i + m_depth_offset, m_rowIndex, m_colIndex,
                                    m_otherIndex);
  }
  EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE Packet loadPacket(Index i,
                                                          Index j) const {
   std::vector<std::string> mht_24_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSeigen_spatial_convolutionsSLinlDTh mht_24(mht_24_v, 987, "", "./tensorflow/core/kernels/eigen_spatial_convolutions-inl.h", "loadPacket");

    return m_base_mapper.template loadPacket<Alignment>(i + m_depth_offset,
                                                        j + m_col_offset);
  }
  EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE Scalar
  loadCoeffStandard(Index i) const {
   std::vector<std::string> mht_25_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSeigen_spatial_convolutionsSLinlDTh mht_25(mht_25_v, 995, "", "./tensorflow/core/kernels/eigen_spatial_convolutions-inl.h", "loadCoeffStandard");

    return m_base_mapper.loadCoeffStandard(i + m_depth_offset, m_rowIndex,
                                           m_colIndex, m_otherIndex);
  }

  EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE Packet loadPacketFast(Index i) const {
   std::vector<std::string> mht_26_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSeigen_spatial_convolutionsSLinlDTh mht_26(mht_26_v, 1003, "", "./tensorflow/core/kernels/eigen_spatial_convolutions-inl.h", "loadPacketFast");

    return m_base_mapper.loadPacketFast(i + m_depth_offset, m_rowIndex,
                                        m_colIndex, m_otherIndex);
  }
  EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE Packet
  loadPacketStandard(Index i) const {
   std::vector<std::string> mht_27_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSeigen_spatial_convolutionsSLinlDTh mht_27(mht_27_v, 1011, "", "./tensorflow/core/kernels/eigen_spatial_convolutions-inl.h", "loadPacketStandard");

    typedef decltype(m_base_mapper.m_impl) TensorEvaluatorT;
    return m_base_mapper.template loadPacketStandard<Packet, TensorEvaluatorT>(
        i + m_depth_offset, m_rowIndex, m_colIndex, m_otherIndex);
  }
  template <typename Packet>
  EIGEN_DEVICE_FUNC bool aligned(Index) const {
   std::vector<std::string> mht_28_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSeigen_spatial_convolutionsSLinlDTh mht_28(mht_28_v, 1020, "", "./tensorflow/core/kernels/eigen_spatial_convolutions-inl.h", "aligned");

    return false;
  }

  EIGEN_DEVICE_FUNC
  EIGEN_ALWAYS_INLINE bool nonStandardPatches() const {
   std::vector<std::string> mht_29_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSeigen_spatial_convolutionsSLinlDTh mht_29(mht_29_v, 1028, "", "./tensorflow/core/kernels/eigen_spatial_convolutions-inl.h", "nonStandardPatches");

    return m_base_mapper.nonStandardPatches();
  }

  // Max(Col|Row|Depth): compute the upper limit for the column, row and depth
  // index respectively that fits into the peeled_k elements starting at
  // m_depth_offset.

  EIGEN_DEVICE_FUNC
  EIGEN_ALWAYS_INLINE Index maxCol(const Index peeled_k) const {
   std::vector<std::string> mht_30_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSeigen_spatial_convolutionsSLinlDTh mht_30(mht_30_v, 1040, "", "./tensorflow/core/kernels/eigen_spatial_convolutions-inl.h", "maxCol");

    const Index max_col =
        (m_depth_offset + (peeled_k == 0 ? 0 : peeled_k - 1)) /
        fastPatchColStride();
    return std::min<Index>(1 + max_col, patchCols());
  }

  EIGEN_DEVICE_FUNC
  EIGEN_ALWAYS_INLINE Index maxRow(const Index peeled_k,
                                   const Index col) const {
   std::vector<std::string> mht_31_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSeigen_spatial_convolutionsSLinlDTh mht_31(mht_31_v, 1052, "", "./tensorflow/core/kernels/eigen_spatial_convolutions-inl.h", "maxRow");

    const Index max_row = (m_depth_offset + (peeled_k == 0 ? 0 : peeled_k - 1) -
                           col * patchColStride()) /
                          fastPatchRowStride();
    return std::min<Index>(1 + max_row, patchRows());
  }

  EIGEN_DEVICE_FUNC
  EIGEN_ALWAYS_INLINE Index maxDepth(const Index peeled_k, const Index col,
                                     Index row) const {
   std::vector<std::string> mht_32_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSeigen_spatial_convolutionsSLinlDTh mht_32(mht_32_v, 1064, "", "./tensorflow/core/kernels/eigen_spatial_convolutions-inl.h", "maxDepth");

    const Index max_depth = m_depth_offset + peeled_k -  //
                            col * patchColStride() -     //
                            row * patchRowStride();
    return std::min<Index>(max_depth, patchDepth());
  }

  // MaxDepth uses only the remaining number of elements in the peeled_k.
  EIGEN_DEVICE_FUNC
  EIGEN_ALWAYS_INLINE Index maxDepth(const Index num_elements,
                                     const Index start_depth) const {
   std::vector<std::string> mht_33_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSeigen_spatial_convolutionsSLinlDTh mht_33(mht_33_v, 1077, "", "./tensorflow/core/kernels/eigen_spatial_convolutions-inl.h", "maxDepth");

    return std::min<Index>(start_depth + num_elements, patchDepth());
  }

  // Every register matters in this code, so sometimes to prevent register
  // spilling, instead of the variable that you would expect to see, we use
  // another one, that is guaranteed to have the same value. E.g. patch depth is
  // always the same as input depth, and it's also the same as input row stride.
  // Bunch of other parameters have similar relations.

  typedef internal::TensorIntDivisor<Index> IndexDivisor;

  EIGEN_DEVICE_FUNC
  EIGEN_ALWAYS_INLINE Index patchDepth() const {
   std::vector<std::string> mht_34_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSeigen_spatial_convolutionsSLinlDTh mht_34(mht_34_v, 1093, "", "./tensorflow/core/kernels/eigen_spatial_convolutions-inl.h", "patchDepth");

    return m_base_mapper.m_rowInputStride;
  }
  EIGEN_DEVICE_FUNC
  EIGEN_ALWAYS_INLINE Index patchRows() const {
   std::vector<std::string> mht_35_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSeigen_spatial_convolutionsSLinlDTh mht_35(mht_35_v, 1100, "", "./tensorflow/core/kernels/eigen_spatial_convolutions-inl.h", "patchRows");

    return m_base_mapper.m_colStride;
  }
  EIGEN_DEVICE_FUNC
  EIGEN_ALWAYS_INLINE Index patchCols() const {
   std::vector<std::string> mht_36_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSeigen_spatial_convolutionsSLinlDTh mht_36(mht_36_v, 1107, "", "./tensorflow/core/kernels/eigen_spatial_convolutions-inl.h", "patchCols");

    return m_base_mapper.m_patch_cols;
  }

  EIGEN_DEVICE_FUNC
  EIGEN_ALWAYS_INLINE Index patchRowStride() const {
   std::vector<std::string> mht_37_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSeigen_spatial_convolutionsSLinlDTh mht_37(mht_37_v, 1115, "", "./tensorflow/core/kernels/eigen_spatial_convolutions-inl.h", "patchRowStride");

    eigen_assert(patchDepth() == m_base_mapper.m_patch_row_stride &&
                 "Patch depth must be equal to patch row stride.");
    return patchDepth();
  }
  EIGEN_DEVICE_FUNC
  EIGEN_ALWAYS_INLINE Index patchColStride() const {
   std::vector<std::string> mht_38_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSeigen_spatial_convolutionsSLinlDTh mht_38(mht_38_v, 1124, "", "./tensorflow/core/kernels/eigen_spatial_convolutions-inl.h", "patchColStride");

    return m_base_mapper.m_patch_col_stride;
  }

  EIGEN_DEVICE_FUNC
  EIGEN_ALWAYS_INLINE IndexDivisor fastPatchRowStride() const {
   std::vector<std::string> mht_39_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSeigen_spatial_convolutionsSLinlDTh mht_39(mht_39_v, 1132, "", "./tensorflow/core/kernels/eigen_spatial_convolutions-inl.h", "fastPatchRowStride");

    eigen_assert(patchDepth() == m_base_mapper.m_patch_row_stride &&
                 "Patch depth must be equal to patch row stride.");
    return m_base_mapper.m_fastDimZero;  // patch_depth
  }
  EIGEN_DEVICE_FUNC
  EIGEN_ALWAYS_INLINE IndexDivisor fastPatchColStride() const {
   std::vector<std::string> mht_40_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSeigen_spatial_convolutionsSLinlDTh mht_40(mht_40_v, 1141, "", "./tensorflow/core/kernels/eigen_spatial_convolutions-inl.h", "fastPatchColStride");

    return m_base_mapper.m_fastPatchColStride;
  }

  EIGEN_DEVICE_FUNC
  EIGEN_ALWAYS_INLINE Packet packetNoPadding(const Index depth,
                                             const Index baseIndex) const {
   std::vector<std::string> mht_41_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSeigen_spatial_convolutionsSLinlDTh mht_41(mht_41_v, 1150, "", "./tensorflow/core/kernels/eigen_spatial_convolutions-inl.h", "packetNoPadding");

    const Index inputIndex = depth + baseIndex;
    return m_base_mapper.m_impl.template packet<Unaligned>(inputIndex);
  }
  EIGEN_DEVICE_FUNC
  EIGEN_ALWAYS_INLINE Scalar coeffNoPadding(const Index depth,
                                            const Index baseIndex) const {
   std::vector<std::string> mht_42_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSeigen_spatial_convolutionsSLinlDTh mht_42(mht_42_v, 1159, "", "./tensorflow/core/kernels/eigen_spatial_convolutions-inl.h", "coeffNoPadding");

    const Index inputIndex = depth + baseIndex;
    return m_base_mapper.m_impl.coeff(inputIndex);
  }
  template <typename PacketT = Packet>
  EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE typename std::enable_if<
      TensorEvaluatorHasPartialPacket<TensorEvaluatorT, PacketT, Index>::value,
      PacketT>::type
  partialPacketNoPadding(const Index depth, const Index baseIndex,
                         Index num_coeffs) const {
   std::vector<std::string> mht_43_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSeigen_spatial_convolutionsSLinlDTh mht_43(mht_43_v, 1171, "", "./tensorflow/core/kernels/eigen_spatial_convolutions-inl.h", "partialPacketNoPadding");

    const Index inputIndex = depth + baseIndex;
    return m_base_mapper.m_impl.template partialPacket<PacketT>(
        inputIndex, mask<PacketT>(0, num_coeffs));
  }
  EIGEN_DEVICE_FUNC
  EIGEN_ALWAYS_INLINE bool hasPadding() const {
   std::vector<std::string> mht_44_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSeigen_spatial_convolutionsSLinlDTh mht_44(mht_44_v, 1180, "", "./tensorflow/core/kernels/eigen_spatial_convolutions-inl.h", "hasPadding");

    // TODO(ezhulenev): It does seems that for inflated filter it's still
    // possible to guarantee "no padding or skipping" for non-standard packing.
    if (nonStandardPatches()) return true;

    // Non zero padding before.
    if (m_base_mapper.m_rowPaddingTop > 0) return true;
    if (m_base_mapper.m_colPaddingLeft > 0) return true;

    // Non zero padding after in rows.
    const Index last_row =
        (m_base_mapper.m_outputRows - 1) * m_base_mapper.m_row_strides;
    if (last_row + (patchRows() - 1) >= m_base_mapper.m_inputRows) return true;

    // Non zero padding after in cols.
    const Index last_col =
        (m_base_mapper.m_outputCols - 1) * m_base_mapper.m_col_strides;
    if (last_col + (patchCols() - 1) >= m_base_mapper.m_inputCols) return true;

    return false;
  }
  EIGEN_DEVICE_FUNC
  EIGEN_ALWAYS_INLINE bool padRow(const Index row) const {
   std::vector<std::string> mht_45_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSeigen_spatial_convolutionsSLinlDTh mht_45(mht_45_v, 1205, "", "./tensorflow/core/kernels/eigen_spatial_convolutions-inl.h", "padRow");

    const Index r = m_rowIndex + row;
    return r < 0 || r >= m_base_mapper.m_inputRows;
  }
  EIGEN_DEVICE_FUNC
  EIGEN_ALWAYS_INLINE bool padAnyRow(const Index first_row,
                                     const Index last_row) const {
   std::vector<std::string> mht_46_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSeigen_spatial_convolutionsSLinlDTh mht_46(mht_46_v, 1214, "", "./tensorflow/core/kernels/eigen_spatial_convolutions-inl.h", "padAnyRow");

    return m_rowIndex + first_row < 0 ||
           m_rowIndex + last_row >= m_base_mapper.m_inputRows;
  }
  EIGEN_DEVICE_FUNC
  EIGEN_ALWAYS_INLINE bool padOrSkipRow(const Index row,
                                        Index* orig_row) const {
   std::vector<std::string> mht_47_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSeigen_spatial_convolutionsSLinlDTh mht_47(mht_47_v, 1223, "", "./tensorflow/core/kernels/eigen_spatial_convolutions-inl.h", "padOrSkipRow");

    eigen_assert(nonStandardPatches());

    const Index input_row = m_rowIndex + row * m_base_mapper.m_in_row_strides;
    *orig_row = (m_base_mapper.m_patch_row_inflate_strides == 1)
                    ? input_row
                    : ((input_row >= 0)
                           ? (input_row / m_base_mapper.m_fastInputRowStride)
                           : 0);

    return (*orig_row < 0 || *orig_row >= m_base_mapper.m_inputRows) ||
           (input_row != *orig_row * m_base_mapper.m_patch_row_inflate_strides);
  }
  EIGEN_DEVICE_FUNC
  EIGEN_ALWAYS_INLINE bool padCol(const Index col) const {
   std::vector<std::string> mht_48_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSeigen_spatial_convolutionsSLinlDTh mht_48(mht_48_v, 1240, "", "./tensorflow/core/kernels/eigen_spatial_convolutions-inl.h", "padCol");

    const Index c = m_colIndex + col;
    return c < 0 || c >= m_base_mapper.m_inputCols;
  }
  EIGEN_DEVICE_FUNC
  EIGEN_ALWAYS_INLINE bool padOrSkipCol(const Index col,
                                        Index* orig_col) const {
   std::vector<std::string> mht_49_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSeigen_spatial_convolutionsSLinlDTh mht_49(mht_49_v, 1249, "", "./tensorflow/core/kernels/eigen_spatial_convolutions-inl.h", "padOrSkipCol");

    eigen_assert(nonStandardPatches());

    const Index input_col = m_colIndex + col * m_base_mapper.m_in_col_strides;
    *orig_col = (m_base_mapper.m_patch_col_inflate_strides == 1)
                    ? input_col
                    : ((input_col >= 0)
                           ? (input_col / m_base_mapper.m_fastInputColStride)
                           : 0);

    return (*orig_col < 0 || *orig_col >= m_base_mapper.m_inputCols) ||
           (input_col != *orig_col * m_base_mapper.m_patch_col_inflate_strides);
  }
  EIGEN_DEVICE_FUNC
  EIGEN_ALWAYS_INLINE Index baseIndex(const Index row, const Index col) const {
   std::vector<std::string> mht_50_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSeigen_spatial_convolutionsSLinlDTh mht_50(mht_50_v, 1266, "", "./tensorflow/core/kernels/eigen_spatial_convolutions-inl.h", "baseIndex");

    const Index r = m_rowIndex + row;
    const Index c = m_colIndex + col;
    return r * m_base_mapper.m_rowInputStride +
           c * m_base_mapper.m_colInputStride + m_otherIndex;
  }
  // Compute a base index when original input row and column were precomputed
  // using padOrSkipRow and padOrSkipCol. Used only for non standard patches.
  EIGEN_DEVICE_FUNC
  EIGEN_ALWAYS_INLINE Index origBaseIndex(const Index orig_row,
                                          const Index orig_col) const {
   std::vector<std::string> mht_51_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSeigen_spatial_convolutionsSLinlDTh mht_51(mht_51_v, 1279, "", "./tensorflow/core/kernels/eigen_spatial_convolutions-inl.h", "origBaseIndex");

    return orig_row * m_base_mapper.m_rowInputStride +
           orig_col * m_base_mapper.m_colInputStride + m_otherIndex;
  }

  EIGEN_DEVICE_FUNC
  EIGEN_ALWAYS_INLINE Index rowStride() const {
   std::vector<std::string> mht_52_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSeigen_spatial_convolutionsSLinlDTh mht_52(mht_52_v, 1288, "", "./tensorflow/core/kernels/eigen_spatial_convolutions-inl.h", "rowStride");

    return m_base_mapper.m_row_strides;
  }
  EIGEN_DEVICE_FUNC
  EIGEN_ALWAYS_INLINE Index colStride() const {
   std::vector<std::string> mht_53_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSeigen_spatial_convolutionsSLinlDTh mht_53(mht_53_v, 1295, "", "./tensorflow/core/kernels/eigen_spatial_convolutions-inl.h", "colStride");

    return m_base_mapper.m_col_strides;
  }

  EIGEN_DEVICE_FUNC
  EIGEN_ALWAYS_INLINE Index rowOffset() const {
   std::vector<std::string> mht_54_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSeigen_spatial_convolutionsSLinlDTh mht_54(mht_54_v, 1303, "", "./tensorflow/core/kernels/eigen_spatial_convolutions-inl.h", "rowOffset");

    const Index patchOffset = m_depth_offset / m_base_mapper.m_fastDimZero;
    const Index colOffset = patchOffset / m_base_mapper.m_fastColStride;
    return patchOffset - colOffset * m_base_mapper.m_colStride;
  }

  EIGEN_DEVICE_FUNC
  EIGEN_ALWAYS_INLINE Index colOffset() const {
   std::vector<std::string> mht_55_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSeigen_spatial_convolutionsSLinlDTh mht_55(mht_55_v, 1313, "", "./tensorflow/core/kernels/eigen_spatial_convolutions-inl.h", "colOffset");

    const Index patchOffset = m_depth_offset / m_base_mapper.m_fastDimZero;
    const Index colOffset = patchOffset / m_base_mapper.m_fastColStride;
    return colOffset;
  }

  EIGEN_DEVICE_FUNC
  EIGEN_ALWAYS_INLINE Index depthOffset() const {
   std::vector<std::string> mht_56_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSeigen_spatial_convolutionsSLinlDTh mht_56(mht_56_v, 1323, "", "./tensorflow/core/kernels/eigen_spatial_convolutions-inl.h", "depthOffset");

    return m_depth_offset % patchDepth();
  }

  EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE LinearMapper
  getLinearMapper(Index i, Index j) const {
   std::vector<std::string> mht_57_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSeigen_spatial_convolutionsSLinlDTh mht_57(mht_57_v, 1331, "", "./tensorflow/core/kernels/eigen_spatial_convolutions-inl.h", "getLinearMapper");

    return LinearMapper(m_base_mapper, i + m_depth_offset, j + m_col_offset);
  }

 private:
  Index m_depth_offset;  // First row in the input matrix
  Index m_col_offset;    // First col in the input matrix

  // Knowing that: col_offset == patchIndex * OTHERS, we keep precomputed base
  // indices for the first element in a patch specified by col_offset
  // (see computeBaseIndices(...) for details).
  Index m_rowIndex;
  Index m_colIndex;
  Index m_otherIndex;

  const ParentMapper m_base_mapper;  // Keeping a copy instead of a reference
                                     // performs better in benchmarks.
};

// Arrange a block of the right input matrix (in our case it's always a "virtual
// matrix" constructed from extracted image patches) in contiguous memory.
//
// Given column major input (A0 beside A1 in memory):
// A0 B0 C0 D0  E0 F0 G0 H0 ... Z0
// A1 B1 C1 D1  E1 F1 G1 H1 ... Z1
// A2 B2 C2 D2  E2 F2 G2 H2 ... Z2
// A3 B3 C3 D3  E3 F3 G3 H3 ... Z3
// A4 B4 C4 D4  E4 F4 G4 H4 ... Z4
// A5 B5 C5 D5  E5 F5 G5 H5 ... Z5
// A6 B6 C6 D6  E6 F6 G6 H6 ... Z6
// A7 B7 C7 D7  E7 F7 G7 H7 ... Z7
// A8 ...
// ...
//
// *) A, B, C, ... - patches extracted from the original input.
// *) A0, A1, A2 ... - values from the same patch at different offsets.
//
// The traversal (packed rhs memory) order (B0 besides A0 in memory):
// A0 B0 C0 D0 A1 B1 C1 D1 ...
// E0 F0 G0 H0 E1 F1 G1 H1 ...
// ...
// Z0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 ... <- doesn't belong to any block (nr = 4)
//
// This traversal order must be the same as in default gemm_pack_rhs defined in
// GeneralBlockPanelKernel.h.
//
// *) nr - number of registers along the 'n' dimension.
//    See GeneralBlockPanelKernel.h and "Anatomy of High-Performance Matrix
//    Multiplication" paper.
template <typename NewDimension, Index Rows, Index Cols, typename ArgType,
          typename Device, typename Scalar, typename Index,
          typename nocontract_t, typename contract_t, int packet_size,
          bool inner_dim_contiguous, bool inner_dim_reordered, int Alignment,
          int nr>
struct gemm_pack_rhs<
    Scalar, Index,
    TensorContractionSubMapper<
        Scalar, Index, Rhs,
        TensorEvaluator<
            const TensorReshapingOp<
                NewDimension, const TensorImagePatchOp<Rows, Cols, ArgType> >,
            Device>,
        nocontract_t, contract_t, packet_size, inner_dim_contiguous,
        inner_dim_reordered, Alignment>,
    nr, ColMajor, false, false> {
  typedef TensorContractionSubMapper<
      Scalar, Index, Rhs,
      TensorEvaluator<
          const TensorReshapingOp<
              NewDimension, const TensorImagePatchOp<Rows, Cols, ArgType> >,
          Device>,
      nocontract_t, contract_t, packet_size, inner_dim_contiguous,
      inner_dim_reordered, Alignment>
      SubMapper;
  typedef SubMapper DataMapper;
  typedef typename packet_traits<Scalar>::type Packet;

  EIGEN_STATIC_ASSERT((nr == 4), YOU_MADE_A_PROGRAMMING_MISTAKE)

  EIGEN_DEVICE_FUNC
  EIGEN_DONT_INLINE void operator()(Scalar* block, const DataMapper& rhs,
                                    Index depth, Index cols, Index stride = 0,
                                    Index offset = 0) const {
    eigen_assert(stride == 0);
    eigen_assert(offset == 0);

    const Index packet_cols4 = (cols / 4) * 4;
    const Index peeled_k = (depth / packet_size) * packet_size;
    const bool non_standard_patches = rhs.nonStandardPatches();

    for (Index j2 = 0; j2 < packet_cols4; j2 += 4) {
      const SubMapper dm0 = rhs.getLinearMapper(0, j2 + 0);
      const SubMapper dm1 = rhs.getLinearMapper(0, j2 + 1);
      const SubMapper dm2 = rhs.getLinearMapper(0, j2 + 2);
      const SubMapper dm3 = rhs.getLinearMapper(0, j2 + 3);

      Index k = 0;
      if ((packet_size % 4) == 0 && !non_standard_patches) {
        // FAST PATH:
        // Iterate over patch columns and rows, if we know that a single
        // packet do not span across multiple rows or columns.
        if ((rhs.patchDepth() % packet_size) == 0) {
          const Index start_col = rhs.colOffset();
          const Index max_col = rhs.maxCol(peeled_k);

          for (Index c = start_col; c < max_col; ++c) {
            eigen_assert(k <= peeled_k);

            const Index start_row = (c == start_col) ? rhs.rowOffset() : 0;
            const Index max_row = rhs.maxRow(peeled_k, c);

            const bool pad_col0 = dm0.padCol(c);
            const bool pad_col1 = dm1.padCol(c);
            const bool pad_col2 = dm2.padCol(c);
            const bool pad_col3 = dm3.padCol(c);

            // Check if we can squeeze reads along the `row` and `depth`
            // dimensions (two innermost dimensions).
            if (!pad_col0 && !pad_col1 && !pad_col2 && !pad_col3 &&    //
                !dm0.padRow(start_row) && !dm0.padRow(max_row - 1) &&  //
                !dm1.padRow(start_row) && !dm1.padRow(max_row - 1) &&  //
                !dm2.padRow(start_row) && !dm2.padRow(max_row - 1) &&  //
                !dm3.padRow(start_row) && !dm3.padRow(max_row - 1)) {
              // Compute how many elements we can squeeze read.
              const Index start_depth =
                  (c == start_col) ? rhs.depthOffset() : 0;

              // Upper bound for the number of elements in the depth dimension
              // that we can squeeze read.
              const Index squeeze_length =
                  (max_row - start_row) * rhs.patchDepth() - start_depth;

              // Do not overshoot beyond the block size.
              const Index max_depth =
                  start_depth + std::min<Index>(peeled_k - k, squeeze_length);
              eigen_assert((max_depth - start_depth) % packet_size == 0);

              const Index idx0 = dm0.baseIndex(start_row, c);
              const Index idx1 = dm1.baseIndex(start_row, c);
              const Index idx2 = dm2.baseIndex(start_row, c);
              const Index idx3 = dm3.baseIndex(start_row, c);

              for (Index d = start_depth; d < max_depth; d += packet_size) {
                eigen_assert(k < peeled_k);
                PacketBlock<Packet, 4> kernel;
                kernel.packet[0] = rhs.packetNoPadding(d, idx0);
                kernel.packet[1] = rhs.packetNoPadding(d, idx1);
                kernel.packet[2] = rhs.packetNoPadding(d, idx2);
                kernel.packet[3] = rhs.packetNoPadding(d, idx3);
                ptranspose(kernel);
                pstoreu(block + 0 * packet_size, kernel.packet[0]);
                pstoreu(block + 1 * packet_size, kernel.packet[1]);
                pstoreu(block + 2 * packet_size, kernel.packet[2]);
                pstoreu(block + 3 * packet_size, kernel.packet[3]);
                block += 4 * packet_size;
                k += packet_size;
              }

              // Go to the next column.
              continue;
            }

            // If we can't squeeze reads, process rows one by one.
            for (Index r = start_row; r < max_row; ++r) {
              eigen_assert(k <= peeled_k);

              const bool pad0 = pad_col0 || dm0.padRow(r);
              const bool pad1 = pad_col1 || dm1.padRow(r);
              const bool pad2 = pad_col2 || dm2.padRow(r);
              const bool pad3 = pad_col3 || dm3.padRow(r);

              const Index idx0 = dm0.baseIndex(r, c);
              const Index idx1 = dm1.baseIndex(r, c);
              const Index idx2 = dm2.baseIndex(r, c);
              const Index idx3 = dm3.baseIndex(r, c);

              const Index start_depth = ((c == start_col) && (r == start_row))
                                            ? rhs.depthOffset()
                                            : 0;
              const Index max_depth = rhs.maxDepth(peeled_k - k, start_depth);
              eigen_assert((max_depth - start_depth) % packet_size == 0);

              for (Index d = start_depth; d < max_depth; d += packet_size) {
                eigen_assert(k < peeled_k);
                PacketBlock<Packet, 4> kernel;
                kernel.packet[0] = pad0 ? pset1<Packet>(Scalar(0))
                                        : rhs.packetNoPadding(d, idx0);
                kernel.packet[1] = pad1 ? pset1<Packet>(Scalar(0))
                                        : rhs.packetNoPadding(d, idx1);
                kernel.packet[2] = pad2 ? pset1<Packet>(Scalar(0))
                                        : rhs.packetNoPadding(d, idx2);
                kernel.packet[3] = pad3 ? pset1<Packet>(Scalar(0))
                                        : rhs.packetNoPadding(d, idx3);
                ptranspose(kernel);
                pstoreu(block + 0 * packet_size, kernel.packet[0]);
                pstoreu(block + 1 * packet_size, kernel.packet[1]);
                pstoreu(block + 2 * packet_size, kernel.packet[2]);
                pstoreu(block + 3 * packet_size, kernel.packet[3]);
                block += 4 * packet_size;
                k += packet_size;
              }
            }
          }

          // The loop above should fill peeled_k elements.
          eigen_assert(peeled_k == k);

        } else {
          for (; k < peeled_k; k += packet_size) {
            PacketBlock<Packet, 4> kernel;
            kernel.packet[0] = dm0.loadPacketStandard(k);
            kernel.packet[1] = dm1.loadPacketStandard(k);
            kernel.packet[2] = dm2.loadPacketStandard(k);
            kernel.packet[3] = dm3.loadPacketStandard(k);
            ptranspose(kernel);
            pstoreu(block + 0 * packet_size, kernel.packet[0]);
            pstoreu(block + 1 * packet_size, kernel.packet[1]);
            pstoreu(block + 2 * packet_size, kernel.packet[2]);
            pstoreu(block + 3 * packet_size, kernel.packet[3]);
            block += 4 * packet_size;
          }
        }
      }

      // Copy the remaining coefficients of the column block after the peeled_k.
      if (!rhs.nonStandardPatches()) {
        for (; k < depth; k++) {
          block[0] = dm0.loadCoeffStandard(k);
          block[1] = dm1.loadCoeffStandard(k);
          block[2] = dm2.loadCoeffStandard(k);
          block[3] = dm3.loadCoeffStandard(k);
          block += 4;
        }
      } else {
        for (; k < depth; k++) {
          block[0] = dm0(k);
          block[1] = dm1(k);
          block[2] = dm2(k);
          block[3] = dm3(k);
          block += 4;
        }
      }
    }

    // copy the remaining columns one at a time (nr==1)
    for (Index j2 = packet_cols4; j2 < cols; ++j2) {
      const SubMapper dm0 = rhs.getLinearMapper(0, j2);
      for (Index k = 0; k < depth; k++) {
        *block = dm0(k);
        block += 1;
      }
    }
  }
};

// Template specialization for packet_size = 2. We must special-case packet
// blocks with nr > packet_size, e.g. PacketBlock<Packet2d, 4>.
template <typename NewDimension, Index Rows, Index Cols, typename ArgType,
          typename Device, typename Scalar, typename Index,
          typename nocontract_t, typename contract_t, bool inner_dim_contiguous,
          bool inner_dim_reordered, int Alignment, int nr>
struct gemm_pack_rhs<
    Scalar, Index,
    TensorContractionSubMapper<
        Scalar, Index, Rhs,
        TensorEvaluator<
            const TensorReshapingOp<
                NewDimension, const TensorImagePatchOp<Rows, Cols, ArgType> >,
            Device>,
        nocontract_t, contract_t, 2, inner_dim_contiguous, inner_dim_reordered,
        Alignment>,
    nr, ColMajor, false, false> {
  typedef TensorContractionSubMapper<
      Scalar, Index, Rhs,
      TensorEvaluator<
          const TensorReshapingOp<
              NewDimension, const TensorImagePatchOp<Rows, Cols, ArgType> >,
          Device>,
      nocontract_t, contract_t, 2, inner_dim_contiguous, inner_dim_reordered,
      Alignment>
      SubMapper;
  typedef SubMapper DataMapper;
  typedef typename packet_traits<Scalar>::type Packet;

  EIGEN_STATIC_ASSERT((nr == 4), YOU_MADE_A_PROGRAMMING_MISTAKE)

  EIGEN_DEVICE_FUNC
  EIGEN_DONT_INLINE void operator()(Scalar* block, const DataMapper& rhs,
                                    Index depth, Index cols, Index stride = 0,
                                    Index offset = 0) const {
    eigen_assert(stride == 0);
    eigen_assert(offset == 0);

    const int packet_size = 2;
    const Index packet_cols4 = (cols / 4) * 4;
    const Index peeled_k = (depth / packet_size) * packet_size;
    const bool non_standard_patches = rhs.nonStandardPatches();

    for (Index j2 = 0; j2 < packet_cols4; j2 += 4) {
      const SubMapper dm0 = rhs.getLinearMapper(0, j2 + 0);
      const SubMapper dm1 = rhs.getLinearMapper(0, j2 + 1);
      const SubMapper dm2 = rhs.getLinearMapper(0, j2 + 2);
      const SubMapper dm3 = rhs.getLinearMapper(0, j2 + 3);

      Index k = 0;
      if (!non_standard_patches) {
        // FAST PATH:
        // Iterate over patch columns and rows if we know that a single
        // packet do not span across multiple rows or columns.
        if ((rhs.patchDepth() % packet_size) == 0) {
          const Index start_col = rhs.colOffset();
          const Index max_col = rhs.maxCol(peeled_k);

          for (Index c = start_col; c < max_col; ++c) {
            eigen_assert(k <= peeled_k);

            const Index start_row = (c == start_col) ? rhs.rowOffset() : 0;
            const Index max_row = rhs.maxRow(peeled_k, c);

            const bool pad_col0 = dm0.padCol(c);
            const bool pad_col1 = dm1.padCol(c);
            const bool pad_col2 = dm2.padCol(c);
            const bool pad_col3 = dm3.padCol(c);

            // We can squeeze reads along the `row` and `depth` dimensions if
            // the row stride is `1`, which means that `row` and `depth`
            // dimensions are contiguous (two innermost dimensions).
            if (rhs.rowStride() == 1 &&                                //
                !pad_col0 && !pad_col1 && !pad_col2 && !pad_col3 &&    //
                !dm0.padRow(start_row) && !dm0.padRow(max_row - 1) &&  //
                !dm1.padRow(start_row) && !dm1.padRow(max_row - 1) &&  //
                !dm2.padRow(start_row) && !dm2.padRow(max_row - 1) &&  //
                !dm3.padRow(start_row) && !dm3.padRow(max_row - 1)) {
              // Compute how many elements we can squeeze read.
              const Index start_depth =
                  (c == start_col) ? rhs.depthOffset() : 0;

              // Upper bound for the number of elements in the depth dimension
              // that we can squeeze read.
              const Index squeeze_length =
                  (max_row - start_row) * rhs.patchDepth() - start_depth;

              // Do not overshoot beyond the block size.
              const Index max_depth =
                  start_depth + std::min<Index>(peeled_k - k, squeeze_length);
              eigen_assert((max_depth - start_depth) % packet_size == 0);

              const Index idx0 = dm0.baseIndex(start_row, c);
              const Index idx1 = dm1.baseIndex(start_row, c);
              const Index idx2 = dm2.baseIndex(start_row, c);
              const Index idx3 = dm3.baseIndex(start_row, c);

              for (Index d = start_depth; d < max_depth; d += packet_size) {
                PacketBlock<Packet, 2> kernel0;
                PacketBlock<Packet, 2> kernel1;
                kernel0.packet[0] = rhs.packetNoPadding(d, idx0);
                kernel0.packet[1] = rhs.packetNoPadding(d, idx1);
                kernel1.packet[0] = rhs.packetNoPadding(d, idx2);
                kernel1.packet[1] = rhs.packetNoPadding(d, idx3);
                ptranspose(kernel0);
                ptranspose(kernel1);
                pstoreu(block + 0 * packet_size, kernel0.packet[0]);
                pstoreu(block + 1 * packet_size, kernel1.packet[0]);
                pstoreu(block + 2 * packet_size, kernel0.packet[1]);
                pstoreu(block + 3 * packet_size, kernel1.packet[1]);
                block += 4 * packet_size;
                k += packet_size;
              }

              // Go to the next column.
              continue;
            }

            // If we can't squeeze reads, process rows one by one.
            for (Index r = start_row; r < max_row; ++r) {
              eigen_assert(k <= peeled_k);

              const bool pad0 = pad_col0 || dm0.padRow(r);
              const bool pad1 = pad_col1 || dm1.padRow(r);
              const bool pad2 = pad_col2 || dm2.padRow(r);
              const bool pad3 = pad_col3 || dm3.padRow(r);

              const Index idx0 = dm0.baseIndex(r, c);
              const Index idx1 = dm1.baseIndex(r, c);
              const Index idx2 = dm2.baseIndex(r, c);
              const Index idx3 = dm3.baseIndex(r, c);

              const Index start_depth = ((c == start_col) && (r == start_row))
                                            ? rhs.depthOffset()
                                            : 0;
              const Index max_depth = rhs.maxDepth(peeled_k - k, start_depth);
              eigen_assert((max_depth - start_depth) % packet_size == 0);

              for (Index d = start_depth; d < max_depth; d += packet_size) {
                eigen_assert(k < peeled_k);
                PacketBlock<Packet, 2> kernel0;
                PacketBlock<Packet, 2> kernel1;
                kernel0.packet[0] = pad0 ? pset1<Packet>(Scalar(0))
                                         : rhs.packetNoPadding(d, idx0);
                kernel0.packet[1] = pad1 ? pset1<Packet>(Scalar(0))
                                         : rhs.packetNoPadding(d, idx1);
                kernel1.packet[0] = pad2 ? pset1<Packet>(Scalar(0))
                                         : rhs.packetNoPadding(d, idx2);
                kernel1.packet[1] = pad3 ? pset1<Packet>(Scalar(0))
                                         : rhs.packetNoPadding(d, idx3);
                ptranspose(kernel0);
                ptranspose(kernel1);
                pstoreu(block + 0 * packet_size, kernel0.packet[0]);
                pstoreu(block + 1 * packet_size, kernel1.packet[0]);
                pstoreu(block + 2 * packet_size, kernel0.packet[1]);
                pstoreu(block + 3 * packet_size, kernel1.packet[1]);
                block += 4 * packet_size;
                k += packet_size;
              }
            }
          }

          // The loop above should fill peeled_k elements.
          eigen_assert(peeled_k == k);

        } else {
          // Packet can span multiple rows or columns, so we have to go
          // though the slower "standard" path.
          for (; k < peeled_k; k += packet_size) {
            PacketBlock<Packet, 2> kernel0;
            PacketBlock<Packet, 2> kernel1;
            kernel0.packet[0] = dm0.loadPacketStandard(k);
            kernel0.packet[1] = dm1.loadPacketStandard(k);
            kernel1.packet[0] = dm2.loadPacketStandard(k);
            kernel1.packet[1] = dm3.loadPacketStandard(k);
            ptranspose(kernel0);
            ptranspose(kernel1);
            pstoreu(block + 0 * packet_size, kernel0.packet[0]);
            pstoreu(block + 1 * packet_size, kernel1.packet[0]);
            pstoreu(block + 2 * packet_size, kernel0.packet[1]);
            pstoreu(block + 3 * packet_size, kernel1.packet[1]);
            block += 4 * packet_size;
          }
        }
      }

      // Copy the remaining coefficients of the column block after the peeled_k.
      if (!non_standard_patches) {
        for (; k < depth; k++) {
          block[0] = dm0.loadCoeffStandard(k);
          block[1] = dm1.loadCoeffStandard(k);
          block[2] = dm2.loadCoeffStandard(k);
          block[3] = dm3.loadCoeffStandard(k);
          block += 4;
        }
      } else {
        for (; k < depth; k++) {
          block[0] = dm0(k);
          block[1] = dm1(k);
          block[2] = dm2(k);
          block[3] = dm3(k);
          block += 4;
        }
      }
    }

    // Copy the remaining columns one at a time (nr==1).
    for (Index j2 = packet_cols4; j2 < cols; ++j2) {
      const SubMapper dm0 = rhs.getLinearMapper(0, j2);
      for (Index k = 0; k < depth; k++) {
        *block = dm0(k);
        block += 1;
      }
    }
  }
};

// Special case for non-vectorized types such as float16.
template <typename NewDimension, Index Rows, Index Cols, typename ArgType,
          typename Device, typename Scalar, typename Index,
          typename nocontract_t, typename contract_t, bool inner_dim_contiguous,
          bool inner_dim_reordered, int Alignment, int nr>
struct gemm_pack_rhs<
    Scalar, Index,
    TensorContractionSubMapper<
        Scalar, Index, Rhs,
        TensorEvaluator<
            const TensorReshapingOp<
                NewDimension, const TensorImagePatchOp<Rows, Cols, ArgType> >,
            Device>,
        nocontract_t, contract_t, 1, inner_dim_contiguous, inner_dim_reordered,
        Alignment>,
    nr, ColMajor, false, false> {
  typedef TensorContractionSubMapper<
      Scalar, Index, Rhs,
      TensorEvaluator<
          const TensorReshapingOp<
              NewDimension, const TensorImagePatchOp<Rows, Cols, ArgType> >,
          Device>,
      nocontract_t, contract_t, 1, inner_dim_contiguous, inner_dim_reordered,
      Alignment>
      SubMapper;
  typedef SubMapper DataMapper;

  EIGEN_STATIC_ASSERT((nr == 4), YOU_MADE_A_PROGRAMMING_MISTAKE)

  EIGEN_DEVICE_FUNC
  EIGEN_DONT_INLINE void operator()(Scalar* block, const DataMapper& rhs,
                                    Index depth, Index cols, Index stride = 0,
                                    Index offset = 0) const {
    eigen_assert(stride == 0);
    eigen_assert(offset == 0);

    const Index packet_cols4 = (cols / 4) * 4;

    for (Index j2 = 0; j2 < packet_cols4; j2 += 4) {
      const SubMapper dm0 = rhs.getLinearMapper(0, j2 + 0);
      const SubMapper dm1 = rhs.getLinearMapper(0, j2 + 1);
      const SubMapper dm2 = rhs.getLinearMapper(0, j2 + 2);
      const SubMapper dm3 = rhs.getLinearMapper(0, j2 + 3);

      if (!rhs.nonStandardPatches()) {
        for (Index k = 0; k < depth; k++) {
          block[0] = dm0.loadCoeffStandard(k);
          block[1] = dm1.loadCoeffStandard(k);
          block[2] = dm2.loadCoeffStandard(k);
          block[3] = dm3.loadCoeffStandard(k);
          block += 4;
        }
      } else {
        for (Index k = 0; k < depth; k++) {
          block[0] = dm0(k);
          block[1] = dm1(k);
          block[2] = dm2(k);
          block[3] = dm3(k);
          block += 4;
        }
      }
    }

    // Copy the remaining columns one at a time (nr==1).
    for (Index j2 = packet_cols4; j2 < cols; ++j2) {
      const SubMapper dm0 = rhs.getLinearMapper(0, j2);
      for (Index k = 0; k < depth; k++) {
        *block = dm0(k);
        block += 1;
      }
    }
  }
};
#endif
}  // end namespace internal

/** SpatialConvolution
 * \ingroup CXX11_NeuralNetworks_Module
 *
 * \brief Applies a 2D convolution over a multichannel input image.
 *
 * The input parameter is expected to be a tensor with a rank of 3 or more
 * (channels, height, width, and optionally others)
 * The kernel parameter is expected to be a 4D tensor (filters, channels,
 * kernel_height, kernel_width)
 * The input and the kernel must both be in col-major layout. The result will
 * also be in col-major layout.
 *
 * If col_in_stride, row_in_stride > 1, then applies convolution with holes
 * (aka atrous convolution), sampling every col_in_stride, row_in_stride input
 * pixels.
 *
 * If padding_top, padding_bottom, padding_left, or padding_right is specified,
 * then those paddings will be used to pad the input, and padding_type must be
 * PADDING_VALID.
 *
 * The result can be assigned to a tensor of rank equal to the rank of the
 * input. The dimensions of the result will be filters, height, width (and
 * others if applicable).
 *
 * It is possible to swap the order of the width and height dimensions provided
 * that the same order is used in the input, the kernel, and the output.
 *
 * It is also possible to add an output kernel to the contraction, output
 * kernel is called by Eigen when it "finalizes" the block of an output tensor.
 *
 */
template <typename Input, typename Kernel,
          typename OutputKernel = const NoOpOutputKernel>
EIGEN_DEVICE_FUNC
    EIGEN_ALWAYS_INLINE static const typename internal::conditional<
        internal::traits<Input>::Layout == ColMajor,
        TensorReshapingOp<
            const DSizes<typename internal::traits<Input>::Index,
                         internal::traits<Input>::NumDimensions>,
            const TensorContractionOp<
                const array<IndexPair<typename internal::traits<Input>::Index>,
                            1>,
                const TensorReshapingOp<
                    const DSizes<typename internal::traits<Input>::Index, 2>,
                    const Kernel>,
                const TensorReshapingOp<
                    const DSizes<typename internal::traits<Input>::Index, 2>,
                    const TensorImagePatchOp<Dynamic, Dynamic, const Input> >,
                const OutputKernel> >,
        TensorReshapingOp<
            const DSizes<typename internal::traits<Input>::Index,
                         internal::traits<Input>::NumDimensions>,
            const TensorContractionOp<
                const array<IndexPair<typename internal::traits<Input>::Index>,
                            1>,
                const TensorReshapingOp<
                    const DSizes<typename internal::traits<Input>::Index, 2>,
                    const TensorImagePatchOp<Dynamic, Dynamic, const Input> >,
                const TensorReshapingOp<
                    const DSizes<typename internal::traits<Input>::Index, 2>,
                    const Kernel>,
                const OutputKernel> > >::type
    SpatialConvolution(const Input& input, const Kernel& kernel,
                       const Index row_stride = 1, const Index col_stride = 1,
                       const PaddingType padding_type = PADDING_SAME,
                       const Index row_in_stride = 1,
                       const Index col_in_stride = 1,
                       const OutputKernel& output_kernel = OutputKernel(),
                       Index padding_top = 0, Index padding_bottom = 0,
                       Index padding_left = 0, Index padding_right = 0) {
   std::vector<std::string> mht_58_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSeigen_spatial_convolutionsSLinlDTh mht_58(mht_58_v, 1951, "", "./tensorflow/core/kernels/eigen_spatial_convolutions-inl.h", "SpatialConvolution");

  typedef typename internal::traits<Input>::Index TensorIndex;
  typedef typename internal::traits<Input>::Scalar InputScalar;
  TensorRef<Tensor<InputScalar, internal::traits<Input>::NumDimensions,
                   internal::traits<Input>::Layout, TensorIndex> >
      in(input);
  TensorRef<Tensor<typename internal::traits<Kernel>::Scalar,
                   internal::traits<Kernel>::NumDimensions,
                   internal::traits<Kernel>::Layout, TensorIndex> >
      kern(kernel);

  EIGEN_STATIC_ASSERT(
      internal::traits<Input>::Layout == internal::traits<Kernel>::Layout,
      YOU_MADE_A_PROGRAMMING_MISTAKE)
  const bool isColMajor = (internal::traits<Input>::Layout == ColMajor);

  const int NumDims = internal::traits<Input>::NumDimensions;

  // Number of filters to apply. This is the same as the output depth of the
  // result
  const TensorIndex kernelFilters =
      isColMajor ? kern.dimensions()[0] : kern.dimensions()[3];
  // Number of channels. This is the same as the input depth.
  const TensorIndex kernelChannels =
      isColMajor ? kern.dimensions()[1] : kern.dimensions()[2];
  const TensorIndex kernelRows =
      isColMajor ? kern.dimensions()[2] : kern.dimensions()[1];
  const TensorIndex kernelCols =
      isColMajor ? kern.dimensions()[3] : kern.dimensions()[0];

  const Index kernelRowsEff =
      kernelRows + (kernelRows - 1) * (row_in_stride - 1);
  const Index kernelColsEff =
      kernelCols + (kernelCols - 1) * (col_in_stride - 1);

  array<IndexPair<TensorIndex>, 1> contract_dims;
  contract_dims[0] = IndexPair<TensorIndex>(1, 0);

  const TensorIndex InputRows =
      isColMajor ? in.dimension(1) : in.dimension(NumDims - 2);
  const TensorIndex InputCols =
      isColMajor ? in.dimension(2) : in.dimension(NumDims - 3);
  const bool padding_explicit =
      (padding_top || padding_bottom || padding_left || padding_right);

  TensorIndex out_height;
  TensorIndex out_width;
  switch (padding_type) {
    case PADDING_VALID: {
      const TensorIndex InputRowsEff = InputRows + padding_top + padding_bottom;
      const TensorIndex InputColsEff = InputCols + padding_left + padding_right;
      out_height = divup(InputRowsEff - kernelRowsEff + 1, row_stride);
      out_width = divup(InputColsEff - kernelColsEff + 1, col_stride);
      break;
    }
    case PADDING_SAME: {
      eigen_assert(!padding_explicit);
      out_height = divup(InputRows, row_stride);
      out_width = divup(InputCols, col_stride);
      break;
    }
    default: {
      // Initialize unused variables to avoid a compiler warning
      out_height = 0;
      out_width = 0;
      eigen_assert(false && "unexpected padding");
    }
  }

  // Molds the output of the patch extraction code into a 2d tensor:
  // - the first dimension (dims[0]): the patch values to be multiplied with the
  // kernels
  // - the second dimension (dims[1]): everything else
  DSizes<TensorIndex, 2> pre_contract_dims;
  if (isColMajor) {
    pre_contract_dims[0] = kernelChannels * kernelRows * kernelCols;
    pre_contract_dims[1] = out_height * out_width;
    for (int i = 3; i < NumDims; ++i) {
      pre_contract_dims[1] *= in.dimension(i);
    }
  } else {
    pre_contract_dims[1] = kernelChannels * kernelRows * kernelCols;
    pre_contract_dims[0] = out_height * out_width;
    for (int i = 0; i < NumDims - 3; ++i) {
      pre_contract_dims[0] *= in.dimension(i);
    }
  }

  // Molds the output of the contraction into the shape expected by the used
  // (assuming this is ColMajor):
  // - 1st dim: kernel filters
  // - 2nd dim: output height
  // - 3rd dim: output width
  // - 4th dim and beyond: everything else including batch size
  DSizes<TensorIndex, NumDims> post_contract_dims;
  if (isColMajor) {
    post_contract_dims[0] = kernelFilters;
    post_contract_dims[1] = out_height;
    post_contract_dims[2] = out_width;
    for (int i = 3; i < NumDims; ++i) {
      post_contract_dims[i] = in.dimension(i);
    }
  } else {
    post_contract_dims[NumDims - 1] = kernelFilters;
    post_contract_dims[NumDims - 2] = out_height;
    post_contract_dims[NumDims - 3] = out_width;
    for (int i = 0; i < NumDims - 3; ++i) {
      post_contract_dims[i] = in.dimension(i);
    }
  }

  DSizes<TensorIndex, 2> kernel_dims;
  if (isColMajor) {
    kernel_dims[0] = kernelFilters;
    kernel_dims[1] = kernelChannels * kernelRows * kernelCols;
  } else {
    kernel_dims[0] = kernelChannels * kernelRows * kernelCols;
    kernel_dims[1] = kernelFilters;
  }
  if (padding_explicit) {
    return choose(
        Cond<internal::traits<Input>::Layout == ColMajor>(),
        kernel.reshape(kernel_dims)
            .contract(input
                          .extract_image_patches(
                              kernelRows, kernelCols, row_stride, col_stride,
                              row_in_stride, col_in_stride,
                              /*row_inflate_stride=*/1,
                              /*col_inflate_stride=*/1, padding_top,
                              padding_bottom, padding_left, padding_right,
                              /*padding_value=*/static_cast<InputScalar>(0))
                          .reshape(pre_contract_dims),
                      contract_dims, output_kernel)
            .reshape(post_contract_dims),
        input
            .extract_image_patches(
                kernelRows, kernelCols, row_stride, col_stride, row_in_stride,
                col_in_stride,
                /*row_inflate_stride=*/1,
                /*col_inflate_stride=*/1, padding_top, padding_bottom,
                padding_left, padding_right,
                /*padding_value=*/static_cast<InputScalar>(0))
            .reshape(pre_contract_dims)
            .contract(kernel.reshape(kernel_dims), contract_dims, output_kernel)
            .reshape(post_contract_dims));
  } else {
    return choose(
        Cond<internal::traits<Input>::Layout == ColMajor>(),
        kernel.reshape(kernel_dims)
            .contract(input
                          .extract_image_patches(
                              kernelRows, kernelCols, row_stride, col_stride,
                              row_in_stride, col_in_stride, padding_type)
                          .reshape(pre_contract_dims),
                      contract_dims, output_kernel)
            .reshape(post_contract_dims),
        input
            .extract_image_patches(kernelRows, kernelCols, row_stride,
                                   col_stride, row_in_stride, col_in_stride,
                                   padding_type)
            .reshape(pre_contract_dims)
            .contract(kernel.reshape(kernel_dims), contract_dims, output_kernel)
            .reshape(post_contract_dims));
  }
}

}  // end namespace Eigen

#endif  // TENSORFLOW_CORE_KERNELS_EIGEN_SPATIAL_CONVOLUTIONS_INL_H_
