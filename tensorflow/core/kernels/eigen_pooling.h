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

#ifndef TENSORFLOW_CORE_KERNELS_EIGEN_POOLING_H_
#define TENSORFLOW_CORE_KERNELS_EIGEN_POOLING_H_
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
class MHTracer_DTPStensorflowPScorePSkernelsPSeigen_poolingDTh {
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
   MHTracer_DTPStensorflowPScorePSkernelsPSeigen_poolingDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSkernelsPSeigen_poolingDTh() {
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


#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

namespace Eigen {

/** SpatialMaxPooling
 * \ingroup CXX11_NeuralNetworks_Module
 *
 * \brief Applies a max-pooling over a multichannel input image.
 *
 * The input parameter is expected to be a with a rank of 4 (channels, height,
 * width, others in col-major, and the reverse of that in row-major).
 *
 * The result can be assigned to a tensor of rank equal to the rank of the
 * input. The dimensions of the result will be channels, height, width, and
 * others (in col-major, and the reverse of that if the input was row-major).
 *
 * The order of the width and height dimensions can be swapped if needed.
 *
 */
template <typename Input>
EIGEN_ALWAYS_INLINE static const TensorReshapingOp<
    const Eigen::DSizes<typename internal::traits<Input>::Index,
                        internal::traits<Input>::NumDimensions>,
    const TensorReductionOp<
        internal::MaxReducer<typename internal::remove_const<
            typename internal::traits<Input>::Scalar>::type>,
        typename internal::conditional<
            internal::traits<Input>::Layout == ColMajor,
            const Eigen::IndexList<Eigen::type2index<1>, Eigen::type2index<2> >,
            const Eigen::IndexList<Eigen::type2index<2>,
                                   Eigen::type2index<3> > >::type,
        const TensorImagePatchOp<Dynamic, Dynamic, const Input> > >
SpatialMaxPooling(const Input& input, DenseIndex patchRows,
                  DenseIndex patchCols, DenseIndex strideRows,
                  DenseIndex strideCols, const PaddingType padding_type,
                  DenseIndex in_strideRows = 1, DenseIndex in_strideCols = 1) {
  EIGEN_STATIC_ASSERT(internal::traits<Input>::NumDimensions == 4,
                      YOU_MADE_A_PROGRAMMING_MISTAKE);

  typedef typename internal::traits<Input>::Index TensorIndex;
  TensorRef<Tensor<typename internal::traits<Input>::Scalar,
                   internal::traits<Input>::NumDimensions,
                   internal::traits<Input>::Layout, TensorIndex> >
      in(input);

  const DenseIndex patchRowsEff =
      patchRows + (patchRows - 1) * (in_strideRows - 1);
  const DenseIndex patchColsEff =
      patchCols + (patchCols - 1) * (in_strideCols - 1);

  static const bool isColMajor = (internal::traits<Input>::Layout == ColMajor);
  static const int idxRows = isColMajor ? 1 : 2;
  static const int idxCols = isColMajor ? 2 : 1;

  // Molds the output of the reduction into the shape expected by the user.
  // (assuming col-major):
  // - 1st dim: channels
  // - 2nd dim: output height
  // - 3rd dim: output width
  // - 4th dim and beyond: everything else including batch size
  Eigen::DSizes<TensorIndex, internal::traits<Input>::NumDimensions>
      post_reduce_dims;
  post_reduce_dims[0] = in.dimension(0);
  if (padding_type == PADDING_VALID) {
    post_reduce_dims[idxRows] = Eigen::divup(
        static_cast<DenseIndex>(in.dimension(idxRows)) - patchRowsEff + 1,
        strideRows);
    post_reduce_dims[idxCols] = Eigen::divup(
        static_cast<DenseIndex>(in.dimension(idxCols)) - patchColsEff + 1,
        strideCols);
  } else {
    post_reduce_dims[idxRows] = Eigen::divup(
        static_cast<DenseIndex>(in.dimension(idxRows)), strideRows);
    post_reduce_dims[idxCols] = Eigen::divup(
        static_cast<DenseIndex>(in.dimension(idxCols)), strideCols);
  }
  post_reduce_dims[3] = in.dimension(3);

  // Take advantage of cxx11 to give the compiler information it can use to
  // optimize the code.
  typename internal::conditional<
      internal::traits<Input>::Layout == ColMajor,
      const Eigen::IndexList<Eigen::type2index<1>, Eigen::type2index<2> >,
      const Eigen::IndexList<Eigen::type2index<2>,
                             Eigen::type2index<3> > >::type reduction_dims;

  return input
      .extract_image_patches(
          patchRows, patchCols, strideRows, strideCols, in_strideRows,
          in_strideCols, padding_type,
          Eigen::NumTraits<typename internal::remove_const<
              typename internal::traits<Input>::Scalar>::type>::lowest())
      .maximum(reduction_dims)
      .reshape(post_reduce_dims);
}

/** CuboidMaxPooling
 * \ingroup CXX11_NeuralNetworks_Module
 *
 * \brief Applies a max-pooling over a multichannel input volume.
 *
 * The input parameter is expected to be a tensor with a rank of 5 (channels,
 * depth, height, width, others in col-major, and the reverse of that in
 * row-major).
 *
 * The result can be assigned to a tensor of rank equal to the rank of the
 * input. The dimensions of the result will be channels, depth, height, width,
 * and others (in col-major, and the reverse of that if the input was
 * row-major).
 *
 * The order of the depth, width and height dimensions can be swapped if
 * needed.
 *
 */
template <typename Input>
EIGEN_ALWAYS_INLINE static const TensorReshapingOp<
    const Eigen::DSizes<DenseIndex, internal::traits<Input>::NumDimensions>,
    const TensorReductionOp<
        internal::MaxReducer<float>,
        const Eigen::IndexList<Eigen::type2index<1> >,
        const TensorReshapingOp<
            const Eigen::DSizes<DenseIndex, 3>,
            const TensorVolumePatchOp<Dynamic, Dynamic, Dynamic,
                                      const Input> > > >
CuboidMaxPooling(const Input& input, DenseIndex patchPlanes,
                 DenseIndex patchRows, DenseIndex patchCols,
                 DenseIndex stridePlanes, DenseIndex strideRows,
                 DenseIndex strideCols, const PaddingType padding_type) {
  EIGEN_STATIC_ASSERT(internal::traits<Input>::NumDimensions == 5,
                      YOU_MADE_A_PROGRAMMING_MISTAKE);
  static const bool isColMajor = (internal::traits<Input>::Layout == ColMajor);

  typedef typename internal::traits<Input>::Index TensorIndex;
  TensorRef<Tensor<typename internal::traits<Input>::Scalar,
                   internal::traits<Input>::NumDimensions,
                   internal::traits<Input>::Layout, TensorIndex> >
      in(input);

  static const int idxPlanes = isColMajor ? 1 : 3;
  static const int idxRows = 2;
  static const int idxCols = isColMajor ? 3 : 1;

  // Molds the output of the reduction into the shape expected by the used
  // (assuming col-major):
  // - 1st dim: channels
  // - 2nd dim: output depth
  // - 3rd dim: output height
  // - 4th dim: output width
  // - 5th dim and beyond: everything else including batch size
  Eigen::DSizes<DenseIndex, internal::traits<Input>::NumDimensions>
      post_reduce_dims;
  post_reduce_dims[0] = in.dimension(0);
  if (padding_type == PADDING_VALID) {
    post_reduce_dims[idxPlanes] = Eigen::divup(
        static_cast<DenseIndex>(in.dimension(idxPlanes)) - patchPlanes + 1,
        stridePlanes);
    post_reduce_dims[idxRows] = Eigen::divup(
        static_cast<DenseIndex>(in.dimension(idxRows)) - patchRows + 1,
        strideRows);
    post_reduce_dims[idxCols] = Eigen::divup(
        static_cast<DenseIndex>(in.dimension(idxCols)) - patchCols + 1,
        strideCols);
  } else {
    post_reduce_dims[idxPlanes] = Eigen::divup(
        static_cast<DenseIndex>(in.dimension(idxPlanes)), stridePlanes);
    post_reduce_dims[idxRows] = Eigen::divup(
        static_cast<DenseIndex>(in.dimension(idxRows)), strideRows);
    post_reduce_dims[idxCols] = Eigen::divup(
        static_cast<DenseIndex>(in.dimension(idxCols)), strideCols);
  }
  post_reduce_dims[4] = in.dimension(4);

  Eigen::DSizes<DenseIndex, 3> pre_reduce_dims;
  pre_reduce_dims[1] = patchRows * patchCols * patchPlanes;
  if (isColMajor) {
    pre_reduce_dims[0] = post_reduce_dims[0];
    pre_reduce_dims[2] = post_reduce_dims[1] * post_reduce_dims[2] *
                         post_reduce_dims[3] * post_reduce_dims[4];
  } else {
    pre_reduce_dims[0] = post_reduce_dims[0] * post_reduce_dims[1] *
                         post_reduce_dims[2] * post_reduce_dims[3];
    pre_reduce_dims[2] = post_reduce_dims[4];
  }

  // Take advantage of cxx11 to give the compiler information it can use to
  // optimize the code.
  Eigen::IndexList<Eigen::type2index<1> > reduction_dims;
  return input
      .extract_volume_patches(patchPlanes, patchRows, patchCols, stridePlanes,
                              strideRows, strideCols, padding_type,
                              -Eigen::NumTraits<float>::highest())
      .reshape(pre_reduce_dims)
      .maximum(reduction_dims)
      .reshape(post_reduce_dims);
}

/** SpatialAvgPooling
 * \ingroup CXX11_NeuralNetworks_Module
 *
 * \brief Applies an average pooling over a multichannel input image.
 *
 * The input parameter is expected to be a tensor with a rank of 4 (channels,
 * height, width, others in col-major, and the reverse of that in row-major).
 *
 * The result can be assigned to a tensor of rank equal to the rank of the
 * input. The dimensions of the result will be channels, height, width, and
 * others (in col-major, and the reverse of that if the input was row-major).
 *
 * The order of the width and height dimensions can be swapped if needed.
 *
 */
namespace internal {

template <typename T>
struct AvgPoolMeanReducer {
#if (EIGEN_ARCH_i386 || EIGEN_ARCH_x86_64) && !defined(__CUDACC__) && \
    !defined(__HIPCC__)
  // We only support packet access for floats.
  static constexpr bool PacketAccess = internal::is_same<T, float>::value;
#else
  static const bool PacketAccess = false;
#endif
  static constexpr bool IsStateful = true;

  EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE AvgPoolMeanReducer() : scalarCount_(0) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSeigen_poolingDTh mht_0(mht_0_v, 412, "", "./tensorflow/core/kernels/eigen_pooling.h", "AvgPoolMeanReducer");

    typedef typename packet_traits<T>::type Packet;
#if defined(__HIPCC__)
    packetCount_ = 0;
#else
    packetCount_ = pset1<Packet>(T(0.0));
#endif
  }

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE void reduce(const T t, T* accum) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSeigen_poolingDTh mht_1(mht_1_v, 424, "", "./tensorflow/core/kernels/eigen_pooling.h", "reduce");

    if (t != -Eigen::NumTraits<T>::highest()) {
      (*accum) = (*accum) + t;
      scalarCount_++;
    }
  }

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE T initialize() const {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSeigen_poolingDTh mht_2(mht_2_v, 434, "", "./tensorflow/core/kernels/eigen_pooling.h", "initialize");

    return static_cast<T>(0);
  }

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE T finalize(const T accum) const {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSeigen_poolingDTh mht_3(mht_3_v, 441, "", "./tensorflow/core/kernels/eigen_pooling.h", "finalize");

    eigen_assert(scalarCount_ > 0);
    return accum / T(scalarCount_);
  }

#if (EIGEN_ARCH_i386 || EIGEN_ARCH_x86_64) && !defined(__CUDACC__) && \
    !defined(__HIPCC__)
#ifdef EIGEN_VECTORIZE_AVX512
#define pequal(a, b)   \
  _mm512_castsi512_ps( \
      _mm512_maskz_set1_epi32(_mm512_cmp_ps_mask(a, b, _CMP_EQ_UQ), -1))

  // The ternarylogic function immediate determines the values in the result
  // In the case below, 0xd8 implies (false_mask) ? (b) : (a)
  // For details, refer to the vpternlogd instruction table at
  // http://www.intel.com/content/dam/www/public/us/en/documents/manuals/64-ia-32-architectures-software-developer-vol-2c-manual.pdf

#define psel(a, b, false_mask)                        \
  _mm512_castsi512_ps(_mm512_ternarylogic_epi32(      \
      _mm512_castps_si512(a), _mm512_castps_si512(b), \
      _mm512_castps_si512(false_mask), 0xd8))
#elif defined EIGEN_VECTORIZE_AVX
#define pequal(a, b) _mm256_cmp_ps(a, b, _CMP_EQ_UQ)
#define psel(a, b, false_mask) _mm256_blendv_ps(a, b, false_mask)
#else
#define pequal(a, b) _mm_cmpeq_ps(a, b)
#define psel(a, b, false_mask) \
  _mm_or_ps(_mm_andnot_ps(false_mask, a), _mm_and_ps(false_mask, b))
#endif

  template <typename Packet>
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE void reducePacket(const Packet& p,
                                                          Packet* accum) {
    reducePacketWithType(static_cast<T>(0), p, accum);
  }

  template <typename Packet>
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE void reducePacketWithType(
      T, const Packet& p, Packet* accum) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSeigen_poolingDTh mht_4(mht_4_v, 482, "", "./tensorflow/core/kernels/eigen_pooling.h", "reducePacketWithType");

    Packet skip_mask =
        pequal(p, pset1<Packet>(-Eigen::NumTraits<T>::highest()));
    (*accum) = padd<Packet>(*accum, psel(p, pset1<Packet>(0), skip_mask));
    packetCount_ = padd<Packet>(
        packetCount_, psel(pset1<Packet>(1), pset1<Packet>(0), skip_mask));
  }

  template <typename Packet>
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Packet initializePacket() const {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSeigen_poolingDTh mht_5(mht_5_v, 494, "", "./tensorflow/core/kernels/eigen_pooling.h", "initializePacket");

    return pset1<Packet>(0);
  }

  template <typename Packet>
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Packet
  finalizePacket(const Packet& vaccum) const {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSeigen_poolingDTh mht_6(mht_6_v, 503, "", "./tensorflow/core/kernels/eigen_pooling.h", "finalizePacket");

    return pdiv(vaccum, packetCount_);
  }
  template <typename Packet>
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE T
  finalizeBoth(const T saccum, const Packet& vaccum) const {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSeigen_poolingDTh mht_7(mht_7_v, 511, "", "./tensorflow/core/kernels/eigen_pooling.h", "finalizeBoth");

    return (saccum + predux(vaccum)) / (scalarCount_ + predux(packetCount_));
  }
#endif

 protected:
  typedef typename packet_traits<T>::type Packet;
  int scalarCount_;
#if defined(__HIPCC__)
  int packetCount_;
#else
  Packet packetCount_;
#endif
};

template <typename Device>
struct reducer_traits<AvgPoolMeanReducer<float>, Device> {
  enum {
    Cost = 1,
#if (EIGEN_ARCH_i386 || EIGEN_ARCH_x86_64) && !defined(__CUDACC__) && \
    !defined(__HIPCC__)
    // We only support packet access for floats.
    PacketAccess = true,
#else
    PacketAccess = false,
#endif
    IsStateful = true,
    IsExactlyAssociative = false
  };
};

template <>
struct reducer_traits<AvgPoolMeanReducer<float>, GpuDevice> {
  enum {
    Cost = 1,
    PacketAccess = false,
    IsStateful = true,
    IsExactlyAssociative = false
  };
};

}  // namespace internal

template <typename Input>
EIGEN_ALWAYS_INLINE static const TensorReshapingOp<
    const Eigen::DSizes<typename internal::traits<Input>::Index,
                        internal::traits<Input>::NumDimensions>,
    const TensorReductionOp<
        internal::AvgPoolMeanReducer<typename internal::remove_const<
            typename internal::traits<Input>::Scalar>::type>,
        typename internal::conditional<
            internal::traits<Input>::Layout == ColMajor,
            const Eigen::IndexList<Eigen::type2index<1>, Eigen::type2index<2> >,
            const Eigen::IndexList<Eigen::type2index<2>,
                                   Eigen::type2index<3> > >::type,
        const TensorImagePatchOp<Dynamic, Dynamic, const Input> > >
SpatialAvgPooling(const Input& input, DenseIndex patchRows,
                  DenseIndex patchCols, DenseIndex strideRows,
                  DenseIndex strideCols, const PaddingType padding_type,
                  DenseIndex in_strideRows = 1, DenseIndex in_strideCols = 1) {
  EIGEN_STATIC_ASSERT(internal::traits<Input>::NumDimensions == 4,
                      YOU_MADE_A_PROGRAMMING_MISTAKE);

  typedef typename internal::traits<Input>::Index TensorIndex;
  TensorRef<Tensor<typename internal::traits<Input>::Scalar,
                   internal::traits<Input>::NumDimensions,
                   internal::traits<Input>::Layout, TensorIndex> >
      in(input);

  const DenseIndex patchRowsEff =
      patchRows + (patchRows - 1) * (in_strideRows - 1);
  const DenseIndex patchColsEff =
      patchCols + (patchCols - 1) * (in_strideCols - 1);

  static const bool isColMajor = (internal::traits<Input>::Layout == ColMajor);
  static const int idxRows = isColMajor ? 1 : 2;
  static const int idxCols = isColMajor ? 2 : 1;

  // Molds the output of the reduction into the shape expected by the user.
  // (assuming col-major):
  // - 1st dim: channels
  // - 2nd dim: output height
  // - 3rd dim: output width
  // - 4th dim and beyond: everything else including batch size
  Eigen::DSizes<TensorIndex, internal::traits<Input>::NumDimensions>
      post_reduce_dims;
  post_reduce_dims[0] = in.dimension(0);
  if (padding_type == PADDING_VALID) {
    post_reduce_dims[idxRows] = Eigen::divup(
        static_cast<DenseIndex>(in.dimension(idxRows)) - patchRowsEff + 1,
        strideRows);
    post_reduce_dims[idxCols] = Eigen::divup(
        static_cast<DenseIndex>(in.dimension(idxCols)) - patchColsEff + 1,
        strideCols);
  } else {
    post_reduce_dims[idxRows] = Eigen::divup(
        static_cast<DenseIndex>(in.dimension(idxRows)), strideRows);
    post_reduce_dims[idxCols] = Eigen::divup(
        static_cast<DenseIndex>(in.dimension(idxCols)), strideCols);
  }
  post_reduce_dims[3] = in.dimension(3);

  typedef typename internal::remove_const<
      typename internal::traits<Input>::Scalar>::type CoeffReturnType;
  internal::AvgPoolMeanReducer<CoeffReturnType> mean_with_nan;

  // Take advantage of cxx11 to give the compiler information it can use to
  // optimize the code.
  typename internal::conditional<
      internal::traits<Input>::Layout == ColMajor,
      const Eigen::IndexList<Eigen::type2index<1>, Eigen::type2index<2> >,
      const Eigen::IndexList<Eigen::type2index<2>,
                             Eigen::type2index<3> > >::type reduction_dims;
  return input
      .extract_image_patches(patchRows, patchCols, strideRows, strideCols,
                             in_strideRows, in_strideCols, padding_type,
                             -Eigen::NumTraits<CoeffReturnType>::highest())
      .reduce(reduction_dims, mean_with_nan)
      .reshape(post_reduce_dims);
}

/** CuboidAvgPooling
 * \ingroup CXX11_NeuralNetworks_Module
 *
 * \brief Applies an average pooling over a multichannel input volume.
 *
 * The input parameter is expected to be a tensor with a rank of 5 (channels,
 * depth, height, width, others, and the reverse of that in row-major).
 *
 * The result can be assigned to a tensor of rank equal to the rank of the
 * input. The dimensions of the result will be channels, depth, width, and
 * others (in col-major, and the reverse of that if the input was row-major).
 *
 * The order of the depth, width and height dimensions can be swapped if
 * needed.
 *
 */
template <typename Input>
EIGEN_ALWAYS_INLINE static const TensorReshapingOp<
    const Eigen::DSizes<DenseIndex, internal::traits<Input>::NumDimensions>,
    const TensorReductionOp<
        internal::AvgPoolMeanReducer<float>,
        const Eigen::IndexList<Eigen::type2index<1> >,
        const TensorReshapingOp<
            const Eigen::DSizes<DenseIndex, 3>,
            const TensorVolumePatchOp<Dynamic, Dynamic, Dynamic,
                                      const Input> > > >
CuboidAvgPooling(const Input& input, DenseIndex patchPlanes,
                 DenseIndex patchRows, DenseIndex patchCols,
                 DenseIndex stridePlanes, DenseIndex strideRows,
                 DenseIndex strideCols, const PaddingType padding_type) {
  EIGEN_STATIC_ASSERT(internal::traits<Input>::NumDimensions == 5,
                      YOU_MADE_A_PROGRAMMING_MISTAKE);
  static const bool isColMajor = (internal::traits<Input>::Layout == ColMajor);

  typedef typename internal::traits<Input>::Index TensorIndex;
  TensorRef<Tensor<typename internal::traits<Input>::Scalar,
                   internal::traits<Input>::NumDimensions,
                   internal::traits<Input>::Layout, TensorIndex> >
      in(input);

  static const int idxPlanes = isColMajor ? 1 : 3;
  static const int idxRows = 2;
  static const int idxCols = isColMajor ? 3 : 1;
  // Molds the output of the reduction into the shape expected by the used
  // (assuming col-major):
  // - 1st dim: channels
  // - 2nd dim: outupt depth
  // - 3rd dim: output height
  // - 4th dim: output width
  // - 5th dim and beyond: everything else including batch size
  Eigen::DSizes<DenseIndex, internal::traits<Input>::NumDimensions>
      post_reduce_dims;
  post_reduce_dims[0] = in.dimension(0);
  if (padding_type == PADDING_VALID) {
    post_reduce_dims[idxPlanes] = Eigen::divup(
        static_cast<DenseIndex>(in.dimension(idxPlanes)) - patchPlanes + 1,
        stridePlanes);
    post_reduce_dims[idxRows] = Eigen::divup(
        static_cast<DenseIndex>(in.dimension(idxRows)) - patchRows + 1,
        strideRows);
    post_reduce_dims[idxCols] = Eigen::divup(
        static_cast<DenseIndex>(in.dimension(idxCols)) - patchCols + 1,
        strideCols);
  } else {
    post_reduce_dims[idxPlanes] = Eigen::divup(
        static_cast<DenseIndex>(in.dimension(idxPlanes)), stridePlanes);
    post_reduce_dims[idxRows] = Eigen::divup(
        static_cast<DenseIndex>(in.dimension(idxRows)), strideRows);
    post_reduce_dims[idxCols] = Eigen::divup(
        static_cast<DenseIndex>(in.dimension(idxCols)), strideCols);
  }
  post_reduce_dims[4] = in.dimension(4);

  Eigen::DSizes<DenseIndex, 3> pre_reduce_dims;
  pre_reduce_dims[1] = patchRows * patchCols * patchPlanes;
  if (isColMajor) {
    pre_reduce_dims[0] = post_reduce_dims[0];
    pre_reduce_dims[2] = post_reduce_dims[1] * post_reduce_dims[2] *
                         post_reduce_dims[3] * post_reduce_dims[4];
  } else {
    pre_reduce_dims[0] = post_reduce_dims[0] * post_reduce_dims[1] *
                         post_reduce_dims[2] * post_reduce_dims[3];
    pre_reduce_dims[2] = post_reduce_dims[4];
  }

  typedef typename internal::remove_const<
      typename internal::traits<Input>::Scalar>::type CoeffReturnType;
  internal::AvgPoolMeanReducer<CoeffReturnType> mean_with_nan;

  // Take advantage of cxx11 to give the compiler information it can use to
  // optimize the code.
  Eigen::IndexList<Eigen::type2index<1> > reduction_dims;
  return input
      .extract_volume_patches(patchPlanes, patchRows, patchCols, stridePlanes,
                              strideRows, strideCols, padding_type,
                              -Eigen::NumTraits<float>::highest())
      .reshape(pre_reduce_dims)
      .reduce(reduction_dims, mean_with_nan)
      .reshape(post_reduce_dims);
}

}  // end namespace Eigen

#endif  // TENSORFLOW_CORE_KERNELS_EIGEN_POOLING_H_
