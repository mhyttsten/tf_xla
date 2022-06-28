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

#ifndef TENSORFLOW_CORE_FRAMEWORK_TENSOR_TYPES_H_
#define TENSORFLOW_CORE_FRAMEWORK_TENSOR_TYPES_H_
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
class MHTracer_DTPStensorflowPScorePSframeworkPStensor_typesDTh {
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
   MHTracer_DTPStensorflowPScorePSframeworkPStensor_typesDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSframeworkPStensor_typesDTh() {
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
#include "tensorflow/core/platform/logging.h"

namespace tensorflow {

// Helper to define Tensor types given that the scalar is of type T.
template <typename T, int NDIMS = 1, typename IndexType = Eigen::DenseIndex>
struct TTypes {
  // Rank-<NDIMS> tensor of scalar type T.
  typedef Eigen::TensorMap<Eigen::Tensor<T, NDIMS, Eigen::RowMajor, IndexType>,
                           Eigen::Aligned>
      Tensor;
  typedef Eigen::TensorMap<
      Eigen::Tensor<const T, NDIMS, Eigen::RowMajor, IndexType>, Eigen::Aligned>
      ConstTensor;

  // Unaligned Rank-<NDIMS> tensor of scalar type T.
  typedef Eigen::TensorMap<Eigen::Tensor<T, NDIMS, Eigen::RowMajor, IndexType> >
      UnalignedTensor;
  typedef Eigen::TensorMap<
      Eigen::Tensor<const T, NDIMS, Eigen::RowMajor, IndexType> >
      UnalignedConstTensor;

  typedef Eigen::TensorMap<Eigen::Tensor<T, NDIMS, Eigen::RowMajor, int>,
                           Eigen::Aligned>
      Tensor32Bit;

  // Scalar tensor (implemented as a rank-0 tensor) of scalar type T.
  typedef Eigen::TensorMap<
      Eigen::TensorFixedSize<T, Eigen::Sizes<>, Eigen::RowMajor, IndexType>,
      Eigen::Aligned>
      Scalar;
  typedef Eigen::TensorMap<Eigen::TensorFixedSize<const T, Eigen::Sizes<>,
                                                  Eigen::RowMajor, IndexType>,
                           Eigen::Aligned>
      ConstScalar;

  // Unaligned Scalar tensor of scalar type T.
  typedef Eigen::TensorMap<
      Eigen::TensorFixedSize<T, Eigen::Sizes<>, Eigen::RowMajor, IndexType> >
      UnalignedScalar;
  typedef Eigen::TensorMap<Eigen::TensorFixedSize<const T, Eigen::Sizes<>,
                                                  Eigen::RowMajor, IndexType> >
      UnalignedConstScalar;

  // Rank-1 tensor (vector) of scalar type T.
  typedef Eigen::TensorMap<Eigen::Tensor<T, 1, Eigen::RowMajor, IndexType>,
                           Eigen::Aligned>
      Flat;
  typedef Eigen::TensorMap<
      Eigen::Tensor<const T, 1, Eigen::RowMajor, IndexType>, Eigen::Aligned>
      ConstFlat;
  typedef Eigen::TensorMap<Eigen::Tensor<T, 1, Eigen::RowMajor, IndexType>,
                           Eigen::Aligned>
      Vec;
  typedef Eigen::TensorMap<
      Eigen::Tensor<const T, 1, Eigen::RowMajor, IndexType>, Eigen::Aligned>
      ConstVec;

  // Unaligned Rank-1 tensor (vector) of scalar type T.
  typedef Eigen::TensorMap<Eigen::Tensor<T, 1, Eigen::RowMajor, IndexType> >
      UnalignedFlat;
  typedef Eigen::TensorMap<
      Eigen::Tensor<const T, 1, Eigen::RowMajor, IndexType> >
      UnalignedConstFlat;
  typedef Eigen::TensorMap<Eigen::Tensor<T, 1, Eigen::RowMajor, IndexType> >
      UnalignedVec;
  typedef Eigen::TensorMap<
      Eigen::Tensor<const T, 1, Eigen::RowMajor, IndexType> >
      UnalignedConstVec;

  // Rank-2 tensor (matrix) of scalar type T.
  typedef Eigen::TensorMap<Eigen::Tensor<T, 2, Eigen::RowMajor, IndexType>,
                           Eigen::Aligned>
      Matrix;
  typedef Eigen::TensorMap<
      Eigen::Tensor<const T, 2, Eigen::RowMajor, IndexType>, Eigen::Aligned>
      ConstMatrix;

  // Unaligned Rank-2 tensor (matrix) of scalar type T.
  typedef Eigen::TensorMap<Eigen::Tensor<T, 2, Eigen::RowMajor, IndexType> >
      UnalignedMatrix;
  typedef Eigen::TensorMap<
      Eigen::Tensor<const T, 2, Eigen::RowMajor, IndexType> >
      UnalignedConstMatrix;
};

typedef typename TTypes<float, 1>::Tensor32Bit::Index Index32;

template <typename Index, int NumDims>
bool SafeFor32BitIndexing(const Eigen::DSizes<Index, NumDims>& in) {
  for (int i = 0; i < NumDims; ++i) {
    if (in[i] > std::numeric_limits<Index32>::max()) return false;
  }
  return true;
}

template <typename Index, size_t NumDims>
bool SafeFor32BitIndexing(const Eigen::array<Index, NumDims>& in) {
  for (size_t i = 0; i < NumDims; ++i) {
    if (in[i] > std::numeric_limits<Index32>::max()) return false;
  }
  return true;
}

template <typename TensorType,
          typename Enable = typename TTypes<
              typename TensorType::Scalar, TensorType::NumIndices>::Tensor32Bit>
bool SafeFor32BitIndexing(TensorType in) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSframeworkPStensor_typesDTh mht_0(mht_0_v, 296, "", "./tensorflow/core/framework/tensor_types.h", "SafeFor32BitIndexing");

  return in.size() <= std::numeric_limits<Index32>::max();
}

template <typename Index, int NumDims>
Eigen::DSizes<Index32, NumDims> To32Bit(
    const Eigen::DSizes<Index, NumDims>& in) {
  DCHECK(SafeFor32BitIndexing(in));
  Eigen::DSizes<Index32, NumDims> out;
  for (int i = 0; i < NumDims; ++i) {
    out[i] = static_cast<Index32>(in[i]);
  }
  return out;
}

template <typename Index, size_t NumDims>
Eigen::array<Index32, NumDims> To32Bit(const Eigen::array<Index, NumDims>& in) {
  DCHECK(SafeFor32BitIndexing(in));
  Eigen::array<Index32, NumDims> out;
  for (size_t i = 0; i < NumDims; ++i) {
    out[i] = static_cast<Index32>(in[i]);
  }
  return out;
}

template <typename TensorType>
typename TTypes<typename TensorType::Scalar,
                TensorType::NumIndices>::Tensor32Bit
To32Bit(TensorType in) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSframeworkPStensor_typesDTh mht_1(mht_1_v, 327, "", "./tensorflow/core/framework/tensor_types.h", "To32Bit");

  typedef typename TTypes<typename TensorType::Scalar,
                          TensorType::NumIndices>::Tensor32Bit RetType;
  DCHECK(SafeFor32BitIndexing(in));
  return RetType(in.data(), To32Bit(in.dimensions()));
}

namespace internal {

template <typename Device>
struct MaybeWith32BitIndexingImpl {
  template <typename Func, typename... Args>
  void operator()(Func func, Args&&... args) const {
    func(std::forward<Args>(args)...);
  }
};

template <>
struct MaybeWith32BitIndexingImpl<Eigen::GpuDevice> {
  template <typename Func, typename... Args>
  void operator()(Func func, Args&&... args) const {
    auto all = [](const auto&... bool_vals) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSframeworkPStensor_typesDTh mht_2(mht_2_v, 351, "", "./tensorflow/core/framework/tensor_types.h", "lambda");

      for (bool b : {bool_vals...}) {
        if (!b) return false;
      }
      return true;
    };
    if (all(SafeFor32BitIndexing(std::forward<Args>(args))...)) {
      func(To32Bit(std::forward<Args>(args))...);
    } else {
      func(std::forward<Args>(args)...);
    }
  }
};

}  // namespace internal

template <typename Device, typename Func, typename... Args>
void MaybeWith32BitIndexing(Func func, Args&&... args) {
  return internal::MaybeWith32BitIndexingImpl<Device>()(
      func, std::forward<Args>(args)...);
}

}  // namespace tensorflow
#endif  // TENSORFLOW_CORE_FRAMEWORK_TENSOR_TYPES_H_
