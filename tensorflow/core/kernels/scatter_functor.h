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

#ifndef TENSORFLOW_CORE_KERNELS_SCATTER_FUNCTOR_H_
#define TENSORFLOW_CORE_KERNELS_SCATTER_FUNCTOR_H_
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
class MHTracer_DTPStensorflowPScorePSkernelsPSscatter_functorDTh {
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
   MHTracer_DTPStensorflowPScorePSkernelsPSscatter_functorDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSkernelsPSscatter_functorDTh() {
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


#include <type_traits>

#include "third_party/eigen3/Eigen/Core"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/bounds_check.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/variant_op_registry.h"
#include "tensorflow/core/kernels/dense_update_functor.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/util/determinism.h"
#include "tensorflow/core/util/work_sharder.h"

namespace tensorflow {

class OpKernelContext;
typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;

namespace scatter_op {

enum class UpdateOp { ASSIGN, ADD, SUB, MUL, DIV, MIN, MAX };

namespace internal {

template <scatter_op::UpdateOp Op>
struct Assign {};
template <>
struct Assign<scatter_op::UpdateOp::ASSIGN> {
  template <typename Params, typename Update>
  static void Run(Params p, Update u) {
    p = u;
  }
  template <typename Params, typename Update>
  static void RunScalar(Params p, Update u) {
    p.setConstant(u);
  }
};
template <>
struct Assign<scatter_op::UpdateOp::ADD> {
  template <typename Params, typename Update>
  static void Run(Params p, Update u) {
    p += u;
  }
  template <typename Params, typename Update>
  static void RunScalar(Params p, Update u) {
    p = p + u;
  }
};
template <>
struct Assign<scatter_op::UpdateOp::SUB> {
  template <typename Params, typename Update>
  static void Run(Params p, Update u) {
    p -= u;
  }
  template <typename Params, typename Update>
  static void RunScalar(Params p, Update u) {
    p = p + static_cast<Update>(-u);
  }
};
template <>
struct Assign<scatter_op::UpdateOp::MUL> {
  template <typename Params, typename Update>
  static void Run(Params p, Update u) {
    p *= u;
  }
  template <typename Params, typename Update>
  static void RunScalar(Params p, Update u) {
    p = p * u;
  }
};
template <>
struct Assign<scatter_op::UpdateOp::DIV> {
  template <typename Params, typename Update>
  static void Run(Params p, Update u) {
    p /= u;
  }
  template <typename Params, typename Update>
  static void RunScalar(Params p, Update u) {
    p = p / u;
  }
};
template <>
struct Assign<scatter_op::UpdateOp::MIN> {
  // This method requires that Params and Update are tensor types.
  template <typename Params, typename Update>
  static void Run(Params p, Update u) {
    p = p.cwiseMin(u);
  }
  // Same thing, but for Update being a scalar type.
  template <typename Params, typename Update>
  static void RunScalar(Params p, Update u) {
    p = p.cwiseMin(u);
  }
};
template <>
struct Assign<scatter_op::UpdateOp::MAX> {
  template <typename Params, typename Update>
  static void Run(Params p, Update u) {
    p = p.cwiseMax(u);
  }
  template <typename Params, typename Update>
  static void RunScalar(Params p, Update u) {
    p = p.cwiseMax(u);
  }
};


}  // namespace internal
}  // namespace scatter_op

namespace functor {
template <typename Device, typename T, typename Index, scatter_op::UpdateOp op>
struct ScatterFunctor {
  Index operator()(OpKernelContext* c, const Device& d,
                   typename TTypes<T>::Matrix params,
                   typename TTypes<T>::ConstMatrix updates,
                   typename TTypes<Index>::ConstFlat indices);
};

template <typename Device, typename T, typename Index, scatter_op::UpdateOp op>
struct ScatterFunctorBase {
  Index ParallelExecute(OpKernelContext* c, const Device& d,
                        typename TTypes<T>::Matrix params,
                        typename TTypes<T>::ConstMatrix updates,
                        typename TTypes<Index>::ConstFlat indices) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSscatter_functorDTh mht_0(mht_0_v, 313, "", "./tensorflow/core/kernels/scatter_functor.h", "ParallelExecute");

    const Index N = static_cast<Index>(indices.size());
    const Index limit = static_cast<Index>(params.dimension(0));
    const Index kMaxLocks = 1024;
    const Index entries_per_lock = (limit + kMaxLocks - 1) / kMaxLocks;
    // To reduce the number of locks and the memory usage, we divide the whole
    // index space into kMaxLocks regions with each lock serializing access to
    // a region.
    mutex accessed[kMaxLocks];
    std::atomic<Index> bad_index(-1);
    auto ParallelScatter = [&](Index start, Index end) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSscatter_functorDTh mht_1(mht_1_v, 326, "", "./tensorflow/core/kernels/scatter_functor.h", "lambda");

      for (Index i = start; i < end; ++i) {
        // Grab the index and check its validity.  Do this carefully,
        // to avoid checking the value and grabbing it again from
        // memory a second time (a security risk since it may change in
        // between).
        const Index index = ::tensorflow::internal::SubtleMustCopy(indices(i));
        if (!FastBoundsCheck(index, limit)) {
          bad_index = i;
          return;
        }
        const Index lock_id = index / entries_per_lock;
        // Copy last Ndim-1 dimensions of updates[i] to params[index]
        {
          mutex_lock l(accessed[lock_id]);
          scatter_op::internal::Assign<op>::Run(params.template chip<0>(index),
                                                updates.template chip<0>(i));
        }
      }
    };
    const float kMovingCost = 2.5f;
    float shard_cost = kMovingCost * params.dimension(1);
    const DeviceBase::CpuWorkerThreads& worker_threads =
        *(c->device()->tensorflow_cpu_worker_threads());
    Shard(worker_threads.num_threads, worker_threads.workers, N, shard_cost,
          ParallelScatter);  // TODO: Come up with a good cost estimate.
    return bad_index;
  }
  Index SerialExecute(OpKernelContext* c, const Device& d,
                      typename TTypes<T>::Matrix params,
                      typename TTypes<T>::ConstMatrix updates,
                      typename TTypes<Index>::ConstFlat indices) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSscatter_functorDTh mht_2(mht_2_v, 360, "", "./tensorflow/core/kernels/scatter_functor.h", "SerialExecute");

    const Index N = static_cast<Index>(indices.size());
    const Index limit = static_cast<Index>(params.dimension(0));
    for (Index i = 0; i < N; ++i) {
      // Grab the index and check its validity.  Do this carefully,
      // to avoid checking the value and grabbing it again from
      // memory a second time (a security risk since it may change in
      // between).
      const Index index = ::tensorflow::internal::SubtleMustCopy(indices(i));
      if (!FastBoundsCheck(index, limit)) return i;
      // Copy last Ndim-1 dimensions of updates[i] to params[index]
      scatter_op::internal::Assign<op>::Run(params.template chip<0>(index),
                                            updates.template chip<0>(i));
    }
    return -1;
  }

  Index operator()(OpKernelContext* c, const Device& d,
                   typename TTypes<T>::Matrix params,
                   typename TTypes<T>::ConstMatrix updates,
                   typename TTypes<Index>::ConstFlat indices) {
#ifdef PLATFORM_GOOGLE
    // The parallel version is significantly slower internally. Only call the
    // serial version for now.
    // TODO(penporn): Avoid locking in parallelization (sort beforehand).
    return SerialExecute(c, d, params, updates, indices);
#else
    // indices and params sizes were validated in DoCompute().
    const Index N = static_cast<Index>(indices.size());
    const Index limit = static_cast<Index>(params.dimension(0));
    const Index min_n_threshold = 1024;
    const Index ser_par_ratio = 10000;
    // For parallelizing the updates, duplicate entries need to be handled
    // correctly. Multiple updates to the same index has to be serialized.
    // This can lead to lock contention which may nullify the benefits of
    // parallelization. Assuming uniform random distribution of the indices, we
    // come up with a rough heuristic and determine whether the updates execute
    // serially or parallelly. Also if 'N' is small, overheads of parallel
    // execution outweigh its benefits and hence we check the value of N.
    const bool execute_serial = N < min_n_threshold ||
                                (N / limit) > ser_par_ratio ||
                                OpDeterminismRequired();
    if (execute_serial)
      return SerialExecute(c, d, params, updates, indices);
    else
      return ParallelExecute(c, d, params, updates, indices);
#endif  // PLATFORM_GOOGLE
  }
};

template <typename Device, typename Index>
struct ScatterFunctorVariantAssignBase {
  Index operator()(OpKernelContext* c, const Device& d,
                   typename TTypes<Variant>::Matrix params,
                   typename TTypes<Variant>::ConstMatrix updates,
                   typename TTypes<Index>::ConstFlat indices) {
    // indices and params sizes were validated in DoCompute().
    const Index N = static_cast<Index>(indices.size());
    const Index limit = static_cast<Index>(params.dimension(0));
    const Index cols = static_cast<Index>(params.dimension(1));
    DCHECK_EQ(N, updates.dimension(0));
    DCHECK_EQ(cols, updates.dimension(1));
    for (Index i = 0; i < N; i++) {
      // Grab the index and check its validity.  Do this carefully,
      // to avoid checking the value and grabbing it again from
      // memory a second time (a security risk since it may change in between).
      const Index index = ::tensorflow::internal::SubtleMustCopy(indices(i));
      if (!FastBoundsCheck(index, limit)) return i;
      // Copy last Ndim-1 dimensions of updates[i] to params[index]
      for (int j = 0; j < cols; ++j) {
        const Variant& to_scatter = updates(i, j);
        params(index, j) = to_scatter;
      }
    }
    return -1;
  }
};

template <typename Index>
struct ScatterFunctor<CPUDevice, Variant, Index, scatter_op::UpdateOp::ASSIGN>
    : ScatterFunctorVariantAssignBase<CPUDevice, Index> {};

template <typename Index>
struct ScatterFunctor<GPUDevice, Variant, Index, scatter_op::UpdateOp::ASSIGN>
    : ScatterFunctorVariantAssignBase<GPUDevice, Index> {};


template <typename T, typename Index>
struct ScatterFunctorBase<CPUDevice, T, Index, scatter_op::UpdateOp::ASSIGN> {
  Index operator()(OpKernelContext* c, const CPUDevice& d,
                   typename TTypes<T>::Matrix params,
                   typename TTypes<T>::ConstMatrix updates,
                   typename TTypes<Index>::ConstFlat indices) {
    // indices and params sizes were validated in DoCompute().
    const Index N = static_cast<Index>(indices.size());
    const Index limit = static_cast<Index>(params.dimension(0));
    if (!std::is_same<T, tstring>::value) {
      for (Index i = 0; i < N; i++) {
        // Grab the index and check its validity.  Do this carefully,
        // to avoid checking the value and grabbing it again from
        // memory a second time (a security risk since it may change in
        // between).
        const Index index = ::tensorflow::internal::SubtleMustCopy(indices(i));
        if (!FastBoundsCheck(index, limit)) return i;
        memmove(params.data() + index * params.dimension(1),
                updates.data() + i * updates.dimension(1),
                updates.dimension(1) * sizeof(T));
      }
    } else {
      for (Index i = 0; i < N; i++) {
        // Grab the index and check its validity.  Do this carefully,
        // to avoid checking the value and grabbing it again from
        // memory a second time (a security risk since it may change in
        // between).
        const Index index = ::tensorflow::internal::SubtleMustCopy(indices(i));
        if (!FastBoundsCheck(index, limit)) return i;
        // Copy last Ndim-1 dimensions of updates[i] to params[index]
        scatter_op::internal::Assign<scatter_op::UpdateOp::ASSIGN>::Run(
            params.template chip<0>(index), updates.template chip<0>(i));
      }
    }
    return -1;
  }
};

template <typename T, typename Index, scatter_op::UpdateOp op>
struct ScatterFunctor<CPUDevice, T, Index, op>
    : ScatterFunctorBase<CPUDevice, T, Index, op> {};


template <typename Device, typename T, typename Index, scatter_op::UpdateOp op>
struct ScatterScalarFunctor {
  Index operator()(OpKernelContext* c, const Device& d,
                   typename TTypes<T>::Matrix params,
                   const typename TTypes<T>::ConstScalar update,
                   typename TTypes<Index>::ConstFlat indices);
};

template <typename Device, typename T, typename Index, scatter_op::UpdateOp op>
struct ScatterScalarFunctorBase {
  Index operator()(OpKernelContext* c, const Device& d,
                   typename TTypes<T>::Matrix params,
                   const typename TTypes<T>::ConstScalar update,
                   typename TTypes<Index>::ConstFlat indices) {
    // indices and params sizes were validated in DoCompute().
    const Index N = static_cast<Index>(indices.size());
    const Index limit = static_cast<Index>(params.dimension(0));
    for (Index i = 0; i < N; i++) {
      // Grab the index and check its validity.  Do this carefully,
      // to avoid checking the value and grabbing it again from
      // memory a second time (a security risk since it may change in between).
      const Index index = ::tensorflow::internal::SubtleMustCopy(indices(i));
      if (!FastBoundsCheck(index, limit)) return i;
      // Broadcast update to params[index]
      scatter_op::internal::Assign<op>::RunScalar(
          params.template chip<0>(index), update());
    }
    return -1;
  }
};

template <typename Device, typename Index>
struct ScatterScalarFunctorVariantAssignBase {
  Index operator()(OpKernelContext* c, const Device& d,
                   typename TTypes<Variant>::Matrix params,
                   const typename TTypes<Variant>::ConstScalar update,
                   typename TTypes<Index>::ConstFlat indices) {
    // indices and params sizes were validated in DoCompute().
    const Index N = static_cast<Index>(indices.size());
    const Index limit = static_cast<Index>(params.dimension(0));
    const Index cols = static_cast<Index>(params.dimension(1));
    const Variant& to_scatter = update();
    for (Index i = 0; i < N; i++) {
      // Grab the index and check its validity.  Do this carefully,
      // to avoid checking the value and grabbing it again from
      // memory a second time (a security risk since it may change in between).
      const Index index = ::tensorflow::internal::SubtleMustCopy(indices(i));
      if (!FastBoundsCheck(index, limit)) return i;
      // Broadcast update to params[index]
      for (Index j = 0; j < cols; ++j) {
        params(index, j) = to_scatter;
      }
    }
    return -1;
  }
};

template <typename Index>
struct ScatterScalarFunctor<CPUDevice, Variant, Index,
                            scatter_op::UpdateOp::ASSIGN>
    : ScatterScalarFunctorVariantAssignBase<CPUDevice, Index> {};
template <typename Index>
struct ScatterScalarFunctor<GPUDevice, Variant, Index,
                            scatter_op::UpdateOp::ASSIGN>
    : ScatterScalarFunctorVariantAssignBase<GPUDevice, Index> {};


template <typename T, typename Index>
struct ScatterScalarFunctorBase<CPUDevice, T, Index,
                                scatter_op::UpdateOp::ASSIGN> {
  Index operator()(OpKernelContext* c, const CPUDevice& d,
                   typename TTypes<T>::Matrix params,
                   const typename TTypes<T>::ConstScalar update,
                   typename TTypes<Index>::ConstFlat indices) {
    // indices and params sizes were validated in DoCompute().
    const Index N = static_cast<Index>(indices.size());
    const Index limit = static_cast<Index>(params.dimension(0));
    for (Index i = 0; i < N; i++) {
      // Grab the index and check its validity.  Do this carefully,
      // to avoid checking the value and grabbing it again from
      // memory a second time (a security risk since it may change in between).
      const Index index = ::tensorflow::internal::SubtleMustCopy(indices(i));
      if (!FastBoundsCheck(index, limit)) return i;
      // Broadcast update to params[index]
      scatter_op::internal::Assign<scatter_op::UpdateOp::ASSIGN>::RunScalar(
          params.template chip<0>(index), update());
    }
    return -1;
  }
};

template <typename T, typename Index, scatter_op::UpdateOp op>
struct ScatterScalarFunctor<CPUDevice, T, Index, op>
    : ScatterScalarFunctorBase<CPUDevice, T, Index, op> {};


}  // namespace functor
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_KERNELS_SCATTER_FUNCTOR_H_
