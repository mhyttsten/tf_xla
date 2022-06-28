/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_CORE_KERNELS_RNN_LSTM_OPS_H_
#define TENSORFLOW_CORE_KERNELS_RNN_LSTM_OPS_H_
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
class MHTracer_DTPStensorflowPScorePSkernelsPSrnnPSlstm_opsDTh {
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
   MHTracer_DTPStensorflowPScorePSkernelsPSrnnPSlstm_opsDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSkernelsPSrnnPSlstm_opsDTh() {
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
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/kernels/eigen_activations.h"
#include "tensorflow/core/kernels/rnn/blas_gemm.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {
class OpKernelContext;

enum GateLayout { ICFO, IFCO };

constexpr int gate_c_offset(GateLayout gate_layout, int cell_size) {
  return (gate_layout == ICFO) ? cell_size : cell_size * 2;
}

constexpr int gate_f_offset(GateLayout gate_layout, int cell_size) {
  return (gate_layout == ICFO) ? cell_size * 2 : cell_size;
}

namespace functor {

template <typename Device, typename T>
struct TensorZero {
  void operator()(const Device& d, typename TTypes<T>::Flat t) {
    t.device(d) = t.constant(T(0));
  }
};

template <typename Device, typename T>
struct TensorUnalignedZero {
  void operator()(const Device& d, typename TTypes<T>::UnalignedFlat t) {
    t.device(d) = t.constant(T(0));
  }
};

template <typename Device, typename T>
struct TensorCopy {
  void operator()(const Device& d, typename TTypes<T>::ConstFlat src,
                  typename TTypes<T>::Flat dst) {
    dst.device(d) = src;
  }
};

template <typename Device, typename T>
struct TensorCopyUnaligned {
  void operator()(const Device& d, typename TTypes<T>::UnalignedConstFlat src,
                  typename TTypes<T>::Flat dst) {
    dst.device(d) = src;
  }
};

template <typename Device, typename T>
struct TensorCopyToUnaligned {
  void operator()(const Device& d, typename TTypes<T>::ConstFlat src,
                  typename TTypes<T>::UnalignedFlat dst) {
    dst.device(d) = src;
  }
};

template <typename Device, typename T>
struct TensorAdd {
  void operator()(const Device& d, typename TTypes<T>::ConstFlat a,
                  typename TTypes<T>::ConstFlat b, typename TTypes<T>::Flat c) {
    c.device(d) = a + b;
  }
};

template <typename Device, typename T>
struct TensorZeroPadding {
  void operator()(const Device& d, const int64_t time_idx,
                  typename TTypes<int64_t>::ConstVec seq_len,
                  typename TTypes<T>::Vec mask, typename TTypes<T>::Matrix m) {
    // mask is shape [batch_size].
    mask.device(d) = seq_len.constant(time_idx) < seq_len;

    // m_shape is [batch_size, 1].
    Eigen::array<Eigen::DenseIndex, 2> m_shape({m.dimensions()[0], 1});
    // broadcast_shape is [1, units].
    Eigen::array<Eigen::DenseIndex, 2> broadcast_shape({1, m.dimensions()[1]});

    // m is shape [batch_size, units].
    m.device(d) = m * mask.reshape(m_shape).broadcast(broadcast_shape);
  }
};

struct LSTMBlockCell {
  LSTMBlockCell(const int batch_size, const int input_size, const int cell_size)
      : batch_size_(batch_size),
        input_size_(input_size),
        cell_size_(cell_size) {}

  int batch_size() const {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSrnnPSlstm_opsDTh mht_0(mht_0_v, 279, "", "./tensorflow/core/kernels/rnn/lstm_ops.h", "batch_size");
 return batch_size_; }

  int input_size() const {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSrnnPSlstm_opsDTh mht_1(mht_1_v, 284, "", "./tensorflow/core/kernels/rnn/lstm_ops.h", "input_size");
 return input_size_; }

  int cell_size() const {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSrnnPSlstm_opsDTh mht_2(mht_2_v, 289, "", "./tensorflow/core/kernels/rnn/lstm_ops.h", "cell_size");
 return cell_size_; }

  inline Eigen::array<Eigen::DenseIndex, 2> gates_i_offsets() const {
    return {0, 0};
  }

  inline Eigen::array<Eigen::DenseIndex, 2> gates_c_offsets(
      const GateLayout gate_layout) const {
    return {0, gate_c_offset(gate_layout, cell_size_)};
  }

  inline Eigen::array<Eigen::DenseIndex, 2> gates_f_offsets(
      const GateLayout gate_layout) const {
    return {0, gate_f_offset(gate_layout, cell_size_)};
  }

  inline Eigen::array<Eigen::DenseIndex, 2> gates_o_offsets() const {
    return {0, cell_size_ * 3};
  }

  inline Eigen::array<Eigen::DenseIndex, 2> cell_extents() const {
    return {batch_size_, cell_size_};
  }

  inline Eigen::array<Eigen::DenseIndex, 2> xh_x_offsets() const {
    return {0, 0};
  }

  inline Eigen::array<Eigen::DenseIndex, 2> xh_x_extents() const {
    return {batch_size_, input_size_};
  }

  inline Eigen::array<Eigen::DenseIndex, 2> xh_h_offsets() const {
    return {0, input_size_};
  }

  inline Eigen::array<Eigen::DenseIndex, 2> xh_h_extents() const {
    return {batch_size_, cell_size_};
  }

 protected:
  const int batch_size_;
  const int input_size_;
  const int cell_size_;
};

// See lstm_ops.cc for CPUDevice implementation and lstm_ops_gpu.cu.cc for
// GPUDevice implementation.
template <typename Device, typename T, bool USE_CUBLAS, GateLayout gate_layout>
struct LSTMBlockCellFprop : public LSTMBlockCell {
  LSTMBlockCellFprop(const int batch_size, const int input_size,
                     const int cell_size)
      : LSTMBlockCell(batch_size, input_size, cell_size) {}

  void operator()(OpKernelContext* ctx, const Device& d,
                  const float forget_bias, const float cell_clip,
                  bool use_peephole, typename TTypes<T>::ConstMatrix x,
                  typename TTypes<T>::ConstMatrix cs_prev,
                  typename TTypes<T>::ConstMatrix h_prev,
                  typename TTypes<T>::ConstMatrix w,
                  typename TTypes<T>::ConstVec wci,
                  typename TTypes<T>::ConstVec wcf,
                  typename TTypes<T>::ConstVec wco,
                  typename TTypes<T>::ConstVec b, typename TTypes<T>::Matrix xh,
                  typename TTypes<T>::Matrix i, typename TTypes<T>::Matrix cs,
                  typename TTypes<T>::Matrix f, typename TTypes<T>::Matrix o,
                  typename TTypes<T>::Matrix ci, typename TTypes<T>::Matrix co,
                  typename TTypes<T>::Matrix gates,
                  typename TTypes<T>::Matrix h);
};

// See lstm_ops.cc for CPUDevice implementation and lstm_ops_gpu.cu.cc for
// GPUDevice implementation.
template <typename Device, typename T, bool USE_CUBLAS, GateLayout gate_layout>
struct LSTMBlockCellBprop : public LSTMBlockCell {
  LSTMBlockCellBprop(const int batch_size, const int input_size,
                     const int cell_size)
      : LSTMBlockCell(batch_size, input_size, cell_size) {}

  void operator()(
      OpKernelContext* ctx, const Device& d, bool use_peephole,
      typename TTypes<T>::ConstMatrix x,
      typename TTypes<T>::ConstMatrix cs_prev,
      typename TTypes<T>::ConstMatrix h_prev, typename TTypes<T>::ConstMatrix w,
      typename TTypes<T>::ConstVec wci, typename TTypes<T>::ConstVec wcf,
      typename TTypes<T>::ConstVec wco, typename TTypes<T>::ConstVec b,
      typename TTypes<T>::ConstMatrix i, typename TTypes<T>::ConstMatrix cs,
      typename TTypes<T>::ConstMatrix f, typename TTypes<T>::ConstMatrix o,
      typename TTypes<T>::ConstMatrix ci, typename TTypes<T>::ConstMatrix co,
      typename TTypes<T>::ConstMatrix cs_grad,
      typename TTypes<T>::ConstMatrix h_grad, typename TTypes<T>::Matrix do_,
      typename TTypes<T>::Matrix dcs, typename TTypes<T>::Matrix dci,
      typename TTypes<T>::Matrix df, typename TTypes<T>::Matrix di,
      typename TTypes<T>::Matrix dgates,
      typename TTypes<T>::Matrix cs_prev_grad, typename TTypes<T>::Vec wci_grad,
      typename TTypes<T>::Vec wcf_grad, typename TTypes<T>::Vec wco_grad);
};

template <typename Device, typename T, bool USE_CUBLAS, GateLayout gate_layout>
struct BlockLSTMBprop : public LSTMBlockCell {
  BlockLSTMBprop(const int batch_size, const int input_size,
                 const int cell_size)
      : LSTMBlockCell(batch_size, input_size, cell_size) {}

  void operator()(
      OpKernelContext* ctx, const Device& d, bool use_peephole,
      typename TTypes<T>::ConstMatrix x,
      typename TTypes<T>::ConstMatrix cs_prev,
      typename TTypes<T>::ConstMatrix h_prev, typename TTypes<T>::ConstMatrix w,
      typename TTypes<T>::ConstVec wci, typename TTypes<T>::ConstVec wcf,
      typename TTypes<T>::ConstVec wco, typename TTypes<T>::ConstVec b,
      typename TTypes<T>::Matrix xh, typename TTypes<T>::ConstMatrix i,
      typename TTypes<T>::ConstMatrix cs, typename TTypes<T>::ConstMatrix f,
      typename TTypes<T>::ConstMatrix o, typename TTypes<T>::ConstMatrix ci,
      typename TTypes<T>::ConstMatrix co,
      typename TTypes<T>::ConstMatrix cs_grad,
      typename TTypes<T>::ConstMatrix h_grad, typename TTypes<T>::Matrix do_,
      typename TTypes<T>::Matrix dcs, typename TTypes<T>::Matrix dci,
      typename TTypes<T>::Matrix df, typename TTypes<T>::Matrix di,
      typename TTypes<T>::Matrix dgates,
      typename TTypes<T>::Matrix cs_prev_grad,
      typename TTypes<T>::Matrix h_prev_grad,
      typename TTypes<T>::Matrix xh_grad, typename TTypes<T>::Matrix x_grad,
      typename TTypes<T>::Matrix w_grad, typename TTypes<T>::Vec wci_grad,
      typename TTypes<T>::Vec wcf_grad, typename TTypes<T>::Vec wco_grad,
      typename TTypes<T>::Vec b_grad) {
    // do[t] = sigm'(o[t]) .* dh[t] .* co[t]
    do_.device(d) = o * (o.constant(T(1)) - o) * h_grad * co;

    // dcs[t] += tanh'(cs[t]) .* dh[t] .* o[t] + dcs[t + 1] .* f[t + 1]
    dcs.device(d) = (co.constant(T(1)) - co * co) * h_grad * o + cs_grad;

    Eigen::array<Eigen::DenseIndex, 2> p_shape({1, cell_size_});
    Eigen::array<Eigen::DenseIndex, 2> p_broadcast_shape({batch_size_, 1});
    if (use_peephole) {
      dcs.device(d) =
          dcs + do_ * wco.reshape(p_shape).broadcast(p_broadcast_shape);
    }

    // dci[t] = tanh'(ci[t]) dcs[t] i[t]
    dci.device(d) = (ci.constant(T(1)) - ci * ci) * dcs * i;

    // df[t] = sigm'(f[t]) dcs[t] cs[t - 1]
    df.device(d) = f * (f.constant(T(1)) - f) * dcs * cs_prev;

    // di[t] = sigm'(i[t]) dcs[t] ci[t]
    di.device(d) = i * (i.constant(T(1)) - i) * dcs * ci;

    dgates.slice(gates_i_offsets(), cell_extents()).device(d) = di;
    dgates.slice(gates_c_offsets(gate_layout), cell_extents()).device(d) = dci;
    dgates.slice(gates_f_offsets(gate_layout), cell_extents()).device(d) = df;
    dgates.slice(gates_o_offsets(), cell_extents()).device(d) = do_;

    cs_prev_grad.device(d) = dcs * f;
    if (use_peephole) {
      cs_prev_grad.device(d) =
          cs_prev_grad +
          di * wci.reshape(p_shape).broadcast(p_broadcast_shape) +
          df * wcf.reshape(p_shape).broadcast(p_broadcast_shape);
    }

    // xh_grad.
    typename TTypes<T>::ConstMatrix const_dgates(dgates.data(),
                                                 dgates.dimensions());
    TensorBlasGemm<Device, T, USE_CUBLAS>::compute(
        ctx, d, false, true, 1.f, const_dgates, w, 0.f, xh_grad);

    // xh.
    xh.slice(xh_x_offsets(), xh_x_extents()).device(d) = x;
    xh.slice(xh_h_offsets(), xh_h_extents()).device(d) = h_prev;
    typename TTypes<T>::ConstMatrix const_xh(xh.data(), xh.dimensions());

    // x_grad.
    x_grad.device(d) = xh_grad.slice(xh_x_offsets(), xh_x_extents());
    h_prev_grad.device(d) = xh_grad.slice(xh_h_offsets(), xh_h_extents());

    // w_grad.
    TensorBlasGemm<Device, T, USE_CUBLAS>::compute(
        ctx, d, true, false, 1.f, const_xh, const_dgates, 1.f, w_grad);

    // b_grad.
    b_grad.device(d) += dgates.sum(Eigen::array<int, 1>({0}));

    if (use_peephole) {
      wci_grad.device(d) += (di * cs_prev).sum(Eigen::array<int, 1>({0}));
      wcf_grad.device(d) += (df * cs_prev).sum(Eigen::array<int, 1>({0}));
      wco_grad.device(d) += (do_ * cs).sum(Eigen::array<int, 1>({0}));
    }
  }
};

}  // namespace functor
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_KERNELS_RNN_LSTM_OPS_H_
