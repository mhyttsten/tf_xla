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
class MHTracer_DTPStensorflowPScorePSkernelsPSmklPSmkl_einsum_opDTcc {
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
   MHTracer_DTPStensorflowPScorePSkernelsPSmklPSmkl_einsum_opDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSkernelsPSmklPSmkl_einsum_opDTcc() {
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
#ifdef INTEL_MKL
#define EIGEN_USE_THREADS
#define EIGEN_DONT_PARALLELIZE

#include "mkl_batch_matmul_helper.h"
#include "tensorflow/core/kernels/linalg/einsum_op_impl.h"

namespace tensorflow {

typedef Eigen::ThreadPoolDevice CPUDevice;

// MklEinsum Op is same as Eigen implementation of Einsum Op except,
// it calls MKL's BatchMatMul implementation instead of Eigen implementation
// for now, We are going to replace other Eigen Ops used here, with their
// equivalent MKL ops for better performance.

struct MklEinsumHelper {
  // Contracts the inputs along the last axis. (or the second last if the
  // corresponding value of swap_free_and_contract is true). The batch
  // dimensions are broadcast to the output shape.
  // TODO(intel-tf): BatchMatMul might devolve into a component-wise
  // multiplication when the matrix shape is [1,1]; in this case BatchMatMul
  // functor would be very inefficient. The functor should detect if this is the
  // case and perform componentwise multiplication functor instead.

  template <typename Device, typename T>
  static Status MKLContractOperands(
      OpKernelContext* ctx, absl::Span<const Tensor> inputs,
      absl::Span<const bool> swap_free_and_contract, Tensor* output) {
    if (inputs.size() == 1)
      return EinsumHelper::CopyFrom(inputs[0], inputs[0].shape(), output);
    MatMulBCast bcast(inputs[0].shape().dim_sizes(),
                      inputs[1].shape().dim_sizes());

    Tensor lhs = inputs[0];
    Tensor rhs = inputs[1];

    TensorShape output_shape = bcast.output_batch_shape();
    for (int i = 0; i < inputs.size(); ++i) {
      const int64 free_axis =
          inputs[i].dims() - (swap_free_and_contract[i] ? 1 : 2);
      output_shape.AddDim(inputs[i].dim_size(free_axis));
    }
    bool trans_x = swap_free_and_contract[0];
    bool trans_y = !swap_free_and_contract[1];
    TF_RETURN_IF_ERROR(
        ctx->allocate_temp(DataTypeToEnum<T>::value, output_shape, output));

    if (!(lhs.dims() >= 2))
      return errors::InvalidArgument("In[0] ndims must be >= 2: ", lhs.dims());

    if (!(rhs.dims() >= 2))
      return errors::InvalidArgument("In[1] ndims must be >= 2: ", rhs.dims());

    const auto ndims_lhs = lhs.dims();
    const auto ndims_rhs = rhs.dims();
    // In[0] and In[1] must have compatible batch dimensions
    if (!(bcast.IsValid()))
      return errors::InvalidArgument(
          "In[0] and In[1] must have compatible batch dimensions: ",
          lhs.shape().DebugString(), " vs. ", rhs.shape().DebugString());

    TensorShape out_shape = bcast.output_batch_shape();
    auto lhs_rows = lhs.dim_size(ndims_lhs - 2);
    auto lhs_cols = lhs.dim_size(ndims_lhs - 1);
    auto rhs_rows = rhs.dim_size(ndims_rhs - 2);
    auto rhs_cols = rhs.dim_size(ndims_rhs - 1);

    if (trans_x) std::swap(lhs_rows, lhs_cols);
    if (trans_y) std::swap(rhs_rows, rhs_cols);
    // lhs mismatch rhs shape: lhs_cols, " vs. ", rhs_rows
    if (lhs_cols != rhs_rows)
      return errors::InvalidArgument(
          "lhs mismatch rhs shape: ", lhs_cols, " vs. ", rhs_rows, ": ",
          lhs.shape().DebugString(), " ", rhs.shape().DebugString(), " ",
          trans_x, " ", trans_y);

    out_shape.AddDim(lhs_rows);
    out_shape.AddDim(rhs_cols);
    // The maximum number of dimensions for a tensor in DNNL is
    // DNNL_MAX_NDIMS = 12.
    if (!(out_shape.dims() <= DNNL_MAX_NDIMS))
      return errors::InvalidArgument(
          "Rank of output tensor must be <= 12, ", "but is ", out_shape.dims(),
          ". Current implementation supports upto ", "rank 12 tensors.");

    if (lhs.NumElements() == 0 || rhs.NumElements() == 0) {
      functor::SetZeroFunctor<Device, T> f;
      f(ctx->eigen_device<Device>(), output->flat<T>());
      return Status::OK();
    }

    // Compute parameters for DNNL matmul primitive.
    MklBatchMatMulHelper bmm;
    auto params = bmm.CreateMatMulParams(lhs.shape(), rhs.shape(), out_shape,
                                         trans_x, trans_y);

    // Create or retrieve matmul primitive from cache.
    MklMatMulPrimitive<T, T, T>* matmul_prim =
        MklMatMulPrimitiveFactory<T, T, T, T>::Get(
            *params, false /* value for do_not_cache */);

    UserScratchPad<unsigned char> scratch_pad;
    scratch_pad.AllocateSPTensor(matmul_prim, ctx);
    // Execute matmul primitive.
    std::shared_ptr<stream> cpu_stream;
    MklDnnThreadPool eigen_tp(ctx);
    cpu_stream.reset(CreateStream(&eigen_tp, matmul_prim->GetEngine()));

    matmul_prim->Execute(cpu_stream, lhs.flat<T>().data(), rhs.flat<T>().data(),
                         output->flat<T>().data(), scratch_pad.Get());

    Tensor output_reshaped;
    if (output->dims() != 3) {
      TF_RETURN_IF_ERROR(EinsumHelper::ReshapeToRank3(
          *output, bcast.output_batch_size(), &output_reshaped));
    }
    return Status::OK();
  }
};

template <typename Device, typename T>
class MklEinsum : public OpKernel {
 public:
  explicit MklEinsum(OpKernelConstruction* c) : OpKernel(c) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSmklPSmkl_einsum_opDTcc mht_0(mht_0_v, 308, "", "./tensorflow/core/kernels/mkl/mkl_einsum_op.cc", "MklEinsum");

    OP_REQUIRES_OK(c, c->GetAttr("equation", &mkl_equation_));
    OP_REQUIRES_OK(c, ParseEinsumEquation(
                          mkl_equation_, &mkl_input_labels_,
                          &mkl_output_labels_, &mkl_label_types_,
                          &mkl_input_label_counts_, &mkl_output_label_counts_,
                          &mkl_input_has_ellipsis_, &mkl_output_has_ellipsis_));
  }

  virtual ~MklEinsum() {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSmklPSmkl_einsum_opDTcc mht_1(mht_1_v, 320, "", "./tensorflow/core/kernels/mkl/mkl_einsum_op.cc", "~MklEinsum");
}

  void Compute(OpKernelContext* ctx) override {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSmklPSmkl_einsum_opDTcc mht_2(mht_2_v, 325, "", "./tensorflow/core/kernels/mkl/mkl_einsum_op.cc", "Compute");

    OpInputList inputs;
    OP_REQUIRES_OK(ctx, ctx->input_list("inputs", &inputs));

    OperandLabels input_labels(mkl_input_labels_);
    Labels output_labels(mkl_output_labels_);
    std::vector<EinsumDimensionType> label_types(mkl_label_types_);
    OperandLabelCounts input_label_counts(mkl_input_label_counts_);
    LabelCounts output_label_counts(mkl_output_label_counts_);
    LabelToDimSizes label_to_dim_sizes;

    OP_REQUIRES_OK(ctx, EinsumHelper::ProcessDimensions(
                            inputs, mkl_input_has_ellipsis_,
                            mkl_output_has_ellipsis_, &input_labels,
                            &output_labels, &label_types, &input_label_counts,
                            &output_label_counts, &label_to_dim_sizes));

    // The reduction phase (a) sums across reduction dimensions, (b) takes
    // generalized diagonals, and (c) reshapes it into shape
    //   [(broadcasting) batch shape] + [F,C]
    // where F and C denote the total (compacted) size of free and contract
    // dimensions, respectively.
    const int num_inputs = inputs.size();
    OperandLabels free_labels(num_inputs);
    gtl::InlinedVector<Tensor, 2> inputs_reduced(num_inputs);
    gtl::InlinedVector<bool, 2> swap_free_and_contract(num_inputs);
    for (int i = 0; i < num_inputs; ++i) {
      OP_REQUIRES_OK(ctx,
                     EinsumHelper::ReduceOperand<Device, T>(
                         ctx, inputs[i], label_types, input_label_counts[i],
                         &input_labels[i], &free_labels[i],
                         &swap_free_and_contract[i], &inputs_reduced[i]));
    }

    // After reduction, the inputs should be reshaped to Tensors suitable for
    // contraction. If num_inputs is 1, the reduced input is simply forwarded to
    // the output.
    Tensor contraction_output_reshaped;
    OP_REQUIRES_OK(ctx, MklEinsumHelper::MKLContractOperands<Device, T>(
                            ctx, inputs_reduced, swap_free_and_contract,
                            &contraction_output_reshaped));

    // Copy the batch labels from the contraction output. Recover the batch
    // shape, which may have been broadcasted.
    TensorShape result_shape = contraction_output_reshaped.shape();
    result_shape.RemoveLastDims(2);
    int num_labels = label_types.size();
    Labels result_labels;
    // All batch dimensions should be present in the contracted result. First
    // the broadcasting dimensions, then the named batch dimensions.
    for (int label = 0; label < num_labels; ++label) {
      if (label_types[label] == EinsumDimensionType::kBroadcasting)
        result_labels.push_back(label);
    }
    for (int label = 0; label < num_labels; ++label) {
      if (label_types[label] == EinsumDimensionType::kBatch)
        result_labels.push_back(label);
    }
    for (int i = 0; i < num_inputs; ++i) {
      for (int label : free_labels[i]) {
        result_labels.push_back(label);
        result_shape.AddDim(label_to_dim_sizes[label]);
      }
    }
    // Reshape the contraction (or reduction) result to its expanded shape:
    // [(broadcasted) batch shape] + [free shape 0] + [free shape 1].
    Tensor contraction_output;
    OP_REQUIRES_OK(
        ctx, EinsumHelper::CopyFrom(contraction_output_reshaped, result_shape,
                                    &contraction_output));
    // Inflate the output if necessary. (E.g. for the equation 'i->iii' which
    // may arise while computing gradient of a regular Einsum).
    // TODO(intel-tf): It's possible that Eigen's contract and inflate can be
    // chained here to avoid materializing an intermediate.
    Tensor output_inflated;
    OP_REQUIRES_OK(
        ctx, EinsumHelper::StrideOrInflate<Device, T>(
                 ctx, contraction_output, result_labels, output_label_counts,
                 true /* should_inflate */, &output_inflated));
    if (output_inflated.dims() > contraction_output.dims()) {
      // We inflated the output. Modify result labels accordingly.
      Labels inflated_labels;
      for (int label : result_labels) {
        inflated_labels.insert(inflated_labels.end(),
                               output_label_counts[label], label);
      }
      result_labels.swap(inflated_labels);
    }
    // Find the permutation to map the result labels to the output labels. Note
    // that both the result and the final output may have the repeated labels,
    // in which case the permutation preserves the left-to-right ordering.
    // E.g. if result labels are [0, 0, 1] and output is [0, l, 0] then the
    // permutation should be [0, 2, 1]. We also use the fact that repeated
    // labels in the result are adjacent to each other.
    std::vector<int> output_permutation(output_labels.size());
    std::vector<int> label_to_position(num_labels, -1);
    for (int i = 0; i < result_labels.size(); ++i) {
      // Remember the position of only the leftmost result label.
      if (label_to_position[result_labels[i]] == -1) {
        label_to_position[result_labels[i]] = i;
      }
    }
    for (int i = 0; i < output_labels.size(); ++i) {
      output_permutation[i] = label_to_position[output_labels[i]];
      // We have found the leftmost occurrence. The next one would be adjacent.
      label_to_position[output_labels[i]] += 1;
    }
    Tensor output;
    OP_REQUIRES_OK(ctx, EinsumHelper::TransposeOperand<Device, T>(
                            ctx, output_inflated, output_permutation, &output));
    ctx->set_output(0, output);
  }

 private:
  string mkl_equation_;
  OperandLabels mkl_input_labels_;
  Labels mkl_output_labels_;
  std::vector<EinsumDimensionType> mkl_label_types_;
  OperandLabelCounts mkl_input_label_counts_;
  LabelCounts mkl_output_label_counts_;
  gtl::InlinedVector<bool, 2> mkl_input_has_ellipsis_;
  bool mkl_output_has_ellipsis_ = false;
};

#define REGISTER_EINSUM_MKL(TYPE)                                             \
  REGISTER_KERNEL_BUILDER(Name("_MklEinsum")                                  \
                              .Device(DEVICE_CPU)                             \
                              .TypeConstraint<TYPE>("T")                      \
                              .Label(mkl_op_registry::kMklNameChangeOpLabel), \
                          MklEinsum<CPUDevice, TYPE>)
#ifdef ENABLE_MKL
TF_CALL_float(REGISTER_EINSUM_MKL);
TF_CALL_bfloat16(REGISTER_EINSUM_MKL);
#endif  // ENABLE_MKL
}  // namespace tensorflow
#endif  // INTEL_MKL
