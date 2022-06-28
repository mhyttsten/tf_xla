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
class MHTracer_DTPStensorflowPScorePSkernelsPSmklPSmkl_slice_opDTcc {
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
   MHTracer_DTPStensorflowPScorePSkernelsPSmklPSmkl_slice_opDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSkernelsPSmklPSmkl_slice_opDTcc() {
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

// See docs in ../ops/array_ops.cc.

#ifdef INTEL_MKL

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "dnnl.hpp"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/kernels/ops_util.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/gtl/array_slice.h"
#include "tensorflow/core/platform/prefetch.h"
#include "tensorflow/core/util/mkl_util.h"
#ifdef DNNL_AARCH64_USE_ACL
#include "tensorflow/core/platform/mutex.h"
#endif

using dnnl::stream;

namespace tensorflow {

namespace {

gtl::InlinedVector<int64, 4> IntTensorToInt64Vec(const Tensor& tensor) {
  gtl::InlinedVector<int64, 4> out;
  if (tensor.dtype() == DT_INT32) {
    for (int64 i = 0; i < tensor.NumElements(); ++i) {
      out.push_back(tensor.flat<int32>()(i));
    }
  } else if (tensor.dtype() == DT_INT64) {
    for (int64 i = 0; i < tensor.NumElements(); ++i) {
      out.push_back(tensor.flat<int64_t>()(i));
    }
  } else {
    // tensor must be either int32 or int64
    DCHECK(false);
  }
  return out;
}

}  // namespace

typedef Eigen::ThreadPoolDevice CPUDevice;

// A version of SharedValidation (slice_op.h) written for input that is in
// either Mkl layout or Tensorflow layout. A shared code to validate input
// shapes and check for identity, which is not dependent on the type of T.
// We do this to reduce code size by not duplicating all this for all T
// (float, double, int32, etc.)
static void ValidateMklInputs(OpKernelContext* context, bool* is_identity,
                              gtl::InlinedVector<int64, 4>* begin,
                              gtl::InlinedVector<int64, 4>* size) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSmklPSmkl_slice_opDTcc mht_0(mht_0_v, 237, "", "./tensorflow/core/kernels/mkl/mkl_slice_op.cc", "ValidateMklInputs");

  const int kInputTensorIndex = 0;
  const int kInputBeginIndex = 1;
  const int kInputSizeIndex = 2;
  const Tensor& input = MklGetInput(context, kInputTensorIndex);
  const Tensor& begin_tensor = MklGetInput(context, kInputBeginIndex);
  const Tensor& size_tensor = MklGetInput(context, kInputSizeIndex);

  MklDnnShape input_mkl_shape, begin_mkl_shape, size_mkl_shape;
  GetMklShape(context, kInputTensorIndex, &input_mkl_shape);
  GetMklShape(context, kInputBeginIndex, &begin_mkl_shape);
  GetMklShape(context, kInputSizeIndex, &size_mkl_shape);

  // Begin and size tensors cannot be in MklDnn layout.
  DCHECK_EQ(begin_mkl_shape.IsMklTensor(), false);
  DCHECK_EQ(size_mkl_shape.IsMklTensor(), false);

  TensorShape input_tf_shape = input_mkl_shape.IsMklTensor()
                                   ? input_mkl_shape.GetTfShape()
                                   : input.shape();
  const int input_dims = input_tf_shape.dims();

  OP_REQUIRES(
      context,
      TensorShapeUtils::IsVector(begin_tensor.shape()) &&
          TensorShapeUtils::IsVector(size_tensor.shape()) &&
          begin_tensor.NumElements() == input_dims &&
          size_tensor.NumElements() == input_dims,
      errors::InvalidArgument(
          "Expected begin and size arguments to be 1-D tensors of size ",
          input_dims, ", but got shapes ", begin_tensor.shape().DebugString(),
          " and ", size_tensor.shape().DebugString(), " instead."));

  *begin = IntTensorToInt64Vec(begin_tensor);
  *size = IntTensorToInt64Vec(size_tensor);
  for (int i = 0; i < input_dims; ++i) {
    if ((*size)[i] == -1) {
      // A size[i] of -1 means "all elements from begin[i] to dim_size(i)".
      (*size)[i] = input_tf_shape.dim_size(i) - (*begin)[i];
    }
  }

  *is_identity = true;
  for (int i = 0; i < input_dims; ++i) {
    int64 b = (*begin)[i];
    int64 s = (*size)[i];
    if (input_tf_shape.dim_size(i) == 0) {
      OP_REQUIRES(
          context, b == 0 && s == 0,
          errors::InvalidArgument("Expected begin[", i, "] == 0 (got ", b,
                                  ") and size[", i, "] == 0 ", "(got ", s,
                                  ") when ", "input.dim_size(", i, ") == 0"));
    } else {
      OP_REQUIRES(context, 0 <= b && b <= input_tf_shape.dim_size(i),
                  errors::InvalidArgument("Expected begin[", i, "] in [0, ",
                                          input_tf_shape.dim_size(i),
                                          "], but got ", b));
      OP_REQUIRES(context, 0 <= s && b + s <= input_tf_shape.dim_size(i),
                  errors::InvalidArgument("Expected size[", i, "] in [0, ",
                                          input_tf_shape.dim_size(i) - b,
                                          "], but ", "got ", s));
    }
    const bool take_all = (b == 0) && (s == input_tf_shape.dim_size(i));
    (*is_identity) &= take_all;
  }
}

// A version of SharedSliceCommonCases function written for input tensor
// that may be in MklDnn layout or in Tensorflow layout.
template <typename T>
static void CheckCommonCasesForMklInputs(OpKernelContext* context,
                                         gtl::InlinedVector<int64, 4>* begin,
                                         gtl::InlinedVector<int64, 4>* size,
                                         bool* done) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSmklPSmkl_slice_opDTcc mht_1(mht_1_v, 313, "", "./tensorflow/core/kernels/mkl/mkl_slice_op.cc", "CheckCommonCasesForMklInputs");

  bool is_identity = true;
  *done = false;

  ValidateMklInputs(context, &is_identity, begin, size);
  if (!context->status().ok()) return;

  const Tensor& input = MklGetInput(context, 0);
  MklDnnShape input_mkl_shape;
  GetMklShape(context, 0, &input_mkl_shape);

  if (is_identity) {
    VLOG(1) << "Slice identity";
    context->set_output(0, input);
    // Mkl metadata tensor in this case can just be forwarded from input to
    // output.
    AllocateOutputSetMklShape(context, 0, input_mkl_shape);
    *done = true;
  }
}

// This structure aggregates multiple inputs to Slice methods.
struct MklSliceParams {
  // Parameters from & to represents memory pointing to reorder.
  const memory* from;
  const memory* to;

  // Parameters begin_dims & size_dims represents offset and length
  // passed to view primitive.
  memory::dims begin_dims;
  memory::dims size_dims;

  MklSliceParams(const memory* from, const memory* to, memory::dims begin_dims,
                 memory::dims size_dims)
      : from(from), to(to), begin_dims(begin_dims), size_dims(size_dims) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSmklPSmkl_slice_opDTcc mht_2(mht_2_v, 350, "", "./tensorflow/core/kernels/mkl/mkl_slice_op.cc", "MklSliceParams");
}
};

// This implements the shared interface of Slice reorders.
template <typename T>
class MklSlicePrimitive : public MklPrimitive {
 public:
  explicit MklSlicePrimitive(const MklSliceParams& sliceParams)
      : MklPrimitive(engine(engine::kind::cpu, 0)) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSmklPSmkl_slice_opDTcc mht_3(mht_3_v, 361, "", "./tensorflow/core/kernels/mkl/mkl_slice_op.cc", "MklSlicePrimitive");

    Setup(sliceParams);
  }

  ~MklSlicePrimitive() {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSmklPSmkl_slice_opDTcc mht_4(mht_4_v, 368, "", "./tensorflow/core/kernels/mkl/mkl_slice_op.cc", "~MklSlicePrimitive");
}

  void Execute(const MklSliceParams& sliceParams,
               std::shared_ptr<stream> slice_stream) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSmklPSmkl_slice_opDTcc mht_5(mht_5_v, 374, "", "./tensorflow/core/kernels/mkl/mkl_slice_op.cc", "Execute");

#ifdef DNNL_AARCH64_USE_ACL
    mutex_lock lock(primitive_execution_mu_);
#endif
#ifndef ENABLE_ONEDNN_OPENMP
    context_.src_mem->set_data_handle(sliceParams.from->get_data_handle(),
                                      *slice_stream);
    context_.dst_mem->set_data_handle(sliceParams.to->get_data_handle(),
                                      *slice_stream);
#else
    context_.src_mem->set_data_handle(sliceParams.from->get_data_handle());
    context_.dst_mem->set_data_handle(sliceParams.to->get_data_handle());
#endif  // !ENABLE_ONEDNN_OPENMP

    execute_primitives(context_.slice_primitives, slice_stream,
                       context_.slice_primitives_args);

    // We should set it back to DummyData so as to make the primitive
    // in cache pool stateless. Otherwise, if the result for previous
    // iteration is kept, problems of current iteration won't be
    // thrown immediately, and wrong data would be reused.
    context_.src_mem->set_data_handle(DummyData);
    context_.dst_mem->set_data_handle(DummyData);
    return;
  }

  std::shared_ptr<primitive> GetPrimitive() { return context_.reorder_prim; }

 private:
  struct SliceContext {
    std::shared_ptr<dnnl::memory> src_mem;
    std::shared_ptr<dnnl::memory> dst_mem;
    std::shared_ptr<primitive> reorder_prim;
    std::shared_ptr<reorder::primitive_desc> reorder_pd;
    std::shared_ptr<dnnl::stream> slice_stream;
    std::vector<dnnl::primitive> slice_primitives;
    std::shared_ptr<dnnl::memory> src_sub_mem;
    std::vector<std::unordered_map<int, memory>> slice_primitives_args;
    SliceContext()
        : src_mem(nullptr), dst_mem(nullptr), reorder_prim(nullptr) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSmklPSmkl_slice_opDTcc mht_6(mht_6_v, 416, "", "./tensorflow/core/kernels/mkl/mkl_slice_op.cc", "SliceContext");
}
  } context_;

  void Setup(const MklSliceParams& sliceParams) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSmklPSmkl_slice_opDTcc mht_7(mht_7_v, 422, "", "./tensorflow/core/kernels/mkl/mkl_slice_op.cc", "Setup");

    // Actually, DummyData will not be used in computation,
    // because the real data will be filled before execution.
    context_.src_mem.reset(
        new memory(sliceParams.from->get_desc(), cpu_engine_, DummyData));
    context_.dst_mem.reset(
        new memory(sliceParams.to->get_desc(), cpu_engine_, DummyData));

    auto src_sub_desc = context_.src_mem->get_desc().submemory_desc(
        sliceParams.size_dims, sliceParams.begin_dims);
    context_.src_sub_mem.reset(new memory(src_sub_desc, cpu_engine_, nullptr));
    context_.reorder_pd = std::make_shared<reorder::primitive_desc>(
        reorder::primitive_desc(*context_.src_sub_mem, *context_.dst_mem));
    context_.reorder_prim =
        std::make_shared<dnnl::reorder>(reorder(*context_.reorder_pd));

    context_.slice_primitives_args.push_back(
        {{DNNL_ARG_SRC, *context_.src_mem}, {DNNL_ARG_DST, *context_.dst_mem}});
    context_.slice_primitives.push_back(*context_.reorder_prim);
  }

#ifdef DNNL_AARCH64_USE_ACL
  mutex primitive_execution_mu_;
#endif
};

template <typename T>
class MklSlicePrimitiveFactory : public MklPrimitiveFactory<T> {
 public:
  static MklSlicePrimitive<T>* Get(const MklSliceParams& sliceParams) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSmklPSmkl_slice_opDTcc mht_8(mht_8_v, 454, "", "./tensorflow/core/kernels/mkl/mkl_slice_op.cc", "Get");

    auto reorderPrim = static_cast<MklSlicePrimitive<T>*>(
        MklSlicePrimitiveFactory<T>::GetInstance().GetReorder(sliceParams));
    if (reorderPrim == nullptr) {
      reorderPrim = new MklSlicePrimitive<T>(sliceParams);
      MklSlicePrimitiveFactory<T>::GetInstance().SetReorder(sliceParams,
                                                            reorderPrim);
    }
    return reorderPrim;
  }

  static MklSlicePrimitiveFactory& GetInstance() {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSmklPSmkl_slice_opDTcc mht_9(mht_9_v, 468, "", "./tensorflow/core/kernels/mkl/mkl_slice_op.cc", "GetInstance");

    static MklSlicePrimitiveFactory instance_;
    return instance_;
  }

 private:
  MklSlicePrimitiveFactory() {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSmklPSmkl_slice_opDTcc mht_10(mht_10_v, 477, "", "./tensorflow/core/kernels/mkl/mkl_slice_op.cc", "MklSlicePrimitiveFactory");
}
  ~MklSlicePrimitiveFactory() {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSmklPSmkl_slice_opDTcc mht_11(mht_11_v, 481, "", "./tensorflow/core/kernels/mkl/mkl_slice_op.cc", "~MklSlicePrimitiveFactory");
}

  static string CreateKey(const MklSliceParams& sliceParams) {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSmklPSmkl_slice_opDTcc mht_12(mht_12_v, 486, "", "./tensorflow/core/kernels/mkl/mkl_slice_op.cc", "CreateKey");

    string prefix = "reorder";
    FactoryKeyCreator key_creator;
    auto const& from_desc = sliceParams.from->get_desc().data;
    auto const& to_desc = sliceParams.to->get_desc().data;
    memory::dims from_dims(from_desc.dims, &from_desc.dims[from_desc.ndims]);
    memory::dims to_dims(to_desc.dims, &to_desc.dims[to_desc.ndims]);

    auto from_strides = from_desc.format_desc.blocking.strides;
    auto to_strides = to_desc.format_desc.blocking.strides;
    memory::dims from_strides_outer_blocks(from_strides,
                                           &from_strides[from_desc.ndims]);
    memory::dims to_strides_outer_blocks(to_strides,
                                         &to_strides[to_desc.ndims]);

    key_creator.AddAsKey(prefix);
    key_creator.AddAsKey(static_cast<int>(from_desc.data_type));
    key_creator.AddAsKey(from_dims);
    key_creator.AddAsKey(from_strides_outer_blocks);
    key_creator.AddAsKey(static_cast<int>(to_desc.data_type));
    key_creator.AddAsKey(to_dims);
    key_creator.AddAsKey(to_strides_outer_blocks);
    key_creator.AddAsKey(sliceParams.begin_dims);
    key_creator.AddAsKey(sliceParams.size_dims);
    return key_creator.GetKey();
  }

  MklPrimitive* GetReorder(const MklSliceParams& sliceParams) {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSmklPSmkl_slice_opDTcc mht_13(mht_13_v, 516, "", "./tensorflow/core/kernels/mkl/mkl_slice_op.cc", "GetReorder");

    string key = CreateKey(sliceParams);
    return this->GetOp(key);
  }

  void SetReorder(const MklSliceParams& sliceParams, MklPrimitive* op) {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSmklPSmkl_slice_opDTcc mht_14(mht_14_v, 524, "", "./tensorflow/core/kernels/mkl/mkl_slice_op.cc", "SetReorder");

    string key = CreateKey(sliceParams);
    this->SetOp(key, op);
  }
};

// oneDNN implementation of Slice
template <typename Device, typename T>
class MklSliceOp : public OpKernel {
 public:
  explicit MklSliceOp(OpKernelConstruction* context) : OpKernel(context) {
   std::vector<std::string> mht_15_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSmklPSmkl_slice_opDTcc mht_15(mht_15_v, 537, "", "./tensorflow/core/kernels/mkl/mkl_slice_op.cc", "MklSliceOp");
}

  ~MklSliceOp() {
   std::vector<std::string> mht_16_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSmklPSmkl_slice_opDTcc mht_16(mht_16_v, 542, "", "./tensorflow/core/kernels/mkl/mkl_slice_op.cc", "~MklSliceOp");
}

  void Compute(OpKernelContext* context) override {
   std::vector<std::string> mht_17_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSmklPSmkl_slice_opDTcc mht_17(mht_17_v, 547, "", "./tensorflow/core/kernels/mkl/mkl_slice_op.cc", "Compute");

    gtl::InlinedVector<int64, 4> begin;
    gtl::InlinedVector<int64, 4> size;
    bool done = false;

    CheckCommonCasesForMklInputs<T>(context, &begin, &size, &done);

    if (!context->status().ok() || done == true) return;

    // oneDNN supports more than 8 dimension and less than 12 dimension tensor.
    // But we are mimicking functionality of Eigen Slice op for CPU.
    if (begin.size() >= 8) {
      OP_REQUIRES(
          context, false,
          errors::Unimplemented("MklSliceOp : Unhandled input dimensions"));
    }

    ComputeMklSlice(context, begin, size);
  }

 private:
  // Slice op implemented using oneDNN APIs.
  void ComputeMklSlice(OpKernelContext* context,
                       const gtl::InlinedVector<int64, 4>& begin,
                       const gtl::InlinedVector<int64, 4>& size) {
   std::vector<std::string> mht_18_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSmklPSmkl_slice_opDTcc mht_18(mht_18_v, 574, "", "./tensorflow/core/kernels/mkl/mkl_slice_op.cc", "ComputeMklSlice");

    try {
      // oneDNN API usage below is guided by description at:
      //  https://github.com/01org/mkl-dnn/issues/69
      //
      // Relevant part of the description is copied below:
      //
      // Let's say you want to copy a part of memory into another buffer (and
      // probably change the format). Then your steps are:
      //
      // 1. create memory primitive descriptor in_mem_pd and memory primitive
      //    in_mem_p for the entire source data. create view primitive
      //    descriptor in_submem_pd based on in_mem_pd, initial offsets,
      //    and sub-sizes
      // 2. create memory primitive descriptor out_mem_pd and memory primitive
      //    out_mem_p for the output (the logical sizes should match sub-sizes
      //    used in step 1, but the format might be arbitrary)
      // 3. create reorder primitive descriptor reorder_pd based on in_submem_pd
      //    and out_mem_pd. create reorder primitive itself based on reorder_pd,
      //    in_mem_p, and out_mem_p.
      //
      // Please notice that there is no view primitive. There is only view
      // primitive descriptor. And the reorder uses source memory as input but
      // traverses it according to a view in_submem_pd.

      auto cpu_engine = engine(engine::kind::cpu, 0);
      MklDnnData<T> src(&cpu_engine);
      MklDnnData<T> output(&cpu_engine);

      // Populate offsets and sizes in memory::dims format based on vector.
      memory::dims begin_dims = {};
      begin_dims.resize(begin.size());
      for (size_t i = 0; i < begin.size(); ++i) begin_dims[i] = begin[i];
      memory::dims size_dims = {};
      bool empty = false;
      size_dims.resize(size.size());
      for (size_t i = 0; i < size.size(); ++i) {
        size_dims[i] = size[i];
        if (size_dims[i] == 0) empty = true;
      }

      Tensor* output_tensor = nullptr;
      MklDnnShape output_mkl_shape;

      // If no dimension is selected in slice, the result should be empty.
      // Just return an empty output tensor, and a dummy Mkl-shape tensor.
      if (empty) {  // for empty dims
        auto shape_to = MklDnnDimsToTFShape(size_dims);
        AllocateOutputSetMklShape(context, 0, &output_tensor, shape_to,
                                  output_mkl_shape);
        return;
      }

      // Step 1 (as per above description) - Create memory for user data.
      // We use blocked format here to describe input tensor.
      const Tensor& input_tensor = MklGetInput(context, 0);
      memory::dims input_dims, input_strides;
      MklDnnShape input_mkl_shape;
      GetMklShape(context, 0, &input_mkl_shape);

      if (input_mkl_shape.IsMklTensor()) {
        auto input_mkl_format = input_mkl_shape.GetTfDataFormat();
        auto input_tf_format = MklDnnDataFormatToTFDataFormat(input_mkl_format);

        bool is_slice2d = (input_mkl_shape.GetDimension() == 4);
        begin_dims = is_slice2d
                         ? MklDnnDimsInNCHW(begin_dims, input_tf_format)
                         : MklDnnDimsInNCDHW(begin_dims, input_tf_format);
        size_dims = is_slice2d ? MklDnnDimsInNCHW(size_dims, input_tf_format)
                               : MklDnnDimsInNCDHW(size_dims, input_tf_format);
        auto input_md = input_mkl_shape.GetMklLayout();
        src.SetUsrMem(input_md, &input_tensor);

        // Handle data format safely, change them to block format.
        // Compute parameters of reorder primitive first.
        input_dims = input_mkl_shape.GetSizesAsMklDnnDims();
        input_strides = CalculateTFStrides(input_dims);
      } else {
        // Initialize input dimensions and strides to be used when input is not
        // in MklDnn layout.
        input_dims = TFShapeToMklDnnDims(input_tensor.shape());
        input_strides = CalculateTFStrides(input_dims);
        // Create input memory descriptor.
        auto input_md =
            MklDnnData<T>::CreateBlockedMemDesc(input_dims, input_strides);
        src.SetUsrMem(input_md, &input_tensor);
      }

      // If format not equal to block format, execute reorder.
      // Or else do nothing for it.
      auto op_md =
          MklDnnData<T>::CreateBlockedMemDesc(input_dims, input_strides);
      src.CheckReorderToOpMem(op_md, cpu_engine, context);

      // Step 2 - Create memory for output.
      auto output_strides = CalculateTFStrides(size_dims);
      auto output_md =
          MklDnnData<T>::CreateBlockedMemDesc(size_dims, output_strides);
      auto output_pd = output_md;
      AllocateOutputTensor(context, input_mkl_shape, &output_pd, size_dims,
                           &output_tensor, &output_mkl_shape);
      DCHECK(output_tensor);
      DCHECK_EQ(input_mkl_shape.IsMklTensor(), output_mkl_shape.IsMklTensor());
      output.SetUsrMem(output_md, output_tensor);

      // Step 3 - create reorder primitive.
      MklSliceParams sliceParams(&src.GetOpMem(), output.GetUsrMem(),
                                 begin_dims, size_dims);
      MklSlicePrimitive<T>* reorder_prim =
          MklSlicePrimitiveFactory<T>::Get(sliceParams);
      // Execute slice reorder.
      std::shared_ptr<stream> slice_stream;
      MklDnnThreadPool eigen_tp(context);
      slice_stream.reset(CreateStream(&eigen_tp, reorder_prim->GetEngine()));
      reorder_prim->Execute(sliceParams, slice_stream);
    } catch (dnnl::error& e) {
      string error_msg = "Status: " + std::to_string(e.status) +
                         ", message: " + string(e.message) + ", in file " +
                         string(__FILE__) + ":" + std::to_string(__LINE__);
      OP_REQUIRES_OK(
          context,
          errors::Aborted("Operation received an exception:", error_msg));
    }
  }

 private:
  void AllocateOutputTensor(OpKernelContext* context,
                            const MklDnnShape& input_mkl_shape,
                            memory::desc* output_pd,
                            const memory::dims& output_dims,
                            Tensor** output_tensor,
                            MklDnnShape* output_mkl_shape) {
   std::vector<std::string> mht_19_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSmklPSmkl_slice_opDTcc mht_19(mht_19_v, 708, "", "./tensorflow/core/kernels/mkl/mkl_slice_op.cc", "AllocateOutputTensor");

    DCHECK(output_tensor);
    DCHECK(output_mkl_shape);

    TensorShape output_tf_shape;

    if (input_mkl_shape.IsMklTensor()) {
      // Since input tensor is in Mkl layout, output tensor will be in Mkl
      // layout.

      // Allocate shape of Mkl tensor.
      output_mkl_shape->SetMklTensor(true);
      output_mkl_shape->SetMklLayout(output_pd);
      output_mkl_shape->SetElemType(MklDnnType<T>());
      output_mkl_shape->SetTfLayout(input_mkl_shape.GetDimension(), output_dims,
                                    input_mkl_shape.GetTfDataFormat());

      output_tf_shape.AddDim(output_pd->get_size() / sizeof(T));
    } else {
      // If input is not in Mkl layout, then output won't be in Mkl layout.
      output_mkl_shape->SetMklTensor(false);
      output_tf_shape = MklDnnDimsToTFShape(output_dims);
    }

    AllocateOutputSetMklShape(context, 0, output_tensor, output_tf_shape,
                              *output_mkl_shape);
  }
};

// oneDNN Slice registration
#define REGISTER_MKL_SLICE(type)                               \
  REGISTER_KERNEL_BUILDER(                                     \
      Name("_MklSlice")                                        \
          .Device(DEVICE_CPU)                                  \
          .TypeConstraint<type>("T")                           \
          .HostMemory("begin")                                 \
          .HostMemory("size")                                  \
          .Label(mkl_op_registry::kMklLayoutDependentOpLabel), \
      MklSliceOp<CPUDevice, type>);

TF_CALL_float(REGISTER_MKL_SLICE);
TF_CALL_bfloat16(REGISTER_MKL_SLICE);
#undef REGISTER_MKL_SLICE

}  // namespace tensorflow

#endif  // INTEL_MKL
