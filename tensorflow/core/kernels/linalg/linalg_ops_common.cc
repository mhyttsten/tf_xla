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
class MHTracer_DTPStensorflowPScorePSkernelsPSlinalgPSlinalg_ops_commonDTcc {
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
   MHTracer_DTPStensorflowPScorePSkernelsPSlinalgPSlinalg_ops_commonDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSkernelsPSlinalgPSlinalg_ops_commonDTcc() {
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

#include "tensorflow/core/kernels/linalg/linalg_ops_common.h"

#include <utility>

#include "third_party/eigen3/Eigen/Core"
#include "tensorflow/core/framework/device_base.h"
#include "tensorflow/core/framework/kernel_def_builder.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {

// static
template <class InputScalar, class OutputScalar>
void LinearAlgebraOp<InputScalar, OutputScalar>::ValidateSingleMatrix(
    OpKernelContext* context, const TensorShapes& input_matrix_shapes) {
  OP_REQUIRES(context, input_matrix_shapes.size() == 1,
              errors::InvalidArgument("Expected a single input matrix, got %d.",
                                      input_matrix_shapes.size()));
  OP_REQUIRES(context, TensorShapeUtils::IsMatrix(input_matrix_shapes[0]),
              errors::InvalidArgument("Input must be a matrix."));
}

// static
template <class InputScalar, class OutputScalar>
void LinearAlgebraOp<InputScalar, OutputScalar>::ValidateSingleSquareMatrix(
    OpKernelContext* context, const TensorShapes& input_matrix_shapes) {
  OP_REQUIRES(context, input_matrix_shapes.size() == 1,
              errors::InvalidArgument("Expected a single input matrix, got %d.",
                                      input_matrix_shapes.size()));
  OP_REQUIRES(context, TensorShapeUtils::IsSquareMatrix(input_matrix_shapes[0]),
              errors::InvalidArgument("Input matrix must be square."));
}

// static
template <class InputScalar, class OutputScalar>
void LinearAlgebraOp<InputScalar, OutputScalar>::ValidateSolver(
    OpKernelContext* context, const TensorShapes& input_matrix_shapes) {
  OP_REQUIRES(context, input_matrix_shapes.size() == 2,
              errors::InvalidArgument("Expected two input matrices, got %d.",
                                      input_matrix_shapes.size()));
  OP_REQUIRES(context, TensorShapeUtils::IsMatrix(input_matrix_shapes[0]),
              errors::InvalidArgument("First input (lhs) must be a matrix."));
  OP_REQUIRES(context, TensorShapeUtils::IsMatrix(input_matrix_shapes[1]),
              errors::InvalidArgument("Second input (rhs) must be a matrix."));
  OP_REQUIRES(
      context,
      input_matrix_shapes[0].dim_size(0) == input_matrix_shapes[1].dim_size(0),
      errors::InvalidArgument("Input matrix and rhs are incompatible."));
}

// static
template <class InputScalar, class OutputScalar>
void LinearAlgebraOp<InputScalar, OutputScalar>::ValidateSquareSolver(
    OpKernelContext* context, const TensorShapes& input_matrix_shapes) {
  OP_REQUIRES(context, input_matrix_shapes.size() == 2,
              errors::InvalidArgument("Expected two input matrices, got %d.",
                                      input_matrix_shapes.size()));
  OP_REQUIRES(
      context, TensorShapeUtils::IsSquareMatrix(input_matrix_shapes[0]),
      errors::InvalidArgument("First input (lhs) must be a square matrix."));
  OP_REQUIRES(context, TensorShapeUtils::IsMatrix(input_matrix_shapes[1]),
              errors::InvalidArgument("Second input (rhs) must be a matrix."));
  OP_REQUIRES(
      context,
      input_matrix_shapes[0].dim_size(0) == input_matrix_shapes[1].dim_size(0),
      errors::InvalidArgument("Input matrix and rhs are incompatible."));
}

template <class InputScalar, class OutputScalar>
void LinearAlgebraOp<InputScalar, OutputScalar>::Compute(
    OpKernelContext* context) {
  TensorInputs inputs;
  TensorShapes input_matrix_shapes;
  TensorShape batch_shape;
  AnalyzeInputs(context, &inputs, &input_matrix_shapes, &batch_shape);
  if (!context->status().ok()) return;

  TensorShapes output_matrix_shapes;
  TensorOutputs outputs;
  PrepareOutputs(context, input_matrix_shapes, batch_shape, &outputs,
                 &output_matrix_shapes);
  if (!context->status().ok()) return;

  // Process the individual matrix problems in parallel using a threadpool.
  auto shard = [this, &inputs, &input_matrix_shapes, &outputs,
                &output_matrix_shapes, context](int64_t begin, int64_t end) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSlinalgPSlinalg_ops_commonDTcc mht_0(mht_0_v, 274, "", "./tensorflow/core/kernels/linalg/linalg_ops_common.cc", "lambda");

    for (int64_t i = begin; i < end; ++i) {
      ComputeTensorSlice(context, i, inputs, input_matrix_shapes, outputs,
                         output_matrix_shapes);
    }
  };
  auto worker_threads = *(context->device()->tensorflow_cpu_worker_threads());
  Shard(worker_threads.num_threads, worker_threads.workers,
        batch_shape.num_elements(), GetCostPerUnit(input_matrix_shapes), shard);
}

template <class InputScalar, class OutputScalar>
void LinearAlgebraOp<InputScalar, OutputScalar>::AnalyzeInputs(
    OpKernelContext* context, TensorInputs* inputs,
    TensorShapes* input_matrix_shapes, TensorShape* batch_shape) {
  int input_rank = -1;
  for (int i = 0; i < NumMatrixInputs(context); ++i) {
    const Tensor& in = context->input(i);
    if (i == 0) {
      input_rank = in.dims();
      OP_REQUIRES(
          context, input_rank >= 2,
          errors::InvalidArgument("Input tensor ", i,
                                  " must have rank >= 2, got ", input_rank));
      // If the tensor rank is greater than 2, we consider the inner-most
      // dimensions as matrices, and loop over all the other outer ("batch")
      // dimensions to compute the results.
      for (int dim = 0; dim < input_rank - 2; ++dim) {
        batch_shape->AddDim(in.dim_size(dim));
      }
    } else {
      // Make sure that all inputs have the same rank and outer dimensions.
      OP_REQUIRES(context, input_rank == in.dims(),
                  errors::InvalidArgument(
                      "All input tensors must have the same rank."));
      for (int dim = 0; dim < input_rank - 2; ++dim) {
        OP_REQUIRES(
            context, in.dim_size(dim) == batch_shape->dim_size(dim),
            errors::InvalidArgument(
                "All input tensors must have the same outer dimensions."));
      }
    }

    const int row_dimension = input_rank - 2;
    const int col_dimension = input_rank - 1;
    const int64_t num_rows = in.dim_size(row_dimension);
    const int64_t num_cols = in.dim_size(col_dimension);
    input_matrix_shapes->emplace_back(
        std::initializer_list<int64_t>({num_rows, num_cols}));
    inputs->emplace_back(&in);
  }
  // Have the derived class validate that the inputs are as expected.
  ValidateInputMatrixShapes(context, *input_matrix_shapes);
}

template <class InputScalar, class OutputScalar>
void LinearAlgebraOp<InputScalar, OutputScalar>::PrepareOutputs(
    OpKernelContext* context, const TensorShapes& input_matrix_shapes,
    const TensorShape& batch_shape, TensorOutputs* outputs,
    TensorShapes* output_matrix_shapes) {
  // Get shape for each of the matrix outputs produced by the derived class.
  *output_matrix_shapes = GetOutputMatrixShapes(input_matrix_shapes);
  const int num_outputs = output_matrix_shapes->size();

  // Make sure the number of op outputs is what the derived class expects.
  OP_REQUIRES(
      context, num_outputs <= context->num_outputs(),
      errors::Internal(
          "Derived class expected more outputs (%d) that the op has (%d).",
          num_outputs, context->num_outputs()));

  // Allocate outputs.
  std::set<int> unused_inputs;
  for (int input_idx = 0; input_idx < context->num_inputs(); ++input_idx) {
    unused_inputs.insert(input_idx);
  }
  for (int output_idx = 0; output_idx < context->num_outputs(); ++output_idx) {
    TensorShape output_tensor_shape({});
    if (output_idx < num_outputs) {
      // This output is used, set up output shape and allocate it.
      const TensorShape& output_matrix_shape =
          output_matrix_shapes->at(output_idx);
      OP_REQUIRES(context, output_matrix_shape.dims() <= 2,
                  errors::InvalidArgument(
                      "Rank of matrix output no. %d must be 0, 1 or 2, got %d.",
                      output_idx, output_matrix_shape.dims()));

      // The final output has the shape of the outer batch dimensions
      // concatenated with the output_matrix_shape (if the output is not
      // scalar).
      output_tensor_shape = batch_shape;
      output_tensor_shape.AppendShape(output_matrix_shape);
    }
    Tensor* out = nullptr;
    // See if there is an input buffer we can reuse for this output.
    bool reused_input = false;
    if (EnableInputForwarding()) {
      for (int input_idx : unused_inputs) {
        if (context->forward_input_to_output_with_shape(
                input_idx, output_idx, output_tensor_shape, &out)) {
          reused_input = true;
          unused_inputs.erase(input_idx);
          break;
        }
      }
    }
    if (!reused_input) {
      OP_REQUIRES_OK(context, context->allocate_output(
                                  output_idx, output_tensor_shape, &out));
    }
    outputs->emplace_back(out);
  }
}

template <class InputScalar, class OutputScalar>
void LinearAlgebraOp<InputScalar, OutputScalar>::ComputeTensorSlice(
    OpKernelContext* context, int64_t matrix_index, const TensorInputs& inputs,
    const TensorShapes& input_matrix_shapes, const TensorOutputs& outputs,
    const TensorShapes& output_matrix_shapes) {
  InputConstMatrixMaps matrix_inputs;
  for (size_t i = 0; i < inputs.size(); ++i) {
    // TODO(kalakris): Handle alignment if possible. Eigen::Map is
    // unaligned by default.
    matrix_inputs.emplace_back(
        inputs[i]->flat<InputScalar>().data() +
            matrix_index * input_matrix_shapes[i].num_elements(),
        input_matrix_shapes[i].dim_size(0), input_matrix_shapes[i].dim_size(1));
  }

  OutputMatrixMaps matrix_outputs;
  for (size_t i = 0; i < output_matrix_shapes.size(); ++i) {
    // The output matrix shape may not be a matrix.
    int num_output_rows = output_matrix_shapes[i].dims() >= 1
                              ? output_matrix_shapes[i].dim_size(0)
                              : 1;
    int num_output_cols = output_matrix_shapes[i].dims() == 2
                              ? output_matrix_shapes[i].dim_size(1)
                              : 1;
    matrix_outputs.emplace_back(
        outputs[i]->flat<OutputScalar>().data() +
            matrix_index * output_matrix_shapes[i].num_elements(),
        num_output_rows, num_output_cols);
  }
  ComputeMatrix(context, matrix_inputs, &matrix_outputs);
}

// Explicitly instantiate LinearAlgebraOp for the scalar types we expect to use.
template class LinearAlgebraOp<float>;
template class LinearAlgebraOp<double>;
template class LinearAlgebraOp<complex64>;
template class LinearAlgebraOp<complex128>;
template class LinearAlgebraOp<float, complex64>;
template class LinearAlgebraOp<double, complex128>;

}  // namespace tensorflow
