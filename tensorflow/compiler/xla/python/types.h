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

#ifndef TENSORFLOW_COMPILER_XLA_PYTHON_TYPES_H_
#define TENSORFLOW_COMPILER_XLA_PYTHON_TYPES_H_
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
class MHTracer_DTPStensorflowPScompilerPSxlaPSpythonPStypesDTh {
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
   MHTracer_DTPStensorflowPScompilerPSxlaPSpythonPStypesDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSxlaPSpythonPStypesDTh() {
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


#include <memory>
#include <vector>

#include "absl/container/inlined_vector.h"
#include "absl/types/optional.h"
#include "pybind11/numpy.h"
#include "pybind11/pybind11.h"
#include "pybind11/pytypes.h"
#include "pybind11/stl.h"
#include "pybind11_abseil/absl_casters.h"  // from @pybind11_abseil
#include "tensorflow/compiler/xla/literal.h"
#include "tensorflow/compiler/xla/python/status_casters.h"
#include "tensorflow/compiler/xla/shape.h"
#include "tensorflow/compiler/xla/status.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/platform/protobuf.h"

namespace xla {

// Converts a NumPy dtype to a PrimitiveType.
StatusOr<PrimitiveType> DtypeToPrimitiveType(const pybind11::dtype& np_type);

// Converts a PrimitiveType to a Numpy dtype.
StatusOr<pybind11::dtype> PrimitiveTypeToDtype(PrimitiveType type);

// Returns a numpy-style format descriptor string for `type`.
StatusOr<std::string> FormatDescriptorForPrimitiveType(PrimitiveType type);

// Returns a numpy-style typestr for `type`, as returned by np.dtype(...).str
StatusOr<pybind11::str> TypeDescriptorForPrimitiveType(PrimitiveType type);

struct NumpyScalarTypes {
  pybind11::object np_bool;
  pybind11::object np_int8;
  pybind11::object np_int16;
  pybind11::object np_int32;
  pybind11::object np_int64;
  pybind11::object np_uint8;
  pybind11::object np_uint16;
  pybind11::object np_uint32;
  pybind11::object np_uint64;
  pybind11::object np_bfloat16;
  pybind11::object np_float16;
  pybind11::object np_float32;
  pybind11::object np_float64;
  pybind11::object np_complex64;
  pybind11::object np_complex128;
  pybind11::object np_longlong;
  pybind11::object np_intc;
};
const NumpyScalarTypes& GetNumpyScalarTypes();

// For S64/U64/F64/C128 types, returns the largest 32-bit equivalent.
PrimitiveType Squash64BitTypes(PrimitiveType type);

// Returns the strides for `shape`.
std::vector<ssize_t> ByteStridesForShape(const Shape& shape);
std::vector<int64_t> ByteStridesForShapeInt64(const Shape& shape);

// Converts a literal to (possibly-nested tuples of) NumPy arrays.
// The literal's leaf arrays are not copied; instead the NumPy arrays share
// buffers with the literals. Takes ownership of `literal` and keeps the
// necessary pieces alive using Python reference counting.
// Requires the GIL.
StatusOr<pybind11::object> LiteralToPython(std::shared_ptr<Literal> literal);

// Converts a Python object into an XLA shape and a vector of leaf buffers.
// The leaf buffers correspond to a depth-first, left-to-right traversal of
// the Python value.
// Requires the GIL.
struct PythonBufferTree {
  // Holds a reference to the arrays pointed to by `leaves`, since we may
  // need to make a copy if the array is not in a C-style layout.
  absl::InlinedVector<pybind11::object, 1> arrays;
  absl::InlinedVector<BorrowingLiteral, 1> leaves;
  Shape shape;
};
StatusOr<PythonBufferTree> GetPythonBufferTree(
    const pybind11::object& argument);

// Converts a sequence of C++ ints to a Python tuple of ints.
// Pybind11 by default converts a std::vector<T> to a Python list;
// we frequently want a tuple instead e.g. for shapes.
template <typename T>
pybind11::tuple SpanToTuple(absl::Span<T const> xs) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpythonPStypesDTh mht_0(mht_0_v, 274, "", "./tensorflow/compiler/xla/python/types.h", "SpanToTuple");

  pybind11::tuple out(xs.size());
  for (int i = 0; i < xs.size(); ++i) {
    out[i] = pybind11::cast(xs[i]);
  }
  return out;
}
template <>
pybind11::tuple SpanToTuple(absl::Span<int const> xs);
template <>
pybind11::tuple SpanToTuple(absl::Span<int64_t const> xs);

// Converts a Python iterable/sequence of T to std::vector<T>
template <typename T>
std::vector<T> IterableToVector(const pybind11::iterable& iterable) {
  std::vector<T> output;
  for (auto item : iterable) {
    output.push_back(item.cast<T>());
  }
  return output;
}
template <typename T>
std::vector<T> SequenceToVector(const pybind11::sequence& sequence) {
  std::vector<T> output;
  output.reserve(sequence.size());
  for (auto item : sequence) {
    output.push_back(item.cast<T>());
  }
  return output;
}

// Private helper function used in the implementation of the type caster for
// xla::BorrowingLiteral. Converts a Python array-like object into a buffer
// pointer and shape.
struct CastToArrayResult {
  pybind11::object array;  // Holds a reference to the array to keep it alive.
  const char* buf_ptr;
  xla::Shape shape;
};
absl::optional<CastToArrayResult> CastToArray(pybind11::handle h);

}  // namespace xla

// This namespace is a documented pybind11 extension point.
// Caution: Unusually for Google code, this code uses C++ exceptions because
// they are the only mechanism for reporting cast failures to pybind11. However,
// the exceptions are local to the binding code.
namespace pybind11 {
namespace detail {

// Literals.
// Literal data can be passed to XLA as a NumPy array; its value can be
// cast to an xla::BorrowingLiteral or xla::LiteralSlice in a zero-copy way.
// We don't have any literal -> numpy conversions here, since all the methods
// that want to return arrays build Python objects directly.

template <>
struct type_caster<xla::BorrowingLiteral> {
 public:
  PYBIND11_TYPE_CASTER(xla::BorrowingLiteral, _("xla::BorrowingLiteral"));

  // Pybind appears to keep type_casters alive until the callee has run.
  absl::InlinedVector<pybind11::array, 1> arrays;

  bool load(handle input, bool) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpythonPStypesDTh mht_1(mht_1_v, 341, "", "./tensorflow/compiler/xla/python/types.h", "load");

    // TODO(b/79707221): support nested tuples if/when XLA adds support for
    // nested BorrowingLiterals.
    if (pybind11::isinstance<pybind11::tuple>(input)) {
      pybind11::tuple tuple =
          pybind11::reinterpret_borrow<pybind11::tuple>(input);
      std::vector<xla::Shape> shapes;
      std::vector<const char*> buffers;
      arrays.reserve(tuple.size());
      shapes.reserve(tuple.size());
      buffers.reserve(tuple.size());
      for (pybind11::handle entry : tuple) {
        auto c = xla::CastToArray(entry);
        if (!c) {
          return false;
        }
        arrays.push_back(c->array);
        buffers.push_back(c->buf_ptr);
        shapes.push_back(c->shape);
      }
      value = xla::BorrowingLiteral(buffers,
                                    xla::ShapeUtil::MakeTupleShape(shapes));
    } else {
      auto c = xla::CastToArray(input);
      if (!c) {
        return false;
      }
      arrays.push_back(c->array);
      value = xla::BorrowingLiteral(c->buf_ptr, c->shape);
    }
    return true;
  }
};

template <>
struct type_caster<xla::LiteralSlice> {
 public:
  PYBIND11_TYPE_CASTER(xla::LiteralSlice, _("xla::LiteralSlice"));

  // Pybind appears to keep type_casters alive until the callee has run.
  type_caster<xla::BorrowingLiteral> literal_caster;

  bool load(handle handle, bool convert) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpythonPStypesDTh mht_2(mht_2_v, 386, "", "./tensorflow/compiler/xla/python/types.h", "load");

    if (!literal_caster.load(handle, convert)) {
      return false;
    }
    value = static_cast<const xla::BorrowingLiteral&>(literal_caster);
    return true;
  }
};

// XLA protocol buffers
// We don't actually care that these are the protocol buffers, we merely want
// objects that duck type as protocol buffers. The client code currently avoids
// depending on Python protocol buffers to avoid conflicting definitions from
// different modules that both include XLA.

template <>
struct type_caster<xla::ConvolutionDimensionNumbers> {
 public:
  PYBIND11_TYPE_CASTER(xla::ConvolutionDimensionNumbers,
                       _("xla::ConvolutionDimensionNumbers"));

  // PyObject -> C++ conversion.
  bool load(handle handle, bool) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpythonPStypesDTh mht_3(mht_3_v, 411, "", "./tensorflow/compiler/xla/python/types.h", "load");

    value.set_input_batch_dimension(
        getattr(handle, "input_batch_dimension").cast<int64_t>());
    value.set_input_feature_dimension(
        getattr(handle, "input_feature_dimension").cast<int64_t>());
    value.set_output_batch_dimension(
        getattr(handle, "output_batch_dimension").cast<int64_t>());
    value.set_output_feature_dimension(
        getattr(handle, "output_feature_dimension").cast<int64_t>());
    value.set_kernel_input_feature_dimension(
        getattr(handle, "kernel_input_feature_dimension").cast<int64_t>());
    value.set_kernel_output_feature_dimension(
        getattr(handle, "kernel_output_feature_dimension").cast<int64_t>());
    std::vector<int64_t> dims;
    dims = getattr(handle, "input_spatial_dimensions")
               .cast<std::vector<int64_t>>();
    std::copy(dims.begin(), dims.end(),
              tensorflow::protobuf::RepeatedFieldBackInserter(
                  value.mutable_input_spatial_dimensions()));
    dims = getattr(handle, "kernel_spatial_dimensions")
               .cast<std::vector<int64_t>>();
    std::copy(dims.begin(), dims.end(),
              tensorflow::protobuf::RepeatedFieldBackInserter(
                  value.mutable_kernel_spatial_dimensions()));
    dims = getattr(handle, "output_spatial_dimensions")
               .cast<std::vector<int64_t>>();
    std::copy(dims.begin(), dims.end(),
              tensorflow::protobuf::RepeatedFieldBackInserter(
                  value.mutable_output_spatial_dimensions()));
    return true;
  }
};

template <>
struct type_caster<xla::DotDimensionNumbers> {
 public:
  PYBIND11_TYPE_CASTER(xla::DotDimensionNumbers, _("xla::DotDimensionNumbers"));

  // PyObject -> C++ conversion.
  bool load(handle handle, bool) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpythonPStypesDTh mht_4(mht_4_v, 453, "", "./tensorflow/compiler/xla/python/types.h", "load");

    std::vector<int64_t> dims;
    dims = getattr(handle, "lhs_contracting_dimensions")
               .cast<std::vector<int64_t>>();
    std::copy(dims.begin(), dims.end(),
              tensorflow::protobuf::RepeatedFieldBackInserter(
                  value.mutable_lhs_contracting_dimensions()));
    dims = getattr(handle, "rhs_contracting_dimensions")
               .cast<std::vector<int64_t>>();
    std::copy(dims.begin(), dims.end(),
              tensorflow::protobuf::RepeatedFieldBackInserter(
                  value.mutable_rhs_contracting_dimensions()));
    dims = getattr(handle, "lhs_batch_dimensions").cast<std::vector<int64_t>>();
    std::copy(dims.begin(), dims.end(),
              tensorflow::protobuf::RepeatedFieldBackInserter(
                  value.mutable_lhs_batch_dimensions()));
    dims = getattr(handle, "rhs_batch_dimensions").cast<std::vector<int64_t>>();
    std::copy(dims.begin(), dims.end(),
              tensorflow::protobuf::RepeatedFieldBackInserter(
                  value.mutable_rhs_batch_dimensions()));
    return true;
  }
};

template <>
struct type_caster<xla::GatherDimensionNumbers> {
 public:
  PYBIND11_TYPE_CASTER(xla::GatherDimensionNumbers,
                       _("xla::GatherDimensionNumbers"));

  // PyObject -> C++ conversion.
  bool load(handle handle, bool) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpythonPStypesDTh mht_5(mht_5_v, 487, "", "./tensorflow/compiler/xla/python/types.h", "load");

    std::vector<int64_t> dims;
    dims = getattr(handle, "offset_dims").cast<std::vector<int64_t>>();
    std::copy(dims.begin(), dims.end(),
              tensorflow::protobuf::RepeatedFieldBackInserter(
                  value.mutable_offset_dims()));
    dims = getattr(handle, "collapsed_slice_dims").cast<std::vector<int64_t>>();
    std::copy(dims.begin(), dims.end(),
              tensorflow::protobuf::RepeatedFieldBackInserter(
                  value.mutable_collapsed_slice_dims()));
    dims = getattr(handle, "start_index_map").cast<std::vector<int64_t>>();
    std::copy(dims.begin(), dims.end(),
              tensorflow::protobuf::RepeatedFieldBackInserter(
                  value.mutable_start_index_map()));
    value.set_index_vector_dim(
        getattr(handle, "index_vector_dim").cast<int64_t>());
    return true;
  }
};

template <>
struct type_caster<xla::ScatterDimensionNumbers> {
 public:
  PYBIND11_TYPE_CASTER(xla::ScatterDimensionNumbers,
                       _("xla::ScatterDimensionNumbers"));

  // PyObject -> C++ conversion.
  bool load(handle handle, bool) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpythonPStypesDTh mht_6(mht_6_v, 517, "", "./tensorflow/compiler/xla/python/types.h", "load");

    std::vector<int64_t> dims;
    dims = getattr(handle, "update_window_dims").cast<std::vector<int64_t>>();
    std::copy(dims.begin(), dims.end(),
              tensorflow::protobuf::RepeatedFieldBackInserter(
                  value.mutable_update_window_dims()));
    dims = getattr(handle, "inserted_window_dims").cast<std::vector<int64_t>>();
    std::copy(dims.begin(), dims.end(),
              tensorflow::protobuf::RepeatedFieldBackInserter(
                  value.mutable_inserted_window_dims()));
    dims = getattr(handle, "scatter_dims_to_operand_dims")
               .cast<std::vector<int64_t>>();
    std::copy(dims.begin(), dims.end(),
              tensorflow::protobuf::RepeatedFieldBackInserter(
                  value.mutable_scatter_dims_to_operand_dims()));
    value.set_index_vector_dim(
        getattr(handle, "index_vector_dim").cast<int64_t>());
    return true;
  }
};

template <>
struct type_caster<xla::ReplicaGroup> {
 public:
  PYBIND11_TYPE_CASTER(xla::ReplicaGroup, _("xla::ReplicaGroup"));

  // PyObject -> C++ conversion.
  bool load(handle handle, bool) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpythonPStypesDTh mht_7(mht_7_v, 547, "", "./tensorflow/compiler/xla/python/types.h", "load");

    std::vector<int64_t> dims;
    dims = getattr(handle, "replica_ids").cast<std::vector<int64_t>>();
    std::copy(dims.begin(), dims.end(),
              tensorflow::protobuf::RepeatedFieldBackInserter(
                  value.mutable_replica_ids()));
    return true;
  }
};

template <>
struct type_caster<xla::PaddingConfig> {
 public:
  PYBIND11_TYPE_CASTER(xla::PaddingConfig, _("xla::PaddingConfig"));

  // PyObject -> C++ conversion.
  bool load(handle handle, bool) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpythonPStypesDTh mht_8(mht_8_v, 566, "", "./tensorflow/compiler/xla/python/types.h", "load");

    sequence dimensions =
        reinterpret_borrow<sequence>(getattr(handle, "dimensions"));

    for (const auto& dimension : dimensions) {
      xla::PaddingConfig::PaddingConfigDimension* config_dim =
          value.add_dimensions();
      config_dim->set_edge_padding_low(
          getattr(dimension, "edge_padding_low").cast<int64_t>());
      config_dim->set_edge_padding_high(
          getattr(dimension, "edge_padding_high").cast<int64_t>());
      config_dim->set_interior_padding(
          getattr(dimension, "interior_padding").cast<int64_t>());
    }
    return true;
  }
};

template <>
struct type_caster<xla::OpMetadata> {
 public:
  PYBIND11_TYPE_CASTER(xla::OpMetadata, _("xla::OpMetadata"));

  // PyObject -> C++ conversion.
  bool load(handle handle, bool) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpythonPStypesDTh mht_9(mht_9_v, 593, "", "./tensorflow/compiler/xla/python/types.h", "load");

    pybind11::handle op_type = getattr(handle, "op_type");
    if (!op_type.is_none()) {
      value.set_op_type(op_type.cast<std::string>());
    }
    pybind11::handle op_name = getattr(handle, "op_name");
    if (!op_name.is_none()) {
      value.set_op_name(op_name.cast<std::string>());
    }
    pybind11::handle source_file = getattr(handle, "source_file");
    if (!source_file.is_none()) {
      value.set_source_file(source_file.cast<std::string>());
    }
    pybind11::handle source_line = getattr(handle, "source_line");
    if (!source_line.is_none()) {
      value.set_source_line(source_line.cast<int32_t>());
    }
    return true;
  }
};

template <>
struct type_caster<xla::PrecisionConfig> {
 public:
  PYBIND11_TYPE_CASTER(xla::PrecisionConfig, _("xla::PrecisionConfig"));

  // PyObject -> C++ conversion.
  bool load(handle handle, bool) {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpythonPStypesDTh mht_10(mht_10_v, 623, "", "./tensorflow/compiler/xla/python/types.h", "load");

    if (handle.is_none()) {
      return true;
    }

    sequence operand_precisions =
        reinterpret_borrow<sequence>(getattr(handle, "operand_precision"));

    for (const auto& operand_precision : operand_precisions) {
      value.add_operand_precision(
          operand_precision.cast<xla::PrecisionConfig::Precision>());
    }
    return true;
  }
};

}  // namespace detail
}  // namespace pybind11

#endif  // TENSORFLOW_COMPILER_XLA_PYTHON_TYPES_H_
