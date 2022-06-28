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
class MHTracer_DTPStensorflowPScompilerPSxlaPSpythonPStypesDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSxlaPSpythonPStypesDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSxlaPSpythonPStypesDTcc() {
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

#include "tensorflow/compiler/xla/python/types.h"

#include "absl/container/flat_hash_map.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/python/lib/core/bfloat16.h"

namespace xla {

namespace py = pybind11;

xla::StatusOr<PrimitiveType> DtypeToPrimitiveType(const py::dtype& np_type) {
  static auto* types =
      new absl::flat_hash_map<std::pair<char, int>, PrimitiveType>({
          {{'b', 1}, PRED},
          {{'i', 1}, S8},
          {{'i', 2}, S16},
          {{'i', 4}, S32},
          {{'i', 8}, S64},
          {{'u', 1}, U8},
          {{'u', 2}, U16},
          {{'u', 4}, U32},
          {{'u', 8}, U64},
          {{'V', 2}, BF16},  // array protocol code for raw data (void*)
          {{'f', 2}, F16},
          {{'f', 4}, F32},
          {{'f', 8}, F64},
          {{'c', 8}, C64},
          {{'c', 16}, C128},
      });
  auto it = types->find({np_type.kind(), np_type.itemsize()});
  if (it == types->end()) {
    return InvalidArgument("Unknown NumPy type %c size %d", np_type.kind(),
                           np_type.itemsize());
  }
  return it->second;
}

xla::StatusOr<py::dtype> PrimitiveTypeToDtype(PrimitiveType type) {
  switch (type) {
    case PRED:
      return py::dtype::of<bool>();
    case S8:
      return py::dtype::of<int8_t>();
    case S16:
      return py::dtype::of<int16_t>();
    case S32:
      return py::dtype::of<int32_t>();
    case S64:
      return py::dtype::of<int64_t>();
    case U8:
      return py::dtype::of<uint8_t>();
    case U16:
      return py::dtype::of<uint16_t>();
    case U32:
      return py::dtype::of<uint32_t>();
    case U64:
      return py::dtype::of<uint64_t>();
    case BF16: {
      py::handle bfloat16(tensorflow::Bfloat16Dtype());
      return py::dtype::from_args(py::reinterpret_borrow<py::object>(bfloat16));
    }
    case F16:
      return py::dtype("e");  // PEP 3118 code for "float16
    case F32:
      return py::dtype::of<float>();
    case F64:
      return py::dtype::of<double>();
    case C64:
      return py::dtype::of<std::complex<float>>();
    case C128:
      return py::dtype::of<std::complex<double>>();
    default:
      return Unimplemented("Unimplemented primitive type %s",
                           PrimitiveType_Name(type));
  }
}

const NumpyScalarTypes& GetNumpyScalarTypes() {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpythonPStypesDTcc mht_0(mht_0_v, 262, "", "./tensorflow/compiler/xla/python/types.cc", "GetNumpyScalarTypes");

  static const NumpyScalarTypes* singleton = []() {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpythonPStypesDTcc mht_1(mht_1_v, 266, "", "./tensorflow/compiler/xla/python/types.cc", "lambda");

    NumpyScalarTypes* dtypes = new NumpyScalarTypes();
    const auto numpy = py::module::import("numpy");
    dtypes->np_bool = py::object(numpy.attr("bool_"));
    dtypes->np_int8 = py::object(numpy.attr("int8"));
    dtypes->np_int16 = py::object(numpy.attr("int16"));
    dtypes->np_int32 = py::object(numpy.attr("int32"));
    dtypes->np_int64 = py::object(numpy.attr("int64"));
    dtypes->np_uint8 = py::object(numpy.attr("uint8"));
    dtypes->np_uint16 = py::object(numpy.attr("uint16"));
    dtypes->np_uint32 = py::object(numpy.attr("uint32"));
    dtypes->np_uint64 = py::object(numpy.attr("uint64"));
    dtypes->np_bfloat16 =
        py::reinterpret_borrow<py::object>(tensorflow::Bfloat16Dtype());
    dtypes->np_float16 = py::object(numpy.attr("float16"));
    dtypes->np_float32 = py::object(numpy.attr("float32"));
    dtypes->np_float64 = py::object(numpy.attr("float64"));
    dtypes->np_complex64 = py::object(numpy.attr("complex64"));
    dtypes->np_complex128 = py::object(numpy.attr("complex128"));
    dtypes->np_longlong = py::object(numpy.attr("longlong"));
    dtypes->np_intc = py::object(numpy.attr("intc"));
    return dtypes;
  }();
  return *singleton;
}

// Returns a numpy-style format descriptor string for `type`.
StatusOr<std::string> FormatDescriptorForPrimitiveType(PrimitiveType type) {
  // We use an "=" prefix to indicate that we prefer "standard" types like
  // np.int32 rather than "native" types like np.cint. pybind11 does not qualify
  // its format descriptors.
  switch (type) {
    case PRED:
      return std::string("?");
    case S8:
      return std::string("=b");
    case S16:
      return std::string("=h");
    case S32:
      return std::string("=i");
    case S64:
      return std::string("=q");
    case U8:
      return std::string("=B");
    case U16:
      return std::string("=H");
    case U32:
      return std::string("=I");
    case U64:
      return std::string("=Q");
    case F16:
      return std::string("=e");
    case F32:
      return std::string("=f");
    case F64:
      return std::string("=d");
    case C64:
      return std::string("=Zf");
    case C128:
      return std::string("=Zd");
    default:
      return Unimplemented("Unimplemented primitive type %s",
                           PrimitiveType_Name(type));
  }
}

StatusOr<py::str> TypeDescriptorForPrimitiveType(PrimitiveType type) {
  static_assert(__BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__,
                "Big endian support not implemented");
  switch (type) {
    case PRED:
      return py::str("|b1");
    case S8:
      return py::str("|i1");
    case S16:
      return py::str("<i2");
    case S32:
      return py::str("<i4");
    case S64:
      return py::str("<i8");
    case U8:
      return py::str("|u1");
    case U16:
      return py::str("<u2");
    case U32:
      return py::str("<u4");
    case U64:
      return py::str("<u8");
    case BF16:
      return py::str("<V2");
    case F16:
      return py::str("<f2");
    case F32:
      return py::str("<f4");
    case F64:
      return py::str("<f8");
    case C64:
      return py::str("<c8");
    case C128:
      return py::str("<c16");
    default:
      return Unimplemented("Unimplemented primitive type %s",
                           PrimitiveType_Name(type));
  }
}

PrimitiveType Squash64BitTypes(PrimitiveType type) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpythonPStypesDTcc mht_2(mht_2_v, 375, "", "./tensorflow/compiler/xla/python/types.cc", "Squash64BitTypes");

  switch (type) {
    case S64:
      return S32;
    case U64:
      return U32;
    case F64:
      return F32;
    case C128:
      return C64;
    default:
      return type;
  }
}

// Returns the strides for `shape`.
std::vector<ssize_t> ByteStridesForShape(const Shape& shape) {
  std::vector<ssize_t> strides;
  CHECK(shape.IsArray());
  CHECK(shape.has_layout());

  strides.resize(shape.dimensions_size());
  ssize_t stride = ShapeUtil::ByteSizeOfPrimitiveType(shape.element_type());
  for (int i : shape.layout().minor_to_major()) {
    strides.at(i) = stride;
    stride *= shape.dimensions(i);
  }
  return strides;
}

std::vector<int64_t> ByteStridesForShapeInt64(const Shape& shape) {
  std::vector<int64_t> strides;
  CHECK(shape.IsArray());
  CHECK(shape.has_layout());

  strides.resize(shape.dimensions_size());
  int64_t stride = ShapeUtil::ByteSizeOfPrimitiveType(shape.element_type());
  for (int i : shape.layout().minor_to_major()) {
    strides.at(i) = stride;
    stride *= shape.dimensions(i);
  }
  return strides;
}

StatusOr<py::object> LiteralToPython(std::shared_ptr<xla::Literal> literal) {
  xla::Literal& m = *literal;
  if (m.shape().IsTuple()) {
    std::vector<Literal> elems = m.DecomposeTuple();
    std::vector<py::object> arrays(elems.size());
    for (int i = 0; i < elems.size(); ++i) {
      TF_ASSIGN_OR_RETURN(
          arrays[i],
          LiteralToPython(absl::make_unique<Literal>(std::move(elems[i]))));
    }
    py::tuple result(elems.size());
    for (int i = 0; i < elems.size(); ++i) {
      PyTuple_SET_ITEM(result.ptr(), i, arrays[i].release().ptr());
    }
    return result;
  }
  TF_RET_CHECK(m.shape().IsArray());

  py::object literal_object = py::cast(literal);
  TF_ASSIGN_OR_RETURN(py::dtype dtype,
                      PrimitiveTypeToDtype(m.shape().element_type()));
  return py::array(dtype, m.shape().dimensions(),
                   ByteStridesForShape(m.shape()), m.untyped_data(),
                   literal_object);
}

StatusOr<PythonBufferTree> GetPythonBufferTree(const py::object& argument) {
  PythonBufferTree tree;
  if (py::isinstance<py::tuple>(argument)) {
    py::tuple tuple = py::reinterpret_borrow<py::tuple>(argument);
    std::vector<Shape> host_shapes(tuple.size());
    for (int i = 0; i < host_shapes.size(); ++i) {
      TF_ASSIGN_OR_RETURN(PythonBufferTree subtree,
                          GetPythonBufferTree(tuple[i]));
      tree.leaves.reserve(tree.leaves.size() + subtree.leaves.size());
      std::move(subtree.leaves.begin(), subtree.leaves.end(),
                std::back_inserter(tree.leaves));
      tree.arrays.reserve(tree.arrays.size() + subtree.arrays.size());
      std::move(subtree.arrays.begin(), subtree.arrays.end(),
                std::back_inserter(tree.arrays));
      host_shapes[i] = std::move(subtree.shape);
    }
    tree.shape = ShapeUtil::MakeTupleShape(host_shapes);
  } else {
    pybind11::detail::type_caster<BorrowingLiteral> caster;
    if (!caster.load(argument, /*convert=*/true)) {
      return InvalidArgument("Invalid array value.");
    }
    DCHECK_EQ(caster.arrays.size(), 1);
    tree.arrays.push_back(std::move(caster.arrays.front()));
    tree.leaves.push_back(std::move(*caster));
    tree.shape = tree.leaves.front().shape();
  }
  return tree;
}

template <typename IntType>
static py::tuple IntSpanToTupleHelper(absl::Span<IntType const> xs) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpythonPStypesDTcc mht_3(mht_3_v, 479, "", "./tensorflow/compiler/xla/python/types.cc", "IntSpanToTupleHelper");

  py::tuple out(xs.size());
  for (int i = 0; i < xs.size(); ++i) {
    out[i] = py::int_(xs[i]);
  }
  return out;
}

template <>
pybind11::tuple SpanToTuple(absl::Span<int const> xs) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpythonPStypesDTcc mht_4(mht_4_v, 491, "", "./tensorflow/compiler/xla/python/types.cc", "SpanToTuple");

  return IntSpanToTupleHelper(xs);
}
template <>
pybind11::tuple SpanToTuple(absl::Span<int64_t const> xs) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpythonPStypesDTcc mht_5(mht_5_v, 498, "", "./tensorflow/compiler/xla/python/types.cc", "SpanToTuple");

  return IntSpanToTupleHelper(xs);
}

absl::optional<CastToArrayResult> CastToArray(py::handle h) {
  py::array array = py::array::ensure(
      h, py::array::c_style | py::detail::npy_api::NPY_ARRAY_ALIGNED_);
  if (!array) {
    return absl::nullopt;
  }
  auto type_or_status = DtypeToPrimitiveType(array.dtype());
  if (!type_or_status.ok()) {
    throw std::runtime_error(type_or_status.status().ToString());
  }
  PrimitiveType type = type_or_status.ValueOrDie();

  absl::InlinedVector<int64_t, 4> dims(array.ndim());
  for (int i = 0; i < array.ndim(); ++i) {
    dims[i] = array.shape(i);
  }
  Shape shape = ShapeUtil::MakeShape(type, dims);
  if (array.size() * array.itemsize() != ShapeUtil::ByteSizeOf(shape)) {
    throw std::runtime_error(absl::StrCat(
        "Size mismatch for buffer: ", array.size() * array.itemsize(), " vs. ",
        ShapeUtil::ByteSizeOf(shape)));
  }
  return CastToArrayResult{array, static_cast<const char*>(array.data()),
                           shape};
}

}  // namespace xla
