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
class MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSlibPSqrDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSlibPSqrDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSlibPSqrDTcc() {
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

#include "tensorflow/compiler/xla/client/lib/qr.h"

#include <memory>
#include <vector>

#include "tensorflow/compiler/xla/client/lib/arithmetic.h"
#include "tensorflow/compiler/xla/client/lib/constants.h"
#include "tensorflow/compiler/xla/client/lib/loops.h"
#include "tensorflow/compiler/xla/client/lib/math.h"
#include "tensorflow/compiler/xla/client/lib/matrix.h"
#include "tensorflow/compiler/xla/client/lib/slicing.h"
#include "tensorflow/compiler/xla/client/xla_builder.h"
#include "tensorflow/compiler/xla/literal_util.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/core/lib/core/errors.h"

namespace xla {

QrDecomposition Qr(XlaOp a) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSlibPSqrDTcc mht_0(mht_0_v, 205, "", "./tensorflow/compiler/xla/client/lib/qr.cc", "Qr");

  auto result = [&]() -> StatusOr<QrDecomposition> {
    XlaBuilder* builder = a.builder();
    TF_ASSIGN_OR_RETURN(Shape a_shape, builder->GetShape(a));
    const int num_dims = a_shape.rank();
    if (num_dims < 2) {
      return InvalidArgument(
          "Arguments to QR must have rank >= 2: got shape %s",
          a_shape.ToString());
    }
    const int64_t m = ShapeUtil::GetDimension(a_shape, -2);
    const int64_t n = ShapeUtil::GetDimension(a_shape, -1);

    std::vector<int64_t> taus_dims(a_shape.dimensions().begin(),
                                   a_shape.dimensions().end());
    taus_dims.pop_back();
    taus_dims.back() = std::min(m, n);
    auto taus_shape = ShapeUtil::MakeShape(a_shape.element_type(), taus_dims);

    Shape qr_shape = ShapeUtil::MakeTupleShape({a_shape, taus_shape});
    auto qr = CustomCall(a.builder(), "Qr", {a}, qr_shape);
    a = GetTupleElement(qr, 0);
    auto taus = GetTupleElement(qr, 1);

    return QrDecomposition{a, taus};
  }();
  if (!result.ok()) {
    XlaOp error = a.builder()->ReportError(result.status());
    return QrDecomposition{error, error};
  }
  return result.ValueOrDie();
}

XlaOp ProductOfElementaryHouseholderReflectors(XlaOp a, XlaOp taus) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSlibPSqrDTcc mht_1(mht_1_v, 241, "", "./tensorflow/compiler/xla/client/lib/qr.cc", "ProductOfElementaryHouseholderReflectors");

  XlaBuilder* builder = a.builder();
  return builder->ReportErrorOrReturn([&]() -> StatusOr<XlaOp> {
    TF_ASSIGN_OR_RETURN(Shape a_shape, builder->GetShape(a));
    TF_ASSIGN_OR_RETURN(Shape taus_shape, builder->GetShape(taus));
    if (a_shape.rank() < 2) {
      return InvalidArgument(
          "Matrix `a` must have >= 2 dimensions: got shape %s",
          a_shape.ToString());
    }
    if (taus_shape.rank() + 1 != a_shape.rank()) {
      return InvalidArgument(
          "Matrix `taus` must have one fewer dimension than `a`: got shapes "
          "%s and %s",
          taus_shape.ToString(), a_shape.ToString());
    }
    const int64_t m = ShapeUtil::GetDimension(a_shape, -2);
    const int64_t n = ShapeUtil::GetDimension(a_shape, -1);
    if (m < n) {
      return InvalidArgument(
          "Argument to product of elementary Householder "
          "reflectors must have m >= n, got shape %s",
          a_shape.ToString());
    }
    absl::Span<const int64_t> a_batch_dims =
        absl::MakeConstSpan(a_shape.dimensions().begin(),
                            a_shape.dimensions().begin() + a_shape.rank() - 2);
    absl::Span<const int64_t> taus_batch_dims = absl::MakeConstSpan(
        taus_shape.dimensions().begin(),
        taus_shape.dimensions().begin() + taus_shape.rank() - 1);
    const int64_t k = ShapeUtil::GetDimension(taus_shape, -1);
    if (a_shape.element_type() != taus_shape.element_type() ||
        a_batch_dims != taus_batch_dims || k > n) {
      return InvalidArgument("Invalid shape for `taus`, got a=%s and taus=%s",
                             taus_shape.ToString(), a_shape.ToString());
    }
    return CustomCall(a.builder(), "ProductOfElementaryHouseholderReflectors",
                      {a, taus}, a_shape);
  });
}

void QrExplicit(XlaOp a, bool full_matrices, XlaOp& q, XlaOp& r) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSlibPSqrDTcc mht_2(mht_2_v, 285, "", "./tensorflow/compiler/xla/client/lib/qr.cc", "QrExplicit");

  StatusOr<Shape> a_shape_or = a.builder()->GetShape(a);
  if (!a_shape_or.ok()) {
    q = a.builder()->ReportError(a_shape_or.status());
    r = q;
    return;
  }
  Shape a_shape = a_shape_or.ValueOrDie();
  const int64_t m = ShapeUtil::GetDimension(a_shape, -2);
  const int64_t n = ShapeUtil::GetDimension(a_shape, -1);
  const int64_t p = std::min(m, n);

  auto qr = Qr(a);
  if (full_matrices) {
    XlaOp t;
    if (m < n) {
      t = SliceInMinorDims(qr.q_and_r, {0, 0}, {m, m});
    } else {
      t = PadInDim(qr.q_and_r, Zero(a.builder(), a_shape.element_type()),
                   a_shape.dimensions_size() - 1, /*pad_lo=*/0,
                   /*pad_hi=*/m - n);
    }
    q = ProductOfElementaryHouseholderReflectors(t, qr.taus);
    r = UpperTriangle(qr.q_and_r);
  } else {
    XlaOp t;
    if (m < n) {
      t = SliceInMinorDims(qr.q_and_r, {0, 0}, {m, m});
    } else {
      t = qr.q_and_r;
    }
    q = ProductOfElementaryHouseholderReflectors(t, qr.taus);
    q = SliceInMinorDims(q, {0, 0}, {m, p});
    r = UpperTriangle(SliceInMinorDims(qr.q_and_r, {0, 0}, {p, n}));
  }
}

}  // namespace xla
