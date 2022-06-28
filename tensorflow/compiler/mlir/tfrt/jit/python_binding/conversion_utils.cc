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
class MHTracer_DTPStensorflowPScompilerPSmlirPStfrtPSjitPSpython_bindingPSconversion_utilsDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSmlirPStfrtPSjitPSpython_bindingPSconversion_utilsDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSmlirPStfrtPSjitPSpython_bindingPSconversion_utilsDTcc() {
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

/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/mlir/tfrt/jit/python_binding/conversion_utils.h"

#include <stdexcept>

#include "pybind11/numpy.h"
#include "tfrt/jitrt/types.h"  // from @tf_runtime
#include "tfrt/dtype/dtype.h"  // from @tf_runtime

namespace tensorflow {

using ::tfrt::DType;

using ::tfrt::jitrt::MemrefDesc;

// Returns Python buffer protocol's type string from TFRT's dtype.
const char* ToPythonStructFormat(DType dtype_kind) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStfrtPSjitPSpython_bindingPSconversion_utilsDTcc mht_0(mht_0_v, 200, "", "./tensorflow/compiler/mlir/tfrt/jit/python_binding/conversion_utils.cc", "ToPythonStructFormat");

  // Reference: https://docs.python.org/3/library/struct.html

  switch (dtype_kind) {
    case DType::Invalid:
      throw std::runtime_error("Invalid dtype.");
    case DType::Unsupported:
      throw std::runtime_error("Unsupported dtype.");
    case DType::UI8:
      return "B";
    case DType::UI16:
      return "H";
    case DType::UI32:
      return "I";
    case DType::UI64:
      return "Q";
    case DType::I1:
      return "?";
    case DType::I8:
      return "b";
    case DType::I16:
      return "h";
    case DType::I32:
      return "i";
    case DType::I64:
      return "q";
    case DType::F32:
      return "f";
    case DType::F64:
      return "d";
    case DType::Complex64:
      throw std::runtime_error("Unimplemented.");
    case DType::Complex128:
      throw std::runtime_error("Unimplemented.");
    case DType::F16:
      throw std::runtime_error("Unimplemented.");
    case DType::BF16:
      throw std::runtime_error("Unimplemented.");
    case DType::String:
      throw std::runtime_error("Unimplemented.");
    default:
      throw std::runtime_error("Unimplemented.");
  }
}

// Returns TFRT's dtype for the Python buffer protocol's type string.
DType FromPythonStructFormat(char dtype) {
   std::vector<std::string> mht_1_v;
   mht_1_v.push_back("dtype: '" + std::string(1, dtype) + "'");
   MHTracer_DTPStensorflowPScompilerPSmlirPStfrtPSjitPSpython_bindingPSconversion_utilsDTcc mht_1(mht_1_v, 250, "", "./tensorflow/compiler/mlir/tfrt/jit/python_binding/conversion_utils.cc", "FromPythonStructFormat");

  // Reference: https://docs.python.org/3/library/struct.html
  switch (dtype) {
    case 'B':
      return DType::UI8;
    case 'H':
      return DType::UI16;
    case 'I':
      return DType::UI32;
    case 'Q':
      return DType::UI64;
    case '?':
      return DType::I1;
    case 'b':
      return DType::I8;
    case 'h':
      return DType::I16;
    case 'i':
      return DType::I32;
    case 'q':
      return DType::I64;
    case 'f':
      return DType::F32;
    case 'd':
      return DType::F64;
    default:
      throw std::runtime_error("Unsupported python dtype.");
  }
}

// Converts Python array to the Memref Descriptor.
void ConvertPyArrayMemrefDesc(const pybind11::array& array,
                              MemrefDesc* memref) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStfrtPSjitPSpython_bindingPSconversion_utilsDTcc mht_2(mht_2_v, 285, "", "./tensorflow/compiler/mlir/tfrt/jit/python_binding/conversion_utils.cc", "ConvertPyArrayMemrefDesc");

  auto py_dtype = [](pybind11::dtype dtype) -> char {
    // np.int64 array for some reason has `i` dtype, however according to the
    // documentation it must be `q`.
    if (dtype.kind() == 'i' && dtype.itemsize() == 8) return 'q';

    return dtype.char_();
  };

  memref->dtype = DType(FromPythonStructFormat(py_dtype(array.dtype())));
  memref->data = const_cast<void*>(array.data());
  memref->offset = 0;

  auto rank = array.ndim();
  memref->sizes.resize(rank);
  memref->strides.resize(rank);

  for (ssize_t d = 0; d < rank; ++d) {
    memref->sizes[d] = array.shape(d);
    memref->strides[d] = array.strides(d) / array.itemsize();
  }
}

}  // namespace tensorflow
