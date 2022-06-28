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
class MHTracer_DTPStensorflowPScompilerPSmlirPStfrtPSjitPSpython_bindingPStfrt_fallbackDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSmlirPStfrtPSjitPSpython_bindingPStfrt_fallbackDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSmlirPStfrtPSjitPSpython_bindingPStfrt_fallbackDTcc() {
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

#include "tensorflow/compiler/mlir/tfrt/jit/python_binding/tfrt_fallback.h"

#include <string>
#include <vector>

#include "llvm/ADT/SmallVector.h"
#include "pybind11/numpy.h"
#include "pybind11/pybind11.h"
#include "pybind11/stl.h"
#include "tensorflow/compiler/mlir/tfrt/jit/python_binding/conversion_utils.h"
#include "tensorflow/compiler/mlir/tfrt/jit/tf_jitrt.h"
#include "tensorflow/compiler/mlir/tfrt/runtime_fallback/runtime_fallback_executor.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/platform/refcount.h"
#include "tensorflow/core/runtime_fallback/util/type_util.h"
#include "tfrt/dtype/dtype.h"  // from @tf_runtime

namespace tensorflow {

namespace py = pybind11;

using ::tfrt::jitrt::MemrefDesc;

static py::array ConvertTensorToPyArray(const Tensor& tensor) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStfrtPSjitPSpython_bindingPStfrt_fallbackDTcc mht_0(mht_0_v, 209, "", "./tensorflow/compiler/mlir/tfrt/jit/python_binding/tfrt_fallback.cc", "ConvertTensorToPyArray");

  auto tensor_sizes = tensor.shape().dim_sizes();

  auto dtype = tfd::GetTfrtDtype(tensor.dtype());
  std::vector<ssize_t> sizes(tensor_sizes.begin(), tensor_sizes.end());
  std::vector<ssize_t> strides(tensor_sizes.size(), tfrt::GetHostSize(dtype));
  if (strides.size() > 1) {
    for (size_t d = strides.size() - 1; d > 0; --d) {
      strides[d - 1] = strides[d] * tensor_sizes[d];
    }
  }

  return py::array(py::buffer_info(tensor.data(), tfrt::GetHostSize(dtype),
                                   ToPythonStructFormat(dtype), strides.size(),
                                   sizes, strides));
}

std::vector<py::array> RunTfrtFallback(
    const std::string& module_ir, const std::string& entrypoint,
    const std::vector<py::array>& arguments) {
  // Convert arguments to memrefs.
  std::vector<MemrefDesc> memrefs(arguments.size());
  for (size_t i = 0; i < arguments.size(); ++i) {
    ConvertPyArrayMemrefDesc(arguments[i], &memrefs[i]);
  }

  // Convert memrefs to tensors.
  llvm::SmallVector<Tensor> tensor_arguments;
  tensor_arguments.reserve(arguments.size());
  for (const auto& memref : memrefs) {
    size_t size = tfrt::GetHostSize(memref.dtype);
    // memref.data is still owned by the py::array. Therefore we pass nullptr as
    // base_ptr, because we don't need to keep track of it for deallocation.
    // The tensor will take ownership of the buffer from the reference counted
    // pointer.
    auto* buffer = new MemrefTensorBuffer(/*base_ptr=*/nullptr, memref.data,
                                          size, /*owner=*/false);
    auto ptr = core::RefCountPtr<MemrefTensorBuffer>(buffer);
    TensorShape shape;
    auto st = TensorShapeUtils::MakeShape(memref.sizes, &shape);
    (void)st;
    tensor_arguments.emplace_back(tfd::GetTfDataType(memref.dtype),
                                  std::move(shape), std::move(ptr));
  }

  RuntimeFallbackExecutor executor(/*num_threads=*/4);
  executor.Prepare(module_ir);
  auto results = executor.Execute(entrypoint, tensor_arguments);
  std::vector<py::array> ret_values;
  ret_values.reserve(results.size());
  for (const auto& tensor : results) {
    ret_values.push_back(ConvertTensorToPyArray(tensor));
  }
  return ret_values;
}

PYBIND11_MODULE(_tfrt_fallback, m) {
  m.def("run_tfrt_fallback", &RunTfrtFallback);
}

}  // namespace tensorflow
