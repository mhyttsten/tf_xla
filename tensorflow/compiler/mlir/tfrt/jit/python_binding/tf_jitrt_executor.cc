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
class MHTracer_DTPStensorflowPScompilerPSmlirPStfrtPSjitPSpython_bindingPStf_jitrt_executorDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSmlirPStfrtPSjitPSpython_bindingPStf_jitrt_executorDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSmlirPStfrtPSjitPSpython_bindingPStf_jitrt_executorDTcc() {
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

/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/mlir/tfrt/jit/python_binding/tf_jitrt_executor.h"

#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>

#include "mlir/Dialect/Bufferization/Transforms/Bufferize.h"
#include "mlir/ExecutionEngine/CRunnerUtils.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/compiler/mlir/tensorflow/dialect_registration.h"
#include "tensorflow/compiler/mlir/tfrt/jit/python_binding/conversion_utils.h"
#include "tensorflow/compiler/mlir/tfrt/jit/tf_jitrt_pipeline.h"
#include "tensorflow/compiler/mlir/tfrt/python_tests/python_test_attrs_registration.h"
#include "tensorflow/core/platform/dynamic_annotations.h"
#include "tfrt/jitrt/jitrt.h"  // from @tf_runtime
#include "tfrt/jitrt/jitrt_compiler.h"  // from @tf_runtime
#include "tfrt/dtype/dtype.h"  // from @tf_runtime
#include "tfrt/host_context/async_value.h"  // from @tf_runtime
#include "tfrt/host_context/concurrent_work_queue.h"  // from @tf_runtime
#include "tfrt/host_context/execution_context.h"  // from @tf_runtime
#include "tfrt/host_context/host_allocator.h"  // from @tf_runtime
#include "tfrt/host_context/kernel_utils.h"  // from @tf_runtime
#include "tfrt/support/ref_count.h"  // from @tf_runtime
#include "tfrt/support/string_util.h"  // from @tf_runtime

namespace py = pybind11;

using ::tfrt::AsyncValue;
using ::tfrt::AsyncValuePtr;
using ::tfrt::CreateMallocAllocator;
using ::tfrt::CreateMultiThreadedWorkQueue;
using ::tfrt::DecodedDiagnostic;
using ::tfrt::DType;
using ::tfrt::GetDType;
using ::tfrt::RCReference;
using ::tfrt::RemainingResults;
using ::tfrt::RequestContext;
using ::tfrt::RequestContextBuilder;
using ::tfrt::StrCat;

using ::tfrt::jitrt::CompilationOptions;
using ::tfrt::jitrt::CompilationPipelineOptions;
using ::tfrt::jitrt::CreateDefaultJitRtCompilationPipeline;
using ::tfrt::jitrt::Executable;
using ::tfrt::jitrt::HostContextAsyncTaskRunner;
using ::tfrt::jitrt::JitExecutable;
using ::tfrt::jitrt::MemrefDesc;
using ::tfrt::jitrt::RegisterDefaultJitRtDialects;
using ::tfrt::jitrt::ReturnStridedMemref;
using ::tfrt::jitrt::ReturnValueConverter;

namespace tensorflow {

TfJitRtExecutor::TfJitRtExecutor()
    : host_context_(
          [](const DecodedDiagnostic& diag) {
            llvm::errs() << "Encountered runtime error: " << diag.message
                         << "\n";
          },
          CreateMallocAllocator(), CreateMultiThreadedWorkQueue(4, 4)) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStfrtPSjitPSpython_bindingPStf_jitrt_executorDTcc mht_0(mht_0_v, 245, "", "./tensorflow/compiler/mlir/tfrt/jit/python_binding/tf_jitrt_executor.cc", "TfJitRtExecutor::TfJitRtExecutor");
}

TfJitRtExecutor::Handle TfJitRtExecutor::Compile(const std::string& mlir_module,
                                                 const std::string& entrypoint,
                                                 Specialization specialization,
                                                 bool vectorize,
                                                 bool codegen_transpose,
                                                 bool legalize_i1_tensors) {
   std::vector<std::string> mht_1_v;
   mht_1_v.push_back("mlir_module: \"" + mlir_module + "\"");
   mht_1_v.push_back("entrypoint: \"" + entrypoint + "\"");
   MHTracer_DTPStensorflowPScompilerPSmlirPStfrtPSjitPSpython_bindingPStf_jitrt_executorDTcc mht_1(mht_1_v, 257, "", "./tensorflow/compiler/mlir/tfrt/jit/python_binding/tf_jitrt_executor.cc", "TfJitRtExecutor::Compile");

  // Options for the default JitRt compilation pipeline (lowering to LLVM).
  CompilationPipelineOptions copts;
  copts.alignment = EIGEN_MAX_ALIGN_BYTES;
  copts.num_worker_threads = 4;

  CompilationOptions opts;
  opts.register_dialects = [](mlir::DialectRegistry& registry) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStfrtPSjitPSpython_bindingPStf_jitrt_executorDTcc mht_2(mht_2_v, 267, "", "./tensorflow/compiler/mlir/tfrt/jit/python_binding/tf_jitrt_executor.cc", "lambda");

    mlir::RegisterAllTensorFlowDialects(registry);
    RegisterDefaultJitRtDialects(registry);
    // Needed to verify function argument attributes which are used to
    // annotate dynamic shaped types with static type information.
    mlir::tfrt::RegisterPythonTestAttrsDialect(registry);
  };
  opts.create_compilation_pipeline = [=](mlir::PassManager& pm) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStfrtPSjitPSpython_bindingPStf_jitrt_executorDTcc mht_3(mht_3_v, 277, "", "./tensorflow/compiler/mlir/tfrt/jit/python_binding/tf_jitrt_executor.cc", "lambda");

    tensorflow::TfJitRtPipelineOptions opts;
    opts.vectorize = vectorize;
    opts.codegen_transpose = codegen_transpose;
    opts.legalize_i1_tensors = legalize_i1_tensors;
    tensorflow::CreateTfJitRtPipeline(pm, opts);
    CreateDefaultJitRtCompilationPipeline(pm, copts);
  };
  opts.create_specialization_pipeline = CreateJitRtSpecializationPipeline;
  opts.specialization = specialization;
  opts.calling_convention = CompilationOptions::DefaultCallingConvention(
      mlir::bufferization::BufferizeTypeConverter());

  // Instantiate new JitExecutable from the MLIR source.
  llvm::Expected<JitExecutable> jit_executable =
      JitExecutable::Instantiate(mlir_module, entrypoint, opts);
  if (auto err = jit_executable.takeError())
    throw std::runtime_error(
        StrCat("Failed to instantiate JitExecutable: ", err));

  Handle hdl = jit_executables_.size();
  jit_executables_.insert({hdl, std::move(*jit_executable)});
  return hdl;
}

template <typename T, int rank>
static llvm::ArrayRef<int64_t> Sizes(StridedMemRefType<T, rank>* memref) {
  return memref->sizes;
}

template <typename T, int rank>
static llvm::ArrayRef<int64_t> Strides(StridedMemRefType<T, rank>* memref) {
  return memref->strides;
}

template <typename T>
static llvm::ArrayRef<int64_t> Sizes(StridedMemRefType<T, 0>* memref) {
  return {};
}

template <typename T>
static llvm::ArrayRef<int64_t> Strides(StridedMemRefType<T, 0>* memref) {
  return {};
}

namespace {
struct PyBindingConversionContext {};

using PyBindingReturnValueConverter =
    ReturnValueConverter<PyBindingConversionContext>;
}  // namespace

template <typename T>
static bool IsAligned(const T* ptr) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStfrtPSjitPSpython_bindingPStf_jitrt_executorDTcc mht_4(mht_4_v, 333, "", "./tensorflow/compiler/mlir/tfrt/jit/python_binding/tf_jitrt_executor.cc", "IsAligned");

#if EIGEN_MAX_ALIGN_BYTES == 0
  return true;
#else
  return reinterpret_cast<intptr_t>(ptr) % EIGEN_MAX_ALIGN_BYTES == 0;
#endif
}

// Converts StridedMemrefType to the Python array. This struct satisfies
// ReturnStridedMemref's concept (see jitrt.h).
//
// TODO(ezhulenev): Currently this converter transfers ownership of the memref
// to the Python array. This is not correct in general, because memref does not
// imply ownership, for example it can be one of the forwarded inputs or a
// global memref that is owned by the compiled kernel.
struct MemrefToPyArray {
  using ResultType = py::array;
  using ConversionContext = PyBindingConversionContext;

  template <typename T, int rank>
  static py::array Convert(const ConversionContext&, void* memref_ptr) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStfrtPSjitPSpython_bindingPStf_jitrt_executorDTcc mht_5(mht_5_v, 356, "", "./tensorflow/compiler/mlir/tfrt/jit/python_binding/tf_jitrt_executor.cc", "Convert");

    auto* memref = static_cast<StridedMemRefType<T, rank>*>(memref_ptr);
    assert(IsAligned(memref->data) && "returned memref must be aligned");

    auto memref_sizes = Sizes(memref);
    auto memref_strides = Strides(memref);

    std::vector<ssize_t> sizes(memref_sizes.begin(), memref_sizes.end());
    std::vector<ssize_t> strides(memref_strides.begin(), memref_strides.end());

    // Python expects strides in bytes.
    auto dtype = GetDType<T>();
    for (size_t d = 0; d < strides.size(); ++d)
      strides[d] *= GetHostSize(dtype);

    return py::array(py::buffer_info(memref->data, GetHostSize(dtype),
                                     ToPythonStructFormat(dtype), rank, sizes,
                                     strides));
  }
};

std::vector<py::array> TfJitRtExecutor::Execute(
    Handle handle, const std::vector<py::array>& arguments) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStfrtPSjitPSpython_bindingPStf_jitrt_executorDTcc mht_6(mht_6_v, 381, "", "./tensorflow/compiler/mlir/tfrt/jit/python_binding/tf_jitrt_executor.cc", "TfJitRtExecutor::Execute");

  // Verify that we have a compilation result for the handle.
  auto it = jit_executables_.find(handle);
  if (it == jit_executables_.end())
    throw std::runtime_error(StrCat("Unknown jit executable handle: ", handle));

  JitExecutable& jit_executable = it->getSecond();

  // Build an ExecutionContext from the HostContext.
  llvm::Expected<RCReference<RequestContext>> req_ctx =
      RequestContextBuilder(&host_context_, /*resource_context=*/nullptr)
          .build();
  tfrt::ExecutionContext exec_ctx(std::move(*req_ctx));

  // Convert arguments to memrefs.
  std::vector<MemrefDesc> memrefs(arguments.size());
  for (int i = 0; i < arguments.size(); ++i)
    ConvertPyArrayMemrefDesc(arguments[i], &memrefs[i]);

  // Get an executable that might be specialized to the operands.
  llvm::Expected<AsyncValuePtr<Executable>> executable =
      jit_executable.GetExecutable(memrefs);
  if (auto err = executable.takeError())
    throw std::runtime_error(
        StrCat("Failed to get Executable: ", std::move(err)));

  // Wait for the compilation completion.
  host_context_.Await({executable->CopyRef()});

  if (executable->IsError())
    throw std::runtime_error(
        StrCat("Failed to get Executable: ", executable->GetError()));

  // Prepare storage for returned values.
  unsigned num_results = (*executable)->num_results();
  std::vector<RCReference<AsyncValue>> result_storage(num_results);

  RemainingResults results(result_storage);

  // Execute async tasks in the HostContext work queue.
  Executable::ExecuteOpts opts;
  HostContextAsyncTaskRunner async_task_runner(&host_context_);
  opts.async_task_runner = &async_task_runner;

  // Convert returned memrefs to python arrays.
  PyBindingConversionContext results_ctx;
  PyBindingReturnValueConverter converter(results, results_ctx);
  converter.AddConversion(ReturnStridedMemref<MemrefToPyArray>);
  if (auto err = (*executable)->Execute(memrefs, converter, opts))
    throw std::runtime_error(StrCat("Unsupported argument: ", err));

  // Pull Python arrays out of async values.
  std::vector<py::array> ret_values;
  ret_values.reserve(result_storage.size());
  for (auto& result : result_storage) {
    if (result->IsError())
      throw std::runtime_error(StrCat("result error: ", result->GetError()));
    py::array& result_array = result->get<py::array>();
    TF_ANNOTATE_MEMORY_IS_INITIALIZED(result_array.data(),
                                      result_array.nbytes());
    ret_values.emplace_back(result_array);
  }

  return ret_values;
}

bool TfJitRtExecutor::BuiltWith(const std::string& cpu_feature) {
   std::vector<std::string> mht_7_v;
   mht_7_v.push_back("cpu_feature: \"" + cpu_feature + "\"");
   MHTracer_DTPStensorflowPScompilerPSmlirPStfrtPSjitPSpython_bindingPStf_jitrt_executorDTcc mht_7(mht_7_v, 451, "", "./tensorflow/compiler/mlir/tfrt/jit/python_binding/tf_jitrt_executor.cc", "TfJitRtExecutor::BuiltWith");

  if (cpu_feature == "AVX2") {
#ifdef __AVX2__
    return true;
#else
    return false;
#endif
  }
  return false;
}

}  // namespace tensorflow

PYBIND11_MODULE(_tf_jitrt_executor, m) {
  py::enum_<tensorflow::TfJitRtExecutor::Specialization>(m, "Specialization")
      .value("ENABLED", tensorflow::TfJitRtExecutor::Specialization::kEnabled)
      .value("DISABLED", tensorflow::TfJitRtExecutor::Specialization::kDisabled)
      .value("ALWAYS", tensorflow::TfJitRtExecutor::Specialization::kAlways);

  py::class_<tensorflow::TfJitRtExecutor>(m, "TfJitRtExecutor")
      .def(py::init<>())
      .def("compile", &tensorflow::TfJitRtExecutor::Compile,
           py::arg("mlir_module"), py::arg("entrypoint"),
           py::arg("specialization") =
               tensorflow::TfJitRtExecutor::Specialization::kEnabled,
           py::arg("vectorize") = false, py::arg("codegen_transpose") = false,
           py::arg("legalize_i1_tensors") = false)
      .def("execute", &tensorflow::TfJitRtExecutor::Execute)
      .def("built_with", &tensorflow::TfJitRtExecutor::BuiltWith,
           py::arg("cpu_feature"));
}
