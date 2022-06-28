/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_COMPILER_XLA_TESTS_CLIENT_LIBRARY_TEST_BASE_H_
#define TENSORFLOW_COMPILER_XLA_TESTS_CLIENT_LIBRARY_TEST_BASE_H_
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
class MHTracer_DTPStensorflowPScompilerPSxlaPStestsPSclient_library_test_baseDTh {
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
   MHTracer_DTPStensorflowPScompilerPSxlaPStestsPSclient_library_test_baseDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSxlaPStestsPSclient_library_test_baseDTh() {
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
#include <string>
#include <type_traits>
#include <vector>

#include "absl/memory/memory.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "tensorflow/compiler/xla/array2d.h"
#include "tensorflow/compiler/xla/array3d.h"
#include "tensorflow/compiler/xla/array4d.h"
#include "tensorflow/compiler/xla/client/client_library.h"
#include "tensorflow/compiler/xla/client/global_data.h"
#include "tensorflow/compiler/xla/client/xla_builder.h"
#include "tensorflow/compiler/xla/client/xla_computation.h"
#include "tensorflow/compiler/xla/literal.h"
#include "tensorflow/compiler/xla/literal_util.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/compiler/xla/tests/literal_test_util.h"
#include "tensorflow/compiler/xla/tests/manifest_checking_test.h"
#include "tensorflow/compiler/xla/tests/test_utils.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/lib/core/bitmap.h"
#include "tensorflow/core/platform/stream_executor_no_cuda.h"
#include "tensorflow/core/platform/test.h"

namespace xla {

// Sets the use_bfloat16 on a container of test cases according to the values in
// use_bfloat16_params. Generates one set of test cases for each values in
// use_bfloat16_params with that value. Returns the result.
template <typename TestCase>
std::vector<TestCase> ExpandUseBfloat16(
    absl::Span<const bool> use_bfloat16_params,
    absl::Span<const TestCase> specs) {
  std::vector<TestCase> expanded;
  for (bool use_bfloat16 : use_bfloat16_params) {
    for (const auto& spec : specs) {
      expanded.push_back(spec);
      expanded.back().use_bfloat16 = use_bfloat16;
    }
  }
  return expanded;
}

// A client library test establishes an in-process XLA client connection.
class ClientLibraryTestBase : public ManifestCheckingTest {
 protected:
  explicit ClientLibraryTestBase(se::Platform* platform = nullptr);

  // Creates a new ClientLibraryTestBase with custom client options.
  ClientLibraryTestBase(se::Platform* platform,
                        const LocalClientOptions& client_options);

  // Returns the name of the test currently being run.
  std::string TestName() const;

  void SetFastMathDisabled(bool disabled) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPStestsPSclient_library_test_baseDTh mht_0(mht_0_v, 245, "", "./tensorflow/compiler/xla/tests/client_library_test_base.h", "SetFastMathDisabled");

    auto* opts = execution_options_.mutable_debug_options();
    opts->set_xla_cpu_enable_fast_math(!disabled);
    opts->set_xla_cpu_enable_fast_min_max(!disabled);
    opts->set_xla_gpu_enable_fast_min_max(!disabled);
  }

  void SetSeed(uint64_t seed) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPStestsPSclient_library_test_baseDTh mht_1(mht_1_v, 255, "", "./tensorflow/compiler/xla/tests/client_library_test_base.h", "SetSeed");
 execution_options_.set_seed(seed); }

  // Provides mutable access to the execution DebugOptions field; this lets
  // tests tweak the options that will be used to compile/run the graph.
  DebugOptions* mutable_debug_options() {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPStestsPSclient_library_test_baseDTh mht_2(mht_2_v, 262, "", "./tensorflow/compiler/xla/tests/client_library_test_base.h", "mutable_debug_options");

    return execution_options_.mutable_debug_options();
  }

  // TODO(b/25566808): Add helper that populates a literal from a testdata file.

  // Convenience methods for building and running a computation with the member
  // execution options. Modify execution_options_ in your test if you want to
  // customize the options.
  StatusOr<std::unique_ptr<GlobalData>> Execute(
      XlaBuilder* builder, absl::Span<GlobalData* const> arguments);

  StatusOr<Literal> ExecuteAndTransfer(
      XlaBuilder* builder, absl::Span<GlobalData* const> arguments,
      const Shape* shape_with_output_layout = nullptr);

  StatusOr<Literal> ExecuteAndTransfer(
      const XlaComputation& computation,
      absl::Span<GlobalData* const> arguments,
      const Shape* shape_with_output_layout = nullptr);

  // This executes the computation via the reference client (which connects a
  // interpreter backend). The result is used as the expected value of the
  // computation.
  StatusOr<Literal> ExecuteAndTransferReference(
      const XlaComputation& computation,
      absl::Span<GlobalData* const> arguments,
      const Shape* shape_with_output_layout = nullptr);

  // Run a computation and return its value as a string. If an error
  // occurs, then instead return the error as a string.
  std::string ExecuteToString(XlaBuilder* builder,
                              absl::Span<GlobalData* const> arguments);

  // Convenience methods for building and running a computation, transferring
  // the result, and comparing it to the expected value(s). Methods are
  // templated on the native host type which maps to specific XLA types (See
  // XlaBuilder for details). For each rank, two forms are
  // provided: one for floating point types with an ErrorSpec parameter, and one
  // for integral types without the ErrorSpec parameter.
  template <typename NativeT>
  void ComputeAndCompareR0(XlaBuilder* builder, NativeT expected,
                           absl::Span<GlobalData* const> arguments);
  template <typename NativeT>
  void ComputeAndCompareR0(XlaBuilder* builder, NativeT expected,
                           absl::Span<GlobalData* const> arguments,
                           ErrorSpec error);

  template <typename NativeT>
  void ComputeAndCompareR1(XlaBuilder* builder,
                           absl::Span<const NativeT> expected,
                           absl::Span<GlobalData* const> arguments);
  template <typename NativeT>
  void ComputeAndCompareR1(XlaBuilder* builder,
                           absl::Span<const NativeT> expected,
                           absl::Span<GlobalData* const> arguments,
                           ErrorSpec error);

  // As above, but uses a bitmap to hold the predicate vector to avoid
  // deficiencies of vector<bool>.
  void ComputeAndCompareR1(XlaBuilder* builder,
                           const tensorflow::core::Bitmap& expected,
                           absl::Span<GlobalData* const> arguments);

  template <typename NativeT>
  void ComputeAndCompareR2(XlaBuilder* builder,
                           const Array2D<NativeT>& expected,
                           absl::Span<GlobalData* const> arguments);
  template <typename NativeT>
  void ComputeAndCompareR2(XlaBuilder* builder,
                           const Array2D<NativeT>& expected,
                           absl::Span<GlobalData* const> arguments,
                           ErrorSpec error);

  template <typename NativeT>
  void ComputeAndCompareR3(XlaBuilder* builder,
                           const Array3D<NativeT>& expected,
                           absl::Span<GlobalData* const> arguments);
  template <typename NativeT>
  void ComputeAndCompareR3(XlaBuilder* builder,
                           const Array3D<NativeT>& expected,
                           absl::Span<GlobalData* const> arguments,
                           ErrorSpec error);

  template <typename NativeT>
  void ComputeAndCompareR4(XlaBuilder* builder,
                           const Array4D<NativeT>& expected,
                           absl::Span<GlobalData* const> arguments);
  template <typename NativeT>
  void ComputeAndCompareR4(XlaBuilder* builder,
                           const Array4D<NativeT>& expected,
                           absl::Span<GlobalData* const> arguments,
                           ErrorSpec error);

  // Build and run the computation and compare the result with the given
  // literal. shape_with_layout indicates the result layout to request when
  // calling Execute.
  void ComputeAndCompareLiteral(XlaBuilder* builder, const Literal& expected,
                                absl::Span<GlobalData* const> arguments,
                                const Shape* shape_with_layout = nullptr);
  void ComputeAndCompareLiteral(XlaBuilder* builder, const Literal& expected,
                                absl::Span<GlobalData* const> arguments,
                                ErrorSpec error,
                                const Shape* shape_with_layout = nullptr);

  // Build and run the computation and return the result as a literal.
  // shape_with_layout indicates the result layout to request when calling
  // Execute.
  StatusOr<Literal> ComputeAndTransfer(
      XlaBuilder* builder, absl::Span<GlobalData* const> arguments,
      const Shape* shape_with_layout = nullptr);

  // ComputeAndCompare variant which returns an error status.
  Status ComputeAndCompareLiteralWithStatus(
      XlaBuilder* builder, const Literal& expected,
      absl::Span<GlobalData* const> arguments,
      const Shape* shape_with_layout = nullptr);
  Status ComputeAndCompareLiteralWithStatus(
      XlaBuilder* builder, const Literal& expected,
      absl::Span<GlobalData* const> arguments, ErrorSpec error,
      const Shape* shape_with_layout = nullptr);

  // Compare the result of the computation to a strings. In XLA strings are
  // represented using rank-1 U8 shapes.
  void ComputeAndCompareR1U8(XlaBuilder* builder, absl::string_view expected,
                             absl::Span<GlobalData* const> arguments);

  // Convenience method for running a built computation, transferring the
  // result, and comparing it to the expected tuple literal.
  void ComputeAndCompareTuple(XlaBuilder* builder, const Literal& expected,
                              absl::Span<GlobalData* const> arguments);
  void ComputeAndCompareTuple(XlaBuilder* builder, const Literal& expected,
                              absl::Span<GlobalData* const> arguments,
                              ErrorSpec error);

  // Convenience method for running a built computation and comparing the result
  // with the reference result.
  void ComputeAndCompare(XlaBuilder* builder,
                         absl::Span<const Literal> arguments);
  void ComputeAndCompare(XlaBuilder* builder,
                         absl::Span<const Literal> arguments, ErrorSpec error);
  template <typename NativeT>
  void ComputeAndCompare(XlaBuilder* builder, const Array<NativeT>& expected,
                         absl::Span<GlobalData* const> arguments);
  template <typename NativeT>
  void ComputeAndCompare(XlaBuilder* builder, const Array<NativeT>& expected,
                         absl::Span<GlobalData* const> arguments,
                         ErrorSpec error);
  // Create scalar operations for use in reductions.
  XlaComputation CreateScalarRelu();
  XlaComputation CreateScalarMax();
  XlaComputation CreateScalarReluSensitivity();

  // Special case convenience functions for creating filled arrays.

  // Creates an array of pseudorandom values lying between the given minimum and
  // maximum values.
  template <typename NativeT>
  std::vector<NativeT> CreatePseudorandomR1(const int width, NativeT min_value,
                                            NativeT max_value, uint32_t seed);
  template <typename NativeT>
  std::unique_ptr<Array2D<NativeT>> CreatePseudorandomR2(const int rows,
                                                         const int cols,
                                                         NativeT min_value,
                                                         NativeT max_value,
                                                         uint32_t seed);

  // Creates a (rows x cols) array filled in the following form:
  //
  //  [      0              1 ...                   cols-1]
  //  [  1,000          1,001 ...          1000.0 + cols-1]
  //  [    ...            ... ...                      ...]
  //  [(rows-1)*1000.0    ... ... (rows-1)*1000.0 + cols-1]
  //
  // If provided, offset is added uniformly to every element (e.g. an offset of
  // 64 would cause 0 in the above to be 64, 1 to be 65, 1000 to be 1064, etc.)
  std::unique_ptr<Array2D<float>> CreatePatternedMatrix(const int rows,
                                                        const int cols,
                                                        float offset = 0.0);

  // Creates a (rows x cols) array as above, padded out to
  // (rows_padded x cols_padded) with zeroes.  Requires rows_padded >= rows
  // and cols_padded > cols.
  std::unique_ptr<Array2D<float>> CreatePatternedMatrixWithZeroPadding(
      const int rows, const int cols, const int rows_padded,
      const int cols_padded);

  // Creates a parameter instruction, transfers the literal for the parameter to
  // server, then stores into "data_handle" the global handle for that
  // parameter. When the use_bfloat16 flag is set but the literal has F32
  // elements, the literal will be converted to BF16 before being transferred.
  StatusOr<std::unique_ptr<GlobalData>> CreateParameterAndTransferLiteral(
      int64_t parameter_number, const Literal& literal, const std::string& name,
      XlaBuilder* builder, XlaOp* data_handle);

  // As above, but the caller can specify the device that the literal is
  // transferred to. If device_handle is nullptr, the literal will be
  // transferred to the default device.
  StatusOr<std::unique_ptr<GlobalData>> CreateParameterAndTransferLiteral(
      int64_t parameter_number, const Literal& literal, const std::string& name,
      const DeviceHandle* device_handle, XlaBuilder* builder,
      XlaOp* data_handle);

  // Creates a parameter instruction and sets the value that will be passed to
  // the computation as specified. This function must be used for all parameters
  // or none and no parameters must be passed when invoking the computation if
  // using this mechanism. If using this mechanism, then each parameter must be
  // set exactly once. The first added parameter gets index 0, then 1 and so on.
  XlaOp AddParam(const Literal& argument, XlaBuilder* builder);

  template <class T>
  XlaOp AddParam(const Array<T>& argument, XlaBuilder* builder) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPStestsPSclient_library_test_baseDTh mht_3(mht_3_v, 476, "", "./tensorflow/compiler/xla/tests/client_library_test_base.h", "AddParam");

    return AddParam(LiteralUtil::CreateFromArray(argument), builder);
  }

  // Creates a constant instruction with the given literal. When the
  // use_bfloat16 flag is set but the literal has F32 elements, the elements
  // will be converted to BF16s.
  XlaOp CreateConstantFromLiteral(const Literal& literal, XlaBuilder* builder);

  // Creates a constant instruction with the given array. When the use_bfloat16
  // flag is set but the array has float elements, the elements will be
  // converted to bfloat16s.

  template <typename NativeT>
  XlaOp CreateConstantFromArray(const Array<NativeT>& array,
                                XlaBuilder* builder) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPStestsPSclient_library_test_baseDTh mht_4(mht_4_v, 494, "", "./tensorflow/compiler/xla/tests/client_library_test_base.h", "CreateConstantFromArray");

    return CreateConstantFromLiteral(LiteralUtil::CreateFromArray(array),
                                     builder);
  }

  // Same as CreateConstantFromArray, but for scalars.
  template <typename NativeT>
  XlaOp CreateConstantFromScalar(NativeT value, XlaBuilder* builder) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPStestsPSclient_library_test_baseDTh mht_5(mht_5_v, 504, "", "./tensorflow/compiler/xla/tests/client_library_test_base.h", "CreateConstantFromScalar");

    return CreateConstantFromLiteral(LiteralUtil::CreateR0<NativeT>(value),
                                     builder);
  }

  // Creates a parameter instruction that wraps a given value and then stores
  // into "data_handle" the global handle for that parameter.
  //
  // "parameter_number" is the parameter number.
  // "name" is the name of the parameter instruction.
  //
  // When the use_bfloat16 flag is set but NativeT is float, the data will be
  // converted to bfloat16.
  template <typename NativeT>
  std::unique_ptr<GlobalData> CreateR0Parameter(NativeT value,
                                                int64_t parameter_number,
                                                const std::string& name,
                                                XlaBuilder* builder,
                                                XlaOp* data_handle);

  // Creates a parameter instruction that wraps the given values and then stores
  // into "data_handle" the global handle for that parameter.
  //
  // "parameter_number" is the parameter number.
  // "name" is the name of the parameter instruction.
  //
  // When the use_bfloat16 flag is set but NativeT is float, the data will be
  // converted to bfloat16.
  template <typename NativeT>
  std::unique_ptr<GlobalData> CreateR1Parameter(
      absl::Span<const NativeT> values, int64_t parameter_number,
      const std::string& name, XlaBuilder* builder, XlaOp* data_handle);

  // Creates a parameter instruction that wraps the given constant array
  // "array_2d" and then stores it to the global handle for that parameter
  // "data_handle".
  //
  // "parameter_number" is the parameter number.
  // "name" is the name of the parameter instruction.
  //
  // When the use_bfloat16 flag is set but NativeT is float, the data will be
  // converted to bfloat16.
  template <typename NativeT>
  std::unique_ptr<GlobalData> CreateR2Parameter(
      const Array2D<NativeT>& array_2d, int64_t parameter_number,
      const std::string& name, XlaBuilder* builder, XlaOp* data_handle);

  // Creates a parameter instruction that wraps the given constant array
  // "array_3d" and then stores it to the global handle for that parameter
  // "data_handle".
  //
  // "parameter_number" is the parameter number.
  // "name" is the name of the parameter instruction.
  //
  // When the use_bfloat16 flag is set but NativeT is float, the data will be
  // converted to bfloat16.
  template <typename NativeT>
  std::unique_ptr<GlobalData> CreateR3Parameter(
      const Array3D<NativeT>& array_3d, int64_t parameter_number,
      const std::string& name, XlaBuilder* builder, XlaOp* data_handle);

  // Creates a parameter instruction that wraps the given constant array
  // "array_4d" and then stores it to the global handle for that parameter
  // "data_handle".
  //
  // "parameter_number" is the parameter number.
  // "name" is the name of the parameter instruction.
  //
  // When the use_bfloat16 flag is set but NativeT is float, the data will be
  // converted to bfloat16.
  template <typename NativeT>
  std::unique_ptr<GlobalData> CreateR4Parameter(
      const Array4D<NativeT>& array_4d, int64_t parameter_number,
      const std::string& name, XlaBuilder* builder, XlaOp* data_handle);

  template <typename NativeT>
  std::unique_ptr<GlobalData> CreateParameter(const Array<NativeT>& array_4d,
                                              int64_t parameter_number,
                                              const std::string& name,
                                              XlaBuilder* builder,
                                              XlaOp* data_handle);

  // Getter and setter for the use_bfloat16 flag, which indicates whether to run
  // tests with all float-type input/output converted to bfloat16.
  bool use_bfloat16() const {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPStestsPSclient_library_test_baseDTh mht_6(mht_6_v, 591, "", "./tensorflow/compiler/xla/tests/client_library_test_base.h", "use_bfloat16");
 return use_bfloat16_; }
  void set_use_bfloat16(bool value) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPStestsPSclient_library_test_baseDTh mht_7(mht_7_v, 595, "", "./tensorflow/compiler/xla/tests/client_library_test_base.h", "set_use_bfloat16");
 use_bfloat16_ = value; }

  // The float type used in this test, BF16 or F32 according to use_bfloat16.
  PrimitiveType FloatType() const {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPStestsPSclient_library_test_baseDTh mht_8(mht_8_v, 601, "", "./tensorflow/compiler/xla/tests/client_library_test_base.h", "FloatType");
 return use_bfloat16_ ? BF16 : F32; }

  // Executes the computation and calculates the expected reference value using
  // the reference client. Returns two literals in the order of (expected,
  // actual).
  StatusOr<std::pair<Literal, Literal>> ComputeValueAndReference(
      XlaBuilder* builder, absl::Span<const Literal> arguments);

  // Converts an f32 literal to bf16 if use_bfloat16_ is true.
  Literal MaybeConvertLiteralToBfloat16(const Literal& literal);

  LocalClient* client_;
  LocalClient* ref_client_;  // To compute reference result.
  ExecutionOptions execution_options_;

 private:
  Status ComputeAndCompareLiteralWithAllOutputLayouts(
      const xla::XlaComputation& computation, const Literal& expected,
      absl::Span<GlobalData* const> arguments,
      const std::function<void(const Literal& actual,
                               const std::string& error_message)>&
          verify_output);
  Status ComputeAndCompareLiteralWithAllInputLayouts(
      const xla::XlaComputation& computation, const Literal& expected,
      absl::Span<GlobalData* const> arguments,
      const std::function<void(const Literal& actual,
                               const std::string& error_message)>&
          verify_output,
      const Shape* output_with_layout = nullptr);

  // Converts an f32 shape to bf16 if use_bfloat16_ is true.
  Shape MaybeConvertShapeToBfloat16(const Shape& shape);

  // Whether to run tests with all float-type input/output converted to
  // bfloat16.
  bool use_bfloat16_ = false;

  // Arguments to be passed to the computation when it runs.
  std::vector<Literal> arguments_;
};

template <typename NativeT>
void ClientLibraryTestBase::ComputeAndCompareR0(
    XlaBuilder* builder, NativeT expected,
    absl::Span<GlobalData* const> arguments) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPStestsPSclient_library_test_baseDTh mht_9(mht_9_v, 648, "", "./tensorflow/compiler/xla/tests/client_library_test_base.h", "ClientLibraryTestBase::ComputeAndCompareR0");

  Literal expected_literal = LiteralUtil::CreateR0<NativeT>(expected);
  ClientLibraryTestBase::ComputeAndCompareLiteral(builder, expected_literal,
                                                  arguments);
}

template <typename NativeT>
void ClientLibraryTestBase::ComputeAndCompareR0(
    XlaBuilder* builder, NativeT expected,
    absl::Span<GlobalData* const> arguments, ErrorSpec error) {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPStestsPSclient_library_test_baseDTh mht_10(mht_10_v, 660, "", "./tensorflow/compiler/xla/tests/client_library_test_base.h", "ClientLibraryTestBase::ComputeAndCompareR0");

  static_assert(std::is_same<NativeT, float>::value ||
                    std::is_same<NativeT, double>::value ||
                    std::is_same<NativeT, bfloat16>::value ||
                    std::is_same<NativeT, half>::value ||
                    std::is_same<NativeT, complex64>::value ||
                    std::is_same<NativeT, complex128>::value,
                "Float or complex type required when specifying an ErrorSpec");
  Literal expected_literal = LiteralUtil::CreateR0<NativeT>(expected);
  ClientLibraryTestBase::ComputeAndCompareLiteral(builder, expected_literal,
                                                  arguments, error);
}

template <typename NativeT>
void ClientLibraryTestBase::ComputeAndCompareR1(
    XlaBuilder* builder, absl::Span<const NativeT> expected,
    absl::Span<GlobalData* const> arguments) {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPStestsPSclient_library_test_baseDTh mht_11(mht_11_v, 679, "", "./tensorflow/compiler/xla/tests/client_library_test_base.h", "ClientLibraryTestBase::ComputeAndCompareR1");

  Literal expected_literal = LiteralUtil::CreateR1<NativeT>(expected);
  ClientLibraryTestBase::ComputeAndCompareLiteral(builder, expected_literal,
                                                  arguments);
}

template <typename NativeT>
void ClientLibraryTestBase::ComputeAndCompareR1(
    XlaBuilder* builder, absl::Span<const NativeT> expected,
    absl::Span<GlobalData* const> arguments, ErrorSpec error) {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPStestsPSclient_library_test_baseDTh mht_12(mht_12_v, 691, "", "./tensorflow/compiler/xla/tests/client_library_test_base.h", "ClientLibraryTestBase::ComputeAndCompareR1");

  static_assert(std::is_same<NativeT, float>::value ||
                    std::is_same<NativeT, double>::value ||
                    std::is_same<NativeT, bfloat16>::value ||
                    std::is_same<NativeT, half>::value ||
                    std::is_same<NativeT, complex64>::value ||
                    std::is_same<NativeT, complex128>::value,
                "Float or complex type required when specifying an ErrorSpec");
  Literal expected_literal = LiteralUtil::CreateR1<NativeT>(expected);
  ClientLibraryTestBase::ComputeAndCompareLiteral(builder, expected_literal,
                                                  arguments, error);
}

template <typename NativeT>
void ClientLibraryTestBase::ComputeAndCompareR2(
    XlaBuilder* builder, const Array2D<NativeT>& expected,
    absl::Span<GlobalData* const> arguments) {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPStestsPSclient_library_test_baseDTh mht_13(mht_13_v, 710, "", "./tensorflow/compiler/xla/tests/client_library_test_base.h", "ClientLibraryTestBase::ComputeAndCompareR2");

  Literal expected_literal =
      LiteralUtil::CreateR2FromArray2D<NativeT>(expected);
  ClientLibraryTestBase::ComputeAndCompareLiteral(builder, expected_literal,
                                                  arguments);
}

template <typename NativeT>
void ClientLibraryTestBase::ComputeAndCompareR2(
    XlaBuilder* builder, const Array2D<NativeT>& expected,
    absl::Span<GlobalData* const> arguments, ErrorSpec error) {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPStestsPSclient_library_test_baseDTh mht_14(mht_14_v, 723, "", "./tensorflow/compiler/xla/tests/client_library_test_base.h", "ClientLibraryTestBase::ComputeAndCompareR2");

  static_assert(std::is_same<NativeT, float>::value ||
                    std::is_same<NativeT, double>::value ||
                    std::is_same<NativeT, bfloat16>::value ||
                    std::is_same<NativeT, half>::value ||
                    std::is_same<NativeT, complex64>::value ||
                    std::is_same<NativeT, complex128>::value,
                "Float or complex type required when specifying an ErrorSpec");
  Literal expected_literal =
      LiteralUtil::CreateR2FromArray2D<NativeT>(expected);
  ClientLibraryTestBase::ComputeAndCompareLiteral(builder, expected_literal,
                                                  arguments, error);
}

template <typename NativeT>
void ClientLibraryTestBase::ComputeAndCompareR3(
    XlaBuilder* builder, const Array3D<NativeT>& expected,
    absl::Span<GlobalData* const> arguments) {
   std::vector<std::string> mht_15_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPStestsPSclient_library_test_baseDTh mht_15(mht_15_v, 743, "", "./tensorflow/compiler/xla/tests/client_library_test_base.h", "ClientLibraryTestBase::ComputeAndCompareR3");

  Literal expected_literal =
      LiteralUtil::CreateR3FromArray3D<NativeT>(expected);
  ClientLibraryTestBase::ComputeAndCompareLiteral(builder, expected_literal,
                                                  arguments);
}

template <typename NativeT>
void ClientLibraryTestBase::ComputeAndCompareR3(
    XlaBuilder* builder, const Array3D<NativeT>& expected,
    absl::Span<GlobalData* const> arguments, ErrorSpec error) {
   std::vector<std::string> mht_16_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPStestsPSclient_library_test_baseDTh mht_16(mht_16_v, 756, "", "./tensorflow/compiler/xla/tests/client_library_test_base.h", "ClientLibraryTestBase::ComputeAndCompareR3");

  static_assert(std::is_same<NativeT, float>::value ||
                    std::is_same<NativeT, double>::value ||
                    std::is_same<NativeT, bfloat16>::value ||
                    std::is_same<NativeT, half>::value ||
                    std::is_same<NativeT, complex64>::value ||
                    std::is_same<NativeT, complex128>::value,
                "Float or complex type required when specifying an ErrorSpec");
  Literal expected_literal =
      LiteralUtil::CreateR3FromArray3D<NativeT>(expected);
  ClientLibraryTestBase::ComputeAndCompareLiteral(builder, expected_literal,
                                                  arguments, error);
}

template <typename NativeT>
void ClientLibraryTestBase::ComputeAndCompareR4(
    XlaBuilder* builder, const Array4D<NativeT>& expected,
    absl::Span<GlobalData* const> arguments) {
   std::vector<std::string> mht_17_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPStestsPSclient_library_test_baseDTh mht_17(mht_17_v, 776, "", "./tensorflow/compiler/xla/tests/client_library_test_base.h", "ClientLibraryTestBase::ComputeAndCompareR4");

  Literal expected_literal =
      LiteralUtil::CreateR4FromArray4D<NativeT>(expected);
  ClientLibraryTestBase::ComputeAndCompareLiteral(builder, expected_literal,
                                                  arguments);
}

template <typename NativeT>
void ClientLibraryTestBase::ComputeAndCompareR4(
    XlaBuilder* builder, const Array4D<NativeT>& expected,
    absl::Span<GlobalData* const> arguments, ErrorSpec error) {
   std::vector<std::string> mht_18_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPStestsPSclient_library_test_baseDTh mht_18(mht_18_v, 789, "", "./tensorflow/compiler/xla/tests/client_library_test_base.h", "ClientLibraryTestBase::ComputeAndCompareR4");

  static_assert(std::is_same<NativeT, float>::value ||
                    std::is_same<NativeT, double>::value ||
                    std::is_same<NativeT, bfloat16>::value ||
                    std::is_same<NativeT, half>::value ||
                    std::is_same<NativeT, complex64>::value ||
                    std::is_same<NativeT, complex128>::value,
                "Float or complex type required when specifying an ErrorSpec");
  Literal expected_literal =
      LiteralUtil::CreateR4FromArray4D<NativeT>(expected);
  ClientLibraryTestBase::ComputeAndCompareLiteral(builder, expected_literal,
                                                  arguments, error);
}

template <typename NativeT>
void ClientLibraryTestBase::ComputeAndCompare(
    XlaBuilder* builder, const Array<NativeT>& expected,
    absl::Span<GlobalData* const> arguments) {
   std::vector<std::string> mht_19_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPStestsPSclient_library_test_baseDTh mht_19(mht_19_v, 809, "", "./tensorflow/compiler/xla/tests/client_library_test_base.h", "ClientLibraryTestBase::ComputeAndCompare");

  Literal expected_literal = LiteralUtil::CreateFromArray<NativeT>(expected);
  ClientLibraryTestBase::ComputeAndCompareLiteral(builder, expected_literal,
                                                  arguments);
}

template <typename NativeT>
void ClientLibraryTestBase::ComputeAndCompare(
    XlaBuilder* builder, const Array<NativeT>& expected,
    absl::Span<GlobalData* const> arguments, ErrorSpec error) {
   std::vector<std::string> mht_20_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPStestsPSclient_library_test_baseDTh mht_20(mht_20_v, 821, "", "./tensorflow/compiler/xla/tests/client_library_test_base.h", "ClientLibraryTestBase::ComputeAndCompare");

  static_assert(std::is_same<NativeT, float>::value ||
                    std::is_same<NativeT, double>::value ||
                    std::is_same<NativeT, bfloat16>::value ||
                    std::is_same<NativeT, half>::value ||
                    std::is_same<NativeT, complex64>::value ||
                    std::is_same<NativeT, complex128>::value,
                "Float or complex type required when specifying an ErrorSpec");
  Literal expected_literal = LiteralUtil::CreateFromArray<NativeT>(expected);
  ClientLibraryTestBase::ComputeAndCompareLiteral(builder, expected_literal,
                                                  arguments, error);
}

template <typename NativeT>
std::unique_ptr<GlobalData> ClientLibraryTestBase::CreateR0Parameter(
    NativeT value, int64_t parameter_number, const std::string& name,
    XlaBuilder* builder, XlaOp* data_handle) {
   std::vector<std::string> mht_21_v;
   mht_21_v.push_back("name: \"" + name + "\"");
   MHTracer_DTPStensorflowPScompilerPSxlaPStestsPSclient_library_test_baseDTh mht_21(mht_21_v, 841, "", "./tensorflow/compiler/xla/tests/client_library_test_base.h", "ClientLibraryTestBase::CreateR0Parameter");

  Literal literal = LiteralUtil::CreateR0(value);
  if (use_bfloat16_ && literal.shape().element_type() == F32) {
    literal = LiteralUtil::ConvertF32ToBF16(literal);
  }
  std::unique_ptr<GlobalData> data =
      client_->TransferToServer(literal).ConsumeValueOrDie();
  *data_handle = Parameter(builder, parameter_number, literal.shape(), name);
  return data;
}

template <typename NativeT>
std::unique_ptr<GlobalData> ClientLibraryTestBase::CreateR1Parameter(
    absl::Span<const NativeT> values, int64_t parameter_number,
    const std::string& name, XlaBuilder* builder, XlaOp* data_handle) {
   std::vector<std::string> mht_22_v;
   mht_22_v.push_back("name: \"" + name + "\"");
   MHTracer_DTPStensorflowPScompilerPSxlaPStestsPSclient_library_test_baseDTh mht_22(mht_22_v, 859, "", "./tensorflow/compiler/xla/tests/client_library_test_base.h", "ClientLibraryTestBase::CreateR1Parameter");

  Literal literal = LiteralUtil::CreateR1(values);
  if (use_bfloat16_ && literal.shape().element_type() == F32) {
    literal = LiteralUtil::ConvertF32ToBF16(literal);
  }
  std::unique_ptr<GlobalData> data =
      client_->TransferToServer(literal).ConsumeValueOrDie();
  *data_handle = Parameter(builder, parameter_number, literal.shape(), name);
  return data;
}

template <typename NativeT>
std::unique_ptr<GlobalData> ClientLibraryTestBase::CreateR2Parameter(
    const Array2D<NativeT>& array_2d, int64_t parameter_number,
    const std::string& name, XlaBuilder* builder, XlaOp* data_handle) {
   std::vector<std::string> mht_23_v;
   mht_23_v.push_back("name: \"" + name + "\"");
   MHTracer_DTPStensorflowPScompilerPSxlaPStestsPSclient_library_test_baseDTh mht_23(mht_23_v, 877, "", "./tensorflow/compiler/xla/tests/client_library_test_base.h", "ClientLibraryTestBase::CreateR2Parameter");

  Literal literal = LiteralUtil::CreateR2FromArray2D(array_2d);
  if (use_bfloat16_ && literal.shape().element_type() == F32) {
    literal = LiteralUtil::ConvertF32ToBF16(literal);
  }
  std::unique_ptr<GlobalData> data =
      client_->TransferToServer(literal).ConsumeValueOrDie();
  *data_handle = Parameter(builder, parameter_number, literal.shape(), name);
  return data;
}

template <typename NativeT>
std::unique_ptr<GlobalData> ClientLibraryTestBase::CreateR3Parameter(
    const Array3D<NativeT>& array_3d, int64_t parameter_number,
    const std::string& name, XlaBuilder* builder, XlaOp* data_handle) {
   std::vector<std::string> mht_24_v;
   mht_24_v.push_back("name: \"" + name + "\"");
   MHTracer_DTPStensorflowPScompilerPSxlaPStestsPSclient_library_test_baseDTh mht_24(mht_24_v, 895, "", "./tensorflow/compiler/xla/tests/client_library_test_base.h", "ClientLibraryTestBase::CreateR3Parameter");

  Literal literal = LiteralUtil::CreateR3FromArray3D(array_3d);
  if (use_bfloat16_ && literal.shape().element_type() == F32) {
    literal = LiteralUtil::ConvertF32ToBF16(literal);
  }
  std::unique_ptr<GlobalData> data =
      client_->TransferToServer(literal).ConsumeValueOrDie();
  *data_handle = Parameter(builder, parameter_number, literal.shape(), name);
  return data;
}

template <typename NativeT>
std::unique_ptr<GlobalData> ClientLibraryTestBase::CreateR4Parameter(
    const Array4D<NativeT>& array_4d, int64_t parameter_number,
    const std::string& name, XlaBuilder* builder, XlaOp* data_handle) {
   std::vector<std::string> mht_25_v;
   mht_25_v.push_back("name: \"" + name + "\"");
   MHTracer_DTPStensorflowPScompilerPSxlaPStestsPSclient_library_test_baseDTh mht_25(mht_25_v, 913, "", "./tensorflow/compiler/xla/tests/client_library_test_base.h", "ClientLibraryTestBase::CreateR4Parameter");

  Literal literal = LiteralUtil::CreateR4FromArray4D(array_4d);
  if (use_bfloat16_ && literal.shape().element_type() == F32) {
    literal = LiteralUtil::ConvertF32ToBF16(literal);
  }
  std::unique_ptr<GlobalData> data =
      client_->TransferToServer(literal).ConsumeValueOrDie();
  *data_handle = Parameter(builder, parameter_number, literal.shape(), name);
  return data;
}

template <typename NativeT>
std::unique_ptr<GlobalData> ClientLibraryTestBase::CreateParameter(
    const Array<NativeT>& array, int64_t parameter_number,
    const std::string& name, XlaBuilder* builder, XlaOp* data_handle) {
   std::vector<std::string> mht_26_v;
   mht_26_v.push_back("name: \"" + name + "\"");
   MHTracer_DTPStensorflowPScompilerPSxlaPStestsPSclient_library_test_baseDTh mht_26(mht_26_v, 931, "", "./tensorflow/compiler/xla/tests/client_library_test_base.h", "ClientLibraryTestBase::CreateParameter");

  Literal literal = LiteralUtil::CreateFromArray(array);
  if (use_bfloat16_ && literal.shape().element_type() == F32) {
    literal = LiteralUtil::ConvertF32ToBF16(literal);
  }
  std::unique_ptr<GlobalData> data =
      client_->TransferToServer(literal).ConsumeValueOrDie();
  *data_handle = Parameter(builder, parameter_number, literal.shape(), name);
  return data;
}

template <typename NativeT>
std::vector<NativeT> ClientLibraryTestBase::CreatePseudorandomR1(
    const int width, NativeT min_value, NativeT max_value, uint32_t seed) {
   std::vector<std::string> mht_27_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPStestsPSclient_library_test_baseDTh mht_27(mht_27_v, 947, "", "./tensorflow/compiler/xla/tests/client_library_test_base.h", "ClientLibraryTestBase::CreatePseudorandomR1");

  std::vector<NativeT> result(width);
  PseudorandomGenerator<NativeT> generator(min_value, max_value, seed);
  for (int i = 0; i < width; ++i) {
    result[i] = generator.get();
  }
  return result;
}

template <typename NativeT>
std::unique_ptr<Array2D<NativeT>> ClientLibraryTestBase::CreatePseudorandomR2(
    const int rows, const int cols, NativeT min_value, NativeT max_value,
    uint32_t seed) {
   std::vector<std::string> mht_28_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPStestsPSclient_library_test_baseDTh mht_28(mht_28_v, 962, "", "./tensorflow/compiler/xla/tests/client_library_test_base.h", "ClientLibraryTestBase::CreatePseudorandomR2");

  auto result = absl::make_unique<Array2D<NativeT>>(rows, cols);
  PseudorandomGenerator<NativeT> generator(min_value, max_value, seed);
  for (int y = 0; y < rows; ++y) {
    for (int x = 0; x < cols; ++x) {
      (*result)(y, x) = generator.get();
    }
  }
  return result;
}

}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_TESTS_CLIENT_LIBRARY_TEST_BASE_H_
