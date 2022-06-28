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
class MHTracer_DTPStensorflowPScompilerPSmlirPStfrtPStransformsPSlmhlo_to_gpuPSfft_patternDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSmlirPStfrtPStransformsPSlmhlo_to_gpuPSfft_patternDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSmlirPStfrtPStransformsPSlmhlo_to_gpuPSfft_patternDTcc() {
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

// Copyright 2022 The TensorFlow Runtime Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// Pattern to lower lmhlo.fft op to tfrt dialect.
#include <cstdint>
#include <functional>
#include <numeric>
#include <string>
#include <utility>

#include "mlir-hlo/Dialect/lhlo_gpu/IR/lhlo_gpu_ops.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/Support/ScopedPrinter.h"
#include "tensorflow/compiler/mlir/hlo/include/mlir-hlo/Dialect/lhlo/IR/lhlo_ops.h"
#include "tensorflow/compiler/mlir/tfrt/transforms/lmhlo_to_gpu/pattern_utils.h"
#include "tensorflow/compiler/mlir/xla/type_to_shape.h"
#include "tensorflow/compiler/xla/service/gpu/ir_emission_utils.h"
#include "tensorflow/compiler/xla/shape.h"
#include "tfrt/gpu/kernels/gpu_ops.h"  // from @tf_runtime
#include "tfrt/gpu/passes/passes.h"  // from @tf_runtime
#include "tfrt/gpu/wrapper/cufft_wrapper.h"  // from @tf_runtime
#include "tfrt/basic_kernels/opdefs/types.h"  // from @tf_runtime

namespace tensorflow {

static llvm::Expected<tfrt::gpu::wrapper::FftType> GetFftType(
    llvm::StringRef type, bool double_precision) {
  llvm::Expected<int> value =
      llvm::StringSwitch<llvm::Expected<int>>(type)
          .Case("FFT", double_precision ? CUFFT_Z2Z : CUFFT_C2C)
          .Case("IFFT", double_precision ? CUFFT_Z2Z : CUFFT_C2C)
          .Case("RFFT", double_precision ? CUFFT_D2Z : CUFFT_R2C)
          .Case("IRFFT", double_precision ? CUFFT_Z2D : CUFFT_C2R)
          .Default(tfrt::MakeStringError("Unsupported FFT type: ", type));
  if (!value) return value.takeError();
  return tfrt::gpu::wrapper::FftType(*value, kGpuTargetPlatform);
}

static llvm::Expected<tfrt::gpu::wrapper::FftDirection> GetFftDirection(
    llvm::StringRef type) {
  llvm::Expected<int> value =
      llvm::StringSwitch<llvm::Expected<int>>(type)
          .Case("FFT", CUFFT_FORWARD)
          .Case("IFFT", CUFFT_INVERSE)
          .Case("RFFT", CUFFT_FORWARD)
          .Case("IRFFT", CUFFT_INVERSE)
          .Default(tfrt::MakeStringError("Unsupported FFT type: ", type));
  if (!value) return value.takeError();
  return tfrt::gpu::wrapper::FftDirection(*value, kGpuTargetPlatform);
}

namespace {

struct FftRewritePattern
    : tfrt::gpu::GpuAsyncOpConversionPattern<lmhlo::FftOp> {
  using tfrt::gpu::GpuAsyncOpConversionPattern<lmhlo::FftOp>::OpAdaptor;
  using tfrt::gpu::GpuAsyncOpConversionPattern<
      lmhlo::FftOp>::GpuAsyncOpConversionPattern;
  FailureOr<Value> matchAndRewriteOp(
      lmhlo::FftOp op, OpAdaptor adaptor, Value chain, Value stream,
      ConversionPatternRewriter& rewriter) const override {
    xla::Shape input_shape = xla::gpu::GetShape(op.operand());
    xla::Shape output_shape = xla::gpu::GetShape(op.output());
    if (input_shape.is_dynamic() || output_shape.is_dynamic())
      return rewriter.notifyMatchFailure(op, "expected static shapes");
    if (!xla::LayoutUtil::IsMonotonicWithDim0Major(input_shape.layout()) ||
        !xla::LayoutUtil::IsMonotonicWithDim0Major(output_shape.layout())) {
      return rewriter.notifyMatchFailure(op, "expected dense row-major");
    }

    bool double_precision = input_shape.element_type() == xla::F64 ||
                            input_shape.element_type() == xla::C128;
    auto type = GetFftType(mlir::mhlo::stringifyFftType(adaptor.fft_type()),
                           double_precision);
    auto direction =
        GetFftDirection(mlir::mhlo::stringifyFftType(adaptor.fft_type()));
    if (!type || !direction) {
      auto error = joinErrors(type.takeError(), direction.takeError());
      return rewriter.notifyMatchFailure(op, llvm::toString(std::move(error)));
    }

    llvm::SmallVector<int64_t, 3> dimensions;
    llvm::copy(op.fft_length().getValues<int64_t>(),
               std::back_inserter(dimensions));
    int rank = dimensions.size();

    auto batch_dims = input_shape.dimensions();
    uint64_t batch =
        std::accumulate(batch_dims.begin(), batch_dims.end() - rank, 1,
                        std::multiplies<int64_t>());

    auto get_strides = [](absl::Span<const int64_t> dims) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStfrtPStransformsPSlmhlo_to_gpuPSfft_patternDTcc mht_0(mht_0_v, 273, "", "./tensorflow/compiler/mlir/tfrt/transforms/lmhlo_to_gpu/fft_pattern.cc", "lambda");

      llvm::SmallVector<int64_t, 4> strides(dims.size() + 1, 1);
      std::partial_sum(dims.rbegin(), dims.rend(), strides.rbegin() + 1,
                       std::multiplies<int64_t>());
      return strides;
    };
    llvm::SmallVector<int64_t, 4> input_strides =
        get_strides(input_shape.dimensions().last(rank));
    llvm::SmallVector<int64_t, 4> output_strides =
        get_strides(output_shape.dimensions().last(rank));

    mlir::Location loc = op->getLoc();
    Value context = rewriter.create<tfrt::gpu::StreamGetContextOp>(loc, stream);

    auto handle = rewriter.create<tfrt::gpu::FftCreateOp>(
        loc, context, *type, batch, rewriter.getI64ArrayAttr(dimensions),
        rewriter.getI64ArrayAttr(input_strides),
        rewriter.getI64ArrayAttr(output_strides));

    // Note: we could determine the workspace size during lowering similar to
    // convolutions because the dimensions are static. But it's unclear if we
    // really want the compiler to depend on cuFFT/hipFFT, and the expensive
    // part is the allocation, which is currently not hoisted.
    mlir::Value workspace_size =
        rewriter.create<tfrt::gpu::FftGetWorkspaceSizeOp>(loc, handle);
    mlir::Value allocator =
        rewriter.create<tfrt::gpu::AllocatorCreateOp>(loc, context);
    mlir::Value workspace = rewriter.create<tfrt::gpu::MemAllocateOp>(
        loc, allocator, stream, workspace_size, chain);

    chain = rewriter.create<tfrt::gpu::FftExecuteOp>(
        loc, stream, handle, adaptor.operand(), adaptor.output(), workspace,
        *direction, chain);

    rewriter.eraseOp(op);
    return chain;
  }
};

}  // namespace

void populateFftConversionPattern(RewritePatternSet& patterns,
                                  TypeConverter& converter) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStfrtPStransformsPSlmhlo_to_gpuPSfft_patternDTcc mht_1(mht_1_v, 318, "", "./tensorflow/compiler/mlir/tfrt/transforms/lmhlo_to_gpu/fft_pattern.cc", "populateFftConversionPattern");

  patterns.add<FftRewritePattern>(converter, patterns.getContext());
}

}  // namespace tensorflow
