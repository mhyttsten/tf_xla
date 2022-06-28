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
class MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPScPSc_api_unified_experimental_mlirDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPScPSc_api_unified_experimental_mlirDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPScPSc_api_unified_experimental_mlirDTcc() {
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

/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#include <cstddef>
#include <memory>

#include "absl/strings/str_cat.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/iterator_range.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/IR/Location.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/IR/Operation.h"  // from @llvm-project
#include "mlir/IR/OperationSupport.h"  // from @llvm-project
#include "mlir/IR/TypeUtilities.h"  // from @llvm-project
#include "mlir/Pass/PassManager.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "tensorflow/c/c_api.h"
#include "tensorflow/c/eager/abstract_context.h"
#include "tensorflow/c/eager/abstract_operation.h"
#include "tensorflow/c/eager/abstract_tensor_handle.h"
#include "tensorflow/c/eager/c_api.h"
#include "tensorflow/c/eager/c_api_internal.h"
#include "tensorflow/c/eager/c_api_unified_experimental_internal.h"
#include "tensorflow/c/tensor_interface.h"
#include "tensorflow/c/tf_status.h"
#include "tensorflow/c/tf_status_helper.h"
#include "tensorflow/c/tf_status_internal.h"
#include "tensorflow/compiler/mlir/tensorflow/dialect_registration.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_device.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_executor.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_types.h"
#include "tensorflow/compiler/mlir/tensorflow/transforms/passes.h"
#include "tensorflow/compiler/mlir/tensorflow/translate/export_graphdef.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/convert_type.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/dump_mlir_util.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/error_util.h"
#include "tensorflow/core/framework/node_def_util.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/lib/llvm_rtti/llvm_rtti.h"
#include "tensorflow/core/platform/errors.h"

namespace mlir {
namespace TF {
using tensorflow::AbstractFunction;
using tensorflow::AbstractOperation;
using tensorflow::AbstractTensorHandle;
using tensorflow::AbstractTensorInterface;
using tensorflow::dyn_cast;
using tensorflow::OutputList;
using tensorflow::string;
using tensorflow::errors::FailedPrecondition;
using tensorflow::errors::InvalidArgument;
using tensorflow::errors::Unimplemented;
using tensorflow::tracing::TracingContext;
using tensorflow::tracing::TracingOperation;
using tensorflow::tracing::TracingTensorHandle;

namespace {

void RegisterDialects(mlir::MLIRContext& ctx) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPScPSc_api_unified_experimental_mlirDTcc mht_0(mht_0_v, 248, "", "./tensorflow/compiler/mlir/tensorflow/c/c_api_unified_experimental_mlir.cc", "RegisterDialects");

  mlir::DialectRegistry registry;
  mlir::RegisterAllTensorFlowDialects(registry);
  ctx.appendDialectRegistry(registry);
  ctx.loadAllAvailableDialects();
}

Status ConvertDataTypeToTensor(tensorflow::DataType dtype, Builder builder,
                               Type* type) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPScPSc_api_unified_experimental_mlirDTcc mht_1(mht_1_v, 259, "", "./tensorflow/compiler/mlir/tensorflow/c/c_api_unified_experimental_mlir.cc", "ConvertDataTypeToTensor");

  Status s = tensorflow::ConvertDataType(dtype, builder, type);
  if (s.ok()) *type = UnrankedTensorType::get(*type);
  return s;
}

class MlirTensor : public TracingTensorHandle {
 public:
  explicit MlirTensor(Value value)
      : TracingTensorHandle(kMlir), value_(value) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPScPSc_api_unified_experimental_mlirDTcc mht_2(mht_2_v, 271, "", "./tensorflow/compiler/mlir/tensorflow/c/c_api_unified_experimental_mlir.cc", "MlirTensor");
}

  tensorflow::DataType DataType() const override {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPScPSc_api_unified_experimental_mlirDTcc mht_3(mht_3_v, 276, "", "./tensorflow/compiler/mlir/tensorflow/c/c_api_unified_experimental_mlir.cc", "DataType");

    tensorflow::DataType type;
    Status s = ConvertToDataType(value_.getType(), &type);
    if (!s.ok()) {
      return tensorflow::DT_INVALID;
    }
    return type;
  }

  tensorflow::Status Shape(
      tensorflow::PartialTensorShape* shape) const override {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPScPSc_api_unified_experimental_mlirDTcc mht_4(mht_4_v, 289, "", "./tensorflow/compiler/mlir/tensorflow/c/c_api_unified_experimental_mlir.cc", "Shape");

    // TODO(b/173074167): Implement this and enable tests in
    // unified_api_test.cc.
    return Unimplemented("MlirTensor::Shape is not implemented yet.");
  }

  Value getValue() {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPScPSc_api_unified_experimental_mlirDTcc mht_5(mht_5_v, 298, "", "./tensorflow/compiler/mlir/tensorflow/c/c_api_unified_experimental_mlir.cc", "getValue");
 return value_; }
  Type getElementType() {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPScPSc_api_unified_experimental_mlirDTcc mht_6(mht_6_v, 302, "", "./tensorflow/compiler/mlir/tensorflow/c/c_api_unified_experimental_mlir.cc", "getElementType");

    return value_.getType().cast<ShapedType>().getElementType();
  }

  // For LLVM style RTTI.
  static bool classof(const AbstractTensorHandle* ptr) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPScPSc_api_unified_experimental_mlirDTcc mht_7(mht_7_v, 310, "", "./tensorflow/compiler/mlir/tensorflow/c/c_api_unified_experimental_mlir.cc", "classof");

    return ptr->getKind() == kMlir;
  }

 private:
  Value value_;
};

class MlirFunctionContext;

class MlirAbstractOp : public TracingOperation {
 public:
  explicit MlirAbstractOp(MLIRContext* context,
                          MlirFunctionContext* function_context)
      : TracingOperation(kMlir),
        context_(context),
        function_context_(function_context) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPScPSc_api_unified_experimental_mlirDTcc mht_8(mht_8_v, 329, "", "./tensorflow/compiler/mlir/tensorflow/c/c_api_unified_experimental_mlir.cc", "MlirAbstractOp");
}

  void Release() override {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPScPSc_api_unified_experimental_mlirDTcc mht_9(mht_9_v, 334, "", "./tensorflow/compiler/mlir/tensorflow/c/c_api_unified_experimental_mlir.cc", "Release");
 delete this; }

  Status Reset(const char* op, const char* raw_device_name) override;

  const string& Name() const override;

  const string& DeviceName() const override;

  Status SetDeviceName(const char* name) override;

  Status AddInput(AbstractTensorHandle* input) override;
  Status AddInputList(absl::Span<AbstractTensorHandle* const> inputs) override;
  Status Execute(absl::Span<AbstractTensorHandle*> retvals,
                 int* num_retvals) override;

  Status SetAttrString(const char* attr_name, const char* data,
                       size_t length) override;
  Status SetAttrInt(const char* attr_name, int64_t value) override;
  Status SetAttrFloat(const char* attr_name, float value) override;
  Status SetAttrBool(const char* attr_name, bool value) override;
  Status SetAttrType(const char* attr_name,
                     tensorflow::DataType dtype) override;
  Status SetAttrShape(const char* attr_name, const int64_t* dims,
                      const int num_dims) override;
  Status SetAttrFunction(const char* attr_name,
                         const AbstractOperation* value) override;
  Status SetAttrFunctionName(const char* attr_name, const char* value,
                             size_t length) override;
  Status SetAttrTensor(const char* attr_name,
                       AbstractTensorInterface* tensor) override;
  Status SetAttrStringList(const char* attr_name, const void* const* values,
                           const size_t* lengths, int num_values) override;
  Status SetAttrFloatList(const char* attr_name, const float* values,
                          int num_values) override;
  Status SetAttrIntList(const char* attr_name, const int64_t* values,
                        int num_values) override;
  Status SetAttrTypeList(const char* attr_name,
                         const tensorflow::DataType* values,
                         int num_values) override;
  Status SetAttrBoolList(const char* attr_name, const unsigned char* values,
                         int num_values) override;
  Status SetAttrShapeList(const char* attr_name, const int64_t** dims,
                          const int* num_dims, int num_values) override;
  Status SetAttrFunctionList(
      const char* attr_name,
      absl::Span<const AbstractOperation*> values) override;

  Status SetOpName(const char* const op_name) override;

  MLIRContext* GetContext() {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPScPSc_api_unified_experimental_mlirDTcc mht_10(mht_10_v, 386, "", "./tensorflow/compiler/mlir/tensorflow/c/c_api_unified_experimental_mlir.cc", "GetContext");
 return context_; }

  Status AddRef(Type type, Type* output_type);

  Status Create(ArrayRef<Value> operands, OperationState**);

  // For LLVM style RTTI.
  static bool classof(const AbstractOperation* ptr) {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPScPSc_api_unified_experimental_mlirDTcc mht_11(mht_11_v, 396, "", "./tensorflow/compiler/mlir/tensorflow/c/c_api_unified_experimental_mlir.cc", "classof");

    return ptr->getKind() == kMlir;
  }

 private:
  // Return true is there are still unfilled ODS slots for adding more inputs.
  bool IsNextODSArgAvailable();

  MLIRContext* context_;
  MlirFunctionContext* function_context_;
  SmallVector<Value, 8> operands_;
  llvm::StringMap<Attribute> attrs_;
  std::unique_ptr<OperationState> state_;
  // This is the index of the next ODS operand that will be added with AddInput
  // or AddInput;
  int current_ods_input_ = 0;
  const tensorflow::OpDef* op_def_ = nullptr;
  const char* op_name_ = nullptr;
  string tf_op_type_;
  // TODO(srbs): Use this.
  string device_name_;
};

// MlirFunction is a thin wrapper over a FuncOp.
class MlirFunction : public AbstractFunction {
 public:
  explicit MlirFunction(std::unique_ptr<MLIRContext> context,
                        OwningOpRef<mlir::ModuleOp> module, FuncOp func)
      : AbstractFunction(kMlir),
        context_(std::move(context)),
        module_(std::move(module)),
        func_(func) {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPScPSc_api_unified_experimental_mlirDTcc mht_12(mht_12_v, 430, "", "./tensorflow/compiler/mlir/tensorflow/c/c_api_unified_experimental_mlir.cc", "MlirFunction");
}

  Status GetFunctionDef(tensorflow::FunctionDef** f) override;

  // For LLVM style RTTI.
  static bool classof(const AbstractFunction* ptr) {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPScPSc_api_unified_experimental_mlirDTcc mht_13(mht_13_v, 438, "", "./tensorflow/compiler/mlir/tensorflow/c/c_api_unified_experimental_mlir.cc", "classof");

    return ptr->getKind() == kMlir;
  }

 private:
  std::unique_ptr<MLIRContext> context_;
  OwningOpRef<mlir::ModuleOp> module_;
  FuncOp func_;
  std::unique_ptr<tensorflow::FunctionDef> fdef_;
};

class MlirFunctionContext : public TracingContext {
 public:
  explicit MlirFunctionContext(const char* name)
      : TracingContext(kMlir),
        context_(std::make_unique<MLIRContext>()),
        builder_(context_.get()) {
   std::vector<std::string> mht_14_v;
   mht_14_v.push_back("name: \"" + (name == nullptr ? std::string("nullptr") : std::string((char*)name)) + "\"");
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPScPSc_api_unified_experimental_mlirDTcc mht_14(mht_14_v, 458, "", "./tensorflow/compiler/mlir/tensorflow/c/c_api_unified_experimental_mlir.cc", "MlirFunctionContext");

    RegisterDialects(*context_);
    // TODO(aminim) figure out the location story here
    module_ = ModuleOp::create(builder_.getUnknownLoc());
    func_ = FuncOp::create(builder_.getUnknownLoc(), name,
                           builder_.getFunctionType(llvm::None, llvm::None));
    module_->push_back(func_);
    builder_ = OpBuilder::atBlockBegin(func_.addEntryBlock());
  }

  void Release() override {
   std::vector<std::string> mht_15_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPScPSc_api_unified_experimental_mlirDTcc mht_15(mht_15_v, 471, "", "./tensorflow/compiler/mlir/tensorflow/c/c_api_unified_experimental_mlir.cc", "Release");
 delete this; }

  AbstractOperation* CreateOperation() override {
   std::vector<std::string> mht_16_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPScPSc_api_unified_experimental_mlirDTcc mht_16(mht_16_v, 476, "", "./tensorflow/compiler/mlir/tensorflow/c/c_api_unified_experimental_mlir.cc", "CreateOperation");

    return new MlirAbstractOp(context_.get(), this);
  }
  Status AddParameter(tensorflow::DataType dtype,
                      const tensorflow::PartialTensorShape& shape,
                      TracingTensorHandle** handle) override;

  Status Finalize(OutputList* outputs, AbstractFunction** f) override;

  Status RegisterFunction(AbstractFunction* func) override {
   std::vector<std::string> mht_17_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPScPSc_api_unified_experimental_mlirDTcc mht_17(mht_17_v, 488, "", "./tensorflow/compiler/mlir/tensorflow/c/c_api_unified_experimental_mlir.cc", "RegisterFunction");

    return Unimplemented(
        "Registering graph functions has not been implemented yet.");
  }

  Status RemoveFunction(const string& func) override {
   std::vector<std::string> mht_18_v;
   mht_18_v.push_back("func: \"" + func + "\"");
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPScPSc_api_unified_experimental_mlirDTcc mht_18(mht_18_v, 497, "", "./tensorflow/compiler/mlir/tensorflow/c/c_api_unified_experimental_mlir.cc", "RemoveFunction");

    return Unimplemented(
        "MlirFunctionContext::RemoveFunction has not been implemented yet.");
  }

  Operation* CreateOperationFromState(const OperationState& state);

 private:
  std::unique_ptr<MLIRContext> context_;
  OpBuilder builder_;
  FuncOp func_;
  OwningOpRef<mlir::ModuleOp> module_;
};

Status MlirAbstractOp::Reset(const char* op, const char* device_name) {
   std::vector<std::string> mht_19_v;
   mht_19_v.push_back("op: \"" + (op == nullptr ? std::string("nullptr") : std::string((char*)op)) + "\"");
   mht_19_v.push_back("device_name: \"" + (device_name == nullptr ? std::string("nullptr") : std::string((char*)device_name)) + "\"");
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPScPSc_api_unified_experimental_mlirDTcc mht_19(mht_19_v, 516, "", "./tensorflow/compiler/mlir/tensorflow/c/c_api_unified_experimental_mlir.cc", "MlirAbstractOp::Reset");

  if (state_) {
    return FailedPrecondition("Reset called on already built op.");
  }
  TF_RETURN_IF_ERROR(
      tensorflow::OpRegistry::Global()->LookUpOpDef(op, &op_def_));
  assert(op_def_);

  tf_op_type_ = op;
  std::string name = "tf.";
  name += op;
  // TODO(aminim) figure out the location story here
  state_ = std::make_unique<OperationState>(UnknownLoc::get(context_), name);
  return Status::OK();
}

Status MlirAbstractOp::SetAttrType(const char* attr_name,
                                   tensorflow::DataType dtype) {
   std::vector<std::string> mht_20_v;
   mht_20_v.push_back("attr_name: \"" + (attr_name == nullptr ? std::string("nullptr") : std::string((char*)attr_name)) + "\"");
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPScPSc_api_unified_experimental_mlirDTcc mht_20(mht_20_v, 537, "", "./tensorflow/compiler/mlir/tensorflow/c/c_api_unified_experimental_mlir.cc", "MlirAbstractOp::SetAttrType");

  if (!state_)
    return FailedPrecondition(
        "op_type must be specified before specifying attrs.");
  Type mlir_type;
  Builder builder(context_);
  TF_RETURN_IF_ERROR(ConvertDataType(dtype, builder, &mlir_type));
  attrs_[attr_name] = TypeAttr::get(mlir_type);
  return Status::OK();
}

Status MlirAbstractOp::SetOpName(const char* const op_name) {
   std::vector<std::string> mht_21_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPScPSc_api_unified_experimental_mlirDTcc mht_21(mht_21_v, 551, "", "./tensorflow/compiler/mlir/tensorflow/c/c_api_unified_experimental_mlir.cc", "MlirAbstractOp::SetOpName");

  // TODO(aminim): should we use a location?
  if (op_name_) {
    return FailedPrecondition("SetOpName called on already built op.");
  }
  op_name_ = op_name;
  return Status::OK();
}

Status MlirAbstractOp::AddRef(Type type, Type* output_type) {
   std::vector<std::string> mht_22_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPScPSc_api_unified_experimental_mlirDTcc mht_22(mht_22_v, 563, "", "./tensorflow/compiler/mlir/tensorflow/c/c_api_unified_experimental_mlir.cc", "MlirAbstractOp::AddRef");

  Type elt_type = getElementTypeOrSelf(type);
  if (elt_type.isa<mlir::TF::TensorFlowRefType>()) {
    return InvalidArgument("Requested reference to a reference type");
  }
  elt_type = TensorFlowRefType::get(elt_type);
  if (RankedTensorType tensor_type = type.dyn_cast<RankedTensorType>()) {
    *output_type = RankedTensorType::get(tensor_type.getShape(), elt_type);
  }
  *output_type = UnrankedTensorType::get(elt_type);
  return Status::OK();
}

Status MlirAbstractOp::Create(ArrayRef<Value> operands,
                              OperationState** state) {
   std::vector<std::string> mht_23_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPScPSc_api_unified_experimental_mlirDTcc mht_23(mht_23_v, 580, "", "./tensorflow/compiler/mlir/tensorflow/c/c_api_unified_experimental_mlir.cc", "MlirAbstractOp::Create");

  state_->operands = llvm::to_vector<4>(operands);
  Builder builder(context_);

  if (current_ods_input_ != op_def_->input_arg_size())
    return InvalidArgument(absl::StrCat("Mismatch in operands number: got ",
                                        current_ods_input_, " expected ",
                                        op_def_->input_arg_size(), " ; for op ",
                                        state_->name.getStringRef().str()));

  // Process results according to the op_def and infer types for derived
  // attributes.
  for (const tensorflow::OpDef::ArgDef& output_arg : op_def_->output_arg()) {
    int original_size = state_->types.size();
    if (!output_arg.number_attr().empty()) {
      // Same type repeated "repeats" times.
      Attribute repeats_attr = attrs_[output_arg.number_attr()];
      if (!repeats_attr)
        return InvalidArgument("Missing attribute '", output_arg.number_attr(),
                               "' required for output list '",
                               output_arg.name(), "'");
      if (!repeats_attr.isa<IntegerAttr>())
        return InvalidArgument("Attribute '", output_arg.number_attr(),
                               "' required for output list '",
                               output_arg.name(), "' isn't an integer");
      int64_t repeats = repeats_attr.cast<IntegerAttr>().getInt();

      if (!output_arg.type_attr().empty()) {
        // Same type repeated "repeats" times.
        Attribute attr = attrs_[output_arg.type_attr()];
        if (!attr)
          return InvalidArgument("Missing attribute '", output_arg.type_attr(),
                                 "' required for output '", output_arg.name(),
                                 "'");
        TypeAttr type_attr = attr.dyn_cast<TypeAttr>();
        if (!type_attr)
          return InvalidArgument("Attribute '", output_arg.type_attr(),
                                 "' required for output '", output_arg.name(),
                                 "' isn't a type attribute");
        for (int i = 0; i < repeats; ++i)
          state_->types.push_back(UnrankedTensorType::get(type_attr.getType()));
      } else if (output_arg.type() != tensorflow::DT_INVALID) {
        for (int i = 0; i < repeats; ++i) {
          Type type;
          TF_RETURN_IF_ERROR(
              ConvertDataType(output_arg.type(), builder, &type));
          state_->types.push_back(type);
        }
      } else {
        return InvalidArgument("Missing type or type_attr field in ",
                               output_arg.ShortDebugString());
      }
    } else if (!output_arg.type_attr().empty()) {
      Attribute attr = attrs_[output_arg.type_attr()];
      if (!attr)
        return InvalidArgument("Missing attribute '", output_arg.type_attr(),
                               "' required for output '", output_arg.name(),
                               "'");
      TypeAttr type_attr = attr.dyn_cast<TypeAttr>();
      if (!type_attr)
        return InvalidArgument("Attribute '", output_arg.type_attr(),
                               "' required for output '", output_arg.name(),
                               "' isn't a type attribute");
      state_->types.push_back(UnrankedTensorType::get(type_attr.getValue()));
    } else if (!output_arg.type_list_attr().empty()) {
      // This is pointing to an attribute which is an array of types.
      Attribute attr = attrs_[output_arg.type_list_attr()];
      if (!attr)
        return InvalidArgument(
            "Missing attribute '", output_arg.type_list_attr(),
            "' required for output '", output_arg.name(), "'");
      ArrayAttr array_attr = attr.dyn_cast<ArrayAttr>();
      if (!array_attr)
        return InvalidArgument("Attribute '", output_arg.type_list_attr(),
                               "' required for output '", output_arg.name(),
                               "' isn't an array attribute");
      for (Attribute attr : array_attr) {
        TypeAttr type_attr = attr.dyn_cast<TypeAttr>();
        if (!type_attr)
          return InvalidArgument("Array Attribute '",
                                 output_arg.type_list_attr(),
                                 "' required for output '", output_arg.name(),
                                 "' has a non-Type element");
        state_->types.push_back(UnrankedTensorType::get(type_attr.getValue()));
      }
    } else if (output_arg.type() != tensorflow::DT_INVALID) {
      Type type;
      Builder builder(context_);
      TF_RETURN_IF_ERROR(ConvertDataType(output_arg.type(), builder, &type));
      state_->types.push_back(type);
    } else {
      return InvalidArgument("No type fields in ",
                             output_arg.ShortDebugString());
    }
    if (output_arg.is_ref()) {
      // For all types that were added by this function call, make them refs.
      for (Type& type : llvm::make_range(&state_->types[original_size],
                                         state_->types.end())) {
        Type output_type;
        TF_RETURN_IF_ERROR(AddRef(type, &output_type));
        type = output_type;
      }
    }
  }
  for (auto& it : attrs_) state_->addAttribute(it.first(), it.second);
  *state = state_.get();
  return Status::OK();
}

const string& MlirAbstractOp::Name() const {
   std::vector<std::string> mht_24_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPScPSc_api_unified_experimental_mlirDTcc mht_24(mht_24_v, 692, "", "./tensorflow/compiler/mlir/tensorflow/c/c_api_unified_experimental_mlir.cc", "MlirAbstractOp::Name");
 return tf_op_type_; }

const string& MlirAbstractOp::DeviceName() const {
   std::vector<std::string> mht_25_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPScPSc_api_unified_experimental_mlirDTcc mht_25(mht_25_v, 697, "", "./tensorflow/compiler/mlir/tensorflow/c/c_api_unified_experimental_mlir.cc", "MlirAbstractOp::DeviceName");
 return device_name_; }

Status MlirAbstractOp::SetDeviceName(const char* name) {
   std::vector<std::string> mht_26_v;
   mht_26_v.push_back("name: \"" + (name == nullptr ? std::string("nullptr") : std::string((char*)name)) + "\"");
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPScPSc_api_unified_experimental_mlirDTcc mht_26(mht_26_v, 703, "", "./tensorflow/compiler/mlir/tensorflow/c/c_api_unified_experimental_mlir.cc", "MlirAbstractOp::SetDeviceName");

  device_name_ = name;
  return Status::OK();
}

Status MlirAbstractOp::SetAttrString(const char* attr_name, const char* data,
                                     size_t length) {
   std::vector<std::string> mht_27_v;
   mht_27_v.push_back("attr_name: \"" + (attr_name == nullptr ? std::string("nullptr") : std::string((char*)attr_name)) + "\"");
   mht_27_v.push_back("data: \"" + (data == nullptr ? std::string("nullptr") : std::string((char*)data)) + "\"");
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPScPSc_api_unified_experimental_mlirDTcc mht_27(mht_27_v, 714, "", "./tensorflow/compiler/mlir/tensorflow/c/c_api_unified_experimental_mlir.cc", "MlirAbstractOp::SetAttrString");

  return Unimplemented("SetAttrString has not been implemented yet.");
}
Status MlirAbstractOp::SetAttrInt(const char* attr_name, int64_t value) {
   std::vector<std::string> mht_28_v;
   mht_28_v.push_back("attr_name: \"" + (attr_name == nullptr ? std::string("nullptr") : std::string((char*)attr_name)) + "\"");
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPScPSc_api_unified_experimental_mlirDTcc mht_28(mht_28_v, 721, "", "./tensorflow/compiler/mlir/tensorflow/c/c_api_unified_experimental_mlir.cc", "MlirAbstractOp::SetAttrInt");

  return Unimplemented("SetAttrInt has not been implemented yet.");
}
Status MlirAbstractOp::SetAttrFloat(const char* attr_name, float value) {
   std::vector<std::string> mht_29_v;
   mht_29_v.push_back("attr_name: \"" + (attr_name == nullptr ? std::string("nullptr") : std::string((char*)attr_name)) + "\"");
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPScPSc_api_unified_experimental_mlirDTcc mht_29(mht_29_v, 728, "", "./tensorflow/compiler/mlir/tensorflow/c/c_api_unified_experimental_mlir.cc", "MlirAbstractOp::SetAttrFloat");

  return Unimplemented("SetAttrFloat has not been implemented yet.");
}
Status MlirAbstractOp::SetAttrBool(const char* attr_name, bool value) {
   std::vector<std::string> mht_30_v;
   mht_30_v.push_back("attr_name: \"" + (attr_name == nullptr ? std::string("nullptr") : std::string((char*)attr_name)) + "\"");
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPScPSc_api_unified_experimental_mlirDTcc mht_30(mht_30_v, 735, "", "./tensorflow/compiler/mlir/tensorflow/c/c_api_unified_experimental_mlir.cc", "MlirAbstractOp::SetAttrBool");

  attrs_[attr_name] = BoolAttr::get(context_, value);
  return Status::OK();
}
Status MlirAbstractOp::SetAttrShape(const char* attr_name, const int64_t* dims,
                                    const int num_dims) {
   std::vector<std::string> mht_31_v;
   mht_31_v.push_back("attr_name: \"" + (attr_name == nullptr ? std::string("nullptr") : std::string((char*)attr_name)) + "\"");
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPScPSc_api_unified_experimental_mlirDTcc mht_31(mht_31_v, 744, "", "./tensorflow/compiler/mlir/tensorflow/c/c_api_unified_experimental_mlir.cc", "MlirAbstractOp::SetAttrShape");

  return Unimplemented("SetAttrShape has not been implemented yet.");
}
Status MlirAbstractOp::SetAttrFunction(const char* attr_name,
                                       const AbstractOperation* value) {
   std::vector<std::string> mht_32_v;
   mht_32_v.push_back("attr_name: \"" + (attr_name == nullptr ? std::string("nullptr") : std::string((char*)attr_name)) + "\"");
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPScPSc_api_unified_experimental_mlirDTcc mht_32(mht_32_v, 752, "", "./tensorflow/compiler/mlir/tensorflow/c/c_api_unified_experimental_mlir.cc", "MlirAbstractOp::SetAttrFunction");

  return Unimplemented("SetAttrFunction has not been implemented yet.");
}
Status MlirAbstractOp::SetAttrFunctionName(const char* attr_name,
                                           const char* value, size_t length) {
   std::vector<std::string> mht_33_v;
   mht_33_v.push_back("attr_name: \"" + (attr_name == nullptr ? std::string("nullptr") : std::string((char*)attr_name)) + "\"");
   mht_33_v.push_back("value: \"" + (value == nullptr ? std::string("nullptr") : std::string((char*)value)) + "\"");
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPScPSc_api_unified_experimental_mlirDTcc mht_33(mht_33_v, 761, "", "./tensorflow/compiler/mlir/tensorflow/c/c_api_unified_experimental_mlir.cc", "MlirAbstractOp::SetAttrFunctionName");

  return Unimplemented("SetAttrFunctionName has not been implemented yet.");
}
Status MlirAbstractOp::SetAttrTensor(const char* attr_name,
                                     AbstractTensorInterface* tensor) {
   std::vector<std::string> mht_34_v;
   mht_34_v.push_back("attr_name: \"" + (attr_name == nullptr ? std::string("nullptr") : std::string((char*)attr_name)) + "\"");
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPScPSc_api_unified_experimental_mlirDTcc mht_34(mht_34_v, 769, "", "./tensorflow/compiler/mlir/tensorflow/c/c_api_unified_experimental_mlir.cc", "MlirAbstractOp::SetAttrTensor");

  return Unimplemented("SetAttrTensor has not been implemented yet.");
}
Status MlirAbstractOp::SetAttrStringList(const char* attr_name,
                                         const void* const* values,
                                         const size_t* lengths,
                                         int num_values) {
   std::vector<std::string> mht_35_v;
   mht_35_v.push_back("attr_name: \"" + (attr_name == nullptr ? std::string("nullptr") : std::string((char*)attr_name)) + "\"");
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPScPSc_api_unified_experimental_mlirDTcc mht_35(mht_35_v, 779, "", "./tensorflow/compiler/mlir/tensorflow/c/c_api_unified_experimental_mlir.cc", "MlirAbstractOp::SetAttrStringList");

  return Unimplemented("SetAttrStringList has not been implemented yet.");
}
Status MlirAbstractOp::SetAttrFloatList(const char* attr_name,
                                        const float* values, int num_values) {
   std::vector<std::string> mht_36_v;
   mht_36_v.push_back("attr_name: \"" + (attr_name == nullptr ? std::string("nullptr") : std::string((char*)attr_name)) + "\"");
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPScPSc_api_unified_experimental_mlirDTcc mht_36(mht_36_v, 787, "", "./tensorflow/compiler/mlir/tensorflow/c/c_api_unified_experimental_mlir.cc", "MlirAbstractOp::SetAttrFloatList");

  return Unimplemented("SetAttrFloatList has not been implemented yet.");
}
Status MlirAbstractOp::SetAttrIntList(const char* attr_name,
                                      const int64_t* values, int num_values) {
   std::vector<std::string> mht_37_v;
   mht_37_v.push_back("attr_name: \"" + (attr_name == nullptr ? std::string("nullptr") : std::string((char*)attr_name)) + "\"");
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPScPSc_api_unified_experimental_mlirDTcc mht_37(mht_37_v, 795, "", "./tensorflow/compiler/mlir/tensorflow/c/c_api_unified_experimental_mlir.cc", "MlirAbstractOp::SetAttrIntList");

  return Unimplemented("SetAttrIntList has not been implemented yet.");
}
Status MlirAbstractOp::SetAttrTypeList(const char* attr_name,
                                       const tensorflow::DataType* values,
                                       int num_values) {
   std::vector<std::string> mht_38_v;
   mht_38_v.push_back("attr_name: \"" + (attr_name == nullptr ? std::string("nullptr") : std::string((char*)attr_name)) + "\"");
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPScPSc_api_unified_experimental_mlirDTcc mht_38(mht_38_v, 804, "", "./tensorflow/compiler/mlir/tensorflow/c/c_api_unified_experimental_mlir.cc", "MlirAbstractOp::SetAttrTypeList");

  return Unimplemented("SetAttrTypeList has not been implemented yet.");
}
Status MlirAbstractOp::SetAttrBoolList(const char* attr_name,
                                       const unsigned char* values,
                                       int num_values) {
   std::vector<std::string> mht_39_v;
   mht_39_v.push_back("attr_name: \"" + (attr_name == nullptr ? std::string("nullptr") : std::string((char*)attr_name)) + "\"");
   mht_39_v.push_back("values: \"" + (values == nullptr ? std::string("nullptr") : std::string((char*)values)) + "\"");
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPScPSc_api_unified_experimental_mlirDTcc mht_39(mht_39_v, 814, "", "./tensorflow/compiler/mlir/tensorflow/c/c_api_unified_experimental_mlir.cc", "MlirAbstractOp::SetAttrBoolList");

  return Unimplemented("SetAttrBoolList has not been implemented yet.");
}
Status MlirAbstractOp::SetAttrShapeList(const char* attr_name,
                                        const int64_t** dims,
                                        const int* num_dims, int num_values) {
   std::vector<std::string> mht_40_v;
   mht_40_v.push_back("attr_name: \"" + (attr_name == nullptr ? std::string("nullptr") : std::string((char*)attr_name)) + "\"");
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPScPSc_api_unified_experimental_mlirDTcc mht_40(mht_40_v, 823, "", "./tensorflow/compiler/mlir/tensorflow/c/c_api_unified_experimental_mlir.cc", "MlirAbstractOp::SetAttrShapeList");

  return Unimplemented("SetAttrShapeList has not been implemented yet.");
}
Status MlirAbstractOp::SetAttrFunctionList(
    const char* attr_name, absl::Span<const AbstractOperation*> values) {
   std::vector<std::string> mht_41_v;
   mht_41_v.push_back("attr_name: \"" + (attr_name == nullptr ? std::string("nullptr") : std::string((char*)attr_name)) + "\"");
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPScPSc_api_unified_experimental_mlirDTcc mht_41(mht_41_v, 831, "", "./tensorflow/compiler/mlir/tensorflow/c/c_api_unified_experimental_mlir.cc", "MlirAbstractOp::SetAttrFunctionList");

  return Unimplemented("SetAttrFunctionList has not been implemented yet.");
}

Status MlirFunction::GetFunctionDef(tensorflow::FunctionDef** f) {
   std::vector<std::string> mht_42_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPScPSc_api_unified_experimental_mlirDTcc mht_42(mht_42_v, 838, "", "./tensorflow/compiler/mlir/tensorflow/c/c_api_unified_experimental_mlir.cc", "MlirFunction::GetFunctionDef");

  if (fdef_) {
    *f = fdef_.get();
    return Status::OK();
  }
  PassManager pm(func_.getContext());
  ::tensorflow::applyTensorflowAndCLOptions(pm);
  pm.addNestedPass<FuncOp>(CreateFunctionalToExecutorDialectConversionPass());
  pm.addPass(CreateBreakUpIslandsPass());

  // In case of failure, the `diag_handler` converts MLIR errors emitted to
  // the MLIRContext into a tensorflow::Status.
  StatusScopedDiagnosticHandler diag_handler(func_.getContext());
  LogicalResult result = pm.run(func_->getParentOfType<ModuleOp>());
  (void)result;
  TF_RETURN_IF_ERROR(diag_handler.ConsumeStatus());

  tensorflow::GraphExportConfig configs;
  fdef_.reset(new tensorflow::FunctionDef());
  TF_RETURN_IF_ERROR(
      ConvertMlirFunctionToFunctionLibraryDef(func_, configs, fdef_.get()));
  *f = fdef_.get();
  return Status::OK();
}

Status MlirAbstractOp::Execute(absl::Span<AbstractTensorHandle*> retvals,
                               int* num_retvals) {
   std::vector<std::string> mht_43_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPScPSc_api_unified_experimental_mlirDTcc mht_43(mht_43_v, 867, "", "./tensorflow/compiler/mlir/tensorflow/c/c_api_unified_experimental_mlir.cc", "MlirAbstractOp::Execute");

  OperationState* state;
  TF_RETURN_IF_ERROR(Create(operands_, &state));
  Operation* op = function_context_->CreateOperationFromState(*state);
  *num_retvals = op->getNumResults();
  for (int i = 0; i < *num_retvals; i++)
    retvals[i] = new MlirTensor(op->getResult(i));
  return Status::OK();
}

Operation* MlirFunctionContext::CreateOperationFromState(
    const OperationState& state) {
   std::vector<std::string> mht_44_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPScPSc_api_unified_experimental_mlirDTcc mht_44(mht_44_v, 881, "", "./tensorflow/compiler/mlir/tensorflow/c/c_api_unified_experimental_mlir.cc", "MlirFunctionContext::CreateOperationFromState");

  return builder_.create(state);
}

Status MlirFunctionContext::AddParameter(
    tensorflow::DataType dtype, const tensorflow::PartialTensorShape& shape,
    TracingTensorHandle** handle) {
   std::vector<std::string> mht_45_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPScPSc_api_unified_experimental_mlirDTcc mht_45(mht_45_v, 890, "", "./tensorflow/compiler/mlir/tensorflow/c/c_api_unified_experimental_mlir.cc", "MlirFunctionContext::AddParameter");

  // TODO(b/173073199): Use shape. Enable tests in unified_api_test.cc once
  // resolved.
  Type type;
  TF_RETURN_IF_ERROR(ConvertDataTypeToTensor(dtype, builder_, &type));
  *handle =
      new MlirTensor(func_.getBody().front().addArgument(type, func_.getLoc()));
  return Status::OK();
}

Status MlirAbstractOp::AddInput(AbstractTensorHandle* input) {
   std::vector<std::string> mht_46_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPScPSc_api_unified_experimental_mlirDTcc mht_46(mht_46_v, 903, "", "./tensorflow/compiler/mlir/tensorflow/c/c_api_unified_experimental_mlir.cc", "MlirAbstractOp::AddInput");

  if (current_ods_input_ >= op_def_->input_arg_size())
    return InvalidArgument(
        absl::StrCat("More Input() (", current_ods_input_, ") calls than the ",
                     op_def_->input_arg_size(), " allowed input_args ; for op ",
                     state_->name.getStringRef().str()));

  auto* operand = dyn_cast<MlirTensor>(input);
  if (!operand) return InvalidArgument("Unable to cast input to MlirTensor");
  operands_.push_back(operand->getValue());

  // Get the next ArgDef and use it to infer the derived attributes associated
  // to this input.
  const tensorflow::OpDef::ArgDef& arg_def =
      op_def_->input_arg(current_ods_input_++);
  Type expected_type;
  if (arg_def.type() != tensorflow::DT_INVALID) {
    Builder builder(context_);
    TF_RETURN_IF_ERROR(
        tensorflow::ConvertDataType(arg_def.type(), builder, &expected_type));
    if (arg_def.is_ref()) {
      Type output_type;
      TF_RETURN_IF_ERROR(AddRef(expected_type, &output_type));
      expected_type = output_type;
    }
  } else {
    expected_type = cast<MlirTensor>(input)->getElementType();
  }
  if (!arg_def.type_attr().empty())
    attrs_[arg_def.type_attr()] = TypeAttr::get(expected_type);

  return Status::OK();
}

Status MlirAbstractOp::AddInputList(
    absl::Span<AbstractTensorHandle* const> inputs) {
   std::vector<std::string> mht_47_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPScPSc_api_unified_experimental_mlirDTcc mht_47(mht_47_v, 941, "", "./tensorflow/compiler/mlir/tensorflow/c/c_api_unified_experimental_mlir.cc", "MlirAbstractOp::AddInputList");

  if (current_ods_input_ >= op_def_->input_arg_size())
    return InvalidArgument(
        absl::StrCat("More Input() (", current_ods_input_, ") calls than the ",
                     op_def_->input_arg_size(), " allowed input_args"));

  for (AbstractTensorHandle* input : inputs) {
    auto* operand = dyn_cast<MlirTensor>(input);
    if (!operand) return InvalidArgument("Unable to cast input to MlirTensor");
    operands_.push_back(operand->getValue());
  }

  // Get the next ArgDef and use it to infer the derived attributes associated
  // to this input.
  const tensorflow::OpDef::ArgDef& arg_def =
      op_def_->input_arg(current_ods_input_++);
  if (!arg_def.number_attr().empty()) {
    Builder builder(context_);
    attrs_[arg_def.number_attr()] = builder.getI32IntegerAttr(inputs.size());
    // TODO(aminim): handle ref variable.
    if (arg_def.type() != tensorflow::DT_INVALID) {
      // TODO(aminim): check type wrt input
      Type arg_def_type;
      TF_RETURN_IF_ERROR(
          ConvertDataType(arg_def.type(), builder, &arg_def_type));
      // Ensure each of the type in the list matches the op def type.
      // TODO(aminim): can we improve the error message with the actual types?
      for (AbstractTensorHandle* input : inputs)
        if (arg_def_type != cast<MlirTensor>(input)->getElementType())
          return InvalidArgument(
              "Invalid input list: type mismatch the op def expectation");
    } else if (!inputs.empty()) {
      if (arg_def.type_attr().empty())
        return FailedPrecondition(
            "Invalid opdef type constraint: either type or type_attr required");

      attrs_[arg_def.type_attr()] =
          TypeAttr::get(cast<MlirTensor>(inputs.front())->getElementType());
    }
  } else if (!arg_def.type_list_attr().empty()) {
    // TODO(aminim): handle ref variable.
    SmallVector<Attribute, 8> types;
    types.reserve(inputs.size());
    for (AbstractTensorHandle* input : inputs)
      types.push_back(TypeAttr::get(cast<MlirTensor>(input)->getElementType()));
    attrs_[arg_def.type_list_attr()] = ArrayAttr::get(GetContext(), types);
  }
  return Status::OK();
}

Status MlirFunctionContext::Finalize(OutputList* outputs,
                                     AbstractFunction** f) {
   std::vector<std::string> mht_48_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPScPSc_api_unified_experimental_mlirDTcc mht_48(mht_48_v, 995, "", "./tensorflow/compiler/mlir/tensorflow/c/c_api_unified_experimental_mlir.cc", "MlirFunctionContext::Finalize");

  Block& body = func_.getBody().front();
  SmallVector<Value, 8> ret_operands;
  for (auto* output : outputs->outputs) {
    auto* operand = dyn_cast<MlirTensor>(output);
    if (!operand)
      return InvalidArgument("Capturing eager tensors is not supported yet.");
    if (operand->getValue().getContext() != context_.get())
      return InvalidArgument(
          "Capturing tensors from other context is not supported.");
    ret_operands.push_back(operand->getValue());
  }
  builder_.create<func::ReturnOp>(func_.getLoc(), ret_operands);

  auto arg_types = body.getArgumentTypes();
  auto result_types = body.getTerminator()->getOperandTypes();
  func_.setType(FunctionType::get(func_.getContext(), arg_types, result_types));
  *f = new MlirFunction(std::move(context_), std::move(module_), func_);
  return Status::OK();
}

extern "C" {
TracingContext* MlirTracingFactory(const char* fn_name, TF_Status* s) {
   std::vector<std::string> mht_49_v;
   mht_49_v.push_back("fn_name: \"" + (fn_name == nullptr ? std::string("nullptr") : std::string((char*)fn_name)) + "\"");
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPScPSc_api_unified_experimental_mlirDTcc mht_49(mht_49_v, 1021, "", "./tensorflow/compiler/mlir/tensorflow/c/c_api_unified_experimental_mlir.cc", "MlirTracingFactory");

  return new MlirFunctionContext(fn_name);
}
}

}  // namespace
}  // namespace TF
}  // namespace mlir
