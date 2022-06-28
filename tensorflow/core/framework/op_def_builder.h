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

// Class and associated machinery for specifying an Op's OpDef and shape
// inference function for Op registration.

#ifndef TENSORFLOW_CORE_FRAMEWORK_OP_DEF_BUILDER_H_
#define TENSORFLOW_CORE_FRAMEWORK_OP_DEF_BUILDER_H_
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
class MHTracer_DTPStensorflowPScorePSframeworkPSop_def_builderDTh {
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
   MHTracer_DTPStensorflowPScorePSframeworkPSop_def_builderDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSframeworkPSop_def_builderDTh() {
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


#include <string>
#include <vector>

#include "tensorflow/core/framework/full_type.pb.h"
#include "tensorflow/core/framework/op_def.pb.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/core/stringpiece.h"
#include "tensorflow/core/platform/macros.h"

namespace tensorflow {

// TODO(b/62899350): Refactor without proto dependencies.
typedef std::function<Status(OpDef* c)> OpTypeConstructor;

typedef std::vector<std::reference_wrapper<const FullTypeDef>> TypeRefVector;
typedef std::map<std::string, std::reference_wrapper<const FullTypeDef>>
    TypeRefMap;

// A type inference function, called for each node during type inference
// (possibly multiple times).
// The first argument (input_types) will hold the type of each of the node's
// inputs. The second argument (type_vars) will hold the return type of
// each function referred from any type variable (e.g. `FuncVar`) present
// in the node's corresponding op definition.
//
// TODO(mdan): Consider a vector-in, vector-out contract.
typedef std::function<StatusOr<FullTypeDef>(const TypeRefVector&,
                                            const TypeRefMap&)>
    ForwardTypeInferenceFn;

class FunctionDefHelper;

namespace shape_inference {
class InferenceContext;
}
typedef std::function<Status(shape_inference::InferenceContext* c)>
    OpShapeInferenceFn;

struct OpRegistrationData {
 public:
  OpRegistrationData() {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSop_def_builderDTh mht_0(mht_0_v, 232, "", "./tensorflow/core/framework/op_def_builder.h", "OpRegistrationData");
}
  OpRegistrationData(const OpDef& def) : op_def(def) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSop_def_builderDTh mht_1(mht_1_v, 236, "", "./tensorflow/core/framework/op_def_builder.h", "OpRegistrationData");
}
  OpRegistrationData(const OpDef& def, const OpShapeInferenceFn& fn,
                     bool is_function = false)
      : op_def(def), shape_inference_fn(fn), is_function_op(is_function) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSop_def_builderDTh mht_2(mht_2_v, 242, "", "./tensorflow/core/framework/op_def_builder.h", "OpRegistrationData");
}

  OpDef op_def;
  OpShapeInferenceFn shape_inference_fn;

  // Type constructor. This callable initializes the type of this op.
  // It is provided as a programmatic mechanism for defining an op's
  // type, as part of its registration. It is to be eventually replaced by a
  // textual language.
  //
  // Important: historically, op registrations only contained partial
  // input/output type information in non-standardized attribute declarations
  // (e.g. typically, input types were held in a `dtype` attribute). The type
  // constructor currently duplicates such attribute information, with the aim
  // of entirely subsuming it, and eventually deprecating all type-related
  // attributes.
  //
  // Since ops are typically parametrized, the type created by this constructor
  // is also parametric.
  //
  // Example: for an op `Foo(x: T) -> Bar[T]`:
  //
  //  * typically, its op registration included a single attribute `T: type`;
  //    then the respective input was defined as `x: T`; the output type `Bar`
  //    was implied by the op name.
  //  * the type constructor creates a FullType object containing `Bar[T]`; this
  //    still relies on the `T` attribute which it references.
  //  * in the future, the type constructor will create a FullType containing
  //    `Callable[(x: T), Bar[T]]`, and the attribute `T` will be deprecated.
  OpTypeConstructor type_ctor;

  // Forward type inference function. This callable infers the return type of an
  // op based on its input types.
  //
  // Note that the type constructor and forward inference functions need not be
  // mutually exclusive: if there is some static information that can be set
  // based on attributes, then that should be set in the constructor. If more
  // information can be extracted from inputs, that should be done in the
  // forward inference function.
  //
  // This is similar to the shape function, but is more general, and applied
  // directly to NodeDefs, rather than working on the ShapeAndType structures.
  // Note that the op input/output declarations may specify some implicit type
  // constraints through attribute references (i.e. two inputs pointing to the
  // same type attribute). Those constraints may duplicate what this function
  // specifies in its body. That's intended, for a gradual transition to a more
  // formal type system.
  //
  // These type inference functions are intermediate solutions as well: once the
  // op registration has a complete, formal type definition, along with
  // a solver-based type inference, it will replace these functions.
  //
  // TODO(mdan): Merge with shape inference.
  // TODO(mdan): Replace with a union-based type inference algorithm.
  ForwardTypeInferenceFn fwd_type_fn;

  bool is_function_op = false;
};

// Builder class passed to the REGISTER_OP() macro.
class OpDefBuilder {
 public:
  // Constructs an OpDef with just the name field set.
  explicit OpDefBuilder(std::string op_name);

  // Adds an attr to this OpDefBuilder (and returns *this). The spec has
  // format "<name>:<type>" or "<name>:<type>=<default>"
  // where <name> matches regexp [a-zA-Z][a-zA-Z0-9_]*
  // (by convention only using capital letters for attrs that can be inferred)
  // <type> can be:
  //   "string", "int", "float", "bool", "type", "shape", or "tensor"
  //   "numbertype", "realnumbertype", "quantizedtype"
  //       (meaning "type" with a restriction on valid values)
  //   "{int32,int64}" or {realnumbertype,quantizedtype,string}"
  //       (meaning "type" with a restriction containing unions of value types)
  //   "{\"foo\", \"bar\n baz\"}", or "{'foo', 'bar\n baz'}"
  //       (meaning "string" with a restriction on valid values)
  //   "list(string)", ..., "list(tensor)", "list(numbertype)", ...
  //       (meaning lists of the above types)
  //   "int >= 2" (meaning "int" with a restriction on valid values)
  //   "list(string) >= 2", "list(int) >= 2"
  //       (meaning "list(string)" / "list(int)" with length at least 2)
  // <default>, if included, should use the Proto text format
  // of <type>.  For lists use [a, b, c] format.
  //
  // Note that any attr specifying the length of an input or output will
  // get a default minimum of 1 unless the >= # syntax is used.
  //
  // TODO(josh11b): Perhaps support restrictions and defaults as optional
  // extra arguments to Attr() instead of encoding them in the spec string.
  // TODO(josh11b): Would like to have better dtype handling for tensor attrs:
  // * Ability to say the type of an input/output matches the type of
  //   the tensor.
  // * Ability to restrict the type of the tensor like the existing
  //   restrictions for type attrs.
  // Perhaps by linking the type of the tensor to a type attr?
  OpDefBuilder& Attr(std::string spec);

  // Adds an input or output to this OpDefBuilder (and returns *this).
  // The spec has form "<name>:<type-expr>" or "<name>:Ref(<type-expr>)"
  // where <name> matches regexp [a-z][a-z0-9_]* and <type-expr> can be:
  // * For a single tensor: <type>
  // * For a sequence of tensors with the same type: <number>*<type>
  // * For a sequence of tensors with different types: <type-list>
  // Where:
  //   <type> is either one of "float", "int32", "string", ...
  //                 or the name of an attr (see above) with type "type".
  //   <number> is the name of an attr with type "int".
  //   <type-list> is the name of an attr with type "list(type)".
  // TODO(josh11b): Indicate Ref() via an optional argument instead of
  // in the spec?
  // TODO(josh11b): SparseInput() and SparseOutput() matching the Python
  // handling?
  OpDefBuilder& Input(std::string spec);
  OpDefBuilder& Output(std::string spec);

  // Turns on the indicated boolean flag in this OpDefBuilder (and
  // returns *this).
  OpDefBuilder& SetIsCommutative();
  OpDefBuilder& SetIsAggregate();
  OpDefBuilder& SetIsStateful();
  OpDefBuilder& SetAllowsUninitializedInput();
  OpDefBuilder& SetIsDistributedCommunication();

  // Deprecate the op at a certain GraphDef version.
  OpDefBuilder& Deprecated(int version, std::string explanation);

  // Adds docs to this OpDefBuilder (and returns *this).
  // Docs have the format:
  //   <1-line summary>
  //   <rest of the description>
  //   <name>: <description of name>
  //   <name>: <description of name>
  //     <if long, indent the description on subsequent lines>
  // Where <name> is the name of an attr, input, or output.  Please
  // wrap docs at 72 columns so that it may be indented in the
  // generated output.  For tensor inputs or outputs (not attrs), you
  // may start the description with an "=" (like name:= <description>)
  // to suppress the automatically-generated type documentation in
  // generated output.
  OpDefBuilder& Doc(std::string text);

  // Sets the function to be used as type constructor.
  // See OpRegistrationData::type_ctor.
  OpDefBuilder& SetTypeConstructor(OpTypeConstructor c);

  // Sets the function to be used for forward type inference.
  // See OpRegistrationData::fwd_type_fn.
  OpDefBuilder& SetForwardTypeFn(ForwardTypeInferenceFn f);

  // Sets the shape function to be used for shape inference.
  //
  // Note that currently (October 2016), python code still requires a
  // RegisterShape call to invoke this; see call_cpp_shape_fn in
  // python/framework/common_shapes.py
  OpDefBuilder& SetShapeFn(OpShapeInferenceFn fn);

  // Allows the `<type>` in calls to `Attr()` to be "any".
  // This is used by PythonAPIWrapper for pass-through parameters.
  OpDefBuilder& AllowAttrTypeAny();

  // Sets op_reg_data->op_def to the requested OpDef and
  // op_reg_data->shape_inference_fn to the requested shape inference function,
  // or returns an error.
  // Must be called after all of the above methods.
  //
  // Note that OpDefBuilder only reports parsing errors.  You should also
  // call ValidateOpDef() to detect other problems.
  Status Finalize(OpRegistrationData* op_reg_data) const;

 private:
  friend class FunctionDefHelper;

  // Adds control output to this OpDefBuilder (and returns *this).
  // The <name> must be a valid node name (matches regexp
  // [a-zA-Z][a-zA-Z0-9_]*). Named control output can only exist for functions.
  OpDefBuilder& ControlOutput(std::string name);

  OpDef* op_def() {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSop_def_builderDTh mht_3(mht_3_v, 423, "", "./tensorflow/core/framework/op_def_builder.h", "op_def");
 return &op_reg_data_.op_def; }

  OpRegistrationData op_reg_data_;
  std::vector<string> attrs_;
  std::vector<string> inputs_;
  std::vector<string> outputs_;
  std::vector<string> control_outputs_;
  std::string doc_;
  std::vector<string> errors_;
  bool allow_attr_type_any_ = false;
};

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_FRAMEWORK_OP_DEF_BUILDER_H_
