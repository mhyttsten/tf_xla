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
class MHTracer_DTPStensorflowPSpythonPSframeworkPSpython_op_gen_internalDTcc {
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
   MHTracer_DTPStensorflowPSpythonPSframeworkPSpython_op_gen_internalDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSpythonPSframeworkPSpython_op_gen_internalDTcc() {
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

#include "tensorflow/python/framework/python_op_gen_internal.h"

#include <float.h>
#include <stdio.h>

#include <iomanip>
#include <sstream>
#include <unordered_map>

#include "absl/strings/escaping.h"
#include "absl/strings/str_replace.h"
#include "tensorflow/core/framework/api_def.pb.h"
#include "tensorflow/core/framework/attr_value.pb.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_def.pb.h"
#include "tensorflow/core/framework/op_def_util.h"
#include "tensorflow/core/framework/op_gen_lib.h"
#include "tensorflow/core/framework/tensor.pb.h"
#include "tensorflow/core/framework/tensor_shape.pb.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/lib/gtl/map_util.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/lib/strings/stringprintf.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {
namespace python_op_gen_internal {

const int kRightMargin = 78;
// Names specified in tf_export decorators are exported to
// TensorFlow 2.0 by default.
const int kLatestAPIExportVersion = 2;

bool IsPythonReserved(const string& s) {
   std::vector<std::string> mht_0_v;
   mht_0_v.push_back("s: \"" + s + "\"");
   MHTracer_DTPStensorflowPSpythonPSframeworkPSpython_op_gen_internalDTcc mht_0(mht_0_v, 223, "", "./tensorflow/python/framework/python_op_gen_internal.cc", "IsPythonReserved");

  static const std::set<string>* const kPythonReserved = new std::set<string>(
      {// Keywords in Python, from:
       //   import keyword
       //   print keyword.kwlist
       "and", "as", "assert", "break", "class", "continue", "def", "del",
       "elif", "else", "except", "exec", "finally", "for", "from", "global",
       "if", "import", "in", "is", "lambda", "not", "or", "pass", "print",
       "raise", "return", "try", "while", "with", "yield",
       // Built-in functions and types in Python, from:
       //   [x for x in dir(__builtins__) if not x[0].islower()]
       "ArithmeticError", "AssertionError", "AttributeError", "BaseException",
       "BufferError", "BytesWarning", "DeprecationWarning", "EOFError",
       "Ellipsis", "EnvironmentError", "Exception", "False",
       "FloatingPointError", "FutureWarning", "GeneratorExit", "IOError",
       "ImportError", "ImportWarning", "IndentationError", "IndexError",
       "KeyError", "KeyboardInterrupt", "LookupError", "MemoryError",
       "NameError", "None", "NotImplemented", "NotImplementedError", "OSError",
       "OverflowError", "PendingDeprecationWarning", "ReferenceError",
       "RuntimeError", "RuntimeWarning", "StandardError", "StopIteration",
       "SyntaxError", "SyntaxWarning", "SystemError", "SystemExit", "TabError",
       "True", "TypeError", "UnboundLocalError", "UnicodeDecodeError",
       "UnicodeEncodeError", "UnicodeError", "UnicodeTranslateError",
       "UnicodeWarning", "UserWarning", "ValueError", "Warning",
       "ZeroDivisionError", "__debug__", "__doc__", "__import__", "__name__",
       "__package__"});

  return kPythonReserved->count(s) > 0;
}

bool IsOpWithUnderscorePrefix(const string& s) {
   std::vector<std::string> mht_1_v;
   mht_1_v.push_back("s: \"" + s + "\"");
   MHTracer_DTPStensorflowPSpythonPSframeworkPSpython_op_gen_internalDTcc mht_1(mht_1_v, 257, "", "./tensorflow/python/framework/python_op_gen_internal.cc", "IsOpWithUnderscorePrefix");

  static const std::set<string>* const kUnderscoreOps = new std::set<string>(
      {// Lowercase built-in functions and types in Python, from:
       // [x for x in dir(__builtins__) if x[0].islower()] except "round".
       // These need to be excluded so they don't conflict with actual built-in
       // functions since we use '*' imports.
       "abs", "all", "any", "apply", "bin", "bool", "buffer", "bytearray",
       "bytes", "callable", "chr", "classmethod", "cmp", "coerce", "compile",
       "complex", "copyright", "credits", "delattr", "dict", "dir", "divmod",
       "enumerate", "eval", "execfile", "exit", "file", "filter", "float",
       "format", "frozenset", "getattr", "globals", "hasattr", "hash", "help",
       "hex", "id", "input", "int", "intern", "isinstance", "issubclass",
       "iter", "len", "license", "list", "locals", "long", "map", "max",
       "memoryview", "min", "next", "object", "oct", "open", "ord", "pow",
       "print", "property", "quit", "range", "raw_input", "reduce", "reload",
       "repr", "reversed", "set", "setattr", "slice", "sorted", "staticmethod",
       "str", "sum", "super", "tuple", "type", "unichr", "unicode", "vars",
       "xrange", "zip",
       // These have the same name as ops defined in Python and might be used
       // incorrectly depending on order of '*' imports.
       // TODO(annarev): reduce usage of '*' imports and remove these from the
       // list.
       "fused_batch_norm", "histogram_fixed_width", "stack",
       "batch_norm_with_global_normalization", "clip_by_value"});
  return kUnderscoreOps->count(s) > 0;
}

string AvoidPythonReserved(const string& s) {
   std::vector<std::string> mht_2_v;
   mht_2_v.push_back("s: \"" + s + "\"");
   MHTracer_DTPStensorflowPSpythonPSframeworkPSpython_op_gen_internalDTcc mht_2(mht_2_v, 288, "", "./tensorflow/python/framework/python_op_gen_internal.cc", "AvoidPythonReserved");

  // Convert namespace separators ('>' characters) to joiners
  string result = absl::StrReplaceAll(s, {{">", "_"}});

  if (IsPythonReserved(result)) return strings::StrCat(result, "_");
  return result;
}

// Indent the first line by "initial" spaces and all following lines
// by "rest" spaces.
string Indent(int initial, int rest, StringPiece in) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPSpythonPSframeworkPSpython_op_gen_internalDTcc mht_3(mht_3_v, 301, "", "./tensorflow/python/framework/python_op_gen_internal.cc", "Indent");

  // TODO(josh11b): Also word-wrapping?
  string copy(in.data(), in.size());
  absl::StripTrailingAsciiWhitespace(&copy);
  std::vector<string> v = str_util::Split(copy, '\n');

  string result;
  bool first = true;
  for (const string& line : v) {
    if (first) {
      result = strings::StrCat(Spaces(initial), line, "\n");
      first = false;
    } else {
      if (line.empty()) {
        strings::StrAppend(&result, "\n");
      } else {
        strings::StrAppend(&result, Spaces(rest), line, "\n");
      }
    }
  }
  return result;
}

// Adds append to *dest, with a space if the first line will be <= width,
// or a newline otherwise.
void AppendWithinWidth(string* dest, StringPiece append, int width) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPSpythonPSframeworkPSpython_op_gen_internalDTcc mht_4(mht_4_v, 329, "", "./tensorflow/python/framework/python_op_gen_internal.cc", "AppendWithinWidth");

  auto first_line = append.find('\n');
  if (first_line == string::npos) first_line = append.size();
  if (dest->size() + first_line + 1 /* space */ > static_cast<size_t>(width)) {
    strings::StrAppend(dest, "\n", append);
  } else {
    strings::StrAppend(dest, " ", append);
  }
}

// Like DataTypeString() but uses the Python names for the
// float types.
string PythonDataTypeString(DataType dtype) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPSpythonPSframeworkPSpython_op_gen_internalDTcc mht_5(mht_5_v, 344, "", "./tensorflow/python/framework/python_op_gen_internal.cc", "PythonDataTypeString");

  switch (dtype) {
    case DT_FLOAT:
      return "float32";
    case DT_DOUBLE:
      return "float64";
    default:
      return DataTypeString(dtype);
  }
}

string TypeString(DataType dtype, bool ref) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPSpythonPSframeworkPSpython_op_gen_internalDTcc mht_6(mht_6_v, 358, "", "./tensorflow/python/framework/python_op_gen_internal.cc", "TypeString");

  if (ref) {
    return strings::StrCat("mutable `", PythonDataTypeString(dtype), "`");
  } else {
    return strings::StrCat("`", PythonDataTypeString(dtype), "`");
  }
}

string TypeListString(const AttrValue& value) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPSpythonPSframeworkPSpython_op_gen_internalDTcc mht_7(mht_7_v, 369, "", "./tensorflow/python/framework/python_op_gen_internal.cc", "TypeListString");

  string ret;
  for (int t : value.list().type()) {
    if (!ret.empty()) strings::StrAppend(&ret, ", ");
    DataType dtype = static_cast<DataType>(t);
    if (IsRefType(dtype)) {
      strings::StrAppend(&ret, PythonDataTypeString(RemoveRefType(dtype)),
                         " mutable");
    } else {
      strings::StrAppend(&ret, "`", PythonDataTypeString(dtype), "`");
    }
  }
  return ret;
}

string SingleTensorName(DataType dtype, bool is_ref) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPSpythonPSframeworkPSpython_op_gen_internalDTcc mht_8(mht_8_v, 387, "", "./tensorflow/python/framework/python_op_gen_internal.cc", "SingleTensorName");

  const string type_str = TypeString(dtype, is_ref);
  return strings::StrCat("A `Tensor` of type ", type_str, ".");
}

const char kUnknownTensorType[] = {"A `Tensor`."};

string ArgTypeName(const OpDef& op_def, const OpDef::ArgDef& arg,
                   const std::unordered_map<string, string>& inferred_attrs,
                   bool is_output) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPSpythonPSframeworkPSpython_op_gen_internalDTcc mht_9(mht_9_v, 399, "", "./tensorflow/python/framework/python_op_gen_internal.cc", "ArgTypeName");

  if (!arg.number_attr().empty()) {
    // N Tensors with the same type
    const string* original_arg =
        gtl::FindOrNull(inferred_attrs, arg.number_attr());
    string prefix;
    if (original_arg == nullptr) {
      prefix = strings::StrCat("A list of `", arg.number_attr(), "`");
    } else if (*original_arg == arg.name()) {
      const OpDef::AttrDef* attr = FindAttr(arg.number_attr(), op_def);
      if (attr->has_minimum() && attr->minimum() > 0) {
        prefix = strings::StrCat("A list of at least ", attr->minimum());
      } else {
        prefix = "A list of";
      }
    } else {
      prefix = strings::StrCat("A list with the same length as `",
                               AvoidPythonReserved(*original_arg), "` of");
    }

    if (arg.type() != DT_INVALID) {
      return strings::StrCat(prefix, " `Tensor` objects with type ",
                             TypeString(arg.type(), arg.is_ref()), ".");
    } else {
      original_arg = gtl::FindOrNull(inferred_attrs, arg.type_attr());
      if (arg.is_ref()) {
        strings::StrAppend(&prefix, " mutable");
      }
      if (original_arg == nullptr) {
        return strings::StrCat(prefix, " `Tensor` objects with type `",
                               arg.type_attr(), "`.");
      } else if (*original_arg == arg.name()) {
        const OpDef::AttrDef* attr = FindAttr(arg.type_attr(), op_def);
        if (attr->has_allowed_values()) {
          return strings::StrCat(prefix,
                                 " `Tensor` objects with the same type in: ",
                                 TypeListString(attr->allowed_values()), ".");
        } else {
          return strings::StrCat(prefix,
                                 " `Tensor` objects with the same type.");
        }
      } else {
        return strings::StrCat(prefix,
                               " `Tensor` objects with the same type as `",
                               AvoidPythonReserved(*original_arg), "`.");
      }
    }
  } else if (!arg.type_attr().empty() || !arg.type_list_attr().empty()) {
    const bool is_list = !arg.type_list_attr().empty();
    const string attr_name = is_list ? arg.type_list_attr() : arg.type_attr();
    const OpDef::AttrDef* attr = FindAttr(attr_name, op_def);
    const string mutable_str = arg.is_ref() ? "mutable " : "";
    const string prefix =
        is_list ? strings::StrCat("A list of ", mutable_str, "`Tensor` objects")
                : strings::StrCat("A ", mutable_str, "`Tensor`");
    const string* original_arg = gtl::FindOrNull(inferred_attrs, attr_name);
    if (original_arg == nullptr) {
      return strings::StrCat(prefix, " of type `", attr_name, "`.");
    } else if (*original_arg == arg.name()) {
      if (attr->has_allowed_values()) {
        if (is_list) {
          return strings::StrCat(prefix, " with types from: ",
                                 TypeListString(attr->allowed_values()), ".");
        } else {
          return strings::StrCat(
              prefix, is_output ? ". Has one of the following types: "
                                : ". Must be one of the following types: ",
              TypeListString(attr->allowed_values()), ".");
        }
      } else {
        return strings::StrCat(prefix, ".");
      }
    } else {
      return strings::StrCat(prefix,
                             is_output ? ". Has the same type as `"
                                       : ". Must have the same type as `",
                             AvoidPythonReserved(*original_arg), "`.");
    }
  } else {
    return SingleTensorName(arg.type(), arg.is_ref());
  }
}

string GetReturns(const OpDef& op_def,
                  const std::vector<string>& output_type_string) {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPSpythonPSframeworkPSpython_op_gen_internalDTcc mht_10(mht_10_v, 486, "", "./tensorflow/python/framework/python_op_gen_internal.cc", "GetReturns");

  string result;
  DCHECK_EQ(op_def.output_arg_size(), output_type_string.size());
  const int num_outs = op_def.output_arg_size();
  strings::StrAppend(&result, "\n  Returns:\n");
  if (num_outs == 0) {
    strings::StrAppend(&result, "    The created Operation.\n");
  } else {
    if (num_outs == 1) {
      StringPiece description = op_def.output_arg(0).description();
      if (ConsumeEquals(&description)) {  // Skip the generated type info.
        strings::StrAppend(&result, Indent(4, 4, description));
      } else {
        // Special case of one output, don't use the name of the output unless
        // there is no description.
        string desc = output_type_string.empty() ? kUnknownTensorType
                                                 : output_type_string[0];
        if (desc == kUnknownTensorType) {
          // Special case where we don't understand how the output tensor type
          // depends on the input tensor types, just use the output arg
          // description if we can.
          if (!description.empty()) {
            desc = op_def.output_arg(0).description();
          } else if (!op_def.output_arg(0).name().empty()) {
            desc = strings::StrCat(" The ", op_def.output_arg(0).name(),
                                   " `Tensor`.");
          }
        } else if (!description.empty()) {
          AppendWithinWidth(&desc, description, kRightMargin - 4 /* indent */);
        }
        strings::StrAppend(&result, Indent(4, 4, desc));
      }
    } else {
      std::vector<string> out_names(num_outs);
      for (int i = 0; i < num_outs; ++i) {
        if (!op_def.output_arg(i).name().empty()) {
          out_names[i] = op_def.output_arg(i).name();
        } else {
          out_names[i] = strings::StrCat("output", i);
        }
      }
      strings::StrAppend(&result, "    A tuple of `Tensor` objects (",
                         absl::StrJoin(out_names, ", "), ").\n\n");
      for (int i = 0; i < num_outs; ++i) {
        string desc = strings::StrCat(out_names[i], ": ");
        StringPiece description = op_def.output_arg(i).description();
        if (ConsumeEquals(&description)) {  // Skip the generated type info.
          strings::StrAppend(&desc, description);
        } else {
          const string type = static_cast<size_t>(i) < output_type_string.size()
                                  ? output_type_string[i]
                                  : kUnknownTensorType;
          if (!description.empty()) {
            if (type == kUnknownTensorType) {
              // Special case where we don't understand how the output tensor
              // type depends on the input tensor types, so we just use the
              // output arg description.
              strings::StrAppend(&desc, description);
            } else {
              strings::StrAppend(&desc, type, " ", description);
            }
          } else {
            strings::StrAppend(&desc, type);
          }
        }
        strings::StrAppend(&result, Indent(4, 6, desc));
      }
    }
  }
  return result;
}

string StringToPython(const string& str) {
   std::vector<std::string> mht_11_v;
   mht_11_v.push_back("str: \"" + str + "\"");
   MHTracer_DTPStensorflowPSpythonPSframeworkPSpython_op_gen_internalDTcc mht_11(mht_11_v, 562, "", "./tensorflow/python/framework/python_op_gen_internal.cc", "StringToPython");

  return strings::StrCat("\"", absl::CEscape(str), "\"");
}

string DataTypeToPython(DataType dtype, const string& dtype_module) {
   std::vector<std::string> mht_12_v;
   mht_12_v.push_back("dtype_module: \"" + dtype_module + "\"");
   MHTracer_DTPStensorflowPSpythonPSframeworkPSpython_op_gen_internalDTcc mht_12(mht_12_v, 570, "", "./tensorflow/python/framework/python_op_gen_internal.cc", "DataTypeToPython");

  return strings::StrCat(dtype_module, PythonDataTypeString(dtype));
}

string ShapeToPython(const TensorShapeProto& shape) {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPSpythonPSframeworkPSpython_op_gen_internalDTcc mht_13(mht_13_v, 577, "", "./tensorflow/python/framework/python_op_gen_internal.cc", "ShapeToPython");

  if (shape.unknown_rank()) {
    return "None";
  }
  string python = "[";
  for (const auto& dim : shape.dim()) {
    if (python.size() > 1) strings::StrAppend(&python, ", ");
    if (!dim.name().empty()) {
      strings::StrAppend(&python, "(", StringToPython(dim.name()), ", ",
                         dim.size(), ")");
    } else {
      strings::StrAppend(&python, dim.size());
    }
  }
  strings::StrAppend(&python, "]");
  return python;
}

string TensorToPython(const TensorProto& proto) {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPSpythonPSframeworkPSpython_op_gen_internalDTcc mht_14(mht_14_v, 598, "", "./tensorflow/python/framework/python_op_gen_internal.cc", "TensorToPython");

  return proto.ShortDebugString();
}

string AttrListToPython(const AttrValue& value,
                        const string& dtype_module = "tf.") {
   std::vector<std::string> mht_15_v;
   MHTracer_DTPStensorflowPSpythonPSframeworkPSpython_op_gen_internalDTcc mht_15(mht_15_v, 606, "", "./tensorflow/python/framework/python_op_gen_internal.cc", "AttrListToPython");

  string ret;
  if (value.list().s_size() > 0) {
    for (int i = 0; i < value.list().s_size(); ++i) {
      if (i > 0) strings::StrAppend(&ret, ", ");
      strings::StrAppend(&ret, StringToPython(value.list().s(i)));
    }
  } else if (value.list().i_size() > 0) {
    for (int i = 0; i < value.list().i_size(); ++i) {
      if (i > 0) strings::StrAppend(&ret, ", ");
      strings::StrAppend(&ret, value.list().i(i));
    }
  } else if (value.list().f_size() > 0) {
    for (int i = 0; i < value.list().f_size(); ++i) {
      if (i > 0) strings::StrAppend(&ret, ", ");
      strings::StrAppend(&ret, value.list().f(i));
    }
  } else if (value.list().b_size() > 0) {
    for (int i = 0; i < value.list().b_size(); ++i) {
      if (i > 0) strings::StrAppend(&ret, ", ");
      strings::StrAppend(&ret, value.list().b(i) ? "True" : "False");
    }
  } else if (value.list().type_size() > 0) {
    for (int i = 0; i < value.list().type_size(); ++i) {
      if (i > 0) strings::StrAppend(&ret, ", ");
      strings::StrAppend(&ret,
                         DataTypeToPython(value.list().type(i), dtype_module));
    }
  } else if (value.list().shape_size() > 0) {
    for (int i = 0; i < value.list().shape_size(); ++i) {
      if (i > 0) strings::StrAppend(&ret, ", ");
      strings::StrAppend(&ret, ShapeToPython(value.list().shape(i)));
    }
  } else if (value.list().tensor_size() > 0) {
    for (int i = 0; i < value.list().tensor_size(); ++i) {
      if (i > 0) strings::StrAppend(&ret, ", ");
      strings::StrAppend(&ret, TensorToPython(value.list().tensor(i)));
    }
  } else if (value.list().func_size() > 0) {
    for (int i = 0; i < value.list().func_size(); ++i) {
      if (i > 0) strings::StrAppend(&ret, ", ");
      strings::StrAppend(&ret, StringToPython(value.list().func(i).name()));
    }
  }
  return ret;
}

// NOTE: The return value may contain spaces (for example, it could be
// a string "foo bar" with an embedded space) and is not safe to pass
// to WordWrap().
string AttrValueToPython(const string& type, const AttrValue& value,
                         const string& dtype_module) {
   std::vector<std::string> mht_16_v;
   mht_16_v.push_back("type: \"" + type + "\"");
   mht_16_v.push_back("dtype_module: \"" + dtype_module + "\"");
   MHTracer_DTPStensorflowPSpythonPSframeworkPSpython_op_gen_internalDTcc mht_16(mht_16_v, 662, "", "./tensorflow/python/framework/python_op_gen_internal.cc", "AttrValueToPython");

  if (type == "string") {
    return StringToPython(value.s());
  } else if (type == "int") {
    return strings::StrCat(value.i());
  } else if (type == "float") {
    if (std::isnan(value.f()) || std::isinf(value.f())) {
      return strings::StrCat("float('", value.f(), "')");
    } else {
      // Use locale-independent conversion.
      static_assert(FLT_DIG < 10, "FLT_DIG is too big");
      std::ostringstream s;
      s.imbue(std::locale::classic());
      s << std::setprecision(FLT_DIG) << value.f();
      // If there is no I/O error for `std::ostringstream s` return s.str(),
      // otherwise fallback to strings::StrCat(value.f()).
      if (s.good()) {
        return s.str();
      }
      return strings::StrCat(value.f());
    }
  } else if (type == "bool") {
    return value.b() ? "True" : "False";
  } else if (type == "type") {
    return DataTypeToPython(value.type(), dtype_module);
  } else if (type == "shape") {
    return ShapeToPython(value.shape());
  } else if (type == "tensor") {
    return TensorToPython(value.tensor());
  } else if (type == "func") {
    return StringToPython(value.func().name());
  } else if (absl::StartsWith(type, "list(")) {
    return strings::StrCat("[", AttrListToPython(value, dtype_module), "]");
  } else {
    return "?";
  }
}

void GenerateLowerCaseOpName(const string& str, string* result) {
   std::vector<std::string> mht_17_v;
   mht_17_v.push_back("str: \"" + str + "\"");
   MHTracer_DTPStensorflowPSpythonPSframeworkPSpython_op_gen_internalDTcc mht_17(mht_17_v, 704, "", "./tensorflow/python/framework/python_op_gen_internal.cc", "GenerateLowerCaseOpName");

  const char joiner = '_';
  const char namespace_separator = '>';
  const int last_index = str.size() - 1;
  for (int i = 0; i <= last_index; ++i) {
    const char c = str[i];
    // Convert namespace separators ('>' characters) to joiners
    if (c == namespace_separator) {
      result->push_back(joiner);
      continue;
    }

    // Emit a joiner only if a previous-lower-to-now-upper or a
    // now-upper-to-next-lower transition happens.
    // (But don't emit an extra joiner if we just saw a namespace separator
    if (isupper(c) && (i > 0)) {
      if (islower(str[i - 1]) || ((i < last_index) && islower(str[i + 1]))) {
        if (!(str[i - 1] == namespace_separator)) {
          result->push_back(joiner);
        }
      }
    }
    result->push_back(tolower(c));
  }
}

static void AddDelimiter(string* append_to, const string& delim) {
   std::vector<std::string> mht_18_v;
   mht_18_v.push_back("delim: \"" + delim + "\"");
   MHTracer_DTPStensorflowPSpythonPSframeworkPSpython_op_gen_internalDTcc mht_18(mht_18_v, 734, "", "./tensorflow/python/framework/python_op_gen_internal.cc", "AddDelimiter");

  if (!append_to->empty()) strings::StrAppend(append_to, delim);
}

const ApiDef::Attr* FindAttr(StringPiece name, const ApiDef& api_def) {
   std::vector<std::string> mht_19_v;
   MHTracer_DTPStensorflowPSpythonPSframeworkPSpython_op_gen_internalDTcc mht_19(mht_19_v, 741, "", "./tensorflow/python/framework/python_op_gen_internal.cc", "FindAttr");

  for (int i = 0; i < api_def.attr_size(); ++i) {
    if (api_def.attr(i).name() == name) {
      return &api_def.attr(i);
    }
  }
  return nullptr;
}

GenPythonOp::GenPythonOp(const OpDef& op_def, const ApiDef& api_def,
                         const string& function_name, bool add_type_annotations)
    : op_def_(op_def),
      api_def_(api_def),
      function_name_(function_name),
      add_type_annotations_(add_type_annotations),
      num_outs_(op_def.output_arg_size()) {
   std::vector<std::string> mht_20_v;
   mht_20_v.push_back("function_name: \"" + function_name + "\"");
   MHTracer_DTPStensorflowPSpythonPSframeworkPSpython_op_gen_internalDTcc mht_20(mht_20_v, 760, "", "./tensorflow/python/framework/python_op_gen_internal.cc", "GenPythonOp::GenPythonOp");
}

GenPythonOp::~GenPythonOp() {
   std::vector<std::string> mht_21_v;
   MHTracer_DTPStensorflowPSpythonPSframeworkPSpython_op_gen_internalDTcc mht_21(mht_21_v, 765, "", "./tensorflow/python/framework/python_op_gen_internal.cc", "GenPythonOp::~GenPythonOp");
}

string GenPythonOp::Code() {
   std::vector<std::string> mht_22_v;
   MHTracer_DTPStensorflowPSpythonPSframeworkPSpython_op_gen_internalDTcc mht_22(mht_22_v, 770, "", "./tensorflow/python/framework/python_op_gen_internal.cc", "GenPythonOp::Code");

  // This has all the input args followed by those attrs that don't have
  // defaults.
  std::vector<ParamNames> params_no_default;
  // The parameters with defaults (these have to be listed after those without).
  // No input args are included, just attrs.
  std::vector<ParamNames> params_with_default;

  for (int i = 0; i < api_def_.arg_order_size(); ++i) {
    const auto& arg = *FindInputArg(api_def_.arg_order(i), op_def_);
    const auto& api_def_arg = *FindInputArg(api_def_.arg_order(i), api_def_);
    params_no_default.emplace_back(api_def_arg.name(), api_def_arg.rename_to());
    if (!arg.type_attr().empty()) {
      gtl::InsertIfNotPresent(&inferred_attrs_, arg.type_attr(), arg.name());
    } else if (!arg.type_list_attr().empty()) {
      gtl::InsertIfNotPresent(&inferred_attrs_, arg.type_list_attr(),
                              arg.name());
    }
    if (!arg.number_attr().empty()) {
      gtl::InsertIfNotPresent(&inferred_attrs_, arg.number_attr(), arg.name());
    }
  }
  for (int i = 0; i < api_def_.attr_size(); ++i) {
    const auto& attr(api_def_.attr(i));
    // Do not add inferred attrs to the Python function signature.
    if (inferred_attrs_.find(attr.name()) == inferred_attrs_.end()) {
      if (attr.has_default_value()) {
        params_with_default.emplace_back(attr.name(), attr.rename_to());
      } else {
        params_no_default.emplace_back(attr.name(), attr.rename_to());
      }
    }
  }

  // Save the list of attr parameters (attrs that won't be inferred),
  // those with defaults go at the end.
  // Get the attrs in the order we want by taking the attrs without defaults
  // from the end of args_no_default, and adding args_no_default.
  attrs_.reserve(params_no_default.size() - op_def_.input_arg_size() +
                 params_with_default.size());
  for (int i = op_def_.input_arg_size(), end = params_no_default.size();
       i < end; ++i) {
    attrs_.push_back(params_no_default[i].GetName());
  }
  for (int i = 0, end = params_with_default.size(); i < end; ++i) {
    attrs_.push_back(params_with_default[i].GetName());
  }

  param_names_.reserve(params_no_default.size() + params_with_default.size());
  param_names_.insert(param_names_.begin(), params_no_default.begin(),
                      params_no_default.end());
  for (const auto& param : params_with_default) {
    param_names_.push_back(param);
  }

  string parameters;
  for (const auto& param : params_no_default) {
    AddDelimiter(&parameters, ", ");
    strings::StrAppend(&parameters, param.GetRenameTo());
  }
  for (const auto& param_and_default : params_with_default) {
    AddDelimiter(&parameters, ", ");
    strings::StrAppend(&parameters, param_and_default.GetRenameTo(), "=None");
  }
  AddDelimiter(&parameters, ", ");
  strings::StrAppend(&parameters, "name=None");

  AddExport();
  AddDefLine(parameters);
  AddDocStringDescription();
  AddDocStringArgs();
  AddDocStringInputs();
  AddDocStringAttrs();
  AddDocStringNameArg();
  AddOutputGlobals();
  AddDocStringOutputs();
  strings::StrAppend(&result_, "  \"\"\"\n");
  AddBody("  ");
  strings::StrAppend(&result_, "\n\n");

  return prelude_ + result_;
}

void GenPythonOp::AddExport() {
   std::vector<std::string> mht_23_v;
   MHTracer_DTPStensorflowPSpythonPSframeworkPSpython_op_gen_internalDTcc mht_23(mht_23_v, 856, "", "./tensorflow/python/framework/python_op_gen_internal.cc", "GenPythonOp::AddExport");

  if (api_def_.visibility() != ApiDef::VISIBLE) {
    return;
  }
  // Whether op should be available in latest export version.
  bool op_available_in_latest =
      !api_def_.deprecation_version() ||
      api_def_.deprecation_version() > kLatestAPIExportVersion;

  string names;
  string names_v1;
  string deprecated_endpoints;

  for (const auto& endpoint : api_def_.endpoint()) {
    string endpoint_name;
    python_op_gen_internal::GenerateLowerCaseOpName(endpoint.name(),
                                                    &endpoint_name);
    if (endpoint.deprecated() || endpoint.deprecation_version() > 0) {
      AddDelimiter(&deprecated_endpoints, ", ");
      strings::StrAppend(&deprecated_endpoints, "'", endpoint_name, "'");
    }
    // Add all endpoints to TensorFlow 1.* API.
    AddDelimiter(&names_v1, ", ");
    strings::StrAppend(&names_v1, "'", endpoint_name, "'");
    // Add non-deprecated endpoints to TensorFlow 2.* API.
    if (op_available_in_latest &&
        (!endpoint.deprecation_version() ||
         endpoint.deprecation_version() > kLatestAPIExportVersion)) {
      AddDelimiter(&names, ", ");
      strings::StrAppend(&names, "'", endpoint_name, "'");
    }
  }

  // tf_export decorator has the following format:
  // @tf_export(v2_name, v2_name, v1=[v1_name, v1_name])
  if (names != names_v1) {
    AddDelimiter(&names, ", ");
    strings::StrAppend(&names, "v1=[", names_v1, "]");
  }
  strings::StrAppend(&result_, "@tf_export(", names, ")\n");

  // If all endpoints are deprecated, add @deprecated decorator.
  if (!api_def_.deprecation_message().empty()) {
    const string instructions = api_def_.deprecation_message();
    strings::StrAppend(&result_, "@deprecated(None, '", instructions, "')\n");
  }
  // Add @deprecated_endpoints decorator.
  if (!deprecated_endpoints.empty()) {
    strings::StrAppend(&result_, "@deprecated_endpoints(", deprecated_endpoints,
                       ")\n");
  }
}

void GenPythonOp::AddDefLine(const string& function_name,
                             const string& parameters) {
   std::vector<std::string> mht_24_v;
   mht_24_v.push_back("function_name: \"" + function_name + "\"");
   mht_24_v.push_back("parameters: \"" + parameters + "\"");
   MHTracer_DTPStensorflowPSpythonPSframeworkPSpython_op_gen_internalDTcc mht_24(mht_24_v, 915, "", "./tensorflow/python/framework/python_op_gen_internal.cc", "GenPythonOp::AddDefLine");

  strings::StrAppend(&result_, "def ", function_name, "(", parameters, "):\n");
}

void GenPythonOp::AddDefLine(const string& parameters) {
   std::vector<std::string> mht_25_v;
   mht_25_v.push_back("parameters: \"" + parameters + "\"");
   MHTracer_DTPStensorflowPSpythonPSframeworkPSpython_op_gen_internalDTcc mht_25(mht_25_v, 923, "", "./tensorflow/python/framework/python_op_gen_internal.cc", "GenPythonOp::AddDefLine");

  AddDefLine(function_name_, parameters);
}

void GenPythonOp::AddDocStringDescription() {
   std::vector<std::string> mht_26_v;
   MHTracer_DTPStensorflowPSpythonPSframeworkPSpython_op_gen_internalDTcc mht_26(mht_26_v, 930, "", "./tensorflow/python/framework/python_op_gen_internal.cc", "GenPythonOp::AddDocStringDescription");

  string comment;
  if (api_def_.summary().empty()) {
    comment = "TODO: add doc.\n";
  } else {
    comment = strings::StrCat(api_def_.summary(), "\n");
    if (!api_def_.description().empty()) {
      strings::StrAppend(&comment, "\n", Indent(2, 2, api_def_.description()));
    }
  }
  strings::StrAppend(&result_, "  r\"\"\"", comment, "\n");
}

void GenPythonOp::AddDocStringArgs() {
   std::vector<std::string> mht_27_v;
   MHTracer_DTPStensorflowPSpythonPSframeworkPSpython_op_gen_internalDTcc mht_27(mht_27_v, 946, "", "./tensorflow/python/framework/python_op_gen_internal.cc", "GenPythonOp::AddDocStringArgs");

  strings::StrAppend(&result_, "  Args:\n");
}

void GenPythonOp::AddDocStringInputs() {
   std::vector<std::string> mht_28_v;
   MHTracer_DTPStensorflowPSpythonPSframeworkPSpython_op_gen_internalDTcc mht_28(mht_28_v, 953, "", "./tensorflow/python/framework/python_op_gen_internal.cc", "GenPythonOp::AddDocStringInputs");

  for (int i = 0; i < api_def_.arg_order_size(); ++i) {
    const auto& arg = *FindInputArg(api_def_.arg_order(i), op_def_);
    const auto& api_def_arg = *FindInputArg(api_def_.arg_order(i), api_def_);
    StringPiece description = api_def_arg.description();
    string desc;
    if (ConsumeEquals(&description)) {  // Skip the generated type info.
      desc = strings::StrCat(param_names_[i].GetRenameTo(), ": ");
    } else {
      desc = strings::StrCat(param_names_[i].GetRenameTo(), ": ",
                             ArgTypeName(op_def_, arg, inferred_attrs_, false));
    }
    if (!description.empty()) {
      AppendWithinWidth(&desc, description, kRightMargin - 4 /* indent */);
    }
    strings::StrAppend(&result_, Indent(4, 6, desc));
  }
}

void GenPythonOp::AddDocStringAttrs() {
   std::vector<std::string> mht_29_v;
   MHTracer_DTPStensorflowPSpythonPSframeworkPSpython_op_gen_internalDTcc mht_29(mht_29_v, 975, "", "./tensorflow/python/framework/python_op_gen_internal.cc", "GenPythonOp::AddDocStringAttrs");

  for (const string& name : attrs_) {
    const auto& attr = *FindAttr(name, op_def_);
    const auto& api_def_attr = *FindAttr(name, api_def_);
    string desc =
        strings::StrCat(AvoidPythonReserved(api_def_attr.rename_to()), ": ");

    static const char* const kAttrTypeName[][2] = {
        {"string", "`string`"},
        {"list(string)", "list of `strings`"},
        {"int", "`int`"},
        {"list(int)", "list of `ints`"},
        {"float", "`float`"},
        {"list(float)", "list of `floats`"},
        {"bool", "`bool`"},
        {"list(bool)", "list of `bools`"},
        {"type", "`tf.DType`"},
        {"list(type)", "list of `tf.DTypes`"},
        {"shape", "`tf.TensorShape` or list of `ints`"},
        {"list(shape)",
         "list of shapes (each a `tf.TensorShape` or list of `ints`)"},
        {"tensor", "`tf.TensorProto`"},
        {"list(tensor)", "list of `tf.TensorProto` objects"},
        {"func", "function decorated with @Defun"},
        {"list(func)", "list of functions decorated with @Defun"},
    };
    for (size_t i = 0; i < TF_ARRAYSIZE(kAttrTypeName); ++i) {
      if (attr.type() == kAttrTypeName[i][0]) {
        string s;
        if (api_def_attr.has_default_value()) {
          s = strings::StrCat("optional ", kAttrTypeName[i][1]);
        } else {
          s = kAttrTypeName[i][1];
        }
        if (s[0] == 'o' || (s[0] == '`' && (s[1] == 'i' || s[1] == 'o'))) {
          strings::StrAppend(&desc, "An ", s);
        } else {
          strings::StrAppend(&desc, "A ", s);
        }
        break;
      }
    }

    if (attr.has_allowed_values()) {
      strings::StrAppend(&desc, " from: `",
                         AttrListToPython(attr.allowed_values()), "`");
    }

    if (attr.has_minimum()) {
      if (attr.type() == "int") {
        strings::StrAppend(&desc, " that is `>= ", attr.minimum(), "`");
      } else if (attr.minimum() > 0) {
        strings::StrAppend(&desc, " that has length `>= ", attr.minimum(), "`");
      }
    }

    strings::StrAppend(&desc, ".");

    if (api_def_attr.has_default_value()) {
      strings::StrAppend(
          &desc, " Defaults to `",
          AttrValueToPython(attr.type(), api_def_attr.default_value()), "`.");
    }
    if (!api_def_attr.description().empty()) {
      AppendWithinWidth(&desc, api_def_attr.description(),
                        kRightMargin - 4 /* indent */);
    }
    strings::StrAppend(&result_, Indent(4, 6, desc));
  }
}

void GenPythonOp::AddDocStringNameArg() {
   std::vector<std::string> mht_30_v;
   MHTracer_DTPStensorflowPSpythonPSframeworkPSpython_op_gen_internalDTcc mht_30(mht_30_v, 1049, "", "./tensorflow/python/framework/python_op_gen_internal.cc", "GenPythonOp::AddDocStringNameArg");

  strings::StrAppend(&result_,
                     "    name: A name for the operation (optional).\n");
}

void GenPythonOp::AddOutputGlobals() {
   std::vector<std::string> mht_31_v;
   MHTracer_DTPStensorflowPSpythonPSframeworkPSpython_op_gen_internalDTcc mht_31(mht_31_v, 1057, "", "./tensorflow/python/framework/python_op_gen_internal.cc", "GenPythonOp::AddOutputGlobals");

  // Generate a namedtuple class to hold the outputs, if there are multiple.
  // Example:
  //
  // _OpOutputs = collections.namedtuple(
  //     "_OpOutputs",
  //     "out1 out2 out3")
  if (num_outs_ > 1) {
    std::vector<string> out_names;
    out_names.reserve(num_outs_);
    for (int i = 0; i < num_outs_; ++i) {
      const string out_name = !api_def_.out_arg(i).rename_to().empty()
                                  ? api_def_.out_arg(i).rename_to()
                                  : strings::StrCat("output", i);
      out_names.push_back(strings::StrCat("\"", out_name, "\""));
    }

    strings::StrAppend(&prelude_, "_", AvoidPythonReserved(op_def_.name()),
                       "Output = collections.namedtuple(\n");
    strings::StrAppend(&prelude_, "    \"", AvoidPythonReserved(op_def_.name()),
                       "\",\n");
    strings::StrAppend(&prelude_, "    [", absl::StrJoin(out_names, ", "),
                       "])");
    strings::StrAppend(&prelude_, "\n\n");
  }
  strings::StrAppend(&prelude_, "\n");
}

void GenPythonOp::AddDocStringOutputs() {
   std::vector<std::string> mht_32_v;
   MHTracer_DTPStensorflowPSpythonPSframeworkPSpython_op_gen_internalDTcc mht_32(mht_32_v, 1088, "", "./tensorflow/python/framework/python_op_gen_internal.cc", "GenPythonOp::AddDocStringOutputs");

  std::vector<string> output_type_string;
  output_type_string.reserve(num_outs_);
  for (int i = 0; i < num_outs_; ++i) {
    output_type_string.push_back(
        ArgTypeName(op_def_, op_def_.output_arg(i), inferred_attrs_, true));
  }
  strings::StrAppend(&result_, GetReturns(op_def_, output_type_string));
}

void GenPythonOp::AddBody(const string& prefix) {
   std::vector<std::string> mht_33_v;
   mht_33_v.push_back("prefix: \"" + prefix + "\"");
   MHTracer_DTPStensorflowPSpythonPSframeworkPSpython_op_gen_internalDTcc mht_33(mht_33_v, 1102, "", "./tensorflow/python/framework/python_op_gen_internal.cc", "GenPythonOp::AddBody");

  const string apply_prefix = strings::StrCat(
      prefix, "_result = _op_def_lib.apply_op(\"", op_def_.name(), "\", ");
  AddBodyNoReturn(apply_prefix);
  if (num_outs_ > 1) {
    strings::StrAppend(&result_, prefix, "_result = _",
                       AvoidPythonReserved(op_def_.name()),
                       "Output._make(_result)\n");
  }
  strings::StrAppend(&result_, prefix, "return _result\n");
}

void GenPythonOp::AddBodyNoReturn(const string& apply_prefix) {
   std::vector<std::string> mht_34_v;
   mht_34_v.push_back("apply_prefix: \"" + apply_prefix + "\"");
   MHTracer_DTPStensorflowPSpythonPSframeworkPSpython_op_gen_internalDTcc mht_34(mht_34_v, 1118, "", "./tensorflow/python/framework/python_op_gen_internal.cc", "GenPythonOp::AddBodyNoReturn");

  string args;
  for (size_t i = 0; i < param_names_.size(); ++i) {
    strings::StrAppend(&args, AvoidPythonReserved(param_names_[i].GetName()),
                       "=", param_names_[i].GetRenameTo(), ", ");
  }
  strings::StrAppend(&args, "name=name)");

  strings::StrAppend(&result_,
                     // Wrap the arguments, and indent to the (.
                     WordWrap(apply_prefix, args, kRightMargin), "\n");
}

}  // namespace python_op_gen_internal
}  // namespace tensorflow
