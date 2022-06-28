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

#ifndef TENSORFLOW_JAVA_SRC_GEN_CC_JAVA_DEFS_H_
#define TENSORFLOW_JAVA_SRC_GEN_CC_JAVA_DEFS_H_
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
class MHTracer_DTPStensorflowPSjavaPSsrcPSgenPSccPSjava_defsDTh {
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
   MHTracer_DTPStensorflowPSjavaPSsrcPSgenPSccPSjava_defsDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSjavaPSsrcPSgenPSccPSjava_defsDTh() {
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


#include <list>
#include <map>
#include <string>
#include <utility>

#include "tensorflow/core/framework/types.h"

namespace tensorflow {
namespace java {

// An enumeration of different modifiers commonly used in Java
enum Modifier {
  PACKAGE = 0,
  PUBLIC = (1 << 0),
  PROTECTED = (1 << 1),
  PRIVATE = (1 << 2),
  STATIC = (1 << 3),
  FINAL = (1 << 4),
};

class Annotation;

// A definition of any kind of Java type (classes, interfaces...)
//
// Note that most of the data fields of this class are only useful in specific
// contexts and are not required in many cases. For example, annotations and
// supertypes are only useful when declaring a type.
class Type {
 public:
  enum Kind {
    PRIMITIVE, CLASS, INTERFACE, ENUM, GENERIC, ANNOTATION
  };
  static const Type Byte() {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSjavaPSsrcPSgenPSccPSjava_defsDTh mht_0(mht_0_v, 220, "", "./tensorflow/java/src/gen/cc/java_defs.h", "Byte");

    return Type(Type::PRIMITIVE, "byte");
  }
  static const Type Char() {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSjavaPSsrcPSgenPSccPSjava_defsDTh mht_1(mht_1_v, 226, "", "./tensorflow/java/src/gen/cc/java_defs.h", "Char");

    return Type(Type::PRIMITIVE, "char");
  }
  static const Type Short() {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSjavaPSsrcPSgenPSccPSjava_defsDTh mht_2(mht_2_v, 232, "", "./tensorflow/java/src/gen/cc/java_defs.h", "Short");

    return Type(Type::PRIMITIVE, "short");
  }
  static const Type Int() {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPSjavaPSsrcPSgenPSccPSjava_defsDTh mht_3(mht_3_v, 238, "", "./tensorflow/java/src/gen/cc/java_defs.h", "Int");

    return Type(Type::PRIMITIVE, "int");
  }
  static const Type Long() {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPSjavaPSsrcPSgenPSccPSjava_defsDTh mht_4(mht_4_v, 244, "", "./tensorflow/java/src/gen/cc/java_defs.h", "Long");

    return Type(Type::PRIMITIVE, "long");
  }
  static const Type Float() {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPSjavaPSsrcPSgenPSccPSjava_defsDTh mht_5(mht_5_v, 250, "", "./tensorflow/java/src/gen/cc/java_defs.h", "Float");

    return Type(Type::PRIMITIVE, "float");
  }
  static const Type Double() {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPSjavaPSsrcPSgenPSccPSjava_defsDTh mht_6(mht_6_v, 256, "", "./tensorflow/java/src/gen/cc/java_defs.h", "Double");

    return Type(Type::PRIMITIVE, "double");
  }
  static const Type Boolean() {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPSjavaPSsrcPSgenPSccPSjava_defsDTh mht_7(mht_7_v, 262, "", "./tensorflow/java/src/gen/cc/java_defs.h", "Boolean");

    return Type(Type::PRIMITIVE, "boolean");
  }
  static const Type Void() {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPSjavaPSsrcPSgenPSccPSjava_defsDTh mht_8(mht_8_v, 268, "", "./tensorflow/java/src/gen/cc/java_defs.h", "Void");

    // For simplicity, we consider 'void' as a primitive type, like the Java
    // Reflection API does
    return Type(Type::PRIMITIVE, "void");
  }
  static Type Generic(const string& name) {
   std::vector<std::string> mht_9_v;
   mht_9_v.push_back("name: \"" + name + "\"");
   MHTracer_DTPStensorflowPSjavaPSsrcPSgenPSccPSjava_defsDTh mht_9(mht_9_v, 277, "", "./tensorflow/java/src/gen/cc/java_defs.h", "Generic");
 return Type(Type::GENERIC, name); }
  static Type Wildcard() {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPSjavaPSsrcPSgenPSccPSjava_defsDTh mht_10(mht_10_v, 281, "", "./tensorflow/java/src/gen/cc/java_defs.h", "Wildcard");
 return Type(Type::GENERIC, ""); }
  static Type Class(const string& name, const string& package = "") {
   std::vector<std::string> mht_11_v;
   mht_11_v.push_back("name: \"" + name + "\"");
   MHTracer_DTPStensorflowPSjavaPSsrcPSgenPSccPSjava_defsDTh mht_11(mht_11_v, 286, "", "./tensorflow/java/src/gen/cc/java_defs.h", "Class");

    return Type(Type::CLASS, name, package);
  }
  static Type Interface(const string& name, const string& package = "") {
   std::vector<std::string> mht_12_v;
   mht_12_v.push_back("name: \"" + name + "\"");
   MHTracer_DTPStensorflowPSjavaPSsrcPSgenPSccPSjava_defsDTh mht_12(mht_12_v, 293, "", "./tensorflow/java/src/gen/cc/java_defs.h", "Interface");

    return Type(Type::INTERFACE, name, package);
  }
  static Type Enum(const string& name, const string& package = "") {
   std::vector<std::string> mht_13_v;
   mht_13_v.push_back("name: \"" + name + "\"");
   MHTracer_DTPStensorflowPSjavaPSsrcPSgenPSccPSjava_defsDTh mht_13(mht_13_v, 300, "", "./tensorflow/java/src/gen/cc/java_defs.h", "Enum");

    return Type(Type::ENUM, name, package);
  }
  static Type ClassOf(const Type& type) {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPSjavaPSsrcPSgenPSccPSjava_defsDTh mht_14(mht_14_v, 306, "", "./tensorflow/java/src/gen/cc/java_defs.h", "ClassOf");

    return Class("Class").add_parameter(type);
  }
  static Type ListOf(const Type& type) {
   std::vector<std::string> mht_15_v;
   MHTracer_DTPStensorflowPSjavaPSsrcPSgenPSccPSjava_defsDTh mht_15(mht_15_v, 312, "", "./tensorflow/java/src/gen/cc/java_defs.h", "ListOf");

    return Interface("List", "java.util").add_parameter(type);
  }
  static Type IterableOf(const Type& type) {
   std::vector<std::string> mht_16_v;
   MHTracer_DTPStensorflowPSjavaPSsrcPSgenPSccPSjava_defsDTh mht_16(mht_16_v, 318, "", "./tensorflow/java/src/gen/cc/java_defs.h", "IterableOf");

    return Interface("Iterable").add_parameter(type);
  }
  static Type ForDataType(DataType data_type) {
   std::vector<std::string> mht_17_v;
   MHTracer_DTPStensorflowPSjavaPSsrcPSgenPSccPSjava_defsDTh mht_17(mht_17_v, 324, "", "./tensorflow/java/src/gen/cc/java_defs.h", "ForDataType");

    switch (data_type) {
      case DataType::DT_BOOL:
        return Class("Boolean");
      case DataType::DT_STRING:
        return Class("String");
      case DataType::DT_FLOAT:
        return Class("Float");
      case DataType::DT_DOUBLE:
        return Class("Double");
      case DataType::DT_UINT8:
        return Class("UInt8", "org.tensorflow.types");
      case DataType::DT_INT32:
        return Class("Integer");
      case DataType::DT_INT64:
        return Class("Long");
      case DataType::DT_RESOURCE:
        // TODO(karllessard) create a Resource utility class that could be
        // used to store a resource and its type (passed in a second argument).
        // For now, we need to force a wildcard and we will unfortunately lose
        // track of the resource type.
        // Falling through...
      default:
        // Any other datatypes does not have a equivalent in Java and must
        // remain a wildcard (e.g. DT_COMPLEX64, DT_QINT8, ...)
        return Wildcard();
    }
  }
  const Kind& kind() const {
   std::vector<std::string> mht_18_v;
   MHTracer_DTPStensorflowPSjavaPSsrcPSgenPSccPSjava_defsDTh mht_18(mht_18_v, 355, "", "./tensorflow/java/src/gen/cc/java_defs.h", "kind");
 return kind_; }
  const string& name() const {
   std::vector<std::string> mht_19_v;
   MHTracer_DTPStensorflowPSjavaPSsrcPSgenPSccPSjava_defsDTh mht_19(mht_19_v, 359, "", "./tensorflow/java/src/gen/cc/java_defs.h", "name");
 return name_; }
  const string& package() const {
   std::vector<std::string> mht_20_v;
   MHTracer_DTPStensorflowPSjavaPSsrcPSgenPSccPSjava_defsDTh mht_20(mht_20_v, 363, "", "./tensorflow/java/src/gen/cc/java_defs.h", "package");
 return package_; }
  const string canonical_name() const {
   std::vector<std::string> mht_21_v;
   MHTracer_DTPStensorflowPSjavaPSsrcPSgenPSccPSjava_defsDTh mht_21(mht_21_v, 367, "", "./tensorflow/java/src/gen/cc/java_defs.h", "canonical_name");

    return package_.empty() ? name_ : package_ + "." + name_;
  }
  bool wildcard() const {
   std::vector<std::string> mht_22_v;
   MHTracer_DTPStensorflowPSjavaPSsrcPSgenPSccPSjava_defsDTh mht_22(mht_22_v, 373, "", "./tensorflow/java/src/gen/cc/java_defs.h", "wildcard");
 return name_.empty(); }  // only wildcards has no name
  const std::list<Type>& parameters() const {
   std::vector<std::string> mht_23_v;
   MHTracer_DTPStensorflowPSjavaPSsrcPSgenPSccPSjava_defsDTh mht_23(mht_23_v, 377, "", "./tensorflow/java/src/gen/cc/java_defs.h", "parameters");
 return parameters_; }
  Type& add_parameter(const Type& parameter) {
   std::vector<std::string> mht_24_v;
   MHTracer_DTPStensorflowPSjavaPSsrcPSgenPSccPSjava_defsDTh mht_24(mht_24_v, 381, "", "./tensorflow/java/src/gen/cc/java_defs.h", "add_parameter");

    parameters_.push_back(parameter);
    return *this;
  }
  const std::list<Annotation>& annotations() const {
   std::vector<std::string> mht_25_v;
   MHTracer_DTPStensorflowPSjavaPSsrcPSgenPSccPSjava_defsDTh mht_25(mht_25_v, 388, "", "./tensorflow/java/src/gen/cc/java_defs.h", "annotations");
 return annotations_; }
  Type& add_annotation(const Annotation& annotation) {
   std::vector<std::string> mht_26_v;
   MHTracer_DTPStensorflowPSjavaPSsrcPSgenPSccPSjava_defsDTh mht_26(mht_26_v, 392, "", "./tensorflow/java/src/gen/cc/java_defs.h", "add_annotation");

    annotations_.push_back(annotation);
    return *this;
  }
  const std::list<Type>& supertypes() const {
   std::vector<std::string> mht_27_v;
   MHTracer_DTPStensorflowPSjavaPSsrcPSgenPSccPSjava_defsDTh mht_27(mht_27_v, 399, "", "./tensorflow/java/src/gen/cc/java_defs.h", "supertypes");
 return supertypes_; }
  Type& add_supertype(const Type& type) {
   std::vector<std::string> mht_28_v;
   MHTracer_DTPStensorflowPSjavaPSsrcPSgenPSccPSjava_defsDTh mht_28(mht_28_v, 403, "", "./tensorflow/java/src/gen/cc/java_defs.h", "add_supertype");

    if (type.kind_ == CLASS) {
      supertypes_.push_front(type);  // keep superclass at the front of the list
    } else if (type.kind_ == INTERFACE) {
      supertypes_.push_back(type);
    }
    return *this;
  }

 protected:
  Type(Kind kind, const string& name, const string& package = "")
    : kind_(kind), name_(name), package_(package) {
   std::vector<std::string> mht_29_v;
   mht_29_v.push_back("name: \"" + name + "\"");
   MHTracer_DTPStensorflowPSjavaPSsrcPSgenPSccPSjava_defsDTh mht_29(mht_29_v, 418, "", "./tensorflow/java/src/gen/cc/java_defs.h", "Type");
}

 private:
  Kind kind_;
  string name_;
  string package_;
  std::list<Type> parameters_;
  std::list<Annotation> annotations_;
  std::list<Type> supertypes_;
};

// Definition of a Java annotation
//
// This class only defines the usage of an annotation in a specific context,
// giving optionally a set of attributes to initialize.
class Annotation : public Type {
 public:
  static Annotation Create(const string& type_name, const string& pkg = "") {
   std::vector<std::string> mht_30_v;
   mht_30_v.push_back("type_name: \"" + type_name + "\"");
   MHTracer_DTPStensorflowPSjavaPSsrcPSgenPSccPSjava_defsDTh mht_30(mht_30_v, 439, "", "./tensorflow/java/src/gen/cc/java_defs.h", "Create");

    return Annotation(type_name, pkg);
  }
  const string& attributes() const {
   std::vector<std::string> mht_31_v;
   MHTracer_DTPStensorflowPSjavaPSsrcPSgenPSccPSjava_defsDTh mht_31(mht_31_v, 445, "", "./tensorflow/java/src/gen/cc/java_defs.h", "attributes");
 return attributes_; }
  Annotation& attributes(const string& attributes) {
   std::vector<std::string> mht_32_v;
   mht_32_v.push_back("attributes: \"" + attributes + "\"");
   MHTracer_DTPStensorflowPSjavaPSsrcPSgenPSccPSjava_defsDTh mht_32(mht_32_v, 450, "", "./tensorflow/java/src/gen/cc/java_defs.h", "attributes");

    attributes_ = attributes;
    return *this;
  }

 private:
  string attributes_;

  Annotation(const string& name, const string& package)
    : Type(Kind::ANNOTATION, name, package) {
   std::vector<std::string> mht_33_v;
   mht_33_v.push_back("name: \"" + name + "\"");
   mht_33_v.push_back("package: \"" + package + "\"");
   MHTracer_DTPStensorflowPSjavaPSsrcPSgenPSccPSjava_defsDTh mht_33(mht_33_v, 464, "", "./tensorflow/java/src/gen/cc/java_defs.h", "Annotation");
}
};

// A definition of a Java variable
//
// This class declares an instance of a type, such as a class field or a
// method argument, which can be documented.
class Variable {
 public:
  static Variable Create(const string& name, const Type& type) {
   std::vector<std::string> mht_34_v;
   mht_34_v.push_back("name: \"" + name + "\"");
   MHTracer_DTPStensorflowPSjavaPSsrcPSgenPSccPSjava_defsDTh mht_34(mht_34_v, 477, "", "./tensorflow/java/src/gen/cc/java_defs.h", "Create");

    return Variable(name, type, false);
  }
  static Variable Varargs(const string& name, const Type& type) {
   std::vector<std::string> mht_35_v;
   mht_35_v.push_back("name: \"" + name + "\"");
   MHTracer_DTPStensorflowPSjavaPSsrcPSgenPSccPSjava_defsDTh mht_35(mht_35_v, 484, "", "./tensorflow/java/src/gen/cc/java_defs.h", "Varargs");

    return Variable(name, type, true);
  }
  const string& name() const {
   std::vector<std::string> mht_36_v;
   MHTracer_DTPStensorflowPSjavaPSsrcPSgenPSccPSjava_defsDTh mht_36(mht_36_v, 490, "", "./tensorflow/java/src/gen/cc/java_defs.h", "name");
 return name_; }
  const Type& type() const {
   std::vector<std::string> mht_37_v;
   MHTracer_DTPStensorflowPSjavaPSsrcPSgenPSccPSjava_defsDTh mht_37(mht_37_v, 494, "", "./tensorflow/java/src/gen/cc/java_defs.h", "type");
 return type_; }
  bool variadic() const {
   std::vector<std::string> mht_38_v;
   MHTracer_DTPStensorflowPSjavaPSsrcPSgenPSccPSjava_defsDTh mht_38(mht_38_v, 498, "", "./tensorflow/java/src/gen/cc/java_defs.h", "variadic");
 return variadic_; }

 private:
  string name_;
  Type type_;
  bool variadic_;

  Variable(const string& name, const Type& type, bool variadic)
    : name_(name), type_(type), variadic_(variadic) {
   std::vector<std::string> mht_39_v;
   mht_39_v.push_back("name: \"" + name + "\"");
   MHTracer_DTPStensorflowPSjavaPSsrcPSgenPSccPSjava_defsDTh mht_39(mht_39_v, 510, "", "./tensorflow/java/src/gen/cc/java_defs.h", "Variable");
}
};

// A definition of a Java class method
//
// This class defines the signature of a method, including its name, return
// type and arguments.
class Method {
 public:
  static Method Create(const string& name, const Type& return_type) {
   std::vector<std::string> mht_40_v;
   mht_40_v.push_back("name: \"" + name + "\"");
   MHTracer_DTPStensorflowPSjavaPSsrcPSgenPSccPSjava_defsDTh mht_40(mht_40_v, 523, "", "./tensorflow/java/src/gen/cc/java_defs.h", "Create");

    return Method(name, return_type, false);
  }
  static Method ConstructorFor(const Type& clazz) {
   std::vector<std::string> mht_41_v;
   MHTracer_DTPStensorflowPSjavaPSsrcPSgenPSccPSjava_defsDTh mht_41(mht_41_v, 529, "", "./tensorflow/java/src/gen/cc/java_defs.h", "ConstructorFor");

    return Method(clazz.name(), clazz, true);
  }
  bool constructor() const {
   std::vector<std::string> mht_42_v;
   MHTracer_DTPStensorflowPSjavaPSsrcPSgenPSccPSjava_defsDTh mht_42(mht_42_v, 535, "", "./tensorflow/java/src/gen/cc/java_defs.h", "constructor");
 return constructor_; }
  const string& name() const {
   std::vector<std::string> mht_43_v;
   MHTracer_DTPStensorflowPSjavaPSsrcPSgenPSccPSjava_defsDTh mht_43(mht_43_v, 539, "", "./tensorflow/java/src/gen/cc/java_defs.h", "name");
 return name_; }
  const Type& return_type() const {
   std::vector<std::string> mht_44_v;
   MHTracer_DTPStensorflowPSjavaPSsrcPSgenPSccPSjava_defsDTh mht_44(mht_44_v, 543, "", "./tensorflow/java/src/gen/cc/java_defs.h", "return_type");
 return return_type_; }
  const std::list<Variable>& arguments() const {
   std::vector<std::string> mht_45_v;
   MHTracer_DTPStensorflowPSjavaPSsrcPSgenPSccPSjava_defsDTh mht_45(mht_45_v, 547, "", "./tensorflow/java/src/gen/cc/java_defs.h", "arguments");
 return arguments_; }
  Method& add_argument(const Variable& var) {
   std::vector<std::string> mht_46_v;
   MHTracer_DTPStensorflowPSjavaPSsrcPSgenPSccPSjava_defsDTh mht_46(mht_46_v, 551, "", "./tensorflow/java/src/gen/cc/java_defs.h", "add_argument");

    arguments_.push_back(var);
    return *this;
  }
  const std::list<Annotation>& annotations() const {
   std::vector<std::string> mht_47_v;
   MHTracer_DTPStensorflowPSjavaPSsrcPSgenPSccPSjava_defsDTh mht_47(mht_47_v, 558, "", "./tensorflow/java/src/gen/cc/java_defs.h", "annotations");
 return annotations_; }
  Method& add_annotation(const Annotation& annotation) {
   std::vector<std::string> mht_48_v;
   MHTracer_DTPStensorflowPSjavaPSsrcPSgenPSccPSjava_defsDTh mht_48(mht_48_v, 562, "", "./tensorflow/java/src/gen/cc/java_defs.h", "add_annotation");

    annotations_.push_back(annotation);
    return *this;
  }

 private:
  string name_;
  Type return_type_;
  bool constructor_;
  std::list<Variable> arguments_;
  std::list<Annotation> annotations_;

  Method(const string& name, const Type& return_type, bool constructor)
    : name_(name), return_type_(return_type), constructor_(constructor) {
   std::vector<std::string> mht_49_v;
   mht_49_v.push_back("name: \"" + name + "\"");
   MHTracer_DTPStensorflowPSjavaPSsrcPSgenPSccPSjava_defsDTh mht_49(mht_49_v, 579, "", "./tensorflow/java/src/gen/cc/java_defs.h", "Method");
}
};

// A definition of a documentation bloc for a Java element (JavaDoc)
class Javadoc {
 public:
  static Javadoc Create(const string& brief = "") {
   std::vector<std::string> mht_50_v;
   MHTracer_DTPStensorflowPSjavaPSsrcPSgenPSccPSjava_defsDTh mht_50(mht_50_v, 588, "", "./tensorflow/java/src/gen/cc/java_defs.h", "Create");
 return Javadoc(brief); }
  const string& brief() const {
   std::vector<std::string> mht_51_v;
   MHTracer_DTPStensorflowPSjavaPSsrcPSgenPSccPSjava_defsDTh mht_51(mht_51_v, 592, "", "./tensorflow/java/src/gen/cc/java_defs.h", "brief");
 return brief_; }
  const string& details() const {
   std::vector<std::string> mht_52_v;
   MHTracer_DTPStensorflowPSjavaPSsrcPSgenPSccPSjava_defsDTh mht_52(mht_52_v, 596, "", "./tensorflow/java/src/gen/cc/java_defs.h", "details");
 return details_; }
  Javadoc& details(const string& details) {
   std::vector<std::string> mht_53_v;
   mht_53_v.push_back("details: \"" + details + "\"");
   MHTracer_DTPStensorflowPSjavaPSsrcPSgenPSccPSjava_defsDTh mht_53(mht_53_v, 601, "", "./tensorflow/java/src/gen/cc/java_defs.h", "details");

    details_ = details;
    return *this;
  }
  const std::list<std::pair<string, string>>& tags() const { return tags_; }
  Javadoc& add_tag(const string& tag, const string& text) {
   std::vector<std::string> mht_54_v;
   mht_54_v.push_back("tag: \"" + tag + "\"");
   mht_54_v.push_back("text: \"" + text + "\"");
   MHTracer_DTPStensorflowPSjavaPSsrcPSgenPSccPSjava_defsDTh mht_54(mht_54_v, 611, "", "./tensorflow/java/src/gen/cc/java_defs.h", "add_tag");

    tags_.push_back(std::make_pair(tag, text));
    return *this;
  }
  Javadoc& add_param_tag(const string& name, const string& text) {
   std::vector<std::string> mht_55_v;
   mht_55_v.push_back("name: \"" + name + "\"");
   mht_55_v.push_back("text: \"" + text + "\"");
   MHTracer_DTPStensorflowPSjavaPSsrcPSgenPSccPSjava_defsDTh mht_55(mht_55_v, 620, "", "./tensorflow/java/src/gen/cc/java_defs.h", "add_param_tag");

    return add_tag("param", name + " " + text);
  }

 private:
  string brief_;
  string details_;
  std::list<std::pair<string, string>> tags_;

  explicit Javadoc(const string& brief) : brief_(brief) {
   std::vector<std::string> mht_56_v;
   mht_56_v.push_back("brief: \"" + brief + "\"");
   MHTracer_DTPStensorflowPSjavaPSsrcPSgenPSccPSjava_defsDTh mht_56(mht_56_v, 633, "", "./tensorflow/java/src/gen/cc/java_defs.h", "Javadoc");
}
};

}  // namespace java
}  // namespace tensorflow

#endif  // TENSORFLOW_JAVA_SRC_GEN_CC_JAVA_DEFS_H_
