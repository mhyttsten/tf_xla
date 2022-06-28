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
class MHTracer_DTPStensorflowPSjavaPSsrcPSgenPSccPSsource_writerDTcc {
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
   MHTracer_DTPStensorflowPSjavaPSsrcPSgenPSccPSsource_writerDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSjavaPSsrcPSgenPSccPSsource_writerDTcc() {
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

#include <string>
#include <algorithm>
#include <list>

#include "tensorflow/java/src/gen/cc/source_writer.h"

namespace tensorflow {
namespace java {

SourceWriter::SourceWriter() {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSjavaPSsrcPSgenPSccPSsource_writerDTcc mht_0(mht_0_v, 194, "", "./tensorflow/java/src/gen/cc/source_writer.cc", "SourceWriter::SourceWriter");

  // Push an empty generic namespace at start, for simplification.
  generic_namespaces_.push(new GenericNamespace());
}

SourceWriter::~SourceWriter() {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSjavaPSsrcPSgenPSccPSsource_writerDTcc mht_1(mht_1_v, 202, "", "./tensorflow/java/src/gen/cc/source_writer.cc", "SourceWriter::~SourceWriter");

  // Remove empty generic namespace added at start as well as any other
  // namespace objects that haven't been removed.
  while (!generic_namespaces_.empty()) {
    GenericNamespace* generic_namespace = generic_namespaces_.top();
    generic_namespaces_.pop();
    delete generic_namespace;
  }
}

SourceWriter& SourceWriter::Indent(int tab) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSjavaPSsrcPSgenPSccPSsource_writerDTcc mht_2(mht_2_v, 215, "", "./tensorflow/java/src/gen/cc/source_writer.cc", "SourceWriter::Indent");

  left_margin_.resize(
      std::max(static_cast<int>(left_margin_.size() + tab), 0), ' ');
  return *this;
}

SourceWriter& SourceWriter::Prefix(const char* line_prefix) {
   std::vector<std::string> mht_3_v;
   mht_3_v.push_back("line_prefix: \"" + (line_prefix == nullptr ? std::string("nullptr") : std::string((char*)line_prefix)) + "\"");
   MHTracer_DTPStensorflowPSjavaPSsrcPSgenPSccPSsource_writerDTcc mht_3(mht_3_v, 225, "", "./tensorflow/java/src/gen/cc/source_writer.cc", "SourceWriter::Prefix");

  line_prefix_ = line_prefix;
  return *this;
}

SourceWriter& SourceWriter::Write(const StringPiece& str) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPSjavaPSsrcPSgenPSccPSsource_writerDTcc mht_4(mht_4_v, 233, "", "./tensorflow/java/src/gen/cc/source_writer.cc", "SourceWriter::Write");

  size_t line_pos = 0;
  do {
    size_t start_pos = line_pos;
    line_pos = str.find('\n', start_pos);
    if (line_pos != string::npos) {
      ++line_pos;
      Append(str.substr(start_pos, line_pos - start_pos));
      newline_ = true;
    } else {
      Append(str.substr(start_pos, str.size() - start_pos));
    }
  } while (line_pos != string::npos && line_pos < str.size());

  return *this;
}

SourceWriter& SourceWriter::WriteFromFile(const string& fname, Env* env) {
   std::vector<std::string> mht_5_v;
   mht_5_v.push_back("fname: \"" + fname + "\"");
   MHTracer_DTPStensorflowPSjavaPSsrcPSgenPSccPSsource_writerDTcc mht_5(mht_5_v, 254, "", "./tensorflow/java/src/gen/cc/source_writer.cc", "SourceWriter::WriteFromFile");

  string data_;
  TF_CHECK_OK(ReadFileToString(env, fname, &data_));
  return Write(data_);
}

SourceWriter& SourceWriter::Append(const StringPiece& str) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPSjavaPSsrcPSgenPSccPSsource_writerDTcc mht_6(mht_6_v, 263, "", "./tensorflow/java/src/gen/cc/source_writer.cc", "SourceWriter::Append");

  if (!str.empty()) {
    if (newline_) {
      DoAppend(left_margin_ + line_prefix_);
      newline_ = false;
    }
    DoAppend(str);
  }
  return *this;
}

SourceWriter& SourceWriter::AppendType(const Type& type) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPSjavaPSsrcPSgenPSccPSsource_writerDTcc mht_7(mht_7_v, 277, "", "./tensorflow/java/src/gen/cc/source_writer.cc", "SourceWriter::AppendType");

  if (type.wildcard()) {
    Append("?");
  } else {
    Append(type.name());
    if (!type.parameters().empty()) {
      Append("<");
      bool first = true;
      for (const Type& t : type.parameters()) {
        if (!first) {
          Append(", ");
        }
        AppendType(t);
        first = false;
      }
      Append(">");
    }
  }
  return *this;
}

SourceWriter& SourceWriter::EndLine() {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPSjavaPSsrcPSgenPSccPSsource_writerDTcc mht_8(mht_8_v, 301, "", "./tensorflow/java/src/gen/cc/source_writer.cc", "SourceWriter::EndLine");

  Append("\n");
  newline_ = true;
  return *this;
}

SourceWriter& SourceWriter::BeginBlock(const string& expression) {
   std::vector<std::string> mht_9_v;
   mht_9_v.push_back("expression: \"" + expression + "\"");
   MHTracer_DTPStensorflowPSjavaPSsrcPSgenPSccPSsource_writerDTcc mht_9(mht_9_v, 311, "", "./tensorflow/java/src/gen/cc/source_writer.cc", "SourceWriter::BeginBlock");

  if (!expression.empty()) {
    Append(expression + " {");
  } else {
    Append(newline_ ? "{" : " {");
  }
  return EndLine().Indent(2);
}

SourceWriter& SourceWriter::EndBlock() {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPSjavaPSsrcPSgenPSccPSsource_writerDTcc mht_10(mht_10_v, 323, "", "./tensorflow/java/src/gen/cc/source_writer.cc", "SourceWriter::EndBlock");

  return Indent(-2).Append("}").EndLine();
}

SourceWriter& SourceWriter::BeginMethod(const Method& method, int modifiers,
                                        const Javadoc* javadoc) {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPSjavaPSsrcPSgenPSccPSsource_writerDTcc mht_11(mht_11_v, 331, "", "./tensorflow/java/src/gen/cc/source_writer.cc", "SourceWriter::BeginMethod");

  GenericNamespace* generic_namespace = PushGenericNamespace(modifiers);
  if (!method.constructor()) {
    generic_namespace->Visit(method.return_type());
  }
  for (const Variable& v : method.arguments()) {
    generic_namespace->Visit(v.type());
  }
  EndLine();
  if (javadoc != nullptr) {
    WriteJavadoc(*javadoc);
  }
  if (!method.annotations().empty()) {
    WriteAnnotations(method.annotations());
  }
  WriteModifiers(modifiers);
  if (!generic_namespace->declared_types().empty()) {
    WriteGenerics(generic_namespace->declared_types());
    Append(" ");
  }
  if (!method.constructor()) {
    AppendType(method.return_type()).Append(" ");
  }
  Append(method.name()).Append("(");
  bool first = true;
  for (const Variable& v : method.arguments()) {
    if (!first) {
      Append(", ");
    }
    AppendType(v.type()).Append(v.variadic() ? "... " : " ").Append(v.name());
    first = false;
  }
  return Append(")").BeginBlock();
}

SourceWriter& SourceWriter::EndMethod() {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPSjavaPSsrcPSgenPSccPSsource_writerDTcc mht_12(mht_12_v, 369, "", "./tensorflow/java/src/gen/cc/source_writer.cc", "SourceWriter::EndMethod");

  EndBlock();
  PopGenericNamespace();
  return *this;
}

SourceWriter& SourceWriter::BeginType(const Type& type, int modifiers,
                                      const std::list<Type>* extra_dependencies,
                                      const Javadoc* javadoc) {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPSjavaPSsrcPSgenPSccPSsource_writerDTcc mht_13(mht_13_v, 380, "", "./tensorflow/java/src/gen/cc/source_writer.cc", "SourceWriter::BeginType");

  if (!type.package().empty()) {
    Append("package ").Append(type.package()).Append(";").EndLine();
  }
  TypeImporter type_importer(type.package());
  type_importer.Visit(type);
  if (extra_dependencies != nullptr) {
    for (const Type& t : *extra_dependencies) {
      type_importer.Visit(t);
    }
  }
  if (!type_importer.imports().empty()) {
    EndLine();
    for (const string& s : type_importer.imports()) {
      Append("import ").Append(s).Append(";").EndLine();
    }
  }
  return BeginInnerType(type, modifiers, javadoc);
}

SourceWriter& SourceWriter::BeginInnerType(const Type& type, int modifiers,
                                           const Javadoc* javadoc) {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPSjavaPSsrcPSgenPSccPSsource_writerDTcc mht_14(mht_14_v, 404, "", "./tensorflow/java/src/gen/cc/source_writer.cc", "SourceWriter::BeginInnerType");

  GenericNamespace* generic_namespace = PushGenericNamespace(modifiers);
  generic_namespace->Visit(type);
  EndLine();
  if (javadoc != nullptr) {
    WriteJavadoc(*javadoc);
  }
  if (!type.annotations().empty()) {
    WriteAnnotations(type.annotations());
  }
  WriteModifiers(modifiers);
  CHECK_EQ(Type::Kind::CLASS, type.kind()) << ": Not supported yet";
  Append("class ").Append(type.name());
  if (!generic_namespace->declared_types().empty()) {
    WriteGenerics(generic_namespace->declared_types());
  }
  if (!type.supertypes().empty()) {
    bool first_interface = true;
    for (const Type& t : type.supertypes()) {
      if (t.kind() == Type::CLASS) {  // superclass is always first in list
        Append(" extends ");
      } else if (first_interface) {
        Append(" implements ");
        first_interface = false;
      } else {
        Append(", ");
      }
      AppendType(t);
    }
  }
  return BeginBlock();
}

SourceWriter& SourceWriter::EndType() {
   std::vector<std::string> mht_15_v;
   MHTracer_DTPStensorflowPSjavaPSsrcPSgenPSccPSsource_writerDTcc mht_15(mht_15_v, 440, "", "./tensorflow/java/src/gen/cc/source_writer.cc", "SourceWriter::EndType");

  EndBlock();
  PopGenericNamespace();
  return *this;
}

SourceWriter& SourceWriter::WriteField(const Variable& field, int modifiers,
                                       const Javadoc* javadoc) {
   std::vector<std::string> mht_16_v;
   MHTracer_DTPStensorflowPSjavaPSsrcPSgenPSccPSsource_writerDTcc mht_16(mht_16_v, 450, "", "./tensorflow/java/src/gen/cc/source_writer.cc", "SourceWriter::WriteField");

  // If present, write field javadoc only as one brief line
  if (javadoc != nullptr && !javadoc->brief().empty()) {
    Append("/** ").Append(javadoc->brief()).Append(" */").EndLine();
  }
  WriteModifiers(modifiers);
  AppendType(field.type()).Append(" ").Append(field.name()).Append(";");
  EndLine();
  return *this;
}

SourceWriter& SourceWriter::WriteModifiers(int modifiers) {
   std::vector<std::string> mht_17_v;
   MHTracer_DTPStensorflowPSjavaPSsrcPSgenPSccPSsource_writerDTcc mht_17(mht_17_v, 464, "", "./tensorflow/java/src/gen/cc/source_writer.cc", "SourceWriter::WriteModifiers");

  if (modifiers & PUBLIC) {
    Append("public ");
  } else if (modifiers & PROTECTED) {
    Append("protected ");
  } else if (modifiers & PRIVATE) {
    Append("private ");
  }
  if (modifiers & STATIC) {
    Append("static ");
  }
  if (modifiers & FINAL) {
    Append("final ");
  }
  return *this;
}

SourceWriter& SourceWriter::WriteJavadoc(const Javadoc& javadoc) {
   std::vector<std::string> mht_18_v;
   MHTracer_DTPStensorflowPSjavaPSsrcPSgenPSccPSsource_writerDTcc mht_18(mht_18_v, 484, "", "./tensorflow/java/src/gen/cc/source_writer.cc", "SourceWriter::WriteJavadoc");

  Append("/**").Prefix(" * ").EndLine();
  bool do_line_break = false;
  if (!javadoc.brief().empty()) {
    Write(javadoc.brief()).EndLine();
    do_line_break = true;
  }
  if (!javadoc.details().empty()) {
    if (do_line_break) {
      Append("<p>").EndLine();
    }
    Write(javadoc.details()).EndLine();
    do_line_break = true;
  }
  if (!javadoc.tags().empty()) {
    if (do_line_break) {
      EndLine();
    }
    for (const auto& p : javadoc.tags()) {
      Append("@" + p.first);
      if (!p.second.empty()) {
        Append(" ").Write(p.second);
      }
      EndLine();
    }
  }
  return Prefix("").Append(" */").EndLine();
}

SourceWriter& SourceWriter::WriteAnnotations(
    const std::list<Annotation>& annotations) {
   std::vector<std::string> mht_19_v;
   MHTracer_DTPStensorflowPSjavaPSsrcPSgenPSccPSsource_writerDTcc mht_19(mht_19_v, 517, "", "./tensorflow/java/src/gen/cc/source_writer.cc", "SourceWriter::WriteAnnotations");

  for (const Annotation& a : annotations) {
    Append("@" + a.name());
    if (!a.attributes().empty()) {
      Append("(").Append(a.attributes()).Append(")");
    }
    EndLine();
  }
  return *this;
}

SourceWriter& SourceWriter::WriteGenerics(
    const std::list<const Type*>& generics) {
   std::vector<std::string> mht_20_v;
   MHTracer_DTPStensorflowPSjavaPSsrcPSgenPSccPSsource_writerDTcc mht_20(mht_20_v, 532, "", "./tensorflow/java/src/gen/cc/source_writer.cc", "SourceWriter::WriteGenerics");

  Append("<");
  bool first = true;
  for (const Type* pt : generics) {
    if (!first) {
      Append(", ");
    }
    Append(pt->name());
    if (!pt->supertypes().empty()) {
      Append(" extends ").AppendType(pt->supertypes().front());
    }
    first = false;
  }
  return Append(">");
}

SourceWriter::GenericNamespace* SourceWriter::PushGenericNamespace(
    int modifiers) {
   std::vector<std::string> mht_21_v;
   MHTracer_DTPStensorflowPSjavaPSsrcPSgenPSccPSsource_writerDTcc mht_21(mht_21_v, 552, "", "./tensorflow/java/src/gen/cc/source_writer.cc", "SourceWriter::PushGenericNamespace");

  GenericNamespace* generic_namespace;
  if (modifiers & STATIC) {
    generic_namespace = new GenericNamespace();
  } else {
    generic_namespace = new GenericNamespace(generic_namespaces_.top());
  }
  generic_namespaces_.push(generic_namespace);
  return generic_namespace;
}

void SourceWriter::PopGenericNamespace() {
   std::vector<std::string> mht_22_v;
   MHTracer_DTPStensorflowPSjavaPSsrcPSgenPSccPSsource_writerDTcc mht_22(mht_22_v, 566, "", "./tensorflow/java/src/gen/cc/source_writer.cc", "SourceWriter::PopGenericNamespace");

  GenericNamespace* generic_namespace = generic_namespaces_.top();
  generic_namespaces_.pop();
  delete generic_namespace;
}

void SourceWriter::TypeVisitor::Visit(const Type& type) {
   std::vector<std::string> mht_23_v;
   MHTracer_DTPStensorflowPSjavaPSsrcPSgenPSccPSsource_writerDTcc mht_23(mht_23_v, 575, "", "./tensorflow/java/src/gen/cc/source_writer.cc", "SourceWriter::TypeVisitor::Visit");

  DoVisit(type);
  for (const Type& t : type.parameters()) {
    Visit(t);
  }
  for (const Annotation& t : type.annotations()) {
    DoVisit(t);
  }
  for (const Type& t : type.supertypes()) {
    Visit(t);
  }
}

void SourceWriter::GenericNamespace::DoVisit(const Type& type) {
   std::vector<std::string> mht_24_v;
   MHTracer_DTPStensorflowPSjavaPSsrcPSgenPSccPSsource_writerDTcc mht_24(mht_24_v, 591, "", "./tensorflow/java/src/gen/cc/source_writer.cc", "SourceWriter::GenericNamespace::DoVisit");

  // ignore non-generic parameters, wildcards and generics already declared
  if (type.kind() == Type::GENERIC && !type.wildcard() &&
      generic_names_.find(type.name()) == generic_names_.end()) {
    declared_types_.push_back(&type);
    generic_names_.insert(type.name());
  }
}

void SourceWriter::TypeImporter::DoVisit(const Type& type) {
   std::vector<std::string> mht_25_v;
   MHTracer_DTPStensorflowPSjavaPSsrcPSgenPSccPSsource_writerDTcc mht_25(mht_25_v, 603, "", "./tensorflow/java/src/gen/cc/source_writer.cc", "SourceWriter::TypeImporter::DoVisit");

  if (!type.package().empty() && type.package() != current_package_) {
    imports_.insert(type.canonical_name());
  }
}

}  // namespace java
}  // namespace tensorflow
