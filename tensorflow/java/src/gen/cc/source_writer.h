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

#ifndef TENSORFLOW_JAVA_SRC_GEN_CC_SOURCE_WRITER_H_
#define TENSORFLOW_JAVA_SRC_GEN_CC_SOURCE_WRITER_H_
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
class MHTracer_DTPStensorflowPSjavaPSsrcPSgenPSccPSsource_writerDTh {
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
   MHTracer_DTPStensorflowPSjavaPSsrcPSgenPSccPSsource_writerDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSjavaPSsrcPSgenPSccPSsource_writerDTh() {
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
#include <stack>
#include <list>
#include <set>

#include "tensorflow/core/lib/core/stringpiece.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/java/src/gen/cc/java_defs.h"

namespace tensorflow {
namespace java {

// A class for writing Java source code.
class SourceWriter {
 public:
  SourceWriter();

  virtual ~SourceWriter();

  // Indents following lines with white spaces.
  //
  // Indentation is cumulative, i.e. the provided tabulation is added to the
  // current indentation value. If the tabulation is negative, the operation
  // will outdent the source code, until the indentation reaches 0 again.
  //
  // For example, calling Indent(2) twice will indent code with 4 white
  // spaces. Then calling Indent(-2) will outdent the code back to 2 white
  // spaces.
  SourceWriter& Indent(int tab);

  // Prefixes following lines with provided character(s).
  //
  // A common use case of a prefix is for commenting or documenting the code.
  //
  // The prefix is written after the indentation, For example, invoking
  // Indent(2)->Prefix("//") will result in prefixing lines with "  //".
  //
  // An empty value ("") will remove any line prefix that was previously set.
  SourceWriter& Prefix(const char* line_prefix);

  // Writes a source code snippet.
  //
  // The data might potentially contain newline characters, therefore it will
  // be scanned to ensure that each line is indented and prefixed properly,
  // making it a bit slower than Append().
  SourceWriter& Write(const StringPiece& str);

  // Writes a source code snippet read from a file.
  //
  // All lines of the file at the provided path will be read and written back
  // to the output of this writer in regard of its current attributes (e.g.
  // the indentation, prefix, etc.)
  SourceWriter& WriteFromFile(const string& fname, Env* env = Env::Default());

  // Appends a piece of source code.
  //
  // It is expected that no newline character is present in the data provided,
  // otherwise Write() must be used.
  SourceWriter& Append(const StringPiece& str);

  // Appends a type to the current line.
  //
  // The type is written in its simple form (i.e. not prefixed by its package)
  // and followed by any parameter types it has enclosed in brackets (<>).
  SourceWriter& AppendType(const Type& type);

  // Appends a newline character.
  //
  // Data written after calling this method will start on a new line, in respect
  // of the current indentation.
  SourceWriter& EndLine();

  // Begins a block of source code.
  //
  // This method appends a new opening brace to the current data and indent the
  // next lines according to Google Java Style Guide. The block can optionally
  // be preceded by an expression (e.g. Append("if(true)").BeginBlock();)
  SourceWriter& BeginBlock(const string& expression = "");

  // Ends the current block of source code.
  //
  // This method appends a new closing brace to the current data and outdent the
  // next lines back to the margin used before BeginBlock() was invoked.
  SourceWriter& EndBlock();

  // Begins to write a method.
  //
  // This method outputs the signature of the Java method from the data passed
  // in the 'method' parameter and starts a new block. Modifiers are also passed
  // in parameter to define the access scope of this method and, optionally,
  // a Javadoc.
  SourceWriter& BeginMethod(const Method& method, int modifiers,
                            const Javadoc* javadoc = nullptr);

  // Ends the current method.
  //
  // This method ends the block of code that has begun when invoking
  // BeginMethod() prior to this.
  SourceWriter& EndMethod();

  // Begins to write the main type of a source file.
  //
  // This method outputs the declaration of the Java type from the data passed
  // in the 'type' parameter and starts a new block. Modifiers are also passed
  // in parameter to define the access scope of this type and, optionally,
  // a Javadoc.
  //
  // If not null, all types found in the 'extra_dependencies' list will be
  // imported before declaring the new type.
  SourceWriter& BeginType(const Type& type, int modifiers,
                          const std::list<Type>* extra_dependencies = nullptr,
                          const Javadoc* javadoc = nullptr);

  // Begins to write a new inner type.
  //
  // This method outputs the declaration of the Java type from the data passed
  // in the 'type' parameter and starts a new block. Modifiers are also passed
  // in parameter to define the accesses and the scope of this type and,
  // optionally, a Javadoc.
  SourceWriter& BeginInnerType(const Type& type, int modifiers,
                               const Javadoc* javadoc = nullptr);

  // Ends the current type.
  //
  // This method ends the block of code that has begun when invoking
  // BeginType() or BeginInnerType() prior to this.
  SourceWriter& EndType();

  // Writes a variable as fields of a type.
  //
  // This method must be called within the definition of a type (see BeginType()
  // or BeginInnerType()). Modifiers are also be passed in parameter to define
  // the accesses and the scope of this field and, optionally, a Javadoc.
  SourceWriter& WriteField(const Variable& field, int modifiers,
                           const Javadoc* javadoc = nullptr);

 protected:
  virtual void DoAppend(const StringPiece& str) = 0;

 private:
  // A utility base class for visiting elements of a type.
  class TypeVisitor {
   public:
    virtual ~TypeVisitor() = default;
    void Visit(const Type& type);

   protected:
    virtual void DoVisit(const Type& type) = 0;
  };

  // A utility class for keeping track of declared generics in a given scope.
  class GenericNamespace : public TypeVisitor {
   public:
    GenericNamespace() = default;
    explicit GenericNamespace(const GenericNamespace* parent)
      : generic_names_(parent->generic_names_) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSjavaPSsrcPSgenPSccPSsource_writerDTh mht_0(mht_0_v, 343, "", "./tensorflow/java/src/gen/cc/source_writer.h", "GenericNamespace");
}
    std::list<const Type*> declared_types() {
      return declared_types_;
    }
   protected:
    virtual void DoVisit(const Type& type);

   private:
    std::list<const Type*> declared_types_;
    std::set<string> generic_names_;
  };

  // A utility class for collecting a list of import statements to declare.
  class TypeImporter : public TypeVisitor {
   public:
    explicit TypeImporter(const string& current_package)
      : current_package_(current_package) {
   std::vector<std::string> mht_1_v;
   mht_1_v.push_back("current_package: \"" + current_package + "\"");
   MHTracer_DTPStensorflowPSjavaPSsrcPSgenPSccPSsource_writerDTh mht_1(mht_1_v, 363, "", "./tensorflow/java/src/gen/cc/source_writer.h", "TypeImporter");
}
    virtual ~TypeImporter() = default;
    const std::set<string> imports() {
      return imports_;
    }
   protected:
    virtual void DoVisit(const Type& type);

   private:
    string current_package_;
    std::set<string> imports_;
  };

  string left_margin_;
  string line_prefix_;
  bool newline_ = true;
  std::stack<GenericNamespace*> generic_namespaces_;

  SourceWriter& WriteModifiers(int modifiers);
  SourceWriter& WriteJavadoc(const Javadoc& javadoc);
  SourceWriter& WriteAnnotations(const std::list<Annotation>& annotations);
  SourceWriter& WriteGenerics(const std::list<const Type*>& generics);
  GenericNamespace* PushGenericNamespace(int modifiers);
  void PopGenericNamespace();
};

// A writer that outputs source code into a file.
//
// Note: the writer does not acquire the ownership of the file being passed in
// parameter.
class SourceFileWriter : public SourceWriter {
 public:
  explicit SourceFileWriter(WritableFile* file) : file_(file) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSjavaPSsrcPSgenPSccPSsource_writerDTh mht_2(mht_2_v, 398, "", "./tensorflow/java/src/gen/cc/source_writer.h", "SourceFileWriter");
}
  virtual ~SourceFileWriter() = default;

 protected:
  void DoAppend(const StringPiece& str) override {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPSjavaPSsrcPSgenPSccPSsource_writerDTh mht_3(mht_3_v, 405, "", "./tensorflow/java/src/gen/cc/source_writer.h", "DoAppend");

    TF_CHECK_OK(file_->Append(str));
  }

 private:
  WritableFile* file_;
};

// A writer that outputs source code into a string buffer.
class SourceBufferWriter : public SourceWriter {
 public:
  SourceBufferWriter() : owns_buffer_(true), buffer_(new string()) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPSjavaPSsrcPSgenPSccPSsource_writerDTh mht_4(mht_4_v, 419, "", "./tensorflow/java/src/gen/cc/source_writer.h", "SourceBufferWriter");
}
  explicit SourceBufferWriter(string* buffer)
      : owns_buffer_(false), buffer_(buffer) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPSjavaPSsrcPSgenPSccPSsource_writerDTh mht_5(mht_5_v, 424, "", "./tensorflow/java/src/gen/cc/source_writer.h", "SourceBufferWriter");
}
  virtual ~SourceBufferWriter() {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPSjavaPSsrcPSgenPSccPSsource_writerDTh mht_6(mht_6_v, 428, "", "./tensorflow/java/src/gen/cc/source_writer.h", "~SourceBufferWriter");

    if (owns_buffer_) delete buffer_;
  }
  const string& str() {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPSjavaPSsrcPSgenPSccPSsource_writerDTh mht_7(mht_7_v, 434, "", "./tensorflow/java/src/gen/cc/source_writer.h", "str");
 return *buffer_; }

 protected:
  void DoAppend(const StringPiece& str) override {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPSjavaPSsrcPSgenPSccPSsource_writerDTh mht_8(mht_8_v, 440, "", "./tensorflow/java/src/gen/cc/source_writer.h", "DoAppend");

    buffer_->append(str.begin(), str.end());
  }

 private:
  bool owns_buffer_;
  string* buffer_;
};

}  // namespace java
}  // namespace tensorflow

#endif  // TENSORFLOW_JAVA_SRC_GEN_CC_SOURCE_WRITER_H_
