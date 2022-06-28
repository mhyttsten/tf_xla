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
class MHTracer_DTPStensorflowPScompilerPSaotPSembedded_protocol_buffersDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSaotPSembedded_protocol_buffersDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSaotPSembedded_protocol_buffersDTcc() {
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

#include "tensorflow/compiler/aot/embedded_protocol_buffers.h"

#include <memory>
#include <string>

#include "absl/memory/memory.h"
#include "absl/strings/str_replace.h"
#include "llvm/ADT/Triple.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Module.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetOptions.h"
#include "tensorflow/compiler/xla/service/llvm_ir/llvm_type_conversion_util.h"
#include "tensorflow/compiler/xla/util.h"

namespace tensorflow {
namespace tfcompile {

using xla::llvm_ir::AsStringRef;

static void AddEmbeddedProtocolBufferToLlvmModule(
    llvm::Module* module, const ::tensorflow::protobuf::MessageLite& proto,
    absl::string_view unique_identifier, string* protobuf_array_symbol_name,
    int64_t* protobuf_array_size) {
   std::vector<std::string> mht_0_v;
   mht_0_v.push_back("unique_identifier: \"" + std::string(unique_identifier.data(), unique_identifier.size()) + "\"");
   MHTracer_DTPStensorflowPScompilerPSaotPSembedded_protocol_buffersDTcc mht_0(mht_0_v, 213, "", "./tensorflow/compiler/aot/embedded_protocol_buffers.cc", "AddEmbeddedProtocolBufferToLlvmModule");

  string protobuf_array_contents = proto.SerializeAsString();
  *protobuf_array_symbol_name =
      absl::StrCat(unique_identifier, "_protobuf_array_contents");
  *protobuf_array_size = protobuf_array_contents.size();

  llvm::Constant* protobuf_array_initializer =
      llvm::ConstantDataArray::getString(module->getContext(),
                                         AsStringRef(protobuf_array_contents),
                                         /*AddNull=*/false);
  new llvm::GlobalVariable(
      *module, protobuf_array_initializer->getType(),
      /*isConstant=*/true, llvm::GlobalValue::ExternalLinkage,
      protobuf_array_initializer, AsStringRef(*protobuf_array_symbol_name));
}

static string CreateCPPShimExpression(
    absl::string_view qualified_cpp_protobuf_name,
    absl::string_view protobuf_array_symbol_name, int64_t protobuf_array_size) {
   std::vector<std::string> mht_1_v;
   mht_1_v.push_back("qualified_cpp_protobuf_name: \"" + std::string(qualified_cpp_protobuf_name.data(), qualified_cpp_protobuf_name.size()) + "\"");
   mht_1_v.push_back("protobuf_array_symbol_name: \"" + std::string(protobuf_array_symbol_name.data(), protobuf_array_symbol_name.size()) + "\"");
   MHTracer_DTPStensorflowPScompilerPSaotPSembedded_protocol_buffersDTcc mht_1(mht_1_v, 236, "", "./tensorflow/compiler/aot/embedded_protocol_buffers.cc", "CreateCPPShimExpression");

  string code =
      "[]() {\n"
      "    {{PROTOBUF_NAME}}* proto = new {{PROTOBUF_NAME}};\n"
      "    proto->ParseFromArray(&{{ARRAY_SYMBOL}}[0], {{ARRAY_SIZE}});\n"
      "    return proto;\n"
      "  }()";

  return absl::StrReplaceAll(
      code,
      {
          {"{{ARRAY_SYMBOL}}", absl::StrCat(protobuf_array_symbol_name)},
          {"{{ARRAY_SIZE}}", absl::StrCat(protobuf_array_size)},
          {"{{PROTOBUF_NAME}}", absl::StrCat(qualified_cpp_protobuf_name)},
      });
}

static StatusOr<string> CodegenModule(llvm::TargetMachine* target_machine,
                                      std::unique_ptr<llvm::Module> module) {
  llvm::SmallVector<char, 0> stream_buffer;
  llvm::raw_svector_ostream ostream(stream_buffer);
  llvm::legacy::PassManager codegen_passes;

  if (target_machine->addPassesToEmitFile(codegen_passes, ostream, nullptr,
                                          llvm::CGFT_ObjectFile)) {
    return xla::InternalError(
        "Could not create pass pipeline to generate object file");
  }

  codegen_passes.run(*module);

  return string(stream_buffer.begin(), stream_buffer.end());
}

static StatusOr<std::unique_ptr<llvm::TargetMachine>>
GetTargetMachineFromTriple(absl::string_view target_triple) {
  std::string error;
  std::string normalized_triple =
      llvm::Triple::normalize(AsStringRef(absl::string_view(target_triple)));
  const llvm::Target* target =
      llvm::TargetRegistry::lookupTarget(normalized_triple, error);
  if (target == nullptr) {
    return xla::InternalError("TargetRegistry::lookupTarget failed: %s",
                              error.c_str());
  }

  return absl::WrapUnique(target->createTargetMachine(
      normalized_triple, /*CPU=*/"",
      /*Features=*/"", llvm::TargetOptions(), llvm::None));
}

StatusOr<EmbeddedProtocolBuffers> CreateEmbeddedProtocolBuffers(
    absl::string_view target_triple,
    absl::Span<const ProtobufToEmbed> protobufs_to_embed) {
  TF_ASSIGN_OR_RETURN(std::unique_ptr<llvm::TargetMachine> target_machine,
                      GetTargetMachineFromTriple(target_triple));

  llvm::LLVMContext llvm_context;
  std::unique_ptr<llvm::Module> module_with_serialized_proto =
      absl::make_unique<llvm::Module>("embedded_data_module", llvm_context);

  EmbeddedProtocolBuffers result;

  for (const ProtobufToEmbed& protobuf_to_embed : protobufs_to_embed) {
    string cpp_shim, cpp_variable_decl;
    if (protobuf_to_embed.message) {
      string protobuf_array_symbol_name;
      int64_t protobuf_array_size;

      AddEmbeddedProtocolBufferToLlvmModule(
          module_with_serialized_proto.get(), *protobuf_to_embed.message,
          protobuf_to_embed.symbol_prefix, &protobuf_array_symbol_name,
          &protobuf_array_size);
      cpp_shim = CreateCPPShimExpression(
          protobuf_to_embed.qualified_cpp_protobuf_name,
          protobuf_array_symbol_name, protobuf_array_size);

      cpp_variable_decl =
          absl::StrCat("extern \"C\" char ", protobuf_array_symbol_name, "[];");
    } else {
      cpp_shim = "nullptr";
    }
    result.cpp_shims.push_back({cpp_shim, cpp_variable_decl});
  }

  TF_ASSIGN_OR_RETURN(result.object_file_data,
                      CodegenModule(target_machine.get(),
                                    std::move(module_with_serialized_proto)));
  return result;
}

}  // namespace tfcompile
}  // namespace tensorflow
