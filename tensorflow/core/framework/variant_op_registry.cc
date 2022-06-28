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
class MHTracer_DTPStensorflowPScorePSframeworkPSvariant_op_registryDTcc {
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
   MHTracer_DTPStensorflowPScorePSframeworkPSvariant_op_registryDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSframeworkPSvariant_op_registryDTcc() {
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

#include "tensorflow/core/framework/variant_op_registry.h"

#include <string>

#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/type_index.h"
#include "tensorflow/core/framework/variant.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/public/version.h"

namespace tensorflow {

const char* VariantUnaryOpToString(VariantUnaryOp op) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSvariant_op_registryDTcc mht_0(mht_0_v, 198, "", "./tensorflow/core/framework/variant_op_registry.cc", "VariantUnaryOpToString");

  switch (op) {
    case INVALID_VARIANT_UNARY_OP:
      return "INVALID";
    case ZEROS_LIKE_VARIANT_UNARY_OP:
      return "ZEROS_LIKE";
    case CONJ_VARIANT_UNARY_OP:
      return "CONJ";
  }
}

const char* VariantBinaryOpToString(VariantBinaryOp op) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSvariant_op_registryDTcc mht_1(mht_1_v, 212, "", "./tensorflow/core/framework/variant_op_registry.cc", "VariantBinaryOpToString");

  switch (op) {
    case INVALID_VARIANT_BINARY_OP:
      return "INVALID";
    case ADD_VARIANT_BINARY_OP:
      return "ADD";
  }
}

std::unordered_set<string>* UnaryVariantOpRegistry::PersistentStringStorage() {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSvariant_op_registryDTcc mht_2(mht_2_v, 224, "", "./tensorflow/core/framework/variant_op_registry.cc", "UnaryVariantOpRegistry::PersistentStringStorage");

  static std::unordered_set<string>* string_storage =
      new std::unordered_set<string>();
  return string_storage;
}

// Get a pointer to a global UnaryVariantOpRegistry object
UnaryVariantOpRegistry* UnaryVariantOpRegistryGlobal() {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSvariant_op_registryDTcc mht_3(mht_3_v, 234, "", "./tensorflow/core/framework/variant_op_registry.cc", "UnaryVariantOpRegistryGlobal");

  static UnaryVariantOpRegistry* global_unary_variant_op_registry = nullptr;

  if (global_unary_variant_op_registry == nullptr) {
    global_unary_variant_op_registry = new UnaryVariantOpRegistry;
  }
  return global_unary_variant_op_registry;
}

UnaryVariantOpRegistry::VariantDecodeFn* UnaryVariantOpRegistry::GetDecodeFn(
    StringPiece type_name) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSvariant_op_registryDTcc mht_4(mht_4_v, 247, "", "./tensorflow/core/framework/variant_op_registry.cc", "UnaryVariantOpRegistry::GetDecodeFn");

  auto found = decode_fns.find(type_name);
  if (found == decode_fns.end()) return nullptr;
  return &found->second;
}

void UnaryVariantOpRegistry::RegisterDecodeFn(
    const string& type_name, const VariantDecodeFn& decode_fn) {
   std::vector<std::string> mht_5_v;
   mht_5_v.push_back("type_name: \"" + type_name + "\"");
   MHTracer_DTPStensorflowPScorePSframeworkPSvariant_op_registryDTcc mht_5(mht_5_v, 258, "", "./tensorflow/core/framework/variant_op_registry.cc", "UnaryVariantOpRegistry::RegisterDecodeFn");

  CHECK(!type_name.empty()) << "Need a valid name for UnaryVariantDecode";
  VariantDecodeFn* existing = GetDecodeFn(type_name);
  CHECK_EQ(existing, nullptr)
      << "Unary VariantDecodeFn for type_name: " << type_name
      << " already registered";
  decode_fns.insert(std::pair<StringPiece, VariantDecodeFn>(
      GetPersistentStringPiece(type_name), decode_fn));
}

bool DecodeUnaryVariant(Variant* variant) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSvariant_op_registryDTcc mht_6(mht_6_v, 271, "", "./tensorflow/core/framework/variant_op_registry.cc", "DecodeUnaryVariant");

  CHECK_NOTNULL(variant);
  if (variant->TypeName().empty()) {
    VariantTensorDataProto* t = variant->get<VariantTensorDataProto>();
    if (t == nullptr || !t->metadata().empty() || !t->tensors().empty()) {
      // Malformed variant.
      return false;
    } else {
      // Serialization of an empty Variant.
      variant->clear();
      return true;
    }
  }
  UnaryVariantOpRegistry::VariantDecodeFn* decode_fn =
      UnaryVariantOpRegistry::Global()->GetDecodeFn(variant->TypeName());
  if (decode_fn == nullptr) {
    return false;
  }
  const string type_name = variant->TypeName();
  bool decoded = (*decode_fn)(variant);
  if (!decoded) return false;
  if (variant->TypeName() != type_name) {
    LOG(ERROR) << "DecodeUnaryVariant: Variant type_name before decoding was: "
               << type_name
               << " but after decoding was: " << variant->TypeName()
               << ".  Treating this as a failure.";
    return false;
  }
  return true;
}

// Add some basic registrations for use by others, e.g., for testing.

#define REGISTER_VARIANT_DECODE_TYPE(T) \
  REGISTER_UNARY_VARIANT_DECODE_FUNCTION(T, TF_STR(T));

// No encode/decode registered for std::complex<> and Eigen::half
// objects yet.
REGISTER_VARIANT_DECODE_TYPE(int);
REGISTER_VARIANT_DECODE_TYPE(float);
REGISTER_VARIANT_DECODE_TYPE(bool);
REGISTER_VARIANT_DECODE_TYPE(double);

#undef REGISTER_VARIANT_DECODE_TYPE

Status VariantDeviceCopy(
    const VariantDeviceCopyDirection direction, const Variant& from,
    Variant* to,
    const UnaryVariantOpRegistry::AsyncTensorDeviceCopyFn& copy_fn) {
  UnaryVariantOpRegistry::AsyncVariantDeviceCopyFn* device_copy_fn =
      UnaryVariantOpRegistry::Global()->GetDeviceCopyFn(direction,
                                                        from.TypeId());
  if (device_copy_fn == nullptr) {
    return errors::Internal(
        "No unary variant device copy function found for direction: ",
        direction, " and Variant type_index: ",
        port::MaybeAbiDemangle(from.TypeId().name()));
  }
  return (*device_copy_fn)(from, to, copy_fn);
}

namespace {
template <typename T>
Status DeviceCopyPrimitiveType(
    const T& in, T* out,
    const UnaryVariantOpRegistry::AsyncTensorDeviceCopyFn& copier) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSvariant_op_registryDTcc mht_7(mht_7_v, 339, "", "./tensorflow/core/framework/variant_op_registry.cc", "DeviceCopyPrimitiveType");

  // Dummy copy, we don't actually bother copying to the device and back for
  // testing.
  *out = in;
  return Status::OK();
}
}  // namespace

#define REGISTER_VARIANT_DEVICE_COPY_TYPE(T)            \
  INTERNAL_REGISTER_UNARY_VARIANT_DEVICE_COPY_FUNCTION( \
      T, VariantDeviceCopyDirection::HOST_TO_DEVICE,    \
      DeviceCopyPrimitiveType<T>);                      \
  INTERNAL_REGISTER_UNARY_VARIANT_DEVICE_COPY_FUNCTION( \
      T, VariantDeviceCopyDirection::DEVICE_TO_HOST,    \
      DeviceCopyPrimitiveType<T>);                      \
  INTERNAL_REGISTER_UNARY_VARIANT_DEVICE_COPY_FUNCTION( \
      T, VariantDeviceCopyDirection::DEVICE_TO_DEVICE,  \
      DeviceCopyPrimitiveType<T>);

// No zeros_like registered for std::complex<> or Eigen::half objects yet.
REGISTER_VARIANT_DEVICE_COPY_TYPE(int);
REGISTER_VARIANT_DEVICE_COPY_TYPE(float);
REGISTER_VARIANT_DEVICE_COPY_TYPE(double);
REGISTER_VARIANT_DEVICE_COPY_TYPE(bool);

#undef REGISTER_VARIANT_DEVICE_COPY_TYPE

namespace {
template <typename T>
Status ZerosLikeVariantPrimitiveType(OpKernelContext* ctx, const T& t,
                                     T* t_out) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSvariant_op_registryDTcc mht_8(mht_8_v, 372, "", "./tensorflow/core/framework/variant_op_registry.cc", "ZerosLikeVariantPrimitiveType");

  *t_out = T(0);
  return Status::OK();
}
}  // namespace

#define REGISTER_VARIANT_ZEROS_LIKE_TYPE(T)                             \
  REGISTER_UNARY_VARIANT_UNARY_OP_FUNCTION(ZEROS_LIKE_VARIANT_UNARY_OP, \
                                           DEVICE_CPU, T,               \
                                           ZerosLikeVariantPrimitiveType<T>);

// No zeros_like registered for std::complex<> or Eigen::half objects yet.
REGISTER_VARIANT_ZEROS_LIKE_TYPE(int);
REGISTER_VARIANT_ZEROS_LIKE_TYPE(float);
REGISTER_VARIANT_ZEROS_LIKE_TYPE(double);
REGISTER_VARIANT_ZEROS_LIKE_TYPE(bool);

#undef REGISTER_VARIANT_ZEROS_LIKE_TYPE

namespace {
template <typename T>
Status AddVariantPrimitiveType(OpKernelContext* ctx, const T& a, const T& b,
                               T* out) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSvariant_op_registryDTcc mht_9(mht_9_v, 397, "", "./tensorflow/core/framework/variant_op_registry.cc", "AddVariantPrimitiveType");

  *out = a + b;
  return Status::OK();
}
}  // namespace

#define REGISTER_VARIANT_ADD_TYPE(T)                                           \
  REGISTER_UNARY_VARIANT_BINARY_OP_FUNCTION(ADD_VARIANT_BINARY_OP, DEVICE_CPU, \
                                            T, AddVariantPrimitiveType<T>);

// No add registered for std::complex<> or Eigen::half objects yet.
REGISTER_VARIANT_ADD_TYPE(int);
REGISTER_VARIANT_ADD_TYPE(float);
REGISTER_VARIANT_ADD_TYPE(double);
REGISTER_VARIANT_ADD_TYPE(bool);

#undef REGISTER_VARIANT_ADD_TYPE

}  // namespace tensorflow
