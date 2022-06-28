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
class MHTracer_DTPStensorflowPScorePSframeworkPStypes_testDTcc {
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
   MHTracer_DTPStensorflowPScorePSframeworkPStypes_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSframeworkPStypes_testDTcc() {
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

#include "tensorflow/core/framework/types.h"

#include "tensorflow/core/framework/type_traits.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/platform/protobuf.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {
namespace {

TEST(TypesTest, DeviceTypeName) {
  EXPECT_EQ("CPU", DeviceTypeString(DeviceType(DEVICE_CPU)));
  EXPECT_EQ("GPU", DeviceTypeString(DeviceType(DEVICE_GPU)));
}

TEST(TypesTest, kDataTypeRefOffset) {
  // Basic sanity check
  EXPECT_EQ(DT_FLOAT + kDataTypeRefOffset, DT_FLOAT_REF);

  // Use the meta-data provided by proto2 to iterate through the basic
  // types and validate that adding kDataTypeRefOffset gives the
  // corresponding reference type.
  const auto* enum_descriptor = DataType_descriptor();
  int e = DataType_MIN;
  if (e == DT_INVALID) ++e;
  int e_ref = e + kDataTypeRefOffset;
  EXPECT_FALSE(DataType_IsValid(e_ref - 1))
      << "Reference enum "
      << enum_descriptor->FindValueByNumber(e_ref - 1)->name()
      << " without corresponding base enum with value " << e - 1;
  for (;
       DataType_IsValid(e) && DataType_IsValid(e_ref) && e_ref <= DataType_MAX;
       ++e, ++e_ref) {
    string enum_name = enum_descriptor->FindValueByNumber(e)->name();
    string enum_ref_name = enum_descriptor->FindValueByNumber(e_ref)->name();
    EXPECT_EQ(enum_name + "_REF", enum_ref_name)
        << enum_name << "_REF should have value " << e_ref << " not "
        << enum_ref_name;
    // Validate DataTypeString() as well.
    DataType dt_e = static_cast<DataType>(e);
    DataType dt_e_ref = static_cast<DataType>(e_ref);
    EXPECT_EQ(DataTypeString(dt_e) + "_ref", DataTypeString(dt_e_ref));

    // Test DataTypeFromString reverse conversion
    DataType dt_e2, dt_e2_ref;
    EXPECT_TRUE(DataTypeFromString(DataTypeString(dt_e), &dt_e2));
    EXPECT_EQ(dt_e, dt_e2);
    EXPECT_TRUE(DataTypeFromString(DataTypeString(dt_e_ref), &dt_e2_ref));
    EXPECT_EQ(dt_e_ref, dt_e2_ref);
  }
  ASSERT_FALSE(DataType_IsValid(e))
      << "Should define " << enum_descriptor->FindValueByNumber(e)->name()
      << "_REF to be " << e_ref;
  ASSERT_FALSE(DataType_IsValid(e_ref))
      << "Extra reference enum "
      << enum_descriptor->FindValueByNumber(e_ref)->name()
      << " without corresponding base enum with value " << e;
  ASSERT_LT(DataType_MAX, e_ref)
      << "Gap in reference types, missing value for " << e_ref;

  // Make sure there are no enums defined after the last regular type before
  // the first reference type.
  for (; e < DataType_MIN + kDataTypeRefOffset; ++e) {
    EXPECT_FALSE(DataType_IsValid(e))
        << "Discontinuous enum value "
        << enum_descriptor->FindValueByNumber(e)->name() << " = " << e;
  }
}

TEST(TypesTest, DataTypeFromString) {
  DataType dt;
  ASSERT_TRUE(DataTypeFromString("int32", &dt));
  EXPECT_EQ(DT_INT32, dt);
  ASSERT_TRUE(DataTypeFromString("int32_ref", &dt));
  EXPECT_EQ(DT_INT32_REF, dt);
  EXPECT_FALSE(DataTypeFromString("int32_ref_ref", &dt));
  EXPECT_FALSE(DataTypeFromString("foo", &dt));
  EXPECT_FALSE(DataTypeFromString("foo_ref", &dt));
  ASSERT_TRUE(DataTypeFromString("int64", &dt));
  EXPECT_EQ(DT_INT64, dt);
  ASSERT_TRUE(DataTypeFromString("int64_ref", &dt));
  EXPECT_EQ(DT_INT64_REF, dt);
  ASSERT_TRUE(DataTypeFromString("quint8_ref", &dt));
  EXPECT_EQ(DT_QUINT8_REF, dt);
  ASSERT_TRUE(DataTypeFromString("bfloat16", &dt));
  EXPECT_EQ(DT_BFLOAT16, dt);
}

template <typename T>
static bool GetQuantized() {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSframeworkPStypes_testDTcc mht_0(mht_0_v, 274, "", "./tensorflow/core/framework/types_test.cc", "GetQuantized");

  return is_quantized<T>::value;
}

TEST(TypesTest, QuantizedTypes) {
  // NOTE: GUnit cannot parse is::quantized<TYPE>::value() within the
  // EXPECT_TRUE() clause, so we delegate through a template function.
  EXPECT_TRUE(GetQuantized<qint8>());
  EXPECT_TRUE(GetQuantized<quint8>());
  EXPECT_TRUE(GetQuantized<qint32>());

  EXPECT_FALSE(GetQuantized<int8>());
  EXPECT_FALSE(GetQuantized<uint8>());
  EXPECT_FALSE(GetQuantized<int16>());
  EXPECT_FALSE(GetQuantized<int32>());

  EXPECT_TRUE(DataTypeIsQuantized(DT_QINT8));
  EXPECT_TRUE(DataTypeIsQuantized(DT_QUINT8));
  EXPECT_TRUE(DataTypeIsQuantized(DT_QINT32));

  EXPECT_FALSE(DataTypeIsQuantized(DT_INT8));
  EXPECT_FALSE(DataTypeIsQuantized(DT_UINT8));
  EXPECT_FALSE(DataTypeIsQuantized(DT_UINT16));
  EXPECT_FALSE(DataTypeIsQuantized(DT_INT16));
  EXPECT_FALSE(DataTypeIsQuantized(DT_INT32));
  EXPECT_FALSE(DataTypeIsQuantized(DT_BFLOAT16));
}

TEST(TypesTest, ComplexTypes) {
  EXPECT_TRUE(DataTypeIsComplex(DT_COMPLEX64));
  EXPECT_TRUE(DataTypeIsComplex(DT_COMPLEX128));
  EXPECT_FALSE(DataTypeIsComplex(DT_FLOAT));
  EXPECT_FALSE(DataTypeIsComplex(DT_DOUBLE));
}

TEST(TypesTest, IntegerTypes) {
  for (auto dt : AllTypes()) {
    const string name = DataTypeString(dt);
    EXPECT_EQ(DataTypeIsInteger(dt),
              absl::StartsWith(name, "int") || absl::StartsWith(name, "uint"))
        << "DataTypeInteger failed for " << name;
  }
}

}  // namespace
}  // namespace tensorflow
