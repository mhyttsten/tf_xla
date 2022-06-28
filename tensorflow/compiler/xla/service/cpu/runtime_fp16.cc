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
class MHTracer_DTPStensorflowPScompilerPSxlaPSservicePScpuPSruntime_fp16DTcc {
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
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePScpuPSruntime_fp16DTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSxlaPSservicePScpuPSruntime_fp16DTcc() {
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

/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/service/cpu/runtime_fp16.h"

#include <cstring>

#include "absl/base/attributes.h"

namespace {

// Helper class that lets us access the underlying bit representation
// of a float without breaking C++ strict aliasing.
class AliasedFloatInt {
 public:
  static_assert(sizeof(float) == sizeof(uint32_t), "");

  static AliasedFloatInt FromFloat(float f) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePScpuPSruntime_fp16DTcc mht_0(mht_0_v, 199, "", "./tensorflow/compiler/xla/service/cpu/runtime_fp16.cc", "FromFloat");

    AliasedFloatInt value;
    value.set_float(f);
    return value;
  }

  static AliasedFloatInt FromUInt(uint32_t u) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePScpuPSruntime_fp16DTcc mht_1(mht_1_v, 208, "", "./tensorflow/compiler/xla/service/cpu/runtime_fp16.cc", "FromUInt");

    AliasedFloatInt value;
    value.set_uint(u);
    return value;
  }

  void set_float(float f) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePScpuPSruntime_fp16DTcc mht_2(mht_2_v, 217, "", "./tensorflow/compiler/xla/service/cpu/runtime_fp16.cc", "set_float");
 memcpy(&value_, &f, sizeof(f)); }
  float as_float() const {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePScpuPSruntime_fp16DTcc mht_3(mht_3_v, 221, "", "./tensorflow/compiler/xla/service/cpu/runtime_fp16.cc", "as_float");

    float f;
    memcpy(&f, &value_, sizeof(f));
    return f;
  }

  void set_uint(uint32_t u) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePScpuPSruntime_fp16DTcc mht_4(mht_4_v, 230, "", "./tensorflow/compiler/xla/service/cpu/runtime_fp16.cc", "set_uint");
 value_ = u; }
  uint32_t as_uint() const {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePScpuPSruntime_fp16DTcc mht_5(mht_5_v, 234, "", "./tensorflow/compiler/xla/service/cpu/runtime_fp16.cc", "as_uint");
 return value_; }

 private:
  uint32_t value_;
};
}  // namespace

// __gnu_f2h_ieee and __gnu_h2f_ieee are marked as weak symbols so if XLA is
// built with compiler-rt (that also defines these symbols) we don't get a
// duplicate definition linker error.  Making these symbols weak also ensures
// that the compiler-rt definitions "win", but that isn't essential.

// Algorithm copied from Eigen.
uint16_t ABSL_ATTRIBUTE_WEAK __gnu_f2h_ieee(float float_value) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePScpuPSruntime_fp16DTcc mht_6(mht_6_v, 250, "", "./tensorflow/compiler/xla/service/cpu/runtime_fp16.cc", "__gnu_f2h_ieee");

  AliasedFloatInt f = AliasedFloatInt::FromFloat(float_value);

  const AliasedFloatInt f32infty = AliasedFloatInt::FromUInt(255 << 23);
  const AliasedFloatInt f16max = AliasedFloatInt::FromUInt((127 + 16) << 23);
  const AliasedFloatInt denorm_magic =
      AliasedFloatInt::FromUInt(((127 - 15) + (23 - 10) + 1) << 23);
  unsigned int sign_mask = 0x80000000u;
  uint32_t o = static_cast<uint16_t>(0x0u);

  unsigned int sign = f.as_uint() & sign_mask;
  f.set_uint(f.as_uint() ^ sign);

  // NOTE all the integer compares in this function can be safely
  // compiled into signed compares since all operands are below
  // 0x80000000. Important if you want fast straight SSE2 code
  // (since there's no unsigned PCMPGTD).

  if (f.as_uint() >=
      f16max.as_uint()) {  // result is Inf or NaN (all exponent bits set)
    o = (f.as_uint() > f32infty.as_uint()) ? 0x7e00
                                           : 0x7c00;  // NaN->qNaN and Inf->Inf
  } else {                            // (De)normalized number or zero
    if (f.as_uint() < (113 << 23)) {  // resulting FP16 is subnormal or zero
      // use a magic value to align our 10 mantissa bits at the bottom of
      // the float. as long as FP addition is round-to-nearest-even this
      // just works.
      f.set_float(f.as_float() + denorm_magic.as_float());

      // and one integer subtract of the bias later, we have our final float!
      o = static_cast<uint16_t>(f.as_uint() - denorm_magic.as_uint());
    } else {
      unsigned int mant_odd =
          (f.as_uint() >> 13) & 1;  // resulting mantissa is odd

      // update exponent, rounding bias part 1
      f.set_uint(f.as_uint() + (static_cast<unsigned int>(15 - 127) << 23) +
                 0xfff);
      // rounding bias part 2
      f.set_uint(f.as_uint() + mant_odd);
      // take the bits!
      o = static_cast<uint16_t>(f.as_uint() >> 13);
    }
  }

  o |= static_cast<uint16_t>(sign >> 16);
  return o;
}

// Algorithm copied from Eigen.
float ABSL_ATTRIBUTE_WEAK __gnu_h2f_ieee(uint16_t h) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePScpuPSruntime_fp16DTcc mht_7(mht_7_v, 303, "", "./tensorflow/compiler/xla/service/cpu/runtime_fp16.cc", "__gnu_h2f_ieee");

  const AliasedFloatInt magic = AliasedFloatInt::FromUInt(113 << 23);
  const unsigned int shifted_exp = 0x7c00 << 13;  // exponent mask after shift
  AliasedFloatInt o;

  o.set_uint((h & 0x7fff) << 13);                // exponent/mantissa bits
  unsigned int exp = shifted_exp & o.as_uint();  // just the exponent
  o.set_uint(o.as_uint() + ((127 - 15) << 23));  // exponent adjust

  // handle exponent special cases
  if (exp == shifted_exp) {                        // Inf/NaN?
    o.set_uint(o.as_uint() + ((128 - 16) << 23));  // extra exp adjust
  } else if (exp == 0) {                           // Zero/Denormal?
    o.set_uint(o.as_uint() + (1 << 23));           // extra exp adjust
    o.set_float(o.as_float() - magic.as_float());  // renormalize
  }

  o.set_uint(o.as_uint() | (h & 0x8000) << 16);  // sign bit
  return o.as_float();
}

uint16_t ABSL_ATTRIBUTE_WEAK __truncdfhf2(double d) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePScpuPSruntime_fp16DTcc mht_8(mht_8_v, 327, "", "./tensorflow/compiler/xla/service/cpu/runtime_fp16.cc", "__truncdfhf2");

  // This does a double rounding step, but it's precise enough for our use
  // cases.
  return __gnu_f2h_ieee(static_cast<float>(d));
}
