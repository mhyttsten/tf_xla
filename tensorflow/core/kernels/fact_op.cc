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
class MHTracer_DTPStensorflowPScorePSkernelsPSfact_opDTcc {
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
   MHTracer_DTPStensorflowPScorePSkernelsPSfact_opDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSkernelsPSfact_opDTcc() {
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

#include "tensorflow/core/framework/op_kernel.h"

namespace tensorflow {

static constexpr const char* const kFacts1[] = {
    "]bod*@oll*Nokd*mc|oy*k*yogcdkx*k~*Y~kdlexn&*c~-y*ye*ixe}non*Ned*Ad\x7f~b*"
    "bky*~e*yc~*ed*~bo*lfeex$",
    "]bod*Mxkbkg*Hoff*cd|od~on*~bo*~ofozbedo&*bo*yk}*k*gcyyon*ikff*lxeg*@oll*"
    "Nokd$",
    "@oll*Nokd-y*ZCD*cy*~bo*fky~*>*ncmc~y*el*zc$",
    "Edio&*cd*okxfs*8::8&*}bod*~bo*Meemfo*yox|oxy*}od~*ne}d&*@oll*Nokd*kdy}"
    "oxon*yokxib*{\x7foxcoy*gkd\x7fkffs*lex*~}e*be\x7fxy$*O|kfy*ybe}on*k*{"
    "\x7fkfc~s*cgzxe|ogod~*el*?*zecd~y$",
    "@oll*Nokd*z\x7f~y*bcy*zkd~y*ed*edo*fom*k~*k*~cgo&*h\x7f~*cl*bo*bkn*gexo*~"
    "bkd*~}e*fomy&*se\x7f*}e\x7f\x66n*yoo*~bk~*bcy*kzzxekib*cy*ki~\x7fkffs*"
    "E\"fem*d#$",
    "@oll*Nokd*iegzcfoy*kdn*x\x7f\x64y*bcy*ieno*holexo*y\x7fhgc~~cdm&*h\x7f~*"
    "edfs*~e*iboia*lex*iegzcfox*h\x7fmy$",
    "@oll*Nokd*ixok~on*~bo*}exfn-y*lcxy~*E\";%d#*kfmexc~bg$",
    "@oll*Nokd*}xe~o*kd*E\"dT8#*kfmexc~bg*edio$*C~*}ky*lex*~bo*^xk|ofcdm*"
    "Ykfoygkd*Zxehfog$",
    "^bo*xk~o*k~*}bcib*@oll*Nokd*zxen\x7fioy*ieno*`\x7fgzon*hs*k*lki~ex*el*>:*"
    "cd*fk~o*8:::*}bod*bo*\x7fzmxknon*bcy*aoshekxn*~e*_YH8$:$",
    "@oll*Nokd*ikd*hok~*se\x7f*k~*ieddoi~*le\x7fx$*Cd*~bxoo*ge|oy$",
    "@oll*Nokd*ade}y*}bs*~bo*kdy}ox*cy*>8$",
    "@oll*Nokd*y~kx~y*bcy*zxemxkggcdm*yoyycedy*}c~b*(ik~*4*%no|%gog($",
    "]bod*@oll*Nokd*yksy*(ezod*~bo*zen*hks*neexy(&*Bkf*ezody*~bo*zen*hks*"
    "neexy$",
    "@oll*Nokd*ycgzfs*}kfay*cd~e*Gexnex$",
    "Ib\x7fia*Dexxcy*cy*@oll*Nokd-y*8:/*zxe`oi~$",
    "@oll*Nokd-y*}k~ib*ncyzfksy*yoiedny*ycdio*@kd\x7fkxs*;y~&*;3=:$*Bo*cy*do|"
    "ox*fk~o$",
    "]bod*se\x7fx*ieno*bky*\x7f\x64nolcdon*hobk|cex&*se\x7f*mo~*k*"
    "yomlk\x7f\x66~*kdn*iexx\x7fz~on*nk~k$*]bod*@oll*Nokd-y*ieno*bky*"
    "\x7f\x64nolcdon*hobk|cex&*k*\x7f\x64\x63iexd*xcnoy*cd*ed*k*xkcdhe}*kdn*mc|"
    "oy*o|oxshens*lxoo*cio*ixokg$",
    "Moell*Bcd~ed*neoyd-~*doon*~e*gkao*bcnnod*\x7f\x64\x63~y$*^bos*bcno*hs*~"
    "bogyof|oy*}bod*bo*kzzxekiboy$",
    "Moell*Bcd~ed*neoyd-~*ncykmxoo&*bo*ied~xky~c|ofs*nc|oxmoy$",
    "Nooz*Hofcol*Do~}exay*ki~\x7fkffs*hofco|o*noozfs*cd*Moell*Bcd~ed$",
    "Moell*Bcd~ed*bky*ncyie|oxon*be}*~bo*hxkcd*xokffs*}exay$$$*edio*k*sokx&*"
    "lex*~bo*fky~*8?*sokxy$",
    "Gkxae|*xkdneg*lcofny*~bcda*Moell*Bcd~ed*cy*cd~xki~khfo$",
    "Moell*Bcd~ed*ncnd-~*cd|od~*femci&*h\x7f~*bcy*mxok~'mxok~'mxkdnlk~box*ncn$*"
    "\"^x\x7fo+#",
    "Moell*Bcd~ed*bky*}xc~~od*~}e*zkzoxy*~bk~*kxo*noy~cdon*~e*xo|ef\x7f~cedcpo*"
    "gkibcdo*fokxdcdm$*Dehens*ade}y*}bcib*~}e$"};
static constexpr uint64 kNum1 = sizeof(kFacts1) / sizeof(kFacts1[0]);

static constexpr const char* const kFacts2[] = {
    "Yoxmos*Hxcd*kdn*Hk~gkd*bk|o*do|ox*hood*yood*k~*~bo*ykgo*zfkio*k~*~bo*ykgo*"
    "~cgo$"};
static constexpr uint64 kNum2 = sizeof(kFacts2) / sizeof(kFacts2[0]);

static void E(string* s) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSfact_opDTcc mht_0(mht_0_v, 239, "", "./tensorflow/core/kernels/fact_op.cc", "E");

  for (size_t j = 0; j < s->size(); ++j) {
    (*s)[j] ^= '\n';
  }
}

class FactOpKernel : public OpKernel {
 public:
  explicit FactOpKernel(OpKernelConstruction* context) : OpKernel(context) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSfact_opDTcc mht_1(mht_1_v, 250, "", "./tensorflow/core/kernels/fact_op.cc", "FactOpKernel");
}

  void Compute(OpKernelContext* context) override = 0;

 protected:
  void Compute(OpKernelContext* context, const char* const facts[],
               uint64 count) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSfact_opDTcc mht_2(mht_2_v, 259, "", "./tensorflow/core/kernels/fact_op.cc", "Compute");

    Tensor* output_tensor = nullptr;
    OP_REQUIRES_OK(
        context, context->allocate_output(0, TensorShape({}), &output_tensor));
    auto output = output_tensor->template scalar<tstring>();

    string coded = facts[context->env()->NowMicros() % count];
    E(&coded);
    output() = coded;
  }
};

class FactOpKernel1 : public FactOpKernel {
 public:
  explicit FactOpKernel1(OpKernelConstruction* context)
      : FactOpKernel(context) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSfact_opDTcc mht_3(mht_3_v, 277, "", "./tensorflow/core/kernels/fact_op.cc", "FactOpKernel1");
}

  void Compute(OpKernelContext* context) override {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSfact_opDTcc mht_4(mht_4_v, 282, "", "./tensorflow/core/kernels/fact_op.cc", "Compute");

    FactOpKernel::Compute(context, kFacts1, kNum1);
  }
};

class FactOpKernel2 : public FactOpKernel {
 public:
  explicit FactOpKernel2(OpKernelConstruction* context)
      : FactOpKernel(context) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSfact_opDTcc mht_5(mht_5_v, 293, "", "./tensorflow/core/kernels/fact_op.cc", "FactOpKernel2");
}

  void Compute(OpKernelContext* context) override {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSfact_opDTcc mht_6(mht_6_v, 298, "", "./tensorflow/core/kernels/fact_op.cc", "Compute");

    FactOpKernel::Compute(context, kFacts2, kNum2);
  }
};

REGISTER_KERNEL_BUILDER(Name("Fact").Device(DEVICE_GPU).HostMemory("fact"),
                        FactOpKernel1);
REGISTER_KERNEL_BUILDER(Name("Fact").Device(DEVICE_DEFAULT).HostMemory("fact"),
                        FactOpKernel1);

static string D(const char* s) {
   std::vector<std::string> mht_7_v;
   mht_7_v.push_back("s: \"" + (s == nullptr ? std::string("nullptr") : std::string((char*)s)) + "\"");
   MHTracer_DTPStensorflowPScorePSkernelsPSfact_opDTcc mht_7(mht_7_v, 312, "", "./tensorflow/core/kernels/fact_op.cc", "D");

  string ret(s);
  E(&ret);
  return ret;
}

REGISTER_KERNEL_BUILDER(
    Name("Fact").Device(DEVICE_CPU).Label(D("Yoxmos").c_str()), FactOpKernel2);
REGISTER_KERNEL_BUILDER(
    Name("Fact").Device(DEVICE_CPU).Label(D("yoxmos").c_str()), FactOpKernel2);

}  // namespace tensorflow
