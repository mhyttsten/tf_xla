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
class MHTracer_DTPStensorflowPScorePSkernelsPSxsmm_conv2d_testDTcc {
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
   MHTracer_DTPStensorflowPScorePSkernelsPSxsmm_conv2d_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSkernelsPSxsmm_conv2d_testDTcc() {
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

/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

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

#include "include/libxsmm.h"
#include "tensorflow/core/framework/fake_input.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/graph/node_builder.h"
#include "tensorflow/core/kernels/conv_ops.h"
#include "tensorflow/core/kernels/ops_testutil.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {
namespace {

typedef struct {
  int nImg;
  int nIfm;
  int nOfm;
  int ifhp;
  int ifwp;
  int ifh;
  int ifw;
  int ofhp;
  int ofwp;
  int ofh;
  int ofw;
  int pad_h;
  int pad_w;
  int pad_h_in;
  int pad_w_in;
  int pad_h_out;
  int pad_w_out;
  int kh;
  int kw;
  int stride_h;
  int stride_w;
} naive_conv_t;

LIBXSMM_INLINE void naive_copy_NCHW_to_NHWC(const float* nchw, Tensor& nhwc,
                                            int N, int H, int W, int C) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSxsmm_conv2d_testDTcc mht_0(mht_0_v, 221, "", "./tensorflow/core/kernels/xsmm_conv2d_test.cc", "naive_copy_NCHW_to_NHWC");

  LIBXSMM_VLA_DECL(4, const float, input, nchw, C, H, W);
  int n, h, w, c;
  auto output = nhwc.flat<float>();
  for (n = 0; n < N; n++) {
    for (h = 0; h < H; h++) {
      for (w = 0; w < W; w++) {
        for (c = 0; c < C; c++) {
          output(n * H * W * C + h * W * C + w * C + c) =
              LIBXSMM_VLA_ACCESS(4, input, n, c, h, w, C, H, W);
        }
      }
    }
  }
}

LIBXSMM_INLINE void naive_copy_KCRS_to_RSCK(const float* kcrs, Tensor& rsck,
                                            int R, int S, int C, int K) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSxsmm_conv2d_testDTcc mht_1(mht_1_v, 241, "", "./tensorflow/core/kernels/xsmm_conv2d_test.cc", "naive_copy_KCRS_to_RSCK");

  LIBXSMM_VLA_DECL(4, const float, input, kcrs, C, R, S);
  int r, s, c, k;
  auto output = rsck.flat<float>();

  for (r = 0; r < R; r++) {
    for (s = 0; s < S; s++) {
      for (c = 0; c < C; c++) {
        for (k = 0; k < K; k++) {
          output(r * S * C * K + s * C * K + c * K + k) =
              LIBXSMM_VLA_ACCESS(4, input, k, c, r, s, C, R, S);
        }
      }
    }
  }
}

LIBXSMM_INLINE void zero_buf(float* buf, long size) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSxsmm_conv2d_testDTcc mht_2(mht_2_v, 261, "", "./tensorflow/core/kernels/xsmm_conv2d_test.cc", "zero_buf");

  int i;
  for (i = 0; i < size; ++i) {
    buf[i] = 0.0f;
  }
}

LIBXSMM_INLINE void copy_buf(Tensor& dst, float* src, long size) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSxsmm_conv2d_testDTcc mht_3(mht_3_v, 271, "", "./tensorflow/core/kernels/xsmm_conv2d_test.cc", "copy_buf");

  long i;
  auto output = dst.flat<float>();
  for (i = 0; i < size; ++i) output(i) = src[i];
}

LIBXSMM_INLINE void init_buf(float* buf, long size, int initPos, int initOne) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSxsmm_conv2d_testDTcc mht_4(mht_4_v, 280, "", "./tensorflow/core/kernels/xsmm_conv2d_test.cc", "init_buf");

  int i;
  zero_buf(buf, size);
  for (i = 0; i < size; ++i) {
    buf[i] =
        (float)((initOne != 0)
                    ? 1.0
                    : ((initPos != 0) ? drand48() : (0.05 - drand48() / 10.0)));
  }
}

LIBXSMM_INLINE void naive_conv_fp(naive_conv_t* param, const float* input,
                                  float* output, const float* filter) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSxsmm_conv2d_testDTcc mht_5(mht_5_v, 295, "", "./tensorflow/core/kernels/xsmm_conv2d_test.cc", "naive_conv_fp");

  int nImg = param->nImg;
  int nIfm = param->nIfm;
  int nOfm = param->nOfm;
  int ifhp = param->ifhp;
  int ifwp = param->ifwp;
  int ofhp = param->ofhp;
  int ofwp = param->ofwp;
  int ifh = param->ifh;
  int ifw = param->ifw;
  int ofh = param->ofh;
  int ofw = param->ofw;
  int pad_h = param->pad_h;
  int pad_w = param->pad_w;
  int pad_h_in = param->pad_h_in;
  int pad_w_in = param->pad_w_in;
  int pad_h_out = param->pad_h_out;
  int pad_w_out = param->pad_w_out;
  int kh = param->kh;
  int kw = param->kw;
  int stride_h = param->stride_h;
  int stride_w = param->stride_w;
  /* loop counters */
  int img, ofm, ifm, oj, oi, ij, ii, kj, ki;

  LIBXSMM_VLA_DECL(4, float, output_t, output + (pad_w_out * ofwp + pad_h_out),
                   nOfm, ofhp, ofwp);
  LIBXSMM_VLA_DECL(4, const float, input_t,
                   input + (pad_w_in * ifwp + pad_h_in), nIfm, ifhp, ifwp);
  LIBXSMM_VLA_DECL(4, const float, filter_t, filter, nIfm, kh, kw);

  for (img = 0; img < nImg; ++img) {
    for (ofm = 0; ofm < nOfm; ++ofm) {
      for (ifm = 0; ifm < nIfm; ++ifm) {
        for (oj = 0; oj < ofh; ++oj) {
          ij = oj * stride_h - pad_h;
          for (oi = 0; oi < ofw; ++oi) {
            ii = oi * stride_w - pad_w;
            for (kj = 0; kj < kh; ++kj) {
              if (ij + kj < 0 || ij + kj >= ifh) continue;
              for (ki = 0; ki < kw; ++ki) {
                if (ii + ki < 0 || ii + ki >= ifw) continue;
                LIBXSMM_VLA_ACCESS(4, output_t, img, ofm, oj, oi, nOfm, ofhp,
                                   ofwp) +=
                    LIBXSMM_VLA_ACCESS(4, input_t, img, ifm, ij + kj, ii + ki,
                                       nIfm, ifhp, ifwp) *
                    LIBXSMM_VLA_ACCESS(4, filter_t, ofm, ifm, kj, ki, nIfm, kh,
                                       kw);
              }
            }
          }
        }
      }
    }
  }
}

void RunXsmmVsGeneric() {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSxsmm_conv2d_testDTcc mht_6(mht_6_v, 355, "", "./tensorflow/core/kernels/xsmm_conv2d_test.cc", "RunXsmmVsGeneric");
}

class XsmmConv2DTest : public OpsTestBase {
 protected:
  void MakeOp(int stride) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSxsmm_conv2d_testDTcc mht_7(mht_7_v, 362, "", "./tensorflow/core/kernels/xsmm_conv2d_test.cc", "MakeOp");

    TF_CHECK_OK(NodeDefBuilder("xsmm", "Conv2D")
                    .Input(FakeInput(DT_FLOAT))
                    .Input(FakeInput(DT_FLOAT))
                    .Attr("strides", {1, stride, stride, 1})
                    .Attr("padding", "VALID")
                    .Finalize(node_def()));

    TF_ASSERT_OK(InitOp());
  }
};

TEST_F(XsmmConv2DTest, Basic) {
  MakeOp(1);

  // setup scoped allocator, which uses cpu_allocator() for this scope
  const libxsmm_tf_allocator<libxsmm_scratch_allocator> tf_allocator;

  int ifw = 14;   /* input width, "W" */
  int ifh = 14;   /* input height, "H" */
  int nImg = 32;  /* mini-batch size, "N" */
  int nIfm = 64;  /* number of input feature maps, "C" */
  int nOfm = 64;  /* number of output feature maps, "K" */
  int kh = 3;     /* filter height, "R" */
  int kw = 3;     /* filter width, "S" */
  int pad = 0;    /* padding in output */
  int stride = 1; /* stride when accessing inputs */

  int stride_w = stride;
  int stride_h = stride;
  int pad_h = pad;
  int pad_w = pad;

  int pad_h_in = pad_h;
  int pad_w_in = pad_w;

  int pad_h_out = 0;
  int pad_w_out = 0;

  /* deriving some values for naive code */
  int ofh = (ifh + 2 * pad_h - kh) / stride_h + 1;
  int ofw = (ifw + 2 * pad_w - kw) / stride_w + 1;
  int ifhp = ifh + 2 * pad_h_in;
  int ifwp = ifw + 2 * pad_w_in;
  int ofhp = ofh + 2 * pad_h_out;
  int ofwp = ofw + 2 * pad_w_out;

  // Initialization of Filter and Image

  /* allocate data */
  float* naive_input = (float*)libxsmm_aligned_scratch(
      nImg * nIfm * ifhp * ifwp * sizeof(float), 2097152);
  float* naive_output = (float*)libxsmm_aligned_scratch(
      nImg * nOfm * ofhp * ofwp * sizeof(float), 2097152);
  float* naive_filter = (float*)libxsmm_aligned_scratch(
      nOfm * nIfm * kh * kw * sizeof(float), 2097152);
  /* initialize data */
  init_buf(naive_input, nImg * nIfm * ifhp * ifwp, 0, 0);
  zero_buf(naive_output, nImg * nOfm * ofhp * ofwp);
  init_buf(naive_filter, nOfm * nIfm * kh * kw, 0, 0);

  Tensor image(DT_FLOAT, {nImg, ifhp, ifwp, nIfm});

  Tensor filter(DT_FLOAT, {kh, kw, nIfm, nOfm});

  naive_copy_NCHW_to_NHWC(naive_input, image, nImg, ifhp, ifwp, nIfm);
  naive_copy_KCRS_to_RSCK(naive_filter, filter, kh, kw, nIfm, nOfm);

  // Run naive convolution

  naive_conv_t naive_param;

  naive_param.nImg = nImg;
  naive_param.nIfm = nIfm;
  naive_param.nOfm = nOfm;
  naive_param.ifhp = ifhp;
  naive_param.ifwp = ifwp;
  naive_param.ofhp = ofhp;
  naive_param.ofwp = ofwp;
  naive_param.ifh = ifh;
  naive_param.ifw = ifw;
  naive_param.ofh = ofh;
  naive_param.ofw = ofw;
  naive_param.pad_h = pad_h;
  naive_param.pad_w = pad_w;
  naive_param.pad_h_in = pad_h_in;
  naive_param.pad_w_in = pad_w_in;
  naive_param.pad_h_out = pad_h_out;
  naive_param.pad_w_out = pad_w_out;
  naive_param.kh = kh;
  naive_param.kw = kw;
  naive_param.stride_h = stride_h;
  naive_param.stride_w = stride_w;

  naive_conv_fp(&naive_param, naive_input, naive_output, naive_filter);

  AddInputFromArray<float>(image.shape(), image.flat<float>());
  AddInputFromArray<float>(filter.shape(), filter.flat<float>());

  // Run Op (TF)
  TF_ASSERT_OK(RunOpKernel());

  // Check the output.
  Tensor expected(DT_FLOAT, {nImg, ofhp, ofwp, nOfm});
  naive_copy_NCHW_to_NHWC(naive_output, expected, nImg, ofhp, ofwp, nOfm);

  test::ExpectTensorNear<float>(expected, *GetOutput(0), 1e-5);
  libxsmm_free(naive_input);
  libxsmm_free(naive_output);
  libxsmm_free(naive_filter);
}

/*


TEST(XsmmConv2DTest, Basic) {

    auto num_threads =
        ctx->device()->tensorflow_cpu_worker_threads()->num_threads;
    // See libxsmm_dnn.h for this struct definition.
    libxsmm_dnn_conv_desc desc;
    desc.N = batch;
    desc.C = in_depth;
    desc.H = input_rows;
    desc.W = input_cols;
    desc.K = out_depth;
    desc.R = filter_rows;
    desc.S = filter_cols;
    desc.u = stride_rows;
    desc.v = stride_cols;
    desc.pad_h = pad_rows;
    desc.pad_w = pad_cols;
    desc.pad_h_in = pad_rows;  // libxsmm supports only physical padding for now
    desc.pad_w_in = pad_cols;  // libxsmm supports only physical padding for now
    desc.pad_h_out = 0;
    desc.pad_w_out = 0;
    desc.threads = num_threads;
    desc.algo = LIBXSMM_DNN_CONV_ALGO_DIRECT;
    desc.buffer_format = LIBXSMM_DNN_TENSOR_FORMAT_NHWC;
    desc.filter_format =
LIBXSMM_DNN_TENSOR_FORMAT_LIBXSMM;//LIBXSMM_DNN_TENSOR_FORMAT_RSCK;
    desc.fuse_ops = LIBXSMM_DNN_CONV_FUSE_NONE;
    desc.options = LIBXSMM_DNN_CONV_OPTION_NONE;
    desc.datatype_out = LIBXSMM_DNN_DATATYPE_F32;
    desc.datatype_in = LIBXSMM_DNN_DATATYPE_F32;
    if (!CanUseXsmmConv2D(desc, data_format)) {
      return false;
    }

    auto input_ptr = input.template flat<float>().data();
    auto filter_ptr = filter.template flat<float>().data();
    auto output_ptr = output->template flat<float>().data();

    bool success = functor::XsmmFwdConv2D<CPUDevice, float>()(
        ctx, desc, input_ptr, filter_ptr, output_ptr);
    return success;







}
*/
}  // namespace
}  // namespace tensorflow
