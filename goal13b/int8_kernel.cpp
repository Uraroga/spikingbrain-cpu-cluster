#include <torch/extension.h>
#include <ATen/ParallelOpenMP.h>
#include <immintrin.h>
#include <cstdint>
#include <vector>

static inline int hsum128(__m128i v) {
  v = _mm_add_epi32(v, _mm_srli_si128(v, 8));
  v = _mm_add_epi32(v, _mm_srli_si128(v, 4));
  return _mm_cvtsi128_si32(v);
}

static inline int dot_i8_i8_sse(const int8_t* w, const int8_t* a) {
  __m128i sum = _mm_setzero_si128();
  for (int k = 0; k < 128; k += 8) {
    __m128i wb = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(w + k));
    __m128i ab = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(a + k));
    __m128i w16 = _mm_cvtepi8_epi16(wb);
    __m128i a16 = _mm_cvtepi8_epi16(ab);
    sum = _mm_add_epi32(sum, _mm_madd_epi16(w16, a16));
  }
  return hsum128(sum);
}

static inline int dot_i8_i16_sse(const int8_t* w, const int16_t* a) {
  __m128i sum = _mm_setzero_si128();
  for (int k = 0; k < 128; k += 8) {
    __m128i wb = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(w + k));
    __m128i w16 = _mm_cvtepi8_epi16(wb);
    __m128i a16 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(a + k));
    sum = _mm_add_epi32(sum, _mm_madd_epi16(w16, a16));
  }
  return hsum128(sum);
}

__attribute__((target("avx2"))) static int dot_i8_i8_avx2(const int8_t* w, const int8_t* a) {
  __m256i sum = _mm256_setzero_si256();
  for (int k = 0; k < 128; k += 16) {
    __m128i wb = _mm_loadu_si128(reinterpret_cast<const __m128i*>(w + k));
    __m128i ab = _mm_loadu_si128(reinterpret_cast<const __m128i*>(a + k));
    sum = _mm256_add_epi32(sum, _mm256_madd_epi16(_mm256_cvtepi8_epi16(wb), _mm256_cvtepi8_epi16(ab)));
  }
  __m128i s = _mm_add_epi32(_mm256_castsi256_si128(sum), _mm256_extracti128_si256(sum, 1));
  return hsum128(s);
}

__attribute__((target("avx2"))) static int dot_i8_i16_avx2(const int8_t* w, const int16_t* a) {
  __m256i sum = _mm256_setzero_si256();
  for (int k = 0; k < 128; k += 16) {
    __m128i wb = _mm_loadu_si128(reinterpret_cast<const __m128i*>(w + k));
    __m256i a16 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a + k));
    sum = _mm256_add_epi32(sum, _mm256_madd_epi16(_mm256_cvtepi8_epi16(wb), a16));
  }
  __m128i s = _mm_add_epi32(_mm256_castsi256_si128(sum), _mm256_extracti128_si256(sum, 1));
  return hsum128(s);
}

torch::Tensor group_acc_i8(torch::Tensor weights, torch::Tensor activation, bool use_avx2) {
  TORCH_CHECK(weights.device().is_cpu() && activation.device().is_cpu());
  TORCH_CHECK(weights.scalar_type() == at::kChar && activation.scalar_type() == at::kChar);
  TORCH_CHECK(weights.is_contiguous() && activation.is_contiguous());
  TORCH_CHECK(weights.dim() == 2 && activation.dim() == 1 && weights.size(1) == activation.size(0));
  TORCH_CHECK(weights.size(1) % 128 == 0);
  const int64_t O = weights.size(0), K = weights.size(1), G = K / 128;
  auto out = torch::empty({O, G}, torch::TensorOptions().dtype(torch::kInt32));
  const auto* wp = weights.data_ptr<int8_t>(); const auto* ap = activation.data_ptr<int8_t>(); auto* op = out.data_ptr<int32_t>();
  at::parallel_for(0, O, 0, [&](int64_t begin, int64_t end) {
    for (int64_t o = begin; o < end; ++o) for (int64_t g = 0; g < G; ++g) {
      const int8_t* w = wp + o*K + g*128; const int8_t* a = ap + g*128;
      op[o*G+g] = use_avx2 ? dot_i8_i8_avx2(w,a) : dot_i8_i8_sse(w,a);
    }
  });
  return out;
}

torch::Tensor group_acc_i16(torch::Tensor weights, torch::Tensor activation, bool use_avx2) {
  TORCH_CHECK(weights.device().is_cpu() && activation.device().is_cpu());
  TORCH_CHECK(weights.scalar_type() == at::kChar && activation.scalar_type() == at::kShort);
  TORCH_CHECK(weights.is_contiguous() && activation.is_contiguous());
  TORCH_CHECK(weights.dim() == 2 && activation.dim() == 1 && weights.size(1) == activation.size(0));
  TORCH_CHECK(weights.size(1) % 128 == 0);
  const int64_t O = weights.size(0), K = weights.size(1), G = K / 128;
  auto out = torch::empty({O, G}, torch::TensorOptions().dtype(torch::kInt32));
  const auto* wp = weights.data_ptr<int8_t>(); const auto* ap = activation.data_ptr<int16_t>(); auto* op = out.data_ptr<int32_t>();
  at::parallel_for(0, O, 0, [&](int64_t begin, int64_t end) {
    for (int64_t o = begin; o < end; ++o) for (int64_t g = 0; g < G; ++g) {
      const int8_t* w = wp + o*K + g*128; const int16_t* a = ap + g*128;
      op[o*G+g] = use_avx2 ? dot_i8_i16_avx2(w,a) : dot_i8_i16_sse(w,a);
    }
  });
  return out;
}

torch::Tensor outlier_correction(torch::Tensor rowptr, torch::Tensor indices, torch::Tensor residuals, torch::Tensor activation, int64_t groups) {
  TORCH_CHECK(rowptr.scalar_type()==at::kLong && indices.scalar_type()==at::kInt && residuals.scalar_type()==at::kShort && activation.scalar_type()==at::kShort);
  TORCH_CHECK(rowptr.is_contiguous() && indices.is_contiguous() && residuals.is_contiguous() && activation.is_contiguous());
  int64_t O=rowptr.numel()-1, K=activation.numel();
  auto out=torch::zeros({O,groups},torch::TensorOptions().dtype(torch::kInt32));
  auto* rp=rowptr.data_ptr<int64_t>(); auto* ip=indices.data_ptr<int32_t>(); auto* dp=residuals.data_ptr<int16_t>(); auto* ap=activation.data_ptr<int16_t>(); auto* op=out.data_ptr<int32_t>();
  at::parallel_for(0,O,0,[&](int64_t begin,int64_t end){
    for(int64_t o=begin;o<end;++o) for(int64_t j=rp[o];j<rp[o+1];++j){ int k=ip[j]; op[o*groups+k/128] += int32_t(dp[j])*int32_t(ap[k]); }
  });
  return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("group_acc_i8", &group_acc_i8);
  m.def("group_acc_i16", &group_acc_i16);
  m.def("outlier_correction", &outlier_correction);
}
