//
// Created by carlosad on 26/09/24.
//

#include <algorithm>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <map>
#include <mutex>
#include <source_location>
#include "CKKS/Context.cuh"
#include "CKKS/KeySwitchingKey.cuh"
#include "CKKS/RNSPoly.cuh"
#if defined(__clang__)
#include <experimental/source_location>
using sc = std::experimental::source_location;
constexpr int PREFIX_SIZE = 0;
#else
#include <source_location>
using sc = std::source_location;
constexpr int PREFIX_SIZE = 23;
#endif

namespace FIDESlib::CKKS {

namespace {
/* KSK bit-packing (FIDESLIB_KSK_PACK, default 1): uniform packed width for this chain's keys, or
 * 0 = dense. Lossless; saves 12.5% key device memory at the cost of an unpack in the dot kernels.
 * Eligible: single-GPU, all-u32 (type==0) chains whose max prime width has a packed kernel arm. */
int kskPackBitsPolicy(Context& cc) {
    static const bool enabled = [] {
        const char* e = std::getenv("FIDESLIB_KSK_PACK");
        return e == nullptr || std::atoi(e) != 0;
    }();
    if (!enabled || cc->GPUid.size() != 1)
        return 0;
    const auto& hc = cc->precom.constants[0];
    if (hc.type != 0)
        return 0;
    const int K = (int)cc->splitSpecialMeta.at(0).size();
    int W = 0;
    for (int i = 0; i < cc->L + K; ++i)
        W = std::max<int>(W, (int)hc.prime_bits[i]);
    // Widths are compile-time in the dot kernels: arm only the instantiated set; other chains
    // fall back dense (add an instantiation to extend).
    return (W == 27 || W == 28) ? W : 0;
}
}  // namespace

void KeySwitchingKey::Initialize(RawKeySwitchKey& rkk, int q_band) {
    CudaNvtxRange r(std::string{sc::current().function_name()}.substr());
    CKKS::SetCurrentContext(cc);
    keyID = rkk.keyid;

    if (q_band >= 0 && cc->GPUid.size() > 1)
        q_band = -1;  // banding is single-GPU only
    // A seeded key's `a` component is regenerated on-GPU from its 256-bit seed (bit-identical to
    // rkk.r_key[0]), skipping that half of the key upload. `b`, and seedless keys, load normally.
    const bool seeded_a = !rkk.a_seed.empty() && cc->GPUid.size() == 1;
    // At regen level >= 2 both `a`-readers (hoistedRotateDotKSK and fusedDotKSK) regenerate from the
    // seed, so `a` is never read and is released instead of expanded. The N condition is the regen
    // kernels' whole-threads requirement; if it fails the launch gates stream, so `a` must exist.
    const bool release_a = seeded_a && kskRegenLevel() >= 2 &&
                           cc->N % (128 * (cc->precom.constants[0].type == 0 ? 16 : 8)) == 0;

    // When released, `a`'s rows are never allocated: the pointer tables the host staging paths read
    // live in bufferAUXptrs and are built by the LimbPartition constructor from the metas.
    if (!release_a)
        a.generateDecompAndDigit(true, q_band);
    b.generateDecompAndDigit(true, q_band);
    if (cc->GPUid.size() > 1) {
        a.grow(cc->L, false, true);
        b.grow(cc->L, false, true);
    }
    if (release_a)
        a.GPU.at(0).adoptKskASeed(rkk.a_seed, q_band);
    else if (seeded_a)
        a.GPU.at(0).expandKskADigits(rkk.a_seed);
    else {
        if (rkk.r_key[0].empty() && rkk.fill_a)
            rkk.fill_a(rkk);
        a.loadDecompDigit(rkk.r_key[0], rkk.r_key_moduli[0]);
    }
    b.loadDecompDigit(rkk.r_key[1], rkk.r_key_moduli[1]);

    if (const int W = kskPackBitsPolicy(cc)) {
        if (release_a)
            // `a` has no rows to pack; the width still has to match `b` (the compile-time KSK_BITS
            // the dot kernels select on).
            a.GPU.at(0).key_pack_bits = W;
        else
            a.GPU.at(0).packKeyLimbs(W);
        b.GPU.at(0).packKeyLimbs(W);
    }

    cudaDeviceSynchronize();
}

namespace {
// L8: upload `src` (images of limbptr then each DIGITlimbptr[i], as KeyHostImage lays them out) into the key's
// pointer tables on the legacy stream. `src` is pinned, so the copies are asynchronous.
void uploadKeyTables(LimbPartition& p, const KeyHostImage& h, void* const* src) {
    size_t k = 0;
    if (!h.limbTab.empty())
        cudaMemcpyAsync(p.limbptr.data, src, h.limbTab.size() * sizeof(void*), cudaMemcpyHostToDevice, 0);
    k += h.limbTab.size();
    for (size_t i = 0; i < h.digitTab.size(); ++i) {
        if (!h.digitTab[i].empty())
            cudaMemcpyAsync(p.DIGITlimbptr[i].data, src + k, h.digitTab[i].size() * sizeof(void*),
                            cudaMemcpyHostToDevice, 0);
        k += h.digitTab[i].size();
    }
}
}  // namespace

KeyHostImage::~KeyHostImage() {
    // a fetch or eviction may still be reading the pinned images: they ran on the legacy stream
    cudaStreamSynchronize(0);
    if (done)
        cudaEventDestroy(done);
    pinnedHostFree(host);
}

bool KeySwitchingKey::hostOffload() {
    if (host)
        return true;
    if (a.GPU.size() != 1 || b.GPU.size() != 1)
        return false;
    LimbPartition& p = b.GPU[0];
    if (!p.key_pack_bits || !p.bufferKSKPACK || !a.GPU[0].ksk_a_released)
        return false;
    cudaSetDevice(p.device);
    cudaDeviceSynchronize();
    auto img = std::make_unique<KeyHostImage>();
    img->bytes = p.bufferKSKPACKbytes;
    img->base = reinterpret_cast<uintptr_t>(p.bufferKSKPACK);
    auto readTab = [](const VectorGPU<void*>& t) {
        std::vector<uintptr_t> v(t.data != nullptr && t.size > 0 ? (size_t)t.size : 0);
        if (!v.empty())
            cudaMemcpy(v.data(), t.data, v.size() * sizeof(void*), cudaMemcpyDeviceToHost);
        return v;
    };
    img->limbTab = readTab(p.limbptr);
    for (auto& t : p.DIGITlimbptr)
        img->digitTab.push_back(readTab(t));
    size_t ntab = img->limbTab.size();
    for (auto& t : img->digitTab)
        ntab += t.size();
    img->host = pinnedHostAlloc(img->bytes + 2 * ntab * sizeof(void*));
    if (!img->host || cudaEventCreateWithFlags(&img->done, cudaEventDisableTiming) != cudaSuccess) {
        cudaGetLastError();
        img->done = nullptr;
        return false;
    }
    cudaMemcpy(img->host, p.bufferKSKPACK, img->bytes, cudaMemcpyDeviceToHost);
    // bufferKSKPACKbytes is a multiple of 16 (packKeyLimbs' slots), so the tables are aligned
    img->scratch = reinterpret_cast<void**>(static_cast<char*>(img->host) + img->bytes);
    img->nulls = img->scratch + ntab;
    const uintptr_t lo = img->base, hi = img->base + img->bytes;
    size_t k = 0;
    for (const uintptr_t v : img->limbTab)
        img->nulls[k++] = (v >= lo && v < hi) ? nullptr : reinterpret_cast<void*>(v);
    for (auto& t : img->digitTab)
        for (const uintptr_t v : t)
            img->nulls[k++] = (v >= lo && v < hi) ? nullptr : reinterpret_cast<void*>(v);
    if (cudaGetLastError() != cudaSuccess)
        return false;
    host = std::move(img);
    hostEvict();
    return true;
}

bool KeySwitchingKey::hostResident() const {
    return b.GPU.at(0).bufferKSKPACK != nullptr;
}

void KeySwitchingKey::hostFetch() {
    if (!host)
        return;
    LimbPartition& p = b.GPU.at(0);
    if (p.bufferKSKPACK)
        return;
    KeyHostImage& h = *host;
    cudaSetDevice(p.device);
    // unpooled size: cudaMallocAsync on the legacy stream, like the copies below
    void* d = GPUmalloc(p.device, (int)h.bytes, nullptr);
    cudaMemcpyAsync(d, h.host, h.bytes, cudaMemcpyHostToDevice, 0);
    cudaEventSynchronize(h.done);   // the previous fetch's table upload has read `scratch`
    const uintptr_t lo = h.base, hi = h.base + h.bytes, nb = reinterpret_cast<uintptr_t>(d);
    auto rebase = [&](uintptr_t v) { return reinterpret_cast<void*>((v >= lo && v < hi) ? v - lo + nb : v); };
    size_t k = 0;
    for (const uintptr_t v : h.limbTab)
        h.scratch[k++] = rebase(v);
    for (auto& t : h.digitTab)
        for (const uintptr_t v : t)
            h.scratch[k++] = rebase(v);
    uploadKeyTables(p, h, h.scratch);
    cudaEventRecord(h.done, 0);
    p.bufferKSKPACK = static_cast<uint64_t*>(d);
}

void KeySwitchingKey::hostEvict() {
    if (!host)
        return;
    LimbPartition& p = b.GPU.at(0);
    if (!p.bufferKSKPACK)
        return;
    cudaSetDevice(p.device);
    uploadKeyTables(p, *host, host->nulls);
    GPUfree(p.bufferKSKPACK, p.id, (int)host->bytes, nullptr);
    p.bufferKSKPACK = nullptr;
}

KeySwitchingKey::KeySwitchingKey(Context& cc)
    : my_range(loc, LIFETIME),
      keyID(""),
      cc((assert(cc != nullptr), CudaNvtxStart(std::string{sc::current().function_name()}.substr()), cc)),
      a(*cc, -1, false, true),
      b(*cc, -1, false, true) {
    CudaNvtxStop();
    /*
    if (cc.GPUid.size() > 1) {
        for (int j = 0; j < cc.dnum; ++j) {
            mgpu_a.emplace_back(cc, -1);
            mgpu_b.emplace_back(cc, -1);
        }
    }
     */
}

namespace {
const char* rotLimbLogPath() {
    static const char* p = [] {
        const char* e = std::getenv("FIDESLIB_ROT_LIMB_LOG");
        return (e && *e) ? e : static_cast<const char*>(nullptr);
    }();
    return p;
}
std::mutex& rotLimbMu() {
    static std::mutex m;
    return m;
}
std::map<int, int>& rotLimbMax() {
    static std::map<int, int> m;
    return m;
}
}  // namespace

void NoteRotKeyUse(const KeySwitchingKey& k, int limbs) {
    if (!rotLimbLogPath() || k.rot_index < 0)
        return;
    // the statics exist before the handler is registered, so it runs before their destructors
    static const bool at_exit = [] {
        rotLimbMu();
        rotLimbMax();
        std::atexit([] { WriteRotLimbLog(); });
        return true;
    }();
    (void)at_exit;
    std::lock_guard<std::mutex> g(rotLimbMu());
    int& m = rotLimbMax()[k.rot_index];
    m = std::max(m, limbs);
}

int WriteRotLimbLog(const std::string& path) {
    const char* env = rotLimbLogPath();
    if (!env)
        return -1;
    const std::string out = path.empty() ? std::string(env) : path;
    std::map<int, int> snap;
    {
        std::lock_guard<std::mutex> g(rotLimbMu());
        snap = rotLimbMax();
    }
    std::ofstream f(out);
    if (!f) {
        std::cout << "[N2] cannot write the rotation limb log to " << out << std::endl;
        return -1;
    }
    f << "{";
    bool first = true;
    for (const auto& [idx, limbs] : snap) {
        f << (first ? "" : ", ") << "\"" << idx << "\": " << limbs;
        first = false;
    }
    f << "}\n";
    std::cout << "[N2] rotation limb log: " << snap.size() << " keys -> " << out << std::endl;
    return static_cast<int>(snap.size());
}
}  // namespace FIDESlib::CKKS
