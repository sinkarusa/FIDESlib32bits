//
// Created by carlosad on 26/09/24.
//

#ifndef GPUCKKS_KEYSWITCHINGKEY_CUH
#define GPUCKKS_KEYSWITCHINGKEY_CUH

#include <cinttypes>
#include <memory>
#include <string>
#include <vector>

#include "RNSPoly.cuh"
#include "openfhe-interface/RawCiphertext.cuh"

namespace FIDESlib {
namespace CKKS {

/* FIDESLIB_ROT_KEYS_HOST=1 (lever L8): a host-backed rotation key keeps its packed `b` stream (the only device storage
 * of a packed key whose `a` is seed-regenerated) in pinned host memory. ContextData::GetRotationKey fetches it into a
 * device LRU cache (FIDESLIB_ROT_KEYS_DEV_GB) and evicts it again; the device pointer tables stay, rebased to the
 * fetched copy and nulled while evicted. All copies run on the legacy default stream, i.e. after every earlier op on
 * the (blocking) FIDESlib streams and before every later one. */
struct KeyHostImage {
    void* host = nullptr;          // pinned: the packed stream (`bytes`), then `scratch`, then `nulls`
    size_t bytes = 0;              // the packed stream (bufferKSKPACKbytes)
    uintptr_t base = 0;            // device address of the stream the table images were read against
    std::vector<uintptr_t> limbTab;                // limbptr as loaded
    std::vector<std::vector<uintptr_t>> digitTab;  // DIGITlimbptr[i] as loaded
    void** scratch = nullptr;      // pinned: the rebased tables of the last fetch
    void** nulls = nullptr;        // pinned: the tables with every entry into the stream nulled (eviction)
    cudaEvent_t done = nullptr;    // after the last table upload (scratch may be rewritten once it completed)
    uint64_t lastUse = 0;          // ContextData clock of the last GetRotationKey
    uint64_t scope = 0;            // ContextData::RotKeyScope the key was last used in (kept resident within it)
    KeyHostImage() = default;
    KeyHostImage(const KeyHostImage&) = delete;
    KeyHostImage& operator=(const KeyHostImage&) = delete;
    ~KeyHostImage();
};

class KeySwitchingKey {
    static constexpr const char* loc{"KeySwitchingKey"};
    CudaNvtxRange my_range;

   public:
    KeyHash keyID;
    Context& cc;
    RNSPoly a;
    RNSPoly b;
    // Rotation keys: the normalized index ContextData::AddRotationKey stored it under (-1 otherwise); keys the
    // FIDESLIB_ROT_LIMB_LOG record.
    int rot_index = -1;
    // std::vector<RNSPoly> mgpu_a;
    // std::vector<RNSPoly> mgpu_b;
    // L8: non-null while this key is host-backed (KeyHostImage).
    std::unique_ptr<KeyHostImage> host;

    explicit KeySwitchingKey(Context& cc);

    void Initialize(RawKeySwitchKey& rkk, int q_band = -1);
    /* L8: copy the packed `b` stream to pinned host memory and free the device copy (evicted state). False (key left
     * as is) unless single-GPU, packed, `a` released, and the pinned allocation succeeds. */
    bool hostOffload();
    bool hostResident() const;
    void hostFetch();   // device copy + rebased tables, on the legacy stream; no-op if resident
    void hostEvict();   // nulled tables + device copy freed, on the legacy stream; no-op if evicted
};

/* FIDESLIB_ROT_LIMB_LOG=<path> (lever N2a): record, per rotation key (its normalized index), the largest limb count
 * (ciphertext level + 1) a key switch used it at; written as JSON {"index": max_limbs} to <path> at process exit or by
 * WriteRotLimbLog. FIDESLIB_ROT_KEY_BANDS reads that file back (LoadContext). No-op when the env is unset. */
void NoteRotKeyUse(const KeySwitchingKey& k, int limbs);
/* Writes the log to `path` (empty: the env path); returns the number of keys written, -1 if logging is off. */
int WriteRotLimbLog(const std::string& path = "");

}  // namespace CKKS
}  // namespace FIDESlib

#endif  //GPUCKKS_KEYSWITCHINGKEY_CUH
