#ifndef MPS_PROJECT_CONSTANTS_H_
#define MPS_PROJECT_CONSTANTS_H_

#define NUM_DIMENSIONS 3

namespace mps {

constexpr double kPclDst = 0.02;
constexpr double kMinX = 0.0 - kPclDst * 3;
constexpr double kMinY = 0.0 - kPclDst * 3;
constexpr double kMinZ = 0.0 - kPclDst * 3;
constexpr double kMaxX = 1.0 + kPclDst * 3;
constexpr double kMaxY = 0.2 + kPclDst * 3;
constexpr double kMaxZ = 0.6 + kPclDst * 30;

constexpr int kGST = -1;
constexpr int kFLD = 0;
constexpr int kWLL = 1;
constexpr int kNumTyp = 2;

constexpr double kDnsFld = 1000.0;
constexpr double kDnsWll = 1000.0;
constexpr double kDt = 0.0005;
constexpr double kFinTim = 1.0;
constexpr double kSnd = 22.0;
constexpr int kOptFqc = 100;
constexpr double kKnmVsc = 0.000001;
constexpr int kDim = 3;
constexpr double kCrtNum = 0.1;
constexpr double kColRat = 0.2;
constexpr double kDstLmtRat = 0.9;
constexpr double kGx = 0.0;
constexpr double kGy = 0.0;
constexpr double kGz = -9.8;

}  // namespace mps

#endif  // MPS_PROJECT_CONSTANTS_H_
