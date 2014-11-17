//
// Copyright (C) 2013 OpenSim Ltd.
//
// This program is free software; you can redistribute it and/or
// modify it under the terms of the GNU Lesser General Public License
// as published by the Free Software Foundation; either version 2
// of the License, or (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public License
// along with this program; if not, see <http://www.gnu.org/licenses/>.
//

#ifndef __INET_CUDASTRATEGY_H
#define __INET_CUDASTRATEGY_H

#include "inet/physicallayer/contract/IParallelStrategy.h"
#include "inet/physicallayer/contract/IRadio.h"
#include "inet/physicallayer/contract/IRadioMedium.h"
#include "inet/physicallayer/contract/ITransmission.h"
#include "inet/physicallayer/contract/IReception.h"
#include "inet/physicallayer/contract/IListening.h"
#include "inet/physicallayer/contract/IReceptionDecision.h"

namespace inet {

namespace physicallayer {

class RadioMedium;

#define CUDA_ERROR_CHECK(block) { cudaError_t code = block; if (code != cudaSuccess) throw cRuntimeError("CUDA error: %s %s %d\n", cudaGetErrorString(code), __FILE__, __LINE__); }

typedef int64_t cuda_simtime_t;

/**
 * Assuming single channel narrowband scalar transmissions, scalar background
 * noise, scalar receivers, free space path loss, and ignoring movement during
 * transmission, propagation and reception.
 */
class INET_API CUDAStrategy : public cModule, public IParallelStrategy
{
    protected:
        RadioMedium *radioMedium;

        double *hostRadioPositionXs;
        double *hostRadioPositionYs;
        double *hostRadioPositionZs;
        cuda_simtime_t *hostPropagationTimes;
        cuda_simtime_t *hostReceptionTimes;
        double *hostReceptionPowers;

        double *deviceRadioPositionXs;
        double *deviceRadioPositionYs;
        double *deviceRadioPositionZs;
        cuda_simtime_t *devicePropagationTimes;
        cuda_simtime_t *deviceReceptionTimes;
        double *deviceReceptionPowers;

        double pathLossAlpha;
        double backgroundNoisePower;

    protected:
        virtual int numInitStages() const { return INITSTAGE_LAST; }
        virtual void initialize(int stage);
        virtual void computeAllReceptionsForTransmission(const std::vector<const IRadio *> *radios, const std::vector<const ITransmission *> *transmissions) const;
        virtual void computeAllMinSNIRsForAllReceptions(const std::vector<const IRadio *> *radios, const std::vector<const ITransmission *> *transmissions) const;

    public:
        CUDAStrategy();
        virtual ~CUDAStrategy();

        virtual void printToStream(std::ostream& stream) const;
        virtual void computeCache(const std::vector<const IRadio *> *radios, const std::vector<const ITransmission *> *transmissions) const;
};

} // namespace physicallayer

} // namespace inet

#endif // ifndef __INET_CUDASTRATEGY_H

