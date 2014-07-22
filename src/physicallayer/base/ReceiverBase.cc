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

#include "ReceiverBase.h"
#include "IRadio.h"
#include "IRadioMedium.h"

namespace inet {

namespace physicallayer {

bool ReceiverBase::computeIsSynchronizationPossible(const ITransmission *transmission) const
{
    return true;
}

bool ReceiverBase::computeIsSynchronizationAttempted(const IListening *listening, const IReception *reception, const std::vector<const IReception *> *interferingReceptions) const
{
    // TODO: factor with reception?
    if (!computeIsSynchronizationPossible(listening, reception))
        return false;
    else if (simTime() == reception->getStartTime())
        // TODO: isn't there a better way for this optimization? see also in RadioMedium::isSynchronizationAttempted
        return !reception->getReceiver()->getSynchronizationInProgress();
    else {
        const IRadio *radio = reception->getReceiver();
        const IRadioMedium *channel = radio->getMedium();
        for (std::vector<const IReception *>::const_iterator it = interferingReceptions->begin(); it != interferingReceptions->end(); it++) {
            const IReception *interferingReception = *it;
            // TODO: check if synchronization duration is also interfering
            bool isPrecedingReception = interferingReception->getStartTime() < reception->getStartTime() ||
                (interferingReception->getStartTime() == reception->getStartTime() &&
                 interferingReception->getTransmission()->getId() < reception->getTransmission()->getId());
            if (isPrecedingReception) {
                const ITransmission *interferingTransmission = interferingReception->getTransmission();
                if (interferingReception->getStartTime() <= simTime()) {
                    if (radio->getReceptionInProgress() == interferingTransmission)
                        return false;
                }
                else if (channel->isSynchronizationAttempted(radio, interferingTransmission))
                    return false;
            }
        }
        return true;
    }
}

bool ReceiverBase::computeIsReceptionPossible(const ITransmission *transmission) const
{
    return true;
}

bool ReceiverBase::computeIsReceptionAttempted(const IListening *listening, const IReception *reception, const std::vector<const IReception *> *interferingReceptions) const
{
    if (!computeIsReceptionPossible(listening, reception))
        return false;
    else if (simTime() == reception->getStartTime())
        // TODO: isn't there a better way for this optimization? see also in RadioMedium::isReceptionAttempted
        return !reception->getReceiver()->getReceptionInProgress();
    else {
        const IRadio *radio = reception->getReceiver();
        const IRadioMedium *channel = radio->getMedium();
        for (std::vector<const IReception *>::const_iterator it = interferingReceptions->begin(); it != interferingReceptions->end(); it++) {
            const IReception *interferingReception = *it;
            bool isPrecedingReception = interferingReception->getStartTime() < reception->getStartTime() ||
                (interferingReception->getStartTime() == reception->getStartTime() &&
                 interferingReception->getTransmission()->getId() < reception->getTransmission()->getId());
            if (isPrecedingReception) {
                const ITransmission *interferingTransmission = interferingReception->getTransmission();
                if (interferingReception->getStartTime() <= simTime()) {
                    if (radio->getReceptionInProgress() == interferingTransmission)
                        return false;
                }
                else if (channel->isReceptionAttempted(radio, interferingTransmission))
                    return false;
            }
        }
        return true;
    }
}

} // namespace physicallayer

} // namespace inet

