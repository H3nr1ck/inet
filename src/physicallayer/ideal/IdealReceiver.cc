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

#include "IdealReceiver.h"
#include "IdealListening.h"
#include "IdealReception.h"
#include "ListeningDecision.h"
#include "SynchronizationDecision.h"
#include "ReceptionDecision.h"

namespace inet {

namespace physicallayer {

Define_Module(IdealReceiver);

IdealReceiver::IdealReceiver() :
    ignoreInterference(false)
{
}

void IdealReceiver::initialize(int stage)
{
    if (stage == INITSTAGE_LOCAL) {
        ignoreInterference = par("ignoreInterference");
    }
}

bool IdealReceiver::computeIsSynchronizationPossible(const IListening *listening, const IReception *reception) const
{
    const IdealReception::Power power = check_and_cast<const IdealReception *>(reception)->getPower();
    return power == IdealReception::POWER_RECEIVABLE;
}

bool IdealReceiver::computeIsSynchronizationAttempted(const IListening *listening, const IReception *reception, const std::vector<const IReception *> *interferingReceptions) const
{
    if (ignoreInterference)
        return computeIsSynchronizationPossible(listening, reception);
    else
        return ReceiverBase::computeIsSynchronizationAttempted(listening, reception, interferingReceptions);
}

bool IdealReceiver::computeIsReceptionPossible(const IListening *listening, const IReception *reception) const
{
    const IdealReception::Power power = check_and_cast<const IdealReception *>(reception)->getPower();
    return power == IdealReception::POWER_RECEIVABLE;
}

bool IdealReceiver::computeIsReceptionAttempted(const IListening *listening, const IReception *reception, const std::vector<const IReception *> *interferingReceptions) const
{
    if (ignoreInterference)
        return computeIsReceptionPossible(listening, reception);
    else
        return ReceiverBase::computeIsReceptionAttempted(listening, reception, interferingReceptions);
}

void IdealReceiver::printToStream(std::ostream& stream) const
{
    stream << "ideal receiver, " << (ignoreInterference ? "ignore interference" : "compute interference");
}

const IListening *IdealReceiver::createListening(const IRadio *radio, const simtime_t startTime, const simtime_t endTime, const Coord startPosition, const Coord endPosition) const
{
    return new IdealListening(radio, startTime, endTime, startPosition, endPosition);
}

const IListeningDecision *IdealReceiver::computeListeningDecision(const IListening *listening, const std::vector<const IReception *> *interferingReceptions, const INoise *backgroundNoise) const
{
    for (std::vector<const IReception *>::const_iterator it = interferingReceptions->begin(); it != interferingReceptions->end(); it++) {
        const IReception *interferingReception = *it;
        IdealReception::Power interferingPower = check_and_cast<const IdealReception *>(interferingReception)->getPower();
        if (interferingPower != IdealReception::POWER_UNDETECTABLE)
            return new ListeningDecision(listening, true);
    }
    return new ListeningDecision(listening, false);
}

const ISynchronizationDecision *IdealReceiver::computeSynchronizationDecision(const IListening *listening, const IReception *reception, const std::vector<const IReception *> *interferingReceptions, const INoise *backgroundNoise) const
{
    // TODO: factor
    const IdealReception::Power power = check_and_cast<const IdealReception *>(reception)->getPower();
    SynchronizationIndication *indication = new SynchronizationIndication();
    if (power == IdealReception::POWER_RECEIVABLE) {
        if (ignoreInterference)
            return new SynchronizationDecision(reception, indication, true, true, true);
        else {
            for (std::vector<const IReception *>::const_iterator it = interferingReceptions->begin(); it != interferingReceptions->end(); it++) {
                const IReception *interferingReception = *it;
                // TODO: check if synchronization duration is also interfering
                IdealReception::Power interferingPower = check_and_cast<const IdealReception *>(interferingReception)->getPower();
                if (interferingPower == IdealReception::POWER_RECEIVABLE || interferingPower == IdealReception::POWER_INTERFERING)
                    return new SynchronizationDecision(reception, indication, true, true, false);
            }
            return new SynchronizationDecision(reception, indication, true, true, true);
        }
    }
    else
        return new SynchronizationDecision(reception, indication, false, false, false);
}

const IReceptionDecision *IdealReceiver::computeReceptionDecision(const IListening *listening, const IReception *reception, const std::vector<const IReception *> *interferingReceptions, const INoise *backgroundNoise) const
{
    const IdealReception::Power power = check_and_cast<const IdealReception *>(reception)->getPower();
    ReceptionIndication *indication = new ReceptionIndication();
    if (power == IdealReception::POWER_RECEIVABLE) {
        if (ignoreInterference)
            return new ReceptionDecision(reception, indication, true, true, true);
        else {
            for (std::vector<const IReception *>::const_iterator it = interferingReceptions->begin(); it != interferingReceptions->end(); it++) {
                const IReception *interferingReception = *it;
                IdealReception::Power interferingPower = check_and_cast<const IdealReception *>(interferingReception)->getPower();
                if (interferingPower == IdealReception::POWER_RECEIVABLE || interferingPower == IdealReception::POWER_INTERFERING)
                    return new ReceptionDecision(reception, indication, true, true, false);
            }
            return new ReceptionDecision(reception, indication, true, true, true);
        }
    }
    else
        return new ReceptionDecision(reception, indication, false, false, false);
}

} // namespace physicallayer

} // namespace inet

