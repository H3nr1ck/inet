//
// Copyright (C) 2014 OpenSim Ltd.
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

#include "OFDMSymbol.h"

namespace inet {
namespace physicallayer {

void physicallayer::OFDMSymbol::pushAPSKSymbol(const APSKSymbol* apskSymbol, int subcarrierIndex)
{
    if (subcarrierIndex >= 52)
        throw cRuntimeError("Out of range with subcarrierIndex = %d", subcarrierIndex);
    subcarrierSymbols[subcarrierIndex] = apskSymbol;
}

OFDMSymbol::~OFDMSymbol()
{
    // Dynamically created pilot symbols need to be deleted
    // TODO: Ieee80211 specific
    delete subcarrierSymbols[5];
    delete subcarrierSymbols[19];
    delete subcarrierSymbols[33];
    delete subcarrierSymbols[47];
}

} /* namespace physicallayer */
} /* namespace inet */
