/**
 * This file is part of SuperSLAM.
 *
 * Copyright (C) Aditya Wagh <adityamwagh at outlook dot com>
 * For more information see <https://github.com/adityamwagh/SuperSLAM>
 *
 * SuperSLAM is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * SuperSLAM is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with SuperSLAM. If not, see <http://www.gnu.org/licenses/>.
 */

#ifndef SPVOCABULARY_H
#define SPVOCABULARY_H

// #include"thirdparty/DBoW2/DBoW2/FSP.h"
// #include"thirdparty/DBoW2/DBoW2/TemplatedVocabulary.h"

#include "thirdparty/DBoW3/src/DBoW3.h"

namespace SuperSLAM {

// typedef DBoW2::TemplatedVocabulary<DBoW2::FSP::TDescriptor, DBoW2::FSP>
//   SPVocabulary;

typedef DBoW3::Vocabulary SPVocabulary;

typedef SPVocabulary ORBVocabulary;

}  // namespace SuperSLAM

#endif  // ORBVOCABULARY_H
