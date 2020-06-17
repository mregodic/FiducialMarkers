/*=========================================================================
 *
 *  Copyright Insight Software Consortium
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *         http://www.apache.org/licenses/LICENSE-2.0.txt
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 *
 *=========================================================================*/

 // TODO: check and fix "inIndex" and "outIndex" role; and what happens
 // when inSize != outSize

#ifdef DIM_1
__kernel void BinaryDilateFilter(const __global INTYPE* in,
	__constant int *inIndex,
	__constant int *inSize,
	__global OUTTYPE* out,
	__constant int *outIndex,
	__constant int *outSize,
	__constant OPTYPE* op,
	const int radiusx,
	const INTYPE fgValue,
	const INTYPE bgValue,
	const BOOL borderFg
)
{
	int gidx = get_global_id(0);
	int inGix = gidx + outIndex[0] - inIndex[0]; //index conversion
	unsigned int opIdx = (2 * radiusx + 1) - 1;

	if (gix < outSize[0])
	{
		OUTTYPE result = in[ingidx];
		for (int x = -radiusx; x <= radiusx; x++)
		{
			if (op[opIdx] == 1) {
				int tx = gix - x;
				if (tx < 0 || tx >= inSize[0]) {
					if (borderFg) { result = fgValue; break; }
					--opIdx; continue;
				}

				unsigned int tidx = tx;
				if (in[tidx] == fgValue) {
					result = fgValue;
					break;
				}
			}
			--opIdx;
		}

		out[gidx] = result;
	}
}

#endif

#ifdef DIM_2
__kernel void BinaryDilateFilter(const __global INTYPE* in,
	__constant int *inIndex,
	__constant int *inSize,
	__global OUTTYPE* out,
	__constant int *outIndex,
	__constant int *outSize,
	__constant OPTYPE* op,
	const int radiusx,
	const int radiusy,
	const INTYPE fgValue,
	const INTYPE bgValue,
	const BOOL borderFg
)
{
	int gix = get_global_id(0);
	int giy = get_global_id(1);

	int inGix = gix + outIndex[0] - inIndex[0]; //index conversion
	int inGiy = giy + outIndex[1] - inIndex[1]; //index conversion

	unsigned int ingidx = inSize[0] * inGiy + inGix;
	unsigned int gidx = outSize[0] * giy + gix;
	unsigned int opIdx = (2 * radiusy + 1)*(2 * radiusx + 1) - 1;

	OUTTYPE result = in[ingidx];
	bool finished = false;
	if (gix < outSize[0] && giy < outSize[1])
	{
		for (int y = -radiusy; y <= radiusy; y++)
		{
			for (int x = -radiusx; x <= radiusx; x++)
			{
				if (op[opIdx] == 1) {
					int tx = gix - x + outIndex[0] - inIndex[0];
					if (tx < 0 || tx >= inSize[0]) {
						if (borderFg) { result = fgValue; finished = true; break; }
						--opIdx; continue;
					}

					int ty = giy - y + outIndex[1] - inIndex[1];
					if (ty < 0 || ty >= inSize[1]) {
						if (borderFg) { result = fgValue; finished = true; break; }
						--opIdx; continue;
					}

					unsigned int tidx = inSize[0] * ty + tx;
					if (in[tidx] == fgValue) {
						result = fgValue;
						finished = true;
						break;
					}
				}

				--opIdx;
			}

			if (finished)
				break;
		}

		out[gidx] = result;
	}
}

#endif



#ifdef DIM_3

__kernel void BinaryDilateFilter(const __global INTYPE* in,
	__constant int *inIndex,
	__constant int *inSize,
	__global OUTTYPE* out,
	__constant int *outIndex,
	__constant int *outSize,
	__constant OPTYPE* op,
	const int radiusx,
	const int radiusy,
	const int radiusz,
	const INTYPE fgValue,
	const INTYPE bgValue,
	const BOOL borderFg
)
{
	int gix = get_global_id(0);
	int giy = get_global_id(1);
	int giz = get_global_id(2);
	int inGix = gix + outIndex[0] - inIndex[0]; //index conversion
	int inGiy = giy + outIndex[1] - inIndex[1]; //index conversion
	int inGiz = giz + outIndex[2] - inIndex[2]; //index conversion
	unsigned int gidx = outSize[0] * (giz*outSize[1] + giy) + gix;
	unsigned int opIdx = (2 * radiusz + 1)*(2 * radiusy + 1)*(2 * radiusx + 1) - 1;

	/* NOTE: More than three-level nested conditional statements (e.g.,
	   if A && B && C..) invalidates command queue during kernel
	   execution on Apple OpenCL 1.0 (such Macbook Pro with NVIDIA 9600M
	   GT). Therefore, we flattened conditional statements. */
	bool isValid = true;
	if (gix < 0 || gix >= outSize[0]) isValid = false;
	if (giy < 0 || giy >= outSize[1]) isValid = false;
	if (giz < 0 || giz >= outSize[2]) isValid = false;

	OUTTYPE result = 0;
	bool finished = false;
	if (isValid)
	{
		for (int z = -radiusz; z <= radiusz; z++)
		{
			for (int y = -radiusy; y <= radiusy; y++)
			{
				for (int x = -radiusx; x <= radiusx; x++)
				{
					if (op[opIdx] == 1) {
						int tx = gix - x + outIndex[0] - inIndex[0];
						if (tx < 0 || tx >= inSize[0]) {
							if (borderFg) { result = fgValue; finished = true; break; }
							--opIdx; continue;
						}

						int ty = giy - y + outIndex[1] - inIndex[1];
						if (ty < 0 || ty >= inSize[1]) {
							if (borderFg) { result = fgValue; finished = true; break; }
							--opIdx; continue;
						}

						int tz = giz - z + outIndex[2] - inIndex[2];
						if (tz < 0 || tz >= inSize[2]) {
							if (borderFg) { result = fgValue; finished = true; break; }
							--opIdx; continue;
						}

						unsigned int tidx = inSize[0] * inSize[1] * tz + inSize[0] * ty + tx;
						if (in[tidx] == fgValue) {
							result = fgValue;
							finished = true;
							break;
						}
					}
					--opIdx;
				}
				if (finished) break;
			}
			if (finished) break;
		}

		out[gidx] = result;
	}
}

#endif

