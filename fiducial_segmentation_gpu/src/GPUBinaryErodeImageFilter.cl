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

// TODO: handle 'BoundaryToForeground' bool option


#ifdef DIM_1
__kernel void BinaryErodeFilter(const __global INTYPE* in,
				     __constant int *inIndex,
				     __constant int *inSize,
				     __global OUTTYPE* out,
				     __constant int *outIndex,
				     __constant int *outSize,
				     __constant OPTYPE* op,
                                     const int radiusx
                                     const int sesum,
                                     const INTYPE fgValue,
                                     const INTYPE bgValue,
                                     const BOOL borderFg
				     )
{
  int gidx = get_global_id(0);
  int inGix = gidx + outIndex[0] - inIndex[0]; //index conversion
  unsigned int sum = 0;
  unsigned int opIdx = 0;

  int width = inSize[0];

  if(gidx < outSize[0])
  {
    // Clamping boundary condition
    for(int x = max((int)0, (int)(gidx-radiusx)); x <= min((int)(width-1), (int)(gidx+radiusx)); x++)
    {
      unsigned int cidx = x;

      if(op[opIdx] == 1)
	if(in[cidx] == fgValue)
	    ++sum;
	else
	    break;

      opIdx++;
    }
  }

  if(sum == sesum)
      out[gidx] = fgValue;
  else
      out[gidx] = bgValue;
}

#endif


#ifdef DIM_2
__kernel void BinaryErodeFilter(const __global INTYPE* in,
				     __constant int *inIndex,
				     __constant int *inSize,
				     __global OUTTYPE* out,
				     __constant int *outIndex,
				     __constant int *outSize,
				     __constant OPTYPE* op,
                                     const int radiusx,
                                     const int radiusy,
                                     const int sesum,
                                     const INTYPE fgValue,
                                     const INTYPE bgValue,
                                     const BOOL borderFg
				     )
{
  int gix = get_global_id(0);
  int giy = get_global_id(1);
  int inGix = gix + outIndex[0] - inIndex[0]; //index conversion
  int inGiy = giy + outIndex[1] - inIndex[1]; //index conversion

  unsigned int ingidx = inSize[0]*inGiy + inGix;
  unsigned int gidx = outSize[0]*giy + gix;
  unsigned int sum = 0;
  unsigned int opIdx = 0;
  bool finished = false;
  OPTYPE result = in[ingidx];
  if(gix < outSize[0] && giy < outSize[1])
  {
    for(int y = -radiusy; y <= radiusy; ++y)
    {
      for(int x = -radiusx; x <= radiusy; ++x)
      {
        if(op[opIdx] == 1)
        {
            int tx = gix-x + outIndex[0] - inIndex[0];
            if(tx < 0 || tx >= inSize[0]){
                  if(!borderFg){ finished = true; break; }
                  ++opIdx; ++sum; continue;
            }

            int ty = giy-y + outIndex[1] - inIndex[1];
            if(ty < 0 || ty >= inSize[1]){
                if(!borderFg){ finished = true; break; }
                ++opIdx; ++sum; continue;
            }

            unsigned int tidx = inSize[0]*ty + tx;

            if(in[tidx] == fgValue){
              ++sum;
            } else {
              finished = true;
              break;
            }
        }

	opIdx++;
      }

      if(finished) break;
    }

    if(in[ingidx] == fgValue && sum < sesum)
        result = bgValue;

    out[gidx] = result;
  }
}

#endif


#ifdef DIM_3
__kernel void BinaryErodeFilter(const __global INTYPE* in,
				     __constant int *inIndex,
				     __constant int *inSize,
				     __global OUTTYPE* out,
				     __constant int *outIndex,
				     __constant int *outSize,
				     __constant OPTYPE* op,
                                     const int radiusx,
                                     const int radiusy,
                                     const int radiusz,
                                     const int sesum,
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

  unsigned int ingidx = inSize[1]*inSize[0]*inGiz + inSize[0]*inGiy + inGix;
  unsigned int gidx = outSize[0]*(giz*outSize[1] + giy) + gix;
  unsigned int sum = 0;
  unsigned int opIdx = 0;

  /* NOTE: More than three-level nested conditional statements (e.g.,
     if A && B && C..) invalidates command queue during kernel
     execution on Apple OpenCL 1.0 (such Macbook Pro with NVIDIA 9600M
     GT). Therefore, we flattened conditional statements. */
  bool isValid = true;
  if(gix < 0 || gix >= outSize[0]) isValid = false;
  if(giy < 0 || giy >= outSize[1]) isValid = false;
  if(giz < 0 || giz >= outSize[2]) isValid = false;

  if( isValid )
  {
    OPTYPE result = in[ingidx];
    bool finished = false;
    for(int z = -radiusz; z <= radiusz; ++z)
    {
      for(int y = -radiusy; y <= radiusy; ++y)
      {
        for(int x = -radiusx; x <= radiusy; ++x)
        {
          if(op[opIdx] == 1)
          {
              int tx = gix-x + outIndex[0] - inIndex[0];
              if(tx < 0 || tx >= inSize[0]){
                    if(!borderFg){ finished = true; break; }
                    ++opIdx; ++sum; continue;
              }

              int ty = giy-y + outIndex[1] - inIndex[1];
              if(ty < 0 || ty >= inSize[1]){
                  if(!borderFg){ finished = true; break; }
                  ++opIdx; ++sum; continue;
              }

              int tz = giz-z + outIndex[2] - inIndex[2];
              if(tz < 0 || tz >= inSize[2]){
                  if(!borderFg){ finished = true; break; }
                  ++opIdx; ++sum; continue;
              }

              unsigned int tidx = inSize[1]*inSize[0]*tz + inSize[0]*ty + tx;

              if(in[tidx] == fgValue){
                ++sum;
              } else {
                finished = true;
                break;
              }
          }

          opIdx++;
        }

        if(finished) break;
      }
      if(finished) break;
    }

    if(in[ingidx] == fgValue && sum < sesum)
        result = bgValue;

    out[gidx] = result;
  }
}
#endif
