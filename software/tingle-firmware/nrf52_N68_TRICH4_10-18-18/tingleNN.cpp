/**************************************************************************/
/*!
    @file		tingleNN.cpp
	@author		Curt White <curtpw@gmail.com>
	@version	0.1.0
	
	Gesture Detection for Tingle using SynapticJS MLP neural networks pre-trained for targets
	
	@section HISTORY
	
	v0.1.0	- First Release

*/
/**************************************************************************/

#include <nrf_sdm.h>
#include <nrf_soc.h>

#include "tingleNN.h"

//pre-trained neural network weights and activation functions
#include "NN_LHAND_SHORT_BACK_5521.h"
#include "NN_LHAND_SHORT_EYES_5521.h"
#include "NN_LHAND_SHORT_FRONT_5521.h"
#include "NN_LHAND_SHORT_MOUTH_5521.h"
#include "NN_LHAND_SHORT_SIDE_5521.h"
#include "NN_LHAND_SHORT_TOP_5521.h"

#include "NN_LHAND_SHORT_BACK_7521.h"
#include "NN_LHAND_SHORT_EYES_7521.h"
#include "NN_LHAND_SHORT_FRONT_7521.h"
#include "NN_LHAND_SHORT_MOUTH_7521.h"
#include "NN_LHAND_SHORT_SIDE_7521.h"
#include "NN_LHAND_SHORT_TOP_7521.h"

#include "NN_RHAND_SHORT_BACK_5521.h"
#include "NN_RHAND_SHORT_EYES_5521.h"
#include "NN_RHAND_SHORT_FRONT_5521.h"
#include "NN_RHAND_SHORT_MOUTH_5521.h"
#include "NN_RHAND_SHORT_SIDE_5521.h"
#include "NN_RHAND_SHORT_TOP_5521.h"

#include "NN_RHAND_SHORT_BACK_7521.h"
#include "NN_RHAND_SHORT_EYES_7521.h"
#include "NN_RHAND_SHORT_FRONT_7521.h"
#include "NN_RHAND_SHORT_MOUTH_7521.h"
#include "NN_RHAND_SHORT_SIDE_7521.h"
#include "NN_RHAND_SHORT_TOP_7521.h"

#include "NN_LHAND_MEDIUM_BACK_5521.h"
#include "NN_LHAND_MEDIUM_EYES_5521.h"
#include "NN_LHAND_MEDIUM_FRONT_5521.h"
#include "NN_LHAND_MEDIUM_MOUTH_5521.h"
#include "NN_LHAND_MEDIUM_SIDE_5521.h"
#include "NN_LHAND_MEDIUM_TOP_5521.h"

#include "NN_LHAND_MEDIUM_BACK_7521.h"
#include "NN_LHAND_MEDIUM_EYES_7521.h"
#include "NN_LHAND_MEDIUM_FRONT_7521.h"
#include "NN_LHAND_MEDIUM_MOUTH_7521.h"
#include "NN_LHAND_MEDIUM_SIDE_7521.h"
#include "NN_LHAND_MEDIUM_TOP_7521.h"

#include "NN_RHAND_MEDIUM_BACK_5521.h"
#include "NN_RHAND_MEDIUM_EYES_5521.h"
#include "NN_RHAND_MEDIUM_FRONT_5521.h"
#include "NN_RHAND_MEDIUM_MOUTH_5521.h"
#include "NN_RHAND_MEDIUM_SIDE_5521.h"
#include "NN_RHAND_MEDIUM_TOP_5521.h"

#include "NN_RHAND_MEDIUM_BACK_7521.h"
#include "NN_RHAND_MEDIUM_EYES_7521.h"
#include "NN_RHAND_MEDIUM_FRONT_7521.h"
#include "NN_RHAND_MEDIUM_MOUTH_7521.h"
#include "NN_RHAND_MEDIUM_SIDE_7521.h"
#include "NN_RHAND_MEDIUM_TOP_7521.h"

#include "NN_LHAND_LONG_BACK_5521.h"
#include "NN_LHAND_LONG_EYES_5521.h"
#include "NN_LHAND_LONG_FRONT_5521.h"
#include "NN_LHAND_LONG_MOUTH_5521.h"
#include "NN_LHAND_LONG_SIDE_5521.h"
#include "NN_LHAND_LONG_TOP_5521.h"

#include "NN_LHAND_LONG_BACK_7521.h"
#include "NN_LHAND_LONG_EYES_7521.h"
#include "NN_LHAND_LONG_FRONT_7521.h"
#include "NN_LHAND_LONG_MOUTH_7521.h"
#include "NN_LHAND_LONG_SIDE_7521.h"
#include "NN_LHAND_LONG_TOP_7521.h"

#include "NN_RHAND_LONG_BACK_5521.h"
#include "NN_RHAND_LONG_EYES_5521.h"
#include "NN_RHAND_LONG_FRONT_5521.h"
#include "NN_RHAND_LONG_MOUTH_5521.h"
#include "NN_RHAND_LONG_SIDE_5521.h"
#include "NN_RHAND_LONG_TOP_5521.h"

#include "NN_RHAND_LONG_BACK_7521.h"
#include "NN_RHAND_LONG_EYES_7521.h"
#include "NN_RHAND_LONG_FRONT_7521.h"
#include "NN_RHAND_LONG_MOUTH_7521.h"
#include "NN_RHAND_LONG_SIDE_7521.h"
#include "NN_RHAND_LONG_TOP_7521.h"

     extern   float _F1_5in[500];
       extern    float _F1_7in[575];
       extern    float _F2_5in[500];
       extern    float _F2_7in[575];

/**************************************************************************/
/*!
    @brief apply data to 5 INPUT NO ANGULAR POSITION neural network activation function and apply threshold to neural network output
*/
/**************************************************************************/
void tingleNeuralNetwork::checkSettings() { 


  
}


/**************************************************************************/
/*!
    @brief apply data to 5 INPUT NO ANGULAR POSITION neural network activation function and apply threshold to neural network output
*/
/**************************************************************************/
float tingleNeuralNetwork::detect(float t1, float t2, float t3, float  t4, float distance) { 
  float F[500];
//  F = _F1_5in;

     //  copy 'len' elements from 'src' to 'dst'
  //  memcpy(dst, src, sizeof(src[0])*len);
    memcpy(F, _F1_5in, sizeof(_F1_5in[0])*500);
    
Serial.print("First 10 Weights:\t");
for(int i = 0; i < 10; i++){Serial.print(F[i]); Serial.print(" ");}
Serial.print("inside NN5:\t"); Serial.print(t1); Serial.print("\t"); Serial.print(t2); Serial.print("\t"); Serial.print(t3); Serial.print("\t"); Serial.print(t4); Serial.print("\t"); Serial.println(distance);

  
  F[3] = t1;
  F[5] = t2;
  F[7] = t3;
  F[9] = t4;
  F[11] = distance;

F[0] = F[1];F[1] = F[2];F[1] += F[3] * F[4];F[1] += F[5] * F[6];F[1] += F[7] * F[8];F[1] += F[9] * F[10];F[1] += F[11] * F[12];F[1] += F[13] * F[14];F[1] += F[15] * F[16];F[1] += F[17] * F[18];F[1] += F[19] * F[20];F[1] += F[21] * F[22];F[23] = (1 / (1 + exp(-F[1])));F[24] = F[23] * (1 - F[23]);F[25] = F[23];F[26] = F[23];F[27] = F[23];F[28] = F[23];F[29] = F[23];
F[30] = F[31];F[31] = F[32];F[31] += F[3] * F[33];F[31] += F[5] * F[34];F[31] += F[7] * F[35];F[31] += F[9] * F[36];F[31] += F[11] * F[37];F[31] += F[13] * F[38];F[31] += F[15] * F[39];F[31] += F[17] * F[40];F[31] += F[19] * F[41];F[31] += F[21] * F[42];F[43] = (1 / (1 + exp(-F[31])));F[44] = F[43] * (1 - F[43]);F[45] = F[43];F[46] = F[43];F[47] = F[43];F[48] = F[43];F[49] = F[43];
F[50] = F[51];F[51] = F[52];F[51] += F[3] * F[53];F[51] += F[5] * F[54];F[51] += F[7] * F[55];F[51] += F[9] * F[56];F[51] += F[11] * F[57];F[51] += F[13] * F[58];F[51] += F[15] * F[59];F[51] += F[17] * F[60];F[51] += F[19] * F[61];F[51] += F[21] * F[62];F[63] = (1 / (1 + exp(-F[51])));F[64] = F[63] * (1 - F[63]);F[65] = F[63];F[66] = F[63];F[67] = F[63];F[68] = F[63];F[69] = F[63];
F[70] = F[71];F[71] = F[72];F[71] += F[3] * F[73];F[71] += F[5] * F[74];F[71] += F[7] * F[75];F[71] += F[9] * F[76];F[71] += F[11] * F[77];F[71] += F[13] * F[78];F[71] += F[15] * F[79];F[71] += F[17] * F[80];F[71] += F[19] * F[81];F[71] += F[21] * F[82];F[83] = (1 / (1 + exp(-F[71])));F[84] = F[83] * (1 - F[83]);F[85] = F[83];F[86] = F[83];F[87] = F[83];F[88] = F[83];F[89] = F[83];
F[90] = F[91];F[91] = F[92];F[91] += F[3] * F[93];F[91] += F[5] * F[94];F[91] += F[7] * F[95];F[91] += F[9] * F[96];F[91] += F[11] * F[97];F[91] += F[13] * F[98];F[91] += F[15] * F[99];F[91] += F[17] * F[100];F[91] += F[19] * F[101];F[91] += F[21] * F[102];F[103] = (1 / (1 + exp(-F[91])));F[104] = F[103] * (1 - F[103]);F[105] = F[103];F[106] = F[103];F[107] = F[103];F[108] = F[103];F[109] = F[103];
F[110] = F[111];F[111] = F[112];F[111] += F[3] * F[113];F[111] += F[5] * F[114];F[111] += F[7] * F[115];F[111] += F[9] * F[116];F[111] += F[11] * F[117];F[111] += F[13] * F[118];F[111] += F[15] * F[119];F[111] += F[17] * F[120];F[111] += F[19] * F[121];F[111] += F[21] * F[122];F[123] = (1 / (1 + exp(-F[111])));F[124] = F[123] * (1 - F[123]);F[125] = F[123];
F[126] = F[127];F[127] = F[128];F[127] += F[3] * F[129];F[127] += F[5] * F[130];F[127] += F[7] * F[131];F[127] += F[9] * F[132];F[127] += F[11] * F[133];F[127] += F[13] * F[134];F[127] += F[15] * F[135];F[127] += F[17] * F[136];F[127] += F[19] * F[137];F[127] += F[21] * F[138];F[139] = (1 / (1 + exp(-F[127])));F[140] = F[139] * (1 - F[139]);F[141] = F[139];
F[142] = F[143];F[143] = F[144];F[143] += F[3] * F[145];F[143] += F[5] * F[146];F[143] += F[7] * F[147];F[143] += F[9] * F[148];F[143] += F[11] * F[149];F[143] += F[13] * F[150];F[143] += F[15] * F[151];F[143] += F[17] * F[152];F[143] += F[19] * F[153];F[143] += F[21] * F[154];F[155] = (1 / (1 + exp(-F[143])));F[156] = F[155] * (1 - F[155]);F[157] = F[155];
F[158] = F[159];F[159] = F[160];F[159] += F[3] * F[161];F[159] += F[5] * F[162];F[159] += F[7] * F[163];F[159] += F[9] * F[164];F[159] += F[11] * F[165];F[159] += F[13] * F[166];F[159] += F[15] * F[167];F[159] += F[17] * F[168];F[159] += F[19] * F[169];F[159] += F[21] * F[170];F[171] = (1 / (1 + exp(-F[159])));F[172] = F[171] * (1 - F[171]);F[173] = F[171];
F[174] = F[175];F[175] = F[176];F[175] += F[3] * F[177];F[175] += F[5] * F[178];F[175] += F[7] * F[179];F[175] += F[9] * F[180];F[175] += F[11] * F[181];F[175] += F[13] * F[182];F[175] += F[15] * F[183];F[175] += F[17] * F[184];F[175] += F[19] * F[185];F[175] += F[21] * F[186];F[187] = (1 / (1 + exp(-F[175])));F[188] = F[187] * (1 - F[187]);F[189] = F[187];
F[190] = F[191];F[191] = F[125] * F[192] * F[191] + F[193];F[191] += F[3] * F[194] * F[25];F[191] += F[5] * F[195] * F[26];F[191] += F[7] * F[196] * F[27];F[191] += F[9] * F[197] * F[28];F[191] += F[11] * F[198] * F[29];F[13] = (1 / (1 + exp(-F[191])));F[199] = F[13] * (1 - F[13]);
F[200] = F[201];F[201] = F[141] * F[202] * F[201] + F[203];F[201] += F[3] * F[204] * F[45];F[201] += F[5] * F[205] * F[46];F[201] += F[7] * F[206] * F[47];F[201] += F[9] * F[207] * F[48];F[201] += F[11] * F[208] * F[49];F[15] = (1 / (1 + exp(-F[201])));F[209] = F[15] * (1 - F[15]);
F[210] = F[211];F[211] = F[157] * F[212] * F[211] + F[213];F[211] += F[3] * F[214] * F[65];F[211] += F[5] * F[215] * F[66];F[211] += F[7] * F[216] * F[67];F[211] += F[9] * F[217] * F[68];F[211] += F[11] * F[218] * F[69];F[17] = (1 / (1 + exp(-F[211])));F[219] = F[17] * (1 - F[17]);
F[220] = F[221];F[221] = F[173] * F[222] * F[221] + F[223];F[221] += F[3] * F[224] * F[85];F[221] += F[5] * F[225] * F[86];F[221] += F[7] * F[226] * F[87];F[221] += F[9] * F[227] * F[88];F[221] += F[11] * F[228] * F[89];F[19] = (1 / (1 + exp(-F[221])));F[229] = F[19] * (1 - F[19]);
F[230] = F[231];F[231] = F[189] * F[232] * F[231] + F[233];F[231] += F[3] * F[234] * F[105];F[231] += F[5] * F[235] * F[106];F[231] += F[7] * F[236] * F[107];F[231] += F[9] * F[237] * F[108];F[231] += F[11] * F[238] * F[109];F[21] = (1 / (1 + exp(-F[231])));F[239] = F[21] * (1 - F[21]);
F[240] = F[241];F[241] = F[242];F[241] += F[3] * F[243];F[241] += F[5] * F[244];F[241] += F[7] * F[245];F[241] += F[9] * F[246];F[241] += F[11] * F[247];F[241] += F[13] * F[248];F[241] += F[15] * F[249];F[241] += F[17] * F[250];F[241] += F[19] * F[251];F[241] += F[21] * F[252];F[253] = (1 / (1 + exp(-F[241])));F[254] = F[253] * (1 - F[253]);F[255] = F[253];
F[256] = F[257];F[257] = F[258];F[257] += F[3] * F[259];F[257] += F[5] * F[260];F[257] += F[7] * F[261];F[257] += F[9] * F[262];F[257] += F[11] * F[263];F[257] += F[13] * F[264];F[257] += F[15] * F[265];F[257] += F[17] * F[266];F[257] += F[19] * F[267];F[257] += F[21] * F[268];F[269] = (1 / (1 + exp(-F[257])));F[270] = F[269] * (1 - F[269]);F[271] = F[269];
F[272] = F[273];F[273] = F[274];F[273] += F[3] * F[275];F[273] += F[5] * F[276];F[273] += F[7] * F[277];F[273] += F[9] * F[278];F[273] += F[11] * F[279];F[273] += F[13] * F[280];F[273] += F[15] * F[281];F[273] += F[17] * F[282];F[273] += F[19] * F[283];F[273] += F[21] * F[284];F[285] = (1 / (1 + exp(-F[273])));F[286] = F[285] * (1 - F[285]);F[287] = F[285];
F[288] = F[289];F[289] = F[290];F[289] += F[3] * F[291];F[289] += F[5] * F[292];F[289] += F[7] * F[293];F[289] += F[9] * F[294];F[289] += F[11] * F[295];F[289] += F[13] * F[296];F[289] += F[15] * F[297];F[289] += F[17] * F[298];F[289] += F[19] * F[299];F[289] += F[21] * F[300];F[301] = (1 / (1 + exp(-F[289])));F[302] = F[301] * (1 - F[301]);F[303] = F[301];
F[304] = F[305];F[305] = F[306];F[305] += F[3] * F[307];F[305] += F[5] * F[308];F[305] += F[7] * F[309];F[305] += F[9] * F[310];F[305] += F[11] * F[311];F[305] += F[13] * F[312];F[305] += F[15] * F[313];F[305] += F[17] * F[314];F[305] += F[19] * F[315];F[305] += F[21] * F[316];F[317] = (1 / (1 + exp(-F[305])));F[318] = F[317] * (1 - F[317]);F[319] = F[317];
F[320] = F[321];F[321] = F[322];F[321] += F[3] * F[323];F[321] += F[5] * F[324];F[321] += F[7] * F[325];F[321] += F[9] * F[326];F[321] += F[11] * F[327];F[321] += F[13] * F[328];F[321] += F[15] * F[329];F[321] += F[17] * F[330];F[321] += F[19] * F[331];F[321] += F[21] * F[332];F[321] += F[333] * F[334];F[321] += F[335] * F[336];F[337] = (1 / (1 + exp(-F[321])));F[338] = F[337] * (1 - F[337]);F[339] = F[337];F[340] = F[337];F[341] = F[337];F[342] = F[337];F[343] = F[337];F[344] = F[337];F[345] = F[337];F[346] = F[337];F[347] = F[337];F[348] = F[337];
F[349] = F[350];F[350] = F[351];F[350] += F[3] * F[352];F[350] += F[5] * F[353];F[350] += F[7] * F[354];F[350] += F[9] * F[355];F[350] += F[11] * F[356];F[350] += F[13] * F[357];F[350] += F[15] * F[358];F[350] += F[17] * F[359];F[350] += F[19] * F[360];F[350] += F[21] * F[361];F[350] += F[333] * F[362];F[350] += F[335] * F[363];F[364] = (1 / (1 + exp(-F[350])));F[365] = F[364] * (1 - F[364]);F[366] = F[364];F[367] = F[364];F[368] = F[364];F[369] = F[364];F[370] = F[364];F[371] = F[364];F[372] = F[364];F[373] = F[364];F[374] = F[364];F[375] = F[364];
F[376] = F[377];F[377] = F[378];F[377] += F[3] * F[379];F[377] += F[5] * F[380];F[377] += F[7] * F[381];F[377] += F[9] * F[382];F[377] += F[11] * F[383];F[377] += F[13] * F[384];F[377] += F[15] * F[385];F[377] += F[17] * F[386];F[377] += F[19] * F[387];F[377] += F[21] * F[388];F[377] += F[333] * F[389];F[377] += F[335] * F[390];F[391] = (1 / (1 + exp(-F[377])));F[392] = F[391] * (1 - F[391]);F[393] = F[391];
F[394] = F[395];F[395] = F[396];F[395] += F[3] * F[397];F[395] += F[5] * F[398];F[395] += F[7] * F[399];F[395] += F[9] * F[400];F[395] += F[11] * F[401];F[395] += F[13] * F[402];F[395] += F[15] * F[403];F[395] += F[17] * F[404];F[395] += F[19] * F[405];F[395] += F[21] * F[406];F[395] += F[333] * F[407];F[395] += F[335] * F[408];F[409] = (1 / (1 + exp(-F[395])));F[410] = F[409] * (1 - F[409]);F[411] = F[409];
F[412] = F[413];F[413] = F[393] * F[414] * F[413] + F[415];F[413] += F[3] * F[416] * F[339];F[413] += F[5] * F[417] * F[340];F[413] += F[7] * F[418] * F[341];F[413] += F[9] * F[419] * F[342];F[413] += F[11] * F[420] * F[343];F[413] += F[13] * F[421] * F[344];F[413] += F[15] * F[422] * F[345];F[413] += F[17] * F[423] * F[346];F[413] += F[19] * F[424] * F[347];F[413] += F[21] * F[425] * F[348];F[333] = (1 / (1 + exp(-F[413])));F[426] = F[333] * (1 - F[333]);
F[427] = F[428];F[428] = F[411] * F[429] * F[428] + F[430];F[428] += F[3] * F[431] * F[366];F[428] += F[5] * F[432] * F[367];F[428] += F[7] * F[433] * F[368];F[428] += F[9] * F[434] * F[369];F[428] += F[11] * F[435] * F[370];F[428] += F[13] * F[436] * F[371];F[428] += F[15] * F[437] * F[372];F[428] += F[17] * F[438] * F[373];F[428] += F[19] * F[439] * F[374];F[428] += F[21] * F[440] * F[375];F[335] = (1 / (1 + exp(-F[428])));F[441] = F[335] * (1 - F[335]);
F[442] = F[443];F[443] = F[444];F[443] += F[3] * F[445];F[443] += F[5] * F[446];F[443] += F[7] * F[447];F[443] += F[9] * F[448];F[443] += F[11] * F[449];F[443] += F[13] * F[450];F[443] += F[15] * F[451];F[443] += F[17] * F[452];F[443] += F[19] * F[453];F[443] += F[21] * F[454];F[443] += F[333] * F[455];F[443] += F[335] * F[456];F[457] = (1 / (1 + exp(-F[443])));F[458] = F[457] * (1 - F[457]);F[459] = F[457];
F[460] = F[461];F[461] = F[462];F[461] += F[3] * F[463];F[461] += F[5] * F[464];F[461] += F[7] * F[465];F[461] += F[9] * F[466];F[461] += F[11] * F[467];F[461] += F[13] * F[468];F[461] += F[15] * F[469];F[461] += F[17] * F[470];F[461] += F[19] * F[471];F[461] += F[21] * F[472];F[461] += F[333] * F[473];F[461] += F[335] * F[474];F[475] = (1 / (1 + exp(-F[461])));F[476] = F[475] * (1 - F[475]);F[477] = F[475];
F[478] = F[479];F[479] = F[480];F[479] += F[13] * F[481] * F[255];F[479] += F[15] * F[482] * F[271];F[479] += F[17] * F[483] * F[287];F[479] += F[19] * F[484] * F[303];F[479] += F[21] * F[485] * F[319];F[479] += F[333] * F[486] * F[459];F[479] += F[335] * F[487] * F[477];F[479] += F[3] * F[488];F[479] += F[5] * F[489];F[479] += F[7] * F[490];F[479] += F[9] * F[491];F[479] += F[11] * F[492];F[493] = (1 / (1 + exp(-F[479])));F[494] = F[493] * (1 - F[493]);

float output = F[493] * 100;
return output;
}


/**************************************************************************/
/*!
    @brief apply data to 7 INPUT WITH ANGULAR POSITION neural network activation function and apply threshold to neural network output
*/
/**************************************************************************/
float tingleNeuralNetwork::detect(float t1, float t2, float t3, float  t4, float distance, float pitch, float roll) { 

    float F[575];
//  F = _F1_7in;

   //  copy 'len' elements from 'src' to 'dst'
  //  memcpy(dst, src, sizeof(src[0])*len);
    memcpy(F, _F1_7in, sizeof(_F1_7in[0])*575);
  
  F[3] = t1;
  F[5] = t2;
  F[7] = t3;
  F[9] = t4;
  F[11] = distance;
  F[13] = pitch;
  F[15] = roll;
  
  F[0] = F[1];F[1] = F[2];F[1] += F[3] * F[4];F[1] += F[5] * F[6];F[1] += F[7] * F[8];F[1] += F[9] * F[10];F[1] += F[11] * F[12];F[1] += F[13] * F[14];F[1] += F[15] * F[16];F[1] += F[17] * F[18];F[1] += F[19] * F[20];F[1] += F[21] * F[22];F[1] += F[23] * F[24];F[1] += F[25] * F[26];F[27] = (1 / (1 + exp(-F[1])));F[28] = F[27] * (1 - F[27]);F[29] = F[27];F[30] = F[27];F[31] = F[27];F[32] = F[27];F[33] = F[27];F[34] = F[27];F[35] = F[27];
  F[36] = F[37];F[37] = F[38];F[37] += F[3] * F[39];F[37] += F[5] * F[40];F[37] += F[7] * F[41];F[37] += F[9] * F[42];F[37] += F[11] * F[43];F[37] += F[13] * F[44];F[37] += F[15] * F[45];F[37] += F[17] * F[46];F[37] += F[19] * F[47];F[37] += F[21] * F[48];F[37] += F[23] * F[49];F[37] += F[25] * F[50];F[51] = (1 / (1 + exp(-F[37])));F[52] = F[51] * (1 - F[51]);F[53] = F[51];F[54] = F[51];F[55] = F[51];F[56] = F[51];F[57] = F[51];F[58] = F[51];F[59] = F[51];
  F[60] = F[61];F[61] = F[62];F[61] += F[3] * F[63];F[61] += F[5] * F[64];F[61] += F[7] * F[65];F[61] += F[9] * F[66];F[61] += F[11] * F[67];F[61] += F[13] * F[68];F[61] += F[15] * F[69];F[61] += F[17] * F[70];F[61] += F[19] * F[71];F[61] += F[21] * F[72];F[61] += F[23] * F[73];F[61] += F[25] * F[74];F[75] = (1 / (1 + exp(-F[61])));F[76] = F[75] * (1 - F[75]);F[77] = F[75];F[78] = F[75];F[79] = F[75];F[80] = F[75];F[81] = F[75];F[82] = F[75];F[83] = F[75];
  F[84] = F[85];F[85] = F[86];F[85] += F[3] * F[87];F[85] += F[5] * F[88];F[85] += F[7] * F[89];F[85] += F[9] * F[90];F[85] += F[11] * F[91];F[85] += F[13] * F[92];F[85] += F[15] * F[93];F[85] += F[17] * F[94];F[85] += F[19] * F[95];F[85] += F[21] * F[96];F[85] += F[23] * F[97];F[85] += F[25] * F[98];F[99] = (1 / (1 + exp(-F[85])));F[100] = F[99] * (1 - F[99]);F[101] = F[99];F[102] = F[99];F[103] = F[99];F[104] = F[99];F[105] = F[99];F[106] = F[99];F[107] = F[99];
  F[108] = F[109];F[109] = F[110];F[109] += F[3] * F[111];F[109] += F[5] * F[112];F[109] += F[7] * F[113];F[109] += F[9] * F[114];F[109] += F[11] * F[115];F[109] += F[13] * F[116];F[109] += F[15] * F[117];F[109] += F[17] * F[118];F[109] += F[19] * F[119];F[109] += F[21] * F[120];F[109] += F[23] * F[121];F[109] += F[25] * F[122];F[123] = (1 / (1 + exp(-F[109])));F[124] = F[123] * (1 - F[123]);F[125] = F[123];F[126] = F[123];F[127] = F[123];F[128] = F[123];F[129] = F[123];F[130] = F[123];F[131] = F[123];
  F[132] = F[133];F[133] = F[134];F[133] += F[3] * F[135];F[133] += F[5] * F[136];F[133] += F[7] * F[137];F[133] += F[9] * F[138];F[133] += F[11] * F[139];F[133] += F[13] * F[140];F[133] += F[15] * F[141];F[133] += F[17] * F[142];F[133] += F[19] * F[143];F[133] += F[21] * F[144];F[133] += F[23] * F[145];F[133] += F[25] * F[146];F[147] = (1 / (1 + exp(-F[133])));F[148] = F[147] * (1 - F[147]);F[149] = F[147];
  F[150] = F[151];F[151] = F[152];F[151] += F[3] * F[153];F[151] += F[5] * F[154];F[151] += F[7] * F[155];F[151] += F[9] * F[156];F[151] += F[11] * F[157];F[151] += F[13] * F[158];F[151] += F[15] * F[159];F[151] += F[17] * F[160];F[151] += F[19] * F[161];F[151] += F[21] * F[162];F[151] += F[23] * F[163];F[151] += F[25] * F[164];F[165] = (1 / (1 + exp(-F[151])));F[166] = F[165] * (1 - F[165]);F[167] = F[165];
  F[168] = F[169];F[169] = F[170];F[169] += F[3] * F[171];F[169] += F[5] * F[172];F[169] += F[7] * F[173];F[169] += F[9] * F[174];F[169] += F[11] * F[175];F[169] += F[13] * F[176];F[169] += F[15] * F[177];F[169] += F[17] * F[178];F[169] += F[19] * F[179];F[169] += F[21] * F[180];F[169] += F[23] * F[181];F[169] += F[25] * F[182];F[183] = (1 / (1 + exp(-F[169])));F[184] = F[183] * (1 - F[183]);F[185] = F[183];
  F[186] = F[187];F[187] = F[188];F[187] += F[3] * F[189];F[187] += F[5] * F[190];F[187] += F[7] * F[191];F[187] += F[9] * F[192];F[187] += F[11] * F[193];F[187] += F[13] * F[194];F[187] += F[15] * F[195];F[187] += F[17] * F[196];F[187] += F[19] * F[197];F[187] += F[21] * F[198];F[187] += F[23] * F[199];F[187] += F[25] * F[200];F[201] = (1 / (1 + exp(-F[187])));F[202] = F[201] * (1 - F[201]);F[203] = F[201];
  F[204] = F[205];F[205] = F[206];F[205] += F[3] * F[207];F[205] += F[5] * F[208];F[205] += F[7] * F[209];F[205] += F[9] * F[210];F[205] += F[11] * F[211];F[205] += F[13] * F[212];F[205] += F[15] * F[213];F[205] += F[17] * F[214];F[205] += F[19] * F[215];F[205] += F[21] * F[216];F[205] += F[23] * F[217];F[205] += F[25] * F[218];F[219] = (1 / (1 + exp(-F[205])));F[220] = F[219] * (1 - F[219]);F[221] = F[219];
  F[222] = F[223];F[223] = F[149] * F[224] * F[223] + F[225];F[223] += F[3] * F[226] * F[29];F[223] += F[5] * F[227] * F[30];F[223] += F[7] * F[228] * F[31];F[223] += F[9] * F[229] * F[32];F[223] += F[11] * F[230] * F[33];F[223] += F[13] * F[231] * F[34];F[223] += F[15] * F[232] * F[35];F[17] = (1 / (1 + exp(-F[223])));F[233] = F[17] * (1 - F[17]);
  F[234] = F[235];F[235] = F[167] * F[236] * F[235] + F[237];F[235] += F[3] * F[238] * F[53];F[235] += F[5] * F[239] * F[54];F[235] += F[7] * F[240] * F[55];F[235] += F[9] * F[241] * F[56];F[235] += F[11] * F[242] * F[57];F[235] += F[13] * F[243] * F[58];F[235] += F[15] * F[244] * F[59];F[19] = (1 / (1 + exp(-F[235])));F[245] = F[19] * (1 - F[19]);
  F[246] = F[247];F[247] = F[185] * F[248] * F[247] + F[249];F[247] += F[3] * F[250] * F[77];F[247] += F[5] * F[251] * F[78];F[247] += F[7] * F[252] * F[79];F[247] += F[9] * F[253] * F[80];F[247] += F[11] * F[254] * F[81];F[247] += F[13] * F[255] * F[82];F[247] += F[15] * F[256] * F[83];F[21] = (1 / (1 + exp(-F[247])));F[257] = F[21] * (1 - F[21]);
  F[258] = F[259];F[259] = F[203] * F[260] * F[259] + F[261];F[259] += F[3] * F[262] * F[101];F[259] += F[5] * F[263] * F[102];F[259] += F[7] * F[264] * F[103];F[259] += F[9] * F[265] * F[104];F[259] += F[11] * F[266] * F[105];F[259] += F[13] * F[267] * F[106];F[259] += F[15] * F[268] * F[107];F[23] = (1 / (1 + exp(-F[259])));F[269] = F[23] * (1 - F[23]);
  F[270] = F[271];F[271] = F[221] * F[272] * F[271] + F[273];F[271] += F[3] * F[274] * F[125];F[271] += F[5] * F[275] * F[126];F[271] += F[7] * F[276] * F[127];F[271] += F[9] * F[277] * F[128];F[271] += F[11] * F[278] * F[129];F[271] += F[13] * F[279] * F[130];F[271] += F[15] * F[280] * F[131];F[25] = (1 / (1 + exp(-F[271])));F[281] = F[25] * (1 - F[25]);
  F[282] = F[283];F[283] = F[284];F[283] += F[3] * F[285];F[283] += F[5] * F[286];F[283] += F[7] * F[287];F[283] += F[9] * F[288];F[283] += F[11] * F[289];F[283] += F[13] * F[290];F[283] += F[15] * F[291];F[283] += F[17] * F[292];F[283] += F[19] * F[293];F[283] += F[21] * F[294];F[283] += F[23] * F[295];F[283] += F[25] * F[296];F[297] = (1 / (1 + exp(-F[283])));F[298] = F[297] * (1 - F[297]);F[299] = F[297];
  F[300] = F[301];F[301] = F[302];F[301] += F[3] * F[303];F[301] += F[5] * F[304];F[301] += F[7] * F[305];F[301] += F[9] * F[306];F[301] += F[11] * F[307];F[301] += F[13] * F[308];F[301] += F[15] * F[309];F[301] += F[17] * F[310];F[301] += F[19] * F[311];F[301] += F[21] * F[312];F[301] += F[23] * F[313];F[301] += F[25] * F[314];F[315] = (1 / (1 + exp(-F[301])));F[316] = F[315] * (1 - F[315]);F[317] = F[315];
  F[318] = F[319];F[319] = F[320];F[319] += F[3] * F[321];F[319] += F[5] * F[322];F[319] += F[7] * F[323];F[319] += F[9] * F[324];F[319] += F[11] * F[325];F[319] += F[13] * F[326];F[319] += F[15] * F[327];F[319] += F[17] * F[328];F[319] += F[19] * F[329];F[319] += F[21] * F[330];F[319] += F[23] * F[331];F[319] += F[25] * F[332];F[333] = (1 / (1 + exp(-F[319])));F[334] = F[333] * (1 - F[333]);F[335] = F[333];
  F[336] = F[337];F[337] = F[338];F[337] += F[3] * F[339];F[337] += F[5] * F[340];F[337] += F[7] * F[341];F[337] += F[9] * F[342];F[337] += F[11] * F[343];F[337] += F[13] * F[344];F[337] += F[15] * F[345];F[337] += F[17] * F[346];F[337] += F[19] * F[347];F[337] += F[21] * F[348];F[337] += F[23] * F[349];F[337] += F[25] * F[350];F[351] = (1 / (1 + exp(-F[337])));F[352] = F[351] * (1 - F[351]);F[353] = F[351];
  F[354] = F[355];F[355] = F[356];F[355] += F[3] * F[357];F[355] += F[5] * F[358];F[355] += F[7] * F[359];F[355] += F[9] * F[360];F[355] += F[11] * F[361];F[355] += F[13] * F[362];F[355] += F[15] * F[363];F[355] += F[17] * F[364];F[355] += F[19] * F[365];F[355] += F[21] * F[366];F[355] += F[23] * F[367];F[355] += F[25] * F[368];F[369] = (1 / (1 + exp(-F[355])));F[370] = F[369] * (1 - F[369]);F[371] = F[369];
  F[372] = F[373];F[373] = F[374];F[373] += F[3] * F[375];F[373] += F[5] * F[376];F[373] += F[7] * F[377];F[373] += F[9] * F[378];F[373] += F[11] * F[379];F[373] += F[13] * F[380];F[373] += F[15] * F[381];F[373] += F[17] * F[382];F[373] += F[19] * F[383];F[373] += F[21] * F[384];F[373] += F[23] * F[385];F[373] += F[25] * F[386];F[373] += F[387] * F[388];F[373] += F[389] * F[390];F[391] = (1 / (1 + exp(-F[373])));F[392] = F[391] * (1 - F[391]);F[393] = F[391];F[394] = F[391];F[395] = F[391];F[396] = F[391];F[397] = F[391];F[398] = F[391];F[399] = F[391];F[400] = F[391];F[401] = F[391];F[402] = F[391];F[403] = F[391];F[404] = F[391];
  F[405] = F[406];F[406] = F[407];F[406] += F[3] * F[408];F[406] += F[5] * F[409];F[406] += F[7] * F[410];F[406] += F[9] * F[411];F[406] += F[11] * F[412];F[406] += F[13] * F[413];F[406] += F[15] * F[414];F[406] += F[17] * F[415];F[406] += F[19] * F[416];F[406] += F[21] * F[417];F[406] += F[23] * F[418];F[406] += F[25] * F[419];F[406] += F[387] * F[420];F[406] += F[389] * F[421];F[422] = (1 / (1 + exp(-F[406])));F[423] = F[422] * (1 - F[422]);F[424] = F[422];F[425] = F[422];F[426] = F[422];F[427] = F[422];F[428] = F[422];F[429] = F[422];F[430] = F[422];F[431] = F[422];F[432] = F[422];F[433] = F[422];F[434] = F[422];F[435] = F[422];
  F[436] = F[437];F[437] = F[438];F[437] += F[3] * F[439];F[437] += F[5] * F[440];F[437] += F[7] * F[441];F[437] += F[9] * F[442];F[437] += F[11] * F[443];F[437] += F[13] * F[444];F[437] += F[15] * F[445];F[437] += F[17] * F[446];F[437] += F[19] * F[447];F[437] += F[21] * F[448];F[437] += F[23] * F[449];F[437] += F[25] * F[450];F[437] += F[387] * F[451];F[437] += F[389] * F[452];F[453] = (1 / (1 + exp(-F[437])));F[454] = F[453] * (1 - F[453]);F[455] = F[453];
  F[456] = F[457];F[457] = F[458];F[457] += F[3] * F[459];F[457] += F[5] * F[460];F[457] += F[7] * F[461];F[457] += F[9] * F[462];F[457] += F[11] * F[463];F[457] += F[13] * F[464];F[457] += F[15] * F[465];F[457] += F[17] * F[466];F[457] += F[19] * F[467];F[457] += F[21] * F[468];F[457] += F[23] * F[469];F[457] += F[25] * F[470];F[457] += F[387] * F[471];F[457] += F[389] * F[472];F[473] = (1 / (1 + exp(-F[457])));F[474] = F[473] * (1 - F[473]);F[475] = F[473];
  F[476] = F[477];F[477] = F[455] * F[478] * F[477] + F[479];F[477] += F[3] * F[480] * F[393];F[477] += F[5] * F[481] * F[394];F[477] += F[7] * F[482] * F[395];F[477] += F[9] * F[483] * F[396];F[477] += F[11] * F[484] * F[397];F[477] += F[13] * F[485] * F[398];F[477] += F[15] * F[486] * F[399];F[477] += F[17] * F[487] * F[400];F[477] += F[19] * F[488] * F[401];F[477] += F[21] * F[489] * F[402];F[477] += F[23] * F[490] * F[403];F[477] += F[25] * F[491] * F[404];F[387] = (1 / (1 + exp(-F[477])));F[492] = F[387] * (1 - F[387]);
  F[493] = F[494];F[494] = F[475] * F[495] * F[494] + F[496];F[494] += F[3] * F[497] * F[424];F[494] += F[5] * F[498] * F[425];F[494] += F[7] * F[499] * F[426];F[494] += F[9] * F[500] * F[427];F[494] += F[11] * F[501] * F[428];F[494] += F[13] * F[502] * F[429];F[494] += F[15] * F[503] * F[430];F[494] += F[17] * F[504] * F[431];F[494] += F[19] * F[505] * F[432];F[494] += F[21] * F[506] * F[433];F[494] += F[23] * F[507] * F[434];F[494] += F[25] * F[508] * F[435];F[389] = (1 / (1 + exp(-F[494])));F[509] = F[389] * (1 - F[389]);
  F[510] = F[511];F[511] = F[512];F[511] += F[3] * F[513];F[511] += F[5] * F[514];F[511] += F[7] * F[515];F[511] += F[9] * F[516];F[511] += F[11] * F[517];F[511] += F[13] * F[518];F[511] += F[15] * F[519];F[511] += F[17] * F[520];F[511] += F[19] * F[521];F[511] += F[21] * F[522];F[511] += F[23] * F[523];F[511] += F[25] * F[524];F[511] += F[387] * F[525];F[511] += F[389] * F[526];F[527] = (1 / (1 + exp(-F[511])));F[528] = F[527] * (1 - F[527]);F[529] = F[527];
  F[530] = F[531];F[531] = F[532];F[531] += F[3] * F[533];F[531] += F[5] * F[534];F[531] += F[7] * F[535];F[531] += F[9] * F[536];F[531] += F[11] * F[537];F[531] += F[13] * F[538];F[531] += F[15] * F[539];F[531] += F[17] * F[540];F[531] += F[19] * F[541];F[531] += F[21] * F[542];F[531] += F[23] * F[543];F[531] += F[25] * F[544];F[531] += F[387] * F[545];F[531] += F[389] * F[546];F[547] = (1 / (1 + exp(-F[531])));F[548] = F[547] * (1 - F[547]);F[549] = F[547];
  F[550] = F[551];F[551] = F[552];F[551] += F[17] * F[553] * F[299];F[551] += F[19] * F[554] * F[317];F[551] += F[21] * F[555] * F[335];F[551] += F[23] * F[556] * F[353];F[551] += F[25] * F[557] * F[371];F[551] += F[387] * F[558] * F[529];F[551] += F[389] * F[559] * F[549];F[551] += F[3] * F[560];F[551] += F[5] * F[561];F[551] += F[7] * F[562];F[551] += F[9] * F[563];F[551] += F[11] * F[564];F[551] += F[13] * F[565];F[551] += F[15] * F[566];F[567] = (1 / (1 + exp(-F[551])));F[568] = F[567] * (1 - F[567]);
  float output = F[567] * 100;
  return output;
}


/**************************************************************************/
/*!
    @brief set single target gesture
*/
/**************************************************************************/
void tingleNeuralNetwork::selectTarget(tingleNN_targets_t target) { 
    _target1 = target;
    _target2 = NO_TARGET;
    _numNetworks = 1;

      float _F1_5in_TEMP[500];
      float _F1_7in_TEMP[575];
 Serial.print("*******************SETUP NEW TARGET:\t"); Serial.println(target);

        switch(target) {
        case RHAND_SHORT_MOUTH:
            Serial.println("load x hand y gesture z inputs q hair length");
            set_RHand_ShortHair_MouthPosition_5Weights(_F1_5in); 
            set_RHand_ShortHair_MouthPosition_7Weights(_F1_7in);
            break;
        case RHAND_SHORT_EYES:
            Serial.println("load x hand y gesture z inputs q hair length");
            set_RHand_ShortHair_EyesPosition_5Weights(_F1_5in);
            set_RHand_ShortHair_EyesPosition_7Weights(_F1_7in);
            break;
        case RHAND_SHORT_FRONT:
            Serial.println("load x hand y gesture z inputs q hair length");
            set_RHand_ShortHair_FrontPosition_5Weights(_F1_5in); 
            set_RHand_ShortHair_FrontPosition_7Weights(_F1_7in);
            break;
        case RHAND_SHORT_SIDE:
            Serial.println("load x hand y gesture z inputs q hair length");
            set_RHand_ShortHair_SidePosition_5Weights(_F1_5in);
            set_RHand_ShortHair_SidePosition_7Weights(_F1_7in);
            break;
        case RHAND_SHORT_TOP:
            Serial.println("load x hand y gesture z inputs q hair length");
            set_RHand_ShortHair_TopPosition_5Weights(_F1_5in); 
            set_RHand_ShortHair_TopPosition_7Weights(_F1_7in);
            break;
        case RHAND_SHORT_BACK:
            Serial.println("load x hand y gesture z inputs q hair length");
            set_RHand_ShortHair_BackPosition_5Weights(_F1_5in);
            set_RHand_ShortHair_BackPosition_7Weights(_F1_7in);
            break;


        case LHAND_SHORT_MOUTH:
            Serial.println("load x hand y gesture z inputs q hair length");
            set_LHand_ShortHair_MouthPosition_5Weights(_F1_5in); 
            set_LHand_ShortHair_MouthPosition_7Weights(_F1_7in);
            break;
        case LHAND_SHORT_EYES:
            Serial.println("load x hand y gesture z inputs q hair length");
            set_LHand_ShortHair_EyesPosition_5Weights(_F1_5in);
            set_LHand_ShortHair_EyesPosition_7Weights(_F1_7in);
            break;
        case LHAND_SHORT_FRONT:
            Serial.println("load x hand y gesture z inputs q hair length");
            set_LHand_ShortHair_FrontPosition_5Weights(_F1_5in); 
            set_LHand_ShortHair_FrontPosition_7Weights(_F1_7in);
            break;
        case LHAND_SHORT_SIDE:
            Serial.println("load x hand y gesture z inputs q hair length");
            set_LHand_ShortHair_SidePosition_5Weights(_F1_5in);
            set_LHand_ShortHair_SidePosition_7Weights(_F1_7in);
            break;
        case LHAND_SHORT_TOP:
            Serial.println("load x hand y gesture z inputs q hair length");
            set_LHand_ShortHair_TopPosition_5Weights(_F1_5in); 
            set_LHand_ShortHair_TopPosition_7Weights(_F1_7in);
            break;
        case LHAND_SHORT_BACK:
            Serial.println("load x hand y gesture z inputs q hair length");
            set_LHand_ShortHair_BackPosition_5Weights(_F1_5in);
            set_LHand_ShortHair_BackPosition_7Weights(_F1_7in);
            break;
    }

    //  loadNetworkWeights(_target1, _F1_5in_TEMP, _F1_7in_TEMP); 


     //  copy 'len' elements from 'src' to 'dst'
  //  memcpy(dst, src, sizeof(src[0])*len);
  //  memcpy(_F1_5in, _F1_5in_TEMP, sizeof(_F1_5in_TEMP[0])*500);

        

   //  copy 'len' elements from 'src' to 'dst'
  //  memcpy(dst, src, sizeof(src[0])*len);
 //   memcpy(_F1_7in, _F1_7in_TEMP, sizeof(_F1_7in_TEMP[0])*575);

    Serial.print("*******************SETUP Weights:\t");
for(int i = 0; i < 500; i++){Serial.print(_F1_5in[i]); Serial.print(" ");}
Serial.println(" ");
delay(5000);
    
}


/**************************************************************************/
/*!
    @brief set two target gestures
*/
/**************************************************************************/
void tingleNeuralNetwork::selectTarget(tingleNN_targets_t target1 , tingleNN_targets_t target2) { 
    _target1 = target1;
    _target2 = target2;
    _numNetworks = 2;
}


/**************************************************************************/
/*!
    @brief set sensitivity ie detection threshold
*/
/**************************************************************************/
void tingleNeuralNetwork::sensitivity(tingleNN_sensitivity_t sense) {  
    _threshold = sense;
}


/**************************************************************************/
/*!
    @brief lookup neural network weights using specified target
*/
/**************************************************************************/
void tingleNeuralNetwork::loadNetworkWeights(tingleNN_targets_t target, float F_5in[], float F_7in[]) { 
    switch(target) {
        case RHAND_SHORT_MOUTH:
            Serial.println("load x hand y gesture z inputs q hair length");
            set_RHand_ShortHair_MouthPosition_5Weights(_F1_5in); 
            set_RHand_ShortHair_MouthPosition_7Weights(_F1_7in);
            break;
        case RHAND_SHORT_EYES:
            Serial.println("load x hand y gesture z inputs q hair length");
            set_RHand_ShortHair_EyesPosition_5Weights(_F1_5in);
            set_RHand_ShortHair_EyesPosition_7Weights(_F1_7in);
            break;
        case RHAND_SHORT_FRONT:
            Serial.println("load x hand y gesture z inputs q hair length");
            set_RHand_ShortHair_FrontPosition_5Weights(_F1_5in); 
            set_RHand_ShortHair_FrontPosition_7Weights(_F1_7in);
            break;
        case RHAND_SHORT_SIDE:
            Serial.println("load x hand y gesture z inputs q hair length");
            set_RHand_ShortHair_SidePosition_5Weights(_F1_5in);
            set_RHand_ShortHair_SidePosition_7Weights(_F1_7in);
            break;
        case RHAND_SHORT_TOP:
            Serial.println("load x hand y gesture z inputs q hair length");
            set_RHand_ShortHair_TopPosition_5Weights(_F1_5in); 
            set_RHand_ShortHair_TopPosition_7Weights(_F1_7in);
            break;
        case RHAND_SHORT_BACK:
            Serial.println("load x hand y gesture z inputs q hair length");
            set_RHand_ShortHair_BackPosition_5Weights(_F1_5in);
            set_RHand_ShortHair_BackPosition_7Weights(_F1_7in);
            break;


        case LHAND_SHORT_MOUTH:
            Serial.println("load x hand y gesture z inputs q hair length");
            set_LHand_ShortHair_MouthPosition_5Weights(_F1_5in); 
            set_LHand_ShortHair_MouthPosition_7Weights(_F1_7in);
            break;
        case LHAND_SHORT_EYES:
            Serial.println("load x hand y gesture z inputs q hair length");
            set_LHand_ShortHair_EyesPosition_5Weights(_F1_5in);
            set_LHand_ShortHair_EyesPosition_7Weights(_F1_7in);
            break;
        case LHAND_SHORT_FRONT:
            Serial.println("load x hand y gesture z inputs q hair length");
            set_LHand_ShortHair_FrontPosition_5Weights(_F1_5in); 
            set_LHand_ShortHair_FrontPosition_7Weights(_F1_7in);
            break;
        case LHAND_SHORT_SIDE:
            Serial.println("load x hand y gesture z inputs q hair length");
            set_LHand_ShortHair_SidePosition_5Weights(_F1_5in);
            set_LHand_ShortHair_SidePosition_7Weights(_F1_7in);
            break;
        case LHAND_SHORT_TOP:
            Serial.println("load x hand y gesture z inputs q hair length");
            set_LHand_ShortHair_TopPosition_5Weights(_F1_5in); 
            set_LHand_ShortHair_TopPosition_7Weights(_F1_7in);
            break;
        case LHAND_SHORT_BACK:
            Serial.println("load x hand y gesture z inputs q hair length");
            set_LHand_ShortHair_BackPosition_5Weights(_F1_5in);
            set_LHand_ShortHair_BackPosition_7Weights(_F1_7in);
            break;
    }
}



/**************************************************************************/
/*!
    @brief 5 input node neural network activation function (generated by synapticJS)
*/
/**************************************************************************/

float activateFiveInNN(float t1, float t2, float t3, float  t4, float distance) {
//  float F[] = {0,0,0};
//  F = _F1_5in;
/*  float *t1_p = &t1;
  float *t2_p = &t2;
  float *t3_p = &t3;
  float *t4_p = &t4;
  float *distance_p = &distance; */
  /*
  F[3] = t1;
  F[5] = t2;
  F[7] = t3;
  F[9] = t4;
  F[11] = distance;

F[0] = F[1];F[1] = F[2];F[1] += F[3] * F[4];F[1] += F[5] * F[6];F[1] += F[7] * F[8];F[1] += F[9] * F[10];F[1] += F[11] * F[12];F[1] += F[13] * F[14];F[1] += F[15] * F[16];F[1] += F[17] * F[18];F[1] += F[19] * F[20];F[1] += F[21] * F[22];F[23] = (1 / (1 + exp(-F[1])));F[24] = F[23] * (1 - F[23]);F[25] = F[23];F[26] = F[23];F[27] = F[23];F[28] = F[23];F[29] = F[23];
F[30] = F[31];F[31] = F[32];F[31] += F[3] * F[33];F[31] += F[5] * F[34];F[31] += F[7] * F[35];F[31] += F[9] * F[36];F[31] += F[11] * F[37];F[31] += F[13] * F[38];F[31] += F[15] * F[39];F[31] += F[17] * F[40];F[31] += F[19] * F[41];F[31] += F[21] * F[42];F[43] = (1 / (1 + exp(-F[31])));F[44] = F[43] * (1 - F[43]);F[45] = F[43];F[46] = F[43];F[47] = F[43];F[48] = F[43];F[49] = F[43];
F[50] = F[51];F[51] = F[52];F[51] += F[3] * F[53];F[51] += F[5] * F[54];F[51] += F[7] * F[55];F[51] += F[9] * F[56];F[51] += F[11] * F[57];F[51] += F[13] * F[58];F[51] += F[15] * F[59];F[51] += F[17] * F[60];F[51] += F[19] * F[61];F[51] += F[21] * F[62];F[63] = (1 / (1 + exp(-F[51])));F[64] = F[63] * (1 - F[63]);F[65] = F[63];F[66] = F[63];F[67] = F[63];F[68] = F[63];F[69] = F[63];
F[70] = F[71];F[71] = F[72];F[71] += F[3] * F[73];F[71] += F[5] * F[74];F[71] += F[7] * F[75];F[71] += F[9] * F[76];F[71] += F[11] * F[77];F[71] += F[13] * F[78];F[71] += F[15] * F[79];F[71] += F[17] * F[80];F[71] += F[19] * F[81];F[71] += F[21] * F[82];F[83] = (1 / (1 + exp(-F[71])));F[84] = F[83] * (1 - F[83]);F[85] = F[83];F[86] = F[83];F[87] = F[83];F[88] = F[83];F[89] = F[83];
F[90] = F[91];F[91] = F[92];F[91] += F[3] * F[93];F[91] += F[5] * F[94];F[91] += F[7] * F[95];F[91] += F[9] * F[96];F[91] += F[11] * F[97];F[91] += F[13] * F[98];F[91] += F[15] * F[99];F[91] += F[17] * F[100];F[91] += F[19] * F[101];F[91] += F[21] * F[102];F[103] = (1 / (1 + exp(-F[91])));F[104] = F[103] * (1 - F[103]);F[105] = F[103];F[106] = F[103];F[107] = F[103];F[108] = F[103];F[109] = F[103];
F[110] = F[111];F[111] = F[112];F[111] += F[3] * F[113];F[111] += F[5] * F[114];F[111] += F[7] * F[115];F[111] += F[9] * F[116];F[111] += F[11] * F[117];F[111] += F[13] * F[118];F[111] += F[15] * F[119];F[111] += F[17] * F[120];F[111] += F[19] * F[121];F[111] += F[21] * F[122];F[123] = (1 / (1 + exp(-F[111])));F[124] = F[123] * (1 - F[123]);F[125] = F[123];
F[126] = F[127];F[127] = F[128];F[127] += F[3] * F[129];F[127] += F[5] * F[130];F[127] += F[7] * F[131];F[127] += F[9] * F[132];F[127] += F[11] * F[133];F[127] += F[13] * F[134];F[127] += F[15] * F[135];F[127] += F[17] * F[136];F[127] += F[19] * F[137];F[127] += F[21] * F[138];F[139] = (1 / (1 + exp(-F[127])));F[140] = F[139] * (1 - F[139]);F[141] = F[139];
F[142] = F[143];F[143] = F[144];F[143] += F[3] * F[145];F[143] += F[5] * F[146];F[143] += F[7] * F[147];F[143] += F[9] * F[148];F[143] += F[11] * F[149];F[143] += F[13] * F[150];F[143] += F[15] * F[151];F[143] += F[17] * F[152];F[143] += F[19] * F[153];F[143] += F[21] * F[154];F[155] = (1 / (1 + exp(-F[143])));F[156] = F[155] * (1 - F[155]);F[157] = F[155];
F[158] = F[159];F[159] = F[160];F[159] += F[3] * F[161];F[159] += F[5] * F[162];F[159] += F[7] * F[163];F[159] += F[9] * F[164];F[159] += F[11] * F[165];F[159] += F[13] * F[166];F[159] += F[15] * F[167];F[159] += F[17] * F[168];F[159] += F[19] * F[169];F[159] += F[21] * F[170];F[171] = (1 / (1 + exp(-F[159])));F[172] = F[171] * (1 - F[171]);F[173] = F[171];
F[174] = F[175];F[175] = F[176];F[175] += F[3] * F[177];F[175] += F[5] * F[178];F[175] += F[7] * F[179];F[175] += F[9] * F[180];F[175] += F[11] * F[181];F[175] += F[13] * F[182];F[175] += F[15] * F[183];F[175] += F[17] * F[184];F[175] += F[19] * F[185];F[175] += F[21] * F[186];F[187] = (1 / (1 + exp(-F[175])));F[188] = F[187] * (1 - F[187]);F[189] = F[187];
F[190] = F[191];F[191] = F[125] * F[192] * F[191] + F[193];F[191] += F[3] * F[194] * F[25];F[191] += F[5] * F[195] * F[26];F[191] += F[7] * F[196] * F[27];F[191] += F[9] * F[197] * F[28];F[191] += F[11] * F[198] * F[29];F[13] = (1 / (1 + exp(-F[191])));F[199] = F[13] * (1 - F[13]);
F[200] = F[201];F[201] = F[141] * F[202] * F[201] + F[203];F[201] += F[3] * F[204] * F[45];F[201] += F[5] * F[205] * F[46];F[201] += F[7] * F[206] * F[47];F[201] += F[9] * F[207] * F[48];F[201] += F[11] * F[208] * F[49];F[15] = (1 / (1 + exp(-F[201])));F[209] = F[15] * (1 - F[15]);
F[210] = F[211];F[211] = F[157] * F[212] * F[211] + F[213];F[211] += F[3] * F[214] * F[65];F[211] += F[5] * F[215] * F[66];F[211] += F[7] * F[216] * F[67];F[211] += F[9] * F[217] * F[68];F[211] += F[11] * F[218] * F[69];F[17] = (1 / (1 + exp(-F[211])));F[219] = F[17] * (1 - F[17]);
F[220] = F[221];F[221] = F[173] * F[222] * F[221] + F[223];F[221] += F[3] * F[224] * F[85];F[221] += F[5] * F[225] * F[86];F[221] += F[7] * F[226] * F[87];F[221] += F[9] * F[227] * F[88];F[221] += F[11] * F[228] * F[89];F[19] = (1 / (1 + exp(-F[221])));F[229] = F[19] * (1 - F[19]);
F[230] = F[231];F[231] = F[189] * F[232] * F[231] + F[233];F[231] += F[3] * F[234] * F[105];F[231] += F[5] * F[235] * F[106];F[231] += F[7] * F[236] * F[107];F[231] += F[9] * F[237] * F[108];F[231] += F[11] * F[238] * F[109];F[21] = (1 / (1 + exp(-F[231])));F[239] = F[21] * (1 - F[21]);
F[240] = F[241];F[241] = F[242];F[241] += F[3] * F[243];F[241] += F[5] * F[244];F[241] += F[7] * F[245];F[241] += F[9] * F[246];F[241] += F[11] * F[247];F[241] += F[13] * F[248];F[241] += F[15] * F[249];F[241] += F[17] * F[250];F[241] += F[19] * F[251];F[241] += F[21] * F[252];F[253] = (1 / (1 + exp(-F[241])));F[254] = F[253] * (1 - F[253]);F[255] = F[253];
F[256] = F[257];F[257] = F[258];F[257] += F[3] * F[259];F[257] += F[5] * F[260];F[257] += F[7] * F[261];F[257] += F[9] * F[262];F[257] += F[11] * F[263];F[257] += F[13] * F[264];F[257] += F[15] * F[265];F[257] += F[17] * F[266];F[257] += F[19] * F[267];F[257] += F[21] * F[268];F[269] = (1 / (1 + exp(-F[257])));F[270] = F[269] * (1 - F[269]);F[271] = F[269];
F[272] = F[273];F[273] = F[274];F[273] += F[3] * F[275];F[273] += F[5] * F[276];F[273] += F[7] * F[277];F[273] += F[9] * F[278];F[273] += F[11] * F[279];F[273] += F[13] * F[280];F[273] += F[15] * F[281];F[273] += F[17] * F[282];F[273] += F[19] * F[283];F[273] += F[21] * F[284];F[285] = (1 / (1 + exp(-F[273])));F[286] = F[285] * (1 - F[285]);F[287] = F[285];
F[288] = F[289];F[289] = F[290];F[289] += F[3] * F[291];F[289] += F[5] * F[292];F[289] += F[7] * F[293];F[289] += F[9] * F[294];F[289] += F[11] * F[295];F[289] += F[13] * F[296];F[289] += F[15] * F[297];F[289] += F[17] * F[298];F[289] += F[19] * F[299];F[289] += F[21] * F[300];F[301] = (1 / (1 + exp(-F[289])));F[302] = F[301] * (1 - F[301]);F[303] = F[301];
F[304] = F[305];F[305] = F[306];F[305] += F[3] * F[307];F[305] += F[5] * F[308];F[305] += F[7] * F[309];F[305] += F[9] * F[310];F[305] += F[11] * F[311];F[305] += F[13] * F[312];F[305] += F[15] * F[313];F[305] += F[17] * F[314];F[305] += F[19] * F[315];F[305] += F[21] * F[316];F[317] = (1 / (1 + exp(-F[305])));F[318] = F[317] * (1 - F[317]);F[319] = F[317];
F[320] = F[321];F[321] = F[322];F[321] += F[3] * F[323];F[321] += F[5] * F[324];F[321] += F[7] * F[325];F[321] += F[9] * F[326];F[321] += F[11] * F[327];F[321] += F[13] * F[328];F[321] += F[15] * F[329];F[321] += F[17] * F[330];F[321] += F[19] * F[331];F[321] += F[21] * F[332];F[321] += F[333] * F[334];F[321] += F[335] * F[336];F[337] = (1 / (1 + exp(-F[321])));F[338] = F[337] * (1 - F[337]);F[339] = F[337];F[340] = F[337];F[341] = F[337];F[342] = F[337];F[343] = F[337];F[344] = F[337];F[345] = F[337];F[346] = F[337];F[347] = F[337];F[348] = F[337];
F[349] = F[350];F[350] = F[351];F[350] += F[3] * F[352];F[350] += F[5] * F[353];F[350] += F[7] * F[354];F[350] += F[9] * F[355];F[350] += F[11] * F[356];F[350] += F[13] * F[357];F[350] += F[15] * F[358];F[350] += F[17] * F[359];F[350] += F[19] * F[360];F[350] += F[21] * F[361];F[350] += F[333] * F[362];F[350] += F[335] * F[363];F[364] = (1 / (1 + exp(-F[350])));F[365] = F[364] * (1 - F[364]);F[366] = F[364];F[367] = F[364];F[368] = F[364];F[369] = F[364];F[370] = F[364];F[371] = F[364];F[372] = F[364];F[373] = F[364];F[374] = F[364];F[375] = F[364];
F[376] = F[377];F[377] = F[378];F[377] += F[3] * F[379];F[377] += F[5] * F[380];F[377] += F[7] * F[381];F[377] += F[9] * F[382];F[377] += F[11] * F[383];F[377] += F[13] * F[384];F[377] += F[15] * F[385];F[377] += F[17] * F[386];F[377] += F[19] * F[387];F[377] += F[21] * F[388];F[377] += F[333] * F[389];F[377] += F[335] * F[390];F[391] = (1 / (1 + exp(-F[377])));F[392] = F[391] * (1 - F[391]);F[393] = F[391];
F[394] = F[395];F[395] = F[396];F[395] += F[3] * F[397];F[395] += F[5] * F[398];F[395] += F[7] * F[399];F[395] += F[9] * F[400];F[395] += F[11] * F[401];F[395] += F[13] * F[402];F[395] += F[15] * F[403];F[395] += F[17] * F[404];F[395] += F[19] * F[405];F[395] += F[21] * F[406];F[395] += F[333] * F[407];F[395] += F[335] * F[408];F[409] = (1 / (1 + exp(-F[395])));F[410] = F[409] * (1 - F[409]);F[411] = F[409];
F[412] = F[413];F[413] = F[393] * F[414] * F[413] + F[415];F[413] += F[3] * F[416] * F[339];F[413] += F[5] * F[417] * F[340];F[413] += F[7] * F[418] * F[341];F[413] += F[9] * F[419] * F[342];F[413] += F[11] * F[420] * F[343];F[413] += F[13] * F[421] * F[344];F[413] += F[15] * F[422] * F[345];F[413] += F[17] * F[423] * F[346];F[413] += F[19] * F[424] * F[347];F[413] += F[21] * F[425] * F[348];F[333] = (1 / (1 + exp(-F[413])));F[426] = F[333] * (1 - F[333]);
F[427] = F[428];F[428] = F[411] * F[429] * F[428] + F[430];F[428] += F[3] * F[431] * F[366];F[428] += F[5] * F[432] * F[367];F[428] += F[7] * F[433] * F[368];F[428] += F[9] * F[434] * F[369];F[428] += F[11] * F[435] * F[370];F[428] += F[13] * F[436] * F[371];F[428] += F[15] * F[437] * F[372];F[428] += F[17] * F[438] * F[373];F[428] += F[19] * F[439] * F[374];F[428] += F[21] * F[440] * F[375];F[335] = (1 / (1 + exp(-F[428])));F[441] = F[335] * (1 - F[335]);
F[442] = F[443];F[443] = F[444];F[443] += F[3] * F[445];F[443] += F[5] * F[446];F[443] += F[7] * F[447];F[443] += F[9] * F[448];F[443] += F[11] * F[449];F[443] += F[13] * F[450];F[443] += F[15] * F[451];F[443] += F[17] * F[452];F[443] += F[19] * F[453];F[443] += F[21] * F[454];F[443] += F[333] * F[455];F[443] += F[335] * F[456];F[457] = (1 / (1 + exp(-F[443])));F[458] = F[457] * (1 - F[457]);F[459] = F[457];
F[460] = F[461];F[461] = F[462];F[461] += F[3] * F[463];F[461] += F[5] * F[464];F[461] += F[7] * F[465];F[461] += F[9] * F[466];F[461] += F[11] * F[467];F[461] += F[13] * F[468];F[461] += F[15] * F[469];F[461] += F[17] * F[470];F[461] += F[19] * F[471];F[461] += F[21] * F[472];F[461] += F[333] * F[473];F[461] += F[335] * F[474];F[475] = (1 / (1 + exp(-F[461])));F[476] = F[475] * (1 - F[475]);F[477] = F[475];
F[478] = F[479];F[479] = F[480];F[479] += F[13] * F[481] * F[255];F[479] += F[15] * F[482] * F[271];F[479] += F[17] * F[483] * F[287];F[479] += F[19] * F[484] * F[303];F[479] += F[21] * F[485] * F[319];F[479] += F[333] * F[486] * F[459];F[479] += F[335] * F[487] * F[477];F[479] += F[3] * F[488];F[479] += F[5] * F[489];F[479] += F[7] * F[490];F[479] += F[9] * F[491];F[479] += F[11] * F[492];F[493] = (1 / (1 + exp(-F[479])));F[494] = F[493] * (1 - F[493]);

float output = F[493] * 100;
return output;
*/
}

float activateSevenInNN(float t1, float t2, float t3, float  t4, float distance, float pitch, float roll, float *F[]) {
  //float F[] = {0,0,0};
  /*
  float *t1_p = &t1;
  float *t2_p = &t2;
  float *t3_p = &t3;
  float *t4_p = &t4;
  float *distance_p = &distance;
  float *pitch_p = &pitch;
  float *roll_p = &roll;
  F[3] = t1_p;
  F[5] = t2_p;
  F[7] = t3_p;
  F[9] = t4_p;
  F[11] = distance_p;
  F[13] = pitch_p;
  F[15] = roll_p;
  
  F[0] = F[1];F[1] = F[2];F[1] += F[3] * F[4];F[1] += F[5] * F[6];F[1] += F[7] * F[8];F[1] += F[9] * F[10];F[1] += F[11] * F[12];F[1] += F[13] * F[14];F[1] += F[15] * F[16];F[1] += F[17] * F[18];F[1] += F[19] * F[20];F[1] += F[21] * F[22];F[1] += F[23] * F[24];F[1] += F[25] * F[26];F[27] = (1 / (1 + exp(-F[1])));F[28] = F[27] * (1 - F[27]);F[29] = F[27];F[30] = F[27];F[31] = F[27];F[32] = F[27];F[33] = F[27];F[34] = F[27];F[35] = F[27];
  F[36] = F[37];F[37] = F[38];F[37] += F[3] * F[39];F[37] += F[5] * F[40];F[37] += F[7] * F[41];F[37] += F[9] * F[42];F[37] += F[11] * F[43];F[37] += F[13] * F[44];F[37] += F[15] * F[45];F[37] += F[17] * F[46];F[37] += F[19] * F[47];F[37] += F[21] * F[48];F[37] += F[23] * F[49];F[37] += F[25] * F[50];F[51] = (1 / (1 + exp(-F[37])));F[52] = F[51] * (1 - F[51]);F[53] = F[51];F[54] = F[51];F[55] = F[51];F[56] = F[51];F[57] = F[51];F[58] = F[51];F[59] = F[51];
  F[60] = F[61];F[61] = F[62];F[61] += F[3] * F[63];F[61] += F[5] * F[64];F[61] += F[7] * F[65];F[61] += F[9] * F[66];F[61] += F[11] * F[67];F[61] += F[13] * F[68];F[61] += F[15] * F[69];F[61] += F[17] * F[70];F[61] += F[19] * F[71];F[61] += F[21] * F[72];F[61] += F[23] * F[73];F[61] += F[25] * F[74];F[75] = (1 / (1 + exp(-F[61])));F[76] = F[75] * (1 - F[75]);F[77] = F[75];F[78] = F[75];F[79] = F[75];F[80] = F[75];F[81] = F[75];F[82] = F[75];F[83] = F[75];
  F[84] = F[85];F[85] = F[86];F[85] += F[3] * F[87];F[85] += F[5] * F[88];F[85] += F[7] * F[89];F[85] += F[9] * F[90];F[85] += F[11] * F[91];F[85] += F[13] * F[92];F[85] += F[15] * F[93];F[85] += F[17] * F[94];F[85] += F[19] * F[95];F[85] += F[21] * F[96];F[85] += F[23] * F[97];F[85] += F[25] * F[98];F[99] = (1 / (1 + exp(-F[85])));F[100] = F[99] * (1 - F[99]);F[101] = F[99];F[102] = F[99];F[103] = F[99];F[104] = F[99];F[105] = F[99];F[106] = F[99];F[107] = F[99];
  F[108] = F[109];F[109] = F[110];F[109] += F[3] * F[111];F[109] += F[5] * F[112];F[109] += F[7] * F[113];F[109] += F[9] * F[114];F[109] += F[11] * F[115];F[109] += F[13] * F[116];F[109] += F[15] * F[117];F[109] += F[17] * F[118];F[109] += F[19] * F[119];F[109] += F[21] * F[120];F[109] += F[23] * F[121];F[109] += F[25] * F[122];F[123] = (1 / (1 + exp(-F[109])));F[124] = F[123] * (1 - F[123]);F[125] = F[123];F[126] = F[123];F[127] = F[123];F[128] = F[123];F[129] = F[123];F[130] = F[123];F[131] = F[123];
  F[132] = F[133];F[133] = F[134];F[133] += F[3] * F[135];F[133] += F[5] * F[136];F[133] += F[7] * F[137];F[133] += F[9] * F[138];F[133] += F[11] * F[139];F[133] += F[13] * F[140];F[133] += F[15] * F[141];F[133] += F[17] * F[142];F[133] += F[19] * F[143];F[133] += F[21] * F[144];F[133] += F[23] * F[145];F[133] += F[25] * F[146];F[147] = (1 / (1 + exp(-F[133])));F[148] = F[147] * (1 - F[147]);F[149] = F[147];
  F[150] = F[151];F[151] = F[152];F[151] += F[3] * F[153];F[151] += F[5] * F[154];F[151] += F[7] * F[155];F[151] += F[9] * F[156];F[151] += F[11] * F[157];F[151] += F[13] * F[158];F[151] += F[15] * F[159];F[151] += F[17] * F[160];F[151] += F[19] * F[161];F[151] += F[21] * F[162];F[151] += F[23] * F[163];F[151] += F[25] * F[164];F[165] = (1 / (1 + exp(-F[151])));F[166] = F[165] * (1 - F[165]);F[167] = F[165];
  F[168] = F[169];F[169] = F[170];F[169] += F[3] * F[171];F[169] += F[5] * F[172];F[169] += F[7] * F[173];F[169] += F[9] * F[174];F[169] += F[11] * F[175];F[169] += F[13] * F[176];F[169] += F[15] * F[177];F[169] += F[17] * F[178];F[169] += F[19] * F[179];F[169] += F[21] * F[180];F[169] += F[23] * F[181];F[169] += F[25] * F[182];F[183] = (1 / (1 + exp(-F[169])));F[184] = F[183] * (1 - F[183]);F[185] = F[183];
  F[186] = F[187];F[187] = F[188];F[187] += F[3] * F[189];F[187] += F[5] * F[190];F[187] += F[7] * F[191];F[187] += F[9] * F[192];F[187] += F[11] * F[193];F[187] += F[13] * F[194];F[187] += F[15] * F[195];F[187] += F[17] * F[196];F[187] += F[19] * F[197];F[187] += F[21] * F[198];F[187] += F[23] * F[199];F[187] += F[25] * F[200];F[201] = (1 / (1 + exp(-F[187])));F[202] = F[201] * (1 - F[201]);F[203] = F[201];
  F[204] = F[205];F[205] = F[206];F[205] += F[3] * F[207];F[205] += F[5] * F[208];F[205] += F[7] * F[209];F[205] += F[9] * F[210];F[205] += F[11] * F[211];F[205] += F[13] * F[212];F[205] += F[15] * F[213];F[205] += F[17] * F[214];F[205] += F[19] * F[215];F[205] += F[21] * F[216];F[205] += F[23] * F[217];F[205] += F[25] * F[218];F[219] = (1 / (1 + exp(-F[205])));F[220] = F[219] * (1 - F[219]);F[221] = F[219];
  F[222] = F[223];F[223] = F[149] * F[224] * F[223] + F[225];F[223] += F[3] * F[226] * F[29];F[223] += F[5] * F[227] * F[30];F[223] += F[7] * F[228] * F[31];F[223] += F[9] * F[229] * F[32];F[223] += F[11] * F[230] * F[33];F[223] += F[13] * F[231] * F[34];F[223] += F[15] * F[232] * F[35];F[17] = (1 / (1 + exp(-F[223])));F[233] = F[17] * (1 - F[17]);
  F[234] = F[235];F[235] = F[167] * F[236] * F[235] + F[237];F[235] += F[3] * F[238] * F[53];F[235] += F[5] * F[239] * F[54];F[235] += F[7] * F[240] * F[55];F[235] += F[9] * F[241] * F[56];F[235] += F[11] * F[242] * F[57];F[235] += F[13] * F[243] * F[58];F[235] += F[15] * F[244] * F[59];F[19] = (1 / (1 + exp(-F[235])));F[245] = F[19] * (1 - F[19]);
  F[246] = F[247];F[247] = F[185] * F[248] * F[247] + F[249];F[247] += F[3] * F[250] * F[77];F[247] += F[5] * F[251] * F[78];F[247] += F[7] * F[252] * F[79];F[247] += F[9] * F[253] * F[80];F[247] += F[11] * F[254] * F[81];F[247] += F[13] * F[255] * F[82];F[247] += F[15] * F[256] * F[83];F[21] = (1 / (1 + exp(-F[247])));F[257] = F[21] * (1 - F[21]);
  F[258] = F[259];F[259] = F[203] * F[260] * F[259] + F[261];F[259] += F[3] * F[262] * F[101];F[259] += F[5] * F[263] * F[102];F[259] += F[7] * F[264] * F[103];F[259] += F[9] * F[265] * F[104];F[259] += F[11] * F[266] * F[105];F[259] += F[13] * F[267] * F[106];F[259] += F[15] * F[268] * F[107];F[23] = (1 / (1 + exp(-F[259])));F[269] = F[23] * (1 - F[23]);
  F[270] = F[271];F[271] = F[221] * F[272] * F[271] + F[273];F[271] += F[3] * F[274] * F[125];F[271] += F[5] * F[275] * F[126];F[271] += F[7] * F[276] * F[127];F[271] += F[9] * F[277] * F[128];F[271] += F[11] * F[278] * F[129];F[271] += F[13] * F[279] * F[130];F[271] += F[15] * F[280] * F[131];F[25] = (1 / (1 + exp(-F[271])));F[281] = F[25] * (1 - F[25]);
  F[282] = F[283];F[283] = F[284];F[283] += F[3] * F[285];F[283] += F[5] * F[286];F[283] += F[7] * F[287];F[283] += F[9] * F[288];F[283] += F[11] * F[289];F[283] += F[13] * F[290];F[283] += F[15] * F[291];F[283] += F[17] * F[292];F[283] += F[19] * F[293];F[283] += F[21] * F[294];F[283] += F[23] * F[295];F[283] += F[25] * F[296];F[297] = (1 / (1 + exp(-F[283])));F[298] = F[297] * (1 - F[297]);F[299] = F[297];
  F[300] = F[301];F[301] = F[302];F[301] += F[3] * F[303];F[301] += F[5] * F[304];F[301] += F[7] * F[305];F[301] += F[9] * F[306];F[301] += F[11] * F[307];F[301] += F[13] * F[308];F[301] += F[15] * F[309];F[301] += F[17] * F[310];F[301] += F[19] * F[311];F[301] += F[21] * F[312];F[301] += F[23] * F[313];F[301] += F[25] * F[314];F[315] = (1 / (1 + exp(-F[301])));F[316] = F[315] * (1 - F[315]);F[317] = F[315];
  F[318] = F[319];F[319] = F[320];F[319] += F[3] * F[321];F[319] += F[5] * F[322];F[319] += F[7] * F[323];F[319] += F[9] * F[324];F[319] += F[11] * F[325];F[319] += F[13] * F[326];F[319] += F[15] * F[327];F[319] += F[17] * F[328];F[319] += F[19] * F[329];F[319] += F[21] * F[330];F[319] += F[23] * F[331];F[319] += F[25] * F[332];F[333] = (1 / (1 + exp(-F[319])));F[334] = F[333] * (1 - F[333]);F[335] = F[333];
  F[336] = F[337];F[337] = F[338];F[337] += F[3] * F[339];F[337] += F[5] * F[340];F[337] += F[7] * F[341];F[337] += F[9] * F[342];F[337] += F[11] * F[343];F[337] += F[13] * F[344];F[337] += F[15] * F[345];F[337] += F[17] * F[346];F[337] += F[19] * F[347];F[337] += F[21] * F[348];F[337] += F[23] * F[349];F[337] += F[25] * F[350];F[351] = (1 / (1 + exp(-F[337])));F[352] = F[351] * (1 - F[351]);F[353] = F[351];
  F[354] = F[355];F[355] = F[356];F[355] += F[3] * F[357];F[355] += F[5] * F[358];F[355] += F[7] * F[359];F[355] += F[9] * F[360];F[355] += F[11] * F[361];F[355] += F[13] * F[362];F[355] += F[15] * F[363];F[355] += F[17] * F[364];F[355] += F[19] * F[365];F[355] += F[21] * F[366];F[355] += F[23] * F[367];F[355] += F[25] * F[368];F[369] = (1 / (1 + exp(-F[355])));F[370] = F[369] * (1 - F[369]);F[371] = F[369];
  F[372] = F[373];F[373] = F[374];F[373] += F[3] * F[375];F[373] += F[5] * F[376];F[373] += F[7] * F[377];F[373] += F[9] * F[378];F[373] += F[11] * F[379];F[373] += F[13] * F[380];F[373] += F[15] * F[381];F[373] += F[17] * F[382];F[373] += F[19] * F[383];F[373] += F[21] * F[384];F[373] += F[23] * F[385];F[373] += F[25] * F[386];F[373] += F[387] * F[388];F[373] += F[389] * F[390];F[391] = (1 / (1 + exp(-F[373])));F[392] = F[391] * (1 - F[391]);F[393] = F[391];F[394] = F[391];F[395] = F[391];F[396] = F[391];F[397] = F[391];F[398] = F[391];F[399] = F[391];F[400] = F[391];F[401] = F[391];F[402] = F[391];F[403] = F[391];F[404] = F[391];
  F[405] = F[406];F[406] = F[407];F[406] += F[3] * F[408];F[406] += F[5] * F[409];F[406] += F[7] * F[410];F[406] += F[9] * F[411];F[406] += F[11] * F[412];F[406] += F[13] * F[413];F[406] += F[15] * F[414];F[406] += F[17] * F[415];F[406] += F[19] * F[416];F[406] += F[21] * F[417];F[406] += F[23] * F[418];F[406] += F[25] * F[419];F[406] += F[387] * F[420];F[406] += F[389] * F[421];F[422] = (1 / (1 + exp(-F[406])));F[423] = F[422] * (1 - F[422]);F[424] = F[422];F[425] = F[422];F[426] = F[422];F[427] = F[422];F[428] = F[422];F[429] = F[422];F[430] = F[422];F[431] = F[422];F[432] = F[422];F[433] = F[422];F[434] = F[422];F[435] = F[422];
  F[436] = F[437];F[437] = F[438];F[437] += F[3] * F[439];F[437] += F[5] * F[440];F[437] += F[7] * F[441];F[437] += F[9] * F[442];F[437] += F[11] * F[443];F[437] += F[13] * F[444];F[437] += F[15] * F[445];F[437] += F[17] * F[446];F[437] += F[19] * F[447];F[437] += F[21] * F[448];F[437] += F[23] * F[449];F[437] += F[25] * F[450];F[437] += F[387] * F[451];F[437] += F[389] * F[452];F[453] = (1 / (1 + exp(-F[437])));F[454] = F[453] * (1 - F[453]);F[455] = F[453];
  F[456] = F[457];F[457] = F[458];F[457] += F[3] * F[459];F[457] += F[5] * F[460];F[457] += F[7] * F[461];F[457] += F[9] * F[462];F[457] += F[11] * F[463];F[457] += F[13] * F[464];F[457] += F[15] * F[465];F[457] += F[17] * F[466];F[457] += F[19] * F[467];F[457] += F[21] * F[468];F[457] += F[23] * F[469];F[457] += F[25] * F[470];F[457] += F[387] * F[471];F[457] += F[389] * F[472];F[473] = (1 / (1 + exp(-F[457])));F[474] = F[473] * (1 - F[473]);F[475] = F[473];
  F[476] = F[477];F[477] = F[455] * F[478] * F[477] + F[479];F[477] += F[3] * F[480] * F[393];F[477] += F[5] * F[481] * F[394];F[477] += F[7] * F[482] * F[395];F[477] += F[9] * F[483] * F[396];F[477] += F[11] * F[484] * F[397];F[477] += F[13] * F[485] * F[398];F[477] += F[15] * F[486] * F[399];F[477] += F[17] * F[487] * F[400];F[477] += F[19] * F[488] * F[401];F[477] += F[21] * F[489] * F[402];F[477] += F[23] * F[490] * F[403];F[477] += F[25] * F[491] * F[404];F[387] = (1 / (1 + exp(-F[477])));F[492] = F[387] * (1 - F[387]);
  F[493] = F[494];F[494] = F[475] * F[495] * F[494] + F[496];F[494] += F[3] * F[497] * F[424];F[494] += F[5] * F[498] * F[425];F[494] += F[7] * F[499] * F[426];F[494] += F[9] * F[500] * F[427];F[494] += F[11] * F[501] * F[428];F[494] += F[13] * F[502] * F[429];F[494] += F[15] * F[503] * F[430];F[494] += F[17] * F[504] * F[431];F[494] += F[19] * F[505] * F[432];F[494] += F[21] * F[506] * F[433];F[494] += F[23] * F[507] * F[434];F[494] += F[25] * F[508] * F[435];F[389] = (1 / (1 + exp(-F[494])));F[509] = F[389] * (1 - F[389]);
  F[510] = F[511];F[511] = F[512];F[511] += F[3] * F[513];F[511] += F[5] * F[514];F[511] += F[7] * F[515];F[511] += F[9] * F[516];F[511] += F[11] * F[517];F[511] += F[13] * F[518];F[511] += F[15] * F[519];F[511] += F[17] * F[520];F[511] += F[19] * F[521];F[511] += F[21] * F[522];F[511] += F[23] * F[523];F[511] += F[25] * F[524];F[511] += F[387] * F[525];F[511] += F[389] * F[526];F[527] = (1 / (1 + exp(-F[511])));F[528] = F[527] * (1 - F[527]);F[529] = F[527];
  F[530] = F[531];F[531] = F[532];F[531] += F[3] * F[533];F[531] += F[5] * F[534];F[531] += F[7] * F[535];F[531] += F[9] * F[536];F[531] += F[11] * F[537];F[531] += F[13] * F[538];F[531] += F[15] * F[539];F[531] += F[17] * F[540];F[531] += F[19] * F[541];F[531] += F[21] * F[542];F[531] += F[23] * F[543];F[531] += F[25] * F[544];F[531] += F[387] * F[545];F[531] += F[389] * F[546];F[547] = (1 / (1 + exp(-F[531])));F[548] = F[547] * (1 - F[547]);F[549] = F[547];
  F[550] = F[551];F[551] = F[552];F[551] += F[17] * F[553] * F[299];F[551] += F[19] * F[554] * F[317];F[551] += F[21] * F[555] * F[335];F[551] += F[23] * F[556] * F[353];F[551] += F[25] * F[557] * F[371];F[551] += F[387] * F[558] * F[529];F[551] += F[389] * F[559] * F[549];F[551] += F[3] * F[560];F[551] += F[5] * F[561];F[551] += F[7] * F[562];F[551] += F[9] * F[563];F[551] += F[11] * F[564];F[551] += F[13] * F[565];F[551] += F[15] * F[566];F[567] = (1 / (1 + exp(-F[551])));F[568] = F[567] * (1 - F[567]);
  float output = F[567] * 100;
  return output;
  */
}

tingleNeuralNetwork neuralNetwork; //Arduino_nRF5x_lowPower nRF5x_lowPower;
