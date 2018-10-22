/**************************************************************************/
/*!
    @file       tingleNN.h
    @author     Curt White <curtpw@gmail.com>
    @version    0.1.0
    
    Gesture Detection for Tingle using SynapticJS MLP neural networks pre-trained for targets
    
    @section HISTORY
    
    v0.1.0  - First Release
*/
/**************************************************************************/

#ifndef TINGLE_NN_H
#define TINGLE_NN_H

#include "Arduino.h"

typedef enum {
    RHAND_SHORT_MOUTH            = 0x01, 
    RHAND_SHORT_EYES             = 0x02, 
    RHAND_SHORT_FRONT            = 0x03, 
    RHAND_SHORT_SIDE             = 0x04, 
    RHAND_SHORT_TOP              = 0x05, 
    RHAND_SHORT_BACK             = 0x06, 

    LHAND_SHORT_MOUTH            = 0x07, 
    LHAND_SHORT_EYES             = 0x08, 
    LHAND_SHORT_FRONT            = 0x09, 
    LHAND_SHORT_SIDE             = 0x10, 
    LHAND_SHORT_TOP              = 0x11, 
    LHAND_SHORT_BACK             = 0x12, 

    RHAND_MEDIUM_MOUTH           = 0x13, 
    RHAND_MEDIUMT_EYES           = 0x14, 
    RHAND_MEDIUM_FRONT           = 0x15, 
    RHAND_MEDIUM_SIDE            = 0x16, 
    RHAND_MEDIUMT_TOP            = 0x17, 
    RHAND_MEDIUM_BACK            = 0x18, 

    LHAND_MEDIUM_MOUTH           = 0x19, 
    LHAND_MEDIUM_EYES            = 0x20, 
    LHAND_MEDIUM_FRONT           = 0x21, 
    LHAND_MEDIUM_SIDE            = 0x22, 
    LHAND_MEDIUM_TOP             = 0x23, 
    LHAND_MEDIUM_BACK            = 0x24, 

    RHAND_LONG_MOUTH             = 0x25, 
    RHAND_LONG_EYES              = 0x26, 
    RHAND_LONG_FRONT             = 0x27, 
    RHAND_LONG_SIDE              = 0x28, 
    RHAND_LONG_TOP               = 0x29, 
    RHAND_LONG_BACK              = 0x30, 

    LHAND_LONG_MOUTH             = 0x31, 
    LHAND_LONG_EYES              = 0x32, 
    LHAND_LONG_FRONT             = 0x33, 
    LHAND_LONG_SIDE              = 0x34, 
    LHAND_LONG_TOP               = 0x35, 
    LHAND_LONG_BACK              = 0x36, 



    NO_TARGET                      = 0x99,
} tingleNN_targets_t; //nRF5x_powermodes_t;

typedef enum {
    SENSE_VERY_LOW           = 0x01, 
    SENSE_LOW                = 0x02, 
    SENSE_AVERAGE            = 0x03, 
    SENSE_HIGH               = 0x04, 
    SENSE_VERY_HIGH          = 0x05, 
} tingleNN_sensitivity_t;

class tingleNeuralNetwork { //Arduino_nRF5x_lowPower {
    public:
        void selectTarget(tingleNN_targets_t target); //void powerMode(nRF5x_powermodes_t mode);
        void selectTarget(tingleNN_targets_t target1, tingleNN_targets_t target2); //if two targets 
        void sensitivity(tingleNN_sensitivity_t sense); 
        float detect(float t1, float t2, float t3, float  t4, float distance);
        float detect(float t1, float t2, float t3, float  t4, float distance, float pitch, float roll);  
        void checkSettings();

    private:
        float activateFiveInNN(float t1, float t2, float t3, float  t4, float distance, float F[]);
        float activateSevenInNN(float t1, float t2, float t3, float  t4, float distance, float pitch, float roll, float F[]);

        void loadNetworkWeights(tingleNN_targets_t target,  float F_5in[], float F_7in[]);

        //SHORT HAIR NN WEIGHTS
        void set_RHand_ShortHair_MouthPosition_5Weights(float F[]);  
        void set_RHand_ShortHair_MouthPosition_7Weights(float F[]); 

        void set_RHand_ShortHair_EyesPosition_5Weights(float F[]);  
        void set_RHand_ShortHair_EyesPosition_7Weights(float F[]);  

        void set_RHand_ShortHair_FrontPosition_5Weights(float F[]);  
        void set_RHand_ShortHair_FrontPosition_7Weights(float F[]);  

        void set_RHand_ShortHair_SidePosition_5Weights(float F[]);  
        void set_RHand_ShortHair_SidePosition_7Weights(float F[]);  

        void set_RHand_ShortHair_TopPosition_5Weights(float F[]);  
        void set_RHand_ShortHair_TopPosition_7Weights(float F[]);  

        void set_RHand_ShortHair_BackPosition_5Weights(float F[]);  
        void set_RHand_ShortHair_BackPosition_7Weights(float F[]);   

        void set_LHand_ShortHair_MouthPosition_5Weights(float F[]);  
        void set_LHand_ShortHair_MouthPosition_7Weights(float F[]); 

        void set_LHand_ShortHair_EyesPosition_5Weights(float F[]);  
        void set_LHand_ShortHair_EyesPosition_7Weights(float F[]);  

        void set_LHand_ShortHair_FrontPosition_5Weights(float F[]);  
        void set_LHand_ShortHair_FrontPosition_7Weights(float F[]);  

        void set_LHand_ShortHair_SidePosition_5Weights(float F[]);  
        void set_LHand_ShortHair_SidePosition_7Weights(float F[]);  

        void set_LHand_ShortHair_TopPosition_5Weights(float F[]);  
        void set_LHand_ShortHair_TopPosition_7Weights(float F[]);  

        void set_LHand_ShortHair_BackPosition_5Weights(float F[]);  
        void set_LHand_ShortHair_BackPosition_7Weights(float F[]);  

        //MEDIUM HAIR NN WEIGHTS
        void set_RHand_MediumHair_MouthPosition_5Weights(float F[]);  
        void set_RHand_MediumHair_MouthPosition_7Weights(float F[]); 

        void set_RHand_MediumHair_EyesPosition_5Weights(float F[]);  
        void set_RHand_MediumHair_EyesPosition_7Weights(float F[]);  

        void set_RHand_MediumHair_FrontPosition_5Weights(float F[]);  
        void set_RHand_MediumHair_FrontPosition_7Weights(float F[]);  

        void set_RHand_MediumHair_SidePosition_5Weights(float F[]);  
        void set_RHand_MediumHair_SidePosition_7Weights(float F[]);  

        void set_RHand_MediumHair_TopPosition_5Weights(float F[]);  
        void set_RHand_MediumHair_TopPosition_7Weights(float F[]);  

        void set_RHand_MediumHair_BackPosition_5Weights(float F[]);  
        void set_RHand_MediumHair_BackPosition_7Weights(float F[]);   

        void set_LHand_MediumHair_MouthPosition_5Weights(float F[]);  
        void set_LHand_MediumHair_MouthPosition_7Weights(float F[]); 

        void set_LHand_MediumHair_EyesPosition_5Weights(float F[]);  
        void set_LHand_MediumHair_EyesPosition_7Weights(float F[]);  

        void set_LHand_MediumHair_FrontPosition_5Weights(float F[]);  
        void set_LHand_MediumHair_FrontPosition_7Weights(float F[]);  

        void set_LHand_MediumHair_SidePosition_5Weights(float F[]);  
        void set_LHand_MediumHair_SidePosition_7Weights(float F[]);  

        void set_LHand_MediumHair_TopPosition_5Weights(float F[]);  
        void set_LHand_MediumHair_TopPosition_7Weights(float F[]);  

        void set_LHand_MediumHair_BackPosition_5Weights(float F[]);  
        void set_LHand_MediumHair_BackPosition_7Weights(float F[]);  

        //LONG HAIR NN WEIGHTS
        void set_RHand_LongHair_MouthPosition_5Weights(float F[]);  
        void set_RHand_LongHair_MouthPosition_7Weights(float F[]); 

        void set_RHand_LongHair_EyesPosition_5Weights(float F[]);  
        void set_RHand_LongHair_EyesPosition_7Weights(float F[]);  

        void set_RHand_LongHair_FrontPosition_5Weights(float F[]);  
        void set_RHand_LongHair_FrontPosition_7Weights(float F[]);  

        void set_RHand_LongHair_SidePosition_5Weights(float F[]);  
        void set_RHand_LongHair_SidePosition_7Weights(float F[]);  

        void set_RHand_LongHair_TopPosition_5Weights(float F[]);  
        void set_RHand_LongHair_TopPosition_7Weights(float F[]);  

        void set_RHand_LongHair_BackPosition_5Weights(float F[]);  
        void set_RHand_LongHair_BackPosition_7Weights(float F[]);   

        void set_LHand_LongHair_MouthPosition_5Weights(float F[]);  
        void set_LHand_LongHair_MouthPosition_7Weights(float F[]); 

        void set_LHand_LongHair_EyesPosition_5Weights(float F[]);  
        void set_LHand_LongHair_EyesPosition_7Weights(float F[]);  

        void set_LHand_LongHair_FrontPosition_5Weights(float F[]);  
        void set_LHand_LongHair_FrontPosition_7Weights(float F[]);  

        void set_LHand_LongHair_SidePosition_5Weights(float F[]);  
        void set_LHand_LongHair_SidePosition_7Weights(float F[]);  

        void set_LHand_LongHair_TopPosition_5Weights(float F[]);  
        void set_LHand_LongHair_TopPosition_7Weights(float F[]);  

        void set_LHand_LongHair_BackPosition_5Weights(float F[]);  
        void set_LHand_LongHair_BackPosition_7Weights(float F[]);  

        tingleNN_targets_t _target1;
        tingleNN_targets_t _target2;
        tingleNN_sensitivity_t _threshold;



        int _numNetworks = 1;
        //neural network weights
        float _F1_5in[500];
        float _F1_7in[575];
        float _F2_5in[500];
        float _F2_7in[575];
     /*   void powerOff(void);
        void constLat(void);
        void lowPower(void);
        uint8_t checkForSoftDevice(void); */
};

extern tingleNeuralNetwork neuralNetwork; //Arduino_nRF5x_lowPower nRF5x_lowPower;

#endif
