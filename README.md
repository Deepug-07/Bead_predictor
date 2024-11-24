# Modeling of Bead Geometries of GMAW-Based Wire–Arc Additive Manufacturing of 316L Series Stainless Steels

## Authors
- Deepu Kumar Gupta (2101047)
- Ravi Raj (2101051)

## Supervised by
Vikas Upadhyay  
Associate Professor  
Department of Mechanical Engineering  
National Institute of Technology Patna  
Patna - 800005  

## Abstract
This project focuses on the modeling of bead geometries in Gas Metal Arc Welding (GMAW)-based Wire-Arc Additive Manufacturing (WAAM) of 316L stainless steel. The study investigates the effects of process parameters such as Wire Feed Speed (WFS), Travel Speed (TS), and Voltage on the resulting bead height (BH) and bead width (BW) using a neural network approach.
Additive manufacturing (AM) systems, which form 3D structures by layer deposition, have gained popularity due to reduced production time and material wastage. The WAAM process, particularly using GMAW, exhibits a higher deposition rate and cost-effectiveness compared to traditional methods. This research aims to model the bead geometries resulting from this process to enhance the quality of fabricated components.

The dataset utilized in this research is sourced from the paper titled "Experimental Investigations on Mechanical Properties of Multi-layered Structure Fabricated by GMAW-based WAAM of SS316L." The dataset includes:
- **Input Variables**: WFS, TS, Voltage.
- **Target Variables**: Bead Height (BH), Bead Width (BW).

The dataset was split into training (80%) and testing (20%) subsets.

## Neural Network Architecture
The neural network model consists of:
- **Input Layer**: Accepts features for WFS, TS, and Voltage.
- **Hidden Layers**: 
  - 1st Layer: 128 neurons (ReLU activation)
  - 2nd Layer: 64 neurons (ReLU activation, Batch Normalization, Dropout)
  - 3rd Layer: 32 neurons (ReLU activation, Batch Normalization, Dropout)
  - 4th Layer: 16 neurons (ReLU activation)
- **Output Layer**: 2 neurons (linear activation for BH and BW).

The model is compiled using the Adam optimizer with Mean Squared Error (MSE) as the loss function.
