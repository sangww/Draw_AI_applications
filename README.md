# Description
This repository contains two IPython Notebooks and related scripts, data for two pen stroke sequence generation models.

1. helping_hand.ipynb is an autoencoder model that learns a mapping from 2-DOF sketches (only x,y coordinates) to 5-DOF sketches (x,y coordinates, pen pressure, x,y tilt). The result is a replica of "HelpingHand: Example-based Stroke Stylization" (Lu, 2012)

2. handwriting.ipynb is an autoencoder model that learns to reconstruct readable handwritings of a certain style from simulated dysgraphia data. 

# Usage
Training and executing the models requires PyTorch with CUDA support. Google Colab notebook with GPU mode is recommended. To use Colab notebook, upload every files and folders to Google Drive, open the notebook file and import scripts from Google Drive. Notebooks contain necessary descriptions and instructions of code snippets.

# Repository structure

- /scripts
Python scripts for data processing and PyTorch models

- /models
Trained ML models. Can be loaded with PyTorch.

- /IAM_data
IAM online handwriting data, downloadable from https://fki.tic.heia-fr.ch/databases/iam-on-line-handwriting-database

Raw data is large does not need to be downloaded unless want to change the storage format. `handwriting.p` file is serialized numpy arrays containing over 10,000 handwriting sequences.

- /helping-hand
    - /HelpingHandDataset
    - /hhdemoRelease

Demo executable from "HelpingHand: Example-based Stroke Stylization" (Lu, 2012) https://gfx.cs.princeton.edu/pubs/Lu_2012_HES/index.php

Contains 6D (coordinates+pressure+tilt) drawing data in .cyn text format and software tool for sketch rendering

- /results
Some drawing data generated with ML models. Can be visualized using the Helping Hand tool.



# How to use HelpingHand visualization tool

The following is quoted from HelpingHand docs

## Data format

=============================================
The .cyn file format
=============================================

For example:

1					###number of strokes in the file  

stroke 0				###keyword "stroke" and the index of the stroke

3					###number of samples in the stroke

0	0	0.0361797	0.050332	0.270588	29.7969		0.406232	-0.281259

1    	0.007	0.0356365	0.0532817	0.481119	28.7381		0.406232	-0.29501

2	0.014	0.0350476	0.0562231	0.509804	28.5938		0.406232	-0.296884


###the eight numbers are: (from left to right)

1) sample index (start from 0)

2) time stamp

3) x position (0-1)

4) y position (0-1)

5) pressure (0-1) 

6) rotation (in angles) 

7) tiltX (-1, 1)

8) tiltY (-1, 1)

## Visualization tool

Supported Platforms (64 bit):

* Windows 7

* Mac OS X 10.7 Snow Leopard


Quick Start:

1. binary -> [Platform] -> x64 -> Release -> hhdemo

2. Start drawing on the white canvas with stylus or mouse.

3. Or load query strokes from file and synthesize the loaded strokes.

UI:

1. "Load Library"	: Load library exemplars (.cyn files with 6D gesture data) from "data" folder.

2. "Draw Library"	: Draw the library exemplars on the canvas.

3. "Load Query"		: Load query strokes (.cyn files with 2D position data) from "query" folder.

4. "Draw Query"		: Draw the original query strokes on the canvas.

5. "Synthesis"		: Select pose, trajectory or pose+trajectory synthesis method

6. "Enrich Lib"		: Enrich the library with different scales of the strokes

7. "Syn At Once"	: After loading library exemplars and query strokes from files, perform the synthesis and draw the entire synthesis results all at once

8. "Syn And Draw"	: After loading library exemplars and query strokes from files, perform the synthesis stroke by stroke, and draw the synthesized stroke one by one.

9. "Save Results"	: Save the synthesized query strokes into .cyn file in "results" folder.