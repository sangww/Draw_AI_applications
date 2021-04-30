HelpingHand Readme - Build dated 2 August, 2012

This demo accompanies the publication on the HelpingHand project at ACM SIGGRAPH 2012.  The full citation for the work is:

Jingwan Lu, Fisher Yu, Adam Finkelstein, and Stephen DiVerdi. HelpingHand: Example-based Stroke Stylization. ACM Transactions on Graphics (Proc. SIGGRAPH), August 2012.

Please use this citation when referencing the demo.  More information is available on the project at  http://gfx.cs.princeton.edu/pubs/Lu_2012_HES/ .

This software and all associated files are made available under a Creative Commons Attribution-NonCommercial-ShareAlike license. 
http://creativecommons.org/licenses/by-nc-sa/3.0/

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