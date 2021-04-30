HelpingHand: Example-based Stroke Stylization
Jingwan Lu, Fisher Yu, Adam Finkelstein, Stephen DiVerdi

Contact: jingwan.lu.cynthia@gmail.com

The dataset includes the hand writtings and line drawings we collected from artists and amateur users.
To visualize the dataset, download our application executable, http://gfx.cs.princeton.edu/pubs/Lu_2012_HES/hhDemo.zip

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