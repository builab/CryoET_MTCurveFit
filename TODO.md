- For cilia (9), perform a renumber rlnHelicalID (later) cilia_cluster_renumber (Make a core utils sort.py). If > 9 filament, don't do it.
- PRIORITY: it is still not fully smooth now for predict.py
- Edge Case (Low priority): for microtubule a bit horizontal, then the line fit can have two lines that overlap a lot but not fully. Perhaps, the cleaning should modify to clean if overlap > 10 particles' length?
- There can be another kinds of prediction (--dphi and --dz) like the filament model in ChimeraX (later) (Determine polarity). Need to validate
- ChimeraX bundle (Does this worth the effort? Probably not)
- Test with data

Less priorty
- Do a basic rlnAnglePsi and rlnAngleRot for microtubule prediction (no template).
- Later on, need to check for polarity (does the 14PF template get right 13PF polarity)
- visualize_angles.py Function to plot with line fitting rlnAngleRot, rlnAnglePsi, rlnAngleTilt for diagnosis.

v0.9.2
 - Fix bugs to longer tubes
 - Fix fit.py, predict.py to output right rlnAngleTilt and rlnAnglePsi
 - Implement filter by psi in clean.py to remove horizontal tubes.
 
v0.9.3
 - Make visualize_star_angles.py to see the smoothness of angles to improve predict.py using outlier detection.
 - Implement smooth_angles for predict.py and work so well