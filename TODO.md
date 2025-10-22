- Do a basic rlnAnglePsi and rlnAngleRot for microtubule prediction (no template).
- Later on, need to check for polarity (does the 14PF template get right 13PF polarity)
- For cilia (9), perform a renumber rlnHelicalID (later) cilia_cluster_renumber (Make a core utils sort.py). If > 9 filament, don't do it.
- Further more for angle prediction (for other filament), we can do median filter and then smoothing, option --smooth (NOT priority)
- PRIORITY: it is still not fully smooth now for predict.py
- Edge Case (Low priority): for microtubule a bit horizontal, then the line fit can have two lines that overlap a lot but not fully. Perhaps, the cleaning should modify to clean if overlap > 10 particles' length?
- There can be another kinds of prediction (--dphi and --dz) like the filament model in ChimeraX (later) (Determine polarity). Need to validate
- ChimeraX bundle (Does this worth the effort? Probably not)
- Test with data

v0.9.1
 - Fix bugs to longer tubes
 - Fix fit.py, predict.py to output right rlnAngleTilt and rlnAnglePsi
 - Implement filter by psi in clean.py to remove horizontal tubes.