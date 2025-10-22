- Do a basic rlnAnglePsi and rlnAngleRot for microtubule prediction (no template).
- Later on, need to check for polarity (does the 14PF template get right 13PF polarity)
- PRIORITY: For final prediction.py or maybe in clean.py, use a direction to get rid of wrong prediction --psi_range -90,90
- For cilia (9), perform a renumber rlnHelicalID (later) cilia_cluster_renumber (Make a core utils sort.py)
- Further more for angle prediction (for other filament), we can do median filter and then smoothing, option --smooth (NOT priority)
- Edge Case: for microtubule a bit horizontal, then the line fit can have two lines that overlap a lot but not fully. Perhaps, the cleaning should modify to clean if overlap > 10 particles' length?
- There can be another kinds of prediction (--dphi and --dz) like the filament model in ChimeraX (later) (Determine polarity). Need to validate
- ChimeraX bundle (Does this worth the effort? Probably not)
- Test with data

v0.9.1
 - Fix bugs to longer tubes