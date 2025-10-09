CryoET_MTFit/
├── install.sh			# Installing script
├── source-env.sh		# Source script
├── bundle_info.xml			# TO DO
├── README.md			
├── structure.md		# Source script
├── examples/			# Test example
│   ├── CCDC146C_001_particles.star     # Template matching file
└── src/
	├──	utils/
    │   ├── __init__.py     # Export core functions
    │   ├── fit.py          # Initial Curve fitting logic
    │   ├── clean.py        # Overlap detection & filtering logic
    │   ├── connect.py      # Line connecting logic
    │   ├── view.py      	# Visualize star file
    │   └── io.py           # I/O utilities
    ├── mtfit_cmd.py       # ChimeraX command script
    ├── mt_fit.py       # CLI wrapper to fit, clean & connect
    ├── view_star_.py   # CLI wrapper to visualize star file
    └── __init__.py     # Package marker