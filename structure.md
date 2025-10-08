CryoET_MTFit/
├── install.sh			# Installing script
├── source-env.sh		# Source script
├── TODO.md			# TO DO
├── README.md			
├── structure.md		# Source script
├── examples/			# Test example
│   ├── CCDC146C_001_particles.star     # Template matching file
├── utils/
│   ├── __init__.py     # Module exports
│   ├── fit.py          # Initial Curve fitting logic
│   ├── clean.py        # Overlap detection & filtering logic
│   ├── connect.py      # Line connecting logic
│   ├── view.py      	# Visualize star file
│   └── io.py           # I/O utilities
└── scripts/
    ├── mt_fit.py       # CLI wrapper to fit, clean & connect
    ├── view_star_.py   # CLI wrapper to visualize star file
    └── mtfitchimerax.py     # ChimeraX interface of mt_fit.py pipeline only