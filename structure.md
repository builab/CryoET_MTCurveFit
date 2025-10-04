CryoET_MTFit/
├── install.sh			# Installing script
├── source-env.sh		# Source script
├── README.md			
├── structure.md		# Source script
├── examples/			# Test example
│   ├── CCDC146C_001_particles.star     # Template matching file
├── utils/
│   ├── __init__.py     # Module exports
│   ├── fit.py          # Initial Curve fitting logic
│   ├── clean.py        # Overlap detection & filtering logic
│   ├── connect.py      # Line connecting logic
│   └── io.py           # I/O utilities
└── scripts/
    ├── mt_init_fit.py       # Initial Fit from Template CLI wrapper
    ├── mt_clean.py          # Clean CLI wrapper
    ├── mt_connect.py        # Connect MT wrapper
    ├── mt_clean_connect.py  # Clean & Connect Combined CLI wrapper
    ├── mt_fit_simple.py     # Combine Fit, Clean, Connect CLI wrapper with limited arguments
    └── mtfitchimerax.py     # ChimeraX interface of mt_fit_simple.py