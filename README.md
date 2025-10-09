# CRYOET_MTFIT

## Introduction
Line fitting based on 3D template matching of filaments (MT).  

Some codes and ideas are based on:  
👉 [https://github.com/PengxinChai/multi-curve-fitting](https://github.com/PengxinChai/multi-curve-fitting)

> **Note:** The code is not yet fully done or tested.

---

## Installation
_To be added._

---

## Usage

### Initial Fit
```bash
mt_fit.py fit CCDC147C_001_particles.star --angpix 14 --sample_step 82 --min_seed 6
```

Note: Increase the min_seed to 6 is a lot cleaner than 5 but might ignore some MTs

### Clean duplicate
mt_fit.py clean CCDC147C_001_particles_fitted.star --angpix 14 --dist_thres 50 

### Connect lines
mt_fit.py connect CCDC147C_001_particles_fitted_cleaned.star --dist_extrapolate 1500 --angpix 14 --min_seed 5 --overlap_thres 80 --sample_step 82 

### Predict (To be added)
mt_fit.py predict CCDC147C_001_particles_fitted_cleaned_connected.star --template CCDC147C_001_particles.star --range 100

### One commandline for all
mt_fit.py pipeline CCDC147C_001_particles.star --angpix 14 --sample_step 82 --min_seed 6 --poly_order 3 --clean_dist_thres 50 --dist_extrapolate 2000 --overlap_thres 100 


## VISUALIZING RESULTS
You can use ChimeraX with ArtiaX installed to visualize star files. On the other hand, you can use our simple star visualizer

### Visualize initial fit
view_star.py CCDC147C_001_particles.star

### View final results
view_star.py CCDC147C_001_particles_processed.star

### Or write out if running from a server
view_star.py --output final.html CCDC147C_001_particles_processed.star

## USING INSDIDE CHIMERAX
Open ChimeraX with ArtiaX, load your template matching star file.

For now, we would use like this:
cd ~/Documents/GitHub/CryoET_MTCurveFit/scripts
runscript mtfitchimerax.py #1.2.1 voxelSize 14 sampleStep 82 minseed 6 poly 3 cleanDistThres 50 distExtrapolate 2000 overlapThres 100
