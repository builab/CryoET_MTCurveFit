Some code is based on 
https://github.com/PengxinChai/multi-curve-fitting

Command line step by step (Use the example file CCDC147C_001_particles.star):

Initial Fit: 
mt_fit.py fit CCDC147C_001_particles.star --angpix 14 --sample_step 82 --min_seed 6 

Increase the min_seed to 6 is a lot cleaner than 5 but might ignore some MTs

Clean duplicate
mt_fit.py clean CCDC147C_001_particles_fitted.star --angpix 14 --dist_thres 50 

Connect lines
mt_fit.py connect CCDC147C_001_particles_fitted_cleaned.star --dist_extrapolate 1500 --angpix 14 --min_seed 5 --overlap_thres 80 --sample_step 82 

Predict (To be added)
mt_fit.py predict CCDC147C_001_particles_fitted_cleaned_connected.star --template CCDC147C_001_particles.star --range 100

Combining everything in a pipeline (to add predict later)
mt_fit.py pipeline CCDC147C_001_particles.star --angpix 14 --sample_step 82 --min_seed 6 --poly_order 3 --clean_dist_thres 50 --dist_extrapolate 2000 --overlap_thres 100 

Using inside ChimeraX
cd ~/Documents/GitHub/CryoET_MTCurveFit/scripts
runscript mtfitchimerax.py #1.2.1 voxelSize 14 sampleStep 82 minseed 6 poly 3 cleanDistThres 50 distExtrapolate 2000 overlapThres 100
