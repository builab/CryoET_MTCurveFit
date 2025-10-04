Some code is based on 
https://github.com/PengxinChai/multi-curve-fitting


Command line step by step (Use the example file CCDC147C_001_particles.star):

Initial Fit: 
mt_fit.py --angpix 14 --sample_step 82 --min_seed 6 CCDC147C_001_particles.star

Increase the min_seed to 6 is a lot cleaner than 5 but might ignore some MTs

Clean duplicate
mt_clean.py --angpix 14 --dist_thres 50 CCDC147C_001_particles_init_fit.star 

Connect lines
m_connect.py --dist_extrapolate 1500 --angpix 14 --min_seed 5 --overlap_thres 80 --sample_step 82 CCDC147C_001_particles_init_fit_filtered50.0A.star

Combining clean & connect
mt_clean_connect.py --clean_dist_thres 50 --dist_extrapolate 1500 --angpix 14 --min_seed 5 --overlap_thres 80 --sample_step 82 CCDC147C_001_particles_init_fit.star

Combining everything in 1
mt_fit_simple.py --angpix 14 --sample_step 82 --min_seed 6 --poly_order 3 --clean_dist_thres 50 --dist_extrapolate 2000 --overlap_thres 100 CCDC147C_001_particles.star

Using inside ChimeraX
mtfitchimerax.py #1.2.1 voxelSize 14 sampleStep 82 minseed 6 poly 3 cleanDistThres 50 distExtrapolate 2000 overlapThres 100
