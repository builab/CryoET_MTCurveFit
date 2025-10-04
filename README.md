Some code is based on 
https://github.com/PengxinChai/multi-curve-fitting

In command line:
mt_fit.py --angpix 14 --sample_step 82 --min_seed 6 --poly_order 3 CCDC147C_001_particles.star

Simple
mt_fit.py --angpix 14 CCDC147C_001_particles.star

Increase the min_seed to 6 is a lot cleaner than 5 but might ignore some MTs

In ChimeraX, navigate to the script folder and type:
runscript mcurvefit.py #1.2.1 voxelSize 1 sampleStep 82

runscript mcurvefit.py #1.2.1 voxelSize 1 minseed 5 poly 3 sampleStep 82

minseed 5 (default)
poly 3 (default)
sampleStep 82 (default)

mt_clean.py --angpix 14 --dist_thres 50 CCDC147C_001_particles_init_fit.star 

time python ~/Documents/GitHub/CryoET_MTCurveFit/mcurve_connect.py --dist_extrapolate 1500 --angpix 14 --min_seed 5 --overlap_thres 80 --sample_step 82 CCDC147C_001_particles_init_fit_filtered50A.star

For star_predict_angles.py

runscript mtpredictangles.py #1.2.1 template #1.2.2

example.md

python ~/Documents/GitHub/CryoET_MTCurveFit/scripts/mt_clean_connect.py --clean_dist_thres 50 --dist_extrapolate 1500 --angpix 14 --min_seed 5 --overlap_thres 80 --sample_step 82 CCDC147C_001_particles_init_fit.star

python ~/Documents/GitHub/CryoET_MTCurveFit/scripts/mt_fit_all.py --angpix 14 --sample_step 82 --min_seed 6 --poly_order 3 --clean_dist_thres 50 --dist_extrapolate 2000 --overlap_thres 100 CCDC147C_001_particles.star