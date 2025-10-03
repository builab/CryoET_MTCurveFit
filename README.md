Some code is based on 
https://github.com/PengxinChai/multi-curve-fitting

In command line (need to fix the star file)
mcurve_fitting_3D.py --pixel_size_ang 1 --sample_step_ang 82 --min_number_seed 6 --poly_expon 3 MT_8.48Apx_warp.star

Simple
python mcurve_fitting_3D.py --pixel_size_ang 1 MT_8.48Apx_warp.star

Increase the min_seed to 6 is a lot cleaner than 5 but might ignore some MTs

In ChimeraX, navigate to the script folder and type:
runscript mcurvefit.py #1.2.1 voxelSize 1 sampleStep 82

runscript mcurvefit.py #1.2.1 voxelSize 1 minseed 5 poly 3 sampleStep 82

minseed 5 (default)
poly 3 (default)
sampleStep 82 (default)

time python ~/Documents/GitHub/CryoET_MTCurveFit/helical_overlap.py --angpix 14 --dist_thres 60 CCDC147C_001_particles_init_fit.star 

time python ~/Documents/GitHub/CryoET_MTCurveFit/connect_lines.py --dist_extrapolate 1500 --angpix 14 --min_seed 5 --overlap_thres 80 --sample_step 82 CCDC147C_001_particles_init_fit_filtered50A.star

For star_predict_angles.py

runscript mtpredictangles.py #1.2.1 template #1.2.2

example.md
