import rospkg
import os

rospack = rospkg.RosPack()
pkg_path = rospack.get_path('curriculum_learning')
outdir = pkg_path + '/training_results'

 # Create directories for outputs
model_path = pkg_path + '/trained_models'
reports_dir = pkg_path + '/training_reports'
if not os.path.exists(model_path):
    os.makedirs(model_path)
if not os.path.exists(reports_dir):
    os.makedirs(reports_dir)